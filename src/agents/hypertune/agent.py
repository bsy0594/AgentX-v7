"""
Hyperparameter Tuner Agent — Optuna-based hyperparameter optimization.
Receives: tar.gz (competition data) + best model name (text).
Returns: optimized hyperparameters as JSON.
Falls back to curated defaults if Optuna is unavailable.
"""
import asyncio
import base64
import io
import json
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from a2a.server.tasks import TaskUpdater
from a2a.types import FilePart, FileWithBytes, Message, Part, TaskState, TextPart
from a2a.utils import get_message_text, new_agent_text_message


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced feature engineering matching Feature Engineer Agent output."""
    out = pd.DataFrame(index=df.index)

    if "Cabin" in df.columns:
        cabin = df["Cabin"].astype(str).str.split("/", expand=True)
        out["Cabin_Deck"] = cabin[0].where(cabin[0] != "nan", other="__NA__")
        raw_num = pd.to_numeric(cabin.get(1), errors="coerce")
        out["Cabin_Num"]    = raw_num
        out["CabinNum_bin"] = pd.cut(raw_num, bins=10, labels=False).astype("float")
        out["Cabin_Side"]   = (cabin[2].where(cabin[2] != "nan", other="__NA__")
                               if 2 in cabin.columns else "__NA__")

    if "PassengerId" in df.columns:
        pid = df["PassengerId"].astype(str).str.split("_", expand=True)
        out["Group"] = pd.to_numeric(pid[0], errors="coerce")
        grp_series = df["PassengerId"].astype(str).str.split("_").str[0]
        grp_size   = grp_series.map(grp_series.value_counts())
        out["GroupSize"] = grp_size.values
        out["IsSolo"]    = (grp_size == 1).astype(int).values

    spend_cols    = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    existing_spend = [c for c in spend_cols if c in df.columns]
    if existing_spend:
        spend = df[existing_spend].apply(pd.to_numeric, errors="coerce").fillna(0)
        total = spend.sum(axis=1)
        out["TotalSpend"]    = total
        out["AnySpend"]      = (total > 0).astype(int)
        out["LogTotalSpend"] = np.log1p(total)
        grp_size_safe = out.get("GroupSize", pd.Series(1, index=df.index)).replace(0, 1)
        out["SpendPerPerson"] = total / grp_size_safe
        for c in existing_spend:
            out[c]           = spend[c]
            out[f"Log{c}"]   = np.log1p(spend[c])

    for col in df.columns:
        uniq = set(str(v).lower() for v in df[col].dropna().unique())
        if uniq <= {"true", "false"}:
            out[f"{col}_int"] = df[col].map(
                {"True": 1, "False": 0, True: 1, False: 0}
            ).fillna(-1).astype(int)

    if "CryoSleep" in df.columns and existing_spend:
        cryo = df["CryoSleep"].map({"True": 1, "False": 0, True: 1, False: 0}).fillna(0)
        out["CryoSpend_flag"] = ((cryo == 1) & (out.get("TotalSpend", pd.Series(0, index=df.index)) > 0)).astype(int)

    skip = {"Cabin", "PassengerId", "Name"}
    for c in df.columns:
        if c not in skip and c not in out.columns:
            out[c] = df[c]

    return out


class HyperparameterTunerAgent:
    async def run(self, message: Message, updater: TaskUpdater) -> None:
        tar_bytes  = self._extract_tar(message)
        model_name = self._extract_model_name(message)

        if not tar_bytes:
            await updater.failed(new_agent_text_message("[Tuner] No data received."))
            return

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"[Tuner] Optimizing {model_name} hyperparameters..."),
        )

        with tempfile.TemporaryDirectory(prefix="tuner_") as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
                tar.extractall(data_dir)
            result_json = await asyncio.to_thread(self._tune, data_dir, model_name)

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"TUNED_PARAMS_START\n{result_json}\nTUNED_PARAMS_END"),
        )

    # ── tuning logic ──────────────────────────────────────────────────────────

    def _tune(self, data_dir: Path, model_name: str) -> str:
        train_path  = next(data_dir.rglob("train.csv"))
        sample_path = next(data_dir.rglob("sample_submission.csv"), None)

        train = pd.read_csv(train_path)
        if sample_path:
            sample     = pd.read_csv(sample_path)
            id_col     = sample.columns[0]
            target_col = sample.columns[1]
        else:
            id_col     = train.columns[0]
            target_col = train.columns[-1]

        feat_cols = [c for c in train.columns if c not in (id_col, target_col)]
        X_raw = engineer(train[feat_cols].copy())
        y     = train[target_col].copy()

        num_cols = X_raw.select_dtypes(include="number").columns.tolist()
        cat_cols = [c for c in X_raw.columns if c not in num_cols]
        X = X_raw.assign(**{c: X_raw[c].astype(str) for c in cat_cols})

        num_pipe = Pipeline([("imp", SimpleImputer(strategy="median"))])
        cat_pipe = Pipeline([
            ("imp", SimpleImputer(strategy="constant", fill_value="__NA__")),
            ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ])
        transformers = []
        if num_cols: transformers.append(("num", num_pipe, num_cols))
        if cat_cols: transformers.append(("cat", cat_pipe, cat_cols))
        pre = ColumnTransformer(transformers=transformers, remainder="drop")

        try:
            return self._tune_with_optuna(pre, X, y, model_name)
        except ImportError:
            print("[Tuner] Optuna not available — using curated defaults")
            return self._curated_defaults(model_name)

    def _tune_with_optuna(self, pre, X, y, model_name: str) -> str:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            if model_name == "XGBoost":
                from xgboost import XGBClassifier
                model = XGBClassifier(
                    n_estimators=trial.suggest_int("n_estimators", 200, 800),
                    max_depth=trial.suggest_int("max_depth", 3, 8),
                    learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                    subsample=trial.suggest_float("subsample", 0.6, 1.0),
                    colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
                    gamma=trial.suggest_float("gamma", 0.0, 0.5),
                    reg_alpha=trial.suggest_float("reg_alpha", 0.0, 1.0),
                    reg_lambda=trial.suggest_float("reg_lambda", 0.5, 2.0),
                    random_state=42, eval_metric="logloss", verbosity=0,
                )
            elif model_name == "LightGBM":
                from lightgbm import LGBMClassifier
                model = LGBMClassifier(
                    n_estimators=trial.suggest_int("n_estimators", 200, 800),
                    max_depth=trial.suggest_int("max_depth", 3, 8),
                    learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                    subsample=trial.suggest_float("subsample", 0.6, 1.0),
                    colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    min_child_samples=trial.suggest_int("min_child_samples", 10, 50),
                    num_leaves=trial.suggest_int("num_leaves", 20, 100),
                    reg_alpha=trial.suggest_float("reg_alpha", 0.0, 1.0),
                    reg_lambda=trial.suggest_float("reg_lambda", 0.0, 1.0),
                    random_state=42, verbose=-1,
                )
            elif model_name == "GradientBoosting":
                from sklearn.ensemble import GradientBoostingClassifier
                model = GradientBoostingClassifier(
                    n_estimators=trial.suggest_int("n_estimators", 100, 500),
                    max_depth=trial.suggest_int("max_depth", 3, 7),
                    learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                    subsample=trial.suggest_float("subsample", 0.6, 1.0),
                    min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
                    random_state=42,
                )
            else:  # RandomForest
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(
                    n_estimators=trial.suggest_int("n_estimators", 100, 500),
                    max_depth=trial.suggest_int("max_depth", 5, 20),
                    min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
                    max_features=trial.suggest_float("max_features", 0.3, 1.0),
                    random_state=42, n_jobs=1,
                )

            pipe   = Pipeline([("pre", pre), ("model", model)])
            scores = cross_val_score(pipe, X, y, cv=5, scoring="accuracy", n_jobs=1)
            return scores.mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=100, show_progress_bar=False)

        best_params = study.best_params
        best_score  = study.best_value
        print(f"[Tuner] Best CV accuracy: {best_score:.5f} | params: {best_params}")

        return json.dumps({
            "model": model_name,
            "params": best_params,
            "best_cv_accuracy": round(best_score, 5),
            "n_trials": 100,
        }, indent=2)

    def _curated_defaults(self, model_name: str) -> str:
        defaults = {
            "XGBoost": {
                "n_estimators": 600, "max_depth": 6, "learning_rate": 0.03,
                "subsample": 0.8, "colsample_bytree": 0.7,
                "min_child_weight": 3, "gamma": 0.1,
                "reg_alpha": 0.1, "reg_lambda": 1.0,
            },
            "LightGBM": {
                "n_estimators": 600, "max_depth": 6, "learning_rate": 0.03,
                "subsample": 0.8, "colsample_bytree": 0.7,
                "min_child_samples": 20, "num_leaves": 63,
                "reg_alpha": 0.1, "reg_lambda": 0.5,
            },
            "GradientBoosting": {
                "n_estimators": 400, "max_depth": 5, "learning_rate": 0.05,
                "subsample": 0.8,
            },
            "RandomForest": {
                "n_estimators": 500, "max_depth": None,
                "min_samples_leaf": 2,
            },
        }
        params = defaults.get(model_name, defaults["XGBoost"])
        return json.dumps({
            "model": model_name,
            "params": params,
            "best_cv_accuracy": None,
            "n_trials": 0,
            "source": "curated_defaults",
        }, indent=2)

    def _extract_tar(self, message: Message) -> bytes | None:
        for part in message.parts:
            if isinstance(part.root, FilePart):
                fd = part.root.file
                if isinstance(fd, FileWithBytes):
                    return base64.b64decode(fd.bytes)
        return None

    def _extract_model_name(self, message: Message) -> str:
        for part in message.parts:
            if isinstance(part.root, TextPart):
                text = part.root.text or ""
                if text.startswith("MODEL:"):
                    return text.replace("MODEL:", "").strip()
        text = get_message_text(message) or ""
        for name in ["XGBoost", "LightGBM", "GradientBoosting", "RandomForest"]:
            if name in text:
                return name
        return "XGBoost"
