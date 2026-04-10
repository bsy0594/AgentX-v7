"""
Model Selector Agent — CV-based model competition.
Tries XGBoost, LightGBM, GradientBoosting, RandomForest on the actual data.
Returns best model config as JSON. No GPT needed.
"""
import asyncio
import io
import json
import tarfile
import tempfile
import base64
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from a2a.server.tasks import TaskUpdater
from a2a.types import FilePart, FileWithBytes, Message, TaskState
from a2a.utils import get_message_text, new_agent_text_message


# ── Feature engineering (same logic as ML agent fallback) ─────────────────────
def engineer(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    # Cabin → Deck / Num / Side
    if "Cabin" in df.columns:
        cabin = df["Cabin"].astype(str).str.split("/", expand=True)
        out["Cabin_Deck"] = cabin[0].where(cabin[0] != "nan", other="__NA__")
        out["Cabin_Num"]  = pd.to_numeric(cabin.get(1), errors="coerce")
        out["Cabin_Side"] = cabin[2].where(cabin[2] != "nan", other="__NA__") if 2 in cabin.columns else "__NA__"

    # PassengerId → Group
    if "PassengerId" in df.columns:
        pid = df["PassengerId"].astype(str).str.split("_", expand=True)
        out["Group"] = pd.to_numeric(pid[0], errors="coerce")

    # Spending features
    spend_cols    = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    existing_spend = [c for c in spend_cols if c in df.columns]
    if existing_spend:
        spend = df[existing_spend].apply(pd.to_numeric, errors="coerce").fillna(0)
        out["TotalSpend"] = spend.sum(axis=1)
        out["AnySpend"]   = (spend.sum(axis=1) > 0).astype(int)
        for c in existing_spend:
            out[c] = spend[c]

    # Keep all other raw cols (except high-cardinality)
    skip = {"Cabin", "PassengerId", "Name"}
    for c in df.columns:
        if c not in skip and c not in out.columns:
            out[c] = df[c]

    return out


class ModelSelectorAgent:
    async def run(self, message: Message, updater: TaskUpdater) -> None:
        tar_bytes = self._extract_tar(message)
        if not tar_bytes:
            await updater.failed(new_agent_text_message("[ModelSelector] No data received."))
            return

        await updater.update_status(
            TaskState.working,
            new_agent_text_message("[ModelSelector] Running cross-validation tournament..."),
        )

        with tempfile.TemporaryDirectory(prefix="model_sel_") as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
                tar.extractall(data_dir)

            result_json = await asyncio.to_thread(self._select, data_dir)

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"MODEL_SELECTION_START\n{result_json}\nMODEL_SELECTION_END"),
        )

    # ── CV logic ──────────────────────────────────────────────────────────────

    def _select(self, data_dir: Path) -> str:
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

        y    = train[target_col].copy()
        feat = [c for c in train.columns if c not in (id_col, target_col)]
        X    = engineer(train[feat].copy())

        num_cols = X.select_dtypes(include="number").columns.tolist()
        cat_cols = [c for c in X.columns if c not in num_cols]
        X        = X.assign(**{c: X[c].astype(str) for c in cat_cols})

        num_pipe = Pipeline([("imp", SimpleImputer(strategy="median"))])
        cat_pipe = Pipeline([
            ("imp",  SimpleImputer(strategy="constant", fill_value="__NA__")),
            ("enc",  OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ])
        transformers = []
        if num_cols: transformers.append(("num", num_pipe, num_cols))
        if cat_cols: transformers.append(("cat", cat_pipe, cat_cols))
        pre = ColumnTransformer(transformers=transformers, remainder="drop")

        # ── Model candidates ──────────────────────────────────────────────────
        candidates: dict[str, object] = {}

        try:
            from xgboost import XGBClassifier
            candidates["XGBoost"] = XGBClassifier(
                n_estimators=400, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                min_child_weight=3, gamma=0.1,
                random_state=42, eval_metric="logloss", verbosity=0,
            )
        except ImportError:
            print("[ModelSelector] xgboost not available")

        try:
            from lightgbm import LGBMClassifier
            candidates["LightGBM"] = LGBMClassifier(
                n_estimators=400, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                min_child_samples=20,
                random_state=42, verbose=-1,
            )
        except ImportError:
            print("[ModelSelector] lightgbm not available")

        candidates["GradientBoosting"] = GradientBoostingClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=42,
        )
        candidates["RandomForest"] = RandomForestClassifier(
            n_estimators=300, max_depth=None,
            random_state=42, n_jobs=-1,
        )

        # ── Run CV ────────────────────────────────────────────────────────────
        results: dict[str, dict] = {}
        for name, model in candidates.items():
            try:
                pipe   = Pipeline([("pre", pre), ("model", model)])
                scores = cross_val_score(pipe, X, y, cv=5, scoring="accuracy", n_jobs=1)
                results[name] = {
                    "cv_mean":  round(float(scores.mean()), 5),
                    "cv_std":   round(float(scores.std()), 5),
                    "cv_scores": [round(float(s), 5) for s in scores],
                }
                print(f"[ModelSelector] {name}: {scores.mean():.5f} ± {scores.std():.5f}")
            except Exception as e:
                print(f"[ModelSelector] {name} failed: {e}")
                results[name] = {"error": str(e)}

        ok = {k: v for k, v in results.items() if "cv_mean" in v}
        if not ok:
            return json.dumps({"error": "All models failed CV", "details": results})

        best = max(ok, key=lambda k: ok[k]["cv_mean"])
        print(f"[ModelSelector] Winner: {best} ({ok[best]['cv_mean']:.5f})")

        return json.dumps({
            "best_model":   best,
            "best_cv_mean": ok[best]["cv_mean"],
            "best_cv_std":  ok[best]["cv_std"],
            "all_scores":   results,
        }, indent=2)

    def _extract_tar(self, message: Message) -> bytes | None:
        for part in message.parts:
            if isinstance(part.root, FilePart):
                fd = part.root.file
                if isinstance(fd, FileWithBytes):
                    return base64.b64decode(fd.bytes)
        return None
