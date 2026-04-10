"""
Threshold Optimizer Agent — CV-based classification threshold optimization.
Receives: tar.gz (competition data) + tuned params JSON + output path (text).
Trains model with predict_proba, finds optimal threshold via OOF CV, applies to test.
Returns: THRESHOLD_PATH:/path/to/submission
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
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from a2a.server.tasks import TaskUpdater
from a2a.types import FilePart, FileWithBytes, Message, TaskState, TextPart
from a2a.utils import get_message_text, new_agent_text_message


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced feature engineering — matches Feature Engineer Agent."""
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

    spend_cols     = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
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
            out[c]         = spend[c]
            out[f"Log{c}"] = np.log1p(spend[c])

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


class ThresholdOptimizerAgent:
    async def run(self, message: Message, updater: TaskUpdater) -> None:
        tar_bytes   = self._extract_tar(message)
        params_info = self._extract_params(message)
        output_path = self._extract_output_path(message)

        if not tar_bytes:
            await updater.failed(new_agent_text_message("[Thresh] No data received."))
            return

        model_name = params_info.get("model", "XGBoost")
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"[Thresh] Finding optimal threshold for {model_name}..."),
        )

        with tempfile.TemporaryDirectory(prefix="thresh_") as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
                tar.extractall(data_dir)
            result = await asyncio.to_thread(
                self._optimize, data_dir, params_info, output_path
            )

        if result:
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"THRESHOLD_PATH:{result}"),
            )
        else:
            await updater.failed(new_agent_text_message("[Thresh] Optimization failed."))

    # ── optimization logic ────────────────────────────────────────────────────

    def _optimize(self, data_dir: Path, params_info: dict, output_path: str | None) -> str | None:
        train_path  = next(data_dir.rglob("train.csv"), None)
        test_path   = next(data_dir.rglob("test.csv"), None)
        sample_path = next(data_dir.rglob("sample_submission.csv"), None)

        if not train_path or not test_path:
            print("[Thresh] Missing train or test CSV")
            return None

        train = pd.read_csv(train_path)
        test  = pd.read_csv(test_path)

        if sample_path:
            sample     = pd.read_csv(sample_path)
            id_col     = sample.columns[0]
            target_col = sample.columns[1]
        else:
            id_col     = train.columns[0]
            target_col = train.columns[-1]

        feat_cols = [c for c in train.columns if c not in (id_col, target_col)]
        X_raw = engineer(train[feat_cols].copy())
        y_raw = train[target_col].copy()

        # Encode target to 0/1
        classes    = sorted(y_raw.unique(), key=str)
        class_map  = {v: i for i, v in enumerate(classes)}
        y          = y_raw.map(class_map).fillna(0).astype(int)
        inv_map    = {i: v for v, i in class_map.items()}

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

        model = self._build_model(params_info)
        if model is None:
            return None

        # ── OOF probabilities ─────────────────────────────────────────────────
        skf        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof_probs  = np.zeros(len(y))
        pipe       = Pipeline([("pre", pre), ("model", model)])

        for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr        = y.iloc[tr_idx]
            pipe.fit(X_tr, y_tr)
            probs = pipe.predict_proba(X_val)
            # probability of positive class (index 1)
            pos_idx = list(pipe.classes_).index(1) if hasattr(pipe, "classes_") else 1
            oof_probs[val_idx] = probs[:, pos_idx]
            print(f"[Thresh] Fold {fold+1} done")

        # ── Find optimal threshold ────────────────────────────────────────────
        thresholds = np.arange(0.30, 0.70, 0.005)
        best_thresh, best_acc = 0.5, 0.0
        for t in thresholds:
            preds = (oof_probs >= t).astype(int)
            acc   = (preds == y.values).mean()
            if acc > best_acc:
                best_acc, best_thresh = acc, t

        print(f"[Thresh] Optimal threshold: {best_thresh:.3f} (OOF acc: {best_acc:.5f} vs 0.5: {(oof_probs >= 0.5).astype(int).eq(y).mean():.5f})")

        # ── Predict on test ───────────────────────────────────────────────────
        test_feat = engineer(test[[c for c in test.columns if c != id_col]].copy())
        test_cat  = [c for c in test_feat.columns if c not in test_feat.select_dtypes(include="number").columns]
        X_test    = test_feat.assign(**{c: test_feat[c].astype(str) for c in test_cat})

        pipe.fit(X, y)  # refit on full train
        test_probs = pipe.predict_proba(X_test)
        pos_idx    = list(pipe.classes_).index(1) if hasattr(pipe, "classes_") else 1
        test_preds = (test_probs[:, pos_idx] >= best_thresh).astype(int)
        test_labels = [inv_map.get(p, p) for p in test_preds]

        ids = test[id_col].reset_index(drop=True)
        sub = pd.DataFrame({id_col: ids, target_col: test_labels})

        # Save submission
        out_path = Path(output_path) if output_path else Path(tempfile.mktemp(suffix="_thresh.csv"))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sub.to_csv(out_path, index=False)
        print(f"[Thresh] Saved {len(sub)} rows → {out_path}")
        return str(out_path)

    def _build_model(self, params_info: dict):
        model_name = params_info.get("model", "XGBoost")
        params     = params_info.get("params", {})
        try:
            if model_name == "XGBoost":
                from xgboost import XGBClassifier
                return XGBClassifier(**{**params, "random_state": 42, "eval_metric": "logloss", "verbosity": 0})
            elif model_name == "LightGBM":
                from lightgbm import LGBMClassifier
                return LGBMClassifier(**{**params, "random_state": 42, "verbose": -1})
            elif model_name == "GradientBoosting":
                from sklearn.ensemble import GradientBoostingClassifier
                return GradientBoostingClassifier(**{**params, "random_state": 42})
            else:
                from sklearn.ensemble import RandomForestClassifier
                return RandomForestClassifier(**{**params, "random_state": 42, "n_jobs": -1})
        except Exception as e:
            print(f"[Thresh] Model build failed: {e}")
            return None

    def _extract_tar(self, message: Message) -> bytes | None:
        for part in message.parts:
            if isinstance(part.root, FilePart):
                fd = part.root.file
                if isinstance(fd, FileWithBytes):
                    return base64.b64decode(fd.bytes)
        return None

    def _extract_params(self, message: Message) -> dict:
        text = get_message_text(message) or ""
        if "TUNED_PARAMS:" in text:
            try:
                start = text.index("TUNED_PARAMS:") + len("TUNED_PARAMS:")
                end   = text.index("\nOUTPUT_PATH:") if "\nOUTPUT_PATH:" in text else len(text)
                return json.loads(text[start:end].strip())
            except Exception as e:
                print(f"[Thresh] Failed to parse params: {e}")
        return {"model": "XGBoost", "params": {}}

    def _extract_output_path(self, message: Message) -> str | None:
        text = get_message_text(message) or ""
        if "OUTPUT_PATH:" in text:
            try:
                start = text.index("OUTPUT_PATH:") + len("OUTPUT_PATH:")
                return text[start:].strip().splitlines()[0]
            except Exception:
                pass
        return None
