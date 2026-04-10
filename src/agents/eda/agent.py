"""
EDA Agent — Runs actual pandas/numpy analysis on competition data.
No GPT needed: deterministic statistical report for other agents to consume.
"""
import base64
import io
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from a2a.server.tasks import TaskUpdater
from a2a.types import FilePart, FileWithBytes, Message, TaskState
from a2a.utils import new_agent_text_message


class EDAAgent:
    async def run(self, message: Message, updater: TaskUpdater) -> None:
        tar_bytes = self._extract_tar(message)
        if not tar_bytes:
            await updater.failed(new_agent_text_message("[EDA] No data received."))
            return

        await updater.update_status(
            TaskState.working,
            new_agent_text_message("[EDA] Extracting and analyzing data..."),
        )

        with tempfile.TemporaryDirectory(prefix="eda_") as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
                tar.extractall(data_dir)
            report = self._analyze(data_dir)

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"EDA_REPORT_START\n{report}\nEDA_REPORT_END"),
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    def _extract_tar(self, message: Message) -> bytes | None:
        for part in message.parts:
            if isinstance(part.root, FilePart):
                fd = part.root.file
                if isinstance(fd, FileWithBytes):
                    return base64.b64decode(fd.bytes)
        return None

    def _analyze(self, data_dir: Path) -> str:
        parts: list[str] = []

        train_path  = next(data_dir.rglob("train.csv"), None)
        test_path   = next(data_dir.rglob("test.csv"), None)
        sample_path = next(data_dir.rglob("sample_submission.csv"), None)

        if not train_path:
            return "ERROR: could not find train.csv"

        train = pd.read_csv(train_path)
        test  = pd.read_csv(test_path) if test_path else None

        if sample_path:
            sample     = pd.read_csv(sample_path)
            id_col     = sample.columns[0]
            target_col = sample.columns[1]
        else:
            id_col     = train.columns[0]
            target_col = train.columns[-1]

        feat_cols = [c for c in train.columns if c not in (id_col, target_col)]
        num_cols  = train[feat_cols].select_dtypes(include="number").columns.tolist()
        cat_cols  = [c for c in feat_cols if c not in num_cols]

        # ── Overview ──────────────────────────────────────────────────────────
        parts.append("=== DATASET OVERVIEW ===")
        parts.append(f"Train : {train.shape[0]} rows × {train.shape[1]} cols")
        if test is not None:
            parts.append(f"Test  : {test.shape[0]} rows × {test.shape[1]} cols")
        parts.append(f"ID col: {id_col}   Target: {target_col}")

        # ── Target distribution ───────────────────────────────────────────────
        parts.append("\n=== TARGET DISTRIBUTION ===")
        vc = train[target_col].value_counts(normalize=True)
        for val, pct in vc.items():
            parts.append(f"  {val}: {pct:.1%} ({int(pct * len(train))} rows)")
        majority_pct = vc.max()
        parts.append(f"  Imbalanced: {'YES' if majority_pct > 0.65 else 'NO'} (majority {majority_pct:.1%})")

        # ── Null rates ────────────────────────────────────────────────────────
        parts.append("\n=== NULL RATES (train) ===")
        null_rates = (train.isnull().mean() * 100).sort_values(ascending=False)
        has_nulls = null_rates[null_rates > 0]
        if has_nulls.empty:
            parts.append("  No nulls found.")
        for col, rate in has_nulls.items():
            parts.append(f"  {col}: {rate:.1f}%")

        # ── Numeric features ──────────────────────────────────────────────────
        parts.append(f"\n=== NUMERIC FEATURES ({len(num_cols)}) ===")
        for col in num_cols:
            s = train[col].describe()
            skew = train[col].skew()
            parts.append(
                f"  {col}: mean={s['mean']:.2f}, std={s['std']:.2f}, "
                f"range=[{s['min']:.1f},{s['max']:.1f}], "
                f"null={train[col].isnull().sum()}, skew={skew:.2f}"
            )

        # ── Categorical features ───────────────────────────────────────────────
        parts.append(f"\n=== CATEGORICAL FEATURES ({len(cat_cols)}) ===")
        for col in cat_cols:
            n_unique = train[col].nunique()
            top = train[col].value_counts().head(4).to_dict()
            parts.append(f"  {col}: {n_unique} unique | top={top} | null={train[col].isnull().sum()}")

        # ── Feature-target correlation ────────────────────────────────────────
        parts.append("\n=== FEATURE-TARGET CORRELATION (|r|) ===")
        try:
            y_numeric = pd.to_numeric(train[target_col].map({v: i for i, v in enumerate(train[target_col].unique())}), errors="coerce")
            corrs: dict[str, float] = {}
            for col in num_cols:
                c = train[col].corr(y_numeric)
                if not np.isnan(c):
                    corrs[col] = abs(c)
            # One-hot of categoricals (top 2 values each)
            for col in cat_cols[:8]:
                vc2 = train[col].value_counts().head(2).index
                for val in vc2:
                    dummy = (train[col] == val).astype(float)
                    c = dummy.corr(y_numeric)
                    if not np.isnan(c):
                        corrs[f"{col}=={val}"] = abs(c)
            for feat, corr in sorted(corrs.items(), key=lambda x: x[1], reverse=True)[:20]:
                parts.append(f"  {feat}: {corr:.4f}")
        except Exception as e:
            parts.append(f"  (skipped: {e})")

        # ── Feature engineering hints ─────────────────────────────────────────
        parts.append("\n=== FEATURE ENGINEERING OPPORTUNITIES ===")

        if "Cabin" in train.columns:
            sample_val = train["Cabin"].dropna().iloc[0] if not train["Cabin"].dropna().empty else "?"
            n_sep = str(sample_val).count("/")
            parts.append(f"  'Cabin' example: '{sample_val}' → {n_sep+1}-part split (Deck/Num/Side)")

        if "PassengerId" in train.columns:
            sample_pid = str(train["PassengerId"].iloc[0])
            parts.append(f"  'PassengerId' example: '{sample_pid}' → extract group prefix")

        if "Name" in train.columns:
            parts.append("  'Name' column present → extract family name (last token)")

        spend_cols    = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
        found_spends  = [c for c in spend_cols if c in train.columns]
        if found_spends:
            zero_rates = {c: (train[c] == 0).mean() for c in found_spends}
            parts.append(f"  Spending cols: {found_spends}")
            parts.append(f"  Zero-spend rates: {zero_rates}")
            parts.append("  → Recommend: TotalSpend, AnySpend, LogSpend per col")

        # Detect boolean columns stored as object
        for col in cat_cols:
            uniq = set(str(v).lower() for v in train[col].dropna().unique())
            if uniq <= {"true", "false"}:
                parts.append(f"  '{col}' is boolean-as-object → cast carefully before encoding")

        return "\n".join(parts)
