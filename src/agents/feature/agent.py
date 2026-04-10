"""
Feature Engineer Agent — Deterministic domain-specific feature engineering for tabular competitions.
Analyzes the data and returns:
  1. An enhanced engineer() function (Python code)
  2. A human-readable feature engineering report
No LLM needed.
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


class FeatureEngineerAgent:
    async def run(self, message: Message, updater: TaskUpdater) -> None:
        tar_bytes = self._extract_tar(message)
        if not tar_bytes:
            await updater.failed(new_agent_text_message("[FE] No data received."))
            return

        await updater.update_status(
            TaskState.working,
            new_agent_text_message("[FE] Analyzing data for feature engineering opportunities..."),
        )

        with tempfile.TemporaryDirectory(prefix="fe_") as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
                tar.extractall(data_dir)
            report, code = self._engineer(data_dir)

        payload = f"FE_REPORT_START\n{report}\nFE_REPORT_END\nFE_CODE_START\n{code}\nFE_CODE_END"
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(payload),
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    def _extract_tar(self, message: Message) -> bytes | None:
        for part in message.parts:
            if isinstance(part.root, FilePart):
                fd = part.root.file
                if isinstance(fd, FileWithBytes):
                    return base64.b64decode(fd.bytes)
        return None

    def _engineer(self, data_dir: Path) -> tuple[str, str]:
        train_path  = next(data_dir.rglob("train.csv"), None)
        sample_path = next(data_dir.rglob("sample_submission.csv"), None)

        if not train_path:
            return "ERROR: train.csv not found", ""

        train = pd.read_csv(train_path)
        if sample_path:
            sample     = pd.read_csv(sample_path)
            id_col     = sample.columns[0]
            target_col = sample.columns[1]
        else:
            id_col     = train.columns[0]
            target_col = train.columns[-1]

        report_parts: list[str] = ["=== FEATURE ENGINEERING REPORT ==="]
        features_added: list[str] = []

        # ── Detect features ───────────────────────────────────────────────────
        has_cabin      = "Cabin" in train.columns
        has_pid        = "PassengerId" in train.columns
        has_name       = "Name" in train.columns
        spend_cols     = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
        existing_spend = [c for c in spend_cols if c in train.columns]
        bool_cols      = []
        for col in train.columns:
            if col in (id_col, target_col):
                continue
            uniq = set(str(v).lower() for v in train[col].dropna().unique())
            if uniq <= {"true", "false"}:
                bool_cols.append(col)

        # ── Group features (from PassengerId) ─────────────────────────────────
        if has_pid:
            train["_Group"] = train["PassengerId"].astype(str).str.split("_").str[0]
            group_size = train.groupby("_Group")["_Group"].transform("count")
            report_parts.append(f"\nGROUP ANALYSIS (from PassengerId):")
            report_parts.append(f"  Group size: mean={group_size.mean():.2f}, max={group_size.max()}")
            report_parts.append(f"  Solo travelers: {(group_size == 1).mean():.1%}")
            features_added += ["Group (group id as int)", "GroupSize", "IsSolo (GroupSize==1)"]

            if existing_spend and target_col in train.columns:
                # Group CryoSleep rate
                if "CryoSleep" in train.columns:
                    cryo_num = train["CryoSleep"].map({"True": 1, "False": 0, True: 1, False: 0}).fillna(0)
                    grp_cryo = cryo_num.groupby(train["_Group"]).transform("mean")
                    report_parts.append(f"  Group CryoSleep rate: mean={grp_cryo.mean():.3f}")
                    features_added.append("GroupCryoSleepRate")

        # ── Spending features ─────────────────────────────────────────────────
        if existing_spend:
            spend = train[existing_spend].apply(pd.to_numeric, errors="coerce").fillna(0)
            total = spend.sum(axis=1)
            nonzero = (total > 0)
            report_parts.append(f"\nSPENDING ANALYSIS:")
            report_parts.append(f"  Passengers with any spend: {nonzero.mean():.1%}")
            report_parts.append(f"  TotalSpend: mean={total.mean():.1f}, max={total.max():.1f}")
            for col in existing_spend:
                s = spend[col]
                report_parts.append(f"  {col}: zero_rate={( s==0).mean():.1%}, mean={s[s>0].mean():.1f}")
            features_added += [
                "TotalSpend", "AnySpend", "LogTotalSpend (log1p)",
                "SpendPerPerson (TotalSpend / GroupSize)",
            ] + [f"Log{c} (log1p)" for c in existing_spend]

        # ── Cabin features ────────────────────────────────────────────────────
        if has_cabin:
            cabin_split = train["Cabin"].astype(str).str.split("/", expand=True)
            decks = cabin_split[0].value_counts()
            report_parts.append(f"\nCABIN ANALYSIS:")
            report_parts.append(f"  Decks found: {decks.to_dict()}")
            if 1 in cabin_split.columns:
                nums = pd.to_numeric(cabin_split[1], errors="coerce")
                report_parts.append(f"  CabinNum range: [{nums.min():.0f}, {nums.max():.0f}]")
            features_added += ["Cabin_Deck", "Cabin_Num", "Cabin_Side", "CabinNum_bin (qcut 10)"]

        # ── Boolean features ──────────────────────────────────────────────────
        if bool_cols:
            report_parts.append(f"\nBOOLEAN COLUMNS (cast to int): {bool_cols}")
            features_added += [f"{c}_int" for c in bool_cols]

        # ── CryoSleep interactions ────────────────────────────────────────────
        if "CryoSleep" in train.columns and existing_spend:
            report_parts.append(f"\nCRYOSLEEP INTERACTION:")
            cryo = train["CryoSleep"].map({"True": 1, "False": 0, True: 1, False: 0}).fillna(0)
            spend_nonzero_during_cryo = (train[existing_spend[0]].fillna(0) > 0) & (cryo == 1)
            report_parts.append(f"  Spend while CryoSleep=True: {spend_nonzero_during_cryo.mean():.1%} (anomaly)")
            features_added.append("CryoSleep_TotalSpend_inconsistency (flag)")

        report_parts.append(f"\nFEATURES TO ADD ({len(features_added)} total):")
        for f in features_added:
            report_parts.append(f"  + {f}")

        report = "\n".join(report_parts)
        code   = self._build_code(has_cabin, has_pid, existing_spend, bool_cols)
        return report, code

    def _build_code(
        self,
        has_cabin: bool,
        has_pid: bool,
        existing_spend: list[str],
        bool_cols: list[str],
    ) -> str:
        """Return a complete engineer(df, group_stats=None) function."""
        spend_list = str(existing_spend)
        bool_list  = str(bool_cols)
        return f'''def engineer(df, group_stats=None):
    """Enhanced feature engineering — Feature Engineer Agent v2."""
    import numpy as np
    import pandas as pd

    out = pd.DataFrame(index=df.index)

    # ── Cabin split ───────────────────────────────────────────────────────────
    if {has_cabin} and "Cabin" in df.columns:
        cabin = df["Cabin"].astype(str).str.split("/", expand=True)
        out["Cabin_Deck"] = cabin[0].where(cabin[0] != "nan", other="__NA__")
        raw_num = pd.to_numeric(cabin.get(1), errors="coerce")
        out["Cabin_Num"]  = raw_num
        out["CabinNum_bin"] = pd.cut(raw_num, bins=10, labels=False).astype("float")
        out["Cabin_Side"] = (cabin[2].where(cabin[2] != "nan", other="__NA__")
                             if 2 in cabin.columns else "__NA__")

    # ── Group features ────────────────────────────────────────────────────────
    if {has_pid} and "PassengerId" in df.columns:
        pid = df["PassengerId"].astype(str).str.split("_", expand=True)
        out["Group"] = pd.to_numeric(pid[0], errors="coerce")
        grp_size = df["PassengerId"].astype(str).str.split("_").str[0].map(
            df["PassengerId"].astype(str).str.split("_").str[0].value_counts()
        )
        out["GroupSize"] = grp_size.values
        out["IsSolo"]    = (grp_size == 1).astype(int).values

    # ── Spending features ─────────────────────────────────────────────────────
    spend_cols = {spend_list}
    existing   = [c for c in spend_cols if c in df.columns]
    if existing:
        spend = df[existing].apply(pd.to_numeric, errors="coerce").fillna(0)
        total = spend.sum(axis=1)
        out["TotalSpend"]    = total
        out["AnySpend"]      = (total > 0).astype(int)
        out["LogTotalSpend"] = np.log1p(total)
        grp_size_safe = out.get("GroupSize", pd.Series(1, index=df.index)).replace(0, 1)
        out["SpendPerPerson"] = total / grp_size_safe
        for c in existing:
            out[c]              = spend[c]
            out[f"Log{{c}}"]    = np.log1p(spend[c])

    # ── Boolean columns ───────────────────────────────────────────────────────
    bool_cols = {bool_list}
    for col in bool_cols:
        if col in df.columns:
            out[f"{{col}}_int"] = df[col].map(
                {{"True": 1, "False": 0, True: 1, False: 0}}
            ).fillna(-1).astype(int)

    # ── CryoSleep interaction ─────────────────────────────────────────────────
    if "CryoSleep" in df.columns and existing:
        cryo = df["CryoSleep"].map({{"True": 1, "False": 0, True: 1, False: 0}}).fillna(0)
        out["CryoSpend_flag"] = ((cryo == 1) & (out.get("TotalSpend", pd.Series(0, index=df.index)) > 0)).astype(int)

    # ── Keep remaining raw columns ────────────────────────────────────────────
    skip = {{"Cabin", "PassengerId", "Name"}}
    for c in df.columns:
        if c not in skip and c not in out.columns:
            out[c] = df[c]

    return out


def compute_group_stats(train_df):
    """Compute group-level aggregates from training data only (no leakage)."""
    if "PassengerId" not in train_df.columns:
        return None
    grp = train_df["PassengerId"].astype(str).str.split("_").str[0]
    stats = train_df.copy()
    stats["_grp"] = grp
    return stats.groupby("_grp").size().rename("_grp_size")
'''
