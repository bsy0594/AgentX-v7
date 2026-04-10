"""
Ensemble Agent — Blends multiple submission CSVs with weighted voting.
For binary classification: weighted majority vote (stacking=2x, others=1x).
For regression: weighted average.
Input: text message containing SUBMISSION_PATHS and optional WEIGHTS.
Output: blended submission written to OUTPUT_PATH.
"""
import re
from pathlib import Path

import pandas as pd
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState
from a2a.utils import get_message_text, new_agent_text_message


class EnsembleAgent:
    async def run(self, message: Message, updater: TaskUpdater) -> None:
        payload = get_message_text(message) or ""

        # Parse paths and output path
        csv_paths_match = re.search(r"SUBMISSION_PATHS:\n(.*?)(?:\nOUTPUT_PATH:|\nWEIGHTS:|\Z)", payload, re.DOTALL)
        output_match    = re.search(r"OUTPUT_PATH:(.*?)(?:\n|\Z)", payload)
        weights_match   = re.search(r"WEIGHTS:(.*?)(?:\n|\Z)", payload)

        if not csv_paths_match:
            await updater.failed(new_agent_text_message("[Ensemble] No SUBMISSION_PATHS provided."))
            return

        paths_text  = csv_paths_match.group(1).strip()
        csv_paths   = [Path(p.strip()) for p in paths_text.splitlines() if p.strip()]
        output_path = Path(output_match.group(1).strip()) if output_match else csv_paths[0].parent / "ensemble.csv"

        # Parse weights (optional, comma-separated floats matching path count)
        weights = None
        if weights_match:
            try:
                weights = [float(w.strip()) for w in weights_match.group(1).split(",")]
                if len(weights) != len(csv_paths):
                    weights = None
            except ValueError:
                weights = None

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"[Ensemble] Blending {len(csv_paths)} submissions → {output_path.name}"),
        )

        try:
            result_path = self._blend(csv_paths, output_path, weights)
        except Exception as e:
            await updater.failed(new_agent_text_message(f"[Ensemble] Blending failed: {e}"))
            return

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"ENSEMBLE_PATH:{result_path}"),
        )

    # ── blending logic ────────────────────────────────────────────────────────

    def _blend(self, paths: list[Path], output: Path, weights: list[float] | None = None) -> str:
        dfs = []
        valid_weights = []
        for i, p in enumerate(paths):
            if not p.exists():
                print(f"[Ensemble] WARNING: {p} not found, skipping")
                continue
            dfs.append(pd.read_csv(p))
            w = weights[i] if weights and i < len(weights) else 1.0
            valid_weights.append(w)
            print(f"[Ensemble] Loaded {p.name}: {len(dfs[-1])} rows (weight={w})")

        if not dfs:
            raise ValueError("No valid submission files found")

        id_col     = dfs[0].columns[0]
        target_col = dfs[0].columns[1]
        ids        = dfs[0][id_col]

        # Detect if binary classification (handles True/False strings, bools, and 0/1 integers)
        unique_vals = set(str(v).strip().lower() for df in dfs for v in df[target_col].unique())
        BINARY_VALS = {"true", "false", "1", "0"}
        is_bool_labels = unique_vals <= BINARY_VALS and len(unique_vals) <= 4

        # Determine output format: use the first file's format as reference
        ref_vals = set(str(v).strip().lower() for v in dfs[0][target_col].unique())
        output_as_bool = ref_vals <= {"true", "false"}

        if is_bool_labels:
            # Weighted vote: normalize all formats to 0/1, accumulate, threshold at 0.5
            VOTE_MAP = {"true": 1, "false": 0, "1": 1, "0": 0, True: 1, False: 0, 1: 1, 0: 0}
            total_weight = sum(valid_weights)
            weighted_sum = None
            for df, w in zip(dfs, valid_weights):
                votes = df[target_col].map(
                    lambda v: VOTE_MAP.get(str(v).strip().lower(), VOTE_MAP.get(v, 0))
                ).fillna(0).reset_index(drop=True)
                if weighted_sum is None:
                    weighted_sum = votes * w
                else:
                    weighted_sum += votes * w

            # Majority: threshold at 0.5 * total_weight
            majority = (weighted_sum / total_weight) >= 0.5
            # Restore to original format (True/False or 1/0)
            blended = majority if output_as_bool else majority.astype(int)

            # Agreement stats
            if len(dfs) > 1:
                pred_matrix = pd.concat(
                    [df[target_col].map(lambda v: VOTE_MAP.get(str(v).strip().lower(), VOTE_MAP.get(v, 0)))
                     .reset_index(drop=True) for df in dfs], axis=1
                )
                unanimous = (pred_matrix.nunique(axis=1) == 1).mean()
                print(f"[Ensemble] Weighted vote unanimous agreement: {unanimous:.1%}")
        else:
            # Regression or numeric: weighted average
            weighted_sum = None
            total_weight = sum(valid_weights)
            for df, w in zip(dfs, valid_weights):
                vals = pd.to_numeric(df[target_col], errors="coerce").reset_index(drop=True) * w
                if weighted_sum is None:
                    weighted_sum = vals
                else:
                    weighted_sum += vals
            blended = weighted_sum / total_weight

        result = pd.DataFrame({id_col: ids, target_col: blended})
        output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output, index=False)
        print(f"[Ensemble] Saved {len(result)} rows → {output}")
        return str(output)
