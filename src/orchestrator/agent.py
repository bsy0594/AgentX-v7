"""
ML Orchestrator — Coordinates EDA, Critic, Model Selector, Error Handler, and Ensemble agents.

Flow:
  1. Extract competition data
  2. EDA Agent  → real data insights          (parallel with step 3)
  3. Model Selector Agent → CV best model     (parallel with step 2)
  4. Form plan (GPT-4o, with EDA report)
  5. Critic Agent → review plan
  6. Generate code (GPT-4o, with EDA + critique + best model)
  7. Run with self-repair loop (Error Handler Agent)
  8. Deterministic fallback (best model from CV)
  9. Ensemble Agent → blend GPT + fallback predictions
 10. Return best submission
"""
import asyncio
import base64
import io
import json
import re
import subprocess
import sys
import tarfile
import tempfile
import textwrap
from pathlib import Path
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    FilePart, FileWithBytes, Message, Part, Role, TaskState, TextPart,
    TaskStatusUpdateEvent,
)
from a2a.utils import get_message_text, new_agent_text_message
from openai import AsyncOpenAI

import os
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

CRITIC_URL        = "http://127.0.0.1:8001"
ERROR_HANDLER_URL = "http://127.0.0.1:8002"
EDA_URL           = "http://127.0.0.1:8003"
MODEL_SEL_URL     = "http://127.0.0.1:8005"
ENSEMBLE_URL      = "http://127.0.0.1:8006"
HP_TUNER_URL      = "http://127.0.0.1:8007"
FE_URL            = "http://127.0.0.1:8008"
THRESHOLD_URL     = "http://127.0.0.1:8009"
PLANNER_URL       = "http://127.0.0.1:8010"
CODE_GEN_URL      = "http://127.0.0.1:8011"
STACKING_URL      = "http://127.0.0.1:8012"

SYSTEM_PROMPT = """\
You are a Machine Learning Engineer with 10 years of experience in Kaggle competitions \
and production ML systems.

Your strengths:
- Deep expertise in feature engineering and preprocessing for tabular data
- Strong intuition for model selection (tree-based vs. linear vs. neural)
- Methodical approach: always inspect data before writing code
- Awareness of common pitfalls: data leakage, class imbalance, metric mismatch

When given a task you:
1. Carefully read the competition instructions and evaluation metric
2. Inspect the data schema and sample rows
3. Form a clear written plan BEFORE writing any code
4. Write clean, well-structured code that follows your plan
5. Validate your output against the sample submission format
"""

# Available libraries (installed in this conda env)
CODING_RULES = """\
MANDATORY: Use sklearn Pipeline + ColumnTransformer for ALL preprocessing.
This avoids pandas Copy-on-Write errors entirely. Never modify DataFrames in-place.

AVAILABLE LIBRARIES: pandas, numpy, sklearn, xgboost, lightgbm, scipy

REQUIRED CODE STRUCTURE (follow exactly, replace placeholders):
from pathlib import Path
import pandas as pd, numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier  # or LGBMClassifier / GradientBoostingClassifier

data_dir        = Path("DATA_DIR_PLACEHOLDER")
submission_path = Path("SUBMISSION_PATH_PLACEHOLDER")

# 1. Load
train_path = next(data_dir.rglob("train.csv"))
test_path  = next(data_dir.rglob("test.csv"))
sample     = pd.read_csv(next(data_dir.rglob("sample_submission.csv")))
id_col, target_col = sample.columns[0], sample.columns[1]
train = pd.read_csv(train_path)
test  = pd.read_csv(test_path)
ids   = test[id_col].reset_index(drop=True)
y     = train[target_col].copy()

# 2. Feature engineering — ENHANCED (use exactly as written)
def engineer(df):
    out = pd.DataFrame(index=df.index)
    cabin_deck = None
    if "Cabin" in df.columns:
        cabin = df["Cabin"].astype(str).str.split("/", expand=True)
        cabin_deck = cabin[0].where(cabin[0] != "nan", other="__NA__")
        out["Cabin_Deck"] = cabin_deck
        raw_num = pd.to_numeric(cabin.get(1), errors="coerce")
        out["Cabin_Num"]    = raw_num
        out["CabinNum_bin"] = pd.cut(raw_num, bins=10, labels=False).astype("float")
        out["Cabin_Side"]   = cabin[2].where(cabin[2] != "nan", other="__NA__") if 2 in cabin.columns else "__NA__"
    if "PassengerId" in df.columns:
        pid = df["PassengerId"].astype(str).str.split("_", expand=True)
        out["Group"] = pd.to_numeric(pid[0], errors="coerce")
        grp_series = df["PassengerId"].astype(str).str.split("_").str[0]
        grp_size   = grp_series.map(grp_series.value_counts())
        out["GroupSize"] = grp_size.values
        out["IsSolo"]    = (grp_size == 1).astype(int).values
    spend_cols = ["RoomService","FoodCourt","ShoppingMall","Spa","VRDeck"]
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
            out[c] = spend[c]
            out[f"Log{c}"] = np.log1p(spend[c])
    for col in df.columns:
        uniq = set(str(v).lower() for v in df[col].dropna().unique())
        if uniq <= {"true", "false"}:
            out[f"{col}_int"] = df[col].map({"True":1,"False":0,True:1,False:0}).fillna(-1).astype(int)
    cryo_int = None
    if "CryoSleep" in df.columns:
        cryo_int = df["CryoSleep"].map({"True":1,"False":0,True:1,False:0}).fillna(0)
        if existing:
            out["CryoSpend_flag"] = ((cryo_int==1)&(out.get("TotalSpend",pd.Series(0,index=df.index))>0)).astype(int)
    if "Age" in df.columns:
        age = pd.to_numeric(df["Age"], errors="coerce")
        out["Age_bin"] = pd.cut(age, bins=[0,12,18,35,60,200], labels=False).astype("float")
        if cryo_int is not None:
            out["Age_x_Cryo"] = age.fillna(age.median()) * cryo_int
        if existing:
            total_spend = out.get("TotalSpend", pd.Series(0, index=df.index))
            out["Age_x_Spend"] = age.fillna(age.median()) * np.log1p(total_spend)
    if cabin_deck is not None and "Destination" in df.columns:
        out["Deck_x_Dest"] = cabin_deck.astype(str) + "_" + df["Destination"].astype(str).fillna("__NA__")
    if "HomePlanet" in df.columns and "Destination" in df.columns:
        out["Home_x_Dest"] = df["HomePlanet"].astype(str).fillna("__NA__") + "_" + df["Destination"].astype(str).fillna("__NA__")
    skip = {"Cabin", "PassengerId", "Name"}
    for c in df.columns:
        if c not in skip and c not in out.columns:
            out[c] = df[c]
    return out

_train_feat = train.drop(columns=[c for c in [id_col, target_col] if c in train.columns])
_test_feat  = test.drop(columns=[c for c in [id_col, target_col] if c in test.columns])
X_tr_raw = engineer(_train_feat)
X_te_raw = engineer(_test_feat)

num_cols = X_tr_raw.select_dtypes(include="number").columns.tolist()
cat_cols = [c for c in X_tr_raw.columns if c not in num_cols]
X_tr = X_tr_raw.assign(**{c: X_tr_raw[c].astype(str) for c in cat_cols})
X_te = X_te_raw.assign(**{c: X_te_raw[c].astype(str) for c in cat_cols})

# 3. Pipeline (no in-place modifications)
num_pipe = Pipeline([("imp", SimpleImputer(strategy="median"))])
cat_pipe = Pipeline([
    ("imp", SimpleImputer(strategy="constant", fill_value="__NA__")),
    ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
])
transformers = []
if num_cols: transformers.append(("num", num_pipe, num_cols))
if cat_cols: transformers.append(("cat", cat_pipe, cat_cols))
pre = ColumnTransformer(transformers=transformers, remainder="drop")

# 4. Model — use XGBoost or LightGBM for best results
pipe = Pipeline([("pre", pre), ("model", XGBClassifier(...))])
pipe.fit(X_tr, y)

# 5. Predict & save
preds = pipe.predict(X_te)
pd.DataFrame({id_col: ids, target_col: preds}).to_csv(submission_path, index=False)
"""


class MLAgent:
    def __init__(self):
        self.client  = AsyncOpenAI(api_key=OPENAI_API_KEY)  # fallback only
        self.workdir: Path | None = None

    # ── main flow ─────────────────────────────────────────────────────────────

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        await updater.update_status(
            TaskState.working,
            new_agent_text_message("[MLE] Received task. Extracting competition data..."),
        )

        tar_bytes = self._extract_tar(message)
        if not tar_bytes:
            await updater.failed(new_agent_text_message("[MLE] No competition data found."))
            return

        instructions  = get_message_text(message) or ""
        self.workdir  = Path(tempfile.mkdtemp(prefix="ml_agent_"))
        data_dir      = self.workdir / "data"
        data_dir.mkdir()

        await asyncio.to_thread(self._extract_tar_to_dir, tar_bytes, data_dir)

        await updater.update_status(
            TaskState.working,
            new_agent_text_message("[MLE] Data extracted. Running EDA + Model Selection in parallel..."),
        )

        csv_files    = list(data_dir.rglob("*.csv"))
        data_summary = await asyncio.to_thread(self._summarize_data, csv_files)
        submission_path      = self.workdir / "submission.csv"
        submission_path_xgb  = self.workdir / "submission_xgb.csv"
        ensemble_path        = self.workdir / "submission_ensemble.csv"

        # ── Step 1: EDA + Model Selection + FE + Stacking in parallel ───────────
        eda_task      = asyncio.create_task(self._call_eda_agent(tar_bytes))
        model_task    = asyncio.create_task(self._call_model_selector(tar_bytes))
        fe_task       = asyncio.create_task(self._call_feature_engineer(tar_bytes))
        stacking_path = self.workdir / "submission_stacking.csv"
        stacking_task = asyncio.create_task(
            self._call_stacking_agent(tar_bytes, stacking_path)
        )

        # ── Step 2: Plan (GPT, while agents run) ──────────────────────────────
        await updater.update_status(TaskState.working, new_agent_text_message("[MLE] Forming solution plan..."))
        plan = await self._form_plan(instructions, data_summary)
        await updater.update_status(TaskState.working, new_agent_text_message(f"[MLE] Plan:\n\n{plan}"))

        # ── Step 3: Wait for EDA + FE reports ────────────────────────────────
        eda_report = await eda_task
        fe_report, fe_code = await fe_task
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"[EDA] Report received ({len(eda_report)} chars)"),
        )
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"[FE] Feature engineer report received ({len(fe_report)} chars)"),
        )

        # ── Step 4: Critic ────────────────────────────────────────────────────
        await updater.update_status(TaskState.working, new_agent_text_message("[MLE] Consulting Critic Agent..."))
        critique = await self._consult_critic(plan, instructions, data_summary, eda_report)
        await updater.update_status(TaskState.working, new_agent_text_message(f"[Critic] Feedback:\n\n{critique}"))

        # ── Step 5: Generate code (with FE code injected) ─────────────────────
        await updater.update_status(TaskState.working, new_agent_text_message("[MLE] Writing solution code..."))
        code = await self._generate_code(instructions, data_summary, plan, critique, eda_report, fe_code, data_dir, submission_path)

        # ── Step 6: Execute with self-repair ──────────────────────────────────
        success, output, code = await self._run_with_repair(code, submission_path, updater, max_repairs=2)

        # ── Step 7: Wait for model selector result ────────────────────────────
        model_selection = await model_task
        best_model = model_selection.get("best_model", "XGBoost")
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"[ModelSelector] {best_model} "
                                   f"CV={model_selection.get('best_cv_mean','?')}"),
        )

        # ── Step 8: Hyperparameter tuning ─────────────────────────────────────
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"[MLE] Tuning {best_model} hyperparameters with Optuna..."),
        )
        tuned_params = await self._call_hyperparameter_tuner(tar_bytes, best_model)
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"[Tuner] Best CV: {tuned_params.get('best_cv_accuracy','?')} | params tuned"),
        )

        # ── Step 9: Deterministic fallback with tuned params ──────────────────
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"[MLE] Running {best_model} fallback with tuned params..."),
        )
        fallback_code = self._build_fallback(data_dir, submission_path_xgb, best_model, tuned_params)
        fb_success, fb_output = await asyncio.to_thread(self._run_code, fallback_code)
        print(f"[MLE] Fallback ({best_model}) output:\n{fb_output[:500]}")

        # ── Step 10: Threshold optimization on fallback ───────────────────────
        submission_path_thresh = self.workdir / "submission_thresh.csv"
        thresh_path = None
        if fb_success:
            await updater.update_status(
                TaskState.working,
                new_agent_text_message("[MLE] Optimizing classification threshold..."),
            )
            thresh_path = await self._call_threshold_optimizer(tar_bytes, tuned_params, submission_path_thresh)
            if thresh_path:
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(f"[Thresh] Threshold-optimized submission ready"),
                )

        # ── Step 11: Wait for stacking result ─────────────────────────────────
        stacking_result = await stacking_task
        if stacking_result:
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"[Stacking] Ensemble submission ready: {Path(stacking_result).name}"),
            )

        if not success and not fb_success and not stacking_result:
            await updater.failed(new_agent_text_message(
                f"[MLE] All strategies failed.\nGPT error: {output[:300]}\nFallback error: {fb_output[:300]}"
            ))
            return

        # ── Step 12: Ensemble all successful submissions ───────────────────────
        # Priority weights: stacking (best) > threshold > single model > GPT
        ensemble_inputs = []
        ensemble_weights = []

        # Stacking is our best deterministic result — highest weight
        if stacking_result and Path(stacking_result).exists():
            ensemble_inputs.append(Path(stacking_result))
            ensemble_weights.append(2.0)

        # Threshold-optimized fallback
        if thresh_path and Path(thresh_path).exists():
            ensemble_inputs.append(Path(thresh_path))
            ensemble_weights.append(1.5)
        elif fb_success and submission_path_xgb.exists():
            ensemble_inputs.append(submission_path_xgb)
            ensemble_weights.append(1.0)

        # LLM-generated code (lower weight — may be noisier)
        if success and submission_path.exists():
            ensemble_inputs.append(submission_path)
            ensemble_weights.append(1.0)

        if len(ensemble_inputs) > 1:
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(
                    f"[MLE] Ensembling {len(ensemble_inputs)} submissions "
                    f"(weights={ensemble_weights})..."
                ),
            )
            ens_path = await self._call_ensemble_agent(
                ensemble_inputs, ensemble_path, weights=ensemble_weights
            )
            if ens_path and Path(ens_path).exists():
                final_path = Path(ens_path)
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(f"[Ensemble] Blended submission ready: {final_path.name}"),
                )
            else:
                # Fallback to best single
                final_path = ensemble_inputs[0]
        elif ensemble_inputs:
            final_path = ensemble_inputs[0]
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"[MLE] Using single best submission: {final_path.name}"),
            )
        else:
            final_path = submission_path  # last resort

        # ── Step 10: Upload artifact ──────────────────────────────────────────
        if not final_path.exists():
            await updater.failed(new_agent_text_message("[MLE] No submission file produced."))
            return

        submission_bytes = final_path.read_bytes()
        row_count        = submission_bytes.count(b"\n")
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"[MLE] Submission ready ({row_count} rows). Uploading..."),
        )
        await updater.add_artifact(
            parts=[Part(root=FilePart(file=FileWithBytes(
                bytes=base64.b64encode(submission_bytes).decode("ascii"),
                name="submission.csv",
                mime_type="text/csv",
            )))],
            name="submission",
        )

    # ── EDA Agent call ────────────────────────────────────────────────────────

    async def _call_eda_agent(self, tar_bytes: bytes) -> str:
        try:
            async with httpx.AsyncClient(timeout=120) as http:
                resolver = A2ACardResolver(httpx_client=http, base_url=EDA_URL)
                card     = await resolver.get_agent_card()
                client   = ClientFactory(ClientConfig(httpx_client=http, streaming=True)).create(card)
                msg = Message(
                    kind="message", role=Role.user,
                    parts=[Part(root=FilePart(file=FileWithBytes(
                        bytes=base64.b64encode(tar_bytes).decode("ascii"),
                        name="data.tar.gz", mime_type="application/gzip",
                    )))],
                    message_id=uuid4().hex,
                )
                async for event in client.send_message(msg):
                    match event:
                        case (_, TaskStatusUpdateEvent() as upd) if upd.status.message:
                            text = get_message_text(upd.status.message) or ""
                            if "EDA_REPORT_START" in text:
                                start = text.index("EDA_REPORT_START") + len("EDA_REPORT_START\n")
                                end   = text.index("EDA_REPORT_END")
                                return text[start:end].strip()
        except Exception as e:
            print(f"[MLE] EDA Agent unreachable: {e}")
        return "(EDA unavailable — proceeding without report)"

    # ── Feature Engineer call ─────────────────────────────────────────────────

    async def _call_feature_engineer(self, tar_bytes: bytes) -> tuple[str, str]:
        try:
            async with httpx.AsyncClient(timeout=120) as http:
                resolver = A2ACardResolver(httpx_client=http, base_url=FE_URL)
                card     = await resolver.get_agent_card()
                client   = ClientFactory(ClientConfig(httpx_client=http, streaming=True)).create(card)
                msg = Message(
                    kind="message", role=Role.user,
                    parts=[Part(root=FilePart(file=FileWithBytes(
                        bytes=base64.b64encode(tar_bytes).decode("ascii"),
                        name="data.tar.gz", mime_type="application/gzip",
                    )))],
                    message_id=uuid4().hex,
                )
                async for event in client.send_message(msg):
                    match event:
                        case (_, TaskStatusUpdateEvent() as upd) if upd.status.message:
                            text = get_message_text(upd.status.message) or ""
                            if "FE_REPORT_START" in text and "FE_CODE_START" in text:
                                try:
                                    report_start = text.index("FE_REPORT_START") + len("FE_REPORT_START\n")
                                    report_end   = text.index("FE_REPORT_END")
                                    code_start   = text.index("FE_CODE_START") + len("FE_CODE_START\n")
                                    code_end     = text.index("FE_CODE_END")
                                    return text[report_start:report_end].strip(), text[code_start:code_end].strip()
                                except ValueError:
                                    pass
        except Exception as e:
            print(f"[MLE] Feature Engineer unreachable: {e}")
        return "(FE unavailable)", ""

    # ── Hyperparameter Tuner call ─────────────────────────────────────────────

    async def _call_hyperparameter_tuner(self, tar_bytes: bytes, model_name: str) -> dict:
        try:
            async with httpx.AsyncClient(timeout=600) as http:
                resolver = A2ACardResolver(httpx_client=http, base_url=HP_TUNER_URL)
                card     = await resolver.get_agent_card()
                client   = ClientFactory(ClientConfig(httpx_client=http, streaming=True)).create(card)
                msg = Message(
                    kind="message", role=Role.user,
                    parts=[
                        Part(root=FilePart(file=FileWithBytes(
                            bytes=base64.b64encode(tar_bytes).decode("ascii"),
                            name="data.tar.gz", mime_type="application/gzip",
                        ))),
                        Part(root=TextPart(text=f"MODEL:{model_name}")),
                    ],
                    message_id=uuid4().hex,
                )
                async for event in client.send_message(msg):
                    match event:
                        case (_, TaskStatusUpdateEvent() as upd) if upd.status.message:
                            text = get_message_text(upd.status.message) or ""
                            if "TUNED_PARAMS_START" in text:
                                start = text.index("TUNED_PARAMS_START") + len("TUNED_PARAMS_START\n")
                                end   = text.index("TUNED_PARAMS_END")
                                return json.loads(text[start:end].strip())
        except Exception as e:
            print(f"[MLE] Hyperparameter Tuner unreachable: {e}")
        return {"model": model_name, "params": {}, "best_cv_accuracy": None}

    # ── Threshold Optimizer call ──────────────────────────────────────────────

    async def _call_threshold_optimizer(self, tar_bytes: bytes, tuned_params: dict, output_path: Path) -> str | None:
        try:
            async with httpx.AsyncClient(timeout=600) as http:
                resolver = A2ACardResolver(httpx_client=http, base_url=THRESHOLD_URL)
                card     = await resolver.get_agent_card()
                client   = ClientFactory(ClientConfig(httpx_client=http, streaming=True)).create(card)
                payload = f"TUNED_PARAMS:{json.dumps(tuned_params)}\nOUTPUT_PATH:{output_path.as_posix()}"
                msg = Message(
                    kind="message", role=Role.user,
                    parts=[
                        Part(root=FilePart(file=FileWithBytes(
                            bytes=base64.b64encode(tar_bytes).decode("ascii"),
                            name="data.tar.gz", mime_type="application/gzip",
                        ))),
                        Part(root=TextPart(text=payload)),
                    ],
                    message_id=uuid4().hex,
                )
                async for event in client.send_message(msg):
                    match event:
                        case (_, TaskStatusUpdateEvent() as upd) if upd.status.message:
                            text = get_message_text(upd.status.message) or ""
                            if "THRESHOLD_PATH:" in text:
                                return text.split("THRESHOLD_PATH:")[1].strip()
        except Exception as e:
            print(f"[MLE] Threshold Optimizer unreachable: {e}")
        return None

    # ── Model Selector call ───────────────────────────────────────────────────

    async def _call_model_selector(self, tar_bytes: bytes) -> dict:
        try:
            async with httpx.AsyncClient(timeout=600) as http:
                resolver = A2ACardResolver(httpx_client=http, base_url=MODEL_SEL_URL)
                card     = await resolver.get_agent_card()
                client   = ClientFactory(ClientConfig(httpx_client=http, streaming=True)).create(card)
                msg = Message(
                    kind="message", role=Role.user,
                    parts=[Part(root=FilePart(file=FileWithBytes(
                        bytes=base64.b64encode(tar_bytes).decode("ascii"),
                        name="data.tar.gz", mime_type="application/gzip",
                    )))],
                    message_id=uuid4().hex,
                )
                async for event in client.send_message(msg):
                    match event:
                        case (_, TaskStatusUpdateEvent() as upd) if upd.status.message:
                            text = get_message_text(upd.status.message) or ""
                            if "MODEL_SELECTION_START" in text:
                                start = text.index("MODEL_SELECTION_START") + len("MODEL_SELECTION_START\n")
                                end   = text.index("MODEL_SELECTION_END")
                                return json.loads(text[start:end].strip())
        except Exception as e:
            print(f"[MLE] Model Selector unreachable: {e}")
        return {"best_model": "XGBoost", "best_cv_mean": None, "all_scores": {}}

    # ── Stacking Agent call ───────────────────────────────────────────────────

    async def _call_stacking_agent(self, tar_bytes: bytes, output_path: Path) -> str | None:
        try:
            async with httpx.AsyncClient(timeout=900) as http:
                resolver = A2ACardResolver(httpx_client=http, base_url=STACKING_URL)
                card     = await resolver.get_agent_card()
                client   = ClientFactory(ClientConfig(httpx_client=http, streaming=True)).create(card)
                msg = Message(
                    kind="message", role=Role.user,
                    parts=[
                        Part(root=FilePart(file=FileWithBytes(
                            bytes=base64.b64encode(tar_bytes).decode("ascii"),
                            name="data.tar.gz", mime_type="application/gzip",
                        ))),
                        Part(root=TextPart(text=f"OUTPUT_PATH:{output_path.as_posix()}")),
                    ],
                    message_id=uuid4().hex,
                )
                async for event in client.send_message(msg):
                    match event:
                        case (_, TaskStatusUpdateEvent() as upd) if upd.status.message:
                            text = get_message_text(upd.status.message) or ""
                            if "STACKING_PATH:" in text:
                                return text.split("STACKING_PATH:")[1].strip()
        except Exception as e:
            print(f"[MLE] Stacking Agent unreachable: {e}")
        return None

    # ── Ensemble Agent call ───────────────────────────────────────────────────

    async def _call_ensemble_agent(
        self,
        csv_paths: list[Path],
        output: Path,
        weights: list[float] | None = None,
    ) -> str | None:
        weights_line = ""
        if weights:
            weights_line = f"\nWEIGHTS:{','.join(str(w) for w in weights)}"
        payload = (
            "SUBMISSION_PATHS:\n"
            + "\n".join(str(p) for p in csv_paths)
            + f"\nOUTPUT_PATH:{output}"
            + weights_line
        )
        try:
            async with httpx.AsyncClient(timeout=60) as http:
                resolver = A2ACardResolver(httpx_client=http, base_url=ENSEMBLE_URL)
                card     = await resolver.get_agent_card()
                client   = ClientFactory(ClientConfig(httpx_client=http, streaming=True)).create(card)
                msg = Message(
                    kind="message", role=Role.user,
                    parts=[Part(TextPart(text=payload))],
                    message_id=uuid4().hex,
                )
                async for event in client.send_message(msg):
                    match event:
                        case (_, TaskStatusUpdateEvent() as upd) if upd.status.message:
                            text = get_message_text(upd.status.message) or ""
                            if "ENSEMBLE_PATH:" in text:
                                return text.split("ENSEMBLE_PATH:")[1].strip()
        except Exception as e:
            print(f"[MLE] Ensemble Agent unreachable: {e}")
        return None

    # ── Repair loop ───────────────────────────────────────────────────────────

    async def _run_with_repair(
        self,
        code: str,
        submission_path: Path,
        updater: TaskUpdater,
        max_repairs: int = 2,
    ) -> tuple[bool, str, str]:
        for attempt in range(1, max_repairs + 2):
            label = "Running solution..." if attempt == 1 else f"Repair attempt {attempt - 1}..."
            await updater.update_status(TaskState.working, new_agent_text_message(f"[MLE] {label}"))

            success, output = await asyncio.to_thread(self._run_code, code)
            print(f"[MLE] Attempt {attempt}:\n{output[:600]}")

            if success and submission_path.exists():
                return True, output, code

            if attempt <= max_repairs:
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(
                        f"[MLE] Attempt {attempt} failed. Delegating to Error Handler...\n"
                        f"Error snippet: {output[:300]}"
                    ),
                )
                code = await self._fix_code(code, output)

        return False, output, code

    async def _fix_code(self, code: str, error: str) -> str:
        payload = f"CODE:\n{code}\n\nERROR:\n{error[:2000]}"
        try:
            async with httpx.AsyncClient(timeout=90) as http:
                resolver = A2ACardResolver(httpx_client=http, base_url=ERROR_HANDLER_URL)
                card     = await resolver.get_agent_card()
                handler  = ClientFactory(ClientConfig(httpx_client=http, streaming=True)).create(card)
                msg = Message(
                    kind="message", role=Role.user,
                    parts=[Part(TextPart(text=payload))],
                    message_id=uuid4().hex,
                )
                async for event in handler.send_message(msg):
                    match event:
                        case (_, TaskStatusUpdateEvent() as upd) if upd.status.message:
                            text = get_message_text(upd.status.message) or ""
                            if "FIXED_CODE_START" in text:
                                start = text.index("FIXED_CODE_START") + len("FIXED_CODE_START\n")
                                end   = text.index("FIXED_CODE_END")
                                return text[start:end].strip()
        except Exception as e:
            print(f"[MLE] Error Handler unreachable: {e}")

        # Self-repair fallback (direct OpenAI call — Error Handler unreachable)
        try:
            resp = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": (
                        f"Fix this failing ML code.\n\nFAILING CODE:\n{code}\n\n"
                        f"ERROR:\n{error[:1500]}\n\n{CODING_RULES}\n\n"
                        f"Return ONLY corrected Python code. No markdown."
                    )},
                ],
                temperature=0.05, max_tokens=3500,
            )
            fixed = resp.choices[0].message.content.strip()
            if fixed.startswith("```"):
                lines = fixed.splitlines()
                fixed = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            return fixed
        except Exception as e:
            print(f"[MLE] GPT fix fallback failed: {e}")
            return code

    # ── Planning ──────────────────────────────────────────────────────────────

    async def _form_plan(self, instructions: str, data_summary: str) -> str:
        payload = (
            f"INSTRUCTIONS:\n{instructions[:2000]}\n\n"
            f"DATA SUMMARY:\n{data_summary[:3000]}"
        )
        try:
            async with httpx.AsyncClient(timeout=120) as http:
                resolver = A2ACardResolver(httpx_client=http, base_url=PLANNER_URL)
                card     = await resolver.get_agent_card()
                client   = ClientFactory(ClientConfig(httpx_client=http, streaming=True)).create(card)
                msg = Message(
                    kind="message", role=Role.user,
                    parts=[Part(TextPart(text=payload))],
                    message_id=uuid4().hex,
                )
                async for event in client.send_message(msg):
                    match event:
                        case (_, TaskStatusUpdateEvent() as upd) if upd.status.message:
                            text = get_message_text(upd.status.message) or ""
                            if "PLAN_START" in text:
                                start = text.index("PLAN_START") + len("PLAN_START\n")
                                end   = text.index("PLAN_END")
                                return text[start:end].strip()
        except Exception as e:
            print(f"[MLE] Planner unreachable: {e} — using fallback")
        # Fallback: direct OpenAI call
        try:
            resp = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": payload},
                ],
                temperature=0.3, max_tokens=700,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"[MLE] GPT plan fallback failed: {e}")
            return "Use a simple LightGBM classifier with default preprocessing."

    # ── Critic ────────────────────────────────────────────────────────────────

    async def _consult_critic(self, plan: str, instructions: str, data_summary: str, eda_report: str) -> str:
        payload = (
            f"COMPETITION INSTRUCTIONS:\n{instructions[:1000]}\n\n"
            f"DATA SUMMARY:\n{data_summary[:1500]}\n\n"
            f"EDA REPORT:\n{eda_report[:2000]}\n\n"
            f"ML ENGINEER'S PLAN:\n{plan}"
        )
        try:
            async with httpx.AsyncClient(timeout=60) as http:
                resolver = A2ACardResolver(httpx_client=http, base_url=CRITIC_URL)
                card     = await resolver.get_agent_card()
                critic   = ClientFactory(ClientConfig(httpx_client=http, streaming=True)).create(card)
                msg = Message(
                    kind="message", role=Role.user,
                    parts=[Part(TextPart(text=payload))],
                    message_id=uuid4().hex,
                )
                parts = []
                async for event in critic.send_message(msg):
                    match event:
                        case (_, TaskStatusUpdateEvent() as upd) if upd.status.message:
                            text = get_message_text(upd.status.message) or ""
                            if text.strip() and "Reviewing" not in text:
                                parts.append(text)
                return "\n".join(parts) or "No critique returned."
        except Exception as e:
            print(f"[MLE] Critic unreachable: {e}")
            return f"Critic unavailable. Proceeding with plan as-is."

    # ── Code generation ───────────────────────────────────────────────────────

    async def _generate_code(
        self,
        instructions: str,
        data_summary: str,
        plan: str,
        critique: str,
        eda_report: str,
        fe_code: str,
        data_dir: Path,
        submission_path: Path,
    ) -> str:
        data_dir_str = data_dir.as_posix()
        sub_path_str = submission_path.as_posix()
        rules = (
            CODING_RULES
            .replace("DATA_DIR_PLACEHOLDER", data_dir_str)
            .replace("SUBMISSION_PATH_PLACEHOLDER", sub_path_str)
        )
        # Inject paths as an explicit constraint — GPT must not override these
        path_constraint = (
            f"CRITICAL PATH CONSTRAINT — copy these EXACTLY, do NOT change them:\n"
            f'  data_dir        = Path("{data_dir_str}")\n'
            f'  submission_path = Path("{sub_path_str}")\n'
            f"This is a Windows machine. These are the real paths. Do NOT use /home or /tmp.\n"
        )
        fe_section = (
            f"FEATURE ENGINEERING CODE (use this engineer() function exactly — do NOT simplify it):\n"
            f"{fe_code[:3000]}\n\n"
        ) if fe_code else ""
        codegen_payload = (
            f"{path_constraint}\n"
            f"Write a complete Python ML solution.\n\n"
            f"INSTRUCTIONS:\n{instructions[:1500]}\n\n"
            f"DATA SUMMARY:\n{data_summary[:1500]}\n\n"
            f"EDA REPORT:\n{eda_report[:2000]}\n\n"
            f"{fe_section}"
            f"YOUR PLAN:\n{plan}\n\n"
            f"CRITIC FEEDBACK (address every issue):\n{critique}\n\n"
            f"{rules}\n\n"
            f"Output ONLY raw Python code. No markdown fences. No explanations."
        )
        code = None
        try:
            async with httpx.AsyncClient(timeout=180) as http:
                resolver = A2ACardResolver(httpx_client=http, base_url=CODE_GEN_URL)
                card     = await resolver.get_agent_card()
                client   = ClientFactory(ClientConfig(httpx_client=http, streaming=True)).create(card)
                msg = Message(
                    kind="message", role=Role.user,
                    parts=[Part(TextPart(text=codegen_payload))],
                    message_id=uuid4().hex,
                )
                async for event in client.send_message(msg):
                    match event:
                        case (_, TaskStatusUpdateEvent() as upd) if upd.status.message:
                            text = get_message_text(upd.status.message) or ""
                            if "CODE_START" in text:
                                start = text.index("CODE_START") + len("CODE_START\n")
                                end   = text.index("CODE_END")
                                code  = text[start:end].strip()
        except Exception as e:
            print(f"[MLE] Code Generator unreachable: {e} — using fallback")

        if not code:
            # Fallback: direct OpenAI call
            try:
                resp = await self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": codegen_payload},
                    ],
                    temperature=0.1, max_tokens=4000,
                )
                code = resp.choices[0].message.content.strip()
            except Exception as e:
                print(f"[MLE] GPT code gen fallback failed: {e}")
                code = ""

        if code.startswith("```"):
            lines = code.splitlines()
            code  = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        if code.startswith("```"):
            lines = code.splitlines()
            code  = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        return code

    # ── Deterministic fallback ────────────────────────────────────────────────

    def _build_fallback(
        self,
        data_dir: Path,
        submission_path: Path,
        best_model: str,
        tuned_params: dict,
    ) -> str:
        d = data_dir.as_posix()
        s = submission_path.as_posix()

        params = tuned_params.get("params", {})

        # Build model instantiation using tuned params (fall back to good defaults)
        if best_model == "XGBoost":
            model_import = "from xgboost import XGBClassifier"
            p = {
                "n_estimators": params.get("n_estimators", 600),
                "max_depth": params.get("max_depth", 6),
                "learning_rate": params.get("learning_rate", 0.03),
                "subsample": params.get("subsample", 0.8),
                "colsample_bytree": params.get("colsample_bytree", 0.7),
                "min_child_weight": params.get("min_child_weight", 3),
                "gamma": params.get("gamma", 0.1),
                "reg_alpha": params.get("reg_alpha", 0.1),
                "reg_lambda": params.get("reg_lambda", 1.0),
            }
            model_line = (f"XGBClassifier(n_estimators={p['n_estimators']}, max_depth={p['max_depth']}, "
                          f"learning_rate={p['learning_rate']:.4f}, subsample={p['subsample']:.2f}, "
                          f"colsample_bytree={p['colsample_bytree']:.2f}, min_child_weight={p['min_child_weight']}, "
                          f"gamma={p['gamma']:.3f}, reg_alpha={p['reg_alpha']:.3f}, reg_lambda={p['reg_lambda']:.3f}, "
                          f"random_state=42, eval_metric='logloss', verbosity=0)")
        elif best_model == "LightGBM":
            model_import = "from lightgbm import LGBMClassifier"
            p = {
                "n_estimators": params.get("n_estimators", 600),
                "max_depth": params.get("max_depth", 6),
                "learning_rate": params.get("learning_rate", 0.03),
                "subsample": params.get("subsample", 0.8),
                "colsample_bytree": params.get("colsample_bytree", 0.7),
                "min_child_samples": params.get("min_child_samples", 20),
                "num_leaves": params.get("num_leaves", 63),
                "reg_alpha": params.get("reg_alpha", 0.1),
                "reg_lambda": params.get("reg_lambda", 0.5),
            }
            model_line = (f"LGBMClassifier(n_estimators={p['n_estimators']}, max_depth={p['max_depth']}, "
                          f"learning_rate={p['learning_rate']:.4f}, subsample={p['subsample']:.2f}, "
                          f"colsample_bytree={p['colsample_bytree']:.2f}, min_child_samples={p['min_child_samples']}, "
                          f"num_leaves={p['num_leaves']}, reg_alpha={p['reg_alpha']:.3f}, "
                          f"reg_lambda={p['reg_lambda']:.3f}, random_state=42, verbose=-1)")
        elif best_model == "GradientBoosting":
            model_import = "from sklearn.ensemble import GradientBoostingClassifier"
            p = {
                "n_estimators": params.get("n_estimators", 400),
                "max_depth": params.get("max_depth", 5),
                "learning_rate": params.get("learning_rate", 0.05),
                "subsample": params.get("subsample", 0.8),
            }
            model_line = (f"GradientBoostingClassifier(n_estimators={p['n_estimators']}, "
                          f"max_depth={p['max_depth']}, learning_rate={p['learning_rate']:.4f}, "
                          f"subsample={p['subsample']:.2f}, random_state=42)")
        else:  # RandomForest
            model_import = "from sklearn.ensemble import RandomForestClassifier"
            p = {
                "n_estimators": params.get("n_estimators", 500),
                "min_samples_leaf": params.get("min_samples_leaf", 2),
            }
            model_line = (f"RandomForestClassifier(n_estimators={p['n_estimators']}, "
                          f"min_samples_leaf={p['min_samples_leaf']}, random_state=42, n_jobs=-1)")

        cv_info = f"# Tuned: {best_model} (CV={tuned_params.get('best_cv_accuracy','?')})"

        return textwrap.dedent(f"""
import pandas as pd
import numpy as np
from pathlib import Path
{model_import}
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
{cv_info}

data_dir        = Path("{d}")
submission_path = Path("{s}")

train = pd.read_csv(next(data_dir.rglob("train.csv")))
test  = pd.read_csv(next(data_dir.rglob("test.csv")))
sample_paths = list(data_dir.rglob("sample_submission.csv"))
if sample_paths:
    sample     = pd.read_csv(sample_paths[0])
    id_col     = sample.columns[0]
    target_col = sample.columns[1]
else:
    id_col     = train.columns[0]
    target_col = train.columns[-1]

print(f"id={{id_col}}  target={{target_col}}  train={{train.shape}}  test={{test.shape}}")
ids = test[id_col].reset_index(drop=True)
y   = train[target_col].copy()

def engineer(df):
    out = pd.DataFrame(index=df.index)
    cabin_deck = None
    if "Cabin" in df.columns:
        cabin = df["Cabin"].astype(str).str.split("/", expand=True)
        cabin_deck = cabin[0].where(cabin[0] != "nan", other="__NA__")
        out["Cabin_Deck"] = cabin_deck
        raw_num = pd.to_numeric(cabin.get(1), errors="coerce")
        out["Cabin_Num"]    = raw_num
        out["CabinNum_bin"] = pd.cut(raw_num, bins=10, labels=False).astype("float")
        out["Cabin_Side"]   = cabin[2].where(cabin[2] != "nan", other="__NA__") if 2 in cabin.columns else "__NA__"
    if "PassengerId" in df.columns:
        pid = df["PassengerId"].astype(str).str.split("_", expand=True)
        out["Group"] = pd.to_numeric(pid[0], errors="coerce")
        grp_series = df["PassengerId"].astype(str).str.split("_").str[0]
        grp_size   = grp_series.map(grp_series.value_counts())
        out["GroupSize"] = grp_size.values
        out["IsSolo"]    = (grp_size == 1).astype(int).values
    spend_cols = ["RoomService","FoodCourt","ShoppingMall","Spa","VRDeck"]
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
            out[c]           = spend[c]
            out[f"Log{{c}}"] = np.log1p(spend[c])
    for col in df.columns:
        uniq = set(str(v).lower() for v in df[col].dropna().unique())
        if uniq <= {{"true", "false"}}:
            out[f"{{col}}_int"] = df[col].map({{"True": 1, "False": 0, True: 1, False: 0}}).fillna(-1).astype(int)
    cryo_int = None
    if "CryoSleep" in df.columns:
        cryo_int = df["CryoSleep"].map({{"True": 1, "False": 0, True: 1, False: 0}}).fillna(0)
        if existing:
            out["CryoSpend_flag"] = ((cryo_int == 1) & (out.get("TotalSpend", pd.Series(0, index=df.index)) > 0)).astype(int)
    if "Age" in df.columns:
        age = pd.to_numeric(df["Age"], errors="coerce")
        out["Age_bin"] = pd.cut(age, bins=[0,12,18,35,60,200], labels=False).astype("float")
        if cryo_int is not None:
            out["Age_x_Cryo"] = age.fillna(age.median()) * cryo_int
        if existing:
            total_spend = out.get("TotalSpend", pd.Series(0, index=df.index))
            out["Age_x_Spend"] = age.fillna(age.median()) * np.log1p(total_spend)
    if cabin_deck is not None and "Destination" in df.columns:
        out["Deck_x_Dest"] = cabin_deck.astype(str) + "_" + df["Destination"].astype(str).fillna("__NA__")
    if "HomePlanet" in df.columns and "Destination" in df.columns:
        out["Home_x_Dest"] = df["HomePlanet"].astype(str).fillna("__NA__") + "_" + df["Destination"].astype(str).fillna("__NA__")
    skip = {{"Cabin", "PassengerId", "Name"}}
    for c in df.columns:
        if c not in skip and c not in out.columns:
            out[c] = df[c]
    return out

_tr_feat = train.drop(columns=[c for c in [id_col, target_col] if c in train.columns])
_te_feat = test.drop(columns=[c for c in [id_col, target_col] if c in test.columns])
X_tr_raw = engineer(_tr_feat)
X_te_raw = engineer(_te_feat)

print(f"Features ({{len(X_tr_raw.columns)}}): {{list(X_tr_raw.columns)}}")
num_cols = X_tr_raw.select_dtypes(include="number").columns.tolist()
cat_cols = [c for c in X_tr_raw.columns if c not in num_cols]

X_tr = X_tr_raw.assign(**{{c: X_tr_raw[c].astype(str) for c in cat_cols}})
X_te = X_te_raw.assign(**{{c: X_te_raw[c].astype(str) for c in cat_cols}})

num_pipe = Pipeline([("imp", SimpleImputer(strategy="median"))])
cat_pipe = Pipeline([
    ("imp", SimpleImputer(strategy="constant", fill_value="__NA__")),
    ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
])
transformers = []
if num_cols: transformers.append(("num", num_pipe, num_cols))
if cat_cols: transformers.append(("cat", cat_pipe, cat_cols))
pre = ColumnTransformer(transformers=transformers, remainder="drop")

clf = Pipeline([("pre", pre), ("model", {model_line})])
print(f"Fitting {best_model}...")
clf.fit(X_tr, y)
preds = clf.predict(X_te)

sub = pd.DataFrame({{id_col: ids, target_col: preds}})
sub.to_csv(submission_path, index=False)
print(f"Done. {{len(sub)}} predictions saved to {{submission_path}}")
""")

    # ── helpers ───────────────────────────────────────────────────────────────

    def _extract_tar(self, message: Message) -> bytes | None:
        for part in message.parts:
            if isinstance(part.root, FilePart):
                fd = part.root.file
                if isinstance(fd, FileWithBytes):
                    return base64.b64decode(fd.bytes)
        return None

    def _extract_tar_to_dir(self, tar_bytes: bytes, dest: Path) -> None:
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
            tar.extractall(dest)

    def _summarize_data(self, csv_files: list[Path]) -> str:
        parts = []
        for p in csv_files:
            try:
                df = pd.read_csv(p, nrows=5)
                parts.append(
                    f"\n### {p.name}\nShape: {df.shape}\n"
                    f"Columns: {list(df.columns)}\n"
                    f"Dtypes:\n{df.dtypes.to_string()}\n"
                    f"Sample:\n{df.head(3).to_string()}"
                )
            except Exception as e:
                parts.append(f"\n### {p.name}: error ({e})")
        return "\n".join(parts)

    def _run_code(self, code: str) -> tuple[bool, str]:
        code_file = self.workdir / "solution.py"
        code_file.write_text(code, encoding="utf-8")
        try:
            result = subprocess.run(
                [sys.executable, str(code_file)],
                capture_output=True, text=True,
                timeout=300, cwd=str(self.workdir),
            )
            return result.returncode == 0, result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return False, "Timed out after 300s"
        except Exception as e:
            return False, str(e)
