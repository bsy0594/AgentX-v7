"""
Stacking Agent — General-purpose multi-model stacking ensemble for tabular ML.
Input : FilePart (tar.gz) + TextPart("OUTPUT_PATH:/path/to/submission.csv")
Output: status message containing "STACKING_PATH:/path/to/submission.csv"

Supports:
  - Binary classification  (bool True/False, str labels, or 0/1 integers)
  - Multi-class classification
  - Regression

Pipeline:
  1. Auto-detect task type from target column
  2. LLM-driven feature engineering (GPT generates competition-specific features)
     with fallback to generic FE (datetime, bool, ID-drop, missing indicators)
  3. CV-based target encoding for high-cardinality categoricals (classification only)
  4. Impute + ordinal-encode remaining categoricals
  5. 5-fold OOF: XGBoost, LightGBM, GradientBoosting, CatBoost
  6. LogisticRegression (classification) or Ridge (regression) meta-learner
     with proper 5-fold OOF to avoid leakage in threshold search
  7. Optimal threshold search for binary classification
  8. Submission written in original target format
"""
import asyncio
import base64
import io
import re
import tarfile
import tempfile
import traceback
from enum import Enum, auto
from pathlib import Path

import numpy as np
import pandas as pd
from a2a.server.tasks import TaskUpdater
from a2a.types import FilePart, FileWithBytes, Message, TaskState
from a2a.utils import get_message_text, new_agent_text_message
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# ── LLM client ─────────────────────────────────────────────────────────────────

import os
_OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")


# ── task type ──────────────────────────────────────────────────────────────────

class TaskType(Enum):
    BINARY     = auto()
    MULTICLASS = auto()
    REGRESSION = auto()


def detect_task(y: pd.Series) -> TaskType:
    """Infer task type from the target column."""
    y_str    = y.astype(str).str.strip().str.lower()
    unique   = set(y_str.unique())
    n_unique = len(unique)

    # Bool-like strings → binary classification
    if unique <= {"true", "false"}:
        return TaskType.BINARY

    # Try numeric
    y_num = pd.to_numeric(y, errors="coerce")
    if y_num.notna().mean() > 0.95:          # mostly numeric
        if n_unique == 2:
            return TaskType.BINARY
        if n_unique <= 25:
            return TaskType.MULTICLASS
        return TaskType.REGRESSION           # many distinct numeric values

    # Categorical strings
    if n_unique == 2:
        return TaskType.BINARY
    if n_unique <= 25:
        return TaskType.MULTICLASS
    return TaskType.REGRESSION


def encode_target(y: pd.Series, task: TaskType) -> tuple[np.ndarray, dict]:
    """
    Convert target to a numeric ndarray.
    Returns (encoded_array, label_to_int_mapping).
    For regression the mapping is empty.
    """
    if task == TaskType.REGRESSION:
        return pd.to_numeric(y, errors="coerce").values.astype(float), {}

    y_str  = y.astype(str).str.strip().str.lower()
    labels = sorted(y_str.unique())
    mapping = {lbl: i for i, lbl in enumerate(labels)}
    return y_str.map(mapping).values.astype(int), mapping


def decode_predictions(
    preds: np.ndarray,
    task: TaskType,
    original_y: pd.Series,
    mapping: dict,
) -> pd.Series:
    """Convert numeric predictions back to the original target format."""
    if task == TaskType.REGRESSION:
        return pd.Series(preds, dtype=float)

    reverse = {v: k for k, v in mapping.items()}

    # Reconstruct original-case mapping from the actual target column
    original_case: dict[int, object] = {}
    for val in original_y.unique():
        key = str(val).strip().lower()
        if key in mapping:
            original_case[mapping[key]] = val

    def _convert(code: int):
        if code in original_case:
            return original_case[code]
        if code in reverse:
            return reverse[code]
        return code

    return pd.Series(preds).map(_convert)


# ── generic feature engineering ────────────────────────────────────────────────

def engineer(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Dataset-agnostic feature engineering — minimal, high-signal only:
      1. Drop near-unique identifier columns (>97% unique, non-numeric)
      2. Parse datetime columns → year, month, day, hour, weekday
      3. Convert bool-like text columns → int
      4. Missing indicators — binary flags for columns with >1% and <99% NaN
    """
    tr = train_df.copy()
    te = test_df.copy()
    n_rows = len(tr)

    dropped_cols = []
    datetime_cols = []

    for col in list(tr.columns):
        series = tr[col]

        # Drop ID-like columns (high cardinality, non-numeric)
        if series.dtype == object:
            n_unique = series.nunique()
            if n_unique > 0.97 * n_rows and n_unique > 50:
                dropped_cols.append(col)
                continue

        # Parse datetime columns
        if series.dtype == object:
            sample = series.dropna().head(100)
            try:
                pd.to_datetime(sample, infer_datetime_format=True, errors="raise")
                full_tr = pd.to_datetime(series, infer_datetime_format=True, errors="coerce")
                if full_tr.notna().mean() > 0.8:
                    datetime_cols.append(col)
                    full_te = pd.to_datetime(te[col], infer_datetime_format=True, errors="coerce")
                    for sfx, acc in [("year", "year"), ("month", "month"), ("day", "day"),
                                     ("hour", "hour"), ("weekday", "weekday")]:
                        tr[f"{col}_{sfx}"] = getattr(full_tr.dt, acc)
                        te[f"{col}_{sfx}"] = getattr(full_te.dt, acc)
                    continue
            except Exception:
                pass

        # Bool-like strings → int
        if series.dtype == object:
            uniq = set(str(v).strip().lower() for v in series.dropna().unique())
            if uniq <= {"true", "false", "yes", "no", "1", "0"}:
                bmap = {"true": 1, "false": 0, "yes": 1, "no": 0, "1": 1, "0": 0}
                tr[col] = series.astype(str).str.strip().str.lower().map(bmap).fillna(-1).astype(int)
                te[col] = te[col].astype(str).str.strip().str.lower().map(bmap).fillna(-1).astype(int)

    tr.drop(columns=[c for c in dropped_cols + datetime_cols if c in tr.columns], inplace=True)
    te.drop(columns=[c for c in dropped_cols + datetime_cols if c in te.columns], inplace=True)

    # Missing indicators (binary flags for columns with meaningful missing rates)
    for col in list(tr.columns):
        miss_rate = tr[col].isna().mean()
        if 0.01 < miss_rate < 0.99:
            tr[f"{col}__miss"] = tr[col].isna().astype(int)
            te[f"{col}__miss"] = te[col].isna().astype(int)

    return tr, te


# ── LLM-driven feature engineering ────────────────────────────────────────────

_FE_SYSTEM_PROMPT = """\
You are a Kaggle Grandmaster feature engineer. Given a dataset description, \
write a Python function `fe(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]` \
that creates powerful features for the prediction task.

Rules:
- The function receives COPIES of the raw train/test DataFrames (target column already removed).
- Return modified (train, test) with new columns added. Keep all original columns.
- Use only pandas, numpy, re. No model training. No sklearn.
- Handle NaN gracefully — never raise on missing values.
- Use train statistics (e.g., value_counts, median) when encoding; apply same mapping to test.
- Do NOT include import statements — pd, np, and re are already imported.
- Output ONLY the function body inside ```python ... ``` fences. No explanation.
- Create 10-20 high-quality features. Do NOT create more than 25 new columns.
- Focus on features with strong predictive signal based on the correlations shown.
"""


def _build_data_summary(train: pd.DataFrame, target_col: str, y: pd.Series) -> str:
    """Build a compact data summary with target correlations for the LLM prompt."""
    lines = []
    lines.append(f"Shape: {train.shape[0]} rows × {train.shape[1]} cols")
    lines.append(f"Target: '{target_col}' — {y.nunique()} unique values, distribution: {y.value_counts().head(5).to_dict()}")

    # Encode target for correlation
    y_num = pd.to_numeric(y, errors="coerce")
    if y_num.isna().all():
        y_str = y.astype(str).str.strip().str.lower()
        labels = sorted(y_str.unique())
        y_num = y_str.map({lbl: i for i, lbl in enumerate(labels)})

    lines.append("")
    lines.append("Columns (with target correlation where computable):")
    for col in train.columns:
        if col == target_col:
            continue
        s = train[col]
        dtype = str(s.dtype)
        miss = f"{s.isna().mean():.0%} missing" if s.isna().any() else "no nulls"
        corr_str = ""
        if s.dtype in ("float64", "int64", "float32", "int32"):
            try:
                c = s.corr(y_num)
                if pd.notna(c):
                    corr_str = f", corr={c:+.3f}"
            except Exception:
                pass
        if s.dtype == object:
            uniq = s.nunique()
            examples = s.dropna().unique()[:5].tolist()
            lines.append(f"  {col} ({dtype}, {uniq} unique, {miss}{corr_str}): {examples}")
        else:
            lines.append(f"  {col} ({dtype}, {miss}{corr_str}): min={s.min()}, median={s.median()}, max={s.max()}")

    lines.append("")
    lines.append("First 5 rows:")
    lines.append(train.head(5).to_string())
    return "\n".join(lines)


def _call_llm(messages: list[dict]) -> str:
    """Call GPT-4o (synchronous — use via asyncio.to_thread)."""
    import openai
    try:
        client = openai.OpenAI(api_key=_OPENAI_KEY)
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.3,
            max_tokens=4000,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        print(f"[Stacking] LLM call failed: {e} — falling back to generic FE")
        return ""


def _extract_and_run_fe(content: str, train: pd.DataFrame, test: pd.DataFrame, target_col: str):
    """Extract fe() from LLM response and execute it. Returns (tr_fe, te_fe) or raises."""
    match = re.search(r"```python\s*\n(.*?)```", content, re.DOTALL)
    if not match:
        match = re.search(r"```\s*\n(.*?)```", content, re.DOTALL)
    if not match:
        raise ValueError("No code block found")

    code = match.group(1).strip()
    print(f"[LLM-FE] Generated code ({len(code)} chars)")

    namespace = {"pd": pd, "np": np, "re": re}
    exec(code, namespace)

    if "fe" not in namespace:
        raise ValueError("No fe() function found")

    tr_copy = train.drop(columns=[target_col], errors="ignore").copy()
    te_copy = test.copy()
    tr_fe, te_fe = namespace["fe"](tr_copy, te_copy)

    if not isinstance(tr_fe, pd.DataFrame) or not isinstance(te_fe, pd.DataFrame):
        raise ValueError("fe() didn't return DataFrames")
    if len(tr_fe) != len(train) or len(te_fe) != len(test):
        raise ValueError(f"Row count mismatch: {len(tr_fe)} vs {len(train)}")

    shared = [c for c in tr_fe.columns if c in te_fe.columns]
    return tr_fe[shared], te_fe[shared]


def _quick_cv_score(X_tr: pd.DataFrame, y_arr: np.ndarray, task: "TaskType") -> float:
    """Quick 3-fold LightGBM CV score for LLM feedback."""
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OrdinalEncoder as OE

    num_cols = X_tr.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X_tr.columns if c not in num_cols]

    # Simple preprocessing
    X = X_tr.copy()
    for c in cat_cols:
        X[c] = X[c].astype(str)
    transformers = []
    if num_cols:
        transformers.append(("n", SimpleImputer(strategy="median"), num_cols))
    if cat_cols:
        transformers.append(("c", Pipeline([
            ("i", SimpleImputer(strategy="constant", fill_value="__NA__")),
            ("e", OE(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), cat_cols))
    if not transformers:
        return 0.0
    ct = ColumnTransformer(transformers=transformers, remainder="drop")
    Xarr = ct.fit_transform(X)

    try:
        from lightgbm import LGBMClassifier, LGBMRegressor
        if task == TaskType.REGRESSION:
            mdl_cls = LGBMRegressor
        else:
            mdl_cls = LGBMClassifier
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42) if task != TaskType.REGRESSION else KFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        for tr_idx, val_idx in skf.split(Xarr, y_arr):
            m = mdl_cls(n_estimators=100, learning_rate=0.1, max_depth=5, verbose=-1, n_jobs=1, random_state=42)
            m.fit(Xarr[tr_idx], y_arr[tr_idx])
            if task == TaskType.REGRESSION:
                from sklearn.metrics import mean_squared_error
                scores.append(-mean_squared_error(y_arr[val_idx], m.predict(Xarr[val_idx])))
            else:
                scores.append((m.predict(Xarr[val_idx]) == y_arr[val_idx]).mean())
        return float(np.mean(scores))
    except Exception as e:
        print(f"[LLM-FE] Quick CV failed: {e}")
        return 0.0


async def llm_feature_engineer(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str,
    y: pd.Series,
    task: "TaskType",
    y_arr: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Iterative LLM-driven feature engineering (2 rounds with GPT-4o):
      Round 1: Generate features from data summary + target correlations
      Round 2: Refine based on quick CV feedback
    Falls back to generic engineer() on failure.
    """
    summary = _build_data_summary(train, target_col, y)
    messages = [
        {"role": "system", "content": _FE_SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"Dataset summary:\n{summary}\n\n"
            f"Write a `fe(train, test)` function that creates the best possible features "
            f"for predicting '{target_col}'."
        )},
    ]

    tr_feat = train.drop(columns=[target_col], errors="ignore")
    te_feat = test.copy()
    best_tr, best_te = engineer(tr_feat, te_feat)
    best_score = _quick_cv_score(best_tr, y_arr, task)
    baseline_score = best_score
    best_msg = f"[LLM-FE] Generic baseline CV: {best_score:.4f}"
    print(best_msg)

    for round_i in range(2):
        try:
            content = await asyncio.wait_for(
                asyncio.to_thread(_call_llm, messages),
                timeout=90,
            )
            messages.append({"role": "assistant", "content": content})

            tr_fe, te_fe = _extract_and_run_fe(content, train, test, target_col)
            score = _quick_cv_score(tr_fe, y_arr, task)
            print(f"[LLM-FE] Round {round_i+1}: {tr_fe.shape[1]} features, CV={score:.4f}")

            if score > best_score:
                best_tr, best_te = tr_fe, te_fe
                best_score = score
                best_msg = f"[LLM-FE] Round {round_i+1}: {tr_fe.shape[1]} features, CV={score:.4f}"

            if round_i == 0:
                messages.append({"role": "user", "content": (
                    f"Your features achieved CV accuracy = {score:.4f} with {tr_fe.shape[1]} features. "
                    f"The baseline (no feature engineering) scored {baseline_score:.4f}. "
                    f"Try to improve: add more interaction features, group-level aggregations, "
                    f"or ratio features. Remove any features that might add noise. "
                    f"Write an improved `fe(train, test)` function."
                )})

        except Exception as e:
            print(f"[LLM-FE] Round {round_i+1} failed: {e}")
            if round_i == 0:
                messages.append({"role": "user", "content": (
                    f"Your code failed with: {type(e).__name__}: {e}. "
                    f"Fix the error and write a corrected `fe(train, test)` function."
                )})

    return best_tr, best_te, best_msg


# ── CV-based target encoding ───────────────────────────────────────────────────

def cv_target_encode(
    X_train: pd.DataFrame,
    y_encoded: np.ndarray,
    X_test: pd.DataFrame,
    cat_cols: list[str],
    n_splits: int = 5,
    smoothing: float = 30.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leak-free target encoding with smoothed leave-one-fold-out means."""
    y_series    = pd.Series(y_encoded)
    global_mean = float(y_series.mean())
    X_tr_enc    = X_train.copy()
    X_te_enc    = X_test.copy()
    skf         = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for col in cat_cols:
        oof_enc      = np.full(len(X_train), global_mean, dtype=float)
        te_fold_preds = np.zeros(len(X_test))

        for tr_idx, val_idx in skf.split(X_train, y_series):
            tr_x   = X_train.iloc[tr_idx][col].astype(str)
            tr_y   = y_series.iloc[tr_idx]
            val_x  = X_train.iloc[val_idx][col].astype(str)
            stats  = tr_y.groupby(tr_x.values).agg(["sum", "count"])
            smooth = stats["count"] / (stats["count"] + smoothing)
            enc    = smooth * (stats["sum"] / stats["count"]) + (1 - smooth) * global_mean

            oof_enc[val_idx]  = val_x.map(enc).fillna(global_mean).values
            te_fold_preds    += X_test[col].astype(str).map(enc).fillna(global_mean).values

        te_fold_preds /= n_splits
        X_tr_enc[col]  = oof_enc
        X_te_enc[col]  = te_fold_preds

    return X_tr_enc, X_te_enc


# ── impute + encode ────────────────────────────────────────────────────────────

def prepare_arrays(
    X_tr: pd.DataFrame,
    X_te: pd.DataFrame,
    encoded_cat_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Impute missing values and ordinal-encode any remaining categorical columns.
    Columns already target-encoded are treated as numeric.
    """
    num_cols = X_tr.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X_tr.columns if c not in num_cols]

    # Target-encoded cols are already numeric — reclassify
    for c in encoded_cat_cols:
        if c in cat_cols:
            cat_cols.remove(c)
            if c not in num_cols:
                num_cols.append(c)

    X_tr = X_tr.assign(**{c: X_tr[c].astype(str) for c in cat_cols})
    X_te = X_te.assign(**{c: X_te[c].astype(str) for c in cat_cols})

    transformers = []
    if num_cols:
        transformers.append(("num", SimpleImputer(strategy="median"), num_cols))
    if cat_cols:
        transformers.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="constant", fill_value="__NA__")),
            ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), cat_cols))

    ct      = ColumnTransformer(transformers=transformers, remainder="drop")
    Xtr_arr = ct.fit_transform(X_tr)
    Xte_arr = ct.transform(X_te)
    return Xtr_arr, Xte_arr


# ── OOF prediction ─────────────────────────────────────────────────────────────

def oof_predict(
    model_cls,
    model_kwargs: dict,
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    task: TaskType,
    n_splits: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    5-fold OOF predictions.
    Returns:
      oof_proba : (n_train,)  for binary, (n_train, n_classes) for multiclass
      te_proba  : (n_test,)   for binary, (n_test, n_classes)  for multiclass
    For regression returns predicted values instead of probabilities.
    """
    if task == TaskType.REGRESSION:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    is_multiclass = task == TaskType.MULTICLASS
    n_classes     = len(np.unique(ytr)) if is_multiclass else None

    if is_multiclass:
        oof_proba = np.zeros((len(Xtr), n_classes))
        te_proba  = np.zeros((len(Xte), n_classes))
    else:
        oof_proba = np.zeros(len(Xtr))
        te_proba  = np.zeros(len(Xte))

    for tr_idx, val_idx in cv.split(Xtr, ytr):
        mdl = model_cls(**model_kwargs)
        mdl.fit(Xtr[tr_idx], ytr[tr_idx])

        if task == TaskType.REGRESSION:
            oof_proba[val_idx]  = mdl.predict(Xtr[val_idx])
            te_proba           += mdl.predict(Xte)
        elif is_multiclass:
            oof_proba[val_idx]  = mdl.predict_proba(Xtr[val_idx])
            te_proba           += mdl.predict_proba(Xte)
        else:
            oof_proba[val_idx]  = mdl.predict_proba(Xtr[val_idx])[:, 1]
            te_proba           += mdl.predict_proba(Xte)[:, 1]

    te_proba /= n_splits
    return oof_proba, te_proba


# ── meta-learner OOF ───────────────────────────────────────────────────────────

def meta_oof_predict(
    S_tr: np.ndarray,
    ytr: np.ndarray,
    S_te: np.ndarray,
    task: TaskType,
    n_splits: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Proper cross-validated meta-learner predictions (no train-on-all leakage).
    Returns (oof_meta_proba, te_meta_proba).
    """
    scaler = StandardScaler()

    if task == TaskType.REGRESSION:
        cv          = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        oof_meta    = np.zeros(len(S_tr))
        te_meta     = np.zeros(len(S_te))
        for tr_idx, val_idx in cv.split(S_tr):
            sc      = StandardScaler()
            S_tr_s  = sc.fit_transform(S_tr[tr_idx])
            S_val_s = sc.transform(S_tr[val_idx])
            S_te_s  = sc.transform(S_te)
            meta    = Ridge(alpha=1.0)
            meta.fit(S_tr_s, ytr[tr_idx])
            oof_meta[val_idx] = meta.predict(S_val_s)
            te_meta          += meta.predict(S_te_s)
        te_meta /= n_splits
    else:
        cv          = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        n_classes   = len(np.unique(ytr))
        is_multi    = task == TaskType.MULTICLASS

        if is_multi:
            oof_meta = np.zeros((len(S_tr), n_classes))
            te_meta  = np.zeros((len(S_te), n_classes))
        else:
            oof_meta = np.zeros(len(S_tr))
            te_meta  = np.zeros(len(S_te))

        for tr_idx, val_idx in cv.split(S_tr, ytr):
            sc      = StandardScaler()
            S_tr_s  = sc.fit_transform(S_tr[tr_idx])
            S_val_s = sc.transform(S_tr[val_idx])
            S_te_s  = sc.transform(S_te)
            meta    = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
            meta.fit(S_tr_s, ytr[tr_idx])
            if is_multi:
                oof_meta[val_idx] = meta.predict_proba(S_val_s)
                te_meta          += meta.predict_proba(S_te_s)
            else:
                oof_meta[val_idx] = meta.predict_proba(S_val_s)[:, 1]
                te_meta          += meta.predict_proba(S_te_s)[:, 1]
        te_meta /= n_splits

    return oof_meta, te_meta


# ── Stacking Agent ─────────────────────────────────────────────────────────────

class StackingAgent:
    async def run(self, message: Message, updater: TaskUpdater) -> None:

        # ── Parse input ──────────────────────────────────────────────────────
        tar_bytes   = self._extract_tar(message)
        payload     = get_message_text(message) or ""
        output_path = None
        m = re.search(r"OUTPUT_PATH:(.*?)(?:\n|$)", payload)
        if m:
            output_path = Path(m.group(1).strip())

        if not tar_bytes:
            await updater.failed(new_agent_text_message("[Stacking] No data received."))
            return

        await updater.update_status(
            TaskState.working,
            new_agent_text_message("[Stacking] Extracting data..."),
        )

        # ── Extract data ──────────────────────────────────────────────────────
        tmpdir   = Path(tempfile.mkdtemp(prefix="stacking_"))
        data_dir = tmpdir / "data"
        data_dir.mkdir()
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
            tar.extractall(data_dir)

        train_path  = next(data_dir.rglob("train.csv"))
        test_path   = next(data_dir.rglob("test.csv"))
        sample_path = next(data_dir.rglob("sample_submission.csv"), None)

        train  = pd.read_csv(train_path)
        test   = pd.read_csv(test_path)
        if sample_path:
            sample = pd.read_csv(sample_path)
            id_col, target_col = sample.columns[0], sample.columns[1]
        else:
            id_col, target_col = train.columns[0], train.columns[-1]

        ids = test[id_col].reset_index(drop=True)
        y   = train[target_col].copy()

        # ── Task detection ────────────────────────────────────────────────────
        task = detect_task(y)
        y_encoded, label_mapping = encode_target(y, task)
        y_arr = y_encoded

        print(f"[Stacking] train={train.shape} test={test.shape} "
              f"task={task.name} n_classes={len(label_mapping) or 'N/A'}")

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"[Stacking] Task detected: {task.name}. Building ensemble..."),
        )

        # ── Feature engineering (LLM-driven with generic fallback) ────────────
        await updater.update_status(
            TaskState.working,
            new_agent_text_message("[Stacking] LLM feature engineering (2 rounds)..."),
        )
        te_clean = test.drop(columns=[c for c in [id_col, target_col] if c in test.columns])
        X_tr_raw, X_te_raw, fe_msg = await llm_feature_engineer(train, te_clean, target_col, y, task, y_arr)

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(fe_msg),
        )

        # Align columns
        shared_cols = [c for c in X_tr_raw.columns if c in X_te_raw.columns]
        X_tr_raw    = X_tr_raw[shared_cols]
        X_te_raw    = X_te_raw[shared_cols]
        print(f"[Stacking] Features after engineering: {X_tr_raw.shape[1]}")

        # ── CV target encoding (classification only, high-cardinality cols) ───
        if task != TaskType.REGRESSION:
            candidate_cat = [
                c for c in X_tr_raw.select_dtypes(include="object").columns
                if X_tr_raw[c].nunique() > 5
            ]
            if candidate_cat:
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(f"[Stacking] Target-encoding {len(candidate_cat)} cols..."),
                )
                X_tr_enc, X_te_enc = cv_target_encode(X_tr_raw, y_arr, X_te_raw, candidate_cat)
            else:
                X_tr_enc, X_te_enc = X_tr_raw.copy(), X_te_raw.copy()
                candidate_cat = []
        else:
            X_tr_enc, X_te_enc = X_tr_raw.copy(), X_te_raw.copy()
            candidate_cat = []

        # ── Prepare numeric arrays ────────────────────────────────────────────
        Xtr_arr, Xte_arr = prepare_arrays(X_tr_enc, X_te_enc, candidate_cat)

        # ── Base model OOF ────────────────────────────────────────────────────
        successful_oof_tr: list[np.ndarray] = []
        successful_oof_te: list[np.ndarray] = []

        async def fit_base(label: str, model_cls, kwargs: dict) -> None:
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"[Stacking] Fitting {label} OOF..."),
            )
            try:
                oof, te = oof_predict(model_cls, kwargs, Xtr_arr, y_arr, Xte_arr, task)
                if task == TaskType.BINARY:
                    acc = ((oof >= 0.5).astype(int) == y_arr).mean()
                    print(f"[Stacking] {label} OOF acc: {acc:.4f}")
                elif task == TaskType.MULTICLASS:
                    acc = (oof.argmax(axis=1) == y_arr).mean()
                    print(f"[Stacking] {label} OOF acc: {acc:.4f}")
                else:
                    rmse = np.sqrt(((oof - y_arr) ** 2).mean())
                    print(f"[Stacking] {label} OOF RMSE: {rmse:.4f}")
                successful_oof_tr.append(oof)
                successful_oof_te.append(te)
            except Exception as e:
                print(f"[Stacking] {label} failed: {e}")

        try:
            from xgboost import XGBClassifier, XGBRegressor
            cls = XGBRegressor if task == TaskType.REGRESSION else XGBClassifier
            xgb_kwargs = dict(
                n_estimators=500, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
                gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
                random_state=42, verbosity=0,
                **({} if task == TaskType.REGRESSION else {"eval_metric": "logloss"}),
            )
            await fit_base("XGBoost", cls, xgb_kwargs)
        except Exception as e:
            print(f"[Stacking] XGBoost import failed: {e}")

        try:
            from lightgbm import LGBMClassifier, LGBMRegressor
            cls = LGBMRegressor if task == TaskType.REGRESSION else LGBMClassifier
            lgb_kwargs = dict(
                n_estimators=500, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.7, num_leaves=63,
                reg_alpha=0.1, reg_lambda=0.5, min_child_samples=20,
                random_state=42, verbose=-1,
            )
            await fit_base("LightGBM", cls, lgb_kwargs)
        except Exception as e:
            print(f"[Stacking] LightGBM import failed: {e}")

        try:
            from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
            cls = GradientBoostingRegressor if task == TaskType.REGRESSION else GradientBoostingClassifier
            gbm_kwargs = dict(
                n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, random_state=42,
            )
            await fit_base("GBM", cls, gbm_kwargs)
        except Exception as e:
            print(f"[Stacking] GBM failed: {e}")

        try:
            from catboost import CatBoostClassifier, CatBoostRegressor
            cls = CatBoostRegressor if task == TaskType.REGRESSION else CatBoostClassifier
            cat_kwargs = dict(
                iterations=500, depth=6, learning_rate=0.05, l2_leaf_reg=3,
                random_seed=42, verbose=0,
            )
            await fit_base("CatBoost", cls, cat_kwargs)
        except Exception as e:
            print(f"[Stacking] CatBoost import failed: {e}")

        if not successful_oof_tr:
            await updater.failed(new_agent_text_message("[Stacking] All base models failed."))
            return

        # ── Stack features ────────────────────────────────────────────────────
        S_tr = np.column_stack(successful_oof_tr)
        S_te = np.column_stack(successful_oof_te)

        # ── Meta-learner (proper CV OOF — no leakage) ─────────────────────────
        await updater.update_status(
            TaskState.working,
            new_agent_text_message("[Stacking] Training meta-learner (CV OOF)..."),
        )
        meta_oof, meta_te = meta_oof_predict(S_tr, y_arr, S_te, task)

        # ── Threshold search (binary only) ────────────────────────────────────
        if task == TaskType.BINARY:
            best_thresh, best_acc = 0.5, 0.0
            for t in np.arange(0.30, 0.71, 0.01):
                acc = ((meta_oof >= t).astype(int) == y_arr).mean()
                if acc > best_acc:
                    best_acc, best_thresh = acc, t
            print(f"[Stacking] OOF meta accuracy: {best_acc:.4f} @ threshold={best_thresh:.2f}")
            preds_encoded = (meta_te >= best_thresh).astype(int)

        elif task == TaskType.MULTICLASS:
            preds_encoded = meta_oof.argmax(axis=1) if meta_oof.ndim > 1 else meta_te.argmax(axis=1)
            preds_encoded = meta_te.argmax(axis=1)
            oof_acc = (meta_oof.argmax(axis=1) == y_arr).mean()
            print(f"[Stacking] OOF meta accuracy: {oof_acc:.4f}")

        else:  # REGRESSION
            preds_encoded = meta_te
            oof_rmse = np.sqrt(((meta_oof - y_arr) ** 2).mean())
            print(f"[Stacking] OOF meta RMSE: {oof_rmse:.4f}")

        # ── Decode predictions to original format ─────────────────────────────
        preds_out = decode_predictions(preds_encoded, task, y, label_mapping)

        # ── Save submission ────────────────────────────────────────────────────
        if output_path is None:
            output_path = tmpdir / "stacking_submission.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        sub = pd.DataFrame({id_col: ids, target_col: preds_out})
        sub.to_csv(output_path, index=False)
        print(f"[Stacking] Saved {len(sub)} rows → {output_path.name}")

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                f"[Stacking] Done. task={task.name} n_models={len(successful_oof_tr)} "
                f"predictions={len(sub)}\n"
                f"STACKING_PATH:{output_path}"
            ),
        )

    def _extract_tar(self, message: Message) -> bytes | None:
        for part in message.parts:
            if isinstance(part.root, FilePart):
                fd = part.root.file
                if isinstance(fd, FileWithBytes):
                    return base64.b64decode(fd.bytes)
        return None
