import re
from pathlib import Path

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState
from a2a.utils import get_message_text, new_agent_text_message
from openai import AsyncOpenAI

import os
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

SYSTEM_PROMPT = """\
You are an expert ML Python debugger specializing in data science pipelines.

You receive a failing Python script and its error traceback. Your job is to:
1. Diagnose the exact root cause
2. Apply the correct fix
3. Return ONLY the corrected Python code — no markdown, no explanations

KNOWN ERROR PATTERNS AND FIXES:

[pandas Copy-on-Write / ChainedAssignment]
Symptoms: ChainedAssignmentError, column_setitem, _setitem_single_column
Cause: Mutating a DataFrame slice in-place (pandas 2.x CoW)
Fix: Use .assign() to create new columns instead of df[col] = ... on a slice.
     Never use inplace=True on a subset. Reassign: df = df.assign(col=df[col].fillna(v))

[Mixed bool/str in OrdinalEncoder]
Symptoms: TypeError: Encoders require uniformly strings or numbers. Got ['bool', 'str']
Cause: object-dtype columns (CryoSleep, VIP) contain real Python bool + NaN.
       After imputing NaN with a string, the column has mixed bool/str types.
Fix: Before encoding, cast ALL categorical columns to str uniformly:
     X = X.assign(**{col: X[col].astype(str) for col in cat_cols})

[KeyError on column access]
Symptoms: KeyError, pandas IndexEngine.get_loc
Cause: Code references a column that doesn't exist — usually from:
  - Case mismatch (e.g. 'cabin' vs 'Cabin')
  - Feature engineered in train but not test
  - Column dropped before referenced
Fix: Only use feat = [c for c in train.columns if c not in (id_col, target_col)
     and c in test.columns]. Apply ALL feature engineering to both train and test.

[SimpleImputer sort crash on mixed types]
Symptoms: TypeError: '<' not supported between 'str' and 'bool'
Cause: SimpleImputer(strategy='most_frequent') sorts values to find mode,
       fails when column has both bool and str entries.
Fix: Cast to str before imputing, or use ColumnTransformer to separate
     numeric and categorical pipelines.

[sklearn pipeline column mismatch]
Symptoms: ValueError: X has N features but model expects M
Cause: Train/test preprocessing produced different column counts.
Fix: Ensure identical feature list and transformations for both train and test.
     Use ColumnTransformer with explicit column lists.

[MANDATORY CODE STYLE — always use this pattern]
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

num_pipe = Pipeline([("imp", SimpleImputer(strategy="median"))])
cat_pipe = Pipeline([
    ("imp", SimpleImputer(strategy="constant", fill_value="__NA__")),
    ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
])
X_tr = X_tr.assign(**{col: X_tr[col].astype(str) for col in cat_cols})
X_te = X_te.assign(**{col: X_te[col].astype(str) for col in cat_cols})
"""

# Patterns for fast rule-based pre-diagnosis (before calling GPT)
KNOWN_ERRORS = [
    (
        r"ChainedAssignment|column_setitem|_setitem_single_column",
        "pandas Copy-on-Write: in-place mutation on a DataFrame slice"
    ),
    (
        r"uniformly strings or numbers.*bool.*str|bool.*str.*uniformly",
        "Mixed bool/str types in categorical column — cast to str before encoding"
    ),
    (
        r"get_loc|KeyError",
        "Column not found — likely case mismatch or feature engineered in train but missing in test"
    ),
    (
        r"'<' not supported between instances of 'str' and 'bool'",
        "SimpleImputer sort crash — mixed bool/str in object column"
    ),
    (
        r"X has \d+ features but \w+ is expecting \d+",
        "Feature count mismatch between train and test preprocessing"
    ),
    (
        r"could not convert string to float",
        "Categorical column not encoded before passing to model"
    ),
    (
        r"Input contains NaN",
        "NaN values not imputed before model training"
    ),
]


def diagnose(error: str) -> str:
    """Rule-based pre-diagnosis from known error patterns."""
    for pattern, description in KNOWN_ERRORS:
        if re.search(pattern, error, re.IGNORECASE):
            return description
    return "Unknown error — requires LLM analysis"


class ErrorHandlerAgent:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        payload = get_message_text(message) or ""
        if not payload.strip():
            await updater.failed(new_agent_text_message("No code/error provided."))
            return

        # Parse structured input
        code, error = self._parse_payload(payload)

        # Rule-based diagnosis first
        diagnosis = diagnose(error)
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                f"[ErrorHandler] Diagnosing...\n"
                f"Root cause: {diagnosis}"
            ),
        )

        # GPT fix with specialized system prompt
        fixed_code = await self._fix(code, error, diagnosis)

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"[ErrorHandler] Fix applied. Root cause was:\n{diagnosis}"),
        )

        # Return the fixed code as a status message (ML agent reads it)
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"FIXED_CODE_START\n{fixed_code}\nFIXED_CODE_END"),
        )

    def _parse_payload(self, payload: str) -> tuple[str, str]:
        """Extract CODE and ERROR sections from the payload."""
        code_match = re.search(r"CODE:\n(.*?)(?=\nERROR:|\Z)", payload, re.DOTALL)
        error_match = re.search(r"ERROR:\n(.*)", payload, re.DOTALL)
        code  = code_match.group(1).strip() if code_match else payload
        error = error_match.group(1).strip() if error_match else ""
        return code, error

    async def _fix(self, code: str, error: str, diagnosis: str) -> str:
        resp = await self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"Fix this failing ML Python code.\n\n"
                    f"PRE-DIAGNOSED ROOT CAUSE: {diagnosis}\n\n"
                    f"ERROR TRACEBACK:\n{error}\n\n"
                    f"FAILING CODE:\n{code}\n\n"
                    f"Return ONLY the corrected Python code. No markdown. No explanations."
                )},
            ],
            temperature=0.05,
            max_tokens=3500,
        )
        fixed = resp.choices[0].message.content.strip()
        if fixed.startswith("```"):
            lines = fixed.splitlines()
            fixed = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        return fixed
