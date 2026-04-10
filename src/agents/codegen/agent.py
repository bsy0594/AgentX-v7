"""
Code Generator Agent — Sole responsibility: write complete ML Python code.
Input : text message with PLAN, CRITIQUE, EDA_REPORT, FE_CODE, DATA_DIR, SUBMISSION_PATH, INSTRUCTIONS.
Output: CODE_START\n{python_code}\nCODE_END
"""
from pathlib import Path

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState
from a2a.utils import get_message_text, new_agent_text_message
from openai import AsyncOpenAI

import os
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

SYSTEM_PROMPT = """\
You are a Machine Learning Engineer with 10 years of experience in Kaggle competitions.

Your sole task right now is to write COMPLETE, RUNNABLE Python ML code.

Rules:
- Use sklearn Pipeline + ColumnTransformer for ALL preprocessing (no in-place mutations)
- Use exactly the engineer() function provided in FE_CODE if present
- Follow the DATA_DIR and SUBMISSION_PATH exactly — do NOT change them
- Cast all categorical columns to str before encoding
- Output ONLY raw Python code — no markdown fences, no explanations
"""


class CodeGeneratorAgent:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        payload = get_message_text(message) or ""
        if not payload.strip():
            await updater.failed(new_agent_text_message("[CodeGen] No input received."))
            return

        await updater.update_status(
            TaskState.working,
            new_agent_text_message("[CodeGen] Writing ML solution code..."),
        )

        try:
            resp = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": payload},
                ],
                temperature=0.1,
                max_tokens=4000,
            )
            code = resp.choices[0].message.content.strip()
        except Exception as e:
            await updater.failed(new_agent_text_message(f"[CodeGen] OpenAI call failed: {e}"))
            return

        if code.startswith("```"):
            lines = code.splitlines()
            code  = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        print(f"[CodeGen] Code generated ({len(code)} chars)")

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"CODE_START\n{code}\nCODE_END"),
        )
