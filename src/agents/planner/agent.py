"""
Planner Agent — Sole responsibility: form a concise ML solution plan.
Input : text message with INSTRUCTIONS + DATA_SUMMARY sections.
Output: PLAN_START\n{plan}\nPLAN_END
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

Your sole task right now is to write a CONCISE solution plan — no code.

Cover exactly these 5 points:
1. Task type & evaluation metric
2. Key preprocessing steps
3. Feature engineering priorities (based on data summary)
4. Model choice and why
5. Potential pitfalls to watch for

Be specific, actionable, and brief (under 400 words).
"""


class PlannerAgent:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        payload = get_message_text(message) or ""
        if not payload.strip():
            await updater.failed(new_agent_text_message("[Planner] No input received."))
            return

        await updater.update_status(
            TaskState.working,
            new_agent_text_message("[Planner] Forming solution plan..."),
        )

        try:
            resp = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": payload},
                ],
                temperature=0.3,
                max_tokens=600,
            )
            plan = resp.choices[0].message.content.strip()
        except Exception as e:
            await updater.failed(new_agent_text_message(f"[Planner] OpenAI call failed: {e}"))
            return

        print(f"[Planner] Plan formed ({len(plan)} chars)")

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"PLAN_START\n{plan}\nPLAN_END"),
        )
