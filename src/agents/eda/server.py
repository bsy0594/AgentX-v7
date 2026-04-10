import argparse
import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from executor import Executor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8003)
    args = parser.parse_args()

    skill = AgentSkill(
        id="eda-analysis",
        name="Exploratory Data Analysis",
        description=(
            "Receives competition data (tar.gz). Returns a structured EDA report: "
            "null rates, feature distributions, target correlation, and feature engineering hints."
        ),
        tags=["eda", "data-analysis", "pandas", "statistics"],
    )

    agent_card = AgentCard(
        name="EDA Agent",
        description=(
            "Deterministic data analyst. Runs real pandas/numpy analysis — no LLM needed. "
            "Provides null rates, correlations, distribution stats, and FE opportunities "
            "to other agents so they can make data-driven decisions."
        ),
        url=f"http://{args.host}:{args.port}/",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )

    handler = DefaultRequestHandler(
        agent_executor=Executor(),
        task_store=InMemoryTaskStore(),
    )
    app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=handler,
        max_content_length=None,
    )
    uvicorn.run(app.build(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
