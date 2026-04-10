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
    parser.add_argument("--port", type=int, default=8008)
    args = parser.parse_args()

    skill = AgentSkill(
        id="feature-engineering",
        name="Feature Engineering",
        description=(
            "Receives competition data (tar.gz). Returns an enhanced engineer() function "
            "and a feature engineering report with domain-specific features."
        ),
        tags=["feature-engineering", "pandas", "tabular"],
    )

    agent_card = AgentCard(
        name="Feature Engineer Agent",
        description=(
            "Deterministic feature engineer. Analyzes competition data and returns "
            "ready-to-use enhanced engineer() code with group features, log transforms, "
            "boolean casts, and interaction terms."
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
