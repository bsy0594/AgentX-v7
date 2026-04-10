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
    parser.add_argument("--port", type=int, default=8011)
    args = parser.parse_args()

    skill = AgentSkill(
        id="ml-code-generation",
        name="ML Code Generation",
        description=(
            "Receives a solution plan, critique, EDA report, FE code, and data paths. "
            "Returns complete runnable Python ML code using sklearn Pipeline."
        ),
        tags=["code-generation", "python", "sklearn", "xgboost"],
    )

    agent_card = AgentCard(
        name="Code Generator Agent",
        description=(
            "Single-responsibility code writer. Uses gpt-4o to generate complete, "
            "runnable ML Python code from a given plan and context. "
            "No planning — code only."
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
