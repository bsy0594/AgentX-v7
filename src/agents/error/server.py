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
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()

    skill = AgentSkill(
        id="ml-error-fix",
        name="ML Error Fixer",
        description=(
            "Receives failing ML Python code and its error traceback. "
            "Diagnoses the root cause and returns corrected, runnable code."
        ),
        tags=["debugging", "error-handling", "pandas", "sklearn", "ml"],
    )

    agent_card = AgentCard(
        name="ML Error Handler Agent",
        description=(
            "Specialized ML debugging agent. Expert in pandas CoW errors, "
            "sklearn API misuse, dtype mismatches, chained assignment, "
            "feature engineering bugs, and numpy/scipy issues."
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
