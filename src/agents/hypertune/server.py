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
    parser.add_argument("--port", type=int, default=8007)
    args = parser.parse_args()

    skill = AgentSkill(
        id="hyperparameter-tuning",
        name="Hyperparameter Tuning",
        description=(
            "Receives competition data (tar.gz) and a model name. "
            "Uses Optuna to find optimal hyperparameters. Returns tuned params as JSON."
        ),
        tags=["hyperparameter-tuning", "optuna", "xgboost", "lightgbm"],
    )

    agent_card = AgentCard(
        name="Hyperparameter Tuner Agent",
        description=(
            "Optuna-based hyperparameter optimizer. Runs 40 trials of Bayesian optimization "
            "on the actual competition data to find best model parameters. "
            "Falls back to curated defaults if Optuna is unavailable."
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
