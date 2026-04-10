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
    parser.add_argument("--port", type=int, default=8012)
    args = parser.parse_args()

    skill = AgentSkill(
        id="ml-stacking-ensemble",
        name="ML Stacking Ensemble",
        description=(
            "Receives competition data (tar.gz). "
            "Builds a multi-model stacking ensemble: XGBoost + LightGBM + GradientBoosting "
            "base learners with CV-based target encoding + LogisticRegression meta-learner. "
            "Returns path to optimized submission CSV."
        ),
        tags=["stacking", "ensemble", "xgboost", "lightgbm", "meta-learning"],
    )

    agent_card = AgentCard(
        name="Stacking Ensemble Agent",
        description=(
            "Multi-model stacking agent. Runs 5-fold OOF predictions from three base models, "
            "applies CV-based target encoding, trains LogisticRegression meta-learner, "
            "and finds optimal classification threshold. Deterministic — no LLM calls."
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
