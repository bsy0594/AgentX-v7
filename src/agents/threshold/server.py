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
    parser.add_argument("--port", type=int, default=8009)
    args = parser.parse_args()

    skill = AgentSkill(
        id="threshold-optimization",
        name="Threshold Optimization",
        description=(
            "Receives competition data (tar.gz) and tuned model params. "
            "Finds optimal classification threshold via OOF CV probabilities. "
            "Returns an optimized submission CSV path."
        ),
        tags=["threshold", "classification", "optimization", "predict_proba"],
    )

    agent_card = AgentCard(
        name="Threshold Optimizer Agent",
        description=(
            "CV-based threshold optimizer. Trains model with predict_proba, "
            "sweeps thresholds 0.30-0.70 on OOF predictions to maximize accuracy, "
            "then applies optimal threshold to test predictions."
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
