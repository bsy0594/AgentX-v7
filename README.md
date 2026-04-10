# AgentX

Multi-agent ML competition solver built on the [A2A (Agent-to-Agent) protocol](https://a2a-protocol.org/). AgentX coordinates 13 specialized agents to automatically solve Kaggle-style machine learning competitions from the [MLE-Bench](https://github.com/openai/mle-bench) benchmark.

## Architecture

AgentX uses a hub-and-spoke architecture where a central **orchestrator** coordinates **11 specialized sub-agents** and is evaluated by an external **evaluator** agent.

```
                          run_test.py
                               |
                               v
                    +---------------------+
                    |     Evaluator       |  port 9009
                    |  (MLE-Bench Green)  |
                    +---------------------+
                               |
                    A2A: tar.gz + instructions
                               |
                               v
                    +---------------------+
                    |    Orchestrator     |  port 8000
                    |   (ML Agent)       |
                    |   GPT-4o powered   |
                    +---------------------+
                       |    |    |    |
          +------------+    |    |    +-----------+
          |                 |    |                |
          v                 v    v                v
   +-----------+    +-----------+-----------+  +-----------+
   |    EDA    |    |  Critic   |   Error   |  | Stacking  |
   |   :8003   |    |   :8001   |   :8002   |  |   :8012   |
   +-----------+    +-----------+-----------+  +-----------+
   |   Model   |    |  Planner  |  CodeGen  |  | Ensemble  |
   |   :8005   |    |   :8010   |   :8011   |  |   :8006   |
   +-----------+    +-----------+-----------+  +-----------+
   |  Feature  |    | HyperTune | Threshold |
   |   :8008   |    |   :8007   |   :8009   |
   +-----------+    +-----------+-----------+
```

## Pipeline Flow

When a competition is submitted for evaluation, the following steps execute:

### Phase 1: Parallel Analysis (Steps 1-3)

| Agent | Port | Role |
|-------|------|------|
| **EDA** | 8003 | Analyzes dataset: column types, missing rates, class balance, distributions |
| **Model Selector** | 8005 | Runs cross-validation on multiple model families, returns best model + CV scores |
| **Feature Engineer** | 8008 | Generates feature engineering code snippets based on data structure |
| **Stacking** | 8012 | Runs the full stacking pipeline independently (see below) |

### Phase 2: LLM Planning (Steps 4-6)

1. **Orchestrator** forms a solution plan using GPT-4o (informed by EDA report)
2. **Critic Agent** (8001) reviews the plan and provides feedback
3. **Code Generator** (8011) produces a complete Python solution
4. Code is executed with a **self-repair loop** — if it fails, the **Error Handler** (8002) diagnoses the issue and the orchestrator retries (up to 2 repairs)

### Phase 3: Tuning & Fallback (Steps 7-10)

5. **Hyperparameter Tuner** (8007) runs Optuna on the best model from Phase 1
6. A **deterministic fallback** model is trained with tuned parameters
7. **Threshold Optimizer** (8009) searches for the optimal classification threshold

### Phase 4: Final Ensemble (Steps 11-12)

8. **Stacking Agent** result is collected (runs in parallel since Step 1)
9. **Ensemble Agent** (8006) blends all successful submissions via weighted majority vote:
   - Stacking output: weight **2.0** (highest priority)
   - Threshold-optimized: weight **1.5**
   - Fallback model: weight **1.0**
   - LLM-generated code: weight **1.0**

## Stacking Agent (Core ML Engine)

The stacking agent (`src/agents/stacking/`) is the primary scoring engine. It runs a complete ML pipeline without any LLM dependency for model training:

```
Competition Data (tar.gz)
       |
       v
  Task Detection -----> Binary / Multi-class / Regression
       |
       v
  LLM Feature Engineering (GPT-4o, 2-round iterative)
  |  Round 1: Generate features from data summary + correlations
  |  Quick CV: 3-fold LightGBM evaluation
  |  Round 2: Refine based on CV feedback
  |  Fallback: Generic FE (datetime, bool, ID-drop) if LLM fails
       |
       v
  CV Target Encoding (5-fold, leak-free, classification only)
       |
       v
  Impute + OrdinalEncode remaining categoricals
       |
       v
  5-Fold OOF Base Models
  |  XGBoost     (n=500, depth=6, lr=0.05)
  |  LightGBM    (n=500, depth=6, lr=0.05)
  |  GBM         (n=300, depth=5, lr=0.05)
  |  CatBoost    (iter=500, depth=6, lr=0.05)
       |
       v
  Meta-Learner (5-fold OOF, no leakage)
  |  Classification: LogisticRegression
  |  Regression: Ridge
       |
       v
  Threshold Search (binary only, 0.30-0.70)
       |
       v
  submission.csv (in original target format)
```

### LLM Feature Engineering

The stacking agent uses GPT-4o to generate **competition-specific** features at runtime. This keeps the agent general-purpose while adapting to each dataset:

1. A data summary is built (column types, missing rates, target correlations, sample rows)
2. GPT-4o generates a Python `fe(train, test)` function with domain-specific features
3. The generated code is executed and evaluated with quick 3-fold LightGBM CV
4. GPT-4o refines the features in a second round based on the CV score
5. The best feature set (LLM-generated or generic fallback) is selected

### Generic Fallback FE

If the LLM fails or produces worse features than the baseline, the agent falls back to generic feature engineering:
- Drop near-unique identifier columns (>97% unique, non-numeric)
- Parse datetime columns into year, month, day, hour, weekday components
- Convert bool-like text columns (`True`/`False`, `Yes`/`No`) to integers

## Agent Communication (A2A Protocol)

All agents communicate via the [A2A protocol](https://a2a-protocol.org/), an open standard for agent-to-agent communication over HTTP.

### Each Agent Has 3 Files

```
agent_name/
├── server.py      # Uvicorn HTTP server, publishes AgentCard
├── executor.py    # A2A request handler (message -> task -> events)
└── agent.py       # Business logic
```

### Message Types

| Type | Direction | Purpose |
|------|-----------|---------|
| `Message` (FilePart) | Evaluator -> Orchestrator | Competition data as tar.gz |
| `TaskStatusUpdateEvent` | Agent -> Caller | Progress updates, text results |
| `TaskArtifactUpdateEvent` | Agent -> Caller | File outputs (submission CSV) |

### Agent Discovery

Each agent publishes an `AgentCard` at `/.well-known/agent.json` describing its capabilities. The orchestrator discovers sub-agents by resolving their cards at startup.

## Scores

Performance on the **spaceship-titanic** competition from MLE-Bench:

| Date | Score | Medal | Notes |
|------|-------|-------|-------|
| 2026-04-07 | 0.82529 | Gold | v6 with competition-specific FE |
| 2026-04-08 | 0.82414 | Gold | LLM-driven FE (general agent) |
| 2026-04-08 | 0.81954 | Silver | LLM-driven FE (general agent) |

Medal thresholds: Gold >= 0.82066, Silver >= 0.81388, Bronze >= 0.80967

## Project Structure

```
AgentX/
├── src/
│   ├── evaluator/              # MLE-Bench Green evaluator (port 9009)
│   │   ├── server.py
│   │   ├── executor.py
│   │   ├── agent.py            # Downloads competition, grades submissions
│   │   ├── messenger.py        # A2A message helpers
│   │   └── instructions.txt    # Competition instructions template
│   ├── orchestrator/           # ML Orchestrator (port 8000)
│   │   ├── server.py
│   │   ├── executor.py
│   │   └── agent.py            # GPT-4o powered coordinator
│   └── agents/                 # 11 specialized sub-agents
│       ├── critic/      (8001) # Reviews solution plans
│       ├── error/       (8002) # Diagnoses execution failures
│       ├── eda/         (8003) # Exploratory data analysis
│       ├── model/       (8005) # Cross-validation model selection
│       ├── ensemble/    (8006) # Weighted submission blending
│       ├── hypertune/   (8007) # Optuna hyperparameter tuning
│       ├── feature/     (8008) # Feature engineering code generation
│       ├── threshold/   (8009) # Classification threshold optimization
│       ├── planner/     (8010) # Solution strategy planning
│       ├── codegen/     (8011) # GPT-4o code generation
│       └── stacking/    (8012) # Multi-model stacking ensemble
├── tests/                      # Test suite
├── docs/                       # Documentation and reference PDFs
├── run_test.py                 # Evaluation test harness
├── start_all.sh                # Start all 13 services
├── stop_all.sh                 # Stop all services
├── pyproject.toml              # Project dependencies
└── .gitignore
```

## Quick Start

### Prerequisites

- Python 3.11+ with conda
- conda environment `test` with dependencies installed
- OpenAI API key at the configured path

### Install Dependencies

```bash
conda create -n test python=3.11
conda activate test
pip install -e ".[test]"
```

### Run

```bash
# Start all 13 agent services
bash start_all.sh test

# Run evaluation on spaceship-titanic
conda run -n test python run_test.py

# Stop all services
bash stop_all.sh
```

### Port Map

| Port | Service | Type |
|------|---------|------|
| 9009 | MLE-Bench Evaluator | Evaluator |
| 8000 | ML Orchestrator | Orchestrator |
| 8001 | Critic Agent | Sub-agent |
| 8002 | Error Handler | Sub-agent |
| 8003 | EDA Agent | Sub-agent |
| 8005 | Model Selector | Sub-agent |
| 8006 | Ensemble Agent | Sub-agent |
| 8007 | Hyperparameter Tuner | Sub-agent |
| 8008 | Feature Engineer | Sub-agent |
| 8009 | Threshold Optimizer | Sub-agent |
| 8010 | Planner Agent | Sub-agent |
| 8011 | Code Generator | Sub-agent |
| 8012 | Stacking Agent | Sub-agent |

## Key Design Decisions

1. **Multi-agent over monolithic**: Each agent is independently deployable, testable, and replaceable. The stacking agent can be swapped without touching the orchestrator.

2. **LLM for features, not for modeling**: GPT-4o generates feature engineering code, but model training is deterministic (XGBoost, LightGBM, CatBoost with fixed hyperparameters). This gives the best of both worlds: adaptive features + reproducible training.

3. **Ensemble as safety net**: Even if the LLM-generated code fails completely, the stacking agent provides a strong deterministic baseline. The weighted ensemble ensures the best available submission is always selected.

4. **A2A protocol**: Standard HTTP-based communication means agents can be written in any language, deployed anywhere, and replaced independently. No proprietary RPC or message queue required.
