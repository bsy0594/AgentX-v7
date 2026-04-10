#!/bin/bash
# Start all AgentX services
# Usage: bash start_all.sh [conda_env]

ENV=${1:-test}
BASE="$(cd "$(dirname "$0")" && pwd)"

declare -A services=(
    ["src/evaluator"]=9009
    ["src/orchestrator"]=8000
    ["src/agents/critic"]=8001
    ["src/agents/error"]=8002
    ["src/agents/eda"]=8003
    ["src/agents/model"]=8005
    ["src/agents/ensemble"]=8006
    ["src/agents/hypertune"]=8007
    ["src/agents/feature"]=8008
    ["src/agents/threshold"]=8009
    ["src/agents/planner"]=8010
    ["src/agents/codegen"]=8011
    ["src/agents/stacking"]=8012
)

echo "Starting AgentX services (env=$ENV)..."
for svc in "${!services[@]}"; do
    port=${services[$svc]}
    cd "$BASE/$svc" && conda run -n "$ENV" python server.py --port "$port" > /dev/null 2>&1 &
    echo "  $svc → port $port (PID $!)"
done

echo "Waiting for servers to start..."
sleep 8

echo "Status:"
for svc in "${!services[@]}"; do
    port=${services[$svc]}
    if netstat -ano 2>/dev/null | grep -q ":$port.*LISTENING"; then
        echo "  ✓ $svc → :$port"
    else
        echo "  ✗ $svc → :$port FAILED"
    fi
done
