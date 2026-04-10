#!/bin/bash
# Start AgentX v7 participant services: orchestrator + all sub-agents
# Usage: bash start_participant.sh [--host HOST] [--port PORT] [--card-url URL]

BASE="$(cd "$(dirname "$0")" && pwd)"
ORCHESTRATOR_ARGS="$@"

HOST="0.0.0.0"
while [[ $# -gt 0 ]]; do
  case $1 in
    --host) HOST="$2"; shift 2;;
    *) shift;;
  esac
done

echo "Starting AgentX v7 sub-agents (host=$HOST)..."

declare -A agents=(
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

for svc in "${!agents[@]}"; do
    port=${agents[$svc]}
    cd "$BASE/$svc" && python server.py --host "$HOST" --port "$port" &
    echo "  $svc → port $port (PID $!)"
done

echo "Waiting for sub-agents to start..."
sleep 5

echo "Starting orchestrator..."
cd "$BASE/src/orchestrator" && exec python server.py $ORCHESTRATOR_ARGS
