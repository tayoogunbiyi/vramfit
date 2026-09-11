#!/usr/bin/env bash
# Start one pinned model under vLLM and wait until it serves.
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: bash validation/serve.sh MODEL REVISION RUN_DIR" >&2
  exit 2
fi
model=$1
revision=$2
run_dir=$3
port=${PORT:-8000}
if [[ ! $revision =~ ^[0-9a-f]{40}$ ]]; then
  echo "REVISION must be a full model commit SHA." >&2
  exit 2
fi
if [[ -e $run_dir ]]; then
  echo "Refusing to overwrite $run_dir" >&2
  exit 1
fi
mkdir -p -- "$run_dir"
run_dir=$(cd -- "$run_dir" && pwd)


command=(vllm serve "$model"
  --revision "$revision"
  --port "$port"
  --dtype bfloat16
  --kv-cache-dtype auto
  --max-model-len 8192
  --max-num-batched-tokens 8192
  --max-num-seqs 8
  --gpu-memory-utilization 0.90
  --no-enable-prefix-caching
  --enable-log-requests)

printf '%q ' "${command[@]}" > "$run_dir/launch-command.txt"
printf '\n' >> "$run_dir/launch-command.txt"

nohup "${command[@]}" > "$run_dir/server.log" 2>&1 &
pid=$!
echo "$pid" > "$run_dir/server.pid"

deadline=$((SECONDS + ${STARTUP_TIMEOUT:-900}))
until curl -fsS "http://localhost:$port/health" > /dev/null 2>&1; do
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "vLLM exited during startup; see $run_dir/server.log" >&2
    exit 1
  fi
  if (( SECONDS >= deadline )); then
    echo "vLLM did not become ready within ${STARTUP_TIMEOUT:-900}s; see $run_dir/server.log" >&2
    kill "$pid" 2>/dev/null || true
    exit 1
  fi
  sleep 2
done

echo "Server ready on port $port (pid $pid); log in $run_dir/server.log"
echo "Stop before loading a new model with: kill $pid"
