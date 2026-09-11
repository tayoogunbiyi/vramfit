#!/usr/bin/env bash
# Run the warm-up and the three BF16 baseline cases against a running vLLM server.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: bash validation/run.sh MODEL RUN_DIR" >&2
  exit 2
fi
model=$1
run_dir=$2
port=${PORT:-8000}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

if [[ ! -d $run_dir ]]; then
  echo "$run_dir does not exist; start the server with validation/serve.sh first." >&2
  exit 1
fi
run_dir=$(cd -- "$run_dir" && pwd)
if ! curl -fsS "http://localhost:$port/v1/models" > /dev/null 2>&1; then
  echo "No server responding on http://localhost:$port" >&2
  exit 1
fi

date -u +%Y-%m-%dT%H:%M:%SZ > "$run_dir/started-at.txt"

while read -r case_name streams prompt output seconds; do
  case_dir="$run_dir/$case_name"
  if [[ -e $case_dir ]]; then
    echo "Refusing to overwrite $case_dir" >&2
    exit 1
  fi
  mkdir -p -- "$case_dir"

  # Full snapshots bracket the case: cache_config_info carries num_gpu_blocks and
  # block_size (allocated KV capacity), and the counters give a per-case delta.
  curl -fsS "http://localhost:$port/metrics" > "$case_dir/metrics-start.txt" || true

  bash "$script_dir/metrics.sh" "$case_dir/metrics.jsonl" &
  metrics_pid=$!
  trap 'kill "$metrics_pid" 2>/dev/null || true' EXIT

  command=(guidellm run
    --backend "kind=openai_http,target=http://localhost:$port,model=$model"
    --data "kind=synthetic_text,prompt_tokens=$prompt,output_tokens=$output"
    --profile "kind=concurrent,streams=$streams"
    --constraint "kind=max_duration,seconds=$seconds"
    --constraint kind=max_errors,count=1
    --seed kind=static,value=42
    --output "kind=json,path=$case_dir/benchmarks.json"
    --output "kind=csv,path=$case_dir/benchmarks.csv"
    --disable-console-interactive)
  printf '%q ' "${command[@]}" > "$case_dir/command.txt"
  printf '\n' >> "$case_dir/command.txt"

  status=0
  "${command[@]}" < /dev/null > "$case_dir/guidellm.log" 2>&1 || status=$?
  kill "$metrics_pid" 2>/dev/null || true
  wait "$metrics_pid" 2>/dev/null || true
  trap - EXIT
  curl -fsS "http://localhost:$port/metrics" > "$case_dir/metrics-end.txt" || true

  if (( status != 0 )); then
    echo "Case $case_name failed (exit $status); see $case_dir/guidellm.log" >&2
    exit 1
  fi
  echo "Saved case $case_name"
done <<'CASES'
warmup 1 512 128 30
A 1 512 128 180
B 8 512 128 180
C 8 4096 256 180
CASES

date -u +%Y-%m-%dT%H:%M:%SZ > "$run_dir/finished-at.txt"
echo "Done. $run_dir/warmup is the warm-up and is excluded from results."
