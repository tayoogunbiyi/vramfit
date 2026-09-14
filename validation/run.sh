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

if [[ ! -e $run_dir/started-at.txt ]]; then
  date -u +%Y-%m-%dT%H:%M:%SZ > "$run_dir/started-at.txt"
fi

# Prints the successful request count from a guidellm JSON report, or nothing if unreadable.
successful_requests() {
  python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["benchmarks"][0]["metrics"]["request_totals"]["successful"])' "$1" 2>/dev/null || true
}

while read -r case_name streams prompt output seconds; do
  case_dir="$run_dir/$case_name"
  if [[ -e $case_dir ]]; then
    if [[ -e $case_dir/metrics-end.txt && $(successful_requests "$case_dir/benchmarks.json") =~ ^[1-9] ]]; then
      echo "Skipping case $case_name; already saved"
      continue
    fi
    echo "Refusing to overwrite incomplete $case_dir; move it aside to rerun the case" >&2
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

  # Requests cut off at the time limit are aborted, not finished; let the engine drain them
  # so the closing snapshot shows an idle engine.
  for _ in {1..30}; do
    idle=$(curl -fsS "http://localhost:$port/metrics" 2>/dev/null |
      awk '$1 ~ /^vllm:num_requests_(running|waiting)/ { seen = 1; busy += $2 } END { print (seen && busy == 0) }' || true)
    [[ $idle == 1 ]] && break
    sleep 1
  done
  curl -fsS "http://localhost:$port/metrics" > "$case_dir/metrics-end.txt" || true

  if (( status != 0 )); then
    echo "Case $case_name failed (exit $status); see $case_dir/guidellm.log" >&2
    exit 1
  fi
  # Incomplete requests do not count as errors, so max_errors cannot catch a case where
  # nothing finished.
  successful=$(successful_requests "$case_dir/benchmarks.json")
  if [[ ! $successful =~ ^[1-9] ]]; then
    echo "Case $case_name finished no requests (successful: ${successful:-unreadable}); see $case_dir/benchmarks.json" >&2
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
