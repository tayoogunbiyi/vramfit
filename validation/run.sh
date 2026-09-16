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
# CASES_FILE replaces the case table below, e.g. for a short smoke run.
if [[ -n ${CASES_FILE:-} && ! -r $CASES_FILE ]]; then
  echo "CASES_FILE $CASES_FILE is not readable" >&2
  exit 1
fi
if ! curl -fsS "http://localhost:$port/v1/models" > /dev/null 2>&1; then
  echo "No server responding on http://localhost:$port" >&2
  exit 1
fi

if [[ ! -e $run_dir/started-at.txt ]]; then
  date -u +%Y-%m-%dT%H:%M:%SZ > "$run_dir/started-at.txt"
fi

# shellcheck source=lib/case.sh
source "$script_dir/lib/case.sh"

while read -r case_name streams prompt output seconds; do
  case_dir="$run_dir/$case_name"
  if [[ -e $case_dir ]]; then
    if [[ -e $case_dir/status.txt ]] && ! grep -q '^drained=true$' "$case_dir/status.txt"; then
      echo "Refusing to skip undrained $case_dir" >&2
      exit 1
    fi
    if [[ -e $case_dir/metrics-end.txt && $(vramfit_successful_requests "$case_dir/benchmarks.json") =~ ^[1-9] ]]; then
      echo "Skipping case $case_name; already saved"
      continue
    fi
    echo "Refusing to overwrite incomplete $case_dir; move it aside to rerun the case" >&2
    exit 1
  fi
  mkdir -p -- "$case_dir"

  # Full snapshots bracket the case: cache_config_info carries num_gpu_blocks and
  # block_size (allocated KV capacity), and the counters give a per-case delta.
  vramfit_run_case "$case_dir" "$model" "$port" "$streams" "$prompt" "$output" "$seconds" || exit 1
  echo "Saved case $case_name"
done < <(cat -- "${CASES_FILE:-/dev/stdin}" <<'CASES'
warmup 1 512 128 30
A 1 512 128 180
B 8 512 128 180
C 8 4096 256 180
CASES
)

date -u +%Y-%m-%dT%H:%M:%SZ > "$run_dir/finished-at.txt"
echo "Done. $run_dir/warmup is the warm-up and is excluded from results."
