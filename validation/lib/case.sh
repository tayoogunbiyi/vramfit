#!/usr/bin/env bash
# Shared per-case load runner. Meant to be sourced by validation/run.sh and the
# boundary orchestrator, not executed directly.
#
# vramfit_run_case writes evidence into an existing case directory and always
# records case_dir/status.txt, even when the case fails or does not drain, so a
# caller can preserve a failed point without losing later evidence. It never
# calls exit; callers decide what a nonzero return means for their run.
VRAMFIT_LIB_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# Prints the successful request count from a guidellm JSON report, or nothing if unreadable.
vramfit_successful_requests() {
  python3 -c 'import json, sys; t=json.load(open(sys.argv[1]))["benchmarks"][0]["metrics"]["request_totals"]; assert t["errored"] == 0; print(t["successful"])' "$1" 2>/dev/null || true
}

# vramfit_run_case CASE_DIR MODEL PORT STREAMS PROMPT OUTPUT SECONDS
#
# DRAIN_TIMEOUT (seconds, default 30) bounds how long this waits for the engine
# to report zero running/waiting requests after the load finishes. Metrics keep
# sampling through that wait, so the drain phase is not a gap in the series.
# Returns 0 only when guidellm exited 0, the engine drained within the bound,
# and at least one request succeeded.
vramfit_run_case() (
  local case_dir=$1 case_model=$2 case_port=$3 streams=$4 prompt=$5 output=$6 seconds=$7
  local drain_timeout=${DRAIN_TIMEOUT:-30}
  local metrics_pid="" status=130 deadline remaining idle=0 successful=0 command
  [[ $drain_timeout =~ ^[1-9][0-9]*$ ]] || { echo "Invalid DRAIN_TIMEOUT" >&2; return 2; }
  # shellcheck disable=SC2329 # Invoked by the EXIT trap.
  cleanup_case() {
    [[ -z $metrics_pid ]] || { kill "$metrics_pid" 2>/dev/null || true; wait "$metrics_pid" 2>/dev/null || true; }
    printf "guidellm_exit=%s\ndrained=%s\ndrain_timeout_seconds=%s\nsuccessful_requests=%s\n" \
      "$status" "$([[ $idle == 1 ]] && echo true || echo false)" "$drain_timeout" "${successful:-0}" > "$case_dir/status.txt"
  }
  trap cleanup_case EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM

  curl --max-time 5 -fsS "http://localhost:$case_port/metrics" > "$case_dir/metrics-start.txt" || true

  bash "$VRAMFIT_LIB_DIR/../metrics.sh" "$case_dir/metrics.jsonl" &
  metrics_pid=$!

  command=(guidellm run
    --backend "kind=openai_http,target=http://localhost:$case_port,model=$case_model"
    --data "kind=synthetic_text,prompt_tokens=$prompt,output_tokens=$output"
    --profile "kind=concurrent,streams=$streams"
    --constraint "kind=max_duration,seconds=$seconds"
    --constraint "kind=max_errors,count=1"
    --seed "kind=static,value=42"
    --output "kind=json,path=$case_dir/benchmarks.json"
    --output "kind=csv,path=$case_dir/benchmarks.csv"
    --disable-console-interactive)
  printf '%q ' "${command[@]}" > "$case_dir/command.txt"
  printf '\n' >> "$case_dir/command.txt"

  status=0
  "${command[@]}" < /dev/null > "$case_dir/guidellm.log" 2>&1 || status=$?

  # Requests cut off at the time limit are aborted, not finished; let the engine drain
  # them, bounded so a stuck engine cannot hang the run. Metrics stay alive throughout.
  deadline=$((SECONDS + drain_timeout))
  while (( SECONDS < deadline )); do
    remaining=$((deadline - SECONDS))
    (( remaining <= 5 )) || remaining=5
    idle=$(curl --max-time "$remaining" -fsS "http://localhost:$case_port/metrics" 2>/dev/null |
      awk '$1 ~ /^vllm:num_requests_(running|waiting)/ { seen = 1; busy += $2 } END { print (seen && busy == 0) }' || true)
    [[ $idle == 1 ]] && break
    sleep 1
  done
  kill "$metrics_pid" 2>/dev/null || true
  wait "$metrics_pid" 2>/dev/null || true
  metrics_pid=
  curl --max-time 5 -fsS "http://localhost:$case_port/metrics" > "$case_dir/metrics-end.txt" || true

  successful=$(vramfit_successful_requests "$case_dir/benchmarks.json")

  if (( status != 0 )); then
    echo "Case $(basename -- "$case_dir") failed (exit $status); see $case_dir/guidellm.log" >&2
    return 1
  fi
  if [[ $idle != 1 ]]; then
    echo "Case $(basename -- "$case_dir") did not drain within ${drain_timeout}s; see $case_dir/metrics-end.txt" >&2
    return 1
  fi
  # Incomplete requests do not count as errors, so max_errors cannot catch a case where
  # nothing finished.
  if [[ ! $successful =~ ^[1-9] ]]; then
    echo "Case $(basename -- "$case_dir") finished no requests (successful: ${successful:-unreadable}); see $case_dir/benchmarks.json" >&2
    return 1
  fi
  return 0
)
