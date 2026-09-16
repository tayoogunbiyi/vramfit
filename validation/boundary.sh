#!/usr/bin/env bash
# Stage 6 capacity-boundary orchestrator. Probes a fixed, deliberately small KV-cache
# pool with concurrency below/at/above its predicted full-residency limit, then repeats
# the pressure case against a spacious control cache. See
# reports/script_exercise/06-capacity-boundary-experiment.md for the full protocol.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash validation/boundary.sh RUNS_DIR" >&2
  exit 2
fi
runs_dir=$1
port=${PORT:-8000}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
model="Qwen/Qwen3-4B"
revision="1cfa9a7208912126459214e8b04321603b3df60c"
protocol_version="stage6-boundary-v1"

mkdir -p -- "$runs_dir/boundary"
runs_dir=$(cd -- "$runs_dir" && pwd)

# A fresh invocation always gets a new experiment directory and never overwrites an
# earlier one. Reattachment to a still-running job happens one layer up in
# validation/pod.sh, by staying attached to this same process rather than restarting it.
experiment_dir=$(mktemp -d "$runs_dir/boundary/$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")

echo "Boundary experiment: $experiment_dir"

# shellcheck source=lib/case.sh
source "$script_dir/lib/case.sh"

{
  echo "protocol_version=$protocol_version"
  echo "script_commit=$(cat "$script_dir/../COMMIT.txt" 2>/dev/null || git -C "$script_dir/.." rev-parse HEAD)"
  echo "model=$model"
  echo "revision=$revision"
  echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$experiment_dir/manifest.txt"

manifest_line() {
  printf '%s\n' "$1" >> "$experiment_dir/manifest.txt"
}

for record in COMMIT.txt pip-freeze.txt nvidia-smi.txt; do
  cp "$runs_dir/$record" "$experiment_dir/$record"
done
python3 "$script_dir/analyze_boundary.py" --predictions > "$experiment_dir/predictions.md"
server_pid=
stop_server() {
  [[ -n $server_pid ]] || return 0
  # serve.sh and all its descendants share this dedicated process group, including
  # engines left behind when startup fails or the API process exits first.
  kill -- "-$server_pid" 2>/dev/null || true
  for _ in {1..60}; do
    kill -0 -- "-$server_pid" 2>/dev/null || break
    sleep 2
  done
  kill -9 -- "-$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
  for _ in {1..5}; do
    if ! ps -eo pgid=,stat= | awk -v pg="$server_pid" '$1 == pg && $2 !~ /^Z/ { alive=1 } END { exit !alive }'; then
      server_pid=
      return 0
    fi
    sleep 1
  done
  echo "Server process group did not stop; refusing another launch" >&2
  trap - EXIT
  exit 1
}
trap stop_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# run_launch LAUNCH_NAME KV_CACHE_BYTES CASE_TABLE
# CASE_TABLE lines: name streams prompt output seconds
# Drains are bounded to 120s here, well above the baseline default, since a launch
# under deliberate cache pressure is expected to take longer to settle.
run_launch() {
  local launch_name=$1 kv_bytes=$2 case_table=$3
  local launch_dir="$experiment_dir/$launch_name"

  manifest_line "launch=$launch_name kv_cache_bytes=$kv_bytes"
  echo "Starting $launch_name launch (KV_CACHE_BYTES=$kv_bytes)"
  KV_CACHE_BYTES=$kv_bytes setsid bash "$script_dir/serve.sh" "$model" "$revision" "$launch_dir" &
  server_pid=$!
  local startup_status=0
  wait "$server_pid" || startup_status=$?
  if (( startup_status != 0 )); then
    manifest_line "case=$launch_name-startup launch=$launch_name status=server-startup-failed"
    echo "$launch_name server failed to start; preserving evidence and continuing" >&2
    stop_server
    return 1
  fi

  if ! curl --max-time 5 -fsS "http://localhost:$port/v1/models" > /dev/null 2>&1; then
    manifest_line "case=$launch_name-startup launch=$launch_name status=server-not-responding"
    stop_server
    return 1
  fi

  curl --max-time 5 -fsS "http://localhost:$port/metrics" > "$launch_dir/layout.txt" || { stop_server; return 1; }
  if ! python3 "$script_dir/analyze_boundary.py" --check-layout "$launch_dir/layout.txt" "$kv_bytes" > "$launch_dir/layout-check.txt" 2>&1; then
    manifest_line "launch=$launch_name status=layout-mismatch"
    stop_server
    return 1
  fi
  nvidia-smi > "$launch_dir/gpu-before.txt"
  local overall_status=0 case_name streams prompt output seconds case_dir
  while read -r case_name streams prompt output seconds; do
    [[ -n $case_name ]] || continue
    case_dir="$launch_dir/$case_name"
    mkdir -p -- "$case_dir"
    if METRICS_INTERVAL=${BOUNDARY_METRICS_INTERVAL:-0.5} DRAIN_TIMEOUT=${BOUNDARY_DRAIN_TIMEOUT:-120} \
        vramfit_run_case "$case_dir" "$model" "$port" "$streams" "$prompt" "$output" "$seconds"; then
      manifest_line "case=$case_name launch=$launch_name status=ok"
      echo "Saved $launch_name/$case_name"
    else
      overall_status=1
      if grep -q '^drained=false$' "$case_dir/status.txt" 2>/dev/null; then
        manifest_line "case=$case_name launch=$launch_name status=drain-timeout"
        echo "$launch_name did not drain after $case_name; stopping this launch, evidence preserved" >&2
        stop_server
        return 1
      fi
      manifest_line "case=$case_name launch=$launch_name status=failed"
    fi
  done <<< "$case_table"

  stop_server
  nvidia-smi > "$launch_dir/gpu-after.txt"
  return "$overall_status"
}

main_status=0
run_launch cache-2g 2147483648 "$(cat <<'CASES'
warmup 1 4096 256 30
below-1 3 4096 256 180
above 4 4096 256 180
below-2 3 4096 256 180
CASES
)" || main_status=$?

control_status=0
run_launch cache-4g 4294967296 "$(cat <<'CASES'
control-warmup 1 4096 256 30
control-four 4 4096 256 180
CASES
)" || control_status=$?

manifest_line "finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
# Ordinary completion marker: the orchestrator ran end to end. It says nothing about
# whether the boundary was actually observed; that classification is a separate,
# explicit step over the fetched evidence (validation/analyze_boundary.py).
date -u +%Y-%m-%dT%H:%M:%SZ > "$experiment_dir/finished-at.txt"

echo "Boundary experiment saved to $experiment_dir"
if (( main_status != 0 || control_status != 0 )); then
  echo "One or more cases did not complete cleanly; see $experiment_dir/manifest.txt" >&2
  exit 1
fi
