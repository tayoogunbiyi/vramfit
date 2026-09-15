#!/usr/bin/env bash
# Serve each pinned model in turn and run its cases. smoke runs Qwen3-4B with short cases.
set -euo pipefail

if [[ $# -ne 1 || ! $1 =~ ^(smoke|full)$ ]]; then
  echo "Usage: bash validation/run-all.sh smoke|full" >&2
  exit 2
fi
mode=$1
# Full evaluations must use the committed baseline table.
unset CASES_FILE
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
runs_dir=$script_dir/runs

models=(
  "Qwen/Qwen3-4B 1cfa9a7208912126459214e8b04321603b3df60c"
  "deepseek-ai/DeepSeek-R1-Distill-Llama-8B 6a6f4aa4197940add57724a7707d069478df56b1"
)
if [[ $mode == smoke ]]; then
  models=("${models[0]}")
  runs_dir=$runs_dir/smoke
  mkdir -p -- "$runs_dir"
  cat > "$runs_dir/cases.txt" <<'CASES'
warmup 1 64 8 10
A 1 512 128 20
B 8 512 128 20
C 8 4096 256 60
CASES
  export CASES_FILE=$runs_dir/cases.txt
fi

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
  server_pid=
}
trap stop_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for entry in "${models[@]}"; do
  read -r model revision <<< "$entry"
  run_dir=$runs_dir/${model##*/}
  if [[ $mode == full && -e $run_dir/finished-at.txt ]]; then
    if ! cmp -s "$script_dir/../COMMIT.txt" "$run_dir/COMMIT.txt"; then
      echo "Completed $run_dir belongs to a different commit; move it aside to rerun" >&2
      exit 1
    fi
    echo "Skipping $model; already finished"
    continue
  fi
  # Cases must share one server launch with its startup log, so a partial run starts over.
  if [[ -e $run_dir ]]; then
    aside=$(mktemp -d "$run_dir.old-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")
    rmdir -- "$aside"
    mv -- "$run_dir" "$aside"
    echo "Moved previous $run_dir to $aside"
  fi

  echo "Serving $model"
  setsid bash "$script_dir/serve.sh" "$model" "$revision" "$run_dir" &
  server_pid=$!
  startup_status=0
  wait "$server_pid" || startup_status=$?
  if [[ -d $run_dir ]]; then
    cp "$script_dir/../COMMIT.txt" "$run_dir/COMMIT.txt"
    for record in nvidia-smi.txt pip-freeze.txt; do
      cp "$script_dir/runs/$record" "$run_dir/$record"
    done
  fi
  (( startup_status == 0 )) || exit "$startup_status"
  bash "$script_dir/run.sh" "$model" "$run_dir"
  stop_server
done

echo "All $mode runs finished in $runs_dir"
