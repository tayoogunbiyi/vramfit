#!/usr/bin/env bash
# Drive the GPU validation on a rented pod over SSH from this machine.
set -euo pipefail

if [[ $# -ne 3 || ! $1 =~ ^(ship|setup|smoke|run|boundary|fetch)$ ]]; then
  echo "Usage: bash validation/pod.sh ship|setup|smoke|run|boundary|fetch HOST SSH_PORT" >&2
  exit 2
fi
subcommand=$1
host=$2
port=$3
if [[ -z $host || ! $port =~ ^[0-9]+$ ]]; then
  echo "HOST and a numeric SSH_PORT are required" >&2
  exit 2
fi
pod_root=/workspace/vramfit
repo=$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)

# Runs a bash script from stdin on the pod, with the remaining arguments as its arguments.
on_pod() {
  {
    # Arguments expand in the remote shell.
    # shellcheck disable=SC2016
    printf '%s\n' 'set -euo pipefail' 'mkdir -p "$1"' 'exec 9>"$1/control.lock"' 'flock -x 9'
    cat
  } | ssh -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
    -p "$port" "root@$host" "bash -s --$(printf ' %q' "$pod_root" "$@")"
}

# Copies the pod's validation/runs, including smoke/ and the pod records, into the local one.
fetch() {
  local staging previous
  staging=$(mktemp -d "$repo/validation/.fetch-XXXXXX")
  if ! on_pod <<'POD' | tar -xf - -C "$staging"
cd "$1/validation/runs" && tar -cf - .
POD
  then
    rm -rf -- "$staging"
    return 1
  fi
  # Replace the snapshot: merging tar output leaves stale success markers locally.
  if [[ -e $repo/validation/runs ]]; then
    previous=$(mktemp -d "$repo/validation/runs.old-XXXXXX")
    rmdir -- "$previous"
    mv -- "$repo/validation/runs" "$previous"
  fi
  mv -- "$staging" "$repo/validation/runs"
  echo "Fetched pod validation/runs into validation/runs"
}

# Starts run-all.sh on the pod detached from SSH, or reattaches to it, then streams its log
# until it exits.
run_on_pod() {
  local mode=$1 shown=0 failures=0 output status
  # Explicit return: callers use ||, which turns off set -e in here.
  on_pod "$mode" <<'POD' || return 1
set -euo pipefail
cd "$1"
mkdir -p jobs
exec 8>gpu.lock
if ! flock -n 8; then
  job=$(cat jobs/active-mode 2>/dev/null || true)
  if [[ $job == "$2" && ! -e jobs/$job.exit ]]; then
    echo "Reattaching to running $2 job"
    exit 0
  fi
  if [[ -n $job ]]; then
    echo "A $job job is still running on the pod" >&2
    exit 1
  fi
  echo "The GPU is busy; retry after the current operation finishes" >&2
  exit 1
fi
echo "$2" > jobs/active-mode
rm -f "jobs/$2.exit"
export HF_HOME=$1/.cache/huggingface PATH=$1/.venv/bin:$PATH
# boundary owns its own orchestration (validation/boundary.sh); smoke/full share run-all.sh.
nohup setsid bash -c '
if [[ $1 == boundary ]]; then
  bash validation/boundary.sh validation/runs
else
  bash validation/run-all.sh "$1"
fi
echo $? > "jobs/$1.exit"
' _ "$2" > "jobs/$2.log" 2>&1 < /dev/null 9>&- &
echo $! > "jobs/$2.pid"
echo "Started $2 job on the pod"
POD

  while true; do
    if output=$(on_pod "$mode" "$shown" <<'POD'
cd "$1/jobs"
# Check the exit file first (a completed process may remain a zombie), and
# recheck after liveness to handle a job exiting between those two checks.
if [[ -e $2.exit ]]; then
  status="exit $(cat "$2.exit")"
elif kill -0 "$(cat "$2.pid")" 2>/dev/null; then
  status=running
elif [[ -e $2.exit ]]; then
  status="exit $(cat "$2.exit")"
else
  status=lost
fi
# Read logs after status so a completed job includes its final lines.
tail -n "+$(($3 + 1))" "$2.log"
echo "@status $status"
POD
    ); then
      failures=0
      status=${output##*@status }
      output=${output%@status *}
      if [[ -n $output ]]; then
        printf '%s' "$output"
        shown=$((shown + $(printf '%s' "$output" | wc -l)))
      fi
      case $status in
        running) ;;
        "exit 0") return 0 ;;
        lost) echo "The $mode job died without recording an exit code" >&2; return 1 ;;
        *) echo "The $mode job failed ($status); see $pod_root/jobs/$mode.log" >&2; return 1 ;;
      esac
    elif (( ++failures >= 5 )); then
      echo "Lost SSH to the pod; the job keeps running there. Rerun to reattach." >&2
      return 1
    fi
    sleep "${POLL_INTERVAL:-30}"
  done
}

case $subcommand in
  ship)
    if [[ -n $(git -C "$repo" status --porcelain --untracked-files=all -- validation) ]]; then
      echo "validation/ has uncommitted changes; commit them so the pod runs a known commit" >&2
      exit 1
    fi
    commit=$(git -C "$repo" rev-parse HEAD)
    # The archive takes stdin, so this remote command is inline rather than a heredoc.
    # shellcheck disable=SC2016
    git -C "$repo" archive "$commit" validation |
      ssh -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
        -p "$port" "root@$host" "bash -c $(printf '%q' '
set -euo pipefail
mkdir -p "$1"
cd "$1"
exec 9>control.lock
flock -x 9
exec 8>gpu.lock
if ! flock -n 8; then
  if [[ $(cat COMMIT.txt) == "$2" ]]; then
    cat > /dev/null
    exit 0
  fi
  echo "Cannot ship a different commit while a job is running" >&2
  exit 1
fi
tar -xf -
echo "$2" > COMMIT.txt
') _ $pod_root $commit"
    echo "Shipped validation/ at $commit to $pod_root"
    ;;

  setup)
    on_pod <<'POD'
set -euo pipefail
cd "$1"
exec 8>gpu.lock
if ! flock -n 8; then
  echo "Job running; preserving its installed environment"
  exit 0
fi
mkdir -p validation/runs
for tool in curl nvidia-smi python3 timeout ps; do
  command -v "$tool" > /dev/null || { echo "$tool is missing on the pod" >&2; exit 1; }
done
nvidia-smi > validation/runs/nvidia-smi.txt
export UV_CACHE_DIR=$1/.cache/uv UV_PYTHON_INSTALL_DIR=$1/.cache/python
packages='vllm==0.11.0 guidellm==0.7.3 transformers==4.57.6'
if [[ $(cat .venv/installed 2>/dev/null || true) != "$packages" ]]; then
  python3 -m pip install --quiet uv
  # Rebuild incomplete installs; downloaded packages remain in the uv cache.
  python3 -m uv venv --clear --python 3.12 .venv
  # vLLM 0.11 uses tokenizer APIs removed in transformers 5.
  python3 -m uv pip install --python .venv/bin/python vllm==0.11.0 guidellm==0.7.3 transformers==4.57.6
  echo "$packages" > .venv/installed
fi
python3 -m uv pip freeze --python .venv/bin/python > validation/runs/pip-freeze.txt
cp COMMIT.txt validation/runs/COMMIT.txt
head -n 12 validation/runs/nvidia-smi.txt
echo "Pod ready; installed packages listed in validation/runs/pip-freeze.txt"
POD
    ;;

  # Results are fetched even when the job fails, so the logs can be read locally.
  smoke)
    status=0
    run_on_pod smoke || status=$?
    fetch
    exit "$status"
    ;;

  run)
    if [[ ${YES:-} != 1 ]]; then
      read -r -p "Run the full evaluation? Two models, about 30 minutes of GPU time. [y/N] " answer
      [[ $answer =~ ^[yY]$ ]] || { echo "Cancelled"; exit 1; }
    fi
    status=0
    run_on_pod full || status=$?
    fetch
    exit "$status"
    ;;

  boundary)
    if [[ ${YES:-} != 1 ]]; then
      read -r -p "Run the capacity-boundary experiment? One model, two server launches, about 30-45 minutes of GPU time. [y/N] " answer
      [[ $answer =~ ^[yY]$ ]] || { echo "Cancelled"; exit 1; }
    fi
    status=0
    run_on_pod boundary || status=$?
    fetch
    exit "$status"
    ;;

  fetch)
    fetch
    ;;
esac
