#!/usr/bin/env bash
#
# model-sweep.sh - run the vramfit CLI over a roster of ~20 supported and
# unsupported Hugging Face models, assert outcomes, and write a full text report.
#
# Usage: scripts/model-sweep.sh [-o OUTFILE] [-v VRAM] [-d DTYPE] [-p PROMPT_LEN]
#                               [-m MAX_OUTPUT_LEN] [-h]
#   -o  report path (default: $VRAMFIT_SWEEP_OUT, else ${TMPDIR:-/tmp}/vramfit-model-sweep.txt)
#   -v  --vram GiB          (default: 80)
#   -d  --dtype             (default: bfloat16)
#   -p  --prompt-length     (default: 1024)
#   -m  --max-output-length (default: 512)
#   -h  show this help
# Exit: 0 = all assertions passed (access-dependent skips allowed),
#       1 = unexpected model outcomes, 2 = invocation/report setup failure.
#
# Env: VRAMFIT_CMD overrides how the CLI is invoked (default "uv run --project <repo> vramfit");
#      HF_TOKEN is used for gated repos, and read from <repo>/.env when unset.

set -u

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

if [[ -n "${VRAMFIT_CMD:-}" ]]; then
  VRAMFIT_CMD_STR=$VRAMFIT_CMD
  read -r -a VRAMFIT_CMD <<< "$VRAMFIT_CMD_STR"
else
  VRAMFIT_CMD=(uv run --project "$REPO_DIR" vramfit)
  VRAMFIT_CMD_STR="uv run --project $REPO_DIR vramfit"
fi

# Drop uv's benign "active venv does not match the project" warning; no-op otherwise.
strip_noise() { sed -e '/^warning: .*VIRTUAL_ENV=.*does not match the project environment/d'; }

if [[ -z "${HF_TOKEN:-}" && -f "$REPO_DIR/.env" ]]; then
  _tok=$(grep -E '^HF_TOKEN=' "$REPO_DIR/.env" | head -1 | cut -d= -f2-)
  [[ -n "$_tok" ]] && export HF_TOKEN="$_tok"
  unset _tok
fi

OUTFILE="${VRAMFIT_SWEEP_OUT:-${TMPDIR:-/tmp}/vramfit-model-sweep.txt}"
VRAM=80
DTYPE=bfloat16
PROMPT_LEN=1024
MAX_OUTPUT_LEN=512

while getopts ":o:v:d:p:m:h" opt; do
  case "$opt" in
    o) OUTFILE=$OPTARG ;;
    v) VRAM=$OPTARG ;;
    d) DTYPE=$OPTARG ;;
    p) PROMPT_LEN=$OPTARG ;;
    m) MAX_OUTPUT_LEN=$OPTARG ;;
    h) grep -E '^#( |$)' "$0" | sed -E 's/^# ?//'; exit 0 ;;
    \?) echo "Unknown option: -$OPTARG" >&2; exit 2 ;;
    :) echo "Option -$OPTARG requires an argument" >&2; exit 2 ;;
  esac
done
shift $((OPTIND - 1))
if (( $# )); then
  echo "Unexpected positional arguments: $*" >&2
  exit 2
fi

# Roster: label<TAB>model_id<TAB>expectation<TAB>adapter<TAB>rejection reason.
# Accept means a known estimate, NOT that the chosen workload must fit the GPU.
ROSTER=(
  # --- expected to be supported ---
  $'llama\tTinyLlama/TinyLlama-1.1B-Chat-v1.0\taccept\tllama\t-'
  $'llama (MHA, tied)\tHuggingFaceTB/SmolLM2-1.7B\taccept\tllama\t-'
  $'llama 3.1\tNousResearch/Meta-Llama-3.1-8B-Instruct\taccept\tllama\t-'
  $'llama 3.2\tunsloth/Llama-3.2-1B-Instruct\taccept\tllama\t-'
  $'llama (distill)\tdeepseek-ai/DeepSeek-R1-Distill-Llama-8B\taccept\tllama\t-'
  $'llama (Falcon3)\ttiiuae/Falcon3-7B-Base\taccept\tllama\t-'
  $'qwen2\tQwen/Qwen2.5-0.5B\taccept\tqwen2\t-'
  $'qwen2\tQwen/Qwen2.5-7B-Instruct\taccept\tqwen2\t-'
  $'qwen2 (QwQ)\tQwen/QwQ-32B\taccept\tqwen2\t-'
  $'qwen3\tQwen/Qwen3-0.6B\taccept\tqwen3\t-'
  $'qwen3 (head_dim != h/n)\tQwen/Qwen3-4B\taccept\tqwen3\t-'
  $'mistral\tmistralai/Mistral-7B-Instruct-v0.2\taccept\tmistral\t-'
  $'mistral (Nemo)\tmistralai/Mistral-Nemo-Instruct-2407\taccept\tmistral\t-'

  # --- expected to be rejected ---
  $'quantized\tQwen/Qwen2.5-7B-Instruct-AWQ\treject\tqwen2\tQuantized checkpoints are not supported.'
  $'dual-chunk attention\tQwen/Qwen2.5-7B-Instruct-1M\treject\tqwen2\tDual-chunk attention is not supported.'
  $'qwen3_moe\tQwen/Qwen3-30B-A3B\treject\tqwen3_moe\tNo adapter is registered for this model type.'
  $'sliding-window\tmistralai/Mistral-7B-v0.1\treject\tmistral\tSliding-window attention is not supported by this adapter.'
  $'mistral3 composite\tmistralai/Mistral-Small-3.2-24B-Instruct-2506\treject\tmistral3\tNo adapter is registered for this model type.'
  $'mixtral MoE\tmistralai/Mixtral-8x7B-Instruct-v0.1\treject\tmixtral\tNo adapter is registered for this model type.'

  # Supported architecture, but access depends on the caller's credentials.
  $'gated (Meta)\tmeta-llama/Llama-3.1-8B\tgated\tllama\t-'
)

# Assertions intentionally use the CLI's labelled output. Keep these checks in
# sync with presentation changes; empty/unknown output must never pass as fit.
has_line() { grep -Eq -- "$1" <<< "$output"; }

known_estimate() {
  (( status == 0 )) &&
    grep -Fxq -- "Hugging Face model: $model" <<< "$output" &&
    grep -Fxq -- "Adapter: $adapter (uniform full attention)" <<< "$output" &&
    has_line '^Resolved revision: [[:xdigit:]]{40}$' &&
    has_line '^Learned parameters: [1-9][0-9,]*$' &&
    has_line '^Parameter evidence: (hub_safetensors|safetensors_headers)$' &&
    has_line '^KV per token: [1-9][0-9,]* bytes$' &&
    has_line '^Known memory \(weights \+ KV\): [0-9]+\.[0-9]+ GiB \([1-9][0-9,]* bytes\)$' &&
    has_line '^Known workload fits: (yes, within|no, exceeds) the calculated budget$' &&
    has_line '^Theoretical maximum concurrency: [0-9][0-9,]*$'
}

assert_outcome() {
  verdict=FAIL
  detail="expected a known $adapter estimate; received exit $status or incomplete/unknown output"
  case "$expectation" in
    accept|gated)
      if known_estimate; then
        verdict=PASS
        detail="known $adapter estimate"
      elif [[ "$expectation" == gated ]] && (( status == 1 )) &&
           has_line "^Error: Could not .* for '$model' at '[^']+' \(HTTP (401|403)\)\."; then
        verdict=SKIP
        detail="repository access unavailable; architecture not verified"
      fi
      ;;
    reject)
      detail="expected unsupported $adapter: $reason; received exit $status or a different failure"
      if (( status == 1 )) &&
         has_line "^Error: Unsupported architecture '$adapter': " &&
         grep -Fq -- "$reason" <<< "$output"; then
        verdict=PASS
        detail="expected architecture rejection"
      fi
      ;;
    *) detail="invalid roster expectation: $expectation" ;;
  esac
}

version_output=$("${VRAMFIT_CMD[@]}" --version 2>&1)
if (( $? != 0 )); then
  printf 'Cannot invoke CLI: %s\n%s\n' "$VRAMFIT_CMD_STR" "$version_output" >&2
  exit 2
fi

RULE_HEAVY=$(printf '=%.0s' {1..100})
RULE_LIGHT=$(printf -- '-%.0s' {1..100})
RULE_HASH=$(printf '#%.0s' {1..100})

total=${#ROSTER[@]}
start_epoch=$SECONDS

{
  echo "$RULE_HASH"
  echo "# vramfit model sweep"
  echo "# generated : $(date +"%Y-%m-%dT%H:%M:%S%z")"
  echo "# host      : $(uname -srm)"
  echo "# repo      : $REPO_DIR"
  echo "# git       : $(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo n/a)"
  echo "# invoke    : $VRAMFIT_CMD_STR"
  echo "# cli       : $(printf '%s\n' "$version_output" | strip_noise)"
  echo "# args      : --vram $VRAM --dtype $DTYPE --prompt-length $PROMPT_LEN --max-output-length $MAX_OUTPUT_LEN"
  echo "# roster    : $total models"
  echo "# hf token  : $([[ -n "${HF_TOKEN:-}" ]] && echo present || echo absent)"
  echo "$RULE_HASH"
  echo
} > "$OUTFILE" || exit 2

STATUS_LINES=()
passed=0
failed=0
skipped=0
i=0
for entry in "${ROSTER[@]}"; do
  ((i++))
  IFS=$'\t' read -r label model expectation adapter reason <<< "$entry"
  printf '[%2d/%2d] %-24s %s\n' "$i" "$total" "$label" "$model" >&2

  run_start=$SECONDS
  output=$("${VRAMFIT_CMD[@]}" "$model" \
    --vram "$VRAM" --dtype "$DTYPE" \
    --prompt-length "$PROMPT_LEN" --max-output-length "$MAX_OUTPUT_LEN" 2>&1)
  status=$?
  run_secs=$((SECONDS - run_start))
  output=$(printf '%s\n' "$output" | strip_noise)
  assert_outcome
  case "$verdict" in
    PASS) ((passed+=1)) ;;
    FAIL) ((failed+=1)) ;;
    SKIP) ((skipped+=1)) ;;
  esac
  printf '        %s: %s\n' "$verdict" "$detail" >&2

  {
    echo "$RULE_HEAVY"
    printf '[%2d/%2d]  %s  |  %s\n' "$i" "$total" "$label" "$model"
    echo "$RULE_HEAVY"
    echo "\$ ${VRAMFIT_CMD[*]} $model --vram $VRAM --dtype $DTYPE --prompt-length $PROMPT_LEN --max-output-length $MAX_OUTPUT_LEN"
    echo "$RULE_LIGHT"
    echo "$output"
    echo "$RULE_LIGHT"
    echo "exit status : $status"
    echo "duration    : ${run_secs}s"
    echo "assertion   : $verdict — $detail"
    echo
    echo
  } >> "$OUTFILE" || exit 2

  STATUS_LINES+=("$(printf '  %-4s %-24s %-46s exit %d — %s' "$verdict" "$label" "$model" "$status" "$detail")")
done

{
  echo "$RULE_HASH"
  echo "# ASSERTIONS (PASS = expected outcome, FAIL = unexpected, SKIP = access unavailable)"
  echo "$RULE_HASH"
  printf '%s\n' "${STATUS_LINES[@]:-}"
  echo "$RULE_LIGHT"
  echo "models run : $total"
  echo "assertions : $passed passed, $failed failed, $skipped skipped"
  echo "elapsed    : $((SECONDS - start_epoch))s"
  echo "finished   : $(date +"%Y-%m-%dT%H:%M:%S%z")"
  echo "$RULE_HASH"
} >> "$OUTFILE" || exit 2

echo >&2
echo "Report written to: $OUTFILE" >&2
echo "Assertions: $passed passed, $failed failed, $skipped skipped" >&2
(( failed == 0 ))
