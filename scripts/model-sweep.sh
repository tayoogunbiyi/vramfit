#!/usr/bin/env bash
#
# model-sweep.sh - run the vramfit CLI over a roster of ~20 supported and
# unsupported Hugging Face models and write every full result to one text report.
#
# Usage: scripts/model-sweep.sh [-o OUTFILE] [-v VRAM] [-d DTYPE] [-p PROMPT_LEN]
#                               [-m MAX_OUTPUT_LEN] [-h]
#   -o  report path (default: $VRAMFIT_SWEEP_OUT, else ${TMPDIR:-/tmp}/vramfit-model-sweep.txt)
#   -v  --vram GiB          (default: 80)
#   -d  --dtype             (default: bfloat16)
#   -p  --prompt-length     (default: 1024)
#   -m  --max-output-length (default: 512)
#   -h  show this help
#
# Env: VRAMFIT_CMD overrides how the CLI is invoked (default "uv run --project <repo> vramfit");
#      HF_TOKEN is used for gated repos, and read from <repo>/.env when unset.

set -u

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

VRAMFIT_CMD_STR="${VRAMFIT_CMD:-uv run --project "$REPO_DIR" vramfit}"
read -r -a VRAMFIT_CMD <<< "$VRAMFIT_CMD_STR"

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

# Roster entries: "label<TAB>model_id". The label is only a note in the report.
ROSTER=(
  # --- expected to be supported ---
  $'llama\tTinyLlama/TinyLlama-1.1B-Chat-v1.0'
  $'llama (MHA, tied)\tHuggingFaceTB/SmolLM2-1.7B'
  $'llama 3.1\tNousResearch/Meta-Llama-3.1-8B-Instruct'
  $'llama 3.2\tunsloth/Llama-3.2-1B-Instruct'
  $'llama (distill)\tdeepseek-ai/DeepSeek-R1-Distill-Llama-8B'
  $'llama (Falcon3)\ttiiuae/Falcon3-7B-Base'
  $'qwen2\tQwen/Qwen2.5-0.5B'
  $'qwen2\tQwen/Qwen2.5-7B-Instruct'
  $'qwen2 (QwQ)\tQwen/QwQ-32B'
  $'qwen3\tQwen/Qwen3-0.6B'
  $'qwen3 (head_dim != h/n)\tQwen/Qwen3-4B'
  $'mistral\tmistralai/Mistral-7B-Instruct-v0.2'
  $'mistral (Nemo)\tmistralai/Mistral-Nemo-Instruct-2407'

  # --- expected to be rejected ---
  $'quantized\tQwen/Qwen2.5-7B-Instruct-AWQ'
  $'dual-chunk attention\tQwen/Qwen2.5-7B-Instruct-1M'
  $'qwen3_moe\tQwen/Qwen3-30B-A3B'
  $'sliding-window\tmistralai/Mistral-7B-v0.1'
  $'mistral3 composite\tmistralai/Mistral-Small-3.2-24B-Instruct-2506'
  $'mixtral MoE\tmistralai/Mixtral-8x7B-Instruct-v0.1'
  $'gated (Meta)\tmeta-llama/Llama-3.1-8B'
)

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
  echo "# cli       : $("${VRAMFIT_CMD[@]}" --version 2>&1 | strip_noise || echo n/a)"
  echo "# args      : --vram $VRAM --dtype $DTYPE --prompt-length $PROMPT_LEN --max-output-length $MAX_OUTPUT_LEN"
  echo "# roster    : $total models"
  echo "# hf token  : $([[ -n "${HF_TOKEN:-}" ]] && echo present || echo absent)"
  echo "$RULE_HASH"
  echo
} > "$OUTFILE"

STATUS_LINES=()
i=0
for entry in "${ROSTER[@]}"; do
  ((i++))
  IFS=$'\t' read -r label model <<< "$entry"
  printf '[%2d/%2d] %-24s %s\n' "$i" "$total" "$label" "$model" >&2

  run_start=$SECONDS
  output=$("${VRAMFIT_CMD[@]}" "$model" \
    --vram "$VRAM" --dtype "$DTYPE" \
    --prompt-length "$PROMPT_LEN" --max-output-length "$MAX_OUTPUT_LEN" 2>&1)
  status=$?
  run_secs=$((SECONDS - run_start))
  output=$(printf '%s\n' "$output" | strip_noise)

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
    echo
    echo
  } >> "$OUTFILE"

  STATUS_LINES+=("$(printf '  %-24s %-46s exit %d' "$label" "$model" "$status")")
done

{
  echo "$RULE_HASH"
  echo "# INDEX (exit 0 = estimate produced, non-zero = declined)"
  echo "$RULE_HASH"
  printf '%s\n' "${STATUS_LINES[@]:-}"
  echo "$RULE_LIGHT"
  echo "models run : $total"
  echo "elapsed    : $((SECONDS - start_epoch))s"
  echo "finished   : $(date +"%Y-%m-%dT%H:%M:%S%z")"
  echo "$RULE_HASH"
} >> "$OUTFILE"

echo >&2
echo "Report written to: $OUTFILE" >&2
