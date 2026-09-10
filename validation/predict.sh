#!/usr/bin/env bash
# Regenerate the three BF16 baseline reports for one pinned model.
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "Usage: bash validation/predict.sh MODEL REVISION OUTPUT_DIR VRAM_GIB [HEADROOM_PERCENT=10]" >&2
  exit 2
fi
model=$1
revision=$2
output_dir=$3
vram=$4
headroom=${5:-10}
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [[ ! $revision =~ ^[0-9a-f]{40}$ ]]; then
  echo "REVISION must be a full model commit SHA." >&2
  exit 2
fi
if [[ -e $output_dir ]]; then
  echo "Refusing to overwrite $output_dir" >&2
  exit 1
fi
if [[ -n $(git -C "$repo_dir" status --porcelain -- src pyproject.toml uv.lock) ]]; then
  echo "Commit estimator/dependency changes before recording predictions." >&2
  exit 1
fi
mkdir -p -- "$output_dir"
output_dir=$(cd -- "$output_dir" && pwd)
cd -- "$repo_dir"
git rev-parse HEAD > "$output_dir/estimator-commit.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$output_dir/created-at.txt"
while read -r case_name concurrency prompt output; do
  command=(uv run --frozen vramfit "$model" --revision "$revision"
    --vram "$vram" --dtype bfloat16 --headroom "$headroom"
    --target-concurrency "$concurrency" --prompt-length "$prompt"
    --max-output-length "$output" --detailed)
  printf '%q ' "${command[@]}" >> "$output_dir/commands.sh"
  printf '\n' >> "$output_dir/commands.sh"
  if ! "${command[@]}" > "$output_dir/$case_name.txt" 2> "$output_dir/$case_name.stderr.txt"; then
    echo "Case $case_name failed; see $output_dir/$case_name.stderr.txt" >&2
    exit 1
  fi
  echo "Saved case $case_name"
done <<'CASES'
A 1 512 128
B 8 512 128
C 8 4096 256
CASES

# Extract exact byte counts from the reports; do not duplicate estimator formulas.
uv run --frozen python - "$output_dir" "$model" "$revision" "$vram" "$headroom" <<'PY'
import csv
from pathlib import Path
import re
import sys

directory, model, revision, vram, headroom = sys.argv[1:]
directory = Path(directory)
rows = []
for case, concurrency, prompt, output in [('A', 1, 512, 128), ('B', 8, 512, 128), ('C', 8, 4096, 256)]:
    text = (directory / f'{case}.txt').read_text()
    if f'Resolved revision: {revision}' not in text:
        raise SystemExit(f'Case {case}: revision mismatch')
    row = dict(model_id=model, revision=revision,
               estimator_commit=(directory / 'estimator-commit.txt').read_text().strip(),
               case=case, concurrency=concurrency, prompt_tokens=prompt,
               output_tokens=output, resident_tokens=concurrency * (prompt + output),
               weight_dtype='bfloat16', kv_dtype='bfloat16', physical_vram_gib=vram,
               headroom_percent=headroom, residency='full', overhead_gib='unmodelled')
    for field, label in [
        ('weights_gib', 'Learned weight memory'), ('kv_gib', 'Workload KV memory'),
        ('known_total_gib', 'Known memory (weights + KV)'),
        ('safety_margin_gib', f'Reserved headroom ({float(headroom):g}%)'),
        ('usable_vram_gib', 'Usable VRAM'),
        ('remaining_budget_gib', 'Remaining known budget (negative means deficit)'),
    ]:
        match = re.search(r'^' + re.escape(label) + r': .*\(([-\d,]+) bytes\)$', text, re.M)
        if not match:
            raise SystemExit(f'Case {case}: missing numeric component {label}')
        row[field] = format(int(match[1].replace(',', '')) / 2**30, '.10f')
    row['output_file'] = f'{case}.txt'
    rows.append(row)
with (directory / 'predictions.csv').open('w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=rows[0])
    writer.writeheader()
    writer.writerows(rows)
PY
