#!/usr/bin/env python3
"""Summarize one Qwen boundary experiment locally; residency conclusions need review."""
import argparse
import datetime as dt
import json
import math
import re
from pathlib import Path

BYTES_PER_BLOCK = 147456 * 16
CASES = {'cache-2g': ['warmup', 'below-1', 'above', 'below-2'],
         'cache-4g': ['control-warmup', 'control-four']}


def predictions():
    return '''# Frozen boundary predictions

Qwen3-4B BF16: 147,456 bytes/token, 16-token blocks, 4,104 actual prompt
and 256 output tokens. Rounded requirement: 273 blocks/sequence.

| Cache | Total blocks | Usable (one null block excluded) |
| --- | ---: | ---: |
| 2 GiB | 910 | 909 |
| 4 GiB | 1820 | 1819 |

Three sequences require 819 blocks; four require 1,092. Only three fit
fully in 2 GiB. Four fit in the 4 GiB control. This is a cache boundary,
not a physical GPU capacity or total runtime-memory prediction.
'''


def check_layout(path, pool_bytes):
    lines = [l for l in path.read_text().splitlines()
             if l.startswith('vllm:cache_config_info{')]
    if len(lines) != 1:
        raise ValueError('Expected one engine cache_config_info sample')
    fields = dict(re.findall(r'(\w+)="([^"]*)"', lines[0]))
    # Byte-count fields render with a trailing ".0" in real vLLM 0.11 output (confirmed
    # against validation/runs/Qwen3-4B/*/metrics-start.txt's "swap_space_bytes":
    # "4294967296.0"), while block counts render as plain integers. Compare numerically
    # so a benign float-vs-int rendering difference doesn't look like a layout mismatch.
    expected_numeric = {'block_size': 16, 'num_gpu_blocks': pool_bytes // BYTES_PER_BLOCK,
                        'kv_cache_memory_bytes': pool_bytes}
    for key, expected in expected_numeric.items():
        try:
            observed = float(fields[key])
        except (KeyError, ValueError):
            raise ValueError(f'{key}: expected {expected}, observed {fields.get(key)!r}')
        if observed != expected:
            raise ValueError(f'{key}: expected {expected}, observed {fields[key]}')
    if fields.get('enable_prefix_caching') != 'False':
        raise ValueError('enable_prefix_caching: expected False, observed '
                        f'{fields.get("enable_prefix_caching")!r}')
    return int(float(fields['num_gpu_blocks']))


def counter(path):
    values = re.findall(r'^vllm:num_preemptions_total(?:\{[^}]*\})?\s+(\S+)',
                        path.read_text(), re.M)
    if len(values) != 1:
        raise ValueError('Missing/ambiguous preemption counter')
    value = float(values[0])
    if not math.isfinite(value) or value < 0:
        raise ValueError('Invalid preemption counter')
    return value


def case_summary(path, total_blocks):
    status = dict(l.split('=', 1) for l in (path/'status.txt').read_text().splitlines())
    b = json.loads((path/'benchmarks.json').read_text())['benchmarks'][0]
    m = b['metrics']; totals = m['request_totals']
    if status.get('guidellm_exit') != '0' or status.get('drained') != 'true':
        raise ValueError('Case failed or did not drain; inspect status.txt')
    if totals['successful'] <= 0 or totals['errored'] != 0:
        raise ValueError('No successful requests or nonzero request errors')
    expected_seconds = 30 if 'warmup' in path.name else 180
    if b['duration'] != expected_seconds:
        raise ValueError('Unexpected benchmark duration')
    for key, expected in [('prompt_token_count', 4104), ('output_token_count', 256)]:
        if any(m[key]['successful'][k] != expected for k in ['min', 'max']):
            raise ValueError(f'Unexpected {key}; inspect actual request lengths')
    rows = []; missing = 0
    for line in (path/'metrics.jsonl').read_text().splitlines():
        try:
            r = json.loads(line)
            timestamp = dt.datetime.fromisoformat(r['timestamp'].replace('Z', '+00:00')).timestamp()
            if not b['start_time'] <= timestamp <= b['end_time']:
                continue
            keys = ['num_requests_running', 'num_requests_waiting', 'kv_cache_usage_perc',
                    'num_preemptions_total']
            if r.get('scrape_failed') or any(not isinstance(r.get(k), (float, int)) or
                    not math.isfinite(r[k]) or r[k] < 0 for k in keys) or r['kv_cache_usage_perc'] > 1:
                raise ValueError('Missing or invalid engine fields')
            rows.append(r)
        except (ValueError, TypeError, KeyError):
            missing += 1
    if not rows:
        raise ValueError('No valid samples within benchmark window')
    delta = counter(path/'metrics-end.txt') - counter(path/'metrics-start.txt')
    if delta < 0:
        raise ValueError('Preemption counter reset during case')
    peak = max(r['kv_cache_usage_perc'] for r in rows)
    queue = sum(r['num_requests_waiting'] > 0 for r in rows)/len(rows)
    running = max(r['num_requests_running'] for r in rows)
    latency = m['request_latency']['successful']['mean']
    rate = m['output_tokens_per_second']['successful']['mean']
    return (f'{totals["successful"]} / {totals["incomplete"]} | {running:g} | '
            f'{queue:.1%} | {round(peak*(total_blocks-1))} | {delta:g} | '
            f'{latency:.2f} | {rate:.1f} | {len(rows)} / {missing}')


def analyze(experiment):
    lines = ['# Boundary evidence review', '', predictions(),
             '## Measurements', '',
             'Queue fractions and peaks use only the benchmark time window. Counts are',
             'GuideLLM summary successes/incompletes; cutoff cancellations are not errors.',
             'Rates are successful-request aggregate means. Missing samples remain visible.', '']
    problems = []
    for launch, names in CASES.items():
        directory = experiment/launch
        lines += [f'### {launch}', '',
                  f'[Server log]({launch}/server.log) · [Layout gate]({launch}/layout-check.txt)', '',
                  '| Case | OK / incomplete | Peak running | Queue | Peak blocks | Preemptions Δ | Latency s | Output tok/s | Samples / missing |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        try:
            blocks = check_layout(directory/'layout.txt', (2 if launch == 'cache-2g' else 4)*2**30)
        except (OSError, ValueError) as e:
            problems.append(f'{launch}: {e}')
            continue
        for name in names:
            try:
                row = case_summary(directory/name, blocks)
                lines.append(f'| [{name}]({launch}/{name}/benchmarks.json) | {row} |')
            except (OSError, ValueError, KeyError, TypeError, IndexError) as e:
                problems.append(f'{launch}/{name}: {e}')
        lines += ['']
    lines += ['## Interpretation requires review', '',
              'No automatic claim that a boundary was observed is made. Check:', '',
              '- Below-1 and below-2: repeated three-sequence residency, near 819 blocks, no preemption.',
              '- Above: cache-linked pressure (near capacity and/or preemptions); queueing alone is insufficient.',
              '- Control-four: four-sequence residency near 1,092 blocks with pressure relieved.',
              '- Inspect metrics.jsonl and metrics-start/end.txt for each case; running counts alone do not prove full residency.',
              '- If those observations do not distinguish memory from scheduling, report unresolved.', '']
    if problems:
        lines += ['## Missing or invalid evidence', ''] + ['- '+p for p in problems] + ['']
    return '\n'.join(lines), bool(problems)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', nargs='?', type=Path)
    parser.add_argument('--predictions', action='store_true')
    parser.add_argument('--check-layout', nargs=2, metavar=('SNAPSHOT', 'BYTES'))
    args = parser.parse_args()
    if args.predictions:
        print(predictions()); return 0
    if args.check_layout:
        try:
            blocks = check_layout(Path(args.check_layout[0]), int(args.check_layout[1]))
            print(f'Layout verified: {blocks} total / {blocks-1} usable blocks'); return 0
        except (OSError, ValueError) as e:
            print(f'Layout rejected: {e}'); return 1
    if args.experiment is None or not args.experiment.is_dir():
        parser.error('Pass the fetched experiment directory')
    report, invalid = analyze(args.experiment)
    (args.experiment/'analysis.md').write_text(report)
    print(report)
    return int(invalid)


if __name__ == '__main__':
    raise SystemExit(main())
