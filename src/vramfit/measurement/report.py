"""Portable, plain-data summaries suitable for subsequent findings analysis."""

import csv
import statistics


def summarize(report):
    groups = {}
    for phase in report['phases']:
        if phase['kind'] != 'measured':
            continue
        key = (phase['prompt_length'], phase['max_output_length'], phase['target_concurrency'])
        groups.setdefault(key, []).append(phase)
    rows = []
    for (prompt, output, concurrency), phases in groups.items():
        success = [phase for phase in phases if phase['status'] == 'ok']
        peaks = [phase['observations']['sampled_device_peak_bytes'] for phase in success
                 if phase['observations']['sampled_device_peak_bytes'] is not None]
        running = [phase['observations']['observed_running_requests_max'] for phase in success
                   if phase['observations']['observed_running_requests_max'] is not None]
        estimates = {phase['estimate']['known_memory_bytes'] for phase in phases}
        rows.append({
            'engine': report['engine'], 'synthetic': report['synthetic'],
            'prompt_length': prompt, 'max_output_length': output, 'target_concurrency': concurrency,
            'successful_repetitions': len(success), 'failed_repetitions': len(phases) - len(success),
            'known_memory_bytes': next(iter(estimates)) if len(estimates) == 1 else None,
            'sampled_device_peak_bytes_max': max(peaks) if peaks else None,
            'sampled_device_peak_bytes_min': min(peaks) if peaks else None,
            'observed_running_requests_max': max(running) if running else None,
            'batch_duration_seconds_median': statistics.median(p['duration_seconds'] for p in success) if success else None,
            'output_tokens_per_second_median': statistics.median(p['output_tokens_per_second'] for p in success) if success else None,
        })
    return rows


def write_summary(directory, report):
    rows = summarize(report)
    with (directory / 'summary.csv').open('w', newline='') as file:
        fields = list(rows[0]) if rows else ['engine', 'synthetic', 'successful_repetitions']
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    def gib(value):
        return 'unavailable' if value is None else f'{value / 1024**3:.4f}'
    lines = [f"# Measurement: {report['engine']}", '',
             f"Status: **{report['status']}**. Synthetic: **{report['synthetic']}**. Mode: **{report['measurement_mode']}**.", '',
             'Peaks are sampled device totals including the allocated KV pool; they are not weight/KV estimation error.', '',
             '| Prompt / output | Submitted concurrency | Successful repeats | Known weights + KV (GiB) | Sampled device peak (GiB) |',
             '|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['prompt_length']} / {row['max_output_length']} | {row['target_concurrency']} | "
                     f"{row['successful_repetitions']} | {gib(row['known_memory_bytes'])} | "
                     f"{gib(row['sampled_device_peak_bytes_max'])} |")
    lines += ['', '## Limitations', ''] + [f'- {item}' for item in report['limitations']]
    (directory / 'summary.md').write_text('\n'.join(lines) + '\n')
