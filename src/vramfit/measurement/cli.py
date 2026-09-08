"""Commands for preparing and executing the same experiment against either engine."""

import asyncio
import json
import os
from pathlib import Path

import click
import httpx

from vramfit.errors import ModelInspectionError
from vramfit.hub import ModelHubError
from vramfit.measurement.backends import endpoint
from vramfit.measurement.config import PILOT, prepare, read_json, validate
from vramfit.measurement.runner import Settings, headers, run
from vramfit.measurement.telemetry import validate_snapshot


@click.group()
def main():
    """Prepare exact-token workloads and measure a remote vLLM or SGLang server."""


@main.command('prepare')
@click.argument('model')
@click.option('--revision', default='main', show_default=True)
@click.option('--served-name', help='Name advertised by /v1/models; defaults to model ID.')
@click.option('--dtype', type=click.Choice(['bfloat16', 'float16', 'float32']), default='bfloat16')
@click.option('--vram', type=click.FloatRange(min=0, min_open=True), default=24.0, show_default=True)
@click.option('--headroom', type=click.FloatRange(min=0, max=100, max_open=True), default=10.0)
@click.option('--workloads', type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option('--output', type=click.Path(path_type=Path), required=True)
def prepare_command(model, revision, served_name, dtype, vram, headroom, workloads, output):
    """Resolve and save pinned metadata, without downloading weights."""
    try:
        if output.exists():
            raise ValueError('Output already exists; choose a new manifest path')
        document = prepare(model, revision, served_name, dtype, vram, headroom,
                           read_json(workloads) if workloads else PILOT)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x') as file:
            json.dump(document, file, indent=2, allow_nan=False)
            file.write('\n')
        click.echo(f"Prepared {model}@{document['model']['revision']} in {output}")
    except (ValueError, OSError, ModelHubError, ModelInspectionError) as exc:
        raise click.ClickException(str(exc)) from exc


def environment_metadata(path, manifest, engine):
    data = read_json(path)
    required = ('engine', 'engine_version', 'model_revision', 'dtype', 'kv_dtype',
                'tensor_parallel_size', 'prefix_caching', 'max_model_len', 'launch_command',
                'image_digest', 'gpu_model', 'cuda_version', 'pytorch_version',
                'graph_mode', 'memory_settings', 'scheduler_settings')
    if not isinstance(data, dict) or any(key not in data for key in required):
        raise ValueError('Environment metadata missing required fields; see examples/environment.json')
    if data['engine'] != engine or data['model_revision'] != manifest['model']['revision']:
        raise ValueError('Environment engine/revision differs from manifest or selected engine')
    if data['dtype'] != manifest['model']['dtype'] or data['kv_dtype'] != data['dtype']:
        raise ValueError('Environment weight/KV dtype must match the manifest')
    if type(data['tensor_parallel_size']) is not int or data['tensor_parallel_size'] != 1:
        raise ValueError('This pilot requires tensor_parallel_size=1')
    if data['prefix_caching'] is not False:
        raise ValueError('Disable prefix caching for this pilot')
    context = data['max_model_len']
    if type(context) is not int or context < max(sum(pair) for pair in manifest['workloads']['pairs']):
        raise ValueError('Environment context length is below a requested workload')
    for key in ('engine_version', 'launch_command', 'gpu_model', 'graph_mode'):
        if not isinstance(data[key], str) or not data[key].strip() or data[key].startswith('REPLACE'):
            raise ValueError(f'Fill environment {key}')
    if data['graph_mode'] not in ('eager', 'cuda_graphs'):
        raise ValueError('graph_mode must be eager or cuda_graphs')
    for key in ('memory_settings', 'scheduler_settings'):
        if not isinstance(data[key], dict) or not data[key]:
            raise ValueError(f'Fill environment {key} with the actual engine settings')
    allocation = data.get('allocation_evidence', {})
    if not isinstance(allocation, dict):
        raise ValueError('allocation_evidence must be an object')
    for key in ('learned_weight_runtime_bytes', 'allocated_kv_pool_bytes'):
        value = allocation.get(key)
        if value is not None:
            if type(value) is not int or value < 0 or not allocation.get('source'):
                raise ValueError('Allocation bytes require nonnegative integers and an evidence source')
    # Environment manifests are explicit experiment metadata, not dumps of os.environ.
    serialized = json.dumps(data).lower()
    if any(word in serialized for word in ('authorization', 'bearer ', 'hf_token=', 'api_key=', 'api-key ')):
        raise ValueError('Remove credentials from environment metadata')
    return data


@main.command('run')
@click.option('--manifest', type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option('--engine', type=click.Choice(['vllm', 'sglang']), required=True)
@click.option('--endpoint', 'url', required=True, help='Server root URL, e.g. http://127.0.0.1:8000')
@click.option('--telemetry-endpoint', help='Collector root URL, e.g. http://127.0.0.1:9109')
@click.option('--environment', type=click.Path(exists=True, dir_okay=False, path_type=Path), help='Attested server environment JSON; required for real runs.')
@click.option('--baseline', type=click.Path(exists=True, dir_okay=False, path_type=Path), help='Pre-start collector snapshot JSON.')
@click.option('--output', type=click.Path(path_type=Path), required=True, help='New output directory; existing directories are never overwritten.')
@click.option('--api-key-env', default='VRAMFIT_API_KEY', show_default=True)
@click.option('--telemetry-key-env', default='VRAMFIT_COLLECTOR_TOKEN', show_default=True)
@click.option('--request-timeout', type=float, default=120.0, show_default=True)
@click.option('--ready-timeout', type=float, default=120.0, show_default=True)
@click.option('--max-duration', type=float, default=1800.0, show_default=True, help='Client deadline in seconds; does not stop cloud billing.')
@click.option('--sample-interval', type=float, default=0.2, show_default=True)
@click.option('--synthetic', is_flag=True, help='Explicitly label test-server runs; never use for real-model findings.')
def run_command(manifest, engine, url, telemetry_endpoint, environment, baseline, output,
                api_key_env, telemetry_key_env, request_timeout, ready_timeout, max_duration,
                sample_interval, synthetic):
    """Run warm-ups and repeated concurrent batches; no automatic inference retries."""
    try:
        document = validate(read_json(manifest))
        if not synthetic and not environment:
            raise ValueError('Real measurements require --environment; test servers require --synthetic')
        metadata = environment_metadata(environment, document, engine) if environment else None
        baseline_data = validate_snapshot(read_json(baseline)) if baseline else None
        settings = Settings(engine, endpoint(url), endpoint(telemetry_endpoint) if telemetry_endpoint else None,
                            request_timeout, sample_interval, max_duration, ready_timeout,
                            os.environ.get(api_key_env, ''), os.environ.get(telemetry_key_env, ''), synthetic)
        report = asyncio.run(run(document, settings, output, metadata, baseline_data))
        click.echo(f"{report['status']}: {output / 'run.json'}")
        if report['status'] != 'completed':
            raise click.ClickException('Measurement incomplete; inspect run.json and samples.jsonl')
    except (ValueError, OSError) as exc:
        raise click.ClickException(str(exc)) from exc


@main.command('baseline')
@click.option('--telemetry-endpoint', required=True)
@click.option('--output', type=click.Path(path_type=Path), required=True)
@click.option('--telemetry-key-env', default='VRAMFIT_COLLECTOR_TOKEN')
def baseline_command(telemetry_endpoint, output, telemetry_key_env):
    """Capture device usage BEFORE starting the inference engine."""
    try:
        if output.exists():
            raise ValueError('Baseline output already exists')
        with httpx.Client(base_url=endpoint(telemetry_endpoint), timeout=10, trust_env=False,
                          headers=headers(os.environ.get(telemetry_key_env, ''))) as client:
            response = client.get('/snapshot')
            response.raise_for_status()
            data = validate_snapshot(response.json())
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x') as file:
            json.dump(data, file, indent=2, allow_nan=False)
        click.echo(f"Saved baseline to {output}; device available: {data.get('device') is not None}")
    except (ValueError, OSError, httpx.HTTPError) as exc:
        raise click.ClickException(f'Baseline failed: {type(exc).__name__}') from exc
