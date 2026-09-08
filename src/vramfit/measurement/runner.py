"""Bounded asynchronous workload runner with durable per-phase observations."""

import asyncio
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import time

import httpx

from vramfit.measurement.backends import Backend
from vramfit.measurement.config import estimate, finite, validate
from vramfit.measurement.telemetry import engine_metrics, utc_now, validate_snapshot
from vramfit.measurement.report import write_summary


LIMITATIONS = [
    "Device memory is a sampled whole-GPU observation, not an exact allocation peak or per-process attribution.",
    "Loaded-idle memory includes engine allocations and KV pool; it is not learned-weight memory.",
    "KV pool allocation and active KV usage are separate; total device usage minus the analytical estimate is not estimation error.",
    "Client concurrency is submitted requests; scheduler concurrency may differ or be missed between samples.",
    "Latency is client end-to-end wall time; non-streaming measurements do not provide TTFT or inter-token latency.",
    "Server revision, dtype and launch settings are operator-attested unless independently captured; /v1/models only verifies served identity.",
]


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as file:
        file.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        file.flush()
        os.fsync(file.fileno())
    temp.replace(path)


@dataclass(frozen=True)
class Settings:
    engine: str
    endpoint: str
    telemetry_endpoint: str | None = None
    request_timeout: float = 120
    sample_interval: float = 0.2
    max_duration: float = 1800
    ready_timeout: float = 120
    token: str = ""
    telemetry_token: str = ""
    synthetic: bool = False

    def __post_init__(self):
        for name in ("request_timeout", "sample_interval", "max_duration", "ready_timeout"):
            finite(getattr(self, name), name)
        if self.sample_interval < 0.05:
            raise ValueError("sample_interval must be at least 0.05 seconds")


def headers(token):
    return {"Authorization": f"Bearer {token}"} if token else {}


async def wait_ready(client, backend, timeout):
    deadline = time.monotonic() + timeout
    last_error = "unavailable"
    while True:
        try:
            return await backend.identity(client)
        except (httpx.HTTPError, ValueError) as exc:
            # Authentication and wrong identity are not transient startup failures.
            if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in (401, 403):
                raise ValueError(f"Endpoint authentication failed ({exc.response.status_code})") from exc
            if isinstance(exc, ValueError):
                raise
            last_error = type(exc).__name__
        if time.monotonic() >= deadline:
            raise ValueError(f"Endpoint not ready before timeout ({last_error})")
        await asyncio.sleep(min(1, max(0, deadline - time.monotonic())))


class Sampler:
    def __init__(self, metrics_client, device_client, settings, model, events):
        self.metrics_client, self.device_client = metrics_client, device_client
        self.settings, self.model, self.events = settings, model, events

    async def sample(self, phase):
        observation = {"phase": phase, "observed_at": utc_now(), "client_monotonic": time.monotonic(),
                       "engine": None, "device_snapshot": None, "errors": []}

        async def metrics():
            try:
                response = await self.metrics_client.get('/metrics')
                response.raise_for_status()
                parsed = engine_metrics(response.text, self.settings.engine, self.model)
                observation['engine'] = parsed
                observation['raw_metrics'] = response.text
            except (httpx.HTTPError, ValueError) as exc:
                observation['errors'].append(f"engine_metrics:{type(exc).__name__}")

        async def device():
            if self.device_client is None:
                observation['errors'].append('device_collector:not_configured')
                return
            try:
                response = await self.device_client.get('/snapshot')
                response.raise_for_status()
                observation['device_snapshot'] = validate_snapshot(response.json())
            except (httpx.HTTPError, ValueError) as exc:
                observation['errors'].append(f"device_collector:{type(exc).__name__}")

        await asyncio.gather(metrics(), device())
        observation['collection_seconds'] = time.monotonic() - observation['client_monotonic']
        self.events.write(json.dumps(observation, allow_nan=False) + '\n')
        self.events.flush()
        return observation


def summarize_samples(samples):
    devices = [s['device_snapshot']['device'] for s in samples
               if s.get('device_snapshot') and s['device_snapshot'].get('device')]
    uuids = {d['uuid'] for d in devices}
    totals = {d['total_bytes'] for d in devices}
    consistent = len(uuids) == 1 and len(totals) == 1
    def peak(field):
        values = [s['engine']['values'][field] for s in samples
                  if s.get('engine') and s['engine']['values'].get(field) is not None]
        return max(values) if values else None
    return {
        'sample_count': len(samples), 'device_sample_count': len(devices),
        'device_uuid': next(iter(uuids)) if consistent else None,
        'device_total_bytes': next(iter(totals)) if consistent else None,
        'sampled_device_peak_bytes': max(d['used_bytes'] for d in devices) if consistent else None,
        'device_identity_consistent': consistent,
        'observed_running_requests_max': peak('running_requests'),
        'observed_waiting_requests_max': peak('waiting_requests'),
        'observed_cache_usage_ratio_max': peak('cache_usage_ratio'),
        'collection_error_count': sum(len(s['errors']) for s in samples),
        'preemptions_counter_first': next((s['engine']['values']['preemptions_total'] for s in samples
                                           if s.get('engine') and s['engine']['values'].get('preemptions_total') is not None), None),
        'preemptions_counter_last': next((s['engine']['values']['preemptions_total'] for s in reversed(samples)
                                          if s.get('engine') and s['engine']['values'].get('preemptions_total') is not None), None),
    }


async def request_one(client, backend, ids, output, index):
    started = time.monotonic()
    record = {'index': index, 'started_at': utc_now(), 'status': 'failed',
              'prompt_tokens': None, 'output_tokens': None}
    try:
        path, payload = backend.request(ids, output)
        response = await client.post(path, json=payload)
        if response.is_error:
            # Keep arbitrary server bodies (potential secrets) out of artifacts.
            oom = 'out of memory' in response.text[:65536].lower() or 'outofmemory' in response.text[:65536].lower()
            record.update(error='out_of_memory' if oom else 'http_error', http_status=response.status_code)
        else:
            record.update(backend.usage(response.json()))
            record['status'] = 'ok' if (record['prompt_tokens'], record['output_tokens']) == (len(ids), output) else 'token_mismatch'
    except (httpx.HTTPError, ValueError) as exc:
        record['error'] = type(exc).__name__
    record['duration_seconds'] = time.monotonic() - started
    return record


async def phase(client, backend, sampler, manifest, prompt, output, concurrency, name, seed):
    samples = [await sampler.sample(name)]
    stop = asyncio.Event()
    async def poll():
        while not stop.is_set():
            samples.append(await sampler.sample(name))
            try:
                await asyncio.wait_for(stop.wait(), sampler.settings.sample_interval)
            except TimeoutError:
                pass
    monitor = asyncio.create_task(poll())
    started = time.monotonic()
    try:
        # Distinct deterministic prompts per request/repetition; disable prefix
        # caching on the engine as well, since accidental shared prefixes exist.
        rng = random.Random(seed)
        prompts = [rng.choices(manifest['token_ids'], k=prompt) for _ in range(concurrency)]
        requests = await asyncio.gather(*(request_one(client, backend, ids, output, i)
                                          for i, ids in enumerate(prompts)))
    finally:
        stop.set()
        await monitor
    elapsed = time.monotonic() - started
    samples.append(await sampler.sample(name))
    counts_ok = all(row['status'] == 'ok' for row in requests)
    observations = summarize_samples(samples)
    estimate_manifest = manifest
    if observations['device_total_bytes'] is not None:
        estimate_manifest = manifest | {'gpu': manifest['gpu'] | {'vram_gib': observations['device_total_bytes'] / 1024**3}}
    return {'phase': name, 'prompt_length': prompt, 'max_output_length': output,
            'target_concurrency': concurrency, 'seed': seed, 'requests': requests,
            'duration_seconds': elapsed, 'status': 'ok' if counts_ok else 'failed',
            'output_tokens_per_second': sum(r['output_tokens'] or 0 for r in requests) / elapsed if counts_ok else None,
            'observations': observations,
            'estimate_capacity_source': 'collector' if observations['device_total_bytes'] is not None else 'manifest',
            'estimate': estimate(estimate_manifest, prompt, output, concurrency)}


async def run(manifest, settings, destination, environment=None, baseline=None):
    validate(manifest)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / 'manifest.json', manifest)
    report = {'schema_version': 1, 'status': 'running', 'started_at': utc_now(),
              'synthetic': settings.synthetic, 'engine': settings.engine, 'endpoint': settings.endpoint,
              'telemetry_endpoint': settings.telemetry_endpoint,
              'manifest_sha256': hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
              'client': {'platform': platform.platform(), 'python': platform.python_version()},
              'settings': {name: getattr(settings, name) for name in ('sample_interval', 'request_timeout', 'max_duration', 'ready_timeout')},
              'environment': environment, 'pre_start_baseline': baseline,
              'loaded_idle': None, 'phases': [], 'limitations': LIMITATIONS,
              'unavailable': ['learned_weight_runtime_bytes (requires engine profiling logs)',
                              'framework_allocated_reserved_bytes (requires worker instrumentation)',
                              'allocated_kv_pool_bytes (requires engine allocation evidence)']}
    allocation = (environment or {}).get('allocation_evidence', {})
    report['allocation_evidence'] = allocation or None
    report['unavailable'] = [item for item in report['unavailable'] if allocation.get(item.split(' ')[0]) is None]
    write_json(destination / 'run.json', report)
    backend = Backend(settings.engine, manifest['model']['served_name'])
    limits = httpx.Limits(max_connections=max(manifest['workloads']['concurrency']), max_keepalive_connections=20)
    try:
        async with httpx.AsyncClient(base_url=settings.endpoint, headers=headers(settings.token),
                                    timeout=settings.request_timeout, limits=limits, trust_env=False) as client, \
                   httpx.AsyncClient(base_url=settings.endpoint, headers=headers(settings.token), timeout=5, trust_env=False) as metrics_client, \
                   httpx.AsyncClient(base_url=settings.telemetry_endpoint or settings.endpoint,
                                     headers=headers(settings.telemetry_token), timeout=5, trust_env=False) as device_client:
            # Deadline bounds readiness + all phases, not the provider's billing.
            async with asyncio.timeout(settings.max_duration):
                report['identity'] = await asyncio.wait_for(wait_ready(client, backend, settings.ready_timeout), settings.ready_timeout)
                with (destination / 'samples.jsonl').open('x') as events:
                    sampler = Sampler(metrics_client, device_client if settings.telemetry_endpoint else None,
                                      settings, backend.model, events)
                    idle = [await sampler.sample('loaded_idle') for _ in range(3)]
                    report['loaded_idle'] = summarize_samples(idle)
                    write_json(destination / 'run.json', report)
                    number = 0
                    for prompt, output in manifest['workloads']['pairs']:
                        for concurrency in manifest['workloads']['concurrency']:
                            for kind, count in (('warmup', manifest['workloads']['warmups']),
                                                ('measured', manifest['workloads']['repetitions'])):
                                for repetition in range(count):
                                    number += 1
                                    name = f'{kind}-p{prompt}-o{output}-c{concurrency}-r{repetition}'
                                    row = await phase(client, backend, sampler, manifest, prompt, output, concurrency, name, number)
                                    row.update(kind=kind, repetition=repetition)
                                    report['phases'].append(row)
                                    write_json(destination / 'run.json', report)
                                    if row['status'] != 'ok':
                                        raise ValueError('Request failed or token counts differed; stopped without retrying paid work')
                report['status'] = 'completed'
    except (httpx.HTTPError, ValueError, TimeoutError, asyncio.CancelledError) as exc:
        report['status'] = 'interrupted' if isinstance(exc, asyncio.CancelledError) else 'failed'
        report['error'] = type(exc).__name__
        if isinstance(exc, ValueError):
            report['error_detail'] = str(exc)
    finally:
        report['finished_at'] = utc_now()
        measured = [p for p in report['phases'] if p['kind'] == 'measured']
        report['coverage'] = {
            'measured_phases': len(measured),
            'phases_with_device_samples': sum(p['observations']['device_sample_count'] > 0 for p in measured),
            'phases_with_scheduler_samples': sum(p['observations']['observed_running_requests_max'] is not None for p in measured),
            'real_device_and_scheduler_evidence_available': False,
        }
        # Completing HTTP requests is distinct from collecting validation evidence.
        # Operator metadata cannot prove absence of unrelated GPU work.
        report['coverage']['real_device_and_scheduler_evidence_available'] = (
            not settings.synthetic and report['status'] == 'completed' and bool(measured)
            and all(p['observations']['device_identity_consistent'] for p in measured)
            and all(p['observations']['observed_running_requests_max'] is not None for p in measured)
            and environment is not None
        )
        if baseline and report['loaded_idle']:
            baseline_device = baseline.get('device')
            report['baseline_device_matches'] = bool(baseline_device and
                baseline_device['uuid'] == report['loaded_idle']['device_uuid'])
        write_json(destination / 'run.json', report)
        write_summary(destination, report)
    return report
