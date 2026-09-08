"""Sampled device memory and engine metrics; missing observations are never zero."""

import csv
from datetime import datetime, timezone
import io
import math
import platform
import subprocess

from prometheus_client.parser import text_string_to_metric_families


METRICS = {
    "vllm": {
        "running_requests": ("vllm:num_requests_running",),
        "waiting_requests": ("vllm:num_requests_waiting",),
        "cache_usage_ratio": ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"),
        "preemptions_total": ("vllm:num_preemptions_total",),
    },
    "sglang": {
        "running_requests": ("sglang:num_running_reqs",),
        "waiting_requests": ("sglang:num_queue_reqs",),
        "cache_usage_ratio": ("sglang:token_usage",),
        "preemptions_total": ("sglang:num_retracted_reqs_total",),
    },
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def engine_metrics(text, engine, model):
    """Keep raw series; summarize only unambiguous single-instance observations."""
    if len(text) > 4_000_000:
        raise ValueError("metrics response exceeds 4 MB")
    series = []
    wanted = {name for names in METRICS[engine].values() for name in names}
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            if sample.name in wanted and math.isfinite(sample.value):
                label_model = sample.labels.get("model_name", sample.labels.get("model"))
                if label_model is None or label_model == model:
                    series.append({"name": sample.name, "labels": sample.labels, "value": sample.value})
    normalized = {}
    for field, names in METRICS[engine].items():
        normalized[field] = None
        for name in names:
            values = [row["value"] for row in series if row["name"] == name]
            if values:
                # Multiple workers/replicas may report duplicate or different scopes.
                # Do not sum them into invented single-GPU measurements.
                if len(values) == 1 and values[0] >= 0 and (field != "cache_usage_ratio" or values[0] <= 1):
                    normalized[field] = values[0]
                break
    return {"values": normalized, "series": series}


def device_snapshot(gpu_index=0):
    base = {"schema_version": 1, "observed_at": utc_now(), "host": platform.node(),
            "platform": platform.system(), "source": "nvidia-smi", "device": None}
    try:
        result = subprocess.run([
            "nvidia-smi", f"--id={gpu_index}",
            "--query-gpu=uuid,name,driver_version,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ], capture_output=True, text=True, timeout=5, check=True)
        rows = list(csv.reader(io.StringIO(result.stdout), skipinitialspace=True))
        if len(rows) != 1 or len(rows[0]) != 5:
            raise ValueError("expected one GPU row")
        uuid, name, driver, total, used = [value.strip() for value in rows[0]]
        total, used = float(total), float(used)
        if not all(math.isfinite(x) for x in (total, used)) or not 0 <= used <= total or total <= 0:
            raise ValueError("invalid device memory")
        base["device"] = {"uuid": uuid, "name": name, "driver_version": driver,
                          "total_bytes": int(total * 1024**2), "used_bytes": int(used * 1024**2)}
        base["unavailable_reason"] = None
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        base["unavailable_reason"] = f"{type(exc).__name__}: NVIDIA device memory unavailable"
    return base


def validate_snapshot(data):
    if not isinstance(data, dict) or data.get("schema_version") != 1 or not isinstance(data.get("observed_at"), str):
        raise ValueError("invalid collector response")
    device = data.get("device")
    if device is not None:
        if not isinstance(device, dict) or not isinstance(device.get("uuid"), str) or not device['uuid']:
            raise ValueError("collector device UUID missing")
        total, used = device.get("total_bytes"), device.get("used_bytes")
        if type(total) is not int or type(used) is not int or not 0 <= used <= total or total <= 0:
            raise ValueError("collector memory values invalid")
    return data
