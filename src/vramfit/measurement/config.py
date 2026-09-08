"""Versioned, pinned experiment manifests and exact synthetic token workloads."""

from dataclasses import asdict
import json
import math
from pathlib import Path
import re

from vramfit.inspection import inspect_model
from vramfit.memory import GPUCapacity, Workload, calculate_memory
from vramfit.models import DecoderSpec, ParameterEstimate


PILOT = {"pairs": [[256, 128], [1024, 512]], "concurrency": [1, 4], "warmups": 1, "repetitions": 3}


def positive(value, name, *, zero=False):
    if type(value) is not int or value < (0 if zero else 1):
        raise ValueError(f"{name} must be an integer >= {0 if zero else 1}")
    return value


def finite(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= minimum:
        raise ValueError(f"{name} must be finite and > {minimum}")
    return value


def matrix(value):
    if not isinstance(value, dict) or set(value) != set(PILOT):
        raise ValueError(f"workloads must contain exactly {', '.join(PILOT)}")
    for name in ("pairs", "concurrency"):
        if not isinstance(value[name], list) or not value[name]:
            raise ValueError(f"{name} must be a nonempty list")
    for pair in value["pairs"]:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError("each pair must contain prompt and output token lengths")
        for length in pair:
            positive(length, "token length")
    for count in value["concurrency"]:
        positive(count, "concurrency")
    positive(value["warmups"], "warmups", zero=True)
    positive(value["repetitions"], "repetitions")
    return value


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read JSON from {path}: {exc}") from exc


def decoded_model(document):
    decoder = dict(document['decoder'])
    parameters = dict(document['parameters'])
    for data, fields in ((decoder, ('assumptions',)), (parameters, ('assumptions', 'warnings'))):
        for field in fields:
            values = data.get(field, ())
            if not isinstance(values, (list, tuple)) or any(not isinstance(value, str) for value in values):
                raise ValueError(f'{field} must contain strings')
            data[field] = tuple(values)
    return DecoderSpec(**decoder), ParameterEstimate(**parameters)


def validate(document):
    try:
        if not isinstance(document, dict):
            raise ValueError("manifest must be an object")
        if type(document["schema_version"]) is not int or document["schema_version"] != 1:
            raise ValueError("unsupported manifest schema_version")
        model = document["model"]
        if not re.fullmatch(r"[0-9a-fA-F]{40}", model["revision"]):
            raise ValueError("model revision must be a resolved 40-character SHA")
        for key in ("id", "served_name"):
            if not isinstance(model[key], str) or not model[key].strip():
                raise ValueError(f"model {key} must be nonempty")
        spec, parameters = decoded_model(document)
        gpu = GPUCapacity(document["gpu"]["vram_gib"], document["gpu"]["headroom_percent"])
        ids = document["token_ids"]
        if not isinstance(ids, list) or not ids:
            raise ValueError("token_ids must be a nonempty list")
        vocab = positive(document["vocab_size"], "vocab_size")
        for token in ids:
            positive(token, "token ID", zero=True)
            if token >= vocab:
                raise ValueError("token ID outside vocabulary")
        for prompt, output in matrix(document["workloads"])["pairs"]:
            for concurrency in document["workloads"]["concurrency"]:
                calculate_memory(spec, parameters, Workload(prompt, output, concurrency, model["dtype"]), gpu)
        return document
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Malformed experiment manifest: {exc}") from exc


def prepare(model_id, revision, served_name, dtype, vram, headroom, workloads):
    matrix(workloads)
    inspection = inspect_model(model_id, revision=revision)
    vocab = positive(inspection.snapshot.config.get("vocab_size"), "config vocab_size")
    special = {inspection.snapshot.config.get(name) for name in ("bos_token_id", "pad_token_id")
               if isinstance(inspection.snapshot.config.get(name), int)}
    eos = inspection.snapshot.config.get("eos_token_id", [])
    special.update(eos if isinstance(eos, list) else [eos])
    token_ids = [token for token in range(min(100, vocab), min(1000, vocab)) if token not in special]
    if not token_ids:
        token_ids = [token for token in range(vocab) if token not in special]
    parameters = asdict(inspection.parameters)
    parameters.pop("checkpoint_buffers")  # buffer disclosure is retained separately
    return validate({
        "schema_version": 1,
        "model": {"id": model_id, "revision": inspection.snapshot.revision,
                  "served_name": served_name or model_id, "dtype": dtype},
        "gpu": {"vram_gib": vram, "headroom_percent": headroom},
        "decoder": asdict(inspection.spec), "parameters": parameters,
        "checkpoint_buffers": [asdict(t) for t in inspection.parameters.checkpoint_buffers],
        "vocab_size": vocab, "token_ids": token_ids, "workloads": workloads,
        "prompt_kind": "synthetic token IDs; memory workload, not a quality benchmark",
    })


def estimate(document, prompt, output, concurrency):
    spec, parameters = decoded_model(document)
    return asdict(calculate_memory(
        spec, parameters,
        Workload(prompt, output, concurrency, document["model"]["dtype"]),
        GPUCapacity(**document["gpu"]),
    ))
