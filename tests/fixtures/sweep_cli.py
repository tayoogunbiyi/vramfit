"""Fake CLI process for shell-sweep tests; never performs network or file I/O."""

import os
import sys

mode = os.environ.get("SWEEP_TEST_MODE", "normal")
if sys.argv[1:] == ["--version"]:
    print("vramfit, version test")
    sys.exit(1 if mode == "version_failure" else 0)

model = sys.argv[1]
rejections = {
    "Qwen/Qwen2.5-7B-Instruct-AWQ": ("qwen2", "Quantized checkpoints are not supported."),
    "Qwen/Qwen2.5-7B-Instruct-1M": ("qwen2", "Dual-chunk attention is not supported."),
    "Qwen/Qwen3-30B-A3B": ("qwen3_moe", "No adapter is registered for this model type."),
    "mistralai/Mistral-7B-v0.1": ("mistral", "Sliding-window attention is not supported by this adapter."),
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506": ("mistral3", "No adapter is registered for this model type."),
    "mistralai/Mixtral-8x7B-Instruct-v0.1": ("mixtral", "No adapter is registered for this model type."),
}
target = model == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
gate = model == "meta-llama/Llama-3.1-8B"
rejected_target = model == "Qwen/Qwen2.5-7B-Instruct-AWQ"
if gate and mode != "gate_success":
    if mode == "gate_network":
        print("Error: Could not resolve model revision. Check your connection.")
    elif mode == "gate_unknown":
        print("Learned parameters: unknown")
        sys.exit(0)
    else:
        status = 401 if mode == "gate_401" else 403
        print(f"Error: Could not download config.json for '{model}' at 'main' (HTTP {status}). Request access.")
    sys.exit(1)

if target and mode in ("network", "public_auth"):
    if mode == "public_auth":
        print(f"Error: Could not download config.json for '{model}' at 'main' (HTTP 401). Authenticate.")
    else:
        print("Error: Could not resolve model revision. Check your connection.")
    sys.exit(1)
if target and mode == "empty":
    sys.exit(0)

if model in rejections and not (rejected_target and mode == "unexpected_accept"):
    family, reason = rejections[model]
    if rejected_target:
        if mode == "rejection_network":
            print("Error: Could not resolve model revision. Check your connection.")
            sys.exit(1)
        if mode == "wrong_reason":
            reason = "Custom model-code mappings have unverified semantics."
    print(f"Error: Unsupported architecture '{family}': {reason}")
    sys.exit(2 if rejected_target and mode == "rejection_exit2" else 1)

family = "llama"
if model.startswith("Qwen/Qwen3"):
    family = "qwen3"
elif model.startswith("Qwen/"):
    family = "qwen2"
elif model.startswith("mistralai/"):
    family = "mistral"
if target and mode == "wrong_family":
    family = "qwen2"
print(f"Hugging Face model: {'wrong/model' if target and mode == 'wrong_identity' else model}")
print("Resolved revision: " + ("main" if target and mode == "bad_revision" else "a" * 40))
print(f"Adapter: {family} (uniform full attention)")
count = "unknown" if target and mode == "unknown" else "15"
print(f"Learned parameters: {count}")
print("Parameter evidence: hub_safetensors")
print("KV per token: 64 bytes")
if not (target and mode == "missing_memory"):
    print("Known memory (weights + KV): 0.0000 GiB (2,910 bytes)")
fit = "no, exceeds" if mode == "no_fit" else "yes, within"
print(f"Known workload fits: {fit} the calculated budget")
print("Theoretical maximum concurrency: 0" if mode == "no_fit" else "Theoretical maximum concurrency: 1,118,481")
