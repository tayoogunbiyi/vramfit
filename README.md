# vramfit

Estimate the GPU memory needed to serve an LLM at your target concurrency.
vramfit reads Hugging Face model metadata to budget weights and KV cache for your
prompt and output lengths without downloading model weights.

## Installation

Requires Python 3.11+. From a checkout of this repository:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
```

## Usage

```sh
# One 80 GiB GPU; eight requests held in memory simultaneously.
# Budget for each request to hold up to 8,192 input + 2,048 generated tokens.
# Defaults: dtype=float16, target-concurrency=1, headroom=10%.
vramfit Qwen/Qwen3-32B \
  --vram 80 \
  --dtype bfloat16 \
  --target-concurrency 8 \
  --headroom 15 \
  --prompt-length 8192 \
  --max-output-length 2048
```

Output looks like:

```text
Qwen/Qwen3-32B · EXCEEDS ESTIMATED BUDGET

┌──────────────────┬─────────────────────────────────┐
│ Estimated memory │ 81.02 GiB / 68.00 GiB usable    │
│ Budget deficit   │ 13.02 GiB                       │
│ GPU              │ 80 GiB · 15% reserved           │
│ Precision        │ bfloat16 weights + KV           │
│ Workload         │ 8 concurrent requests           │
│ Tokens / request │ 8,192 prompt + 2,048 max output │
└──────────────────┴─────────────────────────────────┘

Weights fit, but KV cache at the requested concurrency exceeds the budget.
Runtime overhead excluded; not a runtime guarantee.
Use --detailed for memory breakdown and model evidence.
```

Add `--detailed` for the breakdown and assumptions, or `--revision BRANCH_TAG_OR_SHA`
to pin a model revision (defaults to `main`). Inspection requires internet access to
Hugging Face. For private/gated models, use `hf auth login` with an account that has access.

## Scope and assumptions

**Fits means weights + KV cache ≤ GPU memory − reserved headroom.** Activations,
CUDA and framework overhead are excluded; headroom may not cover them. This is a
memory estimate and not a guarantee of serving capacity, throughput or latency.

Supports non-quantized dense text decoders with uniform full attention:

| Adapter (`model_type`) | Supported family |
| --- | --- |
| `llama` | Llama-compatible causal language models |
| `qwen2` | Qwen2 and Qwen2.5 |
| `qwen3` | Dense Qwen3 |
| `mistral` | Mistral with sliding-window attention explicitly disabled |

Each model must pass configuration checks. Quantization, MoE, sliding/mixed
attention, multi-query attention, multimodal/encoder-decoder models and custom
model code are unsupported. Multi-GPU sharding and CPU offloading are not modelled.
Insufficient parameter evidence produces **unknown**.

`--dtype` sets both weight and KV precision: `float32` (4 bytes), `float16` or
`bfloat16` (2 bytes), independently of the checkpoint's stored floating dtype.

## Predicted vs. measured GPU memory

**We ran Qwen3-4B and DeepSeek-R1-Distill-Llama-8B on an NVIDIA A40:
peak occupied KV cache was within 2.5% of predictions across all six workloads.**
Logged model allocations were within 0.9% of predicted weight memory. Runs used
BF16 and vLLM 0.11.0, with no reported request errors or preemptions.

A: 1 request, 512 input + 128 output tokens. B: 8 requests with the same lengths.
C: 8 requests, 4,096 input + 256 output tokens.

| Model | Case | Predicted KV (GiB) | Peak occupied KV (GiB) | Delta |
| --- | --- | ---: | ---: | ---: |
| Qwen3-4B | A | 0.08789 | 0.09009 | +2.50% |
| Qwen3-4B | B | 0.70312 | 0.72070 | +2.50% |
| Qwen3-4B | C | 4.78125 | 4.79883 | +0.37% |
| Distill-Llama-8B | A | 0.07812 | 0.08008 | +2.50% |
| Distill-Llama-8B | B | 0.62500 | 0.64062 | +2.50% |
| Distill-Llama-8B | C | 4.25000 | 4.25391 | +0.09% |

Occupied KV is inferred from sampled cache blocks; delta is relative to prediction.
Chat-template tokens and cache-block rounding explain the small KV differences;
sampled peaks also depend on request progress. These comparisons cover weights
and occupied KV, not total runtime memory.

Full details, limitations and reproduction steps: [validation/README.md](validation/README.md).

## Development

```sh
uv sync
uv run vramfit --help
uv run python -m unittest discover -v
```

### Adding an adapter

For a new family, implement [ModelAdapter](src/vramfit/adapters/base.py) in
`src/vramfit/adapters/` and register it in
[default_registry()](src/vramfit/adapters/registry.py). Validate supported features,
normalize geometry and resolve parameter counts; report missing evidence as unknown.
Add revision-pinned tests for supported and rejected cases. See the
[test adapter](tests/test_adapter_pipeline.py) for an example. Existing families
usually need a regression fixture rather than a new adapter.
