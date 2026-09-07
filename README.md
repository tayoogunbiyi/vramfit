# vramfit

Estimate whether an LLM inference workload will fit in GPU memory.

## Usage

Provide a Hugging Face model ID and one GPU's physical capacity in GiB:

```console
vramfit TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --vram 24 \
  --dtype bfloat16 \
  --target-concurrency 4 \
  --headroom 15 \
  --prompt-length 1024 \
  --max-output-length 512
```

`--dtype` defaults to `float16`, `--target-concurrency` defaults to `1`, and
`--headroom` defaults to `10%`. Prompt and maximum output lengths are required
and are measured in tokens. The selected dtype applies to both weights and KV
cache. Headroom is reserved from the physical capacity before evaluating fit.

The result shows learned weight memory, KV-cache demand, remaining budget and
theoretical concurrency. A calculated fit is **not a runtime guarantee**:
activations, CUDA/framework overhead and other runtime costs are not included.
Concurrency is a memory upper bound, not a throughput estimate.

Supports standard, non-quantized Llama, Qwen2/Qwen2.5, dense Qwen3 and
full-attention Mistral text decoders. Sliding/mixed attention, MoE and multimodal
variants are outside the current scope. Insufficient parameter evidence is
reported as unknown rather than guessed from the model name.

Use `--revision BRANCH_TAG_OR_SHA` to select a revision (default: `main`). The
output records the resolved commit. The command reads Hub metadata, caches
`config.json`, and inspects SafeTensors headers when needed; it does not download
weight payloads or load a model.

For gated or private models, authenticate first:

```console
hf auth login
```

Authentication does not grant gated-repository access. Request access on the
model's Hugging Face page and wait for approval if required.

## Development

```console
uv sync
uv run vramfit --help
```

Run the tests with `uv run python -m unittest discover -v`.

### Model sweep

`scripts/model-sweep.sh` runs the CLI against ~20 supported and unsupported
Hugging Face models and logs every full result, plus an exit-status index, to one
text report. It is metadata-only (no weight downloads) and takes under a minute.

```console
scripts/model-sweep.sh
scripts/model-sweep.sh -o sweep.txt          # choose the report path
```

The report defaults to `${TMPDIR:-/tmp}/vramfit-model-sweep.txt`. Each entry
records the command, its output and its exit status (0 when an estimate is
produced, non-zero when the model is declined); it does no pass/fail judging. Set
`HF_TOKEN` (or add it to `.env`) to cover gated repos.

### Adding an adapter

Adapters support model families, not repository names. For an already supported
family, add a regression fixture instead. For a new compatible family, implement
[ModelAdapter](src/vramfit/adapters/base.py) in `src/vramfit/adapters/<family>.py`:
declare `model_type`, normalize config geometry, decide when headers are needed,
and resolve the learned parameter count.

Keep family rules and evidence handling in the adapter: validate defaults,
reject unsupported features, account for tied weights and buffers, and report
insufficient evidence as unknown. Leave Hub access, memory calculations and
output to the shared pipeline. Register the adapter in
[default_registry()](src/vramfit/adapters/registry.py).

Add revision-pinned fixtures and tests for supported variants, rejection cases
and missing evidence, including the CLI path. See the
[test-only adapter](tests/test_adapter_pipeline.py) for a complete example.
Run `uv run python -m unittest discover -v`; actual GPU memory accuracy still
requires separate runtime measurements.
