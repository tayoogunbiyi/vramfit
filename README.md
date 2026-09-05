# vramfit

Estimate whether an LLM inference workload will fit in GPU memory.

## Usage

Provide a Hugging Face model ID and one GPU's physical capacity in GiB:

```console
vramfit meta-llama/Llama-3.1-8B \
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
