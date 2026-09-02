# vramfit

Estimate whether an LLM inference workload will fit in GPU memory.

## Usage

Provide a Hugging Face model ID and the available GPU memory in GiB:

```console
vramfit meta-llama/Llama-3.1-8B \
  --vram 24 \
  --dtype bfloat16 \
  --target-concurrency 4 \
  --headroom 15
```

`--dtype` defaults to `float16`, `--target-concurrency` defaults to `1`, and
`--headroom` defaults to `10%`.

The command downloads `config.json` from the model repository into the standard
Hugging Face cache. For gated or private models, authenticate first:

```console
hf auth login
```

Memory estimation is not implemented yet.

## Development

```console
uv sync
uv run vramfit --help
```
