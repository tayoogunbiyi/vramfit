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

The default output shows a fit verdict and a compact table with estimated memory,
remaining usable budget and workload settings. Below 60 terminal columns it uses
stacked rows. Add `--detailed` for the full memory breakdown, theoretical concurrency,
resolved revision, model evidence and assumptions. Detailed output starts with the
same compact summary, then adds these sections below it in wrapping terminal tables; redirected output uses complete labelled lines for
saved reports and scripts. A calculated fit is **not a runtime guarantee**:
activations, CUDA/framework overhead and other runtime costs are not included.
Concurrency is a memory upper bound, not a throughput estimate.

Use `--revision BRANCH_TAG_OR_SHA` to select a revision (default: `main`). The
`--detailed` output records the resolved commit. The command reads Hub metadata, caches
`config.json`, and inspects SafeTensors headers when needed; it does not download
weight payloads or load a model.

For gated or private models, authenticate first:

```console
hf auth login
```

Authentication does not grant gated-repository access. Request access on the
model's Hugging Face page and wait for approval if required.

## Scope and assumptions

A **fits** result means that estimated learned weights plus the requested KV
cache fit within one GPU's physical VRAM after reserving `--headroom`. It is a
calculation of these known components, not proof that an inference engine will
start or complete the workload. Memory fit does not guarantee throughput or
latency. Multi-GPU sharding and CPU offloading are outside this estimate.

### Architecture and precision

The built-in adapters support standard, non-quantized dense text decoders with
uniform full attention:

| Adapter (`model_type`) | Supported family |
| --- | --- |
| `llama` | Llama-compatible causal language models |
| `qwen2` | Qwen2 and Qwen2.5 |
| `qwen3` | Dense Qwen3 |
| `mistral` | Mistral with sliding-window attention explicitly disabled |

Family names alone do not establish support: the config must pass the adapter's
feature and geometry checks. MoE, sliding/mixed attention, multi-query attention,
multimodal and encoder-decoder models, and custom model-code mappings are outside
the current scope. Insufficient parameter evidence is reported as **unknown**
rather than guessed from the model name.

`--dtype` accepts only `float32` (4 bytes per element), `float16` and `bfloat16`
(2 bytes per element). It sets the assumed runtime precision for **both learned
weights and KV cache**, independently of the checkpoint's stored floating dtype;
there is no separate KV dtype option. This assumes the runtime uses the selected
precision for both components.

Quantized checkpoints are unsupported, and configs declaring quantization are
rejected. The estimator does not model INT8/INT4/FP8 formats, packed weights,
quantization scales or zero points, or partially unquantized layers. It makes no
quantized-memory accuracy claim.

### Workload residency

`--target-concurrency` means **simultaneously resident sequences** whose KV caches
are held on the GPU.

Every resident sequence is budgeted for the full `--prompt-length` plus
`--max-output-length`, all at once. This is full sequence residency, not average
occupancy during generation; requests at different stages may occupy less cache.
In general, the budget holds
`target_concurrency * (prompt_length + max_output_length)` token positions.

Average lengths do not cover longer requests; use upper bounds to budget for them.

### Memory accounting

| Component | What it covers |
| --- | --- |
| Weights | Model parameters at the selected precision |
| KV cache | Cached keys and values for all resident token positions |
| Runtime overhead | Excluded, including activations, CUDA and runtime buffers |
| Safety margin | `--headroom` reserves a percentage of GPU memory (default 10%) |

The workload fits when **weights + KV cache ≤ GPU memory − safety margin**.
The margin may not cover runtime overhead. Use `--detailed` for the breakdown.

## GPU validation

TBD — no real GPU validation yet.

## Development

```console
uv sync
uv run vramfit --help
```

Run the tests with `uv run python -m unittest discover -v`.

### Model sweep

`scripts/model-sweep.sh` checks expected outcomes across ~20 Hugging Face models
and saves a report with full output and PASS/FAIL/SKIP results. No weights are
downloaded; failed assertions produce a non-zero exit status.

```console
scripts/model-sweep.sh
scripts/model-sweep.sh -o sweep.txt          # choose the report path
```

The report defaults to `${TMPDIR:-/tmp}/vramfit-model-sweep.txt`.

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
