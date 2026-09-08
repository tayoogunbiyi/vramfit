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

Supports standard, non-quantized Llama, Qwen2/Qwen2.5, dense Qwen3 and
full-attention Mistral text decoders. Sliding/mixed attention, MoE and multimodal
variants are outside the current scope. Insufficient parameter evidence is
reported as unknown rather than guessed from the model name.

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

## Measurement pilot (vLLM / SGLang)

The `vramfit-measure` client runs on macOS or Linux without installing either
inference engine. Point it at a dedicated, already-running vLLM or SGLang server.
`vramfit-collector` runs alongside that server and samples one NVIDIA GPU through
`nvidia-smi`. The runner does not provision Pods, install inference engines, or
stop provider billing.

### Verify the client on a Mac

```console
uv sync
uv run python scripts/measurement-smoke.py --output experiments/results/mac-smoke
```

This exercises both engine HTTP protocols with **synthetic local test servers**,
including the full pilot: 256/128 and 1,024/512 prompt/output tokens, concurrency
1 and 4, one warm-up and three measured repetitions per case. It also exercises
the real local collector; NVIDIA memory is **unavailable** on a Mac. It does not
run vLLM/SGLang models on Apple hardware or measure real GPU memory. Run the full
suite with `uv run python -m unittest discover -s tests -v`; tests use localhost
sockets but make no external network requests.

### Prepare any supported model

```console
uv run vramfit-measure prepare Qwen/Qwen2.5-7B-Instruct \
  --vram 24 --dtype bfloat16 --workloads examples/pilot.json \
  --output experiments/results/qwen25-manifest.json
```

Preparation resolves the model revision to a SHA and saves decoder geometry,
parameter evidence, the workload matrix and deterministic token-ID vocabulary.
No weights are downloaded. Use `--revision` to select an existing SHA and
`--served-name` if the server advertises an alias. Change the model argument for
Qwen3-8B, TinyLlama or Qwen3-4B; each model gets its own manifest and server session.
Edit a copy of `examples/pilot.json` to change lengths, concurrency or repeats.
All workloads must fit the model's configured context. A non-fit memory estimate
is permitted so failures can be investigated, but it is not permission to spend
on downloading an unsuitable model.

### Prepare the GPU server

Use a current stable engine image compatible with the actual GPU and driver,
then record its version and image digest. Official setup references:
[vLLM Docker](https://docs.vllm.ai/en/latest/deployment/docker/) and
[SGLang installation](https://docs.sglang.io/docs/get-started/install).
Run each engine separately; reuse a Hugging Face cache where your Pod setup
allows it. Do not assume nested Docker is available inside a RunPod container.

Generate an explicit launch command locally:

```console
uv run vramfit-measure launch-command \
  --manifest experiments/results/qwen25-manifest.json --engine vllm \
  --environment-output experiments/results/vllm-environment.json
```

Run the printed command in the GPU host's engine environment. For SGLang, change
`--engine sglang` and use a different environment-output file. The commands pin
the model SHA and dtype, disable prefix caching and CUDA graphs for this pilot,
and set the context and scheduler limits from the manifest. This is an **eager
execution baseline**, not a throughput-tuned production deployment. Check the
installed engine's `--help` before launching: flags evolve between versions.
[vLLM engine arguments](https://docs.vllm.ai/en/latest/configuration/engine_args/),
[SGLang server arguments](https://docs.sglang.io/docs/advanced_features/server_arguments).

`--memory-fraction` defaults to 0.9. The two engines' fractions have different
allocation semantics; equal values do not imply equal KV pool sizes. Fill the
created environment JSON with actual engine/GPU versions and retain the startup
log. `null` version/digest fields mean unavailable; never substitute guesses.
The runner rejects unfilled engine-version/GPU placeholders, wrong revisions or
dtypes, enabled prefix caching, and tensor parallelism other than one.

On the GPU host, install this project into a separate lightweight environment,
then start the collector **before the inference server**:

```console
uv run vramfit-collector --host 127.0.0.1 --port 9109 --gpu-index 0
```

Use the same physical GPU for the server and collector. If `CUDA_VISIBLE_DEVICES`
remaps CUDA indices, `nvidia-smi --id` still selects the physical index; verify the
UUID in results. Keep unrelated GPU workloads off the host.

From the Mac, forward both ports using the Pod's actual SSH details:

```console
ssh -N -L 8000:127.0.0.1:8000 -L 9109:127.0.0.1:9109 USER@GPU_HOST
```

Add the SSH port/key options supplied by RunPod if needed. A loopback tunnel
keeps the collector private. Non-loopback collector bindings require the
`VRAMFIT_COLLECTOR_TOKEN` environment variable. The client accepts inference
credentials through `VRAMFIT_API_KEY`, and collector credentials through
`VRAMFIT_COLLECTOR_TOKEN`; token values are not saved in reports.

Capture idle GPU memory before starting the engine:

```console
uv run vramfit-measure baseline --telemetry-endpoint http://127.0.0.1:9109 \
  --output experiments/results/before-vllm.json
```

Then launch the engine, fill its environment record, and run from the Mac:

```console
uv run vramfit-measure run \
  --manifest experiments/results/qwen25-manifest.json \
  --engine vllm --endpoint http://127.0.0.1:8000 \
  --telemetry-endpoint http://127.0.0.1:9109 \
  --environment experiments/results/vllm-environment.json \
  --baseline experiments/results/before-vllm.json \
  --max-duration 1800 --output experiments/results/qwen25-vllm
```

For the second engine, stop the first server, capture a new baseline, start
SGLang with its generated command, then change `--engine`, `--environment`,
`--baseline` and `--output`. If the host/port changes, change the two endpoint
URLs (root URLs, without `/v1`). Keep the model manifest unchanged. Inference
requests are not retried automatically. A deadline or failed request preserves
partial results and returns nonzero (an in-flight telemetry scrape may take up to
five additional seconds to close); the client deadline does not terminate the
Pod or guarantee cancellation of server-side work.

### Interpret the artifacts

Every run creates a new directory; existing results are never overwritten:

- `manifest.json`: pinned model and workloads used for this run.
- `run.json`: environment, pre-start baseline, loaded-idle observations, every
  warm-up/measured batch, per-request counts/errors/latency and matching estimates.
- `samples.jsonl`: timestamped raw engine metrics, normalized selected metrics,
  device snapshots and collection failures, labelled by phase.
- `summary.csv` and `summary.md`: measured repetitions only, with warm-ups
  excluded; sampled memory range, median batch duration and output throughput.

Prompts are synthetic integer IDs, identical across engine runs for the same
manifest. Early EOS is disabled and returned token counts must match the requested
lengths exactly. Generated text is not saved. Client latency includes networking;
this non-streaming pilot does not report TTFT or inter-token latency.

Device observations are **whole-GPU sampled totals**, with MiB resolution from
`nvidia-smi`. The polling interval is a delay between collections; sampling and
networking add time, and short peaks can be missed. If GPU capacity is observed,
the per-batch estimate uses that observed total instead of the manifest's nominal
capacity. Loaded-idle usage includes engine buffers and the preallocated KV pool;
it is not the learned-weight allocation. Do not subtract the analytical estimate
from total device memory and call the result estimation error.

Engine cache ratios and running/waiting requests remain separate from device
bytes. Unknown/ambiguous metric series remain unavailable; raw metrics are kept
for version-specific analysis. Metric freshness is engine-dependent, so polling
faster cannot make a slow-updating gauge more precise. Compare observed scheduler
concurrency with submitted concurrency, and inspect preemption counters before
claiming all requests resided concurrently.

Real runs require operator-attested environment metadata plus working device and
scheduler telemetry. The models API verifies a served name, not a revision or
dtype. Preserve startup logs to support those attestations. Optional
`allocation_evidence` in the environment JSON can contain
`learned_weight_runtime_bytes`, `allocated_kv_pool_bytes`, and a `source` pointing
to a retained log and its relevant lines. Without that evidence, those allocations
are explicitly unavailable. PyTorch worker allocated/reserved peaks are also
unavailable without additional worker instrumentation; device samples are not a
substitute. No calibration is applied by the runner.

For a real local engine without NVIDIA telemetry (for example, vLLM with its
[Apple Silicon Metal plugin](https://github.com/vllm-project/vllm-metal)), use
`--requests-only` with a filled environment record and a small manifest prepared
with `--workloads examples/mac-smoke.json`. This records genuine token counts and
request timing as `synthetic: false`, `measurement_mode: requests_only`. It does
not claim GPU-memory validation or equate Apple unified memory with NVIDIA VRAM.
Remove `--requests-only` and supply the collector endpoint for the GPU pilot.
The Metal plugin belongs in a separate engine environment, not this project's
client dependencies. The two-engine synthetic smoke remains the portable check
for both protocol paths; a Metal run validates only the real vLLM path.
