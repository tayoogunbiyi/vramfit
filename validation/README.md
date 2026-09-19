# Validation experiment

| Setting | Value |
| --- | --- |
| Smaller model | `Qwen/Qwen3-4B` |
| Qwen revision | `1cfa9a7208912126459214e8b04321603b3df60c` |
| Larger model | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` |
| Llama revision | `6a6f4aa4197940add57724a7707d069478df56b1` |
| Original estimator prediction commit | `6759da2cfc1acbd5e0e354b8a435ca710c241260` |
| Baseline / boundary script commits | `1adcee8` / `59c970e` |
| vLLM | `0.11.0` |
| guidellm | `0.7.3` |
| transformers | `4.57.6` (vLLM 0.11 breaks on transformers 5) |
| Weight / KV cache dtype | BF16 / BF16 |
| Provider / GPU target | RunPod Secure Cloud / 1× NVIDIA A40 48 GB |
| Actual VRAM / driver | 46,068 MiB (44.99 GiB) / 570.211.01 |

## Measured results — 16 September 2026

All six measured cases completed with zero reported request errors and
zero preemptions. Warmups are excluded.

### Predicted vs observed memory

Case A requests one stream with 512 prompt + 128 output tokens; B requests eight
of those streams; C requests eight streams with 4,096 prompt + 256 output tokens.
Predictions budget the full prompt and output for every stream simultaneously.

| Model | Case | Predicted KV (GiB) | Peak occupied KV (GiB) | Delta (GiB) | Delta (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3-4B | A | 0.08789 | 0.09009 | +0.00220 | +2.50% |
| Qwen3-4B | B | 0.70312 | 0.72070 | +0.01758 | +2.50% |
| Qwen3-4B | C | 4.78125 | 4.79883 | +0.01758 | +0.37% |
| Distill-Llama-8B | A | 0.07812 | 0.08008 | +0.00195 | +2.50% |
| Distill-Llama-8B | B | 0.62500 | 0.64062 | +0.01562 | +2.50% |
| Distill-Llama-8B | C | 4.25000 | 4.25391 | +0.00391 | +0.09% |

Delta is observed minus predicted; percentages use the prediction as the denominator.
Values are rounded after calculation. Observed KV is inferred from sampled occupied
blocks and model geometry, not a measurement of total GPU memory.

**Why the difference?** Chat formatting adds eight prompt tokens for Qwen and four
for DeepSeek. vLLM then rounds cache allocations to 16-token blocks. This explains
the 2.5% increase in short cases and the 0.37% increase in Qwen's long case.
DeepSeek's long case peaked below the full-length allocation for all eight streams:
requests do not necessarily reach their maximum length together, and samples can
miss peaks. Its smaller 0.09% delta is not evidence of a more accurate estimate.

| Model | Predicted weights (GiB) | Logged model allocation (GiB) | Delta (GiB) |
| --- | ---: | ---: | ---: |
| Qwen3-4B | 7.4924 | 7.5552 | +0.0628 |
| Distill-Llama-8B | 14.9575 | 14.9889 | +0.0314 |

The weight estimate counts parameter bytes; vLLM reports a runtime allocation,
so these are related but not identical quantities. Neither comparison includes
all runtime overhead or proves the minimum GPU size needed to serve the workload.

### Request performance

Latency is mean end-to-end response time; output tokens/s is GuideLLM's aggregate
mean for successful requests. Incomplete requests were still outstanding at the
time limit, rather than reported request errors.

| Model | Case | Successful | Incomplete at cutoff | Latency (s) | Output tokens/s |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3-4B | A | 77 | 0 | 2.35 | 54.4 |
| Qwen3-4B | B | 481 | 7 | 2.96 | 342.0 |
| Qwen3-4B | C | 145 | 7 | 9.80 | 199.9 |
| Distill-Llama-8B | A | 47 | 0 | 3.84 | 33.4 |
| Distill-Llama-8B | B | 298 | 6 | 4.82 | 210.2 |
| Distill-Llama-8B | C | 96 | 8 | 14.70 | 138.9 |

### Capacity boundary

Qwen used 4,104 actual prompt tokens and 256 output tokens per request. With
16-token blocks, each full sequence needs 273 blocks. The 2 GiB pool provided
909 usable blocks, enough for three simultaneous sequences in memory (819 blocks),
but not four (1,092 blocks).

The server allowed at most eight running sequences, above both tested concurrency
levels. A ninth incoming request would wait even with spare cache; raising
`--max-num-seqs` to nine would lift that limit but would not create more cache.
Here, cache capacity limited the server to three running requests before the
eight-sequence limit mattered.

| Cache / case | Requested concurrency | Peak running | Peak occupied blocks | Samples queued | Mean latency (s) | Output tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 GiB / below-1 | 3 | 3 | 819 | 0.0% | 6.74 | 111.5 |
| 2 GiB / above | 4 | 3 | 819 | 88.1% | 8.71 | 114.9 |
| 2 GiB / below-2 | 3 | 3 | 819 | 0.0% | 6.70 | 112.0 |
| 4 GiB / control-four | 4 | 4 | 1,092 | 4.9% | 7.37 | 135.1 |

There were no reported request errors or preemptions. Extra demand was handled by
waiting. Pressure appeared below 100% cache occupancy: the remaining 90 blocks
could not hold another complete 257-block prompt. The larger-cache control and
repeated below point support the predicted three-versus-four resident boundary.
This is a restricted-cache A40 experiment, not validation of a smaller physical GPU.

### Evidence and limits

[Baseline evidence](evidence/baseline/) and [boundary evidence](evidence/boundary/)
contain compact results CSVs, launch/benchmark commands and relevant startup log
extracts. CSV peaks and queue percentages use samples within each benchmark window;
prompt lengths are configured lengths before chat formatting. Full benchmark reports,
raw metrics and warmups are kept locally and are not included in the repository.

- Original predictions used a 48 GiB planning capacity; reproduction below uses
  the measured 44.99 GiB. This changes headroom, not predicted weight or KV bytes.
- KV pool reservation differs from occupied cache. Total runtime overhead remains
  unmodelled. Chat formatting and block rounding explain small KV differences.
- GuideLLM cutoff summaries, raw cancellation records and server completions use
  different windows. Qwen baseline server completion deltas exceed client successes
  by 1/1/3/3 across warmup/A/B/C; only the extra warmup request was directly accounted
  for. Exact request-ID reconciliation was not performed.
- Samples can miss peaks; running counts do not alone prove full-length residency.
  One run per configuration does not establish performance confidence intervals.
- PyTorch sampling fallback and model generation defaults were active. NCCL shutdown
  warnings occurred; the boundary run's final GPU check showed memory released.
  A negative control graph-capture allocation delta is not a physical memory estimate.
- Baseline execution took about 47 minutes and boundary about 45 minutes, excluding
  initial package setup/fetch. Timed load was only 19 and 13 minutes respectively.

## Reproduce predictions

With `uv` installed, run from the repository root to regenerate cases A–C.
Arguments: model ID, revision SHA, new output directory, GPU capacity in GiB,
and optional headroom percentage (default 10%). Each directory receives CLI
reports, commands and `predictions.csv`; existing directories are not overwritten.

```sh
bash validation/predict.sh Qwen/Qwen3-4B \
  1cfa9a7208912126459214e8b04321603b3df60c /tmp/qwen-predictions 44.98828125

bash validation/predict.sh deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  6a6f4aa4197940add57724a7707d069478df56b1 /tmp/llama-predictions 44.98828125
```

## Measurement

Load is generated by [guidellm](https://vllm-project.github.io/guidellm/stable/)
against a local vLLM server. guidellm reports client-side numbers only, so
`metrics.sh` samples vLLM's `/metrics` alongside it for resident sequences, KV cache
usage and preemptions.

`serve.sh` is written for vLLM 0.11: earlier versions reject `--enable-log-requests`
and use the older `gpu_cache_usage_perc` metric name.

From the repository root, ship the committed validation scripts and install the pinned
packages on a rented GPU pod over SSH. Uncommitted changes under `validation/` block
shipping.

```sh
make smoke HOST=1.2.3.4 SSH_PORT=22022  # short Qwen3-4B check; run this first
make eval HOST=1.2.3.4 SSH_PORT=22022   # full evaluation of both models; asks first
make fetch HOST=1.2.3.4 SSH_PORT=22022  # copy results again
make boundary HOST=1.2.3.4 SSH_PORT=22022  # Qwen: 3 vs 4 requests with 2/4 GiB cache
```

Results are fetched into `validation/runs/`, including after failure; smoke results
are under `smoke/`. If SSH drops, rerun the same target to reattach. Create and stop
the pod manually.

Boundary results are saved under `validation/runs/boundary/<id>/`. After fetching,
run `python3 validation/analyze_boundary.py validation/runs/boundary/<id>` to write
`analysis.md`. It summarizes evidence; whether the cache boundary was observed
requires comparing the below/above/control cases, not just successful completion.

Or run one model at a time by hand:

```sh
bash validation/serve.sh MODEL REVISION validation/runs/MODEL_SLUG
bash validation/run.sh MODEL validation/runs/MODEL_SLUG
kill "$(cat validation/runs/MODEL_SLUG/server.pid)"
```

`serve.sh` starts vLLM with BF16 weights and KV cache, prefix caching off, and
`--max-num-seqs 8`, then waits for `/health`. `run.sh` runs a
warm-up followed by cases A, B and C.

| Case | Concurrency | Prompt tokens | Output tokens | Seconds |
| --- | ---: | ---: | ---: | ---: |
| warmup | 1 | 512 | 128 | 30 |
| A | 1 | 512 | 128 | 180 |
| B | 8 | 512 | 128 | 180 |
| C | 8 | 4096 | 256 | 180 |

Cases are time-bounded; actual request counts are recorded in the reports.

Each case writes to `validation/runs/MODEL_SLUG/CASE/`:

| File | Contents |
| --- | --- |
| `benchmarks.json` / `.csv` | guidellm results, including per-request token counts |
| `metrics.jsonl` | engine samples: running, waiting, KV usage, preemptions |
| `metrics-start.txt` / `-end.txt` | full `/metrics` data |
| `command.txt`, `guidellm.log` | exact invocation and console output |

`server.log` and `launch-command.txt` sit alongside, per model.

## Reading the results

### Chat template tokens

`prompt_tokens` in the case table is the length of the message content. The server
applies the model's chat template before tokenizing, so each request is slightly
longer than configured:

| Model | Template tokens | Case A/B prompt | Case C prompt |
| --- | ---: | ---: | ---: |
| Qwen3-4B | 8 | 520 | 4104 |
| Distill-Llama-8B | 4 | 516 | 4100 |


### KV blocks round up

vLLM allocates KV cache in 16-token blocks and the estimator multiplies exact token counts, so measured cache use can exceed by up to 15 tokens per sequence. At full residency (prompt,
template and output), for either model:

| Case | Estimator tokens | Engine tokens | Difference |
| --- | ---: | ---: | ---: |
| A/B | 640 | 656 (41 blocks) | +2.5% |
| C | 4352 | 4368 (273 blocks) | +0.37% |


### Step token budget

`--max-num-batched-tokens 8192` caps how many tokens the engine processes in one
step. A generating sequence costs 1 token per step; a prompt costs its length, split
across steps by chunked prefill, which is on by default on GPU.

At startup vLLM profiles a full-size step and reserves that activation memory before
sizing the KV pool, so a larger budget means a smaller pool. This is part of the
runtime overhead the estimator excludes; record the startup log's memory profiling
lines alongside `num_gpu_blocks`.
