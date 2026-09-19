# GPU validation results

**On an NVIDIA A40, peak occupied KV was within 2.5% of predictions across six
workloads; logged model allocations were within 0.9% of predicted weight memory.**
We also tested where cache capacity forces requests to queue.

Runs on 16 September 2026 used 44.99 GiB VRAM, BF16 weights/cache, vLLM 0.11.0,
GuideLLM 0.7.3 and transformers 4.57.6. All measured cases had zero reported
request errors and preemptions. Warmups are excluded below.

## Predicted vs. measured memory

A: 1 request, 512 input + 128 output tokens. B: 8 requests with the same lengths.
C: 8 requests, 4,096 input + 256 output tokens. Each case ran for 180 seconds,
after a 30-second warmup per model. Predictions budget full input and output
lengths for every request simultaneously.

| Model | Case | Predicted KV (GiB) | Peak occupied KV (GiB) | Delta (GiB) | Delta (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3-4B | A | 0.08789 | 0.09009 | +0.00220 | +2.50% |
| Qwen3-4B | B | 0.70312 | 0.72070 | +0.01758 | +2.50% |
| Qwen3-4B | C | 4.78125 | 4.79883 | +0.01758 | +0.37% |
| Distill-Llama-8B | A | 0.07812 | 0.08008 | +0.00195 | +2.50% |
| Distill-Llama-8B | B | 0.62500 | 0.64062 | +0.01562 | +2.50% |
| Distill-Llama-8B | C | 4.25000 | 4.25391 | +0.00391 | +0.09% |

Delta is observed minus predicted, relative to prediction. Occupied KV is inferred
from sampled cache blocks and model geometry, not total GPU memory.

### Why the small differences?

Chat formatting adds eight input tokens for Qwen and four for DeepSeek; vLLM then
rounds allocations to 16-token blocks. A/B therefore need 656 token slots per
request rather than 640 (+2.5%); C needs 4,368 rather than 4,352 (+0.37%).
DeepSeek C's sampled peak was lower because full-length allocations were not
observed together for all eight requests. Its +0.09% delta does not establish
better prediction accuracy.

| Model | Predicted weights (GiB) | Logged model allocation (GiB) | Delta (GiB) |
| --- | ---: | ---: | ---: |
| Qwen3-4B | 7.4924 | 7.5552 | +0.0628 |
| Distill-Llama-8B | 14.9575 | 14.9889 | +0.0314 |

Predicted weights count parameter bytes; logged allocations measure runtime model
loading. Neither table includes all runtime overhead or establishes a minimum GPU size.

## Capacity boundary: three requests vs. four

With Qwen's cache limited to 2 GiB, we predicted room for three long requests,
but not four. Each used 4,104 actual input + 256 output tokens, requiring 273
cache blocks. The pool had 909 usable blocks: three need 819; four need 1,092.

| Cache / case | Requested concurrency | Peak running | Peak occupied blocks | Samples queued | Mean latency (s) | Output tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 GiB / below-1 | 3 | 3 | 819 | 0.0% | 6.74 | 111.5 |
| 2 GiB / above | 4 | 3 | 819 | 88.1% | 8.71 | 114.9 |
| 2 GiB / below-2 | 3 | 3 | 819 | 0.0% | 6.70 | 112.0 |
| 4 GiB / control-four | 4 | 4 | 1,092 | 4.9% | 7.37 | 135.1 |

**Requests completed even when the requested concurrency could not fit in memory.**
The fourth request waited: the remaining 90 blocks could not hold its 257-block
prompt. Doubling the cache allowed four running requests and reduced queueing.
The server limit stayed at eight, so it did not cause this three-request boundary.
This tests a restricted cache on an A40, not a smaller physical GPU.

## Request performance

Latency is mean end-to-end time; output tokens/s is aggregate throughput for
successful requests. Incomplete requests were outstanding at the time limit.

| Model | Case | Successful | Incomplete at cutoff | Latency (s) | Output tokens/s |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3-4B | A | 77 | 0 | 2.35 | 54.4 |
| Qwen3-4B | B | 481 | 7 | 2.96 | 342.0 |
| Qwen3-4B | C | 145 | 7 | 9.80 | 199.9 |
| Distill-Llama-8B | A | 47 | 0 | 3.84 | 33.4 |
| Distill-Llama-8B | B | 298 | 6 | 4.82 | 210.2 |
| Distill-Llama-8B | C | 96 | 8 | 14.70 | 138.9 |

## Records and limits

[Baseline records](evidence/baseline/) and [boundary records](evidence/boundary/)
contain compact results, commands and startup extracts. Raw reports and metrics
are retained locally. Peaks and queue percentages use the benchmark time window.

- These runs cover two models and one GPU. Sampling can miss peaks; one run per
  configuration does not establish performance confidence intervals.
- Reserved cache differs from occupied cache. Activations, CUDA and other runtime
  overhead remain outside the estimator.
- Client cutoff totals and server completions use different windows. Qwen server
  completions exceeded client successes by 1/3/3 in A/B/C; request IDs were not
  reconciled. Use the client totals consistently for performance comparisons.

## Reproduce the runs

Lease a GPU pod and run from the repository root. These targets ship committed
validation scripts, install pinned dependencies, run the workload and fetch results.
Uncommitted changes under `validation/` block shipping.

```sh
make smoke HOST=1.2.3.4 SSH_PORT=22022     # short Qwen check; run first
make eval HOST=1.2.3.4 SSH_PORT=22022      # both models; asks before starting
make boundary HOST=1.2.3.4 SSH_PORT=22022  # Qwen with 2/4 GiB cache
make fetch HOST=1.2.3.4 SSH_PORT=22022     # retrieve results again
```

Results go to `validation/runs/`, including after failure. Rerun the same target
to reattach after an SSH interruption. Create and stop the pod manually.
Our baseline took about 47 minutes and the boundary run 45 minutes, excluding
package setup/fetch; actual timed load was 19 and 13 minutes respectively.

The server uses BF16, prefix caching off, `--max-num-seqs 8` and
`--max-num-batched-tokens 8192`. GuideLLM supplies load; `metrics.sh` samples
vLLM's running/waiting requests, cache usage and preemptions.

| Local file | What to inspect |
| --- | --- |
| `MODEL/CASE/benchmarks.json` / `.csv` | Request counts, latency, throughput and token lengths |
| `MODEL/CASE/metrics.jsonl` | Running requests, queueing, occupied cache and preemptions |
| `MODEL/server.log` | Startup allocations and runtime warnings |
| `MODEL/launch-command.txt`, `MODEL/CASE/command.txt` | Exact settings |

Boundary results use `boundary/<id>/cache-2g/` and `cache-4g/`. Summarize them with:

```sh
python3 validation/analyze_boundary.py validation/runs/boundary/<id>
```

### Reproduce predictions without a GPU

With `uv` installed, run these pinned model revisions. Arguments are model ID,
revision, a new output directory and GPU capacity in GiB; headroom defaults to 10%.
Outputs include CLI reports and `predictions.csv`.

```sh
bash validation/predict.sh Qwen/Qwen3-4B \
  1cfa9a7208912126459214e8b04321603b3df60c /tmp/qwen-predictions 44.98828125

bash validation/predict.sh deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  6a6f4aa4197940add57724a7707d069478df56b1 /tmp/llama-predictions 44.98828125
```

Original predictions used a 48 GiB planning capacity; these commands use measured
44.99 GiB. This changes the available budget, not predicted weight or KV bytes.
