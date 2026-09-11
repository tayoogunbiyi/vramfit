#!/usr/bin/env bash
# Sample vLLM engine metrics into JSONL until killed. guidellm reports client-side
# concurrency only; running/waiting sequences, cache usage and preemptions come from
# here.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash validation/metrics.sh OUTPUT_FILE" >&2
  exit 2
fi
output=$1
port=${PORT:-8000}
interval=${METRICS_INTERVAL:-1}

while true; do
  timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  if scrape=$(curl -fsS "http://localhost:$port/metrics" 2>/dev/null); then
    printf '%s\n' "$scrape" | awk -v ts="$timestamp" '
      $1 ~ /^vllm:num_requests_running/                       { running = $2 }
      $1 ~ /^vllm:num_requests_waiting/                       { waiting = $2 }
      # Renamed from gpu_cache_usage_perc to kv_cache_usage_perc in vLLM 0.11.
      $1 ~ /^vllm:(gpu|kv)_cache_usage_perc/                  { cache = $2 }
      $1 ~ /^vllm:num_preemptions_total/                      { preemptions = $2 }
      END {
        printf "{\"timestamp\":\"%s\",\"num_requests_running\":%s,\"num_requests_waiting\":%s,\"kv_cache_usage_perc\":%s,\"num_preemptions_total\":%s}\n",
          ts,
          (running == "" ? "null" : running),
          (waiting == "" ? "null" : waiting),
          (cache == "" ? "null" : cache),
          (preemptions == "" ? "null" : preemptions)
      }' >> "$output"
  else
    # Record the gap rather than leaving a silent hole in the sample series.
    printf '{"timestamp":"%s","scrape_failed":true}\n' "$timestamp" >> "$output"
  fi
  sleep "$interval"
done
