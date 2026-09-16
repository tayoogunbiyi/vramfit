#!/usr/bin/env bash
# Sample vLLM engine metrics into JSONL until killed. guidellm reports client-side
# concurrency only; running/waiting sequences, cache usage and preemptions come from
# here. Also samples device memory/utilization directly from nvidia-smi, since engine
# cache usage measures active occupancy, not total device allocation.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash validation/metrics.sh OUTPUT_FILE" >&2
  exit 2
fi
output=$1
port=${PORT:-8000}
interval=${METRICS_INTERVAL:-1}
gpu_index=${GPU_INDEX:-0}

# Stop the sampler cleanly when its case ends.
trap 'exit 0' TERM INT

while true; do
  timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  if scrape=$(curl --max-time 5 -fsS "http://localhost:$port/metrics" 2>/dev/null); then
    engine_json=$(printf '%s\n' "$scrape" | awk '
      $1 ~ /^vllm:num_requests_running/                       { running = $2 }
      $1 ~ /^vllm:num_requests_waiting/                       { waiting = $2 }
      # Renamed from gpu_cache_usage_perc to kv_cache_usage_perc in vLLM 0.11.
      $1 ~ /^vllm:(gpu|kv)_cache_usage_perc/                  { cache = $2 }
      $1 ~ /^vllm:num_preemptions_total/                      { preemptions = $2 }
      END {
        printf "\"num_requests_running\":%s,\"num_requests_waiting\":%s,\"kv_cache_usage_perc\":%s,\"num_preemptions_total\":%s",
          (running == "" ? "null" : running),
          (waiting == "" ? "null" : waiting),
          (cache == "" ? "null" : cache),
          (preemptions == "" ? "null" : preemptions)
      }')
    scrape_failed=false
  else
    # Record the gap rather than leaving a silent hole in the sample series.
    engine_json='"num_requests_running":null,"num_requests_waiting":null,"kv_cache_usage_perc":null,"num_preemptions_total":null'
    scrape_failed=true
  fi

  device_json='"device_memory_used_mib":null,"device_memory_total_mib":null,"device_utilization_percent":null'
  if command -v nvidia-smi > /dev/null 2>&1; then
    if device_line=$(timeout 5 nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu \
        --format=csv,noheader,nounits -i "$gpu_index" 2>/dev/null); then
      device_json=$(printf '%s\n' "$device_line" | awk -F', *' '{
        for (i=1; i<=3; i++) { gsub(/^[[:space:]]+|[[:space:]]+$/, "", $i); if ($i !~ /^[0-9]+([.][0-9]+)?$/) $i="null" }
        printf "\"device_memory_used_mib\":%s,\"device_memory_total_mib\":%s,\"device_utilization_percent\":%s", $1, $2, $3
        exit
      }')
    fi
  fi

  printf '{"timestamp":"%s","scrape_failed":%s,%s,%s}\n' \
    "$timestamp" "$scrape_failed" "$engine_json" "$device_json" >> "$output"
  sleep "$interval"
done
