#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/cunyuliu/pc_cng_research}
CONFIG_NAME=${CONFIG_NAME:?CONFIG_NAME is required, e.g. pw20_m000}
GPU=${PC_CNG_GPU:?PC_CNG_GPU is required}
MAX_LOADAVG=${MAX_LOADAVG:-80}
MIN_GPU_FREE_MIB=${MIN_GPU_FREE_MIB:-8192}
POLL_SECONDS=${POLL_SECONDS:-300}

echo "[$(date +%F_%T)] watcher start config=$CONFIG_NAME gpu=$GPU max_loadavg=$MAX_LOADAVG min_gpu_free_mib=$MIN_GPU_FREE_MIB"

while true; do
  load1=$(awk '{print $1}' /proc/loadavg)
  load_ok=$(awk -v loadv="$load1" -v max="$MAX_LOADAVG" 'BEGIN { print (loadv <= max) ? 1 : 0 }')
  gpu_csv=$(nvidia-smi -i "$GPU" --query-gpu=memory.used,memory.total --format=csv,noheader,nounits)
  used_mib=$(printf '%s\n' "$gpu_csv" | awk -F',' '{gsub(/ /, "", $1); print $1}')
  total_mib=$(printf '%s\n' "$gpu_csv" | awk -F',' '{gsub(/ /, "", $2); print $2}')
  free_mib=$((total_mib - used_mib))
  gpu_ok=$(awk -v free="$free_mib" -v min="$MIN_GPU_FREE_MIB" 'BEGIN { print (free >= min) ? 1 : 0 }')

  echo "[$(date +%F_%T)] config=$CONFIG_NAME gpu=$GPU load1=$load1 load_ok=$load_ok used=${used_mib}MiB free=${free_mib}MiB gpu_ok=$gpu_ok"

  if [ "$load_ok" = "1" ] && [ "$gpu_ok" = "1" ]; then
    echo "[$(date +%F_%T)] gates passed; launching config=$CONFIG_NAME on gpu=$GPU"
    exec env ROOT="$ROOT" CONFIG_NAME="$CONFIG_NAME" PC_CNG_GPU="$GPU" bash "$ROOT/scripts/run_v2_pairwise_margin_10seed_selected.sh"
  fi

  sleep "$POLL_SECONDS"
done
