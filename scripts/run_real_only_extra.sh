#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/cunyuliu/pc_cng_research
PY=/home/cunyuliu/miniconda3/envs/pc_cng_gpu/bin/python
cd "$ROOT/chem_negative_sampling"
mkdir -p "$ROOT/results/logs"
COMMON_REAL="--real-csv $ROOT/data/processed/regiosqm20_normalized.csv --real-csv $ROOT/data/processed/hitea_full_normalized.csv"
start_run() {
  name=$1
  gpu=$2
  hidden=$3
  bits=$4
  drop=$5
  out="$ROOT/results/$name"
  log="$ROOT/results/logs/$name.log"
  mkdir -p "$out"
  if [ -f "$out/metrics.json" ]; then echo "$name already complete"; return 0; fi
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=. nohup "$PY" -m pc_cng.train_feasibility_mlp \
    $COMMON_REAL \
    --output-dir "$out" \
    --epochs 80 \
    --batch-size 4096 \
    --lr 0.001 \
    --hidden-dim "$hidden" \
    --dropout "$drop" \
    --n-bits "$bits" \
    --seed 20260709 \
    > "$log" 2>&1 &
  echo "$!" > "$out/pid.txt"
  echo "started $name pid $! on gpu $gpu"
}
start_run full_feasibility_mlp_real_only_h4096_n2048_e80 0 4096 2048 0.20
start_run full_feasibility_mlp_real_only_h2048_n4096_e80 1 2048 4096 0.15
