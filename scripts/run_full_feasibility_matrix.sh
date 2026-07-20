#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/cunyuliu/pc_cng_research
PY=/home/cunyuliu/miniconda3/envs/pc_cng_gpu/bin/python
cd "$ROOT/chem_negative_sampling"
mkdir -p "$ROOT/results/logs"
COMMON_REAL="--real-csv $ROOT/data/processed/regiosqm20_normalized.csv --real-csv $ROOT/data/processed/hitea_full_normalized.csv"
COMMON_SYN="--synthetic-csv $ROOT/results/regiosqm20_full/pc_cng_synthetic_negatives_reviewed.csv --synthetic-csv $ROOT/results/hitea_full_generation/pc_cng_synthetic_negatives_reviewed.csv"
start_run() {
  name=$1
  gpu=$2
  extra=$3
  out="$ROOT/results/$name"
  log="$ROOT/results/logs/$name.log"
  mkdir -p "$out"
  if [ -f "$out/metrics.json" ]; then
    echo "$name already complete"
    return 0
  fi
  echo "starting $name on gpu $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=. nohup "$PY" -m pc_cng.train_feasibility_mlp \
    $COMMON_REAL $extra \
    --output-dir "$out" \
    --epochs 60 \
    --batch-size 4096 \
    --lr 0.001 \
    --n-bits 2048 \
    --seed 20260709 \
    > "$log" 2>&1 &
  echo "$!" > "$out/pid.txt"
}
start_run full_feasibility_mlp_real_only_h2048_e60 0 "--hidden-dim 2048 --dropout 0.15"
start_run full_feasibility_mlp_pc_cng_h2048_e60 1 "$COMMON_SYN --hidden-dim 2048 --dropout 0.15"
start_run full_feasibility_mlp_pc_cng_h4096_e60 2 "$COMMON_SYN --hidden-dim 4096 --dropout 0.20"
