#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/cunyuliu/pc_cng_research
PY=/home/cunyuliu/miniconda3/envs/pc_cng_gpu/bin/python
cd "$ROOT/chem_negative_sampling"
mkdir -p "$ROOT/results/logs"
REAL="--real-csv $ROOT/data/processed/regiosqm20_normalized.csv --real-csv $ROOT/data/processed/hitea_full_normalized.csv"
V2_SYN="--synthetic-csv $ROOT/results/v2_boundary_generation/regiosqm20_boundary_negatives_reviewed.csv --synthetic-csv $ROOT/results/v2_boundary_generation/hitea_full_boundary_negatives_reviewed.csv"
run_pairwise() {
  name=v2_pairwise_reward_h2048_n4096_e80
  out="$ROOT/results/$name"
  log="$ROOT/results/logs/$name.log"
  mkdir -p "$out"
  if [ ! -f "$out/metrics.json" ]; then
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. nohup "$PY" -m pc_cng.train_pairwise_reward_mlp \
      $REAL $V2_SYN \
      --output-dir "$out" \
      --epochs 80 \
      --batch-size 4096 \
      --lr 0.001 \
      --hidden-dim 2048 \
      --n-bits 4096 \
      --dropout 0.20 \
      --pairwise-weight 1.0 \
      --bce-weight 1.0 \
      --seed 20260710 \
      > "$log" 2>&1 &
    echo "$!" > "$out/pid.txt"
  fi
}
run_bce() {
  name=v2_direct_bce_h2048_n4096_e80
  out="$ROOT/results/$name"
  log="$ROOT/results/logs/$name.log"
  mkdir -p "$out"
  if [ ! -f "$out/metrics.json" ]; then
    CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. nohup "$PY" -m pc_cng.train_feasibility_mlp \
      $REAL $V2_SYN \
      --output-dir "$out" \
      --epochs 80 \
      --batch-size 4096 \
      --lr 0.001 \
      --hidden-dim 2048 \
      --n-bits 4096 \
      --dropout 0.20 \
      --seed 20260710 \
      > "$log" 2>&1 &
    echo "$!" > "$out/pid.txt"
  fi
}
run_pairwise
run_bce
