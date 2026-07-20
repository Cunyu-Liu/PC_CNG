#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/cunyuliu/pc_cng_research
PY=/home/cunyuliu/miniconda3/envs/pc_cng_gpu/bin/python
cd "$ROOT/chem_negative_sampling"
mkdir -p "$ROOT/results/logs"
REAL="--real-csv $ROOT/data/processed/regiosqm20_normalized.csv --real-csv $ROOT/data/processed/hitea_full_normalized.csv --real-csv $ROOT/data/processed/uspto_openmolecules_train_only.csv"
SYN="--synthetic-csv $ROOT/results/regiosqm20_full/pc_cng_synthetic_negatives_reviewed.csv --synthetic-csv $ROOT/results/hitea_full_generation/pc_cng_synthetic_negatives_reviewed.csv"
start_run() {
  name=$1
  gpu=$2
  extra=$3
  out="$ROOT/results/$name"
  log="$ROOT/results/logs/$name.log"
  mkdir -p "$out"
  if [ -f "$out/metrics.json" ]; then echo "$name already complete"; return 0; fi
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=. nohup "$PY" -m pc_cng.train_feasibility_mlp \
    $REAL $extra \
    --output-dir "$out" \
    --epochs 40 \
    --batch-size 4096 \
    --lr 0.001 \
    --hidden-dim 4096 \
    --dropout 0.20 \
    --n-bits 2048 \
    --seed 20260709 \
    > "$log" 2>&1 &
  echo "$!" > "$out/pid.txt"
  echo "started $name pid $! on gpu $gpu"
}
start_run full_feasibility_mlp_uspto_real_only_h4096_n2048_e40 0 ""
start_run full_feasibility_mlp_uspto_real_plus_pc_cng_h4096_n2048_e40 1 "$SYN"
