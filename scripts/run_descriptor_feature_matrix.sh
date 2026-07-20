#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/cunyuliu/pc_cng_research
PY=/home/cunyuliu/miniconda3/envs/pc_cng_gpu/bin/python
cd "$ROOT/chem_negative_sampling"
mkdir -p "$ROOT/results/logs"
REAL="--real-csv $ROOT/data/processed/regiosqm20_normalized.csv --real-csv $ROOT/data/processed/hitea_full_normalized.csv"
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
    --epochs 80 \
    --batch-size 4096 \
    --lr 0.001 \
    --dropout 0.20 \
    --seed 20260710 \
    --include-descriptors \
    > "$log" 2>&1 &
  echo "$!" > "$out/pid.txt"
  echo "started $name pid $! on gpu $gpu"
}
start_run descriptor_count_h4096_n2048_e80 4 "--hidden-dim 4096 --n-bits 2048 --fp-mode count"
start_run descriptor_binary_count_h2048_n2048_e80 1 "--hidden-dim 2048 --n-bits 2048 --fp-mode binary_count"
