#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/cunyuliu/pc_cng_research
PY=/home/cunyuliu/miniconda3/envs/pc_cng_gpu/bin/python
cd "$ROOT/chem_negative_sampling"
mkdir -p "$ROOT/results/logs"
BASE_REAL="--real-csv $ROOT/data/processed/regiosqm20_normalized.csv --real-csv $ROOT/data/processed/hitea_full_normalized.csv"
USPTO_REAL="$BASE_REAL --real-csv $ROOT/data/processed/uspto_openmolecules_train_only.csv"
BASE_SYN="--synthetic-csv $ROOT/results/regiosqm20_full/pc_cng_synthetic_negatives_reviewed.csv --synthetic-csv $ROOT/results/hitea_full_generation/pc_cng_synthetic_negatives_reviewed.csv"
USPTO_SYN="--synthetic-csv $ROOT/results/uspto_openmolecules_full_generation/pc_cng_synthetic_negatives_reviewed.csv"
start_run() {
  name=$1
  gpu=$2
  extra=$3
  out="$ROOT/results/$name"
  log="$ROOT/results/logs/$name.log"
  mkdir -p "$out"
  if [ -f "$out/metrics.json" ]; then echo "$name already complete"; return 0; fi
  echo "starting $name on gpu $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=. nohup "$PY" -m pc_cng.train_feasibility_mlp \
    $extra \
    --output-dir "$out" \
    --epochs 80 \
    --batch-size 4096 \
    --lr 0.001 \
    --dropout 0.20 \
    --seed 20260710 \
    > "$log" 2>&1 &
  echo "$!" > "$out/pid.txt"
}
start_run weighted_pc_cng_synth02_h4096_n2048_e80 0 "$BASE_REAL $BASE_SYN --hidden-dim 4096 --n-bits 2048 --origin-weight synthetic=0.2"
start_run weighted_pc_cng_synth02_h2048_n4096_e80 1 "$BASE_REAL $BASE_SYN --hidden-dim 2048 --n-bits 4096 --origin-weight synthetic=0.2"
start_run weighted_uspto50k_w005_real_h4096_n2048_e80 4 "$USPTO_REAL --hidden-dim 4096 --n-bits 2048 --dataset-weight uspto_openmolecules_yield25to150=0.05 --max-train-per-dataset uspto_openmolecules_yield25to150=50000"
start_run weighted_uspto50k_w005_uspto_synth50k_w005_h4096_n2048_e80 0 "$USPTO_REAL $USPTO_SYN --hidden-dim 4096 --n-bits 2048 --dataset-weight uspto_openmolecules_yield25to150=0.05 --origin-weight synthetic=0.05 --max-train-per-dataset uspto_openmolecules_yield25to150=50000 --max-synthetic 50000"
