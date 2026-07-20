#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/cunyuliu/pc_cng_research
PY=/home/cunyuliu/miniconda3/envs/pc_cng_gpu/bin/python
OUT=$ROOT/results/uspto_openmolecules_full_generation
mkdir -p "$OUT" "$ROOT/results/logs"
cd "$ROOT/chem_negative_sampling"
PYTHONPATH=. "$PY" -m pc_cng.run_scale_generation \
  --input "$ROOT/data/processed/uspto_openmolecules_train_only.csv" \
  --output "$OUT/pc_cng_synthetic_negatives.csv" \
  --summary "$OUT/pc_cng_generation_summary.json"
PYTHONPATH=. "$PY" -m pc_cng.false_negative_review \
  --input "$OUT/pc_cng_synthetic_negatives.csv" \
  --output "$OUT/pc_cng_synthetic_negatives_reviewed.csv" \
  --summary "$OUT/false_negative_review_summary.json" \
  --known-positive "$ROOT/data/processed/uspto_openmolecules_train_only.csv" \
  --known-positive "$ROOT/data/processed/regiosqm20_normalized.csv" \
  --known-positive "$ROOT/data/processed/hitea_full_normalized.csv"
