#!/bin/bash
# P3-05: HTE leave-one-out evaluation (3 strategies: PC-CNG / random / none).
set -e
cd /home/cunyuliu/pc_cng_research
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=/home/cunyuliu/pc_cng_research/chem_negative_sampling
mkdir -p results/logs

exec /home/cunyuliu/miniconda3/envs/pc_cng_gpu/bin/python -m evaluation.hte_eval \
    --hte-csv data/processed/hitea_full_normalized.csv \
    --pc-cng-negatives results/uspto_openmolecules_full_generation/pc_cng_synthetic_negatives_reviewed.csv \
    --output-dir results/hte_evaluation_20260720 \
    --n-per-class 50 --min-class-size 20 \
    --seeds 20260710,20260711,20260712,20260713,20260714,20260715,20260716,20260717,20260718,20260719
