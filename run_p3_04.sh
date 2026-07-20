#!/bin/bash
# P3-04: Condition extraction + 10-seed training (翻盘 P2-08 NO-GO).
# Uses REAL ORD conditions instead of P2-08's synthetic conditions.
set -e
cd /home/cunyuliu/pc_cng_research
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/home/cunyuliu/pc_cng_research/chem_negative_sampling
mkdir -p results/logs

echo "[P3-04] Step 1: extract conditions from ORD..."
/home/cunyuliu/miniconda3/envs/pc_cng_gpu/bin/python -m data.extract_conditions \
    --input data/processed/ord_normalized.csv \
    --output data/processed/ord_conditions.json

echo "[P3-04] Step 2: 10-seed condition training..."
exec /home/cunyuliu/miniconda3/envs/pc_cng_gpu/bin/python -m training.train_condition \
    --data data/processed/ord_conditions.json \
    --train-idx data/processed/train_idx_condition.json \
    --val-idx data/processed/val_idx_condition.json \
    --test-idx data/processed/test_idx_condition.json \
    --seeds 20260710,20260711,20260712,20260713,20260714,20260715,20260716,20260717,20260718,20260719 \
    --output-dir results/condition_prediction_v2_ord_20260720
