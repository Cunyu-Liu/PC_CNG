#!/bin/bash
# P3-03 v2: Re-run pairs targeting hitea with the meaningful-groups fix.
# The original run (PID 337569) produces MRR=1.0 for ord->hitea and
# uspto->hitea because hitea has negatives but no source_id with BOTH
# pos+neg. This script uses the fixed check and generates negatives by
# product corruption, making MRR meaningful.
set -e
cd /home/cunyuliu/pc_cng_research
export CUDA_VISIBLE_DEVICES=7
export PYTHONPATH=/home/cunyuliu/pc_cng_research/chem_negative_sampling
mkdir -p results/logs
echo "[P3-03-FIXED-V2] Starting at $(date)"
echo "[P3-03-FIXED-V2] GPU: 7"
echo "[P3-03-FIXED-V2] Pairs: ord->hitea, uspto->hitea (hitea meaningful-groups fix)"
exec /home/cunyuliu/miniconda3/envs/pc_cng_gpu/bin/python /home/cunyuliu/pc_cng_research/chem_negative_sampling/training/p3_03_fix_negatives.py \
    --pairs 'ord->hitea,uspto->hitea' \
    --backbone-ckpt results/pretrained_backbone_chemformer_lora_20260720/seed20260710/model.pt \
    --vocab external/reaction_lm/Chemformer/bart_vocab.json \
    --data-dir data/processed \
    --seeds 20260710,20260711,20260712,20260713,20260714,20260715,20260716,20260717,20260718,20260719 \
    --output-dir results/cross_dataset_finetune_head_fixed_v2_20260721 \
    --n-few-shot 0.1 --epochs 5 --lr 1e-4 --device cuda:0 \
    --train-idx data/processed/train_idx_v3.json --val-idx data/processed/val_idx_v3.json --test-idx data/processed/test_idx_v3.json \
    --n-negatives 4 --bootstrap-iterations 10000
