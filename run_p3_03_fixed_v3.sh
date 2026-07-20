#!/bin/bash
# P3-03 v3: Re-run hitea pairs with subsampled positives (3000) for speed.
# The v2 run (PID 927429) uses all 15498 hitea positives -> 77490 rows,
# which is too slow (~5+ min/seed). This v3 run subsamples to 3000
# positives (matching ord's 2910), giving 15000 rows and ~30s/seed.
set -e
cd /home/cunyuliu/pc_cng_research
export CUDA_VISIBLE_DEVICES=5
export PYTHONPATH=/home/cunyuliu/pc_cng_research/chem_negative_sampling
mkdir -p results/logs
echo "[P3-03-FIXED-V3] Starting at $(date)"
echo "[P3-03-FIXED-V3] GPU: 5"
echo "[P3-03-FIXED-V3] Pairs: ord->hitea, uspto->hitea (subsampled to 3000 positives)"
exec /home/cunyuliu/miniconda3/envs/pc_cng_gpu/bin/python /home/cunyuliu/pc_cng_research/chem_negative_sampling/training/p3_03_fix_negatives.py \
    --pairs 'ord->hitea,uspto->hitea' \
    --backbone-ckpt results/pretrained_backbone_chemformer_lora_20260720/seed20260710/model.pt \
    --vocab external/reaction_lm/Chemformer/bart_vocab.json \
    --data-dir data/processed \
    --seeds 20260710,20260711,20260712,20260713,20260714,20260715,20260716,20260717,20260718,20260719 \
    --output-dir results/cross_dataset_finetune_head_fixed_v3_20260721 \
    --n-few-shot 0.1 --epochs 5 --lr 1e-4 --device cuda:0 \
    --train-idx data/processed/train_idx_v3.json --val-idx data/processed/val_idx_v3.json --test-idx data/processed/test_idx_v3.json \
    --n-negatives 4 --bootstrap-iterations 10000 \
    --max-positives 3000
