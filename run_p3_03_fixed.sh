#!/bin/bash
# P3-03 FIXED: Cross-dataset evaluation with on-the-fly negative generation
#
# BUG FIX: The original P3-03 run found MRR=1.0 for ALL variants because
# the cross-dataset CSVs contain only positive reactions (label_type=positive).
# With each source_id group having exactly 1 positive item, MRR is trivially 1.0.
#
# FIX: Generate 4 negatives per positive by corrupting the product:
#   positive: reactants>>original_product  (label=1)
#   negative: reactants>>random_product    (label=0)
# All examples (positive + its K negatives) share the same source_id.
#
# This makes MRR meaningful: rank the positive among K+1 candidates.
# Random baseline MRR ≈ 0.456 (for K=4).
set -e

cd /home/cunyuliu/pc_cng_research
export CUDA_VISIBLE_DEVICES=6
export PYTHONPATH=/home/cunyuliu/pc_cng_research/chem_negative_sampling
mkdir -p results/logs

echo "[P3-03-FIX] Starting at $(date)"
echo "[P3-03-FIX] GPU: $CUDA_VISIBLE_DEVICES"
echo "[P3-03-FIX] Output: results/cross_dataset_finetune_head_fixed_20260721"

# Run all 7 pairs (including uspto_to_ord which was already "complete" but with
# the MRR=1.0 bug).  Start with small datasets first for quick verification.
exec /home/cunyuliu/miniconda3/envs/pc_cng_gpu/bin/python /tmp/p3_03_fix_negatives.py \
    --pairs 'uspto->ord,hitea->ord,ord->hitea,uspto->hitea,ord->uspto,hitea->uspto,uspto_openmolecules->ord' \
    --backbone-ckpt results/pretrained_backbone_chemformer_lora_20260720/seed20260710/model.pt \
    --vocab external/reaction_lm/Chemformer/bart_vocab.json \
    --data-dir data/processed \
    --seeds 20260710,20260711,20260712,20260713,20260714,20260715,20260716,20260717,20260718,20260719 \
    --output-dir results/cross_dataset_finetune_head_fixed_20260721 \
    --n-few-shot 0.1 \
    --epochs 5 \
    --lr 1e-4 \
    --device cuda:0 \
    --train-idx data/processed/train_idx_v3.json \
    --val-idx data/processed/val_idx_v3.json \
    --test-idx data/processed/test_idx_v3.json \
    --n-negatives 4 \
    --bootstrap-iterations 10000
