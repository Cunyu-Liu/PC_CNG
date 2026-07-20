#!/bin/bash
# P3-03 FIXED: Full 10-seed run with on-the-fly negative generation
# Uses GPU 0 (now free after P3-02 completed)
# Runs 5 manageable pairs (ord/hitea targets), skips uspto target (530K rows too slow)
set -e

cd /home/cunyuliu/pc_cng_research
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/home/cunyuliu/pc_cng_research/chem_negative_sampling
mkdir -p results/logs

echo "[P3-03-FIXED] Starting at $(date)"
echo "[P3-03-FIXED] GPU: $CUDA_VISIBLE_DEVICES"
echo "[P3-03-FIXED] Pairs: uspto->ord, hitea->ord, ord->hitea, uspto->hitea, uspto_openmolecules->ord"

# Run 5 pairs with manageable target sizes (ord: 2910, hitea: 39546)
# Skip ord->uspto and hitea->uspto (uspto target: 530K rows -> 2.65M examples, too slow)
exec /home/cunyuliu/miniconda3/envs/pc_cng_gpu/bin/python /home/cunyuliu/pc_cng_research/chem_negative_sampling/training/p3_03_fix_negatives.py \
    --pairs 'uspto->ord,hitea->ord,ord->hitea,uspto->hitea,uspto_openmolecules->ord' \
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
