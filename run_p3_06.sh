#!/bin/bash
# P3-06: Multi-task joint training (retrosynthesis + condition + yield).
#
# Demonstrates shared representation benefits by training a multi-task model
# (3 heads jointly) vs. single-task baselines (1 head each) on the same
# P3-01 frozen + LoRA backbone. Uses uncertainty weighting (Kendall 2018).
#
# GPU allocation (server):
#   0 = P3-04 done, 1 = P3-05, 2 = P3-03, 3 = P3-06 (this), 4 = calibrate,
#   6/7 = done.
#
# Output: results/multitask_joint_training_20260720/
#   - seed{SEED}/metrics.json  : per-seed multitask + singletask metrics
#   - summary.json             : aggregated metrics + paired bootstrap CI
#   - summary.md               : human-readable summary
set -e
cd /home/cunyuliu/pc_cng_research
export CUDA_VISIBLE_DEVICES=3  # GPU 3 (0=P3-04 done, 1=P3-05, 2=P3-03, 4=calibrate, 6/7=done)
export PYTHONPATH=/home/cunyuliu/pc_cng_research/chem_negative_sampling
mkdir -p results/logs

exec /home/cunyuliu/miniconda3/envs/pc_cng_gpu/bin/python -m models.multitask \
    --backbone-ckpt results/pretrained_backbone_chemformer_lora_20260720/seed20260710/model.pt \
    --vocab external/reaction_lm/Chemformer/bart_vocab.json \
    --data-dir data/processed \
    --seeds 20260710,20260711,20260712,20260713,20260714,20260715,20260716,20260717,20260718,20260719 \
    --output-dir results/multitask_joint_training_20260720 \
    --epochs 5 --lr 1e-4 --batch-size 16 --device cuda:0 \
    --uncertainty-weighting \
    --train-idx data/processed/train_idx_v3.json \
    --val-idx data/processed/val_idx_v3.json \
    --test-idx data/processed/test_idx_v3.json
