#!/bin/bash
set -e
cd /home/cunyuliu/pc_cng_research
export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=/home/cunyuliu/pc_cng_research/chem_negative_sampling
mkdir -p results/logs
exec /home/cunyuliu/miniconda3/envs/pc_cng_gpu/bin/python -m training.finetune_head \
    --pairs all \
    --backbone-ckpt results/pretrained_backbone_chemformer_lora_20260720/seed20260710/model.pt \
    --vocab external/reaction_lm/Chemformer/bart_vocab.json \
    --data-dir data/processed \
    --seeds 20260710,20260711,20260712,20260713,20260714,20260715,20260716,20260717,20260718,20260719 \
    --output-dir results/cross_dataset_finetune_head_20260720 \
    --n-few-shot 0.1 --epochs 5 --lr 1e-4 --device cuda:0 \
    --train-idx data/processed/train_idx_v3.json \
    --val-idx data/processed/val_idx_v3.json \
    --test-idx data/processed/test_idx_v3.json
