#!/bin/bash
# P3-02: Full 10-seed SOTA comparison v2 with B5 Chemformer baseline.
# This script翻盘 P2-06 NO-GO by adding a real SOTA Transformer baseline.
set -e
cd /home/cunyuliu/pc_cng_research
export CUDA_VISIBLE_DEVICES=7
export PYTHONPATH=/home/cunyuliu/pc_cng_research/chem_negative_sampling
mkdir -p results/logs

exec /home/cunyuliu/miniconda3/envs/pc_cng_gpu/bin/python -m pc_cng.run_sota_comparison_v2 \
    --pc-cng-negatives results/uspto_openmolecules_full_generation/pc_cng_synthetic_negatives_reviewed.csv \
    --output-dir results/sota_comparison_v2_uspto_mit_50k_20260720 \
    --methods rdkit_template,heuristic_validator,tanimoto_nn,pc_cng,chemformer_scorer \
    --seeds 20260710,20260711,20260712,20260713,20260714,20260715,20260716,20260717,20260718,20260719 \
    --max-sources 2000 \
    --chemformer-ckpt models/reaction_lm/chemformer_pretrained_hf/model_sanitized.ckpt \
    --chemformer-vocab external/reaction_lm/Chemformer/bart_vocab.json \
    --chemformer-device cuda:0 \
    --chemformer-batch-size 16 \
    --chemformer-epochs 100 \
    --epochs 200 \
    --bootstrap-iterations 10000
