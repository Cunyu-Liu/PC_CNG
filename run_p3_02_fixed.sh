#!/bin/bash
# P3-02 fixed: re-run SOTA comparison with build_train_fingerprints bug fix.
# The bug: dedup by parent_product alone kept only label=1 (gold) entries,
# making Tanimoto-NN always return score=1.0. Fix: dedup by (product, label).
set -e
cd /home/cunyuliu/pc_cng_research
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/home/cunyuliu/pc_cng_research/chem_negative_sampling
mkdir -p results/logs
exec /home/cunyuliu/miniconda3/envs/pc_cng_gpu/bin/python -m pc_cng.run_sota_comparison_v2 \
    --pc-cng-negatives results/uspto_openmolecules_full_generation/pc_cng_synthetic_negatives_reviewed.csv \
    --output-dir results/sota_comparison_v2_fixed_20260721 \
    --methods rdkit_template,heuristic_validator,tanimoto_nn,pc_cng,chemformer_scorer \
    --seeds 20260710,20260711,20260712,20260713,20260714,20260715,20260716,20260717,20260718,20260719 \
    --max-sources 2000 \
    --chemformer-ckpt models/reaction_lm/chemformer_pretrained_hf/model_sanitized.ckpt \
    --chemformer-vocab external/reaction_lm/Chemformer/bart_vocab.json \
    --chemformer-device cuda:0 --chemformer-batch-size 16 --chemformer-epochs 100 \
    --epochs 200 --bootstrap-iterations 10000
