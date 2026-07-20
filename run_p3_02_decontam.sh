#!/bin/bash
# P3-02 (decontaminated): re-run the 10-seed SOTA comparison v2 with the
# decontamination filter enabled. This fixes the Tanimoto-NN data-leakage
# artifact where test-set products appear verbatim in the train set
# (Tanimoto=1.0 nearest neighbour trivially recovers the label, inflating
# Tanimoto-NN MRR to 1.0 and making PC-CNG look unfairly worse).
#
# With --decontaminate:
#   * Primary metrics (summary.json::metrics, paired_significance.json,
#     go_no_go_decision.json) reflect the FAIR decontaminated comparison.
#   * Contaminated reference metrics are preserved under metrics_contam /
#     paired_significance_contam for transparency.
#   * n_leaked / leak_rate recorded in summary.json::decontamination.
#
# Output dir is NEW (dated 20260721) so the existing contaminated run at
# results/sota_comparison_v2_uspto_mit_50k_20260720 is preserved.
set -e
cd /home/cunyuliu/pc_cng_research
export CUDA_VISIBLE_DEVICES=7
export PYTHONPATH=/home/cunyuliu/pc_cng_research/chem_negative_sampling
mkdir -p results/logs

exec /home/cunyuliu/miniconda3/envs/pc_cng_gpu/bin/python -m pc_cng.run_sota_comparison_v2 \
    --pc-cng-negatives results/uspto_openmolecules_full_generation/pc_cng_synthetic_negatives_reviewed.csv \
    --output-dir results/sota_comparison_v2_decontam_20260721 \
    --methods rdkit_template,heuristic_validator,tanimoto_nn,pc_cng,chemformer_scorer \
    --seeds 20260710,20260711,20260712,20260713,20260714,20260715,20260716,20260717,20260718,20260719 \
    --max-sources 2000 \
    --chemformer-ckpt models/reaction_lm/chemformer_pretrained_hf/model_sanitized.ckpt \
    --chemformer-vocab external/reaction_lm/Chemformer/bart_vocab.json \
    --chemformer-device cuda:0 \
    --chemformer-batch-size 16 \
    --chemformer-epochs 100 \
    --epochs 200 \
    --bootstrap-iterations 10000 \
    --decontaminate
