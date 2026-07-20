#!/bin/bash
# P3-07: LLM-as-judge expert agent review (翻盘 P2-03 DEFERRED).
#
# Hybrid approach:
#   1. LLM-judge infrastructure (prompt + API call code) for future use
#   2. LocalExpertJudge (RDKit + atom balance + plausibility heuristics)
#      as offline fallback (server has no internet access)
#   3. Compare with DFT results from P2-02
#
# Output: results/llm_judge_20260720/{judgments.json, summary.md}
set -e
cd /home/cunyuliu/pc_cng_research
export PYTHONPATH=/home/cunyuliu/pc_cng_research/chem_negative_sampling
mkdir -p results/logs

# If the PC-CNG negatives file is missing, fall back to a HITEa positives
# subset so the pipeline degrades gracefully (the CLI handles the
# fallback automatically when --fallback-csv is provided).
FALLBACK_CSV=""
if [ -f "data/processed/hitea_full_positives.csv" ]; then
    FALLBACK_CSV="data/processed/hitea_full_positives.csv"
fi

# DFT results dir is optional -- the CLI skips DFT agreement if missing.
DFT_DIR=""
if [ -d "results/dft_validation_chemoselectivity_20260720" ]; then
    DFT_DIR="results/dft_validation_chemoselectivity_20260720"
fi

exec /home/cunyuliu/miniconda3/envs/pc_cng_gpu/bin/python -m evaluation.llm_judge \
    --pc-cng-negatives results/uspto_openmolecules_full_generation/pc_cng_synthetic_negatives_reviewed.csv \
    --fallback-csv "$FALLBACK_CSV" \
    --dft-results "$DFT_DIR" \
    --output-dir results/llm_judge_20260720 \
    --n-samples 100 \
    --judges local_expert_1,local_expert_2,local_expert_3 \
    --seeds 20260710
