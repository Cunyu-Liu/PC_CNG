#!/bin/bash
# P3-08: Comprehensive benchmark suite covering 6 dimensions
# - Dimension 1: Negative generation quality (validity / uniqueness / diversity)
# - Dimension 2: Downstream task improvement (retrosynthesis MRR / condition / yield)
# - Dimension 3: Cross-dataset generalization (7 pairs x 3 variants)
# - Dimension 4: Computational efficiency (latency / throughput / memory)
# - Dimension 5: Chemical plausibility (DFT rate / LLM-judge agreement)
# - Dimension 6: Ablation studies (PC-CNG components)
#
# Output: results/benchmark_suite_v3_20260720/{metrics.json, report.md}
set -e
cd /home/cunyuliu/pc_cng_research
export CUDA_VISIBLE_DEVICES=5  # GPU 5 (free)
export PYTHONPATH=/home/cunyuliu/pc_cng_research/chem_negative_sampling
mkdir -p results/logs
exec /home/cunyuliu/miniconda3/envs/pc_cng_gpu/bin/python -m evaluation.benchmark_suite \
    --results-dir results \
    --output-dir results/benchmark_suite_v3_20260720 \
    --backbone-ckpt results/pretrained_backbone_chemformer_lora_20260720/seed20260710/model.pt \
    --vocab external/reaction_lm/Chemformer/bart_vocab.json \
    --dimensions all
