#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/cunyuliu/pc_cng_research}
PYTHON_BIN=${PYTHON_BIN:-/home/cunyuliu/miniconda3/envs/pc_cng_gpu/bin/python}
GPU_DECODER=${GPU_DECODER:-0}
GPU_DOWNSTREAM=${GPU_DOWNSTREAM:-1}

CODE_DIR="$ROOT/chem_negative_sampling"
RESULTS_DIR="$ROOT/results/masked_hard_decoder_full"
LOG_DIR="$ROOT/results/logs"

HIT_ALIGNMENT="$ROOT/data/processed/hitea_full_normalized.csv"
REGIO_ALIGNMENT="$ROOT/data/processed/regiosqm20_normalized.csv"
CANDIDATES="$RESULTS_DIR/candidates.csv"
CANDIDATE_SUMMARY="$RESULTS_DIR/candidates_summary.json"
DECODER_DIR="$RESULTS_DIR/train_masked_hard_decoder"
RULE_NEG="$RESULTS_DIR/rule_hard_negatives.csv"
RULE_NEG_SUMMARY="$RESULTS_DIR/rule_hard_negatives_summary.json"
RULE_REVIEWED="$RESULTS_DIR/rule_hard_negatives_reviewed.csv"
RULE_REVIEW_SUMMARY="$RESULTS_DIR/rule_hard_negatives_review_summary.json"
PAIRWISE_DIR="$ROOT/results/rule_hard_negatives_pairwise_reward_h2048_n4096_e80"
BCE_DIR="$ROOT/results/rule_hard_negatives_direct_bce_h2048_n4096_e80"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"
cd "$CODE_DIR"

if [ ! -f "$CANDIDATE_SUMMARY" ]; then
  echo "[1/6] Build known-positive masked candidate dataset"
  PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.build_edit_decoder_dataset \
    --input "$HIT_ALIGNMENT" \
    --input "$REGIO_ALIGNMENT" \
    --output "$CANDIDATES" \
    --summary "$CANDIDATE_SUMMARY" \
    --map-unmapped \
    --known-positive "$HIT_ALIGNMENT" \
    --known-positive "$REGIO_ALIGNMENT" \
    --max-candidates-per-pair 10 \
    > "$LOG_DIR/masked_hard_build_dataset.log" 2>&1
fi

if [ ! -f "$DECODER_DIR/metrics.json" ]; then
  echo "[2/6] Train masked hard-negative decoder"
  mkdir -p "$DECODER_DIR"
  CUDA_VISIBLE_DEVICES="$GPU_DECODER" PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.train_reaction_center_edit_decoder \
    --input "$CANDIDATES" \
    --output-dir "$DECODER_DIR" \
    --epochs 80 \
    --batch-size 1024 \
    --hidden-dim 512 \
    --dropout 0.15 \
    --loss-mode masked_hard_negative \
    --positive-bce-weight 1.0 \
    --hard-bce-weight 1.0 \
    --positive-rank-weight 0.5 \
    --hard-rank-weight 1.0 \
    > "$LOG_DIR/masked_hard_train_decoder.log" 2>&1
fi

if [ ! -f "$RULE_NEG_SUMMARY" ]; then
  echo "[3/6] Export rule-selected hard negatives"
  PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.export_rule_hard_negatives \
    --input "$CANDIDATES" \
    --output "$RULE_NEG" \
    --summary "$RULE_NEG_SUMMARY" \
    --top-k 2 \
    > "$LOG_DIR/masked_hard_export_rule_negatives.log" 2>&1
fi

if [ ! -f "$RULE_REVIEW_SUMMARY" ]; then
  echo "[4/6] Review rule-selected hard negatives"
  PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.false_negative_review \
    --input "$RULE_NEG" \
    --output "$RULE_REVIEWED" \
    --summary "$RULE_REVIEW_SUMMARY" \
    --known-positive "$HIT_ALIGNMENT" \
    --known-positive "$REGIO_ALIGNMENT" \
    > "$LOG_DIR/masked_hard_review_rule_negatives.log" 2>&1
fi

if [ ! -f "$PAIRWISE_DIR/metrics.json" ]; then
  echo "[5/6] Train downstream pairwise reward model"
  mkdir -p "$PAIRWISE_DIR"
  CUDA_VISIBLE_DEVICES="$GPU_DOWNSTREAM" PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.train_pairwise_reward_mlp \
    --real-csv "$REGIO_ALIGNMENT" \
    --real-csv "$HIT_ALIGNMENT" \
    --synthetic-csv "$RULE_REVIEWED" \
    --output-dir "$PAIRWISE_DIR" \
    --epochs 80 \
    --batch-size 4096 \
    --lr 0.001 \
    --hidden-dim 2048 \
    --n-bits 4096 \
    --dropout 0.20 \
    --pairwise-weight 1.0 \
    --bce-weight 1.0 \
    --seed 20260710 \
    > "$LOG_DIR/rule_hard_pairwise_reward.log" 2>&1
fi

if [ ! -f "$BCE_DIR/metrics.json" ]; then
  echo "[6/6] Train downstream BCE model"
  mkdir -p "$BCE_DIR"
  CUDA_VISIBLE_DEVICES="$GPU_DOWNSTREAM" PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.train_feasibility_mlp \
    --real-csv "$REGIO_ALIGNMENT" \
    --real-csv "$HIT_ALIGNMENT" \
    --synthetic-csv "$RULE_REVIEWED" \
    --output-dir "$BCE_DIR" \
    --epochs 80 \
    --batch-size 4096 \
    --lr 0.001 \
    --hidden-dim 2048 \
    --n-bits 4096 \
    --dropout 0.20 \
    --seed 20260710 \
    > "$LOG_DIR/rule_hard_direct_bce.log" 2>&1
fi

if [ -f "$ROOT/evaluate_stacked_ensemble.py" ]; then
  PYTHONPATH=. "$PYTHON_BIN" "$ROOT/evaluate_stacked_ensemble.py" > "$LOG_DIR/masked_hard_stacked_ensemble.log" 2>&1 || true
fi

echo "Known-positive masked hard-negative decoder pipeline complete."
