#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/cunyuliu/pc_cng_research}
PYTHON_BIN=${PYTHON_BIN:-/home/cunyuliu/miniconda3/envs/pc_cng_gpu/bin/python}
GPU_DECODER=${GPU_DECODER:-0}
GPU_DOWNSTREAM=${GPU_DOWNSTREAM:-1}

CODE_DIR="$ROOT/chem_negative_sampling"
RESULTS_DIR="$ROOT/results/edit_decoder_v3_full"
LOG_DIR="$ROOT/results/logs"

HIT_ALIGNMENT="$ROOT/data/processed/hitea_full_normalized.csv"
REGIO_ALIGNMENT="$ROOT/data/processed/regiosqm20_normalized.csv"
HIT_POS="$ROOT/data/processed/hitea_full_positives.csv"
REGIO_POS="$ROOT/data/processed/regiosqm20_positives.csv"

CANDIDATES="$RESULTS_DIR/edit_decoder_candidates.csv"
CANDIDATE_SUMMARY="$RESULTS_DIR/edit_decoder_candidates_summary.json"
DECODER_DIR="$RESULTS_DIR/reaction_center_edit_decoder"
DECODER_CKPT="$DECODER_DIR/best_reaction_center_edit_decoder.pt"

HIT_LEARNED="$RESULTS_DIR/hitea_learned_boundary_negatives.csv"
HIT_LEARNED_SUMMARY="$RESULTS_DIR/hitea_learned_boundary_summary.json"
REGIO_LEARNED="$RESULTS_DIR/regiosqm20_learned_boundary_negatives.csv"
REGIO_LEARNED_SUMMARY="$RESULTS_DIR/regiosqm20_learned_boundary_summary.json"

HIT_REVIEWED="$RESULTS_DIR/hitea_learned_boundary_negatives_reviewed.csv"
HIT_REVIEW_SUMMARY="$RESULTS_DIR/hitea_learned_boundary_review_summary.json"
REGIO_REVIEWED="$RESULTS_DIR/regiosqm20_learned_boundary_negatives_reviewed.csv"
REGIO_REVIEW_SUMMARY="$RESULTS_DIR/regiosqm20_learned_boundary_review_summary.json"

PAIRWISE_DIR="$ROOT/results/v3_learned_pairwise_reward_h2048_n4096_e80"
BCE_DIR="$ROOT/results/v3_learned_direct_bce_h2048_n4096_e80"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"
cd "$CODE_DIR"

if [ ! -f "$CANDIDATE_SUMMARY" ]; then
  echo "[1/7] Building edit-decoder candidate dataset"
  PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.build_edit_decoder_dataset \
    --input "$HIT_ALIGNMENT" \
    --input "$REGIO_ALIGNMENT" \
    --output "$CANDIDATES" \
    --summary "$CANDIDATE_SUMMARY" \
    --map-unmapped \
    --max-candidates-per-pair 10 \
    > "$LOG_DIR/edit_decoder_v3_build_dataset.log" 2>&1
fi

if [ ! -f "$DECODER_DIR/metrics.json" ]; then
  echo "[2/7] Training reaction-center edit decoder"
  mkdir -p "$DECODER_DIR"
  CUDA_VISIBLE_DEVICES="$GPU_DECODER" PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.train_reaction_center_edit_decoder \
    --input "$CANDIDATES" \
    --output-dir "$DECODER_DIR" \
    --epochs 80 \
    --batch-size 1024 \
    --hidden-dim 512 \
    --dropout 0.15 \
    --lr 0.001 \
    > "$LOG_DIR/edit_decoder_v3_train_decoder.log" 2>&1
fi

if [ ! -f "$HIT_LEARNED_SUMMARY" ]; then
  echo "[3/7] Generating learned HiTEA boundary negatives"
  CUDA_VISIBLE_DEVICES="$GPU_DECODER" PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.run_learned_boundary_generation \
    --input "$HIT_POS" \
    --checkpoint "$DECODER_CKPT" \
    --output "$HIT_LEARNED" \
    --summary "$HIT_LEARNED_SUMMARY" \
    --top-k 2 \
    --known-positive "$HIT_ALIGNMENT" \
    --known-positive "$REGIO_ALIGNMENT" \
    > "$LOG_DIR/edit_decoder_v3_generate_hitea.log" 2>&1
fi

if [ ! -f "$REGIO_LEARNED_SUMMARY" ]; then
  echo "[4/7] Generating learned RegioSQM20 boundary negatives"
  CUDA_VISIBLE_DEVICES="$GPU_DECODER" PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.run_learned_boundary_generation \
    --input "$REGIO_POS" \
    --checkpoint "$DECODER_CKPT" \
    --output "$REGIO_LEARNED" \
    --summary "$REGIO_LEARNED_SUMMARY" \
    --top-k 2 \
    --map-unmapped \
    --known-positive "$HIT_ALIGNMENT" \
    --known-positive "$REGIO_ALIGNMENT" \
    > "$LOG_DIR/edit_decoder_v3_generate_regio.log" 2>&1
fi

if [ ! -f "$HIT_REVIEW_SUMMARY" ]; then
  echo "[5/7] Reviewing learned boundary negatives"
  PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.false_negative_review \
    --input "$HIT_LEARNED" \
    --output "$HIT_REVIEWED" \
    --summary "$HIT_REVIEW_SUMMARY" \
    --known-positive "$HIT_ALIGNMENT" \
    --known-positive "$REGIO_ALIGNMENT" \
    > "$LOG_DIR/edit_decoder_v3_review_hitea.log" 2>&1

  PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.false_negative_review \
    --input "$REGIO_LEARNED" \
    --output "$REGIO_REVIEWED" \
    --summary "$REGIO_REVIEW_SUMMARY" \
    --known-positive "$HIT_ALIGNMENT" \
    --known-positive "$REGIO_ALIGNMENT" \
    > "$LOG_DIR/edit_decoder_v3_review_regio.log" 2>&1
fi

if [ ! -f "$PAIRWISE_DIR/metrics.json" ]; then
  echo "[6/7] Training downstream pairwise reward model with learned negatives"
  mkdir -p "$PAIRWISE_DIR"
  CUDA_VISIBLE_DEVICES="$GPU_DOWNSTREAM" PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.train_pairwise_reward_mlp \
    --real-csv "$REGIO_ALIGNMENT" \
    --real-csv "$HIT_ALIGNMENT" \
    --synthetic-csv "$REGIO_REVIEWED" \
    --synthetic-csv "$HIT_REVIEWED" \
    --output-dir "$PAIRWISE_DIR" \
    --epochs 80 \
    --batch-size 4096 \
    --lr 0.001 \
    --hidden-dim 2048 \
    --n-bits 4096 \
    --dropout 0.20 \
    --pairwise-weight 1.0 \
    --bce-weight 1.0 \
    > "$LOG_DIR/edit_decoder_v3_train_pairwise_reward.log" 2>&1
fi

if [ ! -f "$BCE_DIR/metrics.json" ]; then
  echo "[7/7] Training downstream BCE model with learned negatives"
  mkdir -p "$BCE_DIR"
  CUDA_VISIBLE_DEVICES="$GPU_DOWNSTREAM" PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.train_feasibility_mlp \
    --real-csv "$REGIO_ALIGNMENT" \
    --real-csv "$HIT_ALIGNMENT" \
    --synthetic-csv "$REGIO_REVIEWED" \
    --synthetic-csv "$HIT_REVIEWED" \
    --output-dir "$BCE_DIR" \
    --epochs 80 \
    --batch-size 4096 \
    --lr 0.001 \
    --hidden-dim 2048 \
    --n-bits 4096 \
    --dropout 0.20 \
    > "$LOG_DIR/edit_decoder_v3_train_direct_bce.log" 2>&1
fi

if [ -f "$ROOT/evaluate_stacked_ensemble.py" ]; then
  PYTHONPATH=. "$PYTHON_BIN" "$ROOT/evaluate_stacked_ensemble.py" > "$LOG_DIR/edit_decoder_v3_stacked_ensemble.log" 2>&1 || true
fi

echo "PC-CNG v3 edit-decoder pipeline complete."
