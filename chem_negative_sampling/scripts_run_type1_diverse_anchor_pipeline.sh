#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/cunyuliu/pc_cng_research}
PYTHON_BIN=${PYTHON_BIN:-/home/cunyuliu/miniconda3/envs/pc_cng_gpu/bin/python}
GPU_TRAIN=${GPU_TRAIN:-0}

CODE_DIR="$ROOT/chem_negative_sampling"
LOG_DIR="$ROOT/results/logs"
RESULTS_DIR="$ROOT/results/type1_diverse_anchor_full"

REGIO_ALIGNMENT="$ROOT/data/processed/regiosqm20_normalized.csv"
HIT_ALIGNMENT="$ROOT/data/processed/hitea_full_normalized.csv"

CANDIDATES="$RESULTS_DIR/diverse_anchor_candidates.csv"
SUMMARY="$RESULTS_DIR/diverse_anchor_summary.json"
REVIEWED="$RESULTS_DIR/diverse_anchor_candidates_reviewed.csv"
REVIEW_SUMMARY="$RESULTS_DIR/diverse_anchor_review_summary.json"

PAIRWISE_DIR="$ROOT/results/type1_diverse_anchor_pairwise_reward_h2048_n4096_e80"
RERANK_DIR="$ROOT/results/candidate_reranking_eval/type1_diverse_anchor_pairwise"
RERANK_ALL_DIR="$ROOT/results/candidate_reranking_eval_all_group/type1_diverse_anchor_pairwise"
ACTION_FAMILY_DIR="$RESULTS_DIR/action_family_contribution"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"
cd "$CODE_DIR"

if [[ ! -f "$SUMMARY" ]]; then
  echo "[1/6] Generate diverse-anchor type-1 candidates"
  PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.run_hard_negative_actions \
    --input "$REGIO_ALIGNMENT" \
    --input "$HIT_ALIGNMENT" \
    --known-positive "$HIT_ALIGNMENT" \
    --known-positive "$REGIO_ALIGNMENT" \
    --output "$CANDIDATES" \
    --summary "$SUMMARY" \
    --action heteroatom \
    --action regio \
    --map-unmapped \
    --diverse-anchor \
    --max-candidates-per-reaction "${MAX_CANDIDATES_PER_REACTION:-12}" \
    --max-candidates-per-pair "${MAX_CANDIDATES_PER_PAIR:-24}" \
    --max-anchor-distance "${MAX_ANCHOR_DISTANCE:-8}" \
    --min-product-similarity "${MIN_PRODUCT_SIMILARITY:-0.45}" \
    --max-product-similarity "${MAX_PRODUCT_SIMILARITY:-0.995}" \
    --min-atom-balance "${MIN_ATOM_BALANCE:-0.35}" \
    --progress-every "${PROGRESS_EVERY:-1000}" \
    > "$LOG_DIR/type1_diverse_anchor_generate.log" 2>&1
fi

if [[ ! -f "$REVIEW_SUMMARY" ]]; then
  echo "[2/6] Review diverse-anchor candidates"
  PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.false_negative_review \
    --input "$CANDIDATES" \
    --output "$REVIEWED" \
    --summary "$REVIEW_SUMMARY" \
    --known-positive "$HIT_ALIGNMENT" \
    --known-positive "$REGIO_ALIGNMENT" \
    > "$LOG_DIR/type1_diverse_anchor_review.log" 2>&1
fi

if [[ ! -f "$PAIRWISE_DIR/metrics.json" ]]; then
  echo "[3/6] Train full-parameter pairwise reward model"
  mkdir -p "$PAIRWISE_DIR"
  CUDA_VISIBLE_DEVICES="$GPU_TRAIN" PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.train_pairwise_reward_mlp \
    --real-csv "$REGIO_ALIGNMENT" \
    --real-csv "$HIT_ALIGNMENT" \
    --synthetic-csv "$REVIEWED" \
    --output-dir "$PAIRWISE_DIR" \
    --epochs "${EPOCHS:-80}" \
    --batch-size "${BATCH_SIZE:-4096}" \
    --lr "${LR:-0.001}" \
    --hidden-dim "${HIDDEN_DIM:-2048}" \
    --n-bits "${N_BITS:-4096}" \
    --dropout "${DROPOUT:-0.20}" \
    --pairwise-weight "${PAIRWISE_WEIGHT:-1.0}" \
    --bce-weight "${BCE_WEIGHT:-1.0}" \
    --seed "${SEED:-20260710}" \
    > "$LOG_DIR/type1_diverse_anchor_pairwise_train.log" 2>&1
fi

if [[ ! -f "$RERANK_DIR/ranking_metrics.json" ]]; then
  echo "[4/6] Evaluate same-split candidate reranking"
  PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.evaluate_candidate_reranking \
    --real-csv "$REGIO_ALIGNMENT" \
    --real-csv "$HIT_ALIGNMENT" \
    --synthetic-csv "$REVIEWED" \
    --model-dir "$PAIRWISE_DIR" \
    --output-dir "$RERANK_DIR" \
    --candidate-scope same_split \
    --batch-size "${BATCH_SIZE:-4096}" \
    --device cuda \
    > "$LOG_DIR/type1_diverse_anchor_rerank_same_split.log" 2>&1
fi

if [[ ! -f "$RERANK_ALL_DIR/ranking_metrics.json" ]]; then
  echo "[5/6] Evaluate all-group candidate reranking"
  PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.evaluate_candidate_reranking \
    --real-csv "$REGIO_ALIGNMENT" \
    --real-csv "$HIT_ALIGNMENT" \
    --synthetic-csv "$REVIEWED" \
    --model-dir "$PAIRWISE_DIR" \
    --output-dir "$RERANK_ALL_DIR" \
    --candidate-scope all_group \
    --batch-size "${BATCH_SIZE:-4096}" \
    --device cuda \
    > "$LOG_DIR/type1_diverse_anchor_rerank_all_group.log" 2>&1
fi

echo "[6/6] Analyze action-family contribution"
PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.analyze_action_family_contribution \
  --synthetic-csv "$REVIEWED" \
  --real-csv "$REGIO_ALIGNMENT" \
  --real-csv "$HIT_ALIGNMENT" \
  --score-csv "diverse_new_pairwise=$RERANK_DIR/candidate_scores.csv" \
  --output-dir "$ACTION_FAMILY_DIR" \
  > "$LOG_DIR/type1_diverse_anchor_action_family.log" 2>&1

echo "Type-1 diverse-anchor pipeline complete."
