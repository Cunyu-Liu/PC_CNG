#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/cunyuliu/pc_cng_research}
PYTHON_BIN=${PYTHON_BIN:-/home/cunyuliu/miniconda3/envs/pc_cng_gpu/bin/python}
GPU_TRAIN=${GPU_TRAIN:-4}
GPU_EVAL=${GPU_EVAL:-$GPU_TRAIN}

CODE_DIR="$ROOT/chem_negative_sampling"
RESULTS_DIR=${RESULTS_DIR:-$ROOT/results/type1_partial_product_supplement_20260711}
LOG_DIR=${LOG_DIR:-$ROOT/results/logs}

REGIO_ALIGNMENT=${REGIO_ALIGNMENT:-$ROOT/data/processed/regiosqm20_normalized.csv}
HITEA_ALIGNMENT=${HITEA_ALIGNMENT:-$ROOT/data/processed/hitea_full_normalized.csv}
BASE_REVIEWED=${BASE_REVIEWED:-$ROOT/results/type1_diverse_anchor_full/diverse_anchor_candidates_reviewed.csv}
QUOTA_REVIEWED=${QUOTA_REVIEWED:-$ROOT/results/type1_class_quota_supplement_20260711/class_quota_candidates_reviewed.csv}
FALLBACK_REVIEWED=${FALLBACK_REVIEWED:-$ROOT/results/type1_class_fallback_supplement_20260711/class_fallback_candidates_reviewed.csv}

PARTIAL_RAW="$RESULTS_DIR/partial_product_candidates.csv"
PARTIAL_SUMMARY="$RESULTS_DIR/partial_product_summary.json"
PARTIAL_REVIEWED="$RESULTS_DIR/partial_product_candidates_reviewed.csv"
PARTIAL_REVIEW_SUMMARY="$RESULTS_DIR/partial_product_review_summary.json"
OUT_DIR=${OUT_DIR:-$RESULTS_DIR/partial_product_augmented_pairwise_seed20260710}
SEED=${SEED:-20260710}

TARGET_CLASSES=(
  "Hydrogenation"
  "Ni coupling"
  "Amide coupling"
  "Cu coupling"
  "Rh coupling"
)

mkdir -p "$RESULTS_DIR" "$LOG_DIR"
cd "$CODE_DIR"

CLASS_ARGS=()
for reaction_class in "${TARGET_CLASSES[@]}"; do
  CLASS_ARGS+=(--include-reaction-class "$reaction_class")
done

if [[ ! -f "$PARTIAL_SUMMARY" ]]; then
  echo "[generate] partial-product candidates"
  PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.run_hard_negative_actions \
    --input "$HITEA_ALIGNMENT" \
    --known-positive "$HITEA_ALIGNMENT" \
    --known-positive "$REGIO_ALIGNMENT" \
    --exclude-candidate-csv "$BASE_REVIEWED" \
    --exclude-candidate-csv "$QUOTA_REVIEWED" \
    --exclude-candidate-csv "$FALLBACK_REVIEWED" \
    --output "$PARTIAL_RAW" \
    --summary "$PARTIAL_SUMMARY" \
    --action partial_product \
    "${CLASS_ARGS[@]}" \
    --max-candidates-per-reaction 16 \
    --min-product-similarity 0.0 \
    --max-product-similarity 0.98 \
    --progress-every 1000 \
    > "$LOG_DIR/type1_partial_product_supplement_generate.log" 2>&1
fi

if [[ ! -f "$PARTIAL_REVIEW_SUMMARY" ]]; then
  echo "[review] partial-product candidates"
  PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.false_negative_review \
    --input "$PARTIAL_RAW" \
    --output "$PARTIAL_REVIEWED" \
    --summary "$PARTIAL_REVIEW_SUMMARY" \
    --known-positive "$HITEA_ALIGNMENT" \
    --known-positive "$REGIO_ALIGNMENT" \
    > "$LOG_DIR/type1_partial_product_supplement_review.log" 2>&1
fi

if [[ "${STOP_AFTER_REVIEW:-0}" == "1" ]]; then
  echo "Partial-product generation/review complete: $PARTIAL_REVIEWED"
  exit 0
fi

if [[ ! -f "$OUT_DIR/metrics.json" ]]; then
  echo "[train] partial-product augmented pairwise seed=$SEED"
  rm -rf "$OUT_DIR"
  CUDA_VISIBLE_DEVICES="$GPU_TRAIN" PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.train_pairwise_reward_mlp \
    --real-csv "$REGIO_ALIGNMENT" \
    --real-csv "$HITEA_ALIGNMENT" \
    --synthetic-csv "$BASE_REVIEWED" \
    --synthetic-csv "$QUOTA_REVIEWED" \
    --synthetic-csv "$FALLBACK_REVIEWED" \
    --synthetic-csv "$PARTIAL_REVIEWED" \
    --output-dir "$OUT_DIR" \
    --epochs "${EPOCHS:-80}" \
    --batch-size "${BATCH_SIZE:-4096}" \
    --hidden-dim "${HIDDEN_DIM:-2048}" \
    --n-bits "${N_BITS:-4096}" \
    --dropout "${DROPOUT:-0.20}" \
    --seed "$SEED" \
    > "$LOG_DIR/type1_partial_product_augmented_pairwise_train.log" 2>&1
fi

if [[ ! -f "$OUT_DIR/rerank_same_split/ranking_metrics.json" ]]; then
  echo "[rerank] partial-product augmented pairwise seed=$SEED"
  CUDA_VISIBLE_DEVICES="$GPU_EVAL" PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.evaluate_candidate_reranking \
    --real-csv "$REGIO_ALIGNMENT" \
    --real-csv "$HITEA_ALIGNMENT" \
    --synthetic-csv "$BASE_REVIEWED" \
    --synthetic-csv "$QUOTA_REVIEWED" \
    --synthetic-csv "$FALLBACK_REVIEWED" \
    --synthetic-csv "$PARTIAL_REVIEWED" \
    --model-dir "$OUT_DIR" \
    --output-dir "$OUT_DIR/rerank_same_split" \
    --candidate-scope same_split \
    --batch-size "${BATCH_SIZE:-4096}" \
    --device cuda \
    > "$LOG_DIR/type1_partial_product_augmented_pairwise_rerank.log" 2>&1
fi

if [[ ! -f "$RESULTS_DIR/reaction_class_partial_product_trained/reaction_class_benchmark.json" ]]; then
  echo "[class] partial-product trained reaction-class benchmark"
  PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.analyze_reaction_class_benchmark \
    --score-csv "partial_product_trained=$OUT_DIR/rerank_same_split/candidate_scores.csv:score" \
    --output-dir "$RESULTS_DIR/reaction_class_partial_product_trained" \
    --min-groups 20 \
    --weak-top1 0.80 \
    --weak-mrr 0.85 \
    > "$LOG_DIR/type1_partial_product_augmented_pairwise_class.log" 2>&1
fi

echo "Partial-product augmented pairwise experiment complete: $OUT_DIR"
