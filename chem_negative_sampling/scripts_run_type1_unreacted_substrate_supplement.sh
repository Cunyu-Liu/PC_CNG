#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/cunyuliu/pc_cng_research}
PYTHON_BIN=${PYTHON_BIN:-/home/cunyuliu/miniconda3/envs/pc_cng_gpu/bin/python}
GPU_TRAIN=${GPU_TRAIN:-4}
GPU_EVAL=${GPU_EVAL:-$GPU_TRAIN}

CODE_DIR="$ROOT/chem_negative_sampling"
RESULTS_DIR=${RESULTS_DIR:-$ROOT/results/type1_unreacted_substrate_supplement_v2_20260711}
LOG_DIR=${LOG_DIR:-$ROOT/results/logs}

REGIO_ALIGNMENT=${REGIO_ALIGNMENT:-$ROOT/data/processed/regiosqm20_normalized.csv}
HITEA_ALIGNMENT=${HITEA_ALIGNMENT:-$ROOT/data/processed/hitea_full_normalized.csv}
BASE_REVIEWED=${BASE_REVIEWED:-$ROOT/results/type1_diverse_anchor_full/diverse_anchor_candidates_reviewed.csv}
QUOTA_REVIEWED=${QUOTA_REVIEWED:-$ROOT/results/type1_class_quota_supplement_20260711/class_quota_candidates_reviewed.csv}
FALLBACK_REVIEWED=${FALLBACK_REVIEWED:-$ROOT/results/type1_class_fallback_supplement_20260711/class_fallback_candidates_reviewed.csv}
PARTIAL_REVIEWED=${PARTIAL_REVIEWED:-$ROOT/results/type1_partial_product_supplement_20260711/partial_product_candidates_reviewed.csv}

UNREACTED_RAW="$RESULTS_DIR/unreacted_substrate_candidates.csv"
UNREACTED_SUMMARY="$RESULTS_DIR/unreacted_substrate_summary.json"
UNREACTED_REVIEWED="$RESULTS_DIR/unreacted_substrate_candidates_reviewed.csv"
UNREACTED_REVIEW_SUMMARY="$RESULTS_DIR/unreacted_substrate_review_summary.json"
OUT_DIR=${OUT_DIR:-$RESULTS_DIR/unreacted_augmented_pairwise_seed20260710}
SEED=${SEED:-20260710}
RUN_ID=${RUN_ID:-seed${SEED}}
TRAIN_LOG=${TRAIN_LOG:-$LOG_DIR/type1_unreacted_augmented_pairwise_train_${RUN_ID}.log}
RERANK_LOG=${RERANK_LOG:-$LOG_DIR/type1_unreacted_augmented_pairwise_rerank_${RUN_ID}.log}

TARGET_CLASSES=(
  "Hydrogenation"
  "Rh coupling"
)

mkdir -p "$RESULTS_DIR" "$LOG_DIR"
cd "$CODE_DIR"

CLASS_ARGS=()
for reaction_class in "${TARGET_CLASSES[@]}"; do
  CLASS_ARGS+=(--include-reaction-class "$reaction_class")
done

if [[ ! -f "$UNREACTED_SUMMARY" ]]; then
  echo "[generate] unreacted-substrate candidates"
  PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.run_hard_negative_actions \
    --input "$HITEA_ALIGNMENT" \
    --known-positive "$HITEA_ALIGNMENT" \
    --known-positive "$REGIO_ALIGNMENT" \
    --exclude-candidate-csv "$BASE_REVIEWED" \
    --exclude-candidate-csv "$QUOTA_REVIEWED" \
    --exclude-candidate-csv "$FALLBACK_REVIEWED" \
    --exclude-candidate-csv "$PARTIAL_REVIEWED" \
    --exclude-review-status keep_synthetic_negative \
    --exclude-review-status discard_known_positive \
    --output "$UNREACTED_RAW" \
    --summary "$UNREACTED_SUMMARY" \
    --action unreacted_substrate \
    "${CLASS_ARGS[@]}" \
    --max-candidates-per-reaction 4 \
    --min-product-similarity 0.0 \
    --max-product-similarity 0.9999 \
    --progress-every 1000 \
    > "$LOG_DIR/type1_unreacted_substrate_supplement_generate.log" 2>&1
fi

if [[ ! -f "$UNREACTED_REVIEW_SUMMARY" ]]; then
  echo "[review] unreacted-substrate candidates"
  PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.false_negative_review \
    --input "$UNREACTED_RAW" \
    --output "$UNREACTED_REVIEWED" \
    --summary "$UNREACTED_REVIEW_SUMMARY" \
    --known-positive "$HITEA_ALIGNMENT" \
    --known-positive "$REGIO_ALIGNMENT" \
    > "$LOG_DIR/type1_unreacted_substrate_supplement_review.log" 2>&1
fi

if [[ "${STOP_AFTER_REVIEW:-0}" == "1" ]]; then
  echo "Unreacted-substrate generation/review complete: $UNREACTED_REVIEWED"
  exit 0
fi

if [[ ! -f "$OUT_DIR/metrics.json" ]]; then
  echo "[train] unreacted-substrate augmented pairwise seed=$SEED"
  rm -rf "$OUT_DIR"
  CUDA_VISIBLE_DEVICES="$GPU_TRAIN" PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.train_pairwise_reward_mlp \
    --real-csv "$REGIO_ALIGNMENT" \
    --real-csv "$HITEA_ALIGNMENT" \
    --synthetic-csv "$BASE_REVIEWED" \
    --synthetic-csv "$QUOTA_REVIEWED" \
    --synthetic-csv "$FALLBACK_REVIEWED" \
    --synthetic-csv "$PARTIAL_REVIEWED" \
    --synthetic-csv "$UNREACTED_REVIEWED" \
    --output-dir "$OUT_DIR" \
    --epochs "${EPOCHS:-80}" \
    --batch-size "${BATCH_SIZE:-4096}" \
    --hidden-dim "${HIDDEN_DIM:-2048}" \
    --n-bits "${N_BITS:-4096}" \
    --dropout "${DROPOUT:-0.20}" \
    --seed "$SEED" \
    > "$TRAIN_LOG" 2>&1
fi

if [[ ! -f "$OUT_DIR/rerank_same_split/ranking_metrics.json" ]]; then
  echo "[rerank] unreacted-substrate augmented pairwise seed=$SEED"
  CUDA_VISIBLE_DEVICES="$GPU_EVAL" PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.evaluate_candidate_reranking \
    --real-csv "$REGIO_ALIGNMENT" \
    --real-csv "$HITEA_ALIGNMENT" \
    --synthetic-csv "$BASE_REVIEWED" \
    --synthetic-csv "$QUOTA_REVIEWED" \
    --synthetic-csv "$FALLBACK_REVIEWED" \
    --synthetic-csv "$PARTIAL_REVIEWED" \
    --synthetic-csv "$UNREACTED_REVIEWED" \
    --model-dir "$OUT_DIR" \
    --output-dir "$OUT_DIR/rerank_same_split" \
    --candidate-scope same_split \
    --batch-size "${BATCH_SIZE:-4096}" \
    --device cuda \
    > "$RERANK_LOG" 2>&1
fi

if [[ ! -f "$RESULTS_DIR/reaction_class_unreacted_trained/reaction_class_benchmark.json" ]]; then
  echo "[class] unreacted-substrate trained reaction-class benchmark"
  PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.analyze_reaction_class_benchmark \
    --score-csv "unreacted_trained=$OUT_DIR/rerank_same_split/candidate_scores.csv:score" \
    --output-dir "$RESULTS_DIR/reaction_class_unreacted_trained" \
    --min-groups 20 \
    --weak-top1 0.80 \
    --weak-mrr 0.85 \
    > "$LOG_DIR/type1_unreacted_augmented_pairwise_class.log" 2>&1
fi

if [[ ! -f "$RESULTS_DIR/source_support_after_unreacted/source_support_audit.json" ]]; then
  echo "[support] source/molecular support after unreacted-substrate supplement"
  PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.audit_reaction_class_source_support \
    --positive-csv "$HITEA_ALIGNMENT" \
    --synthetic-csv "$BASE_REVIEWED" \
    --synthetic-csv "$QUOTA_REVIEWED" \
    --synthetic-csv "$FALLBACK_REVIEWED" \
    --synthetic-csv "$PARTIAL_REVIEWED" \
    --synthetic-csv "$UNREACTED_REVIEWED" \
    --output-dir "$RESULTS_DIR/source_support_after_unreacted" \
    --min-groups 20 \
    --max-missing-contexts-per-class 50 \
    --include-reaction-class "Hydrogenation" \
    --include-reaction-class "Rh coupling" \
    > "$LOG_DIR/type1_unreacted_source_support_audit.log" 2>&1
fi

echo "Unreacted-substrate augmented pairwise experiment complete: $OUT_DIR"
