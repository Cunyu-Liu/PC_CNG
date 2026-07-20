#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/cunyuliu/pc_cng_research}
PYTHON_BIN=${PYTHON_BIN:-/home/cunyuliu/miniconda3/envs/pc_cng_gpu/bin/python}
GPU_TRAIN=${GPU_TRAIN:-4}
GPU_EVAL=${GPU_EVAL:-$GPU_TRAIN}

CODE_DIR="$ROOT/chem_negative_sampling"
RESULTS_DIR=${RESULTS_DIR:-$ROOT/results/type1_curated_weak_class_contexts_20260711}
LOG_DIR=${LOG_DIR:-$ROOT/results/logs}

REGIO_ALIGNMENT=${REGIO_ALIGNMENT:-$ROOT/data/processed/regiosqm20_normalized.csv}
HITEA_ALIGNMENT=${HITEA_ALIGNMENT:-$ROOT/data/processed/hitea_full_normalized.csv}
USPTO_NORMALIZED=${USPTO_NORMALIZED:-$ROOT/data/processed/uspto_openmolecules_normalized.csv}
HITEA_ULLMANN=${HITEA_ULLMANN:-$ROOT/external/HiTEA/data/cleaned_datasets/ullmann.csv}

BASE_REVIEWED=${BASE_REVIEWED:-$ROOT/results/type1_diverse_anchor_full/diverse_anchor_candidates_reviewed.csv}
QUOTA_REVIEWED=${QUOTA_REVIEWED:-$ROOT/results/type1_class_quota_supplement_20260711/class_quota_candidates_reviewed.csv}
FALLBACK_REVIEWED=${FALLBACK_REVIEWED:-$ROOT/results/type1_class_fallback_supplement_20260711/class_fallback_candidates_reviewed.csv}
PARTIAL_REVIEWED=${PARTIAL_REVIEWED:-$ROOT/results/type1_partial_product_supplement_20260711/partial_product_candidates_reviewed.csv}
UNREACTED_REVIEWED=${UNREACTED_REVIEWED:-$ROOT/results/type1_unreacted_substrate_supplement_v2_20260711/unreacted_substrate_candidates_reviewed.csv}

CURATED_CSV="$RESULTS_DIR/curated_weak_class_contexts.csv"
CURATED_SUMMARY="$RESULTS_DIR/curated_weak_class_contexts_summary.json"
CURATED_RAW="$RESULTS_DIR/curated_class_fallback_candidates.csv"
CURATED_RAW_SUMMARY="$RESULTS_DIR/curated_class_fallback_summary.json"
CURATED_REVIEWED="$RESULTS_DIR/curated_class_fallback_candidates_reviewed.csv"
CURATED_REVIEW_SUMMARY="$RESULTS_DIR/curated_class_fallback_review_summary.json"

OUT_DIR=${OUT_DIR:-$RESULTS_DIR/curated_augmented_pairwise_seed20260711}
SEED=${SEED:-20260711}

mkdir -p "$RESULTS_DIR" "$LOG_DIR"
cd "$CODE_DIR"

if [[ ! -f "$CURATED_SUMMARY" ]]; then
  echo "[build] curated weak-class positive contexts"
  PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.build_curated_weak_class_contexts \
    --hitea-cleaned-csv "Cu coupling=$HITEA_ULLMANN" \
    --uspto-csv "$USPTO_NORMALIZED" \
    --output "$CURATED_CSV" \
    --summary "$CURATED_SUMMARY" \
    --max-uspto-rows-per-class "${MAX_USPTO_ROWS_PER_CLASS:-200}" \
    > "$LOG_DIR/type1_curated_weak_class_context_build.log" 2>&1
fi

if [[ ! -f "$RESULTS_DIR/source_support_with_curated_contexts/source_support_audit.json" ]]; then
  echo "[support] source support after adding curated positives"
  PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.audit_reaction_class_source_support \
    --positive-csv "$HITEA_ALIGNMENT" \
    --positive-csv "$CURATED_CSV" \
    --synthetic-csv "$BASE_REVIEWED" \
    --synthetic-csv "$QUOTA_REVIEWED" \
    --synthetic-csv "$FALLBACK_REVIEWED" \
    --synthetic-csv "$PARTIAL_REVIEWED" \
    --synthetic-csv "$UNREACTED_REVIEWED" \
    --output-dir "$RESULTS_DIR/source_support_with_curated_contexts" \
    --min-groups 20 \
    --max-missing-contexts-per-class 50 \
    --include-reaction-class "Amide coupling" \
    --include-reaction-class "Cu coupling" \
    --include-reaction-class "Ni coupling" \
    --include-reaction-class "Hydrogenation" \
    --include-reaction-class "Rh coupling" \
    > "$LOG_DIR/type1_curated_weak_class_source_support_before_generation.log" 2>&1
fi

if [[ ! -f "$CURATED_RAW_SUMMARY" ]]; then
  echo "[generate] curated Amide/Cu class-fallback candidates"
  PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.run_hard_negative_actions \
    --input "$CURATED_CSV" \
    --known-positive "$HITEA_ALIGNMENT" \
    --known-positive "$REGIO_ALIGNMENT" \
    --known-positive "$CURATED_CSV" \
    --exclude-candidate-csv "$BASE_REVIEWED" \
    --exclude-candidate-csv "$QUOTA_REVIEWED" \
    --exclude-candidate-csv "$FALLBACK_REVIEWED" \
    --exclude-candidate-csv "$PARTIAL_REVIEWED" \
    --exclude-candidate-csv "$UNREACTED_REVIEWED" \
    --output "$CURATED_RAW" \
    --summary "$CURATED_RAW_SUMMARY" \
    --action class_fallback \
    --include-reaction-class "Amide coupling" \
    --include-reaction-class "Cu coupling" \
    --max-candidates-per-reaction 4 \
    --min-product-similarity 0.0 \
    --max-product-similarity 0.98 \
    --progress-every 100 \
    > "$LOG_DIR/type1_curated_weak_class_fallback_generate.log" 2>&1
fi

if [[ ! -f "$CURATED_REVIEW_SUMMARY" ]]; then
  echo "[review] curated Amide/Cu class-fallback candidates"
  PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.false_negative_review \
    --input "$CURATED_RAW" \
    --output "$CURATED_REVIEWED" \
    --summary "$CURATED_REVIEW_SUMMARY" \
    --known-positive "$HITEA_ALIGNMENT" \
    --known-positive "$REGIO_ALIGNMENT" \
    --known-positive "$CURATED_CSV" \
    > "$LOG_DIR/type1_curated_weak_class_fallback_review.log" 2>&1
fi

if [[ ! -f "$RESULTS_DIR/source_support_after_curated_fallback/source_support_audit.json" ]]; then
  echo "[support] source support after curated fallback review"
  PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.audit_reaction_class_source_support \
    --positive-csv "$HITEA_ALIGNMENT" \
    --positive-csv "$CURATED_CSV" \
    --synthetic-csv "$BASE_REVIEWED" \
    --synthetic-csv "$QUOTA_REVIEWED" \
    --synthetic-csv "$FALLBACK_REVIEWED" \
    --synthetic-csv "$PARTIAL_REVIEWED" \
    --synthetic-csv "$UNREACTED_REVIEWED" \
    --synthetic-csv "$CURATED_REVIEWED" \
    --output-dir "$RESULTS_DIR/source_support_after_curated_fallback" \
    --min-groups 20 \
    --max-missing-contexts-per-class 50 \
    --include-reaction-class "Amide coupling" \
    --include-reaction-class "Cu coupling" \
    --include-reaction-class "Ni coupling" \
    --include-reaction-class "Hydrogenation" \
    --include-reaction-class "Rh coupling" \
    > "$LOG_DIR/type1_curated_weak_class_source_support_after_generation.log" 2>&1
fi

if [[ "${STOP_AFTER_REVIEW:-0}" == "1" ]]; then
  echo "Curated weak-class generation/review complete: $CURATED_REVIEWED"
  exit 0
fi

if [[ ! -f "$OUT_DIR/metrics.json" ]]; then
  echo "[train] curated augmented pairwise seed=$SEED"
  rm -rf "$OUT_DIR"
  CUDA_VISIBLE_DEVICES="$GPU_TRAIN" PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.train_pairwise_reward_mlp \
    --real-csv "$REGIO_ALIGNMENT" \
    --real-csv "$HITEA_ALIGNMENT" \
    --real-csv "$CURATED_CSV" \
    --synthetic-csv "$BASE_REVIEWED" \
    --synthetic-csv "$QUOTA_REVIEWED" \
    --synthetic-csv "$FALLBACK_REVIEWED" \
    --synthetic-csv "$PARTIAL_REVIEWED" \
    --synthetic-csv "$UNREACTED_REVIEWED" \
    --synthetic-csv "$CURATED_REVIEWED" \
    --output-dir "$OUT_DIR" \
    --epochs "${EPOCHS:-80}" \
    --batch-size "${BATCH_SIZE:-4096}" \
    --hidden-dim "${HIDDEN_DIM:-2048}" \
    --n-bits "${N_BITS:-4096}" \
    --dropout "${DROPOUT:-0.20}" \
    --seed "$SEED" \
    > "$LOG_DIR/type1_curated_weak_class_pairwise_train.log" 2>&1
fi

if [[ ! -f "$OUT_DIR/rerank_same_split/ranking_metrics.json" ]]; then
  echo "[rerank] curated augmented pairwise seed=$SEED"
  CUDA_VISIBLE_DEVICES="$GPU_EVAL" PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.evaluate_candidate_reranking \
    --real-csv "$REGIO_ALIGNMENT" \
    --real-csv "$HITEA_ALIGNMENT" \
    --real-csv "$CURATED_CSV" \
    --synthetic-csv "$BASE_REVIEWED" \
    --synthetic-csv "$QUOTA_REVIEWED" \
    --synthetic-csv "$FALLBACK_REVIEWED" \
    --synthetic-csv "$PARTIAL_REVIEWED" \
    --synthetic-csv "$UNREACTED_REVIEWED" \
    --synthetic-csv "$CURATED_REVIEWED" \
    --model-dir "$OUT_DIR" \
    --output-dir "$OUT_DIR/rerank_same_split" \
    --candidate-scope same_split \
    --batch-size "${BATCH_SIZE:-4096}" \
    --device cuda \
    > "$LOG_DIR/type1_curated_weak_class_pairwise_rerank.log" 2>&1
fi

if [[ ! -f "$RESULTS_DIR/reaction_class_curated_augmented/reaction_class_benchmark.json" ]]; then
  echo "[class] curated augmented reaction-class benchmark"
  PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.analyze_reaction_class_benchmark \
    --score-csv "curated_augmented=$OUT_DIR/rerank_same_split/candidate_scores.csv:score" \
    --output-dir "$RESULTS_DIR/reaction_class_curated_augmented" \
    --min-groups 20 \
    --weak-top1 0.80 \
    --weak-mrr 0.85 \
    > "$LOG_DIR/type1_curated_weak_class_reaction_class_benchmark.log" 2>&1
fi

echo "Curated weak-class augmented experiment complete: $OUT_DIR"
