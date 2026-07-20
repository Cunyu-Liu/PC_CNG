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
CURATED_CSV=${CURATED_CSV:-$RESULTS_DIR/curated_weak_class_contexts.csv}

BASE_REVIEWED=${BASE_REVIEWED:-$ROOT/results/type1_diverse_anchor_full/diverse_anchor_candidates_reviewed.csv}
QUOTA_REVIEWED=${QUOTA_REVIEWED:-$ROOT/results/type1_class_quota_supplement_20260711/class_quota_candidates_reviewed.csv}
FALLBACK_REVIEWED=${FALLBACK_REVIEWED:-$ROOT/results/type1_class_fallback_supplement_20260711/class_fallback_candidates_reviewed.csv}
PARTIAL_REVIEWED=${PARTIAL_REVIEWED:-$ROOT/results/type1_partial_product_supplement_20260711/partial_product_candidates_reviewed.csv}
UNREACTED_REVIEWED=${UNREACTED_REVIEWED:-$ROOT/results/type1_unreacted_substrate_supplement_v2_20260711/unreacted_substrate_candidates_reviewed.csv}
CURATED_REVIEWED=${CURATED_REVIEWED:-$RESULTS_DIR/curated_class_fallback_candidates_reviewed.csv}

SEED=${SEED:-20260711}
WEAK_CLASS_WEIGHT=${WEAK_CLASS_WEIGHT:-0.5}
WEIGHT_TAG=${WEIGHT_TAG:-classw050}
OUT_DIR=${OUT_DIR:-$RESULTS_DIR/curated_augmented_pairwise_${WEIGHT_TAG}_seed${SEED}}

mkdir -p "$RESULTS_DIR" "$LOG_DIR"
cd "$CODE_DIR"

if [[ ! -f "$OUT_DIR/metrics.json" ]]; then
  echo "[train] curated class-weighted pairwise seed=$SEED weak_class_weight=$WEAK_CLASS_WEIGHT"
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
    --class-weight "Amide coupling=$WEAK_CLASS_WEIGHT" \
    --class-weight "Cu coupling=$WEAK_CLASS_WEIGHT" \
    --seed "$SEED" \
    > "$LOG_DIR/type1_curated_${WEIGHT_TAG}_pairwise_train.log" 2>&1
fi

if [[ ! -f "$OUT_DIR/rerank_expanded_scope/ranking_metrics.json" ]]; then
  echo "[rerank] expanded curated scope"
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
    --output-dir "$OUT_DIR/rerank_expanded_scope" \
    --candidate-scope same_split \
    --batch-size "${BATCH_SIZE:-4096}" \
    --device cuda \
    > "$LOG_DIR/type1_curated_${WEIGHT_TAG}_expanded_rerank.log" 2>&1
fi

if [[ ! -f "$OUT_DIR/rerank_original_scope/ranking_metrics.json" ]]; then
  echo "[rerank] original Regio/HiTEA scope"
  CUDA_VISIBLE_DEVICES="$GPU_EVAL" PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.evaluate_candidate_reranking \
    --real-csv "$REGIO_ALIGNMENT" \
    --real-csv "$HITEA_ALIGNMENT" \
    --synthetic-csv "$BASE_REVIEWED" \
    --synthetic-csv "$QUOTA_REVIEWED" \
    --synthetic-csv "$FALLBACK_REVIEWED" \
    --synthetic-csv "$PARTIAL_REVIEWED" \
    --synthetic-csv "$UNREACTED_REVIEWED" \
    --model-dir "$OUT_DIR" \
    --output-dir "$OUT_DIR/rerank_original_scope" \
    --candidate-scope same_split \
    --batch-size "${BATCH_SIZE:-4096}" \
    --device cuda \
    > "$LOG_DIR/type1_curated_${WEIGHT_TAG}_original_rerank.log" 2>&1
fi

if [[ ! -f "$RESULTS_DIR/reaction_class_curated_${WEIGHT_TAG}_expanded/reaction_class_benchmark.json" ]]; then
  echo "[class] expanded curated scope"
  PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.analyze_reaction_class_benchmark \
    --score-csv "curated_${WEIGHT_TAG}_expanded=$OUT_DIR/rerank_expanded_scope/candidate_scores.csv:score" \
    --output-dir "$RESULTS_DIR/reaction_class_curated_${WEIGHT_TAG}_expanded" \
    --min-groups 20 \
    --weak-top1 0.80 \
    --weak-mrr 0.85 \
    > "$LOG_DIR/type1_curated_${WEIGHT_TAG}_expanded_class.log" 2>&1
fi

if [[ ! -f "$RESULTS_DIR/reaction_class_curated_${WEIGHT_TAG}_original/reaction_class_benchmark.json" ]]; then
  echo "[class] original Regio/HiTEA scope"
  PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.analyze_reaction_class_benchmark \
    --score-csv "curated_${WEIGHT_TAG}_original=$OUT_DIR/rerank_original_scope/candidate_scores.csv:score" \
    --output-dir "$RESULTS_DIR/reaction_class_curated_${WEIGHT_TAG}_original" \
    --min-groups 20 \
    --weak-top1 0.80 \
    --weak-mrr 0.85 \
    > "$LOG_DIR/type1_curated_${WEIGHT_TAG}_original_class.log" 2>&1
fi

echo "Curated class-weighted selection complete: $OUT_DIR"
