#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/cunyuliu/pc_cng_research}
PYTHON_BIN=${PYTHON_BIN:-/home/cunyuliu/miniconda3/envs/pc_cng_gpu/bin/python}
GPU_TRAIN=${GPU_TRAIN:-4}
GPU_EVAL=${GPU_EVAL:-$GPU_TRAIN}

CODE_DIR="$ROOT/chem_negative_sampling"
RESULTS_DIR=${RESULTS_DIR:-$ROOT/results/type1_class_quota_supplement_20260711}
LOG_DIR=${LOG_DIR:-$ROOT/results/logs}

REGIO_ALIGNMENT=${REGIO_ALIGNMENT:-$ROOT/data/processed/regiosqm20_normalized.csv}
HITEA_ALIGNMENT=${HITEA_ALIGNMENT:-$ROOT/data/processed/hitea_full_normalized.csv}
BASE_REVIEWED=${BASE_REVIEWED:-$ROOT/results/type1_diverse_anchor_full/diverse_anchor_candidates_reviewed.csv}
SUPP_REVIEWED=${SUPP_REVIEWED:-$RESULTS_DIR/class_quota_candidates_reviewed.csv}
OUT_DIR=${OUT_DIR:-$RESULTS_DIR/augmented_pairwise_seed20260710}
SEED=${SEED:-20260710}

mkdir -p "$RESULTS_DIR" "$LOG_DIR"
cd "$CODE_DIR"

if [[ ! -f "$SUPP_REVIEWED" ]]; then
  echo "[error] missing supplemental reviewed CSV: $SUPP_REVIEWED" >&2
  exit 2
fi

if [[ ! -f "$OUT_DIR/metrics.json" ]]; then
  echo "[train] augmented pairwise seed=$SEED"
  rm -rf "$OUT_DIR"
  CUDA_VISIBLE_DEVICES="$GPU_TRAIN" PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.train_pairwise_reward_mlp \
    --real-csv "$REGIO_ALIGNMENT" \
    --real-csv "$HITEA_ALIGNMENT" \
    --synthetic-csv "$BASE_REVIEWED" \
    --synthetic-csv "$SUPP_REVIEWED" \
    --output-dir "$OUT_DIR" \
    --epochs "${EPOCHS:-80}" \
    --batch-size "${BATCH_SIZE:-4096}" \
    --hidden-dim "${HIDDEN_DIM:-2048}" \
    --n-bits "${N_BITS:-4096}" \
    --dropout "${DROPOUT:-0.20}" \
    --seed "$SEED" \
    > "$LOG_DIR/type1_class_quota_augmented_pairwise_train.log" 2>&1
fi

if [[ ! -f "$OUT_DIR/rerank_same_split/ranking_metrics.json" ]]; then
  echo "[rerank] augmented pairwise seed=$SEED"
  CUDA_VISIBLE_DEVICES="$GPU_EVAL" PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.evaluate_candidate_reranking \
    --real-csv "$REGIO_ALIGNMENT" \
    --real-csv "$HITEA_ALIGNMENT" \
    --synthetic-csv "$BASE_REVIEWED" \
    --synthetic-csv "$SUPP_REVIEWED" \
    --model-dir "$OUT_DIR" \
    --output-dir "$OUT_DIR/rerank_same_split" \
    --candidate-scope same_split \
    --batch-size "${BATCH_SIZE:-4096}" \
    --device cuda \
    > "$LOG_DIR/type1_class_quota_augmented_pairwise_rerank.log" 2>&1
fi

if [[ ! -f "$RESULTS_DIR/reaction_class_augmented_trained/reaction_class_benchmark.json" ]]; then
  echo "[class] augmented trained reaction-class benchmark"
  PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.analyze_reaction_class_benchmark \
    --score-csv "augmented_trained=$OUT_DIR/rerank_same_split/candidate_scores.csv:score" \
    --output-dir "$RESULTS_DIR/reaction_class_augmented_trained" \
    --min-groups 20 \
    --weak-top1 0.80 \
    --weak-mrr 0.85 \
    > "$LOG_DIR/type1_class_quota_augmented_pairwise_class.log" 2>&1
fi

echo "Class-quota augmented pairwise experiment complete: $OUT_DIR"
