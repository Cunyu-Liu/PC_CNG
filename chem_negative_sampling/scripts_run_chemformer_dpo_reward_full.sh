#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/cunyuliu/pc_cng_research}
PYTHON_BIN=${PYTHON_BIN:-/home/cunyuliu/miniconda3/envs/pc_cng_gpu/bin/python}
GPU_TRAIN=${GPU_TRAIN:-1}

CODE_DIR="$ROOT/chem_negative_sampling"
LOG_DIR="$ROOT/results/logs"
SCORED_CSV=${SCORED_CSV:-$ROOT/results/reaction_lm_scorer_smoke/lm_scores_chemformer_log_likelihood.csv}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/results/chemformer_dpo_reward_synthetic_h2048_n4096_e80}

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
cd "$CODE_DIR"

if [[ ! -f "$OUTPUT_DIR/metrics.json" ]]; then
  echo "[train] Chemformer-reference DPO reward MLP"
  CUDA_VISIBLE_DEVICES="$GPU_TRAIN" PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.train_dpo_reward_mlp \
    --scored-csv "$SCORED_CSV" \
    --output-dir "$OUTPUT_DIR" \
    --epochs "${EPOCHS:-80}" \
    --batch-size "${BATCH_SIZE:-4096}" \
    --lr "${LR:-0.001}" \
    --hidden-dim "${HIDDEN_DIM:-2048}" \
    --n-bits "${N_BITS:-4096}" \
    --dropout "${DROPOUT:-0.20}" \
    --pair-source "${PAIR_SOURCE:-synthetic}" \
    --bce-weight "${BCE_WEIGHT:-1.0}" \
    --pairwise-weight "${PAIRWISE_WEIGHT:-1.0}" \
    --dpo-weight "${DPO_WEIGHT:-1.0}" \
    --dpo-beta "${DPO_BETA:-0.2}" \
    --reference-scale "${REFERENCE_SCALE:-standardize}" \
    --seed "${SEED:-20260710}" \
    > "$LOG_DIR/chemformer_dpo_reward_full.log" 2>&1
fi

echo "Chemformer-reference DPO reward full run complete: $OUTPUT_DIR"
if [[ -f "$OUTPUT_DIR/metrics.json" ]]; then
  "$PYTHON_BIN" - <<PY
import json
path = "$OUTPUT_DIR/metrics.json"
data = json.load(open(path))
print(json.dumps({
    "counts": data.get("counts"),
    "val": data.get("val"),
    "test": data.get("test"),
    "ranking_overall": data.get("ranking", {}).get("overall"),
    "ranking_by_split": data.get("ranking", {}).get("by_split"),
}, indent=2))
PY
fi
