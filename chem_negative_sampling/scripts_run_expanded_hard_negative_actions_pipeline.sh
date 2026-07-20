#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/cunyuliu/pc_cng_research}
PYTHON_BIN=${PYTHON_BIN:-/home/cunyuliu/miniconda3/envs/pc_cng_gpu/bin/python}
GPU_DOWNSTREAM=${GPU_DOWNSTREAM:-1}

CODE_DIR="$ROOT/chem_negative_sampling"
RESULTS_DIR="$ROOT/results/expanded_hard_negative_actions_full"
LOG_DIR="$ROOT/results/logs"

HIT_ALIGNMENT="$ROOT/data/processed/hitea_full_normalized.csv"
REGIO_ALIGNMENT="$ROOT/data/processed/regiosqm20_normalized.csv"

ACTION_NEG="$RESULTS_DIR/expanded_hard_negatives.csv"
ACTION_SUMMARY="$RESULTS_DIR/expanded_hard_negatives_summary.json"
ACTION_REVIEWED="$RESULTS_DIR/expanded_hard_negatives_reviewed.csv"
ACTION_REVIEW_SUMMARY="$RESULTS_DIR/expanded_hard_negatives_review_summary.json"
PIPELINE_SUMMARY="$RESULTS_DIR/pipeline_summary.json"

PAIRWISE_DIR="$ROOT/results/expanded_actions_pairwise_reward_h2048_n4096_e80"
BCE_DIR="$ROOT/results/expanded_actions_direct_bce_h2048_n4096_e80"

MAX_CANDIDATES_PER_REACTION=${MAX_CANDIDATES_PER_REACTION:-6}
MAX_CANDIDATES_PER_PAIR=${MAX_CANDIDATES_PER_PAIR:-12}
MAX_ANCHOR_DISTANCE=${MAX_ANCHOR_DISTANCE:-6}
MAX_TAUTOMERS=${MAX_TAUTOMERS:-4}

mkdir -p "$RESULTS_DIR" "$LOG_DIR"
cd "$CODE_DIR"

LIMIT_ARGS=()
if [ -n "${ACTION_LIMIT:-}" ]; then
  LIMIT_ARGS=(--limit "$ACTION_LIMIT")
fi

if [ ! -f "$ACTION_SUMMARY" ]; then
  echo "[1/5] Generate expanded hard negatives"
  PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.run_hard_negative_actions \
    --input "$HIT_ALIGNMENT" \
    --input "$REGIO_ALIGNMENT" \
    --low-yield-input "$HIT_ALIGNMENT" \
    --low-yield-input "$REGIO_ALIGNMENT" \
    --known-positive "$HIT_ALIGNMENT" \
    --known-positive "$REGIO_ALIGNMENT" \
    --output "$ACTION_NEG" \
    --summary "$ACTION_SUMMARY" \
    --action heteroatom \
    --action regio \
    --action tautomer \
    --action low_yield_seed \
    --max-candidates-per-reaction "$MAX_CANDIDATES_PER_REACTION" \
    --max-candidates-per-pair "$MAX_CANDIDATES_PER_PAIR" \
    --max-anchor-distance "$MAX_ANCHOR_DISTANCE" \
    --max-tautomers "$MAX_TAUTOMERS" \
    --map-unmapped \
    "${LIMIT_ARGS[@]}" \
    > "$LOG_DIR/expanded_actions_generate.log" 2>&1
fi

if [ ! -f "$ACTION_REVIEW_SUMMARY" ]; then
  echo "[2/5] Review expanded hard negatives"
  PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.false_negative_review \
    --input "$ACTION_NEG" \
    --output "$ACTION_REVIEWED" \
    --summary "$ACTION_REVIEW_SUMMARY" \
    --known-positive "$HIT_ALIGNMENT" \
    --known-positive "$REGIO_ALIGNMENT" \
    > "$LOG_DIR/expanded_actions_review.log" 2>&1
fi

if [ ! -f "$PAIRWISE_DIR/metrics.json" ]; then
  echo "[3/5] Train downstream pairwise reward model"
  mkdir -p "$PAIRWISE_DIR"
  CUDA_VISIBLE_DEVICES="$GPU_DOWNSTREAM" PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.train_pairwise_reward_mlp \
    --real-csv "$REGIO_ALIGNMENT" \
    --real-csv "$HIT_ALIGNMENT" \
    --synthetic-csv "$ACTION_REVIEWED" \
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
    > "$LOG_DIR/expanded_actions_pairwise_reward.log" 2>&1
fi

if [ ! -f "$BCE_DIR/metrics.json" ]; then
  echo "[4/5] Train downstream BCE model"
  mkdir -p "$BCE_DIR"
  CUDA_VISIBLE_DEVICES="$GPU_DOWNSTREAM" PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.train_feasibility_mlp \
    --real-csv "$REGIO_ALIGNMENT" \
    --real-csv "$HIT_ALIGNMENT" \
    --synthetic-csv "$ACTION_REVIEWED" \
    --output-dir "$BCE_DIR" \
    --epochs 80 \
    --batch-size 4096 \
    --lr 0.001 \
    --hidden-dim 2048 \
    --n-bits 4096 \
    --dropout 0.20 \
    --seed 20260710 \
    > "$LOG_DIR/expanded_actions_direct_bce.log" 2>&1
fi

if [ -f "$ROOT/evaluate_stacked_ensemble.py" ]; then
  echo "[5/5] Update stacked ensemble"
  PYTHONPATH=. "$PYTHON_BIN" "$ROOT/evaluate_stacked_ensemble.py" \
    > "$LOG_DIR/expanded_actions_stacked_ensemble.log" 2>&1 || true
fi

export ACTION_SUMMARY ACTION_REVIEW_SUMMARY PAIRWISE_DIR BCE_DIR ROOT PIPELINE_SUMMARY
PYTHONPATH=. "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path


def load(path):
    path = Path(path)
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


pairwise_metrics = load(Path(os.environ["PAIRWISE_DIR"]) / "metrics.json")
bce_metrics = load(Path(os.environ["BCE_DIR"]) / "metrics.json")
ensemble_summary = load(Path(os.environ["ROOT"]) / "results" / "stacked_ensemble_summary.json")

summary = {
    "actions": load(os.environ["ACTION_SUMMARY"]),
    "review": load(os.environ["ACTION_REVIEW_SUMMARY"]),
    "pairwise_reward": {
        "dir": os.environ["PAIRWISE_DIR"],
        "counts": (pairwise_metrics or {}).get("counts"),
        "val": (pairwise_metrics or {}).get("val"),
        "test": (pairwise_metrics or {}).get("test"),
    },
    "direct_bce": {
        "dir": os.environ["BCE_DIR"],
        "counts": (bce_metrics or {}).get("counts"),
        "val": (bce_metrics or {}).get("val"),
        "test": (bce_metrics or {}).get("test"),
    },
    "stacked_ensemble": {
        "best_by_val_auc": (ensemble_summary or {}).get("best_by_val_auc"),
        "best_by_val_auprc": (ensemble_summary or {}).get("best_by_val_auprc"),
        "used_runs": (ensemble_summary or {}).get("used_runs"),
    },
}

out = Path(os.environ["PIPELINE_SUMMARY"])
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2, ensure_ascii=False)
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY

echo "Expanded hard-negative actions pipeline complete."
