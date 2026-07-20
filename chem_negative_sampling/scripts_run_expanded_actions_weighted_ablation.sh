#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/cunyuliu/pc_cng_research}
PYTHON_BIN=${PYTHON_BIN:-/home/cunyuliu/miniconda3/envs/pc_cng_gpu/bin/python}
GPU_DOWNSTREAM=${GPU_DOWNSTREAM:-0}

CODE_DIR="$ROOT/chem_negative_sampling"
LOG_DIR="$ROOT/results/logs"

HIT_ALIGNMENT="$ROOT/data/processed/hitea_full_normalized.csv"
REGIO_ALIGNMENT="$ROOT/data/processed/regiosqm20_normalized.csv"
ACTION_REVIEWED="$ROOT/results/expanded_hard_negative_actions_full/expanded_hard_negatives_reviewed.csv"

BCE_SYNTH02_DIR="$ROOT/results/expanded_actions_direct_bce_synth02_h2048_n4096_e80"
BCE_SYNTH05_DIR="$ROOT/results/expanded_actions_direct_bce_synth05_h2048_n4096_e80"
PAIRWISE_FAMILY_DIR="$ROOT/results/expanded_actions_pairwise_family_margin_h2048_n4096_e80"
SUMMARY="$ROOT/results/expanded_hard_negative_actions_full/weighted_ablation_summary.json"

mkdir -p "$LOG_DIR"
cd "$CODE_DIR"

if [ ! -f "$BCE_SYNTH02_DIR/metrics.json" ]; then
  echo "[1/4] Train expanded direct BCE with synthetic weight 0.2"
  mkdir -p "$BCE_SYNTH02_DIR"
  CUDA_VISIBLE_DEVICES="$GPU_DOWNSTREAM" PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.train_feasibility_mlp \
    --real-csv "$REGIO_ALIGNMENT" \
    --real-csv "$HIT_ALIGNMENT" \
    --synthetic-csv "$ACTION_REVIEWED" \
    --output-dir "$BCE_SYNTH02_DIR" \
    --epochs 80 \
    --batch-size 4096 \
    --lr 0.001 \
    --hidden-dim 2048 \
    --n-bits 4096 \
    --dropout 0.20 \
    --origin-weight synthetic=0.2 \
    --seed 20260710 \
    > "$LOG_DIR/expanded_actions_direct_bce_synth02.log" 2>&1
fi

if [ ! -f "$BCE_SYNTH05_DIR/metrics.json" ]; then
  echo "[2/4] Train expanded direct BCE with synthetic weight 0.5"
  mkdir -p "$BCE_SYNTH05_DIR"
  CUDA_VISIBLE_DEVICES="$GPU_DOWNSTREAM" PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.train_feasibility_mlp \
    --real-csv "$REGIO_ALIGNMENT" \
    --real-csv "$HIT_ALIGNMENT" \
    --synthetic-csv "$ACTION_REVIEWED" \
    --output-dir "$BCE_SYNTH05_DIR" \
    --epochs 80 \
    --batch-size 4096 \
    --lr 0.001 \
    --hidden-dim 2048 \
    --n-bits 4096 \
    --dropout 0.20 \
    --origin-weight synthetic=0.5 \
    --seed 20260710 \
    > "$LOG_DIR/expanded_actions_direct_bce_synth05.log" 2>&1
fi

if [ ! -f "$PAIRWISE_FAMILY_DIR/metrics.json" ]; then
  echo "[3/4] Train expanded action-family aware pairwise reward"
  mkdir -p "$PAIRWISE_FAMILY_DIR"
  CUDA_VISIBLE_DEVICES="$GPU_DOWNSTREAM" PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.train_pairwise_reward_mlp \
    --real-csv "$REGIO_ALIGNMENT" \
    --real-csv "$HIT_ALIGNMENT" \
    --synthetic-csv "$ACTION_REVIEWED" \
    --output-dir "$PAIRWISE_FAMILY_DIR" \
    --epochs 80 \
    --batch-size 4096 \
    --lr 0.001 \
    --hidden-dim 2048 \
    --n-bits 4096 \
    --dropout 0.20 \
    --pairwise-weight 1.0 \
    --bce-weight 1.0 \
    --margin 0.05 \
    --family-margin regio=0.30 \
    --family-margin tautomer=0.10 \
    --family-margin low_yield_seed=0.00 \
    --family-weight regio=2.0 \
    --family-weight tautomer=1.0 \
    --family-weight low_yield_seed=0.2 \
    --seed 20260710 \
    > "$LOG_DIR/expanded_actions_pairwise_family_margin.log" 2>&1
fi

if [ -f "$ROOT/evaluate_stacked_ensemble.py" ]; then
  echo "[4/4] Update stacked ensemble with weighted ablations"
  PYTHONPATH=. "$PYTHON_BIN" "$ROOT/evaluate_stacked_ensemble.py" \
    > "$LOG_DIR/expanded_actions_weighted_ablation_ensemble.log" 2>&1 || true
fi

export BCE_SYNTH02_DIR BCE_SYNTH05_DIR PAIRWISE_FAMILY_DIR ROOT SUMMARY
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


def brief(path):
    metrics = load(Path(path) / "metrics.json")
    if not metrics:
        return None
    return {
        "dir": path,
        "counts": metrics.get("counts"),
        "val": metrics.get("val"),
        "test": metrics.get("test"),
        "pair_family_margins": metrics.get("pair_family_margins"),
        "pair_family_weights": metrics.get("pair_family_weights"),
    }


ensemble = load(Path(os.environ["ROOT"]) / "results" / "stacked_ensemble_summary.json")
summary = {
    "expanded_direct_bce_synth02": brief(os.environ["BCE_SYNTH02_DIR"]),
    "expanded_direct_bce_synth05": brief(os.environ["BCE_SYNTH05_DIR"]),
    "expanded_pairwise_family_margin": brief(os.environ["PAIRWISE_FAMILY_DIR"]),
    "stacked_ensemble": {
        "best_by_val_auc": (ensemble or {}).get("best_by_val_auc"),
        "best_by_val_auprc": (ensemble or {}).get("best_by_val_auprc"),
        "used_runs": (ensemble or {}).get("used_runs"),
    },
}
out = Path(os.environ["SUMMARY"])
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2, ensure_ascii=False)
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY

echo "Expanded actions weighted ablation complete."
