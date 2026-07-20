#!/usr/bin/env bash
set -euo pipefail

# Safe full-pipeline entry for a normalized positive reaction CSV.
# It writes outputs to RESULTS_DIR and never deletes existing data.

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
INPUT_POSITIVES="${INPUT_POSITIVES:?Set INPUT_POSITIVES to a CSV with reaction_smiles/source_id}"
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/results/full_pipeline_$(date +%Y%m%d_%H%M%S)}"
LIMIT="${LIMIT:-}"
EPOCHS="${EPOCHS:-300}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "$RESULTS_DIR"

LIMIT_ARGS=()
if [[ -n "$LIMIT" ]]; then
  LIMIT_ARGS=(--limit "$LIMIT")
fi

echo "[pipeline] Input: $INPUT_POSITIVES"
echo "[pipeline] Results: $RESULTS_DIR"

"$PYTHON_BIN" -m pc_cng.run_scale_generation \
  --input "$INPUT_POSITIVES" \
  --output "$RESULTS_DIR/pc_cng_synthetic_negatives.csv" \
  --summary "$RESULTS_DIR/pc_cng_generation_summary.json" \
  "${LIMIT_ARGS[@]}"

"$PYTHON_BIN" -m pc_cng.baselines \
  --input "$INPUT_POSITIVES" \
  --output "$RESULTS_DIR/baseline_negatives.csv" \
  --summary "$RESULTS_DIR/baseline_summary.json" \
  "${LIMIT_ARGS[@]}"

"$PYTHON_BIN" -m pc_cng.false_negative_review \
  --input "$RESULTS_DIR/pc_cng_synthetic_negatives.csv" \
  --output "$RESULTS_DIR/pc_cng_synthetic_negatives_reviewed.csv" \
  --summary "$RESULTS_DIR/false_negative_review_summary.json" \
  --known-positive "$INPUT_POSITIVES"

"$PYTHON_BIN" -m pc_cng.run_experiment_matrix \
  --input "$INPUT_POSITIVES" \
  --output-dir "$RESULTS_DIR/experiment_matrix" \
  --epochs "$EPOCHS" \
  "${LIMIT_ARGS[@]}"

if "$PYTHON_BIN" - <<'PY'
try:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    raise SystemExit(0)
except Exception:
    raise SystemExit(1)
PY
then
  "$PYTHON_BIN" -m pc_cng.train_graph_edit_decoder \
    --input "$RESULTS_DIR/pc_cng_synthetic_negatives_reviewed.csv" \
    --output-dir "$RESULTS_DIR/graph_edit_decoder" \
    --epochs 50
else
  echo "[pipeline] PyTorch not available; skipping graph edit decoder training"
fi

echo "[pipeline] Done: $RESULTS_DIR"
