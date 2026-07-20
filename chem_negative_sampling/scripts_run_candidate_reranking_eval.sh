#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/cunyuliu/pc_cng_research}
PYTHON_BIN=${PYTHON_BIN:-/home/cunyuliu/miniconda3/envs/pc_cng_gpu/bin/python}
GPU_EVAL=${GPU_EVAL:-0}

CODE_DIR="$ROOT/chem_negative_sampling"
RESULTS_DIR="$ROOT/results/candidate_reranking_eval"
LOG_DIR="$ROOT/results/logs"

HIT_ALIGNMENT="$ROOT/data/processed/hitea_full_normalized.csv"
REGIO_ALIGNMENT="$ROOT/data/processed/regiosqm20_normalized.csv"
EXPANDED_REVIEWED="$ROOT/results/expanded_hard_negative_actions_full/expanded_hard_negatives_reviewed.csv"

REAL_ONLY="$ROOT/results/full_feasibility_mlp_real_only_h2048_n4096_e80"
V2_DIRECT="$ROOT/results/v2_direct_bce_h2048_n4096_e80"
RULE_HARD="$ROOT/results/rule_hard_negatives_direct_bce_h2048_n4096_e80"
EXP_PAIRWISE="$ROOT/results/expanded_actions_pairwise_reward_h2048_n4096_e80"
EXP_BCE_SYNTH05="$ROOT/results/expanded_actions_direct_bce_synth05_h2048_n4096_e80"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"
cd "$CODE_DIR"

run_eval() {
  local name="$1"
  shift
  local out_dir="$RESULTS_DIR/$name"
  if [ -f "$out_dir/ranking_metrics.json" ]; then
    echo "[skip] $name"
    return
  fi
  mkdir -p "$out_dir"
  echo "[eval] $name"
  CUDA_VISIBLE_DEVICES="$GPU_EVAL" PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.evaluate_candidate_reranking \
    --real-csv "$REGIO_ALIGNMENT" \
    --real-csv "$HIT_ALIGNMENT" \
    --synthetic-csv "$EXPANDED_REVIEWED" \
    --output-dir "$out_dir" \
    --candidate-scope same_split \
    "$@" \
    > "$LOG_DIR/candidate_reranking_${name}.log" 2>&1
}

run_eval real_only --model-dir "$REAL_ONLY"
run_eval v2_direct_bce --model-dir "$V2_DIRECT"
run_eval rule_hard_direct_bce --model-dir "$RULE_HARD"
run_eval expanded_pairwise --model-dir "$EXP_PAIRWISE"
run_eval expanded_bce_synth05 --model-dir "$EXP_BCE_SYNTH05"
run_eval mean_core_pc_cng \
  --model-dir "$REAL_ONLY" \
  --model-dir "$V2_DIRECT" \
  --model-dir "$RULE_HARD" \
  --model-dir "$EXP_PAIRWISE" \
  --model-dir "$EXP_BCE_SYNTH05"

SUMMARY="$RESULTS_DIR/summary.json"
export RESULTS_DIR SUMMARY
PYTHONPATH=. "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["RESULTS_DIR"])
records = {}
for path in sorted(root.glob("*/ranking_metrics.json")):
    metrics = json.load(open(path))
    records[path.parent.name] = {
        "overall": metrics.get("overall"),
        "by_dataset": metrics.get("by_dataset"),
        "by_split": metrics.get("by_split"),
        "by_candidate_source": metrics.get("by_candidate_source"),
    }

summary = {"runs": records}
out = Path(os.environ["SUMMARY"])
json.dump(summary, open(out, "w"), indent=2, ensure_ascii=False)
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY

echo "Candidate reranking evaluation complete."
