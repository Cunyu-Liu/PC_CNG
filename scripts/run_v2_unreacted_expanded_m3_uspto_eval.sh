#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/cunyuliu/pc_cng_research}
PYTHON_BIN=${PYTHON_BIN:-/home/cunyuliu/miniconda3/envs/pc_cng_gpu/bin/python}
DEVICE=${DEVICE:-cpu}
BATCH_SIZE=${BATCH_SIZE:-4096}

CODE_DIR="$ROOT/chem_negative_sampling"
LOG_DIR="$ROOT/results/logs"
RESULTS_DIR=${RESULTS_DIR:-$ROOT/results/type1_v2_unreacted_expanded_m3_uspto_eval_20260712}

REGIO_REAL="$ROOT/data/processed/regiosqm20_normalized.csv"
HITEA_REAL="$ROOT/data/processed/hitea_full_normalized.csv"
USPTO_EXPANDED_REAL="$ROOT/results/original_test_expansion_uspto_negatives_20260712/uspto_original_test_expansion_positive_parents.csv"

SYNTHETIC_CSVS=(
  "$ROOT/results/type1_diverse_anchor_full/diverse_anchor_candidates_reviewed_knownpos_filtered.csv"
  "$ROOT/results/type1_class_quota_supplement_20260711/class_quota_candidates_reviewed.csv"
  "$ROOT/results/type1_class_fallback_supplement_20260711/class_fallback_candidates_reviewed.csv"
  "$ROOT/results/type1_partial_product_supplement_20260711/partial_product_candidates_reviewed.csv"
  "$ROOT/results/type1_unreacted_substrate_supplement_v2_20260711/unreacted_substrate_candidates_reviewed.csv"
  "$ROOT/results/original_test_expansion_uspto_negatives_20260712/uspto_test_expansion_candidates_reviewed_knownpos_filtered.csv"
)

SEEDS=(20260710 20260711 20260712 20260713 20260714 20260715 20260716 20260717 20260718 20260719)
MODEL_ROOT="$ROOT/results/type1_unreacted_substrate_supplement_v2_20260711"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"
cd "$CODE_DIR"

COMMON_ARGS=(
  --real-csv "$REGIO_REAL"
  --real-csv "$HITEA_REAL"
  --real-csv "$USPTO_EXPANDED_REAL"
)
for synthetic_csv in "${SYNTHETIC_CSVS[@]}"; do
  COMMON_ARGS+=(--synthetic-csv "$synthetic_csv")
done

for seed in "${SEEDS[@]}"; do
  model_dir="$MODEL_ROOT/unreacted_augmented_pairwise_seed${seed}"
  out_dir="$RESULTS_DIR/v2_unreacted_seed${seed}/rerank_same_split"
  log_file="$LOG_DIR/type1_v2_unreacted_expanded_m3_uspto_eval_seed${seed}.log"
  if [[ -f "$out_dir/ranking_metrics.json" ]]; then
    echo "[skip] expanded rerank seed=$seed already exists"
    continue
  fi
  if [[ ! -f "$model_dir/best_pairwise_reward_mlp.pt" && ! -f "$model_dir/best_feasibility_mlp.pt" ]]; then
    echo "[error] missing checkpoint for seed=$seed in $model_dir" >&2
    exit 1
  fi
  mkdir -p "$out_dir"
  echo "[rerank] expanded M3 USPTO benchmark seed=$seed device=$DEVICE"
  PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.evaluate_candidate_reranking \
    "${COMMON_ARGS[@]}" \
    --model-dir "$model_dir" \
    --output-dir "$out_dir" \
    --candidate-scope same_split \
    --batch-size "$BATCH_SIZE" \
    --device "$DEVICE" \
    > "$log_file" 2>&1
done

SUMMARY_DIR="$RESULTS_DIR/v2_unreacted_expanded_m3_uspto_summary"
mkdir -p "$SUMMARY_DIR"
PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.multiseed_summary \
  --exp-dir "$RESULTS_DIR" \
  --prefix "v2_unreacted_seed" \
  --seeds "$(IFS=,; echo "${SEEDS[*]}")" \
  --output "$SUMMARY_DIR/summary.json" \
  > "$SUMMARY_DIR/summary.log" 2>&1

RESULTS_DIR="$RESULTS_DIR" PYTHONPATH=. "$PYTHON_BIN" - <<'PY'
import csv
import json
import os
from pathlib import Path

result_dir = Path(os.environ["RESULTS_DIR"])
seeds = [20260710, 20260711, 20260712, 20260713, 20260714, 20260715, 20260716, 20260717, 20260718, 20260719]
rows = []
for seed in seeds:
    path = result_dir / f"v2_unreacted_seed{seed}/rerank_same_split/ranking_metrics.json"
    if not path.exists():
        rows.append({"seed": seed, "status": "missing"})
        continue
    data = json.loads(path.read_text())
    overall = data["overall"]
    test = data["by_split"].get("test", {})
    uspto = data["by_dataset"].get("uspto_openmolecules_yield25to150", {})
    rows.append(
        {
            "seed": seed,
            "status": "complete",
            "overall_groups": overall.get("groups", 0),
            "overall_top1": overall.get("top1", 0.0),
            "overall_mrr": overall.get("mrr", 0.0),
            "overall_ndcg": overall.get("ndcg", 0.0),
            "test_groups": test.get("groups", 0),
            "test_top1": test.get("top1", 0.0),
            "test_mrr": test.get("mrr", 0.0),
            "test_ndcg": test.get("ndcg", 0.0),
            "uspto_groups": uspto.get("groups", 0),
            "uspto_top1": uspto.get("top1", 0.0),
            "uspto_mrr": uspto.get("mrr", 0.0),
            "uspto_ndcg": uspto.get("ndcg", 0.0),
        }
    )
out_csv = result_dir / "v2_unreacted_expanded_m3_uspto_per_seed.csv"
with out_csv.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps({"per_seed_csv": str(out_csv), "completed": sum(row["status"] == "complete" for row in rows)}, indent=2))
PY

echo "Expanded M3 USPTO v2/unreacted evaluation complete: $RESULTS_DIR"
