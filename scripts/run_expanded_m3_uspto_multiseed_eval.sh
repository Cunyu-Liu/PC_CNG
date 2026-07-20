#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/cunyuliu/pc_cng_research}
PYTHON_BIN=${PYTHON_BIN:-/home/cunyuliu/miniconda3/envs/pc_cng_gpu/bin/python}
DEVICE=${DEVICE:-cpu}
BATCH_SIZE=${BATCH_SIZE:-4096}

EXP_NAME=${EXP_NAME:?Set EXP_NAME, e.g. type1_v2_hidden4096_expanded_m3_uspto_eval_20260712}
MODEL_ROOT=${MODEL_ROOT:?Set MODEL_ROOT containing per-seed model dirs}
MODEL_PREFIX=${MODEL_PREFIX:?Set MODEL_PREFIX, e.g. v2_hidden4096_pairwise_seed}
OUTPUT_PREFIX=${OUTPUT_PREFIX:-$MODEL_PREFIX}
SUMMARY_NAME=${SUMMARY_NAME:-expanded_m3_uspto_summary}

CODE_DIR="$ROOT/chem_negative_sampling"
LOG_DIR="$ROOT/results/logs"
RESULTS_DIR=${RESULTS_DIR:-$ROOT/results/$EXP_NAME}

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
  model_dir="$MODEL_ROOT/${MODEL_PREFIX}${seed}"
  out_dir="$RESULTS_DIR/${OUTPUT_PREFIX}${seed}/rerank_same_split"
  log_file="$LOG_DIR/${EXP_NAME}_seed${seed}.log"
  if [[ -f "$out_dir/ranking_metrics.json" ]]; then
    echo "[skip] expanded rerank seed=$seed already exists"
    continue
  fi
  if [[ ! -f "$model_dir/best_pairwise_reward_mlp.pt" && ! -f "$model_dir/best_feasibility_mlp.pt" ]]; then
    echo "[error] missing checkpoint for seed=$seed in $model_dir" >&2
    exit 1
  fi
  mkdir -p "$out_dir"
  echo "[rerank] $EXP_NAME seed=$seed device=$DEVICE"
  PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.evaluate_candidate_reranking \
    "${COMMON_ARGS[@]}" \
    --model-dir "$model_dir" \
    --output-dir "$out_dir" \
    --candidate-scope same_split \
    --batch-size "$BATCH_SIZE" \
    --device "$DEVICE" \
    > "$log_file" 2>&1
done

SUMMARY_DIR="$RESULTS_DIR/$SUMMARY_NAME"
mkdir -p "$SUMMARY_DIR"
PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.multiseed_summary \
  --exp-dir "$RESULTS_DIR" \
  --prefix "$OUTPUT_PREFIX" \
  --seeds "$(IFS=,; echo "${SEEDS[*]}")" \
  --output "$SUMMARY_DIR/summary.json" \
  > "$SUMMARY_DIR/summary.log" 2>&1

RESULTS_DIR="$RESULTS_DIR" OUTPUT_PREFIX="$OUTPUT_PREFIX" SUMMARY_NAME="$SUMMARY_NAME" PYTHONPATH=. "$PYTHON_BIN" - <<'PY'
import csv
import json
import os
from pathlib import Path
from collections import Counter, defaultdict

from pc_cng.evaluate_candidate_reranking import grouped_metrics, ranking_metrics
from pc_cng.multiseed_paired_significance import build_ensemble_scores, write_scores_csv


result_dir = Path(os.environ["RESULTS_DIR"])
output_prefix = os.environ["OUTPUT_PREFIX"]
summary_name = os.environ["SUMMARY_NAME"]
seeds = [20260710, 20260711, 20260712, 20260713, 20260714, 20260715, 20260716, 20260717, 20260718, 20260719]
rows = []
score_csvs = []
for seed in seeds:
    metrics_path = result_dir / f"{output_prefix}{seed}/rerank_same_split/ranking_metrics.json"
    score_path = result_dir / f"{output_prefix}{seed}/rerank_same_split/candidate_scores.csv"
    if not metrics_path.exists():
        rows.append({"seed": seed, "status": "missing"})
        continue
    score_csvs.append(str(score_path))
    data = json.loads(metrics_path.read_text())
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
per_seed_csv = result_dir / f"{summary_name}_per_seed.csv"
with per_seed_csv.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

ensemble_dir = result_dir / f"{summary_name}_ensemble"
ensemble_dir.mkdir(parents=True, exist_ok=True)
ensemble = build_ensemble_scores(score_csvs)
for row in ensemble:
    row["label"] = int(float(row.get("label", 0) or 0))
    row["score"] = float(row.get("score", 0.0) or 0.0)
score_out = ensemble_dir / "candidate_scores.csv"
write_scores_csv(str(score_out), ensemble)
metrics = {
    "config": {
        "score_csvs": score_csvs,
        "seeds": seeds,
        "method": "mean_score_ensemble_from_existing_seed_candidate_scores",
    },
    "overall": ranking_metrics(ensemble),
    "by_split": grouped_metrics(ensemble, "split"),
    "by_dataset": grouped_metrics(ensemble, "dataset"),
    "by_candidate_source": grouped_metrics(ensemble, "candidate_source"),
    "by_candidate_family": grouped_metrics(ensemble, "candidate_family"),
    "predictions": str(score_out),
}
(ensemble_dir / "ranking_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

support_csv = Path("/home/cunyuliu/pc_cng_research/results/original_benchmark_support_audit_m3_uspto_20260712/combined_reactants_plus_synthetic_groups.csv")
support_counts = Counter()
with support_csv.open(newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    for row in reader:
        if str(row.get("evaluable", "")).lower() != "true":
            continue
        support_counts[(row.get("dataset", ""), row.get("split", ""))] += 1

groups = defaultdict(list)
with score_out.open(newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    for row in reader:
        if row.get("group_id"):
            groups[row["group_id"]].append(row)

scored_any = Counter()
scored_evaluable = Counter()
scored_non_evaluable = Counter()
for gid, group_rows in groups.items():
    parts = gid.split("|")
    if len(parts) < 4:
        continue
    _, dataset, split, _ = parts[:4]
    scored_any[(dataset, split)] += 1
    labels = [int(float(row.get("label", 0) or 0)) for row in group_rows]
    if any(labels) and not all(labels):
        scored_evaluable[(dataset, split)] += 1
    else:
        scored_non_evaluable[(dataset, split)] += 1

coverage_rows = []
for key in sorted(set(support_counts) | set(scored_any) | set(scored_evaluable)):
    support = support_counts.get(key, 0)
    eval_count = scored_evaluable.get(key, 0)
    coverage_rows.append(
        {
            "dataset": key[0],
            "split": key[1],
            "support_evaluable_groups": support,
            "scored_any_groups": scored_any.get(key, 0),
            "scored_evaluable_groups": eval_count,
            "scored_non_evaluable_groups": scored_non_evaluable.get(key, 0),
            "evaluable_coverage": 0.0 if support == 0 else eval_count / support,
            "unscored_or_non_evaluable_groups": max(0, support - eval_count),
        }
    )
coverage = {
    "support_audit_csv": str(support_csv),
    "ensemble_scores_csv": str(score_out),
    "support_total_groups": sum(support_counts.values()),
    "scored_any_total_groups": sum(scored_any.values()),
    "scored_evaluable_total_groups": sum(scored_evaluable.values()),
    "support_test_groups": sum(v for (_, split), v in support_counts.items() if split == "test"),
    "scored_any_test_groups": sum(v for (_, split), v in scored_any.items() if split == "test"),
    "scored_evaluable_test_groups": sum(v for (_, split), v in scored_evaluable.items() if split == "test"),
    "rows": coverage_rows,
    "note": "Support audit counts chemically evaluable groups before model-scored reranking. Reranking metrics count only scored groups with at least one positive and one negative after featurization/scoring.",
}
(ensemble_dir / "scoring_coverage_summary.json").write_text(json.dumps(coverage, indent=2, ensure_ascii=False), encoding="utf-8")

manifest = {
    "result_dir": str(result_dir),
    "summary_json": str(result_dir / summary_name / "summary.json"),
    "per_seed_csv": str(per_seed_csv),
    "ensemble_metrics_json": str(ensemble_dir / "ranking_metrics.json"),
    "coverage_json": str(ensemble_dir / "scoring_coverage_summary.json"),
    "completed": sum(row["status"] == "complete" for row in rows),
}
(result_dir / "expanded_m3_uspto_eval_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(manifest, indent=2, ensure_ascii=False))
PY

echo "Expanded M3 USPTO multiseed evaluation complete: $RESULTS_DIR"
