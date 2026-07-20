#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/cunyuliu/pc_cng_research}
PYTHON_BIN=${PYTHON_BIN:-/home/cunyuliu/miniconda3/envs/pc_cng_gpu/bin/python}
CODE_DIR="$ROOT/chem_negative_sampling"
RESULTS_DIR=${RESULTS_DIR:-$ROOT/results/type1_curated_weak_class_contexts_20260711}
LOG_DIR=${LOG_DIR:-$ROOT/results/logs}

WEAK_CLASS_WEIGHT=${WEAK_CLASS_WEIGHT:-0.5}
WEIGHT_TAG=${WEIGHT_TAG:-classw050_rc}
SEEDS=${SEEDS:-"20260710 20260711 20260712 20260713 20260714"}
SUMMARY_DIR="$RESULTS_DIR/${WEIGHT_TAG}_multiseed_summary"

mkdir -p "$SUMMARY_DIR" "$LOG_DIR"
cd "$CODE_DIR"

for seed in $SEEDS; do
  echo "[seed] $seed weak_class_weight=$WEAK_CLASS_WEIGHT"
  env \
    ROOT="$ROOT" \
    PYTHON_BIN="$PYTHON_BIN" \
    GPU_TRAIN="${GPU_TRAIN:-4}" \
    GPU_EVAL="${GPU_EVAL:-${GPU_TRAIN:-4}}" \
    RESULTS_DIR="$RESULTS_DIR" \
    LOG_DIR="$LOG_DIR" \
    WEAK_CLASS_WEIGHT="$WEAK_CLASS_WEIGHT" \
    WEIGHT_TAG="$WEIGHT_TAG" \
    SEED="$seed" \
    ./scripts_run_type1_curated_class_weight_selection.sh
done

PYTHONPATH=. "$PYTHON_BIN" - <<'PY'
import csv
import json
import math
import os
from pathlib import Path
from statistics import mean, pstdev

root = Path(os.environ.get("RESULTS_DIR", "/home/cunyuliu/pc_cng_research/results/type1_curated_weak_class_contexts_20260711"))
summary_dir = root / f"{os.environ.get('WEIGHT_TAG', 'classw050_rc')}_multiseed_summary"
seeds = os.environ.get("SEEDS", "20260710 20260711 20260712 20260713 20260714").split()
weight_tag = os.environ.get("WEIGHT_TAG", "classw050_rc")

def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)

def metric(payload: dict, split: str, name: str) -> float:
    if split == "overall":
        return float(dict(payload.get("overall", {})).get(name, float("nan")))
    return float(dict(dict(payload.get("by_split", {})).get(split, {})).get(name, float("nan")))

def dataset_metric(payload: dict, dataset: str, name: str) -> float:
    return float(dict(dict(payload.get("by_dataset", {})).get(dataset, {})).get(name, float("nan")))

rows = []
for seed in seeds:
    out_dir = root / f"curated_augmented_pairwise_{weight_tag}_seed{seed}"
    for scope, rel in [
        ("original_regio_hitea", "rerank_original_scope/ranking_metrics.json"),
        ("expanded_curated", "rerank_expanded_scope/ranking_metrics.json"),
    ]:
        payload = load_json(out_dir / rel)
        if not payload:
            continue
        rows.append(
            {
                "seed": seed,
                "scope": scope,
                "groups": int(dict(payload.get("overall", {})).get("groups", 0) or 0),
                "overall_top1": metric(payload, "overall", "top1"),
                "test_top1": metric(payload, "test", "top1"),
                "val_top1": metric(payload, "val", "top1"),
                "overall_mrr": metric(payload, "overall", "mrr"),
                "overall_ndcg": metric(payload, "overall", "ndcg"),
                "regiosqm20_top1": dataset_metric(payload, "regiosqm20", "top1"),
                "hitea_top1": dataset_metric(payload, "hitea_full", "top1"),
                "curated_uspto_top1": dataset_metric(payload, "curated_uspto_openmolecules_rule", "top1"),
            }
        )

fields = [
    "seed",
    "scope",
    "groups",
    "overall_top1",
    "test_top1",
    "val_top1",
    "overall_mrr",
    "overall_ndcg",
    "regiosqm20_top1",
    "hitea_top1",
    "curated_uspto_top1",
]
summary_dir.mkdir(parents=True, exist_ok=True)
with (summary_dir / "per_seed_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

summary_rows = []
for scope in sorted({row["scope"] for row in rows}):
    subset = [row for row in rows if row["scope"] == scope]
    item = {"scope": scope, "n_seeds": len(subset)}
    for name in fields[3:]:
        values = [float(row[name]) for row in subset if not math.isnan(float(row[name]))]
        item[f"{name}_mean"] = mean(values) if values else float("nan")
        item[f"{name}_std"] = pstdev(values) if len(values) > 1 else 0.0
    summary_rows.append(item)

with (summary_dir / "summary.json").open("w", encoding="utf-8") as handle:
    json.dump({"seeds": seeds, "rows": rows, "summary": summary_rows}, handle, indent=2, ensure_ascii=False)

summary_fields = ["scope", "n_seeds"] + [f"{name}_{stat}" for name in fields[3:] for stat in ("mean", "std")]
with (summary_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=summary_fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(summary_rows)

lines = ["| " + " | ".join(summary_fields) + " |", "| " + " | ".join(["---"] * len(summary_fields)) + " |"]
for row in summary_rows:
    rendered = []
    for field in summary_fields:
        value = row.get(field, "")
        if isinstance(value, float):
            value = f"{value * 100.0:.2f}" if field.endswith(("_mean", "_std")) else f"{value:.4f}"
        rendered.append(str(value))
    lines.append("| " + " | ".join(rendered) + " |")
(summary_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps({"summary_dir": str(summary_dir), "n_rows": len(rows), "n_summary_rows": len(summary_rows)}, indent=2))
PY

echo "Curated class-weighted multiseed complete: $SUMMARY_DIR"
