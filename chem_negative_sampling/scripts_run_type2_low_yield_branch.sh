#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/cunyuliu/pc_cng_research}
PYTHON_BIN=${PYTHON_BIN:-/home/cunyuliu/miniconda3/envs/pc_cng_gpu/bin/python}
GPU_TRAIN=${GPU_TRAIN:-0}

CODE_DIR="$ROOT/chem_negative_sampling"
LOG_DIR="$ROOT/results/logs"
RESULTS_DIR=${RESULTS_DIR:-$ROOT/results/type2_low_yield_branch}

REGIO_ALIGNMENT="$ROOT/data/processed/regiosqm20_normalized.csv"
HIT_ALIGNMENT="$ROOT/data/processed/hitea_full_normalized.csv"
SYNTHETIC_CSV=${SYNTHETIC_CSV:-$ROOT/results/expanded_hard_negative_actions_full/expanded_hard_negatives_reviewed.csv}

SEEDS=${SEEDS:-20260710 20260711 20260712 20260713 20260714}
CONFIGS=${CONFIGS:-low_yield_synth02 low_yield_synth05 low_yield_synth10}
EPOCHS=${EPOCHS:-80}
BATCH_SIZE=${BATCH_SIZE:-4096}
HIDDEN_DIM=${HIDDEN_DIM:-2048}
N_BITS=${N_BITS:-4096}

mkdir -p "$RESULTS_DIR" "$LOG_DIR"
cd "$CODE_DIR"

run_config() {
  local name="$1"
  local synthetic_weight="$2"
  local seed="$3"
  local out_dir="$RESULTS_DIR/${name}_seed${seed}"

  if [[ -f "$out_dir/metrics.json" ]]; then
    echo "[skip] completed config=$name seed=$seed"
    return 0
  fi

  echo "[train] type2 low-yield config=$name seed=$seed synthetic_weight=$synthetic_weight"
  mkdir -p "$out_dir"
  CUDA_VISIBLE_DEVICES="$GPU_TRAIN" PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.train_feasibility_mlp \
    --real-csv "$REGIO_ALIGNMENT" \
    --real-csv "$HIT_ALIGNMENT" \
    --synthetic-csv "$SYNTHETIC_CSV" \
    --synthetic-family low_yield_seed \
    --output-dir "$out_dir" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "${LR:-0.001}" \
    --hidden-dim "$HIDDEN_DIM" \
    --n-bits "$N_BITS" \
    --dropout "${DROPOUT:-0.20}" \
    --origin-weight "synthetic=${synthetic_weight}" \
    --seed "$seed" \
    > "$LOG_DIR/type2_low_yield_${name}_${seed}.log" 2>&1
}

for seed in $SEEDS; do
  for config in $CONFIGS; do
    case "$config" in
      low_yield_synth02)
        run_config "$config" 0.2 "$seed"
        ;;
      low_yield_synth05)
        run_config "$config" 0.5 "$seed"
        ;;
      low_yield_synth10)
        run_config "$config" 1.0 "$seed"
        ;;
      *)
        echo "[error] unknown config: $config" >&2
        exit 2
        ;;
    esac
  done
done

SUMMARY_JSON="$RESULTS_DIR/summary.json"
PAPER_DIR="$RESULTS_DIR/paper_summary"
export RESULTS_DIR SUMMARY_JSON PAPER_DIR SEEDS CONFIGS
PYTHONPATH=. "$PYTHON_BIN" - <<'PY'
import csv
import json
import math
import os
import random
from pathlib import Path
from statistics import mean, pstdev


METRICS = ["roc_auc", "auprc", "f1", "accuracy"]


def load(path: Path):
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def percentile(values, q):
    values = sorted(values)
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return values[low]
    w = pos - low
    return values[low] * (1.0 - w) + values[high] * w


def bootstrap_ci(values, seed=20260710, n_iter=10000):
    values = list(values)
    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)
    means = []
    for _ in range(n_iter):
        sample = [rng.choice(values) for _ in values]
        means.append(mean(sample))
    return percentile(means, 0.025), percentile(means, 0.975)


def summarize(values):
    values = [float(v) for v in values if v == v]
    low, high = bootstrap_ci(values) if values else (0.0, 0.0)
    return {
        "mean": mean(values) if values else 0.0,
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "ci95_low": low,
        "ci95_high": high,
        "n": len(values),
    }


def pct(stats):
    return f"{float(stats['mean']) * 100.0:.2f} +/- {float(stats['std']) * 100.0:.2f}"


root = Path(os.environ["RESULTS_DIR"])
records = []
for config in os.environ["CONFIGS"].split():
    for seed in os.environ["SEEDS"].split():
        run_dir = root / f"{config}_seed{seed}"
        metrics = load(run_dir / "metrics.json")
        if not metrics:
            continue
        records.append(
            {
                "run": run_dir.name,
                "setting": config,
                "seed": int(seed),
                "test": metrics.get("test", {}),
                "val": metrics.get("val", {}),
                "test_hitea": metrics.get("test_by_dataset", {}).get("hitea_full", {}),
                "test_regiosqm20": metrics.get("test_by_dataset", {}).get("regiosqm20", {}),
                "counts": metrics.get("counts", {}),
                "paths": {"metrics": str(run_dir / "metrics.json")},
            }
        )

summary = {}
for setting in sorted({record["setting"] for record in records}):
    subset = [record for record in records if record["setting"] == setting]
    raw = {"n": len(subset)}
    for family in ["test", "val", "test_hitea", "test_regiosqm20"]:
        raw[family] = {metric: summarize([record.get(family, {}).get(metric, float("nan")) for record in subset]) for metric in METRICS}
    summary[setting] = raw

payload = {"records": records, "summary": summary}
Path(os.environ["SUMMARY_JSON"]).parent.mkdir(parents=True, exist_ok=True)
with open(os.environ["SUMMARY_JSON"], "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, ensure_ascii=False)

rows = []
for setting, raw in sorted(summary.items()):
    for metric in METRICS:
        rows.append(
            {
                "setting": setting,
                "metric": metric,
                "n": str(raw["n"]),
                "test": pct(raw["test"][metric]),
                "val": pct(raw["val"][metric]),
                "test_hitea": pct(raw["test_hitea"][metric]),
                "test_regiosqm20": pct(raw["test_regiosqm20"][metric]),
            }
        )

paper_dir = Path(os.environ["PAPER_DIR"])
paper_dir.mkdir(parents=True, exist_ok=True)
fields = ["setting", "metric", "n", "test", "val", "test_hitea", "test_regiosqm20"]
with open(paper_dir / "paper_table.csv", "w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
with open(paper_dir / "paper_table.md", "w", encoding="utf-8") as handle:
    handle.write("| " + " | ".join(fields) + " |\n")
    handle.write("| " + " | ".join(["---"] * len(fields)) + " |\n")
    for row in rows:
        handle.write("| " + " | ".join(row[field] for field in fields) + " |\n")

print(json.dumps({"records": len(records), "summary": os.environ["SUMMARY_JSON"]}, indent=2))
PY

echo "Type-2 low-yield branch complete: $RESULTS_DIR"
