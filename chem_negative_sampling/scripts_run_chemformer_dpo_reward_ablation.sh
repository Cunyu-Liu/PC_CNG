#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/cunyuliu/pc_cng_research}
PYTHON_BIN=${PYTHON_BIN:-/home/cunyuliu/miniconda3/envs/pc_cng_gpu/bin/python}
GPU_TRAIN=${GPU_TRAIN:-1}

CODE_DIR="$ROOT/chem_negative_sampling"
LOG_DIR="$ROOT/results/logs"
SCORED_CSV=${SCORED_CSV:-$ROOT/results/reaction_lm_scorer_smoke/lm_scores_chemformer_log_likelihood.csv}
RESULTS_DIR=${RESULTS_DIR:-$ROOT/results/chemformer_dpo_reward_ablation}
SEEDS=${SEEDS:-20260710 20260711 20260712 20260713 20260714}
EPOCHS=${EPOCHS:-80}
BATCH_SIZE=${BATCH_SIZE:-4096}
HIDDEN_DIM=${HIDDEN_DIM:-2048}
N_BITS=${N_BITS:-4096}

mkdir -p "$RESULTS_DIR" "$LOG_DIR"
cd "$CODE_DIR"

run_config() {
  local name="$1"
  local bce_weight="$2"
  local pairwise_weight="$3"
  local dpo_weight="$4"
  local dpo_beta="$5"
  local seed="$6"
  local out_dir="$RESULTS_DIR/${name}_seed${seed}"

  if [[ ! -f "$out_dir/metrics.json" ]]; then
    echo "[train] config=$name seed=$seed"
    mkdir -p "$out_dir"
    CUDA_VISIBLE_DEVICES="$GPU_TRAIN" PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.train_dpo_reward_mlp \
      --scored-csv "$SCORED_CSV" \
      --output-dir "$out_dir" \
      --epochs "$EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --lr "${LR:-0.001}" \
      --hidden-dim "$HIDDEN_DIM" \
      --n-bits "$N_BITS" \
      --dropout "${DROPOUT:-0.20}" \
      --pair-source synthetic \
      --bce-weight "$bce_weight" \
      --pairwise-weight "$pairwise_weight" \
      --dpo-weight "$dpo_weight" \
      --dpo-beta "$dpo_beta" \
      --reference-scale standardize \
      --seed "$seed" \
      > "$LOG_DIR/chemformer_dpo_ablation_${name}_${seed}.log" 2>&1
  fi
}

for seed in $SEEDS; do
  run_config dpo_pairwise_synth 1.0 1.0 1.0 0.2 "$seed"
  run_config pairwise_only_synth 1.0 1.0 0.0 0.2 "$seed"
  run_config dpo_only_synth 1.0 0.0 1.0 0.2 "$seed"
done

SUMMARY_JSON="$RESULTS_DIR/summary.json"
export RESULTS_DIR SUMMARY_JSON SCORED_CSV
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


root = Path(os.environ["RESULTS_DIR"])
baseline = load(Path(os.environ["SCORED_CSV"]).with_name("lm_scores_chemformer_log_likelihood_metrics.json"))
records = []
for run_dir in sorted(root.glob("*_seed*")):
    name, seed_part = run_dir.name.rsplit("_seed", 1)
    ranking = load(run_dir / "ranking_metrics.json")
    metrics = load(run_dir / "metrics.json")
    if not ranking or not metrics:
        continue
    record = {
        "run": run_dir.name,
        "setting": name,
        "seed": int(seed_part),
        "ranking": ranking["overall"],
        "test_ranking": ranking.get("by_split", {}).get("test", {}),
        "synthetic_ranking": ranking.get("by_candidate_source", {}).get("synthetic", {}),
        "binary_test": metrics.get("test", {}),
        "counts": metrics.get("counts", {}),
        "paths": {
            "metrics": str(run_dir / "metrics.json"),
            "ranking": str(run_dir / "ranking_metrics.json"),
        },
    }
    if baseline:
        base_overall = baseline.get("overall", {})
        base_test = baseline.get("by_split", {}).get("test", {})
        record["delta_vs_chemformer_ll"] = {
            metric: ranking["overall"].get(metric, 0.0) - base_overall.get(metric, 0.0)
            for metric in ["top1", "top3", "mrr", "ndcg"]
        }
        record["test_delta_vs_chemformer_ll"] = {
            metric: ranking.get("by_split", {}).get("test", {}).get(metric, 0.0) - base_test.get(metric, 0.0)
            for metric in ["top1", "top3", "mrr", "ndcg"]
        }
    records.append(record)

summary = {
    "baseline": {
        "name": "chemformer_log_likelihood",
        "overall": (baseline or {}).get("overall"),
        "test": (baseline or {}).get("by_split", {}).get("test") if baseline else None,
    },
    "records": records,
}
with open(os.environ["SUMMARY_JSON"], "w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2, ensure_ascii=False)
print(json.dumps({"records": len(records), "summary": os.environ["SUMMARY_JSON"]}, indent=2))
PY

SUMMARY_DIR="$RESULTS_DIR/paper_summary"
PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.summarize_reaction_lm_benchmark \
  --input "$SUMMARY_JSON" \
  --output-dir "$SUMMARY_DIR" \
  --family ranking \
  --family test_ranking \
  --family synthetic_ranking \
  --family delta_vs_chemformer_ll \
  --family test_delta_vs_chemformer_ll

echo "Chemformer DPO reward ablation complete: $RESULTS_DIR"
