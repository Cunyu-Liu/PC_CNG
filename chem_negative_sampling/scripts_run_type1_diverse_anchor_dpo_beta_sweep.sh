#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/cunyuliu/pc_cng_research}
PYTHON_BIN=${PYTHON_BIN:-/home/cunyuliu/miniconda3/envs/pc_cng_gpu/bin/python}
GPU_TRAIN=${GPU_TRAIN:-0}

CODE_DIR="$ROOT/chem_negative_sampling"
LOG_DIR="$ROOT/results/logs"
RESULTS_DIR=${RESULTS_DIR:-$ROOT/results/type1_diverse_anchor_dpo_beta_sweep}
SCORED_CSV=${SCORED_CSV:-$ROOT/results/type1_diverse_anchor_dpo_reference/chemformer_scores/lm_scores_chemformer_log_likelihood.csv}

SEEDS=${SEEDS:-20260710 20260711 20260712}
CONFIGS=${CONFIGS:-dpo_beta005 dpo_beta010 dpo_beta050 dpo_beta020_refnone}
EPOCHS=${EPOCHS:-80}
BATCH_SIZE=${BATCH_SIZE:-4096}
HIDDEN_DIM=${HIDDEN_DIM:-2048}
N_BITS=${N_BITS:-4096}

mkdir -p "$RESULTS_DIR" "$LOG_DIR"
cd "$CODE_DIR"

run_config() {
  local name="$1"
  local dpo_beta="$2"
  local reference_scale="$3"
  local pairwise_weight="$4"
  local seed="$5"
  local out_dir="$RESULTS_DIR/${name}_seed${seed}"

  if [[ -f "$out_dir/metrics.json" && -f "$out_dir/ranking_metrics.json" ]]; then
    echo "[skip] completed config=$name seed=$seed"
    return 0
  fi

  echo "[train] beta sweep config=$name seed=$seed beta=$dpo_beta reference_scale=$reference_scale pairwise_weight=$pairwise_weight"
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
    --bce-weight 1.0 \
    --pairwise-weight "$pairwise_weight" \
    --dpo-weight 1.0 \
    --dpo-beta "$dpo_beta" \
    --reference-scale "$reference_scale" \
    --seed "$seed" \
    > "$LOG_DIR/type1_diverse_anchor_dpo_beta_sweep_${name}_${seed}.log" 2>&1
}

for seed in $SEEDS; do
  for config in $CONFIGS; do
    case "$config" in
      dpo_beta005)
        run_config "$config" 0.05 standardize 0.0 "$seed"
        ;;
      dpo_beta010)
        run_config "$config" 0.10 standardize 0.0 "$seed"
        ;;
      dpo_beta050)
        run_config "$config" 0.50 standardize 0.0 "$seed"
        ;;
      dpo_beta020_refnone)
        run_config "$config" 0.20 none 0.0 "$seed"
        ;;
      dpo_pairwise_beta010)
        run_config "$config" 0.10 standardize 1.0 "$seed"
        ;;
      *)
        echo "[error] unknown config: $config" >&2
        exit 2
        ;;
    esac
  done
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
baseline_path = Path(os.environ["SCORED_CSV"]).with_name("lm_scores_chemformer_log_likelihood_metrics.json")
baseline = load(baseline_path)
base = (baseline or {}).get("overall", {})
records = []
for run_dir in sorted(root.glob("*_seed*")):
    name, seed_part = run_dir.name.rsplit("_seed", 1)
    ranking = load(run_dir / "ranking_metrics.json")
    metrics = load(run_dir / "metrics.json")
    if not ranking or not metrics:
        continue
    reward = ranking.get("overall", {})
    records.append(
        {
            "run": run_dir.name,
            "setting": name,
            "seed": int(seed_part),
            "chemformer_reference": base,
            "reward_model": reward,
            "synthetic_ranking": ranking.get("by_candidate_source", {}).get("synthetic", {}),
            "test_ranking": ranking.get("by_split", {}).get("test", {}),
            "binary_test": metrics.get("test", {}),
            "delta": {
                metric: float(reward.get(metric, 0.0)) - float(base.get(metric, 0.0))
                for metric in ["top1", "top3", "mrr", "ndcg"]
            },
            "paths": {
                "metrics": str(run_dir / "metrics.json"),
                "ranking": str(run_dir / "ranking_metrics.json"),
            },
        }
    )

with open(os.environ["SUMMARY_JSON"], "w", encoding="utf-8") as handle:
    json.dump({"baseline": str(baseline_path), "records": records}, handle, indent=2, ensure_ascii=False)
print(json.dumps({"records": len(records), "summary": os.environ["SUMMARY_JSON"]}, indent=2))
PY

SUMMARY_DIR="$RESULTS_DIR/paper_summary"
PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.summarize_reaction_lm_benchmark \
  --input "$SUMMARY_JSON" \
  --output-dir "$SUMMARY_DIR" \
  --family chemformer_reference \
  --family reward_model \
  --family synthetic_ranking \
  --family test_ranking \
  --family delta

echo "Type-1 diverse-anchor DPO beta sweep complete: $RESULTS_DIR"
