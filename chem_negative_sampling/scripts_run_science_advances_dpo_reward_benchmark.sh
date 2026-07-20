#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/cunyuliu/pc_cng_research}
PYTHON_BIN=${PYTHON_BIN:-/home/cunyuliu/miniconda3/envs/pc_cng_gpu/bin/python}
GPU_TRAIN=${GPU_TRAIN:-1}

CODE_DIR="$ROOT/chem_negative_sampling"
CHEMFORMER_RESULTS_DIR=${CHEMFORMER_RESULTS_DIR:-$ROOT/results/science_advances_chemformer_lm_benchmark_full}
RESULTS_DIR=${RESULTS_DIR:-$ROOT/results/science_advances_dpo_reward_benchmark}
LOG_DIR=${LOG_DIR:-$ROOT/results/logs}

SEEDS=${SEEDS:-20260710 20260711 20260712 20260713 20260714 20260715 20260716 20260717 20260718 20260719}
SETTINGS=${SETTINGS:-k_low k_high}
CONFIGS=${CONFIGS:-dpo_pairwise_synth pairwise_only_synth dpo_only_synth}

EPOCHS=${EPOCHS:-80}
BATCH_SIZE=${BATCH_SIZE:-4096}
LR=${LR:-0.001}
HIDDEN_DIM=${HIDDEN_DIM:-2048}
N_BITS=${N_BITS:-4096}
DROPOUT=${DROPOUT:-0.20}
DPO_BETA=${DPO_BETA:-0.2}

mkdir -p "$RESULTS_DIR" "$LOG_DIR"
cd "$CODE_DIR"

run_config() {
  local setting="$1"
  local seed="$2"
  local name="$3"
  local bce_weight="$4"
  local pairwise_weight="$5"
  local dpo_weight="$6"
  local dpo_beta="$7"

  local run_name="${setting}_seed${seed}"
  local input_csv="$CHEMFORMER_RESULTS_DIR/$run_name/pc_cng_candidates/lm_scores_chemformer_log_likelihood.csv"
  local out_dir="$RESULTS_DIR/$run_name/$name"
  local log_path="$LOG_DIR/scadv_dpo_reward_${setting}_${seed}_${name}.log"

  if [[ ! -f "$input_csv" ]]; then
    echo "[skip] missing Chemformer-scored candidates: $input_csv"
    return 0
  fi

  if [[ -f "$out_dir/metrics.json" && -f "$out_dir/ranking_metrics.json" ]]; then
    echo "[skip] completed $run_name config=$name"
    return 0
  fi

  echo "[train] $run_name config=$name"
  mkdir -p "$out_dir"
  CUDA_VISIBLE_DEVICES="$GPU_TRAIN" PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.train_dpo_reward_mlp \
    --scored-csv "$input_csv" \
    --output-dir "$out_dir" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --hidden-dim "$HIDDEN_DIM" \
    --n-bits "$N_BITS" \
    --dropout "$DROPOUT" \
    --pair-source synthetic \
    --bce-weight "$bce_weight" \
    --pairwise-weight "$pairwise_weight" \
    --dpo-weight "$dpo_weight" \
    --dpo-beta "$dpo_beta" \
    --reference-scale standardize \
    --seed "$seed" \
    > "$log_path" 2>&1
}

for setting in $SETTINGS; do
  for seed in $SEEDS; do
    for config in $CONFIGS; do
      case "$config" in
        dpo_pairwise_synth)
          run_config "$setting" "$seed" "$config" 1.0 1.0 1.0 "$DPO_BETA"
          ;;
        pairwise_only_synth)
          run_config "$setting" "$seed" "$config" 1.0 1.0 0.0 "$DPO_BETA"
          ;;
        dpo_only_synth)
          run_config "$setting" "$seed" "$config" 1.0 0.0 1.0 "$DPO_BETA"
          ;;
        *)
          echo "[error] unknown config: $config" >&2
          exit 2
          ;;
      esac
    done
  done
done

SUMMARY_JSON="$RESULTS_DIR/summary.json"
export CHEMFORMER_RESULTS_DIR RESULTS_DIR SUMMARY_JSON SETTINGS SEEDS CONFIGS
PYTHONPATH=. "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path


def load(path: Path):
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


metrics = ["top1", "top3", "mrr", "ndcg"]
chemformer_root = Path(os.environ["CHEMFORMER_RESULTS_DIR"])
results_root = Path(os.environ["RESULTS_DIR"])
records = []
for setting in os.environ["SETTINGS"].split():
    for seed in os.environ["SEEDS"].split():
        run_name = f"{setting}_seed{seed}"
        baseline_path = chemformer_root / run_name / "pc_cng_candidates" / "lm_scores_chemformer_log_likelihood_metrics.json"
        baseline = load(baseline_path)
        if not baseline:
            continue
        baseline_overall = baseline.get("overall", {})
        for config in os.environ["CONFIGS"].split():
            ranking_path = results_root / run_name / config / "ranking_metrics.json"
            ranking = load(ranking_path)
            if not ranking:
                continue
            reward_overall = ranking.get("overall", {})
            record = {
                "run": f"{run_name}_{config}",
                "setting": f"{setting}_{config}",
                "seed": int(seed),
                "chemformer_pc_cng_candidates": baseline_overall,
                "reward_model": reward_overall,
                "delta": {
                    metric: float(reward_overall.get(metric, 0.0)) - float(baseline_overall.get(metric, 0.0))
                    for metric in metrics
                },
                "paths": {
                    "baseline_metrics": str(baseline_path),
                    "reward_metrics": str(ranking_path),
                },
            }
            records.append(record)

summary = {
    "baseline": "chemformer_log_likelihood_on_pc_cng_candidates",
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
  --family chemformer_pc_cng_candidates \
  --family reward_model

echo "Science Advances-style DPO reward benchmark complete."
