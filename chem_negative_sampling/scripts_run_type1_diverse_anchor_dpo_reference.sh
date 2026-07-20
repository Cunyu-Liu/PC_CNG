#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/cunyuliu/pc_cng_research}
PYTHON_BIN=${PYTHON_BIN:-/home/cunyuliu/miniconda3/envs/pc_cng_gpu/bin/python}
REACTION_LM_PYTHON=${REACTION_LM_PYTHON:-$ROOT/envs/reaction_lm/bin/python}
GPU_EVAL=${GPU_EVAL:-1}
GPU_TRAIN=${GPU_TRAIN:-0}

CODE_DIR="$ROOT/chem_negative_sampling"
LOG_DIR="$ROOT/results/logs"
RESULTS_DIR=${RESULTS_DIR:-$ROOT/results/type1_diverse_anchor_dpo_reference}

REGIO_ALIGNMENT="$ROOT/data/processed/regiosqm20_normalized.csv"
HIT_ALIGNMENT="$ROOT/data/processed/hitea_full_normalized.csv"
SYNTHETIC_CSV=${SYNTHETIC_CSV:-$ROOT/results/type1_diverse_anchor_full/diverse_anchor_candidates_reviewed.csv}

SCORER=${SCORER:-chemformer_log_likelihood}
MODEL_NAME=${MODEL_NAME:-chemformer_forward_uspto50k}
MODEL_PATH=${MODEL_PATH:-$ROOT/models/reaction_lm/chemformer_forward_uspto50k/model_sanitized.ckpt}
VOCABULARY_PATH=${VOCABULARY_PATH:-$ROOT/external/reaction_lm/Chemformer/bart_vocab.json}
LM_BATCH_SIZE=${LM_BATCH_SIZE:-128}

SEEDS=${SEEDS:-20260710 20260711 20260712 20260713 20260714}
CONFIGS=${CONFIGS:-dpo_pairwise_synth pairwise_only_synth dpo_only_synth}
EPOCHS=${EPOCHS:-80}
BATCH_SIZE=${BATCH_SIZE:-4096}
HIDDEN_DIM=${HIDDEN_DIM:-2048}
N_BITS=${N_BITS:-4096}
DPO_BETA=${DPO_BETA:-0.2}

LM_DIR="$RESULTS_DIR/chemformer_scores"
CANDIDATES_CSV="$LM_DIR/lm_candidates.csv"
CANDIDATES_SUMMARY="$LM_DIR/lm_candidates_summary.json"
SCORES_CSV="$LM_DIR/lm_scores_${SCORER}.csv"
SCORES_SUMMARY="$LM_DIR/lm_scores_${SCORER}_summary.json"
LM_METRICS="$LM_DIR/lm_scores_${SCORER}_metrics.json"

mkdir -p "$RESULTS_DIR" "$LM_DIR" "$LOG_DIR"
cd "$CODE_DIR"

if [[ ! -f "$CANDIDATES_SUMMARY" ]]; then
  echo "[1/5] Build diverse-anchor LM candidate set"
  PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.build_reaction_lm_candidate_set \
    --real-csv "$REGIO_ALIGNMENT" \
    --real-csv "$HIT_ALIGNMENT" \
    --synthetic-csv "$SYNTHETIC_CSV" \
    --candidate-scope same_split \
    --output "$CANDIDATES_CSV" \
    --summary "$CANDIDATES_SUMMARY" \
    > "$LOG_DIR/type1_diverse_anchor_dpo_build_candidates.log" 2>&1
fi

if [[ ! -f "$SCORES_SUMMARY" ]]; then
  echo "[2/5] Score candidates with Chemformer"
  CUDA_VISIBLE_DEVICES="$GPU_EVAL" PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.reaction_lm_scorer \
    --input "$CANDIDATES_CSV" \
    --output "$SCORES_CSV" \
    --summary "$SCORES_SUMMARY" \
    --scorer "$SCORER" \
    --model-name "$MODEL_NAME" \
    --model-path "$MODEL_PATH" \
    --vocabulary-path "$VOCABULARY_PATH" \
    --root "$ROOT" \
    --reaction-lm-python "$REACTION_LM_PYTHON" \
    --batch-size "$LM_BATCH_SIZE" \
    --device cuda \
    > "$LOG_DIR/type1_diverse_anchor_dpo_chemformer_score.log" 2>&1
fi

if [[ ! -f "$LM_METRICS" ]]; then
  echo "[3/5] Evaluate frozen Chemformer ranking"
  PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.evaluate_reaction_lm_scores \
    --input "$SCORES_CSV" \
    --output "$LM_METRICS" \
    > "$LOG_DIR/type1_diverse_anchor_dpo_chemformer_metrics.log" 2>&1
fi

run_config() {
  local name="$1"
  local bce_weight="$2"
  local pairwise_weight="$3"
  local dpo_weight="$4"
  local dpo_beta="$5"
  local seed="$6"
  local out_dir="$RESULTS_DIR/${name}_seed${seed}"

  if [[ -f "$out_dir/metrics.json" && -f "$out_dir/ranking_metrics.json" ]]; then
    echo "[skip] completed config=$name seed=$seed"
    return 0
  fi

  echo "[4/5] Train DPO-reference reward config=$name seed=$seed"
  mkdir -p "$out_dir"
  CUDA_VISIBLE_DEVICES="$GPU_TRAIN" PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.train_dpo_reward_mlp \
    --scored-csv "$SCORES_CSV" \
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
    > "$LOG_DIR/type1_diverse_anchor_dpo_${name}_${seed}.log" 2>&1
}

for seed in $SEEDS; do
  for config in $CONFIGS; do
    case "$config" in
      dpo_pairwise_synth)
        run_config "$config" 1.0 1.0 1.0 "$DPO_BETA" "$seed"
        ;;
      pairwise_only_synth)
        run_config "$config" 1.0 1.0 0.0 "$DPO_BETA" "$seed"
        ;;
      dpo_only_synth)
        run_config "$config" 1.0 0.0 1.0 "$DPO_BETA" "$seed"
        ;;
      *)
        echo "[error] unknown config: $config" >&2
        exit 2
        ;;
    esac
  done
done

echo "[5/5] Summarize DPO-reference runs"
SUMMARY_JSON="$RESULTS_DIR/summary.json"
export RESULTS_DIR SUMMARY_JSON LM_METRICS
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
baseline = load(os.environ["LM_METRICS"])
records = []
for run_dir in sorted(root.glob("*_seed*")):
    name, seed_part = run_dir.name.rsplit("_seed", 1)
    ranking = load(run_dir / "ranking_metrics.json")
    metrics = load(run_dir / "metrics.json")
    if not ranking or not metrics:
        continue
    reward = ranking.get("overall", {})
    base = (baseline or {}).get("overall", {})
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
            "counts": metrics.get("counts", {}),
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

summary = {
    "baseline": {
        "name": "chemformer_log_likelihood",
        "metrics": (baseline or {}).get("overall"),
        "path": os.environ["LM_METRICS"],
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
  --family chemformer_reference \
  --family reward_model \
  --family synthetic_ranking \
  --family test_ranking \
  --family delta

echo "Type-1 diverse-anchor DPO-reference benchmark complete: $RESULTS_DIR"
