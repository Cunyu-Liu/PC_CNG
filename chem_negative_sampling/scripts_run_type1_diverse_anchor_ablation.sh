#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/cunyuliu/pc_cng_research}
PYTHON_BIN=${PYTHON_BIN:-/home/cunyuliu/miniconda3/envs/pc_cng_gpu/bin/python}
GPU_TRAIN=${GPU_TRAIN:-0}

CODE_DIR="$ROOT/chem_negative_sampling"
LOG_DIR="$ROOT/results/logs"
RESULTS_DIR=${RESULTS_DIR:-$ROOT/results/type1_diverse_anchor_ablation}

REGIO_ALIGNMENT="$ROOT/data/processed/regiosqm20_normalized.csv"
HIT_ALIGNMENT="$ROOT/data/processed/hitea_full_normalized.csv"
SYNTHETIC_CSV=${SYNTHETIC_CSV:-$ROOT/results/type1_diverse_anchor_full/diverse_anchor_candidates_reviewed.csv}

SEEDS=${SEEDS:-20260710 20260711 20260712 20260713 20260714}
CONFIGS=${CONFIGS:-pairwise_default family_margin}
EPOCHS=${EPOCHS:-80}
BATCH_SIZE=${BATCH_SIZE:-4096}
HIDDEN_DIM=${HIDDEN_DIM:-2048}
N_BITS=${N_BITS:-4096}

mkdir -p "$RESULTS_DIR" "$LOG_DIR"
cd "$CODE_DIR"

run_config() {
  local name="$1"
  local seed="$2"
  shift 2
  local extra_args=("$@")
  local out_dir="$RESULTS_DIR/${name}_seed${seed}"
  local same_dir="$out_dir/rerank_same_split"
  local all_dir="$out_dir/rerank_all_group"
  local family_dir="$out_dir/action_family_contribution"

  if [[ ! -f "$out_dir/metrics.json" ]]; then
    echo "[train] config=$name seed=$seed"
    mkdir -p "$out_dir"
    CUDA_VISIBLE_DEVICES="$GPU_TRAIN" PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.train_pairwise_reward_mlp \
      --real-csv "$REGIO_ALIGNMENT" \
      --real-csv "$HIT_ALIGNMENT" \
      --synthetic-csv "$SYNTHETIC_CSV" \
      --output-dir "$out_dir" \
      --epochs "$EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --lr "${LR:-0.001}" \
      --hidden-dim "$HIDDEN_DIM" \
      --n-bits "$N_BITS" \
      --dropout "${DROPOUT:-0.20}" \
      --pairwise-weight "${PAIRWISE_WEIGHT:-1.0}" \
      --bce-weight "${BCE_WEIGHT:-1.0}" \
      --seed "$seed" \
      "${extra_args[@]}" \
      > "$LOG_DIR/type1_diverse_anchor_ablation_${name}_${seed}_train.log" 2>&1
  fi

  if [[ ! -f "$same_dir/ranking_metrics.json" ]]; then
    echo "[rerank-same] config=$name seed=$seed"
    PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.evaluate_candidate_reranking \
      --real-csv "$REGIO_ALIGNMENT" \
      --real-csv "$HIT_ALIGNMENT" \
      --synthetic-csv "$SYNTHETIC_CSV" \
      --model-dir "$out_dir" \
      --output-dir "$same_dir" \
      --candidate-scope same_split \
      --batch-size "$BATCH_SIZE" \
      --device cuda \
      > "$LOG_DIR/type1_diverse_anchor_ablation_${name}_${seed}_rerank_same.log" 2>&1
  fi

  if [[ ! -f "$all_dir/ranking_metrics.json" ]]; then
    echo "[rerank-all] config=$name seed=$seed"
    PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.evaluate_candidate_reranking \
      --real-csv "$REGIO_ALIGNMENT" \
      --real-csv "$HIT_ALIGNMENT" \
      --synthetic-csv "$SYNTHETIC_CSV" \
      --model-dir "$out_dir" \
      --output-dir "$all_dir" \
      --candidate-scope all_group \
      --batch-size "$BATCH_SIZE" \
      --device cuda \
      > "$LOG_DIR/type1_diverse_anchor_ablation_${name}_${seed}_rerank_all.log" 2>&1
  fi

  if [[ ! -f "$family_dir/action_family_contribution.json" ]]; then
    echo "[family] config=$name seed=$seed"
    PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.analyze_action_family_contribution \
      --synthetic-csv "$SYNTHETIC_CSV" \
      --real-csv "$REGIO_ALIGNMENT" \
      --real-csv "$HIT_ALIGNMENT" \
      --score-csv "${name}_seed${seed}=$same_dir/candidate_scores.csv" \
      --output-dir "$family_dir" \
      > "$LOG_DIR/type1_diverse_anchor_ablation_${name}_${seed}_family.log" 2>&1
  fi
}

for seed in $SEEDS; do
  for config in $CONFIGS; do
    case "$config" in
      pairwise_default)
        run_config "$config" "$seed"
        ;;
      family_margin)
        run_config "$config" "$seed" \
          --margin "${FAMILY_BASE_MARGIN:-0.05}" \
          --family-margin "regio=${REGIO_MARGIN:-0.20}" \
          --family-margin "heteroatom=${HETEROATOM_MARGIN:-0.15}" \
          --family-weight "regio=${REGIO_WEIGHT:-1.5}" \
          --family-weight "heteroatom=${HETEROATOM_WEIGHT:-1.0}"
        ;;
      *)
        echo "[error] unknown config: $config" >&2
        exit 2
        ;;
    esac
  done
done

SUMMARY_JSON="$RESULTS_DIR/summary.json"
export RESULTS_DIR SUMMARY_JSON SEEDS CONFIGS
PYTHONPATH=. "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path


def load(path: Path):
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def family_metrics(payload, family):
    summaries = payload.get("score_summaries", []) if payload else []
    if not summaries:
        return {}
    family_data = summaries[0].get("families", {}).get(family, {})
    out = dict(family_data.get("family_only_challenge_metrics", {}))
    margins = family_data.get("score_margins", {})
    out["positive_wins_rate"] = margins.get("positive_wins_rate", 0.0)
    out["margin_mean"] = margins.get("margin", {}).get("mean", 0.0)
    out["hard_negative_beats_positive"] = margins.get("hard_negative_beats_positive", 0)
    return out


root = Path(os.environ["RESULTS_DIR"])
records = []
for seed in os.environ["SEEDS"].split():
    for config in os.environ["CONFIGS"].split():
        run_dir = root / f"{config}_seed{seed}"
        metrics = load(run_dir / "metrics.json")
        same = load(run_dir / "rerank_same_split" / "ranking_metrics.json")
        all_group = load(run_dir / "rerank_all_group" / "ranking_metrics.json")
        family = load(run_dir / "action_family_contribution" / "action_family_contribution.json")
        if not metrics or not same:
            continue
        records.append(
            {
                "run": run_dir.name,
                "setting": config,
                "seed": int(seed),
                "binary_test": metrics.get("test", {}),
                "counts": metrics.get("counts", {}),
                "ranking": same.get("overall", {}),
                "test_ranking": same.get("by_split", {}).get("test", {}),
                "synthetic_ranking": same.get("by_candidate_source", {}).get("synthetic", {}),
                "all_group_ranking": (all_group or {}).get("overall", {}),
                "regio_challenge": family_metrics(family, "regio"),
                "heteroatom_challenge": family_metrics(family, "heteroatom"),
                "paths": {
                    "metrics": str(run_dir / "metrics.json"),
                    "same_ranking": str(run_dir / "rerank_same_split" / "ranking_metrics.json"),
                    "all_ranking": str(run_dir / "rerank_all_group" / "ranking_metrics.json"),
                    "family": str(run_dir / "action_family_contribution" / "action_family_contribution.json"),
                },
            }
        )

summary = {"records": records}
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
  --family all_group_ranking \
  --family regio_challenge \
  --family heteroatom_challenge

echo "Type-1 diverse-anchor ablation complete: $RESULTS_DIR"
