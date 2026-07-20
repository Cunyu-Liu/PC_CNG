#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/cunyuliu/pc_cng_research}
PYTHON_BIN=${PYTHON_BIN:-/home/cunyuliu/miniconda3/envs/pc_cng_gpu/bin/python}
REACTION_LM_PYTHON=${REACTION_LM_PYTHON:-$ROOT/envs/reaction_lm/bin/python}
GPU_EVAL=${GPU_EVAL:-1}

CODE_DIR="$ROOT/chem_negative_sampling"
BASE_RESULTS_DIR=${BASE_RESULTS_DIR:-$ROOT/results/science_advances_regiosqm_benchmark_full}
RESULTS_DIR=${RESULTS_DIR:-$ROOT/results/science_advances_chemformer_lm_benchmark}
LOG_DIR=${LOG_DIR:-$ROOT/results/logs}

SEEDS=${SEEDS:-20260710 20260711 20260712 20260713 20260714 20260715 20260716 20260717 20260718 20260719}
SETTINGS=${SETTINGS:-k_low k_high}
SCORER=${SCORER:-chemformer_log_likelihood}
MODEL_NAME=${MODEL_NAME:-chemformer_forward_uspto50k}
MODEL_PATH=${MODEL_PATH:-$ROOT/models/reaction_lm/chemformer_forward_uspto50k/model_sanitized.ckpt}
VOCABULARY_PATH=${VOCABULARY_PATH:-$ROOT/external/reaction_lm/Chemformer/bart_vocab.json}
BATCH_SIZE=${BATCH_SIZE:-128}

mkdir -p "$RESULTS_DIR" "$LOG_DIR"
cd "$CODE_DIR"

for setting in $SETTINGS; do
  for seed in $SEEDS; do
    SRC_RUN_DIR="$BASE_RESULTS_DIR/${setting}_seed${seed}"
    SPLIT_CSV="$SRC_RUN_DIR/split/regiosqm20_science_advances_split.csv"
    NEG_CSV="$SRC_RUN_DIR/generated_negatives/boundary_negatives_reviewed.csv"
    RUN_DIR="$RESULTS_DIR/${setting}_seed${seed}"
    mkdir -p "$RUN_DIR"

    if [[ ! -f "$SPLIT_CSV" ]]; then
      echo "[skip] missing split: $SPLIT_CSV"
      continue
    fi

    for family in real_candidates pc_cng_candidates; do
      FAMILY_DIR="$RUN_DIR/$family"
      CANDIDATES_CSV="$FAMILY_DIR/lm_candidates.csv"
      CANDIDATES_SUMMARY="$FAMILY_DIR/lm_candidates_summary.json"
      SCORES_CSV="$FAMILY_DIR/lm_scores_${SCORER}.csv"
      SCORES_SUMMARY="$FAMILY_DIR/lm_scores_${SCORER}_summary.json"
      METRICS_JSON="$FAMILY_DIR/lm_scores_${SCORER}_metrics.json"
      mkdir -p "$FAMILY_DIR"

      if [[ ! -f "$CANDIDATES_SUMMARY" ]]; then
        echo "[candidates] $setting seed=$seed family=$family"
        BUILD_ARGS=(--real-csv "$SPLIT_CSV")
        if [[ "$family" == "pc_cng_candidates" ]]; then
          if [[ ! -f "$NEG_CSV" ]]; then
            echo "[skip] missing negatives: $NEG_CSV"
            continue
          fi
          BUILD_ARGS+=(--synthetic-csv "$NEG_CSV")
        fi
        PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.build_reaction_lm_candidate_set \
          "${BUILD_ARGS[@]}" \
          --candidate-scope same_split \
          --output "$CANDIDATES_CSV" \
          --summary "$CANDIDATES_SUMMARY" \
          > "$LOG_DIR/scadv_lm_${setting}_${seed}_${family}_candidates.log" 2>&1
      fi

      if [[ ! -f "$SCORES_SUMMARY" ]]; then
        echo "[score] $setting seed=$seed family=$family scorer=$SCORER"
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
          --batch-size "$BATCH_SIZE" \
          --device cuda \
          > "$LOG_DIR/scadv_lm_${setting}_${seed}_${family}_score.log" 2>&1
      fi

      if [[ ! -f "$METRICS_JSON" ]]; then
        echo "[metrics] $setting seed=$seed family=$family"
        PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.evaluate_reaction_lm_scores \
          --input "$SCORES_CSV" \
          --output "$METRICS_JSON" \
          > "$LOG_DIR/scadv_lm_${setting}_${seed}_${family}_metrics.log" 2>&1
      fi
    done
  done
done

SUMMARY_JSON="$RESULTS_DIR/summary.json"
export RESULTS_DIR SUMMARY_JSON SCORER
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
records = []
for run_dir in sorted(root.glob("k_*_seed*")):
    setting, seed_part = run_dir.name.rsplit("_seed", 1)
    real_metrics = load(run_dir / "real_candidates" / f"lm_scores_{os.environ['SCORER']}_metrics.json")
    pc_metrics = load(run_dir / "pc_cng_candidates" / f"lm_scores_{os.environ['SCORER']}_metrics.json")
    if not real_metrics or not pc_metrics:
        continue
    record = {
        "run": run_dir.name,
        "setting": setting,
        "seed": int(seed_part),
        "real_candidates": real_metrics["overall"],
        "pc_cng_candidates": pc_metrics["overall"],
        "delta": {
            metric: pc_metrics["overall"].get(metric, 0.0) - real_metrics["overall"].get(metric, 0.0)
            for metric in ["top1", "top3", "mrr", "ndcg"]
        },
        "paths": {
            "real_metrics": str(run_dir / "real_candidates" / f"lm_scores_{os.environ['SCORER']}_metrics.json"),
            "pc_cng_metrics": str(run_dir / "pc_cng_candidates" / f"lm_scores_{os.environ['SCORER']}_metrics.json"),
        },
    }
    records.append(record)

summary = {
    "scorer": os.environ["SCORER"],
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
  --family real_candidates \
  --family pc_cng_candidates

echo "Science Advances-style Chemformer LM benchmark complete."
