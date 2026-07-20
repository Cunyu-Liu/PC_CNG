#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/cunyuliu/pc_cng_research}
PYTHON_BIN=${PYTHON_BIN:-/home/cunyuliu/miniconda3/envs/pc_cng_gpu/bin/python}
GPU_TRAIN=${GPU_TRAIN:-4}
GPU_EVAL=${GPU_EVAL:-$GPU_TRAIN}

CODE_DIR="$ROOT/chem_negative_sampling"
RESULTS_DIR=${RESULTS_DIR:-$ROOT/results/type1_graph_stats_pairwise_full}
LOG_DIR=${LOG_DIR:-$ROOT/results/logs}

REGIO_ALIGNMENT=${REGIO_ALIGNMENT:-$ROOT/data/processed/regiosqm20_normalized.csv}
HITEA_ALIGNMENT=${HITEA_ALIGNMENT:-$ROOT/data/processed/hitea_full_normalized.csv}
SYNTHETIC_CSV=${SYNTHETIC_CSV:-$ROOT/results/type1_diverse_anchor_full/diverse_anchor_candidates_reviewed.csv}

SEEDS=${SEEDS:-20260710 20260711 20260712 20260713 20260714}
EPOCHS=${EPOCHS:-80}
BATCH_SIZE=${BATCH_SIZE:-4096}
HIDDEN_DIM=${HIDDEN_DIM:-1024}
DROPOUT=${DROPOUT:-0.20}

EXTERNAL_CANDIDATES=${EXTERNAL_CANDIDATES:-$ROOT/results/external_product_prediction_benchmark_20260711/full_observed_pc_cng_chemformer_beam_candidates.csv}
EXTERNAL_LM_SCORES=${EXTERNAL_LM_SCORES:-$ROOT/results/external_product_prediction_benchmark_20260711/lm_scores_chemformer_log_likelihood.csv}
RUN_EXTERNAL=${RUN_EXTERNAL:-1}

mkdir -p "$RESULTS_DIR" "$LOG_DIR"
cd "$CODE_DIR"

MODEL_ARGS=()
for seed in $SEEDS; do
  run_dir="$RESULTS_DIR/graph_stats_seed${seed}"
  MODEL_ARGS+=(--model-dir "$run_dir")
  if [[ ! -f "$run_dir/metrics.json" ]]; then
    echo "[train] graph_stats seed=$seed"
    mkdir -p "$run_dir"
    CUDA_VISIBLE_DEVICES="$GPU_TRAIN" PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.train_pairwise_reward_mlp \
      --real-csv "$REGIO_ALIGNMENT" \
      --real-csv "$HITEA_ALIGNMENT" \
      --synthetic-csv "$SYNTHETIC_CSV" \
      --output-dir "$run_dir" \
      --feature-mode graph_stats \
      --epochs "$EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --hidden-dim "$HIDDEN_DIM" \
      --dropout "$DROPOUT" \
      --pairwise-weight 1.0 \
      --bce-weight 1.0 \
      --seed "$seed" \
      > "$LOG_DIR/type1_graph_stats_pairwise_${seed}_train.log" 2>&1
  fi

  same_dir="$run_dir/rerank_same_split"
  if [[ ! -f "$same_dir/ranking_metrics.json" ]]; then
    echo "[rerank] graph_stats seed=$seed"
    CUDA_VISIBLE_DEVICES="$GPU_EVAL" PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.evaluate_candidate_reranking \
      --real-csv "$REGIO_ALIGNMENT" \
      --real-csv "$HITEA_ALIGNMENT" \
      --synthetic-csv "$SYNTHETIC_CSV" \
      --model-dir "$run_dir" \
      --output-dir "$same_dir" \
      --candidate-scope same_split \
      --batch-size "$BATCH_SIZE" \
      --device cuda \
      > "$LOG_DIR/type1_graph_stats_pairwise_${seed}_rerank_same.log" 2>&1
  fi
done

ENSEMBLE_DIR="$RESULTS_DIR/graph_stats_ensemble5"
if [[ ! -f "$ENSEMBLE_DIR/rerank_same_split/ranking_metrics.json" ]]; then
  echo "[rerank] graph_stats ensemble"
  CUDA_VISIBLE_DEVICES="$GPU_EVAL" PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.evaluate_candidate_reranking \
    --real-csv "$REGIO_ALIGNMENT" \
    --real-csv "$HITEA_ALIGNMENT" \
    --synthetic-csv "$SYNTHETIC_CSV" \
    "${MODEL_ARGS[@]}" \
    --output-dir "$ENSEMBLE_DIR/rerank_same_split" \
    --candidate-scope same_split \
    --batch-size "$BATCH_SIZE" \
    --device cuda \
    > "$LOG_DIR/type1_graph_stats_pairwise_ensemble_rerank_same.log" 2>&1
fi

if [[ "$RUN_EXTERNAL" == "1" && -f "$EXTERNAL_CANDIDATES" && -f "$EXTERNAL_LM_SCORES" ]]; then
  if [[ ! -f "$ENSEMBLE_DIR/external_validity_aware/benchmark_summary.json" ]]; then
    echo "[external] graph_stats ensemble validity-aware"
    CUDA_VISIBLE_DEVICES="$GPU_EVAL" PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.evaluate_external_product_prediction_benchmark \
      --candidate-csv "$EXTERNAL_CANDIDATES" \
      --external-score "chemformer_likelihood=$EXTERNAL_LM_SCORES:lm_score" \
      --primary-external-score chemformer_likelihood \
      "${MODEL_ARGS[@]}" \
      --pc-cng-invalid-negative-score 0.0 \
      --output-dir "$ENSEMBLE_DIR/external_validity_aware" \
      --batch-size "$BATCH_SIZE" \
      --device cuda \
      --normalization group_zscore \
      > "$LOG_DIR/type1_graph_stats_pairwise_ensemble_external_validity.log" 2>&1
  fi
fi

SUMMARY_JSON="$RESULTS_DIR/summary.json"
export RESULTS_DIR SUMMARY_JSON SEEDS
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
for seed in os.environ["SEEDS"].split():
    run_dir = root / f"graph_stats_seed{seed}"
    metrics = load(run_dir / "metrics.json")
    ranking = load(run_dir / "rerank_same_split" / "ranking_metrics.json")
    if metrics and ranking:
        records.append(
            {
                "run": run_dir.name,
                "setting": "graph_stats",
                "seed": int(seed),
                "binary_test": metrics.get("test", {}),
                "ranking": ranking.get("overall", {}),
                "test_ranking": ranking.get("by_split", {}).get("test", {}),
                "synthetic_ranking": ranking.get("by_candidate_source", {}).get("synthetic", {}),
                "paths": {
                    "metrics": str(run_dir / "metrics.json"),
                    "ranking": str(run_dir / "rerank_same_split" / "ranking_metrics.json"),
                },
            }
        )

ensemble_dir = root / "graph_stats_ensemble5"
ensemble_ranking = load(ensemble_dir / "rerank_same_split" / "ranking_metrics.json")
ensemble_external = load(ensemble_dir / "external_validity_aware" / "benchmark_summary.json")
summary = {
    "records": records,
    "ensemble": {
        "same_context": ensemble_ranking,
        "external_validity_aware": ensemble_external,
    },
}
with open(os.environ["SUMMARY_JSON"], "w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2, ensure_ascii=False)
print(json.dumps({"records": len(records), "summary": os.environ["SUMMARY_JSON"]}, indent=2))
PY

echo "Graph-stats pairwise benchmark complete: $RESULTS_DIR"
