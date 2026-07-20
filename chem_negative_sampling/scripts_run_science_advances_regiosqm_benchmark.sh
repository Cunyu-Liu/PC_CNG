#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/cunyuliu/pc_cng_research}
PYTHON_BIN=${PYTHON_BIN:-/home/cunyuliu/miniconda3/envs/pc_cng_gpu/bin/python}
GPU_TRAIN=${GPU_TRAIN:-0}

CODE_DIR="$ROOT/chem_negative_sampling"
DATASET="$ROOT/data/processed/regiosqm20_normalized.csv"
RESULTS_DIR=${RESULTS_DIR:-$ROOT/results/science_advances_regiosqm_benchmark}
LOG_DIR=${LOG_DIR:-$ROOT/results/logs}

SEEDS=${SEEDS:-20260710 20260711 20260712}
SETTINGS=${SETTINGS:-k_low k_high}
EPOCHS=${EPOCHS:-40}
BATCH_SIZE=${BATCH_SIZE:-1024}
HIDDEN_DIM=${HIDDEN_DIM:-1024}
N_BITS=${N_BITS:-2048}
MAX_CANDIDATES_PER_REACTION=${MAX_CANDIDATES_PER_REACTION:-4}

mkdir -p "$RESULTS_DIR" "$LOG_DIR"
cd "$CODE_DIR"

for setting in $SETTINGS; do
  for seed in $SEEDS; do
    RUN_DIR="$RESULTS_DIR/${setting}_seed${seed}"
    SPLIT_DIR="$RUN_DIR/split"
    NEG_DIR="$RUN_DIR/generated_negatives"
    REAL_ONLY_DIR="$RUN_DIR/real_only"
    PC_CNG_DIR="$RUN_DIR/pc_cng_augmented"
    REAL_RANK_DIR="$RUN_DIR/rerank_real_only"
    PC_CNG_RANK_DIR="$RUN_DIR/rerank_pc_cng_augmented"
    mkdir -p "$RUN_DIR" "$SPLIT_DIR" "$NEG_DIR"

    if [ ! -f "$SPLIT_DIR/summary.json" ]; then
      echo "[split] $setting seed=$seed"
      PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.build_science_advances_regiosqm_splits \
        --input "$DATASET" \
        --output-dir "$SPLIT_DIR" \
        --setting "$setting" \
        --seed "$seed" \
        > "$LOG_DIR/scadv_${setting}_${seed}_split.log" 2>&1
    fi

    if [ ! -f "$NEG_DIR/review_summary.json" ]; then
      echo "[generate] $setting seed=$seed"
      PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.run_boundary_generation \
        --input "$SPLIT_DIR/train_positives.csv" \
        --output "$NEG_DIR/boundary_negatives.csv" \
        --summary "$NEG_DIR/boundary_negatives_summary.json" \
        --max-candidates-per-reaction "$MAX_CANDIDATES_PER_REACTION" \
        > "$LOG_DIR/scadv_${setting}_${seed}_generate.log" 2>&1

      PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.false_negative_review \
        --input "$NEG_DIR/boundary_negatives.csv" \
        --output "$NEG_DIR/boundary_negatives_reviewed.csv" \
        --summary "$NEG_DIR/review_summary.json" \
        --known-positive "$SPLIT_DIR/regiosqm20_science_advances_split.csv" \
        > "$LOG_DIR/scadv_${setting}_${seed}_review.log" 2>&1
    fi

    if [ ! -f "$REAL_ONLY_DIR/metrics.json" ]; then
      echo "[train real-only] $setting seed=$seed"
      mkdir -p "$REAL_ONLY_DIR"
      CUDA_VISIBLE_DEVICES="$GPU_TRAIN" PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.train_feasibility_mlp \
        --real-csv "$SPLIT_DIR/regiosqm20_science_advances_split.csv" \
        --output-dir "$REAL_ONLY_DIR" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --hidden-dim "$HIDDEN_DIM" \
        --n-bits "$N_BITS" \
        --dropout 0.20 \
        --seed "$seed" \
        > "$LOG_DIR/scadv_${setting}_${seed}_train_real.log" 2>&1
    fi

    if [ ! -f "$PC_CNG_DIR/metrics.json" ]; then
      echo "[train pc-cng] $setting seed=$seed"
      mkdir -p "$PC_CNG_DIR"
      CUDA_VISIBLE_DEVICES="$GPU_TRAIN" PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.train_feasibility_mlp \
        --real-csv "$SPLIT_DIR/regiosqm20_science_advances_split.csv" \
        --synthetic-csv "$NEG_DIR/boundary_negatives_reviewed.csv" \
        --output-dir "$PC_CNG_DIR" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --hidden-dim "$HIDDEN_DIM" \
        --n-bits "$N_BITS" \
        --dropout 0.20 \
        --origin-weight synthetic=0.5 \
        --seed "$seed" \
        > "$LOG_DIR/scadv_${setting}_${seed}_train_pc_cng.log" 2>&1
    fi

    if [ ! -f "$REAL_RANK_DIR/ranking_metrics.json" ]; then
      echo "[rerank real-only] $setting seed=$seed"
      CUDA_VISIBLE_DEVICES="$GPU_TRAIN" PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.evaluate_candidate_reranking \
        --real-csv "$SPLIT_DIR/regiosqm20_science_advances_split.csv" \
        --output-dir "$REAL_RANK_DIR" \
        --candidate-scope same_split \
        --model-dir "$REAL_ONLY_DIR" \
        > "$LOG_DIR/scadv_${setting}_${seed}_rerank_real.log" 2>&1
    fi

    if [ ! -f "$PC_CNG_RANK_DIR/ranking_metrics.json" ]; then
      echo "[rerank pc-cng] $setting seed=$seed"
      CUDA_VISIBLE_DEVICES="$GPU_TRAIN" PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.evaluate_candidate_reranking \
        --real-csv "$SPLIT_DIR/regiosqm20_science_advances_split.csv" \
        --synthetic-csv "$NEG_DIR/boundary_negatives_reviewed.csv" \
        --output-dir "$PC_CNG_RANK_DIR" \
        --candidate-scope same_split \
        --model-dir "$PC_CNG_DIR" \
        > "$LOG_DIR/scadv_${setting}_${seed}_rerank_pc_cng.log" 2>&1
    fi
  done
done

SUMMARY="$RESULTS_DIR/summary.json"
export RESULTS_DIR SUMMARY
PYTHONPATH=. "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path
from statistics import mean, pstdev


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
    split_summary = load(run_dir / "split" / "summary.json")
    neg_summary = load(run_dir / "generated_negatives" / "boundary_negatives_summary.json")
    review_summary = load(run_dir / "generated_negatives" / "review_summary.json")
    real_metrics = load(run_dir / "rerank_real_only" / "ranking_metrics.json")
    pc_metrics = load(run_dir / "rerank_pc_cng_augmented" / "ranking_metrics.json")
    if not real_metrics or not pc_metrics:
        continue
    record = {
        "run": run_dir.name,
        "setting": setting,
        "seed": int(seed_part),
        "split": split_summary,
        "generated": neg_summary,
        "review": review_summary,
        "real_only": real_metrics["overall"],
        "pc_cng_augmented": pc_metrics["overall"],
        "delta": {
            key: pc_metrics["overall"].get(key, 0.0) - real_metrics["overall"].get(key, 0.0)
            for key in ["top1", "top3", "mrr", "ndcg"]
        },
    }
    records.append(record)

aggregate = {}
for setting in sorted({record["setting"] for record in records}):
    subset = [record for record in records if record["setting"] == setting]
    aggregate[setting] = {}
    for family in ["real_only", "pc_cng_augmented", "delta"]:
        aggregate[setting][family] = {}
        for metric in ["top1", "top3", "mrr", "ndcg"]:
            values = [float(record[family].get(metric, 0.0)) for record in subset]
            aggregate[setting][family][metric] = {
                "mean": mean(values) if values else 0.0,
                "std": pstdev(values) if len(values) > 1 else 0.0,
                "n": len(values),
            }

summary = {"records": records, "aggregate": aggregate}
out = Path(os.environ["SUMMARY"])
json.dump(summary, open(out, "w"), indent=2, ensure_ascii=False)
print(json.dumps(summary["aggregate"], indent=2, ensure_ascii=False))
PY

echo "Science Advances-style RegioSQM benchmark complete."
