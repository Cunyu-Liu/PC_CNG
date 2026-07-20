#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/cunyuliu/pc_cng_research}
PY=${PY:-/home/cunyuliu/miniconda3/envs/pc_cng_gpu/bin/python}
GPU=${PC_CNG_GPU:-5}
CONFIG_NAME=${CONFIG_NAME:-pw20_m000}

case "$CONFIG_NAME" in
  pw20_m000)
    PAIRWISE_WEIGHT=2.0
    MARGIN=0.0
    ;;
  pw20_m005)
    PAIRWISE_WEIGHT=2.0
    MARGIN=0.05
    ;;
  *)
    echo "Unsupported CONFIG_NAME=$CONFIG_NAME; expected pw20_m000 or pw20_m005" >&2
    exit 2
    ;;
esac

cd "$ROOT/chem_negative_sampling"

EXP_NAME="type1_v2_pairwise_margin_10seed_20260714"
OUT_ROOT="$ROOT/results/$EXP_NAME"
SMOKE_ROOT="$ROOT/results/type1_v2_pairwise_margin_smoke_20260712"
mkdir -p "$OUT_ROOT" "$ROOT/results/logs"

REAL="--real-csv $ROOT/data/processed/regiosqm20_normalized.csv --real-csv $ROOT/data/processed/hitea_full_normalized.csv"
SYN_CSVS=(
  "$ROOT/results/type1_diverse_anchor_full/diverse_anchor_candidates_reviewed_knownpos_filtered.csv"
  "$ROOT/results/type1_class_quota_supplement_20260711/class_quota_candidates_reviewed.csv"
  "$ROOT/results/type1_class_fallback_supplement_20260711/class_fallback_candidates_reviewed.csv"
  "$ROOT/results/type1_partial_product_supplement_20260711/partial_product_candidates_reviewed.csv"
  "$ROOT/results/type1_unreacted_substrate_supplement_v2_20260711/unreacted_substrate_candidates_reviewed.csv"
)

SYN_ARGS=""
for csv in "${SYN_CSVS[@]}"; do
  SYN_ARGS="$SYN_ARGS --synthetic-csv $csv"
done

SEEDS=(20260710 20260711 20260712 20260713 20260714 20260715 20260716 20260717 20260718 20260719)
SEED_CSV=$(IFS=,; echo "${SEEDS[*]}")

for seed in "${SEEDS[@]}"; do
  seed_name="v2_${CONFIG_NAME}_pairwise_seed${seed}"
  seed_out="$OUT_ROOT/$seed_name"
  smoke_seed_out="$SMOKE_ROOT/$seed_name"
  log="$ROOT/results/logs/${EXP_NAME}_${CONFIG_NAME}_seed${seed}.log"

  if [ ! -d "$seed_out" ] && [ -d "$smoke_seed_out" ]; then
    echo "Reusing smoke seed $CONFIG_NAME/$seed from $smoke_seed_out"
    cp -a "$smoke_seed_out" "$seed_out"
  fi

  if [ -f "$seed_out/rerank_same_split/ranking_metrics.json" ]; then
    echo "Config $CONFIG_NAME seed $seed already done, skipping"
    continue
  fi

  echo "Starting pairwise/margin 10-seed config=$CONFIG_NAME weight=$PAIRWISE_WEIGHT margin=$MARGIN seed=$seed on GPU $GPU..."
  mkdir -p "$seed_out"

  if [ ! -f "$seed_out/best_pairwise_reward_mlp.pt" ]; then
    CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=. "$PY" -m pc_cng.train_pairwise_reward_mlp \
      $REAL $SYN_ARGS \
      --output-dir "$seed_out" \
      --epochs 80 \
      --batch-size 4096 \
      --lr 0.001 \
      --hidden-dim 2048 \
      --n-bits 4096 \
      --dropout 0.20 \
      --feature-mode morgan \
      --pairwise-weight "$PAIRWISE_WEIGHT" \
      --margin "$MARGIN" \
      --checkpoint-metric val_roc_auc \
      --seed "$seed" \
      > "$log" 2>&1
  else
    echo "  Existing checkpoint found, skipping training." >> "$log"
  fi

  echo "  Training done, running same-split rerank..." >> "$log"

  CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=. "$PY" -m pc_cng.evaluate_candidate_reranking \
    $REAL $SYN_ARGS \
    --model-dir "$seed_out" \
    --output-dir "$seed_out/rerank_same_split" \
    --group-by reactants \
    --candidate-scope same_split \
    --batch-size 4096 \
    >> "$log" 2>&1

  echo "Config $CONFIG_NAME seed $seed done."
done

SUMMARY_DIR="$OUT_ROOT/${CONFIG_NAME}_10seed_summary"
mkdir -p "$SUMMARY_DIR"
PYTHONPATH=. "$PY" -m pc_cng.multiseed_summary \
  --exp-dir "$OUT_ROOT" \
  --prefix "v2_${CONFIG_NAME}_pairwise_seed" \
  --seeds "$SEED_CSV" \
  --output "$SUMMARY_DIR/summary.json"

SIG_ARGS=()
for seed in "${SEEDS[@]}"; do
  SIG_ARGS+=(--baseline "$ROOT/results/type1_v2_filtered_baseline_20260712/v2_filtered_pairwise_seed${seed}/rerank_same_split/candidate_scores.csv")
  SIG_ARGS+=(--candidate "$OUT_ROOT/v2_${CONFIG_NAME}_pairwise_seed${seed}/rerank_same_split/candidate_scores.csv")
done

PYTHONPATH=. "$PY" -m pc_cng.multiseed_paired_significance \
  "${SIG_ARGS[@]}" \
  --baseline-name v2_filtered_knownpos \
  --candidate-name "${CONFIG_NAME}_10seed" \
  --output-dir "$OUT_ROOT/paired_significance_filtered_v2_vs_${CONFIG_NAME}_10seed" \
  --bootstrap-iterations 10000 \
  --seed 20260712

cat > "$SUMMARY_DIR/config.json" <<JSON
{
  "experiment": "$EXP_NAME",
  "config_name": "$CONFIG_NAME",
  "pairwise_weight": $PAIRWISE_WEIGHT,
  "margin": $MARGIN,
  "seeds": "$SEED_CSV",
  "reused_smoke_seeds": ["20260710", "20260711", "20260712"]
}
JSON

echo "pairwise/margin 10-seed run complete for $CONFIG_NAME."
