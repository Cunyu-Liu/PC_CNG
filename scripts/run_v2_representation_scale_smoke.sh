#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/cunyuliu/pc_cng_research
PY=/home/cunyuliu/miniconda3/envs/pc_cng_gpu/bin/python
GPU=${PC_CNG_GPU:-5}

cd "$ROOT/chem_negative_sampling"

EXP_NAME="type1_v2_representation_scale_smoke_20260712"
OUT_ROOT="$ROOT/results/$EXP_NAME"
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

SEEDS=(20260710 20260711 20260712)
SEED_CSV=$(IFS=,; echo "${SEEDS[*]}")

CONFIG_NAMES=("nbits8192" "binary_count4096")
CONFIG_ARGS=(
  "--n-bits 8192 --fp-mode binary"
  "--n-bits 4096 --fp-mode binary_count"
)

for idx in "${!CONFIG_NAMES[@]}"; do
  config_name="${CONFIG_NAMES[$idx]}"
  config_args="${CONFIG_ARGS[$idx]}"

  for seed in "${SEEDS[@]}"; do
    seed_name="v2_${config_name}_pairwise_seed${seed}"
    seed_out="$OUT_ROOT/$seed_name"
    log="$ROOT/results/logs/${EXP_NAME}_${config_name}_seed${seed}.log"

    if [ -f "$seed_out/rerank_same_split/ranking_metrics.json" ]; then
      echo "Config $config_name seed $seed already done, skipping"
      continue
    fi

    echo "Starting representation smoke config=$config_name seed=$seed on GPU $GPU..."
    mkdir -p "$seed_out"

    CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=. "$PY" -m pc_cng.train_pairwise_reward_mlp \
      $REAL $SYN_ARGS \
      --output-dir "$seed_out" \
      --epochs 80 \
      --batch-size 4096 \
      --lr 0.001 \
      --hidden-dim 2048 \
      --dropout 0.20 \
      --feature-mode morgan \
      --checkpoint-metric val_roc_auc \
      $config_args \
      --seed "$seed" \
      > "$log" 2>&1

    echo "Training done for config=$config_name seed=$seed; running same-split rerank..."

    CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=. "$PY" -m pc_cng.evaluate_candidate_reranking \
      $REAL $SYN_ARGS \
      --model-dir "$seed_out" \
      --output-dir "$seed_out/rerank_same_split" \
      --group-by reactants \
      --candidate-scope same_split \
      --batch-size 4096 \
      >> "$log" 2>&1

    echo "Config $config_name seed $seed done."
  done

  summary_dir="$OUT_ROOT/${config_name}_smoke_summary"
  mkdir -p "$summary_dir"
  PYTHONPATH=. "$PY" -m pc_cng.multiseed_summary \
    --exp-dir "$OUT_ROOT" \
    --prefix "v2_${config_name}_pairwise_seed" \
    --seeds "$SEED_CSV" \
    --output "$summary_dir/summary.json"

  sig_args=()
  for seed in "${SEEDS[@]}"; do
    sig_args+=(--baseline "$ROOT/results/type1_v2_filtered_baseline_20260712/v2_filtered_pairwise_seed${seed}/rerank_same_split/candidate_scores.csv")
    sig_args+=(--candidate "$OUT_ROOT/v2_${config_name}_pairwise_seed${seed}/rerank_same_split/candidate_scores.csv")
  done

  PYTHONPATH=. "$PY" -m pc_cng.multiseed_paired_significance \
    "${sig_args[@]}" \
    --baseline-name v2_filtered_3seed \
    --candidate-name "${config_name}_3seed" \
    --output-dir "$OUT_ROOT/paired_significance_v2_vs_${config_name}_smoke" \
    --bootstrap-iterations 5000 \
    --seed 20260712
done

echo "representation-scale smoke complete."
