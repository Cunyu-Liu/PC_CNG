#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/cunyuliu/pc_cng_research
PY=/home/cunyuliu/miniconda3/envs/pc_cng_gpu/bin/python
cd "$ROOT/chem_negative_sampling"
mkdir -p "$ROOT/results/logs"
CAND="$ROOT/results/masked_hard_decoder_full/candidates.csv"
RULE_SYN="$ROOT/results/masked_hard_decoder_full/rule_hard_negatives_reviewed.csv"
REAL="--real-csv $ROOT/data/processed/regiosqm20_normalized.csv --real-csv $ROOT/data/processed/hitea_full_normalized.csv"
DECODER_DIR="$ROOT/results/masked_hard_decoder_full/train_masked_hard_decoder"
PAIRWISE_DIR="$ROOT/results/rule_hard_negatives_pairwise_reward_h2048_n4096_e80"
BCE_DIR="$ROOT/results/rule_hard_negatives_direct_bce_h2048_n4096_e80"
if [ ! -f "$DECODER_DIR/metrics.json" ]; then
  mkdir -p "$DECODER_DIR"
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. nohup "$PY" -m pc_cng.train_reaction_center_edit_decoder \
    --input "$CAND" \
    --output-dir "$DECODER_DIR" \
    --epochs 80 \
    --batch-size 1024 \
    --hidden-dim 512 \
    --dropout 0.15 \
    --loss-mode masked_hard_negative \
    --positive-bce-weight 1.0 \
    --hard-bce-weight 1.0 \
    --positive-rank-weight 0.5 \
    --hard-rank-weight 1.0 \
    > "$ROOT/results/logs/masked_hard_decoder_train.log" 2>&1 &
  echo "$!" > "$DECODER_DIR/pid.txt"
fi
if [ ! -f "$PAIRWISE_DIR/metrics.json" ]; then
  mkdir -p "$PAIRWISE_DIR"
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. nohup "$PY" -m pc_cng.train_pairwise_reward_mlp \
    $REAL \
    --synthetic-csv "$RULE_SYN" \
    --output-dir "$PAIRWISE_DIR" \
    --epochs 80 \
    --batch-size 4096 \
    --lr 0.001 \
    --hidden-dim 2048 \
    --n-bits 4096 \
    --dropout 0.20 \
    --pairwise-weight 1.0 \
    --bce-weight 1.0 \
    --seed 20260710 \
    > "$ROOT/results/logs/rule_hard_pairwise_reward.log" 2>&1 &
  echo "$!" > "$PAIRWISE_DIR/pid.txt"
fi
if [ ! -f "$BCE_DIR/metrics.json" ]; then
  mkdir -p "$BCE_DIR"
  CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. nohup "$PY" -m pc_cng.train_feasibility_mlp \
    $REAL \
    --synthetic-csv "$RULE_SYN" \
    --output-dir "$BCE_DIR" \
    --epochs 80 \
    --batch-size 4096 \
    --lr 0.001 \
    --hidden-dim 2048 \
    --n-bits 4096 \
    --dropout 0.20 \
    --seed 20260710 \
    > "$ROOT/results/logs/rule_hard_direct_bce.log" 2>&1 &
  echo "$!" > "$BCE_DIR/pid.txt"
fi
