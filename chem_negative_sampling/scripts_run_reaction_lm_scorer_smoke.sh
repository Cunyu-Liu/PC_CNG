#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/cunyuliu/pc_cng_research}
PYTHON_BIN=${PYTHON_BIN:-/home/cunyuliu/miniconda3/envs/pc_cng_gpu/bin/python}
REACTION_LM_PYTHON=${REACTION_LM_PYTHON:-$ROOT/envs/reaction_lm/bin/python}
SCORER=${SCORER:-length_baseline}
MODEL_NAME=${MODEL_NAME:-$SCORER}
N_BEAMS=${N_BEAMS:-10}
REACTION_LM_DEVICE=${REACTION_LM_DEVICE:-cuda}

CODE_DIR="$ROOT/chem_negative_sampling"
RESULTS_DIR="$ROOT/results/reaction_lm_scorer_smoke"

HIT_ALIGNMENT="$ROOT/data/processed/hitea_full_normalized.csv"
REGIO_ALIGNMENT="$ROOT/data/processed/regiosqm20_normalized.csv"
EXPANDED_REVIEWED="$ROOT/results/expanded_hard_negative_actions_full/expanded_hard_negatives_reviewed.csv"

mkdir -p "$RESULTS_DIR"
cd "$CODE_DIR"

SCORER_ARGS=()
if [[ -n "${MODEL_PATH:-}" ]]; then
  SCORER_ARGS+=(--model-path "$MODEL_PATH")
fi
if [[ -n "${VOCABULARY_PATH:-}" ]]; then
  SCORER_ARGS+=(--vocabulary-path "$VOCABULARY_PATH")
fi
if [[ -n "${CHEMFORMER_ROOT:-}" ]]; then
  SCORER_ARGS+=(--chemformer-root "$CHEMFORMER_ROOT")
fi
if [[ -n "${MOLECULAR_TRANSFORMER_ROOT:-}" ]]; then
  SCORER_ARGS+=(--molecular-transformer-root "$MOLECULAR_TRANSFORMER_ROOT")
fi
if [[ -n "${REACTION_LM_WORK_DIR:-}" ]]; then
  SCORER_ARGS+=(--work-dir "$REACTION_LM_WORK_DIR")
fi
if [[ "${INCLUDE_AGENTS:-1}" == "0" ]]; then
  SCORER_ARGS+=(--exclude-agents)
fi

PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.build_reaction_lm_candidate_set \
  --real-csv "$REGIO_ALIGNMENT" \
  --real-csv "$HIT_ALIGNMENT" \
  --synthetic-csv "$EXPANDED_REVIEWED" \
  --candidate-scope same_split \
  --output "$RESULTS_DIR/lm_candidates.csv" \
  --summary "$RESULTS_DIR/lm_candidates_summary.json"

PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.reaction_lm_scorer \
  --input "$RESULTS_DIR/lm_candidates.csv" \
  --output "$RESULTS_DIR/lm_scores_${SCORER}.csv" \
  --summary "$RESULTS_DIR/lm_scores_${SCORER}_summary.json" \
  --scorer "$SCORER" \
  --model-name "$MODEL_NAME" \
  --root "$ROOT" \
  --reaction-lm-python "$REACTION_LM_PYTHON" \
  --n-beams "$N_BEAMS" \
  --device "$REACTION_LM_DEVICE" \
  "${SCORER_ARGS[@]}"

PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.evaluate_reaction_lm_scores \
  --input "$RESULTS_DIR/lm_scores_${SCORER}.csv" \
  --output "$RESULTS_DIR/lm_scores_${SCORER}_metrics.json"

echo "Reaction LM scorer smoke complete for SCORER=$SCORER."
