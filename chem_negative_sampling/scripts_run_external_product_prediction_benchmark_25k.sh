#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/cunyuliu/pc_cng_research}
PYTHON_BIN=${PYTHON_BIN:-/home/cunyuliu/miniconda3/envs/pc_cng_gpu/bin/python}
REACTION_LM_PYTHON=${REACTION_LM_PYTHON:-$ROOT/envs/reaction_lm/bin/python}
GPU_EVAL=${GPU_EVAL:-0}

CODE_DIR="$ROOT/chem_negative_sampling"
LOG_DIR=${LOG_DIR:-$ROOT/results/logs}
RESULTS_DIR=${RESULTS_DIR:-$ROOT/results/external_product_prediction_benchmark_25k_20260713}

REGIO_ALIGNMENT=${REGIO_ALIGNMENT:-$ROOT/data/processed/regiosqm20_normalized.csv}
HITEA_ALIGNMENT=${HITEA_ALIGNMENT:-$ROOT/data/processed/hitea_full_normalized.csv}
PRESELECTED_CONTEXTS_CSV=${PRESELECTED_CONTEXTS_CSV:-$ROOT/results/external_product_prediction_context_expansion_20260713/external_product_prediction_contexts_merged.csv}
PRESELECTED_CHEMFORMER_INPUT=${PRESELECTED_CHEMFORMER_INPUT:-$ROOT/results/external_product_prediction_context_expansion_20260713/external_product_prediction_contexts_merged_chemformer_input.csv}

DEFAULT_SYNTHETIC_CSVS=(
  "$ROOT/results/type1_diverse_anchor_full/diverse_anchor_candidates_reviewed_knownpos_filtered.csv"
  "$ROOT/results/type1_class_quota_supplement_20260711/class_quota_candidates_reviewed.csv"
  "$ROOT/results/type1_class_fallback_supplement_20260711/class_fallback_candidates_reviewed.csv"
  "$ROOT/results/type1_partial_product_supplement_20260711/partial_product_candidates_reviewed.csv"
  "$ROOT/results/type1_unreacted_substrate_supplement_v2_20260711/unreacted_substrate_candidates_reviewed.csv"
)
if [[ -n "${SYNTHETIC_CSVS:-}" ]]; then
  read -r -a SYNTHETIC_INPUTS <<< "$SYNTHETIC_CSVS"
else
  SYNTHETIC_INPUTS=("${DEFAULT_SYNTHETIC_CSVS[@]}")
fi

PC_CNG_MODEL_DIRS=${PC_CNG_MODEL_DIRS:-$ROOT/results/type1_diverse_anchor_ablation_full/pairwise_default_seed20260710}
TARGETED_PC_CNG_DIR=${TARGETED_PC_CNG_DIR:-$ROOT/results/external_product_prediction_25k_pc_cng_targeted_20260713}
TARGETED_PC_CNG_CSV=${TARGETED_PC_CNG_CSV:-$TARGETED_PC_CNG_DIR/external_product_prediction_25k_pc_cng_targeted_candidates.csv}
TARGETED_PC_CNG_SUMMARY=${TARGETED_PC_CNG_SUMMARY:-$TARGETED_PC_CNG_DIR/external_product_prediction_25k_pc_cng_targeted_summary.json}
TARGETED_PC_CNG_SUMMARY_MD=${TARGETED_PC_CNG_SUMMARY_MD:-$TARGETED_PC_CNG_DIR/external_product_prediction_25k_pc_cng_targeted_summary.md}
CHEMFORMER_ROOT=${CHEMFORMER_ROOT:-$ROOT/external/reaction_lm/Chemformer}
CHEMFORMER_MODEL_PATH=${CHEMFORMER_MODEL_PATH:-$ROOT/models/reaction_lm/chemformer_forward_uspto50k/model_sanitized.ckpt}
CHEMFORMER_VOCABULARY_PATH=${CHEMFORMER_VOCABULARY_PATH:-$CHEMFORMER_ROOT/bart_vocab.json}
CHEMFORMER_MODEL_NAME=${CHEMFORMER_MODEL_NAME:-chemformer_forward_uspto50k}
SCORER=${SCORER:-chemformer_log_likelihood}
N_BEAMS=${N_BEAMS:-10}
BATCH_SIZE=${BATCH_SIZE:-128}
PC_CNG_BATCH_SIZE=${PC_CNG_BATCH_SIZE:-4096}

# Safe default: do not start GPU-heavy beam generation or benchmark unless
# explicitly requested by a watcher/launcher.
GENERATE_BEAMS=${GENERATE_BEAMS:-0}
RUN_BENCHMARK=${RUN_BENCHMARK:-0}
BUILD_BASE_ONLY=${BUILD_BASE_ONLY:-0}
GENERATE_TARGETED_PC_CNG=${GENERATE_TARGETED_PC_CNG:-0}
FORCE_REBUILD_BASE=${FORCE_REBUILD_BASE:-0}
PC_CNG_INVALID_NEGATIVE_SCORE=${PC_CNG_INVALID_NEGATIVE_SCORE:-}

CONTEXTS_CSV="$RESULTS_DIR/product_prediction_contexts.csv"
CHEMFORMER_INPUT="$RESULTS_DIR/chemformer_forward_input.csv"
BASE_CANDIDATES="$RESULTS_DIR/base_observed_pc_cng_candidates.csv"
BASE_SUMMARY="$RESULTS_DIR/base_observed_pc_cng_candidates_summary.json"
BEAM_CSV=${BEAM_CSV:-$RESULTS_DIR/chemformer_forward_beams.tsv}
FULL_CANDIDATES="$RESULTS_DIR/full_observed_pc_cng_chemformer_beam_candidates.csv"
FULL_SUMMARY="$RESULTS_DIR/full_observed_pc_cng_chemformer_beam_candidates_summary.json"
LM_SCORES="$RESULTS_DIR/lm_scores_${SCORER}.csv"
LM_SCORE_SUMMARY="$RESULTS_DIR/lm_scores_${SCORER}_summary.json"
BENCHMARK_DIR="$RESULTS_DIR/benchmark"
VALIDITY_DIR="$RESULTS_DIR/benchmark_validity_aware"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"
cd "$CODE_DIR"

for required in "$REGIO_ALIGNMENT" "$HITEA_ALIGNMENT" "$PRESELECTED_CONTEXTS_CSV" "$PRESELECTED_CHEMFORMER_INPUT"; do
  if [[ ! -f "$required" ]]; then
    echo "[error] missing required input: $required" >&2
    exit 2
  fi
done

SYN_ARGS=()
for synthetic_csv in "${SYNTHETIC_INPUTS[@]}"; do
  if [[ -f "$synthetic_csv" ]]; then
    SYN_ARGS+=(--synthetic-csv "$synthetic_csv")
  else
    echo "[warn] missing synthetic CSV: $synthetic_csv" >&2
  fi
done

if [[ "$GENERATE_TARGETED_PC_CNG" == "1" && ( ! -f "$TARGETED_PC_CNG_CSV" || ! -f "$TARGETED_PC_CNG_SUMMARY" ) ]]; then
  echo "[build] targeted PC-CNG candidates for 25k contexts"
  PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.generate_external_context_pc_cng_candidates \
    --context-csv "$PRESELECTED_CONTEXTS_CSV" \
    --output "$TARGETED_PC_CNG_CSV" \
    --summary "$TARGETED_PC_CNG_SUMMARY" \
    --summary-md "$TARGETED_PC_CNG_SUMMARY_MD" \
    > "$LOG_DIR/external_product_prediction_25k_targeted_pc_cng.log" 2>&1
fi

if [[ -f "$TARGETED_PC_CNG_CSV" ]]; then
  SYN_ARGS+=(--synthetic-csv "$TARGETED_PC_CNG_CSV")
else
  echo "[info] no targeted PC-CNG context candidates yet: $TARGETED_PC_CNG_CSV" >&2
fi

if [[ ! -f "$CONTEXTS_CSV" ]]; then
  cp "$PRESELECTED_CONTEXTS_CSV" "$CONTEXTS_CSV"
fi
if [[ ! -f "$CHEMFORMER_INPUT" ]]; then
  cp "$PRESELECTED_CHEMFORMER_INPUT" "$CHEMFORMER_INPUT"
fi

if [[ "$FORCE_REBUILD_BASE" == "1" || ! -f "$BASE_SUMMARY" ]]; then
  echo "[build] 25k observed + PC-CNG candidate set"
  PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.build_external_product_prediction_candidate_set \
    --real-csv "$REGIO_ALIGNMENT" \
    --real-csv "$HITEA_ALIGNMENT" \
    --beam-context-csv "$CONTEXTS_CSV" \
    "${SYN_ARGS[@]}" \
    --output "$BASE_CANDIDATES" \
    --summary "$BASE_SUMMARY" \
    > "$LOG_DIR/external_product_prediction_25k_build_base.log" 2>&1
fi

if [[ "$BUILD_BASE_ONLY" == "1" ]]; then
  echo "25k external base candidate build complete: $BASE_SUMMARY"
  exit 0
fi

if [[ "$GENERATE_BEAMS" == "1" && ! -f "$BEAM_CSV" ]]; then
  if [[ ! -f "$CHEMFORMER_MODEL_PATH" || ! -f "$CHEMFORMER_VOCABULARY_PATH" ]]; then
    echo "[error] Chemformer model/vocabulary missing; set BEAM_CSV to an existing prediction file or provide Chemformer paths." >&2
    exit 2
  fi
  echo "[beam] 25k Chemformer forward beams"
  CUDA_VISIBLE_DEVICES="$GPU_EVAL" PYTHONPATH="$CHEMFORMER_ROOT:${PYTHONPATH:-}" "$REACTION_LM_PYTHON" -m molbart.predict \
    data_path="$CHEMFORMER_INPUT" \
    output_sampled_smiles="$BEAM_CSV" \
    model_path="$CHEMFORMER_MODEL_PATH" \
    vocabulary_path="$CHEMFORMER_VOCABULARY_PATH" \
    batch_size="$BATCH_SIZE" \
    n_beams="$N_BEAMS" \
    n_gpus=1 \
    data_device=cuda \
    dataset_part=full \
    task=forward_prediction \
    model_type=bart \
    train_mode=eval \
    > "$LOG_DIR/external_product_prediction_25k_chemformer_beams.log" 2>&1
fi

if [[ ! -f "$BEAM_CSV" ]]; then
  echo "[info] no 25k beam CSV yet. Set GENERATE_BEAMS=1 or BEAM_CSV to continue." >&2
  exit 0
fi

if [[ ! -f "$FULL_SUMMARY" ]]; then
  echo "[build] 25k observed + PC-CNG + external beams"
  PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.build_external_product_prediction_candidate_set \
    --real-csv "$REGIO_ALIGNMENT" \
    --real-csv "$HITEA_ALIGNMENT" \
    --beam-context-csv "$CONTEXTS_CSV" \
    "${SYN_ARGS[@]}" \
    --external-beam-csv "$BEAM_CSV" \
    --external-model-name "$CHEMFORMER_MODEL_NAME" \
    --n-beams "$N_BEAMS" \
    --output "$FULL_CANDIDATES" \
    --summary "$FULL_SUMMARY" \
    > "$LOG_DIR/external_product_prediction_25k_build_full.log" 2>&1
fi

if [[ "$RUN_BENCHMARK" != "1" ]]; then
  echo "[info] full 25k candidates built. Set RUN_BENCHMARK=1 to score/evaluate." >&2
  exit 0
fi

if [[ ! -f "$LM_SCORE_SUMMARY" ]]; then
  echo "[score] 25k $SCORER"
  CUDA_VISIBLE_DEVICES="$GPU_EVAL" PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.reaction_lm_scorer \
    --input "$FULL_CANDIDATES" \
    --output "$LM_SCORES" \
    --summary "$LM_SCORE_SUMMARY" \
    --scorer "$SCORER" \
    --model-name "$CHEMFORMER_MODEL_NAME" \
    --model-path "$CHEMFORMER_MODEL_PATH" \
    --vocabulary-path "$CHEMFORMER_VOCABULARY_PATH" \
    --root "$ROOT" \
    --reaction-lm-python "$REACTION_LM_PYTHON" \
    --batch-size "$BATCH_SIZE" \
    --device cuda \
    > "$LOG_DIR/external_product_prediction_25k_lm_score.log" 2>&1
fi

MODEL_ARGS=()
for model_dir in $PC_CNG_MODEL_DIRS; do
  if [[ -d "$model_dir" ]]; then
    MODEL_ARGS+=(--model-dir "$model_dir")
  else
    echo "[warn] missing PC-CNG model dir: $model_dir" >&2
  fi
done
if [[ ${#MODEL_ARGS[@]} -eq 0 ]]; then
  echo "[error] no usable PC-CNG model dirs; set PC_CNG_MODEL_DIRS" >&2
  exit 2
fi

INVALID_NEGATIVE_ARGS=()
if [[ -n "$PC_CNG_INVALID_NEGATIVE_SCORE" ]]; then
  INVALID_NEGATIVE_ARGS=(--pc-cng-invalid-negative-score "$PC_CNG_INVALID_NEGATIVE_SCORE")
fi

echo "[benchmark] 25k strict external product prediction comparison"
CUDA_VISIBLE_DEVICES="$GPU_EVAL" PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.evaluate_external_product_prediction_benchmark \
  --candidate-csv "$FULL_CANDIDATES" \
  --external-score "chemformer_likelihood=$LM_SCORES:lm_score" \
  --primary-external-score chemformer_likelihood \
  "${MODEL_ARGS[@]}" \
  --output-dir "$BENCHMARK_DIR" \
  --batch-size "$PC_CNG_BATCH_SIZE" \
  --device cuda \
  --normalization group_zscore \
  "${INVALID_NEGATIVE_ARGS[@]}" \
  > "$LOG_DIR/external_product_prediction_25k_benchmark.log" 2>&1

echo "[benchmark] 25k validity-aware external product prediction comparison"
CUDA_VISIBLE_DEVICES="$GPU_EVAL" PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.evaluate_external_product_prediction_benchmark \
  --candidate-csv "$FULL_CANDIDATES" \
  --external-score "chemformer_likelihood=$LM_SCORES:lm_score" \
  --primary-external-score chemformer_likelihood \
  "${MODEL_ARGS[@]}" \
  --output-dir "$VALIDITY_DIR" \
  --batch-size "$PC_CNG_BATCH_SIZE" \
  --device cuda \
  --normalization group_zscore \
  --allow-incomplete-groups \
  "${INVALID_NEGATIVE_ARGS[@]}" \
  > "$LOG_DIR/external_product_prediction_25k_validity_aware.log" 2>&1

echo "25k external product prediction benchmark complete: $RESULTS_DIR"
