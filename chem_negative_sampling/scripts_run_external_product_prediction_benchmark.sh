#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/cunyuliu/pc_cng_research}
PYTHON_BIN=${PYTHON_BIN:-/home/cunyuliu/miniconda3/envs/pc_cng_gpu/bin/python}
REACTION_LM_PYTHON=${REACTION_LM_PYTHON:-$ROOT/envs/reaction_lm/bin/python}
GPU_EVAL=${GPU_EVAL:-0}

CODE_DIR="$ROOT/chem_negative_sampling"
LOG_DIR=${LOG_DIR:-$ROOT/results/logs}
RESULTS_DIR=${RESULTS_DIR:-$ROOT/results/external_product_prediction_benchmark}

REGIO_ALIGNMENT=${REGIO_ALIGNMENT:-$ROOT/data/processed/regiosqm20_normalized.csv}
HITEA_ALIGNMENT=${HITEA_ALIGNMENT:-$ROOT/data/processed/hitea_full_normalized.csv}
SYNTHETIC_CSV=${SYNTHETIC_CSV:-$ROOT/results/type1_diverse_anchor_full/diverse_anchor_candidates_reviewed.csv}
PC_CNG_MODEL_DIRS=${PC_CNG_MODEL_DIRS:-$ROOT/results/type1_diverse_anchor_ablation_full/pairwise_default_seed20260710}

CHEMFORMER_ROOT=${CHEMFORMER_ROOT:-$ROOT/external/reaction_lm/Chemformer}
CHEMFORMER_MODEL_PATH=${CHEMFORMER_MODEL_PATH:-$ROOT/models/reaction_lm/chemformer_forward_uspto50k/model_sanitized.ckpt}
CHEMFORMER_VOCABULARY_PATH=${CHEMFORMER_VOCABULARY_PATH:-$CHEMFORMER_ROOT/bart_vocab.json}
CHEMFORMER_MODEL_NAME=${CHEMFORMER_MODEL_NAME:-chemformer_forward_uspto50k}
SCORER=${SCORER:-chemformer_log_likelihood}
N_BEAMS=${N_BEAMS:-10}
BATCH_SIZE=${BATCH_SIZE:-128}
PC_CNG_BATCH_SIZE=${PC_CNG_BATCH_SIZE:-4096}
GENERATE_BEAMS=${GENERATE_BEAMS:-1}
PC_CNG_INVALID_NEGATIVE_SCORE=${PC_CNG_INVALID_NEGATIVE_SCORE:-}

BASE_CANDIDATES="$RESULTS_DIR/base_observed_pc_cng_candidates.csv"
BASE_SUMMARY="$RESULTS_DIR/base_observed_pc_cng_candidates_summary.json"
CONTEXTS_CSV="$RESULTS_DIR/product_prediction_contexts.csv"
CHEMFORMER_INPUT="$RESULTS_DIR/chemformer_forward_input.csv"
BEAM_CSV=${BEAM_CSV:-$RESULTS_DIR/chemformer_forward_beams.tsv}
FULL_CANDIDATES="$RESULTS_DIR/full_observed_pc_cng_chemformer_beam_candidates.csv"
FULL_SUMMARY="$RESULTS_DIR/full_observed_pc_cng_chemformer_beam_candidates_summary.json"
LM_SCORES="$RESULTS_DIR/lm_scores_${SCORER}.csv"
LM_SCORE_SUMMARY="$RESULTS_DIR/lm_scores_${SCORER}_summary.json"
BENCHMARK_DIR="$RESULTS_DIR/benchmark"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"
cd "$CODE_DIR"

for required in "$REGIO_ALIGNMENT" "$HITEA_ALIGNMENT" "$SYNTHETIC_CSV"; do
  if [[ ! -f "$required" ]]; then
    echo "[error] missing required input: $required" >&2
    exit 2
  fi
done

if [[ ! -f "$BASE_SUMMARY" ]]; then
  echo "[build] observed + PC-CNG candidate set"
  PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.build_external_product_prediction_candidate_set \
    --real-csv "$REGIO_ALIGNMENT" \
    --real-csv "$HITEA_ALIGNMENT" \
    --synthetic-csv "$SYNTHETIC_CSV" \
    --output "$BASE_CANDIDATES" \
    --summary "$BASE_SUMMARY" \
    --contexts-output "$CONTEXTS_CSV" \
    --chemformer-input-output "$CHEMFORMER_INPUT" \
    > "$LOG_DIR/external_product_prediction_build_base.log" 2>&1
fi

if [[ "$GENERATE_BEAMS" == "1" && ! -f "$BEAM_CSV" ]]; then
  if [[ ! -f "$CHEMFORMER_MODEL_PATH" || ! -f "$CHEMFORMER_VOCABULARY_PATH" ]]; then
    echo "[error] Chemformer model/vocabulary missing; set BEAM_CSV to an existing prediction file or provide Chemformer paths." >&2
    exit 2
  fi
  echo "[beam] Chemformer forward beams"
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
    > "$LOG_DIR/external_product_prediction_chemformer_beams.log" 2>&1
fi

BEAM_ARGS=()
if [[ -f "$BEAM_CSV" ]]; then
  BEAM_ARGS=(--external-beam-csv "$BEAM_CSV" --beam-context-csv "$CONTEXTS_CSV")
else
  echo "[warn] no beam CSV found; benchmark will use observed positives + PC-CNG candidates only" >&2
fi

if [[ ! -f "$FULL_SUMMARY" ]]; then
  echo "[build] observed + PC-CNG + external beams"
  PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.build_external_product_prediction_candidate_set \
    --real-csv "$REGIO_ALIGNMENT" \
    --real-csv "$HITEA_ALIGNMENT" \
    --synthetic-csv "$SYNTHETIC_CSV" \
    "${BEAM_ARGS[@]}" \
    --external-model-name "$CHEMFORMER_MODEL_NAME" \
    --n-beams "$N_BEAMS" \
    --output "$FULL_CANDIDATES" \
    --summary "$FULL_SUMMARY" \
    > "$LOG_DIR/external_product_prediction_build_full.log" 2>&1
fi

if [[ ! -f "$LM_SCORE_SUMMARY" ]]; then
  echo "[score] $SCORER"
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
    > "$LOG_DIR/external_product_prediction_lm_score.log" 2>&1
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

echo "[benchmark] strict external product prediction comparison"
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
  > "$LOG_DIR/external_product_prediction_benchmark.log" 2>&1

echo "External product prediction benchmark complete: $BENCHMARK_DIR"
