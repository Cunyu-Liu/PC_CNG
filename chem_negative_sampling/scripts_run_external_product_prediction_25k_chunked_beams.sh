#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/cunyuliu/pc_cng_research}
PYTHON_BIN=${PYTHON_BIN:-/home/cunyuliu/miniconda3/envs/pc_cng_gpu/bin/python}
REACTION_LM_PYTHON=${REACTION_LM_PYTHON:-$ROOT/envs/reaction_lm/bin/python}

CHEMFORMER_ROOT=${CHEMFORMER_ROOT:-$ROOT/external/reaction_lm/Chemformer}
CHEMFORMER_MODEL_PATH=${CHEMFORMER_MODEL_PATH:-$ROOT/models/reaction_lm/chemformer_forward_uspto50k/model_sanitized.ckpt}
CHEMFORMER_VOCABULARY_PATH=${CHEMFORMER_VOCABULARY_PATH:-$CHEMFORMER_ROOT/bart_vocab.json}

RESULTS_DIR=${RESULTS_DIR:-$ROOT/results/external_product_prediction_benchmark_25k_20260713}
CHUNK_MANIFEST=${CHUNK_MANIFEST:-$RESULTS_DIR/chemformer_input_chunks/chemformer_forward_input_25k_chunks_manifest.json}
OUTPUT_DIR=${OUTPUT_DIR:-$RESULTS_DIR/chemformer_beam_chunks}
MERGED_BEAM_CSV=${MERGED_BEAM_CSV:-$RESULTS_DIR/chemformer_forward_beams.tsv}
STATUS_JSON=${STATUS_JSON:-$OUTPUT_DIR/chemformer_forward_beam_chunks_status.json}
LOG_DIR=${LOG_DIR:-$ROOT/results/logs}

GPU_EVAL=${GPU_EVAL:-1}
GPU_MEM_LIMIT_MB=${GPU_MEM_LIMIT_MB:-2500}
GPU_UTIL_LIMIT_PCT=${GPU_UTIL_LIMIT_PCT:-10}
GPU_MAX_COMPUTE_APPS=${GPU_MAX_COMPUTE_APPS:-0}
POLL_SECONDS=${POLL_SECONDS:-300}
N_BEAMS=${N_BEAMS:-10}
BATCH_SIZE=${BATCH_SIZE:-128}
DATA_DEVICE=${DATA_DEVICE:-cuda}
DRY_RUN=${DRY_RUN:-0}
MERGE_ONLY=${MERGE_ONLY:-0}
START_CHUNK=${START_CHUNK:-0}
STOP_CHUNK=${STOP_CHUNK:-}

BEAM_PREFIX=${BEAM_PREFIX:-chemformer_forward_beams_25k}

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

if [[ ! -f "$CHUNK_MANIFEST" ]]; then
  echo "[error] missing chunk manifest: $CHUNK_MANIFEST" >&2
  exit 2
fi

if [[ "$DRY_RUN" != "1" && "$MERGE_ONLY" != "1" ]]; then
  for required in "$REACTION_LM_PYTHON" "$CHEMFORMER_MODEL_PATH" "$CHEMFORMER_VOCABULARY_PATH"; do
    if [[ ! -f "$required" ]]; then
      echo "[error] missing required runtime file: $required" >&2
      exit 2
    fi
  done
fi

chunk_beam_path() {
  local idx="$1"
  printf '%s/%s_chunk_%04d.tsv' "$OUTPUT_DIR" "$BEAM_PREFIX" "$idx"
}

write_status() {
  "$PYTHON_BIN" - "$CHUNK_MANIFEST" "$OUTPUT_DIR" "$BEAM_PREFIX" "$MERGED_BEAM_CSV" "$STATUS_JSON" <<'PY'
import json
import os
import sys
from pathlib import Path

manifest_path, output_dir, prefix, merged_path, status_path = sys.argv[1:6]
manifest = json.loads(Path(manifest_path).read_text())
chunks = []
complete = 0
for chunk in manifest["chunks"]:
    idx = int(chunk["chunk_index"])
    expected_rows = int(chunk["rows"])
    beam_path = os.path.join(output_dir, f"{prefix}_chunk_{idx:04d}.tsv")
    exists = os.path.exists(beam_path)
    line_count = 0
    if exists:
        with open(beam_path, encoding="utf-8") as handle:
            line_count = sum(1 for _ in handle)
    valid = exists and line_count == expected_rows + 1
    complete += int(valid)
    chunks.append(
        {
            "chunk_index": idx,
            "input_path": chunk["path"],
            "expected_rows": expected_rows,
            "beam_path": beam_path,
            "beam_exists": exists,
            "beam_line_count": line_count,
            "beam_valid": valid,
        }
    )
payload = {
    "chunk_manifest": manifest_path,
    "output_dir": output_dir,
    "merged_beam_csv": merged_path,
    "chunk_count": len(chunks),
    "complete_chunks": complete,
    "all_chunks_complete": complete == len(chunks),
    "merged_beam_exists": os.path.exists(merged_path),
    "chunks": chunks,
}
os.makedirs(os.path.dirname(status_path), exist_ok=True)
with open(status_path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, ensure_ascii=False)
    handle.write("\n")
print(json.dumps(payload, indent=2, ensure_ascii=False))
PY
}

chunk_valid() {
  local idx="$1"
  local rows="$2"
  local beam_path
  beam_path="$(chunk_beam_path "$idx")"
  [[ -f "$beam_path" ]] || return 1
  local lines
  lines="$(wc -l < "$beam_path" | tr -d ' ')"
  [[ "$lines" == "$((rows + 1))" ]]
}

wait_for_gpu() {
  while true; do
    local raw mem util apps
    raw="$(nvidia-smi -i "$GPU_EVAL" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
    mem="${raw%%,*}"
    util="${raw##*,}"
    if [[ "$util" == "[N/A]" || -z "$util" ]]; then
      util=100
    fi
    apps="$(nvidia-smi -i "$GPU_EVAL" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^$/d' | wc -l | tr -d ' ')"
    echo "[$(date '+%F_%T')] gpu${GPU_EVAL} mem=${mem}MiB util=${util}% compute_apps=${apps}"
    if [[ "$mem" =~ ^[0-9]+$ && "$util" =~ ^[0-9]+$ && "$apps" =~ ^[0-9]+$ && "$mem" -le "$GPU_MEM_LIMIT_MB" && "$util" -le "$GPU_UTIL_LIMIT_PCT" && "$apps" -le "$GPU_MAX_COMPUTE_APPS" ]]; then
      break
    fi
    sleep "$POLL_SECONDS"
  done
}

merge_chunks() {
  local tmp="${MERGED_BEAM_CSV}.tmp"
  rm -f "$tmp"
  local first=1
  while IFS=$'\t' read -r idx path rows; do
    local beam_path
    beam_path="$(chunk_beam_path "$idx")"
    if ! chunk_valid "$idx" "$rows"; then
      echo "[info] cannot merge yet; invalid/missing chunk $idx: $beam_path" >&2
      return 0
    fi
    if [[ "$first" == "1" ]]; then
      cat "$beam_path" >> "$tmp"
      first=0
    else
      tail -n +2 "$beam_path" >> "$tmp"
    fi
  done < <("$PYTHON_BIN" - "$CHUNK_MANIFEST" <<'PY'
import json
import sys
from pathlib import Path
data = json.loads(Path(sys.argv[1]).read_text())
for chunk in data["chunks"]:
    print(f"{int(chunk['chunk_index'])}\t{chunk['path']}\t{int(chunk['rows'])}")
PY
)
  mv "$tmp" "$MERGED_BEAM_CSV"
  echo "[merge] wrote $MERGED_BEAM_CSV"
}

if [[ "$DRY_RUN" == "1" ]]; then
  write_status
  exit 0
fi

if [[ "$MERGE_ONLY" == "1" ]]; then
  merge_chunks
  write_status
  exit 0
fi

while IFS=$'\t' read -r idx chunk_path rows; do
  if [[ "$idx" -lt "$START_CHUNK" ]]; then
    continue
  fi
  if [[ -n "$STOP_CHUNK" && "$idx" -gt "$STOP_CHUNK" ]]; then
    continue
  fi
  beam_path="$(chunk_beam_path "$idx")"
  log_path="$LOG_DIR/external_product_prediction_25k_beam_chunk_${idx}.log"
  if chunk_valid "$idx" "$rows"; then
    echo "[skip] chunk $idx already valid: $beam_path"
    continue
  fi
  echo "[wait] chunk $idx rows=$rows output=$beam_path"
  wait_for_gpu
  echo "[beam] chunk $idx on GPU $GPU_EVAL"
  predict_input="$OUTPUT_DIR/${BEAM_PREFIX}_chunk_${idx}_input_no_header.tsv"
  tail -n +2 "$chunk_path" > "$predict_input"
  CUDA_VISIBLE_DEVICES="$GPU_EVAL" PYTHONPATH="$CHEMFORMER_ROOT:${PYTHONPATH:-}" "$REACTION_LM_PYTHON" -m molbart.predict \
    data_path="$predict_input" \
    output_sampled_smiles="$beam_path" \
    model_path="$CHEMFORMER_MODEL_PATH" \
    vocabulary_path="$CHEMFORMER_VOCABULARY_PATH" \
    batch_size="$BATCH_SIZE" \
    n_beams="$N_BEAMS" \
    n_gpus=1 \
    data_device="$DATA_DEVICE" \
    dataset_part=full \
    task=forward_prediction \
    model_type=bart \
    train_mode=eval \
    > "$log_path" 2>&1
  if ! chunk_valid "$idx" "$rows"; then
    echo "[error] chunk $idx finished but output is missing or has unexpected line count" >&2
    exit 1
  fi
  write_status > /dev/null
done < <("$PYTHON_BIN" - "$CHUNK_MANIFEST" <<'PY'
import json
import sys
from pathlib import Path
data = json.loads(Path(sys.argv[1]).read_text())
for chunk in data["chunks"]:
    print(f"{int(chunk['chunk_index'])}\t{chunk['path']}\t{int(chunk['rows'])}")
PY
)

merge_chunks
write_status
