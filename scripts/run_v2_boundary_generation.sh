#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/cunyuliu/pc_cng_research
PY=/home/cunyuliu/miniconda3/envs/pc_cng_gpu/bin/python
cd "$ROOT/chem_negative_sampling"
OUT=$ROOT/results/v2_boundary_generation
mkdir -p "$OUT" "$ROOT/results/logs"
run_one() {
  name=$1
  input=$2
  output=$OUT/${name}_boundary_negatives.csv
  summary=$OUT/${name}_boundary_summary.json
  log=$ROOT/results/logs/v2_boundary_${name}.log
  if [ -f "$summary" ]; then echo "$name already complete"; return 0; fi
  echo "starting $name"
  PYTHONPATH=. nohup "$PY" -m pc_cng.run_boundary_generation \
    --input "$input" \
    --output "$output" \
    --summary "$summary" \
    --max-candidates-per-reaction 3 \
    > "$log" 2>&1 &
  echo "$!" > "$OUT/${name}.pid"
}
run_one hitea_full "$ROOT/data/processed/hitea_full_positives.csv"
run_one regiosqm20 "$ROOT/data/processed/regiosqm20_positives.csv"
