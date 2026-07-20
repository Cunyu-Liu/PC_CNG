#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python3}
WORKDIR=${WORKDIR:-results/hard_negative_actions_smoke}
mkdir -p "$WORKDIR"

INPUT="$WORKDIR/action_input.csv"
OUTPUT="$WORKDIR/hard_negative_actions.csv"
SUMMARY="$WORKDIR/summary.json"
REVIEWED="$WORKDIR/hard_negative_actions_reviewed.csv"
REVIEW_SUMMARY="$WORKDIR/review_summary.json"

cat > "$INPUT" <<'CSV'
source_id,reaction_smiles,label_type,yield,source,split
alkyl_001,COS(=O)(=O)O[CH3:12].[F:1][c:2]1[cH:3][c:4]([Br:5])[cH:6][c:7]2[cH:8][n:9][nH:10][c:11]12>>[F:1][c:2]1[cH:3][c:4]([Br:5])[cH:6][c:7]2[cH:8][n:9][n:10]([CH3:12])[c:11]12,positive,20,smoke,train
taut_001,CC(=O)C>>CC(=O)C,positive,70,smoke,train
low_yield_001,CCBr.N>>CCN,real_negative,0,smoke,train
CSV

PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.run_hard_negative_actions \
  --input "$INPUT" \
  --low-yield-input "$INPUT" \
  --known-positive "$INPUT" \
  --output "$OUTPUT" \
  --summary "$SUMMARY" \
  --action heteroatom \
  --action regio \
  --action tautomer \
  --action low_yield_seed \
  --max-candidates-per-reaction 8 \
  --max-candidates-per-pair 12 \
  --map-unmapped

PYTHONPATH=. "$PYTHON_BIN" -m pc_cng.false_negative_review \
  --input "$OUTPUT" \
  --output "$REVIEWED" \
  --summary "$REVIEW_SUMMARY" \
  --known-positive "$INPUT"

"$PYTHON_BIN" - <<'PY' "$SUMMARY" "$OUTPUT"
import csv
import json
import sys

summary_path, output_path = sys.argv[1:3]
summary = json.load(open(summary_path))
counts = summary["counts"]
required = ["heteroatom", "regio", "tautomer", "low_yield_seed"]
missing = [name for name in required if counts.get(name, 0) <= 0]
if missing:
    raise SystemExit(f"Missing expected action outputs: {missing}; counts={counts}")

with open(output_path, newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
if not rows:
    raise SystemExit("No hard-negative action rows were generated")
print(json.dumps({"counts": counts, "rows": len(rows)}, indent=2))
PY

echo "Hard-negative action smoke test passed."
