#!/usr/bin/env bash
set -euo pipefail

# Safe public-data preparation entry. This script does not delete data.
# Provide dataset CSV paths through environment variables when available:
#   REGIOSQM_CSV=/path/regiosqm.csv
#   HITEA_CSV=/path/hitea.csv
#   USPTO_CSV=/path/uspto.csv
#   ORD_CSV=/path/ord.csv

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data}"
PROCESSED_DIR="$DATA_ROOT/processed"
SUMMARY_DIR="$DATA_ROOT/summaries"

mkdir -p "$DATA_ROOT/raw" "$PROCESSED_DIR" "$SUMMARY_DIR"

normalize_if_present() {
  local name="$1"
  local path="$2"
  if [[ -n "$path" && -f "$path" ]]; then
    echo "[prepare] Normalizing $name from $path"
    python3 -m pc_cng.data_ingestion \
      --input "$path" \
      --output "$PROCESSED_DIR/${name}_normalized.csv" \
      --summary "$SUMMARY_DIR/${name}_summary.json" \
      --source-name "$name"
  else
    echo "[prepare] Skipping $name; file not provided or not found"
  fi
}

normalize_if_present "regiosqm20" "${REGIOSQM_CSV:-}"
normalize_if_present "hitea" "${HITEA_CSV:-}"
normalize_if_present "uspto" "${USPTO_CSV:-}"
normalize_if_present "ord" "${ORD_CSV:-}"

cat > "$SUMMARY_DIR/README_public_data.md" <<'EOF'
# Public Data Preparation

Expected normalized schema:

- source_id
- reaction_smiles
- reactants
- agents
- products
- label_type
- yield
- source
- split_key
- split

Notes:

- `label_type=synthetic_negative` is never assigned here.
- Unverified unknown reactions must not be treated as real negatives.
- For publishable results, store raw-data provenance and license details.
EOF

echo "[prepare] Done. Processed files are in $PROCESSED_DIR"

