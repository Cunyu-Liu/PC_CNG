"""Pre-map a small batch of USPTO reactions with RXNMapper for G8-C smoke test."""
import csv
import json
import sys
from pathlib import Path

# Read 80 reactions from USPTO CSV
rxns = []
with open("data/processed/uspto_openmolecules_normalized.csv") as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        if i >= 80:
            break
        rxns.append(row["reaction_smiles"])
print(f"Read {len(rxns)} reactions", file=sys.stderr)

# Batch map with RXNMapper
from rxnmapper import RXNMapper
mapper = RXNMapper()
print("RXNMapper loaded, mapping batch...", file=sys.stderr)
results = mapper.get_attention_guided_atom_maps(rxns)
mapped = [r["mapped_rxn"] for r in results if r.get("mapped_rxn")]
print(f"Mapped {len(mapped)} reactions (out of {len(rxns)})", file=sys.stderr)

# Save to JSON
out = Path("data/p4/g8c_smoke_mapped_reactions.json")
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    json.dump(mapped, f)
print(f"Saved to {out}", file=sys.stderr)
