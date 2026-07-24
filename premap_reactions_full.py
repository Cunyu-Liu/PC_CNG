"""Pre-map USPTO reactions with RXNMapper for G8-C, with robust error handling."""
import csv
import json
import sys
from pathlib import Path

# Read reactions from USPTO CSV
rxns = []
with open("data/processed/uspto_openmolecules_normalized.csv") as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        if i >= 500:
            break
        r = row["reaction_smiles"].strip()
        if not r or ">>" not in r:
            continue
        # Skip if reactants or products are empty
        parts = r.split(">>")
        if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
            continue
        rxns.append(r)
print(f"Read {len(rxns)} valid reactions", file=sys.stderr)

# Batch map with RXNMapper, processing in chunks to handle failures
from rxnmapper import RXNMapper
mapper = RXNMapper()
print("RXNMapper loaded, mapping in chunks...", file=sys.stderr)

mapped = []
chunk_size = 50
for start in range(0, len(rxns), chunk_size):
    chunk = rxns[start:start + chunk_size]
    try:
        results = mapper.get_attention_guided_atom_maps(chunk)
        for r in results:
            m = r.get("mapped_rxn", "")
            if m and ">" in m:
                mapped.append(m)
        print(f"  Chunk {start//chunk_size + 1}: {len(mapped)}/{len(rxns)} mapped", file=sys.stderr)
    except Exception as e:
        print(f"  Chunk {start//chunk_size + 1} failed: {e}", file=sys.stderr)
        # Try individual reactions in this chunk
        for rxn in chunk:
            try:
                results = mapper.get_attention_guided_atom_maps([rxn])
                m = results[0].get("mapped_rxn", "")
                if m and ">" in m:
                    mapped.append(m)
            except Exception:
                continue

print(f"Mapped {len(mapped)} reactions (out of {len(rxns)})", file=sys.stderr)

# Save to JSON
out = Path("data/p4/g8c_full_mapped_reactions.json")
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    json.dump(mapped, f)
print(f"Saved to {out}", file=sys.stderr)
