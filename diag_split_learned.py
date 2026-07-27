"""Diagnose learned candidate attrition on the real author_lab split."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "chem_negative_sampling"))

import torch
from pc_cng.run_phase3_external_validation import (
    NegativeGenerator, METHOD_LEARNED, load_g8c_model, load_hitea_split,
)
from pc_cng.p4_g8c_learned_structured_proposal import (
    generate_structured_proposal, _apply_structured_edit,
)

REPO = Path(__file__).resolve().parent

device = torch.device("cuda:0") if torch.cuda.is_available() else None
print("device:", device)
model, err = load_g8c_model(REPO / "results/p4_g8c_phase2_full/model_checkpoint.pt",
                            device=device)
print("model:", "OK" if model is not None else err)

split_data = load_hitea_split(
    REPO / "data/processed/p4_hte_normalized.parquet",
    REPO / "data/ood_splits", "author_lab", 500, 300)
test_rows = split_data["test"]
print("test rows:", len(test_rows))

n_rows = n_parse_ok = n_gen_ok = n_edits = n_applied = 0
exc_types = {}
for _, row in test_rows.head(60).iterrows():
    rxn = row.get("reaction_smiles")
    if not rxn or not isinstance(rxn, str):
        continue
    n_rows += 1
    parts = rxn.split(">")
    if len(parts) != 3:
        continue
    n_parse_ok += 1
    try:
        edits = generate_structured_proposal(
            model, rxn, top_k=8, device=device,
            use_validity_mask=True, risk_rerank=False)
        n_gen_ok += 1
        n_edits += len(edits or [])
        for e in edits or []:
            edited = _apply_structured_edit(rxn, e)
            if edited and isinstance(edited, str):
                n_applied += 1
    except Exception as ex:
        exc_types[type(ex).__name__] = exc_types.get(type(ex).__name__, 0) + 1
        if sum(exc_types.values()) <= 2:
            import traceback
            traceback.print_exc()

print(f"rows={n_rows} parse_ok={n_parse_ok} gen_ok={n_gen_ok} "
      f"edits={n_edits} applied={n_applied} exceptions={exc_types}")

# sample a reaction to inspect format
rxn0 = test_rows.iloc[0]["reaction_smiles"]
print("sample rxn:", rxn0[:120])
