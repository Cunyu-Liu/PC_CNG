"""Diagnose why zero learned_structured candidates appear in Phase 4 pools."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "chem_negative_sampling"))

import pc_cng.run_phase4_fixed_testset as p4
from pc_cng.run_phase3_external_validation import (
    NegativeGenerator, METHOD_LEARNED, load_g8c_model,
)
from pc_cng.p4_g8c_learned_structured_proposal import (
    generate_structured_proposal, _apply_structured_edit,
)
import pandas as pd

REPO = Path(__file__).resolve().parent

model, err = load_g8c_model(REPO / "results/p4_g8c_phase2_full/model_checkpoint.pt",
                            device=None)
print("model:", "OK" if model is not None else err)
gen = NegativeGenerator(METHOD_LEARNED, model=model, top_k=8, device=None, seed=1)
print("gen.model:", type(gen.model).__name__, "gen.device:", gen.device)

df = pd.read_parquet(REPO / "data/processed/p4_hte_normalized.parquet")
rows = df.head(20)

ok = fail = 0
for _, row in rows.iterrows():
    rxn = row["reaction_smiles"]
    if not isinstance(rxn, str):
        continue
    parts = rxn.split(">")
    if len(parts) != 3:
        continue
    try:
        edits = generate_structured_proposal(
            gen.model, rxn, top_k=8, device=gen.device,
            use_validity_mask=True, risk_rerank=False)
        n_added = 0
        for edit in edits or []:
            edited = _apply_structured_edit(rxn, edit)
            if edited and isinstance(edited, str):
                n_added += 1
        ok += 1
        if ok <= 3:
            print(f"rxn ok: n_edits={len(edits or [])} applied={n_added}")
    except Exception:
        fail += 1
        if fail <= 3:
            import traceback
            traceback.print_exc()
print(f"direct block: ok={ok} fail={fail}")

# Now the real generate_union_candidates (learned only, no rule)
cands = p4.generate_union_candidates(
    rows, None, gen, ["CCO", "c1ccccc1"], seed=1, n_rule=8, n_learned=8)
srcs = {}
for rxn, cl in cands.items():
    for c in cl:
        srcs[c["source"]] = srcs.get(c["source"], 0) + 1
print("union candidate sources:", srcs)
pools = {}
for rxn, cl in cands.items():
    for c in cl:
        if c["source"] == "learned_structured":
            pools[c["pool"]] = pools.get(c["pool"], 0) + 1
print("learned candidates by pool:", pools)
