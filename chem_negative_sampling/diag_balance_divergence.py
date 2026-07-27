"""Diagnose the balance-check divergence between the exhaustive generator
and run_phase4_fixed_testset._mk_candidate.

For a handful of random-split rows, run the generator and print, per
candidate: applied_product, sim, d_true (generator ref vs raw ref),
d_neg, and both balance verdicts.
"""
import sys
from pathlib import Path

_ROOT = Path("/home/cunyuliu/pc_cng_research")
_CNS = _ROOT / "chem_negative_sampling"
sys.path.insert(0, str(_CNS))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from pc_cng.chem_utils import atom_count_distance  # noqa: E402
from pc_cng.run_phase3_external_validation import (  # noqa: E402
    DEFAULT_CHECKPOINT, DEFAULT_OOD_DIR, DEFAULT_PARQUET, load_g8c_model,
    load_hitea_split)
from pc_cng.run_phase4_fixed_testset import (  # noqa: E402
    BALANCE_DIST_SLACK, _product_of, _tanimoto)
from pc_cng.phase3_enhanced import morgan_fingerprint  # noqa: E402
from pc_cng.p4_g8c_learned_structured_proposal import (  # noqa: E402
    generate_structured_proposal_exhaustive, _strip_atom_maps)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, device = load_g8c_model(DEFAULT_CHECKPOINT, device)
    split = load_hitea_split(DEFAULT_PARQUET, DEFAULT_OOD_DIR, "random", 40, 12)
    rows = split["test"].head(5)
    for _, row in rows.iterrows():
        rxn = row["reaction_smiles"]
        reactants = rxn.split(">")[0]
        true_raw = _product_of(rxn)
        true_canon = _strip_atom_maps(true_raw)
        d_true_raw = atom_count_distance(reactants, true_raw)
        d_true_canon = atom_count_distance(reactants, true_canon)
        print(f"\n=== rxn: {rxn[:110]}")
        print(f"  true_raw  : {true_raw[:90]}")
        print(f"  true_canon: {true_canon[:90]}")
        print(f"  d_true_raw={d_true_raw}  d_true_canon={d_true_canon}")
        edits = generate_structured_proposal_exhaustive(
            model, rxn, top_k=8, device=device, use_validity_mask=True,
            risk_rerank=False, map_unmapped=True,
            require_atom_balance=True, balance_dist_slack=BALANCE_DIST_SLACK)
        print(f"  n_edits={len(edits)}")
        tfp = morgan_fingerprint(true_raw)
        for e in edits:
            ap = getattr(e, "applied_product", None)
            if not ap:
                continue
            sim = _tanimoto(morgan_fingerprint(ap), tfp)
            d_neg = atom_count_distance(reactants, ap)
            bal_raw = d_neg <= d_true_raw + BALANCE_DIST_SLACK
            bal_canon = d_neg <= d_true_canon + BALANCE_DIST_SLACK
            pool = ("easy" if sim < 0.40 else
                    "semi_hard" if sim <= 0.75 else "hard")
            print(f"    {e.edit_type.name:22s} sim={sim:.3f} pool={pool:9s} "
                  f"d_neg={d_neg} bal_raw={bal_raw} bal_canon={bal_canon} "
                  f"prod={ap[:70]}")


if __name__ == "__main__":
    main()
