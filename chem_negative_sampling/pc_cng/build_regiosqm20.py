"""Build RegioSQM20 positives/negatives without importing repo_utils.

The official negative_learning CLI imports pytorch_lightning via repo_utils even
for data extraction. This lightweight builder reuses the public dictionaries and
generator classes from the cloned repository while avoiding that training stack.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--negative-learning-src", required=True, help="Path to negative_learning/src")
    parser.add_argument("--compounds-smiles", required=True, help="RegioSQM20 compounds.smiles file")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    sys.path.insert(0, args.negative_learning_src)

    from rxn.chemutils.reaction_equation import ReactionEquation, rxn_standardization
    from rxn_negative_learning.data_generation.help_dicts_regiosqm import (
        COMPOUNDS_HALO_REACTANTS_DICT,
        HALO_REACTANT_SMILES_DICT,
    )
    from rxn_negative_learning.data_generation.negative_reactions_generator import (
        PosNegRxnGenerator,
        RegioSQMdatum,
    )

    compounds: Dict[str, Dict[str, str]] = {}
    with open(args.compounds_smiles, encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            name, smiles, centers = parts[0], parts[1], parts[2]
            compounds[name] = {"smiles": smiles, "centers": centers.rstrip(",")}

    rows: List[Dict[str, object]] = []
    idx = 0
    missing = 0
    for name, matches in COMPOUNDS_HALO_REACTANTS_DICT.items():
        if name not in compounds:
            missing += 1
            continue
        main_reactant = compounds[name]["smiles"]
        reaction_centers = [int(pos.strip()) for pos in compounds[name]["centers"].split(",") if pos.strip()]
        halo_reactants = []
        for match in matches:
            halo_reactants.extend(HALO_REACTANT_SMILES_DICT.get(match.strip(), []))
        halo_reactants = sorted(set(halo_reactants))
        if not halo_reactants:
            missing += 1
            continue

        generator = PosNegRxnGenerator(RegioSQMdatum(name, main_reactant, reaction_centers, halo_reactants))
        pos_reactions, pos_targets, neg_targets = generator()
        for pos_rxn, pos_tgt, neg_tgts in zip(pos_reactions, pos_targets, neg_targets):
            positive_rxn = standardize(f"{pos_rxn}>>{pos_tgt}", rxn_standardization, ReactionEquation)
            if positive_rxn:
                rows.append({"idx": idx, "rxn": positive_rxn, "score": 1, "name": name})
            if neg_tgts:
                for neg_tgt in neg_tgts:
                    negative_rxn = standardize(f"{pos_rxn}>>{neg_tgt}", rxn_standardization, ReactionEquation)
                    if negative_rxn:
                        rows.append({"idx": idx, "rxn": negative_rxn, "score": 0, "name": name})
            idx += 1

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["idx", "rxn", "score", "name"])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "output": args.output,
        "rows": len(rows),
        "positives": sum(1 for row in rows if row["score"] == 1),
        "negatives": sum(1 for row in rows if row["score"] == 0),
        "missing_compounds_or_halo_reactants": missing,
    }
    os.makedirs(os.path.dirname(args.summary), exist_ok=True)
    with open(args.summary, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def standardize(raw_rxn: str, rxn_standardization, ReactionEquation) -> str:
    try:
        return rxn_standardization(ReactionEquation.from_string(raw_rxn)).to_string()
    except Exception:
        return raw_rxn


if __name__ == "__main__":
    main()

