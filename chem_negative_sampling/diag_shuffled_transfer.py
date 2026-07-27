"""Diagnose WHY shuffled_parent transfers to boundary negatives (Phase 4 H2).

Hypothesis A (composition shortcut): the shuffled_parent-trained classifier
detects boundary negatives only when the edit changes the molecular formula
(atom transmutation).  On formula-PRESERVING edits (bond-order change /
formed-bond migration) its AUPRC should collapse to ~0.5.

Hypothesis B (plausibility transfer): the classifier learned a general
reactant-product compatibility prior; it detects boundary negatives even
when composition is preserved.  H2's premise would then be wrong and
shuffled_parent is a genuinely strong baseline.

Reads ``per_scenario_records/{scenario}__shuffled_parent__semi_hard.csv``
and reports shuffled-classifier AUPRC stratified by formula preservation,
with per-source counts.

Run:
    python3 diag_shuffled_transfer.py results/phase4_fixed_testset_v41
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    _RDKIT_OK = True
except Exception:  # pragma: no cover
    _RDKIT_OK = False


def _formula(smiles: str):
    if not _RDKIT_OK or not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return rdMolDescriptors.CalcMolFormula(mol)


def _product(rxn: str) -> str:
    parts = rxn.split(">")
    if len(parts) == 3:
        return parts[2]
    if len(parts) == 2:
        return parts[1]
    return ""


def _reactants(rxn: str) -> str:
    return rxn.split(">")[0]


def _auprc(labels, scores) -> float:
    """Rank-based AUPRC (average precision) without sklearn dependency."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    n_pos = int(labels.sum())
    if n_pos == 0 or n_pos == len(labels):
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    lab = labels[order]
    tp = np.cumsum(lab)
    prec = tp / (np.arange(len(lab)) + 1)
    return float((prec * lab).sum() / n_pos)


def _is_pos(v) -> bool:
    return v in ("True", "1", True, 1)


def main(root: str) -> None:
    rec_dir = Path(root) / "per_scenario_records"
    files = sorted(rec_dir.glob("*__shuffled_parent__semi_hard.csv"))
    if not files:
        print(f"no shuffled_parent semi_hard records under {rec_dir}")
        sys.exit(1)

    hdr = (f"{'scenario':<16} {'subset':<24} {'n_pos':>5} {'n_neg':>5} "
           f"{'AUPRC':>7}  neg_sources")
    print(hdr)
    overall_neg = defaultdict(list)   # subset -> neg scores
    overall_pos = []                  # pos scores (pooled)
    for f in files:
        scenario = f.name.split("__")[0]
        rows = list(csv.DictReader(open(f)))

        # reactants -> unique true product (from positive records)
        true_by_reactants = defaultdict(set)
        for r in rows:
            if _is_pos(r.get("is_positive")):
                true_by_reactants[_reactants(r["reaction_smiles"])].add(
                    _product(r["reaction_smiles"]))

        pos_scores, neg = [], defaultdict(list)   # subset -> [(score, source)]
        for r in rows:
            score = float(r["score"])
            if _is_pos(r.get("is_positive")):
                pos_scores.append(score)
                continue
            rxn = r["reaction_smiles"]
            trues = true_by_reactants.get(_reactants(rxn), set())
            src = r.get("source", "?")
            if len(trues) != 1:
                neg["neg_unpaired"].append((score, src))
                continue
            true_prod = next(iter(trues))
            f_neg, f_true = _formula(_product(rxn)), _formula(true_prod)
            if f_neg is None or f_true is None:
                neg["neg_formula_unknown"].append((score, src))
            elif f_neg == f_true:
                neg["formula_preserved"].append((score, src))
            else:
                neg["formula_changed"].append((score, src))

        n_pos = len(pos_scores)
        overall_pos.extend(pos_scores)
        for subset in ("formula_preserved", "formula_changed"):
            entries = neg.get(subset, [])
            if not entries:
                continue
            scores = [s for s, _ in entries]
            srcs = defaultdict(int)
            for _, s in entries:
                srcs[s] += 1
            labels = [1] * n_pos + [0] * len(scores)
            auprc = _auprc(labels, pos_scores + scores)
            overall_neg[subset].extend(scores)
            src_str = ",".join(f"{k}:{v}" for k, v in sorted(srcs.items()))
            print(f"{scenario:<16} {subset:<24} {n_pos:>5} {len(scores):>5} "
                  f"{auprc:>7.4f}  {src_str}")

    print("\n=== overall (pooled across scenarios) ===")
    n_pos = len(overall_pos)
    for subset in ("formula_preserved", "formula_changed"):
        scores = overall_neg.get(subset, [])
        if not scores:
            continue
        labels = [1] * n_pos + [0] * len(scores)
        auprc = _auprc(labels, overall_pos + scores)
        print(f"{'ALL':<16} {subset:<24} {n_pos:>5} {len(scores):>5} "
              f"{auprc:>7.4f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else "results/phase4_fixed_testset_v41")
