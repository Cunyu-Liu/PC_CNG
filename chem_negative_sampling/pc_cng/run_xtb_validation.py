"""P1-10: xTB / DFT / MMFF94 computational validation of synthetic negatives.

Validates that high-confidence synthetic negatives are thermodynamically
infeasible by computing reaction free energies (DeltaG) and rough barrier
estimates using RDKit's MMFF94 force field (xTB / DFT degraded path).

Usage::

    python3 -m pc_cng.run_xtb_validation \
        --candidates results/false_negative_three_layer_20260719/high_confidence_negatives.csv \
        --limit 100 \
        --output-dir results/xtb_dft_validation_20260719 \
        --method mmff94

The runner samples ``--limit`` high-value synthetic negatives (sorted by
``feasibility_score = 1 - hard_score`` ascending, i.e. most confident
negatives first), restricts to rows where the candidate reaction actually
involves a chemical change (``candidate_reactants != candidate_product``),
computes MMFF94 (or UFF fallback) energies for reactants and products, and
reports the fraction whose DeltaG > +5 kcal/mol (thermodynamic support for
"low feasibility").  A paired 10-seed bootstrap significance test compares
synthetic-negative DeltaG against the observed-positive DeltaG from the
``positive_reaction`` column of the same rows.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdForceFieldHelpers import (
    MMFFGetMoleculeForceField,
    MMFFGetMoleculeProperties,
    MMFFHasAllMoleculeParams,
    MMFFOptimizeMolecule,
)

# --------------------------------------------------------------------------- #
# Constants (judgment rule from Section 22.1 P1-10)
# --------------------------------------------------------------------------- #
DELTA_G_SUPPORT_THRESHOLD = 5.0  # kcal/mol  (DeltaG > +5  => supports low feasibility)
BARRIER_SUPPORT_THRESHOLD = 25.0  # kcal/mol  (barrier  > 25 => supports low feasibility)
BARRIER_FORCE_FIELD_CONSTANT = 5.0  # kcal/mol  (rough kinetic contribution added to |DeltaG|)
DEFAULT_RANDOM_SEED = 20260719
DEFAULT_NUM_SEEDS = 10
GO_NO_GO_THRESHOLD = 0.60  # support rate >= 60% => GO


# --------------------------------------------------------------------------- #
# SMILES parsing helpers
# --------------------------------------------------------------------------- #
def parse_reaction_smiles(reaction_smiles: str) -> Tuple[str, str]:
    """Split a reaction SMILES ``"A.B>>C.D"`` into ``(reactants, products)``.

    Returns ``("", "")`` if the string does not contain exactly one ``>>``.
    """
    if not isinstance(reaction_smiles, str) or ">>" not in reaction_smiles:
        return "", ""
    parts = reaction_smiles.split(">>")
    if len(parts) != 2:
        return "", ""
    return parts[0], parts[1]


def strip_atom_maps(smiles: str) -> str:
    """Remove RDKit atom-map numbers (``[C:1]`` -> ``[C]``) from a SMILES.

    Returns the canonical SMILES without atom maps, or ``""`` if parsing
    fails.  Uses ``SetAtomMapNum(0)`` because ``ClearProp("molAtomMap")``
    alone does not prevent ``MolToSmiles`` from re-emitting the map number.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return ""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol)


# --------------------------------------------------------------------------- #
# Energy computation
# --------------------------------------------------------------------------- #
def compute_molecule_energy(
    smiles: str,
    method: str = "mmff94",
    seed: int = 42,
    max_iters: int = 200,
) -> Dict:
    """Compute the single-point energy (kcal/mol) of one molecule SMILES.

    Steps:
      1. Strip atom maps.
      2. Parse + add explicit hydrogens.
      3. Embed a 3D conformation (ETKDG / random coords fallback).
      4. Optimize with MMFF94 (degrade to UFF if MMFF params missing / failure).

    Returns a dict with keys: ``smiles``, ``energy_kcal_per_mol``,
    ``status``, ``method``.  ``energy_kcal_per_mol`` is ``None`` on failure.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return {"smiles": "", "energy_kcal_per_mol": None, "status": "empty", "method": method}

    clean = strip_atom_maps(smiles)
    if not clean:
        return {"smiles": smiles, "energy_kcal_per_mol": None, "status": "parse_error", "method": method}

    mol = Chem.MolFromSmiles(clean)
    if mol is None:
        return {"smiles": smiles, "energy_kcal_per_mol": None, "status": "parse_error", "method": method}

    mol = Chem.AddHs(mol)

    # 3D embedding
    try:
        rid = AllChem.EmbedMolecule(mol, randomSeed=seed, useRandomCoords=True)
        if rid != 0:
            # retry with basic params
            rid = AllChem.EmbedMolecule(mol, randomSeed=seed)
        if rid != 0:
            return {"smiles": smiles, "energy_kcal_per_mol": None, "status": "embed_failed", "method": method}
    except Exception:
        return {"smiles": smiles, "energy_kcal_per_mol": None, "status": "embed_failed", "method": method}

    actual_method = method
    energy: Optional[float] = None

    if actual_method == "xtb":
        # xtb-python not installed in this environment; auto-degrade.
        actual_method = "mmff94"

    if actual_method == "mmff94":
        if not MMFFHasAllMoleculeParams(mol):
            actual_method = "uff"  # degrade
        else:
            try:
                MMFFOptimizeMolecule(mol, maxIters=max_iters)
                props = MMFFGetMoleculeProperties(mol)
                ff = MMFFGetMoleculeForceField(mol, props)
                energy = float(ff.CalcEnergy())
            except Exception:
                actual_method = "uff"  # degrade on failure

    if actual_method == "uff" and energy is None:
        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=max_iters)
            ff = AllChem.UFFGetMoleculeForceField(mol)
            energy = float(ff.CalcEnergy())
        except Exception:
            return {"smiles": smiles, "energy_kcal_per_mol": None, "status": "uff_failed", "method": actual_method}

    if energy is None:
        return {"smiles": smiles, "energy_kcal_per_mol": None, "status": "no_energy", "method": actual_method}

    return {"smiles": smiles, "energy_kcal_per_mol": energy, "status": "ok", "method": actual_method}


def compute_reaction_energy(
    reactants_smiles: str,
    products_smiles: str,
    method: str = "mmff94",
    seed: int = 42,
) -> Dict:
    """Compute ``DeltaG = E(products) - E(reactants)`` for a reaction.

    Multi-component SMILES (``"A.B.C"``) are split on ``"."`` and the
    component energies are summed.  The barrier estimate follows the
    Section 22.1 P1-10 spec::

        barrier_estimate = |E_products - E_reactants| + 5 kcal/mol

    (a rough force-field level estimate; documented as such in the protocol).

    Returns a dict with ``reactant_energy``, ``product_energy``,
    ``delta_g``, ``barrier_estimate``, ``status``, ``method``.
    """
    r_components = [s for s in str(reactants_smiles).split(".") if s]
    p_components = [s for s in str(products_smiles).split(".") if s]

    r_energy: Optional[float] = 0.0
    r_status = "ok"
    r_method_used = method
    for comp in r_components:
        res = compute_molecule_energy(comp, method=method, seed=seed)
        r_method_used = res["method"]  # capture any degradation
        if res["energy_kcal_per_mol"] is None:
            r_energy = None
            r_status = f"reactant_failed:{res['status']}"
            break
        r_energy += res["energy_kcal_per_mol"]

    p_energy: Optional[float] = 0.0
    p_status = "ok"
    p_method_used = method
    for comp in p_components:
        res = compute_molecule_energy(comp, method=method, seed=seed)
        p_method_used = res["method"]
        if res["energy_kcal_per_mol"] is None:
            p_energy = None
            p_status = f"product_failed:{res['status']}"
            break
        p_energy += res["energy_kcal_per_mol"]

    if r_energy is None or p_energy is None:
        return {
            "reactant_energy": r_energy,
            "product_energy": p_energy,
            "delta_g": None,
            "barrier_estimate": None,
            "status": f"{r_status};{p_status}",
            "method": r_method_used or p_method_used,
        }

    delta_g = float(p_energy - r_energy)
    barrier_estimate = abs(delta_g) + BARRIER_FORCE_FIELD_CONSTANT
    return {
        "reactant_energy": float(r_energy),
        "product_energy": float(p_energy),
        "delta_g": delta_g,
        "barrier_estimate": float(barrier_estimate),
        "status": "ok",
        "method": r_method_used or p_method_used,
    }


# --------------------------------------------------------------------------- #
# Judgment rule
# --------------------------------------------------------------------------- #
def judge_support(
    delta_g: Optional[float],
    barrier: Optional[float],
) -> str:
    """Apply the P1-10 low-feasibility support rule.

    Rule (Section 22.1):
      - ``DeltaG > +5 kcal/mol``  => supported (thermodynamically uphill)
      - ``barrier > 25 kcal/mol`` => supported (kinetically hindered)

    To avoid a false-positive where a strongly *exothermic* reaction
    (``DeltaG << 0``) inflates the ``|DeltaG| + 5`` barrier estimate, the
    barrier branch only triggers when ``DeltaG > 0`` (endothermic side).
    """
    if delta_g is None and barrier is None:
        return "inconclusive"
    if delta_g is not None and delta_g > DELTA_G_SUPPORT_THRESHOLD:
        return "supported"
    # Only use the barrier signal when the reaction is at least weakly
    # endothermic (DeltaG > 0); for exothermic reactions the force-field
    # barrier estimate is not physically meaningful.
    if (
        barrier is not None
        and barrier > BARRIER_SUPPORT_THRESHOLD
        and delta_g is not None
        and delta_g > 0
    ):
        return "supported"
    return "not_supported"


def support_reason(
    delta_g: Optional[float],
    barrier: Optional[float],
) -> str:
    """Return a human-readable reason for the support verdict."""
    if delta_g is None and barrier is None:
        return "inconclusive: no energy data"
    if delta_g is not None and delta_g > DELTA_G_SUPPORT_THRESHOLD:
        return f"delta_g>{DELTA_G_SUPPORT_THRESHOLD:.0f} ({delta_g:+.2f} kcal/mol)"
    if (
        barrier is not None
        and barrier > BARRIER_SUPPORT_THRESHOLD
        and delta_g is not None
        and delta_g > 0
    ):
        return f"barrier>{BARRIER_SUPPORT_THRESHOLD:.0f} ({barrier:.2f} kcal/mol, endothermic)"
    return f"not supported (delta_g={delta_g}, barrier={barrier})"


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
def sample_candidates(
    df: pd.DataFrame,
    limit: int = 100,
    require_chemical_change: bool = True,
    deduplicate: bool = True,
) -> pd.DataFrame:
    """Sample high-value synthetic negatives.

    ``feasibility_score = 1 - hard_score`` (lower => less feasible => higher
    confidence the candidate is a true negative).  Sort ascending by
    feasibility_score and take the top ``limit`` rows.

    When ``require_chemical_change`` is True (default), rows whose
    ``candidate_reactants == candidate_product`` are excluded because their
    DeltaG is trivially 0 and provides no thermodynamic evidence.

    When ``deduplicate`` is True (default), duplicate
    ``(candidate_reactants, candidate_product)`` pairs are dropped (keeping
    the first occurrence after sorting, i.e. the highest-confidence one) so
    that the limited compute budget is spent on chemically diverse reactions.
    """
    df = df.copy()
    df["hard_score"] = pd.to_numeric(df["hard_score"], errors="coerce")
    df = df.dropna(subset=["hard_score"])
    df["feasibility_score"] = 1.0 - df["hard_score"].astype(float)

    if require_chemical_change:
        mask = df["candidate_reactants"].astype(str) != df["candidate_product"].astype(str)
        df = df[mask].copy()

    df = df.sort_values("feasibility_score", ascending=True, kind="mergesort")

    if deduplicate:
        df = df.drop_duplicates(
            subset=["candidate_reactants", "candidate_product"],
            keep="first",
        )

    df = df.head(limit)
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Paired 10-seed bootstrap significance test
# --------------------------------------------------------------------------- #
def run_paired_significance_test(
    neg_deltas: Sequence[Optional[float]],
    pos_deltas: Sequence[Optional[float]],
    num_seeds: int = DEFAULT_NUM_SEEDS,
    base_seed: int = DEFAULT_RANDOM_SEED,
) -> Dict:
    """10-seed paired bootstrap significance test.

    Pairs ``neg_deltas[i]`` with ``pos_deltas[i]`` (truncated to the common
    length, dropping pairs where either side is ``None``).  For each of
    ``num_seeds`` seeds we bootstrap-resample the paired differences
    (``neg - pos``) with replacement and record the mean and the percentile
    95% CI.  A seed is "significant" if the CI excludes 0.

    Returns a dict with ``n_pairs``, ``num_seeds``, ``overall_mean_diff``,
    ``n_significant_seeds``, ``significance_rate``, ``seed_results``.
    """
    n = min(len(neg_deltas), len(pos_deltas))
    pairs: List[Tuple[float, float]] = []
    for i in range(n):
        nd = neg_deltas[i]
        pd_ = pos_deltas[i]
        if nd is None or pd_ is None:
            continue
        try:
            pairs.append((float(nd), float(pd_)))
        except (TypeError, ValueError):
            continue

    if len(pairs) < 2:
        return {
            "status": "insufficient_data" if len(pairs) < 2 else "ok",
            "n_pairs": len(pairs),
            "num_seeds": num_seeds,
            "overall_mean_diff": None,
            "n_significant_seeds": 0,
            "significance_rate": 0.0,
            "seed_results": [],
            "interpretation": "insufficient_data",
        }

    diffs = [nd - pd_ for nd, pd_ in pairs]
    overall_mean_diff = sum(diffs) / len(diffs)

    seed_results: List[Dict] = []
    n_significant = 0
    n = len(diffs)

    for seed_idx in range(num_seeds):
        rng = random.Random(base_seed + seed_idx)
        resampled = [rng.choice(diffs) for _ in range(n)]
        mean_diff = sum(resampled) / n
        sorted_rs = sorted(resampled)
        lo_idx = max(0, int(0.025 * n))
        hi_idx = min(n - 1, int(0.975 * n))
        lo = sorted_rs[lo_idx]
        hi = sorted_rs[hi_idx]
        excludes_zero = (lo > 0) or (hi < 0)
        if excludes_zero:
            n_significant += 1
        seed_results.append({
            "seed": base_seed + seed_idx,
            "mean_diff": mean_diff,
            "ci_lo": lo,
            "ci_hi": hi,
            "excludes_zero": bool(excludes_zero),
        })

    significance_rate = n_significant / num_seeds if num_seeds > 0 else 0.0
    interpretation = "significant" if significance_rate >= 0.8 else "not_significant"

    return {
        "status": "ok",
        "n_pairs": len(pairs),
        "num_seeds": num_seeds,
        "overall_mean_diff": overall_mean_diff,
        "n_significant_seeds": n_significant,
        "significance_rate": significance_rate,
        "seed_results": seed_results,
        "interpretation": interpretation,
    }


# --------------------------------------------------------------------------- #
# Main CLI
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="P1-10: xTB / DFT / MMFF94 computational validation of synthetic negatives",
    )
    parser.add_argument(
        "--candidates",
        required=True,
        help="Path to high_confidence_negatives.csv (P1-08 output)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Max number of synthetic negatives to compute (default: 100)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for xtb_results.csv / validation_summary.json",
    )
    parser.add_argument(
        "--method",
        choices=["xtb", "mmff94", "uff"],
        default="mmff94",
        help="Computational method (default: mmff94; xtb auto-degrades to mmff94 if unavailable)",
    )
    parser.add_argument(
        "--control-size",
        type=int,
        default=100,
        help="Control positive sample size (default: 100)",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=DEFAULT_NUM_SEEDS,
        help=f"Number of seeds for paired bootstrap test (default: {DEFAULT_NUM_SEEDS})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=f"Base random seed (default: {DEFAULT_RANDOM_SEED})",
    )
    parser.add_argument(
        "--no-require-chemical-change",
        action="store_true",
        help="Do not require candidate_reactants != candidate_product (include no-reaction rows)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load candidates
    if not os.path.isfile(args.candidates):
        print(f"[error] candidates file not found: {args.candidates}", file=sys.stderr)
        return 2
    df = pd.read_csv(args.candidates)
    print(f"[info] loaded {len(df)} candidates from {args.candidates}")

    # Determine actual method (xtb auto-degrades)
    actual_method = args.method
    if actual_method == "xtb":
        try:
            import xtb  # noqa: F401
        except ImportError:
            print("[warn] xtb-python not available; degrading to MMFF94", file=sys.stderr)
            actual_method = "mmff94"

    # Sample high-value synthetic negatives
    sampled = sample_candidates(
        df,
        limit=args.limit,
        require_chemical_change=not args.no_require_chemical_change,
    )
    print(f"[info] sampled {len(sampled)} high-value synthetic negatives "
          f"(require_chemical_change={not args.no_require_chemical_change})")

    # Compute energies for synthetic negatives + paired control positives
    neg_rows: List[Dict] = []
    pos_rows: List[Dict] = []
    for _, row in sampled.iterrows():
        cand_r = str(row.get("candidate_reactants", ""))
        cand_p = str(row.get("candidate_product", ""))
        pos_rxn = str(row.get("positive_reaction", ""))

        # Synthetic negative energy
        neg_energy = compute_reaction_energy(cand_r, cand_p, method=actual_method, seed=args.seed)
        neg_verdict = judge_support(neg_energy["delta_g"], neg_energy["barrier_estimate"])
        neg_rows.append({
            "source_id": row.get("source_id", ""),
            "failure_type": row.get("failure_type", ""),
            "edit_action": row.get("edit_action", ""),
            "task": row.get("task", ""),
            "hard_score": float(row.get("hard_score", 0.0)) if pd.notna(row.get("hard_score")) else None,
            "false_negative_risk": float(row.get("false_negative_risk", 0.0)) if pd.notna(row.get("false_negative_risk")) else None,
            "feasibility_score": float(row.get("feasibility_score", 0.0)) if pd.notna(row.get("feasibility_score")) else None,
            "candidate_reactants": cand_r,
            "candidate_product": cand_p,
            "reactant_energy": neg_energy["reactant_energy"],
            "product_energy": neg_energy["product_energy"],
            "delta_g": neg_energy["delta_g"],
            "barrier_estimate": neg_energy["barrier_estimate"],
            "method": neg_energy["method"],
            "status": neg_energy["status"],
            "support_verdict": neg_verdict,
            "support_reason": support_reason(neg_energy["delta_g"], neg_energy["barrier_estimate"]),
        })

        # Paired control: observed positive reaction
        pos_r, pos_p = parse_reaction_smiles(pos_rxn)
        if pos_r and pos_p:
            pos_energy = compute_reaction_energy(pos_r, pos_p, method=actual_method, seed=args.seed)
            pos_rows.append({
                "source_id": row.get("source_id", ""),
                "positive_reaction": pos_rxn,
                "reactant_energy": pos_energy["reactant_energy"],
                "product_energy": pos_energy["product_energy"],
                "delta_g": pos_energy["delta_g"],
                "barrier_estimate": pos_energy["barrier_estimate"],
                "status": pos_energy["status"],
                "method": pos_energy["method"],
            })
        else:
            pos_rows.append({
                "source_id": row.get("source_id", ""),
                "positive_reaction": pos_rxn,
                "reactant_energy": None,
                "product_energy": None,
                "delta_g": None,
                "barrier_estimate": None,
                "status": "no_positive_reaction",
                "method": actual_method,
            })

    neg_df = pd.DataFrame(neg_rows)
    pos_df = pd.DataFrame(pos_rows)

    # Truncate control to control_size
    if len(pos_df) > args.control_size:
        pos_df = pos_df.head(args.control_size)

    # Write per-reaction results
    neg_df.to_csv(os.path.join(args.output_dir, "xtb_results.csv"), index=False)
    print(f"[info] wrote xtb_results.csv ({len(neg_df)} rows)")
    pos_df.to_csv(os.path.join(args.output_dir, "control_positive_results.csv"), index=False)
    print(f"[info] wrote control_positive_results.csv ({len(pos_df)} rows)")

    # Summary stats
    total = len(neg_df)
    supported = int((neg_df["support_verdict"] == "supported").sum())
    not_supported = int((neg_df["support_verdict"] == "not_supported").sum())
    inconclusive = int((neg_df["support_verdict"] == "inconclusive").sum())
    support_rate = supported / total if total > 0 else 0.0

    valid_neg_deltas = neg_df["delta_g"].dropna().tolist()
    valid_pos_deltas = [d for d in pos_df["delta_g"].tolist() if d is not None and pd.notna(d)]

    delta_g_stats = {
        "n_valid": len(valid_neg_deltas),
        "mean": float(pd.Series(valid_neg_deltas).mean()) if valid_neg_deltas else None,
        "std": float(pd.Series(valid_neg_deltas).std()) if len(valid_neg_deltas) > 1 else None,
        "min": float(pd.Series(valid_neg_deltas).min()) if valid_neg_deltas else None,
        "max": float(pd.Series(valid_neg_deltas).max()) if valid_neg_deltas else None,
        "median": float(pd.Series(valid_neg_deltas).median()) if valid_neg_deltas else None,
    }

    pos_delta_g_stats = {
        "n_valid": len(valid_pos_deltas),
        "mean": float(pd.Series(valid_pos_deltas).mean()) if valid_pos_deltas else None,
        "std": float(pd.Series(valid_pos_deltas).std()) if len(valid_pos_deltas) > 1 else None,
        "median": float(pd.Series(valid_pos_deltas).median()) if valid_pos_deltas else None,
    }

    # Paired significance (pair by source_id, position-aligned)
    paired = run_paired_significance_test(
        neg_df["delta_g"].tolist(),
        pos_df["delta_g"].tolist(),
        num_seeds=args.num_seeds,
        base_seed=args.seed,
    )

    go_no_go = "GO" if support_rate >= GO_NO_GO_THRESHOLD else "NO_GO_partial_support"

    summary = {
        "task": "P1-10 computational validation",
        "method_requested": args.method,
        "method_actual": actual_method,
        "degraded_from_requested": actual_method != args.method,
        "candidates_path": os.path.abspath(args.candidates),
        "total_candidates_loaded": int(len(df)),
        "n_synthetic_negatives_computed": int(total),
        "n_control_positives_computed": int(len(pos_df)),
        "support_rule": f"delta_g > +{DELTA_G_SUPPORT_THRESHOLD:.0f} kcal/mol OR (barrier > {BARRIER_SUPPORT_THRESHOLD:.0f} AND delta_g > 0)",
        "n_supported": supported,
        "n_not_supported": not_supported,
        "n_inconclusive": inconclusive,
        "support_rate": float(support_rate),
        "delta_g_stats_synthetic_neg": delta_g_stats,
        "delta_g_stats_control_pos": pos_delta_g_stats,
        "go_no_go_threshold": GO_NO_GO_THRESHOLD,
        "go_no_go_verdict": go_no_go,
        "paired_significance": paired,
        "num_seeds": args.num_seeds,
        "base_seed": args.seed,
        "timestamp": pd.Timestamp.now().isoformat(),
    }

    with open(os.path.join(args.output_dir, "validation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[info] wrote validation_summary.json")

    with open(os.path.join(args.output_dir, "paired_significance.json"), "w") as f:
        json.dump(paired, f, indent=2, default=str)
    print(f"[info] wrote paired_significance.json")

    # Console summary
    print("\n" + "=" * 60)
    print("P1-10 Computational Validation Summary")
    print("=" * 60)
    print(f"Method requested : {args.method}")
    print(f"Method actual    : {actual_method}  (degraded: {actual_method != args.method})")
    print(f"Synthetic negs   : {total}  (supported: {supported} = {support_rate:.1%})")
    print(f"Control positives: {len(pos_df)}")
    print(f"Go/No-Go         : {go_no_go}  (threshold: {GO_NO_GO_THRESHOLD:.0%})")
    if delta_g_stats["n_valid"] > 0:
        print(f"DeltaG (neg)     : mean={delta_g_stats['mean']:+.2f}  "
              f"median={delta_g_stats['median']:+.2f}  "
              f"min={delta_g_stats['min']:+.2f}  max={delta_g_stats['max']:+.2f}")
    if pos_delta_g_stats["n_valid"] > 0:
        print(f"DeltaG (pos ctrl): mean={pos_delta_g_stats['mean']:+.2f}  "
              f"median={pos_delta_g_stats['median']:+.2f}")
    print(f"Paired test      : {paired['interpretation']}  "
          f"({paired['n_significant_seeds']}/{paired['num_seeds']} seeds significant, "
          f"mean_diff={paired.get('overall_mean_diff')})")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
