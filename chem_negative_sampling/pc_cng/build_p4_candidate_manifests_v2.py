"""P4 v2 Candidate Manifest Builder — blocking remediation for P4-G3 NO_GO.

The frozen v1 ``hte_feasibility_v1.json`` has three defective negative
sources (see docs/p4_04_generator_scorer_analysis.md §4):

* ``rule_pc_cng`` — v1 used a plain string ``.replace()`` on *atom-mapped*
  gold SMILES, which never matched, so every group silently fell back to the
  random-corruption string (A6 ≡ A2 in all 500 groups).
* ``random_corruption`` — v1 used character-level delete/insert/swap on the
  SMILES string; only 7.4% of candidates are RDKit-parseable.
* ``unconstrained_edit`` — v1 used random string truncation; only 5.0%
  parseable.

This builder regenerates exactly those three sources with validity-enforced,
genuinely structure-based generators:

* ``rule_pc_cng``: reaction-SMARTS functional-group interconversion rules
  (acid→amide, ester→amide, amide→acid, aryl halogen swaps, nitro→amine,
  ketone→alcohol, methyl aryl ether→phenol).  Products that collide with any
  known positive are rejected and the next rule is tried.
* ``random_corruption``: RWMol random atom substitution / terminal-atom
  deletion / bond-order change with sanitization retry.
* ``unconstrained_edit``: RWMol random heavy-atom deletion with
  sanitization retry.

Everything else — group set, gold candidates, splits, and the four
fully-valid sources (``random_mismatch``, ``tanimoto_retrieval``,
``template_perturbation``, ``external_beam``) — is copied byte-identical from
the frozen v1 manifest.  v1 files are never modified; v2 is written to a new
file with ``manifest_version = "v2"`` and a fresh ``manifest_hash``.

CLI::

    python3 -m pc_cng.build_p4_candidate_manifests_v2 \
        --v1-manifest data/p4/manifests/hte_feasibility_v1.json \
        --hte-csv data/processed/hitea_full_normalized.csv \
        --output data/p4/manifests/hte_feasibility_v2.json \
        --audit-output results/p4_manifest_v2_audit/audit_report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

# Reuse v1 builder utilities (record schema, fingerprint cache, hashing).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pc_cng.build_p4_candidate_manifests import (  # noqa: E402
    _FingerprintCache,
    _compute_manifest_hash,
    _load_csv_rows,
    _make_candidate,
    _strip_atom_map,
)

MANIFEST_VERSION_V2 = "v2"
V2_SEED = 20260723

# Sources copied byte-identical from v1 (100% valid in v1, see G4 report §4).
COPIED_SOURCES = (
    "random_mismatch",
    "tanimoto_retrieval",
    "template_perturbation",
    "external_beam",
)
# Sources regenerated in v2.
REGENERATED_SOURCES = (
    "random_corruption",
    "unconstrained_edit",
    "rule_pc_cng",
)


# ---------------------------------------------------------------------------
# Rule-based PC-CNG generator (reaction-SMARTS interconversions)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Rule:
    """A functional-group interconversion rule."""
    name: str
    reaction_smarts: str


RULES: Tuple[Rule, ...] = (
    Rule("acid_to_amide",
         "[CX3:1](=[OX1:2])[OX2H1:3]>>[CX3:1](=[OX1:2])[NX3H2:3]"),
    Rule("ester_to_amide",
         "[CX3:1](=[OX1:2])[OX2H0:3]>>[CX3:1](=[OX1:2])[NX3H2:3]"),
    Rule("amide_to_acid",
         "[CX3:1](=[OX1:2])[NX3:3]>>[CX3:1](=[OX1:2])[OX2H1:3]"),
    Rule("aryl_br_to_cl", "[c:1][Br:2]>>[c:1][Cl:2]"),
    Rule("aryl_cl_to_br", "[c:1][Cl:2]>>[c:1][Br:2]"),
    Rule("aryl_f_to_cl", "[c:1][F:2]>>[c:1][Cl:2]"),
    Rule("nitro_to_amine",
         "[#6:1][N+:2](=[O:3])[O-:4]>>[#6:1][NH2:2]"),
    Rule("ketone_to_alcohol",
         "[#6:3][CX3:1](=[OX1:2])[#6:4]>>[#6:3][CX3:1]([OX2H1:2])[#6:4]"),
    Rule("methyl_aryl_ether_to_phenol",
         "[c:1][OX2:2][CH3:3]>>[c:1][OX2H1:2]"),
    Rule("alcohol_to_ketone",
         "[#6:3][CX4H1:1]([OX2H1:2])[#6:4]"
         ">>[#6:3][CX3:1](=[OX1:2])[#6:4]"),
    Rule("primary_alcohol_to_aldehyde",
         "[CX4H2:1][OX2H1:2]>>[CX3H1:1]=[OX1:2]"),
    Rule("nitrile_to_amine",
         "[CX2:1]#[NX1:2]>>[CX4H2:1][NX3H2:2]"),
    Rule("amide_reduction_to_amine",
         "[CX3:1](=[OX1:2])[NX3:3]>>[CX4H2:1][NX3H2:3]"),
    Rule("aryl_demethylation", "[c:1][CH3:2]>>[c:1]"),
    Rule("n_methyl_demethylation", "[n:1][CH3:2]>>[nH:1]"),
    Rule("aryl_cl_to_amine", "[c:1][Cl:2]>>[c:1][NH2:2]"),
    Rule("amine_acetylation", "[NX3H2:1]>>[NX3:1]C(C)=O"),
)

_COMPILED_RULES: Tuple[Tuple[Rule, Any], ...] = tuple(
    (r, AllChem.ReactionFromSmarts(r.reaction_smarts)) for r in RULES
)


def _canonical(smiles: str) -> Optional[str]:
    """Canonical isomeric SMILES, or None if unparseable."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def _sanitize_product(mol: Chem.Mol) -> Optional[str]:
    """Sanitize a reaction product; return canonical SMILES or None."""
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def _run_rule(mol: Chem.Mol, rule: Rule, rxn: Any,
              rng: random.Random) -> Optional[str]:
    """Apply one reaction-SMARTS rule; return canonical product SMILES."""
    try:
        outcomes = rxn.RunReactants((mol,))
    except Exception:
        return None
    if not outcomes:
        return None
    products = []
    seen = set()
    for outcome in outcomes:
        smi = _sanitize_product(outcome[0])
        if smi and smi not in seen:
            seen.add(smi)
            products.append(smi)
    if not products:
        return None
    return rng.choice(products)


def generate_rule_pccng(
    gold_smiles: str,
    known_positives: frozenset,
    rng: random.Random,
) -> Tuple[Optional[str], Dict[str, Any]]:
    """Generate a rule-based PC-CNG counterfactual candidate.

    Tries the interconversion rules in a per-group shuffled order.  A product
    is accepted only if it (a) sanitizes, (b) differs from the gold
    canonical form, and (c) is not a known positive (collision check).

    Returns ``(candidate_smiles, metadata)``; ``candidate_smiles`` is None
    when every rule failed or was rejected.
    """
    meta: Dict[str, Any] = {"rule": None, "rejected_collisions": 0,
                            "rules_tried": []}
    gold_canon = _canonical(_strip_atom_map(gold_smiles))
    mol = Chem.MolFromSmiles(gold_canon) if gold_canon else None
    if mol is None:
        return None, meta

    order = list(_COMPILED_RULES)
    rng.shuffle(order)
    for rule, rxn in order:
        meta["rules_tried"].append(rule.name)
        smi = _run_rule(mol, rule, rxn, rng)
        if smi is None:
            continue
        if smi == gold_canon:
            continue
        if smi in known_positives:
            meta["rejected_collisions"] += 1
            continue
        meta["rule"] = rule.name
        return smi, meta
    return None, meta


# ---------------------------------------------------------------------------
# Validity-enforced random corruption (RWMol edits, sanitization retry)
# ---------------------------------------------------------------------------

_SUBST_ELEMENTS = [6, 7, 8, 16, 9, 17, 35]  # C N O S F Cl Br


def _try_sanitize_rw(rw: Chem.RWMol) -> Optional[str]:
    """Get molecule from RWMol and sanitize; canonical SMILES or None."""
    try:
        mol = rw.GetMol()
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    except Exception:
        return None


def _random_rw_edit(gold_mol: Chem.Mol, rng: random.Random,
                    allow_delete: bool, allow_bond: bool) -> Optional[str]:
    """One random RWMol edit attempt; canonical SMILES or None."""
    heavy = [a.GetIdx() for a in gold_mol.GetAtoms() if a.GetAtomicNum() > 1]
    if not heavy:
        return None
    ops = ["substitute"]
    terminal = [i for i in heavy if gold_mol.GetAtomWithIdx(i).GetDegree() == 1]
    if allow_delete and terminal:
        ops.append("delete_terminal")
    if allow_bond:
        single_cc = [b.GetIdx() for b in gold_mol.GetBonds()
                     if b.GetBondType() == Chem.BondType.SINGLE
                     and b.GetBeginAtom().GetAtomicNum() == 6
                     and b.GetEndAtom().GetAtomicNum() == 6
                     and not b.IsInRing()]
        if single_cc:
            ops.append("bond_order")
    op = rng.choice(ops)

    rw = Chem.RWMol(gold_mol)
    if op == "substitute":
        idx = rng.choice(heavy)
        atom = rw.GetAtomWithIdx(idx)
        choices = [z for z in _SUBST_ELEMENTS if z != atom.GetAtomicNum()]
        atom.SetAtomicNum(rng.choice(choices))
        atom.SetFormalCharge(0)
        atom.SetNumExplicitHs(0)
        atom.UpdatePropertyCache(strict=False)
    elif op == "delete_terminal":
        rw.RemoveAtom(rng.choice(terminal))
    else:  # bond_order
        bond_idx = rng.choice(single_cc)
        rw.GetBondWithIdx(bond_idx).SetBondType(Chem.BondType.DOUBLE)
    return _try_sanitize_rw(rw)


def generate_valid_corruption(
    gold_smiles: str,
    known_positives: frozenset,
    rng: random.Random,
    max_attempts: int = 200,
) -> Tuple[Optional[str], Dict[str, Any]]:
    """Random structural corruption guaranteed to be RDKit-parseable.

    Applies random atom substitutions / terminal-atom deletions / bond-order
    changes via RWMol, retrying until a sanitizable, non-gold, non-known-
    positive molecule is found.
    """
    meta: Dict[str, Any] = {"attempts": 0}
    gold_canon = _canonical(_strip_atom_map(gold_smiles))
    gold_mol = Chem.MolFromSmiles(gold_canon) if gold_canon else None
    if gold_mol is None:
        return None, meta
    for attempt in range(1, max_attempts + 1):
        meta["attempts"] = attempt
        smi = _random_rw_edit(gold_mol, rng, allow_delete=True,
                              allow_bond=True)
        if smi is None or smi == gold_canon or smi in known_positives:
            continue
        return smi, meta
    return None, meta


def generate_valid_unconstrained_edit(
    gold_smiles: str,
    known_positives: frozenset,
    rng: random.Random,
    max_attempts: int = 200,
) -> Tuple[Optional[str], Dict[str, Any]]:
    """Unconstrained structural edit guaranteed to be RDKit-parseable.

    Deletes 1-2 heavy atoms (preferring terminal atoms) via RWMol with
    sanitization retry; falls back to substitution edits when the molecule
    is too small to delete from.
    """
    meta: Dict[str, Any] = {"attempts": 0}
    gold_canon = _canonical(_strip_atom_map(gold_smiles))
    gold_mol = Chem.MolFromSmiles(gold_canon) if gold_canon else None
    if gold_mol is None:
        return None, meta
    n_heavy = gold_mol.GetNumHeavyAtoms()
    for attempt in range(1, max_attempts + 1):
        meta["attempts"] = attempt
        if n_heavy <= 6:
            # Small molecule: deletion would trivialize; substitute instead.
            smi = _random_rw_edit(gold_mol, rng, allow_delete=False,
                                  allow_bond=False)
        else:
            strategy = attempt % 3
            if strategy == 0:
                # Substitution fallback keeps diversity when the terminal
                # pool is tiny (deterministic deletion products that may
                # collide with known positives).
                smi = _random_rw_edit(gold_mol, rng, allow_delete=False,
                                      allow_bond=True)
            else:
                rw = Chem.RWMol(gold_mol)
                heavy = [a.GetIdx() for a in rw.GetAtoms()
                         if a.GetAtomicNum() > 1]
                if strategy == 1:
                    terminal = [i for i in heavy
                                if rw.GetAtomWithIdx(i).GetDegree() == 1]
                    pool = terminal if terminal else heavy
                else:
                    # Any heavy atom; may fragment the molecule, which is
                    # acceptable for an unconstrained edit.
                    pool = heavy
                n_del = rng.randint(1, min(2, len(pool)))
                for idx in sorted(rng.sample(pool, n_del), reverse=True):
                    rw.RemoveAtom(idx)
                smi = _try_sanitize_rw(rw)
        if smi is None or smi == gold_canon or smi in known_positives:
            continue
        return smi, meta
    return None, meta


# ---------------------------------------------------------------------------
# v2 manifest assembly
# ---------------------------------------------------------------------------

def build_known_positive_pool(hte_csv: Path, v1_manifest: dict) -> frozenset:
    """Canonical SMILES of every observed product (all splits) + v1 golds."""
    pool = set()
    for row in _load_csv_rows(hte_csv):
        canon = _canonical(_strip_atom_map(row.get("products", "")))
        if canon:
            pool.add(canon)
    for group in v1_manifest.get("groups", []):
        for cand in group.get("candidates", []):
            if cand.get("gold_candidate"):
                canon = _canonical(_strip_atom_map(
                    cand.get("candidate_smiles", "")))
                if canon:
                    pool.add(canon)
    return frozenset(pool)


def _regenerate_candidate(
    source: str,
    gold_smiles: str,
    known_positives: frozenset,
    rng: random.Random,
) -> Tuple[Optional[str], Dict[str, Any]]:
    if source == "rule_pc_cng":
        return generate_rule_pccng(gold_smiles, known_positives, rng)
    if source == "random_corruption":
        return generate_valid_corruption(gold_smiles, known_positives, rng)
    if source == "unconstrained_edit":
        return generate_valid_unconstrained_edit(gold_smiles,
                                                 known_positives, rng)
    raise ValueError(f"unknown regenerated source: {source}")


def build_v2_manifest(
    v1_manifest: dict,
    known_positives: frozenset,
    train_products: List[str],
    seed: int = V2_SEED,
) -> Tuple[dict, Dict[str, Any]]:
    """Build the v2 manifest from a frozen v1 manifest.

    Returns ``(v2_manifest, build_report)``.  Raises ``RuntimeError`` if any
    group cannot produce a valid candidate for a regenerated source (the
    fallback chain rule -> corruption -> edit makes this practically
    unreachable, and silence is never acceptable here).
    """
    fp_cache = _FingerprintCache()
    fp_cache.prepare_train_set(train_products, max_n=500)

    report: Dict[str, Any] = {
        "seed": seed,
        "groups_total": 0,
        "rule_usage": {},
        "rejected_collisions": 0,
        "corruption_attempts_mean": 0.0,
        "edit_attempts_mean": 0.0,
        "quarantined": [],
    }
    corruption_attempts: List[int] = []
    edit_attempts: List[int] = []

    v2_groups: List[dict] = []
    for group in v1_manifest.get("groups", []):
        report["groups_total"] += 1
        gid = group["group_id"]
        rng = random.Random(f"{seed}:{gid}")
        gold_cand = next(c for c in group["candidates"]
                         if c.get("gold_candidate"))
        gold_smiles = gold_cand["candidate_smiles"]

        new_candidates: List[dict] = []
        # Canonical SMILES of every candidate already placed in this group;
        # regenerated candidates must differ from all siblings (prevents the
        # v1-style A6≡A2 duplication when the rule fallback fires).
        sibling_excludes: set = set()
        for cand in group["candidates"]:
            src = cand.get("candidate_source", "")
            if cand.get("gold_candidate") or src in COPIED_SOURCES:
                # Byte-identical copy (version/hash backfilled below).
                canon = cand.get("canonical_smiles") or _canonical(
                    _strip_atom_map(cand.get("candidate_smiles", "")))
                if canon:
                    sibling_excludes.add(canon)
                new_candidates.append(dict(cand))
                continue
            if src not in REGENERATED_SOURCES:
                raise RuntimeError(f"{gid}: unexpected source {src!r}")

            exclusion_pool = known_positives | sibling_excludes
            smi, meta = _regenerate_candidate(
                src, gold_smiles, exclusion_pool, rng)
            if smi is None and src == "rule_pc_cng":
                # Fallback chain: no applicable rule -> valid corruption
                # (still excluding every sibling, so A6 never equals A2).
                smi, meta2 = generate_valid_corruption(
                    gold_smiles, exclusion_pool, rng)
                meta = {**meta, "fallback": "valid_corruption",
                        **meta2}
            if smi is None:
                report["quarantined"].append({"group_id": gid,
                                              "source": src})
                raise RuntimeError(
                    f"{gid}: could not generate a valid {src} candidate; "
                    "refusing to emit an invalid or duplicated candidate")

            if src == "rule_pc_cng":
                rule_name = meta.get("rule") or "fallback_valid_corruption"
                report["rule_usage"][rule_name] = \
                    report["rule_usage"].get(rule_name, 0) + 1
                report["rejected_collisions"] += \
                    meta.get("rejected_collisions", 0)
            elif src == "random_corruption":
                corruption_attempts.append(meta["attempts"])
            elif src == "unconstrained_edit":
                edit_attempts.append(meta["attempts"])

            rec = _make_candidate(
                benchmark_name=cand.get("benchmark_name",
                                        "P4-HTE-Feasibility"),
                group_id=gid,
                source_reaction_id=cand.get("source_reaction_id", ""),
                parent_reaction_id=cand.get("parent_reaction_id", ""),
                experimental_group_id=cand.get("experimental_group_id", ""),
                split=cand.get("split", group.get("split", "train")),
                candidate_smiles=smi,
                candidate_source=src,
                candidate_source_rank=cand.get("candidate_source_rank", 0),
                gold=False,
                gold_smiles=gold_smiles,
                reaction_family=cand.get("reaction_family", ""),
                reaction_template=cand.get("reaction_template", ""),
                train_products=train_products,
                manifest_hash="PENDING",
                fp_cache=fp_cache,
            )
            if src == "rule_pc_cng":
                rec["pccng_rule"] = meta.get("rule") or \
                    "fallback_valid_corruption"
            new_candidates.append(rec)
            sibling_excludes.add(rec["canonical_smiles"])

        v2_groups.append({
            "group_id": gid,
            "source_reaction_id": group.get("source_reaction_id", ""),
            "parent_reaction_id": group.get("parent_reaction_id", ""),
            "experimental_group_id": group.get("experimental_group_id", ""),
            "split": group.get("split", "train"),
            "candidates": new_candidates,
        })

    v2 = {
        "benchmark_name": v1_manifest.get("benchmark_name",
                                          "P4-HTE-Feasibility"),
        "manifest_version": MANIFEST_VERSION_V2,
        "derived_from": {
            "manifest_version": v1_manifest.get("manifest_version"),
            "manifest_hash": v1_manifest.get("manifest_hash"),
        },
        "v2_seed": seed,
        "manifest_hash": "",
        "groups": v2_groups,
    }
    v2["manifest_hash"] = _compute_manifest_hash(v2)

    # Backfill version + hash into every candidate record.
    for g in v2_groups:
        for c in g["candidates"]:
            c["manifest_version"] = MANIFEST_VERSION_V2
            c["manifest_hash"] = v2["manifest_hash"]

    if corruption_attempts:
        report["corruption_attempts_mean"] = round(
            sum(corruption_attempts) / len(corruption_attempts), 2)
    if edit_attempts:
        report["edit_attempts_mean"] = round(
            sum(edit_attempts) / len(edit_attempts), 2)
    return v2, report


# ---------------------------------------------------------------------------
# v2 audit
# ---------------------------------------------------------------------------

def audit_v2_manifest(v1_manifest: dict, v2_manifest: dict) -> Dict[str, Any]:
    """Audit v2 against v1.  Every check must pass for the build to be GO."""
    checks: List[Dict[str, Any]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "passed": bool(ok), "detail": detail})

    v1_groups = v1_manifest.get("groups", [])
    v2_groups = v2_manifest.get("groups", [])

    check("manifest_version_is_v2",
          v2_manifest.get("manifest_version") == "v2")
    check("hash_differs_from_v1",
          v2_manifest.get("manifest_hash") != v1_manifest.get("manifest_hash"))
    check("hash_present", bool(v2_manifest.get("manifest_hash")))
    check("same_group_count", len(v1_groups) == len(v2_groups),
          f"v1={len(v1_groups)} v2={len(v2_groups)}")

    v1_by_id = {g["group_id"]: g for g in v1_groups}
    n_gold_identical = 0
    n_split_identical = 0
    n_copied_identical = 0
    n_copied_expected = 0
    one_gold_ok = True
    unique_ids_ok = True
    eight_cands_ok = True
    parse_failures: Dict[str, int] = {}
    a6_eq_a2: List[str] = []
    a6_eq_gold: List[str] = []
    a6_collision = 0

    for g2 in v2_groups:
        gid = g2["group_id"]
        g1 = v1_by_id.get(gid)
        if g1 is None:
            one_gold_ok = False
            continue
        cands2 = g2["candidates"]
        cands1 = {c["candidate_source"]: c for c in g1["candidates"]}

        if len(cands2) != len(g1["candidates"]):
            eight_cands_ok = False
        if sum(bool(c.get("gold_candidate")) for c in cands2) != 1:
            one_gold_ok = False
        ids = [c["candidate_id"] for c in cands2]
        if len(set(ids)) != len(ids):
            unique_ids_ok = False

        if g2.get("split") == g1.get("split"):
            n_split_identical += 1

        gold2 = next(c for c in cands2 if c.get("gold_candidate"))
        gold1 = next(c for c in g1["candidates"] if c.get("gold_candidate"))
        if gold2["candidate_smiles"] == gold1["candidate_smiles"]:
            n_gold_identical += 1

        per_src: Dict[str, str] = {}
        for c in cands2:
            src = c.get("candidate_source", "?")
            per_src[src] = c.get("canonical_smiles") or c.get(
                "candidate_smiles", "")
            mol = Chem.MolFromSmiles(c.get("candidate_smiles", ""))
            if mol is None:
                parse_failures[src] = parse_failures.get(src, 0) + 1

        for src in COPIED_SOURCES:
            n_copied_expected += 1
            c2 = next((c for c in cands2 if c.get("candidate_source") == src),
                      None)
            c1 = cands1.get(src)
            if (c2 is not None and c1 is not None
                    and c2["candidate_smiles"] == c1["candidate_smiles"]):
                n_copied_identical += 1

        a6 = per_src.get("rule_pc_cng")
        if a6 is not None:
            if a6 == per_src.get("random_corruption"):
                a6_eq_a2.append(gid)
            gold_canon = gold2.get("canonical_smiles")
            if gold_canon and a6 == gold_canon:
                a6_eq_gold.append(gid)
            c6 = next(c for c in cands2
                      if c.get("candidate_source") == "rule_pc_cng")
            if c6.get("known_positive_collision"):
                a6_collision += 1

    n = len(v2_groups)
    check("one_gold_per_group", one_gold_ok)
    check("unique_candidate_ids", unique_ids_ok)
    check("candidate_count_matches_v1", eight_cands_ok)
    check("splits_identical_to_v1", n_split_identical == n,
          f"{n_split_identical}/{n}")
    check("gold_smiles_identical_to_v1", n_gold_identical == n,
          f"{n_gold_identical}/{n}")
    check("copied_sources_byte_identical",
          n_copied_identical == n_copied_expected,
          f"{n_copied_identical}/{n_copied_expected}")
    total_parse_fail = sum(parse_failures.values())
    check("all_candidates_parseable", total_parse_fail == 0,
          f"parse_failures={parse_failures}")
    check("rule_pccng_differs_from_corruption", len(a6_eq_a2) == 0,
          f"{len(a6_eq_a2)} groups identical")
    check("rule_pccng_differs_from_gold", len(a6_eq_gold) == 0,
          f"{len(a6_eq_gold)} groups identical")
    check("rule_pccng_no_known_positive_collision", a6_collision == 0,
          f"{a6_collision} collisions")

    # Per-source validity summary (must be 1.0 for every source).
    validity: Dict[str, Dict[str, Any]] = {}
    for g2 in v2_groups:
        for c in g2["candidates"]:
            src = c.get("candidate_source", "?")
            entry = validity.setdefault(src, {"n": 0, "valid": 0})
            entry["n"] += 1
            if Chem.MolFromSmiles(c.get("candidate_smiles", "")) is not None:
                entry["valid"] += 1
    for src, entry in validity.items():
        entry["valid_fraction"] = round(entry["valid"] / entry["n"], 4)

    return {
        "all_passed": all(c["passed"] for c in checks),
        "checks": checks,
        "per_source_validity": validity,
        "n_groups": n,
        "v2_manifest_hash": v2_manifest.get("manifest_hash"),
        "v1_manifest_hash": v1_manifest.get("manifest_hash"),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v1-manifest", required=True, type=Path)
    ap.add_argument("--hte-csv", required=True, type=Path,
                    help="HTEa normalized CSV (known-positive pool + "
                         "train products for nearest-train similarity)")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--audit-output", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=V2_SEED)
    args = ap.parse_args(argv)

    v1_hash_before = _file_sha256(args.v1_manifest)
    v1 = json.loads(args.v1_manifest.read_text())

    known_positives = build_known_positive_pool(args.hte_csv, v1)
    print(f"[v2] known-positive pool: {len(known_positives)} canonical SMILES")

    rows = _load_csv_rows(args.hte_csv)
    train_products = [r["products"] for r in rows
                      if r.get("split") == "train"]

    v2, build_report = build_v2_manifest(v1, known_positives,
                                         train_products, seed=args.seed)
    print(f"[v2] built {len(v2['groups'])} groups; "
          f"rule usage: {build_report['rule_usage']}")
    print(f"[v2] rejected known-positive collisions: "
          f"{build_report['rejected_collisions']}")

    audit = audit_v2_manifest(v1, v2)
    audit["build_report"] = build_report
    audit["v1_file_sha256_before"] = v1_hash_before
    audit["v1_file_sha256_after"] = _file_sha256(args.v1_manifest)
    audit["v1_file_untouched"] = (
        audit["v1_file_sha256_before"] == audit["v1_file_sha256_after"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(v2, indent=1, ensure_ascii=False))
    print(f"[v2] wrote {args.output} (hash {v2['manifest_hash'][:16]}...)")

    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(audit, indent=1,
                                            ensure_ascii=False))
    n_pass = sum(1 for c in audit["checks"] if c["passed"])
    print(f"[v2] audit: {n_pass}/{len(audit['checks'])} checks passed; "
          f"v1 untouched: {audit['v1_file_untouched']}")
    if not audit["all_passed"] or not audit["v1_file_untouched"]:
        print("[v2] AUDIT FAILED", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
