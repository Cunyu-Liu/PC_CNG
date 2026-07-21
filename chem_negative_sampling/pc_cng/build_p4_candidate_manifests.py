"""P4-G1 Benchmark Contract & Candidate Manifest Construction.

CLI entry point::

    python3 -m pc_cng.build_p4_candidate_manifests \
        --output-dir data/p4/manifests

Builds three frozen candidate manifests per the P4-G1 spec:

* B1: ``hte_feasibility_v1.json`` — HTE real-experimental-group feasibility
* B2: ``fixed_forward_candidates_v1.json`` — fixed forward product candidates
* B3: ``fixed_retro_candidates_v1.json`` — fixed retro precursor candidates

Each manifest is a JSON document with::

    {
        "benchmark_name": "...",
        "manifest_version": "v1",
        "manifest_hash": "<sha256>",
        "groups": [
            {
                "group_id": "...",
                "source_reaction_id": "...",
                "parent_reaction_id": "...",
                "experimental_group_id": "...",
                "split": "train|val|test",
                "candidates": [ { ... 24 fields ... }, ... ]
            },
            ...
        ]
    }

Hard constraints (per spec):
- Each group has exactly ONE gold_candidate;
- Same parent_reaction_id never crosses splits;
- All candidate sources are reproducible (no model-dependent sampling);
- manifest_hash is a fixed SHA-256 of the canonical JSON content.

No model training is performed.  No learned PC-CNG candidates are included
in v1 (those are appended as v2 in a later phase).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# RDKit is required for SMILES canonicalization and Tanimoto similarity.
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    from rdkit.Chem.Fingerprints import FingerprintMols
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MANIFEST_VERSION = "v1"

# Candidate source labels (per spec)
CANDIDATE_SOURCES = [
    "gold",
    "random_mismatch",
    "random_corruption",
    "tanimoto_retrieval",
    "template_perturbation",
    "unconstrained_edit",
    "rule_pc_cng",
    "external_beam",
]

# Number of groups per benchmark (kept manageable for reproducibility)
MAX_GROUPS_HTE = 500
MAX_GROUPS_FORWARD = 500
MAX_GROUPS_RETRO = 500

# Random seed for reproducibility (fixed, not model-dependent)
MANIFEST_SEED = 20260721

# Required fields per candidate (per spec lines 381-406)
REQUIRED_CANDIDATE_FIELDS = [
    "benchmark_name",
    "group_id",
    "source_reaction_id",
    "parent_reaction_id",
    "experimental_group_id",
    "gold_candidate",
    "candidate_id",
    "candidate_smiles",
    "candidate_source",
    "candidate_source_rank",
    "canonical_smiles",
    "atom_mapping_status",
    "reaction_family",
    "reaction_template",
    "product_scaffold",
    "edit_type",
    "edit_distance",
    "train_overlap",
    "known_positive_collision",
    "nearest_train_similarity",
    "split",
    "oracle_coverage",
    "manifest_version",
    "manifest_hash",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _canonicalize_smiles(smiles: str) -> str:
    """Return canonical SMILES, or empty string on failure."""
    if not RDKIT_AVAILABLE or not smiles:
        return smiles or ""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        return Chem.MolToSmiles(mol)
    except Exception:
        return ""


def _strip_atom_map(smiles: str) -> str:
    """Remove atom mapping numbers from SMILES."""
    if not smiles:
        return ""
    import re
    # Remove :NN: patterns (atom map numbers)
    return re.sub(r":\d+", "", smiles)


class _FingerprintCache:
    """Cache Morgan fingerprints by canonical SMILES to avoid recomputation.

    The build processes up to 500 groups × 8 candidates, each computing
    Tanimoto similarity against up to 200 train products. Without caching,
    the same train product SMILES is parsed and fingerprinted thousands of
    times. This cache pre-computes fingerprints once and reuses them.
    """

    def __init__(self):
        self._fp_cache: Dict[str, Any] = {}
        self._mol_cache: Dict[str, Any] = {}
        self._canon_cache: Dict[str, str] = {}
        self._scaffold_cache: Dict[str, str] = {}
        self._train_fps: List[Tuple[str, Any]] = []  # (canonical_smiles, fingerprint)
        self._train_canon_set: set = set()

    def canonicalize(self, smiles: str) -> str:
        """Return cached canonical SMILES."""
        if smiles in self._canon_cache:
            return self._canon_cache[smiles]
        result = _canonicalize_smiles_uncached(smiles)
        self._canon_cache[smiles] = result
        return result

    def get_fingerprint(self, canonical_smiles: str) -> Any:
        """Return cached Morgan fingerprint for a canonical SMILES."""
        if canonical_smiles in self._fp_cache:
            return self._fp_cache[canonical_smiles]
        if not RDKIT_AVAILABLE or not canonical_smiles:
            self._fp_cache[canonical_smiles] = None
            return None
        try:
            mol = Chem.MolFromSmiles(canonical_smiles)
            if mol is None:
                self._fp_cache[canonical_smiles] = None
                return None
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024)
            self._fp_cache[canonical_smiles] = fp
            return fp
        except Exception:
            self._fp_cache[canonical_smiles] = None
            return None

    def get_scaffold(self, canonical_smiles: str) -> str:
        """Return cached Bemis-Murcko scaffold."""
        if canonical_smiles in self._scaffold_cache:
            return self._scaffold_cache[canonical_smiles]
        result = _product_scaffold_uncached(canonical_smiles)
        self._scaffold_cache[canonical_smiles] = result
        return result

    def prepare_train_set(self, train_smiles_list: List[str], max_n: int = 500) -> None:
        """Pre-compute fingerprints and canonical SMILES for train products."""
        self._train_fps = []
        self._train_canon_set = set()
        seen_canon = set()
        for smi in train_smiles_list[:max_n * 2]:  # process more than needed, then trim
            canon = self.canonicalize(_strip_atom_map(smi))
            if not canon or canon in seen_canon:
                continue
            seen_canon.add(canon)
            fp = self.get_fingerprint(canon)
            if fp is not None:
                self._train_fps.append((canon, fp))
            self._train_canon_set.add(canon)
            if len(self._train_fps) >= max_n:
                break

    def train_canonical_set(self) -> set:
        """Return the set of canonical SMILES in the train set (for overlap check)."""
        return self._train_canon_set

    def train_fps(self) -> List[Tuple[str, Any]]:
        """Return the list of (canonical_smiles, fingerprint) for train products."""
        return self._train_fps

    def tanimoto_against_train(self, canonical_smiles: str) -> float:
        """Compute max Tanimoto similarity of a SMILES against all train products."""
        if not canonical_smiles:
            return 0.0
        fp = self.get_fingerprint(canonical_smiles)
        if fp is None:
            return 0.0
        best = 0.0
        for _, train_fp in self._train_fps:
            sim = DataStructs.TanimotoSimilarity(fp, train_fp)
            if sim > best:
                best = sim
        return best

    def tanimoto_sim(self, canon_a: str, canon_b: str) -> float:
        """Compute Tanimoto similarity between two canonical SMILES."""
        if not canon_a or not canon_b:
            return 0.0
        fp_a = self.get_fingerprint(canon_a)
        fp_b = self.get_fingerprint(canon_b)
        if fp_a is None or fp_b is None:
            return 0.0
        return DataStructs.TanimotoSimilarity(fp_a, fp_b)

    def best_train_match(self, gold_canon: str, exclude_canon: str = "") -> Tuple[float, str]:
        """Find the train product most similar to gold_canon, excluding exact matches.

        Returns (similarity, original_train_smiles) — but since we work with
        canonical SMILES, we return (similarity, canonical_train_smiles).
        """
        if not gold_canon:
            return (-1.0, "")
        fp_gold = self.get_fingerprint(gold_canon)
        if fp_gold is None:
            return (-1.0, "")
        best_sim = -1.0
        best_smi = ""
        for canon, _ in self._train_fps:
            if canon == gold_canon or canon == exclude_canon:
                continue
            fp = self.get_fingerprint(canon)
            if fp is None:
                continue
            sim = DataStructs.TanimotoSimilarity(fp_gold, fp)
            if sim > best_sim:
                best_sim = sim
                best_smi = canon
        return (best_sim, best_smi)


def _canonicalize_smiles_uncached(smiles: str) -> str:
    """Return canonical SMILES, or empty string on failure (no caching)."""
    if not RDKIT_AVAILABLE or not smiles:
        return smiles or ""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        return Chem.MolToSmiles(mol)
    except Exception:
        return ""


def _product_scaffold_uncached(smiles: str) -> str:
    """Compute Bemis-Murcko scaffold (no caching)."""
    if not RDKIT_AVAILABLE or not smiles:
        return ""
    try:
        from rdkit.Chem.Scaffolds import MurckoScaffold
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        return scaffold or ""
    except Exception:
        return ""


def _tanimoto_sim(smi_a: str, smi_b: str) -> float:
    """Compute Tanimoto similarity between two SMILES (uncached, for tests)."""
    if not RDKIT_AVAILABLE or not smi_a or not smi_b:
        return 0.0
    try:
        mol_a = Chem.MolFromSmiles(smi_a)
        mol_b = Chem.MolFromSmiles(smi_b)
        if mol_a is None or mol_b is None:
            return 0.0
        fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, 2, 1024)
        fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, 2, 1024)
        return DataStructs.TanimotoSimilarity(fp_a, fp_b)
    except Exception:
        return 0.0


def _product_scaffold(smiles: str) -> str:
    """Compute Bemis-Murcko scaffold of the product."""
    if not RDKIT_AVAILABLE or not smiles:
        return ""
    try:
        from rdkit.Chem.Scaffolds import MurckoScaffold
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        return scaffold or ""
    except Exception:
        return ""


def _has_atom_map(smiles: str) -> str:
    """Check if SMILES has atom mapping."""
    if not smiles:
        return "unknown"
    return "mapped" if ":" in smiles else "unmapped"


def _edit_distance(smi_a: str, smi_b: str) -> int:
    """Compute a simple edit distance proxy (character-level Levenshtein)."""
    if not smi_a or not smi_b:
        return -1
    # Simple character-level distance as a proxy
    a, b = smi_a, smi_b
    if len(a) > len(b):
        a, b = b, a
    prev = list(range(len(a) + 1))
    for i, cb in enumerate(b, 1):
        curr = [i]
        for j, ca in enumerate(a, 1):
            curr.append(min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + (ca != cb),
            ))
        prev = curr
    return prev[-1]


def _corrupt_smiles(smiles: str, rng: random.Random) -> str:
    """Create a corrupted version of a SMILES string."""
    if not smiles:
        return ""
    s = list(smiles)
    # Apply 1-3 random mutations
    n_mutations = rng.randint(1, 3)
    for _ in range(n_mutations):
        if not s:
            break
        op = rng.choice(["delete", "insert", "swap"])
        if op == "delete" and len(s) > 5:
            idx = rng.randint(0, len(s) - 1)
            s.pop(idx)
        elif op == "insert":
            idx = rng.randint(0, len(s))
            chars = "CNOSFclbr=#()"
            s.insert(idx, rng.choice(chars))
        elif op == "swap" and len(s) > 1:
            idx = rng.randint(0, len(s) - 2)
            s[idx], s[idx + 1] = s[idx + 1], s[idx]
    return "".join(s)


def _random_mismatch_product(products: List[str], gold: str, rng: random.Random) -> str:
    """Pick a random different product from the pool."""
    candidates = [p for p in products if p != gold]
    if not candidates:
        return gold
    return rng.choice(candidates)


def _strip_manifest_hash(obj: Any) -> Any:
    """Recursively remove ``manifest_hash`` fields from a nested structure.

    The manifest hash must be independent of the (nested) ``manifest_hash``
    fields stored in each candidate, otherwise backfilling the hash into
    candidates would invalidate the manifest hash itself.
    """
    if isinstance(obj, dict):
        return {k: _strip_manifest_hash(v) for k, v in obj.items() if k != "manifest_hash"}
    if isinstance(obj, list):
        return [_strip_manifest_hash(item) for item in obj]
    return obj


def _compute_manifest_hash(manifest: dict) -> str:
    """Compute SHA-256 hash of the manifest's canonical JSON.

    Both the top-level ``manifest_hash`` and any nested ``manifest_hash``
    fields inside candidate records are stripped before hashing, so the hash
    is stable regardless of whether candidate records have been backfilled
    with the manifest hash.
    """
    content = _strip_manifest_hash(manifest)
    canonical = json.dumps(content, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _make_candidate(
    benchmark_name: str,
    group_id: str,
    source_reaction_id: str,
    parent_reaction_id: str,
    experimental_group_id: str,
    split: str,
    candidate_smiles: str,
    candidate_source: str,
    candidate_source_rank: int,
    gold: bool,
    gold_smiles: str,
    reaction_family: str,
    reaction_template: str,
    train_products: List[str],
    manifest_hash: str,
    fp_cache: Optional["_FingerprintCache"] = None,
) -> dict:
    """Build a single candidate record with all 24 required fields.

    If ``fp_cache`` is provided, uses cached fingerprints for Tanimoto and
    scaffold computation (fast path). Otherwise falls back to uncached
    computation (slow, for tests/backward compatibility).
    """
    if fp_cache is not None:
        canonical = fp_cache.canonicalize(_strip_atom_map(candidate_smiles))
        gold_canonical = fp_cache.canonicalize(_strip_atom_map(gold_smiles))
        scaffold = fp_cache.get_scaffold(canonical)
        nearest_sim = fp_cache.tanimoto_against_train(canonical) if train_products else 0.0
        train_overlap = canonical in fp_cache.train_canonical_set() if train_products else False
    else:
        canonical = _canonicalize_smiles(_strip_atom_map(candidate_smiles))
        gold_canonical = _canonicalize_smiles(_strip_atom_map(gold_smiles))
        scaffold = _product_scaffold(canonical)
        nearest_sim = max((_tanimoto_sim(canonical, _canonicalize_smiles(_strip_atom_map(p))) for p in train_products[:200]), default=0.0) if train_products else 0.0
        train_overlap = canonical in {_canonicalize_smiles(_strip_atom_map(p)) for p in train_products} if train_products else False
    known_collision = canonical == gold_canonical and not gold
    edit_type = "none" if gold else candidate_source
    edit_dist = 0 if gold else _edit_distance(canonical, gold_canonical)
    oracle_cov = 1.0 if gold else 0.0

    return {
        "benchmark_name": benchmark_name,
        "group_id": group_id,
        "source_reaction_id": source_reaction_id,
        "parent_reaction_id": parent_reaction_id,
        "experimental_group_id": experimental_group_id,
        "gold_candidate": gold,
        "candidate_id": f"{group_id}_{candidate_source}_{candidate_source_rank}",
        "candidate_smiles": candidate_smiles,
        "candidate_source": candidate_source,
        "candidate_source_rank": candidate_source_rank,
        "canonical_smiles": canonical,
        "atom_mapping_status": _has_atom_map(candidate_smiles),
        "reaction_family": reaction_family,
        "reaction_template": reaction_template,
        "product_scaffold": scaffold,
        "edit_type": edit_type,
        "edit_distance": edit_dist,
        "train_overlap": train_overlap,
        "known_positive_collision": known_collision,
        "nearest_train_similarity": round(nearest_sim, 4),
        "split": split,
        "oracle_coverage": oracle_cov,
        "manifest_version": MANIFEST_VERSION,
        "manifest_hash": manifest_hash,
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_csv_rows(csv_path: Path, max_rows: Optional[int] = None) -> List[dict]:
    """Load rows from a CSV file as dicts."""
    import csv
    if not csv_path.exists():
        return []
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_rows and i >= max_rows:
                break
            rows.append(row)
    return rows


def _load_split_indices(repo_root: Path) -> dict:
    """Load existing split index files."""
    idx = {}
    for name in ["train_idx_v3", "val_idx_v3", "test_idx_v3"]:
        p = repo_root / "data" / "processed" / f"{name}.json"
        if p.exists():
            idx[name] = json.loads(p.read_text())
    return idx


# ---------------------------------------------------------------------------
# B1: HTE Feasibility Manifest
# ---------------------------------------------------------------------------

def build_hte_feasibility_manifest(repo_root: Path, rng: random.Random) -> dict:
    """Build B1: P4-HTE-Feasibility manifest.

    Uses HTEa data (39,546 reactions) with real experimental groups.
    Groups are formed by split_key (parent reaction identifier).
    """
    print("[B1] Building HTE-Feasibility manifest...")
    csv_path = repo_root / "data" / "processed" / "hitea_full_normalized.csv"
    rows = _load_csv_rows(csv_path)
    print(f"[B1] Loaded {len(rows)} HTEa reactions")

    # Group by split_key (experimental group identifier)
    groups_map: Dict[str, List[dict]] = {}
    for row in rows:
        sk = row.get("split_key", "")
        if not sk:
            continue
        groups_map.setdefault(sk, []).append(row)

    # Select up to MAX_GROUPS_HTE groups, prioritizing test split
    all_group_keys = sorted(groups_map.keys())
    rng.shuffle(all_group_keys)
    selected_keys = all_group_keys[:MAX_GROUPS_HTE]

    # Build train product pool for nearest-neighbor computation
    train_products = [r["products"] for r in rows if r.get("split") == "train"]

    fp_cache = _FingerprintCache()
    fp_cache.prepare_train_set(train_products, max_n=500)

    # Use a placeholder hash first, fill in later
    placeholder_hash = "PENDING"

    groups: List[dict] = []
    for gk in selected_keys:
        group_rows = groups_map[gk]
        if not group_rows:
            continue
        # Pick the first row as the gold reaction
        gold_row = group_rows[0]
        gold_smiles = gold_row.get("products", "")
        split = gold_row.get("split", "train")
        reaction_family = gold_row.get("reaction_class", "unknown")
        source_id = gold_row.get("source_id", "")
        experimental_group = gk

        group_id = f"hte_{gk}"
        candidates: List[dict] = []

        # Gold candidate
        candidates.append(_make_candidate(
            benchmark_name="P4-HTE-Feasibility",
            group_id=group_id,
            source_reaction_id=source_id,
            parent_reaction_id=gk,
            experimental_group_id=experimental_group,
            split=split,
            candidate_smiles=gold_smiles,
            candidate_source="gold",
            candidate_source_rank=0,
            gold=True,
            gold_smiles=gold_smiles,
            reaction_family=reaction_family,
            reaction_template="HTEa",
            train_products=train_products,
            manifest_hash=placeholder_hash,
            fp_cache=fp_cache,
        ))

        # random_mismatch
        mismatch = _random_mismatch_product(train_products, gold_smiles, rng)
        candidates.append(_make_candidate(
            benchmark_name="P4-HTE-Feasibility",
            group_id=group_id,
            source_reaction_id=source_id,
            parent_reaction_id=gk,
            experimental_group_id=experimental_group,
            split=split,
            candidate_smiles=mismatch,
            candidate_source="random_mismatch",
            candidate_source_rank=1,
            gold=False,
            gold_smiles=gold_smiles,
            reaction_family=reaction_family,
            reaction_template="HTEa",
            train_products=train_products,
            manifest_hash=placeholder_hash,
            fp_cache=fp_cache,
        ))

        # random_corruption
        corrupted = _corrupt_smiles(gold_smiles, rng)
        candidates.append(_make_candidate(
            benchmark_name="P4-HTE-Feasibility",
            group_id=group_id,
            source_reaction_id=source_id,
            parent_reaction_id=gk,
            experimental_group_id=experimental_group,
            split=split,
            candidate_smiles=corrupted,
            candidate_source="random_corruption",
            candidate_source_rank=2,
            gold=False,
            gold_smiles=gold_smiles,
            reaction_family=reaction_family,
            reaction_template="HTEa",
            train_products=train_products,
            manifest_hash=placeholder_hash,
            fp_cache=fp_cache,
        ))

        # tanimoto_retrieval — pick the most similar product from train
        gold_canon = fp_cache.canonicalize(_strip_atom_map(gold_smiles))
        best_sim, best_smi_canon = fp_cache.best_train_match(gold_canon)
        if best_smi_canon:
            best_smi = best_smi_canon
        else:
            best_smi = mismatch
        candidates.append(_make_candidate(
            benchmark_name="P4-HTE-Feasibility",
            group_id=group_id,
            source_reaction_id=source_id,
            parent_reaction_id=gk,
            experimental_group_id=experimental_group,
            split=split,
            candidate_smiles=best_smi,
            candidate_source="tanimoto_retrieval",
            candidate_source_rank=3,
            gold=False,
            gold_smiles=gold_smiles,
            reaction_family=reaction_family,
            reaction_template="HTEa",
            train_products=train_products,
            manifest_hash=placeholder_hash,
            fp_cache=fp_cache,
        ))

        # template_perturbation — pick a product from the same reaction_class
        same_class = [r["products"] for r in rows if r.get("reaction_class") == reaction_family and r.get("products") != gold_smiles]
        if same_class:
            tp = rng.choice(same_class)
        else:
            tp = mismatch
        candidates.append(_make_candidate(
            benchmark_name="P4-HTE-Feasibility",
            group_id=group_id,
            source_reaction_id=source_id,
            parent_reaction_id=gk,
            experimental_group_id=experimental_group,
            split=split,
            candidate_smiles=tp,
            candidate_source="template_perturbation",
            candidate_source_rank=4,
            gold=False,
            gold_smiles=gold_smiles,
            reaction_family=reaction_family,
            reaction_template="HTEa",
            train_products=train_products,
            manifest_hash=placeholder_hash,
            fp_cache=fp_cache,
        ))

        # unconstrained_edit — random atom deletion from gold
        unmapped = _strip_atom_map(gold_smiles)
        if len(unmapped) > 10:
            edit_smi = unmapped[:rng.randint(5, len(unmapped) - 1)]
        else:
            edit_smi = corrupted
        candidates.append(_make_candidate(
            benchmark_name="P4-HTE-Feasibility",
            group_id=group_id,
            source_reaction_id=source_id,
            parent_reaction_id=gk,
            experimental_group_id=experimental_group,
            split=split,
            candidate_smiles=edit_smi,
            candidate_source="unconstrained_edit",
            candidate_source_rank=5,
            gold=False,
            gold_smiles=gold_smiles,
            reaction_family=reaction_family,
            reaction_template="HTEa",
            train_products=train_products,
            manifest_hash=placeholder_hash,
            fp_cache=fp_cache,
        ))

        # rule_pc_cng — simple functional group substitution
        rule_smi = gold_smiles.replace("C(=O)O", "C(=O)N").replace("[Br]", "[Cl]").replace("[F]", "[Cl]")
        if rule_smi == gold_smiles:
            rule_smi = corrupted
        candidates.append(_make_candidate(
            benchmark_name="P4-HTE-Feasibility",
            group_id=group_id,
            source_reaction_id=source_id,
            parent_reaction_id=gk,
            experimental_group_id=experimental_group,
            split=split,
            candidate_smiles=rule_smi,
            candidate_source="rule_pc_cng",
            candidate_source_rank=6,
            gold=False,
            gold_smiles=gold_smiles,
            reaction_family=reaction_family,
            reaction_template="HTEa",
            train_products=train_products,
            manifest_hash=placeholder_hash,
            fp_cache=fp_cache,
        ))

        # external_beam — use a product from ORD as external source
        external_smi = mismatch  # fallback
        ord_path = repo_root / "data" / "processed" / "ord_normalized.csv"
        if ord_path.exists():
            ord_rows = _load_csv_rows(ord_path, max_rows=100)
            if ord_rows:
                external_smi = rng.choice(ord_rows)["products"]
        candidates.append(_make_candidate(
            benchmark_name="P4-HTE-Feasibility",
            group_id=group_id,
            source_reaction_id=source_id,
            parent_reaction_id=gk,
            experimental_group_id=experimental_group,
            split=split,
            candidate_smiles=external_smi,
            candidate_source="external_beam",
            candidate_source_rank=7,
            gold=False,
            gold_smiles=gold_smiles,
            reaction_family=reaction_family,
            reaction_template="HTEa",
            train_products=train_products,
            manifest_hash=placeholder_hash,
            fp_cache=fp_cache,
        ))

        groups.append({
            "group_id": group_id,
            "source_reaction_id": source_id,
            "parent_reaction_id": gk,
            "experimental_group_id": experimental_group,
            "split": split,
            "candidates": candidates,
        })

    manifest = {
        "benchmark_name": "P4-HTE-Feasibility",
        "manifest_version": MANIFEST_VERSION,
        "manifest_hash": "",  # filled below
        "groups": groups,
    }
    manifest["manifest_hash"] = _compute_manifest_hash(manifest)

    # Backfill the manifest_hash into each candidate
    for g in groups:
        for c in g["candidates"]:
            c["manifest_hash"] = manifest["manifest_hash"]

    print(f"[B1] Built {len(groups)} groups, {sum(len(g['candidates']) for g in groups)} candidates")
    return manifest


# ---------------------------------------------------------------------------
# B2: Fixed Forward Candidates Manifest
# ---------------------------------------------------------------------------

def build_forward_candidates_manifest(repo_root: Path, rng: random.Random) -> dict:
    """Build B2: P4-Fixed-Forward-Candidates manifest.

    Uses USPTO-OM data. For each reaction, the gold candidate is the actual
    product; non-gold candidates are generated from multiple sources.
    """
    print("[B2] Building Fixed-Forward-Candidates manifest...")
    csv_path = repo_root / "data" / "processed" / "uspto_openmolecules_normalized.csv"
    rows = _load_csv_rows(csv_path, max_rows=5000)
    print(f"[B2] Loaded {len(rows)} USPTO-OM reactions (sampled)")

    # Filter to rows that have products
    valid_rows = [r for r in rows if r.get("products") and r.get("split")]
    rng.shuffle(valid_rows)
    selected = valid_rows[:MAX_GROUPS_FORWARD]

    train_products = [r["products"] for r in rows if r.get("split") == "train"]

    fp_cache = _FingerprintCache()
    fp_cache.prepare_train_set(train_products, max_n=500)

    placeholder_hash = "PENDING"

    groups: List[dict] = []
    for row in selected:
        gold_smiles = row.get("products", "")
        split = row.get("split", "train")
        source_id = row.get("source_id", "")
        split_key = row.get("split_key", source_id)
        group_id = f"fwd_{source_id}"

        candidates: List[dict] = []
        # Gold
        candidates.append(_make_candidate(
            benchmark_name="P4-Fixed-Forward-Candidates",
            group_id=group_id,
            source_reaction_id=source_id,
            parent_reaction_id=split_key,
            experimental_group_id=split_key,
            split=split,
            candidate_smiles=gold_smiles,
            candidate_source="gold",
            candidate_source_rank=0,
            gold=True,
            gold_smiles=gold_smiles,
            reaction_family="uspto_om",
            reaction_template="forward",
            train_products=train_products,
            manifest_hash=placeholder_hash,
            fp_cache=fp_cache,
        ))
        # random_mismatch
        mismatch = _random_mismatch_product(train_products, gold_smiles, rng)
        candidates.append(_make_candidate(
            benchmark_name="P4-Fixed-Forward-Candidates",
            group_id=group_id, source_reaction_id=source_id, parent_reaction_id=split_key,
            experimental_group_id=split_key, split=split,
            candidate_smiles=mismatch, candidate_source="random_mismatch",
            candidate_source_rank=1, gold=False, gold_smiles=gold_smiles,
            reaction_family="uspto_om", reaction_template="forward",
            train_products=train_products, manifest_hash=placeholder_hash, fp_cache=fp_cache,
        ))
        # random_corruption
        corrupted = _corrupt_smiles(gold_smiles, rng)
        candidates.append(_make_candidate(
            benchmark_name="P4-Fixed-Forward-Candidates",
            group_id=group_id, source_reaction_id=source_id, parent_reaction_id=split_key,
            experimental_group_id=split_key, split=split,
            candidate_smiles=corrupted, candidate_source="random_corruption",
            candidate_source_rank=2, gold=False, gold_smiles=gold_smiles,
            reaction_family="uspto_om", reaction_template="forward",
            train_products=train_products, manifest_hash=placeholder_hash, fp_cache=fp_cache,
        ))
        # tanimoto_retrieval
        gold_canon = fp_cache.canonicalize(_strip_atom_map(gold_smiles))
        best_sim, best_smi_canon = fp_cache.best_train_match(gold_canon)
        if best_smi_canon:
            best_smi = best_smi_canon
        else:
            best_smi = mismatch
        candidates.append(_make_candidate(
            benchmark_name="P4-Fixed-Forward-Candidates",
            group_id=group_id, source_reaction_id=source_id, parent_reaction_id=split_key,
            experimental_group_id=split_key, split=split,
            candidate_smiles=best_smi, candidate_source="tanimoto_retrieval",
            candidate_source_rank=3, gold=False, gold_smiles=gold_smiles,
            reaction_family="uspto_om", reaction_template="forward",
            train_products=train_products, manifest_hash=placeholder_hash, fp_cache=fp_cache,
        ))
        # template_perturbation
        tp = rng.choice(train_products) if train_products else mismatch
        candidates.append(_make_candidate(
            benchmark_name="P4-Fixed-Forward-Candidates",
            group_id=group_id, source_reaction_id=source_id, parent_reaction_id=split_key,
            experimental_group_id=split_key, split=split,
            candidate_smiles=tp, candidate_source="template_perturbation",
            candidate_source_rank=4, gold=False, gold_smiles=gold_smiles,
            reaction_family="uspto_om", reaction_template="forward",
            train_products=train_products, manifest_hash=placeholder_hash, fp_cache=fp_cache,
        ))
        # unconstrained_edit
        unmapped = _strip_atom_map(gold_smiles)
        edit_smi = unmapped[:max(5, len(unmapped) // 2)] if len(unmapped) > 10 else corrupted
        candidates.append(_make_candidate(
            benchmark_name="P4-Fixed-Forward-Candidates",
            group_id=group_id, source_reaction_id=source_id, parent_reaction_id=split_key,
            experimental_group_id=split_key, split=split,
            candidate_smiles=edit_smi, candidate_source="unconstrained_edit",
            candidate_source_rank=5, gold=False, gold_smiles=gold_smiles,
            reaction_family="uspto_om", reaction_template="forward",
            train_products=train_products, manifest_hash=placeholder_hash, fp_cache=fp_cache,
        ))
        # rule_pc_cng
        rule_smi = gold_smiles.replace("C(=O)O", "C(=O)N").replace("[Br]", "[Cl]")
        if rule_smi == gold_smiles:
            rule_smi = corrupted
        candidates.append(_make_candidate(
            benchmark_name="P4-Fixed-Forward-Candidates",
            group_id=group_id, source_reaction_id=source_id, parent_reaction_id=split_key,
            experimental_group_id=split_key, split=split,
            candidate_smiles=rule_smi, candidate_source="rule_pc_cng",
            candidate_source_rank=6, gold=False, gold_smiles=gold_smiles,
            reaction_family="uspto_om", reaction_template="forward",
            train_products=train_products, manifest_hash=placeholder_hash, fp_cache=fp_cache,
        ))
        # external_beam — use HTEa product as external
        external_smi = mismatch
        hitea_path = repo_root / "data" / "processed" / "hitea_full_normalized.csv"
        if hitea_path.exists():
            hitea_rows = _load_csv_rows(hitea_path, max_rows=100)
            if hitea_rows:
                external_smi = rng.choice(hitea_rows)["products"]
        candidates.append(_make_candidate(
            benchmark_name="P4-Fixed-Forward-Candidates",
            group_id=group_id, source_reaction_id=source_id, parent_reaction_id=split_key,
            experimental_group_id=split_key, split=split,
            candidate_smiles=external_smi, candidate_source="external_beam",
            candidate_source_rank=7, gold=False, gold_smiles=gold_smiles,
            reaction_family="uspto_om", reaction_template="forward",
            train_products=train_products, manifest_hash=placeholder_hash, fp_cache=fp_cache,
        ))

        groups.append({
            "group_id": group_id,
            "source_reaction_id": source_id,
            "parent_reaction_id": split_key,
            "experimental_group_id": split_key,
            "split": split,
            "candidates": candidates,
        })

    manifest = {
        "benchmark_name": "P4-Fixed-Forward-Candidates",
        "manifest_version": MANIFEST_VERSION,
        "manifest_hash": "",
        "groups": groups,
    }
    manifest["manifest_hash"] = _compute_manifest_hash(manifest)
    for g in groups:
        for c in g["candidates"]:
            c["manifest_hash"] = manifest["manifest_hash"]

    print(f"[B2] Built {len(groups)} groups, {sum(len(g['candidates']) for g in groups)} candidates")
    return manifest


# ---------------------------------------------------------------------------
# B3: Fixed Retro Candidates Manifest
# ---------------------------------------------------------------------------

def build_retro_candidates_manifest(repo_root: Path, rng: random.Random) -> dict:
    """Build B3: P4-Fixed-Retro-Candidates manifest.

    Uses USPTO-OM data. For each reaction, the gold candidate is the actual
    reactant set; non-gold candidates are generated from multiple sources.
    """
    print("[B3] Building Fixed-Retro-Candidates manifest...")
    csv_path = repo_root / "data" / "processed" / "uspto_openmolecules_normalized.csv"
    rows = _load_csv_rows(csv_path, max_rows=5000)
    print(f"[B3] Loaded {len(rows)} USPTO-OM reactions (sampled)")

    valid_rows = [r for r in rows if r.get("reactants") and r.get("split")]
    rng.shuffle(valid_rows)
    selected = valid_rows[:MAX_GROUPS_RETRO]

    train_reactants = [r["reactants"] for r in rows if r.get("split") == "train"]

    fp_cache = _FingerprintCache()
    fp_cache.prepare_train_set(train_reactants, max_n=500)

    placeholder_hash = "PENDING"

    groups: List[dict] = []
    for row in selected:
        gold_smiles = row.get("reactants", "")
        split = row.get("split", "train")
        source_id = row.get("source_id", "")
        split_key = row.get("split_key", source_id)
        group_id = f"retro_{source_id}"

        candidates: List[dict] = []
        # Gold
        candidates.append(_make_candidate(
            benchmark_name="P4-Fixed-Retro-Candidates",
            group_id=group_id,
            source_reaction_id=source_id,
            parent_reaction_id=split_key,
            experimental_group_id=split_key,
            split=split,
            candidate_smiles=gold_smiles,
            candidate_source="gold",
            candidate_source_rank=0,
            gold=True,
            gold_smiles=gold_smiles,
            reaction_family="uspto_om",
            reaction_template="retro",
            train_products=train_reactants,
            manifest_hash=placeholder_hash,
            fp_cache=fp_cache,
        ))
        # random_mismatch
        mismatch = _random_mismatch_product(train_reactants, gold_smiles, rng)
        candidates.append(_make_candidate(
            benchmark_name="P4-Fixed-Retro-Candidates",
            group_id=group_id, source_reaction_id=source_id, parent_reaction_id=split_key,
            experimental_group_id=split_key, split=split,
            candidate_smiles=mismatch, candidate_source="random_mismatch",
            candidate_source_rank=1, gold=False, gold_smiles=gold_smiles,
            reaction_family="uspto_om", reaction_template="retro",
            train_products=train_reactants, manifest_hash=placeholder_hash, fp_cache=fp_cache,
        ))
        # random_corruption
        corrupted = _corrupt_smiles(gold_smiles, rng)
        candidates.append(_make_candidate(
            benchmark_name="P4-Fixed-Retro-Candidates",
            group_id=group_id, source_reaction_id=source_id, parent_reaction_id=split_key,
            experimental_group_id=split_key, split=split,
            candidate_smiles=corrupted, candidate_source="random_corruption",
            candidate_source_rank=2, gold=False, gold_smiles=gold_smiles,
            reaction_family="uspto_om", reaction_template="retro",
            train_products=train_reactants, manifest_hash=placeholder_hash, fp_cache=fp_cache,
        ))
        # tanimoto_retrieval
        gold_canon = fp_cache.canonicalize(_strip_atom_map(gold_smiles))
        best_sim, best_smi_canon = fp_cache.best_train_match(gold_canon)
        if best_smi_canon:
            best_smi = best_smi_canon
        else:
            best_smi = mismatch
        candidates.append(_make_candidate(
            benchmark_name="P4-Fixed-Retro-Candidates",
            group_id=group_id, source_reaction_id=source_id, parent_reaction_id=split_key,
            experimental_group_id=split_key, split=split,
            candidate_smiles=best_smi, candidate_source="tanimoto_retrieval",
            candidate_source_rank=3, gold=False, gold_smiles=gold_smiles,
            reaction_family="uspto_om", reaction_template="retro",
            train_products=train_reactants, manifest_hash=placeholder_hash, fp_cache=fp_cache,
        ))
        # template_perturbation
        tp = rng.choice(train_reactants) if train_reactants else mismatch
        candidates.append(_make_candidate(
            benchmark_name="P4-Fixed-Retro-Candidates",
            group_id=group_id, source_reaction_id=source_id, parent_reaction_id=split_key,
            experimental_group_id=split_key, split=split,
            candidate_smiles=tp, candidate_source="template_perturbation",
            candidate_source_rank=4, gold=False, gold_smiles=gold_smiles,
            reaction_family="uspto_om", reaction_template="retro",
            train_products=train_reactants, manifest_hash=placeholder_hash, fp_cache=fp_cache,
        ))
        # unconstrained_edit
        unmapped = _strip_atom_map(gold_smiles)
        edit_smi = unmapped[:max(5, len(unmapped) // 2)] if len(unmapped) > 10 else corrupted
        candidates.append(_make_candidate(
            benchmark_name="P4-Fixed-Retro-Candidates",
            group_id=group_id, source_reaction_id=source_id, parent_reaction_id=split_key,
            experimental_group_id=split_key, split=split,
            candidate_smiles=edit_smi, candidate_source="unconstrained_edit",
            candidate_source_rank=5, gold=False, gold_smiles=gold_smiles,
            reaction_family="uspto_om", reaction_template="retro",
            train_products=train_reactants, manifest_hash=placeholder_hash, fp_cache=fp_cache,
        ))
        # rule_pc_cng
        rule_smi = gold_smiles.replace("C(=O)O", "C(=O)N").replace("[Br]", "[Cl]")
        if rule_smi == gold_smiles:
            rule_smi = corrupted
        candidates.append(_make_candidate(
            benchmark_name="P4-Fixed-Retro-Candidates",
            group_id=group_id, source_reaction_id=source_id, parent_reaction_id=split_key,
            experimental_group_id=split_key, split=split,
            candidate_smiles=rule_smi, candidate_source="rule_pc_cng",
            candidate_source_rank=6, gold=False, gold_smiles=gold_smiles,
            reaction_family="uspto_om", reaction_template="retro",
            train_products=train_reactants, manifest_hash=placeholder_hash, fp_cache=fp_cache,
        ))
        # external_beam
        external_smi = mismatch
        ord_path = repo_root / "data" / "processed" / "ord_normalized.csv"
        if ord_path.exists():
            ord_rows = _load_csv_rows(ord_path, max_rows=100)
            if ord_rows:
                external_smi = rng.choice(ord_rows)["reactants"]
        candidates.append(_make_candidate(
            benchmark_name="P4-Fixed-Retro-Candidates",
            group_id=group_id, source_reaction_id=source_id, parent_reaction_id=split_key,
            experimental_group_id=split_key, split=split,
            candidate_smiles=external_smi, candidate_source="external_beam",
            candidate_source_rank=7, gold=False, gold_smiles=gold_smiles,
            reaction_family="uspto_om", reaction_template="retro",
            train_products=train_reactants, manifest_hash=placeholder_hash, fp_cache=fp_cache,
        ))

        groups.append({
            "group_id": group_id,
            "source_reaction_id": source_id,
            "parent_reaction_id": split_key,
            "experimental_group_id": split_key,
            "split": split,
            "candidates": candidates,
        })

    manifest = {
        "benchmark_name": "P4-Fixed-Retro-Candidates",
        "manifest_version": MANIFEST_VERSION,
        "manifest_hash": "",
        "groups": groups,
    }
    manifest["manifest_hash"] = _compute_manifest_hash(manifest)
    for g in groups:
        for c in g["candidates"]:
            c["manifest_hash"] = manifest["manifest_hash"]

    print(f"[B3] Built {len(groups)} groups, {sum(len(g['candidates']) for g in groups)} candidates")
    return manifest


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_all_manifests(repo_root: Path, output_dir: Path) -> dict:
    """Build all three manifests and write to output_dir."""
    rng = random.Random(MANIFEST_SEED)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Also create splits directory
    splits_dir = repo_root / "data" / "p4" / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    manifests = {}
    # B1
    b1 = build_hte_feasibility_manifest(repo_root, rng)
    (output_dir / "hte_feasibility_v1.json").write_text(
        json.dumps(b1, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    manifests["hte_feasibility"] = b1

    # B2
    b2 = build_forward_candidates_manifest(repo_root, rng)
    (output_dir / "fixed_forward_candidates_v1.json").write_text(
        json.dumps(b2, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    manifests["fixed_forward"] = b2

    # B3
    b3 = build_retro_candidates_manifest(repo_root, rng)
    (output_dir / "fixed_retro_candidates_v1.json").write_text(
        json.dumps(b3, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    manifests["fixed_retro"] = b3

    # Write a summary split file
    split_summary = {
        "manifest_version": MANIFEST_VERSION,
        "seed": MANIFEST_SEED,
        "benchmarks": {
            name: {
                "manifest_hash": m["manifest_hash"],
                "n_groups": len(m["groups"]),
                "n_candidates": sum(len(g["candidates"]) for g in m["groups"]),
            }
            for name, m in manifests.items()
        },
    }
    (splits_dir / "split_summary_v1.json").write_text(
        json.dumps(split_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return split_summary


def main():
    parser = argparse.ArgumentParser(
        description="P4-G1 Benchmark Contract & Candidate Manifest Construction"
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory for manifest output (e.g. data/p4/manifests)"
    )
    parser.add_argument(
        "--repo-root", default=".",
        help="Root of the pc_cng_research repository (default: current dir)"
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not repo_root.exists():
        print(f"ERROR: repo root not found: {repo_root}", file=sys.stderr)
        sys.exit(1)

    if not RDKIT_AVAILABLE:
        print("WARNING: RDKit not available; SMILES canonicalization will be skipped", file=sys.stderr)

    print(f"[P4-G1] Building candidate manifests")
    print(f"[P4-G1] Repo root: {repo_root}")
    print(f"[P4-G1] Output dir: {output_dir}")

    summary = build_all_manifests(repo_root, output_dir)

    print(f"\n[P4-G1] Manifest construction complete:")
    for name, info in summary["benchmarks"].items():
        print(f"  {name}: {info['n_groups']} groups, {info['n_candidates']} candidates, hash={info['manifest_hash'][:16]}...")


if __name__ == "__main__":
    main()
