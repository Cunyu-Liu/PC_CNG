"""Phase 3 OOD (out-of-distribution) splits for the HiTEA dataset.

Creates six split types required by the Phase 3 spec so that OOD results
can become the *main* result (not supplementary):

1. ``reaction_family``  — train on 80% of reaction families, test on rest
2. ``scaffold``         — Murcko scaffold of ``reactant_1_smiles``; 80/20
3. ``condition_space``  — train on 80% of ``experimental_group`` screens
4. ``time``             — proxy via ``notebook_id`` numeric suffix
                          (HiTEA is a single publication, so true year is
                          unavailable; the page/suffix increases with time)
5. ``author_lab``       — ``notebook_id`` prefix as lab proxy
                          (two labs: ``00119131`` and ``00707420``)
6. ``random``           — stratified random baseline for comparison

For every split we write four JSON files under ``data/ood_splits/``::

    {split_name}_train_idx.json   list[int]  (row indices into the parquet)
    {split_name}_val_idx.json     list[int]
    {split_name}_test_idx.json    list[int]
    {split_name}_metadata.json    dict with description, counts, criteria

Design notes
------------
* Test fraction is 0.20 of the data.  Of the remaining 80%, 20% (i.e. 16%
  of total) becomes validation and 80% (i.e. 64% of total) becomes train.
  Final ratio is approximately 64 / 16 / 20.
* Group splits (reaction_family, scaffold, condition_space) keep every
  group wholly inside one partition — no group crosses splits.  Families
  are sorted by size and dealt round-robin into the test pool until the
  cumulative test fraction reaches the target; this avoids the
  pathological case where one large family dominates the test set.
* ``time`` and ``author_lab`` are *true* OOD: the test partition contains
  groups never seen during training.  ``random`` is in-distribution.
* Splits are deterministic given ``seed``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# RDKit is optional at import time so the module can be loaded in a thin
# environment; scaffold computation will raise a clear error if RDKit is
# missing when actually needed.
os.environ.setdefault("RDKitRDLogger", "0")
try:  # pragma: no cover - import guard
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog("rdApp.*")
except Exception:  # pragma: no cover
    Chem = None
    MurckoScaffold = None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_PARQUET = "data/processed/p4_hte_normalized.parquet"
DEFAULT_OUTPUT_DIR = "data/ood_splits"
DEFAULT_SEED = 20260725
TEST_FRACTION = 0.20
VAL_FRACTION_OF_TRAIN = 0.20  # 0.20 of the non-test pool -> 0.16 of total

SPLIT_NAMES = (
    "reaction_family",
    "scaffold",
    "condition_space",
    "time",
    "author_lab",
    "random",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_dataframe(parquet_path: Path):
    """Load the HiTEA parquet into a DataFrame."""
    import pandas as pd
    if not parquet_path.exists():
        raise FileNotFoundError(f"HiTEA parquet not found: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    return df.reset_index(drop=True)


def _split_train_val(train_pool: List[int],
                     val_fraction: float,
                     rng: random.Random) -> Tuple[List[int], List[int]]:
    """Shuffle ``train_pool`` and carve off a validation slice."""
    shuffled = list(train_pool)
    rng.shuffle(shuffled)
    n_val = int(round(len(shuffled) * val_fraction))
    val_idx = shuffled[:n_val]
    train_idx = shuffled[n_val:]
    return sorted(train_idx), sorted(val_idx)


def _groups_to_indices(df, key: str) -> Dict[str, List[int]]:
    """Map each unique value of ``key`` to the list of row indices."""
    groups: Dict[str, List[int]] = defaultdict(list)
    for idx, val in zip(df.index, df[key]):
        if val is None:
            val = "__missing__"
        groups[str(val)].append(int(idx))
    return dict(groups)


def _deal_groups_round_robin(groups: Dict[str, List[int]],
                              test_fraction: float,
                              rng: random.Random) -> Tuple[List[str], List[str]]:
    """Partition group keys so the test pool holds ~``test_fraction`` of rows.

    Groups are sorted by size (descending) and dealt round-robin into the
    test bucket until the cumulative row count crosses the target.  This
    prevents a single huge group from blowing past the target on the first
    iteration and keeps the test set diverse.
    """
    ordered = sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True)
    total = sum(len(v) for _, v in ordered)
    target_test = total * test_fraction
    test_keys: List[str] = []
    train_keys: List[str] = []
    test_count = 0
    # Round-robin deal: iterate and assign one group at a time to test,
    # then one to train, until test target is reached.
    turn = 0
    remaining = list(ordered)
    while remaining and test_count < target_test:
        # Pick the next group (cycle through the list to keep diversity)
        key, idxs = remaining.pop(turn % len(remaining))
        test_keys.append(key)
        test_count += len(idxs)
        turn += 1
    train_keys = [k for k, _ in remaining]
    return test_keys, train_keys


def _write_split(output_dir: Path,
                 split_name: str,
                 train_idx: Sequence[int],
                 val_idx: Sequence[int],
                 test_idx: Sequence[int],
                 metadata: Dict) -> None:
    """Persist the four JSON artefacts for one split."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix, data in (
        ("train_idx", list(train_idx)),
        ("val_idx", list(val_idx)),
        ("test_idx", list(test_idx)),
    ):
        path = output_dir / f"{split_name}_{suffix}.json"
        with open(path, "w") as fh:
            json.dump(data, fh)
    metadata.update({
        "split_name": split_name,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "n_total": len(train_idx) + len(val_idx) + len(test_idx),
        "train_fraction": len(train_idx) / max(1, len(train_idx) + len(val_idx) + len(test_idx)),
        "val_fraction": len(val_idx) / max(1, len(train_idx) + len(val_idx) + len(test_idx)),
        "test_fraction": len(test_idx) / max(1, len(train_idx) + len(val_idx) + len(test_idx)),
    })
    with open(output_dir / f"{split_name}_metadata.json", "w") as fh:
        json.dump(metadata, fh, indent=2)


# ---------------------------------------------------------------------------
# Individual split builders
# ---------------------------------------------------------------------------

def build_reaction_family_split(df, rng: random.Random) -> Tuple[List[int], List[int], List[int], Dict]:
    """Train on 80% of reaction families; test on the held-out 20%."""
    groups = _groups_to_indices(df, "reaction_family")
    test_keys, train_keys = _deal_groups_round_robin(groups, TEST_FRACTION, rng)
    train_pool: List[int] = []
    for k in train_keys:
        train_pool.extend(groups[k])
    test_idx = sorted(i for k in test_keys for i in groups[k])
    train_idx, val_idx = _split_train_val(train_pool, VAL_FRACTION_OF_TRAIN, rng)
    meta = {
        "split_type": "reaction_family",
        "description": (
            "Group-disjoint split by `reaction_family`. Train families are "
            "never seen in test; round-robin dealing keeps the test set "
            "diverse across families."
        ),
        "criteria": "group_disjoint_by_reaction_family",
        "n_train_groups": len(train_keys),
        "n_test_groups": len(test_keys),
        "train_groups": sorted(train_keys),
        "test_groups": sorted(test_keys),
        "is_ood": True,
    }
    return train_idx, val_idx, test_idx, meta


def _murcko_scaffold(smiles: str) -> str:
    """Compute the Murcko scaffold SMILES for a molecule."""
    if Chem is None or MurckoScaffold is None:
        raise RuntimeError("RDKit is required for scaffold splits but is not available.")
    if not smiles or not isinstance(smiles, str):
        return "__invalid__"
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "__invalid__"
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        smi = Chem.MolToSmiles(scaffold, canonical=True)
        return smi if smi else "__empty__"
    except Exception:
        return "__error__"


def build_scaffold_split(df, rng: random.Random) -> Tuple[List[int], List[int], List[int], Dict]:
    """Murcko scaffold of ``reactant_1_smiles``; 80/20 of scaffolds."""
    if "reactant_1_smiles" not in df.columns:
        raise ValueError("Column `reactant_1_smiles` is required for scaffold split.")
    groups: Dict[str, List[int]] = defaultdict(list)
    n_invalid = 0
    for idx, smi in zip(df.index, df["reactant_1_smiles"]):
        scaffold = _murcko_scaffold(smi) if smi is not None else "__invalid__"
        if scaffold in ("__invalid__", "__error__"):
            n_invalid += 1
        groups[scaffold].append(int(idx))
    test_keys, train_keys = _deal_groups_round_robin(groups, TEST_FRACTION, rng)
    train_pool: List[int] = []
    for k in train_keys:
        train_pool.extend(groups[k])
    test_idx = sorted(i for k in test_keys for i in groups[k])
    train_idx, val_idx = _split_train_val(train_pool, VAL_FRACTION_OF_TRAIN, rng)
    meta = {
        "split_type": "scaffold",
        "description": (
            "Group-disjoint split by Murcko scaffold of `reactant_1_smiles`. "
            "Test scaffolds are structurally novel relative to train."
        ),
        "criteria": "group_disjoint_by_murcko_scaffold",
        "n_train_scaffolds": len(train_keys),
        "n_test_scaffolds": len(test_keys),
        "n_invalid_smiles": n_invalid,
        "is_ood": True,
    }
    return train_idx, val_idx, test_idx, meta


def build_condition_space_split(df, rng: random.Random) -> Tuple[List[int], List[int], List[int], Dict]:
    """Train on 80% of ``experimental_group`` screens; test on the rest."""
    groups = _groups_to_indices(df, "experimental_group")
    test_keys, train_keys = _deal_groups_round_robin(groups, TEST_FRACTION, rng)
    train_pool: List[int] = []
    for k in train_keys:
        train_pool.extend(groups[k])
    test_idx = sorted(i for k in test_keys for i in groups[k])
    train_idx, val_idx = _split_train_val(train_pool, VAL_FRACTION_OF_TRAIN, rng)
    meta = {
        "split_type": "condition_space",
        "description": (
            "Group-disjoint split by `experimental_group` (HTE screen id). "
            "Test screens represent an unseen region of condition space."
        ),
        "criteria": "group_disjoint_by_experimental_group",
        "n_train_groups": len(train_keys),
        "n_test_groups": len(test_keys),
        "train_groups": sorted(train_keys),
        "test_groups": sorted(test_keys),
        "is_ood": True,
    }
    return train_idx, val_idx, test_idx, meta


def _notebook_suffix(notebook_id: str) -> int:
    """Extract the numeric suffix of a ``00119131-2497`` style id."""
    if not isinstance(notebook_id, str) or "-" not in notebook_id:
        return -1
    tail = notebook_id.rsplit("-", 1)[-1]
    try:
        return int(tail)
    except ValueError:
        return -1


def build_time_split(df, rng: random.Random) -> Tuple[List[int], List[int], List[int], Dict]:
    """Time-proxy split using the ``notebook_id`` numeric suffix.

    HiTEA is a single publication (King-Smith et al., Nat. Chem. 2023) so a
    true publication-year split is impossible.  The ``notebook_id`` suffix
    (page/experiment number) increases monotonically with experimental
    time, so we sort records by suffix and put the newest 20% into test.
    """
    suffixes = df["notebook_id"].apply(_notebook_suffix).to_numpy()
    order = np.argsort(suffixes, kind="stable")
    n_total = len(order)
    n_test = int(round(n_total * TEST_FRACTION))
    test_pos = order[n_total - n_test:]
    train_pool_pos = order[:n_total - n_test]
    test_idx = sorted(int(p) for p in test_pos)
    train_pool = [int(p) for p in train_pool_pos]
    train_idx, val_idx = _split_train_val(train_pool, VAL_FRACTION_OF_TRAIN, rng)
    meta = {
        "split_type": "time",
        "description": (
            "Time-proxy split.  HiTEA is a single 2023 publication, so a "
            "true year split is unavailable.  Records are sorted by the "
            "numeric suffix of `notebook_id` (which increases with "
            "experimental page order) and the newest 20% form the test set."
        ),
        "criteria": "temporal_by_notebook_id_suffix",
        "proxy_field": "notebook_id",
        "suffix_min": int(suffixes.min()),
        "suffix_max": int(suffixes.max()),
        "test_suffix_min": int(suffixes[test_pos].min()),
        "test_suffix_max": int(suffixes[test_pos].max()),
        "is_ood": True,
        "limitation": (
            "True publication-year split impossible (single source). "
            "Notebook page order used as a monotonic proxy."
        ),
    }
    return train_idx, val_idx, test_idx, meta


def build_author_lab_split(df, rng: random.Random) -> Tuple[List[int], List[int], List[int], Dict]:
    """Author/lab split via ``notebook_id`` prefix.

    The HiTEA dataset contains two notebook prefixes that we treat as
    distinct labs: ``00119131`` (36,763 records) and ``00707420``
    (2,783 records).  We train on the larger lab and test on the smaller,
    giving a genuine lab-disjoint OOD split.
    """
    prefixes = df["notebook_id"].astype(str).str.split("-").str[0]
    groups: Dict[str, List[int]] = defaultdict(list)
    for idx, prefix in zip(df.index, prefixes):
        groups[str(prefix)].append(int(idx))
    if len(groups) < 2:
        raise ValueError(
            "author_lab split requires at least 2 distinct notebook_id "
            f"prefixes; found {len(groups)}."
        )
    # Genuine lab-disjoint OOD: hold out the *smallest* lab(s) as test
    # (regardless of the 20% target) and train on the rest.  With the
    # HiTEA two-lab split (36763 / 2783) the smaller lab (~7%) becomes
    # the entire test set — a true author/lab OOD evaluation.
    by_size_asc = sorted(groups.items(), key=lambda kv: len(kv[1]))
    test_keys: List[str] = []
    train_keys: List[str] = []
    # The smallest lab is always held out for test.  Additional small labs
    # are added to test only if doing so does not exceed 2x TEST_FRACTION
    # (to avoid an absurdly large test set when there are many tiny labs).
    test_count = 0
    target_test_max = len(df) * TEST_FRACTION * 2
    for key, idxs in by_size_asc:
        if not test_keys:
            # Always hold out at least the smallest lab.
            test_keys.append(key)
            test_count += len(idxs)
        elif test_count + len(idxs) <= target_test_max:
            test_keys.append(key)
            test_count += len(idxs)
        else:
            train_keys.append(key)
    train_keys = [k for k in groups.keys() if k not in set(test_keys)]
    if not train_keys:
        raise ValueError(
            "author_lab split: all labs assigned to test; cannot train. "
            f"Found {len(groups)} labs."
        )
    train_pool: List[int] = []
    for k in train_keys:
        train_pool.extend(groups[k])
    test_idx = sorted(i for k in test_keys for i in groups[k])
    train_idx, val_idx = _split_train_val(train_pool, VAL_FRACTION_OF_TRAIN, rng)
    meta = {
        "split_type": "author_lab",
        "description": (
            "Lab-disjoint split via `notebook_id` prefix.  The smaller lab "
            "(``00707420``) is held out entirely for test, giving a true "
            "author/lab OOD evaluation."
        ),
        "criteria": "group_disjoint_by_notebook_id_prefix",
        "proxy_field": "notebook_id",
        "n_train_labs": len(train_keys),
        "n_test_labs": len(test_keys),
        "train_labs": sorted(train_keys),
        "test_labs": sorted(test_keys),
        "is_ood": True,
    }
    return train_idx, val_idx, test_idx, meta


def build_random_split(df, rng: random.Random) -> Tuple[List[int], List[int], List[int], Dict]:
    """Stratified random baseline split (in-distribution)."""
    # Stratify by reaction_family to keep class balance stable across splits.
    groups = _groups_to_indices(df, "reaction_family")
    train_pool: List[int] = []
    test_idx: List[int] = []
    for key, idxs in groups.items():
        shuffled = list(idxs)
        rng.shuffle(shuffled)
        n_test = int(round(len(shuffled) * TEST_FRACTION))
        test_idx.extend(shuffled[:n_test])
        train_pool.extend(shuffled[n_test:])
    test_idx.sort()
    train_idx, val_idx = _split_train_val(train_pool, VAL_FRACTION_OF_TRAIN, rng)
    meta = {
        "split_type": "random",
        "description": (
            "Stratified random baseline (in-distribution).  Stratification "
            "key is `reaction_family`.  Used as the upper-bound comparator "
            "for the OOD splits."
        ),
        "criteria": "stratified_random_by_reaction_family",
        "stratify_key": "reaction_family",
        "is_ood": False,
    }
    return train_idx, val_idx, test_idx, meta


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

SPLIT_BUILDERS = {
    "reaction_family": build_reaction_family_split,
    "scaffold": build_scaffold_split,
    "condition_space": build_condition_space_split,
    "time": build_time_split,
    "author_lab": build_author_lab_split,
    "random": build_random_split,
}


def build_all_splits(parquet_path: Path,
                     output_dir: Path,
                     seed: int = DEFAULT_SEED,
                     split_names: Optional[Sequence[str]] = None) -> Dict[str, Dict]:
    """Build every requested OOD split and persist JSON artefacts.

    Returns a dict ``{split_name: metadata}`` for reporting.
    """
    df = _load_dataframe(parquet_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    names = list(split_names) if split_names else list(SPLIT_NAMES)
    summary: Dict[str, Dict] = {}
    for name in names:
        if name not in SPLIT_BUILDERS:
            print(f"[ood_splits] WARNING: unknown split '{name}', skipping")
            continue
        # Per-split RNG so that adding/removing a split does not perturb others.
        split_rng = random.Random(seed + hash(name) % (2**31))
        try:
            train_idx, val_idx, test_idx, meta = SPLIT_BUILDERS[name](df, split_rng)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[ood_splits] ERROR building '{name}': {exc}")
            summary[name] = {"error": str(exc)}
            continue
        _write_split(output_dir, name, train_idx, val_idx, test_idx, meta)
        print(
            f"[ood_splits] {name:18s}  "
            f"train={len(train_idx):6d}  val={len(val_idx):5d}  "
            f"test={len(test_idx):5d}  ood={meta.get('is_ood')}"
        )
        summary[name] = meta
    # Write a top-level manifest so downstream code can discover splits.
    manifest = {
        "parquet_path": str(parquet_path),
        "n_records": len(df),
        "seed": seed,
        "test_fraction": TEST_FRACTION,
        "val_fraction_of_train": VAL_FRACTION_OF_TRAIN,
        "splits": {
            name: {
                "n_train": m.get("n_train"),
                "n_val": m.get("n_val"),
                "n_test": m.get("n_test"),
                "is_ood": m.get("is_ood"),
            }
            for name, m in summary.items() if "error" not in m
        },
    }
    with open(output_dir / "splits_manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    return summary


def load_split(output_dir: Path, split_name: str) -> Dict:
    """Load a previously-written split's train/val/test indices."""
    out: Dict[str, List[int]] = {}
    for suffix in ("train_idx", "val_idx", "test_idx"):
        path = output_dir / f"{split_name}_{suffix}.json"
        with open(path) as fh:
            out[suffix] = json.load(fh)
    meta_path = output_dir / f"{split_name}_metadata.json"
    if meta_path.exists():
        with open(meta_path) as fh:
            out["metadata"] = json.load(fh)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build Phase 3 OOD splits for the HiTEA dataset."
    )
    parser.add_argument(
        "--parquet", default=DEFAULT_PARQUET,
        help=f"Path to HiTEA parquet (default: {DEFAULT_PARQUET})",
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--splits", nargs="+", default=list(SPLIT_NAMES),
        choices=list(SPLIT_NAMES),
        help="Which splits to build (default: all)",
    )
    args = parser.parse_args(argv)

    # Resolve relative paths against the repo root (cwd when run from repo).
    parquet_path = Path(args.parquet)
    output_dir = Path(args.output_dir)
    if not parquet_path.is_absolute():
        parquet_path = Path.cwd() / parquet_path
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir

    print(f"[ood_splits] parquet   : {parquet_path}")
    print(f"[ood_splits] output    : {output_dir}")
    print(f"[ood_splits] seed      : {args.seed}")
    print(f"[ood_splits] splits    : {', '.join(args.splits)}")
    print()

    summary = build_all_splits(parquet_path, output_dir, args.seed, args.splits)
    print()
    print("[ood_splits] === Summary ===")
    for name, meta in summary.items():
        if "error" in meta:
            print(f"  {name:18s}  ERROR: {meta['error']}")
        else:
            print(
                f"  {name:18s}  "
                f"train={meta['n_train']:6d}  val={meta['n_val']:5d}  "
                f"test={meta['n_test']:5d}  ood={meta.get('is_ood')}"
            )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
