"""Unit tests for ``hte_loader.py`` (P3-05).

These tests use synthetic CSV data (no GPU/remote access needed) and cover:
- :class:`HTEGroup` construction
- :func:`load_hte_by_class` with filtering + yield stats
- :func:`stratified_sample` determinism + edge cases
- :func:`family_cluster_bootstrap` basic shape
- :func:`iter_reactions` CSV parsing
"""

from __future__ import annotations

import csv
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Make ``hte_loader`` importable both when tests live in /tmp/p3_files/p3_05/
# (flat layout) and when they live in chem_negative_sampling/tests/ on the
# server (subdir layout).  We try the data.* import first to mirror the
# server, then fall back to a direct import.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

try:  # server layout: chem_negative_sampling/tests/test_hte_loader.py
    from data.hte_loader import (  # type: ignore[import]
        HTEGroup,
        family_cluster_bootstrap,
        iter_reactions,
        load_hte_by_class,
        stratified_sample,
    )
except ImportError:  # flat local layout
    from hte_loader import (  # type: ignore[no-redef]
        HTEGroup,
        family_cluster_bootstrap,
        iter_reactions,
        load_hte_by_class,
        stratified_sample,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


CSV_HEADER = [
    "source_id",
    "reaction_smiles",
    "reactants",
    "agents",
    "products",
    "label_type",
    "yield",
    "source",
    "split_key",
    "split",
    "reaction_class",
]


def _row(
    source_id: str,
    rxn: str,
    product: str,
    yld: str,
    cls: str,
) -> dict:
    return {
        "source_id": source_id,
        "reaction_smiles": rxn,
        "reactants": rxn.split(">>")[0] if ">>" in rxn else rxn,
        "agents": "",
        "products": product,
        "label_type": "positive",
        "yield": yld,
        "source": "hitea",
        "split_key": "k",
        "split": "train",
        "reaction_class": cls,
    }


@pytest.fixture
def tiny_csv(tmp_path: Path) -> str:
    """A small HTEa-like CSV with 3 classes (sizes 25, 30, 5)."""
    path = tmp_path / "tiny.csv"
    rows: list = []
    # Class "Alkylation" (25 reactions)
    for i in range(25):
        rows.append(_row(f"a{i}", f"CC{i}>>CC{i}O", f"CC{i}O", str(50 + i), "Alkylation"))
    # Class "Acylation" (30 reactions)
    for i in range(30):
        rows.append(_row(f"b{i}", f"CC{i}>>CC{i}=O", f"CC{i}=O", str(60 + i), "Acylation"))
    # Class "Tiny" (5 reactions -- below default min_class_size)
    for i in range(5):
        rows.append(_row(f"c{i}", f"C{i}>>C{i}N", f"C{i}N", "", "Tiny"))
    # One row with an unparseable yield string
    rows.append(_row("bad1", "CC>>CCO", "CCO", "not_a_number", "Alkylation"))
    # One row with empty reaction_smiles (should be skipped)
    rows.append(_row("bad2", "", "", "10", "Acylation"))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_HEADER)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    return str(path)


@pytest.fixture
def empty_csv(tmp_path: Path) -> str:
    path = tmp_path / "empty.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_HEADER)
        writer.writeheader()
    return str(path)


@pytest.fixture
def missing_col_csv(tmp_path: Path) -> str:
    path = tmp_path / "missing.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["source_id", "yield"])
        writer.writeheader()
        writer.writerow({"source_id": "x1", "yield": "10"})
    return str(path)


# ---------------------------------------------------------------------------
# iter_reactions tests
# ---------------------------------------------------------------------------


class TestIterReactions:
    def test_reads_all_rows(self, tiny_csv):
        rows = iter_reactions(tiny_csv)
        # 25 + 30 + 5 + 2 (bad1, bad2) = 62
        assert len(rows) == 62

    def test_handles_empty_csv(self, empty_csv):
        rows = iter_reactions(empty_csv)
        assert rows == []

    def test_normalises_none_to_empty(self, tmp_path: Path):
        """Missing cells in a row should be returned as empty strings."""
        path = tmp_path / "none.csv"
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["a", "b"])
            writer.writeheader()
            writer.writerow({"a": "1"})  # column b missing
        rows = iter_reactions(str(path))
        assert rows[0]["b"] == ""

    def test_preserves_column_values(self, tiny_csv):
        rows = iter_reactions(tiny_csv)
        first = rows[0]
        assert first["source_id"] == "a0"
        assert first["reaction_class"] == "Alkylation"
        assert first["yield"] == "50"


# ---------------------------------------------------------------------------
# load_hte_by_class tests
# ---------------------------------------------------------------------------


class TestLoadHteByClass:
    def test_filters_small_classes(self, tiny_csv):
        groups = load_hte_by_class(tiny_csv, min_class_size=20)
        # Tiny (5 reactions) should be filtered out
        assert set(groups.keys()) == {"Alkylation", "Acylation"}
        assert "Tiny" not in groups

    def test_min_class_size_zero_keeps_all(self, tiny_csv):
        groups = load_hte_by_class(tiny_csv, min_class_size=1)
        assert "Tiny" in groups
        assert groups["Tiny"].n_reactions == 5

    def test_reactions_count(self, tiny_csv):
        groups = load_hte_by_class(tiny_csv, min_class_size=20)
        # Alkylation has 25 + 1 (bad1) = 26 reactions
        assert groups["Alkylation"].n_reactions == 26
        # Acylation has 30 + 1 (bad2) - 1 (bad2 empty smiles) = 30
        assert groups["Acylation"].n_reactions == 30

    def test_yield_stats(self, tiny_csv):
        groups = load_hte_by_class(tiny_csv, min_class_size=20)
        alk = groups["Alkylation"]
        # Yields: 50..74 (25 values) + bad1 (unparseable, skipped) = 25 values
        assert alk.n_reactions == 26
        # mean of 50..74 = 62
        expected_mean = sum(range(50, 75)) / 25
        assert alk.yield_mean == pytest.approx(expected_mean, abs=1e-6)
        # std (sample, ddof=1) of 50..74
        import statistics
        expected_std = statistics.stdev(range(50, 75))
        assert alk.yield_std == pytest.approx(expected_std, abs=1e-6)

    def test_empty_yield_class(self, tiny_csv):
        """The Tiny class has no valid yields -> NaN."""
        groups = load_hte_by_class(tiny_csv, min_class_size=1)
        import math
        assert math.isnan(groups["Tiny"].yield_mean)
        assert math.isnan(groups["Tiny"].yield_std)

    def test_unique_products(self, tiny_csv):
        groups = load_hte_by_class(tiny_csv, min_class_size=20)
        # All 26 Alkylation reactions have distinct products (CC0..CC24, plus CCO from bad1)
        # bad1 product is "CCO" which differs from "CC{i}O"
        assert groups["Alkylation"].n_unique_products == 26

    def test_reaction_smiles_list(self, tiny_csv):
        groups = load_hte_by_class(tiny_csv, min_class_size=20)
        alk = groups["Alkylation"]
        assert len(alk.reaction_smiles_list) == 26
        # All entries should be non-empty strings
        assert all(isinstance(s, str) and s for s in alk.reaction_smiles_list)

    def test_source_ids(self, tiny_csv):
        groups = load_hte_by_class(tiny_csv, min_class_size=20)
        alk = groups["Alkylation"]
        assert len(alk.source_ids) == 26
        assert "a0" in alk.source_ids
        assert "bad1" in alk.source_ids

    def test_missing_required_columns(self, missing_col_csv):
        with pytest.raises(ValueError, match="missing required columns"):
            load_hte_by_class(missing_col_csv, min_class_size=1)

    def test_empty_csv_returns_empty(self, empty_csv):
        groups = load_hte_by_class(empty_csv, min_class_size=1)
        assert groups == {}

    def test_invalid_min_class_size(self, tiny_csv):
        with pytest.raises(ValueError, match="min_class_size"):
            load_hte_by_class(tiny_csv, min_class_size=0)

    def test_dataclass_fields(self, tiny_csv):
        """Smoke test: HTEGroup should be a dataclass with all fields."""
        groups = load_hte_by_class(tiny_csv, min_class_size=20)
        alk = groups["Alkylation"]
        assert isinstance(alk, HTEGroup)
        assert alk.reaction_class == "Alkylation"
        assert isinstance(alk.n_reactions, int)
        assert isinstance(alk.n_unique_products, int)
        assert isinstance(alk.yield_mean, float)
        assert isinstance(alk.yield_std, float)


# ---------------------------------------------------------------------------
# stratified_sample tests
# ---------------------------------------------------------------------------


class TestStratifiedSample:
    def test_basic_sample(self, tiny_csv):
        groups = load_hte_by_class(tiny_csv, min_class_size=20)
        sampled = stratified_sample(groups, n_per_class=10, seed=42)
        # 2 classes * 10 = 20
        assert len(sampled) == 20

    def test_n_per_class_larger_than_class(self, tiny_csv):
        """If n_per_class > class size, the whole class is sampled."""
        groups = load_hte_by_class(tiny_csv, min_class_size=20)
        sampled = stratified_sample(groups, n_per_class=1000, seed=42)
        # 26 + 30 = 56
        assert len(sampled) == 56

    def test_deterministic_with_seed(self, tiny_csv):
        groups = load_hte_by_class(tiny_csv, min_class_size=20)
        s1 = stratified_sample(groups, n_per_class=10, seed=42)
        s2 = stratified_sample(groups, n_per_class=10, seed=42)
        assert s1 == s2

    def test_different_seeds_yield_different_samples(self, tiny_csv):
        groups = load_hte_by_class(tiny_csv, min_class_size=20)
        s1 = stratified_sample(groups, n_per_class=10, seed=42)
        s2 = stratified_sample(groups, n_per_class=10, seed=43)
        assert s1 != s2

    def test_invalid_n_per_class(self, tiny_csv):
        groups = load_hte_by_class(tiny_csv, min_class_size=20)
        with pytest.raises(ValueError, match="n_per_class"):
            stratified_sample(groups, n_per_class=0)

    def test_empty_groups(self):
        sampled = stratified_sample({}, n_per_class=10, seed=42)
        assert sampled == []

    def test_all_entries_are_strings(self, tiny_csv):
        groups = load_hte_by_class(tiny_csv, min_class_size=20)
        sampled = stratified_sample(groups, n_per_class=5, seed=42)
        assert all(isinstance(s, str) for s in sampled)

    def test_sampled_reactions_belong_to_groups(self, tiny_csv):
        """Every sampled SMILES should exist in one of the groups."""
        groups = load_hte_by_class(tiny_csv, min_class_size=20)
        sampled = stratified_sample(groups, n_per_class=5, seed=42)
        all_smiles = set()
        for g in groups.values():
            all_smiles.update(g.reaction_smiles_list)
        for s in sampled:
            assert s in all_smiles


# ---------------------------------------------------------------------------
# family_cluster_bootstrap tests
# ---------------------------------------------------------------------------


class TestFamilyClusterBootstrap:
    def test_returns_correct_number_of_replicates(self, tiny_csv):
        groups = load_hte_by_class(tiny_csv, min_class_size=20)
        reps = family_cluster_bootstrap(groups, seed=0, n_bootstrap=50)
        assert len(reps) == 50
        # Each replicate is the concatenation of len(groups) resampled classes
        # (with replacement), so its length == sum of 2 class sizes (could be
        # any combination: 26+26, 26+30, 30+26, 30+30).
        class_sizes = sorted(g.n_reactions for g in groups.values())  # [26, 30]
        min_possible = len(groups) * class_sizes[0]
        max_possible = len(groups) * class_sizes[-1]
        for rep in reps:
            assert len(rep) > 0
            assert min_possible <= len(rep) <= max_possible

    def test_deterministic_with_seed(self, tiny_csv):
        groups = load_hte_by_class(tiny_csv, min_class_size=20)
        r1 = family_cluster_bootstrap(groups, seed=42, n_bootstrap=10)
        r2 = family_cluster_bootstrap(groups, seed=42, n_bootstrap=10)
        assert r1 == r2

    def test_empty_groups(self):
        reps = family_cluster_bootstrap({}, seed=0, n_bootstrap=10)
        assert reps == []

    def test_invalid_n_bootstrap(self, tiny_csv):
        groups = load_hte_by_class(tiny_csv, min_class_size=20)
        with pytest.raises(ValueError, match="n_bootstrap"):
            family_cluster_bootstrap(groups, n_bootstrap=0)

    def test_replicate_contents_belong_to_input_groups(self, tiny_csv):
        groups = load_hte_by_class(tiny_csv, min_class_size=20)
        reps = family_cluster_bootstrap(groups, seed=0, n_bootstrap=5)
        all_smiles = set()
        for g in groups.values():
            all_smiles.update(g.reaction_smiles_list)
        for rep in reps:
            for s in rep:
                assert s in all_smiles
