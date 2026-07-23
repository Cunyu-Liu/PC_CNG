"""Tests for P4-G6 HTE data schema and normalization.

Run with::

    python3 -m pytest chem_negative_sampling/tests/test_p4_hte_schema.py -v
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CNS_ROOT = _REPO_ROOT / "chem_negative_sampling"
if str(_CNS_ROOT) not in sys.path:
    sys.path.insert(0, str(_CNS_ROOT))

from pc_cng.p4_g6_hte_data import (
    REQUIRED_AUDIT_FIELDS,
    ZERO_TYPES,
    YIELD_BINS,
    LOW_YIELD_THRESHOLDS,
    _classify_yield,
    _yield_bin,
    _grid_hash,
    normalize_hte,
    build_screen_aware_split,
    write_data_audit,
)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestYieldClassification:
    def test_positive_yield(self):
        assert _classify_yield(50.0, 1000.0) == "positive_yield"

    def test_low_yield(self):
        assert _classify_yield(3.0, 500.0) == "low_yield"

    def test_measured_zero(self):
        assert _classify_yield(0.0, 100.0) == "measured_zero"

    def test_below_detection(self):
        assert _classify_yield(0.0, 0.0) == "below_detection"

    def test_missing_measurement(self):
        assert _classify_yield(None, 100.0) == "missing_measurement"

    def test_no_product_recorded(self):
        assert _classify_yield(0.0, None) == "no_product_recorded"

    def test_boundary_5(self):
        # yield == 5.0 is NOT low_yield (it's positive_yield, since < 5 is low)
        assert _classify_yield(5.0, 100.0) == "positive_yield"


class TestYieldBin:
    def test_bin_0(self):
        assert _yield_bin(0.0) == 0
        assert _yield_bin(4.9) == 0

    def test_bin_1(self):
        assert _yield_bin(5.0) == 1
        assert _yield_bin(19.9) == 1

    def test_bin_2(self):
        assert _yield_bin(20.0) == 2
        assert _yield_bin(49.9) == 2

    def test_bin_3(self):
        assert _yield_bin(50.0) == 3
        assert _yield_bin(79.9) == 3

    def test_bin_4(self):
        assert _yield_bin(80.0) == 4
        assert _yield_bin(100.0) == 4

    def test_all_bins_covered(self):
        assert len(YIELD_BINS) == 5

    def test_thresholds(self):
        assert len(LOW_YIELD_THRESHOLDS) >= 2


class TestGridHash:
    def test_deterministic(self):
        assert _grid_hash("a", "b") == _grid_hash("a", "b")

    def test_different_inputs(self):
        assert _grid_hash("a", "b") != _grid_hash("a", "c")

    def test_empty_fields_ignored(self):
        assert _grid_hash("a", "", "b") == _grid_hash("a", "b")


class TestRequiredFields:
    def test_all_16_fields_present(self):
        assert len(REQUIRED_AUDIT_FIELDS) == 16

    def test_fields_match_spec(self):
        expected = {
            "record_id", "source_publication", "license", "measured_yield",
            "yield_unit", "yield_normalization", "experimental_group",
            "plate_id", "substrate_grid", "condition_grid", "replicate",
            "missing_measurement", "reported_zero", "detection_limit",
            "reaction_family", "split",
        }
        assert set(REQUIRED_AUDIT_FIELDS) == expected

    def test_zero_types_complete(self):
        expected_types = {"measured_zero", "below_detection", "missing_measurement",
                          "failed_experiment", "low_yield", "no_product_recorded"}
        assert set(ZERO_TYPES) == expected_types


# ---------------------------------------------------------------------------
# Integration tests (require raw HiTEA data on server)
# ---------------------------------------------------------------------------

class TestNormalizationIntegration:
    @pytest.fixture(scope="class")
    def htea_paths(self):
        repo = Path("/home/cunyuliu/pc_cng_research")
        raw = repo / "external/HiTEA/data/8_SEPT_APPROVED_full_dataset.csv"
        norm = repo / "data/processed/hitea_full_normalized.csv"
        if not raw.exists() or not norm.exists():
            pytest.skip("HiTEA data not available on this machine")
        return raw, norm

    def test_normalize_produces_parquet(self, htea_paths, tmp_path):
        raw, norm = htea_paths
        parquet = tmp_path / "test.parquet"
        summary = normalize_hte(raw, norm, parquet)
        assert parquet.exists()
        assert summary["n_records"] > 0
        assert summary["n_screens"] > 0

    def test_parquet_has_all_fields(self, htea_paths, tmp_path):
        import pyarrow.parquet as pq
        raw, norm = htea_paths
        parquet = tmp_path / "test.parquet"
        normalize_hte(raw, norm, parquet)
        table = pq.read_table(parquet)
        for field in REQUIRED_AUDIT_FIELDS:
            assert field in table.column_names, f"Missing required field: {field}"

    def test_screen_aware_split(self, htea_paths, tmp_path):
        import pyarrow.parquet as pq
        raw, norm = htea_paths
        parquet = tmp_path / "test.parquet"
        manifest = tmp_path / "split.json"
        normalize_hte(raw, norm, parquet)
        result = build_screen_aware_split(parquet, manifest)
        assert result["no_screen_crosses_splits"] is True
        # Verify no screen crosses splits
        df = pq.read_table(parquet).to_pylist()
        screen_splits = {}
        for rec in df:
            sid = rec["experimental_group"]
            sp = rec["split"]
            if sid in screen_splits and screen_splits[sid] != sp:
                pytest.fail(f"Screen {sid} crosses splits: {screen_splits[sid]} and {sp}")
            screen_splits[sid] = sp
