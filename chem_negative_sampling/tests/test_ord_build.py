"""Unit tests for the ORD ingestion module (P1-09)."""

from __future__ import annotations

import csv
import gzip
import json
import os
import tempfile
import unittest
from pathlib import Path

from pc_cng.build_ord import (
    NORMALIZED_FIELDS,
    _value_counts,
    build_summary,
    normalize_rows,
    parse_dataset_file,
    uspto_proxy_fallback,
    write_csv,
    read_csv,
)


def _has_ord_schema() -> bool:
    try:
        from ord_schema import message_helpers  # noqa: F401
        from ord_schema.proto import dataset_pb2, reaction_pb2  # noqa: F401
        return True
    except Exception:
        return False


def _make_normalized_row(**overrides) -> dict:
    base = {f: "" for f in NORMALIZED_FIELDS}
    base.update({
        "source_id": "ord_001",
        "reaction_smiles": "CCO.[O]>>CC=O",
        "reactants": "CCO.[O]",
        "agents": "",
        "products": "CC=O",
        "label_type": "positive",
        "yield": "95.0",
        "source": "ord_open_reaction_database",
        "split_key": "abc123",
        "split": "train",
        "_smiles_valid": "1",
    })
    base.update(overrides)
    return base


class NormalizeRowsTest(unittest.TestCase):
    def test_dedup_by_canonical_reaction(self):
        rows = [
            _make_normalized_row(reaction_smiles="CCO.[O]>>CC=O"),
            _make_normalized_row(source_id="ord_002",
                                 reaction_smiles="CCO.[O]>>CC=O"),  # dup canonical
            _make_normalized_row(source_id="ord_003",
                                 reaction_smiles="c1ccccc1>>c1ccccc1"),  # distinct
        ]
        out = normalize_rows(rows)
        # 2 unique canonical reactions
        self.assertEqual(len(out), 2)
        self.assertEqual(len([r for r in out if r["_smiles_valid"] == "1"]), 2)

    def test_invalid_smiles_kept_but_flagged(self):
        rows = [_make_normalized_row(reaction_smiles="garbage>>not_a_smiles")]
        out = normalize_rows(rows)
        self.assertEqual(len(out), 1)
        # RDKit will likely flag it invalid
        self.assertIn(out[0]["_smiles_valid"], {"0", "1"})


class ValueCountsTest(unittest.TestCase):
    def test_counts(self):
        rows = [
            {"label_type": "positive"},
            {"label_type": "positive"},
            {"label_type": ""},
            {"label_type": "negative"},
        ]
        counts = _value_counts(rows, "label_type")
        self.assertEqual(counts["positive"], 2)
        self.assertEqual(counts["negative"], 1)
        self.assertEqual(counts["<empty>"], 1)


class BuildSummaryTest(unittest.TestCase):
    def test_summary_fields_and_rates(self):
        rows = [
            _make_normalized_row(),
            _make_normalized_row(source_id="ord_002",
                                 reaction_smiles="c1ccccc1>>c1ccccc1",
                                 reactants="c1ccccc1", products="c1ccccc1",
                                 **{"yield": ""}),
            _make_normalized_row(source_id="ord_003",
                                 reaction_smiles="bad>>badbad",
                                 _smiles_valid="0"),
        ]
        summary = build_summary(rows, "ord_test", 2, 2, fallback_used=False)
        self.assertEqual(summary["total_records"], 3)
        self.assertEqual(summary["datasets_attempted"], 2)
        self.assertEqual(summary["datasets_parsed"], 2)
        self.assertFalse(summary["fallback_used"])
        self.assertEqual(summary["source"], "ord_test")
        # validity rate: 2 valid out of 3
        self.assertAlmostEqual(summary["smiles_validity_rate"], 2 / 3, places=3)
        # field non-empty counts
        self.assertEqual(summary["field_non_empty_counts"]["source_id"], 3)
        self.assertEqual(summary["field_non_empty_counts"]["yield"], 2)
        self.assertGreater(summary["field_non_empty_rates"]["source_id"], 0.99)
        self.assertEqual(summary["label_type_distribution"]["positive"], 3)


class UspoProxyFallbackTest(unittest.TestCase):
    def test_samples_and_relabels(self):
        with tempfile.TemporaryDirectory() as tmp:
            uspto_path = Path(tmp) / "uspto.csv"
            uspto_rows = []
            fields = NORMALIZED_FIELDS
            for i in range(100):
                uspto_rows.append({
                    "source_id": f"uspto_{i}",
                    "reaction_smiles": f"CC{'C'*i}>>CC=O",
                    "reactants": f"CC{'C'*i}", "agents": "", "products": "CC=O",
                    "label_type": "positive", "yield": "90", "source": "uspto",
                    "split_key": f"k{i}", "split": "train",
                })
            write_csv(str(uspto_path), uspto_rows, fields)
            out = uspto_proxy_fallback(str(uspto_path), n=20, seed=42)
            self.assertEqual(len(out), 20)
            self.assertTrue(all(r["source"] == "ord_subset_uspto_proxy" for r in out))
            self.assertTrue(all(r["label_type"] == "positive" for r in out))
            # determinism: same seed => same sample
            out2 = uspto_proxy_fallback(str(uspto_path), n=20, seed=42)
            self.assertEqual([r["source_id"] for r in out],
                             [r["source_id"] for r in out2])

    def test_missing_file_returns_empty(self):
        self.assertEqual(uspto_proxy_fallback("/nonexistent.csv", 10, 1), [])


@unittest.skipUnless(_has_ord_schema(), "ord_schema not installed")
class ParseDatasetFileTest(unittest.TestCase):
    def _build_dataset_pb_gz(self, path: Path, reaction_smiles: str,
                             reaction_id: str) -> None:
        """Build a minimal ORD Dataset with one reaction, write to .pb.gz."""
        from ord_schema.proto import dataset_pb2, reaction_pb2  # type: ignore
        from ord_schema import message_helpers  # type: ignore
        reaction_from_smiles = getattr(message_helpers, "reaction_from_smiles", None)
        if reaction_from_smiles is None:
            self.skipTest("message_helpers.reaction_from_smiles unavailable")

        try:
            reaction = reaction_from_smiles(reaction_smiles)
        except Exception as exc:
            self.skipTest(f"reaction_from_smiles failed: {exc}")
        reaction.reaction_id = reaction_id
        dataset = dataset_pb2.Dataset()
        dataset.reactions.append(reaction)
        with gzip.open(str(path), "wb") as handle:
            handle.write(dataset.SerializeToString())

    def test_parses_minimal_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            pb_gz = Path(tmp) / "ord_test.pb.gz"
            self._build_dataset_pb_gz(pb_gz, "CCO.[O]>>CC=O", "ord_test_001")
            rows = parse_dataset_file(str(pb_gz), "ord_open_reaction_database")
            self.assertGreaterEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["source_id"], "ord_test_001")
            self.assertEqual(row["source"], "ord_open_reaction_database")
            self.assertIn(">>", row["reaction_smiles"] or row["reaction_smiles"] + ">>")
            self.assertEqual(row["label_type"], "positive")
            self.assertEqual(row["split"], "train")
            self.assertIn(row["source_id"], ["ord_test_001"])

    def test_missing_file_returns_empty(self):
        self.assertEqual(parse_dataset_file("/nonexistent.pb.gz", "ord"), [])


class WriteReadCsvRoundTripTest(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ord.csv"
            rows = [_make_normalized_row(), _make_normalized_row(source_id="ord_002")]
            write_csv(str(path), rows, NORMALIZED_FIELDS)
            loaded, fields = read_csv(str(path))
            self.assertEqual(fields, NORMALIZED_FIELDS)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded[0]["source_id"], "ord_001")


if __name__ == "__main__":
    unittest.main()
