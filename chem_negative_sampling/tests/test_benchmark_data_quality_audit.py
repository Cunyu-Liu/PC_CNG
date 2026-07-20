from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REAL_FIELDS = [
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

SYN_FIELDS = [
    "source_id",
    "positive_reaction",
    "candidate_reaction",
    "action_family",
    "failure_type",
    "review_status",
]


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


class BenchmarkDataQualityAuditTest(unittest.TestCase):
    def test_filter_removes_only_keep_known_positive_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real = root / "real.csv"
            synthetic = root / "synthetic.csv"
            filtered = root / "synthetic_filtered.csv"
            removed = root / "synthetic_removed.csv"
            summary = root / "filter_summary.json"

            write_csv(
                real,
                REAL_FIELDS,
                [
                    {
                        "source_id": "p1",
                        "reaction_smiles": "CCO>>CC=O",
                        "reactants": "CCO",
                        "products": "CC=O",
                        "label_type": "positive",
                        "source": "toy",
                        "split_key": "k1",
                        "split": "train",
                        "reaction_class": "toy",
                    }
                ],
            )
            write_csv(
                synthetic,
                SYN_FIELDS,
                [
                    {
                        "source_id": "p1",
                        "positive_reaction": "CCO>>CC=O",
                        "candidate_reaction": "CCO>>CC=O",
                        "action_family": "regio",
                        "failure_type": "known_positive_overlap",
                        "review_status": "keep_synthetic_negative",
                    },
                    {
                        "source_id": "p1",
                        "positive_reaction": "CCO>>CC=O",
                        "candidate_reaction": "CCO>>CC=O",
                        "action_family": "regio",
                        "failure_type": "known_positive_overlap",
                        "review_status": "exclude_known_positive",
                    },
                    {
                        "source_id": "p1",
                        "positive_reaction": "CCO>>CC=O",
                        "candidate_reaction": "CCO>>CC",
                        "action_family": "regio",
                        "failure_type": "hard_negative",
                        "review_status": "keep_synthetic_negative",
                    },
                ],
            )

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pc_cng.filter_synthetic_known_positives",
                    "--positive-csv",
                    str(real),
                    "--synthetic-csv",
                    str(synthetic),
                    "--output-csv",
                    str(filtered),
                    "--removed-csv",
                    str(removed),
                    "--summary-json",
                    str(summary),
                ],
                check=True,
            )

            payload = json.loads(summary.read_text(encoding="utf-8"))
            self.assertEqual(payload["removed_rows"], 1)
            self.assertEqual(payload["kept_rows"], 2)
            with filtered.open(newline="", encoding="utf-8") as handle:
                kept_rows = list(csv.DictReader(handle))
            self.assertEqual(len(kept_rows), 2)
            self.assertEqual(kept_rows[0]["review_status"], "exclude_known_positive")
            with removed.open(newline="", encoding="utf-8") as handle:
                removed_rows = list(csv.DictReader(handle))
            self.assertEqual(removed_rows[0]["known_positive_canonical_product"], "CC=O")

    def test_audit_status_changes_after_filtering_known_positive_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real = root / "real.csv"
            synthetic = root / "synthetic.csv"
            filtered = root / "synthetic_filtered.csv"
            removed = root / "synthetic_removed.csv"
            filter_summary = root / "filter_summary.json"
            manifest = root / "manifest.json"
            audit_raw = root / "audit_raw"
            audit_filtered = root / "audit_filtered"

            real_rows = [
                {
                    "source_id": "p1",
                    "reaction_smiles": "CCO>>CC=O",
                    "reactants": "CCO",
                    "products": "CC=O",
                    "label_type": "positive",
                    "source": "toy",
                    "split_key": "k1",
                    "split": "train",
                    "reaction_class": "toy",
                },
                {
                    "source_id": "n1",
                    "reaction_smiles": "CCO>>CC",
                    "reactants": "CCO",
                    "products": "CC",
                    "label_type": "real_negative",
                    "source": "toy",
                    "split_key": "k1",
                    "split": "train",
                    "reaction_class": "toy",
                },
                {
                    "source_id": "p2",
                    "reaction_smiles": "CCC>>CC=C",
                    "reactants": "CCC",
                    "products": "CC=C",
                    "label_type": "positive",
                    "source": "toy",
                    "split_key": "k2",
                    "split": "val",
                    "reaction_class": "toy",
                },
            ]
            write_csv(real, REAL_FIELDS, real_rows)
            write_csv(
                synthetic,
                SYN_FIELDS,
                [
                    {
                        "source_id": "p1",
                        "positive_reaction": "CCO>>CC=O",
                        "candidate_reaction": "CCO>>CC=O",
                        "action_family": "regio",
                        "failure_type": "known_positive_overlap",
                        "review_status": "keep_synthetic_negative",
                    },
                    {
                        "source_id": "p2",
                        "positive_reaction": "CCC>>CC=C",
                        "candidate_reaction": "CCC>>C=C",
                        "action_family": "regio",
                        "failure_type": "val_parent_candidate",
                        "review_status": "keep_synthetic_negative",
                    },
                ],
            )

            def write_manifest(path: Path, synthetic_path: Path) -> None:
                path.write_text(
                    json.dumps(
                        {
                            "server_root": str(root),
                            "datasets": {
                                "real": [{"id": "toy_real", "path": str(real)}],
                                "synthetic": [{"id": "toy_synthetic", "path": str(synthetic_path)}],
                            },
                        }
                    ),
                    encoding="utf-8",
                )

            write_manifest(manifest, synthetic)
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pc_cng.audit_benchmark_data_quality",
                    "--manifest-json",
                    str(manifest),
                    "--output-dir",
                    str(audit_raw),
                ],
                check=True,
            )
            raw_payload = json.loads((audit_raw / "benchmark_data_quality_audit.json").read_text(encoding="utf-8"))
            self.assertEqual(raw_payload["status"], "needs_review")
            self.assertIn("toy_synthetic: keep_synthetic_negative rows overlap known positive products", raw_payload["gate_failures"])

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pc_cng.filter_synthetic_known_positives",
                    "--positive-csv",
                    str(real),
                    "--synthetic-csv",
                    str(synthetic),
                    "--output-csv",
                    str(filtered),
                    "--removed-csv",
                    str(removed),
                    "--summary-json",
                    str(filter_summary),
                ],
                check=True,
            )
            write_manifest(manifest, filtered)
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pc_cng.audit_benchmark_data_quality",
                    "--manifest-json",
                    str(manifest),
                    "--output-dir",
                    str(audit_filtered),
                ],
                check=True,
            )
            filtered_payload = json.loads((audit_filtered / "benchmark_data_quality_audit.json").read_text(encoding="utf-8"))
            self.assertEqual(filtered_payload["status"], "pass_with_warnings")
            self.assertEqual(filtered_payload["gate_failures"], [])
            self.assertTrue(any("non-train parent" in warning for warning in filtered_payload["warnings"]))


if __name__ == "__main__":
    unittest.main()
