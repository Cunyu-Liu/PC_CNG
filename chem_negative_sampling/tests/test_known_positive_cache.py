from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class KnownPositiveCacheTest(unittest.TestCase):
    def test_cache_drives_review_and_product_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            positive = root / "positive.csv"
            raw = root / "raw.csv"
            cache = root / "known_positive_cache.json"
            reviewed = root / "reviewed.csv"
            review_summary = root / "review_summary.json"
            filtered = root / "filtered.csv"
            removed = root / "removed.csv"
            filter_summary = root / "filter_summary.json"

            with positive.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["source_id", "reaction_smiles", "label_type"])
                writer.writerow(["p1", "CCO>>CC=O", "positive"])

            with raw.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "source_id",
                        "positive_reaction",
                        "candidate_reaction",
                        "hard_score",
                        "false_negative_risk",
                        "failure_type",
                    ]
                )
                writer.writerow(["p1", "CCO>>CC=O", "CCO>>CC=O", "0.2", "0.0", "toy"])
                writer.writerow(["p1", "CCO>>CC=O", "CCO>>CC", "0.2", "0.0", "toy"])

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pc_cng.build_known_positive_cache",
                    "--positive-csv",
                    str(positive),
                    "--output-json",
                    str(cache),
                ],
                check=True,
            )
            payload = json.loads(cache.read_text(encoding="utf-8"))
            self.assertEqual(payload["canonical_reaction_count"], 1)
            self.assertEqual(payload["canonical_product_count"], 1)

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pc_cng.false_negative_review",
                    "--input",
                    str(raw),
                    "--output",
                    str(reviewed),
                    "--summary",
                    str(review_summary),
                    "--known-positive",
                    str(cache),
                ],
                check=True,
            )
            review_payload = json.loads(review_summary.read_text(encoding="utf-8"))
            self.assertEqual(review_payload["status_counts"]["discard_known_positive"], 1)
            self.assertEqual(review_payload["status_counts"]["keep_synthetic_negative"], 1)

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pc_cng.filter_synthetic_known_positives",
                    "--positive-csv",
                    str(cache),
                    "--synthetic-csv",
                    str(reviewed),
                    "--output-csv",
                    str(filtered),
                    "--removed-csv",
                    str(removed),
                    "--summary-json",
                    str(filter_summary),
                    "--review-status",
                    "discard_known_positive",
                ],
                check=True,
            )
            filter_payload = json.loads(filter_summary.read_text(encoding="utf-8"))
            self.assertEqual(filter_payload["removed_rows"], 1)
            self.assertEqual(filter_payload["kept_rows"], 1)


if __name__ == "__main__":
    unittest.main()
