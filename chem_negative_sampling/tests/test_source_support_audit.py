from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class SourceSupportAuditTest(unittest.TestCase):
    def test_audit_separates_source_and_molecular_support(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            positives = root / "positives.csv"
            synthetic = root / "synthetic.csv"
            out_dir = root / "out"

            with positives.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["source_id", "reaction_smiles", "label_type", "split", "reaction_class", "source"])
                writer.writerows(
                    [
                        ["dup1", "CCBr.N>>CCN", "positive", "train", "dup_class", "mini"],
                        ["dup2", "CCBr.N>>CCN", "positive", "train", "dup_class", "mini"],
                        ["gap1", "CCBr.O>>CCO", "positive", "train", "gap_class", "mini"],
                        ["gap2", "CCCl.O>>CCO", "positive", "train", "gap_class", "mini"],
                    ]
                )

            with synthetic.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["source_id", "candidate_reaction", "review_status", "action_family"])
                writer.writerows(
                    [
                        ["dup1", "CCBr.N>>CCBr", "keep_synthetic_negative", "class_fallback"],
                        ["dup2", "CCBr.N>>CCBr", "keep_synthetic_negative", "class_fallback"],
                        ["gap1", "CCBr.O>>CCBr", "keep_synthetic_negative", "class_fallback"],
                    ]
                )

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pc_cng.audit_reaction_class_source_support",
                    "--positive-csv",
                    str(positives),
                    "--synthetic-csv",
                    str(synthetic),
                    "--output-dir",
                    str(out_dir),
                    "--min-groups",
                    "2",
                ],
                check=True,
            )
            payload = json.loads((out_dir / "source_support_audit.json").read_text(encoding="utf-8"))
            rows = {row["reaction_class"]: row for row in payload["class_summary"]}

            self.assertEqual(rows["dup_class"]["candidate_sources"], 2)
            self.assertEqual(rows["dup_class"]["candidate_parent_reactions"], 1)
            self.assertEqual(rows["dup_class"]["status"], "source_duplicate_risk")

            self.assertEqual(rows["gap_class"]["candidate_sources"], 1)
            self.assertEqual(rows["gap_class"]["positive_parent_reactions"], 2)
            self.assertEqual(rows["gap_class"]["status"], "generator_coverage_gap")
            self.assertTrue((out_dir / "source_support_audit.md").exists())
            with (out_dir / "missing_parent_contexts.csv").open(newline="", encoding="utf-8") as handle:
                missing = list(csv.DictReader(handle))
            missing_by_class = {row["reaction_class"]: row for row in missing}
            self.assertNotIn("dup_class", missing_by_class)
            self.assertEqual(missing_by_class["gap_class"]["source_id"], "gap2")
            self.assertTrue((out_dir / "missing_parent_contexts.md").exists())


if __name__ == "__main__":
    unittest.main()
