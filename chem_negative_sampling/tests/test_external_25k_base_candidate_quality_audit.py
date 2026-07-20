from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class External25kBaseCandidateQualityAuditTest(unittest.TestCase):
    def test_audit_reports_candidate_quality_warnings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contexts = root / "contexts.csv"
            candidates = root / "candidates.csv"
            out_dir = root / "audit"

            with contexts.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["group_id", "source_id", "reactants", "agents", "observed_product", "split", "dataset"])
                writer.writerows(
                    [
                        ["g1", "s1", "CCO", "", "CC=O", "test", "mini"],
                        ["g2", "s2", "CCBr", "", "CCO", "test", "mini"],
                        ["g3", "s3", "CCCl", "", "CCBr", "test", "mini"],
                    ]
                )

            with candidates.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "group_id",
                        "source_id",
                        "reactants",
                        "agents",
                        "candidate_product",
                        "candidate_reaction",
                        "label",
                        "split",
                        "dataset",
                        "candidate_source",
                    ]
                )
                writer.writerows(
                    [
                        ["g1", "s1", "CCO", "", "CC=O", "CCO>>CC=O", "1", "test", "mini", "observed_positive"],
                        ["g1", "s1", "CCO", "", "CCO", "CCO>>CCO", "0", "test", "mini", "pc_cng"],
                        ["g2", "s2", "CCBr", "", "CCO", "CCBr>>CCO", "1", "test", "mini", "observed_positive"],
                        ["g2", "s2", "CCBr", "", "CCO", "CCBr>>CCO", "0", "test", "mini", "pc_cng"],
                        ["g3", "s3", "CCCl", "", "CCBr", "CCCl>>CCBr", "1", "test", "mini", "observed_positive"],
                    ]
                )

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pc_cng.audit_external_25k_base_candidate_quality",
                    "--contexts-csv",
                    str(contexts),
                    "--candidates-csv",
                    str(candidates),
                    "--output-dir",
                    str(out_dir),
                ],
                check=True,
            )

            payload = json.loads((out_dir / "external_25k_base_candidate_quality_audit.json").read_text())
            self.assertEqual(payload["context_groups"], 3)
            self.assertEqual(payload["observed_positive_groups"], 3)
            self.assertEqual(payload["pc_cng_negative_groups"], 2)
            self.assertEqual(payload["missing_pc_cng_negative_groups"], 1)
            self.assertEqual(payload["same_product_pc_cng_negative_rows"], 1)
            self.assertEqual(payload["decision"], "pass_with_warnings")
            self.assertIn("pc_cng_negative_group_coverage_below_99pct", payload["decision_flags"]["warnings"])
            self.assertTrue((out_dir / "external_25k_base_candidate_quality_audit.md").exists())


if __name__ == "__main__":
    unittest.main()
