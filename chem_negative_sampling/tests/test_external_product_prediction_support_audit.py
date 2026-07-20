from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class ExternalProductPredictionSupportAuditTest(unittest.TestCase):
    def test_audit_reports_context_and_strict_coverage_gaps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contexts = root / "contexts.csv"
            candidates = root / "candidates.csv"
            strict_summary = root / "strict.json"
            validity_summary = root / "validity.json"
            out_dir = root / "out"

            with contexts.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["group_id", "source_id", "split", "dataset"])
                writer.writerows(
                    [
                        ["g1", "s1", "train", "mini"],
                        ["g2", "s2", "test", "mini"],
                        ["g3", "s3", "val", "mini"],
                    ]
                )

            with candidates.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["group_id", "source_id", "split", "dataset", "candidate_source", "label"])
                writer.writerows(
                    [
                        ["g1", "s1", "train", "mini", "observed_positive", "1"],
                        ["g1", "s1", "train", "mini", "pc_cng", "0"],
                        ["g1", "s1", "train", "mini", "chemformer_beam", "0"],
                        ["g2", "s2", "test", "mini", "observed_positive", "1"],
                        ["g2", "s2", "test", "mini", "chemformer_beam", "0"],
                        ["g3", "s3", "val", "mini", "observed_positive", "1"],
                    ]
                )

            strict_summary.write_text(
                json.dumps(
                    {
                        "candidate_rows_requested": 6,
                        "candidate_rows_evaluated": 3,
                        "strict_complete_group_filter": {
                            "kept_groups": 1,
                            "kept_rows": 3,
                            "missing_score_rows": 3,
                        },
                        "pc_cng_score": {
                            "scored_rows": 3,
                            "attach": {"attached": 3, "missing": 3},
                        },
                        "score_metrics": {
                            "external": {
                                "overall": {"groups": 1, "top1": 0.0, "mrr": 0.5, "ndcg": 0.6},
                                "by_split": {"train": {"groups": 1, "top1": 0.0}},
                            },
                            "pc_cng": {
                                "overall": {"groups": 1, "top1": 1.0, "mrr": 1.0, "ndcg": 1.0},
                                "by_split": {"train": {"groups": 1, "top1": 1.0}},
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            validity_summary.write_text(
                json.dumps(
                    {
                        "candidate_rows_requested": 6,
                        "candidate_rows_evaluated": 6,
                        "score_metrics": {
                            "pc_cng": {
                                "overall": {"groups": 3, "top1": 0.67, "mrr": 0.83, "ndcg": 0.9},
                                "by_split": {"test": {"groups": 1, "top1": 1.0}},
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pc_cng.audit_external_product_prediction_support",
                    "--contexts-csv",
                    str(contexts),
                    "--full-candidate-csv",
                    str(candidates),
                    "--strict-summary-json",
                    str(strict_summary),
                    "--validity-summary-json",
                    str(validity_summary),
                    "--output-dir",
                    str(out_dir),
                    "--target-groups",
                    "5",
                ],
                check=True,
            )

            payload = json.loads((out_dir / "external_product_prediction_support_audit.json").read_text())
            self.assertEqual(payload["contexts"]["groups"], 3)
            self.assertEqual(payload["context_target_deficit"], 2)
            self.assertEqual(payload["full_candidates"]["groups_with_pc_cng_candidates"], 1)
            self.assertIn("external_context_target_not_met", payload["decision_flags"])
            self.assertIn("strict_complete_group_target_not_met", payload["decision_flags"])
            self.assertEqual(payload["strict_evaluation"]["best_overall_top1"]["method"], "pc_cng")
            self.assertTrue((out_dir / "external_product_prediction_support_audit.md").exists())


if __name__ == "__main__":
    unittest.main()
