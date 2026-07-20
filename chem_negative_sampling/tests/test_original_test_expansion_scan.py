from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class OriginalTestExpansionScanTest(unittest.TestCase):
    def test_scan_selects_only_heldout_test_contexts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            current = root / "current.csv"
            source = root / "source.csv"
            out_dir = root / "out"

            with current.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "source_id",
                        "reaction_smiles",
                        "reactants",
                        "products",
                        "label_type",
                        "source",
                        "split_key",
                        "split",
                    ]
                )
                writer.writerows(
                    [
                        ["cur_pos", "CCBr.N>>CCN", "CCBr.N", "CCN", "positive", "mini", "a", "train"],
                        ["cur_neg", "CCBr.N>>CCBr", "CCBr.N", "CCBr", "real_negative", "mini", "a", "train"],
                    ]
                )

            with source.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
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
                    ]
                )
                writer.writerows(
                    [
                        ["eligible_low", "CCCl.O>>CCO", "CCCl.O", "", "CCO", "positive", "55", "uspto", "k1", "test"],
                        ["eligible_high", "CCCl.O>>CCOC", "CCCl.O", "", "CCOC", "positive", "90", "uspto", "k1", "test"],
                        ["current_overlap", "CCBr.N>>CCN", "CCBr.N", "", "CCN", "positive", "80", "uspto", "k2", "test"],
                        ["cross_test", "CCF.O>>CCF", "CCF.O", "", "CCF", "positive", "70", "uspto", "k3", "test"],
                        ["cross_train", "CCF.O>>CCF", "CCF.O", "", "CCF", "positive", "70", "uspto", "k3", "train"],
                        ["non_positive", "CCC.O>>CCCO", "CCC.O", "", "CCCO", "real_negative", "", "uspto", "k4", "test"],
                    ]
                )

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pc_cng.scan_original_test_expansion_candidates",
                    "--current-real-csv",
                    str(current),
                    "--source-csv",
                    str(source),
                    "--output-dir",
                    str(out_dir),
                    "--existing-test-groups",
                    "1",
                    "--target-test-groups",
                    "2",
                    "--max-selected",
                    "8",
                ],
                check=True,
            )

            payload = json.loads((out_dir / "uspto_original_test_expansion_scan.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["needed_additional_test_groups"], 1)
            self.assertEqual(payload["selected_candidate_contexts"], 1)
            self.assertTrue(payload["selected_can_cover_deficit"])
            self.assertEqual(payload["status"], "positive_parent_pool_sufficient_for_next_generation_stage")
            self.assertEqual(payload["reject_reason_counts"]["context_seen_in_current_original"], 1)
            self.assertEqual(payload["reject_reason_counts"]["source_context_cross_split"], 2)

            with (out_dir / "uspto_original_test_expansion_candidates.csv").open(newline="", encoding="utf-8") as handle:
                selected = list(csv.DictReader(handle))
            self.assertEqual([row["source_id"] for row in selected], ["eligible_high"])
            self.assertEqual(selected[0]["source_context_positive_rows"], "2")
            self.assertEqual(selected[0]["recommended_next_step"], "generate_and_review_boundary_negatives")


if __name__ == "__main__":
    unittest.main()
