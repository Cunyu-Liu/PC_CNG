from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class ExternalContextExpansionTest(unittest.TestCase):
    def test_selects_safe_unique_contexts_for_external_bridge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            existing = root / "existing_contexts.csv"
            source = root / "uspto.csv"
            out_dir = root / "out"

            with existing.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "row_index",
                        "group_id",
                        "source_id",
                        "reactants",
                        "agents",
                        "observed_product",
                        "split",
                        "dataset",
                        "reaction_class",
                    ]
                )
                writer.writerow(
                    [
                        "0",
                        "external_product_prediction|mini|train|old1",
                        "old1",
                        "CCBr.N",
                        "",
                        "CCN",
                        "train",
                        "mini",
                        "",
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
                        ["eligible_two", "CCF.O>>CCFO", "CCF.O", "", "CCFO", "positive", "70", "uspto", "k2", "val"],
                        ["existing_overlap", "CCBr.N>>CCN", "CCBr.N", "", "CCN", "positive", "80", "uspto", "k3", "test"],
                        ["cross_test", "CCI.O>>CCIO", "CCI.O", "", "CCIO", "positive", "80", "uspto", "k4", "test"],
                        ["cross_train", "CCI.O>>CCIO", "CCI.O", "", "CCIO", "positive", "80", "uspto", "k4", "train"],
                        ["non_positive", "CCC.O>>CCCO", "CCC.O", "", "CCCO", "real_negative", "", "uspto", "k5", "test"],
                    ]
                )

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pc_cng.select_external_product_prediction_contexts",
                    "--existing-context-csv",
                    str(existing),
                    "--source-csv",
                    str(source),
                    "--output-dir",
                    str(out_dir),
                    "--target-total-contexts",
                    "3",
                ],
                check=True,
            )

            payload = json.loads((out_dir / "external_product_prediction_context_expansion_summary.json").read_text())
            self.assertEqual(payload["existing_context_groups"], 1)
            self.assertEqual(payload["needed_additional_contexts"], 2)
            self.assertEqual(payload["selected_contexts"], 2)
            self.assertEqual(payload["merged_contexts"], 3)
            self.assertTrue(payload["selected_can_cover_target"])
            self.assertEqual(payload["reject_reason_counts"]["context_seen_in_existing_external"], 1)
            self.assertEqual(payload["reject_reason_counts"]["source_context_cross_split"], 2)

            with (out_dir / "external_product_prediction_context_expansion.csv").open(newline="", encoding="utf-8") as handle:
                selected = list(csv.DictReader(handle))
            self.assertEqual([row["source_id"] for row in selected], ["eligible_high", "eligible_two"])
            self.assertTrue((out_dir / "external_product_prediction_context_expansion_chemformer_input.csv").exists())
            self.assertTrue((out_dir / "external_product_prediction_contexts_merged.csv").exists())


if __name__ == "__main__":
    unittest.main()
