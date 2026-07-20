from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class ExternalContextPcCngCandidateGenerationTest(unittest.TestCase):
    def test_generate_candidates_for_context_csv_and_rebuild_candidate_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contexts_csv = root / "contexts.csv"
            synthetic_csv = root / "targeted_pc_cng.csv"
            synthetic_summary = root / "targeted_pc_cng_summary.json"
            candidates_csv = root / "candidates.csv"
            build_summary = root / "build_summary.json"

            contexts_csv.write_text(
                "\n".join(
                    [
                        "row_index,group_id,source_id,reactants,agents,observed_product,split,dataset,reaction_class",
                        "0,g1,rxn1,CCO,,CC=O,test,mini,oxidation",
                        "1,g2,rxn2,CCBr,,CCO,test,mini,substitution",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pc_cng.generate_external_context_pc_cng_candidates",
                    "--context-csv",
                    str(contexts_csv),
                    "--output",
                    str(synthetic_csv),
                    "--summary",
                    str(synthetic_summary),
                ],
                check=True,
            )

            payload = json.loads(synthetic_summary.read_text(encoding="utf-8"))
            self.assertEqual(payload["processed_contexts"], 2)
            self.assertEqual(payload["contexts_with_candidates"], 2)
            self.assertGreaterEqual(payload["generated_candidates"], 2)

            with synthetic_csv.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(all(row["review_status"] == "keep_synthetic_negative" for row in rows))
            self.assertTrue(all(row["task"] == "forward_outcome" for row in rows))
            self.assertTrue(all(row["candidate_product"] != row["parent_product"] for row in rows))

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pc_cng.build_external_product_prediction_candidate_set",
                    "--real-csv",
                    str(contexts_csv),
                    "--beam-context-csv",
                    str(contexts_csv),
                    "--synthetic-csv",
                    str(synthetic_csv),
                    "--output",
                    str(candidates_csv),
                    "--summary",
                    str(build_summary),
                ],
                check=True,
            )

            build_payload = json.loads(build_summary.read_text(encoding="utf-8"))
            self.assertEqual(build_payload["contexts"], 2)
            self.assertGreaterEqual(build_payload["candidate_source_counts"].get("pc_cng", 0), 2)

            with candidates_csv.open(newline="", encoding="utf-8") as handle:
                candidate_rows = list(csv.DictReader(handle))
            pc_cng_groups = {
                row["group_id"]
                for row in candidate_rows
                if "pc_cng" in row["candidate_source"] and row["label"] == "0"
            }
            self.assertEqual(pc_cng_groups, {"g1", "g2"})


if __name__ == "__main__":
    unittest.main()
