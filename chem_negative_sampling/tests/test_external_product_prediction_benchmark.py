from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class ExternalProductPredictionBenchmarkTest(unittest.TestCase):
    def test_build_and_evaluate_strict_candidate_benchmark(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real_csv = root / "real.csv"
            synthetic_csv = root / "synthetic.csv"
            beam_csv = root / "beams.tsv"
            candidates_csv = root / "candidates.csv"
            contexts_csv = root / "contexts.csv"
            chemformer_input = root / "chemformer_input.csv"
            build_summary = root / "build_summary.json"

            real_csv.write_text(
                "\n".join(
                    [
                        "source_id,reaction_smiles,label_type,split,source,reaction_class",
                        "rxn1,CCO>>CC=O,positive,val,mini,oxidation",
                        "rxn2,CCBr>>CCO,positive,test,mini,substitution",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            synthetic_csv.write_text(
                "\n".join(
                    [
                        "source_id,candidate_reaction,review_status,action_family",
                        "rxn1,CCO>>CC,keep_synthetic_negative,truncation",
                        "rxn2,CCBr>>CCBr,keep_synthetic_negative,identity",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            beam_csv.write_text(
                "\n".join(
                    [
                        "sampled_smiles_1\tloglikelihood_1\tsampled_smiles_2\tloglikelihood_2",
                        "CC=O\t-0.1\tCC\t-1.0",
                        "CCBr\t-0.2\tCCO\t-0.4",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pc_cng.build_external_product_prediction_candidate_set",
                    "--real-csv",
                    str(real_csv),
                    "--synthetic-csv",
                    str(synthetic_csv),
                    "--external-beam-csv",
                    str(beam_csv),
                    "--output",
                    str(candidates_csv),
                    "--summary",
                    str(build_summary),
                    "--contexts-output",
                    str(contexts_csv),
                    "--chemformer-input-output",
                    str(chemformer_input),
                ],
                check=True,
            )

            build_payload = json.loads(build_summary.read_text(encoding="utf-8"))
            self.assertEqual(build_payload["contexts"], 2)
            self.assertGreaterEqual(build_payload["candidate_rows"], 4)

            with candidates_csv.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(any("chemformer_beam" in row["candidate_source"] for row in rows))
            external_scores = root / "external_scores.csv"
            pc_scores = root / "pc_scores.csv"
            with external_scores.open("w", newline="", encoding="utf-8") as handle:
                fieldnames = list(rows[0]) + ["lm_score"]
                writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for row in rows:
                    score = 2.0 if row["label"] == "1" else 0.0
                    out = dict(row)
                    out["lm_score"] = score
                    writer.writerow(out)
            with pc_scores.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["group_id", "reaction_smiles", "score"])
                writer.writeheader()
                skipped_negative = False
                for row in rows:
                    if row["label"] == "0" and not skipped_negative:
                        skipped_negative = True
                        continue
                    writer.writerow(
                        {
                            "group_id": row["group_id"],
                            "reaction_smiles": row["candidate_reaction"],
                            "score": 1.0 if row["label"] == "1" else 0.1,
                        }
                    )

            out_dir = root / "benchmark"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pc_cng.evaluate_external_product_prediction_benchmark",
                    "--candidate-csv",
                    str(candidates_csv),
                    "--external-score",
                    f"external={external_scores}:lm_score",
                    "--pc-cng-score-csv",
                    str(pc_scores),
                    "--pc-cng-invalid-negative-score",
                    "0.0",
                    "--output-dir",
                    str(out_dir),
                ],
                check=True,
            )

            summary = json.loads((out_dir / "benchmark_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["candidate_rows_evaluated"], len(rows))
            self.assertEqual(summary["score_metrics"]["external"]["overall"]["top1"], 1.0)
            self.assertEqual(
                summary["pc_cng_score"]["invalid_negative_fill"]["filled_negative_rows"],
                1,
            )
            self.assertIn("selected_hybrid", summary)
            self.assertTrue((out_dir / "paper_table.md").exists())

            validity_out = root / "benchmark_validity_aware"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pc_cng.evaluate_external_product_prediction_benchmark",
                    "--candidate-csv",
                    str(candidates_csv),
                    "--external-score",
                    f"external={external_scores}:lm_score",
                    "--pc-cng-score-csv",
                    str(pc_scores),
                    "--output-dir",
                    str(validity_out),
                    "--allow-incomplete-groups",
                ],
                check=True,
            )

            validity_summary = json.loads(
                (validity_out / "benchmark_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(validity_summary["candidate_rows_evaluated"], len(rows))
            self.assertEqual(validity_summary["score_metrics"]["external"]["overall"]["top1"], 1.0)
            self.assertGreater(validity_summary["pc_cng_score"]["attach"]["missing"], 0)
            self.assertFalse(
                validity_summary["hybrid_complete_group_filter"]["same_rows_as_evaluation"]
            )
            self.assertIn("selected_hybrid", validity_summary)
            self.assertTrue((validity_out / "paper_table.md").exists())


if __name__ == "__main__":
    unittest.main()
