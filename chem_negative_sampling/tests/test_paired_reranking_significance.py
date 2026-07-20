from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class PairedRerankingSignificanceTest(unittest.TestCase):
    def test_outputs_paired_delta_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline.csv"
            candidate = root / "candidate.csv"
            out_dir = root / "out"

            for path, scores in [
                (baseline, {"g1_pos": 0.2, "g1_neg": 0.8, "g2_pos": 0.9, "g2_neg": 0.1}),
                (candidate, {"g1_pos": 0.9, "g1_neg": 0.1, "g2_pos": 0.8, "g2_neg": 0.2}),
            ]:
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["group_id", "label", "score", "reaction_smiles"])
                    writer.writerow(["g1", 1, scores["g1_pos"], "A>>B"])
                    writer.writerow(["g1", 0, scores["g1_neg"], "A>>C"])
                    writer.writerow(["g2", 1, scores["g2_pos"], "D>>E"])
                    writer.writerow(["g2", 0, scores["g2_neg"], "D>>F"])

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pc_cng.paired_reranking_significance",
                    "--baseline",
                    str(baseline),
                    "--candidate",
                    str(candidate),
                    "--output-dir",
                    str(out_dir),
                    "--bootstrap-iterations",
                    "100",
                ],
                check=True,
            )

            payload = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["common_evaluable_groups"], 2)
            top1 = payload["summary"]["top1"]
            self.assertAlmostEqual(top1["baseline_mean"], 0.5)
            self.assertAlmostEqual(top1["candidate_mean"], 1.0)
            self.assertAlmostEqual(top1["delta_mean"], 0.5)
            self.assertTrue((out_dir / "paired_group_deltas.csv").exists())
            self.assertTrue((out_dir / "summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
