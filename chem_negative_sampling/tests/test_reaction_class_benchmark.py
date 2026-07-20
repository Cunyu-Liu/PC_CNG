from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class ReactionClassBenchmarkTest(unittest.TestCase):
    def test_reaction_class_flags_and_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scores = root / "scores.csv"
            out_dir = root / "out"
            rows = [
                ["g1", "rxn1", "1", "1.0", "test", "mini", "weak"],
                ["g1", "rxn1n", "0", "0.9", "test", "mini", "weak"],
                ["g2", "rxn2", "1", "1.0", "test", "mini", "strong"],
                ["g2", "rxn2n", "0", "0.1", "test", "mini", "strong"],
                ["g3", "rxn3", "1", "1.0", "val", "mini", "strong"],
                ["g3", "rxn3n", "0", "0.2", "val", "mini", "strong"],
            ]
            with scores.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["group_id", "reaction_smiles", "label", "score", "split", "dataset", "reaction_class"])
                writer.writerows(rows)

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pc_cng.analyze_reaction_class_benchmark",
                    "--score-csv",
                    f"toy={scores}:score",
                    "--output-dir",
                    str(out_dir),
                    "--min-groups",
                    "2",
                    "--weak-top1",
                    "0.95",
                ],
                check=True,
            )
            payload = json.loads((out_dir / "reaction_class_benchmark.json").read_text(encoding="utf-8"))
            classes = {
                row["reaction_class"]: row
                for row in payload["summaries"]["toy"]["class_summary"]
            }
            self.assertEqual(classes["weak"]["status"], "low_support")
            self.assertEqual(classes["strong"]["status"], "ok")
            self.assertTrue((out_dir / "reaction_class_summary.md").exists())
            self.assertTrue((out_dir / "reaction_class_by_split.md").exists())


if __name__ == "__main__":
    unittest.main()
