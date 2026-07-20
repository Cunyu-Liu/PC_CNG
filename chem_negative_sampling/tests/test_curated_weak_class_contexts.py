from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class CuratedWeakClassContextsTest(unittest.TestCase):
    def test_builds_hitea_and_rule_classified_contexts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hitea = root / "ullmann.csv"
            uspto = root / "uspto.csv"
            output = root / "curated.csv"
            summary = root / "summary.json"

            with hitea.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["REACTION_ID", "RXN_SMILES", "Product_Yield_PCT_Area_UV"])
                writer.writerow(["cu1", "Clc1ccccc1.N>>Nc1ccccc1", "75"])

            with uspto.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["source_id", "reaction_smiles", "yield"])
                writer.writerow(["amide1", "CC(=O)Cl.NC>>CC(=O)NC", "80"])

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pc_cng.build_curated_weak_class_contexts",
                    "--hitea-cleaned-csv",
                    f"Cu coupling={hitea}",
                    "--uspto-csv",
                    str(uspto),
                    "--output",
                    str(output),
                    "--summary",
                    str(summary),
                ],
                check=True,
            )

            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            classes = {row["reaction_class"] for row in rows}
            self.assertIn("Cu coupling", classes)
            self.assertIn("Amide coupling", classes)
            self.assertTrue(all(row["label_type"] == "positive" for row in rows))
            payload = json.loads(summary.read_text(encoding="utf-8"))
            self.assertEqual(payload["reaction_classes"]["Cu coupling"], 1)
            self.assertEqual(payload["reaction_classes"]["Amide coupling"], 1)


if __name__ == "__main__":
    unittest.main()
