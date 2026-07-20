from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class NiAtomicSupportAuditTest(unittest.TestCase):
    def test_audit_detects_ni_atoms_and_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "hitea_full_normalized.csv"
            second = root / "uspto_openmolecules_normalized.csv"
            out_dir = root / "out"

            header = ["source_id", "reaction_smiles", "split", "reaction_class"]
            with first.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(header)
                writer.writerows(
                    [
                        ["h0", "CCO>>CC=O", "train", "Alcohol oxidation"],
                        ["h1", "CCBr.[Ni]>>CCBr", "test", "Ni coupling"],
                        ["h2", "CCO>O[Ni]O>CC=O", "val", "Ni catalyst"],
                    ]
                )

            with second.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(header)
                writer.writerows(
                    [
                        ["u0", "CCN>>CCN", "train", ""],
                        ["u1", "CCCl.[Ni+2]>>CCCl", "test", ""],
                    ]
                )

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pc_cng.audit_ni_atomic_support",
                    "--input-csv",
                    str(first),
                    "--input-csv",
                    str(second),
                    "--output-dir",
                    str(out_dir),
                ],
                check=True,
            )

            payload = json.loads((out_dir / "ni_atomic_support_audit.json").read_text())
            datasets = {row["dataset"]: row for row in payload["datasets"]}

            self.assertEqual(datasets["hitea_full_normalized"]["reaction_rows"], 3)
            self.assertEqual(datasets["hitea_full_normalized"]["ni_reactions"], 2)
            self.assertEqual(datasets["hitea_full_normalized"]["distinct_ni_parent_reactants"], 2)
            self.assertEqual(datasets["uspto_openmolecules_normalized"]["ni_reactions"], 1)
            self.assertTrue((out_dir / "ni_atomic_examples.csv").exists())
            self.assertTrue((out_dir / "ni_atomic_support_audit.md").exists())
            self.assertIn("outputs", payload)


if __name__ == "__main__":
    unittest.main()
