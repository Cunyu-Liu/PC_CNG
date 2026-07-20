from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class SplitChemformerInputTest(unittest.TestCase):
    def test_split_preserves_header_and_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "chemformer_input.tsv"
            out_dir = root / "chunks"
            with input_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["reactants", "products", "set"], delimiter="\t")
                writer.writeheader()
                for idx in range(5):
                    writer.writerow({"reactants": f"R{idx}", "products": f"P{idx}", "set": "test"})

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pc_cng.split_chemformer_input",
                    "--input",
                    str(input_path),
                    "--output-dir",
                    str(out_dir),
                    "--chunk-size",
                    "2",
                    "--prefix",
                    "mini",
                ],
                check=True,
            )

            manifest = json.loads((out_dir / "mini_chunks_manifest.json").read_text())
            self.assertEqual(manifest["total_rows"], 5)
            self.assertEqual(manifest["chunk_count"], 3)
            self.assertEqual([row["rows"] for row in manifest["chunks"]], [2, 2, 1])
            self.assertTrue((out_dir / "mini_chunks_manifest.md").exists())
            for chunk in manifest["chunks"]:
                self.assertTrue(chunk["sha256"])
                with Path(chunk["path"]).open(newline="", encoding="utf-8") as handle:
                    reader = csv.DictReader(handle, delimiter="\t")
                    self.assertEqual(reader.fieldnames, ["reactants", "products", "set"])


if __name__ == "__main__":
    unittest.main()
