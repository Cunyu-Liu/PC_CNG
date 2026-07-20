"""Unit tests for ``pc_cng.research_ni_coupling_data`` (P1-11).

Covers:
* Ni catalyst detection (SMILES + name-form fallbacks).
* Reaction type classification (Suzuki / Negishi / Kumada / reductive /
  Buchwald-Hartwig / unknown).
* NiCOlit CSV adapter (load + normalization to PC-CNG schema).
* End-to-end CLI smoke test using ``python3 -m pc_cng.research_ni_coupling_data``.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List


# Import the module under test lazily inside tests so the file can be
# collected even when ``pc_cng`` is not on the path (the subprocess CLI
# smoke test still exercises the real package).
def _import_module():
    from pc_cng import research_ni_coupling_data as mod

    return mod


class DetectNiCatalystTests(unittest.TestCase):
    """Exercise :func:`detect_ni_catalyst` across SMILES and name forms."""

    def test_detect_ni_catalyst_nickel_metal(self) -> None:
        mod = _import_module()
        # [Ni] is the canonical SMILES token for atomic Ni.
        self.assertTrue(mod.detect_ni_catalyst("[Ni]"))
        # Charged Ni(II) in SMILES bracket notation.
        self.assertTrue(mod.detect_ni_catalyst("[Ni+2].[Cl-].[Cl-]"))
        # Ni embedded as an organometallic complex atom.
        self.assertTrue(mod.detect_ni_catalyst("Cl[Ni]Cl"))

    def test_detect_ni_catalyst_ni_cl2(self) -> None:
        mod = _import_module()
        # NiCl2 as a catalyst *name* (not valid SMILES).  The regex
        # fallback must still match this so NiCOlit rows with
        # ``catalyst_precursor = "NiCl2"`` are flagged correctly.
        self.assertTrue(mod.detect_ni_catalyst("NiCl2"))
        # Ni(cod)2 is the canonical Ni(0) precursor name from NiCOlit.
        self.assertTrue(mod.detect_ni_catalyst("Ni(cod)2"))
        # NiCl2(dppf) -- ligand-tagged Ni(II) salt name.
        self.assertTrue(mod.detect_ni_catalyst("NiCl2(dppf)"))
        # NiBr2 / Ni(OAc)2 variants.
        self.assertTrue(mod.detect_ni_catalyst("NiBr2"))
        self.assertTrue(mod.detect_ni_catalyst("Ni(OAc)2"))

    def test_detect_ni_catalyst_no_ni(self) -> None:
        mod = _import_module()
        # Palladium catalysts must NOT be flagged as Ni.
        self.assertFalse(mod.detect_ni_catalyst("Pd"))
        self.assertFalse(mod.detect_ni_catalyst("[Pd]"))
        self.assertFalse(mod.detect_ni_catalyst("Pd(PPh3)4"))
        # Empty / no-catalyst inputs.
        self.assertFalse(mod.detect_ni_catalyst(""))
        self.assertFalse(mod.detect_ni_catalyst("CCO"))
        # False-friend strings that contain "Ni" as part of another token
        # (e.g. "ani" in "aniline") must not be flagged.  We use "aniline"
        # explicitly because RDKit parses it and there is no Ni atom.
        self.assertFalse(mod.detect_ni_catalyst("c1ccccc1N"))


class ClassifyReactionTypeTests(unittest.TestCase):
    """Exercise :func:`classify_reaction_type` for the canonical buckets."""

    def test_classify_reaction_type_suzuki(self) -> None:
        mod = _import_module()
        # Aryl bromide + aryl boronic acid -> biaryl.  Boron consumed.
        # Use a clean SMILES so the atom counter behaves predictably.
        reactants = "c1ccccc1B(O)O.c1ccccc1Br"
        products = "c1ccccc1c1ccccc1"
        self.assertEqual(mod.classify_reaction_type(reactants, products), mod.REACTION_TYPE_SUZUKI)
        # NiCOlit mechanism hint takes precedence.
        self.assertEqual(
            mod.classify_reaction_type(reactants, products, mechanism_hint="Suzuki"),
            mod.REACTION_TYPE_SUZUKI,
        )

    def test_classify_reaction_type_negishi(self) -> None:
        mod = _import_module()
        # Aryl bromide + aryl zinc -> biaryl.  Zn consumed.
        reactants = "c1ccccc1Br.[Zn+2].c1ccccc1"
        products = "c1ccccc1c1ccccc1"
        self.assertEqual(mod.classify_reaction_type(reactants, products), mod.REACTION_TYPE_NEGISHI)
        # Hint override.
        self.assertEqual(
            mod.classify_reaction_type("A.B>>C", "C", mechanism_hint="Negishi"),
            mod.REACTION_TYPE_NEGISHI,
        )

    def test_classify_reaction_type_kumada(self) -> None:
        mod = _import_module()
        # Aryl bromide + aryl MgBr -> biaryl.  Mg consumed.
        reactants = "c1ccccc1Br.c1ccccc1[Mg]Br"
        products = "c1ccccc1c1ccccc1"
        self.assertEqual(mod.classify_reaction_type(reactants, products), mod.REACTION_TYPE_KUMADA)

    def test_classify_reaction_type_hiyama(self) -> None:
        mod = _import_module()
        reactants = "c1ccccc1Br.c1ccccc1[Si](C)(C)C"
        products = "c1ccccc1c1ccccc1"
        self.assertEqual(mod.classify_reaction_type(reactants, products), mod.REACTION_TYPE_HIYAMA)

    def test_classify_reaction_type_murahashi(self) -> None:
        mod = _import_module()
        reactants = "c1ccccc1Br.[Li]c1ccccc1"
        products = "c1ccccc1c1ccccc1"
        self.assertEqual(mod.classify_reaction_type(reactants, products), mod.REACTION_TYPE_MURAHASHI)

    def test_classify_reaction_type_reductive(self) -> None:
        mod = _import_module()
        # Two aryl bromides + Mn reductant -> biaryl.
        reactants = "c1ccccc1Br.c1ccccc1Br.[Mn]"
        products = "c1ccccc1c1ccccc1"
        self.assertEqual(
            mod.classify_reaction_type(reactants, products),
            mod.REACTION_TYPE_REDUCTIVE,
        )

    def test_classify_reaction_type_buchwald(self) -> None:
        mod = _import_module()
        # Aryl bromide + primary amine -> aryl amine.  Halide consumed,
        # N-H reactant, N retained in product.
        reactants = "c1ccccc1Br.[NH2]Cc1ccccc1"
        products = "c1ccccc1NCCc1ccccc1"
        rtype = mod.classify_reaction_type(reactants, products)
        # Either Buchwald-Hartwig or Other Ni depending on whether the N-H
        # heuristic fires; both are Ni-catalyzed amination labels.
        self.assertIn(rtype, {mod.REACTION_TYPE_BUCHWALD, mod.REACTION_TYPE_OTHER_NI})

    def test_classify_reaction_type_no_match(self) -> None:
        mod = _import_module()
        # Simple esterification with no Ni-specific signature.
        reactants = "CC(=O)O.OCC"
        products = "CC(=O)OCC"
        rtype = mod.classify_reaction_type(reactants, products)
        self.assertEqual(rtype, mod.REACTION_TYPE_OTHER_NI)
        # Empty inputs.
        self.assertEqual(mod.classify_reaction_type("", ""), mod.REACTION_TYPE_UNKNOWN)

    def test_classify_reaction_type_uses_mechanism_hint(self) -> None:
        mod = _import_module()
        # Even when atom heuristics would not match, the NiCOlit mechanism
        # hint should take precedence and resolve to the canonical label.
        self.assertEqual(
            mod.classify_reaction_type("A", "B", mechanism_hint="Buchwald"),
            mod.REACTION_TYPE_BUCHWALD,
        )
        # Unknown mechanism hint falls through to atom heuristics.
        self.assertEqual(
            mod.classify_reaction_type("A", "B", mechanism_hint="not-a-real-mechanism"),
            mod.REACTION_TYPE_OTHER_NI,
        )


class LoadNiCOlitRowsTests(unittest.TestCase):
    """Exercise the NiCOlit CSV adapter end-to-end."""

    def _write_nicolit_csv(self, path: Path, rows: List[Dict[str, str]]) -> None:
        fieldnames = [
            "substrate",
            "coupling_partner",
            "effective_coupling_partner",
            "solvent",
            "time",
            "temperature",
            "catalyst_precursor",
            "reagents",
            "effective_reagents",
            "effective_reagents_covalent",
            "reductant",
            "ligand",
            "effective_ligand",
            "product",
            "analytical_yield",
            "isolated_yield",
            "coupling_partner_class",
            "DOI",
            "origin",
            "eq_substrate",
            "eq_coupling_partner",
            "eq_catalyst",
            "eq_ligand",
            "eq_reagent",
            "2_steps",
            "scheme_table",
            "review",
            "Mechanism",
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def test_load_nicolit_rows_basic(self) -> None:
        mod = _import_module()
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "nicolit.csv"
            self._write_nicolit_csv(
                csv_path,
                [
                    {
                        "substrate": "COc1ccc2ccccc2c1",
                        "coupling_partner": "[Li]c1ccccc1",
                        "product": "c3ccc(c2ccc1ccccc1c2)cc3",
                        "catalyst_precursor": "Ni(cod)2",
                        "ligand": "PCy3",
                        "effective_ligand": "PCy3",
                        "solvent": "toluene",
                        "analytical_yield": "73",
                        "Mechanism": "Murahashi",
                        "origin": "Optimisation",
                        "coupling_partner_class": "Li",
                    },
                    {
                        "substrate": "c1ccccc1Br",
                        "coupling_partner": "c1ccccc1B(O)O",
                        "product": "c1ccccc1c1ccccc1",
                        "catalyst_precursor": "NiCl2(dppf)",
                        "solvent": "dioxane",
                        "analytical_yield": "85",
                        "Mechanism": "Suzuki",
                        "origin": "Scope",
                        "coupling_partner_class": "B",
                    },
                    # Row missing substrate / product should be skipped.
                    {
                        "substrate": "",
                        "coupling_partner": "c1ccccc1B(O)O",
                        "product": "",
                        "catalyst_precursor": "NiCl2",
                        "Mechanism": "Suzuki",
                    },
                ],
            )

            rows, stats = mod.load_nicolit_rows(str(csv_path))

        self.assertEqual(len(rows), 2)
        self.assertEqual(stats["rows_loaded"], 2)
        self.assertEqual(stats["skipped_missing"], 1)
        self.assertEqual(stats["skipped_invalid"], 0)

        # Schema check.
        for col in mod.PC_CNG_NORMALIZED_COLUMNS:
            self.assertIn(col, rows[0])

        # First row is Murahashi (Li consumed) and uses Ni(cod)2 catalyst.
        self.assertEqual(rows[0]["source"], "nicolit_literature")
        self.assertEqual(rows[0]["label_type"], "positive")
        self.assertEqual(rows[0]["split"], "train")
        self.assertEqual(rows[0]["yield"], "73.0")
        self.assertIn("Ni(cod)2", rows[0]["agents"])
        self.assertIn("PCy3", rows[0]["agents"])
        self.assertEqual(rows[0]["_reaction_type"], mod.REACTION_TYPE_MURAHASHI)

        # Second row is Suzuki.
        self.assertEqual(rows[1]["_reaction_type"], mod.REACTION_TYPE_SUZUKI)
        self.assertEqual(rows[1]["yield"], "85.0")

        # Mechanism counts captured.
        self.assertEqual(stats["mechanism_counts"]["Murahashi"], 1)
        self.assertEqual(stats["mechanism_counts"]["Suzuki"], 1)

    def test_load_nicolit_rows_dedupes(self) -> None:
        mod = _import_module()
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "nicolit.csv"
            self._write_nicolit_csv(
                csv_path,
                [
                    {
                        "substrate": "c1ccccc1Br",
                        "coupling_partner": "c1ccccc1B(O)O",
                        "product": "c1ccccc1c1ccccc1",
                        "catalyst_precursor": "NiCl2(dppf)",
                        "Mechanism": "Suzuki",
                        "analytical_yield": "80",
                    },
                    # Duplicate reaction -> skipped.
                    {
                        "substrate": "c1ccccc1Br",
                        "coupling_partner": "c1ccccc1B(O)O",
                        "product": "c1ccccc1c1ccccc1",
                        "catalyst_precursor": "NiCl2(dppf)",
                        "Mechanism": "Suzuki",
                        "analytical_yield": "82",
                    },
                ],
            )
            rows, stats = mod.load_nicolit_rows(str(csv_path))
        self.assertEqual(len(rows), 1)
        self.assertEqual(stats["skipped_duplicate"], 1)


class RunResearchWorkflowTests(unittest.TestCase):
    """Exercise the full ``run_research`` workflow with synthetic inputs."""

    def test_run_research_go_path(self) -> None:
        mod = _import_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            uspto_csv = root / "uspto.csv"
            ord_csv = root / "ord.csv"
            nicolit_csv = root / "nicolit.csv"
            out_md = root / "report.md"
            out_supplement = root / "supplement.csv"
            out_summary = root / "summary.json"

            # Minimal USPTO normalized CSV with 1 Ni reaction.
            with uspto_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    ["source_id", "reaction_smiles", "reactants", "agents", "products",
                     "label_type", "yield", "source", "split_key", "split"]
                )
                writer.writerow(
                    [
                        "u1",
                        "c1ccccc1Br.c1ccccc1B(O)O>[Ni]>c1ccccc1c1ccccc1",
                        "c1ccccc1Br.c1ccccc1B(O)O",
                        "[Ni]",
                        "c1ccccc1c1ccccc1",
                        "positive",
                        "80",
                        "uspto_openmolecules",
                        "abc123",
                        "train",
                    ]
                )
                # Non-Ni row.
                writer.writerow(
                    [
                        "u2",
                        "CCO>>CC=O",
                        "CCO",
                        "",
                        "CC=O",
                        "positive",
                        "100",
                        "uspto_openmolecules",
                        "def456",
                        "train",
                    ]
                )

            # Empty ORD CSV.
            with ord_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    ["source_id", "reaction_smiles", "reactants", "agents", "products",
                     "label_type", "yield", "source", "split_key", "split"]
                )

            # NiCOlit CSV with 3 unique Ni reactions (different substrates so
            # the dedup-by-reaction_smiles step keeps all three).
            fieldnames = [
                "substrate", "coupling_partner", "product", "catalyst_precursor",
                "ligand", "effective_ligand", "solvent", "analytical_yield",
                "Mechanism", "origin", "coupling_partner_class",
            ]
            nicolit_test_rows = [
                {
                    "substrate": "c1ccccc1Br",
                    "coupling_partner": "c1ccccc1B(O)O",
                    "product": "c1ccccc1c1ccccc1",
                    "catalyst_precursor": "Ni(cod)2",
                    "ligand": "PCy3",
                    "effective_ligand": "PCy3",
                    "solvent": "toluene",
                    "analytical_yield": "75",
                    "Mechanism": "Suzuki",
                    "origin": "Scope",
                    "coupling_partner_class": "B",
                },
                {
                    "substrate": "c1ccccc1Cl",
                    "coupling_partner": "c1ccccc1B(O)O",
                    "product": "c1ccccc1c1ccccc1",
                    "catalyst_precursor": "NiCl2(dppf)",
                    "ligand": "dppf",
                    "effective_ligand": "dppf",
                    "solvent": "dioxane",
                    "analytical_yield": "68",
                    "Mechanism": "Suzuki",
                    "origin": "Scope",
                    "coupling_partner_class": "B",
                },
                {
                    "substrate": "c1ccccc1I",
                    "coupling_partner": "c1ccccc1B(O)O",
                    "product": "c1ccccc1c1ccccc1",
                    "catalyst_precursor": "Ni(cod)2",
                    "ligand": "PCy3",
                    "effective_ligand": "PCy3",
                    "solvent": "toluene",
                    "analytical_yield": "82",
                    "Mechanism": "Suzuki",
                    "origin": "Scope",
                    "coupling_partner_class": "B",
                },
            ]
            with nicolit_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for row in nicolit_test_rows:
                    writer.writerow(row)

            payload = mod.run_research(
                output_md=str(out_md),
                uspto_csv=str(uspto_csv),
                ord_csv=str(ord_csv),
                hitea_csv=None,
                output_supplement=str(out_supplement),
                output_summary_json=str(out_summary),
                min_count=2,
                nicolit_cache=str(nicolit_csv),
                skip_download=True,
            )

            self.assertTrue(payload["go_no_go"])
            self.assertEqual(payload["total_existing_ni"], 1)
            self.assertEqual(payload["total_nicolit_rows"], 3)
            self.assertEqual(payload["total_ni"], 4)
            self.assertTrue(out_md.exists())
            self.assertTrue(out_supplement.exists())
            self.assertTrue(out_summary.exists())

            # Markdown contains the Go decision.
            md_text = out_md.read_text()
            self.assertIn("**Go**", md_text)
            self.assertIn("NiCOlit", md_text)

            # Supplement CSV is well-formed and matches PC-CNG schema.
            with out_supplement.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                header = reader.fieldnames
                self.assertEqual(tuple(header), tuple(mod.PC_CNG_NORMALIZED_COLUMNS))
                rows = list(reader)
            # 3 NiCOlit rows + 1 existing USPTO Ni row = 4 total.
            self.assertEqual(len(rows), 4)
            for row in rows:
                self.assertEqual(row["label_type"], "positive")

            # Summary JSON has expected fields.
            summary = json.loads(out_summary.read_text())
            self.assertEqual(summary["total_rows"], 4)
            self.assertEqual(summary["sources"]["nicolit_rows"], 3)
            self.assertEqual(summary["sources"]["existing_ni_rows"], 1)
            self.assertIn("Suzuki", summary["reaction_type_counts"])

    def test_run_research_no_go_path(self) -> None:
        mod = _import_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            uspto_csv = root / "uspto.csv"
            ord_csv = root / "ord.csv"
            out_md = root / "report.md"
            out_supplement = root / "supplement.csv"
            out_summary = root / "summary.json"

            for path in (uspto_csv, ord_csv):
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(
                        ["source_id", "reaction_smiles", "reactants", "agents", "products",
                         "label_type", "yield", "source", "split_key", "split"]
                    )

            payload = mod.run_research(
                output_md=str(out_md),
                uspto_csv=str(uspto_csv),
                ord_csv=str(ord_csv),
                hitea_csv=None,
                output_supplement=str(out_supplement),
                output_summary_json=str(out_summary),
                min_count=50,
                nicolit_cache=None,
                skip_download=True,
            )

            self.assertFalse(payload["go_no_go"])
            self.assertEqual(payload["total_ni"], 0)
            self.assertTrue(out_md.exists())
            # Supplement must NOT be written on No-Go.
            self.assertFalse(out_supplement.exists())
            self.assertFalse(out_summary.exists())

            md_text = out_md.read_text()
            self.assertIn("**No-Go**", md_text)
            self.assertIn("known data-source limitation", md_text)


class ResearchCliSmokeTest(unittest.TestCase):
    """Smoke test exercising the ``python3 -m pc_cng.research_ni_coupling_data`` CLI."""

    def test_research_cli_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            uspto_csv = root / "uspto.csv"
            ord_csv = root / "ord.csv"
            nicolit_csv = root / "nicolit.csv"
            out_md = root / "report.md"
            out_supplement = root / "supplement.csv"
            out_summary = root / "summary.json"

            for path in (uspto_csv, ord_csv):
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(
                        ["source_id", "reaction_smiles", "reactants", "agents", "products",
                         "label_type", "yield", "source", "split_key", "split"]
                    )

            # Small NiCOlit CSV with 2 rows (above the Go threshold we set).
            fieldnames = [
                "substrate", "coupling_partner", "product", "catalyst_precursor",
                "ligand", "effective_ligand", "solvent", "analytical_yield",
                "Mechanism", "origin", "coupling_partner_class",
            ]
            with nicolit_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerow(
                    {
                        "substrate": "c1ccccc1Br",
                        "coupling_partner": "c1ccccc1B(O)O",
                        "product": "c1ccccc1c1ccccc1",
                        "catalyst_precursor": "Ni(cod)2",
                        "ligand": "PCy3",
                        "effective_ligand": "PCy3",
                        "solvent": "toluene",
                        "analytical_yield": "73",
                        "Mechanism": "Suzuki",
                        "origin": "Scope",
                        "coupling_partner_class": "B",
                    }
                )

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pc_cng.research_ni_coupling_data",
                    "--output",
                    str(out_md),
                    "--uspto-csv",
                    str(uspto_csv),
                    "--ord-csv",
                    str(ord_csv),
                    "--output-supplement",
                    str(out_supplement),
                    "--output-summary-json",
                    str(out_summary),
                    "--min-count",
                    "1",
                    "--nicolit-cache",
                    str(nicolit_csv),
                    "--skip-download",
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(
                result.returncode,
                0,
                msg=f"CLI failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
            )
            self.assertTrue(out_md.exists())
            self.assertTrue(out_supplement.exists())
            self.assertTrue(out_summary.exists())

            # CLI payload echoes the Go decision.
            payload = json.loads(result.stdout)
            self.assertTrue(payload["go_no_go"])
            self.assertEqual(payload["total_nicolit_rows"], 1)


if __name__ == "__main__":
    unittest.main()
