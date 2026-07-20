"""
Unit tests for extract_conditions.py (P3-04).

Covers main functions:
  - parse_agents_field
  - classify_agent (catalyst/solvent/reagent)
  - extract_conditions_from_row
  - extract_conditions (CSV -> JSON end-to-end)

HC #4: coverage target >=80%.
"""
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

# Make chem_negative_sampling package importable.
_HERE = Path(__file__).resolve().parent
_CNS_ROOT = _HERE.parent
if str(_CNS_ROOT) not in sys.path:
    sys.path.insert(0, str(_CNS_ROOT))

from data.extract_conditions import (
    CATALYST_ELEMENTS,
    COMMON_SOLVENTS,
    classify_agent,
    extract_conditions,
    extract_conditions_from_row,
    main,
    parse_agents_field,
)


def test_parse_agents_empty() -> None:
    """Empty or None inputs should yield empty lists."""
    assert parse_agents_field("") == []
    assert parse_agents_field(None) == []
    assert parse_agents_field("   ") == []


def test_parse_agents_single() -> None:
    """Single SMILES should produce a single-element list."""
    assert parse_agents_field("CCO") == ["CCO"]


def test_parse_agents_multiple() -> None:
    """Comma-separated SMILES should be split and stripped."""
    result = parse_agents_field("CCO, c1ccccc1, [Pd]")
    assert result == ["CCO", "c1ccccc1", "[Pd]"]


def test_parse_agents_trailing_comma() -> None:
    """Trailing commas should not produce empty entries."""
    assert parse_agents_field("CCO,") == ["CCO"]
    assert parse_agents_field(",CCO,") == ["CCO"]


def test_classify_catalyst_elements() -> None:
    """Each catalyst element SMILES should classify as catalyst."""
    for elem in ["[Pd]", "[Pt]", "[Ru]", "[Cu]", "[Ni]", "[Fe]"]:
        assert classify_agent(elem) == "catalyst", f"Failed: {elem}"


def test_classify_catalyst_in_complex_smiles() -> None:
    """Catalyst embedded in complex SMILES should still be detected."""
    assert classify_agent("CC(=O)[Pd]") == "catalyst"
    assert classify_agent("c1ccccc1[Pd]") == "catalyst"
    assert classify_agent("C1CCCC1[Cu]") == "catalyst"


def test_classify_solvent_simple() -> None:
    """Common solvent SMILES should classify as solvent."""
    assert classify_agent("O") == "solvent"  # water
    assert classify_agent("CCO") == "solvent"  # EtOH
    assert classify_agent("CO") == "solvent"  # MeOH
    assert classify_agent("CC#N") == "solvent"  # MeCN


def test_classify_solvent_alternate_forms() -> None:
    """Alternate canonical forms should also classify as solvent."""
    assert classify_agent("CS(=O)C") == "solvent"  # DMSO
    assert classify_agent("ClCCl") == "solvent"  # DCM


def test_classify_reagent() -> None:
    """Non-catalyst, non-solvent SMILES should be reagent."""
    # Acetic anhydride - not in our solvent list
    assert classify_agent("CC(=O)OC(C)=O") == "reagent"
    # Some salt-like reagent
    assert classify_agent("[Na+].[Cl-]") == "reagent"


def test_classify_empty_smiles() -> None:
    """Empty SMILES should default to 'reagent'."""
    assert classify_agent("") == "reagent"


def test_extract_conditions_from_row_full() -> None:
    """Full row with mixed agents should classify correctly.

    Note: c1ccccc1 (benzene) is in our COMMON_SOLVENTS list, so both CCO
    and c1ccccc1 should be classified as solvent.
    """
    row = {
        "source_id": "rxn1",
        "reaction_smiles": "CCO>>CC(=O)OCC",
        "agents": "[Pd],CCO,c1ccccc1",
        "split": "train",
    }
    result = extract_conditions_from_row(row)
    assert result["source_id"] == "rxn1"
    assert result["reaction_smiles"] == "CCO>>CC(=O)OCC"
    assert result["catalyst"] == "[Pd]"
    # CCO and benzene are both in COMMON_SOLVENTS -> joined with '.'.
    assert result["solvent"] == "CCO.c1ccccc1"
    assert result["reagent"] == ""
    assert result["temperature"] is None
    assert result["split"] == "train"


def test_extract_conditions_from_row_with_actual_reagent() -> None:
    """A row where an agent is genuinely a reagent (not solvent/catalyst)."""
    row = {
        "source_id": "rxn1b",
        "reaction_smiles": "CCO>>CC(=O)OCC",
        "agents": "[Pd],CCO,CC(=O)NC",  # acetamide is NOT a solvent
        "split": "train",
    }
    result = extract_conditions_from_row(row)
    assert result["catalyst"] == "[Pd]"
    assert result["solvent"] == "CCO"
    assert result["reagent"] == "CC(=O)NC"


def test_extract_conditions_from_row_no_agents() -> None:
    """Missing/empty agents should yield empty catalyst/solvent/reagent."""
    row = {
        "source_id": "rxn2",
        "reaction_smiles": "A>>B",
        "agents": "",
        "split": "test",
    }
    result = extract_conditions_from_row(row)
    assert result["catalyst"] == ""
    assert result["solvent"] == ""
    assert result["reagent"] == ""
    assert result["split"] == "test"


def test_extract_conditions_from_row_missing_fields() -> None:
    """Missing fields should not raise; defaults applied."""
    row = {"source_id": "rxn3"}
    result = extract_conditions_from_row(row)
    assert result["source_id"] == "rxn3"
    assert result["reaction_smiles"] == ""
    assert result["catalyst"] == ""
    assert result["solvent"] == ""
    assert result["reagent"] == ""
    assert result["split"] == "train"  # default


def test_extract_conditions_csv_to_json() -> None:
    """End-to-end CSV -> JSON should produce correct records."""
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "test.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["source_id", "reaction_smiles", "agents", "split"],
            )
            writer.writeheader()
            writer.writerow({
                "source_id": "1",
                "reaction_smiles": "A>>B",
                "agents": "[Pd],CCO",
                "split": "train",
            })
            writer.writerow({
                "source_id": "2",
                "reaction_smiles": "C>>D",
                "agents": "O",
                "split": "test",
            })

        json_path = Path(tmp) / "out.json"
        n = extract_conditions(str(csv_path), str(json_path))
        assert n == 2
        assert os.path.exists(json_path)

        with open(json_path) as f:
            data = json.load(f)
        assert len(data) == 2
        assert data[0]["catalyst"] == "[Pd]"
        assert data[0]["solvent"] == "CCO"
        assert data[0]["source_id"] == "1"
        assert data[1]["solvent"] == "O"
        assert data[1]["source_id"] == "2"


def test_extract_conditions_missing_input() -> None:
    """Missing input CSV should raise FileNotFoundError."""
    try:
        extract_conditions("/nonexistent/path.csv", "/tmp/out.json")
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_extract_conditions_creates_output_dir() -> None:
    """Parent output directory should be created if missing."""
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "test.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["source_id", "reaction_smiles", "agents", "split"]
            )
            writer.writeheader()
            writer.writerow({
                "source_id": "1", "reaction_smiles": "A>>B",
                "agents": "CCO", "split": "train",
            })
        nested = Path(tmp) / "deep" / "nested" / "out.json"
        n = extract_conditions(str(csv_path), str(nested))
        assert n == 1
        assert nested.exists()


def test_common_solvents_dict_has_water() -> None:
    """Sanity check: water should be in the solvent dictionary."""
    assert "O" in COMMON_SOLVENTS
    assert COMMON_SOLVENTS["O"] == "water"


def test_catalyst_elements_set() -> None:
    """Sanity check: all catalyst elements should be in the set."""
    for elem in ["Pd", "Pt", "Ru", "Cu", "Ni", "Fe"]:
        assert elem in CATALYST_ELEMENTS


def test_main_cli(tmp_path, monkeypatch) -> None:
    """CLI main() should produce JSON output."""
    csv_path = tmp_path / "in.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["source_id", "reaction_smiles", "agents", "split"]
        )
        writer.writeheader()
        writer.writerow({
            "source_id": "1", "reaction_smiles": "A>>B",
            "agents": "[Pd]", "split": "train",
        })
    json_path = tmp_path / "out.json"
    monkeypatch.setattr(
        "sys.argv",
        ["extract_conditions.py",
         "--input", str(csv_path),
         "--output", str(json_path)],
    )
    main()
    assert json_path.exists()
    with open(json_path) as f:
        data = json.load(f)
    assert data[0]["catalyst"] == "[Pd]"


def test_multiple_catalysts_joined() -> None:
    """Multiple catalysts in agents should be joined with '.'."""
    row = {
        "source_id": "rxn_multi",
        "reaction_smiles": "A>>B",
        "agents": "[Pd],[Cu]",
        "split": "train",
    }
    result = extract_conditions_from_row(row)
    assert result["catalyst"] == "[Pd].[Cu]"
    assert result["solvent"] == ""
    assert result["reagent"] == ""
