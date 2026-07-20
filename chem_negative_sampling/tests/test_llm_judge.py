"""Unit tests for ``llm_judge.py`` (P3-07).

These tests use synthetic reactions + tiny CSV/JSON fixtures (no GPU,
no remote access, no LLM API keys required).  They cover:

- :class:`LocalExpertJudge` on valid/invalid/unbalanced/implausible
  reactions
- Atom balance check (via :func:`_sum_counts` / :func:`_atom_counts`)
- :class:`ReactionJudge` orchestration + majority vote
- :func:`cohen_kappa` (2-judge, from scratch) on edge cases
- :func:`compute_inter_judge_agreement` (pairwise matrix + mean)
- :func:`compute_dft_agreement` with synthetic DFT results
- :func:`load_dft_results` with multiple file shapes
- :func:`load_pc_cng_negatives` + :func:`stratified_sample_reactions`
- :func:`call_llm_judge` raises ``NotImplementedError`` offline
- :func:`run_judgment` end-to-end on a tiny CSV

The tests must run with only stdlib + RDKit + numpy installed (no API
keys, no GPU).  They are skipped gracefully when RDKit is unavailable.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

import pytest

# Make ``llm_judge`` importable both when tests live in
# /tmp/p3_files/p3_07/ (flat layout) and when they live in
# chem_negative_sampling/tests/ on the server (subdir layout).
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

try:  # server layout: chem_negative_sampling/tests/test_llm_judge.py
    from evaluation.llm_judge import (  # type: ignore[import]
        REACTION_JUDGE_PROMPT_TEMPLATE,
        AggregatedJudgment,
        Judgment,
        LLMJudgeConfig,
        LocalExpertJudge,
        ReactionJudge,
        _atom_counts,
        _build_local_judges,
        _count_bonds,
        _parse_side,
        _split_reaction,
        _sum_counts,
        call_llm_judge,
        cohen_kappa,
        compute_dft_agreement,
        compute_inter_judge_agreement,
        load_dft_results,
        load_pc_cng_negatives,
        stratified_sample_reactions,
        run_judgment,
        build_arg_parser,
        main,
    )
except ImportError:  # flat local layout
    from llm_judge import (  # type: ignore[no-redef]
        REACTION_JUDGE_PROMPT_TEMPLATE,
        AggregatedJudgment,
        Judgment,
        LLMJudgeConfig,
        LocalExpertJudge,
        ReactionJudge,
        _atom_counts,
        _build_local_judges,
        _count_bonds,
        _parse_side,
        _split_reaction,
        _sum_counts,
        call_llm_judge,
        cohen_kappa,
        compute_dft_agreement,
        compute_inter_judge_agreement,
        load_dft_results,
        load_pc_cng_negatives,
        stratified_sample_reactions,
        run_judgment,
        build_arg_parser,
        main,
    )


# RDKit guard -- most tests need it.
try:
    from rdkit import Chem  # noqa: F401
    from rdkit.Chem import rdMolDescriptors  # noqa: F401
    RDKIT_OK = True
except Exception:  # pragma: no cover
    RDKIT_OK = False

needs_rdkit = pytest.mark.skipif(
    not RDKIT_OK, reason="RDKit not installed in this environment"
)


# ---------------------------------------------------------------------------
# Fixtures: real-ish reactions
# ---------------------------------------------------------------------------

# A simple, valid, balanced, plausible reaction: SN2 of chloromethane
# with hydroxide -> methanol + chloride.
RXN_VALID_BALANCED = "C[Cl-].O>>CO.[Cl-]"

# Valid SMILES but NOT atom-balanced: missing a chlorine on the product
# side (matter creation).
RXN_UNBALANCED = "CCO>>CC"  # O and H disappear

# Invalid SMILES on the product side (garbage token).
RXN_INVALID = "CCO>>XYZnotasmiles"

# Missing '>>' separator entirely.
RXN_NO_ARROW = "CCO"

# A balanced reaction with a large bond count change (implausible by
# the bond-change heuristic).  Cyclohexane -> benzene + 3 H2 is
# technically balanced but the ring count change trips the ring
# heuristic; here we use a deliberately large rearrangement.
RXN_BOND_BLOWN = "C" + ("C" * 20) + ">>c1ccccc1.c1ccccc1.c1ccccc1"

# A balanced, plausible-looking esterification.
RXN_ESTER = "CC(=O)O.OCC>>CC(=O)OCC.O"


# ---------------------------------------------------------------------------
# Prompt template + config
# ---------------------------------------------------------------------------


def test_prompt_template_contains_required_fields():
    """The prompt template must mention all four judgment dimensions."""
    assert "is_valid" in REACTION_JUDGE_PROMPT_TEMPLATE
    assert "is_balanced" in REACTION_JUDGE_PROMPT_TEMPLATE
    assert "is_plausible" in REACTION_JUDGE_PROMPT_TEMPLATE
    assert "score" in REACTION_JUDGE_PROMPT_TEMPLATE
    assert "{reaction_smiles}" in REACTION_JUDGE_PROMPT_TEMPLATE
    # Should be formattable.
    formatted = REACTION_JUDGE_PROMPT_TEMPLATE.format(
        reaction_smiles="A>>B"
    )
    assert "A>>B" in formatted


def test_llm_judge_config_defaults():
    """LLMJudgeConfig should default to offline (no api_key)."""
    cfg = LLMJudgeConfig(model_name="gpt-4")
    assert cfg.model_name == "gpt-4"
    assert cfg.api_key is None
    assert cfg.temperature == 0.0
    assert cfg.max_tokens == 256
    assert cfg.base_url is None


# ---------------------------------------------------------------------------
# call_llm_judge (offline contract)
# ---------------------------------------------------------------------------


def test_call_llm_judge_raises_without_api_key():
    """Without an api_key the LLM judge must raise NotImplementedError."""
    cfg = LLMJudgeConfig(model_name="gpt-4")
    with pytest.raises(NotImplementedError):
        call_llm_judge("dummy prompt", cfg)


def test_call_llm_judge_raises_even_with_key():
    """Even with an api_key the offline build raises NotImplementedError
    (no SDK wired up)."""
    cfg = LLMJudgeConfig(model_name="gpt-4", api_key="sk-dummy")
    with pytest.raises(NotImplementedError):
        call_llm_judge("dummy prompt", cfg)


# ---------------------------------------------------------------------------
# Helpers: _split_reaction / _parse_side / atom counts
# ---------------------------------------------------------------------------


def test_split_reaction_basic():
    """_split_reaction splits on the first '>>'."""
    left, right = _split_reaction("A.B>>C")
    assert left == "A.B"
    assert right == "C"


def test_split_reaction_no_arrow():
    """Missing '>>' returns empty strings."""
    left, right = _split_reaction("just reactants")
    assert left == ""
    assert right == ""


def test_split_reaction_empty():
    assert _split_reaction("") == ("", "")


@needs_rdkit
def test_parse_side_drops_invalid():
    """_parse_side silently drops unparseable tokens."""
    mols = _parse_side("CCO.XYZnotasmiles")
    assert len(mols) == 1  # only CCO parsed


@needs_rdkit
def test_parse_side_empty():
    assert _parse_side("") == []
    assert _parse_side("   ") == []


@needs_rdkit
def test_atom_counts_includes_implicit_h():
    """_atom_counts should add implicit H so balance reflects total H."""
    m = Chem.MolFromSmiles("CCO")  # ethanol C2H6O
    counts = _atom_counts(m)
    assert counts.get("C") == 2
    assert counts.get("O") == 1
    assert counts.get("H") == 6  # 6 implicit H


@needs_rdkit
def test_sum_counts_across_mols():
    mols = [Chem.MolFromSmiles("CO"), Chem.MolFromSmiles("O")]
    total = _sum_counts(mols)
    # CO = CH4O, O = H2O -> total C1 H6 O2
    assert total.get("C") == 1
    assert total.get("O") == 2
    assert total.get("H") == 6


@needs_rdkit
def test_count_bonds():
    """_count_bonds sums GetNumBonds across a list of Mols."""
    mols = [Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("O")]
    # CCO has 2 bonds (C-C, C-O); O has 0 bonds.
    assert _count_bonds(mols) == 2


# ---------------------------------------------------------------------------
# LocalExpertJudge
# ---------------------------------------------------------------------------


@needs_rdkit
def test_local_judge_invalid_reaction():
    """Invalid SMILES -> score 0, is_valid False."""
    judge = LocalExpertJudge(name="t", seed=0)
    j = judge.judge(RXN_INVALID)
    assert isinstance(j, Judgment)
    assert j.judge == "t"
    assert j.is_valid is False
    assert j.is_balanced is False
    assert j.is_plausible is False
    assert j.score == 0
    assert "parse" in j.reasoning.lower() or "failed" in j.reasoning.lower()


@needs_rdkit
def test_local_judge_missing_arrow():
    """Missing '>>' -> invalid with helpful reasoning."""
    judge = LocalExpertJudge(name="t", seed=0)
    j = judge.judge(RXN_NO_ARROW)
    assert j.is_valid is False
    assert j.score == 0
    assert "separator" in j.reasoning.lower() or ">>" in j.reasoning


@needs_rdkit
def test_local_judge_unbalanced():
    """Valid SMILES but unbalanced -> score 2, is_balanced False."""
    judge = LocalExpertJudge(name="t", seed=0)
    j = judge.judge(RXN_UNBALANCED)
    assert j.is_valid is True
    assert j.is_balanced is False
    assert j.is_plausible is False
    assert j.score == 2
    assert "imbalance" in j.reasoning.lower()


@needs_rdkit
def test_local_judge_balanced_plausible():
    """A balanced + plausible reaction should score >= 6."""
    judge = LocalExpertJudge(name="t", seed=0, strictness=0)
    j = judge.judge(RXN_ESTER)
    assert j.is_valid is True
    assert j.is_balanced is True
    # Esterification should pass plausibility heuristics.
    assert j.is_plausible is True
    assert 6 <= j.score <= 10


@needs_rdkit
def test_local_judge_strictness_shifts_score():
    """strictness=+1 should give a score >= strictness=-1 for a
    plausible reaction (the strict expert is 'harder')."""
    j_lenient = LocalExpertJudge(name="L", seed=0, strictness=-1).judge(RXN_ESTER)
    j_strict = LocalExpertJudge(name="S", seed=0, strictness=+1).judge(RXN_ESTER)
    # Both plausible.
    assert j_lenient.is_plausible and j_strict.is_plausible
    # Strict score should be >= lenient (clamping to 10 may equalise).
    assert j_strict.score >= j_lenient.score


@needs_rdkit
def test_local_judge_strictness_invalid_value():
    with pytest.raises(ValueError):
        LocalExpertJudge(name="t", seed=0, strictness=5)


@needs_rdkit
def test_local_judge_seed_jitter_within_bounds():
    """Different seeds may produce different scores but always in
    [6, 10] for a plausible reaction."""
    scores = set()
    for s in range(20):
        j = LocalExpertJudge(name="t", seed=s, strictness=0).judge(RXN_ESTER)
        assert 6 <= j.score <= 10
        scores.add(j.score)
    # With 20 seeds we expect at least 2 distinct jitter values.
    assert len(scores) >= 2


@needs_rdkit
def test_local_judge_bond_change_heuristic():
    """A reaction with a huge bond count change should be flagged
    implausible."""
    judge = LocalExpertJudge(name="t", seed=0, max_bond_change=2)
    j = judge.judge(RXN_BOND_BLOWN)
    # Either invalid (if any fragment fails to parse) or implausible
    # via the bond / ring heuristic.  Either way score <= 4.
    assert j.score <= 4


@needs_rdkit
def test_local_judge_charge_conservation():
    """A reaction that changes total formal charge should be flagged
    implausible (even if atom counts match)."""
    # Na+ + Cl- -> NaCl (neutral) is balanced in atoms but the charge
    # bookkeeping changes: [Na+] + [Cl-] (charge 0) -> [NaCl] (charge 0).
    # This is actually charge-conserved.  Use a clear violation:
    # [Na+] -> [Na] (charge disappears, atom counts equal).
    judge = LocalExpertJudge(name="t", seed=0)
    j = judge.judge("[Na+]>>[Na]")
    # Either balanced but implausible (charge not conserved) or
    # unbalanced depending on how the implicit electron is counted.
    # The score must be <= 4.
    assert j.score <= 4


# ---------------------------------------------------------------------------
# ReactionJudge orchestration
# ---------------------------------------------------------------------------


@needs_rdkit
def test_reaction_judge_three_judges():
    """ReactionJudge should run 3 judges and aggregate their verdicts."""
    judges = [
        LocalExpertJudge(name="local_expert_1", seed=1, strictness=-1),
        LocalExpertJudge(name="local_expert_2", seed=2, strictness=0),
        LocalExpertJudge(name="local_expert_3", seed=3, strictness=+1),
    ]
    rj = ReactionJudge(judges)
    agg = rj.judge_reaction(RXN_ESTER, stratum="easy")
    assert isinstance(agg, AggregatedJudgment)
    assert agg.reaction_smiles == RXN_ESTER
    assert agg.stratum == "easy"
    assert len(agg.judgments) == 3
    # All 3 judges should agree the ester is plausible.
    assert agg.majority_verdict is True
    assert agg.agreement == 1.0
    # Majority score is one of the 3 judges' scores (closest to mean on
    # tie).
    individual_scores = {j.score for j in agg.judgments}
    assert agg.majority_score in individual_scores or agg.majority_score in {
        s for s in individual_scores
    }


@needs_rdkit
def test_reaction_judge_invalid_reaction_majority_false():
    """For an invalid reaction, all 3 judges should return score 0 and
    majority_verdict False."""
    judges = [
        LocalExpertJudge(name="j1", seed=1),
        LocalExpertJudge(name="j2", seed=2),
        LocalExpertJudge(name="j3", seed=3),
    ]
    rj = ReactionJudge(judges)
    agg = rj.judge_reaction(RXN_INVALID)
    assert agg.majority_verdict is False
    assert agg.majority_score == 0
    assert agg.agreement == 1.0


def test_reaction_judge_empty_raises():
    with pytest.raises(ValueError):
        ReactionJudge([])


def test_reaction_judge_name_mismatch_raises():
    j = LocalExpertJudge(name="x", seed=0)
    with pytest.raises(ValueError):
        ReactionJudge([j], judge_names=["a", "b"])


@needs_rdkit
def test_reaction_judge_batch():
    judges = [
        LocalExpertJudge(name="j1", seed=1),
        LocalExpertJudge(name="j2", seed=2),
    ]
    rj = ReactionJudge(judges)
    reactions = [(RXN_ESTER, "easy"), (RXN_UNBALANCED, "hard")]
    batch = rj.judge_batch(reactions)
    assert len(batch) == 2
    assert batch[0].stratum == "easy"
    assert batch[1].stratum == "hard"
    assert all(len(a.judgments) == 2 for a in batch)


def test_reaction_judge_majority_score_tiebreak():
    """When scores are split 1-1-1 the majority is the one closest to
    the mean."""
    # Direct test of the static helper.
    score = ReactionJudge._majority_score([6, 8, 10])
    # mean = 8 -> 8 wins.
    assert score == 8


def test_reaction_judge_majority_score_clear_winner():
    score = ReactionJudge._majority_score([7, 7, 9])
    assert score == 7


def test_reaction_judge_majority_verdict_tie_goes_false():
    """A 1-1 tie on a 2-judge verdict should resolve to False
    (conservative)."""
    assert ReactionJudge._majority_verdict([True, False]) is False
    assert ReactionJudge._majority_verdict([True, True, False]) is True
    assert ReactionJudge._majority_verdict([]) is False


def test_reaction_judge_agreement_fraction():
    assert ReactionJudge._agreement([True, True, True], True) == 1.0
    assert ReactionJudge._agreement([True, False, True], True) == pytest.approx(2 / 3)
    assert ReactionJudge._agreement([], True) == 0.0


# ---------------------------------------------------------------------------
# Cohen's kappa
# ---------------------------------------------------------------------------


def test_cohen_kappa_perfect_agreement():
    """Identical score lists -> kappa == 1.0."""
    a = [8, 7, 9, 6, 8]
    b = [8, 7, 9, 6, 8]
    assert cohen_kappa(a, b) == pytest.approx(1.0)


def test_cohen_kappa_no_agreement_random():
    """Fully disjoint scores with no overlap -> kappa can be 0 or
    negative depending on marginals; check it's <= 0."""
    a = [0, 0, 0, 0]
    b = [10, 10, 10, 10]
    k = cohen_kappa(a, b)
    # p_o = 0, p_e = 0 (no shared categories) -> kappa = 0/1 = 0.
    assert k == pytest.approx(0.0)


def test_cohen_kappa_partial_agreement():
    """A mix of agreement + disagreement should give kappa in (0, 1)."""
    a = [5, 6, 7, 5, 6, 7, 5, 6]
    b = [5, 6, 7, 6, 5, 7, 5, 6]
    k = cohen_kappa(a, b)
    assert 0.0 < k < 1.0


def test_cohen_kappa_empty_lists():
    assert cohen_kappa([], []) == 0.0


def test_cohen_kappa_length_mismatch_raises():
    with pytest.raises(ValueError):
        cohen_kappa([1, 2, 3], [1, 2])


def test_cohen_kappa_degenerate_single_category():
    """If both judges assign the same single score to every item, kappa
    is undefined; we return 1.0 (full agreement)."""
    a = [5, 5, 5]
    b = [5, 5, 5]
    assert cohen_kappa(a, b) == 1.0


def test_cohen_kappa_negative_possible():
    """Worse-than-chance agreement should give kappa < 0."""
    # Judge 1 always says 0, judge 2 says 10 half the time and 0 the
    # other half -- p_o = 0.5, p_e = 0.5 -> kappa = 0.0.  To get a
    # clearly negative kappa we need anti-correlation.
    a = [0, 10, 0, 10]
    b = [10, 0, 10, 0]
    k = cohen_kappa(a, b)
    assert k < 0.0


# ---------------------------------------------------------------------------
# compute_inter_judge_agreement
# ---------------------------------------------------------------------------


def test_inter_judge_agreement_three_judges_perfect():
    """Three identical judges -> kappa matrix is all 1.0, mean 1.0."""
    scores = [[8, 7, 9], [8, 7, 9], [8, 7, 9]]
    result = compute_inter_judge_agreement(scores)
    matrix = result["kappa_pairwise"]
    assert len(matrix) == 3
    assert all(len(row) == 3 for row in matrix)
    for i in range(3):
        for j in range(3):
            assert matrix[i][j] == pytest.approx(1.0)
    assert result["mean_kappa"] == pytest.approx(1.0)


def test_inter_judge_agreement_empty():
    result = compute_inter_judge_agreement([])
    assert result == {"kappa_pairwise": [], "mean_kappa": 0.0}


def test_inter_judge_agreement_single_judge():
    result = compute_inter_judge_agreement([[1, 2, 3]])
    assert result["mean_kappa"] == 0.0
    assert result["kappa_pairwise"] == [[1.0]]


def test_inter_judge_agreement_length_mismatch_raises():
    with pytest.raises(ValueError):
        compute_inter_judge_agreement([[1, 2, 3], [1, 2]])


def test_inter_judge_agreement_symmetric():
    """The kappa matrix should be symmetric."""
    s1 = [8, 7, 9, 6, 5]
    s2 = [8, 6, 9, 7, 5]
    s3 = [7, 7, 8, 6, 5]
    result = compute_inter_judge_agreement([s1, s2, s3])
    matrix = result["kappa_pairwise"]
    for i in range(3):
        for j in range(3):
            assert matrix[i][j] == pytest.approx(matrix[j][i])


# ---------------------------------------------------------------------------
# DFT agreement + load_dft_results
# ---------------------------------------------------------------------------


def _make_agg(rxn: str, verdict: bool) -> AggregatedJudgment:
    """Helper: build a minimal AggregatedJudgment for DFT tests."""
    return AggregatedJudgment(
        reaction_smiles=rxn,
        stratum="easy",
        judgments=[],
        majority_score=8 if verdict else 0,
        majority_verdict=verdict,
        agreement=1.0,
    )


def test_compute_dft_agreement_full():
    judgments = [
        _make_agg("A>>B", True),
        _make_agg("C>>D", False),
    ]
    dft = {
        "A>>B": {"is_plausible": True},
        "C>>D": {"is_plausible": False},
    }
    result = compute_dft_agreement(judgments, dft)
    assert result["pct_agree"] == 1.0
    assert result["n_compared"] == 2
    assert result["n_skipped"] == 0


def test_compute_dft_agreement_partial():
    judgments = [
        _make_agg("A>>B", True),
        _make_agg("C>>D", False),  # disagrees with DFT
        _make_agg("E>>F", True),   # not in DFT (skipped)
    ]
    dft = {
        "A>>B": {"is_plausible": True},
        "C>>D": {"is_plausible": True},  # judgment says False -> disagree
    }
    result = compute_dft_agreement(judgments, dft)
    assert result["n_compared"] == 2
    assert result["n_skipped"] == 1
    assert result["pct_agree"] == 0.5


def test_compute_dft_agreement_empty():
    result = compute_dft_agreement([], {})
    assert result["pct_agree"] == 0.0
    assert result["n_compared"] == 0


def test_load_dft_results_missing_dir(tmp_path):
    """A non-existent directory should yield an empty dict."""
    assert load_dft_results(str(tmp_path / "nope")) == {}


def test_load_dft_results_results_json_shape_a(tmp_path):
    """results.json with {rxn: {is_plausible: ...}} shape."""
    dft_dir = tmp_path / "dft"
    dft_dir.mkdir()
    payload = {
        "A>>B": {"is_plausible": True, "score": -10.5},
        "C>>D": {"is_plausible": False, "score": 25.0},
    }
    (dft_dir / "results.json").write_text(json.dumps(payload))
    out = load_dft_results(str(dft_dir))
    assert set(out.keys()) == {"A>>B", "C>>D"}
    assert out["A>>B"]["is_plausible"] is True
    assert out["C>>D"]["is_plausible"] is False


def test_load_dft_results_results_json_shape_b(tmp_path):
    """results.json with {per_reaction: [...]} shape."""
    dft_dir = tmp_path / "dft"
    dft_dir.mkdir()
    payload = {
        "per_reaction": [
            {"reaction_smiles": "A>>B", "is_plausible": True},
            {"reaction_smiles": "C>>D", "is_plausible": False},
        ]
    }
    (dft_dir / "results.json").write_text(json.dumps(payload))
    out = load_dft_results(str(dft_dir))
    assert len(out) == 2
    assert out["A>>B"]["is_plausible"] is True


def test_load_dft_results_csv(tmp_path):
    """dft_validation.csv with reaction_smiles + is_plausible columns."""
    dft_dir = tmp_path / "dft"
    dft_dir.mkdir()
    rows = [
        {"reaction_smiles": "A>>B", "is_plausible": "true"},
        {"reaction_smiles": "C>>D", "is_plausible": "false"},
    ]
    with open(dft_dir / "dft_validation.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["reaction_smiles", "is_plausible"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    out = load_dft_results(str(dft_dir))
    assert out["A>>B"]["is_plausible"] is True
    assert out["C>>D"]["is_plausible"] is False


def test_load_dft_results_fallback_json_scan(tmp_path):
    """If results.json + csv are absent, scan other *.json files."""
    dft_dir = tmp_path / "dft"
    dft_dir.mkdir()
    payload = {
        "per_reaction": [
            {"reaction_smiles": "X>>Y", "is_plausible": True},
        ]
    }
    (dft_dir / "other.json").write_text(json.dumps(payload))
    out = load_dft_results(str(dft_dir))
    assert "X>>Y" in out


def test_load_dft_results_invalid_json_skipped(tmp_path):
    """Invalid JSON files should be silently skipped."""
    dft_dir = tmp_path / "dft"
    dft_dir.mkdir()
    (dft_dir / "broken.json").write_text("{not valid json")
    out = load_dft_results(str(dft_dir))
    assert out == {}


# ---------------------------------------------------------------------------
# load_pc_cng_negatives + stratified_sample_reactions
# ---------------------------------------------------------------------------


def _write_pc_cng_csv(path: Path, rows: List[dict]) -> None:
    """Write a tiny PC-CNG-style CSV."""
    with open(path, "w", newline="") as fh:
        fieldnames = ["reaction_smiles", "hard_score", "label_type"]
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def test_load_pc_cng_negatives_missing_file(tmp_path):
    """Missing file -> empty list (graceful)."""
    out = load_pc_cng_negatives(str(tmp_path / "nope.csv"))
    assert out == []


def test_load_pc_cng_negatives_missing_column(tmp_path):
    """CSV without reaction_smiles column -> empty list."""
    p = tmp_path / "bad.csv"
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["foo", "bar"])
        w.writeheader()
        w.writerow({"foo": "1", "bar": "2"})
    assert load_pc_cng_negatives(str(p)) == []


def test_load_pc_cng_negatives_basic(tmp_path):
    """A well-formed CSV loads all rows with reaction_smiles + hard_score."""
    p = tmp_path / "negs.csv"
    rows = [
        {"reaction_smiles": "A>>B", "hard_score": "0.1", "label_type": "neg"},
        {"reaction_smiles": "C>>D", "hard_score": "0.9", "label_type": "neg"},
    ]
    _write_pc_cng_csv(p, rows)
    out = load_pc_cng_negatives(str(p))
    assert len(out) == 2
    assert out[0]["reaction_smiles"] == "A>>B"
    assert out[0]["hard_score"] == "0.1"


def test_load_pc_cng_negatives_max_n(tmp_path):
    """max_n should subsample the rows."""
    p = tmp_path / "negs.csv"
    rows = [
        {"reaction_smiles": f"A{i}>>B{i}", "hard_score": str(i / 10), "label_type": "neg"}
        for i in range(10)
    ]
    _write_pc_cng_csv(p, rows)
    out = load_pc_cng_negatives(str(p), max_n=3)
    assert len(out) == 3


def test_stratified_sample_with_hard_score():
    """When hard_score is present, sampling should produce all 3 strata."""
    rows = [
        {"reaction_smiles": f"A{i}>>B{i}", "hard_score": str(i / 30)}
        for i in range(30)
    ]
    sampled = stratified_sample_reactions(rows, n_samples=9, seed=42)
    strata = {s for _, s in sampled}
    assert strata == {"easy", "medium", "hard"}
    assert len(sampled) == 9
    # Each stratum gets 3 samples.
    counts = {s: 0 for s in ("easy", "medium", "hard")}
    for _, s in sampled:
        counts[s] += 1
    assert counts["easy"] == 3
    assert counts["medium"] == 3
    assert counts["hard"] == 3


def test_stratified_sample_without_hard_score():
    """Without hard_score, strata are assigned randomly but all present."""
    rows = [{"reaction_smiles": f"A{i}>>B{i}"} for i in range(30)]
    sampled = stratified_sample_reactions(rows, n_samples=9, seed=42)
    assert len(sampled) == 9
    strata = {s for _, s in sampled}
    assert strata == {"easy", "medium", "hard"}


def test_stratified_sample_zero_n():
    rows = [{"reaction_smiles": "A>>B", "hard_score": "0.5"}]
    assert stratified_sample_reactions(rows, n_samples=0) == []


def test_stratified_sample_empty_rows():
    assert stratified_sample_reactions([], n_samples=10) == []


def test_stratified_sample_more_than_available():
    """Asking for more samples than rows -> returns all rows."""
    rows = [
        {"reaction_smiles": f"A{i}>>B{i}", "hard_score": str(i)}
        for i in range(5)
    ]
    sampled = stratified_sample_reactions(rows, n_samples=100, seed=0)
    assert len(sampled) <= 5  # cannot exceed available rows


# ---------------------------------------------------------------------------
# _build_local_judges
# ---------------------------------------------------------------------------


def test_build_local_judges_basic():
    judges = _build_local_judges(
        ["local_expert_1", "local_expert_2", "local_expert_3"],
        [42],
    )
    assert len(judges) == 3
    assert judges[0].name == "local_expert_1"
    assert judges[1].name == "local_expert_2"
    assert judges[2].name == "local_expert_3"
    # strictness cycles through -1, 0, +1.
    assert judges[0].strictness == -1
    assert judges[1].strictness == 0
    assert judges[2].strictness == 1


def test_build_local_judges_empty_raises():
    with pytest.raises(ValueError):
        _build_local_judges([], [1])


def test_build_local_judges_no_seeds_raises():
    with pytest.raises(ValueError):
        _build_local_judges(["a"], [])


# ---------------------------------------------------------------------------
# run_judgment end-to-end
# ---------------------------------------------------------------------------


@needs_rdkit
def test_run_judgment_end_to_end(tmp_path):
    """Full pipeline on a tiny CSV: 6 rows, 6 samples, 3 judges."""
    p = tmp_path / "negs.csv"
    rows = [
        {"reaction_smiles": "CC(=O)O.OCC>>CC(=O)OCC.O", "hard_score": "0.1", "label_type": "neg"},
        {"reaction_smiles": "CCO>>CO", "hard_score": "0.2", "label_type": "neg"},
        {"reaction_smiles": "CCO>>CC", "hard_score": "0.5", "label_type": "neg"},
        {"reaction_smiles": "C[Cl-].O>>CO.[Cl-]", "hard_score": "0.7", "label_type": "neg"},
        {"reaction_smiles": "XYZ>>ABC", "hard_score": "0.8", "label_type": "neg"},
        {"reaction_smiles": "CC(=O)O.OCC>>CC(=O)OCC.O", "hard_score": "0.9", "label_type": "neg"},
    ]
    _write_pc_cng_csv(p, rows)

    out_dir = tmp_path / "out"
    payload = run_judgment(
        pc_cng_negatives_csv=str(p),
        dft_results_dir=None,
        output_dir=str(out_dir),
        n_samples=6,
        judge_names=("local_expert_1", "local_expert_2", "local_expert_3"),
        seeds=(42,),
    )
    # Check payload shape.
    assert payload["n_samples"] == 6
    assert payload["judges"] == ["local_expert_1", "local_expert_2", "local_expert_3"]
    assert "inter_judge_agreement" in payload
    assert "dft_agreement" in payload
    assert payload["dft_agreement"]["n_compared"] == 0
    assert "go_no_go" in payload
    assert payload["go_no_go"] in {"GO", "NO-GO"}
    # Each judgment has 3 judges.
    for j in payload["judgments"]:
        assert len(j["judges"]) == 3
    # Files written.
    assert (out_dir / "judgments.json").is_file()
    assert (out_dir / "summary.md").is_file()
    # JSON is valid.
    with open(out_dir / "judgments.json") as fh:
        json.load(fh)


@needs_rdkit
def test_run_judgment_with_dft(tmp_path):
    """Pipeline should compute DFT agreement when DFT dir is present."""
    p = tmp_path / "negs.csv"
    rows = [
        {"reaction_smiles": "CC(=O)O.OCC>>CC(=O)OCC.O", "hard_score": "0.1", "label_type": "neg"},
        {"reaction_smiles": "CCO>>CC", "hard_score": "0.9", "label_type": "neg"},
    ]
    _write_pc_cng_csv(p, rows)

    dft_dir = tmp_path / "dft"
    dft_dir.mkdir()
    dft_payload = {
        "CC(=O)O.OCC>>CC(=O)OCC.O": {"is_plausible": True},
        "CCO>>CC": {"is_plausible": False},
    }
    (dft_dir / "results.json").write_text(json.dumps(dft_payload))

    payload = run_judgment(
        pc_cng_negatives_csv=str(p),
        dft_results_dir=str(dft_dir),
        output_dir=None,
        n_samples=2,
        judge_names=("j1", "j2", "j3"),
        seeds=(42,),
    )
    assert payload["dft_agreement"]["n_compared"] == 2
    # All 3 local experts agree on the ester (plausible) and on the
    # unbalanced reaction (not plausible) -> majority verdict matches
    # DFT -> 100% agreement.
    assert payload["dft_agreement"]["pct_agree"] == 1.0


def test_run_judgment_no_negatives_raises(tmp_path):
    """If both pc-cng-negatives and fallback are missing/empty, raise."""
    with pytest.raises(ValueError):
        run_judgment(
            pc_cng_negatives_csv=None,
            dft_results_dir=None,
            output_dir=None,
            n_samples=10,
            fallback_csv=None,
        )


@needs_rdkit
def test_run_judgment_fallback_csv_used(tmp_path):
    """If the primary CSV is missing, the fallback CSV is used."""
    fallback = tmp_path / "fallback.csv"
    rows = [
        {"reaction_smiles": "CC(=O)O.OCC>>CC(=O)OCC.O", "hard_score": "0.1", "label_type": "neg"},
    ]
    _write_pc_cng_csv(fallback, rows)
    payload = run_judgment(
        pc_cng_negatives_csv=str(tmp_path / "missing.csv"),
        dft_results_dir=None,
        output_dir=None,
        n_samples=1,
        fallback_csv=str(fallback),
    )
    assert payload["n_samples"] == 1
    assert payload["meta"]["pc_cng_negatives_csv"] == str(fallback)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_build_arg_parser_defaults():
    parser = build_arg_parser()
    args = parser.parse_args([])
    assert args.n_samples == 100
    assert args.judges == ["local_expert_1", "local_expert_2", "local_expert_3"]
    assert args.seeds == [20260710]


def test_build_arg_parser_custom():
    parser = build_arg_parser()
    args = parser.parse_args([
        "--n-samples", "50",
        "--judges", "a,b,c",
        "--seeds", "1,2,3",
    ])
    assert args.n_samples == 50
    assert args.judges == ["a", "b", "c"]
    assert args.seeds == [1, 2, 3]


def test_main_missing_csv_returns_nonzero(capsys):
    """main() should return 1 and print an error when no CSV is found."""
    rc = main([
        "--pc-cng-negatives", "/tmp/definitely_missing_p3_07.csv",
        "--n-samples", "5",
    ])
    assert rc == 1
    captured = capsys.readouterr()
    assert "ERROR" in captured.err


@needs_rdkit
def test_main_full_run(tmp_path, capsys):
    """A full CLI run with a tiny CSV should succeed (rc 0)."""
    p = tmp_path / "negs.csv"
    rows = [
        {"reaction_smiles": "CC(=O)O.OCC>>CC(=O)OCC.O", "hard_score": "0.1", "label_type": "neg"},
        {"reaction_smiles": "CCO>>CC", "hard_score": "0.9", "label_type": "neg"},
    ]
    _write_pc_cng_csv(p, rows)
    out_dir = tmp_path / "out"
    rc = main([
        "--pc-cng-negatives", str(p),
        "--output-dir", str(out_dir),
        "--n-samples", "2",
        "--judges", "local_expert_1,local_expert_2,local_expert_3",
        "--seeds", "20260710",
    ])
    assert rc == 0
    captured = capsys.readouterr()
    assert "Done" in captured.out
    assert (out_dir / "judgments.json").is_file()
    assert (out_dir / "summary.md").is_file()
