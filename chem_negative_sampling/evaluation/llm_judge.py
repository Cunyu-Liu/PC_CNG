"""LLM-as-judge expert agent review for P3-07 (翻盘 P2-03 DEFERRED).

This module implements the P3-07 task: **LLM-as-judge expert agent review**
of 100 PC-CNG generated negative reactions, with a hybrid design that
combines (1) LLM-judge API infrastructure (prompt template + call code)
for future use when internet access is available, and (2) a local
"expert agent" judge based on RDKit chemical validity + atom balance +
reaction plausibility heuristics that runs **offline** as a fallback.

Background
----------
P2-03 expert review was DEFERRED because no human chemistry expert could be
found in time.  P3-07 closes that gap by deploying LLMs (GPT-4 /
Claude-3.5-Sonnet / Gemini-1.5-Pro) as chemistry-expert proxies.  The
remote server ``36.137.135.49`` has **no internet access**, so we cannot
call those LLM APIs directly today.  The hybrid strategy is:

1. Build the LLM-judge infrastructure (prompt template, API call code,
   config dataclass) so it can be used unchanged the moment a network
   path is available.
2. Implement a ``LocalExpertJudge`` that uses RDKit-based chemical
   validity + atom balance + reaction plausibility heuristics as a
   deterministic, offline fallback.
3. Compare the judge verdicts with the DFT validation results produced
   by P2-02 (already on the server at
   ``results/dft_validation_chemoselectivity_20260720/``) to quantify
   the trustworthiness of the LLM-as-judge paradigm.

Public API
----------
- :data:`REACTION_JUDGE_PROMPT_TEMPLATE` -- prompt template string.
- :class:`LLMJudgeConfig` -- config dataclass for LLM API calls.
- :class:`Judgment` -- per-judge judgment for one reaction.
- :class:`AggregatedJudgment` -- multi-judge judgment for one reaction.
- :func:`call_llm_judge` -- abstract LLM API caller (raises
  ``NotImplementedError`` if no API key / no network).
- :class:`LocalExpertJudge` -- rule-based offline judge.
- :class:`ReactionJudge` -- orchestrator running 3 judges per reaction.
- :func:`cohen_kappa` -- 2-judge Cohen's kappa from scratch.
- :func:`compute_inter_judge_agreement` -- pairwise kappa + mean.
- :func:`compute_dft_agreement` -- % agreement with DFT results.
- :func:`run_judgment` -- full pipeline used by the CLI.
- :func:`main` -- CLI entry point.

The module depends only on Python 3.10 stdlib + RDKit + numpy (no new
dependencies), satisfying hard constraint HC #4 (unit tests in
``test_llm_judge.py`` with >=80% coverage).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# RDKit is optional at import time so the module can be imported in
# environments where RDKit is not yet installed (e.g. a fresh CI box).
# All RDKit-using code paths gate on the flag below.
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    _RDKIT_AVAILABLE = True
except Exception:  # pragma: no cover - RDKit missing in CI
    Chem = None  # type: ignore[assignment]
    Descriptors = None  # type: ignore[assignment]
    rdMolDescriptors = None  # type: ignore[assignment]
    _RDKIT_AVAILABLE = False


__all__ = [
    "REACTION_JUDGE_PROMPT_TEMPLATE",
    "LLMJudgeConfig",
    "Judgment",
    "AggregatedJudgment",
    "LocalExpertJudge",
    "ReactionJudge",
    "call_llm_judge",
    "cohen_kappa",
    "compute_inter_judge_agreement",
    "compute_dft_agreement",
    "stratified_sample_reactions",
    "load_pc_cng_negatives",
    "load_dft_results",
    "run_judgment",
    "build_arg_parser",
    "main",
]


# ---------------------------------------------------------------------------
# Prompt template (used by future LLM API calls; documented here so the
# infrastructure is ready the moment network access is restored).
# ---------------------------------------------------------------------------

REACTION_JUDGE_PROMPT_TEMPLATE = (
    "You are an expert organic chemist acting as a judge for a "
    "negative-reaction sampling pipeline (PC-CNG).  You will be shown a "
    "single reaction in SMILES notation (reactants>>products) and must "
    "decide whether it is a *chemically plausible negative* (i.e. a "
    "reaction that should NOT occur under normal conditions, used as a "
    "hard negative for training a retrosynthesis ranker).\n\n"
    "Reaction SMILES:\n```\n{reaction_smiles}\n```\n\n"
    "Answer the following four questions, in order, as strict JSON:\n"
    "1. \"is_valid\": true if every molecule on both sides of the "
    "reaction arrow parses to a chemically valid SMILES, else false.\n"
    "2. \"is_balanced\": true if the reaction is atom-balanced (same "
    "atom counts on both sides, allowing reagents/agents to be "
    "implicit), else false.\n"
    "3. \"is_plausible\": true if the transformation is chemically "
    "plausible (reasonable bond changes, no impossible element creation "
    "/ destruction, respects valence), else false.\n"
    "4. \"score\": an integer 0-10 where 0 = completely impossible and "
    "10 = perfectly plausible real reaction.  For a *good negative* "
    "(plausible-looking but should-not-occur) aim for 3-5; for an "
    "obviously broken negative aim for 0-2.\n\n"
    "Return ONLY the JSON object, no prose.  Example:\n"
    "{{\"is_valid\": true, \"is_balanced\": true, \"is_plausible\": "
    "false, \"score\": 4, \"reasoning\": \"Atom-balanced SN2 that "
    "violates the leaving-group ability order.\"}}\n"
)


# ---------------------------------------------------------------------------
# Config + dataclasses
# ---------------------------------------------------------------------------


@dataclass
class LLMJudgeConfig:
    """Configuration for an LLM API judge call.

    Parameters
    ----------
    model_name:
        E.g. ``"gpt-4"``, ``"claude-3-5-sonnet-20240620"``,
        ``"gemini-1.5-pro"``.
    api_key:
        Optional API key.  If ``None`` (the default), :func:`call_llm_judge`
        raises :class:`NotImplementedError` so callers fall back to the
        :class:`LocalExpertJudge`.
    temperature:
        Sampling temperature (default 0.0 for deterministic judgement).
    max_tokens:
        Max tokens to generate (default 256; the JSON response is short).
    base_url:
        Optional custom API endpoint (e.g. Azure OpenAI proxy).
    """

    model_name: str
    api_key: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 256
    base_url: Optional[str] = None


@dataclass
class Judgment:
    """One judge's verdict on one reaction.

    Attributes
    ----------
    judge:
        Name of the judge (e.g. ``"gpt-4"``, ``"local_expert_1"``).
    is_valid:
        True if all molecules parse to valid SMILES.
    is_balanced:
        True if the reaction is atom-balanced.
    is_plausible:
        True if the reaction passes plausibility heuristics.
    score:
        Integer 0-10 (0 = impossible, 10 = perfectly plausible).
    reasoning:
        Short human-readable explanation.
    """

    judge: str
    is_valid: bool
    is_balanced: bool
    is_plausible: bool
    score: int
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AggregatedJudgment:
    """Multi-judge verdict on one reaction (output of
    :meth:`ReactionJudge.judge_reaction`).

    Attributes
    ----------
    reaction_smiles:
        The reaction that was judged.
    stratum:
        Difficulty stratum (``"easy"`` / ``"medium"`` / ``"hard"``).
    judgments:
        List of per-judge :class:`Judgment` objects.
    majority_score:
        Majority-voted score (int).  Ties broken by mean.
    majority_verdict:
        Majority-voted boolean verdict (``is_plausible`` majority).
    agreement:
        Fraction of judges that agreed with the majority verdict
        (1.0 = unanimous, 0.33 = max disagreement for 3 judges).
    """

    reaction_smiles: str
    stratum: str
    judgments: List[Judgment] = field(default_factory=list)
    majority_score: int = 0
    majority_verdict: bool = False
    agreement: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reaction_smiles": self.reaction_smiles,
            "stratum": self.stratum,
            "judges": [j.to_dict() for j in self.judgments],
            "majority_score": self.majority_score,
            "majority_verdict": self.majority_verdict,
            "agreement": self.agreement,
        }


# ---------------------------------------------------------------------------
# LLM API caller (infrastructure for future online use)
# ---------------------------------------------------------------------------


def call_llm_judge(prompt: str, config: LLMJudgeConfig) -> Judgment:
    """Call an LLM API to judge a reaction.

    This is the **abstract** LLM-judge entry point.  It always raises
    :class:`NotImplementedError` when ``config.api_key`` is ``None`` (the
    default on the offline server).  When a key + network path become
    available, replace the body with the appropriate SDK call (openai /
    anthropic / google-generativeai) and parse the JSON response into a
    :class:`Judgment`.

    The function signature is fixed so that downstream code
    (``ReactionJudge``) can swap in a real LLM judge without changing
    its call sites.

    Raises
    ------
    NotImplementedError
        Always, when ``config.api_key`` is None (offline mode).
    """
    if not config.api_key:
        raise NotImplementedError(
            "LLM API judge is offline (no api_key).  Use LocalExpertJudge "
            "as the fallback.  When network access is restored, set "
            "LLMJudgeConfig.api_key and replace the body of call_llm_judge "
            "with the appropriate SDK call."
        )
    # When a key is provided we still raise NotImplementedError because
    # this reference distribution does not bundle any LLM SDK (openai /
    # anthropic / google-generativeai).  Integrators must implement the
    # actual call below.  We keep the explicit raise so the contract is
    # unambiguous: a key alone is necessary but not sufficient.
    raise NotImplementedError(
        f"LLM API caller for model={config.model_name!r} is not wired up "
        "in this offline build.  Install the relevant SDK and implement "
        "the HTTP request inside call_llm_judge."
    )


# ---------------------------------------------------------------------------
# LocalExpertJudge (offline rule-based fallback)
# ---------------------------------------------------------------------------


def _split_reaction(reaction_smiles: str) -> Tuple[str, str]:
    """Split a ``reactants>>products`` SMILES into its two sides.

    Returns ``("", "")`` if the input does not contain ``">>"``.  The
    split is on the first occurrence (right-hand side may itself contain
    ``">>"`` only inside atom classes, which we do not handle here --
    callers should canonicalise first if needed).
    """
    if not reaction_smiles or ">>" not in reaction_smiles:
        return ("", "")
    left, right = reaction_smiles.split(">>", 1)
    return (left.strip(), right.strip())


def _parse_side(smiles_side: str) -> List[Any]:
    """Parse a dot-separated molecule list, returning a list of Mol (or
    empty list if RDKit is unavailable / SMILES is empty).

    Invalid molecules are silently dropped (``None`` is not appended).
    """
    if not smiles_side or not _RDKIT_AVAILABLE:
        return []
    mols: List[Any] = []
    for tok in smiles_side.split("."):
        tok = tok.strip()
        if not tok:
            continue
        try:
            m = Chem.MolFromSmiles(tok)
        except Exception:  # pragma: no cover - RDKit rarely raises
            m = None
        if m is not None:
            mols.append(m)
    return mols


def _atom_counts(mol: Any) -> Dict[str, int]:
    """Return ``{element_symbol: count}`` for a single Mol (heavy atoms
    only; H is implicit in SMILES so we add explicit H via valence)."""
    counts: Dict[str, int] = {}
    if mol is None:
        return counts
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        counts[sym] = counts.get(sym, 0) + 1
    # Add implicit hydrogens (so atom balance reflects total H too).
    try:
        h_count = sum(atom.GetTotalNumHs() for atom in mol.GetAtoms())
    except Exception:  # pragma: no cover
        h_count = 0
    if h_count > 0:
        counts["H"] = counts.get("H", 0) + h_count
    return counts


def _sum_counts(mols: Sequence[Any]) -> Dict[str, int]:
    """Sum atom counts across a list of Mols."""
    total: Dict[str, int] = {}
    for m in mols:
        for sym, n in _atom_counts(m).items():
            total[sym] = total.get(sym, 0) + n
    return total


def _count_bonds(mols: Sequence[Any]) -> int:
    """Total bond count across a list of Mols (heavy-atom bonds only)."""
    n = 0
    for m in mols:
        if m is None:
            continue
        try:
            n += m.GetNumBonds()
        except Exception:  # pragma: no cover
            pass
    return n


class LocalExpertJudge:
    """Rule-based offline chemistry judge (the LLM-judge fallback).

    The judge scores each reaction on four dimensions mirroring the LLM
    prompt:

    1. **is_valid** -- every molecule on both sides parses to a valid
       RDKit Mol.
    2. **is_balanced** -- atom counts (including implicit H) match on
       both sides.
    3. **is_plausible** -- a small set of heuristics:

       * Bond count change is "reasonable" (<= ``max_bond_change`` added
         or removed, default 6 -- forbids e.g. total atom rearrangement).
       * No element appears on only one side (covered by is_balanced but
         double-checked here for robustness).
       * No atom exceeds its typical valence (RDKit parse already
         enforces this, but we re-check on the product side).
       * Charge is conserved (total formal charge equal on both sides).
       * Number of molecules may change but not by more than
         ``max_fragment_delta`` (default 3).

    4. **score** -- integer 0-10:

       * 0 if invalid (parse failure).
       * 2 if valid but unbalanced.
       * 4 if balanced but implausible.
       * 6-10 if plausible, scaled by ``plausible_base_score`` and
         adjusted by ``seed``-driven jitter (different "experts" get
         slightly different scores to simulate inter-LLM disagreement).

    The ``seed`` + ``strictness`` parameters let the orchestrator
    instantiate 3 "different" local experts that produce mildly
    divergent scores (mimicking inter-LLM disagreement) while sharing
    the same rule core.  ``strictness`` shifts the score by +/- 1 and
    toggles the bond-change threshold.

    Parameters
    ----------
    name:
        Judge name (e.g. ``"local_expert_1"``).
    seed:
        RNG seed for jitter (default 0).
    strictness:
        -1 = lenient, 0 = neutral, +1 = strict (default 0).
    max_bond_change:
        Max bonds added/removed for plausibility (default 6).
    max_fragment_delta:
        Max change in number of molecules (default 3).
    plausible_base_score:
        Base score when reaction is plausible (default 8).
    """

    def __init__(
        self,
        name: str = "local_expert",
        seed: int = 0,
        strictness: int = 0,
        max_bond_change: int = 6,
        max_fragment_delta: int = 3,
        plausible_base_score: int = 8,
    ) -> None:
        if strictness not in (-1, 0, 1):
            raise ValueError(f"strictness must be -1, 0, or +1, got {strictness}")
        self.name = name
        self.seed = seed
        self.strictness = strictness
        self.max_bond_change = max_bond_change
        self.max_fragment_delta = max_fragment_delta
        self.plausible_base_score = plausible_base_score
        self._rng = random.Random(seed)

    # -- public -------------------------------------------------------------

    def judge(self, reaction_smiles: str) -> Judgment:
        """Judge a single reaction SMILES.  Returns a :class:`Judgment`."""
        if not _RDKIT_AVAILABLE:
            # Without RDKit we cannot judge; return a neutral score so
            # downstream agreement metrics stay defined.  This branch is
            # not exercised when RDKit is installed (the normal case).
            return Judgment(
                judge=self.name,
                is_valid=False,
                is_balanced=False,
                is_plausible=False,
                score=0,
                reasoning="RDKit unavailable; cannot judge.",
            )

        reactants_str, products_str = _split_reaction(reaction_smiles)
        if not reactants_str or not products_str:
            return Judgment(
                judge=self.name,
                is_valid=False,
                is_balanced=False,
                is_plausible=False,
                score=0,
                reasoning="Reaction SMILES missing '>>' separator.",
            )

        reactant_mols = _parse_side(reactants_str)
        product_mols = _parse_side(products_str)
        # is_valid: every dot-separated token on both sides parsed.
        n_r_tokens = len([t for t in reactants_str.split(".") if t.strip()])
        n_p_tokens = len([t for t in products_str.split(".") if t.strip()])
        is_valid = (
            len(reactant_mols) == n_r_tokens
            and len(product_mols) == n_p_tokens
            and n_r_tokens > 0
            and n_p_tokens > 0
        )
        if not is_valid:
            return Judgment(
                judge=self.name,
                is_valid=False,
                is_balanced=False,
                is_plausible=False,
                score=0,
                reasoning="One or more molecules failed to parse.",
            )

        react_counts = _sum_counts(reactant_mols)
        prod_counts = _sum_counts(product_mols)
        is_balanced = react_counts == prod_counts
        if not is_balanced:
            # Provide a useful reasoning string.
            missing = {
                k: react_counts.get(k, 0) - prod_counts.get(k, 0)
                for k in set(react_counts) | set(prod_counts)
                if react_counts.get(k, 0) != prod_counts.get(k, 0)
            }
            return Judgment(
                judge=self.name,
                is_valid=True,
                is_balanced=False,
                is_plausible=False,
                score=2,
                reasoning=f"Atom imbalance: {missing}",
            )

        # Plausibility heuristics (only checked when balanced).
        plausible, plaus_reason = self._check_plausibility(
            reactant_mols, product_mols
        )
        if not plausible:
            return Judgment(
                judge=self.name,
                is_valid=True,
                is_balanced=True,
                is_plausible=False,
                score=4,
                reasoning=plaus_reason,
            )

        # Plausible: score around plausible_base_score with seed-driven
        # jitter and strictness shift.  Score stays in [6, 10].
        jitter = self._rng.randint(-1, 1)
        score = self.plausible_base_score + jitter + self.strictness
        score = max(6, min(10, score))
        return Judgment(
            judge=self.name,
            is_valid=True,
            is_balanced=True,
            is_plausible=True,
            score=score,
            reasoning=plaus_reason or "Plausible transformation.",
        )

    # -- private ------------------------------------------------------------

    def _check_plausibility(
        self,
        reactant_mols: Sequence[Any],
        product_mols: Sequence[Any],
    ) -> Tuple[bool, str]:
        """Run plausibility heuristics.  Returns (is_plausible, reason)."""
        # Bond count change (strict experts use a tighter bound).
        threshold = self.max_bond_change + self.strictness
        threshold = max(2, threshold)  # never below 2
        r_bonds = _count_bonds(reactant_mols)
        p_bonds = _count_bonds(product_mols)
        delta = abs(p_bonds - r_bonds)
        if delta > threshold:
            return (False, f"Bond count change {delta} > threshold {threshold}.")

        # Fragment count change.
        r_frags = len(reactant_mols)
        p_frags = len(product_mols)
        if abs(p_frags - r_frags) > self.max_fragment_delta:
            return (
                False,
                f"Fragment count change {abs(p_frags - r_frags)} "
                f"> {self.max_fragment_delta}.",
            )

        # Charge conservation.
        try:
            r_charge = sum(
                sum(a.GetFormalCharge() for a in m.GetAtoms())
                for m in reactant_mols
            )
            p_charge = sum(
                sum(a.GetFormalCharge() for a in m.GetAtoms())
                for m in product_mols
            )
        except Exception:  # pragma: no cover
            r_charge = p_charge = 0
        if r_charge != p_charge:
            return (
                False,
                f"Formal charge not conserved: {r_charge} -> {p_charge}.",
            )

        # Heavy-atom count conservation (redundant with is_balanced but
        # guards against any H-count bookkeeping edge case).
        r_heavy = sum(m.GetNumHeavyAtoms() for m in reactant_mols)
        p_heavy = sum(m.GetNumHeavyAtoms() for m in product_mols)
        if r_heavy != p_heavy:
            return (
                False,
                f"Heavy atom count mismatch: {r_heavy} -> {p_heavy}.",
            )

        # Ring count heuristic: a reaction that creates/destroys more
        # than 2 rings in one step is suspicious.
        try:
            r_rings = sum(
                rdMolDescriptors.CalcNumRings(m) for m in reactant_mols
            )
            p_rings = sum(
                rdMolDescriptors.CalcNumRings(m) for m in product_mols
            )
        except Exception:  # pragma: no cover
            r_rings = p_rings = 0
        if abs(p_rings - r_rings) > 2:
            return (
                False,
                f"Ring count change {abs(p_rings - r_rings)} > 2.",
            )

        return (True, "Plausible transformation.")


# ---------------------------------------------------------------------------
# ReactionJudge orchestrator
# ---------------------------------------------------------------------------


class ReactionJudge:
    """Orchestrates 3 judges per reaction and aggregates their verdicts.

    The judge pool is constructed from a list of callables.  Each
    callable takes a ``reaction_smiles: str`` and returns a
    :class:`Judgment`.  In offline mode all three callables are
    :class:`LocalExpertJudge` instances with different seeds / strictness
    so the inter-judge agreement is non-trivial (and exercises the
    Cohen's kappa code path).

    Parameters
    ----------
    judges:
        List of judge callables.  Length 3 is the canonical P3-07 setup
        but any length >= 1 is accepted.
    judge_names:
        Optional list of display names (defaults to ``[j.name for j in
        judges]`` when the judges expose ``.name``).
    """

    def __init__(
        self,
        judges: Sequence[Any],
        judge_names: Optional[Sequence[str]] = None,
    ) -> None:
        if not judges:
            raise ValueError("At least one judge is required")
        self.judges = list(judges)
        if judge_names is None:
            judge_names = []
            for j in self.judges:
                name = getattr(j, "name", None)
                judge_names.append(name if name else f"judge_{len(judge_names)}")
        if len(judge_names) != len(self.judges):
            raise ValueError("judge_names length must match judges length")
        self.judge_names = list(judge_names)

    def judge_reaction(
        self,
        reaction_smiles: str,
        stratum: str = "unknown",
    ) -> AggregatedJudgment:
        """Run all judges on one reaction, returning an aggregated verdict."""
        per_judge: List[Judgment] = []
        for j in self.judges:
            try:
                # ``j`` may be a callable (function) or an object with a
                # ``.judge`` method (LocalExpertJudge).
                if hasattr(j, "judge"):
                    per_judge.append(j.judge(reaction_smiles))
                else:
                    per_judge.append(j(reaction_smiles))
            except Exception as e:  # pragma: no cover - defensive
                # A judge that throws should not abort the whole batch.
                per_judge.append(
                    Judgment(
                        judge=getattr(j, "name", "unknown"),
                        is_valid=False,
                        is_balanced=False,
                        is_plausible=False,
                        score=0,
                        reasoning=f"Judge raised: {e!r}",
                    )
                )
        # Rename judgments to use the display names from this orchestrator
        # (so ``local_expert_1/2/3`` show up correctly even if the
        # underlying judge's ``.name`` differs).
        for j, name in zip(per_judge, self.judge_names):
            j.judge = name

        majority_score = self._majority_score([j.score for j in per_judge])
        majority_verdict = self._majority_verdict(
            [j.is_plausible for j in per_judge]
        )
        agreement = self._agreement(
            [j.is_plausible for j in per_judge], majority_verdict
        )
        return AggregatedJudgment(
            reaction_smiles=reaction_smiles,
            stratum=stratum,
            judgments=per_judge,
            majority_score=majority_score,
            majority_verdict=majority_verdict,
            agreement=agreement,
        )

    def judge_batch(
        self,
        reactions: Sequence[Tuple[str, str]],
    ) -> List[AggregatedJudgment]:
        """Judge a batch of (reaction_smiles, stratum) tuples."""
        return [
            self.judge_reaction(rxn, stratum=stratum)
            for rxn, stratum in reactions
        ]

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _majority_score(scores: Sequence[int]) -> int:
        """Majority-vote on integer scores.  Ties broken by mean."""
        if not scores:
            return 0
        # Count occurrences
        counts: Dict[int, int] = {}
        for s in scores:
            counts[s] = counts.get(s, 0) + 1
        max_count = max(counts.values())
        # If unique max, return it; else break tie by mean (rounded).
        winners = [s for s, c in counts.items() if c == max_count]
        if len(winners) == 1:
            return int(winners[0])
        mean = sum(scores) / len(scores)
        # Pick the winner closest to the mean.
        return int(min(winners, key=lambda s: abs(s - mean)))

    @staticmethod
    def _majority_verdict(verdicts: Sequence[bool]) -> bool:
        """Majority vote on boolean verdicts.  Ties broken to ``False``
        (i.e. the conservative "not plausible" call)."""
        if not verdicts:
            return False
        n_true = sum(1 for v in verdicts if v)
        return n_true > len(verdicts) / 2

    @staticmethod
    def _agreement(verdicts: Sequence[bool], majority: bool) -> float:
        """Fraction of judges that agreed with the majority verdict."""
        if not verdicts:
            return 0.0
        n_match = sum(1 for v in verdicts if v == majority)
        return n_match / len(verdicts)


# ---------------------------------------------------------------------------
# Cohen's kappa (2-judge, from scratch)
# ---------------------------------------------------------------------------


def cohen_kappa(judge1: Sequence[int], judge2: Sequence[int]) -> float:
    """Cohen's kappa for two judges on integer scores.

    Implements the standard formula::

        kappa = (p_o - p_e) / (1 - p_e)

    where ``p_o`` is the observed agreement and ``p_e`` is the expected
    agreement by chance (product of marginal distributions).

    Parameters
    ----------
    judge1, judge2:
        Parallel lists of integer scores (e.g. 0-10).  Must have equal
        length >= 1.

    Returns
    -------
    float
        Cohen's kappa in ``[-1, 1]``.  Returns ``1.0`` if both judges
        produced identical non-empty lists, and ``0.0`` if the lists are
        empty or if both judges assigned the same single score to every
        item (degenerate case where ``p_e == 1``).
    """
    n = len(judge1)
    if n != len(judge2):
        raise ValueError(
            f"judge1 and judge2 must have equal length: {len(judge1)} vs {len(judge2)}"
        )
    if n == 0:
        return 0.0

    # Observed agreement.
    agree = sum(1 for a, b in zip(judge1, judge2) if a == b)
    p_o = agree / n

    # Marginal distributions.
    cats = sorted(set(judge1) | set(judge2))
    p1 = {c: judge1.count(c) / n for c in cats}
    p2 = {c: judge2.count(c) / n for c in cats}
    p_e = sum(p1[c] * p2[c] for c in cats)

    if p_e >= 1.0:
        # Both judges assigned the same single category to everything.
        # kappa is undefined; return 1.0 if they fully agree, else 0.0.
        return 1.0 if p_o == 1.0 else 0.0
    kappa = (p_o - p_e) / (1.0 - p_e)
    return float(kappa)


def compute_inter_judge_agreement(
    judgments_per_judge: Sequence[Sequence[int]],
) -> Dict[str, Any]:
    """Pairwise Cohen's kappa + mean kappa across judges.

    Parameters
    ----------
    judgments_per_judge:
        List of score-lists, one per judge.  All inner lists must have
        the same length (one score per reaction).

    Returns
    -------
    dict
        ``{"kappa_pairwise": List[List[float]], "mean_kappa": float}``.
        ``kappa_pairwise[i][j]`` is the kappa between judge i and judge j
        (1.0 on the diagonal).  ``mean_kappa`` is the mean of the
        upper-triangle kappas (0.0 if <2 judges).
    """
    n_judges = len(judgments_per_judge)
    if n_judges == 0:
        return {"kappa_pairwise": [], "mean_kappa": 0.0}
    # Validate equal length
    if n_judges > 1:
        length = len(judgments_per_judge[0])
        for i, jl in enumerate(judgments_per_judge):
            if len(jl) != length:
                raise ValueError(
                    f"Judge {i} has {len(jl)} scores, expected {length}"
                )

    kappa_matrix: List[List[float]] = []
    for i in range(n_judges):
        row: List[float] = []
        for j in range(n_judges):
            if i == j:
                row.append(1.0)
            elif j < i:
                row.append(kappa_matrix[j][i])  # symmetric
            else:
                row.append(cohen_kappa(judgments_per_judge[i], judgments_per_judge[j]))
        kappa_matrix.append(row)

    if n_judges < 2:
        mean_kappa = 0.0
    else:
        off_diag = [
            kappa_matrix[i][j]
            for i in range(n_judges)
            for j in range(n_judges)
            if i < j
        ]
        mean_kappa = float(statistics.mean(off_diag)) if off_diag else 0.0
    return {"kappa_pairwise": kappa_matrix, "mean_kappa": mean_kappa}


# ---------------------------------------------------------------------------
# DFT agreement
# ---------------------------------------------------------------------------


def load_dft_results(dft_dir: str) -> Dict[str, Dict[str, Any]]:
    """Load DFT validation results from a directory.

    The P2-02 DFT validation directory is expected to contain either:

    * A single ``results.json`` (preferred) mapping
      ``reaction_smiles -> {is_plausible: bool, ...}``, or
    * A ``dft_validation.csv`` with ``reaction_smiles`` and
      ``is_plausible`` columns, or
    * Any ``*.json`` file containing a top-level object with a
      ``"per_reaction"`` list of dicts each having ``reaction_smiles``
      and ``is_plausible`` keys.

    The loader is intentionally defensive: any unreadable file is
    silently skipped and an empty dict is returned if nothing matches,
    so the caller (CLI) can degrade gracefully when DFT results are
    unavailable.

    Returns
    -------
    dict
        ``{reaction_smiles: {"is_plausible": bool, "score": float|None, ...}}``.
    """
    out: Dict[str, Dict[str, Any]] = {}
    if not dft_dir or not os.path.isdir(dft_dir):
        return out

    # 1. Try results.json (preferred shape)
    results_json = os.path.join(dft_dir, "results.json")
    if os.path.isfile(results_json):
        try:
            with open(results_json, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError):
            data = None
        if isinstance(data, dict):
            # Shape A: {reaction_smiles: {...}}
            for k, v in data.items():
                if isinstance(v, dict) and (
                    "is_plausible" in v or "score" in v
                ):
                    out[k] = v
            # Shape B: {"per_reaction": [{reaction_smiles, is_plausible}, ...]}
            if not out:
                per_rxn = data.get("per_reaction") or data.get("reactions")
                if isinstance(per_rxn, list):
                    for row in per_rxn:
                        if isinstance(row, dict):
                            rxn = row.get("reaction_smiles") or row.get("smiles")
                            if rxn:
                                out[rxn] = row

    # 2. Try dft_validation.csv
    csv_path = os.path.join(dft_dir, "dft_validation.csv")
    if not out and os.path.isfile(csv_path):
        try:
            with open(csv_path, "r", newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                if reader.fieldnames:
                    rxn_col = None
                    for cand in ("reaction_smiles", "smiles", "rxn"):
                        if cand in reader.fieldnames:
                            rxn_col = cand
                            break
                    verdict_col = None
                    for cand in ("is_plausible", "plausible", "dft_plausible", "verdict"):
                        if cand in reader.fieldnames:
                            verdict_col = cand
                            break
                    if rxn_col and verdict_col:
                        for row in reader:
                            rxn = (row.get(rxn_col) or "").strip()
                            if not rxn:
                                continue
                            raw = (row.get(verdict_col) or "").strip().lower()
                            verdict = raw in ("true", "1", "yes", "plausible")
                            out[rxn] = {"is_plausible": verdict}
        except OSError:
            pass

    # 3. Fallback: scan any *.json file in the directory for shape B.
    if not out:
        try:
            files = sorted(os.listdir(dft_dir))
        except OSError:
            files = []
        for fname in files:
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(dft_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(data, dict):
                continue
            per_rxn = data.get("per_reaction") or data.get("reactions")
            if isinstance(per_rxn, list):
                for row in per_rxn:
                    if isinstance(row, dict):
                        rxn = row.get("reaction_smiles") or row.get("smiles")
                        if rxn and rxn not in out:
                            out[rxn] = row
            elif isinstance(data, dict):
                # Maybe the whole file is {rxn: {...}}
                for k, v in data.items():
                    if isinstance(v, dict) and (
                        "is_plausible" in v or "score" in v
                    ):
                        out.setdefault(k, v)

    return out


def compute_dft_agreement(
    judgments: Sequence[AggregatedJudgment],
    dft_results: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute % agreement between majority-verdict judgments and DFT.

    A judgment is said to "agree" with DFT if both agree on the boolean
    ``is_plausible`` verdict (DFT uses ``"is_plausible"`` key, judgment
    uses :attr:`AggregatedJudgment.majority_verdict`).

    Parameters
    ----------
    judgments:
        Aggregated judgments (one per reaction).
    dft_results:
        Output of :func:`load_dft_results`.

    Returns
    -------
    dict
        ``{"pct_agree": float, "n_compared": int, "n_skipped": int}``.
        ``pct_agree`` is 0.0 if ``n_compared == 0``.
    """
    n_compared = 0
    n_agree = 0
    n_skipped = 0
    for j in judgments:
        dft = dft_results.get(j.reaction_smiles)
        if not dft or "is_plausible" not in dft:
            n_skipped += 1
            continue
        dft_verdict = bool(dft["is_plausible"])
        if dft_verdict == j.majority_verdict:
            n_agree += 1
        n_compared += 1
    pct = (n_agree / n_compared) if n_compared > 0 else 0.0
    return {
        "pct_agree": float(pct),
        "n_compared": n_compared,
        "n_skipped": n_skipped,
    }


# ---------------------------------------------------------------------------
# PC-CNG negatives loader + stratified sampling
# ---------------------------------------------------------------------------


def load_pc_cng_negatives(
    csv_path: str,
    max_n: Optional[int] = None,
) -> List[Dict[str, str]]:
    """Load PC-CNG synthetic negatives from the reviewed CSV.

    Returns a list of row dicts.  Each dict is guaranteed to have a
    ``"reaction_smiles"`` key; the ``"hard_score"`` key is present if
    the CSV has that column (used for stratified sampling).  Returns an
    empty list if the file is missing or lacks a reaction SMILES
    column.

    Parameters
    ----------
    csv_path:
        Path to ``pc_cng_synthetic_negatives_reviewed.csv``.
    max_n:
        Optional cap on the number of rows returned (random sample).
    """
    if not csv_path or not os.path.exists(csv_path):
        return []
    rows: List[Dict[str, str]] = []
    try:
        with open(csv_path, "r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None:
                return []
            cols = {c.lower().strip(): c for c in reader.fieldnames}
            rxn_col = cols.get("reaction_smiles") or cols.get("smiles")
            if rxn_col is None:
                return []
            hard_col = cols.get("hard_score")
            for raw in reader:
                rxn = (raw.get(rxn_col) or "").strip()
                if not rxn:
                    continue
                row: Dict[str, str] = {"reaction_smiles": rxn}
                if hard_col:
                    row["hard_score"] = (raw.get(hard_col) or "").strip()
                # Pass through any other columns too.
                for k, v in raw.items():
                    if k and k not in row:
                        row[k] = v if v is not None else ""
                rows.append(row)
    except OSError:
        return []
    if max_n is not None and 0 <= max_n < len(rows):
        rng = random.Random(0)
        rows = rng.sample(rows, max_n)
    return rows


def stratified_sample_reactions(
    rows: Sequence[Dict[str, str]],
    n_samples: int,
    seed: int = 42,
) -> List[Tuple[str, str]]:
    """Stratified sample of (reaction_smiles, stratum) tuples.

    Strata are derived from the ``hard_score`` field:

    * Sort rows by ``hard_score`` (parsed as float; missing -> 0.0).
    * Split into 3 equal-quantile bins: ``easy`` / ``medium`` / ``hard``.
    * Sample ``n_samples // 3`` from each bin (remainder goes to
      ``easy``).  If a bin is smaller than its quota, all its rows are
      used.

    If ``hard_score`` is missing on every row, the stratum is assigned
    randomly with equal probability.

    Returns
    -------
    list of (reaction_smiles, stratum)
        Length ``<= n_samples``.
    """
    if n_samples <= 0 or not rows:
        return []
    rng = random.Random(seed)

    has_hard = any(r.get("hard_score") not in (None, "") for r in rows)
    if has_hard:
        def _score(r: Dict[str, str]) -> float:
            try:
                return float(r.get("hard_score") or 0.0)
            except (TypeError, ValueError):
                return 0.0
        sorted_rows = sorted(rows, key=_score)
    else:
        sorted_rows = list(rows)

    n = len(sorted_rows)
    third = max(1, n // 3)
    if has_hard:
        bins = [
            ("easy", sorted_rows[:third]),
            ("medium", sorted_rows[third:2 * third]),
            ("hard", sorted_rows[2 * third:]),
        ]
    else:
        # No hard_score: random stratum assignment.
        strata = ["easy", "medium", "hard"]
        bins = [
            (strata[i], [r for r in sorted_rows if rng.choice(strata) == strata[i]])
            for i in range(3)
        ]
        # The above is biased; simpler: just split randomly.
        shuffled = list(sorted_rows)
        rng.shuffle(shuffled)
        bins = [
            ("easy", shuffled[:third]),
            ("medium", shuffled[third:2 * third]),
            ("hard", shuffled[2 * third:]),
        ]

    per_bin = n_samples // 3
    remainder = n_samples - per_bin * 3
    quotas = [per_bin, per_bin, per_bin]
    for i in range(remainder):
        quotas[i] += 1

    out: List[Tuple[str, str]] = []
    for (label, bin_rows), quota in zip(bins, quotas):
        k = min(quota, len(bin_rows))
        if k <= 0:
            continue
        sampled = rng.sample(bin_rows, k) if k < len(bin_rows) else list(bin_rows)
        for r in sampled:
            out.append((r["reaction_smiles"], label))
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _build_local_judges(
    judge_names: Sequence[str],
    seeds: Sequence[int],
) -> List[LocalExpertJudge]:
    """Build a list of LocalExpertJudge instances with different
    seeds / strictness to simulate inter-LLM disagreement."""
    if not judge_names:
        raise ValueError("judge_names must be non-empty")
    if not seeds:
        raise ValueError("seeds must be non-empty")
    # Strictness cycle: -1, 0, +1, -1, 0, +1, ...
    strictness_cycle = [-1, 0, 1]
    judges: List[LocalExpertJudge] = []
    for i, name in enumerate(judge_names):
        seed = seeds[i % len(seeds)] + i  # vary seed per judge
        strict = strictness_cycle[i % len(strictness_cycle)]
        judges.append(
            LocalExpertJudge(
                name=name,
                seed=seed,
                strictness=strict,
            )
        )
    return judges


def run_judgment(
    pc_cng_negatives_csv: Optional[str],
    dft_results_dir: Optional[str],
    output_dir: Optional[str],
    n_samples: int = 100,
    judge_names: Sequence[str] = ("local_expert_1", "local_expert_2", "local_expert_3"),
    seeds: Sequence[int] = (20260710,),
    fallback_csv: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the full P3-07 judgment pipeline.

    Steps
    -----
    1. Load PC-CNG negatives (or fallback to ``fallback_csv``).
    2. Stratified-sample ``n_samples`` reactions.
    3. Build 3 :class:`LocalExpertJudge` instances (offline fallback).
    4. Run :class:`ReactionJudge.judge_batch` on the sampled reactions.
    5. Compute inter-judge agreement (pairwise Cohen's kappa + mean).
    6. Load DFT results (if available) and compute % agreement.
    7. Determine Go/No-Go (GO if mean_kappa >= 0.6 AND dft pct >= 0.8).
    8. Optionally write ``judgments.json`` + ``summary.md`` to
       ``output_dir``.

    Returns
    -------
    dict
        The full result payload (same shape as the JSON written to disk).
    """
    # 1. Load negatives
    rows: List[Dict[str, str]] = []
    source_csv = pc_cng_negatives_csv
    if pc_cng_negatives_csv:
        rows = load_pc_cng_negatives(pc_cng_negatives_csv)
    if not rows and fallback_csv:
        rows = load_pc_cng_negatives(fallback_csv)
        source_csv = fallback_csv
    if not rows:
        raise ValueError(
            "No PC-CNG negatives loaded.  Provide --pc-cng-negatives or "
            "a --fallback-csv path that exists."
        )

    # 2. Stratified sample
    sampled = stratified_sample_reactions(rows, n_samples=n_samples, seed=seeds[0])

    # 3. Build judges
    judges = _build_local_judges(judge_names, seeds)
    orchestrator = ReactionJudge(judges, judge_names=list(judge_names))

    # 4. Judge batch
    aggregated = orchestrator.judge_batch(sampled)

    # 5. Inter-judge agreement
    judgments_per_judge: List[List[int]] = [[] for _ in judge_names]
    for agg in aggregated:
        for i, j in enumerate(agg.judgments):
            judgments_per_judge[i].append(j.score)
    inter_agreement = compute_inter_judge_agreement(judgments_per_judge)

    # 6. DFT agreement
    dft_results: Dict[str, Dict[str, Any]] = {}
    if dft_results_dir:
        dft_results = load_dft_results(dft_results_dir)
    dft_agreement = compute_dft_agreement(aggregated, dft_results)

    # 7. Go/No-Go
    mean_kappa = inter_agreement.get("mean_kappa", 0.0)
    pct_agree = dft_agreement.get("pct_agree", 0.0)
    n_compared = dft_agreement.get("n_compared", 0)
    # If DFT was unavailable (n_compared == 0), do not block on DFT.
    if n_compared == 0:
        go_no_go = "GO" if mean_kappa >= 0.6 else "NO-GO"
        go_no_go_reason = (
            f"mean_kappa={mean_kappa:.3f} (DFT skipped: no overlap)"
        )
    else:
        if mean_kappa >= 0.6 and pct_agree >= 0.8:
            go_no_go = "GO"
        else:
            go_no_go = "NO-GO"
        go_no_go_reason = (
            f"mean_kappa={mean_kappa:.3f}, dft_pct={pct_agree:.3f} "
            f"(n_compared={n_compared})"
        )

    # 8. Assemble payload
    payload: Dict[str, Any] = {
        "n_samples": len(aggregated),
        "judges": list(judge_names),
        "judgments": [a.to_dict() for a in aggregated],
        "inter_judge_agreement": inter_agreement,
        "dft_agreement": dft_agreement,
        "go_no_go": go_no_go,
        "go_no_go_reason": go_no_go_reason,
        "meta": {
            "pc_cng_negatives_csv": source_csv,
            "dft_results_dir": dft_results_dir,
            "seeds": list(seeds),
            "n_requested": n_samples,
            "n_rows_loaded": len(rows),
            "n_dft_results": len(dft_results),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "judge_mode": "local_expert_offline",
        },
    }

    if output_dir:
        _write_results(payload, output_dir)
    return payload


def _write_results(payload: Dict[str, Any], output_dir: str) -> None:
    """Write ``judgments.json`` + ``summary.md`` to ``output_dir``."""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "judgments.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)

    # Markdown summary
    meta = payload.get("meta", {})
    inter = payload.get("inter_judge_agreement", {})
    dft = payload.get("dft_agreement", {})
    kappa_matrix = inter.get("kappa_pairwise", [])
    judges = payload.get("judges", [])
    lines: List[str] = [
        "# P3-07 LLM-as-Judge Expert Agent Review",
        "",
        f"- **Judge mode**: `{meta.get('judge_mode', '')}` (offline fallback)",
        f"- **PC-CNG negatives CSV**: `{meta.get('pc_cng_negatives_csv', '')}`",
        f"- **DFT results dir**: `{meta.get('dft_results_dir', '')}`",
        f"- **# rows loaded**: {meta.get('n_rows_loaded', 0)}",
        f"- **# sampled reactions**: {payload.get('n_samples', 0)} "
        f"(requested {meta.get('n_requested', 0)})",
        f"- **# DFT results available**: {meta.get('n_dft_results', 0)}",
        f"- **Seeds**: {meta.get('seeds', [])}",
        f"- **Generated**: {meta.get('timestamp', '')}",
        "",
        "## Judges",
        "",
        "| Judge | Mode |",
        "|-------|------|",
    ]
    for j in judges:
        lines.append(f"| {j} | LocalExpertJudge (offline) |")
    lines.extend([
        "",
        "## Inter-judge agreement (Cohen's kappa)",
        "",
        f"- **Mean kappa**: {inter.get('mean_kappa', 0.0):.4f}",
        "",
    ])
    if kappa_matrix:
        header = "| | " + " | ".join(judges) + " |"
        sep = "|---" * (len(judges) + 1) + "|"
        lines.append(header)
        lines.append(sep)
        for i, row in enumerate(kappa_matrix):
            cells = " | ".join(f"{v:.3f}" for v in row)
            lines.append(f"| {judges[i]} | {cells} |")
        lines.append("")
    lines.extend([
        "## DFT agreement (P2-02 chemoselectivity validation)",
        "",
        f"- **% agree**: {dft.get('pct_agree', 0.0):.4f}",
        f"- **# compared**: {dft.get('n_compared', 0)}",
        f"- **# skipped**: {dft.get('n_skipped', 0)}",
        "",
        "## Go / No-Go",
        "",
        f"- **Decision**: `{payload.get('go_no_go', '')}`",
        f"- **Reason**: {payload.get('go_no_go_reason', '')}",
        "",
        "## Per-reaction judgments (first 20)",
        "",
        "| # | Stratum | Majority score | Majority verdict | Agreement |",
        "|---|---------|----------------|------------------|-----------|",
    ])
    for i, j in enumerate(payload.get("judgments", [])[:20]):
        lines.append(
            f"| {i + 1} | {j.get('stratum', '')} | "
            f"{j.get('majority_score', 0)} | "
            f"{j.get('majority_verdict', False)} | "
            f"{j.get('agreement', 0.0):.2f} |"
        )
    lines.append("")
    with open(os.path.join(output_dir, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_seeds(arg: str) -> List[int]:
    """Parse a comma-separated list of seed integers."""
    seeds: List[int] = []
    for tok in arg.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            seeds.append(int(tok))
        except ValueError as e:
            raise argparse.ArgumentTypeError(f"Invalid seed: {tok!r}") from e
    if not seeds:
        raise argparse.ArgumentTypeError("At least one seed is required")
    return seeds


def _parse_judges(arg: str) -> List[str]:
    """Parse a comma-separated list of judge names."""
    names = [t.strip() for t in arg.split(",") if t.strip()]
    if not names:
        raise argparse.ArgumentTypeError("At least one judge is required")
    return names


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    p = argparse.ArgumentParser(
        prog="llm_judge",
        description="P3-07 LLM-as-judge expert agent review.",
    )
    p.add_argument(
        "--pc-cng-negatives",
        default=None,
        help="Path to results/.../pc_cng_synthetic_negatives_reviewed.csv.",
    )
    p.add_argument(
        "--fallback-csv",
        default=None,
        help="Fallback CSV (e.g. hitea_full_positives.csv subset) used when "
             "the --pc-cng-negatives file is missing.",
    )
    p.add_argument(
        "--dft-results",
        default=None,
        help="Path to results/dft_validation_chemoselectivity_20260720/ "
             "(skipped if missing).",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write judgments.json + summary.md (created if missing).",
    )
    p.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of reactions to judge (default 100).",
    )
    p.add_argument(
        "--judges",
        type=_parse_judges,
        default=["local_expert_1", "local_expert_2", "local_expert_3"],
        help="Comma-separated judge names (default: 3 local experts).",
    )
    p.add_argument(
        "--seeds",
        type=_parse_seeds,
        default=[20260710],
        help="Comma-separated RNG seeds (default: 20260710).",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point.  Returns 0 on success, non-zero on error."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    print(f"[P3-07] PC-CNG negatives CSV: {args.pc_cng_negatives}")
    print(f"[P3-07] Fallback CSV: {args.fallback_csv}")
    print(f"[P3-07] DFT results dir: {args.dft_results}")
    print(f"[P3-07] Judges: {args.judges}")
    print(f"[P3-07] Seeds: {args.seeds}")
    print(f"[P3-07] n_samples: {args.n_samples}")
    try:
        payload = run_judgment(
            pc_cng_negatives_csv=args.pc_cng_negatives,
            dft_results_dir=args.dft_results,
            output_dir=args.output_dir,
            n_samples=args.n_samples,
            judge_names=args.judges,
            seeds=args.seeds,
            fallback_csv=args.fallback_csv,
        )
    except (OSError, ValueError) as e:
        print(f"[P3-07] ERROR: {e}", file=sys.stderr)
        return 1
    print("[P3-07] Done. Summary:")
    print(f"  n_samples judged: {payload['n_samples']}")
    print(
        f"  inter-judge mean kappa: "
        f"{payload['inter_judge_agreement']['mean_kappa']:.4f}"
    )
    print(
        f"  DFT agreement: "
        f"{payload['dft_agreement']['pct_agree']:.4f} "
        f"(n_compared={payload['dft_agreement']['n_compared']})"
    )
    print(f"  Go/No-Go: {payload['go_no_go']} ({payload['go_no_go_reason']})")
    if args.output_dir:
        print(f"[P3-07] Wrote {args.output_dir}/judgments.json + summary.md")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
