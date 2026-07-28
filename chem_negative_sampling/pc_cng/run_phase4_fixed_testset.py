"""Phase 4: fixed difficulty-controlled test set evaluation.

Motivation
----------
The Phase 3 v2 self-consistent evaluation is semantically confounded: each
arm trains AND tests on its own negatives, so AUPRC conflates "negative
quality" with "task difficulty".  Concretely, the shuffled_parent control
(real product from a different reaction, same reactants) is trivially
detectable through global atom/substructure mismatch (AUPRC 0.95-0.998 even
after the same-reactant fix), while learned_structured negatives are only
detectable through subtle reaction-center chemistry.  Comparing
self-consistent AUPRCs across arms is therefore apples-to-oranges.

Phase 4 resolution (spec: pccng 的分阶段提示词 2.md, Phase 4 lines 644-693):
a FIXED, difficulty-controlled test set shared by all classifiers.

  * Classifiers are TRAINED on each arm's own training negatives
    (rule / random / learned / shuffled), exactly as in Phase 3 v2.
  * All classifiers are EVALUATED on the SAME fixed test records:
    held-out real reactions (positives) paired with difficulty-stratified
    negative pools (easy / semi-hard / hard) drawn from a union of
    generators.  Because the test records are identical across arms, the
    comparison is perfectly paired and measures genuine classifier utility.

Pre-registered hypotheses (frozen before seeing any results):

  H1 (SOTA): on the semi-hard pool, the classifier trained on
      learned_structured negatives achieves significantly higher AUPRC than
      classifiers trained on rule / random / shuffled negatives
      (Holm-adjusted p < 0.05 across scenarios, CI all positive).
  H2 (hard control): the shuffled_parent-trained classifier achieves
      AUPRC ~0.5 on the semi-hard pool (it only learned global-mismatch
      detection, which is absent from boundary negatives).
  H3 (inverted-U utility): the classifier trained on semi-hard negatives
      outperforms classifiers trained on easy and on hard negatives on the
      semi-hard pool (boundary-utility peak).

Difficulty definition (FROZEN, see FROZEN_DIFFICULTY):
  sim = Tanimoto(MorganFP(neg_product, r=2, 2048), MorganFP(true_product))
  easy      : sim <  0.40
  semi_hard : 0.40 <= sim <= 0.75
  hard      : sim >  0.75   (RDKit-valid only)
All candidates must be RDKit-valid; one negative per positive per pool.

Run::

    python3 -m pc_cng.run_phase4_fixed_testset --gpu 0
    python3 -m pc_cng.run_phase4_fixed_testset --gpu 0 --use-gnn
    python3 -m pc_cng.run_phase4_fixed_testset --splits random --max-train 200 --max-test 100
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CNS_ROOT = _REPO_ROOT / "chem_negative_sampling"
if str(_CNS_ROOT) not in sys.path:
    sys.path.insert(0, str(_CNS_ROOT))

os.environ.setdefault("RDKitRDLogger", "0")
try:  # pragma: no cover
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    _RDKIT_OK = True
except Exception:  # pragma: no cover
    Chem = None
    _RDKIT_OK = False

from pc_cng.paired_cluster_inference import (  # noqa: E402
    auprc_metric,
    holm_correction,
    paired_cluster_bootstrap,
)
from pc_cng.chem_utils import atom_count_distance  # noqa: E402
from pc_cng.p4_g8c_learned_structured_proposal import (  # noqa: E402
    _strip_atom_maps,
)
from pc_cng.phase3_enhanced import (  # noqa: E402
    EnhancedMLP,
    morgan_fingerprint,
    reaction_fp_enhanced,
)
from pc_cng.robust_negative_generator import RobustNegativeGenerator  # noqa: E402

try:  # optional GAT classifier
    from pc_cng.reaction_gnn import ReactionAwareClassifier  # noqa: E402
    _HAS_GNN = True
except Exception:  # pragma: no cover
    ReactionAwareClassifier = None
    _HAS_GNN = False

from pc_cng.run_phase3_external_validation import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_NI_CSV,
    DEFAULT_OOD_DIR,
    DEFAULT_PARQUET,
    METHOD_LEARNED,
    METHOD_RANDOM,
    METHOD_RULE,
    MLP_BATCH,
    MLP_EPOCHS,
    MLP_LR,
    NegativeGenerator,
    load_g8c_model,
    load_hitea_split,
)

# ---------------------------------------------------------------------------
# FROZEN difficulty definition (pre-registered; do NOT tune after seeing
# results - spec Phase 4: "在看 test 结果前冻结 difficulty 定义")
# ---------------------------------------------------------------------------
FROZEN_DIFFICULTY = {
    "version": "difficulty_v1_frozen_20260726",
    "similarity_metric": "tanimoto_morgan_radius2_2048bits(neg_product, true_product)",
    "pools": {
        "easy": {"sim_min": 0.0, "sim_max": 0.40, "sim_max_inclusive": False},
        "semi_hard": {"sim_min": 0.40, "sim_max": 0.75, "sim_max_inclusive": True},
        "hard": {"sim_min": 0.75, "sim_max": 1.0, "sim_min_inclusive": False},
    },
    "validity": "rdkit_mol_from_smiles(neg_product) is not None",
    "candidate_count_per_positive_per_pool_per_source": 1,
    "primary_metric": "source_macro_auprc (equal weight per negative source; "
                      "removes pool-composition home-advantage bias)",
    "frozen_note": "Thresholds fixed before any test-set evaluation; "
                   "no post-hoc binning permitted (spec Phase 4).",
}
POOL_NAMES = ("easy", "semi_hard", "hard")
PRIMARY_POOL = "semi_hard"

# Arm names
METHOD_SHUFFLED_PARENT = "shuffled_parent"
DIFF_ARMS = ("diff_easy", "diff_semihard", "diff_hard")
MAIN_ARMS = (METHOD_RULE, METHOD_RANDOM, METHOD_LEARNED, METHOD_SHUFFLED_PARENT)

# Candidates per reaction from each generator (union pool for stratification)
N_RULE_CANDIDATES = 8
N_LEARNED_CANDIDATES = 8
SHUFFLED_OFFSET = 7

# Atom-balance tolerance for fixed-pool eligibility.  Distance-based
# criterion (v4): a candidate is "balanced" when its L1 atom-count distance
# vs the reactants exceeds the TRUE product's own distance by at most
# BALANCE_DIST_SLACK.  Slack=2 admits exactly one atom transmutation
# (one element -1, another +1 -> L1 +2) while foreign products (shuffled /
# random mismatch) differ by far more and stay excluded.  The previous
# ratio-based eps (0.011) silently killed ALL transmutation candidates on
# large multi-component systems (HTE: 2/150 = 0.013 > eps), starving the
# learned arm's semi_hard pool and giving rule_pc_cng a pool-monopoly
# home advantage.
BALANCE_DIST_SLACK = 2

DEFAULT_OUTPUT = _REPO_ROOT / "results" / "phase4_fixed_testset"


# ---------------------------------------------------------------------------
# Difficulty helpers
# ---------------------------------------------------------------------------

def _tanimoto(fp_a: Optional[np.ndarray], fp_b: Optional[np.ndarray]) -> float:
    """Tanimoto similarity between two binary fingerprint arrays."""
    if fp_a is None or fp_b is None:
        return 0.0
    inter = float(np.minimum(fp_a, fp_b).sum())
    union = float(np.maximum(fp_a, fp_b).sum())
    if union <= 0:
        return 0.0
    return inter / union


def _product_of(rxn: str) -> Optional[str]:
    parts = rxn.split(">")
    if len(parts) == 3 and parts[2]:
        return parts[2]
    if len(parts) == 2 and parts[1]:
        return parts[1]
    return None


def _is_valid_mol(smiles: Optional[str]) -> bool:
    if not smiles or Chem is None:
        return False
    try:
        return Chem.MolFromSmiles(smiles) is not None
    except Exception:
        return False


def _pool_of_sim(sim: float) -> Optional[str]:
    """Assign a similarity value to a frozen difficulty pool."""
    if sim < 0.40:
        return "easy"
    if sim <= 0.75:
        return "semi_hard"
    return "hard"


def _row_meta(row) -> Dict[str, Any]:
    return {
        "experimental_group": str(row.get("experimental_group",
                                          row.get("split_key", "default"))),
        "reaction_family": str(row.get("reaction_family", "unknown")),
        "yield_bin": int(row.get("yield_bin", 0)) if "yield_bin" in row else 0,
    }


# ---------------------------------------------------------------------------
# Union candidate generation (for difficulty stratification)
# ---------------------------------------------------------------------------

def generate_union_candidates(
    rows,
    rule_gen: Optional[NegativeGenerator],
    learned_gen: Optional[NegativeGenerator],
    product_pool: List[str],
    seed: int,
    n_rule: int = N_RULE_CANDIDATES,
    n_learned: int = N_LEARNED_CANDIDATES,
    stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, List[Dict]]:
    """Generate a union candidate pool per reaction row.

    Sources (all with the SAME reactants as the positive):
      - rule_pc_cng   : up to ``n_rule`` boundary-edit candidates
      - learned       : up to ``n_learned`` G8-C structured proposals
      - random_real   : 1 real product sampled from the train product pool
      - shuffled_real : 1 real product from row (i+7) within ``rows``

    Returns ``{reaction_smiles: [ {neg_rxn, neg_product, source, sim, valid,
    pool}, ... ] }`` keyed by the positive reaction SMILES.
    """
    rng = random.Random(seed)
    parsed_rows: List[Tuple[str, str]] = []  # (rxn, true_product)
    for _, row in rows.iterrows():
        rxn = row.get("reaction_smiles")
        if not rxn or not isinstance(rxn, str):
            continue
        prod = _product_of(rxn)
        if prod:
            parsed_rows.append((rxn, prod))

    true_fps: Dict[str, Optional[np.ndarray]] = {}
    true_dist: Dict[str, int] = {}
    strip_cache: Dict[str, str] = {}

    def _true_fp(prod: str):
        if prod not in true_fps:
            true_fps[prod] = morgan_fingerprint(prod)
        return true_fps[prod]

    def _norm(smi: str) -> str:
        # Normalise for the atom-count distance: strip atom maps (and the
        # bracket-H bookkeeping that pre-mapped datasets like HiTEA carry)
        # so reactants / true product / candidate are all plain SMILES and
        # the L1 distance reflects composition, not mapping artefacts.
        if smi not in strip_cache:
            strip_cache[smi] = _strip_atom_maps(smi)
        return strip_cache[smi]

    def _true_dist(rxn: str, prod: str) -> int:
        # Keyed by the full reaction: the same product SMILES can arise
        # from different reactant sets, and the distance reference depends
        # on the reactants.
        if rxn not in true_dist:
            true_dist[rxn] = atom_count_distance(
                _norm(rxn.split(">")[0]), _norm(prod))
        return true_dist[rxn]

    def _mk_candidate(neg_rxn: str, source: str, true_prod: str,
                      pos_rxn: str) -> Optional[Dict]:
        neg_prod = _product_of(neg_rxn)
        if not neg_prod or neg_prod == true_prod:
            return None
        if not _is_valid_mol(neg_prod):
            return None
        sim = _tanimoto(morgan_fingerprint(neg_prod), _true_fp(true_prod))
        # Atom-conservation flag (v4, distance-based): a candidate is
        # "balanced" when its L1 atom-count distance vs the reactants
        # exceeds the TRUE product's own distance by at most
        # BALANCE_DIST_SLACK.  Slack=2 admits exactly one atom
        # transmutation (one element -1, another +1 -> L1 +2) while
        # foreign products (shuffled / random mismatch) differ by far
        # more and stay excluded.  Composition-mismatch detection is
        # exactly the shortcut the shuffled_parent control learns, so the
        # fixed pools must exclude non-constructible candidates for H2 to
        # be a valid test (see build_fixed_pools).
        reactants = _norm(pos_rxn.split(">")[0])
        d_neg = atom_count_distance(reactants, _norm(neg_prod))
        balanced = d_neg <= _true_dist(pos_rxn, true_prod) + BALANCE_DIST_SLACK
        return {
            "neg_rxn": neg_rxn,
            "neg_product": neg_prod,
            "source": source,
            "sim": float(sim),
            "atom_dist": int(d_neg),
            "balanced": bool(balanced),
            "valid": True,
            "pool": _pool_of_sim(sim),
        }

    out: Dict[str, List[Dict]] = {}
    n_rows = len(parsed_rows)
    for i, (rxn, true_prod) in enumerate(parsed_rows):
        parts = rxn.split(">")
        if len(parts) != 3:
            continue
        reactants, agents = parts[0], parts[1]
        cands: List[Dict] = []
        seen_products = {true_prod}

        def _try_add(neg_rxn: Optional[str], source: str):
            if not neg_rxn:
                return
            c = _mk_candidate(neg_rxn, source, true_prod, rxn)
            if c is None or c["neg_product"] in seen_products:
                return
            seen_products.add(c["neg_product"])
            cands.append(c)

        # rule candidates (multi)
        if rule_gen is not None:
            try:
                # Enlarge the per-reaction candidate cap for the union pool
                # (default is 4, which starves the balanced semi_hard pool).
                rule_gen._rule_gen.max_candidates_per_reaction = max(
                    n_rule, getattr(rule_gen._rule_gen,
                                    "max_candidates_per_reaction", 4))
                raw = rule_gen._rule_gen.generate_for_reaction(
                    rxn, source_id="phase4", include_failed=False)
                n_rule_added = 0
                for c in (raw or [])[:n_rule]:
                    neg = getattr(c, "candidate_reaction", None)
                    if neg and isinstance(neg, str):
                        _try_add(neg, "rule_pc_cng")
                        n_rule_added += 1
                if stats is not None:
                    key = "rule_ok" if n_rule_added else "rule_empty"
                    stats[key] = stats.get(key, 0) + 1
            except Exception as exc:
                if stats is not None:
                    stats["rule_err"] = stats.get("rule_err", 0) + 1
                    stats.setdefault("rule_err_msg", str(exc)[:160])

        # learned candidates (multi; exhaustive decoder attaches the
        # pre-applied product so no fragile re-application is needed)
        if learned_gen is not None:
            try:
                from pc_cng.p4_g8c_learned_structured_proposal import (
                    generate_structured_proposal_exhaustive,
                    _apply_structured_edit,
                )
                edits = generate_structured_proposal_exhaustive(
                    learned_gen.model, rxn, top_k=n_learned,
                    device=learned_gen.device, use_validity_mask=True,
                    risk_rerank=False,
                    map_unmapped=True,
                    require_atom_balance=True,
                    balance_dist_slack=BALANCE_DIST_SLACK)
                if stats is not None:
                    key = "learned_ok" if edits else "learned_empty"
                    stats[key] = stats.get(key, 0) + 1
                for edit in edits or []:
                    edited = getattr(edit, "applied_product", None)
                    if not edited:
                        edited = _apply_structured_edit(rxn, edit)
                    if edited and isinstance(edited, str):
                        _try_add(f"{reactants}>{agents}>{edited}", "learned_structured")
            except Exception as exc:
                if stats is not None:
                    stats["learned_err"] = stats.get("learned_err", 0) + 1
                    stats.setdefault("learned_err_msg", str(exc)[:160])

        # random real product (from TRAIN product pool - no test leakage)
        if product_pool:
            fake = rng.choice(product_pool)
            attempts = 0
            while fake in seen_products and attempts < 5:
                fake = rng.choice(product_pool)
                attempts += 1
            _try_add(f"{reactants}>{agents}>{fake}", "random_real")

        # shuffled real product (same reactants, product from row i+offset)
        if n_rows >= 4:
            sh_prod = parsed_rows[(i + SHUFFLED_OFFSET) % n_rows][1]
            if sh_prod in seen_products:
                sh_prod = parsed_rows[(i + SHUFFLED_OFFSET + 1) % n_rows][1]
            _try_add(f"{reactants}>{agents}>{sh_prod}", "shuffled_real")

        if cands:
            out[rxn] = cands
    return out


# ---------------------------------------------------------------------------
# Fixed test-set construction
# ---------------------------------------------------------------------------

def build_fixed_pools(
    test_rows,
    cand_by_rxn: Dict[str, List[Dict]],
    seed: int,
) -> Tuple[Dict[str, List[Dict]], Dict[str, Any]]:
    """Build fixed difficulty pools from union candidates.

    Matching across pools (spec Phase 4):
      * validity           : 100% enforced at candidate generation
      * atom conservation  : only stoichiometrically-constructible
                            candidates (``balanced``) enter the pools -
                            removes the composition-mismatch shortcut that
                            inflates the shuffled_parent control (H2)
      * candidate count    : exactly 1 negative per positive per pool
                            per source
      * family             : same positives wherever a candidate exists;
                            family distribution reported per pool
      * similarity         : the stratification variable itself (reported)
      * edit count         : proxied by 1 - sim (monotonic for local edits);
                            reported per pool
      * scorer margin      : reported post-hoc in the audit (G8-C scorer is
                            the downstream model, not available at pool-build)

    Returns ``({pool: records}, audit)`` where records are interleaved
    [pos, neg, pos, neg, ...] with keys: reaction_smiles, negative_smiles,
    label, score, experimental_group, reaction_family, yield_bin, method,
    is_positive, source, sim.
    """
    rng = random.Random(seed + 1)
    row_lookup = {}
    for _, row in test_rows.iterrows():
        rxn = row.get("reaction_smiles")
        if rxn and isinstance(rxn, str):
            row_lookup[rxn] = row

    pools: Dict[str, List[Dict]] = {p: [] for p in POOL_NAMES}
    audit: Dict[str, Any] = {"pools": {}, "frozen_difficulty": FROZEN_DIFFICULTY}

    for pool in POOL_NAMES:
        balance_excluded: Dict[str, int] = defaultdict(int)
        for rxn, cands in cand_by_rxn.items():
            # Atom-conservation eligibility (H2 validity): only
            # stoichiometrically-constructible candidates may enter the
            # fixed test pools.  Unbalanced candidates (foreign products,
            # foreign-atom transmutations) are detectable through global
            # composition mismatch - exactly the shortcut the
            # shuffled_parent control learns - so including them would
            # re-inflate the control's AUPRC and invalidate H2.
            in_pool = [c for c in cands if c["pool"] == pool]
            for c in in_pool:
                if not c["balanced"]:
                    balance_excluded[c["source"]] += 1
            in_pool = [c for c in in_pool if c["balanced"]]
            if not in_pool:
                continue
            row = row_lookup.get(rxn)
            if row is None:
                continue
            # Source-balanced construction: keep up to ONE candidate per
            # (positive, pool, source) instead of a single random choice.
            # The previous rng.choice design let the highest-yield source
            # (rule_pc_cng) dominate the pool, giving the rule-trained
            # classifier a "home advantage" (train distribution == test
            # distribution).  Per-source retention equalises source
            # representation up to availability.
            by_src: Dict[str, Dict] = {}
            for c in in_pool:
                by_src.setdefault(c["source"], c)
            meta = _row_meta(row)
            pools[pool].append({
                "reaction_smiles": rxn, "negative_smiles": rxn,
                "label": 1, "score": 0.0, "method": f"fixed_{pool}",
                "is_positive": True, "source": "real", "sim": 1.0, **meta,
            })
            for src, chosen in sorted(by_src.items()):
                pools[pool].append({
                    "reaction_smiles": chosen["neg_rxn"],
                    "negative_smiles": chosen["neg_product"],
                    "label": 0, "score": 0.0, "method": f"fixed_{pool}",
                    "is_positive": False, "source": src,
                    "sim": chosen["sim"], **meta,
                })

        negs = [r for r in pools[pool] if not r["is_positive"]]
        poss = [r for r in pools[pool] if r["is_positive"]]
        sims = [r["sim"] for r in negs]
        fams = defaultdict(int)
        srcs = defaultdict(int)
        for r in negs:
            fams[r["reaction_family"]] += 1
            srcs[r["source"]] += 1
        audit["pools"][pool] = {
            "n_pos": len(poss),
            "n_neg": len(negs),
            "n_records": len(pools[pool]),
            "sim_mean": float(np.mean(sims)) if sims else None,
            "sim_min": float(min(sims)) if sims else None,
            "sim_max": float(max(sims)) if sims else None,
            "family_distribution": dict(fams),
            "source_composition": dict(srcs),
            "balance_excluded_by_source": dict(balance_excluded),
            "validity_rate": 1.0,
            "candidate_count_per_positive_per_source": 1,
        }
    return pools, audit


# ---------------------------------------------------------------------------
# Source-macro AUPRC (primary metric: equal weight per negative source)
# ---------------------------------------------------------------------------

MIN_PER_SOURCE = 8


def source_macro_auprc_metric(records: List[Dict],
                              min_per_source: int = MIN_PER_SOURCE) -> float:
    """Macro-averaged AUPRC across negative-source slices.

    Positives are shared; negatives are grouped by ``source``; AUPRC is
    computed per slice (all positives + that source's negatives) and
    macro-averaged.  This removes the pool-composition bias: a source that
    yields more candidates does NOT dominate the metric, so no training arm
    gets a "home advantage" from distribution match with the test pool.

    Slices with fewer than ``min_per_source`` negatives are dropped
    (unstable AUPRC); if no slice survives, fall back to pooled AUPRC.
    """
    pos = [r for r in records if r.get("is_positive")]
    neg_by_src: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        if not r.get("is_positive"):
            neg_by_src[str(r.get("source", "unknown"))].append(r)
    vals = [auprc_metric(pos + negs) for negs in neg_by_src.values()
            if len(negs) >= min_per_source]
    if not vals:
        return auprc_metric(records)
    return float(np.mean(vals))


def per_source_auprc(records: List[Dict],
                     min_per_source: int = MIN_PER_SOURCE) -> Dict[str, float]:
    """AUPRC per negative-source slice (audit matrix)."""
    pos = [r for r in records if r.get("is_positive")]
    neg_by_src: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        if not r.get("is_positive"):
            neg_by_src[str(r.get("source", "unknown"))].append(r)
    return {src: float(auprc_metric(pos + negs))
            for src, negs in sorted(neg_by_src.items())
            if len(negs) >= min_per_source}


# ---------------------------------------------------------------------------
# Training-data builders (one per arm)
# ---------------------------------------------------------------------------

def build_main_arm_train(
    arm: str,
    train_rows,
    generators: Dict[str, RobustNegativeGenerator],
    seed: int,
) -> Tuple[Optional[np.ndarray], np.ndarray, List[Dict]]:
    """Build (X, y, records) for a main arm's OWN training negatives.

    Interleaved layout (pos, neg, ...) so records align with X/y.
    """
    fps: List[np.ndarray] = []
    labels: List[int] = []
    records: List[Dict] = []

    if arm == METHOD_SHUFFLED_PARENT:
        # Fully shuffled negatives: both reactants and products come from
        # different reactions, destroying any reactant-product compatibility
        # signal.  This makes shuffled_parent a TRUE null control (AUPRC ~0.5)
        # rather than a compatibility-transfer baseline.
        parsed: List[Tuple[str, str, str, str, dict]] = []
        for _, row in train_rows.iterrows():
            rxn = row.get("reaction_smiles")
            if not rxn or not isinstance(rxn, str):
                continue
            parts = rxn.split(">")
            if len(parts) != 3 or not parts[2]:
                continue
            parsed.append((parts[0], parts[1], parts[2], rxn, _row_meta(row)))
        n = len(parsed)
        for i, (r, a, p, pos_rxn, meta) in enumerate(parsed):
            j = (i + SHUFFLED_OFFSET) % n
            r_shuf = parsed[j][0]
            a_shuf = parsed[j][1]
            p_shuf = parsed[(j + SHUFFLED_OFFSET) % n][2]
            if r_shuf == r and a_shuf == a:
                j2 = (j + 1) % n
                r_shuf = parsed[j2][0]
                a_shuf = parsed[j2][1]
            if p_shuf == p:
                p_shuf = parsed[(j + SHUFFLED_OFFSET + 1) % n][2]
            if r_shuf == r and a_shuf == a:
                continue
            neg_rxn = f"{r_shuf}>{a_shuf}>{p_shuf}"
            pos_fp = reaction_fp_enhanced(pos_rxn)
            neg_fp = reaction_fp_enhanced(neg_rxn)
            if pos_fp is None or neg_fp is None:
                continue
            fps.extend([pos_fp, neg_fp])
            labels.extend([1, 0])
            records.append({"reaction_smiles": pos_rxn, "negative_smiles": pos_rxn,
                            "label": 1, "score": 0.0, "method": arm,
                            "is_positive": True, **meta})
            records.append({"reaction_smiles": neg_rxn, "negative_smiles": p_shuf,
                            "label": 0, "score": 0.0, "method": arm,
                            "is_positive": False, **meta})
    else:
        gen = generators.get(arm)
        if gen is None:
            return None, np.array([]), []
        for _, row in train_rows.iterrows():
            rxn = row.get("reaction_smiles")
            if not rxn or not isinstance(rxn, str):
                continue
            pos_fp = reaction_fp_enhanced(rxn)
            if pos_fp is None:
                continue
            neg_rxn = gen.generate(rxn)
            if neg_rxn is None:
                continue
            neg_fp = reaction_fp_enhanced(neg_rxn)
            if neg_fp is None:
                continue
            meta = _row_meta(row)
            fps.extend([pos_fp, neg_fp])
            labels.extend([1, 0])
            records.append({"reaction_smiles": rxn, "negative_smiles": rxn,
                            "label": 1, "score": 0.0, "method": arm,
                            "is_positive": True, **meta})
            records.append({"reaction_smiles": neg_rxn,
                            "negative_smiles": _product_of(neg_rxn) or neg_rxn,
                            "label": 0, "score": 0.0, "method": arm,
                            "is_positive": False, **meta})

    if not fps:
        return None, np.array([]), []
    X = np.vstack(fps)
    y = np.array(labels, dtype=np.float32)
    assert len(records) == len(y)
    return X, y, records


def build_diff_arm_train(
    arm: str,
    train_rows,
    cand_by_rxn: Dict[str, List[Dict]],
    n_per_arm: int,
    seed: int,
) -> Tuple[Optional[np.ndarray], np.ndarray, List[Dict]]:
    """Build (X, y, records) for a difficulty arm.

    Negatives are sampled from the train-side union candidate pool,
    stratified by the frozen difficulty of the candidate.  All three
    difficulty arms use identical n (budget matched) and the SAME
    classifier architecture (EnhancedMLP) to isolate the difficulty effect.
    """
    pool = {"diff_easy": "easy", "diff_semihard": "semi_hard",
            "diff_hard": "hard"}[arm]
    row_lookup = {}
    for _, row in train_rows.iterrows():
        rxn = row.get("reaction_smiles")
        if rxn and isinstance(rxn, str):
            row_lookup[rxn] = row

    entries: List[Tuple[str, Dict]] = []  # (pos_rxn, candidate)
    for rxn, cands in cand_by_rxn.items():
        for c in cands:
            if c["pool"] == pool:
                entries.append((rxn, c))
    rng = random.Random(seed + 2)
    rng.shuffle(entries)
    entries = entries[:n_per_arm]

    fps: List[np.ndarray] = []
    labels: List[int] = []
    records: List[Dict] = []
    for pos_rxn, cand in entries:
        row = row_lookup.get(pos_rxn)
        if row is None:
            continue
        pos_fp = reaction_fp_enhanced(pos_rxn)
        neg_fp = reaction_fp_enhanced(cand["neg_rxn"])
        if pos_fp is None or neg_fp is None:
            continue
        meta = _row_meta(row)
        fps.extend([pos_fp, neg_fp])
        labels.extend([1, 0])
        records.append({"reaction_smiles": pos_rxn, "negative_smiles": pos_rxn,
                        "label": 1, "score": 0.0, "method": arm,
                        "is_positive": True, **meta})
        records.append({"reaction_smiles": cand["neg_rxn"],
                        "negative_smiles": cand["neg_product"],
                        "label": 0, "score": 0.0, "method": arm,
                        "is_positive": False, "sim": cand["sim"],
                        "source": cand["source"], **meta})
    if not fps:
        return None, np.array([]), []
    X = np.vstack(fps)
    y = np.array(labels, dtype=np.float32)
    return X, y, records


# ---------------------------------------------------------------------------
# Classifier training / scoring
# ---------------------------------------------------------------------------

def train_classifier(X_tr, y_tr, rec_tr, seed: int, use_gnn: bool):
    """Train a classifier; returns the fitted object."""
    if use_gnn and _HAS_GNN:
        clf = ReactionAwareClassifier(input_dim=X_tr.shape[1], seed=seed)
        rxns = [r["reaction_smiles"] for r in rec_tr]
        clf.fit_reactions(rxns, y_tr, epochs=MLP_EPOCHS,
                          batch_size=MLP_BATCH, lr=MLP_LR, verbose=False)
        return clf
    clf = EnhancedMLP(input_dim=X_tr.shape[1], seed=seed)
    clf.train(X_tr, y_tr, epochs=MLP_EPOCHS, batch_size=MLP_BATCH,
              lr=MLP_LR, verbose=False)
    return clf


def score_records(clf, records: List[Dict], use_gnn: bool) -> List[Dict]:
    """Return a COPY of fixed test records with this classifier's scores."""
    out = copy.deepcopy(records)
    if use_gnn and hasattr(clf, "predict_proba_reactions"):
        rxns = [r["reaction_smiles"] for r in out]
        scores = np.asarray(clf.predict_proba_reactions(rxns)).flatten()
        for i, r in enumerate(out):
            if i < len(scores):
                r["score"] = float(scores[i])
        return out
    fps = [reaction_fp_enhanced(r["reaction_smiles"]) for r in out]
    valid = [i for i, f in enumerate(fps) if f is not None]
    if valid:
        X = np.vstack([fps[i] for i in valid])
        scores = clf.predict_proba(X)
        for j, i in enumerate(valid):
            out[i]["score"] = float(scores[j])
    return out


# ---------------------------------------------------------------------------
# Cluster bootstrap of a single metric (for H2: shuffled AUPRC ~ 0.5 CI)
# ---------------------------------------------------------------------------

def cluster_bootstrap_metric(records: List[Dict], metric_fn=auprc_metric,
                             cluster_key: str = "experimental_group",
                             n_bootstrap: int = 1000, confidence: float = 0.95,
                             seed: int = 20260726) -> Dict[str, float]:
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        groups[str(r.get(cluster_key, "default"))].append(r)
    cluster_ids = sorted(groups)
    if not cluster_ids:
        return {"point": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n_clusters": 0}
    rng = random.Random(seed)
    n = len(cluster_ids)
    vals = []
    for _ in range(n_bootstrap):
        sample = []
        for _ in range(n):
            sample.extend(groups[cluster_ids[rng.randrange(n)]])
        vals.append(metric_fn(sample))
    vals.sort()
    alpha = 1.0 - confidence
    return {
        "point": float(metric_fn(records)),
        "ci_low": float(vals[int(alpha / 2 * n_bootstrap)]),
        "ci_high": float(vals[int((1 - alpha / 2) * n_bootstrap)]),
        "n_clusters": n,
    }


# ---------------------------------------------------------------------------
# Scenario runner
# ---------------------------------------------------------------------------

def run_scenario(split_name: str, split_data: Dict, model, device,
                 args, generators_cache: Dict[str, Any]) -> Dict:
    print(f"\n[phase4] === Scenario: {split_name} ===")
    train_rows, test_rows = split_data["train"], split_data["test"]
    print(f"  train={len(train_rows)}  test={len(test_rows)}")

    # Train product pool (for random_real candidates + random arm)
    product_pool: List[str] = []
    for _, row in train_rows.iterrows():
        rxn = row.get("reaction_smiles", "")
        if isinstance(rxn, str):
            prod = _product_of(rxn)
            if prod:
                product_pool.append(prod)

    # Generators (robust-wrapped) for main arms
    generators: Dict[str, RobustNegativeGenerator] = {}
    raw_rule = raw_learned = None
    if METHOD_RULE in generators_cache:
        raw_rule = generators_cache[METHOD_RULE]
        generators[METHOD_RULE] = RobustNegativeGenerator(raw_rule, seed=args.seed)
    if model is not None and METHOD_LEARNED in generators_cache:
        raw_learned = generators_cache[METHOD_LEARNED]
        generators[METHOD_LEARNED] = RobustNegativeGenerator(raw_learned, seed=args.seed)
    if METHOD_RANDOM in generators_cache:
        generators[METHOD_RANDOM] = RobustNegativeGenerator(
            generators_cache[METHOD_RANDOM], seed=args.seed)

    # ----- Union candidate pools (train side for diff arms, test side for
    # ----- fixed evaluation pools)
    t0 = time.time()
    gen_stats: Dict[str, Any] = {}
    cand_train = generate_union_candidates(
        train_rows, raw_rule, raw_learned, product_pool, seed=args.seed,
        stats=gen_stats)
    cand_test = generate_union_candidates(
        test_rows, raw_rule, raw_learned, product_pool, seed=args.seed + 1,
        stats=gen_stats)
    print(f"  [candidates] train={sum(len(v) for v in cand_train.values())} "
          f"test={sum(len(v) for v in cand_test.values())} ({time.time()-t0:.1f}s)")
    if gen_stats:
        print(f"  [gen-stats] {gen_stats}")

    # ----- Fixed test pools
    pools, audit = build_fixed_pools(test_rows, cand_test, seed=args.seed)
    audit["gen_stats"] = gen_stats
    for pool in POOL_NAMES:
        a = audit["pools"][pool]
        print(f"  [pool:{pool}] n_pos={a['n_pos']} sim_mean={a['sim_mean']} "
              f"sources={a['source_composition']} "
              f"bal_excl={a['balance_excluded_by_source']}")
    if audit["pools"][PRIMARY_POOL]["n_pos"] < 20:
        print(f"  [WARN] primary pool '{PRIMARY_POOL}' too small "
              f"({audit['pools'][PRIMARY_POOL]['n_pos']}); scenario degraded")

    # ----- Diff-arm budget matching
    diff_counts = {}
    for arm, pool in (("diff_easy", "easy"), ("diff_semihard", "semi_hard"),
                      ("diff_hard", "hard")):
        diff_counts[arm] = sum(1 for cands in cand_train.values()
                               for c in cands if c["pool"] == pool)
    n_diff = min(min(diff_counts.values()), args.max_train) if min(diff_counts.values()) > 0 else 0
    print(f"  [diff-arms] train candidates: {diff_counts} -> n_per_arm={n_diff}")

    # ----- Train all arms
    scores_by_arm_pool: Dict[str, Dict[str, List[Dict]]] = {}
    arm_meta: Dict[str, Any] = {}

    def _run_arm(arm: str, X_tr, y_tr, rec_tr, use_gnn: bool):
        if X_tr is None or len(X_tr) < 10:
            print(f"  [{arm}] insufficient train data")
            return
        t_a = time.time()
        clf = train_classifier(X_tr, y_tr, rec_tr, args.seed, use_gnn=use_gnn)
        scores_by_arm_pool[arm] = {}
        for pool in POOL_NAMES:
            if not pools[pool]:
                continue
            scored = score_records(clf, pools[pool], use_gnn=use_gnn)
            scores_by_arm_pool[arm][pool] = scored
        arm_meta[arm] = {"n_train": len(X_tr),
                         "train_sec": time.time() - t_a, "gnn": use_gnn}
        primary_auprc = source_macro_auprc_metric(
            scores_by_arm_pool[arm][PRIMARY_POOL]) \
            if PRIMARY_POOL in scores_by_arm_pool[arm] else float("nan")
        print(f"  [{arm}] trained ({len(X_tr)} samples, "
              f"{arm_meta[arm]['train_sec']:.1f}s) "
              f"srcMacroAUPRC[{PRIMARY_POOL}]={primary_auprc:.4f}")

    for arm in MAIN_ARMS:
        if arm == METHOD_LEARNED and model is None:
            continue
        X_tr, y_tr, rec_tr = build_main_arm_train(arm, train_rows, generators,
                                                  args.seed)
        _run_arm(arm, X_tr, y_tr, rec_tr, use_gnn=args.use_gnn)

    for arm in DIFF_ARMS:
        if n_diff < 10:
            print(f"  [{arm}] skipped (n_diff={n_diff} < 10)")
            continue
        X_tr, y_tr, rec_tr = build_diff_arm_train(
            arm, train_rows, cand_train, n_diff, args.seed)
        # Difficulty arms ALWAYS use EnhancedMLP (freeze classifier arch to
        # isolate the difficulty effect - competing-explanation ablation)
        _run_arm(arm, X_tr, y_tr, rec_tr, use_gnn=False)

    # ----- AUPRC matrix (pooled + source-macro + per-source audit)
    auprc_matrix: Dict[str, Dict[str, Any]] = {}
    for arm, by_pool in scores_by_arm_pool.items():
        auprc_matrix[arm] = {}
        for pool, recs in by_pool.items():
            auprc_matrix[arm][pool] = {
                "pooled": float(auprc_metric(recs)),
                "source_macro": float(source_macro_auprc_metric(recs)),
                "per_source": per_source_auprc(recs),
            }

    # ----- Paired bootstrap CIs on the PRIMARY pool (source-macro metric)
    comparisons = [
        (METHOD_LEARNED, METHOD_RULE),
        (METHOD_LEARNED, METHOD_RANDOM),
        (METHOD_LEARNED, METHOD_SHUFFLED_PARENT),
        ("diff_semihard", "diff_easy"),
        ("diff_semihard", "diff_hard"),
    ]
    paired_ci: Dict[str, Any] = {}
    for ch, bl in comparisons:
        if ch not in scores_by_arm_pool or bl not in scores_by_arm_pool:
            continue
        if PRIMARY_POOL not in scores_by_arm_pool[ch] or \
           PRIMARY_POOL not in scores_by_arm_pool[bl]:
            continue
        key = f"{ch}_vs_{bl}"
        try:
            ci = paired_cluster_bootstrap(
                scores_by_arm_pool[ch][PRIMARY_POOL],
                scores_by_arm_pool[bl][PRIMARY_POOL],
                metric_fn=source_macro_auprc_metric,
                cluster_key="experimental_group",
                n_bootstrap=args.n_bootstrap, seed=args.seed)
            paired_ci[key] = ci
            print(f"  [CI:{PRIMARY_POOL}] {key}: delta={ci['delta_mean']:+.4f} "
                  f"CI=[{ci['delta_ci_low']:+.4f}, {ci['delta_ci_high']:+.4f}] "
                  f"p={ci['p_value']:.4f}")
        except Exception as exc:
            paired_ci[key] = {"error": str(exc)}

    # ----- H2: shuffled_parent point + CI on primary pool (source-macro)
    h2_ci = None
    if METHOD_SHUFFLED_PARENT in scores_by_arm_pool and \
       PRIMARY_POOL in scores_by_arm_pool[METHOD_SHUFFLED_PARENT]:
        h2_ci = cluster_bootstrap_metric(
            scores_by_arm_pool[METHOD_SHUFFLED_PARENT][PRIMARY_POOL],
            metric_fn=source_macro_auprc_metric,
            n_bootstrap=args.n_bootstrap, seed=args.seed)
        print(f"  [H2] shuffled_parent srcMacroAUPRC[{PRIMARY_POOL}]="
              f"{h2_ci['point']:.4f} CI=[{h2_ci['ci_low']:.4f}, "
              f"{h2_ci['ci_high']:.4f}]")

    return {
        "split_name": split_name,
        "audit": audit,
        "diff_train_counts": diff_counts,
        "n_diff_per_arm": n_diff,
        "arm_meta": arm_meta,
        "auprc_matrix": auprc_matrix,
        "paired_ci": paired_ci,
        "h2_shuffled_ci": h2_ci,
        "test_records": {arm: by_pool for arm, by_pool in scores_by_arm_pool.items()},
    }


# ---------------------------------------------------------------------------
# Holm correction across scenarios
# ---------------------------------------------------------------------------

def apply_holm(all_results: Dict[str, Dict], alpha: float) -> Dict[str, Any]:
    tests = []
    for scenario, res in all_results.items():
        for key, ci in res.get("paired_ci", {}).items():
            if "error" in ci:
                continue
            tests.append({"scenario": scenario, "pair": key,
                          "p": ci["p_value"], "delta": ci["delta_mean"],
                          "ci_low": ci["delta_ci_low"],
                          "ci_high": ci["delta_ci_high"]})
    if not tests:
        return {"tests": [], "n_significant": 0}
    holm = holm_correction([t["p"] for t in tests], alpha=alpha)
    for t, h in zip(tests, holm):
        t["adjusted_p"] = h["adjusted_p"]
        t["rejected"] = h["rejected"]
    return {"tests": tests,
            "n_significant": sum(1 for t in tests if t["rejected"]),
            "n_total": len(tests), "alpha": alpha}


# ---------------------------------------------------------------------------
# Verdict (pre-registered decision rules)
# ---------------------------------------------------------------------------

def compute_verdict(all_results: Dict[str, Dict], holm_res: Dict[str, Any],
                    scenarios: List[str]) -> Dict[str, Any]:
    rejected = {(t["scenario"], t["pair"]) for t in holm_res.get("tests", [])
                if t["rejected"] and t["delta"] > 0}

    # H1: learned > rule & random & shuffled on semi_hard (Holm, delta>0)
    h1_pairs = [f"{METHOD_LEARNED}_vs_{METHOD_RULE}",
                f"{METHOD_LEARNED}_vs_{METHOD_RANDOM}",
                f"{METHOD_LEARNED}_vs_{METHOD_SHUFFLED_PARENT}"]
    h1_per_scenario = {}
    for sc in scenarios:
        res = all_results.get(sc, {})
        mat = res.get("auprc_matrix", {})
        if METHOD_LEARNED not in mat:
            h1_per_scenario[sc] = {"status": "skipped", "wins": 0, "n": 3}
            continue
        wins = sum(1 for p in h1_pairs if (sc, p) in rejected)
        learned_entry = mat.get(METHOD_LEARNED, {}).get(PRIMARY_POOL, {})
        h1_per_scenario[sc] = {
            "status": "SOTA" if wins == 3 else ("partial" if wins > 0 else "fail"),
            "wins": wins, "n": 3,
            "learned_auprc": learned_entry.get("source_macro")
                             if isinstance(learned_entry, dict) else learned_entry,
        }
    n_sota = sum(1 for v in h1_per_scenario.values()
                 if v.get("status") == "SOTA")

    # H2: shuffled ~ 0.5 on semi_hard
    h2_points = {sc: all_results[sc]["h2_shuffled_ci"]
                 for sc in scenarios
                 if all_results.get(sc, {}).get("h2_shuffled_ci")}
    h2_med = float(np.median([v["point"] for v in h2_points.values()])) \
        if h2_points else None

    # H3: diff_semihard > diff_easy AND > diff_hard
    h3_pairs = ["diff_semihard_vs_diff_easy", "diff_semihard_vs_diff_hard"]
    h3_per_scenario = {}
    for sc in scenarios:
        wins = sum(1 for p in h3_pairs if (sc, p) in rejected)
        h3_per_scenario[sc] = wins
    n_h3 = sum(1 for w in h3_per_scenario.values() if w == 2)

    return {
        "H1_sota": {
            "per_scenario": h1_per_scenario,
            "n_sota_scenarios": n_sota,
            "n_scenarios": len([v for v in h1_per_scenario.values()
                                if v.get("status") != "skipped"]),
            "criterion": "learned beats rule+random+shuffled on semi_hard "
                         "(Holm p<0.05, delta>0) in ALL scenarios",
            "go": n_sota == len([v for v in h1_per_scenario.values()
                                 if v.get("status") != "skipped"]) and n_sota > 0,
        },
        "H2_hard_control": {
            "per_scenario": h2_points,
            "median_auprc": h2_med,
            "criterion": "median shuffled_parent AUPRC on semi_hard in [0.40, 0.70]",
            "achieved": (h2_med is not None and 0.40 <= h2_med <= 0.70),
        },
        "H3_inverted_u": {
            "per_scenario_wins": h3_per_scenario,
            "n_scenarios_both": n_h3,
            "criterion": "diff_semihard beats BOTH diff_easy and diff_hard "
                         "(Holm p<0.05) in a majority of scenarios",
            "supported": n_h3 > len(scenarios) / 2,
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 4: fixed difficulty-controlled test set evaluation.")
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--ni-csv", type=Path, default=DEFAULT_NI_CSV)
    parser.add_argument("--ood-dir", type=Path, default=DEFAULT_OOD_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument("--max-train", type=int, default=500)
    parser.add_argument("--max-test", type=int, default=200)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--no-ni", action="store_true")
    parser.add_argument("--no-uspto", action="store_true")
    parser.add_argument("--uspto-csv", type=Path, default=Path(
        "/home/cunyuliu/pc_cng_research/data/processed/uspto_openmolecules_normalized.csv"))
    parser.add_argument("--use-gnn", action="store_true",
                        help="GAT for main arms (difficulty arms stay MLP)")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args(argv)

    t_start = time.time()
    args.output.mkdir(parents=True, exist_ok=True)
    records_dir = args.output / "per_scenario_records"
    records_dir.mkdir(parents=True, exist_ok=True)

    # GPU audit (hard constraint: all training on GPU)
    import torch
    print(f"[phase4] torch.cuda.is_available() = {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[phase4] device = {torch.cuda.get_device_name(0)}")
    device = None
    if args.gpu is not None and torch.cuda.is_available():
        if args.gpu < torch.cuda.device_count():
            device = torch.device(f"cuda:{args.gpu}")
            free, total = torch.cuda.mem_get_info(args.gpu)
            print(f"[phase4] GPU {args.gpu}: {free/1e9:.1f}/{total/1e9:.1f} GB free")

    print(f"[phase4] output  : {args.output}")
    print(f"[phase4] use_gnn : {args.use_gnn} (available={_HAS_GNN})")
    print(f"[phase4] frozen difficulty: {FROZEN_DIFFICULTY['version']}")
    print(f"[phase4] pools   : easy<0.40 | semi_hard[0.40,0.75] | hard>0.75")

    model, model_err = load_g8c_model(args.checkpoint, device=device)
    if model is not None:
        print("[phase4] G8-C model loaded")
    else:
        print(f"[phase4] G8-C model NOT loaded: {model_err} "
              "(learned arm + learned candidates skipped)")

    # Shared generators (constructed once; rule gen is stateless, learned
    # gen wraps the model, random gen uses per-split product pool)
    generators_cache: Dict[str, Any] = {
        METHOD_RULE: NegativeGenerator(METHOD_RULE, seed=args.seed),
    }
    if model is not None:
        generators_cache[METHOD_LEARNED] = NegativeGenerator(
            METHOD_LEARNED, model=model, top_k=N_LEARNED_CANDIDATES,
            device=device, seed=args.seed)

    # Discover splits
    if args.splits:
        split_names = list(args.splits)
    else:
        split_names = []
        for path in sorted(args.ood_dir.glob("*_metadata.json")):
            name = path.name.replace("_metadata.json", "")
            if name in ("splits_manifest", "patent"):
                continue
            split_names.append(name)

    all_results: Dict[str, Dict] = {}
    scenarios_run: List[str] = []

    def _run_and_store(name: str, split_data: Dict):
        # random generator needs the split-specific product pool
        pool = []
        for _, row in split_data["train"].iterrows():
            rxn = row.get("reaction_smiles", "")
            if isinstance(rxn, str):
                prod = _product_of(rxn)
                if prod:
                    pool.append(prod)
        gens = dict(generators_cache)
        try:
            gens[METHOD_RANDOM] = NegativeGenerator(
                METHOD_RANDOM, seed=args.seed, product_pool=pool)
        except TypeError:
            gens[METHOD_RANDOM] = NegativeGenerator(METHOD_RANDOM, seed=args.seed)
        res = run_scenario(name, split_data, model, device, args, gens)
        all_results[name] = res
        scenarios_run.append(name)
        # persist primary-pool records
        for arm, by_pool in res.get("test_records", {}).items():
            recs = by_pool.get(PRIMARY_POOL)
            if recs:
                _write_records_csv(
                    records_dir / f"{name}__{arm}__{PRIMARY_POOL}.csv", recs)

    for split_name in split_names:
        try:
            split_data = load_hitea_split(args.parquet, args.ood_dir,
                                          split_name, args.max_train,
                                          args.max_test)
        except Exception as exc:
            print(f"[phase4] ERROR loading split '{split_name}': {exc}")
            continue
        _run_and_store(split_name, split_data)

    if not args.no_ni and args.ni_csv.exists():
        from pc_cng.run_phase3_external_validation import load_ni_coupling
        try:
            ni_data = load_ni_coupling(args.ni_csv, args.max_train, args.max_test)
            for part in ("train", "test"):
                ni_data[part]["reaction_family"] = "NI_COUPLING"
                ni_data[part]["experimental_group"] = ni_data[part]["split_key"]
                ni_data[part]["yield_bin"] = 0
            _run_and_store("ni_coupling", ni_data)
        except Exception as exc:
            print(f"[phase4] ERROR on NI Coupling: {exc}")

    if not args.no_uspto and args.uspto_csv.exists():
        from pc_cng.run_phase3_external_validation import load_uspto_patent
        try:
            uspto_data = load_uspto_patent(args.uspto_csv, args.ood_dir,
                                           args.max_train, args.max_test)
            for part in ("train", "test"):
                uspto_data[part]["reaction_family"] = "USPTO_PATENT"
                uspto_data[part]["experimental_group"] = uspto_data[part]["split_key"]
                uspto_data[part]["yield_bin"] = 0
            _run_and_store("uspto_patent", uspto_data)
        except Exception as exc:
            print(f"[phase4] ERROR on USPTO Patent: {exc}")

    # ----- Holm + verdict
    print(f"\n[phase4] === Holm correction (family-wise alpha={args.alpha}) ===")
    holm_res = apply_holm(all_results, args.alpha)
    print(f"  {holm_res.get('n_significant', 0)}/{holm_res.get('n_total', 0)} "
          f"tests significant after Holm")

    verdict = compute_verdict(all_results, holm_res, scenarios_run)

    # ----- Horizontal comparison table (primary pool, source-macro)
    print(f"\n[phase4] === Horizontal comparison table: source-macro "
          f"AUPRC[{PRIMARY_POOL}] (fixed test set) ===")
    arms = [a for a in list(MAIN_ARMS) + list(DIFF_ARMS)
            if any(a in r.get("auprc_matrix", {}) for r in all_results.values())]
    header = f"  {'scenario':<18}" + "".join(f"{a:<20}" for a in arms)
    print(header)
    table_rows = []
    for sc in scenarios_run:
        mat = all_results[sc].get("auprc_matrix", {})
        row = {"scenario": sc}
        line = f"  {sc:<18}"
        for a in arms:
            entry = mat.get(a, {}).get(PRIMARY_POOL)
            v = entry.get("source_macro") if isinstance(entry, dict) else entry
            row[a] = v
            line += f"{v:<20.4f}" if isinstance(v, float) else f"{'-':<20}"
        print(line)
        table_rows.append(row)

    print(f"\n[phase4] === Verdict ===")
    h1 = verdict["H1_sota"]
    print(f"  H1 (learned SOTA on fixed semi_hard): "
          f"{h1['n_sota_scenarios']}/{h1['n_scenarios']} scenarios -> "
          f"{'GO' if h1['go'] else 'NO_GO'}")
    h2 = verdict["H2_hard_control"]
    if h2["median_auprc"] is not None:
        print(f"  H2 (shuffled ~0.5 hard control): median="
          f"{h2['median_auprc']:.4f} -> "
          f"{'ACHIEVED' if h2['achieved'] else 'NOT_ACHIEVED'}")
    h3 = verdict["H3_inverted_u"]
    print(f"  H3 (inverted-U utility): {h3['n_scenarios_both']}/"
          f"{len(scenarios_run)} scenarios semihard>easy&hard -> "
          f"{'SUPPORTED' if h3['supported'] else 'NOT_SUPPORTED'}")

    # ----- Persist
    def _strip(res):
        return {k: v for k, v in res.items() if k != "test_records"}
    with open(args.output / "per_scenario_results.json", "w") as fh:
        json.dump({sc: _strip(r) for sc, r in all_results.items()},
                  fh, indent=2, default=str)
    with open(args.output / "holm_correction.json", "w") as fh:
        json.dump(holm_res, fh, indent=2, default=str)
    with open(args.output / "comparison_table_semihard.json", "w") as fh:
        json.dump(table_rows, fh, indent=2, default=str)
    with open(args.output / "verdict.json", "w") as fh:
        json.dump(verdict, fh, indent=2, default=str)

    script_sha = "unknown"
    try:
        script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except Exception:
        pass
    manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "script_sha256": script_sha,
        "frozen_difficulty": FROZEN_DIFFICULTY,
        "primary_pool": PRIMARY_POOL,
        "use_gnn": args.use_gnn,
        "max_train": args.max_train, "max_test": args.max_test,
        "n_bootstrap": args.n_bootstrap, "seed": args.seed,
        "scenarios": scenarios_run,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "elapsed_sec": time.time() - t_start,
    }
    with open(args.output / "run_manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)

    print(f"\n[phase4] DONE in {time.time() - t_start:.1f}s")
    print(f"[phase4] Results: {args.output}")
    return 0


def _write_records_csv(path: Path, recs: List[Dict]) -> None:
    if not recs:
        return
    fields = ["reaction_smiles", "negative_smiles", "label", "score",
              "experimental_group", "reaction_family", "yield_bin",
              "method", "is_positive", "source", "sim"]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in recs:
            writer.writerow({k: r.get(k, "") for k in fields})


if __name__ == "__main__":
    sys.exit(main())
