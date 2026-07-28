"""Phase 4 follow-up: UNION negative-source arm on the frozen fixed pools.

Motivation (evidence from phase4_v41_smoke + diag_shuffled_transfer.py):
classifiers trained on REAL-product negatives (shuffled_parent /
random_mismatch) learn a broad reactant-product compatibility prior that
transfers to boundary negatives (AUPRC 0.74-0.95), while classifiers
trained on a single synthetic boundary source (learned_structured /
rule_pc_cng) learn narrow detectors that do not fully cover even their own
generator's distribution on held-out reactions.  The union arm tests the
practical SOTA recipe: train on a MIXTURE of negative sources
(boundary-structured: learned + rule; plausibility: shuffled-real) at the
SAME 1:1 positive:negative budget as every other arm.

Design (frozen before running):
  * For each train positive, generate up to three negatives (learned, rule,
    shuffled-real row i+7) and SAMPLE ONE uniformly among available sources.
    => identical training-set size as all other arms; only the source
    mixture differs.
  * Evaluation reuses the SAVED fixed semi_hard pool records from a
    completed run_phase4_fixed_testset output dir
    (per_scenario_records/*__{arm}__semi_hard.csv), so the comparison vs the
    7 existing arms is exactly paired (same records, same order).
  * Classifier architecture matches the base run (--use-gnn must equal the
    base run's use_gnn, checked against run_manifest.json).
  * Reports: source-macro AUPRC on the primary pool, paired cluster
    bootstrap CI for union vs {shuffled_parent, learned_structured,
    rule_pc_cng, random_mismatch, diff_semihard}, Holm correction across
    scenarios.

Run:
    python3 -m pc_cng.run_phase4_union_arm \
        --base-results results/phase4_fixed_testset_v41 --gpu 0 --use-gnn
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CNS_ROOT = _REPO_ROOT / "chem_negative_sampling"
if str(_CNS_ROOT) not in sys.path:
    sys.path.insert(0, str(_CNS_ROOT))

os.environ.setdefault("RDKitRDLogger", "0")

from pc_cng.paired_cluster_inference import (  # noqa: E402
    auprc_metric,
    holm_correction,
    paired_cluster_bootstrap,
)
from pc_cng.phase3_enhanced import (  # noqa: E402
    morgan_fingerprint,
    reaction_fp_enhanced,
)
from pc_cng.robust_negative_generator import RobustNegativeGenerator  # noqa: E402
from pc_cng.run_phase4_fixed_testset import (  # noqa: E402
    METHOD_LEARNED,
    METHOD_RULE,
    METHOD_SHUFFLED_PARENT,
    PRIMARY_POOL,
    SHUFFLED_OFFSET,
    _product_of,
    _row_meta,
    generate_union_candidates,
    per_source_auprc,
    score_records,
    source_macro_auprc_metric,
    train_classifier,
)
from pc_cng.run_phase3_external_validation import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_NI_CSV,
    DEFAULT_OOD_DIR,
    DEFAULT_PARQUET,
    METHOD_RANDOM,
    NegativeGenerator,
    load_g8c_model,
    load_hitea_split,
    load_ni_coupling,
    load_uspto_patent,
)

METHOD_UNION = "learned_union"

# These bounds are inherited from the frozen Phase 4 v4.1 difficulty
# definition.  They are deliberately not tuned by union_v2.
DIFFICULTY_SEMI_HARD_MIN = 0.40
DIFFICULTY_SEMI_HARD_MAX = 0.75
DIFFICULTY_SEMI_HARD_TARGET = 0.575


def _tanimoto(fp_a: Optional[np.ndarray], fp_b: Optional[np.ndarray]) -> float:
    """Return binary Tanimoto similarity without introducing a new dependency."""
    if fp_a is None or fp_b is None:
        return float("nan")
    union = float(np.maximum(fp_a, fp_b).sum())
    if union <= 0:
        return 0.0
    return float(np.minimum(fp_a, fp_b).sum() / union)


def _difficulty_pool(sim: float) -> str:
    if sim < DIFFICULTY_SEMI_HARD_MIN:
        return "easy"
    if sim <= DIFFICULTY_SEMI_HARD_MAX:
        return "semi_hard"
    return "hard"


def _choose_difficulty_matched_candidate(
    candidates: List[Tuple[str, str]],
    true_product: str,
    rng: random.Random,
    stats: Optional[Dict[str, Any]] = None,
) -> Optional[Tuple[str, str, float, bool]]:
    """Choose one candidate using the frozen semi-hard definition.

    The previous union_v2 implementation called each source twice and then
    sampled uniformly.  Because ``NegativeGenerator`` caches one result per
    reaction, those calls were usually duplicates and no difficulty matching
    happened at all.  This helper scores the actually available candidates,
    prefers the frozen semi-hard interval, and records an explicit fallback
    when no source produced a candidate in that interval.
    """
    true_fp = morgan_fingerprint(true_product)
    scored: List[Tuple[str, str, float]] = []
    seen = set()
    for neg_rxn, source in candidates:
        neg_product = _product_of(neg_rxn)
        key = (source, neg_product)
        if not neg_product or neg_product == true_product or key in seen:
            continue
        seen.add(key)
        sim = _tanimoto(morgan_fingerprint(neg_product), true_fp)
        if not np.isfinite(sim):
            continue
        scored.append((neg_rxn, source, sim))

    if not scored:
        if stats is not None:
            stats["no_valid_candidates"] = stats.get("no_valid_candidates", 0) + 1
        return None

    semi = [c for c in scored
            if DIFFICULTY_SEMI_HARD_MIN <= c[2] <= DIFFICULTY_SEMI_HARD_MAX]
    pool = semi or scored
    # The random tie breaker is only used for exact-distance ties and is
    # seeded by the caller; it does not change the frozen difficulty bounds.
    chosen = min(pool, key=lambda c: (abs(c[2] - DIFFICULTY_SEMI_HARD_TARGET),
                                      rng.random()))
    matched = bool(semi)
    if stats is not None:
        stats["candidate_count"] = stats.get("candidate_count", 0) + len(scored)
        stats["semi_hard_available"] = stats.get("semi_hard_available", 0) + int(bool(semi))
        stats["fallback_out_of_band"] = stats.get("fallback_out_of_band", 0) + int(not matched)
    return chosen[0], chosen[1], float(chosen[2]), matched
BASE_ARM_FOR_POOLS = METHOD_SHUFFLED_PARENT  # any arm CSV carries the pool
BASE_ARMS = (METHOD_RULE, METHOD_RANDOM, METHOD_LEARNED,
             METHOD_SHUFFLED_PARENT, "diff_semihard")


# ---------------------------------------------------------------------------
# Union training data
# ---------------------------------------------------------------------------

def build_union_train(
    train_rows,
    generators: Dict[str, RobustNegativeGenerator],
    seed: int,
) -> Tuple[Optional[np.ndarray], np.ndarray, List[Dict]]:
    """One negative per positive, sampled uniformly from available sources.

    Budget-matched to the single-source arms (1:1 pos:neg); only the
    source mixture differs.
    """
    rng = random.Random(seed + 41)

    parsed: List[Tuple[str, str, str, str, Dict]] = []
    for _, row in train_rows.iterrows():
        rxn = row.get("reaction_smiles")
        if not rxn or not isinstance(rxn, str):
            continue
        parts = rxn.split(">")
        if len(parts) != 3 or not parts[2]:
            continue
        parsed.append((rxn, parts[0], parts[1], parts[2], _row_meta(row)))
    n = len(parsed)

    fps: List[np.ndarray] = []
    labels: List[int] = []
    records: List[Dict] = []
    src_counts: Dict[str, int] = defaultdict(int)

    gen_learned = generators.get(METHOD_LEARNED)
    gen_rule = generators.get(METHOD_RULE)

    for i, (rxn, reactants, agents, true_prod, meta) in enumerate(parsed):
        pos_fp = reaction_fp_enhanced(rxn)
        if pos_fp is None:
            continue
        cands: List[Tuple[str, str]] = []  # (neg_rxn, source)
        if gen_learned is not None:
            neg = gen_learned.generate(rxn)
            if neg:
                cands.append((neg, METHOD_LEARNED))
        if gen_rule is not None:
            neg = gen_rule.generate(rxn)
            if neg:
                cands.append((neg, METHOD_RULE))
        p_shuf = parsed[(i + SHUFFLED_OFFSET) % n][3]
        if p_shuf == true_prod:
            p_shuf = parsed[(i + SHUFFLED_OFFSET + 1) % n][3]
        if p_shuf and p_shuf != true_prod:
            cands.append((f"{reactants}>{agents}>{p_shuf}",
                          METHOD_SHUFFLED_PARENT))
        if not cands:
            continue
        neg_rxn, src = cands[rng.randrange(len(cands))]
        neg_fp = reaction_fp_enhanced(neg_rxn)
        if neg_fp is None:
            continue
        fps.extend([pos_fp, neg_fp])
        labels.extend([1, 0])
        src_counts[src] += 1
        records.append({"reaction_smiles": rxn, "negative_smiles": rxn,
                        "label": 1, "score": 0.0, "method": METHOD_UNION,
                        "is_positive": True, **meta})
        records.append({"reaction_smiles": neg_rxn,
                        "negative_smiles": _product_of(neg_rxn) or neg_rxn,
                        "label": 0, "score": 0.0, "method": METHOD_UNION,
                        "is_positive": False, "union_source": src, **meta})
    if not fps:
        return None, np.array([]), []
    print(f"  [union] train source mixture: {dict(src_counts)}")
    X = np.vstack(fps)
    y = np.array(labels, dtype=np.float32)
    assert len(records) == len(y)
    return X, y, records


def build_union_v2_train(
    train_rows,
    generators: Dict[str, RobustNegativeGenerator],
    raw_generators: Dict[str, NegativeGenerator],
    seed: int,
    product_pool: List[str],
    stats: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[np.ndarray], np.ndarray, List[Dict]]:
    """Build a budget-matched union with frozen semi-hard selection.

    ``union_v2`` is an exploratory development arm.  It generates the
    candidates actually available from learned, rule and shuffled-real
    sources, computes the same product Tanimoto similarity used by the
    frozen Phase 4 pools, and selects a candidate in ``[0.40, 0.75]`` when
    one exists.  If no source produces an in-band candidate, it selects the
    closest available candidate and records the out-of-band fallback.

    The previous implementation called cached generators twice and sampled
    uniformly, so it neither over-generated nor difficulty-matched.  Keeping
    the fallback explicit prevents a small dataset from being reported as a
    fully matched arm when it is not.
    """
    rng = random.Random(seed + 41)

    gen_learned = generators.get(METHOD_LEARNED)
    gen_rule = generators.get(METHOD_RULE)

    parsed: List[Tuple[str, str, str, str, Dict]] = []
    for _, row in train_rows.iterrows():
        rxn = row.get("reaction_smiles")
        if not rxn or not isinstance(rxn, str):
            continue
        parts = rxn.split(">")
        if len(parts) != 3 or not parts[2]:
            continue
        parsed.append((rxn, parts[0], parts[1], parts[2], _row_meta(row)))
    n = len(parsed)

    fps: List[np.ndarray] = []
    labels: List[int] = []
    records: List[Dict] = []
    src_counts: Dict[str, int] = defaultdict(int)
    local_stats: Dict[str, Any] = stats if stats is not None else {}

    for i, (rxn, reactants, agents, true_prod, meta) in enumerate(parsed):
        pos_fp = reaction_fp_enhanced(rxn)
        if pos_fp is None:
            continue

        cands: List[Tuple[str, str]] = []

        # RobustNegativeGenerator and NegativeGenerator cache one result per
        # reaction.  Calling them twice therefore does not create two
        # candidates; keep one candidate per source and use two distinct
        # shuffled products only as a genuine fallback diversity source.
        if gen_learned is not None:
            neg = gen_learned.generate(rxn)
            if neg:
                cands.append((neg, METHOD_LEARNED))
        if gen_rule is not None:
            neg = gen_rule.generate(rxn)
            if neg:
                cands.append((neg, METHOD_RULE))
        for offset in (SHUFFLED_OFFSET, SHUFFLED_OFFSET + 1):
            p_shuf = parsed[(i + offset) % n][3]
            if p_shuf == true_prod:
                continue
            if p_shuf:
                cands.append((f"{reactants}>{agents}>{p_shuf}",
                              METHOD_SHUFFLED_PARENT))

        if not cands:
            continue

        chosen = _choose_difficulty_matched_candidate(
            cands, true_prod, rng, stats=local_stats)
        if chosen is None:
            continue
        neg_rxn, src, sim, matched = chosen

        neg_fp = reaction_fp_enhanced(neg_rxn)
        if neg_fp is None:
            continue
        fps.extend([pos_fp, neg_fp])
        labels.extend([1, 0])
        src_counts[src] += 1
        records.append({"reaction_smiles": rxn, "negative_smiles": rxn,
                        "label": 1, "score": 0.0,
                        "method": METHOD_UNION + "_v2",
                        "is_positive": True, **meta})
        records.append({"reaction_smiles": neg_rxn,
                        "negative_smiles": _product_of(neg_rxn) or neg_rxn,
                        "label": 0, "score": 0.0,
                        "method": METHOD_UNION + "_v2",
                        "is_positive": False, "union_source": src,
                        "sim": sim,
                        "difficulty_pool": _difficulty_pool(sim),
                        "difficulty_match": matched,
                        **meta})
    if not fps:
        return None, np.array([]), []
    local_stats["selected_source_counts"] = dict(src_counts)
    print(f"  [union_v2] train source mixture: {dict(src_counts)}")
    X = np.vstack(fps)
    y = np.array(labels, dtype=np.float32)
    assert len(records) == len(y)
    return X, y, records


# ---------------------------------------------------------------------------
# Saved-pool IO
# ---------------------------------------------------------------------------

def _bool(v) -> bool:
    return v in ("True", "1", True, 1)


def load_saved_pool(base_records: Path, scenario: str) -> Optional[List[Dict]]:
    """Load the frozen semi_hard pool (from any arm CSV; scores reset)."""
    f = base_records / f"{scenario}__{BASE_ARM_FOR_POOLS}__{PRIMARY_POOL}.csv"
    if not f.exists():
        return None
    out: List[Dict] = []
    for r in csv.DictReader(open(f)):
        out.append({
            "reaction_smiles": r["reaction_smiles"],
            "negative_smiles": r.get("negative_smiles", ""),
            "label": int(r["label"]),
            "score": 0.0,
            "experimental_group": r.get("experimental_group", "default"),
            "reaction_family": r.get("reaction_family", "unknown"),
            "yield_bin": int(r.get("yield_bin") or 0),
            "method": f"fixed_{PRIMARY_POOL}",
            "is_positive": _bool(r.get("is_positive")),
            "source": r.get("source", "?"),
            "sim": float(r.get("sim") or 0.0),
        })
    return out


def load_baseline_scores(base_records: Path, scenario: str, arm: str
                         ) -> Optional[List[Dict]]:
    f = base_records / f"{scenario}__{arm}__{PRIMARY_POOL}.csv"
    if not f.exists():
        return None
    out: List[Dict] = []
    for r in csv.DictReader(open(f)):
        out.append({
            "reaction_smiles": r["reaction_smiles"],
            "label": int(r["label"]),
            "score": float(r["score"]),
            "experimental_group": r.get("experimental_group", "default"),
            "is_positive": _bool(r.get("is_positive")),
            "source": r.get("source", "?"),
        })
    return out


# ---------------------------------------------------------------------------
# Scenario row loading (mirrors run_phase4_fixed_testset.main)
# ---------------------------------------------------------------------------

def _scenario_rows(args, scenario: str):
    if scenario == "ni_coupling":
        data = load_ni_coupling(args.ni_csv, args.max_train, args.max_test)
        for part in ("train", "test"):
            data[part]["reaction_family"] = "NI_COUPLING"
            data[part]["experimental_group"] = data[part]["split_key"]
            data[part]["yield_bin"] = 0
        return data
    if scenario == "uspto_patent":
        uspto_csv = Path(
            "/home/cunyuliu/pc_cng_research/data/processed/"
            "uspto_openmolecules_normalized.csv")
        data = load_uspto_patent(uspto_csv, args.ood_dir, args.max_train,
                                 args.max_test)
        for part in ("train", "test"):
            data[part]["reaction_family"] = "USPTO_PATENT"
            data[part]["experimental_group"] = data[part]["split_key"]
            data[part]["yield_bin"] = 0
        return data
    return load_hitea_split(args.parquet, args.ood_dir, scenario,
                            args.max_train, args.max_test)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_learned_model(
    model: Any,
    model_error: Optional[str],
    checkpoint: Path,
) -> None:
    """Refuse to turn a learned/rule/shuffled union into a two-source arm."""
    if model is None:
        raise SystemExit(
            "[union][FATAL] learned source expert is unavailable; refusing "
            f"to degrade the union intervention. checkpoint={checkpoint} "
            f"error={model_error}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-results", type=Path, required=True,
                    help="completed run_phase4_fixed_testset output dir")
    ap.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    ap.add_argument("--ni-csv", type=Path, default=DEFAULT_NI_CSV)
    ap.add_argument("--ood-dir", type=Path, default=DEFAULT_OOD_DIR)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--use-gnn", action="store_true",
                    help="must match the classifier used in the base run")
    ap.add_argument("--splits", nargs="+", default=None)
    ap.add_argument("--max-train", type=int, default=500)
    ap.add_argument("--max-test", type=int, default=200)
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260726)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--difficulty-match", action="store_true",
                    help="use v2 difficulty-matched training (semi_hard pool "
                         "+ generator sources); addresses uspto_patent gap")
    args = ap.parse_args()

    base_records = args.base_results / "per_scenario_records"
    if not base_records.exists():
        raise SystemExit(f"no per_scenario_records under {args.base_results}")
    manifest_p = args.base_results / "run_manifest.json"
    if manifest_p.exists():
        manifest = json.load(open(manifest_p))
        print(f"[union] base run manifest: gnn={manifest.get('use_gnn')} "
              f"seed={manifest.get('seed')}")
        if bool(manifest.get("use_gnn")) != bool(args.use_gnn):
            print("[union][WARN] --use-gnn mismatch with base run; scores "
                  "are still paired but architecture differs")

    import torch
    device = None
    print(f"[union] torch.cuda.is_available() = {torch.cuda.is_available()}")
    if args.gpu is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        print(f"[union] device = {torch.cuda.get_device_name(args.gpu)}")

    model, model_err = load_g8c_model(args.checkpoint, device=device)
    print(f"[union] G8-C model {'loaded' if model is not None else model_err}")
    require_learned_model(model, model_err, args.checkpoint)

    scenarios = sorted({f.name.split("__")[0] for f in base_records.glob(
        f"*__{BASE_ARM_FOR_POOLS}__{PRIMARY_POOL}.csv")})
    if args.splits:
        scenarios = [s for s in scenarios if s in set(args.splits)]
    print(f"[union] scenarios: {scenarios}")

    generators_cache: Dict[str, Any] = {
        METHOD_RULE: NegativeGenerator(METHOD_RULE, seed=args.seed),
    }
    if model is not None:
        generators_cache[METHOD_LEARNED] = NegativeGenerator(
            METHOD_LEARNED, model=model, top_k=1, device=device,
            seed=args.seed)

    all_results: Dict[str, Any] = {}
    for scenario in scenarios:
        print(f"\n[union] === Scenario: {scenario} ===")
        try:
            rows = _scenario_rows(args, scenario)
        except Exception as exc:
            print(f"  [skip] cannot reload split rows: {exc}")
            continue
        train_rows = rows["train"]

        generators = {
            k: RobustNegativeGenerator(v, seed=args.seed)
            for k, v in generators_cache.items()
        }

        gen_stats: Dict[str, Any] = {}
        if args.difficulty_match:
            product_pool: List[str] = []
            for _, row in train_rows.iterrows():
                rxn = row.get("reaction_smiles", "")
                if isinstance(rxn, str):
                    prod = _product_of(rxn)
                    if prod:
                        product_pool.append(prod)

            X_tr, y_tr, rec_tr = build_union_v2_train(
                train_rows, generators, generators_cache, args.seed,
                product_pool, stats=gen_stats)
        else:
            X_tr, y_tr, rec_tr = build_union_train(
                train_rows, generators, args.seed)
        if X_tr is None or len(X_tr) < 10:
            print("  [skip] insufficient union train data")
            continue
        t0 = time.time()
        clf = train_classifier(X_tr, y_tr, rec_tr, args.seed,
                               use_gnn=args.use_gnn)
        train_sec = time.time() - t0

        pool_recs = load_saved_pool(base_records, scenario)
        if not pool_recs:
            print("  [skip] no saved pool records")
            continue
        scored = score_records(clf, pool_recs, use_gnn=args.use_gnn)

        arm_suffix = "_v2" if args.difficulty_match else ""
        arm_name = METHOD_UNION + arm_suffix
        out_f = base_records / f"{scenario}__{arm_name}__{PRIMARY_POOL}.csv"
        with open(out_f, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(scored[0].keys()))
            w.writeheader()
            w.writerows(scored)

        union_sma = float(source_macro_auprc_metric(scored))
        train_negatives = [r for r in rec_tr if not r.get("is_positive")]
        matched_count = sum(bool(r.get("difficulty_match"))
                            for r in train_negatives)
        sc_res: Dict[str, Any] = {
            "train_sec": train_sec, "n_train": len(X_tr),
            "source_macro": union_sma,
            "pooled": float(auprc_metric(scored)),
            "per_source": per_source_auprc(scored),
            "generation_stats": gen_stats,
            "train_difficulty": {
                "definition": {
                    "similarity_metric": "tanimoto_morgan_radius2_2048bits",
                    "semi_hard_min": DIFFICULTY_SEMI_HARD_MIN,
                    "semi_hard_max": DIFFICULTY_SEMI_HARD_MAX,
                    "target": DIFFICULTY_SEMI_HARD_TARGET,
                },
                "n_negative": len(train_negatives),
                "n_in_band": matched_count,
                "matched_fraction": (
                    matched_count / len(train_negatives)
                    if train_negatives else None),
                "fallback_count": sum(
                    not bool(r.get("difficulty_match"))
                    for r in train_negatives),
            },
            "paired_ci": {},
        }
        print(f"  [union] trained ({len(X_tr)} samples, {train_sec:.1f}s) "
              f"srcMacroAUPRC[{PRIMARY_POOL}]={union_sma:.4f}")

        for base_arm in BASE_ARMS:
            base_recs = load_baseline_scores(base_records, scenario, base_arm)
            if not base_recs or len(base_recs) != len(scored):
                continue
            try:
                ci = paired_cluster_bootstrap(
                    scored, base_recs,
                    metric_fn=source_macro_auprc_metric,
                    cluster_key="experimental_group",
                    n_bootstrap=args.n_bootstrap, seed=args.seed)
                key = f"{arm_name}_vs_{base_arm}"
                sc_res["paired_ci"][key] = ci
                print(f"  [CI:{PRIMARY_POOL}] {key}: "
                      f"delta={ci['delta_mean']:+.4f} "
                      f"CI=[{ci['delta_ci_low']:+.4f}, "
                      f"{ci['delta_ci_high']:+.4f}] p={ci['p_value']:.4f}")
            except Exception as exc:
                sc_res["paired_ci"][f"{METHOD_UNION}_vs_{base_arm}"] = \
                    {"error": str(exc)}
        all_results[scenario] = sc_res

    # ----- Holm across scenarios
    tests: List[Dict] = []
    for sc, res in all_results.items():
        for key, ci in res.get("paired_ci", {}).items():
            if "error" in ci:
                continue
            tests.append({"scenario": sc, "pair": key, "p": ci["p_value"],
                          "delta": ci["delta_mean"]})
    holm_list = holm_correction([t["p"] for t in tests], alpha=args.alpha) \
        if tests else []
    for t, h in zip(tests, holm_list):
        t["adjusted_p"] = h["adjusted_p"]
        t["significant"] = h["rejected"]
    n_sig = sum(1 for t in tests if t.get("significant"))
    print(f"\n[union] === Holm correction (alpha={args.alpha}) ===")
    print(f"  {n_sig}/{len(tests)} tests significant after Holm")

    # ----- horizontal comparison table vs base arms
    base_json_p = args.base_results / "per_scenario_results.json"
    base_json = json.load(open(base_json_p)) if base_json_p.exists() else {}
    print(f"\n[union] === source-macro AUPRC[{PRIMARY_POOL}] vs base arms ===")
    hdr = f"  {'scenario':<16} {'UNION':>8}"
    for a in BASE_ARMS:
        hdr += f" {a[:12]:>12}"
    print(hdr)
    n_win = 0
    n_eval = 0
    for sc in scenarios:
        res = all_results.get(sc)
        if not res:
            continue
        n_eval += 1
        u = res["source_macro"]
        row = f"  {sc:<16} {u:>8.4f}"
        mat = base_json.get(sc, {}).get("auprc_matrix", {})
        wins = True
        for a in BASE_ARMS:
            v = mat.get(a, {}).get(PRIMARY_POOL, {}).get("source_macro")
            row += f" {v:>12.4f}" if isinstance(v, float) else \
                f" {'nan':>12}"
            if isinstance(v, float) and u <= v:
                wins = False
        if wins:
            n_win += 1
        print(row)
    print(f"\n[union] union numerically beats ALL base arms in "
          f"{n_win}/{n_eval} scenarios")

    out = {"arm": arm_name, "use_gnn": args.use_gnn,
           "difficulty_match_requested": bool(args.difficulty_match),
           "learned_checkpoint": {
               "path": str(args.checkpoint),
               "sha256": _sha256(args.checkpoint),
               "load_status": "loaded",
           },
           "difficulty_definition": {
               "similarity_metric": "tanimoto_morgan_radius2_2048bits",
               "semi_hard_min": DIFFICULTY_SEMI_HARD_MIN,
               "semi_hard_max": DIFFICULTY_SEMI_HARD_MAX,
               "target": DIFFICULTY_SEMI_HARD_TARGET,
           },
           "base_results": str(args.base_results),
           "results": all_results, "holm_tests": tests,
           "n_scenarios_union_wins_outright": n_win}
    with open(args.base_results / "union_arm_results.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[union] saved -> {args.base_results / 'union_arm_results.json'}")


if __name__ == "__main__":
    main()
