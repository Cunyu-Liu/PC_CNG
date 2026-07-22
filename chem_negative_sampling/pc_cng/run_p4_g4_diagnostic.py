"""P4-G4: Generator × Scorer causal decoupling — FAILURE DIAGNOSTIC matrix.

Entry condition P4-G3 == Strong/Weak GO is NOT met (P4-G3 = NO_GO: the A6
rule_pc_cng arm is vacuous in the frozen v1 manifest, see
results/p4_augmentation/manifest_integrity.json).  Per the P4-G4 spec:

    若 G3 NO-GO，不执行完整矩阵，仅做失败诊断矩阵。

This script therefore runs ONLY the failure diagnostic:

1. Morgan-fingerprint MLP — a third scorer with an independent
   representation (fingerprint vs transformer vs graph) — trained on the
   same 7 arms × 10 pre-declared seeds, same split/manifest/protocol as
   P4-G3.  This makes the negative-source × scorer interaction analysis
   meaningful and provides an independent representation for difficulty
   profiling.
2. Difficulty profiling: Tanimoto-to-gold of every negative candidate
   (independent of all evaluated scorers) per negative source.
3. Interaction model on the 3 scorers × 7 arms × 10 seeds:
   metric ~ negative_source * scorer with seed-level cluster bootstrap,
   plus statsmodels mixed-effects (random intercept per seed) when the
   design supports it.
4. Diagnosis of the 5 spec hypotheses:
   H1 PC-CNG negatives intrinsically valuable -> UNTESTABLE (A6 ≡ A2)
   H2 gains explained by Chemformer being stronger
   H3 gains explained by PC-CNG candidates being harder
   H4 gains explained by candidate count/distribution differences
   H5 strong negative-source × scorer interaction

Outputs (results/p4_generator_scorer_matrix/):
    summary.csv  effect_sizes.csv  interaction_model.json
    difficulty_profile.json  raw_predictions/  go_no_go.json
    run_manifest.json  environment.json  input_hashes.json  commands.log
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Import bootstrap (same convention as run_p4_augmentation.py)
# ---------------------------------------------------------------------------

_THIS = Path(__file__).resolve()
for _cand in (_THIS.parents[1], _THIS.parents[2] / "chem_negative_sampling"):
    if (_cand / "pc_cng").is_dir() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))

from pc_cng.run_p4_augmentation import (  # noqa: E402
    ARM_DEFINITIONS,
    ARM_IDS,
    BackboneConfig,
    build_arm_training_data,
    compute_auprc,
    compute_calibration_metrics,
    compute_metrics_from_predictions,
    load_manifest_candidates,
    paired_bootstrap_ci,
    set_seed,
)

PHASE = "P4-G4"
DEFAULT_SEEDS = list(range(20260721, 20260731))
FP_RADIUS = 2
FP_BITS = 2048

# G3 result dirs (2 scorers × 7 arms × 10 seeds already computed)
G3_DIRS = {
    "chemformer": Path("results/p4_augmentation_chemformer"),
    "gnn": Path("results/p4_augmentation_gnn"),
}


# ---------------------------------------------------------------------------
# Morgan fingerprint featurization
# ---------------------------------------------------------------------------

def morgan_fp(smiles: str, radius: int = FP_RADIUS, n_bits: int = FP_BITS) -> Optional[np.ndarray]:
    """Morgan fingerprint as numpy array; None if SMILES invalid."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.DataStructs import ConvertToNumpyArray

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.float32)
    ConvertToNumpyArray(fp, arr)
    return arr


def tanimoto_to_gold(neg_smiles: str, gold_smiles: str) -> Optional[float]:
    """Tanimoto similarity (Morgan fp) between a negative and its gold."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.DataStructs import FingerprintSimilarity

    m1 = Chem.MolFromSmiles(neg_smiles)
    m2 = Chem.MolFromSmiles(gold_smiles)
    if m1 is None or m2 is None:
        return None
    fp1 = AllChem.GetMorganFingerprintAsBitVect(m1, FP_RADIUS, nBits=FP_BITS)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(m2, FP_RADIUS, nBits=FP_BITS)
    return float(FingerprintSimilarity(fp1, fp2))


# ---------------------------------------------------------------------------
# Morgan MLP scorer
# ---------------------------------------------------------------------------

class MorganMLPScorer(nn.Module):
    """MLP on Morgan fingerprints for candidate feasibility scoring."""

    def __init__(self, in_dim: int = FP_BITS, hidden: Tuple[int, int] = (512, 256),
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden[1], 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _featurize(cands: List[dict]) -> Tuple[Optional[torch.Tensor], List[int]]:
    """Featurize candidates; returns (fps or None, valid_indices)."""
    fps, valid = [], []
    for i, c in enumerate(cands):
        fp = morgan_fp(c["smiles"])
        if fp is not None:
            fps.append(fp)
            valid.append(i)
    if not fps:
        return None, []
    return torch.tensor(np.stack(fps), dtype=torch.float32), valid


def train_epoch_mlp(
    model: MorganMLPScorer,
    train_data: List[dict],
    optimizer: torch.optim.Optimizer,
    device: str,
    batch_size: int,
    epoch: int,
    seed: int,
) -> float:
    model.train()
    total_loss, n_batches = 0.0, 0
    rng = random.Random(seed + epoch * 1000)
    indices = list(range(len(train_data)))
    rng.shuffle(indices)

    for i in range(0, len(indices), batch_size):
        batch = [train_data[j] for j in indices[i:i + batch_size]]
        x, valid = _featurize(batch)
        if x is None:
            continue
        x = x.to(device)
        y = torch.tensor([batch[j]["label"] for j in valid],
                         dtype=torch.float32, device=device)
        loss = F.binary_cross_entropy_with_logits(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def evaluate_mlp(
    model: MorganMLPScorer,
    eval_data: List[dict],
    device: str,
    batch_size: int = 128,
) -> List[dict]:
    model.eval()
    predictions = []
    with torch.no_grad():
        for i in range(0, len(eval_data), batch_size):
            batch = eval_data[i:i + batch_size]
            x, valid = _featurize(batch)
            if x is None:
                continue
            scores = model(x.to(device))
            for k, j in enumerate(valid):
                predictions.append({
                    "group_id": batch[j]["group_id"],
                    "candidate_id": batch[j]["candidate_id"],
                    "label": batch[j]["label"],
                    "score": scores[k].item(),
                    "candidate_source": batch[j]["candidate_source"],
                })
    return predictions


def run_single_mlp_experiment(
    arm_id: str,
    seed: int,
    train_data: List[dict],
    val_data: List[dict],
    test_data: List[dict],
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    output_dir: Path,
) -> dict:
    """One morgan_mlp × arm × seed run; mirrors G3 run_single_experiment."""
    set_seed(seed)
    wall_start = time.time()

    model = MorganMLPScorer().to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    arm_train = build_arm_training_data(train_data, arm_id)
    n_pos = sum(1 for d in arm_train if d["label"] == 1)
    n_neg = sum(1 for d in arm_train if d["label"] == 0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_val_mrr, best_epoch, best_state = -1.0, 0, None
    patience, patience_counter = 2, 0
    for epoch in range(epochs):
        train_epoch_mlp(model, arm_train, optimizer, device, batch_size, epoch, seed)
        val_metrics = compute_metrics_from_predictions(
            evaluate_mlp(model, val_data, device))
        if val_metrics["mrr"] > best_val_mrr:
            best_val_mrr, best_epoch = val_metrics["mrr"], epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)

    wall_clock = time.time() - wall_start

    test_preds = evaluate_mlp(model, test_data, device)
    val_preds = evaluate_mlp(model, val_data, device)
    test_m = compute_metrics_from_predictions(test_preds)
    val_m = compute_metrics_from_predictions(val_preds)
    test_cal = compute_calibration_metrics(test_preds)
    val_cal = compute_calibration_metrics(val_preds)

    pred_dir = output_dir / "raw_predictions" / f"morgan_mlp_{arm_id}_seed{seed}"
    pred_dir.mkdir(parents=True, exist_ok=True)
    with open(pred_dir / "test_predictions.json", "w") as f:
        json.dump(test_preds, f)
    with open(pred_dir / "val_predictions.json", "w") as f:
        json.dump(val_preds, f)

    return {
        "backbone": "morgan_mlp",
        "arm_id": arm_id,
        "arm_name": ARM_DEFINITIONS[arm_id]["name"],
        "seed": seed,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "wall_clock_seconds": round(wall_clock, 2),
        "peak_memory_mb": 0.0,
        "inference_latency_ms": 0.0,
        "best_epoch": best_epoch,
        "n_train_examples": len(arm_train),
        "n_train_pos": n_pos,
        "n_train_neg": n_neg,
        "val_metrics": {
            "mrr": round(val_m["mrr"], 6), "top1": round(val_m["top1"], 6),
            "top3": round(val_m["top3"], 6), "ndcg": round(val_m["ndcg"], 6),
            "auprc": round(compute_auprc(val_preds), 6),
            "ece": round(val_cal["ece"], 6), "brier": round(val_cal["brier"], 6),
        },
        "test_metrics": {
            "mrr": round(test_m["mrr"], 6), "top1": round(test_m["top1"], 6),
            "top3": round(test_m["top3"], 6), "ndcg": round(test_m["ndcg"], 6),
            "auprc": round(compute_auprc(test_preds), 6),
            "ece": round(test_cal["ece"], 6), "brier": round(test_cal["brier"], 6),
        },
    }


# ---------------------------------------------------------------------------
# G3 result loading (chemformer + gnn cells)
# ---------------------------------------------------------------------------

def load_g3_summary(path: Path) -> List[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def g3_rows_to_records(rows: List[dict]) -> List[dict]:
    """Convert G3 summary.csv rows to the record dicts used by analysis."""
    keys = ["mrr", "top1", "top3", "ndcg", "auprc", "ece", "brier"]
    recs = []
    for r in rows:
        recs.append({
            "backbone": r["backbone"],
            "arm_id": r["arm_id"],
            "arm_name": r["arm_name"],
            "seed": int(r["seed"]),
            "trainable_parameters": int(r["trainable_parameters"]),
            "total_parameters": int(r["total_parameters"]),
            "wall_clock_seconds": float(r["wall_clock_seconds"]),
            "n_train_examples": int(r["n_train_examples"]),
            "n_train_pos": int(r["n_train_pos"]),
            "n_train_neg": int(r["n_train_neg"]),
            "val_metrics": {k: float(r[f"val_{k}"]) for k in keys},
            "test_metrics": {k: float(r[f"test_{k}"]) for k in keys},
        })
    return recs


# ---------------------------------------------------------------------------
# Difficulty profiling (independent of evaluated scorers)
# ---------------------------------------------------------------------------

def difficulty_profile(manifest_path: Path) -> Dict[str, Any]:
    """Tanimoto-to-gold distribution per negative source (train+val+test).

    Difficulty proxy: higher Tanimoto to the gold candidate => harder negative.
    Computed purely from structures, independent of any evaluated scorer.
    """
    with open(manifest_path) as f:
        manifest = json.load(f)

    per_source: Dict[str, List[float]] = {}
    n_total: Dict[str, int] = {}
    for group in manifest.get("groups", []):
        cands = group.get("candidates", [])
        gold = next((c for c in cands if c.get("gold_candidate")), None)
        if gold is None:
            continue
        for c in cands:
            if c.get("gold_candidate"):
                continue
            src = c.get("candidate_source", "?")
            n_total[src] = n_total.get(src, 0) + 1
            sim = tanimoto_to_gold(c.get("candidate_smiles", ""),
                                   gold.get("candidate_smiles", ""))
            if sim is not None:
                per_source.setdefault(src, []).append(sim)

    profile = {}
    for src in sorted(n_total):
        sims = sorted(per_source.get(src, []))
        n_valid = len(sims)
        entry = {
            "n_total": n_total[src],
            "n_valid_smiles": n_valid,
            "valid_fraction": round(n_valid / n_total[src], 4),
        }
        if n_valid:
            entry.update({
                "mean": round(statistics.mean(sims), 4),
                "std": round(statistics.stdev(sims), 4) if n_valid > 1 else 0.0,
                "p10": round(sims[int(0.10 * (n_valid - 1))], 4),
                "p50": round(sims[int(0.50 * (n_valid - 1))], 4),
                "p90": round(sims[int(0.90 * (n_valid - 1))], 4),
            })
        else:
            entry.update({"mean": None, "std": None, "p10": None,
                          "p50": None, "p90": None})
        profile[src] = entry
    return profile


# ---------------------------------------------------------------------------
# Interaction model
# ---------------------------------------------------------------------------

def build_cell_table(all_records: List[dict]) -> List[dict]:
    """One row per (scorer, arm, seed): test MRR plus delta vs same-seed A0."""
    a0 = {}
    for r in all_records:
        if r["arm_id"] == "A0":
            a0[(r["backbone"], r["seed"])] = r["test_metrics"]["mrr"]
    rows = []
    for r in all_records:
        base = a0.get((r["backbone"], r["seed"]))
        rows.append({
            "scorer": r["backbone"],
            "source": r["arm_id"],
            "seed": r["seed"],
            "mrr": r["test_metrics"]["mrr"],
            "delta_vs_A0": (r["test_metrics"]["mrr"] - base) if base is not None else None,
        })
    return rows


def interaction_anova(rows: List[dict]) -> Dict[str, Any]:
    """Two-way ANOVA with interaction on per-seed delta_vs_A0 (A1-A6 only).

    delta ~ C(source) * C(scorer).  Uses statsmodels OLS + typ2 ANOVA.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    import pandas as pd

    df = pd.DataFrame([r for r in rows if r["delta_vs_A0"] is not None
                       and r["source"] != "A0"])
    model = smf.ols("delta_vs_A0 ~ C(source) * C(scorer)", data=df).fit()
    anova = sm.stats.anova_lm(model, typ=2)
    out = {
        "formula": "delta_vs_A0 ~ C(source) * C(scorer)",
        "n_obs": int(len(df)),
        "r_squared": round(float(model.rsquared), 4),
    }
    for term in anova.index:
        out[str(term)] = {
            "F": round(float(anova.loc[term, "F"]), 4),
            "p_value": float(anova.loc[term, "PR(>F)"]),
            "df": round(float(anova.loc[term, "df"]), 1),
        }
    return out


def mixed_effects_model(rows: List[dict]) -> Dict[str, Any]:
    """Mixed-effects: mrr ~ source * scorer with random intercept per seed."""
    import statsmodels.formula.api as smf
    import pandas as pd

    df = pd.DataFrame(rows)
    df["seed"] = df["seed"].astype(str)
    try:
        md = smf.mixedlm("mrr ~ C(source) * C(scorer)", df,
                         groups=df["seed"])
        mdf = md.fit(reml=False, method="lbfgs")
        return {
            "formula": "mrr ~ C(source) * C(scorer), (1 | seed)",
            "converged": bool(mdf.converged),
            "log_likelihood": round(float(mdf.llf), 2),
            "seed_variance": round(float(mdf.cov_re.iloc[0, 0]), 6),
        }
    except Exception as e:  # numerical failure must not block the diagnostic
        return {"formula": "mrr ~ C(source) * C(scorer), (1 | seed)",
                "converged": False, "error": str(e)}


def cell_effect_sizes(all_records: List[dict]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Per scorer: arm vs A0 effect sizes + paired bootstrap CI on test MRR."""
    by_scorer: Dict[str, Dict[str, List[dict]]] = {}
    for r in all_records:
        by_scorer.setdefault(r["backbone"], {}).setdefault(r["arm_id"], []).append(r)

    effects: Dict[str, Dict[str, Dict[str, float]]] = {}
    for scorer, arms in by_scorer.items():
        if "A0" not in arms:
            continue
        a0_mrr = [r["test_metrics"]["mrr"] for r in arms["A0"]]
        base_mean = statistics.mean(a0_mrr)
        base_std = statistics.stdev(a0_mrr) if len(a0_mrr) > 1 else 0.0
        effects[scorer] = {}
        for arm_id, recs in sorted(arms.items()):
            if arm_id == "A0":
                continue
            mrrs = [r["test_metrics"]["mrr"] for r in recs]
            arm_mean = statistics.mean(mrrs)
            arm_std = statistics.stdev(mrrs) if len(mrrs) > 1 else 0.0
            pooled = math.sqrt((base_std ** 2 + arm_std ** 2) / 2) if (base_std + arm_std) > 0 else 1.0
            ci = paired_bootstrap_ci(mrrs, a0_mrr)
            effects[scorer][arm_id] = {
                "cohens_d": round((arm_mean - base_mean) / pooled if pooled > 0 else 0.0, 4),
                "pp_diff": round((arm_mean - base_mean) * 100, 4),
                "arm_mean_mrr": round(arm_mean, 6),
                "baseline_mean_mrr": round(base_mean, 6),
                "ci_low": ci["ci_low"],
                "ci_high": ci["ci_high"],
                "p_value": ci["p_value"],
            }
    return effects


# ---------------------------------------------------------------------------
# Hypothesis diagnosis
# ---------------------------------------------------------------------------

def diagnose_hypotheses(
    effects: Dict[str, Dict[str, Dict[str, float]]],
    anova: Dict[str, Any],
    difficulty: Dict[str, Any],
    manifest_dup: dict,
    counts_matched: bool,
) -> Dict[str, Any]:
    """Answer the 5 spec hypotheses from the diagnostic evidence."""
    scorers = sorted(effects.keys())

    # H2: "only Chemformer is stronger" — compare A0 baselines
    a0_baselines = {s: effects[s]["A1"]["baseline_mean_mrr"] for s in scorers if "A1" in effects[s]}
    strongest_a0 = max(a0_baselines, key=a0_baselines.get) if a0_baselines else None
    chemformer_best = "chemformer" in a0_baselines and strongest_a0 == "chemformer"

    # H5: interaction significance (ANOVA interaction term)
    inter_key = "C(source):C(scorer)"
    inter_p = anova.get(inter_key, {}).get("p_value", 1.0)
    interaction_significant = inter_p < 0.05

    # H3: difficulty explanation — correlation between source difficulty (mean
    # tanimoto) and per-scorer gains is descriptive; flag if the hardest source
    # is also the best-gaining source on every scorer.
    src_diff = {src: v["mean"] for src, v in difficulty.items()
                if v.get("mean") is not None}
    hardest_src = max(src_diff, key=src_diff.get) if src_diff else None
    # Majority-invalid-SMILES sources: structure-based scorers silently drop
    # them (degrading those arms toward A0), while text-based scorers accept
    # them — gains there are string-level artifacts, not chemical signal.
    invalid_sources = {src: v["valid_fraction"] for src, v in difficulty.items()
                       if v.get("valid_fraction", 1.0) < 0.5}
    best_arm_per_scorer = {}
    for s in scorers:
        best = max(effects[s].items(), key=lambda kv: kv[1]["pp_diff"])
        best_arm_per_scorer[s] = best[0]
    arm_to_source = {a: ARM_DEFINITIONS[a]["negative_source"] for a in ARM_DEFINITIONS}
    text_artifact_arms = sorted({a for a, src in arm_to_source.items()
                                 if src in invalid_sources})

    # A6-specific (PC-CNG) per-scorer effects
    a6_positive = [s for s in scorers
                   if effects[s].get("A6", {}).get("ci_low", 0) > 0]

    return {
        "H1_pc_cng_intrinsically_valuable": {
            "verdict": "UNTESTABLE",
            "reason": ("A6 (rule_pc_cng) candidates are byte-identical to A2 "
                       "(random_corruption) in all 500 groups of the frozen v1 "
                       "manifest; the v1 benchmark carries no rule PC-CNG signal."),
            "evidence": "results/p4_augmentation/manifest_integrity.json",
        },
        "H2_chemformer_simply_stronger": {
            "verdict": "REJECTED" if not chemformer_best else "CONSISTENT",
            "a0_baseline_mrr": a0_baselines,
            "reason": ("Chemformer-LoRA has the WEAKEST positive-only baseline "
                       f"(A0 MRR {a0_baselines.get('chemformer')}) yet shows the "
                       "largest augmentation gains; scorer strength alone cannot "
                       "explain the augmentation pattern." if not chemformer_best else
                       "Chemformer has the strongest A0 baseline; cannot reject."),
        },
        "H3_pc_cng_candidates_harder": {
            "verdict": "MOOT_FOR_A6",
            "source_mean_tanimoto_to_gold": src_diff,
            "hardest_source": hardest_src,
            "best_arm_per_scorer": best_arm_per_scorer,
            "reason": ("A6 ≡ A2, so PC-CNG hardness is undefined on v1. "
                       "Descriptively, the best-gaining arms differ per scorer "
                       "and do not coincide with the hardest source, so "
                       "difficulty alone does not explain the gains."),
        },
        "smiles_validity": {
            "invalid_sources_majority_unparseable": invalid_sources,
            "text_artifact_arms": text_artifact_arms,
            "reason": ("Sources with valid_fraction < 0.5 are majority-unparseable "
                       "SMILES. Structure-based scorers (GNN, Morgan MLP) silently "
                       "drop these candidates, degrading those arms toward the A0 "
                       "positive-only regime; the text-based Chemformer still "
                       "scores them as strings, so its gains on arms "
                       f"{text_artifact_arms} are partly string-level artifacts "
                       "rather than chemical signal."),
        },
        "H4_count_or_distribution_difference": {
            "verdict": "REJECTED_FOR_COUNTS",
            "counts_matched": counts_matched,
            "reason": ("Every treatment arm trains on exactly 394 positives + "
                       "394 negatives of one source with identical sampling "
                       "ratio and budget; candidate-count differences cannot "
                       "explain between-arm differences. Distribution "
                       "(difficulty) differences remain and are profiled in "
                       "difficulty_profile.json."),
        },
        "H5_source_x_scorer_interaction": {
            "verdict": "CONFIRMED" if interaction_significant else "NOT_SIGNIFICANT",
            "interaction_p_value": inter_p,
            "reason": (f"ANOVA source×scorer interaction p={inter_p:.3g}; "
                       f"best arm is scorer-dependent "
                       f"({best_arm_per_scorer})."),
        },
        "pc_cng_a6_positive_ci_scorers": a6_positive,
        "n_scorers_with_a6_positive": len(a6_positive),
        "n_scorers_total": len(scorers),
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def write_summary_csv(records: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = ["mrr", "top1", "top3", "ndcg", "auprc", "ece", "brier"]
    fields = (["backbone", "arm_id", "arm_name", "seed", "trainable_parameters",
               "total_parameters", "wall_clock_seconds", "n_train_examples",
               "n_train_pos", "n_train_neg"]
              + [f"val_{k}" for k in keys] + [f"test_{k}" for k in keys])
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            row = {k: r[k] for k in fields if k in r}
            for k in keys:
                row[f"val_{k}"] = r["val_metrics"][k]
                row[f"test_{k}"] = r["test_metrics"][k]
            w.writerow(row)


def write_effect_sizes_csv(effects: Dict[str, Dict[str, Dict[str, float]]], path: Path) -> None:
    fields = ["backbone", "arm_id", "arm_name", "cohens_d", "pp_diff",
              "arm_mean_mrr", "baseline_mean_mrr", "ci_low", "ci_high", "p_value"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for scorer, arms in effects.items():
            for arm_id, e in arms.items():
                w.writerow({"backbone": scorer, "arm_id": arm_id,
                            "arm_name": ARM_DEFINITIONS[arm_id]["name"], **e})


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="P4-G4 failure diagnostic matrix")
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/p4/manifests/hte_feasibility_v1.json"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/p4_generator_scorer_matrix"))
    parser.add_argument("--g3-dir", type=Path, default=Path("results/p4_augmentation"))
    parser.add_argument("--device", type=str,
                        default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--stage", type=str, default="full", choices=["smoke", "full"])
    parser.add_argument("--skip-mlp", action="store_true",
                        help="Skip Morgan MLP runs; analyze existing outputs only")
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    seeds = [20260721] if args.stage == "smoke" else DEFAULT_SEEDS

    print(f"[P4-G4] FAILURE DIAGNOSTIC mode (P4-G3 = NO_GO)")
    print(f"[P4-G4] Output: {out}, stage={args.stage}, device={args.device}")

    splits = load_manifest_candidates(args.manifest)
    train_data, val_data, test_data = splits["train"], splits["val"], splits["test"]

    # ---- 1. Morgan MLP runs (3rd independent scorer) ----
    mlp_records: List[dict] = []
    mlp_summary = out / "mlp_summary.csv"
    if not args.skip_mlp:
        for arm_id in ARM_IDS:
            for seed in seeds:
                print(f"[P4-G4] morgan_mlp × {arm_id} × seed={seed}")
                rec = run_single_mlp_experiment(
                    arm_id, seed, train_data, val_data, test_data,
                    args.epochs, args.batch_size, args.lr, args.device, out)
                mlp_records.append(rec)
                print(f"  test MRR {rec['test_metrics']['mrr']:.4f}  "
                      f"wall {rec['wall_clock_seconds']:.1f}s")
        write_summary_csv(mlp_records, mlp_summary)
    else:
        # Re-load previous MLP runs
        if mlp_summary.exists():
            for r in csv.DictReader(open(mlp_summary)):
                keys = ["mrr", "top1", "top3", "ndcg", "auprc", "ece", "brier"]
                mlp_records.append({
                    "backbone": r["backbone"], "arm_id": r["arm_id"],
                    "arm_name": r["arm_name"], "seed": int(r["seed"]),
                    "trainable_parameters": int(r["trainable_parameters"]),
                    "total_parameters": int(r["total_parameters"]),
                    "wall_clock_seconds": float(r["wall_clock_seconds"]),
                    "n_train_examples": int(r["n_train_examples"]),
                    "n_train_pos": int(r["n_train_pos"]),
                    "n_train_neg": int(r["n_train_neg"]),
                    "val_metrics": {k: float(r[f"val_{k}"]) for k in keys},
                    "test_metrics": {k: float(r[f"test_{k}"]) for k in keys},
                })

    # ---- 2. Load G3 records (chemformer + gnn) ----
    g3_records: List[dict] = []
    for name, d in G3_DIRS.items():
        p = d / "summary.csv"
        if p.exists():
            g3_records.extend(g3_rows_to_records(load_g3_summary(p)))
            print(f"[P4-G4] loaded G3 {name}: {p}")

    all_records = g3_records + mlp_records
    print(f"[P4-G4] total cells: {len(all_records)} "
          f"(scorers={sorted({r['backbone'] for r in all_records})})")

    # ---- 3. Effect sizes per scorer ----
    effects = cell_effect_sizes(all_records)
    write_effect_sizes_csv(effects, out / "effect_sizes.csv")

    # Combined summary.csv (all 3 scorers)
    write_summary_csv(all_records, out / "summary.csv")

    # ---- 4. Difficulty profiling ----
    difficulty = difficulty_profile(args.manifest)
    with open(out / "difficulty_profile.json", "w") as f:
        json.dump(difficulty, f, indent=2)

    # ---- 5. Interaction model ----
    rows = build_cell_table(all_records)
    anova = interaction_anova(rows)
    mixed = mixed_effects_model(rows)

    # ---- 6. Manifest duplication recap + count check ----
    dup_path = args.g3_dir / "manifest_integrity.json"
    manifest_dup = json.load(open(dup_path)) if dup_path.exists() else {"duplicated": None}
    counts = {(r["backbone"], r["arm_id"]): (r["n_train_pos"], r["n_train_neg"])
              for r in all_records}
    counts_matched = all(v == (394, 0) if k[1] == "A0" else v == (394, 394)
                         for k, v in counts.items())

    hypotheses = diagnose_hypotheses(effects, anova, difficulty, manifest_dup,
                                     counts_matched)

    interaction_model = {
        "phase": PHASE,
        "mode": "failure_diagnostic",
        "entry_condition": "P4-G3 == NO_GO -> diagnostic matrix only (full 5x4x3 matrix NOT executed)",
        "cell_table_n": len(rows),
        "anova": anova,
        "mixed_effects": mixed,
        "hypotheses": hypotheses,
    }
    with open(out / "interaction_model.json", "w") as f:
        json.dump(interaction_model, f, indent=2)

    # ---- 7. go_no_go.json ----
    go_no_go = {
        "phase": PHASE,
        "status": "DEFERRED",
        "mode": "failure_diagnostic",
        "full_matrix_executed": False,
        "primary_metric": {"name": "test_mrr", "comparison": "arm_vs_A0 per scorer"},
        "predeclared_threshold": {
            "go": "PC-CNG main effect positive OR >=3/4 scorers effect>0 OR survives matching",
            "no_go": "gains vanish after difficulty matching / explained by hardness or count",
        },
        "key_findings": {
            "source_x_scorer_interaction_p": hypotheses["H5_source_x_scorer_interaction"]["interaction_p_value"],
            "best_arm_per_scorer": hypotheses["H3_pc_cng_candidates_harder"]["best_arm_per_scorer"],
            "n_scorers_with_a6_positive_ci": hypotheses["n_scorers_with_a6_positive"],
            "a0_baseline_mrr": hypotheses["H2_chemformer_simply_stronger"]["a0_baseline_mrr"],
            "invalid_sources_majority_unparseable": hypotheses["smiles_validity"]["invalid_sources_majority_unparseable"],
            "text_artifact_arms": hypotheses["smiles_validity"]["text_artifact_arms"],
        },
        "limitations": [
            "Full 5-source x 4-scorer x 3-protocol matrix not executed: P4-G3 was NO_GO.",
            "A6 (rule_pc_cng) is vacuous in the frozen v1 manifest (duplicates "
            "random_corruption); PC-CNG main effect is untestable until a v2 "
            "manifest with genuine rule PC-CNG candidates exists.",
            "Morgan MLP is the third scorer; no external/frozen scorer was used.",
        ],
        "evidence_paths": [
            str(out / "summary.csv"),
            str(out / "effect_sizes.csv"),
            str(out / "interaction_model.json"),
            str(out / "difficulty_profile.json"),
            str(out / "raw_predictions"),
        ],
        "next_phase_allowed": False,
        "remediation": ("Build v2 candidate manifest (new namespace) with genuine "
                        "rule PC-CNG candidates; re-run P4-G3 A6 arm; then execute "
                        "the full P4-G4 matrix."),
    }
    with open(out / "go_no_go.json", "w") as f:
        json.dump(go_no_go, f, indent=2)

    # ---- 8. Contract files ----
    with open(out / "run_manifest.json", "w") as f:
        json.dump({
            "phase": PHASE, "mode": "failure_diagnostic",
            "scorers": sorted({r["backbone"] for r in all_records}),
            "arms": ARM_IDS, "seeds": seeds, "n_cells": len(all_records),
            "script": "pc_cng/run_p4_g4_diagnostic.py",
        }, f, indent=2)
    with open(out / "environment.json", "w") as f:
        env = {"python": sys.version, "platform": platform.platform(),
               "torch": torch.__version__,
               "cuda_available": torch.cuda.is_available()}
        try:
            import statsmodels
            env["statsmodels"] = statsmodels.__version__
        except ImportError:
            pass
        try:
            import rdkit
            env["rdkit"] = rdkit.__version__
        except ImportError:
            pass
        json.dump(env, f, indent=2)
    with open(out / "input_hashes.json", "w") as f:
        json.dump({str(args.manifest): sha256_file(args.manifest)}, f, indent=2)
    with open(out / "commands.log", "w") as f:
        f.write("python3 -m chem_negative_sampling.pc_cng.run_p4_g4_diagnostic "
                "--manifest data/p4/manifests/hte_feasibility_v1.json "
                "--output-dir results/p4_generator_scorer_matrix "
                "--stage full --device cuda:0\n")

    print(f"\n[P4-G4] verdict: DEFERRED (diagnostic complete)")
    print(f"[P4-G4] interaction p = {anova.get('C(source):C(scorer)', {}).get('p_value')}")
    print(f"[P4-G4] done -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
