"""P4-G5: Risk-Aware Counterfactual Learning — orchestration.

Two stages::

    # 1. Build risk artifacts (once, deterministic seed 20260723)
    python3 -m pc_cng.run_p4_risk_aware risk-model \
        --manifest data/p4/manifests/hte_feasibility_v2.json \
        --htea-csv data/processed/hitea_full_normalized.csv \
        --output-dir results/p4_risk_aware --device cuda:6

    # 2. Train one / all (method x seed) runs against cached artifacts
    python3 -m pc_cng.run_p4_risk_aware train --method risk_weighted_pairwise --seed 20260721
    python3 -m pc_cng.run_p4_risk_aware train-all --device cuda:6

Data contract:
- Training arm = A6 (gold + rule_pc_cng) from the v2 manifest.
- Backbone = frozen P4-G2 C3 selection (Chemformer-LoRA, 180,737 params).
- Protocol identical to P4-G3: 5 epochs, batch 16, lr 1e-4, early stop
  on val MRR (patience 2), same 10 pre-declared seeds.
- The risk model is calibrated ONLY on observed HTEa train-split rows
  (positive_observed vs negative_observed); synthetic self-labels are
  never used (see pc_cng.models.risk_aware_scorer).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

# ---------------------------------------------------------------------------
# Imports from existing project code
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CNS_ROOT = _REPO_ROOT / "chem_negative_sampling"
if str(_CNS_ROOT) not in sys.path:
    sys.path.insert(0, str(_CNS_ROOT))

from models.pretrained_backbone import (  # noqa: E402
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_VOCAB_PATH,
)
from pc_cng.run_p4_augmentation import (  # noqa: E402
    BackboneConfig,
    build_arm_training_data,
    build_model,
    compute_auprc,
    compute_metrics_from_predictions,
    evaluate_chemformer,
    load_manifest_candidates,
    set_seed,
)
from pc_cng.models.risk_aware_scorer import (  # noqa: E402
    MIN_WEIGHT,
    WEIGHT_COMPONENTS,
    FalseNegativeRiskModel,
    FeasibilityEnsemble,
    RiskFeatureExtractor,
    build_observed_pool,
    compute_sample_weights,
    save_risk_model_manifest,
)
from pc_cng.training.train_risk_aware import (  # noqa: E402
    METHODS,
    compute_loss,
)
from pc_cng.evaluation.false_negative_stress_test import (  # noqa: E402
    build_all_stress_sets,
    collision_sensitivity,
    coverage_risk_curve,
    ece_brier_nll,
    known_positive_metrics,
    near_positive_metrics,
    ood_metrics,
)

PHASE = "P4-G5"
RISK_MODEL_SEED = 20260723
DEFAULT_SEEDS = list(range(20260721, 20260731))  # same 10 seeds as P4-G3
DEFAULT_EPOCHS = 5
DEFAULT_BATCH_SIZE = 16
DEFAULT_LR = 1e-4

# Frozen P4-G2 selection (results/p4_lora_ablation/selected_backbone.json)
C3_CONFIG = BackboneConfig(
    name="chemformer",
    lora_target_patterns=["backbone.encoder_layers.*.self_attn.out_proj"],
    lora_rank=8,
    lora_alpha=16.0,
    lora_dropout=0.0,
)

ABLATION_COMPONENTS = list(WEIGHT_COMPONENTS)


# ---------------------------------------------------------------------------
# Post-hoc temperature scaling (calibration improvement)
# ---------------------------------------------------------------------------

def _fit_temperature(val_logits: List[float], val_labels: List[int]) -> float:
    """Find T* that minimises ECE on the validation set.

    Standard post-hoc calibration: logits / T, search T in [0.5, 5.0].
    Returns T* (1.0 if search fails or no improvement).
    """
    if not val_logits:
        return 1.0
    best_t, best_ece = 1.0, float("inf")
    for t in [round(0.5 + 0.1 * i, 1) for i in range(46)]:  # 0.5 .. 5.0
        probs = [1.0 / (1.0 + math.exp(-max(-30, min(30, l / t)))) for l in val_logits]
        cal = ece_brier_nll(probs, val_labels)
        if cal["ece"] < best_ece:
            best_ece, best_t = cal["ece"], t
    return best_t


def _apply_temperature(logits: List[float], t: float) -> List[float]:
    """Apply temperature scaling to logits -> probabilities."""
    return [1.0 / (1.0 + math.exp(-max(-30, min(30, l / t)))) for l in logits]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_htea_train_rows(htea_csv_path: Path) -> List[Dict[str, Any]]:
    """HTEa train-split rows as feature-extraction entries.

    Observed rows are real reactions: edit_locality anchored at 1.0
    (gold_smiles = own product) and atom_mapping_quality at 1.0.
    """
    rows: List[Dict[str, Any]] = []
    with open(htea_csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("split", "") != "train":
                continue
            prod = (row.get("products") or "").strip()
            if not prod:
                continue
            try:
                y = float(row.get("yield", "") or 0.0)
            except ValueError:
                continue
            rows.append({
                "smiles": prod,
                "gold_smiles": prod,  # locality anchor = 1.0 for observed
                "reaction_family": row.get("reaction_class", "") or "unknown",
                "experimental_group_id": row.get("split_key", ""),
                "atom_mapping_status": "mapped",
                "yield": y,
            })
    return rows


def _gold_smiles_by_group(manifest_path: Path) -> Dict[str, str]:
    import re
    out: Dict[str, str] = {}
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    for group in manifest.get("groups", []):
        gid = group["group_id"]
        for cand in group.get("candidates", []):
            if cand.get("gold_candidate", False):
                out[gid] = re.sub(r":\d+", "", cand.get("candidate_smiles", ""))
    return out


def _experimental_group_id_map(manifest_path: Path) -> Dict[str, str]:
    """Map manifest group_id -> experimental_group_id (HTEa split_key)."""
    out: Dict[str, str] = {}
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    for group in manifest.get("groups", []):
        gid = group["group_id"]
        eid = group.get("experimental_group_id", "") or group.get("parent_reaction_id", "")
        out[gid] = eid
    return out


# ---------------------------------------------------------------------------
# Stage 1: risk model + artifacts
# ---------------------------------------------------------------------------

def stage_risk_model(
    manifest_path: Path,
    htea_csv_path: Path,
    output_dir: Path,
    device: str,
    seed: int = RISK_MODEL_SEED,
    ensemble_members: int = 5,
    ensemble_epochs: int = 20,
) -> Dict[str, Any]:
    """Build observed pool, ensemble, FNR model, weights, stress sets."""
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # --- Observed pool (HTEa train split only; val/test excluded by split_key)
    with open(htea_csv_path, "r", encoding="utf-8") as f:
        excluded: frozenset = frozenset(
            row["split_key"] for row in csv.DictReader(f)
            if row.get("split", "") != "train" and row.get("split_key")
        )
    pool = build_observed_pool(htea_csv_path, excluded, seed=seed)
    print(f"[risk-model] observed pool: {len(pool.pos_smiles)} pos / "
          f"{len(pool.neg_smiles)} neg (HTEa train split only; "
          f"{len(excluded)} val/test split_keys excluded)")

    # --- Feasibility ensemble on observed data
    ensemble = FeasibilityEnsemble(n_members=ensemble_members, seed=seed)
    ens_losses = ensemble.fit(
        pool.pos_smiles, pool.neg_smiles, epochs=ensemble_epochs, device=device,
    )
    print(f"[risk-model] ensemble fitted: {ensemble_members} members, "
          f"mean final loss {statistics.mean(ens_losses):.4f}")

    extractor = RiskFeatureExtractor(pool, ensemble, seed=seed, device=device)

    # --- Features for observed rows (calibration set)
    train_rows = _load_htea_train_rows(htea_csv_path)
    pos_rows = [r for r in train_rows if r["yield"] > 0]
    neg_rows = [r for r in train_rows if r["yield"] == 0]
    # Match the pool's negative downsampling: cap negatives at 2x positives
    rng = random.Random(seed)
    if len(neg_rows) > 2 * len(pos_rows):
        neg_rows = rng.sample(neg_rows, 2 * len(pos_rows))
    pos_feats = extractor.extract_batch(pos_rows)
    neg_feats = extractor.extract_batch(neg_rows)
    print(f"[risk-model] calibration features: {len(pos_feats)} pos / {len(neg_feats)} neg")

    # --- FNR model (observed data only)
    fnr_model = FalseNegativeRiskModel()
    calib = fnr_model.fit(pos_feats, neg_feats, seed=seed)
    print(f"[risk-model] FNR model: logloss={calib['train_logloss']:.4f} "
          f"auroc={calib['train_auroc']:.4f} n={calib['n_train']}")

    # --- Features + weights for ALL manifest candidates
    splits = load_manifest_candidates(manifest_path)
    gold_map = _gold_smiles_by_group(manifest_path)
    exp_gid_map = _experimental_group_id_map(manifest_path)
    all_cands = splits["train"] + splits["val"] + splits["test"]
    for c in all_cands:
        c["gold_smiles"] = gold_map.get(c["group_id"], "")
        # Use the manifest's experimental_group_id (HTEa split_key) so
        # synthetic candidates inherit their parent reaction's experimental
        # support.  Previously this was group_id ("hte_xxx") which never
        # matched any HTEa split_key, giving experimental_support = 0.
        c["experimental_group_id"] = exp_gid_map.get(c["group_id"], "")
    cand_feats = extractor.extract_batch(all_cands)
    cand_fnr = fnr_model.predict_fnr(cand_feats)
    kpc = [bool(c.get("known_positive_collision")) for c in all_cands]
    weights = compute_sample_weights(cand_feats, cand_fnr, known_positive_collision=kpc)

    artifacts: Dict[str, Any] = {"schema": "p4_g5_risk_artifacts/v1", "candidates": {}}
    for c, f, r, w in zip(all_cands, cand_feats, cand_fnr, weights):
        artifacts["candidates"][c["candidate_id"]] = {
            "candidate_id": c["candidate_id"],
            "group_id": c["group_id"],
            "split": c["split"],
            "candidate_source": c["candidate_source"],
            "gold_candidate": c["gold_candidate"],
            "features": {k: round(float(v), 6) for k, v in f.items()},
            "false_negative_risk": round(float(r), 6),
            "weight": {k: round(float(v), 6) for k, v in w.items()},
        }

    # PU prior: mean FNR over train-split rule_pc_cng negatives, capped at 0.3
    # (previous cap was 0.5, which saturated when FNR was inverted on v2
    # near-positive counterfactuals; a conservative prior prevents nnPU
    # from treating too many negatives as hidden positives).
    train_a6_fnr = [
        rec["false_negative_risk"]
        for rec in artifacts["candidates"].values()
        if rec["split"] == "train" and rec["candidate_source"] == "rule_pc_cng"
    ]
    pu_prior = min(0.3, max(0.01, statistics.mean(train_a6_fnr) if train_a6_fnr else 0.1))
    artifacts["pu_prior"] = round(pu_prior, 6)
    artifacts["n_train_a6_negatives"] = len(train_a6_fnr)

    art_path = output_dir / "risk_artifacts.json"
    with open(art_path, "w") as f:
        json.dump(artifacts, f)
    print(f"[risk-model] artifacts: {len(artifacts['candidates'])} candidates, "
          f"pu_prior={pu_prior:.4f}")

    # --- Stress sets
    fnr_by_candidate = {
        cid: rec["false_negative_risk"]
        for cid, rec in artifacts["candidates"].items()
    }
    stress_sets = build_all_stress_sets(
        manifest_path, htea_csv_path, fnr_by_candidate=fnr_by_candidate, seed=seed,
    )
    with open(output_dir / "stress_sets.json", "w") as f:
        json.dump(stress_sets, f)
    print(f"[risk-model] stress sets: known_positive={len(stress_sets['known_positive'])} "
          f"near_positive={len(stress_sets['near_positive'])} "
          f"ood_family={len(stress_sets['ood_family'])} "
          f"collisions={len(stress_sets['collision_candidate_ids'])}")

    # --- Risk model manifest
    save_risk_model_manifest(
        output_dir / "risk_model_manifest.json",
        fnr_model,
        ensemble_meta={
            "n_members": ensemble_members,
            "epochs": ensemble_epochs,
            "member_final_losses": [round(v, 6) for v in ens_losses],
            "backbone": "bagged_mlp_morgan1024",
        },
        pool_meta={
            "n_pos": len(pool.pos_smiles),
            "n_neg": len(pool.neg_smiles),
            "n_families": len(pool.family_counts),
            "source": str(htea_csv_path),
            "split_restriction": "train only (val/test excluded via split column)",
        },
        calibration={k: round(float(v), 6) for k, v in calib.items()},
        input_hashes={
            "manifest_sha256": _sha256_file(manifest_path),
            "htea_csv_sha256": _sha256_file(htea_csv_path),
        },
    )
    print(f"[risk-model] done in {time.time() - t0:.1f}s -> {output_dir}")
    return artifacts


# ---------------------------------------------------------------------------
# Stage 2: training
# ---------------------------------------------------------------------------

def load_artifacts(output_dir: Path) -> Dict[str, Any]:
    with open(output_dir / "risk_artifacts.json", "r") as f:
        return json.load(f)


def attach_weights(
    train_data: List[dict],
    artifacts: Dict[str, Any],
    ablate: Sequence[str] = (),
) -> None:
    """Attach per-example sample_weight in-place.

    Gold -> 1.0. Negatives -> product of non-ablated weight components
    (floored at MIN_WEIGHT), from precomputed artifacts.
    """
    ablate_set = set(ablate)
    for d in train_data:
        if d["label"] == 1:
            d["sample_weight"] = 1.0
            continue
        rec = artifacts["candidates"].get(d["candidate_id"])
        if rec is None:
            raise KeyError(f"missing risk artifact for {d['candidate_id']}")
        w = 1.0
        for comp in WEIGHT_COMPONENTS:
            if comp not in ablate_set:
                w *= rec["weight"][comp]
        d["sample_weight"] = max(w, MIN_WEIGHT)


def _group_batches(
    train_data: List[dict],
    groups_per_batch: int,
    rng: random.Random,
) -> List[List[dict]]:
    """Group-aware batches: each batch holds whole groups (gold + neg)."""
    by_group: Dict[str, List[dict]] = {}
    for d in train_data:
        by_group.setdefault(d["group_id"], []).append(d)
    groups = list(by_group.values())
    rng.shuffle(groups)
    batches = []
    for i in range(0, len(groups), groups_per_batch):
        batch: List[dict] = []
        for g in groups[i:i + groups_per_batch]:
            batch.extend(g)
        batches.append(batch)
    return batches


def train_epoch(
    model,
    tokenizer,
    train_data: List[dict],
    optimizer: torch.optim.Optimizer,
    device: str,
    batch_size: int,
    epoch: int,
    seed: int,
    method: str,
    pu_prior: float,
) -> float:
    """One epoch with the P4-G5 loss dispatch. Returns mean loss."""
    model.train()
    rng = random.Random(seed + epoch * 1000)
    group_methods = ("risk_weighted_pairwise", "risk_weighted_infonce")
    if method in group_methods:
        batches = _group_batches(train_data, max(1, batch_size // 2), rng)
    else:
        indices = list(range(len(train_data)))
        rng.shuffle(indices)
        batches = [
            [train_data[j] for j in indices[i:i + batch_size]]
            for i in range(0, len(indices), batch_size)
        ]

    total, nb = 0.0, 0
    for batch in batches:
        if method in ("pu_nnpu",):
            # nnPU needs at least one P and one U per batch
            if not any(b["label"] == 1 for b in batch) or not any(b["label"] == 0 for b in batch):
                continue
        smiles_list = [b["smiles"] for b in batch]
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.float32, device=device)
        weights = torch.tensor([b["sample_weight"] for b in batch], dtype=torch.float32, device=device)
        group_ids = [b["group_id"] for b in batch]

        token_ids, attn_mask = tokenizer.batch_encode(smiles_list)
        token_ids = token_ids.to(device)
        attn_mask = attn_mask.to(device)
        logits = model(token_ids, attn_mask)

        try:
            loss = compute_loss(
                method, logits, labels,
                group_ids=group_ids, weights=weights, pu_prior=pu_prior,
            )
        except ValueError:
            continue  # e.g. batch without a complete (pos, neg) group

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0,
        )
        optimizer.step()
        total += loss.item()
        nb += 1
    return total / max(nb, 1)


def _to_eval_schema(cands: Sequence[Dict[str, Any]], prefix: str) -> List[dict]:
    """Adapt stress-set candidates to the evaluate_chemformer schema."""
    out = []
    for i, c in enumerate(cands):
        out.append({
            "group_id": c.get("group_id", f"{prefix}_{i}"),
            "candidate_id": c.get("candidate_id", f"{prefix}_{i}"),
            "smiles": c["smiles"],
            "label": c.get("label", 0),
            "candidate_source": prefix,
        })
    return out


def run_single(
    method: str,
    seed: int,
    ablate: Sequence[str],
    manifest_path: Path,
    artifacts: Dict[str, Any],
    stress_sets: Dict[str, Any],
    output_dir: Path,
    checkpoint_path: Path,
    vocab_path: Path,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    fixed_forward_manifest: Optional[Path] = None,
) -> dict:
    """Train one (method, seed, ablate) run and evaluate everything."""
    set_seed(seed)
    t0 = time.time()
    pu_prior = float(artifacts.get("pu_prior", 0.1))

    splits = load_manifest_candidates(manifest_path)
    train_data = build_arm_training_data(splits["train"], "A6")
    attach_weights(train_data, artifacts, ablate)
    val_data, test_data = splits["val"], splits["test"]
    n_pos = sum(1 for d in train_data if d["label"] == 1)

    model, tokenizer, trainable = build_model(C3_CONFIG, checkpoint_path, vocab_path, device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    best_val_mrr, best_state, best_epoch = -1.0, None, 0
    patience, bad = 2, 0
    val_mrr_per_epoch: List[float] = []
    for epoch in range(epochs):
        loss = train_epoch(
            model, tokenizer, train_data, optimizer, device,
            batch_size, epoch, seed, method, pu_prior,
        )
        val_preds = evaluate_chemformer(model, tokenizer, val_data, device)
        val_mrr = compute_metrics_from_predictions(val_preds)["mrr"]
        val_mrr_per_epoch.append(round(val_mrr, 6))
        print(f"  [{method} s{seed}] epoch {epoch}: loss={loss:.4f} val_mrr={val_mrr:.4f}")
        if val_mrr > best_val_mrr:
            best_val_mrr, best_epoch = val_mrr, epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)

    # --- HTE test/val eval with post-hoc temperature scaling
    test_preds = evaluate_chemformer(model, tokenizer, test_data, device)
    val_preds = evaluate_chemformer(model, tokenizer, val_data, device)

    # Fit temperature on val logits, apply to both val and test
    val_logits = [float(p["score"]) for p in val_preds]
    val_labels_raw = [p["label"] for p in val_preds]
    temp_t = _fit_temperature(val_logits, val_labels_raw)
    test_logits = [float(p["score"]) for p in test_preds]
    test_probs_ts = _apply_temperature(test_logits, temp_t)
    val_probs_ts = _apply_temperature(val_logits, temp_t)

    test_metrics = compute_metrics_from_predictions(test_preds)
    val_metrics = compute_metrics_from_predictions(val_preds)
    test_cal = ece_brier_nll(test_probs_ts, [p["label"] for p in test_preds])
    val_cal = ece_brier_nll(val_probs_ts, [p["label"] for p in val_preds])

    # --- Stress tests
    kp_groups_scored = []
    for g in stress_sets["known_positive"]:
        eval_rows = _to_eval_schema(g["candidates"], "kp")
        preds = evaluate_chemformer(model, tokenizer, eval_rows, device, batch_size=64)
        merged = []
        for p, c in zip(preds, g["candidates"]):
            merged.append({**c, "score": p["score"]})
        kp_groups_scored.append(merged)
    kp = known_positive_metrics(kp_groups_scored)

    np_rows = _to_eval_schema(stress_sets["near_positive"], "np")
    np_scored: List[Dict[str, Any]] = []
    if np_rows:
        preds = evaluate_chemformer(model, tokenizer, np_rows, device, batch_size=64)
        by_id = {c.get("candidate_id"): c for c in stress_sets["near_positive"]}
        for p in preds:
            src = by_id.get(p["candidate_id"], {})
            np_scored.append({**src, "score": p["score"]})
    np_metrics = near_positive_metrics(np_scored)

    ood_rows = _to_eval_schema(stress_sets["ood_family"], "ood")
    ood_scored: List[Dict[str, Any]] = []
    if ood_rows:
        preds = evaluate_chemformer(model, tokenizer, ood_rows, device, batch_size=64)
        by_id = {r["candidate_id"]: r for r in ood_rows}
        for p in preds:
            ood_scored.append({"score": p["score"], "label": by_id[p["candidate_id"]]["label"]})
    ood = ood_metrics(ood_scored)

    coll = collision_sensitivity(test_preds, stress_sets["collision_candidate_ids"])

    # --- Selective prediction on test (uncertainty = FNR, not ensemble variance)
    # FNR is a better uncertainty signal than ensemble_variance (which is
    # near-zero for feasible-looking candidates).  High FNR = high uncertainty
    # about whether the candidate is truly negative -> should abstain.
    probs, labels, unc = [], [], []
    for p, prob_ts in zip(test_preds, test_probs_ts):
        rec = artifacts["candidates"].get(p["candidate_id"], {})
        probs.append(prob_ts)
        labels.append(p["label"])
        unc.append(float(rec.get("false_negative_risk", 0.5)))
    cr = coverage_risk_curve(probs, labels, unc)

    # --- Fixed-forward zero-shot eval (skipped for ablation runs)
    fixed_mrr: Optional[float] = None
    if fixed_forward_manifest is not None and fixed_forward_manifest.exists():
        ff_splits = load_manifest_candidates(fixed_forward_manifest)
        ff_test = ff_splits["test"]
        if ff_test:
            ff_preds = evaluate_chemformer(model, tokenizer, ff_test, device, batch_size=64)
            fixed_mrr = round(compute_metrics_from_predictions(ff_preds)["mrr"], 6)

    # --- Training stability: max epoch-over-epoch val-MRR drop
    max_drop = 0.0
    for a, b in zip(val_mrr_per_epoch, val_mrr_per_epoch[1:]):
        max_drop = max(max_drop, a - b)

    result = {
        "phase": PHASE,
        "method": method,
        "seed": seed,
        "ablate": list(ablate),
        "trainable_parameters": trainable,
        "n_train": len(train_data),
        "n_train_pos": n_pos,
        "pu_prior": pu_prior,
        "best_epoch": best_epoch,
        "val_mrr_per_epoch": val_mrr_per_epoch,
        "wall_clock_seconds": round(time.time() - t0, 2),
        "val_metrics": {
            "mrr": round(val_metrics["mrr"], 6),
            "top1": round(val_metrics["top1"], 6),
            "auprc": round(compute_auprc(val_preds), 6),
            "ece": round(val_cal["ece"], 6),
            "brier": round(val_cal["brier"], 6),
            "nll": round(val_cal["nll"], 6),
        },
        "test_metrics": {
            "mrr": round(test_metrics["mrr"], 6),
            "top1": round(test_metrics["top1"], 6),
            "auprc": round(compute_auprc(test_preds), 6),
            "ece": round(test_cal["ece"], 6),
            "brier": round(test_cal["brier"], 6),
            "nll": round(test_cal["nll"], 6),
        },
        "fixed_forward_test_mrr": fixed_mrr,
        "stress": {
            "known_positive": {k: round(v, 6) if isinstance(v, float) else v for k, v in kp.items()},
            "near_positive": {k: round(v, 6) if isinstance(v, float) else v for k, v in np_metrics.items()},
            "ood_family": {k: round(v, 6) if isinstance(v, float) else v for k, v in ood.items()},
            "collision_sensitivity": {k: round(v, 6) if isinstance(v, float) else v for k, v in coll.items()},
        },
        "selective": {
            "risk_at_0p8": round(cr["risk_at_0p8"], 6),
            "auc": round(cr["auc"], 6),
            "coverage": cr["coverage"],
            "risk": [round(v, 6) for v in cr["risk"]],
        },
        "training_stability": {"max_val_mrr_drop": round(max_drop, 6)},
        "temperature": round(temp_t, 4),
    }

    # --- Persist
    tag = f"seed_{seed}"
    if ablate:
        run_dir = output_dir / "ablation" / "_".join(sorted(ablate))
    else:
        run_dir = output_dir / "runs" / method
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / f"{tag}.json", "w") as f:
        json.dump(result, f, indent=2)
    pred_dir = output_dir / "raw_predictions" / (f"ablation_{'_'.join(sorted(ablate))}" if ablate else method)
    pred_dir.mkdir(parents=True, exist_ok=True)
    with open(pred_dir / f"{tag}_test.json", "w") as f:
        json.dump(test_preds, f)
    with open(pred_dir / f"{tag}_val.json", "w") as f:
        json.dump(val_preds, f)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="P4-G5 Risk-Aware Counterfactual Learning")
    sub = parser.add_subparsers(dest="stage", required=True)

    p_rm = sub.add_parser("risk-model", help="Build risk artifacts (once)")
    p_rm.add_argument("--manifest", type=Path, default=Path("data/p4/manifests/hte_feasibility_v2.json"))
    p_rm.add_argument("--htea-csv", type=Path, default=Path("data/processed/hitea_full_normalized.csv"))
    p_rm.add_argument("--output-dir", type=Path, default=Path("results/p4_risk_aware"))
    p_rm.add_argument("--device", type=str, default="cuda:6")
    p_rm.add_argument("--ensemble-epochs", type=int, default=20)

    p_tr = sub.add_parser("train", help="Train one (method, seed[, ablate])")
    p_tr.add_argument("--manifest", type=Path, default=Path("data/p4/manifests/hte_feasibility_v2.json"))
    p_tr.add_argument("--output-dir", type=Path, default=Path("results/p4_risk_aware"))
    p_tr.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    p_tr.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB_PATH)
    p_tr.add_argument("--device", type=str, default="cuda:6")
    p_tr.add_argument("--method", type=str, required=True, choices=METHODS)
    p_tr.add_argument("--seed", type=int, required=True)
    p_tr.add_argument("--ablate", type=str, default="",
                      help="comma-separated weight components to ablate")
    p_tr.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p_tr.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p_tr.add_argument("--lr", type=float, default=DEFAULT_LR)
    p_tr.add_argument("--fixed-forward-manifest", type=Path,
                      default=Path("data/p4/manifests/fixed_forward_candidates_v1.json"))

    p_ta = sub.add_parser("train-all", help="Full matrix: 5 methods x 10 seeds + ablations")
    p_ta.add_argument("--manifest", type=Path, default=Path("data/p4/manifests/hte_feasibility_v2.json"))
    p_ta.add_argument("--output-dir", type=Path, default=Path("results/p4_risk_aware"))
    p_ta.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    p_ta.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB_PATH)
    p_ta.add_argument("--device", type=str, default="cuda:6")
    p_ta.add_argument("--methods", type=str, default=",".join(METHODS))
    p_ta.add_argument("--seeds", type=str, default=",".join(str(s) for s in DEFAULT_SEEDS))
    p_ta.add_argument("--with-ablation", action="store_true",
                      help="also run 4 weight-component ablations x seeds for risk_weighted_pairwise")
    p_ta.add_argument("--ablation-seeds", type=str, default="",
                      help="subset of seeds for ablation (default: all --seeds)")
    p_ta.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p_ta.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p_ta.add_argument("--lr", type=float, default=DEFAULT_LR)
    p_ta.add_argument("--fixed-forward-manifest", type=Path,
                      default=Path("data/p4/manifests/fixed_forward_candidates_v1.json"))
    p_ta.add_argument("--skip-existing", action="store_true", default=True)

    args = parser.parse_args()

    if args.stage == "risk-model":
        stage_risk_model(
            args.manifest, args.htea_csv, args.output_dir, args.device,
            ensemble_epochs=args.ensemble_epochs,
        )
        return

    # train / train-all share artifact + stress loading
    artifacts = load_artifacts(args.output_dir)
    with open(args.output_dir / "stress_sets.json", "r") as f:
        stress_sets = json.load(f)

    if args.stage == "train":
        ablate = [a for a in args.ablate.split(",") if a] if args.ablate else []
        run_single(
            args.method, args.seed, ablate, args.manifest, artifacts, stress_sets,
            args.output_dir, args.checkpoint, args.vocab, args.device,
            args.epochs, args.batch_size, args.lr,
            fixed_forward_manifest=None if ablate else args.fixed_forward_manifest,
        )
        return

    # train-all
    methods = [m for m in args.methods.split(",") if m]
    seeds = [int(s) for s in args.seeds.split(",") if s]
    abl_seeds = ([int(s) for s in args.ablation_seeds.split(",") if s]
                 if args.ablation_seeds else seeds)
    for method in methods:
        for seed in seeds:
            out = args.output_dir / "runs" / method / f"seed_{seed}.json"
            if args.skip_existing and out.exists():
                print(f"[skip] {method} seed {seed} (exists)")
                continue
            print(f"=== {method} seed {seed} ===")
            run_single(
                method, seed, (), args.manifest, artifacts, stress_sets,
                args.output_dir, args.checkpoint, args.vocab, args.device,
                args.epochs, args.batch_size, args.lr,
                fixed_forward_manifest=args.fixed_forward_manifest,
            )
    if args.with_ablation:
        for comp in ABLATION_COMPONENTS:
            for seed in abl_seeds:
                out = args.output_dir / "ablation" / comp / f"seed_{seed}.json"
                if args.skip_existing and out.exists():
                    print(f"[skip] ablation {comp} seed {seed} (exists)")
                    continue
                print(f"=== ablation {comp} seed {seed} (risk_weighted_pairwise) ===")
                run_single(
                    "risk_weighted_pairwise", seed, (comp,), args.manifest,
                    artifacts, stress_sets, args.output_dir, args.checkpoint,
                    args.vocab, args.device, args.epochs, args.batch_size, args.lr,
                    fixed_forward_manifest=None,
                )


if __name__ == "__main__":
    main()
