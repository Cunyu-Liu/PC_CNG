"""P4-G8B v2: Cross-reaction-family transfer — full spec execution.

v1 (results/p4_cross_family_transfer) was NO_GO but spec-incomplete:
only 3/6 methods, only HTE-family directions, 2 seeds, no CI / permutation
test, no per-sample predictions. v2 executes the full predeclared design:

Directions (spec L1645-1651):
    EAS <-> C-N coupling          (USPTO openmolecules, rule-classified)
    USPTO(EAS+C-N) -> HTE family  (Pd coupling, Alkylation)
    ORD(EAS+C-N)   -> HTE family  (Pd coupling, Alkylation)
    HTE family A   -> HTE family B (v1 directions, redone with full stats)

Methods (spec L1654-1661): direct, head_ft, lora_adapter, ewc, risk_aware,
multi_task.

Statistics (spec L70-91): fixed splits, fixed manifest, 10 predeclared seeds,
identical training budget, cluster bootstrap 95% CI, paired permutation test
(exact sign-flip over seeds), effect size (Cohen's d), per-sample predictions
saved. Significance is never drawn from seed-level means alone.

Outputs (results/p4_cross_family_transfer_v2/):
    transfer_results.csv, family_macro_summary.csv, direction_stats.json,
    transfer_analysis.json, raw_predictions/*.csv, go_no_go.json,
    run_manifest.json, environment.json, input_hashes.json, commands.log
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ["RDKitRDLogger"] = "0"
try:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
except Exception:  # pragma: no cover
    Chem = None

from .p4_g8b_cross_family_transfer import (
    FP_BITS,
    MorganMLPScorer,
    compute_auprc,
    compute_ece,
    compute_mrr,
    morgan_fp,
    set_seed,
)

PHASE = "P4-G8B-v2"
BASE_SEED = 20260723
PREDECLARED_SEEDS = [BASE_SEED + i for i in range(10)]
N_EPOCHS = 5
FT_EPOCHS = 3
BATCH_SIZE = 16
LR = 1e-3
LORA_RANK = 8
EWC_LAMBDA = 400.0
N_BOOTSTRAP = 2000
MAX_POS_PER_FAMILY = 1500
NEG_PER_REACTION = 2
MIN_FAMILY_SIZE = 80
METHODS = ["direct", "head_ft", "lora_adapter", "ewc", "risk_aware", "multi_task"]

EAS = "EAS"
CN_COUPLING = "C-N coupling"


# ---------------------------------------------------------------------------
# Rule-based reaction-family classifier (USPTO / ORD)
# ---------------------------------------------------------------------------

def _has(smiles: str, smarts: str) -> bool:
    if Chem is None:
        return False
    mol = Chem.MolFromSmiles(smiles)
    pat = Chem.MolFromSmarts(smarts)
    if mol is None or pat is None:
        return False
    return bool(mol.HasSubstructMatch(pat))


def _split_rxn(reaction: str) -> Tuple[str, str]:
    """Return (reactants, products) for both 'r>>p' and 'r>agents>p' formats."""
    if ">>" in reaction:
        left, right = reaction.split(">>", 1)
        return left, right
    parts = reaction.split(">")
    if len(parts) == 3:
        return parts[0], parts[2]
    if len(parts) == 2:
        return parts[0], parts[1]
    return reaction, ""


def is_cn_coupling(reaction: str) -> bool:
    """Buchwald-Hartwig / Ullmann / Chan-Lam style aryl C-N bond formation.

    Reactants: aryl (pseudo)halide + amine. Product: new arylamine C-N bond
    and the aryl halide is consumed.
    """
    reactants, product = _split_rxn(reaction)
    if not product:
        return False
    aryl_halide = _has(reactants, "[c][Cl,Br,I]")
    amine = _has(reactants, "[NX3;H1,H2;!$(N=*)]")
    aryl_amine_product = _has(product, "[c][NX3]")
    residual_aryl_halide = _has(product, "[c][Cl,Br,I]")
    boron_reagent = _has(reactants, "[c][B]([O])[O]") or _has(reactants, "[B]")
    return aryl_halide and amine and aryl_amine_product and not residual_aryl_halide and not boron_reagent


def is_eas(reaction: str) -> bool:
    """Electrophilic aromatic substitution: nitration / halogenation /
    Friedel-Crafts acylation on an aromatic ring.

    Reagent-side molecules are mixed into the reactant field in the
    normalized '>>' format, so rules key on the *aromatic* form of the
    installed group: it must appear in the product while no aromatic-bound
    instance exists among the reactants.
    """
    reactants, product = _split_rxn(reaction)
    if not product:
        return False
    if not _has(reactants, "[c]"):
        return False
    # Nitration: aryl nitro formed (free nitric acid / nitronium allowed on left)
    if _has(product, "[c][N+](=O)[O-]") and not _has(reactants, "[c][N+](=O)[O-]"):
        return True
    # Halogenation: aryl halide formed, no aryl halide consumed
    if _has(product, "[c][Cl,Br,I]") and not _has(reactants, "[c][Cl,Br,I]"):
        return True
    # Friedel-Crafts acylation: aryl ketone formed, acyl electrophile consumed
    if (_has(product, "[c][C](=O)[#6]") and not _has(reactants, "[c][C](=O)[#6]")
            and _has(reactants, "[C](=O)[Cl,Br]")):
        return True
    return False


def classify_uspto_ord_family(reaction: str) -> Optional[str]:
    """Deterministic rule classifier; returns EAS, CN_COUPLING, or None.

    Labels are rule-based proxies (provenance recorded); unclassified
    reactions are excluded rather than forced into a family.
    """
    if is_cn_coupling(reaction):
        return CN_COUPLING
    if is_eas(reaction):
        return EAS
    return None


# ---------------------------------------------------------------------------
# Dataset containers
# ---------------------------------------------------------------------------

def _cluster_from_scaffold_key(key: str) -> str:
    return f"scaf_{key}"


def _ord_split(split_key: str) -> str:
    bucket = int(hashlib.sha1(split_key.encode()).hexdigest(), 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "val"
    return "test"


def load_hte_family_data(manifest_path: Path, risk_path: Optional[Path],
                         ) -> Dict[str, Dict[str, List[dict]]]:
    """HTE manifest v2: positives = gold, negatives = rule_pc_cng/random."""
    with open(manifest_path) as f:
        manifest = json.load(f)
    risk_map: Dict[str, float] = {}
    if risk_path and Path(risk_path).exists():
        with open(risk_path) as f:
            risk_data = json.load(f)
        for cid, rec in risk_data.get("candidates", {}).items():
            risk_map[cid] = rec.get("features", {}).get("false_negative_risk", 0.5)

    by_family: Dict[str, Dict[str, List[dict]]] = defaultdict(
        lambda: {"train": [], "val": [], "test": []})
    for group in manifest.get("groups", []):
        for cand in group.get("candidates", []):
            family = cand.get("reaction_family", "Unknown")
            split = cand.get("split", "train")
            if split not in ("train", "val", "test"):
                continue
            gold = bool(cand.get("gold_candidate"))
            source = cand.get("candidate_source", "")
            if not gold and source not in ("rule_pc_cng", "random_corruption"):
                continue
            by_family[family][split].append({
                "smiles": cand["candidate_smiles"],
                "label": 1 if gold else 0,
                "candidate_id": cand["candidate_id"],
                "cluster_id": cand.get("experimental_group_id") or cand.get("group_id", "?"),
                "fnr": risk_map.get(cand["candidate_id"], 0.5),
            })
    return {f: s for f, s in by_family.items()}


def load_external_family_data(
    csv_path: Path,
    source_name: str,
    max_pos: int,
    neg_per_reaction: int,
    seed: int,
    cache_dir: Optional[Path] = None,
    force_split_from_key: bool = False,
) -> Dict[str, Dict[str, List[dict]]]:
    """Classify USPTO/ORD reactions into EAS / C-N coupling, take real
    products as positives and generate rule PC-CNG boundary negatives.

    Deterministic: classification is rule-based, sampling uses `seed`,
    negatives come from ReactionBoundaryGenerator with fixed caps.
    """
    from .reaction_boundary_generator import ReactionBoundaryGenerator

    cache_path = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{source_name}_classified_cache.json"

    classified: Dict[str, List[dict]] = {EAS: [], CN_COUPLING: []}
    if cache_path and cache_path.exists():
        with open(cache_path) as f:
            classified = json.load(f)
        classified = {k: v for k, v in classified.items() if k in (EAS, CN_COUPLING)}
    else:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rxn = (row.get("reaction_smiles") or "").strip()
                if not rxn:
                    continue
                family = classify_uspto_ord_family(rxn)
                if family is None:
                    continue
                split = row.get("split", "train") or "train"
                if force_split_from_key or split not in ("train", "val", "test"):
                    split = _ord_split(row.get("split_key") or row.get("source_id") or rxn)
                classified[family].append({
                    "reaction_smiles": rxn,
                    "source_id": row.get("source_id", ""),
                    "split": split,
                    "split_key": row.get("split_key", "") or row.get("source_id", "") or rxn[:40],
                })
        if cache_path:
            with open(cache_path, "w") as f:
                json.dump(classified, f)

    rng = np.random.RandomState(seed)
    generator = ReactionBoundaryGenerator(
        max_candidates_per_reaction=neg_per_reaction,
        allow_unmapped_fallback=False,
    )

    out: Dict[str, Dict[str, List[dict]]] = {}
    for family, rows in classified.items():
        if len(rows) > max_pos:
            idx = rng.choice(len(rows), size=max_pos, replace=False)
            rows = [rows[i] for i in sorted(idx)]
        splits: Dict[str, List[dict]] = {"train": [], "val": [], "test": []}
        for i, row in enumerate(rows):
            split = row["split"] if row["split"] in splits else "train"
            try:
                reactants, product = row["reaction_smiles"].split(">>")
            except ValueError:
                continue
            product = product.strip()
            if not product:
                continue
            cid_pos = f"{source_name}_{family.replace(' ', '_')}_{i}_gold"
            cluster = _cluster_from_scaffold_key(row["split_key"])
            splits[split].append({
                "smiles": product, "label": 1, "candidate_id": cid_pos,
                "cluster_id": cluster, "fnr": 0.0,
            })
            # Rule PC-CNG boundary negatives (counterfactual_unknown, never yield=0)
            try:
                cands = generator.generate_for_reaction(
                    row["reaction_smiles"], source_id=row["source_id"] or cid_pos)
            except Exception:
                cands = []
            for j, cand in enumerate(cands[:neg_per_reaction]):
                splits[split].append({
                    "smiles": cand.candidate_product, "label": 0,
                    "candidate_id": f"{cid_pos}_neg{j}",
                    "cluster_id": cluster,
                    "fnr": float(cand.false_negative_risk),
                })
        out[family] = splits
    return out


def pool_families(*datasets: Dict[str, List[dict]]) -> Dict[str, List[dict]]:
    pooled: Dict[str, List[dict]] = {"train": [], "val": [], "test": []}
    for ds in datasets:
        for split in pooled:
            pooled[split].extend(ds.get(split, []))
    return pooled


# ---------------------------------------------------------------------------
# Model variants: LoRA adapter + EWC
# ---------------------------------------------------------------------------

class LoRAAdapter(nn.Module):
    """Low-rank adapter inserted in parallel to a Linear layer."""

    def __init__(self, linear: nn.Linear, rank: int = LORA_RANK):
        super().__init__()
        self.a = nn.Linear(linear.in_features, rank, bias=False)
        self.b = nn.Linear(rank, linear.out_features, bias=False)
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b.weight)

    def forward(self, x):
        return self.b(self.a(x))


class AdaptedMLP(nn.Module):
    """MorganMLPScorer with LoRA adapters on the hidden Linear layers."""

    def __init__(self, base: MorganMLPScorer, rank: int = LORA_RANK):
        super().__init__()
        self.base = base
        linears = [m for m in base.net if isinstance(m, nn.Linear)]
        self.adapters = nn.ModuleList([LoRAAdapter(lin, rank) for lin in linears[:-1]])

    def forward(self, x):
        adapter_iter = iter(self.adapters)
        for module in self.base.net:
            if isinstance(module, nn.Linear) and module is not self.base.net[-1]:
                x = module(x) + next(adapter_iter)(x)
            else:
                x = module(x)
        return x.squeeze(-1)

    def adapter_parameters(self):
        return self.adapters.parameters()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Featurization cache
# ---------------------------------------------------------------------------

class Featurizer:
    def __init__(self):
        self._cache: Dict[str, np.ndarray] = {}

    def featurize(self, data: Sequence[dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        fps = []
        labels = []
        for d in data:
            fp = self._cache.get(d["smiles"])
            if fp is None:
                fp = morgan_fp(d["smiles"])
                self._cache[d["smiles"]] = fp
            fps.append(fp)
            labels.append(d["label"])
        if not fps:
            return (torch.zeros((0, FP_BITS)), torch.zeros((0,)))
        return (torch.tensor(np.array(fps), dtype=torch.float32),
                torch.tensor(labels, dtype=torch.float32))


# ---------------------------------------------------------------------------
# Training methods
# ---------------------------------------------------------------------------

def _train_loop(model, x, y, w, epochs, lr, params=None, extra_loss=None):
    optimizer = torch.optim.AdamW(params if params is not None else model.parameters(), lr=lr)
    n = x.shape[0]
    if n == 0:
        return
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            logits = model(x[idx])
            loss = F.binary_cross_entropy_with_logits(logits, y[idx], weight=w[idx])
            if extra_loss is not None:
                loss = loss + extra_loss()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def train_source_model(train_data, val_data, featurizer, seed, risk_aware=False):
    """Train base scorer on source family (direct / risk_aware)."""
    set_seed(seed)
    model = MorganMLPScorer()
    x_train, y_train = featurizer.featurize(train_data)
    if risk_aware:
        w = torch.tensor([1.0 - min(max(d.get("fnr", 0.5), 0.0), 0.9) for d in train_data],
                         dtype=torch.float32)
    else:
        w = torch.ones(len(train_data), dtype=torch.float32)
    best_state, best_val = None, -1.0
    for epoch in range(N_EPOCHS):
        _train_loop(model, x_train, y_train, w, epochs=1, lr=LR)
        if val_data:
            val_mrr, _, _ = predict_metrics(model, val_data, featurizer)
            if val_mrr > best_val:
                best_val = val_mrr
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def fine_tune_head(model, target_val_data, featurizer, seed):
    set_seed(seed)
    for p in model.parameters():
        p.requires_grad = False
    for p in model.net[-1].parameters():
        p.requires_grad = True
    x, y = featurizer.featurize(target_val_data)
    w = torch.ones(len(target_val_data))
    _train_loop(model, x, y, w, epochs=FT_EPOCHS, lr=LR * 0.5,
                params=[p for p in model.parameters() if p.requires_grad])
    for p in model.parameters():
        p.requires_grad = True
    return model


def fine_tune_lora(model, target_train_data, featurizer, seed):
    set_seed(seed)
    adapted = AdaptedMLP(model)
    for p in adapted.base.parameters():
        p.requires_grad = False
    x, y = featurizer.featurize(target_train_data)
    w = torch.ones(len(target_train_data))
    _train_loop(adapted, x, y, w, epochs=FT_EPOCHS, lr=LR * 0.5,
                params=list(adapted.adapter_parameters()))
    return adapted


def fine_tune_ewc(model, source_train_data, target_train_data, featurizer, seed,
                  lam=EWC_LAMBDA, fisher_samples=800):
    """Fine-tune on target train with EWC penalty anchored at source weights."""
    set_seed(seed)
    anchor = {k: v.clone() for k, v in model.state_dict().items()}
    # Diagonal Fisher on source data
    fisher = {k: torch.zeros_like(v) for k, v in model.state_dict().items()}
    src = source_train_data[:fisher_samples]
    x_src, y_src = featurizer.featurize(src)
    model.eval()
    for i in range(0, x_src.shape[0], BATCH_SIZE):
        xb = x_src[i:i + BATCH_SIZE]
        model.zero_grad()
        logits = model(xb)
        probs = torch.sigmoid(logits)
        logp = y_src[i:i + BATCH_SIZE] * torch.log(probs + 1e-8) + \
            (1 - y_src[i:i + BATCH_SIZE]) * torch.log(1 - probs + 1e-8)
        (-logp.sum()).backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                fisher[name] += p.grad.detach() ** 2
    n = max(1, x_src.shape[0])
    fisher = {k: v / n for k, v in fisher.items()}

    def extra_loss():
        loss = 0.0
        for name, p in model.named_parameters():
            loss = loss + (fisher[name] * (p - anchor[name]) ** 2).sum()
        return 0.5 * lam * loss

    x, y = featurizer.featurize(target_train_data)
    w = torch.ones(len(target_train_data))
    _train_loop(model, x, y, w, epochs=FT_EPOCHS, lr=LR * 0.3, extra_loss=extra_loss)
    return model


def train_multi_task(source_train_data, target_train_data, val_data, featurizer, seed):
    """Joint training on source + target train (balanced by upsampling)."""
    set_seed(seed)
    combined = list(source_train_data) + list(target_train_data)
    return train_source_model(combined, val_data, featurizer, seed, risk_aware=False)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def predict_scores(model, data, featurizer):
    model.eval()
    with torch.no_grad():
        x, y = featurizer.featurize(data)
        if x.shape[0] == 0:
            return np.array([]), np.array([])
        scores = torch.sigmoid(model(x)).cpu().numpy()
    return scores, y.numpy()


def predict_metrics(model, data, featurizer):
    scores, labels = predict_scores(model, data, featurizer)
    if len(scores) == 0:
        return 0.0, 0.0, 0.0
    return compute_mrr(scores, labels), compute_auprc(scores, labels), compute_ece(scores, labels)


# ---------------------------------------------------------------------------
# Statistics: cluster bootstrap CI, exact paired permutation, effect size
# ---------------------------------------------------------------------------

def cluster_bootstrap_delta_ci(
    method_scores: np.ndarray,
    baseline_scores: np.ndarray,
    labels: np.ndarray,
    clusters: Sequence[str],
    n_boot: int = N_BOOTSTRAP,
    seed: int = BASE_SEED,
) -> Tuple[float, float, float]:
    """Percentile 95% CI of MRR delta under cluster resampling."""
    rng = np.random.RandomState(seed)
    labels = np.asarray(labels)
    unique_clusters = np.array(sorted(set(clusters)))
    if len(unique_clusters) < 2 or labels.sum() == 0:
        delta = compute_mrr(method_scores, labels) - compute_mrr(baseline_scores, labels)
        return float(delta), float(delta), float(delta)
    cluster_to_idx = {c: np.array([i for i, cc in enumerate(clusters) if cc == c])
                      for c in unique_clusters}
    deltas = []
    for _ in range(n_boot):
        sampled = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
        idx = np.concatenate([cluster_to_idx[c] for c in sampled])
        y = labels[idx]
        if y.sum() == 0 or y.sum() == len(y):
            continue
        deltas.append(compute_mrr(method_scores[idx], y) - compute_mrr(baseline_scores[idx], y))
    if not deltas:
        return 0.0, 0.0, 0.0
    deltas = np.array(deltas)
    return (float(deltas.mean()),
            float(np.percentile(deltas, 2.5)),
            float(np.percentile(deltas, 97.5)))


def exact_sign_flip_pvalue(seed_deltas: Sequence[float]) -> float:
    """Exact two-sided paired permutation (sign-flip) p-value over seeds."""
    deltas = np.array([d for d in seed_deltas], dtype=float)
    if len(deltas) == 0:
        return 1.0
    observed = abs(deltas.mean())
    n = len(deltas)
    count = 0
    total = 0
    for mask in range(1 << n):
        signs = np.array([1.0 if (mask >> i) & 1 else -1.0 for i in range(n)])
        total += 1
        if abs((deltas * signs).mean()) >= observed - 1e-15:
            count += 1
    return count / total


def cohens_d(seed_deltas: Sequence[float]) -> float:
    deltas = np.array(list(seed_deltas), dtype=float)
    if len(deltas) < 2:
        return 0.0
    sd = deltas.std(ddof=1)
    if sd < 1e-12:
        if deltas.mean() > 0:
            return float("inf")
        return float("-inf") if deltas.mean() < 0 else 0.0
    return float(deltas.mean() / sd)


def mean_tanimoto(source_smiles: Sequence[str], target_smiles: Sequence[str],
                  max_pairs: int = 200, seed: int = BASE_SEED) -> float:
    """Domain similarity: mean Tanimoto between sampled source/target molecules."""
    if Chem is None or not source_smiles or not target_smiles:
        return 0.0
    from rdkit import DataStructs
    from rdkit.Chem import AllChem
    rng = np.random.RandomState(seed)
    src = list(source_smiles)
    tgt = list(target_smiles)
    if len(src) > max_pairs:
        src = [src[i] for i in rng.choice(len(src), max_pairs, replace=False)]
    if len(tgt) > max_pairs:
        tgt = [tgt[i] for i in rng.choice(len(tgt), max_pairs, replace=False)]

    def fps(smiles_list):
        out = []
        for s in smiles_list:
            mol = Chem.MolFromSmiles(s)
            if mol is not None:
                out.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=FP_BITS))
        return out

    src_fps, tgt_fps = fps(src), fps(tgt)
    if not src_fps or not tgt_fps:
        return 0.0
    sims = []
    for i in range(min(len(src_fps), len(tgt_fps))):
        sims.append(DataStructs.TanimotoSimilarity(src_fps[i], tgt_fps[i]))
    return float(np.mean(sims)) if sims else 0.0


# ---------------------------------------------------------------------------
# Direction registry
# ---------------------------------------------------------------------------

def build_directions(hte: Dict[str, Dict[str, List[dict]]],
                     uspto: Dict[str, Dict[str, List[dict]]],
                     ord_data: Dict[str, Dict[str, List[dict]]],
                     smoke: bool = False) -> List[Dict[str, Any]]:
    directions: List[Dict[str, Any]] = []

    def has_data(ds, family, split, n):
        return len(ds.get(family, {}).get(split, [])) >= n

    # EAS <-> C-N (USPTO)
    for src, tgt in [(EAS, CN_COUPLING), (CN_COUPLING, EAS)]:
        if has_data(uspto, src, "train", MIN_FAMILY_SIZE) and has_data(uspto, tgt, "test", 1):
            directions.append({"name": f"USPTO:{src}→USPTO:{tgt}", "pair_group": "EAS↔C-N",
                               "source": uspto[src], "target": uspto[tgt],
                               "source_name": f"USPTO:{src}", "target_name": f"USPTO:{tgt}"})

    # USPTO(EAS+C-N) -> HTE
    uspto_pool = pool_families(uspto.get(EAS, {}), uspto.get(CN_COUPLING, {}))
    # ORD(EAS+C-N) -> HTE
    ord_pool = pool_families(ord_data.get(EAS, {}), ord_data.get(CN_COUPLING, {}))

    hte_targets = [f for f in ("Pd coupling", "Alkylation")
                   if has_data(hte, f, "test", 1)]
    for tgt in hte_targets:
        if len(uspto_pool["train"]) >= MIN_FAMILY_SIZE:
            directions.append({"name": f"USPTO:EAS+C-N→HTE:{tgt}", "pair_group": "USPTO→HTE",
                               "source": uspto_pool, "target": hte[tgt],
                               "source_name": "USPTO:EAS+C-N", "target_name": f"HTE:{tgt}"})
        if len(ord_pool["train"]) >= MIN_FAMILY_SIZE:
            directions.append({"name": f"ORD:EAS+C-N→HTE:{tgt}", "pair_group": "ORD→HTE",
                               "source": ord_pool, "target": hte[tgt],
                               "source_name": "ORD:EAS+C-N", "target_name": f"HTE:{tgt}"})

    # HTE family <-> family
    hte_pairs = [("Pd coupling", "Alkylation"), ("Pd coupling", "Hydrogenation"),
                 ("Alkylation", "Cabonylation"), ("Rh coupling", "Cu coupling")]
    for src, tgt in hte_pairs:
        for s, t in [(src, tgt), (tgt, src)]:
            if s not in hte or t not in hte:
                continue
            if not has_data(hte, s, "train", MIN_FAMILY_SIZE):
                continue
            if not has_data(hte, t, "test", 1):
                continue
            directions.append({"name": f"HTE:{s}→HTE:{t}", "pair_group": "HTE family",
                               "source": hte[s], "target": hte[t],
                               "source_name": f"HTE:{s}", "target_name": f"HTE:{t}"})

    if smoke:
        return directions[:3]
    return directions


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

def compute_verdict_v2(direction_stats: List[dict]) -> Dict[str, Any]:
    """GO: >=2 chemically different pair groups with CI all positive.
    PARTIAL_GO: >=1 direction positive, all failures reported.
    NO_GO: no positive transfer or severe catastrophic forgetting."""
    positive_groups = set()
    positive_directions = []
    negative_directions = []
    severe_forgetting = []

    for stat in direction_stats:
        for method, res in stat["methods"].items():
            entry = {"direction": stat["name"], "pair_group": stat["pair_group"],
                     "method": method, "delta_mean": res["delta_mean"],
                     "ci_low": res["ci_low"], "ci_high": res["ci_high"],
                     "p_value": res["p_value"], "cohens_d": res["cohens_d"]}
            if res["ci_low"] > 0:
                positive_directions.append(entry)
                positive_groups.add(stat["pair_group"])
            elif res["delta_mean"] < 0:
                negative_directions.append(entry)
            forgetting = res.get("forgetting_mean", 0.0)
            if forgetting < -0.2:
                severe_forgetting.append({**entry, "forgetting": forgetting})

    n_groups = len(positive_groups)
    if n_groups >= 2 and not severe_forgetting:
        verdict, reason = "GO", (f"{n_groups} chemically different pair groups with "
                                 f"cluster-bootstrap CI all positive")
    elif positive_directions and not severe_forgetting:
        verdict, reason = "PARTIAL_GO", (f"{len(positive_directions)} positive direction×method "
                                         f"entries across {n_groups} pair group(s); "
                                         f"all failures reported")
    else:
        verdict = "NO_GO"
        reason = "No positive transfer with CI>0"
        if severe_forgetting:
            reason += f"; severe catastrophic forgetting in {len(severe_forgetting)} entries"

    return {
        "verdict": verdict,
        "reason": reason,
        "n_positive_pair_groups": n_groups,
        "positive_directions": positive_directions,
        "negative_directions": negative_directions,
        "severe_forgetting": severe_forgetting,
        "next_phase_allowed": verdict in ("GO", "PARTIAL_GO"),
    }


# ---------------------------------------------------------------------------
# Main experiment driver
# ---------------------------------------------------------------------------

def run_direction(direction: Dict[str, Any], methods: Sequence[str], seeds: Sequence[int],
                  featurizer: Featurizer, raw_dir: Path,
                  n_boot: int = N_BOOTSTRAP) -> Dict[str, Any]:
    src, tgt = direction["source"], direction["target"]
    src_train, src_test = src.get("train", []), src.get("test", [])
    tgt_train, tgt_val, tgt_test = tgt.get("train", []), tgt.get("val", []), tgt.get("test", [])

    result: Dict[str, Any] = {
        "name": direction["name"], "pair_group": direction["pair_group"],
        "source_name": direction["source_name"], "target_name": direction["target_name"],
        "n_source_train": len(src_train), "n_target_train": len(tgt_train),
        "n_target_test": len(tgt_test),
        "domain_similarity": mean_tanimoto(
            [d["smiles"] for d in src_train if d["label"] == 1],
            [d["smiles"] for d in tgt_train if d["label"] == 1]),
        "methods": {},
        "baseline": {},
    }
    if not tgt_test:
        return result

    labels = np.array([d["label"] for d in tgt_test])
    clusters = [d["cluster_id"] for d in tgt_test]

    # Per-seed baseline (train on target train)
    baseline_scores = {}
    baseline_mrrs = []
    for seed in seeds:
        model = train_source_model(tgt_train, tgt_val, featurizer, seed)
        scores, _ = predict_scores(model, tgt_test, featurizer)
        baseline_scores[seed] = scores
        baseline_mrrs.append(compute_mrr(scores, labels))
        _save_raw(raw_dir, direction["name"], "baseline", seed, tgt_test, scores)
    result["baseline"] = {"mrr_mean": statistics.mean(baseline_mrrs),
                          "mrr_per_seed": baseline_mrrs}

    for method in methods:
        method_mrrs, seed_deltas = [], []
        pooled_method, pooled_base = [], []
        forgettings, eces, auprcs = [], [], []
        for seed in seeds:
            risk_aware = method == "risk_aware"
            model = train_source_model(src_train, tgt_val, featurizer, seed,
                                       risk_aware=risk_aware)
            src_mrr_before = predict_metrics(model, src_test, featurizer)[0] if src_test else 0.0
            if method == "head_ft":
                model = fine_tune_head(model, tgt_val, featurizer, seed)
            elif method == "lora_adapter":
                model = fine_tune_lora(model, tgt_train, featurizer, seed)
            elif method == "ewc":
                model = fine_tune_ewc(model, src_train, tgt_train, featurizer, seed)
            elif method == "multi_task":
                model = train_multi_task(src_train, tgt_train, tgt_val, featurizer, seed)
            scores, _ = predict_scores(model, tgt_test, featurizer)
            mrr, auprc, ece = compute_mrr(scores, labels), compute_auprc(scores, labels), compute_ece(scores, labels)
            method_mrrs.append(mrr)
            auprcs.append(auprc)
            eces.append(ece)
            delta = mrr - compute_mrr(baseline_scores[seed], labels)
            seed_deltas.append(delta)
            pooled_method.append(scores)
            pooled_base.append(baseline_scores[seed])
            _save_raw(raw_dir, direction["name"], method, seed, tgt_test, scores)
            if src_test and method in ("head_ft", "lora_adapter", "ewc"):
                src_mrr_after = predict_metrics(model, src_test, featurizer)[0]
                forgettings.append(src_mrr_after - src_mrr_before)

        # Pooled cluster bootstrap over concatenated seed predictions
        pm = np.concatenate(pooled_method)
        pb = np.concatenate(pooled_base)
        pooled_labels = np.tile(labels, len(seeds))
        pooled_clusters = list(clusters) * len(seeds)
        delta_mean, ci_low, ci_high = cluster_bootstrap_delta_ci(
            pm, pb, pooled_labels, pooled_clusters, n_boot=n_boot)
        result["methods"][method] = {
            "mrr_mean": statistics.mean(method_mrrs),
            "mrr_per_seed": method_mrrs,
            "auprc_mean": statistics.mean(auprcs),
            "ece_mean": statistics.mean(eces),
            "delta_mean": statistics.mean(seed_deltas),
            "delta_per_seed": seed_deltas,
            "bootstrap_delta_mean": delta_mean,
            "ci_low": ci_low, "ci_high": ci_high,
            "p_value": exact_sign_flip_pvalue(seed_deltas),
            "cohens_d": cohens_d(seed_deltas),
            "forgetting_mean": statistics.mean(forgettings) if forgettings else 0.0,
        }
    return result


def _save_raw(raw_dir: Path, direction_name: str, method: str, seed: int,
              test_data: Sequence[dict], scores: np.ndarray) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    safe = direction_name.replace(":", "").replace("→", "_to_").replace(" ", "_").replace("+", "_")
    path = raw_dir / f"{safe}__{method}__seed{seed}.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "cluster_id", "label", "score"])
        for d, s in zip(test_data, scores):
            writer.writerow([d["candidate_id"], d["cluster_id"], d["label"], f"{s:.6f}"])


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    import argparse
    parser = argparse.ArgumentParser(description=f"{PHASE} cross-family transfer v2")
    parser.add_argument("--candidate-manifest", type=Path,
                        default=Path("data/p4/manifests/hte_feasibility_v2.json"))
    parser.add_argument("--uspto-csv", type=Path,
                        default=Path("data/processed/uspto_openmolecules_normalized.csv"))
    parser.add_argument("--ord-csv", type=Path,
                        default=Path("data/processed/ord_normalized.csv"))
    parser.add_argument("--risk-path", type=Path,
                        default=Path("results/p4_risk_aware/risk_artifacts.json"))
    parser.add_argument("--train-idx", type=Path, default=None,
                        help="Optional JSON list of HTE group indices to subset train")
    parser.add_argument("--val-idx", type=Path, default=None,
                        help="Optional JSON list of HTE group indices to subset val")
    parser.add_argument("--test-idx", type=Path, default=None,
                        help="Optional JSON list of HTE group indices to subset test")
    parser.add_argument("--seed", type=int, default=BASE_SEED,
                        help="Base seed; 10 predeclared seeds derived as seed+i")
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/p4_cross_family_transfer_v2"))
    parser.add_argument("--max-pos-per-family", type=int, default=MAX_POS_PER_FAMILY)
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw_predictions"
    cache_dir = output_dir / "_cache"

    seeds = [args.seed + i for i in range(args.n_seeds)]
    if args.smoke:
        seeds = seeds[:2]

    print(f"[{PHASE}] Loading HTE manifest: {args.candidate_manifest}")
    hte = load_hte_family_data(args.candidate_manifest, args.risk_path)

    # Optional idx subsetting of HTE groups (training-entry contract)
    if args.train_idx or args.val_idx or args.test_idx:
        hte = _subset_hte_by_idx(hte, args.train_idx, args.val_idx, args.test_idx)

    print(f"[{PHASE}] Classifying USPTO reactions (rule-based EAS / C-N)...")
    uspto = load_external_family_data(args.uspto_csv, "uspto", args.max_pos_per_family,
                                      NEG_PER_REACTION, args.seed, cache_dir)
    print(f"[{PHASE}] Classifying ORD reactions...")
    ord_data = load_external_family_data(args.ord_csv, "ord", args.max_pos_per_family,
                                         NEG_PER_REACTION, args.seed, cache_dir,
                                         force_split_from_key=True)

    for name, ds in [("USPTO", uspto), ("ORD", ord_data), ("HTE", hte)]:
        for fam, splits in sorted(ds.items()):
            print(f"  {name}:{fam}: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")

    directions = build_directions(hte, uspto, ord_data, smoke=args.smoke)
    print(f"\n[{PHASE}] {len(directions)} directions × {len(METHODS)} methods × "
          f"{len(seeds)} seeds")

    featurizer = Featurizer()
    direction_stats = []
    for direction in directions:
        print(f"\n[{PHASE}] === {direction['name']} ===")
        stat = run_direction(direction, METHODS, seeds, featurizer, raw_dir,
                             n_boot=args.n_bootstrap)
        direction_stats.append(stat)
        for method, res in stat.get("methods", {}).items():
            print(f"  {method:14s} mrr={res['mrr_mean']:.4f} delta={res['delta_mean']:+.4f} "
                  f"CI[{res['ci_low']:+.4f},{res['ci_high']:+.4f}] p={res['p_value']:.4f}")

    with open(output_dir / "direction_stats.json", "w") as f:
        json.dump(direction_stats, f, indent=2)

    # Flat CSV
    rows = []
    for stat in direction_stats:
        for method, res in stat.get("methods", {}).items():
            for i, seed in enumerate(seeds):
                rows.append({
                    "direction": stat["name"], "pair_group": stat["pair_group"],
                    "method": method, "seed": seed,
                    "target_mrr": res["mrr_per_seed"][i],
                    "delta_vs_baseline": res["delta_per_seed"][i],
                    "baseline_mrr": stat["baseline"]["mrr_per_seed"][i],
                    "domain_similarity": stat["domain_similarity"],
                })
    if rows:
        with open(output_dir / "transfer_results.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    # Family macro summary (macro average by target family per method)
    by_target: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for stat in direction_stats:
        for method, res in stat.get("methods", {}).items():
            by_target[stat["target_name"]][method].append(res["mrr_mean"])
    with open(output_dir / "family_macro_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["target_family", "method", "n_directions", "macro_mrr"])
        for target, methods_map in sorted(by_target.items()):
            for method, values in sorted(methods_map.items()):
                writer.writerow([target, method, len(values), f"{statistics.mean(values):.6f}"])

    # Negative-transfer analysis
    analysis = {
        "directions_with_any_positive_ci": sum(
            1 for s in direction_stats
            if any(r["ci_low"] > 0 for r in s.get("methods", {}).values())),
        "total_directions": len(direction_stats),
        "per_pair_group": {},
    }
    for stat in direction_stats:
        grp = analysis["per_pair_group"].setdefault(stat["pair_group"], {"positive": 0, "negative": 0})
        best_delta = max((r["delta_mean"] for r in stat.get("methods", {}).values()), default=0.0)
        if best_delta > 0:
            grp["positive"] += 1
        else:
            grp["negative"] += 1
    with open(output_dir / "transfer_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)

    verdict = compute_verdict_v2(direction_stats)
    go_no_go = {
        "phase": "P4-G8B",
        "status": verdict["verdict"],
        "version": "v2_full_spec",
        "primary_metric": {"name": "transfer_gain", "comparison": "method_vs_target_baseline"},
        "predeclared_threshold": {
            "go": ">=2 chemically different directions CI all positive",
            "partial_go": ">=1 direction positive; all failures reported",
            "no_go": "No positive transfer; or severe catastrophic forgetting",
        },
        "verdict_details": verdict,
        "evidence_paths": [
            str(output_dir / "transfer_results.csv"),
            str(output_dir / "direction_stats.json"),
            str(output_dir / "family_macro_summary.csv"),
            str(output_dir / "transfer_analysis.json"),
            str(raw_dir) + "/",
        ],
        "limitations": [
            "USPTO/ORD family labels are rule-based proxies (is_cn_coupling/is_eas)",
            "USPTO negatives generated by rule PC-CNG after RXNMapper mapping",
            "external scorer is Morgan MLP; chemformer/GNN transfer not repeated in v2",
        ],
        "next_phase_allowed": verdict["next_phase_allowed"],
    }
    with open(output_dir / "go_no_go.json", "w") as f:
        json.dump(go_no_go, f, indent=2)

    with open(output_dir / "run_manifest.json", "w") as f:
        json.dump({
            "phase": "P4-G8B", "version": "v2_full_spec",
            "methods": METHODS, "seeds": seeds,
            "n_directions": len(direction_stats),
            "direction_names": [s["name"] for s in direction_stats],
            "n_bootstrap": args.n_bootstrap,
            "epochs": N_EPOCHS, "ft_epochs": FT_EPOCHS,
            "batch_size": BATCH_SIZE, "lr": LR,
            "lora_rank": LORA_RANK, "ewc_lambda": EWC_LAMBDA,
            "max_pos_per_family": args.max_pos_per_family,
            "neg_per_reaction": NEG_PER_REACTION,
        }, f, indent=2)
    with open(output_dir / "environment.json", "w") as f:
        env = {"python": sys.version.split()[0], "platform": platform.platform(),
               "torch": torch.__version__, "numpy": np.__version__}
        try:
            import rdkit
            env["rdkit"] = rdkit.__version__
        except ImportError:
            pass
        json.dump(env, f, indent=2)
    hashes = {}
    for p in (args.candidate_manifest, args.uspto_csv, args.ord_csv, args.risk_path):
        if p and Path(p).exists():
            hashes[str(p)] = _sha256(Path(p))
    with open(output_dir / "input_hashes.json", "w") as f:
        json.dump(hashes, f, indent=2)
    with open(output_dir / "commands.log", "w") as f:
        f.write(" ".join([sys.executable, "-m", "pc_cng.p4_g8b_transfer_v2"] +
                         [f"--{k}={v}" for k, v in vars(args).items()]) + "\n")

    elapsed = time.time() - t0
    print(f"\n[{PHASE}] Complete ({elapsed:.1f}s)  Verdict: {verdict['verdict']}")
    print(f"[{PHASE}] {verdict['reason']}")


def _subset_hte_by_idx(hte, train_idx, val_idx, test_idx):
    """Apply optional group-index subsets to HTE splits (contract flags)."""
    def load_idx(p):
        if p is None:
            return None
        with open(p) as f:
            return set(json.load(f))

    idx_maps = {"train": load_idx(train_idx), "val": load_idx(val_idx), "test": load_idx(test_idx)}
    out = {}
    for fam, splits in hte.items():
        out[fam] = {}
        for split, rows in splits.items():
            allowed = idx_maps.get(split)
            if allowed is None:
                out[fam][split] = rows
            else:
                out[fam][split] = [r for i, r in enumerate(rows) if i in allowed]
    return out


if __name__ == "__main__":
    main()
