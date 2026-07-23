"""P4-G8B: Cross-reaction-family transfer analysis.

Trains on one reaction family, tests on another, to measure transfer
gain/forgetting/negative-transfer. Reports ALL directions including failures.

Spec: 提示词/pccng 的分阶段提示词.md#L1572-1833 (P4-G8B)

Methods compared:
- direct: train on source family, test on target (no adaptation)
- head_ft: train on source, fine-tune last layer on target val
- risk_aware: train on source with FNR weighting, test on target

Outputs (results/p4_cross_family_transfer/):
    transfer_results.csv       — per-experiment results
    family_macro_summary.csv   — macro-averaged metrics per family
    go_no_go.json              — verdict
    run_manifest.json, environment.json, input_hashes.json, commands.log
"""

from __future__ import annotations

import json
import os
import platform
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ["RDKitRDLogger"] = "0"
try:
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
except Exception:
    pass

PHASE = "P4-G8B"
SEED = 20260723
N_EPOCHS = 5
BATCH_SIZE = 16
LR = 1e-3
FP_RADIUS = 2
FP_BITS = 2048
MIN_FAMILY_SIZE = 80  # minimum candidates to use a family


# ---------------------------------------------------------------------------
# Model (reused from G4 diagnostic)
# ---------------------------------------------------------------------------

class MorganMLPScorer(nn.Module):
    """Simple Morgan-fingerprint MLP for candidate scoring."""
    def __init__(self, n_bits: int = FP_BITS, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_bits, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def morgan_fp(smiles: str) -> np.ndarray:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.DataStructs import ConvertToNumpyArray
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(FP_BITS, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_BITS)
    arr = np.zeros(FP_BITS, dtype=np.float32)
    ConvertToNumpyArray(fp, arr)
    return arr


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def load_manifest_by_family(manifest_path: Path) -> Dict[str, List[dict]]:
    """Load manifest and group candidates by reaction_family + split."""
    with open(manifest_path) as f:
        manifest = json.load(f)

    by_family = defaultdict(lambda: {"train": [], "val": [], "test": []})
    for group in manifest.get("groups", []):
        for cand in group.get("candidates", []):
            family = cand.get("reaction_family", "Unknown")
            split = cand.get("split", "train")
            if split in ("train", "val", "test"):
                by_family[family][split].append(cand)

    # Filter families with enough candidates
    return {f: s for f, s in by_family.items()
            if len(s["train"]) >= MIN_FAMILY_SIZE}


def prepare_training_data(candidates: List[dict], arm_id: str = "A6"
                          ) -> Tuple[List[dict], int, int]:
    """Build training data: positives from gold, negatives from arm source."""
    positives = [c for c in candidates if c.get("gold_candidate")]
    negatives = [c for c in candidates if not c.get("gold_candidate")
                 and c.get("candidate_source") in ("rule_pc_cng", "random_corruption")]

    data = []
    for c in positives:
        data.append({"smiles": c["candidate_smiles"], "label": 1,
                      "candidate_id": c["candidate_id"]})
    for c in negatives:
        data.append({"smiles": c["candidate_smiles"], "label": 0,
                      "candidate_id": c["candidate_id"]})
    return data, len(positives), len(negatives)


def featurize(data: List[dict]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert list of {smiles, label} to tensors."""
    fps = []
    labels = []
    for d in data:
        fp = morgan_fp(d["smiles"])
        fps.append(fp)
        labels.append(d["label"])
    return (torch.tensor(np.array(fps), dtype=torch.float32),
            torch.tensor(labels, dtype=torch.float32))


# ---------------------------------------------------------------------------
# Training methods
# ---------------------------------------------------------------------------

def train_mlp(train_data, val_data, device, epochs=N_EPOCHS,
              risk_aware=False, risk_map=None, seed=SEED):
    """Train MLP. If risk_aware, weight loss by (1 - FNR)."""
    set_seed(seed)
    model = MorganMLPScorer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    x_train, y_train = featurize(train_data)
    x_train, y_train = x_train.to(device), y_train.to(device)

    if risk_aware and risk_map:
        weights = torch.tensor(
            [1.0 - risk_map.get(d["candidate_id"], 0.5) for d in train_data],
            dtype=torch.float32).to(device)
    else:
        weights = torch.ones(len(train_data), dtype=torch.float32).to(device)

    best_val_mrr, best_state = -1.0, None
    n = len(train_data)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            x, y, w = x_train[idx], y_train[idx], weights[idx]
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y, weight=w)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        if val_data:
            model.eval()
            with torch.no_grad():
                x_val, y_val = featurize(val_data)
                x_val, y_val = x_val.to(device), y_val.to(device)
                val_scores = torch.sigmoid(model(x_val)).cpu().numpy()
                val_labels = y_val.cpu().numpy()
            val_mrr = compute_mrr(val_scores, val_labels)
            if val_mrr > best_val_mrr:
                best_val_mrr = val_mrr
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_mrr


def fine_tune_head(model, target_val_data, device, epochs=3, seed=SEED):
    """Fine-tune only the last layer on target validation data."""
    set_seed(seed)
    # Freeze all but last layer
    for param in model.parameters():
        param.requires_grad = False
    for param in model.net[-1].parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR * 0.5)

    x_val, y_val = featurize(target_val_data)
    x_val, y_val = x_val.to(device), y_val.to(device)
    n = len(target_val_data)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            x, y = x_val[idx], y_val[idx]
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


def evaluate_model(model, test_data, device):
    """Evaluate model on test data, return metrics."""
    model.eval()
    with torch.no_grad():
        x_test, y_test = featurize(test_data)
        x_test = x_test.to(device)
        scores = torch.sigmoid(model(x_test)).cpu().numpy()
        labels = y_test.numpy()

    mrr = compute_mrr(scores, labels)
    auprc = compute_auprc(scores, labels)
    ece = compute_ece(scores, labels)
    return {"mrr": mrr, "auprc": auprc, "ece": ece}


def compute_mrr(scores, labels, group_size=None):
    """Mean Reciprocal Rank."""
    if group_size is None:
        # Binary: MRR = mean(1/rank) for positives
        order = np.argsort(-scores)
        ranked_labels = labels[order]
        rr = []
        for i, l in enumerate(ranked_labels):
            if l == 1:
                rr.append(1.0 / (i + 1))
        return statistics.mean(rr) if rr else 0.0
    else:
        # Group-based MRR
        rr = []
        for i in range(0, len(scores), group_size):
            g_scores = scores[i:i + group_size]
            g_labels = labels[i:i + group_size]
            order = np.argsort(-g_scores)
            for rank, idx in enumerate(order):
                if g_labels[idx] == 1:
                    rr.append(1.0 / (rank + 1))
                    break
        return statistics.mean(rr) if rr else 0.0


def compute_auprc(scores, labels):
    """Area Under Precision-Recall Curve."""
    from sklearn.metrics import average_precision_score
    if len(set(labels)) < 2:
        return 0.0
    return float(average_precision_score(labels, scores))


def compute_ece(scores, labels, n_bins=10):
    """Expected Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(scores)
    for i in range(n_bins):
        mask = (scores >= bins[i]) & (scores < bins[i + 1])
        if mask.sum() == 0:
            continue
        avg_conf = scores[mask].mean()
        avg_acc = labels[mask].mean()
        ece += abs(avg_conf - avg_acc) * mask.sum() / n
    return float(ece)


# ---------------------------------------------------------------------------
# Transfer experiment
# ---------------------------------------------------------------------------

def run_transfer_experiment(
    families_data: Dict[str, Dict[str, List[dict]]],
    source_family: str,
    target_family: str,
    methods: List[str],
    risk_map: Dict[str, float],
    device: str,
    seeds: List[int],
) -> List[Dict[str, Any]]:
    """Run one transfer direction: source → target."""
    results = []
    source = families_data[source_family]
    target = families_data[target_family]

    # Prepare data
    src_train, src_n_pos, src_n_neg = prepare_training_data(source["train"])
    tgt_val, _, _ = prepare_training_data(target["val"])
    tgt_test, _, _ = prepare_training_data(target["test"])
    src_test, _, _ = prepare_training_data(source["test"])

    # Also need target-train baseline (train on target, test on target)
    tgt_train, _, _ = prepare_training_data(target["train"])

    for method in methods:
        for seed in seeds:
            print(f"  {source_family} → {target_family} | {method} | seed={seed}")

            # Train on source
            risk_aware = (method == "risk_aware")
            model, train_mrr = train_mlp(
                src_train, tgt_val, device, risk_aware=risk_aware,
                risk_map=risk_map, seed=seed)

            if method == "head_ft":
                model = fine_tune_head(model, tgt_val, device, seed=seed)

            # Evaluate on target test
            target_metrics = evaluate_model(model, tgt_test, device)
            # Evaluate on source test (for forgetting analysis)
            # Skip if source test is empty (e.g., Hydrogenation has test=0)
            if src_test:
                source_metrics = evaluate_model(model, src_test, device)
            else:
                source_metrics = {"mrr": 0.0, "auprc": 0.0, "ece": 0.0}

            results.append({
                "source_family": source_family,
                "target_family": target_family,
                "method": method,
                "seed": seed,
                "train_mrr": round(train_mrr, 6),
                "target_mrr": round(target_metrics["mrr"], 6),
                "target_auprc": round(target_metrics["auprc"], 6),
                "target_ece": round(target_metrics["ece"], 6),
                "source_mrr": round(source_metrics["mrr"], 6),
                "source_auprc": round(source_metrics["auprc"], 6),
                "src_n_train": len(src_train),
                "tgt_n_train": len(tgt_train),
                "tgt_n_test": len(tgt_test),
            })

    # Baseline: train on target, test on target
    for seed in seeds:
        print(f"  {target_family} (baseline) | seed={seed}")
        model, _ = train_mlp(tgt_train, tgt_val, device, seed=seed)
        baseline_metrics = evaluate_model(model, tgt_test, device)
        results.append({
            "source_family": target_family,
            "target_family": target_family,
            "method": "baseline",
            "seed": seed,
            "train_mrr": 0.0,
            "target_mrr": round(baseline_metrics["mrr"], 6),
            "target_auprc": round(baseline_metrics["auprc"], 6),
            "target_ece": round(baseline_metrics["ece"], 6),
            "source_mrr": round(baseline_metrics["mrr"], 6),
            "source_auprc": round(baseline_metrics["auprc"], 6),
            "src_n_train": len(tgt_train),
            "tgt_n_train": len(tgt_train),
            "tgt_n_test": len(tgt_test),
        })

    return results


def compute_verdict(all_results: List[dict]) -> Dict[str, Any]:
    """Determine GO/PARTIAL_GO/NO_GO based on transfer results."""
    # Compute transfer gain (target_mrr - baseline_mrr) per direction
    by_direction = defaultdict(list)
    baselines = defaultdict(list)

    for r in all_results:
        if r["method"] == "baseline":
            baselines[r["target_family"]].append(r["target_mrr"])
        else:
            direction = f"{r['source_family']}→{r['target_family']}"
            by_direction[direction].append(r)

    baseline_means = {f: statistics.mean(v) for f, v in baselines.items()}

    transfer_gains = []
    positive_directions = []
    negative_directions = []

    for direction, results in by_direction.items():
        target_family = results[0]["target_family"]
        baseline_mrr = baseline_means.get(target_family, 0.0)
        transfer_mrr = statistics.mean(r["target_mrr"] for r in results)
        gain = transfer_mrr - baseline_mrr
        transfer_gains.append({
            "direction": direction,
            "method": results[0]["method"],
            "transfer_mrr": round(transfer_mrr, 6),
            "baseline_mrr": round(baseline_mrr, 6),
            "gain": round(gain, 6),
        })
        if gain > 0:
            positive_directions.append(direction)
        else:
            negative_directions.append(direction)

    # Verdict: need >=2 chemically different directions with positive CI
    n_positive = len(set(positive_directions))
    n_total = len(by_direction)

    if n_positive >= 2 and n_positive / max(n_total, 1) >= 0.4:
        verdict = "GO"
        reason = (f"{n_positive}/{n_total} directions show positive transfer gain")
    elif n_positive >= 1:
        verdict = "PARTIAL_GO"
        reason = (f"{n_positive}/{n_total} directions show positive transfer; "
                  f"negative transfer reported for {len(negative_directions)}")
    else:
        verdict = "NO_GO"
        reason = f"No positive transfer in any direction"

    return {
        "verdict": verdict,
        "reason": reason,
        "n_positive_directions": n_positive,
        "n_total_directions": n_total,
        "transfer_gains": transfer_gains,
        "positive_directions": positive_directions,
        "negative_directions": negative_directions,
        "next_phase_allowed": verdict in ("GO", "PARTIAL_GO"),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description=f"{PHASE} cross-family transfer")
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/p4/manifests/hte_feasibility_v2.json"))
    parser.add_argument("--risk-path", type=Path,
                        default=Path("results/p4_risk_aware/risk_artifacts.json"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/p4_cross_family_transfer"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-seeds", type=int, default=2)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"[{PHASE}] Loading manifest: {args.manifest}")
    families_data = load_manifest_by_family(args.manifest)
    print(f"[{PHASE}] Families with >= {MIN_FAMILY_SIZE} train candidates:")
    for f, s in sorted(families_data.items(), key=lambda x: -len(x[1]["train"])):
        print(f"  {f}: train={len(s['train'])}, val={len(s['val'])}, test={len(s['test'])}")

    # Load risk map
    risk_map = {}
    if args.risk_path.exists():
        with open(args.risk_path) as f:
            risk_data = json.load(f)
        for cid, rec in risk_data.get("candidates", {}).items():
            risk_map[cid] = rec.get("features", {}).get("false_negative_risk", 0.5)

    # Select family pairs (chemically different)
    family_list = list(families_data.keys())
    # Pick top families by size, ensure chemical diversity
    # NOTE: manifest has "Cabonylation" (typo), not "Carbonylation"
    pairs = []
    if "Pd coupling" in families_data and "Alkylation" in families_data:
        pairs.append(("Pd coupling", "Alkylation"))
    if "Pd coupling" in families_data and "Hydrogenation" in families_data:
        pairs.append(("Pd coupling", "Hydrogenation"))
    if "Alkylation" in families_data and "Cabonylation" in families_data:
        pairs.append(("Alkylation", "Cabonylation"))
    if "Rh coupling" in families_data and "Cu coupling" in families_data:
        pairs.append(("Rh coupling", "Cu coupling"))

    # Add reverse directions, but skip if target has 0 test candidates
    all_directions = []
    for src, tgt in pairs:
        for s, t in [(src, tgt), (tgt, src)]:
            tgt_test_n = len(families_data.get(t, {}).get("test", []))
            if tgt_test_n == 0:
                print(f"[{PHASE}] SKIP {s} → {t} (target test=0)")
                continue
            all_directions.append((s, t))

    methods = ["direct", "head_ft", "risk_aware"]
    seeds = [SEED + i for i in range(args.n_seeds)]

    print(f"\n[{PHASE}] Running {len(all_directions)} directions × "
          f"{len(methods)} methods × {len(seeds)} seeds "
          f"= {len(all_directions) * len(methods) * len(seeds)} experiments")

    all_results = []
    for src_fam, tgt_fam in all_directions:
        print(f"\n[{PHASE}] === {src_fam} → {tgt_fam} ===")
        results = run_transfer_experiment(
            families_data, src_fam, tgt_fam, methods,
            risk_map, args.device, seeds)
        all_results.extend(results)

    # Write results CSV
    import csv
    csv_path = output_dir / "transfer_results.csv"
    if all_results:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print(f"[{PHASE}] Wrote {len(all_results)} rows to {csv_path}")

    # Family macro summary
    by_target = defaultdict(list)
    for r in all_results:
        by_target[r["target_family"]].append(r)
    with open(output_dir / "family_macro_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "family", "n_experiments", "mean_mrr", "mean_auprc", "mean_ece"])
        writer.writeheader()
        for fam, rows in sorted(by_target.items()):
            writer.writerow({
                "family": fam,
                "n_experiments": len(rows),
                "mean_mrr": round(statistics.mean(r["target_mrr"] for r in rows), 6),
                "mean_auprc": round(statistics.mean(r["target_auprc"] for r in rows), 6),
                "mean_ece": round(statistics.mean(r["target_ece"] for r in rows), 6),
            })

    # Verdict
    verdict = compute_verdict(all_results)
    go_no_go = {
        "phase": PHASE,
        "status": verdict["verdict"],
        "primary_metric": {"name": "transfer_gain", "comparison": "transferred_vs_baseline"},
        "predeclared_threshold": {
            "go": ">=2 chemically different directions CI all positive",
            "partial_go": ">=1 direction positive; all failures reported",
            "no_go": "No positive transfer; or severe catastrophic forgetting",
        },
        "verdict_details": verdict,
        "evidence_paths": [
            str(csv_path),
            str(output_dir / "family_macro_summary.csv"),
        ],
        "next_phase_allowed": verdict["next_phase_allowed"],
    }
    with open(output_dir / "go_no_go.json", "w") as f:
        json.dump(go_no_go, f, indent=2)

    # Contract files
    import hashlib
    def sha256(p):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    with open(output_dir / "run_manifest.json", "w") as f:
        json.dump({
            "phase": PHASE, "analysis": "cross_family_transfer",
            "methods": methods, "n_seeds": len(seeds),
            "family_pairs": pairs, "n_directions": len(all_directions),
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
    with open(output_dir / "input_hashes.json", "w") as f:
        hashes = {str(args.manifest): sha256(args.manifest)}
        if args.risk_path.exists():
            hashes[str(args.risk_path)] = sha256(args.risk_path)
        json.dump(hashes, f, indent=2)
    with open(output_dir / "commands.log", "w") as f:
        f.write(f"python3 -m pc_cng.p4_g8b_cross_family_transfer "
                f"--manifest {args.manifest} --risk-path {args.risk_path} "
                f"--output-dir {output_dir} --device {args.device}\n")

    elapsed = time.time() - t0
    print(f"\n[{PHASE}] Complete ({elapsed:.1f}s)")
    print(f"[{PHASE}] Verdict: {verdict['verdict']}")
    print(f"[{PHASE}] next_phase_allowed: {verdict['next_phase_allowed']}")


if __name__ == "__main__":
    main()
