"""P1-03 (part 2): OOD scaffold/template split evaluation.

Trains rerankers under three split regimes and compares reranking Top-1:

* ``random``: the existing column split (baseline).
* ``scaffold``: held-out BemisMurcko scaffolds of the product (RDKit).
* ``template``: held-out reaction templates, approximated by the canonical
  reactants SMILES (a coarse but deterministic template proxy; atom-mapped
  templates would be a drop-in replacement if rdChiral is available).

For each split type and seed, a FeasibilityMLP is trained on the train portion
and evaluated on the test portion.  The 10-seed mean Top-1 with bootstrap 95%
CI is reported per split type, and the OOD splits are compared against the
random-split baseline via paired bootstrap on per-seed deltas.

This closes the Section 9 E4/E6 OOD gap: it quantifies how much reranking
accuracy degrades when the test set contains scaffolds/templates unseen in
training.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .paired_reranking_significance import bootstrap_ci, mean, percentile
from .train_feasibility_mlp import (
    FeasibilityMLP,
    compute_metrics,
    featurize_rows,
    make_reaction_featurizer,
    predict,
    read_real_rows,
    set_seed,
)


def _try_rdkit():
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol

        return Chem, GetScaffoldForMol
    except Exception:
        return None, None


def murcko_scaffold(smiles: str) -> str:
    """Return the BemisMurcko scaffold SMILES for ``smiles`` (empty on failure)."""
    Chem, GetScaffoldForMol = _try_rdkit()
    if Chem is None:
        return ""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        scaffold = GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold) or ""
    except Exception:
        return ""


def reaction_template(reaction_smiles: str) -> str:
    """Coarse reaction template proxy.

    For atom-mapped reactions we extract the mapped product SMILES (the atom
    map encodes the reaction centre).  For unmapped reactions we fall back to
    the canonical reactants SMILES, which groups reactions that start from the
    same substrates.
    """
    Chem, _ = _try_rdkit()
    try:
        reactants, _, products = _split_reaction(reaction_smiles)
    except ValueError:
        return ""
    if Chem is None:
        return reactants
    # If the reaction is atom-mapped, the product carries the centre info.
    if ":" in products:
        prod_mol = Chem.MolFromSmiles(products)
        if prod_mol is not None:
            return Chem.MolToSmiles(prod_mol) or reactants
    # Otherwise use canonical reactants as a coarse template.
    return Chem.MolToSmiles(Chem.MolFromSmiles(reactants)) if Chem.MolFromSmiles(reactants) else reactants


def _split_reaction(reaction_smiles: str) -> Tuple[str, str, str]:
    """Split ``reactants>agents>products`` (agents may be empty)."""
    parts = reaction_smiles.split(">")
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    if len(parts) == 2:
        return parts[0], "", parts[1]
    raise ValueError(f"Cannot split reaction: {reaction_smiles!r}")


def assign_split_by_key(rows: List[Dict[str, object]], key_field: str, train_frac: float, seed: int) -> None:
    """Reassign the ``split`` field by holding out a fraction of unique keys.

    The split is deterministic per key (sorted), so the same key always lands
    in the same split regardless of row order.  ``seed`` only affects tie
    breaking when keys are not unique strings.
    """
    unique_keys = sorted({str(row.get(key_field, "")) for row in rows})
    if not unique_keys:
        return
    rng = random.Random(seed)
    # Shuffle deterministically so the held-out fraction is seed-dependent.
    shuffled = list(unique_keys)
    rng.shuffle(shuffled)
    n = len(shuffled)
    # Guarantee at least one key lands in train when train_frac > 0, so a
    # single unique key (e.g. all rows sharing one scaffold) does not collapse
    # into the test split (which would make training impossible).
    if train_frac > 0.0 and n > 0:
        train_cutoff = max(1, int(n * train_frac))
    else:
        train_cutoff = int(n * train_frac)
    val_cutoff = int(n * (train_frac + (1.0 - train_frac) / 2.0))
    if val_cutoff < train_cutoff:
        val_cutoff = train_cutoff
    train_keys = set(shuffled[:train_cutoff])
    val_keys = set(shuffled[train_cutoff:val_cutoff])
    for row in rows:
        key = str(row.get(key_field, ""))
        if key in train_keys:
            row["split"] = "train"
        elif key in val_keys:
            row["split"] = "val"
        else:
            row["split"] = "test"


def prepare_splits(
    real_rows: List[Dict[str, object]], train_frac: float, seed: int
) -> Dict[str, List[Dict[str, object]]]:
    """Return three row lists (random, scaffold, template) with reassigned splits.

    The ``random`` variant keeps the original ``split`` column.  The
    ``scaffold`` and ``template`` variants overwrite ``split`` based on the
    BemisMurcko scaffold / reaction template of each row.
    """
    random_rows = [dict(row) for row in real_rows]

    scaffold_rows = [dict(row) for row in real_rows]
    for row in scaffold_rows:
        row["_scaffold"] = murcko_scaffold(str(row.get("products", "")))
    assign_split_by_key(scaffold_rows, "_scaffold", train_frac, seed)

    template_rows = [dict(row) for row in real_rows]
    for row in template_rows:
        row["_template"] = reaction_template(str(row.get("reaction_smiles", "")))
    assign_split_by_key(template_rows, "_template", train_frac, seed)

    return {"random": random_rows, "scaffold": scaffold_rows, "template": template_rows}


def train_and_eval_split(
    rows: List[Dict[str, object]],
    seed: int,
    epochs: int,
    hidden_dim: int,
    n_bits: int,
    batch_size: int,
    device_name: str | None,
) -> Dict[str, object]:
    """Train FeasibilityMLP on the train split and evaluate reranking Top-1."""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    from .evaluate_candidate_reranking import build_real_candidate_rows, ranking_metrics

    set_seed(seed)
    train_rows = [r for r in rows if r.get("split") == "train"]
    val_rows = [r for r in rows if r.get("split") == "val"]
    test_rows = [r for r in rows if r.get("split") == "test"]
    if not train_rows or not test_rows:
        return {"top1": 0.0, "mrr": 0.0, "ndcg": 0.0, "groups": 0, "train_rows": len(train_rows), "test_rows": len(test_rows)}

    featurizer = make_reaction_featurizer(feature_mode="morgan", n_bits=n_bits, fp_mode="binary")
    x_train, y_train, _, _ = featurize_rows(train_rows, featurizer)
    x_val, y_val, _, _ = featurize_rows(val_rows, featurizer)
    if len(x_train) == 0:
        return {"top1": 0.0, "mrr": 0.0, "ndcg": 0.0, "groups": 0, "train_rows": 0, "test_rows": len(test_rows)}

    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = FeasibilityMLP(in_dim=x_train.shape[1], hidden_dim=hidden_dim, dropout=0.15).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    neg = max(float((y_train == 0).sum()), 1.0)
    pos = max(float((y_train == 1).sum()), 1.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], device=device))
    loader = DataLoader(
        TensorDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    best_val = -1.0
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
        if len(y_val):
            val_scores = predict(model, x_val, device, batch_size)
            val_metrics = compute_metrics(y_val, val_scores)
            val_key = val_metrics.get("roc_auc", val_metrics.get("accuracy", 0.0))
            if val_key == val_key and val_key > best_val:
                best_val = float(val_key)
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)

    # Build reranking candidates from test rows and score them.
    # build_real_candidate_rows expects the original CSV schema (string labels).
    test_csv_rows: List[Dict[str, str]] = []
    for r in test_rows:
        test_csv_rows.append(
            {
                "source_id": str(r.get("source_id", "")),
                "reaction_smiles": str(r.get("reaction_smiles", "")),
                "reactants": str(r.get("reactants", "")),
                "products": str(r.get("products", "")),
                "label_type": "positive" if int(r.get("label", 0)) == 1 else "real_negative",
                "source": str(r.get("dataset", "")),
                "split": "test",
                "reaction_class": str(r.get("reaction_class", "")),
                "_input_path": "ood_eval",
            }
        )
    candidates = build_real_candidate_rows(
        real_rows=test_csv_rows,
        group_by="reactants",
        candidate_scope="same_split",
    )
    if not candidates:
        return {"top1": 0.0, "mrr": 0.0, "ndcg": 0.0, "groups": 0, "train_rows": len(train_rows), "test_rows": len(test_rows)}

    # Score candidates with the trained model.  Preserve group_id through
    # featurization so we can compute per-group ranking metrics afterwards.
    cand_rows = [
        {"reaction_smiles": c["reaction_smiles"], "label": c["label"], "group_id": c["group_id"]}
        for c in candidates
    ]
    x_cand, _, _, kept = featurize_rows(cand_rows, featurizer)
    if len(kept) == 0:
        return {"top1": 0.0, "mrr": 0.0, "ndcg": 0.0, "groups": 0, "train_rows": len(train_rows), "test_rows": len(test_rows)}
    scores = predict(model, x_cand, device, batch_size)
    for row, score in zip(kept, scores.tolist()):
        row["score"] = float(score)
    metrics = ranking_metrics(kept)
    return {
        "top1": float(metrics.get("top1", 0.0)),
        "mrr": float(metrics.get("mrr", 0.0)),
        "ndcg": float(metrics.get("ndcg", 0.0)),
        "groups": int(metrics.get("groups", 0)),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
    }


def run_ood_eval(
    seeds: Sequence[int],
    output_dir: str,
    real_csvs: Sequence[str],
    epochs: int,
    hidden_dim: int,
    n_bits: int,
    batch_size: int,
    device_name: str | None,
    train_frac: float,
    bootstrap_iterations: int,
    smoke: bool,
) -> Dict[str, object]:
    os.makedirs(output_dir, exist_ok=True)

    # Load real rows once (splits will be reassigned per seed).
    base_rows: List[Dict[str, object]] = []
    for path in real_csvs:
        rows, _ = read_real_rows(path)
        base_rows.extend(rows)
    print(f"[ood] loaded {len(base_rows)} real rows from {len(real_csvs)} csv files")

    split_types = ["random", "scaffold", "template"]
    per_seed_records: List[Dict[str, object]] = []
    for seed in seeds:
        t0 = time.time()
        splits = prepare_splits(base_rows, train_frac=train_frac, seed=seed)
        seed_record: Dict[str, object] = {"seed": seed}
        for split_type in split_types:
            rows = splits[split_type]
            metrics = train_and_eval_split(
                rows=rows,
                seed=seed,
                epochs=epochs,
                hidden_dim=hidden_dim,
                n_bits=n_bits,
                batch_size=batch_size,
                device_name=device_name,
            )
            seed_record[split_type] = metrics
            print(
                f"[ood] seed={seed} split={split_type} top1={metrics['top1']:.4f} "
                f"groups={metrics['groups']} train={metrics['train_rows']} test={metrics['test_rows']}"
            )
        seed_record["elapsed_sec"] = time.time() - t0
        per_seed_records.append(seed_record)
        if smoke:
            break

    # Aggregate per split type with bootstrap CI.
    aggregate: Dict[str, Dict[str, object]] = {}
    for split_type in split_types:
        top1_values = [float(rec[split_type]["top1"]) for rec in per_seed_records]
        ci_low, ci_high = (
            bootstrap_ci(top1_values, bootstrap_iterations, 20260719)
            if top1_values
            else (0.0, 0.0)
        )
        aggregate[split_type] = {
            "n_seeds": len(top1_values),
            "top1_mean": mean(top1_values),
            "top1_ci95_low": ci_low,
            "top1_ci95_high": ci_high,
            "top1_per_seed": top1_values,
        }

    # Paired deltas: OOD - random (per seed).
    paired_deltas: Dict[str, Dict[str, object]] = {}
    for split_type in ["scaffold", "template"]:
        deltas = [
            float(rec[split_type]["top1"]) - float(rec["random"]["top1"])
            for rec in per_seed_records
        ]
        ci_low, ci_high = (
            bootstrap_ci(deltas, bootstrap_iterations, 20260719 + hash(split_type) % 1000)
            if deltas
            else (0.0, 0.0)
        )
        paired_deltas[split_type] = {
            "mean_delta": mean(deltas),
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "per_seed_deltas": deltas,
            "interpretation": "negative delta => OOD split is harder than random",
        }

    payload = {
        "task": "ood_scaffold_template_split",
        "real_csvs": list(real_csvs),
        "seeds": list(seeds),
        "config": {
            "epochs": epochs,
            "hidden_dim": hidden_dim,
            "n_bits": n_bits,
            "batch_size": batch_size,
            "train_frac": train_frac,
            "bootstrap_iterations": bootstrap_iterations,
            "smoke": smoke,
        },
        "per_seed": per_seed_records,
        "aggregate": aggregate,
        "paired_deltas_vs_random": paired_deltas,
        "split_definitions": {
            "random": "original split column from normalized CSV",
            "scaffold": "BemisMurcko scaffold of the product (held out by scaffold)",
            "template": "canonical reactants SMILES proxy for reaction template (held out by template)",
        },
    }
    with open(os.path.join(output_dir, "ood_split_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(
        f"[ood] random top1={aggregate['random']['top1_mean']:.4f} "
        f"scaffold top1={aggregate['scaffold']['top1_mean']:.4f} "
        f"template top1={aggregate['template']['top1_mean']:.4f}"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="P1-03 OOD scaffold/template split evaluation")
    parser.add_argument(
        "--real-csv",
        action="append",
        default=None,
        help="Normalized CSV to evaluate.  Defaults to regiosqm20 + hitea_full.",
    )
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--n-bits", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", default=None)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--research-root", default=None)
    args = parser.parse_args()

    research_root = args.research_root or os.getcwd()
    if args.real_csv:
        real_csvs = args.real_csv
    else:
        real_csvs = [
            os.path.join(research_root, "data/processed/regiosqm20_normalized.csv"),
            os.path.join(research_root, "data/processed/hitea_full_normalized.csv"),
        ]

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if args.smoke:
        seeds = seeds[:1]

    run_ood_eval(
        seeds=seeds,
        output_dir=args.output_dir,
        real_csvs=real_csvs,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        n_bits=args.n_bits,
        batch_size=args.batch_size,
        device_name=args.device,
        train_frac=args.train_frac,
        bootstrap_iterations=args.bootstrap_iterations,
        smoke=args.smoke,
    )


if __name__ == "__main__":
    main()
