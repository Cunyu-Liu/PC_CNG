"""Evaluation script for the learned GNN graph edit decoder (P1-05).

Loads a trained GNN decoder, generates boundary negatives on a test set,
and compares them with the rule-based baseline on:
  - diversity (unique negative count / total)
  - validity (RDKit-parseable + atom balance)
  - downstream reranking Test Top-1 (optional, trains a small logistic ranker)

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m pc_cng.evaluate_learned_graph_edit_decoder \\
        --checkpoint results/learned_graph_edit_decoder_smoke_20260719/learned_graph_edit_decoder.pt \\
        --test-data data/processed/regiosqm20_normalized.csv \\
        --output-dir results/learned_graph_edit_decoder_smoke_20260719/eval \\
        --limit 50 --top-k 2 --map-unmapped
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .chem_utils import atom_balance_score, canonicalize_reaction, is_valid_smiles, split_reaction
from .learned_graph_edit_decoder import (
    GeneratedNegative,
    LearnedGraphEditDecoder,
    generate_boundary_negatives,
    load_checkpoint,
)
from .reaction_boundary_generator import RXNMapperAdapter
from .reranker import LogisticReactionRanker, evaluate_ranking


def load_test_rows(path: str, limit: Optional[int] = None) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not row.get("reaction_smiles"):
                continue
            if limit is not None and len(rows) >= limit:
                break
            rows.append(row)
    return rows


def _is_valid_negative(negative: GeneratedNegative) -> bool:
    """Check that a generated negative is a valid reaction (RDKit-parseable)."""
    try:
        reactants, _, products = split_reaction(negative.candidate_reaction)
    except ValueError:
        return False
    return is_valid_smiles(reactants) and is_valid_smiles(products)


def _atom_balance(negative: GeneratedNegative) -> float:
    try:
        reactants, _, products = split_reaction(negative.candidate_reaction)
    except ValueError:
        return 0.0
    return atom_balance_score(reactants, products)


def evaluate_generated_negatives(
    negatives: Sequence[GeneratedNegative],
) -> Dict[str, float]:
    """Compute diversity and validity metrics for a set of generated negatives."""
    total = len(negatives)
    if total == 0:
        return {"total": 0, "unique": 0, "diversity": 0.0, "valid": 0, "validity_rate": 0.0,
                "mean_atom_balance": 0.0}
    canonical = set()
    valid_count = 0
    atom_balances: List[float] = []
    for neg in negatives:
        canon = canonicalize_reaction(neg.candidate_reaction)
        if canon:
            canonical.add(canon)
        if _is_valid_negative(neg):
            valid_count += 1
        atom_balances.append(_atom_balance(neg))
    return {
        "total": total,
        "unique": len(canonical),
        "diversity": len(canonical) / total,
        "valid": valid_count,
        "validity_rate": valid_count / total,
        "mean_atom_balance": sum(atom_balances) / len(atom_balances),
    }


def generate_rule_based_negatives(
    rows: Sequence[Dict[str, str]],
    mapper: Optional[RXNMapperAdapter],
    map_unmapped: bool,
    top_k: int,
    max_anchor_distance: int = 6,
) -> List[GeneratedNegative]:
    """Generate negatives using the rule-based decoder (random candidate ranking).

    This is a baseline that picks candidates uniformly at random (no learned scoring),
    matching the candidate generation of ``reaction_center_edit_decoder.build_edit_candidate_groups``
    but without the learned ranker.
    """
    from .learned_graph_edit_decoder import featurize_atom_mapped_reaction
    from .reaction_center_edit_decoder import move_formed_bond_in_product
    from .chem_utils import join_reaction

    negatives: List[GeneratedNegative] = []
    for row in rows:
        reaction = row.get("reaction_smiles", "")
        source_id = row.get("source_id", "")
        graphs, reason = featurize_atom_mapped_reaction(
            reaction,
            source_id=source_id,
            mapper=mapper,
            map_unmapped=map_unmapped,
            max_anchor_distance=max_anchor_distance,
        )
        if reason != "ok":
            continue
        for graph in graphs:
            candidates = [
                (idx, graph.atom_map_nums[idx].item())
                for idx in graph.candidate_anchor_indices
                if idx != graph.true_anchor_idx and graph.atom_map_nums[idx].item() != 0
            ]
            # rule-based: pick first K candidates (closest by distance, as returned by _candidate_anchor_atoms)
            for rank, (cand_idx, cand_map) in enumerate(candidates[:top_k], start=1):
                candidate_product = move_formed_bond_in_product(
                    graph.product, graph.fragment_map, graph.true_anchor_map, int(cand_map)
                )
                if not candidate_product:
                    continue
                candidate_reaction = join_reaction(graph.reactants, candidate_product, "")
                negatives.append(
                    GeneratedNegative(
                        source_id=graph.source_id,
                        pair_id=graph.pair_id,
                        positive_reaction=graph.mapped_reaction,
                        candidate_reaction=candidate_reaction,
                        reactants=graph.reactants,
                        parent_product=graph.product,
                        candidate_product=candidate_product,
                        fragment_map=graph.fragment_map,
                        true_anchor_map=graph.true_anchor_map,
                        candidate_anchor_map=int(cand_map),
                        decoder_score=0.0,
                        decoder_rank=rank,
                    )
                )
    return negatives


def evaluate_downstream_reranking(
    negatives: Sequence[GeneratedNegative],
    test_rows: Sequence[Dict[str, str]],
) -> Dict[str, float]:
    """Train a logistic ranker on (positive + generated negative) and evaluate on held-out test.

    This is a minimal sanity check: can the generated negatives serve as a
    training signal to distinguish positive reactions from negatives?
    """
    train_rows: List[Dict[str, object]] = []
    for neg in negatives:
        train_rows.append({"source_id": neg.source_id, "reaction_smiles": neg.candidate_reaction, "label": 0})
        train_rows.append({"source_id": neg.source_id, "reaction_smiles": neg.positive_reaction, "label": 1})

    test_eval: List[Dict[str, object]] = []
    for row in test_rows:
        source_id = row.get("source_id", "")
        reaction = row.get("reaction_smiles", "")
        test_eval.append({"source_id": source_id, "reaction_smiles": reaction, "label": 1})

    if not train_rows:
        return {"train_rows": 0, "test_rows": 0, "top1": 0.0, "note": "no training data"}

    model = LogisticReactionRanker(learning_rate=0.2, l2=1e-4, epochs=100)
    model.fit(train_rows)

    # evaluate: pair each positive with a random generated negative
    neg_by_source: Dict[str, List[GeneratedNegative]] = defaultdict(list)
    for neg in negatives:
        neg_by_source[neg.source_id].append(neg)

    eval_rows: List[Dict[str, object]] = []
    for row in test_rows:
        source_id = row.get("source_id", "")
        reaction = row.get("reaction_smiles", "")
        eval_rows.append({"source_id": source_id, "reaction_smiles": reaction, "label": 1})
        # pair with a random negative from the same source (or any)
        pool = neg_by_source.get(source_id) or list(negatives)
        if pool:
            import random
            rng = random.Random(42)
            neg = rng.choice(pool)
            eval_rows.append({"source_id": source_id, "reaction_smiles": neg.candidate_reaction, "label": 0})

    metrics = evaluate_ranking(model, eval_rows)
    return {
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "top1": metrics.top1,
        "mrr": metrics.mrr,
        "ndcg": metrics.ndcg,
        "groups": metrics.groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate learned GNN graph edit decoder")
    parser.add_argument("--checkpoint", required=True, help="Path to learned_graph_edit_decoder.pt")
    parser.add_argument("--test-data", required=True, help="CSV with reaction_smiles column")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--map-unmapped", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-anchor-distance", type=int, default=6)
    parser.add_argument("--downstream-reranking", action="store_true", help="Run downstream reranking eval")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(args.gpu)
    else:
        device = torch.device("cpu")
    print(f"[device] {device}")

    model = load_checkpoint(args.checkpoint, device)
    print(f"[model] loaded from {args.checkpoint}")

    rows = load_test_rows(args.test_data, limit=args.limit)
    print(f"[data] {len(rows)} test rows")

    mapper = RXNMapperAdapter() if args.map_unmapped else None

    # GNN negatives
    print("[generate] GNN negatives...")
    gnn_negatives: List[GeneratedNegative] = []
    skip_reasons: Dict[str, int] = {}
    for idx, row in enumerate(rows):
        negs, reason = generate_boundary_negatives(
            model,
            row.get("reaction_smiles", ""),
            source_id=row.get("source_id", ""),
            top_k=args.top_k,
            mapper=mapper,
            map_unmapped=args.map_unmapped,
            device=device,
            max_anchor_distance=args.max_anchor_distance,
        )
        if reason != "ok":
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
        else:
            gnn_negatives.extend(negs)
        if (idx + 1) % 50 == 0:
            print(f"[generate] {idx+1}/{len(rows)} -> {len(gnn_negatives)} negatives")
    print(f"[generate] GNN: {len(gnn_negatives)} negatives (skip_reasons={skip_reasons})")

    # rule-based negatives
    print("[generate] rule-based negatives...")
    rule_negatives = generate_rule_based_negatives(
        rows, mapper, args.map_unmapped, args.top_k, args.max_anchor_distance
    )
    print(f"[generate] rule: {len(rule_negatives)} negatives")

    gnn_metrics = evaluate_generated_negatives(gnn_negatives)
    rule_metrics = evaluate_generated_negatives(rule_negatives)
    print(f"[metrics] GNN: {gnn_metrics}")
    print(f"[metrics] rule: {rule_metrics}")

    summary: Dict[str, object] = {
        "checkpoint": args.checkpoint,
        "test_data": args.test_data,
        "limit": args.limit,
        "top_k": args.top_k,
        "skip_reasons": skip_reasons,
        "gnn_metrics": gnn_metrics,
        "rule_metrics": rule_metrics,
        "comparison": {
            "diversity_diff": gnn_metrics["diversity"] - rule_metrics["diversity"],
            "validity_diff": gnn_metrics["validity_rate"] - rule_metrics["validity_rate"],
        },
    }

    # write negatives CSV
    neg_path = os.path.join(args.output_dir, "gnn_negatives.csv")
    _write_negatives_csv(gnn_negatives, neg_path)
    rule_neg_path = os.path.join(args.output_dir, "rule_negatives.csv")
    _write_negatives_csv(rule_negatives, rule_neg_path)

    # downstream reranking (optional)
    if args.downstream_reranking and gnn_negatives:
        print("[eval] downstream reranking (GNN negatives)...")
        gnn_downstream = evaluate_downstream_reranking(gnn_negatives, rows)
        summary["gnn_downstream"] = gnn_downstream  # type: ignore[assignment]
        print(f"[eval] GNN downstream: {gnn_downstream}")
    if args.downstream_reranking and rule_negatives:
        print("[eval] downstream reranking (rule negatives)...")
        rule_downstream = evaluate_downstream_reranking(rule_negatives, rows)
        summary["rule_downstream"] = rule_downstream  # type: ignore[assignment]
        print(f"[eval] rule downstream: {rule_downstream}")

    with open(os.path.join(args.output_dir, "eval_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))


def _write_negatives_csv(negatives: Sequence[GeneratedNegative], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "source_id", "pair_id", "positive_reaction", "candidate_reaction",
            "fragment_map", "true_anchor_map", "candidate_anchor_map",
            "decoder_score", "decoder_rank",
        ])
        for neg in negatives:
            writer.writerow([
                neg.source_id, neg.pair_id, neg.positive_reaction, neg.candidate_reaction,
                neg.fragment_map, neg.true_anchor_map, neg.candidate_anchor_map,
                f"{neg.decoder_score:.4f}", neg.decoder_rank,
            ])


if __name__ == "__main__":
    main()
