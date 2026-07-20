"""Training script for the learned GNN-based graph edit decoder (P1-05).

Trains a pure-PyTorch MPNN to rank the observed reaction-center anchor above
plausible alternatives. Replaces the shallow MLP in ``train_graph_edit_decoder.py``
and ``train_reaction_center_edit_decoder.py`` for the final paper.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m pc_cng.train_learned_graph_edit_decoder \\
        --train-data data/processed/uspto_openmolecules_normalized.csv \\
        --val-data data/processed/regiosqm20_normalized.csv \\
        --output-dir results/learned_graph_edit_decoder_smoke_20260719 \\
        --epochs 1 --batch-size 16 --limit-train 100 --limit-val 50 \\
        --seed 20260719
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from .learned_graph_edit_decoder import (
    ATOM_FEAT_DIM,
    BOND_FEAT_DIM,
    BatchedGraph,
    LearnedGraphEditDecoder,
    ReactionGraphData,
    collate_graphs,
    featurize_atom_mapped_reaction,
    pairwise_margin_loss,
    predict_batch_anchor_scores,
    save_checkpoint,
)
from .reaction_boundary_generator import RXNMapperAdapter


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_reaction_rows(path: str, limit: Optional[int] = None) -> List[Dict[str, str]]:
    """Load reaction CSV rows (expects columns: reaction_smiles, source_id, split)."""
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


def featurize_reactions(
    rows: Sequence[Dict[str, str]],
    mapper: Optional[RXNMapperAdapter],
    map_unmapped: bool,
    max_anchor_distance: int,
    max_candidates_per_pair: int,
    cache_path: Optional[str] = None,
) -> Tuple[List[List[ReactionGraphData]], Dict[str, int]]:
    """Featurize a list of reaction rows into per-reaction graph data.

    Returns (per_reaction_graphs, stats). ``per_reaction_graphs[i]`` is the list
    of formed-bond groups for row ``i`` (may be empty on failure).
    """
    if cache_path and os.path.exists(cache_path):
        print(f"[featurize] loading cache: {cache_path}")
        cached = torch.load(cache_path, weights_only=False)
        return cached["graphs"], cached["stats"]

    stats: Dict[str, int] = {"total": 0, "ok": 0, "skipped": 0}
    skip_reasons: Dict[str, int] = {}
    all_graphs: List[List[ReactionGraphData]] = []
    t0 = time.time()
    for idx, row in enumerate(rows):
        stats["total"] += 1
        reaction = row.get("reaction_smiles", "")
        source_id = row.get("source_id", f"row_{idx}")
        graphs, reason = featurize_atom_mapped_reaction(
            reaction,
            source_id=source_id,
            split=row.get("split", "train"),
            label_type=row.get("label_type", "positive"),
            mapper=mapper,
            map_unmapped=map_unmapped,
            max_anchor_distance=max_anchor_distance,
            max_candidates_per_pair=max_candidates_per_pair,
        )
        if reason != "ok":
            stats["skipped"] += 1
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            all_graphs.append([])
        else:
            stats["ok"] += 1
            all_graphs.append(graphs)
        if (idx + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"[featurize] {idx+1}/{len(rows)} ok={stats['ok']} skipped={stats['skipped']} ({elapsed:.1f}s)")
    stats["skip_reasons"] = dict(skip_reasons)  # type: ignore[assignment]
    print(f"[featurize] done: {stats['ok']}/{stats['total']} ok in {time.time()-t0:.1f}s")

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        torch.save({"graphs": all_graphs, "stats": stats}, cache_path)
        print(f"[featurize] cached to {cache_path}")
    return all_graphs, stats


def flatten_graphs(per_reaction_graphs: Sequence[List[ReactionGraphData]]) -> List[ReactionGraphData]:
    """Flatten per-reaction graph lists into a single list of valid graphs."""
    out: List[ReactionGraphData] = []
    for graphs in per_reaction_graphs:
        out.extend(graphs)
    return out


def compute_batch_loss(
    model: LearnedGraphEditDecoder,
    batch: List[ReactionGraphData],
    device: torch.device,
    margin: float,
) -> Tuple[torch.Tensor, int]:
    """Compute pairwise margin loss for a batch of graphs.

    Returns (loss, num_pairs).
    """
    if not batch:
        return torch.tensor(0.0, device=device, requires_grad=True), 0
    batched = collate_graphs(batch)
    all_scores = model(
        batched.atom_features.to(device),
        batched.edge_index.to(device),
        batched.edge_features.to(device),
        batched.batch_idx.to(device),
        num_graphs=len(batch),
    )

    total_loss = torch.tensor(0.0, device=device)
    num_pairs = 0
    for graph_idx, graph in enumerate(batch):
        candidates = batched.candidate_anchor_indices_per_graph[graph_idx]
        true_abs = batched.true_anchor_indices[graph_idx]
        if true_abs not in candidates:
            continue
        true_local = candidates.index(true_abs)
        cand_tensor = torch.tensor(candidates, dtype=torch.long, device=device)
        scores = all_scores.index_select(0, cand_tensor)
        true_score = scores[true_local]
        # candidates excluding true
        mask = torch.ones(len(candidates), dtype=torch.bool, device=device)
        mask[true_local] = False
        cand_scores = scores[mask]
        if cand_scores.numel() == 0:
            continue
        loss = pairwise_margin_loss(
            true_score.unsqueeze(0),
            cand_scores.unsqueeze(0),
            margin=margin,
        )
        total_loss = total_loss + loss
        num_pairs += cand_scores.numel()
    if num_pairs > 0:
        total_loss = total_loss / len(batch)
    return total_loss, num_pairs


def evaluate_top1(
    model: LearnedGraphEditDecoder,
    graphs: Sequence[ReactionGraphData],
    device: torch.device,
    batch_size: int = 32,
) -> Dict[str, float]:
    """Evaluate Top-1 accuracy on a set of graphs.

    A graph is correct if the true anchor has the highest score among all candidates.
    """
    model.eval()
    if not graphs:
        return {"top1": 0.0, "num_graphs": 0}
    correct = 0
    total = 0
    with torch.no_grad():
        for start in range(0, len(graphs), batch_size):
            batch = list(graphs[start : start + batch_size])
            batched = collate_graphs(batch)
            all_scores = model(
                batched.atom_features.to(device),
                batched.edge_index.to(device),
                batched.edge_features.to(device),
                batched.batch_idx.to(device),
                num_graphs=len(batch),
            ).cpu()
            for graph_idx, graph in enumerate(batch):
                candidates = batched.candidate_anchor_indices_per_graph[graph_idx]
                true_abs = batched.true_anchor_indices[graph_idx]
                if true_abs not in candidates:
                    continue
                true_local = candidates.index(true_abs)
                cand_tensor = torch.tensor(candidates, dtype=torch.long)
                scores = all_scores.index_select(0, cand_tensor)
                pred_idx = int(scores.argmax().item())
                if pred_idx == true_local:
                    correct += 1
                total += 1
    return {"top1": correct / max(total, 1), "num_graphs": total}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train learned GNN graph edit decoder")
    parser.add_argument("--train-data", required=True, help="CSV with reaction_smiles column")
    parser.add_argument("--val-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-message-passing-rounds", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--gpu", type=int, default=0, help="GPU index (do not use 4)")
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--map-unmapped", action="store_true", help="Use RXNMapper for unmapped reactions")
    parser.add_argument("--max-anchor-distance", type=int, default=6)
    parser.add_argument("--max-candidates-per-pair", type=int, default=8)
    parser.add_argument("--cache-train", default="", help="Path to featurization cache (torch save)")
    parser.add_argument("--cache-val", default="")
    parser.add_argument("--eval-batch-size", type=int, default=32)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    # device
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(args.gpu)
    else:
        device = torch.device("cpu")
    print(f"[device] using {device} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')})")

    # data
    print("[data] loading train/val CSVs...")
    train_rows = load_reaction_rows(args.train_data, limit=args.limit_train)
    val_rows = load_reaction_rows(args.val_data, limit=args.limit_val)
    print(f"[data] train rows={len(train_rows)} val rows={len(val_rows)}")

    mapper = RXNMapperAdapter() if args.map_unmapped else None
    train_cache = args.cache_train or None
    val_cache = args.cache_val or None

    print("[featurize] train...")
    train_per_reaction, train_stats = featurize_reactions(
        train_rows, mapper, args.map_unmapped,
        args.max_anchor_distance, args.max_candidates_per_pair,
        cache_path=train_cache,
    )
    print("[featurize] val...")
    val_per_reaction, val_stats = featurize_reactions(
        val_rows, mapper, args.map_unmapped,
        args.max_anchor_distance, args.max_candidates_per_pair,
        cache_path=val_cache,
    )

    train_graphs = flatten_graphs(train_per_reaction)
    val_graphs = flatten_graphs(val_per_reaction)
    print(f"[graphs] train={len(train_graphs)} val={len(val_graphs)}")

    if not train_graphs:
        raise RuntimeError("No valid train graphs. Check data or use --map-unmapped.")
    if not val_graphs:
        print("[warn] no val graphs; using train subset for validation")
        val_graphs = train_graphs[:50]

    # model
    model = LearnedGraphEditDecoder(
        atom_feat_dim=ATOM_FEAT_DIM,
        bond_feat_dim=BOND_FEAT_DIM,
        hidden_dim=args.hidden_dim,
        num_rounds=args.num_message_passing_rounds,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: List[Dict] = []
    best_top1 = -1.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(train_graphs)
        epoch_loss = 0.0
        epoch_pairs = 0
        num_batches = 0
        t0 = time.time()
        for start in range(0, len(train_graphs), args.batch_size):
            batch = train_graphs[start : start + args.batch_size]
            optimizer.zero_grad()
            loss, pairs = compute_batch_loss(model, batch, device, margin=args.margin)
            if pairs == 0:
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss += float(loss.item())
            epoch_pairs += pairs
            num_batches += 1
        train_loss = epoch_loss / max(num_batches, 1)
        val_metrics = evaluate_top1(model, val_graphs, device, batch_size=args.eval_batch_size)
        elapsed = time.time() - t0
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_pairs": epoch_pairs,
            "val_top1": val_metrics["top1"],
            "val_num_graphs": val_metrics["num_graphs"],
            "elapsed_sec": round(elapsed, 2),
        }
        history.append(record)
        print(f"[epoch {epoch}] loss={train_loss:.4f} pairs={epoch_pairs} val_top1={val_metrics['top1']:.4f} ({elapsed:.1f}s)")

        if val_metrics["top1"] > best_top1:
            best_top1 = val_metrics["top1"]
            best_epoch = epoch
            save_checkpoint(
                model,
                os.path.join(args.output_dir, "learned_graph_edit_decoder.pt"),
                extra={
                    "epoch": epoch,
                    "val_top1": best_top1,
                    "args": vars(args),
                    "train_stats": train_stats,
                    "val_stats": val_stats,
                },
            )

    # final summary
    summary = {
        "output_dir": args.output_dir,
        "args": vars(args),
        "train_stats": train_stats,
        "val_stats": val_stats,
        "train_graphs": len(train_graphs),
        "val_graphs": len(val_graphs),
        "history": history,
        "best_top1": best_top1,
        "best_epoch": best_epoch,
        "final_top1": history[-1]["val_top1"] if history else 0.0,
        "device": str(device),
    }
    with open(os.path.join(args.output_dir, "train_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    # save val predictions
    _write_val_predictions(model, val_graphs, device, os.path.join(args.output_dir, "val_predictions.csv"))

    print(json.dumps({
        "best_top1": best_top1,
        "best_epoch": best_epoch,
        "train_graphs": len(train_graphs),
        "val_graphs": len(val_graphs),
    }, indent=2))


def _write_val_predictions(
    model: LearnedGraphEditDecoder,
    graphs: Sequence[ReactionGraphData],
    device: torch.device,
    path: str,
) -> None:
    """Write per-graph predictions vs ground truth."""
    model.eval()
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "pair_id", "source_id", "true_anchor_rank", "true_anchor_score",
            "num_candidates", "top1_correct", "true_anchor_map",
        ])
        with torch.no_grad():
            for graph in graphs:
                batch_idx = torch.zeros(graph.atom_features.shape[0], dtype=torch.long)
                scores = model(
                    graph.atom_features.to(device),
                    graph.edge_index.to(device),
                    graph.edge_features.to(device),
                    batch_idx.to(device),
                    num_graphs=1,
                ).cpu()
                cand_idx = torch.tensor(graph.candidate_anchor_indices, dtype=torch.long)
                cand_scores = scores.index_select(0, cand_idx)
                true_local = graph.candidate_anchor_indices.index(graph.true_anchor_idx)
                ranking = cand_scores.argsort(descending=True)
                true_rank = int((ranking == true_local).nonzero(as_tuple=True)[0].item()) + 1
                writer.writerow([
                    graph.pair_id,
                    graph.source_id,
                    true_rank,
                    f"{float(cand_scores[true_local].item()):.4f}",
                    len(graph.candidate_anchor_indices),
                    1 if true_rank == 1 else 0,
                    graph.true_anchor_map,
                ])


if __name__ == "__main__":
    main()
