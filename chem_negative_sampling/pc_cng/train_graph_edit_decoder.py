"""First trainable graph-edit decoder entry point.

This is a scale-up scaffold. It trains a small neural classifier over reaction
features to predict failure/edit types from generated candidates. The model is
not the final publishable GNN decoder, but it establishes the training,
checkpointing, and evaluation contract for the later atom-mapped graph model.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List, Tuple

from .reranker import FEATURE_NAMES, featurize_reaction


def _load_torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "PyTorch is required for train_graph_edit_decoder.py. "
            "Run scripts_setup_remote_env.sh in a user venv first."
        ) from exc
    return torch, nn, DataLoader, TensorDataset


def load_rows(path: str) -> Tuple[List[List[float]], List[int], Dict[str, int]]:
    label_to_id: Dict[str, int] = {}
    features: List[List[float]] = []
    labels: List[int] = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            reaction = row.get("candidate_reaction") or row.get("reaction_smiles")
            if not reaction:
                continue
            label_name = row.get("failure_type") or row.get("task") or "unknown"
            if label_name not in label_to_id:
                label_to_id[label_name] = len(label_to_id)
            features.append(featurize_reaction(reaction))
            labels.append(label_to_id[label_name])
    return features, labels, label_to_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV with candidate_reaction/reaction_smiles and failure_type")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    torch, nn, DataLoader, TensorDataset = _load_torch()
    os.makedirs(args.output_dir, exist_ok=True)
    x_rows, y_rows, label_to_id = load_rows(args.input)
    if not x_rows or len(label_to_id) < 2:
        raise ValueError("Need at least two edit/failure classes for training")

    x = torch.tensor(x_rows, dtype=torch.float32)
    y = torch.tensor(y_rows, dtype=torch.long)
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = nn.Sequential(
        nn.Linear(len(FEATURE_NAMES), 128),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(128, len(label_to_id)),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    history = []
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        total = 0
        correct = 0
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_y)
            total += len(batch_y)
            correct += int((logits.argmax(dim=-1) == batch_y).sum().item())
        history.append({"epoch": epoch, "loss": total_loss / total, "accuracy": correct / total})

    checkpoint = {
        "state_dict": model.state_dict(),
        "feature_names": FEATURE_NAMES,
        "label_to_id": label_to_id,
        "history": history,
        "device": str(device),
    }
    torch.save(checkpoint, os.path.join(args.output_dir, "graph_edit_decoder_mlp.pt"))
    with open(os.path.join(args.output_dir, "train_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "input": args.input,
                "rows": len(x_rows),
                "label_to_id": label_to_id,
                "final": history[-1],
                "device": str(device),
                "note": "MLP scaffold for training contract; replace with atom-mapped graph decoder for final paper.",
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    print(json.dumps({"rows": len(x_rows), "label_to_id": label_to_id, "final": history[-1]}, indent=2))


if __name__ == "__main__":
    main()
