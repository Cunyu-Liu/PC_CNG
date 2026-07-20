"""Generate boundary negatives with a trained reaction-center edit decoder."""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List

import torch

from .reaction_boundary_generator import BoundaryCandidate
from .reaction_center_edit_decoder import FEATURE_NAMES, build_edit_candidate_groups
from .reaction_boundary_generator import RXNMapperAdapter
from .train_reaction_center_edit_decoder import EditDecoderMLP
from .false_negative_review import load_known_positive_set
from .chem_utils import canonicalize_reaction


OUTPUT_FIELDS = list(BoundaryCandidate.__dataclass_fields__.keys()) + ["decoder_score", "decoder_rank"]


def load_decoder(path: str, device: torch.device):
    checkpoint = torch.load(path, map_location=device)
    hidden_dim = int(checkpoint.get("hidden_dim", 512))
    dropout = float(checkpoint.get("dropout", 0.15))
    model = EditDecoderMLP(len(FEATURE_NAMES), hidden_dim=hidden_dim, dropout=dropout).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def score_rows(model, rows: List[Dict[str, object]], device: torch.device) -> List[float]:
    if not rows:
        return []
    x = torch.tensor([[float(row.get(name, 0.0) or 0.0) for name in FEATURE_NAMES] for row in rows], dtype=torch.float32, device=device)
    with torch.no_grad():
        return model(x).detach().cpu().numpy().tolist()


def to_boundary_row(row: Dict[str, object], score: float, rank: int) -> Dict[str, object]:
    return {
        "source_id": row["source_id"],
        "positive_reaction": row["positive_reaction"],
        "candidate_reaction": row["candidate_reaction"],
        "task": "forward_outcome",
        "failure_type": "learned_reaction_center_alternative",
        "edit_action": row["edit_action"],
        "parent_reactants": row["reactants"],
        "parent_product": row["parent_product"],
        "candidate_reactants": row["reactants"],
        "candidate_product": row["candidate_product"],
        "valid": 1.0,
        "atom_balance": float(row.get("atom_balance", 0.0) or 0.0),
        "locality": float(row.get("product_similarity", 0.0) or 0.0),
        "closeness": float(row.get("product_similarity", 0.0) or 0.0),
        "hard_score": float(score),
        "false_negative_risk": max(0.0, min(1.0, max(0.0, float(row.get("product_similarity", 0.0) or 0.0) - 0.90) / 0.10)),
        "passes_filter": True,
        "mapped": True,
        "center_maps": f"{row.get('fragment_map', '')};{row.get('true_anchor_map', '')};{row.get('candidate_anchor_map', '')}",
        "label": 0,
        "provenance": "pc_cng_v3_learned_edit_decoder",
        "decoder_score": float(score),
        "decoder_rank": int(rank),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--map-unmapped", action="store_true")
    parser.add_argument("--max-candidates-per-pair", type=int, default=12)
    parser.add_argument("--max-anchor-distance", type=int, default=6)
    parser.add_argument("--allow-different-anchor-atom-type", action="store_true")
    parser.add_argument("--min-product-similarity", type=float, default=0.80)
    parser.add_argument("--max-product-similarity", type=float, default=0.98)
    parser.add_argument("--known-positive", action="append", default=[])
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(os.path.dirname(args.summary), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_decoder(args.checkpoint, device)
    mapper = RXNMapperAdapter() if args.map_unmapped else None
    known_positives = load_known_positive_set(args.known_positive)

    processed = 0
    generated = 0
    skip_reasons: Dict[str, int] = {}

    with open(args.input, newline="", encoding="utf-8") as input_handle, open(
        args.output, "w", newline="", encoding="utf-8"
    ) as output_handle:
        reader = csv.DictReader(input_handle)
        writer = csv.DictWriter(output_handle, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in reader:
            if args.limit is not None and processed >= args.limit:
                break
            if row.get("label_type", "positive") != "positive":
                continue
            processed += 1
            groups, reason = build_edit_candidate_groups(
                reaction_smiles=row["reaction_smiles"],
                source_id=row["source_id"],
                split=row.get("split", "train"),
                label_type=row.get("label_type", "positive"),
                mapper=mapper,
                map_unmapped=args.map_unmapped,
                max_candidates_per_pair=args.max_candidates_per_pair,
                max_anchor_distance=args.max_anchor_distance,
            )
            if reason != "ok":
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                continue
            for group in groups:
                negative_rows = []
                for candidate in group.rows:
                    if int(candidate["is_true_anchor"]) == 1:
                        continue
                    product_similarity = float(candidate.get("product_similarity", 0.0) or 0.0)
                    if product_similarity < args.min_product_similarity or product_similarity > args.max_product_similarity:
                        continue
                    if not args.allow_different_anchor_atom_type and float(candidate.get("candidate_same_atomic_num_as_true", 0.0) or 0.0) < 0.5:
                        continue
                    canonical = canonicalize_reaction(str(candidate.get("candidate_reaction", "")))
                    if canonical and canonical in known_positives:
                        continue
                    negative_rows.append(candidate)
                scores = score_rows(model, negative_rows, device)
                ranked = sorted(zip(negative_rows, scores), key=lambda item: item[1], reverse=True)
                for rank, (candidate, score) in enumerate(ranked[: args.top_k], start=1):
                    writer.writerow(to_boundary_row(candidate, score, rank))
                    generated += 1

    summary = {
        "input": args.input,
        "checkpoint": args.checkpoint,
        "output": args.output,
        "processed": processed,
        "generated": generated,
        "skip_reasons": skip_reasons,
        "top_k": args.top_k,
        "device": str(device),
        "known_positive_count": len(known_positives),
    }
    with open(args.summary, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
