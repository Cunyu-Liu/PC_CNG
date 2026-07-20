"""Transformer-based negative generator (P2-07, L7 fix).

Goal: fix L7 - the GNN learned decoder (P1-05) did not exceed the rule-based
baseline. This script builds a Transformer-based generator that is fine-tuned
on PC-CNG synthetic negatives and compares three generators across 10 seeds:

  G1: Rule-based generator (simple perturbations, no RXNMapper dependency).
  G2: GNN learned decoder (P1-05 checkpoint; falls back to a deterministic
      perturbation if the checkpoint / featurization pipeline is unavailable).
  G3: Transformer encoder-decoder (this task) trained from scratch on
      PC-CNG synthetic negatives (positive_reaction -> candidate_reaction).

Degradation path (Section 26.1):
  - The ``chemformer`` Python package is NOT importable in this environment
    (``importlib.util.find_spec("chemformer")`` returns ``None``), although
    the Chemformer checkpoint files exist under
    ``models/reaction_lm/chemformer_forward_uspto50k/``.
  - Per the task's degradation path, we therefore use option 3: a small
    PyTorch ``nn.Transformer`` encoder-decoder trained from scratch on
    PC-CNG negatives. The Chemformer checkpoint is acknowledged but not
    loaded because the package API is unavailable.

Metrics (10-seed paired):
  - Test Top-1 reranking (logistic ranker trained on each generator's
    negatives, evaluated on the val set).
  - Diversity (unique canonical reactions / total).
  - Validity rate (RDKit-parseable reactants and products).
  - Paired permutation p-value + sign-test p-value on Top-1 delta.

Go/No-Go: G3 mean Top-1 >= G1 mean Top-1 + 1.0 pp.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m pc_cng.transformer_negative_generator \\
        --train-data results/uspto_openmolecules_full_generation/pc_cng_synthetic_negatives_reviewed.csv \\
        --val-data data/processed/regiosqm20_normalized.csv \\
        --output-dir results/transformer_negative_generator_20260720 \\
        --seeds 20260710,20260711,20260712,20260713,20260714,20260715,20260716,20260717,20260718,20260719 \\
        --epochs 5 --batch-size 32 --max-train-samples 10000 --device cuda:0
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from .chem_utils import (
    atom_balance_score,
    canonicalize_reaction,
    is_valid_smiles,
    join_reaction,
    split_reaction,
    string_similarity,
)
from .paired_reranking_significance import (
    bootstrap_ci,
    mean,
    paired_permutation_p_value,
    sign_test_p_value,
)
from .reranker import LogisticReactionRanker, evaluate_ranking


# ---------------------------------------------------------------------------
# Special tokens
# ---------------------------------------------------------------------------

PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
SPECIAL_TOKENS = (PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN)

DEFAULT_SEEDS = (
    "20260710,20260711,20260712,20260713,20260714,"
    "20260715,20260716,20260717,20260718,20260719"
)

GO_NO_GO_THRESHOLD_PP = 1.0  # G3 must beat G1 by at least 1.0 percentage point


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class GeneratedNegative:
    """A single generated negative reaction."""

    source_id: str
    positive_reaction: str
    candidate_reaction: str
    generator: str  # "rule" | "gnn" | "gnn_fallback" | "transformer" | "transformer_fallback"
    decoder_score: float = 0.0


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and Pytorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_train_pairs(path: str, max_samples: Optional[int] = None) -> List[Tuple[str, str]]:
    """Load (positive_reaction, candidate_reaction) pairs from PC-CNG synthetic negatives CSV.

    The CSV is expected to have ``positive_reaction`` and ``candidate_reaction``
    columns (the standard PC-CNG synthetic negatives schema). Rows with
    ``label != 0`` are skipped (we only want negatives here).
    """
    pairs: List[Tuple[str, str]] = []
    if not path or not os.path.exists(path):
        return pairs
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label = str(row.get("label", "0"))
            if label not in ("0", "0.0"):
                continue
            pos = (row.get("positive_reaction") or "").strip()
            cand = (row.get("candidate_reaction") or "").strip()
            if not pos or not cand or ">>" not in pos or ">>" not in cand:
                continue
            pairs.append((pos, cand))
            if max_samples is not None and len(pairs) >= max_samples:
                break
    return pairs


def load_val_rows(path: str, limit: Optional[int] = None) -> List[Dict[str, str]]:
    """Load validation reaction rows (expects ``reaction_smiles`` column)."""
    rows: List[Dict[str, str]] = []
    if not path or not os.path.exists(path):
        return rows
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rxn = (row.get("reaction_smiles") or "").strip()
            if not rxn or ">>" not in rxn:
                continue
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


# ---------------------------------------------------------------------------
# SMILES tokenizer (character-level)
# ---------------------------------------------------------------------------


class SMILESTokenizer:
    """Character-level SMILES tokenizer with special tokens."""

    def __init__(self) -> None:
        self.char2idx: Dict[str, int] = {}
        self.idx2char: List[str] = []
        for tok in SPECIAL_TOKENS:
            self._add_token(tok)
        self.pad_idx = self.char2idx[PAD_TOKEN]
        self.sos_idx = self.char2idx[SOS_TOKEN]
        self.eos_idx = self.char2idx[EOS_TOKEN]
        self.unk_idx = self.char2idx[UNK_TOKEN]

    def _add_token(self, ch: str) -> None:
        if ch not in self.char2idx:
            self.char2idx[ch] = len(self.idx2char)
            self.idx2char.append(ch)

    def build_vocab(self, smiles_iter: Sequence[str]) -> None:
        for smi in smiles_iter:
            for ch in smi:
                self._add_token(ch)

    def encode(self, smi: str, max_len: int) -> torch.Tensor:
        ids = [self.sos_idx]
        for ch in smi:
            ids.append(self.char2idx.get(ch, self.unk_idx))
        ids.append(self.eos_idx)
        ids = ids[:max_len]
        while len(ids) < max_len:
            ids.append(self.pad_idx)
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids: torch.Tensor) -> str:
        out: List[str] = []
        for idx in ids.tolist():
            if idx == self.eos_idx:
                break
            if idx in (self.pad_idx, self.sos_idx):
                continue
            out.append(self.idx2char[idx] if 0 <= idx < len(self.idx2char) else "")
        return "".join(out)

    @property
    def vocab_size(self) -> int:
        return len(self.idx2char)


# ---------------------------------------------------------------------------
# Transformer model
# ---------------------------------------------------------------------------


def make_causal_mask(size: int, device: torch.device) -> torch.Tensor:
    """Standard upper-triangular boolean causal mask (True = mask out)."""
    return torch.triu(
        torch.ones(size, size, device=device, dtype=torch.bool), diagonal=1
    )


class TransformerSeq2Seq(nn.Module):
    """Small Transformer encoder-decoder for SMILES -> SMILES translation.

    Deliberately lightweight so a smoke test (100 samples, 1 epoch) runs in
    a few seconds on CPU/GPU. Architecture mirrors Chemformer's seq2seq
    framing without importing the (unavailable) ``chemformer`` package.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        ff_dim: int = 128,
        pad_idx: int = 0,
        dropout: float = 0.1,
        max_pos: int = 512,
    ) -> None:
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_embedding = nn.Embedding(max_pos, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, ff_dim, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead, ff_dim, dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        batch_size, src_len = src.shape
        tgt_len = tgt.shape[1]
        device = src.device

        src_pos = torch.arange(src_len, device=device).unsqueeze(0).expand(batch_size, src_len)
        tgt_pos = torch.arange(tgt_len, device=device).unsqueeze(0).expand(batch_size, tgt_len)
        src_emb = self.embedding(src) + self.pos_embedding(src_pos)
        tgt_emb = self.embedding(tgt) + self.pos_embedding(tgt_pos)

        src_key_padding_mask = src == self.pad_idx
        tgt_key_padding_mask = tgt == self.pad_idx
        tgt_mask = make_causal_mask(tgt_len, device)

        memory = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        decoded = self.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return self.output_proj(decoded)

    @torch.no_grad()
    def greedy_decode(
        self,
        src: torch.Tensor,
        sos_idx: int,
        eos_idx: int,
        max_len: int = 128,
    ) -> torch.Tensor:
        """Greedy autoregressive decoding. Returns (batch, seq) token ids."""
        self.eval()
        batch_size = src.shape[0]
        device = src.device
        tgt = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        for _ in range(max_len - 1):
            logits = self.forward(src, tgt)
            next_tok = logits[:, -1, :].argmax(dim=-1)
            next_tok = torch.where(
                finished, torch.full_like(next_tok, self.pad_idx), next_tok
            )
            tgt = torch.cat([tgt, next_tok.unsqueeze(1)], dim=1)
            finished = finished | (next_tok == eos_idx)
            if finished.all():
                break
        return tgt


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------


class RuleBasedGenerator:
    """G1: rule-based negative generator (simple perturbations, no RXNMapper).

    Implements three lightweight rules that match common PC-CNG failure modes
    found in the synthetic negatives CSV (``edit_action`` column):
      - drop_reactant: retro_missing_reactant (drop the last reactant).
      - swap_ON: retro_wrong_functional_group (O<->N swap in first reactant).
      - no_reaction: product := reactants (identity reaction).
    """

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def generate(
        self, positive_reaction: str, source_id: str = ""
    ) -> Optional[GeneratedNegative]:
        try:
            reactants, agents, products = split_reaction(positive_reaction)
        except ValueError:
            return None
        reactant_parts = [r for r in reactants.split(".") if r]
        if not reactant_parts or not products:
            return None

        action = self.rng.choice(["drop_reactant", "swap_ON", "no_reaction"])
        if action == "drop_reactant" and len(reactant_parts) > 1:
            new_reactants = ".".join(reactant_parts[:-1])
            cand = join_reaction(new_reactants, products, agents)
        elif action == "swap_ON":
            r0 = reactant_parts[0]
            if "O" in r0:
                new_r0 = r0.replace("O", "N", 1)
            elif "N" in r0:
                new_r0 = r0.replace("N", "O", 1)
            else:
                new_r0 = r0
            new_reactants = ".".join([new_r0] + reactant_parts[1:])
            cand = join_reaction(new_reactants, products, agents)
        else:  # no_reaction
            cand = join_reaction(reactants, reactants, agents)

        return GeneratedNegative(
            source_id=source_id,
            positive_reaction=positive_reaction,
            candidate_reaction=cand,
            generator="rule",
            decoder_score=1.0,
        )


class GNNGenerator:
    """G2: GNN learned decoder (P1-05).

    Attempts to load the P1-05 GNN checkpoint and use
    ``generate_boundary_negatives``. If the checkpoint is missing or the
    featurization pipeline (which needs atom mapping / RXNMapper) fails,
    falls back to a deterministic perturbation that is *different* from
    the rule-based generator (N->O swap instead of O->N, plus reactant
    reordering). This keeps the 3-way comparison well-defined even when
    the GNN cannot run.
    """

    def __init__(
        self,
        seed: int = 0,
        checkpoint: Optional[str] = None,
        device: str = "cpu",
    ) -> None:
        self.seed = seed
        self.rng = random.Random(seed + 1)
        self.checkpoint = checkpoint
        self.device = device
        self.model = None
        self.generate_fn = None
        self.fallback_reason: Optional[str] = None
        if checkpoint and os.path.exists(checkpoint):
            try:
                from .learned_graph_edit_decoder import (
                    generate_boundary_negatives,
                    load_checkpoint,
                )

                self.model = load_checkpoint(checkpoint, torch.device(device))
                self.generate_fn = generate_boundary_negatives
            except Exception as exc:  # pragma: no cover - environment dependent
                self.fallback_reason = f"checkpoint_load_failed: {type(exc).__name__}: {exc}"
        else:
            self.fallback_reason = "no_checkpoint"

    def generate(
        self, positive_reaction: str, source_id: str = ""
    ) -> Optional[GeneratedNegative]:
        if self.model is not None and self.generate_fn is not None:
            try:
                negs, reason = self.generate_fn(
                    self.model,
                    positive_reaction,
                    source_id=source_id,
                    top_k=1,
                    device=torch.device(self.device),
                )
                if negs:
                    n = negs[0]
                    return GeneratedNegative(
                        source_id=n.source_id,
                        positive_reaction=n.positive_reaction,
                        candidate_reaction=n.candidate_reaction,
                        generator="gnn",
                        decoder_score=float(n.decoder_score),
                    )
            except Exception:  # pragma: no cover - environment dependent
                pass
        return self._fallback(positive_reaction, source_id)

    def _fallback(
        self, positive_reaction: str, source_id: str
    ) -> Optional[GeneratedNegative]:
        try:
            reactants, agents, products = split_reaction(positive_reaction)
        except ValueError:
            return None
        reactant_parts = [r for r in reactants.split(".") if r]
        if not reactant_parts:
            return None
        # Inverse of rule-based O->N: swap N->O (or O->S) in first reactant.
        r0 = reactant_parts[0]
        if "N" in r0:
            new_r0 = r0.replace("N", "O", 1)
        elif "O" in r0:
            new_r0 = r0.replace("O", "S", 1)
        elif len(reactant_parts) > 1:
            # Drop the *first* reactant (rule-based drops the last).
            new_reactants = ".".join(reactant_parts[1:])
            cand = join_reaction(new_reactants, products, agents)
            return GeneratedNegative(
                source_id=source_id,
                positive_reaction=positive_reaction,
                candidate_reaction=cand,
                generator="gnn_fallback",
                decoder_score=0.5,
            )
        else:
            new_r0 = r0
        new_reactants = ".".join([new_r0] + reactant_parts[1:])
        cand = join_reaction(new_reactants, products, agents)
        return GeneratedNegative(
            source_id=source_id,
            positive_reaction=positive_reaction,
            candidate_reaction=cand,
            generator="gnn_fallback",
            decoder_score=0.5,
        )


class TransformerGenerator:
    """G3: Transformer encoder-decoder trained on PC-CNG synthetic negatives.

    The model is trained as a seq2seq map positive_reaction -> candidate_reaction
    (teacher forcing). At inference time, greedy decoding produces a candidate
    negative for each input positive. If decoding fails to produce a valid
    reaction, an identity fallback is used and tagged ``transformer_fallback``.
    """

    def __init__(
        self,
        tokenizer: SMILESTokenizer,
        seed: int = 0,
        device: str = "cpu",
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        ff_dim: int = 128,
        max_len: int = 128,
        dropout: float = 0.1,
    ) -> None:
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.max_len = max_len
        self.seed = seed
        self.model = TransformerSeq2Seq(
            tokenizer.vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            ff_dim=ff_dim,
            pad_idx=tokenizer.pad_idx,
            dropout=dropout,
        ).to(self.device)

    def train(
        self,
        pairs: Sequence[Tuple[str, str]],
        epochs: int,
        batch_size: int,
        lr: float = 1e-4,
    ) -> List[Dict[str, float]]:
        """Train the transformer with teacher forcing. Returns per-epoch losses."""
        if not pairs:
            return []
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        history: List[Dict[str, float]] = []
        for epoch in range(epochs):
            rng = random.Random(self.seed + epoch)
            idxs = list(range(len(pairs)))
            rng.shuffle(idxs)
            total_loss = 0.0
            num_batches = 0
            for start in range(0, len(idxs), batch_size):
                batch_idx = idxs[start : start + batch_size]
                src = torch.stack(
                    [self.tokenizer.encode(pairs[i][0], self.max_len) for i in batch_idx]
                ).to(self.device)
                tgt = torch.stack(
                    [self.tokenizer.encode(pairs[i][1], self.max_len) for i in batch_idx]
                ).to(self.device)
                tgt_in = tgt[:, :-1]
                tgt_out = tgt[:, 1:]
                logits = self.model(src, tgt_in)
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    tgt_out.reshape(-1),
                    ignore_index=self.tokenizer.pad_idx,
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()
                total_loss += float(loss.item())
                num_batches += 1
            avg_loss = total_loss / max(num_batches, 1)
            history.append({"epoch": epoch + 1, "loss": avg_loss})
        return history

    def generate(
        self, positive_reaction: str, source_id: str = ""
    ) -> Optional[GeneratedNegative]:
        self.model.eval()
        with torch.no_grad():
            src = self.tokenizer.encode(positive_reaction, self.max_len).unsqueeze(0).to(self.device)
            out = self.model.greedy_decode(
                src, self.tokenizer.sos_idx, self.tokenizer.eos_idx, self.max_len
            )
        cand = self.tokenizer.decode(out[0])
        if not cand or ">>" not in cand:
            # Fallback: identity reaction (tagged so downstream can filter).
            try:
                reactants, agents, products = split_reaction(positive_reaction)
                cand = join_reaction(reactants, products, agents)
            except ValueError:
                return None
            return GeneratedNegative(
                source_id=source_id,
                positive_reaction=positive_reaction,
                candidate_reaction=cand,
                generator="transformer_fallback",
                decoder_score=0.0,
            )
        return GeneratedNegative(
            source_id=source_id,
            positive_reaction=positive_reaction,
            candidate_reaction=cand,
            generator="transformer",
            decoder_score=1.0,
        )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_diversity(negatives: Sequence[GeneratedNegative]) -> float:
    """Fraction of unique canonical reactions among generated negatives.

    Falls back to the raw candidate string when canonicalization fails (e.g.
    for invalid SMILES in unit tests), so distinct strings are still counted.
    """
    if not negatives:
        return 0.0
    unique: set = set()
    for neg in negatives:
        canon = canonicalize_reaction(neg.candidate_reaction)
        unique.add(canon if canon else neg.candidate_reaction)
    return len(unique) / len(negatives)


def compute_validity(negatives: Sequence[GeneratedNegative]) -> float:
    """Fraction of negatives with RDKit-parseable reactants and products."""
    if not negatives:
        return 0.0
    valid = 0
    for neg in negatives:
        try:
            reactants, _, products = split_reaction(neg.candidate_reaction)
        except ValueError:
            continue
        if is_valid_smiles(reactants) and is_valid_smiles(products):
            valid += 1
    return valid / len(negatives)


def compute_top1_reranking(
    negatives: Sequence[GeneratedNegative],
    val_rows: Sequence[Dict[str, str]],
    seed: int,
) -> float:
    """Train a logistic ranker on (positive=1, generated negative=0) and
    evaluate Top-1 accuracy on the val set (each val positive paired with
    a randomly chosen generated negative).

    ``evaluate_ranking`` groups rows by ``source_id``, so each eval row must
    carry the source_id of its val reaction.
    """
    if not negatives or not val_rows:
        return 0.0
    train_rows: List[Dict[str, object]] = []
    for neg in negatives:
        train_rows.append({"reaction_smiles": neg.positive_reaction, "label": 1})
        train_rows.append({"reaction_smiles": neg.candidate_reaction, "label": 0})
    model = LogisticReactionRanker(learning_rate=0.2, l2=1e-4, epochs=100)
    model.fit(train_rows)

    rng = random.Random(seed)
    eval_rows: List[Dict[str, object]] = []
    neg_pool = list(negatives)
    for idx, row in enumerate(val_rows):
        rxn = (row.get("reaction_smiles") or "").strip()
        if not rxn:
            continue
        src = row.get("source_id") or f"val_{idx}"
        eval_rows.append({"source_id": src, "reaction_smiles": rxn, "label": 1})
        neg = rng.choice(neg_pool)
        eval_rows.append({"source_id": src, "reaction_smiles": neg.candidate_reaction, "label": 0})
    if not eval_rows:
        return 0.0
    metrics = evaluate_ranking(model, eval_rows)
    return float(metrics.top1)


def compute_generator_metrics(
    negatives: Sequence[GeneratedNegative],
    val_rows: Sequence[Dict[str, str]],
    seed: int,
) -> Dict[str, float]:
    """Compute the full metric set for one generator's negatives."""
    return {
        "count": float(len(negatives)),
        "top1": compute_top1_reranking(negatives, val_rows, seed),
        "diversity": compute_diversity(negatives),
        "validity": compute_validity(negatives),
    }


# ---------------------------------------------------------------------------
# Per-seed runner
# ---------------------------------------------------------------------------


def run_single_seed(
    seed: int,
    train_pairs: Sequence[Tuple[str, str]],
    val_rows: Sequence[Dict[str, str]],
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, object]:
    """Run G1/G2/G3 generation + metrics for a single seed."""
    set_seed(seed)
    rng = random.Random(seed)

    # Subsample training pairs for the transformer.
    pairs = list(train_pairs)
    if len(pairs) > args.max_train_samples:
        pairs = rng.sample(pairs, args.max_train_samples)
    print(f"[seed {seed}] train_pairs={len(pairs)} val_rows={len(val_rows)}")

    # G1: rule-based
    t0 = time.time()
    g1_gen = RuleBasedGenerator(seed=seed)
    g1_negs = [
        n for n in (
            g1_gen.generate(v.get("reaction_smiles", ""), v.get("source_id", ""))
            for v in val_rows
        )
        if n is not None
    ]
    g1_metrics = compute_generator_metrics(g1_negs, val_rows, seed)
    print(f"[seed {seed}] G1 rule: {len(g1_negs)} negatives, top1={g1_metrics['top1']:.4f} ({time.time()-t0:.1f}s)")

    # G2: GNN learned decoder (or fallback)
    t0 = time.time()
    g2_gen = GNNGenerator(
        seed=seed,
        checkpoint=args.gnn_checkpoint or None,
        device=str(device),
    )
    g2_negs = [
        n for n in (
            g2_gen.generate(v.get("reaction_smiles", ""), v.get("source_id", ""))
            for v in val_rows
        )
        if n is not None
    ]
    g2_metrics = compute_generator_metrics(g2_negs, val_rows, seed)
    g2_fallback_reason = g2_gen.fallback_reason
    print(
        f"[seed {seed}] G2 gnn: {len(g2_negs)} negatives, top1={g2_metrics['top1']:.4f} "
        f"(fallback={g2_fallback_reason}) ({time.time()-t0:.1f}s)"
    )

    # G3: transformer
    t0 = time.time()
    tokenizer = SMILESTokenizer()
    tokenizer.build_vocab([p[0] for p in pairs])
    tokenizer.build_vocab([p[1] for p in pairs])
    # also include val reaction SMILES so decode can always represent them
    tokenizer.build_vocab([v.get("reaction_smiles", "") for v in val_rows])
    g3_gen = TransformerGenerator(
        tokenizer,
        seed=seed,
        device=str(device),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        max_len=args.max_len,
        dropout=args.dropout,
    )
    train_history = g3_gen.train(
        pairs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    g3_negs = [
        n for n in (
            g3_gen.generate(v.get("reaction_smiles", ""), v.get("source_id", ""))
            for v in val_rows
        )
        if n is not None
    ]
    g3_metrics = compute_generator_metrics(g3_negs, val_rows, seed)
    print(
        f"[seed {seed}] G3 transformer: {len(g3_negs)} negatives, top1={g3_metrics['top1']:.4f} "
        f"(epochs={args.epochs}, last_loss={train_history[-1]['loss'] if train_history else 0:.4f}) "
        f"({time.time()-t0:.1f}s)"
    )

    return {
        "seed": seed,
        "g1": g1_metrics,
        "g2": g2_metrics,
        "g3": g3_metrics,
        "g2_fallback_reason": g2_fallback_reason,
        "g3_train_history": train_history,
        "g1_negatives_sample": [asdict(n) for n in g1_negs[:5]],
        "g2_negatives_sample": [asdict(n) for n in g2_negs[:5]],
        "g3_negatives_sample": [asdict(n) for n in g3_negs[:5]],
    }


# ---------------------------------------------------------------------------
# Paired significance
# ---------------------------------------------------------------------------


def paired_significance_test(
    g1_values: Sequence[float],
    g2_values: Sequence[float],
    g3_values: Sequence[float],
    iterations: int = 1000,
    seed: int = 20260720,
) -> Dict[str, object]:
    """Paired significance for Top-1 across the three generators.

    Computes bootstrap CI + paired permutation p + sign-test p on the
    per-seed Top-1 deltas (G3-G1, G3-G2, G2-G1).
    """
    def _paired(a: Sequence[float], b: Sequence[float]) -> Dict[str, object]:
        n = min(len(a), len(b))
        if n == 0:
            return {
                "mean_delta": 0.0,
                "ci_low": 0.0,
                "ci_high": 0.0,
                "paired_permutation_p": 1.0,
                "sign_test_p": 1.0,
                "n_pairs": 0,
            }
        deltas = [float(a[i]) - float(b[i]) for i in range(n)]
        ci_low, ci_high = bootstrap_ci(deltas, iterations, seed)
        return {
            "mean_delta": mean(deltas),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "paired_permutation_p": paired_permutation_p_value(
                deltas, iterations, seed + 1
            ),
            "sign_test_p": sign_test_p_value(deltas),
            "n_pairs": n,
        }

    return {
        "g3_vs_g1": _paired(g3_values, g1_values),
        "g3_vs_g2": _paired(g3_values, g2_values),
        "g2_vs_g1": _paired(g2_values, g1_values),
        "iterations": iterations,
        "seed": seed,
    }


# ---------------------------------------------------------------------------
# Go / No-Go decision
# ---------------------------------------------------------------------------


def go_no_go_decision(
    g1_top1_mean: float,
    g3_top1_mean: float,
    threshold_pp: float = GO_NO_GO_THRESHOLD_PP,
) -> Dict[str, object]:
    """Go/No-Go: G3 mean Top-1 must beat G1 mean Top-1 by >= threshold pp."""
    delta_pp = (g3_top1_mean - g1_top1_mean) * 100.0
    decision = "GO" if delta_pp >= threshold_pp else "NO-GO"
    return {
        "decision": decision,
        "criterion": f"G3 Top-1 >= G1 Top-1 + {threshold_pp} pp",
        "g1_top1_mean": g1_top1_mean,
        "g3_top1_mean": g3_top1_mean,
        "delta_pp": delta_pp,
        "threshold_pp": threshold_pp,
        "passes": decision == "GO",
        "note": (
            "L7 fixed: transformer generator exceeds rule-based baseline."
            if decision == "GO"
            else "L7 not fixed: transformer generator did not exceed rule-based by the required margin."
        ),
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_per_seed_csv(per_seed: Sequence[Dict[str, object]], path: str) -> None:
    """Write per-seed metrics CSV."""
    fieldnames = [
        "seed",
        "g1_top1", "g1_diversity", "g1_validity", "g1_count",
        "g2_top1", "g2_diversity", "g2_validity", "g2_count",
        "g3_top1", "g3_diversity", "g3_validity", "g3_count",
        "g2_fallback_reason",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fieldnames)
        for rec in per_seed:
            row = [rec["seed"]]
            for gen in ("g1", "g2", "g3"):
                m = rec[gen]  # type: ignore[index]
                row.extend([f"{m['top1']:.6f}", f"{m['diversity']:.6f}",
                            f"{m['validity']:.6f}", int(m['count'])])
            row.append(rec.get("g2_fallback_reason", "") or "")
            writer.writerow(row)


def write_generated_sample_csv(
    negatives: Sequence[GeneratedNegative], path: str, limit: int = 100
) -> None:
    """Write the first ``limit`` generated negatives as a CSV sample."""
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_id", "generator", "positive_reaction",
                         "candidate_reaction", "decoder_score"])
        for neg in negatives[:limit]:
            writer.writerow([
                neg.source_id, neg.generator, neg.positive_reaction,
                neg.candidate_reaction, f"{neg.decoder_score:.4f}",
            ])


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------


def resolve_device(device_arg: str) -> torch.device:
    """Resolve a torch device from the CLI string, honoring CUDA_VISIBLE_DEVICES."""
    if device_arg.startswith("cuda") and torch.cuda.is_available():
        # Honor CUDA_VISIBLE_DEVICES: if it restricts to one GPU, use cuda:0.
        vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if vis and "," not in vis:
            return torch.device("cuda:0")
        try:
            return torch.device(device_arg)
        except RuntimeError:
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transformer-based negative generator (P2-07, L7 fix)"
    )
    parser.add_argument("--train-data", required=True,
                        help="PC-CNG synthetic negatives CSV (positive_reaction, candidate_reaction)")
    parser.add_argument("--val-data", required=True,
                        help="Validation CSV with reaction_smiles column")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", default=DEFAULT_SEEDS,
                        help="Comma-separated seed list (default: 10 seeds)")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Transformer fine-tune epochs (default 5)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-train-samples", type=int, default=10000,
                        help="Subsample this many training pairs per seed (default 10000)")
    parser.add_argument("--chemformer-checkpoint", default="",
                        help="Optional Chemformer checkpoint (unused if chemformer pkg unavailable)")
    parser.add_argument("--gnn-checkpoint", default="",
                        help="Optional P1-05 GNN decoder checkpoint for G2")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit-val", type=int, default=None,
                        help="Limit val rows (useful for smoke tests)")
    # Transformer hyperparameters (kept small for fast fine-tune).
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--ff-dim", type=int, default=128)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--significance-iterations", type=int, default=1000)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, object]:
    args = parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    device = resolve_device(args.device)
    print(f"[main] device={device} seeds={seeds} output_dir={args.output_dir}")
    print(f"[main] chemformer_checkpoint={args.chemformer_checkpoint!r}")
    print(f"[main] gnn_checkpoint={args.gnn_checkpoint!r}")

    # Chemformer availability check (for the degradation-path manifest).
    chemformer_status = _check_chemformer_availability(args.chemformer_checkpoint)
    print(f"[main] chemformer_status={chemformer_status}")

    # Load data.
    train_pairs = load_train_pairs(args.train_data, max_samples=None)
    val_rows = load_val_rows(args.val_data, limit=args.limit_val)
    print(f"[data] train_pairs={len(train_pairs)} val_rows={len(val_rows)}")
    if not train_pairs:
        raise RuntimeError(
            f"No training pairs loaded from {args.train_data}. "
            "Check that the CSV has positive_reaction / candidate_reaction columns."
        )
    if not val_rows:
        raise RuntimeError(
            f"No val rows loaded from {args.val_data}. "
            "Check that the CSV has a reaction_smiles column."
        )

    per_seed: List[Dict[str, object]] = []
    all_g3_negatives_for_sample: List[GeneratedNegative] = []
    overall_t0 = time.time()
    for seed in seeds:
        result = run_single_seed(seed, train_pairs, val_rows, args, device)
        per_seed.append(result)
        # Collect G3 negatives from the first seed for the sample CSV.
        if not all_g3_negatives_for_sample and result.get("g3_negatives_sample"):
            # Re-generate is wasteful; instead, we already have samples in the dict.
            # For the sample CSV we re-run G3 on a small subset using the first seed.
            pass

    # If we want a generated_negatives_sample.csv, re-run G3 on the first seed
    # with the first 100 val rows and dump the negatives.
    try:
        first_seed = seeds[0]
        set_seed(first_seed)
        pairs = list(train_pairs)
        rng_sample = random.Random(first_seed)
        if len(pairs) > args.max_train_samples:
            pairs = rng_sample.sample(pairs, args.max_train_samples)
        tokenizer = SMILESTokenizer()
        tokenizer.build_vocab([p[0] for p in pairs])
        tokenizer.build_vocab([p[1] for p in pairs])
        tokenizer.build_vocab([v.get("reaction_smiles", "") for v in val_rows])
        g3_gen = TransformerGenerator(
            tokenizer, seed=first_seed, device=str(device),
            d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers,
            ff_dim=args.ff_dim, max_len=args.max_len, dropout=args.dropout,
        )
        g3_gen.train(pairs, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
        sample_val = val_rows[:100]
        all_g3_negatives_for_sample = [
            n for n in (
                g3_gen.generate(v.get("reaction_smiles", ""), v.get("source_id", ""))
                for v in sample_val
            )
            if n is not None
        ]
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[warn] failed to build G3 sample negatives: {exc}")
        all_g3_negatives_for_sample = []

    # Aggregate metrics.
    g1_top1 = [float(rec["g1"]["top1"]) for rec in per_seed]  # type: ignore[index]
    g2_top1 = [float(rec["g2"]["top1"]) for rec in per_seed]  # type: ignore[index]
    g3_top1 = [float(rec["g3"]["top1"]) for rec in per_seed]  # type: ignore[index]
    g1_div = [float(rec["g1"]["diversity"]) for rec in per_seed]  # type: ignore[index]
    g2_div = [float(rec["g2"]["diversity"]) for rec in per_seed]  # type: ignore[index]
    g3_div = [float(rec["g3"]["diversity"]) for rec in per_seed]  # type: ignore[index]
    g1_val = [float(rec["g1"]["validity"]) for rec in per_seed]  # type: ignore[index]
    g2_val = [float(rec["g2"]["validity"]) for rec in per_seed]  # type: ignore[index]
    g3_val = [float(rec["g3"]["validity"]) for rec in per_seed]  # type: ignore[index]

    significance = paired_significance_test(
        g1_top1, g2_top1, g3_top1,
        iterations=args.significance_iterations,
        seed=seeds[0] if seeds else 20260720,
    )
    decision = go_no_go_decision(
        g1_top1_mean=mean(g1_top1),
        g3_top1_mean=mean(g3_top1),
        threshold_pp=GO_NO_GO_THRESHOLD_PP,
    )

    summary: Dict[str, object] = {
        "task": "P2-07 Transformer-based generator ablation (L7 fix)",
        "output_dir": args.output_dir,
        "args": vars(args),
        "n_seeds": len(seeds),
        "seeds": seeds,
        "chemformer_status": chemformer_status,
        "degradation_path": (
            "small_pytorch_transformer_from_scratch"
            if chemformer_status["package_importable"] is False
            else "chemformer_finetune"
        ),
        "device": str(device),
        "elapsed_sec": round(time.time() - overall_t0, 2),
        "n_train_pairs_loaded": len(train_pairs),
        "n_val_rows_loaded": len(val_rows),
        "g1_top1_mean": mean(g1_top1),
        "g2_top1_mean": mean(g2_top1),
        "g3_top1_mean": mean(g3_top1),
        "g1_diversity_mean": mean(g1_div),
        "g2_diversity_mean": mean(g2_div),
        "g3_diversity_mean": mean(g3_div),
        "g1_validity_mean": mean(g1_val),
        "g2_validity_mean": mean(g2_val),
        "g3_validity_mean": mean(g3_val),
        "g1_top1_per_seed": g1_top1,
        "g2_top1_per_seed": g2_top1,
        "g3_top1_per_seed": g3_top1,
    }

    # Write outputs.
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    write_per_seed_csv(per_seed, os.path.join(args.output_dir, "per_seed_metrics.csv"))
    with open(os.path.join(args.output_dir, "paired_significance.json"), "w", encoding="utf-8") as handle:
        json.dump(significance, handle, indent=2, ensure_ascii=False)
    with open(os.path.join(args.output_dir, "go_no_go_decision.json"), "w", encoding="utf-8") as handle:
        json.dump(decision, handle, indent=2, ensure_ascii=False)
    write_generated_sample_csv(
        all_g3_negatives_for_sample,
        os.path.join(args.output_dir, "generated_negatives_sample.csv"),
        limit=100,
    )

    print("=" * 70)
    print(f"[main] G1 Top-1 mean = {summary['g1_top1_mean']:.4f}")  # type: ignore[index]
    print(f"[main] G2 Top-1 mean = {summary['g2_top1_mean']:.4f}")  # type: ignore[index]
    print(f"[main] G3 Top-1 mean = {summary['g3_top1_mean']:.4f}")  # type: ignore[index]
    print(f"[main] decision = {decision['decision']} (delta_pp={decision['delta_pp']:.2f})")
    print(f"[main] outputs written to {args.output_dir}")
    return summary


def _check_chemformer_availability(checkpoint_path: str) -> Dict[str, object]:
    """Check whether the chemformer package and checkpoint are usable."""
    import importlib.util

    spec = importlib.util.find_spec("chemformer")
    pkg_importable = spec is not None
    checkpoint_exists = bool(checkpoint_path) and os.path.exists(checkpoint_path)
    return {
        "package_importable": bool(pkg_importable),
        "checkpoint_path": checkpoint_path or "",
        "checkpoint_exists": checkpoint_exists,
        "usable": bool(pkg_importable and checkpoint_exists),
        "degradation_reason": (
            "chemformer package not importable"
            if not pkg_importable
            else ("checkpoint missing" if not checkpoint_exists else "ok")
        ),
    }


if __name__ == "__main__":
    main()
