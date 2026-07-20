"""Pretrained Transformer backbone for reaction scoring (P3-01).

Loads a sanitized Chemformer checkpoint (PyTorch-Lightning state_dict with
``weights_only=True``) and exposes the encoder as a frozen (or LoRA-adapted)
reaction-level feature extractor.  The backbone is paired with a lightweight
classification head that scores reaction plausibility (positive vs. PC-CNG
generated negative), enabling a direct 10-seed paired comparison with the
GNN/MLP baselines in :mod:`pc_cng.learned_graph_edit_decoder`.

Key design choices
------------------
* **Offline-first**: the checkpoint and vocabulary live on disk under
  ``models/reaction_lm/chemformer_pretrained_hf/`` and
  ``external/reaction_lm/Chemformer/bart_vocab.json``; no network access is
  required (the server is air-gapped, see P2-06 NO-GO).
* **Encoder-only**: we discard the decoder (110 tensors) and keep the 6-layer
  encoder (74 tensors) + shared token + positional embeddings.  The encoder
  output for the ``^`` (start) token is used as the reaction-level pooled
  representation.
* **Split contract (HC #9)**: the training script in
  :mod:`pc_cng.training.train_pretrained` accepts ``--train-idx/--val-idx/
  --test-idx`` JSON files that pin the exact row indices used for each split.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

__all__ = [
    "ChemformerTokenizer",
    "PretrainedChemformerBackbone",
    "ReactionClassificationHead",
    "PretrainedReactionScorer",
    "DEFAULT_CHECKPOINT_PATH",
    "DEFAULT_VOCAB_PATH",
    "CHEMFORMER_HPARAMS",
    "load_chemformer_state_dict",
]


# ---------------------------------------------------------------------------
# Default paths (resolved relative to the repository root)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]  # chem_negative_sampling/../

DEFAULT_CHECKPOINT_PATH = _REPO_ROOT / (
    "models/reaction_lm/chemformer_pretrained_hf/model_sanitized.ckpt"
)
DEFAULT_VOCAB_PATH = _REPO_ROOT / "external/reaction_lm/Chemformer/bart_vocab.json"

# Hyper-parameters extracted from the sanitized checkpoint
# (checkpoint_summary.json: vocabulary_size=523, d_model=512, ...)
CHEMFORMER_HPARAMS: Dict[str, Any] = {
    "d_model": 512,
    "num_heads": 8,
    "num_layers": 6,
    "d_feedforward": 2048,
    "vocabulary_size": 523,
    "max_seq_len": 512,
    "num_buckets": 12,
    "pad_token_idx": 0,
    "activation": "gelu",
    "dropout": 0.1,
}

# Special token indices (bart_vocab.json ordering)
PAD_IDX = 0
UNK_IDX = 1
BOS_IDX = 2  # "^"
EOS_IDX = 3  # "&"
MASK_IDX = 4
SEP_IDX = 5

# Regex tokeniser pattern (mirrors reaction_lm_scorer.SMILES_TOKEN_PATTERN and
# the ``properties.regex`` field of bart_vocab.json).  We keep the pattern
# conservative so it always matches the same characters as the original
# Chemformer tokeniser.
_SMILES_TOKEN_PATTERN = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|Si?|Se?|Li?|Na?|Mg?|Al?|Ca?|Fe?|Zn?|"
    r"[BCNOPSFIbcnops]|\(|\)|\.|=|#|-|\+|\\\\|/|:|~|@|\?|>|\*|\$|%[0-9]{2}|[0-9])"
)


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------
class ChemformerTokenizer:
    """SMILES tokeniser + vocab mapping for the local Chemformer checkpoint.

    Parameters
    ----------
    vocab_path:
        Path to ``bart_vocab.json`` (or ``bart_vocab_downstream.json``).
    max_seq_len:
        Maximum number of tokens to keep.  Sequences are truncated and the
        EOS token is always appended.
    """

    def __init__(self, vocab_path: str | Path, max_seq_len: int = 256) -> None:
        raw = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
        vocab_list: List[str] = raw["vocabulary"] if isinstance(raw, dict) else raw
        self.token_to_id: Dict[str, int] = {tok: idx for idx, tok in enumerate(vocab_list)}
        self.id_to_token: Dict[int, str] = {idx: tok for tok, idx in self.token_to_id.items()}
        self.vocab_size: int = len(vocab_list)
        self.max_seq_len: int = int(max_seq_len)
        self.pad_idx = PAD_IDX
        self.bos_idx = BOS_IDX
        self.eos_idx = EOS_IDX
        self.unk_idx = UNK_IDX

    # -- core helpers ----------------------------------------------------
    def tokenize(self, smiles: str) -> List[str]:
        """Split a SMILES string into Chemformer tokens."""
        tokens = _SMILES_TOKEN_PATTERN.findall(smiles)
        if tokens and "".join(tokens) == smiles:
            return tokens
        # Fall back to character-level split for any unrecognised chars
        return list(smiles.strip())

    def encode(self, smiles: str, add_special: bool = True) -> List[int]:
        """Convert a SMILES string to a list of vocab indices."""
        tokens = self.tokenize(smiles)
        ids: List[int] = []
        if add_special:
            ids.append(self.bos_idx)
        for tok in tokens:
            ids.append(self.token_to_id.get(tok, self.unk_idx))
        if add_special:
            ids.append(self.eos_idx)
        if len(ids) > self.max_seq_len:
            # Truncate keeping the BOS / EOS markers
            ids = ids[: self.max_seq_len - 1] + [self.eos_idx]
        return ids

    def batch_encode(
        self,
        smiles_list: Sequence[str],
        add_special: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of SMILES and return padded (token_ids, attention_mask)."""
        encoded = [self.encode(s, add_special=add_special) for s in smiles_list]
        max_len = max(len(ids) for ids in encoded)
        padded = torch.full((len(encoded), max_len), self.pad_idx, dtype=torch.long)
        mask = torch.zeros((len(encoded), max_len), dtype=torch.long)
        for i, ids in enumerate(encoded):
            padded[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            mask[i, : len(ids)] = 1
        return padded, mask


# ---------------------------------------------------------------------------
# Positional encoding (relative position buckets, matches Chemformer)
# ---------------------------------------------------------------------------
def _relative_position_bucket(
    relative_position: torch.Tensor,
    num_buckets: int = 12,
    max_distance: int = 128,
) -> torch.Tensor:
    """Translate relative positions into bucket indices (DeBERTa-style)."""
    ret = torch.zeros_like(relative_position)
    n_neg = num_buckets // 2
    neg = -relative_position[relative_position < 0]
    ret[relative_position < 0] = neg.clamp(max=n_neg - 1) + n_neg
    ret[relative_position >= 0] = relative_position[relative_position >= 0].clamp(
        max=num_buckets - n_neg - 1
    )
    return ret


class RelativePositionBias(nn.Module):
    """Learnable relative-position bias table (matches ``pos_emb`` shape)."""

    def __init__(self, num_buckets: int = 12, num_heads: int = 8) -> None:
        super().__init__()
        self.num_buckets = num_buckets
        self.num_heads = num_heads
        self.embedding = nn.Embedding(num_buckets, num_heads)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        ctx_pos = torch.arange(seq_len, device=device)[:, None]
        memory_pos = torch.arange(seq_len, device=device)[None, :]
        relative = ctx_pos - memory_pos
        buckets = _relative_position_bucket(relative, self.num_buckets)
        # (seq_len, seq_len, num_heads) -> (1, num_heads, seq_len, seq_len)
        bias = self.embedding(buckets).permute(2, 0, 1).unsqueeze(0)
        return bias


# ---------------------------------------------------------------------------
# Encoder-only backbone
# ---------------------------------------------------------------------------
class _TransformerEncoderLayer(nn.Module):
    """Minimal nn.TransformerEncoderLayer wrapper that accepts a rel-pos bias."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU() if activation == "gelu" else nn.ReLU()

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn_out, _ = self.self_attn(
            x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False
        )
        x = self.norm1(x + self.dropout(attn_out))
        ff = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.norm2(x + self.dropout(ff))
        return x


class PretrainedChemformerBackbone(nn.Module):
    """Encoder-only backbone built from a sanitized Chemformer checkpoint.

    The module owns: ``emb`` (token embedding), ``pos_emb`` (relative-position
    bias table), and ``encoder.layers.{0..5}`` (6 encoder layers).  The
    decoder and ``token_fc`` output projection are intentionally dropped -- we
    only need the encoder for feature extraction.

    Parameters
    ----------
    checkpoint_path:
        Path to ``model_sanitized.ckpt``.  If ``None`` the module is
        initialised with random weights (used by unit tests).
    hparams:
        Hyper-parameter override dict.
    freeze:
        If ``True`` (default) all backbone parameters have
        ``requires_grad=False`` after loading.  Use :func:`apply_lora` from
        :mod:`pc_cng.models.adapter` to selectively unfreeze LoRA params.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str | Path] = None,
        hparams: Optional[Dict[str, Any]] = None,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        hp = {**CHEMFORMER_HPARAMS, **(hparams or {})}
        self.hparams = hp
        d_model = int(hp["d_model"])
        num_heads = int(hp["num_heads"])
        num_layers = int(hp["num_layers"])
        d_ff = int(hp["d_feedforward"])
        dropout = float(hp["dropout"])
        vocab_size = int(hp["vocabulary_size"])
        num_buckets = int(hp["num_buckets"])
        activation = str(hp.get("activation", "gelu"))

        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        # Chemformer stores ``pos_emb`` as (num_buckets, num_heads); we wrap it
        # in RelativePositionBias to expose a clean interface.
        self.pos_bias = RelativePositionBias(num_buckets=num_buckets, num_heads=num_heads)
        self.encoder_layers = nn.ModuleList(
            [
                _TransformerEncoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    dim_feedforward=d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(num_layers)
            ]
        )
        # Optional final layernorm (Chemformer uses one after the encoder)
        self.final_norm = nn.LayerNorm(d_model)

        if checkpoint_path is not None:
            self._load_sanitized_checkpoint(checkpoint_path)
        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    # -- checkpoint loading ---------------------------------------------
    def _load_sanitized_checkpoint(self, checkpoint_path: str | Path) -> None:
        ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
        state_dict = ckpt.get("state_dict", ckpt)
        # Map Chemformer keys -> our module keys
        own_keys = set(self.state_dict().keys())
        loaded: List[str] = []
        missed: List[str] = []
        for k, v in state_dict.items():
            target = self._map_key(k)
            if target is None:
                continue
            if target in own_keys:
                try:
                    self.state_dict()[target].copy_(v)
                    loaded.append(target)
                except Exception:
                    # Shape mismatch -> skip (happens for pos_emb table layout)
                    missed.append(target)
            else:
                missed.append(target)
        # pos_emb (num_buckets, num_heads) -> RelativePositionBias.embedding.weight
        if "pos_emb" in state_dict:
            pe = state_dict["pos_emb"]
            if pe.shape == self.pos_bias.embedding.weight.shape:
                self.pos_bias.embedding.weight.data.copy_(pe)
                if "pos_bias.embedding.weight" not in loaded:
                    loaded.append("pos_bias.embedding.weight")

    @staticmethod
    def _map_key(chemformer_key: str) -> Optional[str]:
        """Translate a Chemformer state_dict key to our module's key space."""
        k = chemformer_key
        if k == "emb.weight":
            return "emb.weight"
        if k == "pos_emb":
            return "pos_bias.embedding.weight"
        # encoder.layers.{i}.* -> encoder_layers.{i}.*
        if k.startswith("encoder.layers."):
            rest = k[len("encoder.layers.") :]
            return f"encoder_layers.{rest}"
        # Some Chemformer variants use ``encoder.norm.*``
        if k.startswith("encoder.norm."):
            return "final_norm." + k[len("encoder.norm.") :]
        return None

    # -- forward --------------------------------------------------------
    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pool: bool = True,
    ) -> torch.Tensor:
        """Encode a batch of tokenised SMILES.

        Parameters
        ----------
        token_ids:
            ``(batch, seq_len)`` long tensor.
        attention_mask:
            ``(batch, seq_len)`` with 1 for real tokens and 0 for padding.
        pool:
            If True, return the representation of the BOS (``^``) token as a
            reaction-level feature.  Otherwise return the full sequence output.

        Returns
        -------
        torch.Tensor
            ``(batch, d_model)`` if pool=True else ``(batch, seq_len, d_model)``.
        """
        x = self.emb(token_ids)  # (B, L, D)
        # key_padding_mask: True = position to mask (PyTorch convention)
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0
        seq_len = token_ids.size(1)
        device = token_ids.device
        # (1, H, L, L) additive bias for self-attention
        rel_bias = self.pos_bias(seq_len, device)
        attn_mask = rel_bias.expand(token_ids.size(0), -1, -1, -1).reshape(
            token_ids.size(0) * self.pos_bias.num_heads, seq_len, seq_len
        )
        for layer in self.encoder_layers:
            x = layer(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = self.final_norm(x)
        if pool:
            # Use BOS (index 0 after encoding) as the pooled reaction feature
            return x[:, 0, :]
        return x


# ---------------------------------------------------------------------------
# Lightweight classification head + full scorer
# ---------------------------------------------------------------------------
class ReactionClassificationHead(nn.Module):
    """2-layer MLP head that maps pooled encoder features to a plausibility logit."""

    def __init__(self, d_model: int = 512, hidden_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.net(pooled).squeeze(-1)


class PretrainedReactionScorer(nn.Module):
    """Full model: Chemformer backbone + LoRA adapters (optional) + head."""

    def __init__(
        self,
        backbone: PretrainedChemformerBackbone,
        head: Optional[ReactionClassificationHead] = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head or ReactionClassificationHead(d_model=backbone.hparams["d_model"])

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pooled = self.backbone(token_ids, attention_mask=attention_mask, pool=True)
        return self.head(pooled)

    # -- convenience -----------------------------------------------------
    def score_reactions(
        self,
        smiles_list: Sequence[str],
        tokenizer: ChemformerTokenizer,
        device: torch.device,
        batch_size: int = 16,
    ) -> torch.Tensor:
        """Score a list of reaction SMILES; returns logits on ``device``."""
        self.eval()
        scores: List[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(smiles_list), batch_size):
                batch = smiles_list[start : start + batch_size]
                ids, mask = tokenizer.batch_encode(batch)
                ids = ids.to(device)
                mask = mask.to(device)
                logits = self.forward(ids, attention_mask=mask)
                scores.append(logits.cpu())
        return torch.cat(scores)


# ---------------------------------------------------------------------------
# Checkpoint loader helper (used by adapter.apply_lora + train_pretrained)
# ---------------------------------------------------------------------------
def load_chemformer_state_dict(checkpoint_path: str | Path) -> Dict[str, torch.Tensor]:
    """Load a sanitized Chemformer checkpoint and return its state_dict."""
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    return ckpt.get("state_dict", ckpt)
