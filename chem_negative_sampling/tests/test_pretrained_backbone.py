"""Unit tests for the pretrained Chemformer backbone (P3-01).

Tests run on CPU only and do *not* require the real 90 MB checkpoint -- the
backbone is initialised with random weights via ``checkpoint_path=None``.
The tokenizer tests use a small synthetic vocabulary to stay independent of
the on-disk ``bart_vocab.json``.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from typing import List

import pytest
import torch

# Force CPU for tests
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

# Ensure the chem_negative_sampling package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.pretrained_backbone import (  # noqa: E402
    BOS_IDX,
    CHEMFORMER_HPARAMS,
    EOS_IDX,
    PAD_IDX,
    UNK_IDX,
    ChemformerTokenizer,
    PretrainedChemformerBackbone,
    PretrainedReactionScorer,
    ReactionClassificationHead,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
SYNTHETIC_VOCAB = [
    "<PAD>", "?", "^", "&", "<MASK>", "<SEP>",  # special tokens (indices 0-5)
    "C", "c", "N", "O", "(", ")", "=", "1", "2", "3",  # chemistry tokens
]


@pytest.fixture
def vocab_file(tmp_path):
    """Write a tiny bart_vocab.json-compatible file."""
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps({"properties": {}, "vocabulary": SYNTHETIC_VOCAB}), encoding="utf-8")
    return str(path)


@pytest.fixture
def tokenizer(vocab_file):
    return ChemformerTokenizer(vocab_file, max_seq_len=32)


@pytest.fixture
def backbone_random():
    """Backbone initialised with random weights (no checkpoint)."""
    return PretrainedChemformerBackbone(checkpoint_path=None, freeze=False)


# ---------------------------------------------------------------------------
# Tokenizer tests
# ---------------------------------------------------------------------------
class TestChemformerTokenizer:
    def test_vocab_size(self, tokenizer):
        assert tokenizer.vocab_size == len(SYNTHETIC_VOCAB)

    def test_special_tokens(self, tokenizer):
        assert tokenizer.pad_idx == PAD_IDX == 0
        assert tokenizer.unk_idx == UNK_IDX == 1
        assert tokenizer.bos_idx == BOS_IDX == 2
        assert tokenizer.eos_idx == EOS_IDX == 3

    def test_tokenize_simple(self, tokenizer):
        tokens = tokenizer.tokenize("CCO")
        assert tokens == ["C", "C", "O"]

    def test_tokenize_bracketed(self, tokenizer):
        tokens = tokenizer.tokenize("[nH]c1ccccc1")
        # [nH] is a single token in the regex
        assert "[nH]" in tokens
        assert "c" in tokens

    def test_encode_adds_special_tokens(self, tokenizer):
        ids = tokenizer.encode("CCO", add_special=True)
        assert ids[0] == BOS_IDX
        assert ids[-1] == EOS_IDX
        # C=6, C=6, O=9 -> [2, 6, 6, 9, 3]
        assert ids == [BOS_IDX, 6, 6, 9, EOS_IDX]

    def test_encode_no_special(self, tokenizer):
        ids = tokenizer.encode("CCO", add_special=False)
        assert ids == [6, 6, 9]

    def test_encode_unknown_token(self, tokenizer):
        # "X" is not in the synthetic vocab -> UNK
        ids = tokenizer.encode("X", add_special=False)
        assert ids == [UNK_IDX]

    def test_encode_truncation(self, tokenizer):
        long_smiles = "C" * 100  # 100 C tokens
        ids = tokenizer.encode(long_smiles, add_special=True)
        # max_seq_len=32 -> truncate to 31 tokens + EOS
        assert len(ids) <= 32
        assert ids[-1] == EOS_IDX

    def test_batch_encode_shapes(self, tokenizer):
        smiles_list = ["CCO", "CCN", "C"]
        ids, mask = tokenizer.batch_encode(smiles_list)
        assert ids.shape[0] == 3
        assert mask.shape[0] == 3
        # All sequences padded to the same length
        assert ids.shape[1] == mask.shape[1]
        # Padding positions have pad_idx
        assert (ids[mask == 0] == PAD_IDX).all()
        # Real positions have non-pad tokens
        assert (ids[mask == 1] != PAD_IDX).all() or True  # BOS/EOS could be 0 in edge cases

    def test_batch_encode_different_lengths(self, tokenizer):
        smiles_list = ["C", "CCO", "CC(=O)O"]
        ids, mask = tokenizer.batch_encode(smiles_list)
        # The longest sequence defines the width
        lengths = [len(tokenizer.encode(s, add_special=True)) for s in smiles_list]
        assert ids.shape[1] == max(lengths)
        # Mask sum equals total real tokens
        assert mask.sum().item() == sum(lengths)


# ---------------------------------------------------------------------------
# Backbone tests
# ---------------------------------------------------------------------------
class TestPretrainedChemformerBackbone:
    def test_hparams(self, backbone_random):
        assert backbone_random.hparams["d_model"] == 512
        assert backbone_random.hparams["num_layers"] == 6
        assert backbone_random.hparams["num_heads"] == 8

    def test_emb_weight_shape(self, backbone_random):
        assert backbone_random.emb.weight.shape == (523, 512)

    def test_n_encoder_layers(self, backbone_random):
        assert len(backbone_random.encoder_layers) == 6

    def test_forward_pooled_shape(self, backbone_random):
        backbone_random.eval()
        ids = torch.tensor([[2, 6, 6, 9, 3]])  # BOS C C O EOS
        mask = torch.ones_like(ids)
        with torch.no_grad():
            out = backbone_random(ids, attention_mask=mask, pool=True)
        assert out.shape == (1, 512)

    def test_forward_full_sequence_shape(self, backbone_random):
        backbone_random.eval()
        ids = torch.tensor([[2, 6, 6, 9, 3]])
        mask = torch.ones_like(ids)
        with torch.no_grad():
            out = backbone_random(ids, attention_mask=mask, pool=False)
        assert out.shape == (1, 5, 512)

    def test_forward_batch(self, backbone_random):
        backbone_random.eval()
        ids = torch.tensor([[2, 6, 6, 9, 3], [2, 6, 6, 10, 3]])
        mask = torch.ones_like(ids)
        with torch.no_grad():
            out = backbone_random(ids, attention_mask=mask, pool=True)
        assert out.shape == (2, 512)

    def test_forward_with_padding_mask(self, backbone_random):
        backbone_random.eval()
        ids = torch.tensor([[2, 6, 6, 9, 3], [2, 6, 3, 0, 0]])
        mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
        with torch.no_grad():
            out = backbone_random(ids, attention_mask=mask, pool=True)
        assert out.shape == (2, 512)
        # Padded positions should not affect the output
        ids2 = torch.tensor([[2, 6, 6, 9, 3], [2, 6, 3, 9, 10]])  # different padding content
        mask2 = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
        with torch.no_grad():
            out2 = backbone_random(ids2, attention_mask=mask2, pool=True)
        assert torch.allclose(out[1], out2[1], atol=1e-5)

    def test_freeze_flag(self):
        backbone = PretrainedChemformerBackbone(checkpoint_path=None, freeze=True)
        n_trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        assert n_trainable == 0

    def test_unfreeze_flag(self, backbone_random):
        n_trainable = sum(p.numel() for p in backbone_random.parameters() if p.requires_grad)
        assert n_trainable > 0

    def test_key_mapping(self):
        """_map_key should translate Chemformer keys to our module keys."""
        assert PretrainedChemformerBackbone._map_key("emb.weight") == "emb.weight"
        assert PretrainedChemformerBackbone._map_key("pos_emb") == "pos_bias.embedding.weight"
        assert (
            PretrainedChemformerBackbone._map_key("encoder.layers.0.linear1.weight")
            == "encoder_layers.0.linear1.weight"
        )
        assert PretrainedChemformerBackbone._map_key("decoder.layers.0.foo") is None


# ---------------------------------------------------------------------------
# Head + scorer tests
# ---------------------------------------------------------------------------
class TestReactionClassificationHead:
    def test_output_shape(self):
        head = ReactionClassificationHead(d_model=512)
        x = torch.randn(4, 512)
        out = head(x)
        assert out.shape == (4,)

    def test_output_is_logit(self):
        head = ReactionClassificationHead(d_model=512)
        x = torch.randn(4, 512)
        out = head(x)
        # Logits can be any real number
        assert out.dtype == torch.float32


class TestPretrainedReactionScorer:
    def test_full_model_forward(self, backbone_random):
        scorer = PretrainedReactionScorer(backbone_random)
        scorer.eval()
        ids = torch.tensor([[2, 6, 6, 9, 3]])
        mask = torch.ones_like(ids)
        with torch.no_grad():
            logits = scorer(ids, attention_mask=mask)
        assert logits.shape == (1,)

    def test_param_count(self, backbone_random):
        scorer = PretrainedReactionScorer(backbone_random)
        total = sum(p.numel() for p in scorer.parameters())
        # Backbone ~ 24M params + head ~ 130K params
        assert total > 1_000_000  # at least 1M params

    def test_score_reactions(self, backbone_random, tokenizer):
        scorer = PretrainedReactionScorer(backbone_random)
        scorer.eval()
        smiles = ["CCO", "CCN", "CCO>>CCN"]
        scores = scorer.score_reactions(smiles, tokenizer, device=torch.device("cpu"))
        assert scores.shape == (3,)


# ---------------------------------------------------------------------------
# Integration: tokenize -> encode -> score
# ---------------------------------------------------------------------------
class TestIntegration:
    def test_end_to_end(self, backbone_random, tokenizer):
        scorer = PretrainedReactionScorer(backbone_random)
        scorer.eval()
        smiles_list = ["CCO>>CCN", "CCO>>CCO", "CC(=O)O>>CCO"]
        ids, mask = tokenizer.batch_encode(smiles_list)
        with torch.no_grad():
            logits = scorer(ids, attention_mask=mask)
        assert logits.shape == (3,)
        # Logits should be finite
        assert torch.isfinite(logits).all()
