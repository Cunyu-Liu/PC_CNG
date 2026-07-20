"""Unit tests for the Transformer-based negative generator (P2-07, L7 fix).

Tests run on CPU only (``CUDA_VISIBLE_DEVICES=`` is forced by the test runner).
Covers: module imports, CLI args parsing, the three generators, metric
computation, paired significance structure, Go/No-Go decision, fallback when
the Chemformer checkpoint is unavailable, and a small end-to-end smoke run
with a mocked transformer.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from typing import List

import pytest
import torch

# Force CPU for tests.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

# Ensure the chem_negative_sampling package is importable.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pc_cng.transformer_negative_generator import (
    DEFAULT_SEEDS,
    GO_NO_GO_THRESHOLD_PP,
    GeneratedNegative,
    GNNGenerator,
    RuleBasedGenerator,
    SMILESTokenizer,
    TransformerGenerator,
    TransformerSeq2Seq,
    _check_chemformer_availability,
    compute_diversity,
    compute_generator_metrics,
    compute_top1_reranking,
    compute_validity,
    go_no_go_decision,
    load_train_pairs,
    load_val_rows,
    make_causal_mask,
    main,
    paired_significance_test,
    parse_args,
    resolve_device,
    run_single_seed,
    set_seed,
    write_generated_sample_csv,
    write_per_seed_csv,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Realistic small reactions (atom-mapped + unmapped) used across tests.
POS_REACTION_A = "CC(=O)O.CCO>>CCOC(=O)C.O"  # esterification
POS_REACTION_B = "O=C1CCCCCCC1.c1cn[nH]c1>>Brc1cn[nH]c1"  # regio
POS_REACTION_C = "NCC.CC(=O)O>>CC(=O)NCC.O"  # amide formation


@pytest.fixture
def val_rows() -> List[dict]:
    return [
        {"source_id": "src_1", "reaction_smiles": POS_REACTION_A},
        {"source_id": "src_2", "reaction_smiles": POS_REACTION_B},
        {"source_id": "src_3", "reaction_smiles": POS_REACTION_C},
    ]


@pytest.fixture
def train_pairs() -> List[tuple]:
    """(positive, candidate_negative) pairs mirroring the PC-CNG CSV schema."""
    return [
        (POS_REACTION_A, "CC(=O)O.CCO>>CC(=O)O.CCO"),  # no_reaction
        (POS_REACTION_A, "CCO>>CCOC(=O)C.O"),  # drop reactant
        (POS_REACTION_B, "O=C1CCCCCCC1.c1cn[nH]c1>>c1cn[nH]c1"),  # identity-ish
        (POS_REACTION_C, "NCC.CC(=O)O>>NCC.CC(=O)O"),  # no_reaction
        (POS_REACTION_C, "NCC>>CC(=O)NCC.O"),  # drop reactant
    ]


# ---------------------------------------------------------------------------
# 1. Module imports
# ---------------------------------------------------------------------------


def test_module_imports() -> None:
    """Test 1: all public symbols are importable from the module."""
    assert DEFAULT_SEEDS.count(",") == 9, "DEFAULT_SEEDS should list 10 seeds"
    assert GO_NO_GO_THRESHOLD_PP == 1.0
    assert callable(set_seed)
    assert callable(compute_diversity)
    assert callable(compute_validity)
    assert callable(compute_top1_reranking)
    assert callable(go_no_go_decision)
    assert callable(paired_significance_test)
    assert callable(run_single_seed)
    assert callable(main)


def test_default_seeds_are_10() -> None:
    """The default seed list must contain exactly 10 seeds (10-seed paired eval)."""
    seeds = [int(s) for s in DEFAULT_SEEDS.split(",")]
    assert len(seeds) == 10
    assert all(isinstance(s, int) for s in seeds)
    assert len(set(seeds)) == 10  # all unique


# ---------------------------------------------------------------------------
# 2. CLI args parsing
# ---------------------------------------------------------------------------


def test_parse_args_required() -> None:
    """Test 2a: required args are enforced."""
    with pytest.raises(SystemExit):
        parse_args([])  # missing required args


def test_parse_args_defaults() -> None:
    """Test 2b: defaults match the task spec (epochs=5, batch=32, max-train=10000)."""
    args = parse_args([
        "--train-data", "/tmp/train.csv",
        "--val-data", "/tmp/val.csv",
        "--output-dir", "/tmp/out",
    ])
    assert args.train_data == "/tmp/train.csv"
    assert args.val_data == "/tmp/val.csv"
    assert args.output_dir == "/tmp/out"
    assert args.epochs == 5
    assert args.batch_size == 32
    assert args.max_train_samples == 10000
    assert args.device == "cuda:0"
    assert args.chemformer_checkpoint == ""
    # default seeds has 10 entries
    assert len(args.seeds.split(",")) == 10


def test_parse_args_custom_seeds() -> None:
    """Test 2c: custom seed list is parsed correctly."""
    args = parse_args([
        "--train-data", "a", "--val-data", "b", "--output-dir", "c",
        "--seeds", "1,2,3",
        "--epochs", "1", "--batch-size", "8", "--max-train-samples", "5",
    ])
    assert args.seeds == "1,2,3"
    assert args.epochs == 1
    assert args.batch_size == 8
    assert args.max_train_samples == 5


# ---------------------------------------------------------------------------
# 3. Generator comparison logic (3 generators)
# ---------------------------------------------------------------------------


def test_rule_based_generator_produces_valid_schema(val_rows) -> None:
    """Test 3a: G1 rule-based generator produces GeneratedNegative objects."""
    gen = RuleBasedGenerator(seed=42)
    neg = gen.generate(POS_REACTION_A, source_id="test_a")
    assert neg is not None
    assert isinstance(neg, GeneratedNegative)
    assert neg.generator == "rule"
    assert neg.positive_reaction == POS_REACTION_A
    assert ">>" in neg.candidate_reaction
    assert neg.source_id == "test_a"


def test_gnn_generator_fallback_when_no_checkpoint(val_rows) -> None:
    """Test 3b: G2 falls back to a deterministic perturbation when no checkpoint is given."""
    gen = GNNGenerator(seed=42, checkpoint=None, device="cpu")
    assert gen.fallback_reason == "no_checkpoint"
    neg = gen.generate(POS_REACTION_A, source_id="test_a")
    assert neg is not None
    # fallback generator tag must be distinguishable from "gnn"
    assert neg.generator in ("gnn_fallback",)
    assert ">>" in neg.candidate_reaction


def test_gnn_generator_fallback_for_missing_checkpoint_file() -> None:
    """Test 3c: G2 falls back when the checkpoint path does not exist."""
    gen = GNNGenerator(seed=42, checkpoint="/nonexistent/ckpt.pt", device="cpu")
    assert gen.fallback_reason == "no_checkpoint"
    assert gen.model is None


def test_transformer_generator_end_to_end(train_pairs) -> None:
    """Test 3d: G3 transformer trains on tiny data and generates a reaction."""
    set_seed(0)
    tokenizer = SMILESTokenizer()
    tokenizer.build_vocab([p[0] for p in train_pairs])
    tokenizer.build_vocab([p[1] for p in train_pairs])
    gen = TransformerGenerator(
        tokenizer, seed=0, device="cpu",
        d_model=16, nhead=2, num_layers=1, ff_dim=32, max_len=32, dropout=0.0,
    )
    history = gen.train(train_pairs, epochs=1, batch_size=2, lr=1e-3)
    assert len(history) == 1
    assert "loss" in history[0]
    neg = gen.generate(POS_REACTION_A, source_id="test_a")
    assert neg is not None
    assert neg.generator in ("transformer", "transformer_fallback")
    assert neg.positive_reaction == POS_REACTION_A
    assert ">>" in neg.candidate_reaction  # produces something reaction-shaped


def test_three_generators_produce_different_outputs(val_rows, train_pairs) -> None:
    """Test 3e: the three generators produce distinguishable outputs on the same input."""
    g1 = RuleBasedGenerator(seed=42)
    g2 = GNNGenerator(seed=42, checkpoint=None, device="cpu")
    set_seed(0)
    tokenizer = SMILESTokenizer()
    tokenizer.build_vocab([p[0] for p in train_pairs])
    tokenizer.build_vocab([p[1] for p in train_pairs])
    g3 = TransformerGenerator(
        tokenizer, seed=0, device="cpu",
        d_model=16, nhead=2, num_layers=1, ff_dim=32, max_len=32, dropout=0.0,
    )
    g3.train(train_pairs, epochs=1, batch_size=2, lr=1e-3)
    n1 = g1.generate(POS_REACTION_A, source_id="s")
    n2 = g2.generate(POS_REACTION_A, source_id="s")
    n3 = g3.generate(POS_REACTION_A, source_id="s")
    assert n1 is not None and n2 is not None and n3 is not None
    assert n1.generator == "rule"
    assert n2.generator == "gnn_fallback"
    assert n3.generator in ("transformer", "transformer_fallback")


# ---------------------------------------------------------------------------
# 4. Metric computation (Top-1, diversity, validity)
# ---------------------------------------------------------------------------


def test_compute_diversity_empty() -> None:
    assert compute_diversity([]) == 0.0


def test_compute_diversity_all_unique() -> None:
    negs = [
        GeneratedNegative("s1", POS_REACTION_A, "A.B>>C", "rule"),
        GeneratedNegative("s2", POS_REACTION_A, "A.B>>D", "rule"),
    ]
    assert compute_diversity(negs) == 1.0


def test_compute_diversity_all_identical() -> None:
    negs = [
        GeneratedNegative("s1", POS_REACTION_A, "A.B>>C", "rule"),
        GeneratedNegative("s2", POS_REACTION_A, "A.B>>C", "rule"),
    ]
    assert compute_diversity(negs) == pytest.approx(0.5)


def test_compute_validity_empty() -> None:
    assert compute_validity([]) == 0.0


def test_compute_validity_mixed() -> None:
    negs = [
        GeneratedNegative("s1", POS_REACTION_A, "CCO.CC(=O)O>>CCOC(=O)C", "rule"),  # valid
        GeneratedNegative("s2", POS_REACTION_A, "XYZ>>QQQ", "rule"),  # invalid
    ]
    val = compute_validity(negs)
    assert 0.0 < val <= 1.0


def test_compute_top1_reranking_returns_float(val_rows) -> None:
    """Test 4: Top-1 reranking returns a float in [0, 1]."""
    negs = [
        GeneratedNegative("src_1", POS_REACTION_A, "CCO>>CCOC(=O)C", "rule"),
        GeneratedNegative("src_2", POS_REACTION_B, "c1cn[nH]c1>>c1cn[nH]c1", "rule"),
    ]
    top1 = compute_top1_reranking(negs, val_rows, seed=42)
    assert isinstance(top1, float)
    assert 0.0 <= top1 <= 1.0


def test_compute_top1_reranking_empty(val_rows) -> None:
    assert compute_top1_reranking([], val_rows, seed=0) == 0.0
    assert compute_top1_reranking([], [], seed=0) == 0.0


def test_compute_generator_metrics_returns_all_keys(val_rows) -> None:
    negs = [
        GeneratedNegative("src_1", POS_REACTION_A, "CCO>>CCOC(=O)C", "rule"),
    ]
    m = compute_generator_metrics(negs, val_rows, seed=42)
    assert set(m.keys()) == {"count", "top1", "diversity", "validity"}
    assert m["count"] == 1.0
    assert 0.0 <= m["top1"] <= 1.0
    assert 0.0 <= m["diversity"] <= 1.0
    assert 0.0 <= m["validity"] <= 1.0


# ---------------------------------------------------------------------------
# 5. Paired significance test structure
# ---------------------------------------------------------------------------


def test_paired_significance_structure() -> None:
    """Test 5: paired_significance_test returns the expected structure."""
    g1 = [0.50, 0.52, 0.48, 0.51, 0.49]
    g2 = [0.51, 0.53, 0.49, 0.52, 0.50]
    g3 = [0.55, 0.57, 0.53, 0.56, 0.54]
    result = paired_significance_test(g1, g2, g3, iterations=50, seed=42)
    assert "g3_vs_g1" in result
    assert "g3_vs_g2" in result
    assert "g2_vs_g1" in result
    assert "iterations" in result
    assert "seed" in result
    for key in ("g3_vs_g1", "g3_vs_g2", "g2_vs_g1"):
        sub = result[key]
        assert set(sub.keys()) == {
            "mean_delta", "ci_low", "ci_high",
            "paired_permutation_p", "sign_test_p", "n_pairs",
        }
        assert sub["n_pairs"] == 5
        assert 0.0 <= sub["paired_permutation_p"] <= 1.0
        assert 0.0 <= sub["sign_test_p"] <= 1.0
        assert sub["ci_low"] <= sub["mean_delta"] <= sub["ci_high"] or sub["mean_delta"] == 0.0


def test_paired_significance_empty() -> None:
    result = paired_significance_test([], [], [], iterations=10, seed=0)
    assert result["g3_vs_g1"]["n_pairs"] == 0
    assert result["g3_vs_g1"]["paired_permutation_p"] == 1.0


# ---------------------------------------------------------------------------
# 6. Go/No-Go decision
# ---------------------------------------------------------------------------


def test_go_no_go_go_when_g3_beats_g1_by_threshold() -> None:
    """Test 7: Go/No-Go is GO when G3 Top-1 >= G1 Top-1 + 1.0 pp."""
    decision = go_no_go_decision(g1_top1_mean=0.5000, g3_top1_mean=0.5200)
    assert decision["decision"] == "GO"
    assert decision["delta_pp"] == pytest.approx(2.0)
    assert decision["passes"] is True
    assert decision["threshold_pp"] == 1.0


def test_go_no_go_no_go_when_g3_below_threshold() -> None:
    """Test 7b: Go/No-Go is NO-GO when delta < 1.0 pp."""
    decision = go_no_go_decision(g1_top1_mean=0.5000, g3_top1_mean=0.5005)
    assert decision["decision"] == "NO-GO"
    assert decision["delta_pp"] == pytest.approx(0.05, abs=1e-6)
    assert decision["passes"] is False


def test_go_no_go_boundary_exact_threshold_is_go() -> None:
    """Test 7c: exactly +1.0 pp is GO (>=)."""
    decision = go_no_go_decision(g1_top1_mean=0.50, g3_top1_mean=0.51)
    assert decision["decision"] == "GO"
    assert decision["delta_pp"] == pytest.approx(1.0)


def test_go_no_go_custom_threshold() -> None:
    decision = go_no_go_decision(0.5, 0.53, threshold_pp=2.5)
    assert decision["decision"] == "GO"
    assert decision["delta_pp"] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# 7. Fallback when Chemformer checkpoint unavailable
# ---------------------------------------------------------------------------


def test_chemformer_availability_check_missing_pkg_and_checkpoint() -> None:
    """Test 8a: _check_chemformer_availability reports missing pkg/checkpoint."""
    status = _check_chemformer_availability("")
    assert status["package_importable"] is False
    assert status["checkpoint_exists"] is False
    assert status["usable"] is False
    assert "not importable" in status["degradation_reason"]


def test_chemformer_availability_check_missing_checkpoint_only(tmp_path) -> None:
    """Test 8b: if pkg missing but a checkpoint file exists, still not usable."""
    fake_ckpt = tmp_path / "fake.ckpt"
    fake_ckpt.write_text("not a real checkpoint")
    status = _check_chemformer_availability(str(fake_ckpt))
    assert status["package_importable"] is False
    assert status["checkpoint_exists"] is True
    assert status["usable"] is False
    assert status["degradation_reason"] == "chemformer package not importable"


def test_gnn_generator_fallback_produces_valid_reaction() -> None:
    """Test 8c: the GNN fallback path produces a parseable reaction SMILES."""
    gen = GNNGenerator(seed=0, checkpoint=None, device="cpu")
    neg = gen.generate(POS_REACTION_C, source_id="s")
    assert neg is not None
    assert neg.generator == "gnn_fallback"
    assert ">>" in neg.candidate_reaction


# ---------------------------------------------------------------------------
# 8. Synthetic small end-to-end run (mocked Chemformer / tiny transformer)
# ---------------------------------------------------------------------------


def test_run_single_seed_smoke(train_pairs, val_rows) -> None:
    """Test 9: run_single_seed runs end-to-end on tiny data and returns the expected keys."""
    args = parse_args([
        "--train-data", "ignored",  # data already loaded
        "--val-data", "ignored",
        "--output-dir", "ignored",
        "--epochs", "1",
        "--batch-size", "4",
        "--max-train-samples", "5",
        "--d-model", "16", "--nhead", "2", "--num-layers", "1",
        "--ff-dim", "32", "--max-len", "32",
    ])
    result = run_single_seed(
        seed=42, train_pairs=train_pairs, val_rows=val_rows,
        args=args, device=torch.device("cpu"),
    )
    assert result["seed"] == 42
    for gen in ("g1", "g2", "g3"):
        m = result[gen]
        assert set(m.keys()) == {"count", "top1", "diversity", "validity"}
        assert m["count"] >= 0.0
        assert 0.0 <= m["top1"] <= 1.0
        assert 0.0 <= m["diversity"] <= 1.0
        assert 0.0 <= m["validity"] <= 1.0
    # G2 should have run via the fallback path.
    assert result["g2_fallback_reason"] == "no_checkpoint"


def test_main_writes_all_outputs(train_pairs, val_rows, tmp_path) -> None:
    """Test 10: main() writes the 5 required output files."""
    # Write the train CSV in PC-CNG synthetic negatives schema.
    train_csv = tmp_path / "train.csv"
    with open(train_csv, "w", newline="") as handle:
        handle.write("source_id,positive_reaction,candidate_reaction,label\n")
        for i, (pos, cand) in enumerate(train_pairs):
            handle.write(f"s{i},{pos},{cand},0\n")

    val_csv = tmp_path / "val.csv"
    with open(val_csv, "w", newline="") as handle:
        handle.write("source_id,reaction_smiles\n")
        for i, row in enumerate(val_rows):
            handle.write(f"v{i},{row['reaction_smiles']}\n")

    out_dir = tmp_path / "out"
    args_list = [
        "--train-data", str(train_csv),
        "--val-data", str(val_csv),
        "--output-dir", str(out_dir),
        "--seeds", "1,2",
        "--epochs", "1",
        "--batch-size", "4",
        "--max-train-samples", "5",
        "--device", "cpu",
        "--d-model", "16", "--nhead", "2", "--num-layers", "1",
        "--ff-dim", "32", "--max-len", "32",
        "--significance-iterations", "20",
    ]
    summary = main(args_list)

    # Five required outputs.
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "per_seed_metrics.csv").exists()
    assert (out_dir / "paired_significance.json").exists()
    assert (out_dir / "go_no_go_decision.json").exists()
    assert (out_dir / "generated_negatives_sample.csv").exists()

    # summary.json is valid JSON and has the right keys.
    with open(out_dir / "summary.json") as handle:
        loaded = json.load(handle)
    assert loaded["n_seeds"] == 2
    assert loaded["task"].startswith("P2-07")
    assert "g1_top1_mean" in loaded
    assert "g2_top1_mean" in loaded
    assert "g3_top1_mean" in loaded
    assert loaded["chemformer_status"]["package_importable"] is False
    assert loaded["degradation_path"] == "small_pytorch_transformer_from_scratch"

    # per_seed_metrics.csv has a header + 2 rows.
    with open(out_dir / "per_seed_metrics.csv") as handle:
        lines = handle.read().strip().split("\n")
    assert len(lines) == 3  # header + 2 seeds

    # go_no_go_decision.json has the decision field.
    with open(out_dir / "go_no_go_decision.json") as handle:
        decision = json.load(handle)
    assert decision["decision"] in ("GO", "NO-GO")
    assert decision["threshold_pp"] == 1.0
    assert "delta_pp" in decision

    # paired_significance.json structure.
    with open(out_dir / "paired_significance.json") as handle:
        sig = json.load(handle)
    for key in ("g3_vs_g1", "g3_vs_g2", "g2_vs_g1"):
        assert key in sig

    assert isinstance(summary, dict)


# ---------------------------------------------------------------------------
# 9. Tokenizer + model helpers
# ---------------------------------------------------------------------------


def test_smiles_tokenizer_round_trip() -> None:
    tok = SMILESTokenizer()
    smi = "CC(=O)O>>CCOC(=O)C"
    tok.build_vocab([smi])
    encoded = tok.encode(smi, max_len=32)
    assert encoded.shape == (32,)
    assert encoded[0].item() == tok.sos_idx
    # decode (strip special tokens)
    decoded = tok.decode(encoded)
    assert decoded == smi


def test_smiles_tokenizer_handles_unknown_chars() -> None:
    tok = SMILESTokenizer()
    tok.build_vocab(["ABC"])
    encoded = tok.encode("ABZ", max_len=8)  # Z is unknown
    assert tok.unk_idx in encoded.tolist()


def test_make_causal_mask_shape_and_symmetry() -> None:
    mask = make_causal_mask(4, torch.device("cpu"))
    assert mask.shape == (4, 4)
    assert mask.dtype == torch.bool
    # lower triangle (including diagonal) must be False (not masked)
    for i in range(4):
        for j in range(4):
            if j <= i:
                assert not mask[i, j]
            else:
                assert mask[i, j]


def test_transformer_seq2seq_forward_shape() -> None:
    torch.manual_seed(0)
    vocab = 20
    model = TransformerSeq2Seq(
        vocab_size=vocab, d_model=16, nhead=2, num_layers=1, ff_dim=32, pad_idx=0,
    )
    src = torch.randint(1, vocab, (2, 8))
    tgt = torch.randint(1, vocab, (2, 6))
    out = model(src, tgt)
    assert out.shape == (2, 6, vocab)


def test_transformer_greedy_decode_returns_batch_first() -> None:
    torch.manual_seed(0)
    vocab = 15
    model = TransformerSeq2Seq(
        vocab_size=vocab, d_model=16, nhead=2, num_layers=1, ff_dim=32, pad_idx=0,
    )
    src = torch.randint(1, vocab, (3, 8))
    out = model.greedy_decode(src, sos_idx=1, eos_idx=2, max_len=12)
    assert out.shape[0] == 3
    assert out.shape[1] <= 12
    # first token is sos
    assert (out[:, 0] == 1).all()


# ---------------------------------------------------------------------------
# 10. Device resolution + writers
# ---------------------------------------------------------------------------


def test_resolve_device_cpu_when_no_cuda() -> None:
    """resolve_device returns CPU when cuda is not available / forced off."""
    # CUDA_VISIBLE_DEVICES is forced to "" at the top of this test module,
    # so cuda should not be available.
    dev = resolve_device("cuda:0")
    assert dev.type == "cpu"


def test_write_per_seed_csv_format(tmp_path) -> None:
    per_seed = [
        {
            "seed": 1,
            "g1": {"top1": 0.5, "diversity": 0.8, "validity": 1.0, "count": 3.0},
            "g2": {"top1": 0.6, "diversity": 0.9, "validity": 1.0, "count": 3.0},
            "g3": {"top1": 0.55, "diversity": 0.7, "validity": 1.0, "count": 3.0},
            "g2_fallback_reason": "no_checkpoint",
        }
    ]
    path = tmp_path / "per_seed.csv"
    write_per_seed_csv(per_seed, str(path))
    with open(path) as handle:
        content = handle.read()
    assert "seed" in content
    assert "g1_top1" in content
    assert "g2_fallback_reason" in content
    assert "no_checkpoint" in content


def test_write_generated_sample_csv_truncates_to_limit(tmp_path) -> None:
    negs = [
        GeneratedNegative(f"s{i}", POS_REACTION_A, f"A.{i}>>B", "transformer")
        for i in range(150)
    ]
    path = tmp_path / "sample.csv"
    write_generated_sample_csv(negs, str(path), limit=100)
    with open(path) as handle:
        lines = handle.read().strip().split("\n")
    assert len(lines) == 101  # header + 100


# ---------------------------------------------------------------------------
# 11. Data loading helpers
# ---------------------------------------------------------------------------


def test_load_train_pairs_reads_csv(tmp_path) -> None:
    csv_path = tmp_path / "train.csv"
    csv_path.write_text(
        "source_id,positive_reaction,candidate_reaction,label\n"
        "s1,A.B>>C,A.B>>D,0\n"
        "s2,A.B>>C,A.B>>C,1\n"  # skipped: label != 0
        "s3,,A.B>>D,0\n"  # skipped: empty positive
    )
    pairs = load_train_pairs(str(csv_path))
    assert len(pairs) == 1
    assert pairs[0] == ("A.B>>C", "A.B>>D")


def test_load_train_pairs_max_samples(tmp_path) -> None:
    csv_path = tmp_path / "train.csv"
    rows = ["source_id,positive_reaction,candidate_reaction,label"]
    for i in range(10):
        rows.append(f"s{i},A.B>>C,A.B>>D{i},0")
    csv_path.write_text("\n".join(rows))
    pairs = load_train_pairs(str(csv_path), max_samples=3)
    assert len(pairs) == 3


def test_load_train_pairs_missing_file_returns_empty() -> None:
    assert load_train_pairs("/nonexistent/path.csv") == []


def test_load_val_rows_reads_csv(tmp_path) -> None:
    csv_path = tmp_path / "val.csv"
    csv_path.write_text(
        "source_id,reaction_smiles\n"
        "v1,A.B>>C\n"
        "v2,invalid_no_arrows\n"  # skipped
    )
    rows = load_val_rows(str(csv_path))
    assert len(rows) == 1
    assert rows[0]["reaction_smiles"] == "A.B>>C"
