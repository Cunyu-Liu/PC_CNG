# P4 Baseline Lock

**Phase:** P4-G0 (Claim-to-Artifact Evidence Audit)
**Generated:** 2026-07-21
**Lock owner:** Trae (executing `/goal` against `pccng 的分阶段提示词.md` lines 39-520)
**Status:** FROZEN — no P3 artifact may be modified, renamed, or deleted from this point forward

## 1. Purpose

This document freezes the P3 state as the immutable baseline for all P4 work. Per `pccng 的分阶段提示词.md` section 4:

> 不得删除、重命名或覆盖已有 `results/` 子目录；
> manuscript v3 保持不可变；
> Section 0–33 保持历史可追溯；
> 未通过验收不得标记 GO。

Any subsequent P4-GX phase MUST reference this baseline when comparing against P3 results. Any P3 number that disagrees with this baseline is a regression caused by P4 work, not a "correction."

## 2. Git State

| Field | Value |
|---|---|
| Repo root | `/home/cunyuliu/pc_cng_research` |
| Remote | `git@github.com:Cunyu-Liu/PC_CNG.git` |
| Branch | `main` |
| Commit | `392564cac4dd74ea16638d0fd1cca052b3b26006` |

## 3. Immutable Manuscript Artifacts (Hashes)

| Artifact | Path | sha256 (first 16) |
|---|---|---|
| Manuscript v3 | `docs/manuscript_v3_20260720.md` | `c3b68f8510939bff` |
| Supplementary v3 | `docs/manuscript_supplementary_v3_20260720.md` | `b832a5d7d9349228` |
| Journal decision v3 | `docs/target_journal_decision_v3_20260720.md` | `5f3e0d738a14e6d4` |

Full hashes are in `results/p4_initial_state_manifest_20260721.json`. Any byte-level change to these three files after 2026-07-21 invalidates P4 comparisons and must be flagged in `anomaly_report.md` of the offending phase.

## 4. Frozen Headline Result Directories

The following `results/` subdirectories contain the P3 headline numbers cited in manuscript v3. They are FROZEN. P4-G0 audit has already read them; P4-G1+ may add new sibling directories but must not modify these.

| Directory | Experiment | Content sha256 (first 16) |
|---|---|---|
| `results/pretrained_backbone_chemformer_lora_20260720/` | P3-01 PC-CNG + Chemformer-LoRA 10-seed | `54a16141ea5a7420` |
| `results/sota_comparison_v2_fixed_20260721/` | P3-02 SOTA comparison (PC-CNG vs Tanimoto-NN vs Chemformer zero-shot) | `4653da281a75efee` |
| `results/multitask_joint_training_20260720/` | P3-06 multitask joint training | `decf4c5891ee7ed9` |
| `results/cross_dataset_finetune_head_fixed_v2_20260721/` | P3-03 v2 cross-dataset head-FT (10 seeds) | `031e8c5653778402` |
| `results/llm_judge_20260720/` | P3-07 expert judge panel (RDKit `local_expert_offline`, NOT real LLM) | `f41ca393c2aa2771` |
| `results/benchmark_suite_v3_fixed_20260721/` | P3-08 benchmark suite v3 fixed | `74c1830fd956c8e0` |

Per section 4 of the spec, the audit (`results/p4_claim_audit/`) is also a frozen sibling directory.

## 5. Frozen Checkpoint

| Checkpoint | sha256 (first 16) |
|---|---|
| `results/pretrained_backbone_chemformer_lora_20260720/seed20260710/model.pt` | `88706b6e4a1134bb` |

The other 9 seed checkpoints under `pretrained_backbone_chemformer_lora_20260720/seed*/` are part of the frozen directory above (collective content hash `54a16141ea5a7420`).

## 6. Frozen Data Split

| Field | Value |
|---|---|
| Data dir | `data/processed/` |
| Aggregated sha256 (first 16) | `204b962e7949bb45` |
| Note | Aggregated hash of all `*.csv` files under `data/processed/` for tamper detection. P4-G1 may add `data/p4/manifests/` siblings but must not modify `data/processed/`. |

## 7. Frozen Environment

| Tool | Version |
|---|---|
| Python | 3.10 (`/home/cunyuliu/miniconda3/envs/pc_cng_gpu/bin/python`) |
| RDKit | 2025.03.6 |
| PyTorch | 2.6.0+cu124 |
| NumPy | 2.2.6 |
| Pandas | 2.3.3 |
| SciPy | 1.15.2 |
| Conda env | `pc_cng_gpu` |

External tools (AiZynthFinder, DFT, xTB, SOTA comparison baselines) live in their own venvs (`/home/cunyuliu/venvs/aizynthfinder`, `/home/cunyuliu/venvs/dft`, `/home/cunyuliu/venvs/sota`) per the project constraint "外部工具必须在独立环境中安装".

## 8. Frozen Test Suite

| Field | Value |
|---|---|
| Total tests collected | 1115 |
| Claim registry tests (re-run during P4-G0) | 23 passed, 0 failed |
| Audit module | `chem_negative_sampling/pc_cng/audit/run_claim_audit.py` |
| Audit tests | `chem_negative_sampling/tests/test_claim_registry.py` |

## 9. Baseline Numbers (as reported, audit status in parentheses)

These are the manuscript v3 headline numbers as reported, NOT as verified. The P4-G0 audit found **16 MISLABELED or INVALIDATED claims** and **12 UNVERIFIED claims** against this baseline — see `results/p4_claim_audit/anomaly_report.md` for full details. The numbers below are frozen for regression-detection only; they are NOT endorsed as correct.

| Manuscript § | Claim | Reported | Audit status |
|---|---|---|---|
| Abstract | PC-CNG test MRR (10-seed mean) | 0.6120 | PARTIALLY_VERIFIED (recomputed 0.6125) |
| Abstract | Δ vs GNN baseline | +37.00 pp | PARTIALLY_VERIFIED (recomputed +36.94) |
| Abstract | Δ vs Chemformer zero-shot | +21.80 pp | VERIFIED |
| Abstract | LLM-judge κ | 0.646 | MISLABELED (RDKit heuristic, not LLM) |
| Abstract | Nine-dim score | 81/90 = 9.0/10 | UNVERIFIED (decision doc table sums to 67) |
| §5.2 | Backbone params | ~45M | MISLABELED (actual 19.6M) |
| §5.2 | LoRA target modules | attention projections | MISLABELED (actually FFN linear1/linear2) |
| §5.2 | LoRA trainable params | ~1.2M (2.7%) | MISLABELED (actual 377K / 1.9%) |
| §5.2 | Hyperparams | lr=2e-4, batch=64, 50 epochs | MISLABELED (actual lr=1e-4, batch=16, 5 epochs) |

## 10. P4-G0 Verdict

**Status:** `NO_GO`
**`next_phase_allowed`:** `false`

Per the spec's NO-GO criteria:
- There exist un-rebuildable headline claims (12 UNVERIFIED);
- There exist mislabeled implementations not yet corrected (12 MISLABELED);
- There exist invalidated claims from leakage / degenerate tasks / wrong statistics (4 INVALIDATED).

P4-G1 is BLOCKED until the manuscript is corrected (or the offending claims are explicitly deleted/downgraded in a tracked claim diff) and the audit is re-run with a GO verdict. See `docs/p4_claim_audit_20260721.md` section "Required Manuscript Corrections" for the action list.

## 11. Lock Enforcement

Any future Trae session or subagent that:
- modifies `docs/manuscript_v3_20260720.md`, `docs/manuscript_supplementary_v3_20260720.md`, or `docs/target_journal_decision_v3_20260720.md` (unless explicitly instructed by the user as a P4-G0.5 correction pass);
- modifies, renames, or deletes any directory listed in section 4;
- changes the `pc_cng_gpu` conda env (RDKit / PyTorch versions);

is in violation of this lock and must stop and ask the user. The lock can only be lifted by a subsequent `docs/p4_baseline_lock.md` revision that explicitly supersedes this version and records the new hashes.
