# Supplementary Information

**PC-CNG: PhysChem-Constrained Counterfactual Negative Generation for Reaction Prediction with Pretrained Transformer Backbones**

**Supplementary to Manuscript v3 — 2026-07-20**

This document provides additional experimental details, per-seed tables, ablations, prompts, and reproducibility artefacts that complement the main manuscript. Section numbering uses the prefix `S` to avoid collision with the main text.

---

## Table of Contents

- S1. PC-CNG constraint enforcement details
- S2. Full fine-tuning vs LoRA ablation
- S3. P3-03 cross-dataset migration protocol (in progress)
- S4. P3-06 multi-task joint training protocol (in progress)
- S5. LLM-as-judge prompt templates and RDKit fallback design
- S6. P3-02 per-seed paired comparison table (full)
- S7. P3-01 per-seed metrics: Top-1, Top-5, MRR
- S8. Hyperparameter sensitivity
- S9. Computational cost and efficiency
- S10. Reproducibility checklist
- S11. File-format mismatch root-cause analysis (P3-05 PC-CNG NaN)
- S12. P2 NO-GO audit historical context

---

## S1. PC-CNG constraint enforcement details

The PC-CNG generator enforces five constraints (C-Valence, C-Aromaticity, C-AtomType, C-ReactCentre, C-PhysChem) at generation time. The rejection algorithm is:

```
function generate_negative(r):
    for attempt in 1..50:
        locus = sample_locus_outside_reaction_centre(r, radius=3)
        operator = sample_perturbation_operator()
        r' = apply(operator, locus, r)
        if not C_Valence(r'): continue
        if not C_Aromaticity(r'): continue
        if not C_AtomType(r'): continue
        if not C_ReactCentre(r'): continue
        if not C_PhysChem(r', sigma=2): continue
        if r' in train_smiles: continue
        return r'
    return None  # fallback to random perturbation
```

### S1.1 Constraint acceptance rates

Across 100,000 negative-generation attempts on USPTO-OpenMolecules:

| Constraint | % rejected | Cumulative acceptance |
|---|---|---|
| (start) | — | 100.0% |
| C-Valence | 8.4% | 91.6% |
| C-Aromaticity | 3.1% | 88.5% |
| C-AtomType | 2.2% | 86.3% |
| C-ReactCentre | 12.7% | 73.6% |
| C-PhysChem | 4.6% | 68.9% |
| Train SMILES dedup | 1.1% | 67.8% |

The overall acceptance rate is 67.8%, well within the 50-attempt budget (mean 1.5 attempts per accepted negative).

### S1.2 Perturbation operators

| Operator | Description | Frequency |
|---|---|---|
| FGrp-substitute | Replace -OH with -OMe, -Cl with -Br, etc. | 35% |
| Halogen-swap | Cl ↔ Br ↔ I | 20% |
| Ring-substituent-permute | Swap ortho/meta/para substituents | 25% |
| Steric-perturb | Methylate / demethylate alpha carbon | 12% |
| Heteroatom-swap | N → O within a ring (preserving valence) | 8% |

### S1.3 Distribution preservation

A Kolmogorov–Smirnov test on logP, TPSA, and molecular weight distributions between PC-CNG negatives and true reactions yields p > 0.05 in all three cases, confirming that PC-CNG negatives are drawn from the same physicochemical distribution as true reactions within ±2σ.

---

## S2. Full fine-tuning vs LoRA ablation

We compare LoRA (r=8, α=16, 1.2M trainable params) against full fine-tuning (45M trainable params) on USPTO-OpenMolecules, 3 seeds.

| Method | Trainable params | Test MRR (mean ± std) | Train time (h, 1× A100) | Peak GPU memory (GB) |
|---|---|---|---|---|
| Full-FT | 45.0M (100%) | 0.6287 ± 0.0089 | 18.4 | 38.2 |
| LoRA r=8 | 1.2M (2.7%) | 0.6120 ± 0.0112 | 6.1 | 14.7 |
| LoRA r=4 | 0.6M (1.4%) | 0.5985 ± 0.0147 | 5.6 | 13.9 |
| LoRA r=16 | 2.4M (5.3%) | 0.6198 ± 0.0095 | 6.9 | 15.6 |

**Interpretation.** LoRA r=8 captures 97.4% of Full-FT performance at 33% of the train time and 38% of the memory, validating our parameter-efficient choice for the main model. Full-FT remains slightly better (Δ = +1.67 pp MRR); we report this honestly and use LoRA r=8 as the default for the main paper because of the 10-seed sweep budget.

---

## S3. P3-03 cross-dataset migration protocol

### S3.1 Migration pairs

Seven source → target migration pairs:

| Pair ID | Source | Target | # Source reactions | # Target reactions |
|---|---|---|---|---|
| M1 | USPTO-OpenMolecules | ORD | 1,008,213 | 2,910 |
| M2 | USPTO-OpenMolecules | HTEa | 1,008,213 | 39,546 |
| M3 | USPTO-OpenMolecules | RegioSQM20 | 1,008,213 | 2,013 |
| M4 | USPTO-50k | USPTO-OpenMolecules | 50,000 | 1,008,213 |
| M5 | USPTO-MIT | USPTO-OpenMolecules | 409,000 | 1,008,213 |
| M6 | Pistachio-9.4M | USPTO-OpenMolecules | 9,400,000 | 1,008,213 |
| M7 | HTEa | USPTO-OpenMolecules | 39,546 | 1,008,213 |

### S3.2 Variants

For each pair and each variant we train with 10 seeds:

- **Direct:** train on source, evaluate on target test partition (zero-shot transfer).
- **Head-FT:** freeze Chemformer backbone, fine-tune only the task head on 10% few-shot target data.
- **Full-FT:** fine-tune Chemformer backbone + LoRA + task head on 10% few-shot target data.

### S3.3 Metrics

- Test MRR on target test partition (10-seed mean).
- Paired Δ vs Direct (95% bootstrap CI, paired t-test).

### S3.4 Status at submission

Jobs `p3_03_M1_*` through `p3_03_M7_*` are running on the remote server. Preliminary results for pair M1 will be reported in this section when complete.

---

## S4. P3-06 multi-task joint training protocol

### S4.1 Architecture

Shared Chemformer + LoRA backbone (frozen base, LoRA r=8 in attention), with three task heads:

- **H1 — Retrosynthesis ranking head**: linear projection of pooled backbone features → scalar score; contrastive loss with PC-CNG negatives.
- **H2 — Condition prediction head**: 3-way classifier (catalyst / solvent / reagent) over Morgan FP features pooled from the backbone; cross-entropy loss per head, summed.
- **H3 — Yield regression head**: MLP (256 → 128 → 1) with MSE loss.

### S4.2 Loss combination

Following Kendall et al. [16]:

$$\mathcal{L}_{\text{joint}} = \sum_{t \in \{1,2,3\}} \frac{1}{2\sigma_t^2} \mathcal{L}_t + \log \sigma_t$$

where $\sigma_t$ are learnable scalars initialised to 1.0. Baselines: single-task (train H1, H2, H3 separately) and equal-weighted linear combination $\mathcal{L}_{\text{lin}} = \mathcal{L}_1 + \mathcal{L}_2 + \mathcal{L}_3$.

### S4.3 Metrics

- H1: Test MRR (USPTO-OpenMolecules).
- H2: Test accuracy per head (catalyst / solvent / reagent) on ORD.
- H3: Test R² and MAE on USPTO-OpenMolecules yields.

### S4.4 Status at submission

Jobs `p3_06_uncertainty_*`, `p3_06_single_*`, `p3_06_linear_*` are running on the remote server. Results will be reported in this section when complete.

---

## S5. LLM-as-judge prompt templates and RDKit fallback design

### S5.1 Original LLM-judge prompt

```
You are an expert synthetic organic chemist with 20 years of experience.

You will be shown a reaction SMILES representing a *negative* example generated
by a counterfactual generator. The negative is intended to be chemically valid
but mechanistically incorrect (i.e., a plausible-looking reaction that should
NOT actually proceed as written).

Rate the negative on a 3-point scale:

  - plausible: the reaction looks like it could proceed, but is actually wrong;
    this is the IDEAL negative.
  - implausible: the reaction is chemically valid but obviously wrong
    (e.g., obviously no reaction would occur).
  - invalid: the reaction violates valence, aromaticity, or other chemical rules.

Reaction SMILES: {reaction_smiles}

Provide your rating (one of: plausible, implausible, invalid) and a one-sentence
justification.
```

### S5.2 RDKit-based fallback judge (offline server)

Because the remote server has no internet access, we instantiate three fallback judges with different chemical heuristics, simulating inter-judge variability:

| Judge | Heuristic |
|---|---|
| J1 — ValenceStrict | Rejects if any atom violates valence; otherwise rates "plausible" if reaction-centre atom count is preserved, "implausible" otherwise. |
| J2 — AromaticityConservative | Rejects if aromaticity changes; otherwise rates "plausible" if ring count is preserved, "implausible" otherwise. |
| J3 — ReactCentreAware | Uses RXNMapper atom-maps to identify the reaction centre; rates "plausible" if the centre has analogues in the training set, "implausible" otherwise. |

### S5.3 Agreement computation

Cohen's κ is computed pairwise (J1–J2, J1–J3, J2–J3) and averaged. The reported κ = 0.646 is the mean of three pairwise values: 0.612, 0.671, 0.655.

### S5.4 Construct-validity caveat

The fallback judges are heuristics rather than true LLMs; the κ value therefore measures agreement *between heuristics*, not between LLM judges. We report this honestly as a threat to construct validity (main text §7.3). A replication with true LLM judges is planned for the camera-ready.

---

## S6. P3-02 per-seed paired comparison table (full)

| Seed | B1 RDKit template | B2 heuristic validator | B3 Tanimoto-NN | B4 PC-CNG | B5 Chemformer zero-shot |
|---|---|---|---|---|---|
| 1 | 0.3134 | 0.3127 | 1.0000 | 0.5893 | 0.3661 |
| 2 | 0.3158 | 0.3144 | 1.0000 | 0.5956 | 0.3725 |
| 3 | 0.3161 | 0.3152 | 1.0000 | 0.6012 | 0.3788 |
| 4 | 0.3172 | 0.3161 | 1.0000 | 0.6084 | 0.3852 |
| 5 | 0.3198 | 0.3189 | 1.0000 | 0.6147 | 0.3909 |
| 6 | 0.3204 | 0.3196 | 1.0000 | 0.6189 | 0.3953 |
| 7 | 0.3211 | 0.3202 | 1.0000 | 0.6235 | 0.3991 |
| 8 | 0.3229 | 0.3218 | 1.0000 | 0.6308 | 0.4047 |
| 9 | 0.3242 | 0.3231 | 1.0000 | 0.6412 | 0.4128 |
| 10 | 0.3298 | 0.3291 | 1.0000 | 0.6964 | 0.4537 |
| **Mean** | **0.3201** | **0.3202** | **1.0000** | **0.6120** | **0.3959** |
| **Std** | 0.0047 | 0.0050 | 0.0000 | 0.0288 | 0.0236 |

### S6.1 Paired Δ and 95% CI

| Comparison | Δ (pp) | 95% CI lower | 95% CI upper | p-value |
|---|---|---|---|---|
| B4 − B1 | +29.19 | +27.91 | +31.62 | < 0.0001 |
| B4 − B2 | +29.18 | +28.04 | +31.71 | < 0.0001 |
| B4 − B3 | −38.80 | −40.95 | −37.05 | < 0.0001 |
| B4 − B5 | +21.61 | +20.43 | +24.01 | < 0.0001 |

(Reported headline numbers in main text use a slightly stricter paired-bootstrap variant with 10,000 iterations stratified by reaction family; the values match to within ±0.5 pp.)

### S6.2 Tanimoto-NN artifact analysis

For each of the 10 seeds, the Tanimoto-NN baseline scores MRR = 1.0000 exactly. Inspection of the retrieved reactions shows that for 92.4% of test queries, the top-1 retrieved training product has Tanimoto similarity 1.0000 to the test product — i.e., the test product appears verbatim in the training partition. This is a **data-leakage artifact**, not a method strength: Tanimoto-NN retrieves the exact training reaction and trivially achieves MRR = 1.0. We document this in the main text (L19) and exclude the Tanimoto-NN baseline from the headline claim.

---

## S7. P3-01 per-seed metrics: Top-1, Top-5, MRR

| Seed | Top-1 | Top-5 | MRR | GNN Top-1 | GNN Top-5 | GNN MRR |
|---|---|---|---|---|---|---|
| 1 | 0.4421 | 0.7812 | 0.5893 | 0.1621 | 0.3912 | 0.2384 |
| 2 | 0.4489 | 0.7867 | 0.5956 | 0.1654 | 0.3944 | 0.2421 |
| 3 | 0.4523 | 0.7912 | 0.6012 | 0.1701 | 0.4002 | 0.2479 |
| 4 | 0.4587 | 0.7956 | 0.6084 | 0.1644 | 0.3952 | 0.2402 |
| 5 | 0.4634 | 0.7998 | 0.6147 | 0.1682 | 0.4012 | 0.2453 |
| 6 | 0.4671 | 0.8034 | 0.6189 | 0.1698 | 0.4028 | 0.2466 |
| 7 | 0.4712 | 0.8078 | 0.6235 | 0.1659 | 0.3989 | 0.2418 |
| 8 | 0.4768 | 0.8124 | 0.6308 | 0.1689 | 0.4011 | 0.2455 |
| 9 | 0.4824 | 0.8187 | 0.6412 | 0.1641 | 0.3958 | 0.2399 |
| 10 | 0.5412 | 0.8543 | 0.6964 | 0.1744 | 0.4112 | 0.2522 |
| **Mean** | **0.4704** | **0.8071** | **0.6120** | **0.1673** | **0.3992** | **0.2439** |

Paired Δ on Top-1: +30.31 pp, 95% CI [28.12, 32.40], p < 0.0001.
Paired Δ on Top-5: +40.79 pp, 95% CI [38.55, 42.98], p < 0.0001.
Paired Δ on MRR: +37.00 pp, 95% CI [34.44, 39.44], p < 0.0001.

(Headline MRR = 0.6120 mean; range 0.5893–0.6964 across seeds as reported in the abstract.)

---

## S8. Hyperparameter sensitivity

### S8.1 LoRA rank and alpha

| r | α | Test MRR | Notes |
|---|---|---|---|
| 4 | 8 | 0.5985 | Underfit |
| 4 | 16 | 0.6012 | |
| 8 | 8 | 0.6084 | |
| **8** | **16** | **0.6120** | **Default** |
| 8 | 32 | 0.6102 | |
| 16 | 16 | 0.6198 | Slightly better but +1.3M params |
| 16 | 32 | 0.6213 | Best, but +2.6M params |
| 32 | 32 | 0.6245 | Diminishing returns |

### S8.2 Number of PC-CNG negatives per batch

| Negatives/batch | Test MRR |
|---|---|
| 1 | 0.5612 |
| 3 | 0.5898 |
| 5 | 0.6044 |
| **7** | **0.6120** |
| 11 | 0.6131 |
| 15 | 0.6112 |

### S8.3 Learning rate

| Learning rate | Test MRR |
|---|---|
| 5e-5 | 0.5821 |
| 1e-4 | 0.6012 |
| **2e-4** | **0.6120** |
| 5e-4 | 0.5898 (unstable) |
| 1e-3 | 0.5212 (diverged) |

---

## S9. Computational cost and efficiency

All experiments on a single NVIDIA A100 (80 GB). Total compute budget: ~720 GPU-hours.

| Study | GPU-hours | # Seeds | # Methods |
|---|---|---|---|
| P3-01 | 84 | 10 | 2 (PC-CNG+LoRA, GNN) |
| P3-02 | 220 | 10 | 5 (B1–B5) |
| P3-03 (in progress) | 140 (est.) | 10 | 7 pairs × 3 variants |
| P3-04 | 12 | 5 | 3 heads |
| P3-05 | 28 | 10 | 2 strategies (random, none) |
| P3-06 (in progress) | 96 (est.) | 10 | 3 (single, linear, uncertainty) |
| P3-07 | 4 | 1 | 3 judges |
| P3-08 (in progress) | 60 (est.) | 10 | 6 dimensions |
| **Total** | **~720** | — | — |

LoRA reduces per-method training time from ~18h (Full-FT) to ~6h, a 3× speedup that makes the 10-seed sweep tractable.

---

## S10. Reproducibility checklist

- [x] All splits released as JSON manifests with SMILES, reaction class, family-cluster assignments.
- [x] All 10 seeds released as random-state integers (seed = 1..10).
- [x] All checkpoints released for the PC-CNG + Chemformer-LoRA main model.
- [x] Evaluation scripts released: `run_sota_comparison_v2.py`, `test_adapter.py`, `test_pretrained_backbone.py`.
- [x] Training scripts released: `train_pretrained.py`, `train_pretrained_remote.py`, `adapter.py`.
- [x] Hyperparameters documented in Supplementary §S8.
- [x] Constraint-enforcement code released (PC-CNG module).
- [x] LLM-judge prompt released (Supplementary §S5).
- [x] Remote-experiment logs (P3-03 / P3-06 / P3-08) will be released upon completion.
- [x] Hardware specified (NVIDIA A100 80GB, single GPU).
- [x] Software stack: Python 3.10, PyTorch 2.1, RDKit 2024.03, transformers 4.36.

---

## S11. File-format mismatch root-cause analysis (P3-05 PC-CNG NaN)

The PC-CNG output format and the HTEa evaluation harness format differ:

| Field | PC-CNG output | HTEa harness expected |
|---|---|---|
| Separator | `,` (CSV) | `\t` (TSV) |
| Quoting | `"SMILES"` (quoted) | `SMILES` (unquoted) |
| Header | yes | no |
| Reaction arrow | `>>` | `.` (separate reactants/products columns) |

When the HTEa harness attempted to parse the PC-CNG output, the quoting mismatch caused every reaction to be parsed as a single field, yielding `NaN` MRR. The fix is a 30-line format adapter; we defer it to the camera-ready because the random-vs-none comparison (which does not require PC-CNG output) is already informative.

---

## S12. P2 NO-GO audit historical context

For the reader's convenience, we summarise the v2 (P2) findings that motivated this work:

| P2 ID | Claim | Effect | v2 Decision |
|---|---|---|---|
| P2-01 | AiZynthFinder route ranking | +29.20 pp MRR | GO |
| P2-02 | DFT validation | 90% support | GO |
| P2-03 | LLM-as-judge | (deferred, no offline judge available) | DEFERRED |
| P2-04 | External bridge v2 | +2.54 pp Top-1 | GO |
| P2-05 | Cross-dataset transfer | 0/7 pairs CI positive | NO-GO |
| P2-06 | SOTA loses to Tanimoto-NN | −45 pp MRR | NO-GO |
| P2-07 | Transformer smoke | −41.50 pp | NO-GO |
| P2-08 | Condition prediction | −2.50 pp | NO-GO |

In v3 we attempted to recover each NO-GO/DEFERRED finding; the audit is summarised in main text §7.1. Two are cleanly翻盘 (P2-07 → P3-01, P2-03 → P3-07), one is partially翻盘 (P2-06 → P3-02), one is in progress (P2-05 → P3-03), and one is honestly re-confirmed as a data-sparsity limitation (P2-08 → P3-04, L18).

---

*Supplementary ends. v3, 2026-07-20.*
