# Phase 3 G7 Expert Review Pilot Protocol

**Status:** Draft (not yet in `docs/00_当前有效文档/`)
**Owner:** PC-CNG Phase 3 validation
**Created:** 2026-07-25
**Spec reference:** `提示词/pccng 的分阶段提示词.md` (P4-G7 human review)

---

## 1. Purpose

Establish whether a double-blind expert review can reliably discriminate
PC-CNG-generated negative reactions from real reactions and from competing
baselines (random mismatch, template perturbation, rule PC-CNG), *before*
scaling to a full 200–300 item main review. The pilot also estimates
inter-rater reliability so the main review can be powered correctly.

Per the Phase 3 spec, **OOD results should become the main result, not
supplementary**; this pilot is the human-grounded leg of that claim.

## 2. Design overview

| Element | Choice |
|---|---|
| Design | Double-blind, randomized, balanced |
| Sample size | 100 items (80–100 acceptable) |
| Raters | ≥3 independent synthetic chemists |
| Blinding | Raters blind to candidate source; sources randomized per item |
| Scale | Feasibility 1–5, Confidence 1–5, free-text reasoning |
| Primary reliability target | weighted κ ≥ 0.5 (target ≥ 0.6) |
| Pass criteria | See §8 |

## 3. Sample size justification

* 100 items × 3 raters = 300 ratings.
* With 4 candidate sources and 20 items per source, the per-source
  feasibility mean has a 95% CI half-width of roughly
  `1.96 × SD / sqrt(20)`.  Assuming SD ≈ 0.8 on a 1–5 scale, the
  half-width is ≈ 0.35 — sufficient to detect a one-point difference
  between sources.
* 300 ratings give ≥80% power to detect weighted κ = 0.5 vs null κ = 0.3
  at α = 0.05 (two-sided) using a z-test on Fisher-transformed κ.
* The pilot is *not* powered to detect sub-group effects; those are
  deferred to the main review.

## 4. Expert requirements

* **Count:** ≥3 raters, recruited independently (no shared affiliation
  beyond the project).
* **Expertise:** PhD-level or equivalent in synthetic organic /
  organometallic chemistry, with ≥2 years of bench experience in
  cross-coupling, C–H activation, or high-throughput experimentation.
* **Conflict screening:** Raters disclose any prior exposure to the
  HiTEA dataset or PC-CNG codebase; exposed raters are excluded from
  items they recognise.
* **Training:** A 15-minute calibration session using 5 anchor items
  (not counted in the 100) with known labels. Raters must agree with
  the anchor labels on ≥4/5 items before proceeding.
* **Compensation:** Disclosed and time-capped (target ≤90 minutes per
  rater for the 100 items).

## 5. Item composition

100 items, balanced 1:1:1:1:1 across five sources (20 each):

| # | Source | Label | Role |
|---|---|---|---|
| 1 | Real successful HiTEA reactions (high yield) | positive control | validates that raters rate real reactions as feasible |
| 2 | Random product mismatch (rotate product) | obvious negative | floor control; raters should rate these as infeasible |
| 3 | Template perturbation (single-atom swap in product) | template negative | mid-difficulty baseline |
| 4 | PC-CNG learned structured negatives (G8-C model) | candidate | primary subject of evaluation |
| 5 | Rule PC-CNG negatives (ReactionBoundaryGenerator) | rule baseline | direct comparator vs PC-CNG |

* Items are drawn from the OOD test splits (§6) so that the pilot
  simultaneously probes OOD generalisation.
* Each item is a single reaction SMILES rendered as a drawn structure
  (RDKit depiction) plus the conditions (catalyst, solvent,
  temperature) when available.
* The 5 sources are shuffled into a single randomised list per rater;
  no two consecutive items share a source.

## 6. Sampling from OOD splits

* Positive controls: 20 real high-yield reactions sampled from the
  `random` split's test set (stratified by `reaction_family`).
* PC-CNG and rule PC-CNG negatives: 20 each, generated from the same
  20 real reactions drawn from the **`scaffold`** and
  **`reaction_family`** OOD test splits (10 from each), so that the
  pilot directly evaluates OOD negative quality.
* Template and random negatives: generated from the same 20 source
  reactions.
* The 4 negative sources therefore share a *common* source-reaction
  pool, enabling a paired comparison.

## 7. Evaluation form

Each item presents:

1. The reaction (depicted structure + SMILES).
2. Conditions (catalyst, solvent, base, temperature) when available.
3. Three rating fields:

| Field | Type | Scale | Anchors |
|---|---|---|---|
| Feasibility | Likert | 1–5 | 1 = clearly impossible / violates valence or atom balance; 2 = very unlikely; 3 = plausible but doubtful; 4 = likely feasible; 5 = clearly feasible |
| Confidence | Likert | 1–5 | 1 = guessing; 3 = moderate; 5 = certain (literature precedent or mechanistic certainty) |
| Reasoning | Free text | — | One or two sentences citing the specific structural / mechanistic concern |

* Raters may flag an item as "invalid SMILES" or "duplicate" — flagged
  items are excluded from the primary analysis and reported separately.

## 8. Pilot pass criteria

The pilot **passes** if *all* of the following hold:

1. **Inter-rater reliability** — pairwise linearly weighted κ ≥ 0.5 for
   the Feasibility score (averaged across the 3 rater pairs), and
   Krippendorff α (ordinal) ≥ 0.5.  Target ≥ 0.6 for both.
2. **Control discrimination** — a one-sided Wilcoxon signed-rank test
   on per-item Feasibility shows positive controls rated significantly
   higher than random-mismatch negatives (p < 0.01) for each rater.
3. **No severe reviewer drift** — the per-rater mean Feasibility on the
   first 20 items vs the last 20 items differs by < 0.5 on the 1–5
   scale (paired t-test p > 0.05, or |mean diff| < 0.5 whichever is
  stricter).
4. **Coverage** — ≥95% of items receive a valid Feasibility + Confidence
   rating (i.e. <5% missing or flagged-invalid).

If any criterion fails, the protocol is revised (item wording, rater
training, or scale anchors) and the pilot re-run before expanding.

## 9. Metrics

### 9.1 Reliability
* Pairwise linearly weighted κ (Feasibility, Confidence).
* Krippendorff α (ordinal, for Feasibility).
* Intraclass correlation (ICC, two-way random, single-measure) for the
  Feasibility score.

### 9.2 Control discrimination
* Per-rater Wilcoxon signed-rank: positive vs random-mismatch.
* Group-level (3 raters pooled) Kruskal–Wallis across the 5 sources,
  followed by pairwise Dunn tests with Holm correction.

### 9.3 PC-CNG vs baselines
* Primary contrast: PC-CNG (learned structured) vs rule PC-CNG.
  - Paired Wilcoxon signed-rank on per-item Feasibility
    (matched on source reaction).
  - Effect size (rank-biserial correlation).
* Secondary contrasts: PC-CNG vs template, PC-CNG vs random.
* Exploratory: does PC-CNG Feasibility fall *between* positive controls
  and random negatives? (i.e. are the negatives realistic but not
  indistinguishable from real reactions?)

### 9.4 Reasoning analysis
* Free-text reasoning is post-hoc coded into failure-mode categories
  (valence violation, atom-balance violation, mechanistic implausibility,
  missing ligand, unrealistic condition, other).  Inter-coder agreement
  on the category labels is reported (κ on a 20-item sample).

## 10. Main review expansion criteria

The pilot expands to a full 200–300 item main review **only if**:

1. The pilot passes all four criteria in §8.
2. The pilot's PC-CNG vs rule PC-CNG effect size has a 95% CI that
   either excludes zero (significant difference) or is narrow enough
   (half-width < 0.2 on the 1–5 scale) that the main review is powered
   to resolve it.
3. No rater raises a process objection (e.g. ambiguous item rendering)
   that would invalidate the main review's items.

If the pilot's effect size is already significant and large, the main
review may be scoped to confirmatory (200 items).  If the effect is
small or non-significant, the main review expands to 300 items with
additional sub-strata (per reaction family, per OOD split type).

## 11. Pre-registration

Before the first rating is collected, the following are frozen:

* This protocol (version + SHA-256 hash).
* The 100-item set (anonymised IDs + source labels in a sealed envelope;
  the unblinding key is held by a non-rater).
* The analysis script (`chem_negative_sampling/pc_cng/p4_g7_agreement.py`
  or a Phase 3 successor) that computes §9 metrics.
* The pass/fail decision rule (§8).

Any deviation is logged in a deviation register with rationale.

## 12. Limitations & mitigations

| Risk | Mitigation |
|---|---|
| Raters infer source from SMILES style (e.g. atom mapping) | Canonicalise / strip atom maps before depiction; present product-only where possible |
| Small per-source n (20) underpowered for sub-groups | Defer sub-group analysis to main review |
| Rater fatigue after 100 items | Time-capped at 90 min; allow split across two sessions |
| PC-CNG negatives too obviously invalid (floor effect) | Report the distribution; if >50% rated 1, escalate to item redesign |
| PC-CNG negatives indistinguishable from real (ceiling effect) | Report as a positive finding; corroborate with prospective experiment (separate protocol) |

## 13. Artefacts produced

* `docs/phase3_g7_pilot_protocol.md` (this file)
* `data/phase3/g7_pilot_items.csv` — 100 items with anonymised IDs
* `data/phase3/g7_pilot_blinding_key.json` — sealed item→source mapping
* `results/phase3_g7_pilot/ratings.csv` — raw per-rater ratings
* `results/phase3_g7_pilot/reliability.json` — κ, α, ICC
* `results/phase3_g7_pilot/control_discrimination.json` — §9.2 tests
* `results/phase3_g7_pilot/pc_cng_vs_baselines.json` — §9.3 tests
* `results/phase3_g7_pilot/pilot_pass_report.md` — pass/fail decision
