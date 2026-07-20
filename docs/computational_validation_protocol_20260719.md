# Computational Validation Protocol for Synthetic Negatives (P1-10)

**Task**: PC-CNG Section 22.1 P1-10 — small-scale HTE / xTB / DFT computational
validation of high-confidence synthetic negatives.
**Date**: 2026-07-19
**Owner**: PC-CNG research team
**Status**: Completed (MMFF94 degraded path; xTB / DFT unavailable in env)
**Go/No-Go verdict**: **NO_GO_partial_support** (overall support rate 48% < 60%
threshold; `chemoselectivity_error` subset 66.7% ≥ 60%).

---

## 1. Validation Objective

Provide *independent computational evidence* that the synthetic negatives
emitted by P1-08 (`high_confidence_negatives.csv`, 26,517 rows) are genuinely
*low-feasibility* reactions — i.e. that the candidate reactions are
thermodynamically or kinetically disfavoured under standard conditions.

The synthetic-negative pipeline (P1-08) judges a candidate to be a negative
based on graph-edit heuristics (atom-balance, locality, closeness,
ensemble-agreement, database-retrieval, rule-based plausibility).  This
protocol cross-checks those heuristic verdicts against an orthogonal,
physics-based signal: the reaction free energy ΔG and a rough force-field
barrier estimate.

## 2. Computational Methods (Priority & Degradation Path)

| Priority | Method      | Status in env               | Notes |
|----------|-------------|-----------------------------|-------|
| 1        | DFT (B3LYP / ωB97X-D) | Not available (no Gaussian / ORCA / Psi4 install) | Would give the most accurate ΔG and barrier; out of scope for this run. |
| 2        | xTB (GFN2-xTB) | `xtb-python` not installable (`pip install xtb-python` => `No matching distribution found`); no `xtb` binary on PATH | Semi-empirical quantum chemistry; would be the preferred light-weight QC method. |
| 3        | **MMFF94** (Merck Molecular Force Field 94) | **Available** (RDKit 2026.03.3) | **Actual method used for 93/100 reactions.** |
| 4        | UFF (Universal Force Field) | Available (RDKit) | Fallback when MMFF94 lacks parameters (e.g. organometallic atoms Zn, Cu); used for **7/100** reactions. |

**Actual method**: MMFF94 with automatic UFF fallback (`MMFFHasAllMoleculeParams`
gate; 7 reactions containing Zn / Cu / uncommon charge states fell back to UFF).
The `xtb` method requested via `--method xtb` auto-degrades to MMFF94 with a
warning (xtb-python import fails).

### 2.1 Why MMFF94 (not xTB / DFT)

- `xtb-python` is not pip-installable in the `pc_cng` conda env (Python 3.10,
  linux); the upstream wheel is not built for this interpreter.
- A full DFT pipeline (ORCA / Gaussian) would require ≈ 1–10 CPU-hours per
  reaction × 100 reactions = 100–1000 CPU-hours, plus licence setup, which is
  out of scope for the P1-10 budget.
- MMFF94 is fit-for-purpose for a *screening-level* thermodynamic sanity
  check: it reproduces conformational energies to ≈ 0.5–1 kcal/mol for
  drug-like organics and is widely used in reaction-plausibility filters
  (Hartenfeller et al., *J. Chem. Inf. Model.* 2012; Marcou et al.,
  *J. Chem. Inf. Model.* 2015).

### 2.2 Known limitations of MMFF94 for reaction energetics

1. MMFF94 is a *molecular* force field — it has no bonded-interaction terms
   for the transition state, so the barrier estimate is necessarily a rough
   proxy (see §4).
2. MMFF94 does not model bond-making / bond-breaking; ΔG is computed as the
   difference of two independent single-point energies on fully-formed
   reactant and product geometries (the "reactive landscape" is not sampled).
3. Force-field energies are *not* free energies — entropic contributions,
   solvation, and concentration effects are absent.  The absolute ΔG values
   should be interpreted qualitatively (uphill vs downhill), not
   quantitatively.
4. For organometallic species (Zn, Cu, …) MMFF94 has no parameters and UFF
   is used instead; UFF is less reliable for transition metals.

## 3. Evaluation Metrics

For each candidate reaction `R >> P`:

1. **Reactant energy** `E_R` (kcal/mol): sum of MMFF94 single-point energies
   of the optimised 3D conformers of each disconnected reactant component.
2. **Product energy** `E_P` (kcal/mol): same for product components.
3. **Reaction free energy** `ΔG = E_P − E_R` (kcal/mol).
   - `ΔG > 0` => endothermic (thermodynamically disfavoured).
   - `ΔG < 0` => exothermic (thermodynamically favoured).
4. **Barrier estimate** (force-field proxy):
   `barrier ≈ |ΔG| + 5 kcal/mol`.
   The additive constant (5 kcal/mol) is a conservative kinetic residual
   that approximates the minimal distortion energy required to reach a
   transition-state geometry from the nearest minima.  This is **not** a
   computed TS energy — it is explicitly labelled a *force-field estimate*
   and is only consulted on the endothermic side (see §4).
5. **Product stability** is implicitly encoded by `E_P`: a high `E_P`
   relative to `E_R` (i.e. high ΔG) signals an unstable product.

### 3.1 Conformer generation protocol

For each molecule:
1. Strip atom-map numbers (`SetAtomMapNum(0)`).
2. `Chem.AddHs` (explicit hydrogens).
3. `AllChem.EmbedMolecule(randomSeed=20260719, useRandomCoords=True)`; retry
   with default params if the first attempt returns non-zero.
4. `MMFFOptimizeMolecule(maxIters=200)` (or `UFFOptimizeMolecule` on fallback).
5. `MMFFGetMoleculeForceField(mol, props).CalcEnergy()` for the single-point
   energy on the optimised geometry.

The seed is fixed (20260719) for reproducibility.

## 4. Judgment Rule (Support for "Low Feasibility")

A synthetic negative is **supported** by the computational evidence if either:

- **Thermodynamic**: `ΔG > +5 kcal/mol` (reaction is uphill by more than
  5 kcal/mol — disfavoured at standard conditions), **OR**
- **Kinetic**: `barrier > 25 kcal/mol AND ΔG > 0` (endothermic side only).

The `ΔG > 0` guard on the barrier branch prevents a false-positive where a
strongly exothermic reaction (large `|ΔG|`) inflates the `|ΔG| + 5` barrier
estimate without actually being kinetically hindered.  (For exothermic
reactions, the force-field barrier estimate is not physically meaningful
and is not consulted.)

**Inconclusive**: reactions where both `ΔG` and `barrier` are `None`
(parse / embed failure).

**Support rate** = `n_supported / n_computed`.  The Go/No-Go threshold is
60% (Section 22.1 spec).

## 5. Sampling Rule

From the 26,517 high-confidence synthetic negatives:

1. **Filter** to rows where `candidate_reactants != candidate_product`
   (11,499 rows).  Rationale: rows where reactants ≡ products are
   *degenerate* (no reaction) and give ΔG = 0 by construction, which
   provides no thermodynamic evidence under the §4 rule.
2. **Sort** by `feasibility_score = 1 − hard_score` ascending
   (lowest feasibility first = highest confidence negative first).
3. **Deduplicate** by `(candidate_reactants, candidate_product)`, keeping
   the first occurrence (highest-confidence instance) — many source_ids
   share the same candidate reaction and would otherwise waste the compute
   budget.
4. **Take top 100** as the synthetic-negative sample.
5. **Control (observed positives)**: the `positive_reaction` column of the
   same 100 rows, paired 1-to-1 with the synthetic negatives.  This gives
   a paired comparison: for each row, the synthetic-negative ΔG is compared
   against its parent observed-positive ΔG.

### 5.1 Failure-type composition of the sample

| failure_type                | n in sample | % of sample |
|-----------------------------|-------------|-------------|
| retro_missing_reactant      | 45          | 45%         |
| retro_wrong_functional_group| 40          | 40%         |
| chemoselectivity_error      | 15          | 15%         |

The sample over-represents `retro_missing_reactant` (45% vs 28.7% in the
full 26,517-row file) because that failure type has higher `hard_score`
values on average and thus sorts first under the feasibility-ascending rule.

## 6. Results

### 6.1 Headline numbers

| Metric                              | Value |
|-------------------------------------|-------|
| Synthetic negatives computed        | 100   |
| Supported (ΔG > +5 or endothermic barrier > 25) | 48 |
| Not supported                       | 52    |
| Inconclusive                        | 0     |
| **Overall support rate**            | **48.0%** |
| Go/No-Go threshold                  | 60%   |
| **Go/No-Go verdict**                | **NO_GO_partial_support** |
| Method actual                       | MMFF94 (93) + UFF (7) |
| Degraded from requested             | No (MMFF94 requested and used) |

### 6.2 ΔG distribution (synthetic negatives, kcal/mol)

| Statistic | Value    |
|-----------|----------|
| n_valid   | 100      |
| mean      | −0.37    |
| std       | 29.40    |
| min       | −112.47  |
| median    | +3.97    |
| max       | +78.26   |

The distribution is broad and roughly centred near zero (median +3.97),
indicating that the synthetic negatives span the full range from strongly
exothermic to strongly endothermic.

### 6.3 ΔG distribution (control observed positives, kcal/mol)

| Statistic | Value    |
|-----------|----------|
| n_valid   | 100      |
| mean      | +1.70    |
| std       | 30.03    |
| median    | +8.12    |

The control positives have a *higher* median ΔG (+8.12) than the synthetic
negatives (+3.97), which is the opposite of the expected direction if
synthetic negatives were uniformly less feasible.  This reflects the fact
that many observed reactions are endothermic but made feasible by
catalysts / reagents / conditions that are not captured in a bare
force-field ΔG.

### 6.4 Support rate by failure type

| failure_type                | n  | supported | support_rate | ΔG median |
|-----------------------------|----|-----------|--------------|-----------|
| chemoselectivity_error      | 15 | 10        | **66.7%**    | +9.01     |
| retro_wrong_functional_group| 40 | 22        | 55.0%        | +6.87     |
| retro_missing_reactant      | 45 | 16        | 35.6%        | −0.97     |

**Key finding**: the support rate is strongly failure-type-dependent.

- `chemoselectivity_error` (real chemical substitutions, e.g.
  `replace:Cl->Br`) exceeds the 60% Go threshold — the force field agrees
  that these substitutions are thermodynamically uphill.
- `retro_missing_reactant` (a reactant is *dropped* from the precursors)
  has the *lowest* support (35.6%): dropping a reactant often makes the
  residual "reaction" more exothermic (fewer bonds to break), so ΔG
  drops and the §4 rule does not fire.  This is a known limitation of
  energy-based validation for this failure type — the infeasibility is
  *stoichiometric* (missing reagent), not *thermodynamic*.

### 6.5 Support rate by edit action (top 4)

| edit_action                  | n  | supported | support_rate | ΔG median |
|------------------------------|----|-----------|--------------|-----------|
| drop:last_reactant           | 45 | 16        | 35.6%        | −0.97     |
| reactant[0].replace:Br->Cl   | 32 | 19        | 59.4%        | +8.37     |
| replace:Cl->Br               | 15 | 10        | 66.7%        | +9.01     |
| reactant[0].replace:Cl->Br   |  8 |  3        | 37.5%        | −32.09    |

### 6.6 10-seed paired bootstrap significance test

Paired comparison: synthetic-negative ΔG − control-positive ΔG, for each
of the 100 rows; 10 seeds × bootstrap resample (n=100 with replacement) ×
percentile 95% CI.

| Metric                          | Value |
|---------------------------------|-------|
| n_pairs                         | 100   |
| num_seeds                       | 10    |
| overall_mean_diff (neg − pos)   | −2.07 kcal/mol |
| n_significant_seeds (CI excludes 0) | 0 |
| significance_rate               | 0.0%  |
| interpretation                  | **not_significant** |

The paired test is **not significant**: synthetic-negative ΔG is not
systematically higher than observed-positive ΔG.  If anything, the mean
difference is slightly negative (−2.07), again reflecting the
`retro_missing_reactant` drag.

## 7. Go/No-Go Decision

- **Overall**: NO_GO_partial_support (48% < 60%).
- **Subset `chemoselectivity_error`**: GO (66.7% ≥ 60%) — computational
  validation supports the "low feasibility" judgment for genuine chemical
  substitution errors.
- **Subset `retro_wrong_functional_group`**: borderline (55.0%).
- **Subset `retro_missing_reactant`**: not supported by energy validation
  (35.6%) — but this failure type is *stoichiometric*, not thermodynamic,
  and should be validated by a different signal (e.g. database-retrieval
  of the missing reagent, or atom-balance checking, both already done in
  P1-08 Layer 2).

### 7.1 Interpretation for the manuscript

Per Section 22.1 spec:
- The overall support rate (48%) is below the 60% Go threshold, so the
  manuscript must state **"computational validation partial support"**.
- However, the chemoselectivity subset (66.7%) does pass, so the paper
  can claim that for the *chemical-substitution* family of synthetic
  negatives, the MMFF94 force field independently corroborates the
  low-feasibility judgment.
- The `retro_missing_reactant` family should be flagged as validated by
  *graph-level* signals (atom-balance, database-retrieval) rather than by
  energy.
- Journal target: consider downgrading from the top-tier (Nature
  Chemistry / Science) to a venue that accepts partial computational
  validation (e.g. J. Chem. Inf. Model., Digital Discovery), or
  commission a small DFT subset (20–30 reactions) on the
  `chemoselectivity_error` family to strengthen the claim.

## 8. Limitations & Caveats

1. **Force-field level**: MMFF94 / UFF energies are not free energies.
   Solvation, entropy, concentration, and catalyst effects are absent.
2. **No transition-state modelling**: the "barrier" is `|ΔG| + 5`, a
   crude proxy, not a computed TS energy.  A real DFT barrier calculation
   would be needed for a quantitative kinetic claim.
3. **Single conformer**: only one 3D conformer per molecule is optimised;
   conformational ensembles would give a more robust ΔG.
4. **Organometallic fallback**: 7/100 reactions fell back to UFF, which is
   less reliable for Zn / Cu centres.
5. **Missing-reactant confound**: `retro_missing_reactant` is the largest
   failure type in the sample (45%) and is not amenable to energy
   validation by construction (the "reaction" is stoichiometrically
   incomplete, not thermodynamically disfavoured).
6. **Sampling bias**: the feasibility-ascending sort over-samples
   `retro_missing_reactant` relative to its base rate.  A stratified
   sample by failure type would give a fairer picture.

## 9. Reproducibility

### 9.1 Artefacts

- Code: `chem_negative_sampling/pc_cng/run_xtb_validation.py`
- Tests: `chem_negative_sampling/tests/test_xtb_validation.py`
- Results: `results/xtb_dft_validation_20260719/`
  - `xtb_results.csv` — 100 synthetic-negative energies + ΔG + barrier + verdict
  - `control_positive_results.csv` — 100 observed-positive energies
  - `validation_summary.json` — headline metrics + paired-significance block
  - `paired_significance.json` — 10-seed bootstrap details

### 9.2 Acceptance command

```bash
cd /home/cunyuliu/pc_cng_research
/home/cunyuliu/miniconda3/envs/pc_cng/bin/python -m pc_cng.run_xtb_validation \
    --candidates results/false_negative_three_layer_20260719/high_confidence_negatives.csv \
    --limit 100 \
    --output-dir results/xtb_dft_validation_20260719 \
    --method mmff94 \
    --num-seeds 10
/home/cunyuliu/miniconda3/envs/pc_cng/bin/python -m pytest \
    chem_negative_sampling/tests/test_xtb_validation.py -v
```

### 9.3 Environment

- RDKit 2026.03.3
- pandas, numpy (conda env `pc_cng`, Python 3.10.20)
- Seed: 20260719 (base), 10 bootstrap seeds (20260719 … 20260728)
- Hardware: remote CPU node (no GPU used; respects the GPU-4 exclusion)

## 10. Methodology References

1. Halgren, T. A. *Merck molecular force field. I. Basis, form, scope,
   parameterization, and performance of MMFF94.* J. Comput. Chem. 1996,
   17, 490–519.
2. Rappe, A. K.; Casewit, C. J.; Colwell, K. S.; Goddard, W. A.; Skiff, W. M.
   *UFF, a full periodic table force field for molecular mechanics and
   molecular dynamics simulations.* J. Am. Chem. Soc. 1992, 114, 10024–10035.
3. Hartenfeller, M. et al. *A collection of robust organic synthesis
   reactions for in silico molecule design.* J. Cheminform. 2012, 4, 35.
4. Marcou, G. et al. *Expert system for the estimation of regioselectivity
   in organic reactions.* J. Chem. Inf. Model. 2015, 55, 2312–2322.
5. Bannwarth, C.; Ehlert, S.; Grimme, S. *GFN2-xTB — an extended and
   efficient semi-empirical quantum chemistry method.* J. Chem. Theory
   Comput. 2019, 15, 1652–1671. (not used here — xtb unavailable)
