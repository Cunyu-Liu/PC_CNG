# DFT Validation Protocol — P2-02 chemoselectivity_error subset

**Task ID**: P2-02
**Date**: 2026-07-20
**Owner**: pc-cng research
**Status**: Active (supersedes the MMFF94-only path in `computational_validation_protocol_20260719.md` for the `chemoselectivity_error` failure type)

## 1. Purpose

P1-10 used MMFF94 (force field) to validate synthetic-negative candidates and
obtained a support rate of **0.48** (48/100), below the 0.60 GO threshold
(see `results/xtb_dft_validation_20260719/validation_summary.json`). The
partial support was driven by MMFF94's known weaknesses on heteroaromatic
and halogen-rich chemistry — exactly the chemistries that dominate the
`chemoselectivity_error` failure type (1352/26517 ≈ 5.1% of the candidate
pool).

P2-02 re-runs validation on a **20–30 candidate subset** of the
`chemoselectivity_error` rows using a higher-fidelity method (xTB at minimum)
to either (a) confirm the partial-support verdict or (b) recover a GO verdict
that MMFF94 had masked.

## 2. Software stack

| Layer | Software | Version | Environment |
|-------|----------|---------|-------------|
| Driver | `pc_cng.run_dft_validation` | P2-02 | `pc_cng_gpu` conda env (has `rdkit`, `pandas`) |
| Energy worker (xTB) | `xtb-python` (ASE calculator) | 20.2 | `dft` conda env (CPU-only) |
| Optimizer | `ase.optimize.BFGS` | ase 3.29.0 | `dft` |
| SMILES handling | `rdkit` | 2025.03.6 | `pc_cng_gpu` |
| Fallback | `rdkit.Chem.rdForceFieldHelpers.MMFFOptimizeMolecule` | 2025.03.6 | `pc_cng_gpu` |
| Stub | ORCA 4.x | not wired | n/a |

**Bridge mechanism**: the driver process (in `pc_cng_gpu`) parses SMILES,
embeds 3D coordinates with RDKit (ETKDG + MMFF94 pre-optimization), and writes
the geometry as an XYZ block. A separate worker script (`dft_xtb_worker.py`,
written to a temp dir by the driver) is invoked once per batch via
`subprocess.run([dft_python, worker, in.json, out.json])` and reads the XYZ
blocks, runs xTB in the `dft` env, and writes the energies back as JSON.

**CPU-only constraint**: the driver exports `CUDA_VISIBLE_DEVICES=""` at the
start of `main()`, and the worker's subprocess env inherits the same setting
so xTB never accidentally reserves GPU 4 (xTB is CPU-only by design).

## 3. Method priority

```
ORCA 4.x (DFT) > xTB 6.5+ (semi-empirical) > MMFF94 (force field)
```

1. **ORCA 4.x** — full DFT, highest fidelity. **Not implemented** in this
   iteration; the `--method orca` CLI flag exists as a dispatch stub and
   returns `status="not_implemented"`. Wiring ORCA in is left as future
   work because (a) it requires an ORCA binary on the compute host and
   (b) it is ~3 orders of magnitude slower than xTB for the same candidate
   set.

2. **xTB 6.5+ (GFN2-xTB)** — semi-empirical, the **default** for P2-02.
   Good compromise between accuracy and cost (~5–30 s per molecule for
   typical drug-like chemistries). Uses ALPB implicit solvation in water
   by default.

3. **MMFF94** — force field fallback. Used when (a) the user explicitly
   selects `--method mmff94`, or (b) `--method xtb` is selected but the
   xTB batch returns 0/ok results (wholesale failure). In the per-molecule
   fallback path, the original `compute_molecule_energy_mmff94` logic is
   reused with UFF as a secondary fallback when MMFF parameters are missing.

## 4. Computational details

### 4.1 xTB / GFN2-xTB

| Parameter | Value | Notes |
|-----------|-------|-------|
| Hamiltonian | GFN2-xTB | default; switch with `--xtb-method` |
| Solvation | ALPB (water) | implicit; disable with `--xtb-solvent ""` |
| Optimization | BFGS (ASE) | `fmax=0.05 eV/Å`, `max_steps=50` |
| Atomic coordinates | RDKit ETKDG + MMFF94 pre-opt | gives xTB a good starting geometry |
| Thermochemistry | none | P2-02 uses single-point + optimized geometry; vibrational corrections deferred |
| Parallelism | `OMP_NUM_THREADS=4` (env override) | CPU-only |
| Seed | `20260720` (RDKit embedding) | makes 3D geometries reproducible |

### 4.2 MMFF94 (fallback)

| Parameter | Value |
|-----------|-------|
| Force field | MMFF94 (RDKit), UFF as secondary fallback |
| Optimization | `MMFFOptimizeMolecule(maxIters=200)` |
| Seed | `20260720` (ETKDG embedding) |

### 4.3 ORCA (stub)

Not configured. When implemented, the planned settings are:
B3LYP-D3(BJ) / def2-SVP / def2/JK auxiliary basis / COSMO (water) /
` TightSCF` / `Grid4`. Grid size: m4. The XYZ-to-ORCA input writer would
live in `pc_cng.orca_worker` (new module).

## 5. ΔG calculation

For each candidate reaction (single `candidate_reactants` →
`candidate_product` row):

1. **Split** multi-component SMILES on `.`:
   - `reactant_components = candidate_reactants.split(".")`
   - `product_components = candidate_product.split(".")`

2. **Strip atom maps** from every component (RDKit `SetAtomMapNum(0)` +
   `MolToSmiles` canonicalization).

3. **Embed + optimize** each unique component once (results are cached
   across all candidates to avoid recomputing the same molecule).

4. **Sum energies**:
   ```
   E_reactants = Σ E(component_i)        for i in reactant_components
   E_products  = Σ E(component_j)        for j in product_components
   ```

5. **Reaction free energy**:
   ```
   ΔG_reaction = E_products − E_reactants          (kcal/mol)
   ```
   The conversion factor for xTB is `1 Hartree = 627.509474 kcal/mol`.

6. **Barrier approximation** (rough, force-field-derived heuristic kept
   for continuity with P1-10):
   ```
   barrier_estimate = |ΔG_reaction| + 5 kcal/mol
   ```
   Note: this is **not** a transition-state energy. A real TS estimate
   would require a QST2/QST3 search which is out of scope for P2-02.
   The barrier is logged but **not** used by the P2-02 support rule.

## 6. Support criterion (P2-02)

```
ΔG_reaction > 0 kcal/mol  →  supported
ΔG_reaction ≤ 0 kcal/mol  →  not_supported
ΔG_reaction == None        →  inconclusive
```

**Rationale**: a positive ΔG means the candidate reaction is thermodynamically
uphill, so the proposed `chemoselectivity_error` is a true negative — the
edit (e.g. `replace:Cl->Br`) does not produce a chemically favorable
transformation. This is a simpler and stricter rule than P1-10's
`ΔG > +5 kcal/mol` threshold because xTB energies are reliable enough that
a positive sign is already meaningful signal.

The P1-10 barrier-based fallback (`barrier > 25 kcal/mol AND ΔG > 0`) is
**intentionally disabled** in P2-02 to keep the rule purely thermodynamic
and avoid double-counting from the rough `|ΔG|+5` barrier formula.

## 7. Go / No-Go threshold

```
support_rate ≥ 0.60  →  GO
support_rate <  0.60  →  NO_GO_partial_support
```

Identical to P1-10. If P2-02 returns GO, the `chemoselectivity_error` subset
is treated as validated. If it returns NO_GO again, the chemoselectivity_error
edit family is flagged as under-supported and routed back to expert review
(`docs/expert_review_protocol_20260719.md`).

## 8. Inputs and outputs

### Inputs

| File | Source | Notes |
|------|--------|-------|
| `results/false_negative_three_layer_20260719/high_confidence_negatives.csv` | P1-08 | 26517 rows total, 1352 with `failure_type=chemoselectivity_error` |

### Outputs (in `--output-dir`)

| Path | Contents |
|------|----------|
| `per_candidate_results.csv` | One row per candidate with columns: `source_id, failure_type, edit_action, task, hard_score, false_negative_risk, candidate_reactants, candidate_product, dG_reactants, dG_products, dG_reaction, barrier_estimate, method, status, supports_negative, support_verdict, support_reason` |
| `dft_validation_summary.json` | `{task, method_requested, method_actual, degraded_from_requested, candidates_path, failure_type, total_candidates_loaded, n_after_failure_type_filter, n_computed, support_rule, n_supported, n_not_supported, n_inconclusive, support_rate, mean_dg, std_dg, min_dg, max_dg, median_dg, go_no_go_threshold, go_no_go_verdict, xtb_method, xtb_solvent, seed, num_seeds_note, timestamp, notes}` |
| `detailed_logs/xtb_worker.log` | xTB worker stdout/stderr (one combined log per batch) |
| `detailed_logs/<source_id>.log` | Per-candidate log with SMILES, status, energies, verdict |

## 9. CLI

```bash
cd /home/cunyuliu/pc_cng_research/chem_negative_sampling
PYTHONPATH=. /home/cunyuliu/miniconda3/envs/pc_cng_gpu/bin/python -m pc_cng.run_dft_validation \
    --candidates /home/cunyuliu/pc_cng_research/results/false_negative_three_layer_20260719/high_confidence_negatives.csv \
    --failure-type chemoselectivity_error \
    --limit 30 \
    --method xtb \
    --output-dir /home/cunyuliu/pc_cng_research/results/dft_validation_chemoselectivity_20260720 \
    --dft-python /home/cunyuliu/miniconda3/envs/dft/bin/python \
    --xtb-method GFN2-xTB \
    --xtb-solvent water \
    --seed 20260720
```

### CLI flags

| Flag | Default | Purpose |
|------|---------|---------|
| `--candidates PATH` | `…/false_negative_three_layer_20260719/high_confidence_negatives.csv` | Input candidates CSV |
| `--failure-type STR` | `chemoselectivity_error` | Filter rows by `failure_type` column |
| `--limit INT` | `30` | Max candidates to compute (after sampling) |
| `--method {xtb,mmff94,orca}` | `xtb` | Computational method |
| `--output-dir PATH` | required | Output directory (created if missing) |
| `--dft-python PATH` | `/home/cunyuliu/miniconda3/envs/dft/bin/python` | DFT conda env python (used only for `--method xtb`) |
| `--xtb-method STR` | `GFN2-xTB` | xTB Hamiltonian |
| `--xtb-solvent STR` | `water` | ALPB solvent; empty string disables solvation |
| `--no-require-chemical-change` | off | If set, keep rows where `reactants == product` |
| `--seed INT` | `20260720` | RDKit embedding seed |

## 10. Sampling strategy

1. **Filter** to `failure_type == "chemoselectivity_error"` (1352 rows).
2. **Drop** rows where `candidate_reactants == candidate_product` (no
   chemical change; ΔG would be trivially 0).
3. **Sort** by `hard_score` descending (most confident negative first).
4. **Deduplicate** by `(candidate_reactants, candidate_product)` keeping
   the highest-confidence occurrence.
5. **Take top `--limit` rows** (default 30).

## 11. 10-seed paired bootstrap

**Not required for P2-02**. Unlike P1-10 which compared synthetic negatives
against paired control positives (the observed `positive_reaction` column)
with a 10-seed bootstrap, P2-02 is a focused validation of the
chemoselectivity_error subset only. xTB energies are deterministic given
the same input geometry and seed, so a single-seed computation is sufficient
for the GO/NO_GO decision.

If a downstream task re-introduces the paired comparison (e.g.
`chemoselectivity_error` negatives vs `forward_outcome` positives), the
existing `run_paired_significance_test` from `pc_cng.run_xtb_validation` can
be reused with xTB energies as input.

## 12. Degradation path (per Section 26.1)

The runner degrades gracefully:

1. **xTB via ASE preferred** (default `--method xtb`).
2. If the xTB subprocess fails wholesale (0/ok results across all unique
   SMILES in the batch) — typically because `xtb-python` cannot import or
   the ASE calculator cannot be attached — the runner logs a note and
   **per-molecule degrades to MMFF94** (`compute_molecule_energy_mmff94`).
3. If MMFF94 also fails for a specific molecule (missing parameters), the
   existing UFF fallback inside `compute_molecule_energy_mmff94` takes over.
4. If UFF also fails (rare), the molecule is logged as `status="uff_failed"`
   and contributes an `inconclusive` verdict to the summary.
5. If the runner cannot even load the candidates CSV or the `failure_type`
   column is missing, it exits with non-zero code (2 or 3) and writes no
   output files.

### Direct xTB binary fallback (not auto-invoked)

If the ASE-based worker fails but a standalone `xtb` binary is available on
`$PATH`, an alternative invocation would be:

```bash
xtb coords.xyz --opt --gfn2 --alpb water
```

This path is documented for future use; the current runner does not auto-
invoke the binary because (a) it requires writing one XYZ file per molecule
to disk and (b) parsing the energy out of the xTB stdout is brittle. If
auto-degradation to the binary path is needed, it should be added to
`run_xtb_batch` as a second-tier fallback before MMFF94.

### Computational validation deferred

If both xTB and MMFF94 fail for a candidate, the candidate is marked
`inconclusive` in the summary and flagged in the notes. The P2-02 task
then defers computational validation for that candidate to expert review.

## 13. Limitations

1. **No vibrational corrections**: P2-02 uses optimized single-point
   energies, not true Gibbs free energies. Entropic contributions (~1–5
   kcal/mol per component) are ignored. For reactions with a small
   molecule-count change this is a minor error; for reactions with large
   atom-count changes the error can be larger.

2. **No transition-state search**: `barrier_estimate = |ΔG| + 5 kcal/mol`
   is a rough heuristic. Real activation barriers require QST2/QST3 or
   NEB. The barrier value is logged but **not used** by the P2-02 support
   rule.

3. **ALPB implicit solvent**: water is the default. For reactions that
   occur in non-aqueous media (organic solvents, gas phase), the
   `--xtb-solvent` flag should be set accordingly or disabled.

4. **Geometry seed dependence**: xTB is local-optimization-based, so
   different starting geometries can converge to different conformers.
   We mitigate this by MMFF94 pre-optimization in `embed_smiles_to_xyz`
   before passing coordinates to xTB. The seed is fixed (`20260720`)
   for reproducibility.

5. **No conformer search**: only one conformer is sampled per molecule.
   For flexible molecules (≥5 rotatable bonds) this may miss the global
   minimum. A future iteration could use RDKit's conformer enumeration
   (`AllChem.EnumerateStereoisomers` + `EmbedMultipleConfs`) and pick the
   lowest-energy conformer.

6. **No GPU usage**: xTB is CPU-only; the runner explicitly clears
   `CUDA_VISIBLE_DEVICES` to avoid reserving GPU 4.

## 14. Reproducibility checklist

- [x] Random seed fixed (`--seed 20260720`)
- [x] xTB method pinned (`--xtb-method GFN2-xTB`)
- [x] Solvent model pinned (`--xtb-solvent water`)
- [x] MMFF94 / UFF fallback deterministic (same RDKit version)
- [x] `xtb-python` version pinned to `20.2` (conda env `dft`)
- [x] `rdkit` version pinned to `2025.03.6` (conda env `pc_cng_gpu`)
- [x] Output dir contains both the per-candidate CSV and the summary JSON
- [x] Per-candidate log files preserve the inputs and verdict for audit

## 15. Relationship to existing docs

- **Does NOT modify** `docs/00_当前有效文档/*.md` (constraint).
- **Supersedes** the MMFF94-only path in
  `docs/computational_validation_protocol_20260719.md` for the
  `chemoselectivity_error` failure type only. Other failure types
  (`no_reaction`, `retro_missing_reactant`, `retro_no_disconnection`,
  `retro_wrong_functional_group`, `side_product`) continue to use the
  P1-10 MMFF94 protocol.
- **References** `docs/expert_review_protocol_20260719.md` for the
  degradation path when computational validation fails.

## 16. Open issues / future work

1. Wire up ORCA 4.x DFT path (`--method orca`) — currently a stub.
2. Add a direct-`xtb`-binary fallback tier between ASE xTB and MMFF94.
3. Add a conformer-search pre-pass for molecules with ≥5 rotatable bonds.
4. Add vibrational + entropic corrections for true Gibbs free energies.
5. If P2-02 returns GO, port the xTB path back to P1-10 for the other
   failure types and re-evaluate the original 100-candidate support rate.
