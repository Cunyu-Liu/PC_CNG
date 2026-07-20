# Failure Prototype Calibrator - Evaluation Report

- Overall classification accuracy: **0.9500**
- Random baseline accuracy: 0.0500
- Random baseline target hit rate: 0.1000
- Mean per-sample entropy: 0.6111
- Normalized entropy (0=confident, 1=uniform): 0.2654
- Aggregate predicted-class entropy: 0.6881
- Uniform reference entropy (ln 10): 2.3026

## Per-class accuracy

| Failure type | Count | Accuracy | Target hit rate |
|---|---:|---:|---:|
| wrong_anchor | 100 | 0.9167 | 0.5750 |
| broken_atom_balance | 100 | 1.0000 | 0.5250 |
| invalid_valence | 0 | n/a | n/a (too few) |
| fragment_misalignment | 0 | n/a | n/a (too few) |
| wrong_bond_type | 0 | n/a | n/a (too few) |
| aromaticity_violation | 0 | n/a | n/a (too few) |
| stereochemistry_loss | 0 | n/a | n/a (too few) |
| over_reaction | 0 | n/a | n/a (too few) |
| under_reaction | 0 | n/a | n/a (too few) |
| side_product | 0 | n/a | n/a (too few) |

## Go/No-Go assessment (P1-06)

- Accuracy >= 0.70: yes
- Target hit rate >= 0.50 per class: yes (mean=0.5500)
- Mean entropy > 0.230: yes

**Verdict: PASS (eligible for paper Section 6.3)**

> Note: this is a single-seed smoke evaluation. The paper-level
> claim requires a 10-seed paired significance test against the
> random baseline.