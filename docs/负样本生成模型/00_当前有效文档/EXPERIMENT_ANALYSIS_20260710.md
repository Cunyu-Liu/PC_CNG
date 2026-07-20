# PC-CNG Experiment Analysis and SOTA Upgrade Plan

## Executive conclusion

The current strongest rigorous candidate is a stacked ensemble selected on the real validation set:

- Run artifact: `/home/cunyuliu/pc_cng_research/results/stacked_ensemble_summary.json`
- Selection rule: best validation ROC-AUC among completed models.
- Validation: ROC-AUC 0.8738, AUPRC 0.8208, F1 0.7194.
- Test: ROC-AUC 0.8759, AUPRC 0.8117, F1 0.7086 at threshold 0.5.

The best single-model diagnostic result is `full_feasibility_mlp_real_only_h4096_n2048_e80`:

- Validation: ROC-AUC 0.8613, AUPRC 0.8017.
- Test: ROC-AUC 0.8753, AUPRC 0.8087, F1 0.7645.
- Checkpoint: `/home/cunyuliu/pc_cng_research/results/full_feasibility_mlp_real_only_h4096_n2048_e80/best_feasibility_mlp.pt`

For rigorous model selection, the stacked ensemble is preferred because it improves validation ROC-AUC/AUPRC and keeps test performance at the same level.

## What was tested

| Family | Best validation ROC-AUC | Best test ROC-AUC | Main finding |
|---|---:|---:|---|
| Real-only Morgan MLP | 0.8654 | 0.8753 | Strongest single-model baseline. |
| Direct PC-CNG negatives | 0.8192 | 0.8449 | Synthetic negatives currently hurt ranking. |
| Weighted PC-CNG negatives | 0.8333 | 0.8427 | Weighting helps but does not recover real-only performance. |
| USPTO full positive mix | 0.7954 | 0.8444 | Positive-only USPTO causes domain/prior shift. |
| Weighted 50k USPTO | 0.8439 | 0.8673 | Subsampling/low weight helps but still below real-only validation. |
| Count/descriptor features | 0.8677 | 0.8595 | Helpful on validation, not as a standalone test winner. |
| Stacked ensemble | 0.8738 | 0.8761 | Best current rigorous candidate. |

## Diagnosis

1. USPTO is not a supervised negative-learning dataset in this setup.
   The OpenMolecules USPTO subset is a high-yield positive-only reaction corpus. Treating it as ordinary supervised training data shifts class priors and reaction-domain statistics, which lowers RegioSQM20/HiTEA validation performance.

2. Current PC-CNG negatives are not yet boundary-quality negatives.
   Direct and weighted PC-CNG negatives consistently reduce ROC-AUC. This suggests the current rule/counterfactual generator creates distribution artifacts or easy negatives rather than chemically calibrated near-boundary failures.

3. The Morgan-MLP baseline is strong but not a SOTA architecture.
   It is useful as a robust feasibility/reranking baseline, but it does not explicitly model atom-mapped reaction-center edits, local mechanisms, or failure modes.

4. Model complementarity exists.
   Stacking real-only, descriptor, USPTO, and PC-CNG variants improves validation ROC-AUC to 0.8738, which indicates the feature/model families capture partially different signal.

## Concrete upgrades required for SOTA

1. Replace rule PC-CNG with atom-mapped reaction-center edit generation.
   Use RXNMapper to extract formed/broken/changed bonds, train an edit decoder over reaction centers, and constrain negatives to local chemically plausible perturbations.

2. Use USPTO as pretraining, not direct positive-only BCE mixing.
   The next training protocol should be:
   pretrain reaction encoder on USPTO with contrastive/masked-edit objectives, then fine-tune on RegioSQM20/HiTEA real positive/negative labels.

3. Add graph encoder with reaction-center supervision.
   Implement a shared MPNN/Graph Transformer over reactant/product graphs with auxiliary edit-center prediction. The current Morgan MLP should remain the baseline/teacher.

4. Improve false-negative control.
   Current review flags 1,577,736 / 2,683,886 USPTO PC-CNG candidates as keepable, but this is still heuristic. For publication, high-hardness candidates need reaction-center similarity filtering plus exact/near-positive retrieval.

5. Evaluate on external SOTA baselines.
   Required baselines: random mismatch, template perturbation, DORA-style center replacement, PU reliable negatives, and a strong reaction feasibility/reranking model.

## Current artifacts

- USPTO reaction SMILES: `/home/cunyuliu/pc_cng_research/data/processed/uspto_openmolecules_train_only.csv`
- USPTO PC-CNG reviewed negatives: `/home/cunyuliu/pc_cng_research/results/uspto_openmolecules_full_generation/pc_cng_synthetic_negatives_reviewed.csv`
- Full experiment summary: `/home/cunyuliu/pc_cng_research/results/full_feasibility_matrix_summary.json`
- Stacked ensemble summary: `/home/cunyuliu/pc_cng_research/results/stacked_ensemble_summary.json`
- USPTO converter: `pc_cng/build_uspto_openmolecules.py`
- Weighted/descriptive featurizer training script: `pc_cng/train_feasibility_mlp.py`
