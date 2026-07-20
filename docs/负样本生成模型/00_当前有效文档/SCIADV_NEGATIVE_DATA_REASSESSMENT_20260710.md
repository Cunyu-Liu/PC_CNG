# Science Advances Negative-Data Reassessment

## Paper checkpoint

Paper: Toniato, Vaucher, Laino, Graziani, "Negative chemical data boosts language models in reaction outcome prediction", Science Advances 2025, DOI: 10.1126/sciadv.adt5578.

The paper does not claim that arbitrary generated negatives improve retrosynthesis/reaction prediction. Its core claim is narrower and more important:

1. Negative experimental data help most in low-positive-data regimes.
2. The most informative negatives are close to the boundary of successful chemistry.
3. Type-1 negatives are especially valuable: the reaction gives an unexpected but chemically meaningful product.
4. Type-2 negatives are no-reaction/low-yield/starting-material cases, but they are more ambiguous.
5. Randomly pairing reactants with unrelated products gives limited insight and does not clarify the predictive boundary.
6. Their learning mechanism is reward/RL-style tuning of a reaction language model, not simply adding negatives as ordinary BCE labels.

## What they used as negatives

### RegioSQM20

This is the controlled setting. Positives and negatives are well characterized within a narrow organic chemistry domain. The negatives are informative alternative outcomes, especially regio/chemoselective alternatives: products that are chemically meaningful but not the successful observed outcome.

This matches type-1 negative data: same or closely related substrate context, chemically plausible alternative product, near the success boundary.

### HiTEA

This is the real HTE setting. It contains reactions with a range of observed yields, including low/failed outcomes. These are closer to type-2 negatives: intended product not observed or low yield under experimental conditions. The paper explicitly notes that such cases are more ambiguous than type-1 negatives.

## Diagnosis of our PC-CNG negatives

Our current generator produced many samples, but most are not paper-aligned high-value negatives.

Observed counts from generated/reviewed data:

- RegioSQM20 + HiTEA synthetic reviewed rows: 64,646.
- Paper-aligned type-1 filter kept only 1,157 rows.
- Pairwise reward training could use only 1,010 positive/counterfactual pairs.

Reasons:

1. Too many retro artifacts.
   `retro_no_disconnection`, `retro_missing_reactant`, and `retro_wrong_functional_group` keep the target product fixed and perturb precursors. These are not the paper's type-1 forward negative outcomes.

2. Too many no-reaction proxies.
   `product:=reactants` resembles type-2 negatives but is crude and often creates an obvious artifact rather than a measured failed experiment.

3. Side-product generation is too naive.
   `append:O` is not a chemically grounded unexpected product.

4. Chemoselectivity edits are string replacements.
   Br/Cl and N/O replacements can create valid SMILES, but they do not necessarily represent a plausible competing reaction pathway or reaction-center alternative.

5. The training objective was misaligned.
   We initially used synthetic negatives as BCE negative labels. The paper instead uses negative data through a reward model/RL feedback mechanism.

## Experimental confirmation

### Direct synthetic-negative mixing

Directly adding PC-CNG negatives hurt validation ROC-AUC:

- Best real-only validation ROC-AUC: 0.8654.
- Direct PC-CNG validation ROC-AUC: 0.8192.
- Weighted PC-CNG validation ROC-AUC: 0.8333.

### Paper-aligned type-1 filtering

Filtering to type-1 forward-outcome alternatives did not recover performance:

- `paper_aligned_type1_h2048_n4096_e80`: val ROC-AUC 0.8599, test ROC-AUC 0.8486.
- `paper_aligned_type1_h4096_n2048_e80`: val ROC-AUC 0.8546, test ROC-AUC 0.8561.

### Pairwise reward training

Pairwise reward training was closer to the paper's idea, but still did not beat real-only:

- `pairwise_reward_type1_h2048_n4096_e80`: val ROC-AUC 0.8652, test ROC-AUC 0.8571.

This means the remaining bottleneck is not just the loss function. The synthetic type-1 negatives themselves are too weak/noisy.

## Architecture-level fix

The next architecture should replace rule/string edits with a learned reaction-boundary generator.

### PC-CNG v2: Reaction-Boundary Generator

1. Reaction-center encoder
   - Use atom-mapped reactions.
   - Encode reactant graph, product graph, and reaction-center bond edits.
   - Predict formed, broken, and changed bonds.

2. Type-1 counterfactual product decoder
   - Generate alternative products by moving or changing the reaction center within chemically valid local environments.
   - Target regioisomer, chemoselectivity, and competing functional-group outcomes.
   - Do not generate arbitrary product string substitutions.

3. Type-2 no-reaction/low-yield module
   - Use real HTE low-yield data as supervision.
   - Treat type-2 synthetic no-reaction cases as low-confidence auxiliary signals, not hard negatives.

4. Reward model, not BCE-only classifier
   - Train a reward model where observed positive outcome > type-1 counterfactual outcome > random/artifact.
   - Use pairwise/DPO-style loss for generated negatives.
   - Use BCE only as an anchor on real positive/real negative labels.

5. Generator quality filter
   - Keep only candidates with plausible reaction-center locality.
   - Enforce RDKit validity, atom/bond edit locality, product novelty, known-positive retrieval exclusion, and moderate product similarity.

6. USPTO usage
   - Use USPTO positive-only data for reaction encoder pretraining, not direct supervised mixing.
   - Fine-tune on RegioSQM20/HiTEA after pretraining.

## Immediate next experiment

The next meaningful experiment is not another MLP weighting run. It is:

1. Build RXNMapper-based mapped reaction cache.
2. Extract reaction-center edits for positives and real negatives.
3. Train an edit-conditioned graph encoder/reward model.
4. Generate type-1 negatives by learned local reaction-center perturbation.
5. Evaluate whether generated type-1 negatives improve real validation ROC-AUC/AUPRC and retrosynthesis reranking.

If this does not improve over the current stacked ensemble, then the negative-generation hypothesis is not yet supported.
