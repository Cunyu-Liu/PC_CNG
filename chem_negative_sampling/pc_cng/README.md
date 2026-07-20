# PC-CNG MVP

PC-CNG means **PhysChem-Constrained Counterfactual Negative Generator**.

This module is the first executable prototype for the model-design-centered plan:

- generate counterfactual negative reactions from positive reactions;
- separate forward outcome negatives and retrosynthesis precursor negatives;
- attach failure type, edit action, hardness, and false-negative-risk metadata;
- train a tiny dependency-free reranker as a smoke test for downstream utility.

The MVP is intentionally dependency-light. It runs with Python standard library only. If RDKit is installed, `chem_utils.py` automatically uses it for stronger SMILES validation.

## Quick MVP

Run from `code/chem_negative_sampling`:

```bash
python3 -m pc_cng.run_mvp --output-dir results/pc_cng_mvp_demo --epochs 200
```

Outputs:

- `positive_reactions.csv`: demo or input positive reactions.
- `synthetic_counterfactual_negatives.csv`: generated synthetic negatives.
- `ranker_dataset.csv`: positive + negative rows for lightweight reranking.
- `lightweight_ranker.json`: tiny logistic reranker weights.
- `mvp_metrics.json`: generation and ranking smoke-test metrics.

Important: generated rows are `synthetic counterfactual negatives`, not real failed experiments.

## Custom Positive CSV

Input CSV must contain:

- `reaction_smiles`
- optional `source_id` or `id`

```bash
python3 -m pc_cng.run_mvp \
  --input data/my_positive_reactions.csv \
  --output-dir results/my_pc_cng_mvp \
  --epochs 300
```

## Streaming Scale Generation

For larger positive-reaction files:

```bash
python3 -m pc_cng.run_scale_generation \
  --input data/uspto_positives.csv \
  --output results/uspto_pc_cng_negatives.csv \
  --summary results/uspto_pc_cng_summary.json \
  --limit 100000
```

Remove `--limit` for a full run once data quality and runtime are confirmed.

## Data Normalization

Normalize public CSVs into the PC-CNG schema:

```bash
python3 -m pc_cng.data_ingestion \
  --input data/raw/uspto.csv \
  --output data/processed/uspto_normalized.csv \
  --summary data/summaries/uspto_summary.json \
  --source-name uspto
```

Or use the script wrapper:

```bash
export USPTO_CSV=/path/to/uspto.csv
export REGIOSQM_CSV=/path/to/regiosqm.csv
export HITEA_CSV=/path/to/hitea.csv
export ORD_CSV=/path/to/ord.csv
bash scripts_prepare_public_data.sh
```

## Baseline Matrix

Generate baseline negatives:

```bash
python3 -m pc_cng.baselines \
  --input results/pc_cng_mvp_demo/positive_reactions.csv \
  --output results/baseline_smoke/baselines.csv \
  --summary results/baseline_smoke/summary.json
```

Run baseline reranking comparison:

```bash
python3 -m pc_cng.run_experiment_matrix \
  --input results/pc_cng_mvp_demo/positive_reactions.csv \
  --output-dir results/experiment_matrix_smoke \
  --epochs 100
```

Available baselines:

- random product mismatch
- template product perturbation
- DORA-style alternate center
- PU-style reliable negative
- PC-CNG rule MVP

## False-Negative Review

```bash
python3 -m pc_cng.false_negative_review \
  --input results/pc_cng_mvp_demo/synthetic_counterfactual_negatives.csv \
  --output results/fn_review_smoke/reviewed.csv \
  --summary results/fn_review_smoke/summary.json \
  --known-positive results/pc_cng_mvp_demo/positive_reactions.csv
```

Rows marked `needs_review_or_downweight` should not be used as strong
negatives. Rows marked `discard_known_positive` must be removed.

## First Trainable Decoder Scaffold

After PyTorch is installed:

```bash
python3 -m pc_cng.train_graph_edit_decoder \
  --input results/pc_cng_mvp_demo/synthetic_counterfactual_negatives.csv \
  --output-dir results/train_decoder_smoke \
  --epochs 50
```

This is an MLP training-contract scaffold, not the final graph neural decoder.
The final model should consume atom-mapped graph edits from
`atom_mapped_graph_edit.py`.

## Current MVP Scope

The current generator creates two negative families:

- `forward_outcome`: same precursors, counterfactual wrong/no-reaction products.
- `retro_precursor`: same target product, counterfactual wrong precursor sets.

Failure types include:

- `no_reaction`
- `chemoselectivity_error`
- `side_product`
- `retro_no_disconnection`
- `retro_missing_reactant`
- `retro_wrong_functional_group`

## Next Implementation Layer

The publishable model should replace rule edits with learned modules:

1. RDKit/RXNMapper-backed atom mapping and reaction-center extraction.
2. Graph edit vocabulary from positive reactions.
3. Failure-type latent prototypes calibrated by real negatives.
4. Graph edit decoder trained on positive edits and counterfactual preferences.
5. DPO/IPO preference optimization using validator-ranked candidates.
6. Strong downstream validation on retrosynthesis reranking and reaction feasibility.
