# Ni Coupling Data Gap Research (2026-07-20)

Task: PC-CNG P1-11.  This report documents the public-data landscape for Ni-catalyzed cross-coupling reactions and decides whether the PC-CNG benchmark should be supplemented with external Ni coupling reactions.

## TL;DR

- **Go**: 1688 Ni coupling reactions identified (>= 50 threshold).
- Existing PC-CNG datasets contribute 23 Ni reactions.
- Public NiCOlit dataset contributes 1665 Ni reactions.
- Supplement CSV written to `data/processed/ni_coupling_supplement.csv`.

- RDKit available for atomic-number validation: `True`.


## 1. Ni Coupling Reaction Types

Ni-catalyzed cross-coupling is a family of C-C / C-N bond-forming reactions that use a nickel catalyst (often Ni(0)/Ni(II) with a phosphine or N,N-bidentate ligand).  Canonical variants covered by this audit:

| Variant | Nucleophile | Electrophile | Bond formed |
| --- | --- | --- | --- |
| Suzuki-Miyaura | organoboron (B) | aryl/vinyl halide | C-C |
| Negishi | organozinc (Zn) | aryl/vinyl halide | C-C |
| Kumada | organomagnesium (Mg) | aryl/vinyl halide | C-C |
| Hiyama | organosilicon (Si) | aryl/vinyl halide | C-C |
| Murahashi | organolithium (Li) | aryl/vinyl halide | C-C |
| Reductive cross-coupling | two electrophiles + reductant (Mn/Zn) | aryl/vinyl halide | C-C |
| Buchwald-Hartwig amination | amine (N-H) | aryl halide | C-N |

Ni is increasingly attractive vs. Pd because of its low cost and ability to activate aryl chlorides and ethers, but its public data footprint remains far smaller than Pd.


## 2. Public Data Source Survey

### 2.1 Existing PC-CNG benchmark datasets

| Dataset | Total rows | Ni reactions | Ni fraction |
| --- | --- | --- | --- |
| uspto_openmolecules | 530238 | 6 | 0.0011% |
| ord_open_reaction_database | 2910 | 17 | 0.5842% |
| hitea_full | 39546 | 0 | 0.0000% |


### 2.2 Reaction type distribution within existing datasets

| Dataset | Reaction type | Count |
| --- | --- | --- |
| uspto_openmolecules | Other Ni-catalyzed | 4 |
| uspto_openmolecules | Reductive cross-coupling | 2 |
| ord_open_reaction_database | Kumada | 1 |
| ord_open_reaction_database | Murahashi | 1 |
| ord_open_reaction_database | Other Ni-catalyzed | 13 |
| ord_open_reaction_database | Reductive cross-coupling | 1 |
| ord_open_reaction_database | Suzuki | 1 |


### 2.3 Public NiCOlit dataset

- Reference: Schleinitz, J.; Langevin, M.; Smail, Y.; Wehnert, B.; Grimaud, L.; Vuilleumier, R. J. Am. Chem. Soc. 2022, 144, 14722-14730. DOI: 10.1021/jacs.2c05302
- Source URL: https://raw.githubusercontent.com/truejulosdu13/NiCOlit/master/data/NiCOlit.csv
- License: CC-BY-NC-ND (per NiCOlit manuscript).
- Scope: literature-mined Ni-catalyzed C-O / C-C / C-N couplings from primary research articles and review articles, including both scope tables and optimisation tables (failed experiments are represented as low-yield rows).


**NiCOlit ingestion stats**

| Metric | Value |
| --- | --- |
| Rows loaded | 1665 |
| Skipped (missing substrate/product) | 0 |
| Skipped (invalid) | 0 |
| Skipped (duplicate) | 338 |


**NiCOlit Mechanism distribution**

| Mechanism (NiCOlit) | Rows |
| --- | --- |
| Suzuki | 483 |
| Kumada | 314 |
| C-H activation | 274 |
| Review | 145 |
| CO2 Insertion | 72 |
| Hiyama | 62 |
| Negishi | 60 |
| Ni/Cu cooperation | 55 |
| Isocyanates | 47 |
| Murahashi | 46 |
| Al _coupling | 44 |
| P_coupling | 37 |
| Buchwald | 26 |


**NiCOlit coupling_partner_class distribution**

| Coupling partner class | Rows |
| --- | --- |
| B | 545 |
| RMgX | 314 |
| C-H | 274 |
| tbd | 145 |
| CO2 | 72 |
| Zn | 60 |
| Si | 55 |
| NCO | 47 |
| Li | 46 |
| Al | 44 |
| P | 37 |
| NH | 26 |


**NiCOlit top catalyst precursors**

| Catalyst precursor SMILES | Rows |
| --- | --- |
| Ni(cod)2 | 826 |
| NiCl2(PCy3)2 | 340 |
| NiCl2(dppf) | 93 |
| Ni(acac)2 | 55 |
| CCCCN4c1ccccc1N5c6cccc7N3c2ccccc2N(CCCC)C3[Ni](Br)(C45)[n+]67.[Br-] | 42 |
| NiCl2(glyme) | 31 |
| CC(C)[P+](C(C)C)(C(Nc1ccccc1n3nc(c2ccccc2)cc3c4ccccc4)c5ccccc5)[Ni](Cl)(Cl)[P+](C(C)C)(C(C)C)C(Nc6ccccc6n8nc(c7ccccc7)cc8c9ccccc9)c%10ccccc%10 | 28 |
| Cl[Ni](Cl)([P+](C1CCCCC1)(C2CCCCC2)C(Nc3ccccc3n5nc(c4ccccc4)cc5c6ccccc6)c7ccccc7)[P+](C8CCCCC8)(C9CCCCC9)C(Nc%10ccccc%10n%12nc(c%11ccccc%11)cc%12c%13ccccc%13)c%14ccccc%14 | 19 |
| Ni(OTf)2 | 17 |
| NiCl2(PhPCy2)2 | 17 |


### 2.4 Other public sources considered

| Source | Status | Notes |
| --- | --- | --- |
| Open Reaction Database (ord-data) | Already ingested via P1-09 (ord_normalized.csv, 2,910 rows). | 17 Ni reactions found via atomic-number audit. |
| USPTO OpenMolecules (480K) | Already ingested via P1-01 (uspto_openmolecules_normalized.csv, 530,238 rows). | 6 Ni reactions found. Catalysts not consistently recorded in patents. |
| HiTEA (per-question high-throughput) | Already ingested (hitea_full_normalized.csv, 39,546 rows). | 0 Ni reactions; HTE panels are Pd/Cu focused. |
| Reaxys | License-required; not accessible from this project. | Public abstracts only describe aggregate counts. |
| SciFinder | License-required; not accessible. | Same limitation as Reaxys. |
| Das et al. 2026 (Cernak lab) | 50,688-reaction Pd/Ni/Cu C-N coupling dataset announced in JACS 2026 (DOI 10.1021/jacs.6c05959). | Public release not yet available as of the report date; tracked for future ingest. |
| Doyle / MacMillan metallaphotoredox ORD submissions | Available in ord-data; already covered by ord_normalized.csv. | Subset of the 17 ORD Ni reactions. |


## 3. Go/No-Go Decision

**Go.**  1688 Ni coupling reactions identified (>= 50 threshold).  The supplement is written to `data/processed/ni_coupling_supplement.csv` and the per-source statistics are persisted to `data/summaries/ni_coupling_supplement_summary.json`.

### 3.1 Integration strategy with the PC-CNG benchmark

1. The supplement CSV follows the exact PC-CNG normalized schema (`source_id, reaction_smiles, reactants, agents, products, label_type, yield, source, split_key, split`).  It can be concatenated directly with `uspto_openmolecules_normalized.csv` and `ord_normalized.csv` for downstream featurisation.
2. Rows are tagged with `source = 'nicolit_literature'` so downstream tooling can stratify evaluation by data source.
3. The default `split = 'train'` keeps Ni reactions in the training fold.  A reviewer-only split can be derived later by re-hashing `split_key` if a held-out Ni evaluation is desired.
4. Because NiCOlit ships literature-mined yields, performance claims that rely on this supplement must use the existing `multiseed_paired_significance` harness with 10 seeds and a paired test against the no-supplement baseline.


## 4. Reproducibility

Reproduce this report with:

```bash
python3 -m pc_cng.research_ni_coupling_data \
  --output docs/ni_coupling_data_gap_research_YYYYMMDD.md
```

Run the unit tests with:

```bash
python3 -m pytest chem_negative_sampling/tests/test_ni_coupling_research.py -v
```
