# P4-G1 Benchmark Contract & Candidate Manifest Freeze

**Phase:** P4-G1
**Spec reference:** `提示词/pccng 的分阶段提示词.md` lines 327-519
**Generated:** 2026-07-21
**Repo:** `/home/cunyuliu/pc_cng_research` (branch `main`)
**Builder module:** `chem_negative_sampling/pc_cng/build_p4_candidate_manifests.py`
**Audit module:** `chem_negative_sampling/pc_cng/audit_p4_candidate_manifests.py`
**Tests:** `chem_negative_sampling/tests/test_p4_candidate_manifest.py` (42 passed, 4 skipped locally)
**Entry condition:** P4-G0 == GO (satisfied — see `docs/p4_baseline_lock.md` §10)

---

## 1. Purpose

This contract freezes three fixed candidate benchmarks so that **all downstream methods** (PC-CNG, Chemformer-LoRA, SOTA baselines, LLM judge, AiZynthFinder, etc.) **must evaluate against the exact same candidate sets**. This eliminates the confound identified in the P4-G0 audit:

> Different methods used different candidate sets in P3, producing non-comparable performance numbers. Tanimoto-NN's `1.0000` MRR was a direct consequence of a degenerate candidate set (`dedup_key=parent_product`).

P4-G1 freezes the candidate universe so that any MRR / Top-k / NDCG difference observed in P4-G2+ is attributable to the **method**, not to the candidate pool.

---

## 2. Benchmarks

### B1: P4-HTE-Feasibility

**File:** `data/p4/manifests/hte_feasibility_v1.json`

Real experimental groups from HTEa (39,546 reactions). Tasks:
- low-yield classification
- yield-bin prediction
- plate-level ranking
- condition-specific feasibility

Groups are keyed by `split_key` (the experimental plate / reaction family identifier). Each group contains exactly 8 candidates (1 gold + 7 non-gold from different sources).

### B2: P4-Fixed-Forward-Candidates

**File:** `data/p4/manifests/fixed_forward_candidates_v1.json`

Given reactants and conditions, rank fixed candidate products. Source data: USPTO-OpenMolecules (sampled up to 5,000 reactions, capped at 500 groups for v1).

### B3: P4-Fixed-Retro-Candidates

**File:** `data/p4/manifests/fixed_retro_candidates_v1.json`

Given a target product, rank fixed precursor candidates. Source data: USPTO-OpenMolecules (same sampling protocol as B2).

---

## 3. Candidate Sources (8 per group)

Per spec lines 366-375, each group contains candidates from these 8 sources:

| Rank | Source | Description |
|---:|---|---|
| 0 | `gold` | The actual observed product (B1/B2) or reactant set (B3) from the source reaction. Exactly one per group. |
| 1 | `random_mismatch` | A randomly selected different product/reactant from the train pool. |
| 2 | `random_corruption` | Character-level corruption (1-3 mutations: delete/insert/swap) of the gold SMILES. |
| 3 | `tanimoto_retrieval` | The most Tanimoto-similar (Morgan FP r=2, 1024-bit) train molecule to the gold, excluding exact matches. |
| 4 | `template_perturbation` | A molecule drawn from the same reaction class (B1) or a random train molecule (B2/B3). |
| 5 | `unconstrained_edit` | A truncation of the unmapped gold SMILES (first half). |
| 6 | `rule_pc_cng` | Rule-based functional group substitution (e.g., `C(=O)O → C(=O)N`, `[Br] → [Cl]`). |
| 7 | `external_beam` | A molecule drawn from a different dataset (ORD for B1/B3, HTEa for B2). |

**Learned PC-CNG** candidates are NOT included in v1. They will be appended as v2 manifests in a later phase, per spec line 377: "Learned PC-CNG 后续作为新 candidate source 追加，不得修改 v1 manifest".

---

## 4. Required Candidate Fields (24 per spec)

Per spec lines 381-406, every candidate record contains exactly these 24 fields:

```text
benchmark_name, group_id, source_reaction_id, parent_reaction_id,
experimental_group_id, gold_candidate, candidate_id, candidate_smiles,
candidate_source, candidate_source_rank, canonical_smiles,
atom_mapping_status, reaction_family, reaction_template, product_scaffold,
edit_type, edit_distance, train_overlap, known_positive_collision,
nearest_train_similarity, split, oracle_coverage, manifest_version,
manifest_hash
```

Field semantics:

| Field | Type | Description |
|---|---|---|
| `benchmark_name` | str | `P4-HTE-Feasibility` / `P4-Fixed-Forward-Candidates` / `P4-Fixed-Retro-Candidates` |
| `group_id` | str | Unique group identifier (e.g., `hte_<split_key>`, `fwd_<source_id>`, `retro_<source_id>`) |
| `source_reaction_id` | str | ID of the source reaction in the original dataset |
| `parent_reaction_id` | str | Parent reaction identifier used for split isolation |
| `experimental_group_id` | str | Experimental group / plate identifier |
| `gold_candidate` | bool | True if this is the observed correct answer |
| `candidate_id` | str | `<group_id>_<candidate_source>_<rank>` |
| `candidate_smiles` | str | SMILES as stored in source data (may be atom-mapped) |
| `candidate_source` | str | One of the 8 source labels above |
| `candidate_source_rank` | int | 0 (gold) through 7 (external_beam) |
| `canonical_smiles` | str | RDKit-canonicalized SMILES with atom mapping stripped |
| `atom_mapping_status` | str | `mapped` / `unmapped` / `unknown` |
| `reaction_family` | str | Reaction class label (e.g., HTEa reaction_class, `uspto_om`) |
| `reaction_template` | str | `HTEa` / `forward` / `retro` |
| `product_scaffold` | str | Bemis-Murcko scaffold of `canonical_smiles` |
| `edit_type` | str | `none` for gold; otherwise equals `candidate_source` |
| `edit_distance` | int | Character-level Levenshtein distance to gold canonical SMILES (0 for gold) |
| `train_overlap` | bool | True if `canonical_smiles` appears in the train pool |
| `known_positive_collision` | bool | True if a non-gold candidate's canonical SMILES equals the gold canonical SMILES |
| `nearest_train_similarity` | float | Max Tanimoto similarity to the first 200 train molecules |
| `split` | str | `train` / `val` / `test` (inherited from source row) |
| `oracle_coverage` | float | 1.0 for gold, 0.0 otherwise |
| `manifest_version` | str | `v1` |
| `manifest_hash` | str | SHA-256 of the manifest (backfilled into every candidate for traceability) |

---

## 5. Split Isolation Principle

Per spec lines 408-418, split priority (highest first):

1. **Parent reaction isolation** — same `parent_reaction_id` must not appear in multiple splits
2. **Experimental group isolation** — same `experimental_group_id` must not cross splits
3. **Reaction-family isolation** — same `reaction_family` must not cross splits
4. **Template/scaffold isolation** — same `product_scaffold` should not cross splits
5. **Random split** — only as a fallback

**Hard constraint (spec line 418):** "严禁同一 parent reaction 的派生候选跨 train/test".

The audit module (`audit_p4_candidate_manifests.py`) checks this explicitly: any `parent_reaction_id` appearing in more than one split value raises a `Parent leakage` error and forces `NO_GO`.

---

## 6. Manifest Hash

Each manifest's `manifest_hash` is a SHA-256 of the canonical JSON content, computed as:

```python
content = _strip_manifest_hash(manifest)  # recursively removes manifest_hash
canonical = json.dumps(content, sort_keys=True, ensure_ascii=False)
hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
```

The hash is **stable** against backfilling: candidate records store the manifest hash for traceability, but those nested `manifest_hash` fields are stripped before hashing, so adding the hash to candidates does not invalidate the hash itself.

The audit recomputes the hash and verifies it matches the stored value. Any modification to the manifest content (groups, candidates, fields) changes the hash and triggers `NO_GO`.

---

## 7. Reproducibility

- **Random seed:** `MANIFEST_SEED = 20260721` (fixed in source code, not configurable)
- **RDKit version:** 2025.03.6 (canonical SMILES / Tanimoto / scaffold depend on RDKit version)
- **Data sources:** `data/processed/hitea_full_normalized.csv`, `data/processed/uspto_openmolecules_normalized.csv`, `data/processed/ord_normalized.csv`
- **No model dependence:** v1 manifests contain zero learned PC-CNG candidates. All 8 candidate sources are deterministic given the seed and data.

---

## 8. Oracle Coverage

Oracle coverage is reported per group as `oracle_top1_coverage = n_gold / n_groups`. Since every group has exactly 1 gold candidate, the oracle Top-1 coverage is exactly 1.0 for every manifest. A method that ranks the gold candidate first achieves oracle-level performance.

Oracle Top-k (k > 1) is not meaningful here because there is exactly one gold per group; the relevant metric is MRR / Top-1 / NDCG over the 8-candidate ranking.

---

## 9. Audit Outputs

The audit module writes the following to `results/p4_candidate_audit/`:

| File | Contents |
|---|---|
| `audit_report.md` | Human-readable audit summary per manifest |
| `manifest_audit_details.json` | Full structured findings (errors, warnings, stats) per manifest |
| `go_no_go.json` | Phase verdict with `status`, `primary_metric`, `predeclared_threshold`, `evidence_paths`, `limitations`, `next_phase_allowed` |

Per-spec required statistics (spec lines 432-443) are all reported:

- group count
- candidate count
- candidate-count-per-group distribution (min / max / mean)
- candidate-source distribution
- known-positive collisions
- oracle Top-1 coverage
- candidate coverage (scaffold / template / parent diversity)
- reaction-family distribution
- train/test nearest-neighbor similarity (min / max / mean)
- parent / template / scaffold overlap

---

## 10. GO Criteria

Per spec lines 478-485, **GO** requires:

| Criterion | How verified |
|---|---|
| Each group has exactly one gold | `audit_manifest` counts golds per group; any deviation is an error |
| No parent leakage | `audit_manifest` tracks `parent_split_map`; any parent in >1 split is an error |
| All candidate sources reproducible | Fixed seed + deterministic sources + no learned PC-CNG in v1 |
| Manifest has fixed hash | `_compute_manifest_hash` recomputed and compared to stored value |
| Oracle coverage reported | `oracle_top1_coverage` in stats |
| All methods use same manifest | Contract enforced: downstream phases must reference `data/p4/manifests/*_v1.json` by hash |

**NO-GO** triggers (spec lines 487-492):

- gold or candidate set varies by method (not applicable — manifest is method-agnostic)
- parent/template leakage unresolved
- benchmark depends on non-public implicit files
- third party cannot reconstruct candidates

---

## 11. Downstream Contract

All P4-G2+ training and evaluation entry points MUST accept:

```text
--candidate-manifest data/p4/manifests/<benchmark>_v1.json
```

and MUST verify the manifest hash matches the value recorded in this contract before proceeding. Any method that modifies the candidate set is in violation and must publish a v2 manifest with a new hash.

---

## 12. Manifest Hashes (frozen 2026-07-21)

Built and verified on remote server `/home/cunyuliu/pc_cng_research`:

```bash
python3 -m pc_cng.build_p4_candidate_manifests --output-dir data/p4/manifests
```

| Benchmark | File | Full SHA-256 |
|---|---|---|
| B1 HTE-Feasibility | `data/p4/manifests/hte_feasibility_v1.json` | `5701c5c98f79a2f7b27c8eb22be71883850c4c0b98acde0ff3269527d2082a1f` |
| B2 Fixed-Forward-Candidates | `data/p4/manifests/fixed_forward_candidates_v1.json` | `a722572d8e1883366893b23570ce304fd25d803f22e17c52933dd395582e2941` |
| B3 Fixed-Retro-Candidates | `data/p4/manifests/fixed_retro_candidates_v1.json` | `ae9233828e7784900e02cd81049e8724264c7c9d278908b10df58089a398533c` |

Summary also at `data/p4/splits/split_summary_v1.json`.

---

## 13. Acceptance Commands

Per spec lines 447-473:

```bash
# Build
python3 -m pc_cng.build_p4_candidate_manifests --output-dir data/p4/manifests

# Audit
python3 -m pc_cng.audit_p4_candidate_manifests \
  --manifest-dir data/p4/manifests \
  --output-dir results/p4_candidate_audit

# Tests
python3 -m pytest chem_negative_sampling/tests/test_p4_candidate_manifest.py -v

# Spec structural check
python3 - <<'PY'
import json
from pathlib import Path

for path in Path("data/p4/manifests").glob("*.json"):
    data = json.load(open(path))
    assert data["manifest_hash"]
    assert data["groups"]
    for group in data["groups"]:
        assert sum(bool(c["gold_candidate"]) for c in group["candidates"]) == 1
        assert len({c["candidate_id"] for c in group["candidates"]}) == len(group["candidates"])
PY
```

---

## 14. Limitations

1. **Sample size cap:** B2 and B3 sample up to 5,000 USPTO-OM reactions and cap at 500 groups for v1. Full-scale evaluation may require v2 expansion.
2. **Tanimoto retrieval scope:** Nearest-neighbor search is limited to the first 200-500 train molecules per group for compute efficiency. A full-train NN search may surface different (potentially higher-similarity) candidates.
3. **No learned PC-CNG in v1:** Learned PC-CNG candidates will be appended as v2 in a later phase; v1 results should be interpreted as "fixed-candidate" baselines.
4. **External beam scope:** External candidates are drawn from a 100-row sample of ORD (B1/B3) or HTEa (B2). Larger external pools may yield harder negatives.
5. **Edit distance proxy:** Character-level Levenshtein on SMILES strings is a coarse proxy for graph edit distance; refined metrics may be added in v2.

---

## 15. P4-G1 Verdict

**Status:** `GO`
**`next_phase_allowed`:** `true`

### Summary

| Manifest | Groups | Candidates | Golds | Errors | Hash verified | Parent leakage |
|---|---:|---:|---:|---:|---|---:|
| B1 HTE-Feasibility | 500 | 4000 | 500 | 0 | ✅ | 0 |
| B2 Fixed-Forward-Candidates | 500 | 4000 | 500 | 0 | ✅ | 0 |
| B3 Fixed-Retro-Candidates | 500 | 4000 | 500 | 0 | ✅ | 0 |
| **Total** | **1500** | **12000** | **1500** | **0** | — | **0** |

### Per-spec GO criteria verification

| Criterion (spec lines 478-485) | Status | Evidence |
|---|---|---|
| Each group has exactly one gold | ✅ PASS | 1500/1500 groups have exactly 1 gold_candidate |
| No parent leakage | ✅ PASS | `parent_leakage_count=0` across all 3 manifests |
| All candidate sources reproducible | ✅ PASS | Fixed seed `20260721`, deterministic sources, no learned PC-CNG in v1 |
| Manifest has fixed hash | ✅ PASS | All 3 manifest hashes verified by recomputation |
| Oracle coverage reported | ✅ PASS | `oracle_top1_coverage=1.0` for all 3 manifests |
| All methods use same manifest | ✅ CONTRACT | Downstream phases must reference `data/p4/manifests/*_v1.json` by hash |

### Test results

```
46 passed in 50.18s
```

All tests passed including:
- 14 helper function tests
- 5 `_make_candidate` tests
- 6 constants tests
- 7 manifest structure tests
- 7 audit module tests
- 2 go_no_go tests
- 4 end-to-end build tests
- 1 spec structural acceptance test

### Audit evidence

- `results/p4_candidate_audit/go_no_go.json` — verdict file
- `results/p4_candidate_audit/audit_report.md` — human-readable audit
- `results/p4_candidate_audit/manifest_audit_details.json` — structured findings
- `data/p4/manifests/{hte_feasibility,fixed_forward_candidates,fixed_retro_candidates}_v1.json` — frozen manifests
- `data/p4/splits/split_summary_v1.json` — split summary with hashes

Per spec line 518: "完成三个 manifest、审计报告、单元测试和 go_no_go.json 后停止".
