# P4-G1 Candidate Manifest Audit Report

**Manifests audited:** 3


## hte_feasibility_v1.json
- benchmark_name: P4-HTE-Feasibility
- manifest_hash: 5701c5c98f79a2f7... (verified: True)
- n_groups: 500
- n_candidates: 4000
- n_gold: 500
- errors: 0
- warnings: 0
- candidate_source_distribution: {'gold': 500, 'random_mismatch': 500, 'random_corruption': 500, 'tanimoto_retrieval': 500, 'template_perturbation': 500, 'unconstrained_edit': 500, 'rule_pc_cng': 500, 'external_beam': 500}
- split_distribution: {'test': 440, 'train': 3152, 'val': 408}
- candidates_per_group: min=8, max=8, mean=8.0
- known_positive_collisions: 34
- train_overlaps: 539
- nearest_train_similarity: min=0.0, max=1.0, mean=0.2215
- oracle_top1_coverage: 1.0
- n_unique_scaffolds: 257
- n_unique_parents: 500
- parent_leakage_count: 0

## fixed_forward_candidates_v1.json
- benchmark_name: P4-Fixed-Forward-Candidates
- manifest_hash: a722572d8e188336... (verified: True)
- n_groups: 500
- n_candidates: 4000
- n_gold: 500
- errors: 0
- warnings: 0
- candidate_source_distribution: {'gold': 500, 'random_mismatch': 500, 'random_corruption': 500, 'tanimoto_retrieval': 500, 'template_perturbation': 500, 'unconstrained_edit': 500, 'rule_pc_cng': 500, 'external_beam': 500}
- split_distribution: {'val': 416, 'train': 3184, 'test': 400}
- candidates_per_group: min=8, max=8, mean=8.0
- known_positive_collisions: 24
- train_overlaps: 719
- nearest_train_similarity: min=0.0, max=1.0, mean=0.332
- oracle_top1_coverage: 1.0
- n_unique_scaffolds: 714
- n_unique_parents: 499
- parent_leakage_count: 0

## fixed_retro_candidates_v1.json
- benchmark_name: P4-Fixed-Retro-Candidates
- manifest_hash: ae9233828e778490... (verified: True)
- n_groups: 500
- n_candidates: 4000
- n_gold: 500
- errors: 0
- warnings: 0
- candidate_source_distribution: {'gold': 500, 'random_mismatch': 500, 'random_corruption': 500, 'tanimoto_retrieval': 500, 'template_perturbation': 500, 'unconstrained_edit': 500, 'rule_pc_cng': 500, 'external_beam': 500}
- split_distribution: {'train': 3192, 'test': 376, 'val': 432}
- candidates_per_group: min=8, max=8, mean=8.0
- known_positive_collisions: 20
- train_overlaps: 679
- nearest_train_similarity: min=0.0, max=1.0, mean=0.3505
- oracle_top1_coverage: 1.0
- n_unique_scaffolds: 758
- n_unique_parents: 500
- parent_leakage_count: 0