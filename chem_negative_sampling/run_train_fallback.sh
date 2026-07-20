cd /home/cunyuliu/pc_cng_research/chem_negative_sampling || exit 1
OUT=/home/cunyuliu/pc_cng_research/results/type1_class_fallback_supplement_20260711/fallback_augmented_pairwise_seed20260710
rm -rf "$OUT"
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. /home/cunyuliu/miniconda3/envs/pc_cng_gpu/bin/python -m pc_cng.train_pairwise_reward_mlp \
  --real-csv /home/cunyuliu/pc_cng_research/data/processed/regiosqm20_normalized.csv \
  --real-csv /home/cunyuliu/pc_cng_research/data/processed/hitea_full_normalized.csv \
  --synthetic-csv /home/cunyuliu/pc_cng_research/results/type1_diverse_anchor_full/diverse_anchor_candidates_reviewed.csv \
  --synthetic-csv /home/cunyuliu/pc_cng_research/results/type1_class_quota_supplement_20260711/class_quota_candidates_reviewed.csv \
  --synthetic-csv /home/cunyuliu/pc_cng_research/results/type1_class_fallback_supplement_20260711/class_fallback_candidates_reviewed.csv \
  --output-dir "$OUT" \
  --epochs 80 \
  --batch-size 4096 \
  --hidden-dim 2048 \
  --n-bits 4096 \
  --dropout 0.20 \
  --seed 20260710
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. /home/cunyuliu/miniconda3/envs/pc_cng_gpu/bin/python -m pc_cng.evaluate_candidate_reranking \
  --real-csv /home/cunyuliu/pc_cng_research/data/processed/regiosqm20_normalized.csv \
  --real-csv /home/cunyuliu/pc_cng_research/data/processed/hitea_full_normalized.csv \
  --synthetic-csv /home/cunyuliu/pc_cng_research/results/type1_diverse_anchor_full/diverse_anchor_candidates_reviewed.csv \
  --synthetic-csv /home/cunyuliu/pc_cng_research/results/type1_class_quota_supplement_20260711/class_quota_candidates_reviewed.csv \
  --synthetic-csv /home/cunyuliu/pc_cng_research/results/type1_class_fallback_supplement_20260711/class_fallback_candidates_reviewed.csv \
  --model-dir "$OUT" \
  --output-dir "$OUT/rerank_same_split" \
  --candidate-scope same_split \
  --batch-size 4096 \
  --device cuda
PYTHONPATH=. /home/cunyuliu/miniconda3/envs/pc_cng_gpu/bin/python -m pc_cng.analyze_reaction_class_benchmark \
  --score-csv fallback_trained="$OUT/rerank_same_split/candidate_scores.csv:score" \
  --output-dir /home/cunyuliu/pc_cng_research/results/type1_class_fallback_supplement_20260711/reaction_class_fallback_trained \
  --min-groups 20 \
  --weak-top1 0.80 \
  --weak-mrr 0.85
