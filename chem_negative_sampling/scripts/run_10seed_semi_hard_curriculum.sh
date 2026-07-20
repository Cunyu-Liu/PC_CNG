#!/bin/bash
# 10-seed background training launcher for P1-07 Semi-hard Curriculum.
# Runs on GPU 6 (calibrate PID 2544995 on GPU 4 is NOT disturbed).
set -uo pipefail

GPU_ID="${1:-6}"
RESULTS_ROOT="/home/cunyuliu/pc_cng_research/results/semi_hard_curriculum_10seed_20260719"
REAL_CSV="/home/cunyuliu/pc_cng_research/data/processed/regiosqm20_normalized.csv"
SYNTH_CSV="/home/cunyuliu/pc_cng_research/results/v2_boundary_generation/regiosqm20_boundary_negatives_reviewed.csv"
CHEM_DIR="/home/cunyuliu/pc_cng_research/chem_negative_sampling"
PYTHON="/home/cunyuliu/miniconda3/envs/pc_cng_gpu/bin/python"
SEEDS=(20260719 20260720 20260721 20260722 20260723 20260724 20260725 20260726 20260727 20260728)

mkdir -p "$RESULTS_ROOT"
LOG_FILE="$RESULTS_ROOT/launcher.log"
echo "[launcher] start=$(date +%Y-%m-%dT%H:%M:%S) gpu=$GPU_ID" | tee -a "$LOG_FILE"

cd "$CHEM_DIR"
for seed in "${SEEDS[@]}"; do
    out_dir="$RESULTS_ROOT/seed_${seed}"
    if [ -f "$out_dir/comparison.json" ]; then
        echo "[launcher] seed=$seed already done, skipping" | tee -a "$LOG_FILE"
        continue
    fi
    mkdir -p "$out_dir"
    seed_log="$out_dir/seed.log"
    echo "[launcher] seed=$seed start=$(date +%Y-%m-%dT%H:%M:%S)" | tee -a "$LOG_FILE"
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" -m pc_cng.run_semi_hard_curriculum \
        --real-csv "$REAL_CSV" \
        --synthetic-csv "$SYNTH_CSV" \
        --output-dir "$out_dir" \
        --quantile-rounds 4 \
        --epochs-per-round 3 \
        --overlap 0.2 \
        --pairwise-weight 1.0 \
        --bce-weight 1.0 \
        --margin 0.5 \
        --feature-mode morgan \
        --n-bits 1024 \
        --fp-mode binary \
        --hidden-dim 512 \
        --batch-size 512 \
        --lr 0.001 \
        --dropout 0.1 \
        --checkpoint-metric val_top1 \
        --checkpoint-group-by reactants \
        --bootstrap-iterations 2000 \
        --seed "$seed" > "$seed_log" 2>&1
    rc=$?
    echo "[launcher] seed=$seed end=$(date +%Y-%m-%dT%H:%M:%S) rc=$rc" | tee -a "$LOG_FILE"
done

echo "[launcher] all_seeds_done=$(date +%Y-%m-%dT%H:%M:%S)" | tee -a "$LOG_FILE"

# Aggregate per-seed comparison.json into a single summary
"$PYTHON" - <<'PYEOF'
import json, os, glob
root = "/home/cunyuliu/pc_cng_research/results/semi_hard_curriculum_10seed_20260719"
rows = []
for path in sorted(glob.glob(os.path.join(root, "seed_*", "comparison.json"))):
    with open(path) as f:
        d = json.load(f)
    seed = int(os.path.basename(os.path.dirname(path)).split("_")[-1])
    rows.append({
        "seed": seed,
        "curriculum_test_top1_model": d.get("curriculum_test_top1_model"),
        "one_shot_test_top1_model": d.get("one_shot_test_top1_model"),
        "model_diff_pp": d.get("model_diff_pp"),
        "n_paired_groups": d.get("n_paired_groups"),
        "bootstrap_ci_low": d.get("bootstrap_ci_low"),
        "bootstrap_ci_high": d.get("bootstrap_ci_high"),
        "ci_fully_positive": d.get("ci_fully_positive"),
        "permutation_p_value": d.get("permutation_p_value"),
        "sign_test_p_value": d.get("sign_test_p_value"),
        "go_nogo_decision": d.get("go_nogo_decision"),
    })
out = os.path.join(root, "aggregated_10seed.json")
with open(out, "w") as f:
    json.dump({"seeds": rows, "n_seeds": len(rows)}, f, indent=2)
print(f"[launcher] aggregated -> {out}")
PYEOF
