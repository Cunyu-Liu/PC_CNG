#!/usr/bin/env bash
# 10-seed paired significance runner for P1-06 failure prototype calibrator.
# Runs the smoke config across 10 seeds and collects accuracy / target_hit_rate
# for a paired t-test against the random baseline.
set -u

PY=/home/cunyuliu/miniconda3/envs/pc_cng_gpu/bin/python
ROOT=/home/cunyuliu/pc_cng_research/chem_negative_sampling
REAL=/home/cunyuliu/pc_cng_research/data/processed/hitea_full_normalized.csv
ALT=/home/cunyuliu/pc_cng_research/data/processed/regiosqm20_normalized.csv
OUT_ROOT=$ROOT/results/failure_prototype_calibration_10seed_20260719
mkdir -p "$OUT_ROOT"

SEEDS=(20260719 11 22 33 44 55 66 77 88 99)
LIMIT=${1:-200}
EPOCHS=${2:-10}

echo "[$(date '+%F %T')] starting 10-seed run limit=$LIMIT epochs=$EPOCHS" | tee "$OUT_ROOT/run.log"

for seed in "${SEEDS[@]}"; do
  echo "[$(date '+%F %T')] seed=$seed ..." | tee -a "$OUT_ROOT/run.log"
  CUDA_VISIBLE_DEVICES= "$PY" -m pc_cng.train_failure_prototype_calibrator \
    --real-negatives "$REAL" \
    --alt-negatives "$ALT" \
    --output-dir "$OUT_ROOT/seed_$seed" \
    --epochs "$EPOCHS" --batch-size 32 --limit "$LIMIT" --seed "$seed" \
    >> "$OUT_ROOT/seed_$seed.log" 2>&1
  echo "[$(date '+%F %T')] seed=$seed done (exit=$?)" | tee -a "$OUT_ROOT/run.log"
done

echo "[$(date '+%F %T')] aggregating results ..." | tee -a "$OUT_ROOT/run.log"
"$PY" - <<'PYEOF'
import json, os, math
ROOT = "/home/cunyuliu/pc_cng_research/chem_negative_sampling/results/failure_prototype_calibration_10seed_20260719"
rows = []
for d in sorted(os.listdir(ROOT)):
    if not d.startswith("seed_"):
        continue
    summary_path = os.path.join(ROOT, d, "train_summary.json")
    if not os.path.exists(summary_path):
        continue
    s = json.load(open(summary_path))
    seed = int(d.split("_")[1])
    m = s["final_metrics"]
    # Random baseline accuracy = 1 / num_failure_types (uniform).
    baseline = 1.0 / 10
    rows.append({
        "seed": seed,
        "classification_accuracy": m["classification_accuracy"],
        "random_baseline_accuracy": baseline,
        "mean_entropy": m["mean_entropy"],
        "best_val_acc": m["best_val_acc"],
    })

# Paired t-test on accuracy vs baseline (scipy if available, else manual).
accs = [r["classification_accuracy"] for r in rows]
baselines = [r["random_baseline_accuracy"] for r in rows]
n = len(accs)
mean_diff = sum(a - b for a, b in zip(accs, baselines)) / max(n, 1)
var_diff = sum(((a - b) - mean_diff) ** 2 for a, b in zip(accs, baselines)) / max(n - 1, 1)
std_diff = math.sqrt(var_diff)
t_stat = (mean_diff / (std_diff / math.sqrt(max(n, 1)))) if std_diff > 0 else float("inf")

try:
    from scipy import stats  # type: ignore
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=max(n - 1, 1)))
except Exception:
    # Approximate p-value via normal CDF fallback.
    z = abs(t_stat)
    p_value = 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))

summary = {
    "n_seeds": n,
    "seeds": [r["seed"] for r in rows],
    "mean_accuracy": sum(accs) / n,
    "std_accuracy": (sum((a - sum(accs) / n) ** 2 for a in accs) / max(n - 1, 1)) ** 0.5,
    "mean_baseline": sum(baselines) / n,
    "mean_entropy": sum(r["mean_entropy"] for r in rows) / n,
    "mean_diff_vs_baseline": mean_diff,
    "std_diff_vs_baseline": std_diff,
    "t_statistic": t_stat,
    "p_value": p_value,
    "per_seed": rows,
}
out_path = os.path.join(ROOT, "significance_summary.json")
json.dump(summary, open(out_path, "w"), indent=2)
print(json.dumps(summary, indent=2))
print(f"\n[done] -> {out_path}")
PYEOF

echo "[$(date '+%F %T')] 10-seed run complete." | tee -a "$OUT_ROOT/run.log"
