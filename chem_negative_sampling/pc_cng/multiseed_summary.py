"""
Multiseed summary generator for pairwise reward MLP experiments.
Usage:
    python -m pc_cng.multiseed_summary --exp-dir /path/to/results --prefix "exp_prefix_" --seeds 20260710,20260711,...
"""
import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, List


def load_ranking_metrics(seed_dir: Path) -> Dict:
    path = seed_dir / "rerank_same_split" / "ranking_metrics.json"
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def extract_top1_by_split(data: Dict) -> Dict[str, float]:
    out = {}
    out["overall"] = data["overall"]["top1"]
    out["overall_mrr"] = data["overall"]["mrr"]
    out["overall_ndcg"] = data["overall"]["ndcg"]
    for split_name, split_data in data["by_split"].items():
        out[split_name] = split_data["top1"]
        out[f"{split_name}_mrr"] = split_data["mrr"]
        out[f"{split_name}_ndcg"] = split_data["ndcg"]
    return out


def compute_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", type=str, required=True)
    parser.add_argument("--prefix", type=str, required=True, help="Seed dir prefix (e.g. 'v2_hidden4096_pairwise_seed')")
    parser.add_argument("--seeds", type=str, required=True, help="Comma-separated seed list")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]
    exp_dir = Path(args.exp_dir)

    all_metrics = {}
    completed = []
    pending = []

    for seed in seeds:
        seed_dir = exp_dir / f"{args.prefix}{seed}"
        data = load_ranking_metrics(seed_dir)
        if data is None:
            pending.append(seed)
            continue
        completed.append(seed)
        all_metrics[seed] = extract_top1_by_split(data)

    if not all_metrics:
        print("No completed seeds found.")
        return

    metric_keys = list(all_metrics[completed[0]].keys())
    stats = {}
    for key in metric_keys:
        values = [all_metrics[s][key] for s in completed]
        stats[key] = compute_stats(values)

    print(f"\nCompleted: {len(completed)}/{len(seeds)} seeds")
    if pending:
        print(f"Pending: {', '.join(pending)}")
    print()

    print(f"{'Metric':<25} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 70)
    for key in metric_keys:
        s = stats[key]
        print(f"{key:<25} {s['mean']*100:>9.2f}% {s['std']*100:>9.2f}% {s['min']*100:>9.2f}% {s['max']*100:>9.2f}%")

    if args.output:
        out = {
            "completed_seeds": completed,
            "pending_seeds": pending,
            "per_seed": all_metrics,
            "stats": stats,
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSummary written to {args.output}")


if __name__ == "__main__":
    main()
