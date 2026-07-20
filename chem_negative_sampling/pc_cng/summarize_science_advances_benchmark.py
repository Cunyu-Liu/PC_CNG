"""Summarize Science Advances-style PC-CNG benchmark records.

The benchmark script writes per-seed records. This summarizer adds bootstrap
confidence intervals, positive-delta rates, and manuscript-friendly tables.
It uses only the Python standard library so it can run in lightweight envs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Sequence, Tuple


METRICS = ["top1", "top3", "mrr", "ndcg"]
FAMILIES = ["real_only", "pc_cng_augmented", "delta"]


def load_records(path: str) -> List[Dict[str, object]]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    return list(payload.get("records", []))


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_ci(values: Sequence[float], iterations: int, seed: int) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)
    means = []
    values = list(values)
    for _ in range(iterations):
        sample = [rng.choice(values) for _ in values]
        means.append(mean(sample))
    return percentile(means, 0.025), percentile(means, 0.975)


def sign_test_p_value(values: Sequence[float]) -> float:
    positives = sum(1 for value in values if value > 0)
    negatives = sum(1 for value in values if value < 0)
    n = positives + negatives
    if n == 0:
        return 1.0
    observed = min(positives, negatives)
    # Exact two-sided binomial sign test with p=0.5.
    prob = 0.0
    for k in range(0, observed + 1):
        prob += math.comb(n, k) * (0.5**n)
    return min(1.0, 2.0 * prob)


def summarize_values(values: Sequence[float], bootstrap_iterations: int, seed: int) -> Dict[str, float | int]:
    values = list(values)
    ci_low, ci_high = bootstrap_ci(values, bootstrap_iterations, seed)
    return {
        "mean": mean(values) if values else 0.0,
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "n": len(values),
    }


def setting_records(records: Sequence[Dict[str, object]], setting: str) -> List[Dict[str, object]]:
    return [record for record in records if record.get("setting") == setting]


def build_summary(records: Sequence[Dict[str, object]], bootstrap_iterations: int, seed: int) -> Dict[str, object]:
    summary: Dict[str, object] = {}
    for setting in sorted({str(record.get("setting", "")) for record in records}):
        subset = setting_records(records, setting)
        setting_summary: Dict[str, object] = {"n": len(subset)}
        for family in FAMILIES:
            family_summary: Dict[str, object] = {}
            for metric in METRICS:
                values = [float(dict(record.get(family, {})).get(metric, 0.0)) for record in subset]
                family_summary[metric] = summarize_values(values, bootstrap_iterations, seed + len(metric))
            setting_summary[family] = family_summary
        delta_summary: Dict[str, object] = {}
        for metric in METRICS:
            values = [float(dict(record.get("delta", {})).get(metric, 0.0)) for record in subset]
            delta_summary[metric] = {
                "positive_rate": sum(1 for value in values if value > 0) / len(values) if values else 0.0,
                "sign_test_p_value": sign_test_p_value(values),
            }
        setting_summary["delta_tests"] = delta_summary
        summary[setting] = setting_summary
    return summary


def format_pm(stats: Dict[str, float | int], scale: float = 100.0) -> str:
    return f"{float(stats['mean']) * scale:.2f} ± {float(stats['std']) * scale:.2f}"


def format_ci(stats: Dict[str, float | int], scale: float = 100.0) -> str:
    return f"[{float(stats['ci95_low']) * scale:.2f}, {float(stats['ci95_high']) * scale:.2f}]"


def table_rows(summary: Dict[str, object]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for setting, raw_setting in sorted(summary.items()):
        setting_data = dict(raw_setting)
        n = int(setting_data.get("n", 0))
        for metric in METRICS:
            real_stats = dict(dict(setting_data["real_only"])[metric])
            pc_stats = dict(dict(setting_data["pc_cng_augmented"])[metric])
            delta_stats = dict(dict(setting_data["delta"])[metric])
            tests = dict(dict(setting_data["delta_tests"])[metric])
            rows.append(
                {
                    "setting": setting,
                    "metric": metric,
                    "n_seeds": str(n),
                    "real_only_mean_std_pct": format_pm(real_stats),
                    "pc_cng_mean_std_pct": format_pm(pc_stats),
                    "delta_mean_std_pct": format_pm(delta_stats),
                    "delta_ci95_pct": format_ci(delta_stats),
                    "positive_delta_rate": f"{float(tests['positive_rate']):.3f}",
                    "sign_test_p_value": f"{float(tests['sign_test_p_value']):.4f}",
                }
            )
    return rows


def write_csv(path: str, rows: Sequence[Dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "setting",
        "metric",
        "n_seeds",
        "real_only_mean_std_pct",
        "pc_cng_mean_std_pct",
        "delta_mean_std_pct",
        "delta_ci95_pct",
        "positive_delta_rate",
        "sign_test_p_value",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: str, rows: Sequence[Dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    headers = [
        "Setting",
        "Metric",
        "n",
        "Real-only",
        "PC-CNG",
        "Delta",
        "Delta 95% CI",
        "Positive delta rate",
        "Sign-test p",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["setting"],
                    row["metric"],
                    row["n_seeds"],
                    row["real_only_mean_std_pct"],
                    row["pc_cng_mean_std_pct"],
                    row["delta_mean_std_pct"],
                    row["delta_ci95_pct"],
                    row["positive_delta_rate"],
                    row["sign_test_p_value"],
                ]
            )
            + " |"
        )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Benchmark summary.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260710)
    args = parser.parse_args()

    records = load_records(args.input)
    summary = build_summary(records, args.bootstrap_iterations, args.seed)
    rows = table_rows(summary)
    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, "benchmark_with_ci.json")
    csv_path = os.path.join(args.output_dir, "paper_table.csv")
    md_path = os.path.join(args.output_dir, "paper_table.md")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "records": records}, handle, indent=2, ensure_ascii=False)
    write_csv(csv_path, rows)
    write_markdown(md_path, rows)
    print(json.dumps({"summary": summary, "paper_table_csv": csv_path, "paper_table_md": md_path}, indent=2))


if __name__ == "__main__":
    main()
