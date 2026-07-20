"""Summarize multi-seed reaction-LM candidate ranking benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from statistics import mean, pstdev
from typing import Dict, List, Sequence, Tuple


METRICS = ["top1", "top3", "mrr", "ndcg"]
RESERVED_KEYS = {"run", "setting", "seed", "paths", "delta"}


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
    values = list(values)
    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)
    boot_means = []
    for _ in range(iterations):
        sample = [rng.choice(values) for _ in values]
        boot_means.append(mean(sample))
    return percentile(boot_means, 0.025), percentile(boot_means, 0.975)


def sign_test_p_value(values: Sequence[float]) -> float:
    positives = sum(1 for value in values if value > 0)
    negatives = sum(1 for value in values if value < 0)
    n = positives + negatives
    if n == 0:
        return 1.0
    observed = min(positives, negatives)
    probability = 0.0
    for k in range(observed + 1):
        probability += math.comb(n, k) * (0.5**n)
    return min(1.0, 2.0 * probability)


def summarize_values(values: Sequence[float], iterations: int, seed: int) -> Dict[str, float | int]:
    values = list(values)
    low, high = bootstrap_ci(values, iterations, seed)
    return {
        "mean": mean(values) if values else 0.0,
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "ci95_low": low,
        "ci95_high": high,
        "n": len(values),
    }


def infer_families(records: Sequence[Dict[str, object]]) -> List[str]:
    families = set()
    for record in records:
        for key, value in record.items():
            if key in RESERVED_KEYS:
                continue
            if isinstance(value, dict) and any(metric in value for metric in METRICS):
                families.add(key)
    return sorted(families)


def build_summary(
    records: Sequence[Dict[str, object]],
    families: Sequence[str],
    iterations: int,
    seed: int,
) -> Dict[str, object]:
    summary: Dict[str, object] = {}
    settings = sorted({str(record.get("setting", "")) for record in records})
    for setting in settings:
        subset = [record for record in records if str(record.get("setting", "")) == setting]
        setting_summary: Dict[str, object] = {"n": len(subset)}
        for family in families:
            family_summary: Dict[str, object] = {}
            for metric in METRICS:
                values = [float(dict(record.get(family, {})).get(metric, 0.0)) for record in subset]
                family_summary[metric] = summarize_values(values, iterations, seed + len(family) + len(metric))
            setting_summary[family] = family_summary
        if "delta" in families or any("delta" in record for record in subset):
            delta_summary: Dict[str, object] = {}
            for metric in METRICS:
                values = [float(dict(record.get("delta", {})).get(metric, 0.0)) for record in subset]
                stats = summarize_values(values, iterations, seed + 100 + len(metric))
                stats.update(
                    {
                        "positive_rate": sum(1 for value in values if value > 0) / len(values) if values else 0.0,
                        "sign_test_p_value": sign_test_p_value(values),
                    }
                )
                delta_summary[metric] = stats
            setting_summary["delta"] = delta_summary
        summary[setting] = setting_summary
    return summary


def pct(stats: Dict[str, float | int]) -> str:
    return f"{float(stats['mean']) * 100.0:.2f} +/- {float(stats['std']) * 100.0:.2f}"


def ci(stats: Dict[str, float | int]) -> str:
    return f"[{float(stats['ci95_low']) * 100.0:.2f}, {float(stats['ci95_high']) * 100.0:.2f}]"


def table_rows(summary: Dict[str, object], families: Sequence[str]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for setting, raw_setting in sorted(summary.items()):
        setting_data = dict(raw_setting)
        n = str(setting_data.get("n", 0))
        for metric in METRICS:
            row = {"setting": setting, "metric": metric, "n": n}
            for family in families:
                if family not in setting_data:
                    continue
                row[family] = pct(dict(dict(setting_data[family])[metric]))
            if "delta" in setting_data:
                delta_stats = dict(dict(setting_data["delta"])[metric])
                row["delta"] = pct(delta_stats)
                row["delta_ci95"] = ci(delta_stats)
                row["positive_rate"] = f"{float(delta_stats['positive_rate']):.3f}"
                row["sign_test_p"] = f"{float(delta_stats['sign_test_p_value']):.4f}"
            rows.append(row)
    return rows


def write_csv(path: str, rows: Sequence[Dict[str, str]], fieldnames: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: str, rows: Sequence[Dict[str, str]], fieldnames: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = [
        "| " + " | ".join(fieldnames) + " |",
        "| " + " | ".join(["---"] * len(fieldnames)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row.get(field, "") for field in fieldnames) + " |")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--family", action="append", default=[])
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260710)
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as handle:
        payload = json.load(handle)
    records = list(payload.get("records", []))
    families = args.family or infer_families(records)
    summary = build_summary(records, families, args.bootstrap_iterations, args.seed)
    rows = table_rows(summary, families)

    fieldnames = ["setting", "metric", "n"] + list(families)
    if any("delta" in record for record in records):
        fieldnames += ["delta", "delta_ci95", "positive_rate", "sign_test_p"]

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "benchmark_with_ci.json"), "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "records": records, "families": families}, handle, indent=2, ensure_ascii=False)
    write_csv(os.path.join(args.output_dir, "paper_table.csv"), rows, fieldnames)
    write_markdown(os.path.join(args.output_dir, "paper_table.md"), rows, fieldnames)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
