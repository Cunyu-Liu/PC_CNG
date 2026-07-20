"""Repair an external train/val context pool by source_id replacement.

The script removes explicitly invalid observed-positive contexts and appends
pre-selected replacement contexts. It writes a new repaired context CSV plus
removed/added ledgers and a JSON/Markdown summary. It does not mutate the
original context pool.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from typing import Dict, List, Sequence


def read_csv(path: str) -> tuple[List[str], List[Dict[str, str]]]:
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), [dict(row) for row in reader]


def write_csv(path: str, fields: Sequence[str], rows: Sequence[Dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_by(rows: Sequence[Dict[str, str]], field: str) -> Dict[str, int]:
    return dict(sorted(Counter((row.get(field) or "unknown") for row in rows).items()))


def duplicate_values(rows: Sequence[Dict[str, str]], field: str) -> List[str]:
    counts = Counter(row.get(field, "") for row in rows)
    return sorted(value for value, count in counts.items() if value and count > 1)


def write_summary_md(path: str, summary: Dict[str, object]) -> None:
    lines = [
        "# External Train/Val Context Pool Repair",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Original rows | `{summary['original_context_rows']}` |",
        f"| Removed rows | `{summary['removed_context_rows']}` |",
        f"| Added rows | `{summary['added_context_rows']}` |",
        f"| Final rows | `{summary['context_rows']}` |",
        f"| Split counts | `{summary['split_counts']}` |",
        f"| Dataset counts | `{summary['dataset_counts']}` |",
        f"| Removed source IDs | `{', '.join(summary['removed_source_ids'])}` |",
        f"| Added source IDs | `{', '.join(summary['added_source_ids'])}` |",
        f"| Duplicate source IDs | `{summary['duplicate_source_ids']}` |",
        f"| Duplicate group IDs | `{summary['duplicate_group_ids']}` |",
        f"| Decision | `{summary['decision']}` |",
        "",
        "## Outputs",
        "",
        "| Artifact | SHA256 |",
        "|---|---|",
    ]
    for artifact, digest in dict(summary.get("sha256", {})).items():
        lines.append(f"| `{artifact}` | `{digest}` |")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def repair_context_pool(
    input_contexts_csv: str,
    replacement_contexts_csv: str,
    bad_source_ids: Sequence[str],
    output_dir: str,
    output_name: str,
) -> Dict[str, object]:
    fields, original_rows = read_csv(input_contexts_csv)
    replacement_fields, replacement_rows = read_csv(replacement_contexts_csv)
    if fields != replacement_fields:
        raise ValueError(f"CSV field mismatch: input={fields}, replacement={replacement_fields}")

    bad_ids = set(bad_source_ids)
    removed_rows = [row for row in original_rows if row.get("source_id") in bad_ids]
    kept_rows = [row for row in original_rows if row.get("source_id") not in bad_ids]
    removed_ids = {row.get("source_id") for row in removed_rows}
    if removed_ids != bad_ids:
        missing = sorted(bad_ids - removed_ids)
        raise ValueError(f"Bad source ids missing from input contexts: {missing}")
    if len(replacement_rows) != len(removed_rows):
        raise ValueError(
            f"Replacement count mismatch: removed={len(removed_rows)}, replacements={len(replacement_rows)}"
        )
    replacement_bad_ids = sorted(row.get("source_id", "") for row in replacement_rows if row.get("source_id") in bad_ids)
    if replacement_bad_ids:
        raise ValueError(f"Replacement rows include bad source ids: {replacement_bad_ids}")

    repaired_rows = [dict(row) for row in kept_rows + replacement_rows]
    for index, row in enumerate(repaired_rows):
        row["row_index"] = str(index)

    duplicate_sources = duplicate_values(repaired_rows, "source_id")
    duplicate_groups = duplicate_values(repaired_rows, "group_id")
    if duplicate_sources or duplicate_groups:
        raise ValueError(
            f"Duplicates after repair: source_ids={duplicate_sources[:10]}, group_ids={duplicate_groups[:10]}"
        )

    os.makedirs(output_dir, exist_ok=True)
    output_contexts_csv = os.path.join(output_dir, f"{output_name}.csv")
    removed_contexts_csv = os.path.join(output_dir, "removed_invalid_observed_contexts.csv")
    added_contexts_csv = os.path.join(output_dir, "added_replacement_contexts.csv")
    summary_json = os.path.join(output_dir, f"{output_name}_summary.json")
    summary_md = os.path.join(output_dir, f"{output_name}_summary.md")

    write_csv(output_contexts_csv, fields, repaired_rows)
    write_csv(removed_contexts_csv, fields, removed_rows)
    write_csv(added_contexts_csv, fields, replacement_rows)

    summary: Dict[str, object] = {
        "task": "external_trainval_context_pool_repair",
        "decision": "contexts_repaired_ready_for_base_candidate_rebuild",
        "input_contexts_csv": input_contexts_csv,
        "replacement_contexts_csv": replacement_contexts_csv,
        "output_contexts_csv": output_contexts_csv,
        "removed_invalid_observed_contexts_csv": removed_contexts_csv,
        "added_replacement_contexts_csv": added_contexts_csv,
        "summary_json": summary_json,
        "summary_md": summary_md,
        "original_context_rows": len(original_rows),
        "removed_context_rows": len(removed_rows),
        "added_context_rows": len(replacement_rows),
        "context_rows": len(repaired_rows),
        "removed_source_ids": sorted(bad_ids),
        "added_source_ids": [row.get("source_id", "") for row in replacement_rows],
        "split_counts": count_by(repaired_rows, "split"),
        "dataset_counts": count_by(repaired_rows, "dataset"),
        "duplicate_source_ids": duplicate_sources,
        "duplicate_group_ids": duplicate_groups,
    }
    summary["sha256"] = {
        output_contexts_csv: sha256(output_contexts_csv),
        removed_contexts_csv: sha256(removed_contexts_csv),
        added_contexts_csv: sha256(added_contexts_csv),
    }
    with open(summary_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    summary["sha256"][summary_json] = sha256(summary_json)  # type: ignore[index]
    write_summary_md(summary_md, summary)
    summary["sha256"][summary_md] = sha256(summary_md)  # type: ignore[index]
    with open(summary_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-contexts-csv", required=True)
    parser.add_argument("--replacement-contexts-csv", required=True)
    parser.add_argument("--bad-source-id", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-name", default="external_calibration_trainval_contexts_12k_repaired")
    args = parser.parse_args()

    summary = repair_context_pool(
        input_contexts_csv=args.input_contexts_csv,
        replacement_contexts_csv=args.replacement_contexts_csv,
        bad_source_ids=args.bad_source_id,
        output_dir=args.output_dir,
        output_name=args.output_name,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
