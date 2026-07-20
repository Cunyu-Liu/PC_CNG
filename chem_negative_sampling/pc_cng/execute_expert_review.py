"""P2-03 Expert Review Execution Protocol (PC-CNG).

Builds the expert review infrastructure for the PC-CNG project. This script
prepares reviewer forms, parses filled ratings, and computes inter-annotator
agreement (Cohen's kappa for 2 reviewers, Fleiss' kappa for 3+ reviewers).

Per spec P2-03:
- "需用户协助招募 2-3 名化学专家（trae 无法自行执行）"
- "若用户无法提供专家 → 明确标注'expert review protocol specified,
  execution deferred to revision'，论文 limitation 保留"

The script must be BUILT (so infrastructure is ready), but execution is
DEFERRED to revision (requires human chemistry experts).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from sklearn.metrics import cohen_kappa_score
    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    _HAS_SKLEARN = False


# ---------- Constants ----------

LIKERT_COLUMNS: List[str] = [
    "chemical_validity",
    "mechanistic_plausibility",
    "side_product_likelihood",
    "feasibility",
    "overall_verdict",
]

FORM_COLUMNS: List[str] = [
    "sample_id",
    "reactants",
    "products",
    "candidate_reaction",
    "parent_reaction_smiles",
    "failure_type",
    "task",
    "source_origin",
    "true_label",
] + LIKERT_COLUMNS + ["comment", "reviewer_id", "review_timestamp"]

DEFERRED_STATUS: str = (
    "expert review protocol specified, execution deferred to revision"
)

GO_NO_GO_AGREEMENT_THRESHOLD: float = 0.60
GO_NO_GO_PASS_RATE_THRESHOLD: float = 0.70
LIKERT_PASS_THRESHOLD: int = 4


# ---------- I/O helpers ----------

def load_samples(path: Path) -> List[Dict[str, str]]:
    """Load a CSV file (sampled_for_review or filled reviewer form) as list of dicts."""
    with open(path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return list(reader)


def split_reaction_smiles(reaction: str) -> Tuple[str, str]:
    """Split a reaction SMILES into (reactants, products).

    Handles '>>' (irreversible) and '>' (reversible) separators. If the
    input contains no separator, returns (reaction, "").
    """
    if not reaction:
        return "", ""
    if ">>" in reaction:
        left, right = reaction.split(">>", 1)
        return left, right
    if ">" in reaction:
        parts = reaction.split(">")
        if len(parts) >= 2:
            return parts[0], parts[-1]
    return reaction, ""


def build_reviewer_form(rows: List[Dict[str, str]], reviewer_id: int) -> List[Dict[str, str]]:
    """Build a blank reviewer form from sampled rows."""
    form_rows: List[Dict[str, str]] = []
    for row in rows:
        candidate = row.get("reaction_smiles", "")
        reactants, products = split_reaction_smiles(candidate)
        form_row: Dict[str, str] = {
            "sample_id": row.get("sample_id", ""),
            "reactants": reactants,
            "products": products,
            "candidate_reaction": candidate,
            "parent_reaction_smiles": row.get("parent_reaction_smiles", ""),
            "failure_type": row.get("failure_type", ""),
            "task": row.get("task", ""),
            "source_origin": row.get("source_origin", ""),
            "true_label": row.get("true_label", ""),
        }
        for col in LIKERT_COLUMNS:
            form_row[col] = ""
        form_row["comment"] = ""
        form_row["reviewer_id"] = f"reviewer_{reviewer_id}"
        form_row["review_timestamp"] = ""
        form_rows.append(form_row)
    return form_rows


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    """Write rows to a CSV file with the given fieldnames."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


# ---------- Likert parsing ----------

def parse_likert(value: object) -> Optional[int]:
    """Parse a Likert rating from a value. Returns int 1-5 or None.

    Rejects non-integer floats (e.g. "3.5") since Likert scales are integer-only.
    Accepts "3", "3.0", "  3  ", 3 (int).
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        f = float(s)
    except (ValueError, TypeError):
        return None
    # Reject non-integer floats (e.g. 3.5) — Likert is integer-only
    if f != int(f):
        return None
    n = int(f)
    if 1 <= n <= 5:
        return n
    return None


def parse_form_ratings(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    """Parse ratings from a filled reviewer form.

    Returns a list of dicts with parsed integer Likert ratings (or None for blanks).
    """
    parsed: List[Dict[str, object]] = []
    for row in rows:
        out: Dict[str, object] = {
            "sample_id": row.get("sample_id", ""),
            "reviewer_id": row.get("reviewer_id", ""),
        }
        for col in LIKERT_COLUMNS:
            out[col] = parse_likert(row.get(col, ""))
        out["comment"] = row.get("comment", "")
        parsed.append(out)
    return parsed


# ---------- Agreement metrics ----------

def cohen_kappa(labels_a: Sequence[Optional[int]], labels_b: Sequence[Optional[int]]) -> float:
    """Compute Cohen's kappa for two raters.

    Uses sklearn when available, otherwise falls back to a manual implementation.
    Returns float in [-1, 1]; returns 0.0 if undefined (e.g. empty input or
    when inputs have no overlap).
    """
    a = [int(x) for x in labels_a if x is not None]
    b = [int(x) for x in labels_b if x is not None]
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    a = a[:n]
    b = b[:n]
    if _HAS_SKLEARN:
        try:
            kappa = cohen_kappa_score(a, b)
            kappa = float(kappa)
            if np.isnan(kappa):
                # sklearn returns NaN when only one label is present (pe == 1.0);
                # fall back to manual implementation which returns 1.0 in this case.
                return _cohen_kappa_manual(a, b)
            return kappa
        except Exception:
            pass
    return _cohen_kappa_manual(a, b)


def _cohen_kappa_manual(a: List[int], b: List[int]) -> float:
    """Manual Cohen's kappa implementation (numpy-based)."""
    n = len(a)
    if n == 0:
        return 0.0
    labels = sorted(set(a) | set(b))
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    k = len(labels)
    matrix = np.zeros((k, k), dtype=float)
    for ai, bi in zip(a, b):
        matrix[label_to_idx[ai], label_to_idx[bi]] += 1.0
    po = float(np.trace(matrix) / n)
    row_marg = matrix.sum(axis=1) / n
    col_marg = matrix.sum(axis=0) / n
    pe = float(np.dot(row_marg, col_marg))
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


def fleiss_kappa(ratings: Sequence[Sequence[Optional[int]]], n_categories: int = 5) -> float:
    """Compute Fleiss' kappa for 3+ raters.

    Args:
        ratings: list of length N (samples); each entry is a sequence of integer
                 ratings from each rater for that sample. None values are skipped.
        n_categories: number of distinct rating categories (Likert 1..5 default).

    Returns:
        float kappa in [-1, 1]; 0.0 if undefined.
    """
    N = len(ratings)
    if N == 0:
        return 0.0
    # Filter None values per sample; require >= 2 ratings per sample
    cleaned: List[List[int]] = []
    for sample_ratings in ratings:
        sample_clean = [int(r) for r in sample_ratings if r is not None]
        if len(sample_clean) >= 2:
            cleaned.append(sample_clean)
    if not cleaned:
        return 0.0
    # Use the minimum number of raters across samples for consistency
    n = min(len(s) for s in cleaned)
    if n < 2:
        return 0.0
    # Truncate each sample to n raters
    cleaned = [s[:n] for s in cleaned]
    N = len(cleaned)
    matrix = np.zeros((N, n_categories), dtype=float)
    for i, sample_ratings in enumerate(cleaned):
        for r in sample_ratings:
            idx = int(r) - 1  # ratings are 1..n_categories -> 0..n_categories-1
            if 0 <= idx < n_categories:
                matrix[i, idx] += 1.0
    total_ratings = float(N * n)
    if total_ratings == 0:
        return 0.0
    p_j = matrix.sum(axis=0) / total_ratings
    # P_i = (sum_j n_ij^2 - n) / (n * (n - 1))
    P_i = (np.sum(matrix ** 2, axis=1) - n) / (n * (n - 1))
    P_bar = float(np.mean(P_i))
    Pe = float(np.sum(p_j ** 2))
    if Pe == 1.0:
        return 1.0
    return (P_bar - Pe) / (1.0 - Pe)


def compute_pairwise_agreement(
    parsed_per_reviewer: List[List[Dict[str, object]]],
    column: str = "overall_verdict",
) -> Dict[str, object]:
    """Compute inter-annotator agreement between reviewers for a given column.

    For 2 reviewers: returns Cohen's kappa.
    For 3+ reviewers: returns Fleiss' kappa plus pairwise Cohen's kappa matrix.
    """
    n_reviewers = len(parsed_per_reviewer)
    if n_reviewers < 2:
        return {"metric": "none", "values": {}, "reason": "need >=2 reviewers"}
    # Index by sample_id
    by_reviewer: List[Dict[str, Optional[int]]] = []
    for parsed in parsed_per_reviewer:
        idx = {row["sample_id"]: row.get(column) for row in parsed}  # type: ignore[arg-type]
        by_reviewer.append(idx)
    # Find common sample_ids with non-None ratings across all reviewers
    common_ids: Optional[set] = None
    for idx in by_reviewer:
        ids = {sid for sid, v in idx.items() if v is not None}
        common_ids = ids if common_ids is None else (common_ids & ids)
    common_sorted = sorted(common_ids) if common_ids else []
    if len(common_sorted) == 0:
        return {
            "metric": "cohen_kappa" if n_reviewers == 2 else "fleiss_kappa",
            "value": 0.0,
            "n_common_samples": 0,
            "column": column,
            "threshold_substantial_agreement": GO_NO_GO_AGREEMENT_THRESHOLD,
            "passes_threshold": False,
            "reason": "no common samples with non-None ratings",
        }
    if n_reviewers == 2:
        a = [by_reviewer[0][sid] for sid in common_sorted]
        b = [by_reviewer[1][sid] for sid in common_sorted]
        kappa = cohen_kappa(a, b)
        return {
            "metric": "cohen_kappa",
            "value": kappa,
            "n_common_samples": len(common_sorted),
            "column": column,
            "threshold_substantial_agreement": GO_NO_GO_AGREEMENT_THRESHOLD,
            "passes_threshold": bool(kappa >= GO_NO_GO_AGREEMENT_THRESHOLD),
        }
    # 3+ reviewers: pairwise Cohen's kappa matrix + Fleiss' kappa
    pairwise: Dict[str, float] = {}
    for i in range(n_reviewers):
        for j in range(i + 1, n_reviewers):
            a = [by_reviewer[i][sid] for sid in common_sorted]
            b = [by_reviewer[j][sid] for sid in common_sorted]
            pairwise[f"reviewer_{i+1}_vs_reviewer_{j+1}"] = cohen_kappa(a, b)
    fleiss_ratings: List[List[Optional[int]]] = []
    for sid in common_sorted:
        sample_r = [by_reviewer[i][sid] for i in range(n_reviewers)]
        fleiss_ratings.append(sample_r)
    fleiss = fleiss_kappa(fleiss_ratings, n_categories=5)
    return {
        "metric": "fleiss_kappa",
        "value": fleiss,
        "pairwise_cohen_kappa": pairwise,
        "n_common_samples": len(common_sorted),
        "column": column,
        "threshold_substantial_agreement": GO_NO_GO_AGREEMENT_THRESHOLD,
        "passes_threshold": bool(fleiss >= GO_NO_GO_AGREEMENT_THRESHOLD),
    }


# ---------- Pass rate & failure modes ----------

def compute_pass_rate(
    parsed_per_reviewer: List[List[Dict[str, object]]],
    threshold: int = LIKERT_PASS_THRESHOLD,
) -> Dict[str, object]:
    """Compute reviewer pass rate (Likert >= threshold on overall_verdict).

    Per-sample majority pass: a sample is "passed" if at least ceil(n/2)
    reviewers give overall_verdict >= threshold.
    """
    per_reviewer: List[Dict[str, object]] = []
    for parsed in parsed_per_reviewer:
        n_total = 0
        n_pass = 0
        for row in parsed:
            v = row.get("overall_verdict")
            if v is None:
                continue
            n_total += 1
            if int(v) >= threshold:  # type: ignore[arg-type]
                n_pass += 1
        rate = (n_pass / n_total) if n_total > 0 else 0.0
        per_reviewer.append({
            "reviewer_id": (parsed[0]["reviewer_id"] if parsed else ""),
            "n_total": n_total,
            "n_pass": n_pass,
            "pass_rate": rate,
        })
    # Per-sample majority pass
    sample_ratings: Dict[str, List[bool]] = {}
    for parsed in parsed_per_reviewer:
        for row in parsed:
            sid = row["sample_id"]
            v = row.get("overall_verdict")
            if v is None:
                continue
            sample_ratings.setdefault(sid, []).append(int(v) >= threshold)  # type: ignore[arg-type]
    n_reviewers = len(parsed_per_reviewer)
    n_majority = (n_reviewers + 1) // 2  # ceil(n/2)
    sample_pass = 0
    sample_total = 0
    for sid, vals in sample_ratings.items():
        if len(vals) >= 1:
            sample_total += 1
            if sum(1 for v in vals if v) >= n_majority:
                sample_pass += 1
    sample_rate = (sample_pass / sample_total) if sample_total > 0 else 0.0
    return {
        "per_reviewer": per_reviewer,
        "per_sample_majority": {
            "n_total": sample_total,
            "n_pass": sample_pass,
            "pass_rate": sample_rate,
        },
        "threshold": threshold,
        "go_no_go_threshold": GO_NO_GO_PASS_RATE_THRESHOLD,
        "passes_go_no_go": bool(sample_rate >= GO_NO_GO_PASS_RATE_THRESHOLD),
    }


def compute_failure_mode_distribution(
    parsed_per_reviewer: List[List[Dict[str, object]]],
    sample_rows: List[Dict[str, str]],
) -> Dict[str, object]:
    """For samples with overall_verdict < threshold (majority), report failure_type distribution."""
    failure_type_by_sid = {
        row.get("sample_id", ""): row.get("failure_type", "") for row in sample_rows
    }
    sample_ratings: Dict[str, List[int]] = {}
    for parsed in parsed_per_reviewer:
        for row in parsed:
            sid = row["sample_id"]
            v = row.get("overall_verdict")
            if v is None:
                continue
            sample_ratings.setdefault(sid, []).append(int(v))  # type: ignore[arg-type]
    n_reviewers = len(parsed_per_reviewer)
    n_majority = (n_reviewers + 1) // 2
    fail_counts: Dict[str, int] = {}
    total_fails = 0
    for sid, vals in sample_ratings.items():
        if len(vals) < 1:
            continue
        n_fail = sum(1 for v in vals if v < LIKERT_PASS_THRESHOLD)
        is_fail = n_fail >= n_majority
        if is_fail:
            total_fails += 1
            ftype = failure_type_by_sid.get(sid, "unknown")
            fail_counts[ftype] = fail_counts.get(ftype, 0) + 1
    return {
        "n_fail_samples": total_fails,
        "failure_type_counts": fail_counts,
        "failure_type_distribution": {
            k: (v / total_fails if total_fails > 0 else 0.0)
            for k, v in fail_counts.items()
        },
    }


# ---------- Prepare mode ----------

def write_protocol_md(path: Path, reviewer_count: int) -> None:
    """Write the reviewer protocol/instructions document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = f"""# Expert Review Protocol (P2-03)

## Status

{DEFERRED_STATUS}

## Reviewer Count

{reviewer_count} reviewer(s) required (chemistry experts).

## Reviewer Instructions

For each sample row in your reviewer form CSV:

1. **Inspect** the `candidate_reaction` (the reaction SMILES to evaluate).
   Use `parent_reaction_smiles` only as reference context — do **not**
   rate the parent; rate the candidate.
2. **Rate** each of the 5 Likert columns from 1 (lowest) to 5 (highest):
   - `chemical_validity`: Are reactants and products chemically valid SMILES
     with sensible valence/atom types?
   - `mechanistic_plausibility`: Is the transformation mechanistically
     plausible given standard organic chemistry?
   - `side_product_likelihood`: How likely is the displayed product to be the
     major product (5) versus a side product (1)?
   - `feasibility`: Would this reaction proceed under reasonable conditions
     in a wet-lab setting?
   - `overall_verdict`: Overall expert judgment on whether this candidate
     is a *valid* reaction (5) or an *invalid/negative* one (1).
3. **Comment** (optional): add free-form notes in the `comment` column.
4. **Fill** the `review_timestamp` field with an ISO-8601 timestamp when
   you complete the review (e.g. `2026-07-20T15:30:00`).
5. **Do not** modify `sample_id`, `reactants`, `products`,
   `candidate_reaction`, `parent_reaction_smiles`, `failure_type`, `task`,
   `source_origin`, `true_label`, or `reviewer_id` columns.

## Go/No-Go Thresholds

- Inter-annotator agreement (Cohen's kappa or Fleiss' kappa) >= 0.60
- Reviewer pass rate (overall_verdict >= 4) >= 70%

## Output Files

After all reviewer forms are filled and uploaded, run the script in
`--mode aggregate` with `--filled-forms-dir` pointing to the directory
containing the filled reviewer form CSVs. The script will produce:

- `reviewer_ratings_raw.csv`: long-format table of all ratings
- `inter_annotator_agreement.json`: kappa statistics
- `expert_review_summary.json`: pass rate, failure mode distribution,
  per-sample agreement

## Deferral Note

Per spec P2-03, the protocol is fully specified and the forms are
generated, but **execution requires human chemistry experts** which the
automated agent cannot recruit. Execution is therefore deferred to the
revision phase. The paper limitation is preserved.
"""
    path.write_text(content, encoding="utf-8")


def write_deferred_status(path: Path, reviewer_count: int, output_dir: Path) -> None:
    """Write the deferred_status.json file."""
    payload = {
        "task_id": "P2-03",
        "title": "Expert Review Execution",
        "status": "deferred_to_revision",
        "status_message": DEFERRED_STATUS,
        "spec_note_zh_1": "需用户协助招募 2-3 名化学专家（trae 无法自行执行）",
        "spec_note_zh_2": "若用户无法提供专家 → 明确标注'expert review protocol specified, execution deferred to revision'，论文 limitation 保留",
        "protocol_built": True,
        "execution_completed": False,
        "reviewer_count_requested": reviewer_count,
        "reviewer_count_filled": 0,
        "output_dir": str(output_dir),
        "prepared_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        "deferred_reason": (
            "Automated agent cannot recruit human chemistry experts. "
            "Protocol and reviewer forms are generated and ready; "
            "human execution is required during the revision phase."
        ),
        "go_no_go_thresholds": {
            "inter_annotator_agreement_kappa": GO_NO_GO_AGREEMENT_THRESHOLD,
            "reviewer_pass_rate": GO_NO_GO_PASS_RATE_THRESHOLD,
        },
        "limitation_preserved_in_paper": True,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def run_prepare(samples_path: Path, reviewer_count: int, output_dir: Path) -> Dict[str, object]:
    """Prepare reviewer forms, protocol.md, and deferred_status.json."""
    if reviewer_count < 2 or reviewer_count > 3:
        raise ValueError(f"reviewer_count must be 2 or 3, got {reviewer_count}")
    if not samples_path.exists():
        raise FileNotFoundError(f"samples file not found: {samples_path}")
    rows = load_samples(samples_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    forms_dir = output_dir / "reviewer_forms"
    forms_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    for r in range(1, reviewer_count + 1):
        form_rows = build_reviewer_form(rows, r)
        form_path = forms_dir / f"reviewer_{r}_form.csv"
        write_csv(form_path, form_rows, FORM_COLUMNS)
        written.append(str(form_path))
    protocol_path = output_dir / "protocol.md"
    write_protocol_md(protocol_path, reviewer_count)
    deferred_path = output_dir / "deferred_status.json"
    write_deferred_status(deferred_path, reviewer_count, output_dir)
    return {
        "mode": "prepare",
        "n_samples": len(rows),
        "reviewer_count": reviewer_count,
        "reviewer_forms": written,
        "protocol_md": str(protocol_path),
        "deferred_status_json": str(deferred_path),
        "status": "deferred_to_revision",
        "status_message": DEFERRED_STATUS,
    }


# ---------- Aggregate mode ----------

def run_aggregate(
    filled_forms_dir: Path,
    output_dir: Path,
    samples_path: Optional[Path] = None,
) -> Dict[str, object]:
    """Aggregate filled reviewer forms and compute agreement + summary."""
    if not filled_forms_dir.exists():
        raise FileNotFoundError(f"filled-forms-dir not found: {filled_forms_dir}")
    form_paths = sorted(filled_forms_dir.glob("reviewer_*_form.csv"))
    if len(form_paths) < 2:
        raise ValueError(
            f"need >=2 filled reviewer forms, found {len(form_paths)} "
            f"in {filled_forms_dir}"
        )
    parsed_per_reviewer: List[List[Dict[str, object]]] = []
    raw_rows: List[Dict[str, object]] = []
    for fp in form_paths:
        rows = load_samples(fp)
        parsed = parse_form_ratings(rows)
        parsed_per_reviewer.append(parsed)
        for r in parsed:
            r_out: Dict[str, object] = dict(r)
            r_out["form_path"] = str(fp)
            raw_rows.append(r_out)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Write raw ratings CSV
    raw_path = output_dir / "reviewer_ratings_raw.csv"
    raw_fields = ["sample_id", "reviewer_id"] + LIKERT_COLUMNS + ["comment", "form_path"]
    write_csv(raw_path, raw_rows, raw_fields)
    # Inter-annotator agreement
    agreement = compute_pairwise_agreement(parsed_per_reviewer, column="overall_verdict")
    agreement_path = output_dir / "inter_annotator_agreement.json"
    with open(agreement_path, "w", encoding="utf-8") as fh:
        json.dump(agreement, fh, indent=2, ensure_ascii=False)
    # Pass rate
    pass_rate = compute_pass_rate(parsed_per_reviewer, threshold=LIKERT_PASS_THRESHOLD)
    # Failure mode distribution (needs sample rows for failure_type lookup)
    failure_dist: Dict[str, object] = {}
    if samples_path and samples_path.exists():
        sample_rows = load_samples(samples_path)
        failure_dist = compute_failure_mode_distribution(parsed_per_reviewer, sample_rows)
    summary: Dict[str, object] = {
        "n_reviewers": len(parsed_per_reviewer),
        "n_samples_per_reviewer": [len(p) for p in parsed_per_reviewer],
        "agreement": agreement,
        "pass_rate": pass_rate,
        "failure_mode_distribution": failure_dist,
        "go_no_go": {
            "agreement_threshold": GO_NO_GO_AGREEMENT_THRESHOLD,
            "pass_rate_threshold": GO_NO_GO_PASS_RATE_THRESHOLD,
            "agreement_passes": bool(agreement.get("passes_threshold", False)),
            "pass_rate_passes": bool(pass_rate.get("passes_go_no_go", False)),
            "overall_go": bool(
                agreement.get("passes_threshold", False)
                and pass_rate.get("passes_go_no_go", False)
            ),
        },
    }
    summary_path = output_dir / "expert_review_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    return {
        "mode": "aggregate",
        "reviewer_ratings_raw": str(raw_path),
        "inter_annotator_agreement": str(agreement_path),
        "expert_review_summary": str(summary_path),
        "summary": summary,
    }


# ---------- CLI ----------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pc_cng.execute_expert_review",
        description="P2-03 Expert Review Execution Protocol (PC-CNG).",
    )
    p.add_argument(
        "--samples",
        type=Path,
        default=Path("results/expert_review_20260719/sampled_for_review.csv"),
        help="Path to sampled_for_review.csv (default: results/expert_review_20260719/sampled_for_review.csv)",
    )
    p.add_argument(
        "--reviewer-count",
        type=int,
        default=2,
        help="Number of reviewers (2 or 3, default 2)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for forms and results",
    )
    p.add_argument(
        "--mode",
        type=str,
        default="prepare",
        choices=["prepare", "aggregate"],
        help="Mode: 'prepare' generates blank forms; 'aggregate' parses filled forms",
    )
    p.add_argument(
        "--filled-forms-dir",
        type=Path,
        default=None,
        help="Path to filled reviewer forms (required for 'aggregate' mode)",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.reviewer_count < 2 or args.reviewer_count > 3:
        parser.error(f"--reviewer-count must be 2 or 3, got {args.reviewer_count}")
    if args.mode == "prepare":
        result = run_prepare(args.samples, args.reviewer_count, args.output_dir)
    elif args.mode == "aggregate":
        if args.filled_forms_dir is None:
            parser.error("--filled-forms-dir is required for --mode aggregate")
        result = run_aggregate(args.filled_forms_dir, args.output_dir, args.samples)
    else:  # pragma: no cover - argparse choices prevents this
        parser.error(f"unknown mode: {args.mode}")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
