"""Independent artifact verifier for Phase-D source-policy runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from pc_cng.paired_cluster_inference import paired_cluster_bootstrap
from pc_cng.run_phase4_fixed_testset import (
    cluster_bootstrap_metric,
    per_source_auprc,
    source_macro_auprc_metric,
)
from pc_cng.run_phase_d_source_policy import SOURCE_NAMES, _aggregate_exit


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_records(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row["label"] = int(float(row["label"]))
            row["score"] = float(row["score"])
            row["is_positive"] = str(row.get("is_positive", "")).lower() == "true"
            records.append(row)
    return records


def _record_identity(record: Dict[str, Any]) -> tuple:
    return (
        record.get("reaction_smiles"),
        record.get("label"),
        record.get("experimental_group"),
        record.get("source"),
        record.get("is_positive"),
    )


def _numeric_fields_close(
    rebuilt: Dict[str, Any],
    stored: Dict[str, Any],
    fields: Sequence[str],
    tolerance: float = 1e-12,
) -> bool:
    return all(
        key in rebuilt
        and key in stored
        and abs(float(rebuilt[key]) - float(stored[key])) <= tolerance
        for key in fields
    )


def verify_result_directory(result_dir: Path) -> Dict[str, Any]:
    result_dir = result_dir.resolve()
    result_path = result_dir / "phase_d_results.json"
    manifest_path = result_dir / "run_manifest.json"
    verdict_path = result_dir / "verdict.json"
    failures: List[str] = []
    checks: Dict[str, Any] = {}
    for path in (result_path, manifest_path, verdict_path):
        if not path.exists():
            failures.append(f"missing required artifact: {path}")
    if failures:
        return {"verified": False, "failures": failures, "checks": checks}

    result = json.loads(result_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    verdict = json.loads(verdict_path.read_text())
    checks["result_sha256_matches_manifest"] = (
        manifest.get("result_sha256") == _sha256(result_path)
    )
    if not checks["result_sha256_matches_manifest"]:
        failures.append("run_manifest result_sha256 mismatch")

    checkpoint = Path(manifest["phase_c_checkpoint"])
    checks["phase_c_checkpoint_exists"] = checkpoint.exists()
    checks["phase_c_checkpoint_hash_matches"] = (
        checkpoint.exists()
        and _sha256(checkpoint) == manifest["phase_c_checkpoint_sha256"]
    )
    if not checks["phase_c_checkpoint_hash_matches"]:
        failures.append("Phase-C checkpoint hash mismatch")

    expected_sources = list(SOURCE_NAMES)
    checks["source_schema_exact"] = manifest.get("source_names") == expected_sources
    if not checks["source_schema_exact"]:
        failures.append("source schema differs from frozen six-source contract")

    candidate_checks: Dict[str, Any] = {}
    for scenario, audit in result["scenario_audits"].items():
        cache_path = Path(audit["candidate_cache"])
        cache_ok = cache_path.exists() and (
            _sha256(cache_path) == audit["candidate_cache_sha256"]
        )
        entries = json.loads(cache_path.read_text()) if cache_path.exists() else []
        complete = bool(entries) and all(
            sorted(entry.get("candidates", {})) == sorted(expected_sources)
            for entry in entries
        )
        unique_budget = bool(entries) and all(
            len(entry.get("candidates", {})) == len(expected_sources)
            for entry in entries
        )
        expected_n = audit["candidate_generation"]["n_complete_parents"]
        count_ok = len(entries) == expected_n
        candidate_checks[scenario] = {
            "hash_ok": cache_ok,
            "all_six_sources_present": complete,
            "one_candidate_per_source": unique_budget,
            "count_ok": count_ok,
            "n_entries": len(entries),
        }
        if not all((cache_ok, complete, unique_budget, count_ok)):
            failures.append(f"candidate cache contract failed for {scenario}")
    checks["candidate_caches"] = candidate_checks

    result_checks: Dict[str, Any] = {}
    for cell in result["results"]:
        key = f"{cell['scenario']}::{cell['backbone']}"
        arm_checks: Dict[str, Any] = {}
        records_by_arm: Dict[str, List[Dict[str, Any]]] = {}
        reference_identity: Optional[List[tuple]] = None
        for arm, arm_result in cell["arms"].items():
            score_path = Path(arm_result["scored_records"])
            if not score_path.exists():
                failures.append(f"missing scored records: {score_path}")
                continue
            records = _read_records(score_path)
            records_by_arm[arm] = records
            metric = source_macro_auprc_metric(records)
            per_source = per_source_auprc(records)
            metric_ok = abs(metric - arm_result["source_macro_auprc"]) <= 1e-12
            per_source_ok = set(per_source) == set(arm_result["per_source_auprc"]) and all(
                abs(per_source[source] - arm_result["per_source_auprc"][source])
                <= 1e-12
                for source in per_source
            )
            identity = [_record_identity(record) for record in records]
            if reference_identity is None:
                reference_identity = identity
            alignment_ok = identity == reference_identity
            budget = arm_result["train_budget"]
            budget_ok = bool(budget["budget_exact"])
            if arm == "positive_only":
                count_ok = (
                    budget["n_positive"] == cell["n_complete_parents"]
                    and budget["n_negative"] == 0
                )
            else:
                count_ok = (
                    budget["n_positive"]
                    == budget["n_negative"]
                    == cell["n_complete_parents"]
                )
                selected_count = sum(
                    arm_result.get("selected_source_counts", {}).values()
                )
                count_ok = count_ok and selected_count == cell["n_complete_parents"]
            arm_checks[arm] = {
                "metric_rebuilt": metric_ok,
                "per_source_rebuilt": per_source_ok,
                "fixed_record_alignment": alignment_ok,
                "budget_exact": budget_ok,
                "count_ok": count_ok,
                "n_scored_records": len(records),
            }
            if not all((metric_ok, per_source_ok, alignment_ok, budget_ok, count_ok)):
                failures.append(f"arm verification failed: {key}/{arm}")

        gate_checks: Dict[str, Any] = {}
        for variant, gate_audit in cell["gate_training"].items():
            checkpoint_path = Path(gate_audit["checkpoint"])
            checkpoint_ok = checkpoint_path.exists() and (
                _sha256(checkpoint_path) == gate_audit["checkpoint_sha256"]
            )
            selection_ok = (
                sum(gate_audit["selected_source_counts"].values())
                == cell["n_complete_parents"]
            )
            gate_checks[variant] = {
                "checkpoint_hash_ok": checkpoint_ok,
                "selection_count_ok": selection_ok,
            }
            if not checkpoint_ok or not selection_ok:
                failures.append(f"gate artifact failed: {key}/{variant}")
        seed = int(cell["seed"])
        n_bootstrap = int(cell["n_bootstrap"])
        best_source = cell["validation_selected_best_single"]
        rebuilt_inference = {
            "gate_vs_validation_selected_best_single": paired_cluster_bootstrap(
                records_by_arm["learned_source_gate"],
                records_by_arm[best_source],
                source_macro_auprc_metric,
                n_bootstrap=n_bootstrap,
                seed=seed + 2001,
            ),
            "gate_vs_uniform_union": paired_cluster_bootstrap(
                records_by_arm["learned_source_gate"],
                records_by_arm["uniform_union"],
                source_macro_auprc_metric,
                n_bootstrap=n_bootstrap,
                seed=seed + 2002,
            ),
            "gate_vs_global_mixture": paired_cluster_bootstrap(
                records_by_arm["learned_source_gate"],
                records_by_arm["validation_selected_global_mixture"],
                source_macro_auprc_metric,
                n_bootstrap=n_bootstrap,
                seed=seed + 2003,
            ),
        }
        if "gate_no_learned_source" in records_by_arm:
            rebuilt_inference["gate_vs_no_learned_source"] = (
                paired_cluster_bootstrap(
                    records_by_arm["learned_source_gate"],
                    records_by_arm["gate_no_learned_source"],
                    source_macro_auprc_metric,
                    n_bootstrap=n_bootstrap,
                    seed=seed + 2004,
                )
            )
        rebuilt_null = cluster_bootstrap_metric(
            records_by_arm["randomized_label_null"],
            metric_fn=source_macro_auprc_metric,
            n_bootstrap=n_bootstrap,
            seed=seed + 2005,
        )
        inference_checks: Dict[str, bool] = {}
        delta_fields = (
            "challenger_point",
            "baseline_point",
            "delta_mean",
            "delta_ci_low",
            "delta_ci_high",
            "p_value",
        )
        for comparison, rebuilt in rebuilt_inference.items():
            inference_checks[comparison] = _numeric_fields_close(
                rebuilt,
                cell["inference"][comparison],
                delta_fields,
            )
        inference_checks["randomized_label_null"] = _numeric_fields_close(
            rebuilt_null,
            cell["inference"]["randomized_label_null"],
            ("point", "ci_low", "ci_high"),
        )
        if not all(inference_checks.values()):
            failures.append(f"paired inference rebuild failed: {key}")
        result_checks[key] = {
            "arms": arm_checks,
            "gates": gate_checks,
            "paired_inference_rebuilt": inference_checks,
        }
    checks["result_cells"] = result_checks

    formal_run = manifest.get("mode") == "formal"
    rebuilt_verdict = _aggregate_exit(result["results"], formal_run=formal_run)
    checks["verdict_rebuilt"] = rebuilt_verdict == verdict == result["exit_status"]
    if not checks["verdict_rebuilt"]:
        failures.append("verdict does not match independent aggregation")

    if formal_run:
        formal_manifest = manifest.get("formal_manifest") or {}
        formal_contract_ok = (
            formal_manifest.get("status") == "SEALED_UNUSED_FOR_METHOD_DESIGN"
            and formal_manifest.get("labels_unseen_before_model_freeze") is True
            and formal_manifest.get("model_and_analysis_frozen") is True
        )
    else:
        formal_contract_ok = (
            manifest.get("evaluation_status")
            == "DEVELOPMENT_SET_USED_FOR_METHOD_DESIGN"
            and verdict.get("confirmatory_exit_met") is False
        )
    checks["evidence_status_contract"] = formal_contract_ok
    if not formal_contract_ok:
        failures.append("development/formal evidence-status contract failed")
    return {
        "verified": not failures,
        "result_directory": str(result_dir),
        "failures": failures,
        "checks": checks,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = verify_result_directory(args.result_dir)
    output = args.output or (args.result_dir / "independent_verification.json")
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if report["verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
