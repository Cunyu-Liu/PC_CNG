#!/usr/bin/env python3
"""GPU-only formal runner for the G6 v3 reaction-conditioned benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import numpy as np
import torch

from pc_cng.p4_g6_benchmark_v3 import (
    FORMAL_SCHEMA_VERSION,
    ConditionNormalizer,
    FormalAnalysisPlan,
    SharedPretrainedReactionEncoder,
    apply_temperature,
    evaluate_secondary_metrics,
    fit_temperature_scaler,
    load_matched_source_arms,
    metric_records,
    partition_context_complete_records,
    predict_multitask,
    train_multitask_heads,
    validate_cluster_contract,
    validate_formal_analysis_plan,
    validate_reaction_condition_records,
)
from pc_cng.p4_g6_inference_v3 import run_preregistered_primary_inference


REPO_ROOT = Path(__file__).resolve().parents[2]


def _joined(parts: list[Any]) -> str:
    return ".".join(str(x).strip() for x in parts if x not in (None, "") and str(x).strip())


def load_hte_records(parquet_path: Path) -> list[dict[str, Any]]:
    """Load HTE preserving separately auditable condition fields."""
    raw = pq.read_table(str(parquet_path)).to_pylist()
    records: list[dict[str, Any]] = []
    for row in raw:
        reagent_values = [row.get(key, "") for key in ("reagent_1_smiles", "reagent_2_smiles", "reagent_smiles")]
        record = {
            "record_id": str(row.get("record_id", "")),
            "reactants": _joined([row.get("reactant_1_smiles", ""), row.get("reactant_2_smiles", "")]),
            "catalysts": _joined([row.get("catalyst_1_smiles", ""), row.get("catalyst_2_smiles", "")]),
            "solvents": _joined([row.get("solvent", "")]),
            "reagents": _joined(reagent_values),
            "products": str(row.get("products", "") or ""),
            "temperature": row.get("temperature"),
            "reaction_time_hrs": row.get("reaction_time_hrs"),
            "measured_yield": float(row.get("measured_yield", 0.0) or 0.0),
            "split": str(row.get("split", "")),
            "experimental_group": str(row.get("experimental_group", "")),
            "plate_id": str(row.get("plate_id", "")),
            "reaction_family": str(row.get("reaction_family", "")),
            "source_publication": str(row.get("source_publication", "")),
            "reaction_smiles": str(row.get("reaction_smiles", "")),
        }
        records.append(record)
    return records


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def _json_default(value: object) -> object:
    """Preserve JSON scalar types for NumPy results in formal artifacts."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n"
    )


def select_stratified_smoke_records(
    records: list[dict[str, Any]], *, n_records: int, seed: int
) -> list[dict[str, Any]]:
    """Deterministically sample both T5 classes for an integration-only smoke.

    This helper is never called by a formal run.  A positional test slice can
    accidentally contain a single endpoint class, which prevents the smoke
    from exercising the paired AUPRC inference at all.  The formal sealed test
    set remains untouched and is always evaluated in full.
    """
    if n_records <= 0 or n_records >= len(records):
        return list(records)
    buckets = {0: [], 1: []}
    for record in records:
        label = int(float(record.get("measured_yield", 0.0)) >= 50.0)
        buckets[label].append(record)
    if not buckets[0] or not buckets[1]:
        raise RuntimeError("smoke test split lacks one T5 class; cannot exercise AUPRC inference")

    def stable_order(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(
            items,
            key=lambda item: hashlib.sha256(
                f"{seed}:{item.get('record_id', '')}".encode("utf-8")
            ).hexdigest(),
        )

    target_negative = min(len(buckets[0]), n_records // 2)
    target_positive = min(len(buckets[1]), n_records - target_negative)
    selected = stable_order(buckets[0])[:target_negative] + stable_order(buckets[1])[:target_positive]
    if len(selected) < n_records:
        selected_ids = {str(row.get("record_id", "")) for row in selected}
        remainder = [
            item
            for item in stable_order(records)
            if str(item.get("record_id", "")) not in selected_ids
        ]
        selected.extend(remainder[:n_records - len(selected)])
    return sorted(selected, key=lambda item: str(item.get("record_id", "")))


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.formal_run and args.smoke:
        raise ValueError("formal-run and smoke are mutually exclusive")
    if args.formal_run and (not torch.cuda.is_available() or not str(args.device).startswith("cuda")):
        raise RuntimeError("formal G6 v3 runs require CUDA; CPU fallback is forbidden")
    plan_data = json.loads(args.analysis_plan.read_text())
    validate_formal_analysis_plan(plan_data)
    if args.formal_run and int(plan_data["n_seeds"]) != args.n_seeds:
        raise ValueError("formal run seed count must equal frozen analysis plan")
    if args.formal_run and int(plan_data["n_bootstrap"]) != args.n_bootstrap:
        raise ValueError("formal run bootstrap count must equal frozen analysis plan")
    if args.formal_run and int(plan_data["n_permutations"]) != args.n_permutations:
        raise ValueError("formal run permutation count must equal frozen analysis plan")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_records = load_hte_records(args.hte_parquet)
    all_records, excluded_context_records = partition_context_complete_records(raw_records)
    _write_json(args.output_dir / "excluded_reaction_context_records_v3.json", {
        "reason": "records without complete reactants/products cannot be represented by the formal reaction-conditioned encoder",
        "n_input": len(raw_records),
        "n_included": len(all_records),
        "n_excluded": len(excluded_context_records),
        "records": excluded_context_records,
    })
    availability = validate_reaction_condition_records(all_records, formal=args.formal_run)
    train_records = [record for record in all_records if record["split"] == "train"]
    validation_records = [record for record in all_records if record["split"] == "val"]
    test_records = [record for record in all_records if record["split"] == "test"]
    if args.smoke:
        # Keep the full train-side manifest parent contract.  Only the held-out
        # scoring set is shortened and stratified for endpoint coverage; formal
        # testing always scores the complete sealed test set.
        test_records = select_stratified_smoke_records(
            test_records, n_records=args.smoke_per_split, seed=args.seed
        )
    if len(test_records) < 20:
        raise RuntimeError("test split too small for G6 v3 benchmark")
    if len(validation_records) < 20:
        raise RuntimeError("validation split too small for G6 v3 post-hoc calibration")
    cluster_contract = validate_cluster_contract(test_records, formal=args.formal_run)
    arms, arm_audit = load_matched_source_arms(train_records, args.manifest, seed=args.seed)
    normalizer = ConditionNormalizer().fit(arms["positive_only"])
    encoder = SharedPretrainedReactionEncoder(
        checkpoint_path=args.checkpoint,
        vocab_path=args.vocab,
        max_seq_len=args.max_seq_len,
        device=args.device,
        formal=args.formal_run,
    )
    validation_features = encoder.encode_records(validation_records, normalizer, batch_size=args.encode_batch_size)
    test_features = encoder.encode_records(test_records, normalizer, batch_size=args.encode_batch_size)
    output_predictions: dict[str, dict[int, list[dict[str, Any]]]] = {}
    all_metrics: dict[str, dict[int, dict[str, float]]] = {}
    train_history: dict[str, dict[int, dict[str, float]]] = {}
    calibration_by_arm_seed: dict[str, dict[int, dict[str, float]]] = {}

    for arm_name in ("positive_only", "pc_cng", "random", "template_rule", "union"):
        train_arm = arms[arm_name]
        train_features = encoder.encode_records(train_arm, normalizer, batch_size=args.encode_batch_size)
        output_predictions[arm_name] = {}
        all_metrics[arm_name] = {}
        train_history[arm_name] = {}
        calibration_by_arm_seed[arm_name] = {}
        for offset in range(args.n_seeds):
            item_seed = args.seed + offset
            model, history = train_multitask_heads(
                train_features,
                train_arm,
                device=args.device,
                seed=item_seed,
                epochs=args.epochs,
                batch_size=args.train_batch_size,
                learning_rate=args.learning_rate,
                max_rank_pairs=args.max_rank_pairs,
            )
            validation_predictions = predict_multitask(model, validation_features, device=args.device)
            validation_labels = [int(float(record["measured_yield"]) >= 50.0) for record in validation_records]
            temperature = fit_temperature_scaler(
                validation_predictions["T5"], validation_labels, device=args.device
            )
            predictions = predict_multitask(model, test_features, device=args.device)
            predictions["T5"] = apply_temperature(predictions["T5"], temperature)
            output_predictions[arm_name][item_seed] = metric_records("T5", test_records, predictions)
            all_metrics[arm_name][item_seed] = evaluate_secondary_metrics(test_records, predictions)
            train_history[arm_name][item_seed] = history
            calibration_by_arm_seed[arm_name][item_seed] = {
                "temperature": temperature,
                "n_validation": float(len(validation_records)),
                "selection_split": "val",
                "applied_to": "test",
            }

    primary = run_preregistered_primary_inference(
        output_predictions,
        n_bootstrap=args.n_bootstrap,
        n_permutations=args.n_permutations,
        seed=args.seed,
    )
    prediction_path = args.output_dir / "predictions_t5_v3.json"
    _write_json(prediction_path, {arm: {str(seed): rows for seed, rows in values.items()} for arm, values in output_predictions.items()})
    result = {
        "schema_version": FORMAL_SCHEMA_VERSION,
        "scientific_status": "FORMAL" if args.formal_run else "SMOKE_NOT_SCIENTIFIC",
        "analysis_plan": plan_data,
        "git_commit": _git_sha(),
        "input_hashes": {"hte_parquet": _sha256(args.hte_parquet), "manifest": _sha256(args.manifest), "analysis_plan": _sha256(args.analysis_plan)},
        "availability": availability,
        "cluster_contract": cluster_contract,
        "context_exclusions": {"n_input": len(raw_records), "n_included": len(all_records), "n_excluded": len(excluded_context_records), "artifact": str(args.output_dir / "excluded_reaction_context_records_v3.json")},
        "arm_audit": arm_audit,
        "n_train_matched_parents": arm_audit["parent_count"],
        "n_validation": len(validation_records),
        "n_test": len(test_records),
        "encoder": {"checkpoint": str(args.checkpoint), "checkpoint_sha256": _sha256(args.checkpoint), "vocab": str(args.vocab), "vocab_sha256": _sha256(args.vocab), "device": str(args.device), "shared_across_tasks": True, "product_only_baseline": False},
        "runtime": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "ld_library_path": os.environ.get("LD_LIBRARY_PATH", ""),
            "python_executable": os.sys.executable,
        },
        "metrics_by_arm_seed": all_metrics,
        "calibration_by_arm_seed": calibration_by_arm_seed,
        "primary_endpoint_interpretation": (
            "The endpoint is source-macro only when at least two evaluable source-publication slices exist; "
            "this run records the actual slice count alongside every arm/seed metric."
        ),
        "train_history": train_history,
        "primary_inference": primary,
        "prediction_artifact": str(prediction_path),
        "no_test_driven_baseline_selection": True,
    }
    _write_json(args.output_dir / "formal_result_v3.json", result)
    _write_json(args.output_dir / "run_manifest_v3.json", {
        "schema_version": FORMAL_SCHEMA_VERSION,
        "command_contract": {"formal_run": args.formal_run, "smoke": args.smoke, "device": str(args.device), "n_seeds": args.n_seeds, "n_bootstrap": args.n_bootstrap, "n_permutations": args.n_permutations},
        "input_hashes": result["input_hashes"],
        "git_commit": result["git_commit"],
    })
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hte-parquet", type=Path, default=REPO_ROOT / "data/processed/p4_hte_normalized.parquet")
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "data/p4/manifests/hte_feasibility_v2.json")
    parser.add_argument("--analysis-plan", type=Path, default=REPO_ROOT / "docs/phase_b_g6_analysis_plan_v1.json")
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "models/reaction_lm/chemformer_pretrained_hf/model_sanitized.ckpt")
    parser.add_argument("--vocab", type=Path, default=REPO_ROOT / "external/reaction_lm/Chemformer/bart_vocab.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--formal-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-per-split", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--n-permutations", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--encode-batch-size", type=int, default=16)
    parser.add_argument("--max-rank-pairs", type=int, default=4096)
    parser.add_argument("--max-seq-len", type=int, default=256)
    return parser


if __name__ == "__main__":
    namespace = build_parser().parse_args()
    result = run(namespace)
    print(json.dumps(
        {"status": result["scientific_status"], "primary": result["primary_inference"]},
        indent=2,
        default=_json_default,
    ))
