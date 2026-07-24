#!/usr/bin/env python3
"""G6 v2 runner: reads HTE parquet with pyarrow, maps columns, runs evaluation.

FIXES from original p4_g6_hte_eval_v2.py __main__:
1. Uses pyarrow instead of pandas (pandas not available in pc_cng_gpu env)
2. Suppresses RDKit DEPRECATION warnings (major I/O overhead fix)
3. Proper column mapping: reactant_1+reactant_2 → reactants, catalyst+solvent → agents
4. Fingerprint cache stats reporting
"""
from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path

# Suppress RDKit warnings BEFORE any imports
logging.getLogger("rdkit").setLevel(logging.ERROR)
try:
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    pass

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "chem_negative_sampling"))

from pc_cng.p4_g6_hte_eval_v2 import run_g6_v2_evaluation
from pc_cng.p4_g6_task_heads_v2 import get_fp_cache_stats


def load_hte_records(parquet_path: Path) -> list[dict]:
    """Load HTE records from parquet using pyarrow.

    Maps parquet columns to the format expected by G6 task heads:
    - reactant_1_smiles + reactant_2_smiles → reactants
    - catalyst_1_smiles + catalyst_2_smiles + solvent → agents
    - record_id → candidate_id (for FNR mapping)
    """
    table = pq.read_table(str(parquet_path))
    records_raw = table.to_pylist()

    records = []
    for r in records_raw:
        # Combine reactants
        reactants_parts = []
        for k in ("reactant_1_smiles", "reactant_2_smiles"):
            v = r.get(k, "")
            if v and isinstance(v, str) and v.strip():
                reactants_parts.append(v.strip())
        reactants = ".".join(reactants_parts) if reactants_parts else ""

        # Combine agents (catalysts + solvent + temperature)
        agent_parts = []
        for k in ("catalyst_1_smiles", "catalyst_2_smiles", "solvent"):
            v = r.get(k, "")
            if v and isinstance(v, str) and v.strip():
                agent_parts.append(v.strip())
        agents = ".".join(agent_parts) if agent_parts else ""

        # Build record with all fields needed by task heads
        record = {
            "candidate_id": r.get("record_id", ""),
            "record_id": r.get("record_id", ""),
            "products": r.get("products", ""),
            "reactants": reactants,
            "agents": agents,
            "conditions": agents,  # alias for compatibility
            "measured_yield": float(r.get("measured_yield", 0) or 0),
            "yield": float(r.get("measured_yield", 0) or 0),
            "yield_bin": r.get("yield_bin", 0),
            "split": r.get("split", "train"),
            "experimental_group": str(r.get("experimental_group", "")),
            "plate_id": str(r.get("plate_id", "")),
            "reaction_family": r.get("reaction_family", ""),
            "reaction_class": r.get("reaction_class", ""),
            "source_publication": r.get("source_publication", ""),
            "license": r.get("license", ""),
            "reaction_smiles": r.get("reaction_smiles", ""),
            "temperature": r.get("temperature", ""),
            "reaction_time_hrs": r.get("reaction_time_hrs", ""),
        }
        records.append(record)

    return records


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="G6 v2 evaluation (pyarrow + warning suppression)")
    parser.add_argument("--hte-parquet", type=Path,
                        default=_REPO_ROOT / "data/processed/p4_hte_normalized.parquet")
    parser.add_argument("--manifest", type=Path,
                        default=_REPO_ROOT / "data/p4/manifests/hte_feasibility_v2.json")
    parser.add_argument("--risk-artifacts", type=Path,
                        default=_REPO_ROOT / "results/p4_risk_aware/risk_artifacts.json")
    parser.add_argument("--output-dir", type=Path,
                        default=_REPO_ROOT / "results/p4_hte_external_validation_v2")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load HTE data using pyarrow
    print(f"Loading HTE parquet: {args.hte_parquet}")
    hte_records = load_hte_records(args.hte_parquet)
    print(f"Loaded {len(hte_records)} HTE records")

    # Count splits
    from collections import defaultdict
    split_counts = defaultdict(int)
    for r in hte_records:
        split_counts[r.get("split", "unknown")] += 1
    print(f"Split counts: {dict(split_counts)}")

    if args.smoke:
        # Sample proportionally from each split to ensure all splits present
        by_split = defaultdict(list)
        for r in hte_records:
            by_split[r.get("split", "train")].append(r)
        smoke_records = []
        for split in ("train", "val", "test"):
            recs = by_split.get(split, [])
            cap = {"train": 800, "val": 200, "test": 400}[split]
            smoke_records.extend(recs[:cap])
        hte_records = smoke_records
        args.n_bootstrap = 500
        print(f"Smoke mode: {len(hte_records)} records")

    print(f"Manifest: {args.manifest}")
    print(f"Risk artifacts: {args.risk_artifacts}")
    print(f"Output dir: {args.output_dir}")
    print(f"N bootstrap: {args.n_bootstrap}")

    # Run evaluation
    result, scored_data = run_g6_v2_evaluation(
        hte_records=hte_records,
        manifest_path=args.manifest,
        risk_artifacts_path=args.risk_artifacts,
        n_bootstrap=args.n_bootstrap,
    )

    # Write results
    output_file = args.output_dir / "go_no_go_v2.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2, default=str)

    # Save scored records for independent CI reconstruction (Exit Criterion)
    scored_file = args.output_dir / "scored_records_v2.json"
    with open(scored_file, "w") as f:
        json.dump(scored_data, f, indent=2, default=str)

    # Summary CSV
    metric_keys = sorted(
        set(k for r in result["arm_results"].values() for k in r["metrics"])
    )
    summary_lines = ["arm_id,arm_name,n_train,n_pos,n_neg," + ",".join(metric_keys)]
    for arm_id, r in result["arm_results"].items():
        metrics = r["metrics"]
        row = f"{arm_id},{r['arm_name']},{r['n_train']},{r['n_pos']},{r['n_neg']}"
        for k in metric_keys:
            row += f",{metrics.get(k, '')}"
        summary_lines.append(row)
    (args.output_dir / "summary_v2.csv").write_text("\n".join(summary_lines))

    # Print cache stats
    cache_stats = get_fp_cache_stats()
    print(f"\nFingerprint cache: hits={cache_stats['hits']}, misses={cache_stats['misses']}, "
          f"hit_rate={cache_stats['hit_rate']:.1%}, size={cache_stats['cache_size']}")

    print(f"\nVerdict: {result['verdict']}")
    print(f"Reason: {result['reason']}")
    print(f"Primary endpoint: {result['primary_endpoint']}")
    print(f"Collision sensitivity: {result['collision_sensitivity']:.4f}")
    print(f"Output: {output_file}")
    print(f"Scored records: {scored_file}")
