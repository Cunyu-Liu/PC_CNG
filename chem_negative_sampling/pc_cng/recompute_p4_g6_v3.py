#!/usr/bin/env python3
"""Independent reconstruction of G6 v3 preregistered primary inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pc_cng.p4_g6_benchmark_v3 import validate_formal_analysis_plan
from pc_cng.p4_g6_inference_v3 import run_preregistered_primary_inference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--analysis-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.analysis_plan.read_text())
    validate_formal_analysis_plan(plan)
    raw = json.loads(args.predictions.read_text())
    predictions = {arm: {int(seed): rows for seed, rows in seeds.items()} for arm, seeds in raw.items()}
    result = run_preregistered_primary_inference(
        predictions,
        n_bootstrap=int(plan["n_bootstrap"]),
        n_permutations=int(plan["n_permutations"]),
    )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
