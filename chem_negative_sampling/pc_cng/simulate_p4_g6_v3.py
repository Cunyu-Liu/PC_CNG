#!/usr/bin/env python3
"""Pre-run operating-characteristic simulation for the frozen G6 v3 plan."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pc_cng.p4_g6_inference_v3 import simulate_inference_operating_characteristics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-simulations", type=int, default=80)
    parser.add_argument("--n-bootstrap", type=int, default=300)
    parser.add_argument("--n-permutations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260728)
    args = parser.parse_args()
    result = simulate_inference_operating_characteristics(
        n_simulations=args.n_simulations,
        n_bootstrap=args.n_bootstrap,
        n_permutations=args.n_permutations,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
