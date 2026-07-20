"""Build a reusable known-positive reaction/product cache JSON."""

from __future__ import annotations

import argparse
import json

from .known_positive_cache import build_known_positive_cache, write_cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive-csv", action="append", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    payload = build_known_positive_cache(args.positive_csv)
    write_cache(args.output_json, payload)
    print(
        json.dumps(
            {
                "output_json": args.output_json,
                "canonical_reaction_count": payload["canonical_reaction_count"],
                "canonical_product_count": payload["canonical_product_count"],
                "source_csv": args.positive_csv,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
