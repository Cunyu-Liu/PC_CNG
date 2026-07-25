#!/usr/bin/env python3
"""Generate SHA-256 checksums for all manuscript-level summary artifacts.

Phase 0: bring all manuscript-level summary artifacts under version control
with checksums for integrity verification.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path("/home/cunyuliu/pc_cng_research")

# Manuscript-level artifacts: docs with manuscript in name, manuscript figures dirs,
# and key result summary JSONs that feed into manuscript tables.
MANUSCRIPT_PATTERNS = [
    "docs/manuscript_*.md",
    "docs/manuscript_*/",
    "docs/manuscript_figures_*/",
    "results/manuscript_tables_*/",
    "results/current_best_ensemble_detailed_metrics.json",
    "results/ensemble_real_only_summary.json",
    "results/full_feasibility_matrix_summary.json",
    "results/stacked_ensemble_summary.json",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_artifacts() -> list[dict]:
    artifacts: list[dict] = []
    seen: set[Path] = set()

    for pattern in MANUSCRIPT_PATTERNS:
        for p in sorted(REPO_ROOT.glob(pattern)):
            if p.is_file() and p not in seen:
                seen.add(p)
                try:
                    artifacts.append({
                        "path": str(p.relative_to(REPO_ROOT)),
                        "sha256": sha256_file(p),
                        "size_bytes": p.stat().st_size,
                        "type": "file",
                    })
                except Exception as e:
                    artifacts.append({
                        "path": str(p.relative_to(REPO_ROOT)),
                        "error": str(e),
                        "type": "error",
                    })
            elif p.is_dir():
                for f in sorted(p.rglob("*")):
                    if f.is_file() and f not in seen:
                        seen.add(f)
                        try:
                            artifacts.append({
                                "path": str(f.relative_to(REPO_ROOT)),
                                "sha256": sha256_file(f),
                                "size_bytes": f.stat().st_size,
                                "type": "file",
                            })
                        except Exception as e:
                            artifacts.append({
                                "path": str(f.relative_to(REPO_ROOT)),
                                "error": str(e),
                                "type": "error",
                            })
    return artifacts


def main() -> int:
    artifacts = collect_artifacts()
    manifest = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "description": "SHA-256 checksums for all manuscript-level summary artifacts",
        "n_artifacts": len(artifacts),
        "artifacts": artifacts,
    }
    out = REPO_ROOT / "docs" / "manuscript_artifact_checksums.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out} with {len(artifacts)} artifacts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
