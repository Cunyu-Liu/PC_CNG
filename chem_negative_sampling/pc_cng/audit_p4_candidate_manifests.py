"""P4-G1 Candidate Manifest Audit.

CLI entry point::

    python3 -m pc_cng.audit_p4_candidate_manifests \
        --manifest-dir data/p4/manifests \
        --output-dir results/p4_candidate_audit

Audits the three P4-G1 candidate manifests for:
- Each group has exactly 1 gold_candidate;
- All candidate IDs unique within a group;
- manifest_hash present and correct;
- No parent_reaction_id crosses splits (parent leakage);
- All 24 required fields populated;
- Reports all required statistics per spec.

Writes ``go_no_go.json`` with GO/NO-GO verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


# Required candidate fields per spec
REQUIRED_CANDIDATE_FIELDS = [
    "benchmark_name",
    "group_id",
    "source_reaction_id",
    "parent_reaction_id",
    "experimental_group_id",
    "gold_candidate",
    "candidate_id",
    "candidate_smiles",
    "candidate_source",
    "candidate_source_rank",
    "canonical_smiles",
    "atom_mapping_status",
    "reaction_family",
    "reaction_template",
    "product_scaffold",
    "edit_type",
    "edit_distance",
    "train_overlap",
    "known_positive_collision",
    "nearest_train_similarity",
    "split",
    "oracle_coverage",
    "manifest_version",
    "manifest_hash",
]

EXPECTED_MANIFESTS = [
    "hte_feasibility_v1.json",
    "fixed_forward_candidates_v1.json",
    "fixed_retro_candidates_v1.json",
]


def _strip_manifest_hash(obj: Any) -> Any:
    """Recursively remove ``manifest_hash`` fields from a nested structure."""
    if isinstance(obj, dict):
        return {k: _strip_manifest_hash(v) for k, v in obj.items() if k != "manifest_hash"}
    if isinstance(obj, list):
        return [_strip_manifest_hash(item) for item in obj]
    return obj


def _compute_manifest_hash(manifest: dict) -> str:
    """Recompute the manifest hash for verification.

    Strips both the top-level and nested ``manifest_hash`` fields (inside
    candidates) before hashing, so verification is stable against
    backfilled candidate hashes.
    """
    content = _strip_manifest_hash(manifest)
    canonical = json.dumps(content, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def audit_manifest(manifest: dict, manifest_name: str) -> dict:
    """Audit a single manifest and return findings."""
    findings = {
        "manifest_name": manifest_name,
        "benchmark_name": manifest.get("benchmark_name", ""),
        "manifest_version": manifest.get("manifest_version", ""),
        "manifest_hash_reported": manifest.get("manifest_hash", ""),
        "manifest_hash_recomputed": _compute_manifest_hash(manifest),
        "n_groups": 0,
        "n_candidates": 0,
        "n_gold": 0,
        "errors": [],
        "warnings": [],
        "stats": {},
    }

    # 1. Hash verification
    if findings["manifest_hash_reported"] != findings["manifest_hash_recomputed"]:
        findings["errors"].append(
            f"manifest_hash mismatch: reported={findings['manifest_hash_reported'][:16]}... "
            f"recomputed={findings['manifest_hash_recomputed'][:16]}..."
        )

    groups = manifest.get("groups", [])
    findings["n_groups"] = len(groups)

    # Per-group checks
    candidate_source_counts = Counter()
    reaction_family_counts = Counter()
    split_counts = Counter()
    candidate_counts_per_group = []
    known_positive_collisions = 0
    train_overlaps = 0
    nearest_train_sims = []
    oracle_top1_hits = 0
    parent_split_map: Dict[str, set] = defaultdict(set)
    scaffold_set = set()
    template_set = set()
    parent_set = set()

    all_candidate_ids = set()

    for gi, group in enumerate(groups):
        gid = group.get("group_id", f"group_{gi}")
        candidates = group.get("candidates", [])
        candidate_counts_per_group.append(len(candidates))
        findings["n_candidates"] += len(candidates)

        # Check exactly 1 gold
        golds = [c for c in candidates if c.get("gold_candidate")]
        if len(golds) != 1:
            findings["errors"].append(
                f"Group {gid}: expected exactly 1 gold_candidate, found {len(golds)}"
            )
        else:
            findings["n_gold"] += 1

        # Check candidate ID uniqueness within group
        group_cids = [c.get("candidate_id", "") for c in candidates]
        if len(group_cids) != len(set(group_cids)):
            findings["errors"].append(
                f"Group {gid}: duplicate candidate_ids within group"
            )

        # Check all required fields
        for c in candidates:
            for field in REQUIRED_CANDIDATE_FIELDS:
                if field not in c:
                    findings["errors"].append(
                        f"Group {gid}, candidate {c.get('candidate_id', '?')}: missing field '{field}'"
                    )
                    break

            # Collect stats
            cs = c.get("candidate_source", "unknown")
            candidate_source_counts[cs] += 1
            rf = c.get("reaction_family", "unknown")
            reaction_family_counts[rf] += 1
            sp = c.get("split", "unknown")
            split_counts[sp] += 1
            if c.get("known_positive_collision"):
                known_positive_collisions += 1
            if c.get("train_overlap"):
                train_overlaps += 1
            ns = c.get("nearest_train_similarity")
            if isinstance(ns, (int, float)):
                nearest_train_sims.append(ns)
            if c.get("gold_candidate"):
                oracle_top1_hits += 1
            scaffold_set.add(c.get("product_scaffold", ""))
            template_set.add(c.get("reaction_template", ""))
            parent_set.add(c.get("parent_reaction_id", ""))

            # Track parent-split for leakage detection
            parent_id = c.get("parent_reaction_id", "")
            split_val = c.get("split", "")
            if parent_id and split_val:
                parent_split_map[parent_id].add(split_val)

            # Global candidate ID uniqueness
            cid = c.get("candidate_id", "")
            if cid in all_candidate_ids:
                findings["warnings"].append(f"Duplicate global candidate_id: {cid}")
            all_candidate_ids.add(cid)

        # Track parent reaction for split isolation
        parent_id = group.get("parent_reaction_id", "")
        split_val = group.get("split", "")
        if parent_id and split_val:
            parent_split_map[parent_id].add(split_val)

    # 2. Parent leakage check
    parent_leakage = 0
    for parent_id, splits in parent_split_map.items():
        if len(splits) > 1:
            parent_leakage += 1
            findings["errors"].append(
                f"Parent leakage: parent_reaction_id='{parent_id}' appears in splits: {splits}"
            )

    # 3. Compute statistics
    findings["stats"] = {
        "candidate_source_distribution": dict(candidate_source_counts),
        "reaction_family_distribution": dict(reaction_family_counts),
        "split_distribution": dict(split_counts),
        "candidate_count_per_group": {
            "min": min(candidate_counts_per_group) if candidate_counts_per_group else 0,
            "max": max(candidate_counts_per_group) if candidate_counts_per_group else 0,
            "mean": round(sum(candidate_counts_per_group) / len(candidate_counts_per_group), 2) if candidate_counts_per_group else 0,
        },
        "known_positive_collisions": known_positive_collisions,
        "train_overlaps": train_overlaps,
        "nearest_train_similarity": {
            "min": round(min(nearest_train_sims), 4) if nearest_train_sims else 0,
            "max": round(max(nearest_train_sims), 4) if nearest_train_sims else 0,
            "mean": round(sum(nearest_train_sims) / len(nearest_train_sims), 4) if nearest_train_sims else 0,
        },
        "oracle_top1_coverage": round(oracle_top1_hits / findings["n_groups"], 4) if findings["n_groups"] else 0,
        "oracle_top1_hits": oracle_top1_hits,
        "n_unique_scaffolds": len(scaffold_set),
        "n_unique_templates": len(template_set),
        "n_unique_parents": len(parent_set),
        "parent_leakage_count": parent_leakage,
    }

    return findings


def write_audit_report(all_findings: List[dict], output_dir: Path) -> None:
    """Write the audit report and go_no_go.json."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write detailed findings
    (output_dir / "manifest_audit_details.json").write_text(
        json.dumps(all_findings, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Write summary report
    lines = ["# P4-G1 Candidate Manifest Audit Report\n"]
    lines.append(f"**Manifests audited:** {len(all_findings)}\n")

    total_errors = 0
    total_warnings = 0
    for f in all_findings:
        lines.append(f"\n## {f['manifest_name']}")
        lines.append(f"- benchmark_name: {f['benchmark_name']}")
        lines.append(f"- manifest_hash: {f['manifest_hash_reported'][:16]}... (verified: {f['manifest_hash_reported'] == f['manifest_hash_recomputed']})")
        lines.append(f"- n_groups: {f['n_groups']}")
        lines.append(f"- n_candidates: {f['n_candidates']}")
        lines.append(f"- n_gold: {f['n_gold']}")
        lines.append(f"- errors: {len(f['errors'])}")
        lines.append(f"- warnings: {len(f['warnings'])}")
        if f["stats"]:
            s = f["stats"]
            lines.append(f"- candidate_source_distribution: {s['candidate_source_distribution']}")
            lines.append(f"- split_distribution: {s['split_distribution']}")
            lines.append(f"- candidates_per_group: min={s['candidate_count_per_group']['min']}, max={s['candidate_count_per_group']['max']}, mean={s['candidate_count_per_group']['mean']}")
            lines.append(f"- known_positive_collisions: {s['known_positive_collisions']}")
            lines.append(f"- train_overlaps: {s['train_overlaps']}")
            lines.append(f"- nearest_train_similarity: min={s['nearest_train_similarity']['min']}, max={s['nearest_train_similarity']['max']}, mean={s['nearest_train_similarity']['mean']}")
            lines.append(f"- oracle_top1_coverage: {s['oracle_top1_coverage']}")
            lines.append(f"- n_unique_scaffolds: {s['n_unique_scaffolds']}")
            lines.append(f"- n_unique_parents: {s['n_unique_parents']}")
            lines.append(f"- parent_leakage_count: {s['parent_leakage_count']}")
        if f["errors"]:
            lines.append(f"\n### Errors:")
            for e in f["errors"][:10]:
                lines.append(f"- {e}")
            if len(f["errors"]) > 10:
                lines.append(f"- ... and {len(f['errors']) - 10} more errors")
        total_errors += len(f["errors"])
        total_warnings += len(f["warnings"])

    (output_dir / "audit_report.md").write_text("\n".join(lines), encoding="utf-8")

    # Write go_no_go.json
    # GO criteria:
    # - Each group has exactly 1 gold
    # - No parent leakage
    # - All candidate sources reproducible (manifest hash fixed)
    # - Oracle coverage reported
    go_status = "GO" if total_errors == 0 else "NO_GO"

    go_no_go = {
        "phase": "P4-G1",
        "status": go_status,
        "primary_metric": {
            "n_manifests": len(all_findings),
            "n_total_groups": sum(f["n_groups"] for f in all_findings),
            "n_total_candidates": sum(f["n_candidates"] for f in all_findings),
            "n_total_errors": total_errors,
            "n_total_warnings": total_warnings,
        },
        "predeclared_threshold": {
            "max_errors": 0,
            "max_parent_leakage": 0,
            "all_groups_have_gold": True,
            "manifest_hash_fixed": True,
        },
        "evidence_paths": [
            "data/p4/manifests/hte_feasibility_v1.json",
            "data/p4/manifests/fixed_forward_candidates_v1.json",
            "data/p4/manifests/fixed_retro_candidates_v1.json",
            "results/p4_candidate_audit/manifest_audit_details.json",
            "results/p4_candidate_audit/audit_report.md",
        ],
        "limitations": [
            f"{total_errors} errors found across {len(all_findings)} manifests." if total_errors else "No errors found.",
            "Learned PC-CNG candidates are NOT included in v1 (appended as v2 in a later phase).",
            "Tanimoto retrieval uses Morgan fingerprints (radius 2, 1024 bits).",
        ],
        "next_phase_allowed": go_status == "GO",
    }
    (output_dir / "go_no_go.json").write_text(
        json.dumps(go_no_go, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return go_no_go


def main():
    parser = argparse.ArgumentParser(
        description="P4-G1 Candidate Manifest Audit"
    )
    parser.add_argument(
        "--manifest-dir", required=True,
        help="Directory containing manifest JSON files"
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory for audit output files"
    )
    args = parser.parse_args()

    manifest_dir = Path(args.manifest_dir)
    output_dir = Path(args.output_dir)

    if not manifest_dir.exists():
        print(f"ERROR: manifest dir not found: {manifest_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"[P4-G1 Audit] Auditing manifests in {manifest_dir}")

    all_findings = []
    for manifest_name in EXPECTED_MANIFESTS:
        path = manifest_dir / manifest_name
        if not path.exists():
            print(f"WARNING: manifest not found: {path}")
            all_findings.append({
                "manifest_name": manifest_name,
                "benchmark_name": "",
                "manifest_version": "",
                "manifest_hash_reported": "",
                "manifest_hash_recomputed": "",
                "n_groups": 0,
                "n_candidates": 0,
                "n_gold": 0,
                "errors": [f"Manifest file not found: {path}"],
                "warnings": [],
                "stats": {},
            })
            continue

        manifest = json.loads(path.read_text(encoding="utf-8"))
        findings = audit_manifest(manifest, manifest_name)
        all_findings.append(findings)
        print(f"  {manifest_name}: {findings['n_groups']} groups, {findings['n_candidates']} candidates, {len(findings['errors'])} errors")

    go_no_go = write_audit_report(all_findings, output_dir)
    print(f"\n[P4-G1 Audit] Verdict: {go_no_go['status']}")
    print(f"[P4-G1 Audit] next_phase_allowed: {go_no_go['next_phase_allowed']}")

    sys.exit(0 if go_no_go["status"] == "GO" else 1)


if __name__ == "__main__":
    main()
