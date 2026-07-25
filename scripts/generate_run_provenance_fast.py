#!/usr/bin/env python3
"""Fast version: generate run_provenance.json for every results/ subdirectory.

Optimizations vs original:
- Skip conda env export (too slow); use requirements.txt hash + python version
- Limit file hashing to files under 50MB
- Print progress every 10 directories
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path("/home/cunyuliu/pc_cng_research")
RESULTS_DIR = REPO_ROOT / "results"
PROVENANCE_FILENAME = "run_provenance.json"
SCHEMA_VERSION = "1.0"
MAX_HASH_SIZE = 50 * 1024 * 1024  # 50MB

SKIP_HASH_SUFFIXES = {".log", ".pid", ".pt", ".pkl", ".npy", ".npz"}
SKIP_HASH_NAMES = {PROVENANCE_FILENAME, "nmi_audit_status.json", "nohup.out"}


def run(cmd, cwd=None, timeout=15):
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def git_commit_sha():
    return run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT) or "unknown"


def git_branch():
    return run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO_ROOT) or "unknown"


def env_lock_hash():
    req = REPO_ROOT / "chem_negative_sampling" / "requirements.txt"
    content = ""
    if req.exists():
        content = req.read_text(errors="replace")
    py_ver = sys.version
    return hashlib.sha256((content + py_ver).encode("utf-8")).hexdigest()[:16]


def hash_file_fast(path):
    try:
        size = path.stat().st_size
        if size > MAX_HASH_SIZE:
            return f"large_{size}"
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()[:16]
    except Exception:
        return None


def hash_directory(dirpath):
    h = hashlib.sha256()
    files = sorted(
        p for p in dirpath.rglob("*")
        if p.is_file()
        and p.name not in SKIP_HASH_NAMES
        and p.suffix not in SKIP_HASH_SUFFIXES
    )
    for p in files:
        rel = str(p.relative_to(dirpath))
        h.update(rel.encode("utf-8"))
        try:
            size = p.stat().st_size
            if size <= MAX_HASH_SIZE:
                with open(p, "rb") as f:
                    for chunk in iter(lambda: f.read(65536), b""):
                        h.update(chunk)
            else:
                h.update(str(size).encode("utf-8"))
        except Exception:
            h.update(b"<unreadable>")
    return h.hexdigest()[:16]


def extract_command(dirpath, dirname):
    cmd_log = dirpath / "commands.log"
    if cmd_log.exists():
        try:
            content = cmd_log.read_text(errors="replace").strip()
            if content:
                return content.splitlines()[0][:500]
        except Exception:
            pass
    return f"<inferred from directory name: {dirname}>"


def extract_seeds(dirpath):
    seeds = []
    for p in dirpath.glob("*.json"):
        try:
            data = json.loads(p.read_text(errors="replace"))
            if isinstance(data, dict):
                for key in ("seed", "random_seed", "n_seeds", "base_seed"):
                    if key in data and isinstance(data[key], int):
                        seeds.append(data[key])
                if "config" in data and isinstance(data["config"], dict):
                    for key in ("seed", "random_seed", "base_seed"):
                        if key in data["config"] and isinstance(data["config"][key], int):
                            seeds.append(data["config"][key])
        except Exception:
            continue
    seen = set()
    unique = []
    for s in seeds:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique


def output_schema(dirpath):
    schema = {"json": [], "csv": [], "png": [], "log": [], "other": []}
    for p in sorted(dirpath.rglob("*")):
        if not p.is_file() or p.name in SKIP_HASH_NAMES:
            continue
        rel = str(p.relative_to(dirpath))
        suffix = p.suffix.lower().lstrip(".")
        if suffix in schema:
            schema[suffix].append(rel)
        else:
            schema["other"].append(rel)
    return {k: v for k, v in schema.items() if v}


def input_hash(dirpath):
    h = hashlib.sha256()
    found = False
    for name in ("input_manifest.json", "manifest.json", "config.json", "split_manifest.json"):
        p = dirpath / name
        if p.exists():
            try:
                h.update(p.read_bytes())
                found = True
            except Exception:
                pass
    return h.hexdigest()[:16] if found else ""


def frozen_analysis_spec(dirpath):
    for name in ("analysis_spec.json", "frozen_analysis_spec.json", "go_no_go.json"):
        p = dirpath / name
        if p.exists():
            return name
    return ""


def main():
    if not RESULTS_DIR.exists():
        print(f"ERROR: {RESULTS_DIR} does not exist", file=sys.stderr)
        return 1

    sha = git_commit_sha()
    branch = git_branch()
    env_hash = env_lock_hash()
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "unknown")
    n_created = 0
    n_errors = 0
    dirs = sorted(d for d in RESULTS_DIR.iterdir() if d.is_dir())
    total = len(dirs)

    for i, entry in enumerate(dirs):
        provenance_path = entry / PROVENANCE_FILENAME
        try:
            provenance = {
                "schema_version": SCHEMA_VERSION,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "run_id": hash_directory(entry),
                "git_commit_sha": sha,
                "git_branch": branch,
                "env_lock_hash": env_hash,
                "conda_env": conda_env,
                "python_version": sys.version.split()[0],
                "input_hash": input_hash(entry),
                "exact_command": extract_command(entry, entry.name),
                "random_seeds": extract_seeds(entry),
                "output_schema": output_schema(entry),
                "frozen_analysis_spec": frozen_analysis_spec(entry),
                "nmi_audit_status_exists": (entry / "nmi_audit_status.json").exists(),
                "provenance_note": "Generated by Phase 0 evidence freeze; original artifacts unchanged",
            }
            provenance_path.write_text(
                json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            n_created += 1
            if (i + 1) % 10 == 0 or (i + 1) == total:
                print(f"  [{i+1}/{total}] {entry.name}", flush=True)
        except Exception as e:
            n_errors += 1
            print(f"  ERR {entry.name}: {e}", file=sys.stderr, flush=True)

    print(f"\nDone: {n_created} created, {n_errors} errors")
    return 0 if n_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
