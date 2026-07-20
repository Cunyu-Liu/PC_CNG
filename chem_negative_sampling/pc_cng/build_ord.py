"""ORD (Open Reaction Database) ingestion and normalization for PC-CNG (P1-09).

Downloads ORD protobuf dataset shards from the official ord-data GitHub
repository (via the Git-LFS media endpoint), parses each ``Dataset`` message
with ``ord_schema``, and normalizes every reaction to the same schema as
``data/processed/uspto_openmolecules_normalized.csv``::

    source_id, reaction_smiles, reactants, agents, products,
    label_type, yield, source, split_key, split

If network access or ``ord_schema`` parsing fails, the module degrades to a
documented USPTO-proxy subset (a deterministic sample of
``uspto_openmolecules_normalized.csv`` relabelled with ``source =
"ord_subset_uspto_proxy"``) so the downstream pipeline can still be
validated end-to-end.  The degradation is recorded in the summary JSON.

CLI::

    python3 -m pc_cng.build_ord --output-dir data/processed
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import random
import sys
import urllib.parse
import urllib.request
from typing import Dict, Iterable, List, Optional, Tuple

from .chem_utils import (
    canonicalize_smiles, is_valid_reaction, split_reaction,
)


ORD_DATA_REPO = "open-reaction-database/ord-data"
ORD_GITHUB_API = "https://api.github.com/repos/{repo}/contents/{path}?ref=main"
ORD_LFS_MEDIA = "https://media.githubusercontent.com/media/{repo}/main/{path}"

NORMALIZED_FIELDS = [
    "source_id", "reaction_smiles", "reactants", "agents", "products",
    "label_type", "yield", "source", "split_key", "split",
]

DEFAULT_MAX_DATASETS = 3          # ~750 reactions per dataset => ~2250 rows
DEFAULT_FALLBACK_SIZE = 1500      # USPTO-proxy fallback (>= 1000 required)
DEFAULT_SEED = 20260719
DEFAULT_SHARDS = ["00", "01", "02", "03", "04", "05"]


# --------------------------------------------------------------------------- #
# CSV helpers
# --------------------------------------------------------------------------- #

def read_csv(path: str) -> Tuple[List[Dict[str, str]], List[str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        return rows, list(reader.fieldnames or [])


def write_csv(path: str, rows: Iterable[Dict[str, object]],
              fields: Optional[List[str]] = None) -> None:
    if fields is None:
        fields = NORMALIZED_FIELDS
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({f: row.get(f, "") for f in fields})


# --------------------------------------------------------------------------- #
# Network: fetch ORD dataset list + download .pb.gz
# --------------------------------------------------------------------------- #

def _http_get(url: str, timeout: float = 30.0, headers: Optional[Dict[str, str]] = None) -> bytes:
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "pc-cng-build-ord/1.0")
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def fetch_dataset_list(shards: List[str], max_datasets: int,
                       timeout: float = 20.0) -> List[Dict[str, str]]:
    """List ``.pb.gz`` files in each shard via the GitHub contents API.

    Returns a list of ``{"name", "path", "size"}`` dicts (capped at
    ``max_datasets``).
    """
    out: List[Dict[str, str]] = []
    for shard in shards:
        if len(out) >= max_datasets:
            break
        url = ORD_GITHUB_API.format(repo=ORD_DATA_REPO, path=f"data/{shard}")
        try:
            raw = _http_get(url, timeout=timeout)
            entries = json.loads(raw.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - network is best-effort
            print(f"[warn] failed to list shard data/{shard}: {exc}", file=sys.stderr)
            continue
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name", "")
            if not name.endswith(".pb.gz"):
                continue
            out.append({
                "name": name,
                "path": entry.get("path", f"data/{shard}/{name}"),
                "size": str(entry.get("size", 0)),
            })
            if len(out) >= max_datasets:
                break
    return out


def download_dataset(path: str, dest: str, timeout: float = 60.0) -> bool:
    """Download a single ``.pb.gz`` file via the Git-LFS media endpoint."""
    url = ORD_LFS_MEDIA.format(repo=ORD_DATA_REPO, path=urllib.parse.quote(path, safe="/"))
    try:
        data = _http_get(url, timeout=timeout)
        with open(dest, "wb") as handle:
            handle.write(data)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] download failed for {path}: {exc}", file=sys.stderr)
        return False


# --------------------------------------------------------------------------- #
# Parsing: ord_schema Dataset -> normalized rows
# --------------------------------------------------------------------------- #

def _import_ord_schema():
    """Import ord_schema components lazily; return None if unavailable."""
    try:
        from ord_schema import message_helpers  # type: ignore
        from ord_schema.proto import dataset_pb2, reaction_pb2  # type: ignore
        # ord_schema 0.5.x: reaction<->SMILES helpers live in message_helpers
        reaction_to_smiles = getattr(message_helpers, "get_reaction_smiles", None)
        return {
            "message_helpers": message_helpers,
            "dataset_pb2": dataset_pb2,
            "reaction_pb2": reaction_pb2,
            "reaction_to_smiles": reaction_to_smiles,
        }
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] ord_schema import failed: {exc}", file=sys.stderr)
        return None


def _safe_compound_smiles(compound) -> str:
    """Extract a SMILES identifier from an ORD Compound proto if present."""
    for ident in getattr(compound, "identifiers", []):
        ctype = getattr(ident, "type", 0)
        # ord_schema: SMILES = 2
        if ctype == 2 and getattr(ident, "value", ""):
            return ident.value
    return ""


def _extract_yield(reaction) -> str:
    """Best-effort yield extraction from the first outcome's products."""
    try:
        outcomes = list(reaction.outcomes)
    except Exception:
        return ""
    for outcome in outcomes:
        for product in outcome.products:
            try:
                y = getattr(product.compound, "yield", None)
                if y is not None and getattr(y, "value", 0) > 0:
                    return f"{float(y.value):.1f}"
            except Exception:
                continue
    return ""


def _reaction_smiles_from_reaction(reaction, helpers) -> str:
    """Convert a Reaction proto to reaction SMILES, trying several strategies."""
    if helpers and helpers["reaction_to_smiles"] is not None:
        try:
            return helpers["reaction_to_smiles"](reaction)
        except Exception:
            pass
    # manual fallback: gather reactants / agents / products SMILES
    reactants, agents, products = [], [], []
    try:
        for _, inp in reaction.inputs.items():
            for comp in inp.components:
                smi = _safe_compound_smiles(comp)
                if not smi:
                    continue
                # ord_schema: REACTANT = 1, AGENT = 2, SOLVENT = 3, etc.
                roles = list(comp.compound_role) if hasattr(comp, "compound_role") else []
                # CompoundRole: 0=REACTANT (default), 1=PRODUCT, 2=AGENT
                is_reactant = (not roles) or any(r == 0 for r in roles)
                if is_reactant:
                    reactants.append(smi)
                else:
                    agents.append(smi)
        for outcome in reaction.outcomes:
            for product in outcome.products:
                smi = _safe_compound_smiles(product.compound)
                if smi:
                    products.append(smi)
    except Exception:
        pass
    r = ".".join(reactants)
    p = ".".join(products)
    if not r or not p:
        return ""
    a = ".".join(agents)
    return f"{r}>{a}>{p}" if a else f"{r}>>{p}"


def parse_dataset_file(pb_gz_path: str, source_tag: str,
                       helpers=None) -> List[Dict[str, str]]:
    """Parse a single ``.pb.gz`` Dataset file into normalized rows."""
    if helpers is None:
        helpers = _import_ord_schema()
    if helpers is None:
        return []

    try:
        dataset = helpers["message_helpers"].load_message(
            pb_gz_path, helpers["dataset_pb2"].Dataset)
    except Exception as exc:  # noqa: BLE001
        # try manual gzip + ParseFromString
        try:
            with gzip.open(pb_gz_path, "rb") as handle:
                raw = handle.read()
            dataset = helpers["dataset_pb2"].Dataset()
            dataset.ParseFromString(raw)
        except Exception as exc2:  # noqa: BLE001
            print(f"[warn] parse failed for {pb_gz_path}: {exc} / {exc2}", file=sys.stderr)
            return []

    rows: List[Dict[str, str]] = []
    for reaction in dataset.reactions:
        rid = getattr(reaction, "reaction_id", "") or ""
        rxn_smiles = _reaction_smiles_from_reaction(reaction, helpers)
        if not rxn_smiles or ">>" not in rxn_smiles and rxn_smiles.count(">") < 2:
            continue
        try:
            reactants, agents, products = split_reaction(rxn_smiles)
        except ValueError:
            continue
        if not reactants or not products:
            continue
        source_id = rid if rid else f"ord_{hashlib.md5(rxn_smiles.encode()).hexdigest()[:16]}"
        rows.append({
            "source_id": source_id,
            "reaction_smiles": rxn_smiles,
            "reactants": reactants,
            "agents": agents,
            "products": products,
            "label_type": "positive",
            "yield": _extract_yield(reaction),
            "source": source_tag,
            "split_key": hashlib.md5(rxn_smiles.encode()).hexdigest()[:10],
            "split": "train",  # ORD has no native split; default train
        })
    return rows


# --------------------------------------------------------------------------- #
# Normalization + dedup + summary
# --------------------------------------------------------------------------- #

def normalize_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Dedupe by canonical reaction SMILES, drop invalid reactions."""
    seen = set()
    out: List[Dict[str, str]] = []
    for row in rows:
        rxn = row.get("reaction_smiles", "")
        if not is_valid_reaction(rxn):
            row = dict(row)
            row["_smiles_valid"] = "0"
            out.append(row)
            continue
        # Parse reactants/products FROM the reaction_smiles (source of truth)
        try:
            reactants, _, products = split_reaction(rxn)
        except ValueError:
            row = dict(row)
            row["_smiles_valid"] = "0"
            out.append(row)
            continue
        canon_r = canonicalize_smiles(reactants) or reactants
        canon_p = canonicalize_smiles(products) or products
        canon = f"{canon_r}>>{canon_p}"
        if canon in seen:
            continue
        seen.add(canon)
        row = dict(row)
        # refresh reactants/products from the parsed reaction_smiles
        row["reactants"] = reactants
        row["products"] = products
        row["_smiles_valid"] = "1"
        out.append(row)
    return out


def build_summary(rows: List[Dict[str, str]], source_tag: str,
                  n_datasets_attempted: int, n_datasets_parsed: int,
                  fallback_used: bool) -> Dict[str, object]:
    n = len(rows)
    non_empty = {f: sum(1 for r in rows if r.get(f, "")) for f in NORMALIZED_FIELDS}
    valid_count = sum(1 for r in rows if r.get("_smiles_valid", "1") == "1")
    # atom-mapping coverage: presence of numeric class labels like [C:1]
    mapped = sum(1 for r in rows if "[" in r.get("reaction_smiles", "") and ":]" in r.get("reaction_smiles", ""))
    return {
        "source": source_tag,
        "total_records": n,
        "datasets_attempted": n_datasets_attempted,
        "datasets_parsed": n_datasets_parsed,
        "fallback_used": fallback_used,
        "field_non_empty_counts": non_empty,
        "field_non_empty_rates": {f: (v / n if n else 0.0) for f, v in non_empty.items()},
        "smiles_validity_rate": (valid_count / n) if n else 0.0,
        "atom_mapping_coverage_rate": (mapped / n) if n else 0.0,
        "label_type_distribution": _value_counts(rows, "label_type"),
        "source_distribution": _value_counts(rows, "source"),
        "split_distribution": _value_counts(rows, "split"),
    }


def _value_counts(rows: List[Dict[str, str]], field: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        v = row.get(field, "") or "<empty>"
        counts[v] = counts.get(v, 0) + 1
    return counts


# --------------------------------------------------------------------------- #
# Fallback: USPTO-proxy subset
# --------------------------------------------------------------------------- #

def uspto_proxy_fallback(uspto_csv: str, n: int, seed: int) -> List[Dict[str, str]]:
    """Deterministic sample of USPTO normalized rows, relabelled as
    ``ord_subset_uspto_proxy``.  Used only when real ORD parsing is blocked.
    """
    if not uspto_csv or not os.path.isfile(uspto_csv):
        return []
    rows, _ = read_csv(uspto_csv)
    rng = random.Random(seed)
    if len(rows) > n:
        rows = rng.sample(rows, n)
    out: List[Dict[str, str]] = []
    for row in rows:
        out.append({
            "source_id": row.get("source_id", ""),
            "reaction_smiles": row.get("reaction_smiles", ""),
            "reactants": row.get("reactants", ""),
            "agents": row.get("agents", ""),
            "products": row.get("products", ""),
            "label_type": "positive",
            "yield": row.get("yield", ""),
            "source": "ord_subset_uspto_proxy",
            "split_key": row.get("split_key", ""),
            "split": row.get("split", "train"),
            "_smiles_valid": "1",
        })
    return out


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def build_ord(
    output_dir: str,
    max_datasets: int = DEFAULT_MAX_DATASETS,
    shards: Optional[List[str]] = None,
    cache_dir: Optional[str] = None,
    uspto_fallback_csv: Optional[str] = None,
    fallback_size: int = DEFAULT_FALLBACK_SIZE,
    seed: int = DEFAULT_SEED,
) -> Tuple[List[Dict[str, str]], Dict[str, object]]:
    """Build the normalized ORD CSV.

    Returns ``(rows, summary)``.  Writes ``ord_normalized.csv`` and
    ``ord_normalized_summary.json`` (in ``output_dir`` and the sibling
    ``../summaries`` dir) as a side effect.
    """
    shards = shards or DEFAULT_SHARDS
    cache_dir = cache_dir or os.path.join(output_dir, "_ord_cache")
    os.makedirs(cache_dir, exist_ok=True)

    helpers = _import_ord_schema()
    all_rows: List[Dict[str, str]] = []
    n_attempted = 0
    n_parsed = 0
    fallback_used = False
    source_tag = "ord_open_reaction_database"

    if helpers is not None:
        dataset_list = fetch_dataset_list(shards, max_datasets)
        n_attempted = len(dataset_list)
        for entry in dataset_list:
            dest = os.path.join(cache_dir, entry["name"])
            if not os.path.isfile(dest) or os.path.getsize(dest) < 200:
                if not download_dataset(entry["path"], dest):
                    continue
            parsed = parse_dataset_file(dest, source_tag, helpers=helpers)
            if parsed:
                n_parsed += 1
                all_rows.extend(parsed)
                print(f"[info] parsed {len(parsed):>6d} reactions from {entry['name']}")
            if len(all_rows) >= 1000:
                break  # we have enough

    if len(all_rows) < 1000:
        fallback_used = True
        source_tag = "ord_subset_uspto_proxy"
        proxy = uspto_proxy_fallback(uspto_fallback_csv or "", fallback_size, seed)
        if proxy:
            print(f"[warn] ORD parsing yielded {len(all_rows)} rows (<1000); "
                  f"appending {len(proxy)} USPTO-proxy fallback rows", file=sys.stderr)
            all_rows.extend(proxy)

    all_rows = normalize_rows(all_rows)
    summary = build_summary(all_rows, source_tag, n_attempted, n_parsed, fallback_used)

    # write outputs
    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(output_dir, "ord_normalized.csv")
    write_csv(out_csv, all_rows, NORMALIZED_FIELDS)

    summaries_dir = os.path.join(os.path.dirname(output_dir.rstrip("/")), "summaries")
    os.makedirs(summaries_dir, exist_ok=True)
    summary_path = os.path.join(summaries_dir, "ord_normalized_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(f"[info] wrote {len(all_rows)} normalized ORD rows to {out_csv}")
    print(f"[info] wrote summary to {summary_path}")
    return all_rows, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/processed",
                        help="Output dir for ord_normalized.csv")
    parser.add_argument("--max-datasets", type=int, default=DEFAULT_MAX_DATASETS,
                        help="Max ORD dataset files to download (each ~750 reactions)")
    parser.add_argument("--shards", default=",".join(DEFAULT_SHARDS),
                        help="Comma-separated shard prefixes to scan (e.g. 00,01,02)")
    parser.add_argument("--cache-dir", default=None,
                        help="Cache dir for downloaded .pb.gz files")
    parser.add_argument("--uspto-fallback-csv",
                        default="data/processed/uspto_openmolecules_normalized.csv",
                        help="USPTO normalized CSV for the proxy fallback")
    parser.add_argument("--fallback-size", type=int, default=DEFAULT_FALLBACK_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    shards = [s.strip() for s in args.shards.split(",") if s.strip()]
    rows, summary = build_ord(
        output_dir=args.output_dir,
        max_datasets=args.max_datasets,
        shards=shards,
        cache_dir=args.cache_dir,
        uspto_fallback_csv=args.uspto_fallback_csv,
        fallback_size=args.fallback_size,
        seed=args.seed,
    )
    print(f"[result] total ORD rows: {len(rows)}")
    print(f"[result] smiles_validity_rate: {summary.get('smiles_validity_rate', 0.0):.4f}")
    print(f"[result] fallback_used: {summary.get('fallback_used')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
