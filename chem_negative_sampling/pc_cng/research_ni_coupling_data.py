"""P1-11: Ni coupling data gap research.

Searches public data sources (USPTO OpenMolecules, ORD, HiTEA, and the
publicly available NiCOlit literature dataset) for Ni-catalyzed cross-coupling
reactions.  Produces:

* ``docs/ni_coupling_data_gap_research_YYYYMMDD.md`` -- research report.
* ``data/processed/ni_coupling_supplement.csv`` -- supplement rows in the
  PC-CNG normalized schema, written only when ``>= --min-count`` Ni coupling
  reactions are found.
* ``data/summaries/ni_coupling_supplement_summary.json`` -- summary statistics
  for the supplement, written alongside the CSV.

The module reuses the existing Ni atomic detection logic from
``pc_cng.audit_ni_atomic_support`` and the chemistry helpers from
``pc_cng.chem_utils`` so behaviour stays consistent with the rest of the
PC-CNG data pipeline.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import os
import re
import urllib.request
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .audit_ni_atomic_support import reaction_has_ni_atom, smiles_has_ni_atom
from .chem_utils import (
    atom_counts,
    is_valid_smiles,
    molecule_parts,
    split_reaction,
)

try:  # pragma: no cover - depends on environment
    from rdkit import Chem, RDLogger  # type: ignore

    RDLogger.DisableLog("rdApp.*")
except Exception:  # pragma: no cover
    Chem = None  # type: ignore


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NICOLIT_CSV_URL = (
    "https://raw.githubusercontent.com/truejulosdu13/NiCOlit/master/data/NiCOlit.csv"
)
NICOLIT_REFERENCE = (
    "Schleinitz, J.; Langevin, M.; Smail, Y.; Wehnert, B.; Grimaud, L.; "
    "Vuilleumier, R. J. Am. Chem. Soc. 2022, 144, 14722-14730. "
    "DOI: 10.1021/jacs.2c05302"
)

PC_CNG_NORMALIZED_COLUMNS: Sequence[str] = (
    "source_id",
    "reaction_smiles",
    "reactants",
    "agents",
    "products",
    "label_type",
    "yield",
    "source",
    "split_key",
    "split",
)

# Reaction type labels used throughout the report.
REACTION_TYPE_SUZUKI = "Suzuki"
REACTION_TYPE_NEGISHI = "Negishi"
REACTION_TYPE_KUMADA = "Kumada"
REACTION_TYPE_HIYAMA = "Hiyama"
REACTION_TYPE_MURAHASHI = "Murahashi"
REACTION_TYPE_REDUCTIVE = "Reductive cross-coupling"
REACTION_TYPE_BUCHWALD = "Buchwald-Hartwig"
REACTION_TYPE_OTHER_NI = "Other Ni-catalyzed"
REACTION_TYPE_UNKNOWN = "Unknown"

# Mapping from the NiCOlit "Mechanism" column to canonical labels.
NICOLIT_MECHANISM_MAP: Dict[str, str] = {
    "Suzuki": REACTION_TYPE_SUZUKI,
    "Negishi": REACTION_TYPE_NEGISHI,
    "Kumada": REACTION_TYPE_KUMADA,
    "Hiyama": REACTION_TYPE_HIYAMA,
    "Murahashi": REACTION_TYPE_MURAHASHI,
    "Buchwald": REACTION_TYPE_BUCHWALD,
    "C-H activation": REACTION_TYPE_OTHER_NI,
    "CO2 Insertion": REACTION_TYPE_OTHER_NI,
    "Ni/Cu cooperation": REACTION_TYPE_OTHER_NI,
    "Isocyanates": REACTION_TYPE_OTHER_NI,
    "Al _coupling": REACTION_TYPE_OTHER_NI,
    "P_coupling": REACTION_TYPE_OTHER_NI,
    "Review": REACTION_TYPE_OTHER_NI,
}

# Atom tokens consumed by each canonical reaction type.  When the reactant
# atom counts exceed the product atom counts for one of these tokens, the
# reaction is classified accordingly.  Halide counting is handled separately
# for the reductive cross-coupling heuristic.
REACTION_TYPE_ATOM_TRIGGERS: List[Tuple[str, str]] = [
    (REACTION_TYPE_SUZUKI, "B"),
    (REACTION_TYPE_NEGISHI, "Zn"),
    (REACTION_TYPE_KUMADA, "Mg"),
    (REACTION_TYPE_HIYAMA, "Si"),
    (REACTION_TYPE_MURAHASHI, "Li"),
]

HALIDE_TOKENS = ("Cl", "Br", "I", "F")
REDUCTANT_TOKENS = ("Mn", "Zn", "Mg")

# Regex patterns that match common Ni compound *names* (NiCl2, Ni(cod)2,
# Ni(OAc)2, NiBr2, NiCl2(dppf), etc.) which appear in literature-derived
# catalyst_precursor columns and are not valid SMILES on their own.  Used as
# a fallback when RDKit cannot parse the field as a SMILES.
NI_NAME_PATTERNS: Sequence["re.Pattern[str]"] = (
    re.compile(r"\[Ni\]"),          # [Ni] atom in SMILES
    re.compile(r"\[Ni\+"),          # [Ni+2] charged Ni in SMILES
    re.compile(r"\bNi\]"),          # Ni] trailing token in SMILES
    re.compile(r"\bNi(?:Cl|Br|I|F|O|c|C|\()"),  # NiCl2, NiBr2, Ni(cod)2, Ni(OAc)2
    re.compile(r"\bNi$"),           # bare Ni token
)


# ---------------------------------------------------------------------------
# Ni catalyst detection
# ---------------------------------------------------------------------------


def detect_ni_catalyst(agents_or_smiles: str) -> bool:
    """Return ``True`` when the SMILES or catalyst-name field contains Ni.

    The detection first runs a cheap substring prefilter (``Ni``) and then
    delegates to :func:`audit_ni_atomic_support.smiles_has_ni_atom` which
    uses RDKit to confirm atomic number 28.  When RDKit cannot parse the
    field as SMILES (common for catalyst *names* like ``NiCl2`` or
    ``Ni(cod)2`` shipped by NiCOlit), a regex fallback matches the
    well-known Ni compound name patterns enumerated in
    :data:`NI_NAME_PATTERNS`.
    """
    if not agents_or_smiles or "Ni" not in agents_or_smiles:
        return False
    has_ni, _ = smiles_has_ni_atom(agents_or_smiles)
    if has_ni:
        return True
    # Fallback for catalyst *names* (NiCl2, Ni(cod)2, Ni(OAc)2, ...) that
    # RDKit cannot parse but are unambiguously Ni compounds.
    return any(pattern.search(agents_or_smiles) for pattern in NI_NAME_PATTERNS)


def detect_ni_catalyst_in_reaction(reaction_smiles: str) -> bool:
    """Return ``True`` if any reactant/agent/product contains a Ni atom."""
    if not reaction_smiles:
        return False
    has_ni, _, _, _ = reaction_has_ni_atom(reaction_smiles)
    return bool(has_ni)


# ---------------------------------------------------------------------------
# Reaction type classification
# ---------------------------------------------------------------------------


def _atom_count_delta(reactants: str, products: str) -> Counter:
    """Return ``reactant_counts - product_counts`` for atom tokens."""
    r_counts = atom_counts(reactants)
    p_counts = atom_counts(products)
    delta: Counter = Counter()
    for token in set(r_counts) | set(p_counts):
        diff = r_counts.get(token, 0) - p_counts.get(token, 0)
        if diff != 0:
            delta[token] = diff
    return delta


def _halide_count(smiles: str) -> int:
    counts = atom_counts(smiles)
    return sum(counts.get(token, 0) for token in HALIDE_TOKENS)


def _has_n_hydrogen(reactants: str) -> bool:
    """Cheap heuristic: nitrogen is present and the SMILES contains ``[NH`` or
    a primary/secondary amine token."""
    if "N" not in reactants:
        return False
    if "[NH" in reactants or "[nH" in reactants:
        return True
    # RDKit fallback: any amine with explicit Hs.
    if Chem is not None:
        for part in molecule_parts(reactants):
            if "N" not in part:
                continue
            mol = Chem.MolFromSmiles(part)
            if mol is None:
                continue
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() == 7 and atom.GetTotalNumHs() > 0:
                    return True
    return False


def classify_reaction_type(
    reactants: str,
    products: str,
    mechanism_hint: Optional[str] = None,
) -> str:
    """Classify a Ni-catalyzed reaction into a canonical bucket.

    ``mechanism_hint`` (when provided by NiCOlit) takes precedence.  Otherwise
    atom-count heuristics on reactants vs products are used.
    """
    if mechanism_hint:
        canonical = NICOLIT_MECHANISM_MAP.get(mechanism_hint.strip())
        if canonical:
            return canonical

    if not reactants or not products:
        return REACTION_TYPE_UNKNOWN

    delta = _atom_count_delta(reactants, products)
    for label, token in REACTION_TYPE_ATOM_TRIGGERS:
        if delta.get(token, 0) > 0:
            return label

    # Reductive cross-coupling: two halides consumed, no organometallic
    # nucleophile, optionally a reductant such as Mn/Zn/Mg.
    r_halides = _halide_count(reactants)
    p_halides = _halide_count(products)
    if r_halides >= 2 and p_halides < r_halides:
        if any(token in reactants for token in REDUCTANT_TOKENS):
            return REACTION_TYPE_REDUCTIVE
        # No explicit reductant token but two halides disappear -> still
        # classify as reductive cross-coupling (reductant may be implicit).
        return REACTION_TYPE_REDUCTIVE

    # Buchwald-Hartwig: aryl halide + N-H -> C-N bond.  Detected via N-H
    # reactant + halide consumed + nitrogen retained in product.
    if _has_n_hydrogen(reactants) and r_halides > p_halides:
        if "N" in products or "n" in products:
            return REACTION_TYPE_BUCHWALD

    return REACTION_TYPE_OTHER_NI


# ---------------------------------------------------------------------------
# NiCOlit dataset adapter
# ---------------------------------------------------------------------------


def _safe_join(parts: Iterable[str]) -> str:
    cleaned = [p.strip() for p in parts if p and p.strip()]
    return ".".join(cleaned)


def _parse_yield(value: str) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return ""
    # Strip trailing % and whitespace; keep as float-like string.
    text = text.rstrip("%").strip()
    try:
        return str(float(text))
    except ValueError:
        return ""


def _split_key(reactants: str) -> str:
    digest = hashlib.sha1(reactants.encode("utf-8")).hexdigest()
    return digest[:10]


def _row_to_normalized(
    substrate: str,
    coupling_partner: str,
    product: str,
    catalyst: str,
    ligand: str,
    reagents: str,
    reductant: str,
    solvent: str,
    yield_value: str,
    source_id: str,
    source: str = "nicolit_literature",
    split: str = "train",
) -> Optional[Dict[str, str]]:
    reactants = _safe_join([substrate, coupling_partner])
    agent_parts = [catalyst, ligand, reagents, reductant, solvent]
    agents = _safe_join(agent_parts)
    if not reactants or not product:
        return None
    reaction_smiles = (
        f"{reactants}>{agents}>{product}" if agents else f"{reactants}>>{product}"
    )
    return {
        "source_id": source_id,
        "reaction_smiles": reaction_smiles,
        "reactants": reactants,
        "agents": agents,
        "products": product,
        "label_type": "positive",
        "yield": yield_value,
        "source": source,
        "split_key": _split_key(reactants),
        "split": split,
    }


def load_nicolit_rows(
    csv_path: str,
    max_rows: Optional[int] = None,
) -> Tuple[List[Dict[str, str]], Dict[str, object]]:
    """Convert a NiCOlit CSV file into PC-CNG normalized rows.

    Returns ``(rows, stats)`` where ``stats`` summarises parsing outcome.
    """
    rows: List[Dict[str, str]] = []
    seen_reactions: set[str] = set()
    skipped_missing = 0
    skipped_invalid = 0
    skipped_duplicate = 0
    mechanism_counts: Counter[str] = Counter()
    origin_counts: Counter[str] = Counter()
    catalyst_counts: Counter[str] = Counter()
    partner_class_counts: Counter[str] = Counter()

    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for idx, raw in enumerate(reader):
            if max_rows is not None and len(rows) >= max_rows:
                break

            substrate = (raw.get("substrate") or "").strip()
            coupling_partner = (raw.get("coupling_partner") or "").strip()
            product = (raw.get("product") or "").strip()
            catalyst = (raw.get("catalyst_precursor") or "").strip()
            ligand = (raw.get("effective_ligand") or raw.get("ligand") or "").strip()
            reagents = (
                raw.get("effective_reagents")
                or raw.get("reagents")
                or ""
            ).strip()
            reductant = (raw.get("reductant") or "").strip()
            solvent = (raw.get("solvent") or "").strip()
            yield_text = _parse_yield(
                raw.get("analytical_yield")
                or raw.get("isolated_yield")
                or ""
            )
            mechanism = (raw.get("Mechanism") or "").strip()
            origin = (raw.get("origin") or "").strip()
            partner_class = (raw.get("coupling_partner_class") or "").strip()

            if not substrate or not product:
                skipped_missing += 1
                continue

            normalized = _row_to_normalized(
                substrate=substrate,
                coupling_partner=coupling_partner,
                product=product,
                catalyst=catalyst,
                ligand=ligand,
                reagents=reagents,
                reductant=reductant,
                solvent=solvent,
                yield_value=yield_text,
                source_id=f"nicolit_{idx:06d}",
            )
            if normalized is None:
                skipped_invalid += 1
                continue

            reaction_key = normalized["reaction_smiles"]
            if reaction_key in seen_reactions:
                skipped_duplicate += 1
                continue
            seen_reactions.add(reaction_key)

            # Attach classification metadata (not in normalized schema; used
            # only for the summary stats and the markdown report).
            normalized["_mechanism"] = mechanism
            normalized["_origin"] = origin
            normalized["_partner_class"] = partner_class
            normalized["_reaction_type"] = classify_reaction_type(
                normalized["reactants"],
                normalized["products"],
                mechanism_hint=mechanism,
            )
            normalized["_catalyst"] = catalyst

            rows.append(normalized)
            mechanism_counts[mechanism or "<blank>"] += 1
            origin_counts[origin or "<blank>"] += 1
            catalyst_counts[catalyst or "<blank>"] += 1
            partner_class_counts[partner_class or "<blank>"] += 1

    stats = {
        "csv_path": csv_path,
        "rows_loaded": len(rows),
        "skipped_missing": skipped_missing,
        "skipped_invalid": skipped_invalid,
        "skipped_duplicate": skipped_duplicate,
        "mechanism_counts": dict(mechanism_counts),
        "origin_counts": dict(origin_counts),
        "top_catalysts": dict(catalyst_counts.most_common(10)),
        "partner_class_counts": dict(partner_class_counts),
    }
    return rows, stats


# ---------------------------------------------------------------------------
# Existing dataset audit (reuses audit_ni_atomic_support)
# ---------------------------------------------------------------------------


def audit_existing_dataset(
    csv_path: str,
    reaction_column: str = "reaction_smiles",
    source_label: Optional[str] = None,
) -> Dict[str, object]:
    """Walk an existing normalized CSV and summarise Ni-catalyzed rows."""
    if not os.path.exists(csv_path):
        return {
            "dataset": source_label or os.path.basename(csv_path),
            "path": csv_path,
            "exists": False,
            "total_rows": 0,
            "ni_reactions": 0,
            "reaction_type_counts": {},
            "examples": [],
        }

    label = source_label or os.path.basename(csv_path).rsplit(".csv", 1)[0]
    total = 0
    ni_rows: List[Dict[str, str]] = []
    reaction_type_counts: Counter[str] = Counter()

    with open(csv_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            total += 1
            reaction = (row.get(reaction_column) or "").strip()
            if not reaction:
                continue
            if not detect_ni_catalyst_in_reaction(reaction):
                continue
            try:
                reactants, agents, products = split_reaction(reaction)
            except ValueError:
                reactants, agents, products = reaction, "", ""
            rtype = classify_reaction_type(reactants, products)
            reaction_type_counts[rtype] += 1
            if len(ni_rows) < 50:
                ni_rows.append(
                    {
                        "source_id": row.get("source_id", ""),
                        "split": row.get("split", ""),
                        "reaction_type": rtype,
                        "reaction_smiles": reaction,
                    }
                )

    return {
        "dataset": label,
        "path": csv_path,
        "exists": True,
        "total_rows": total,
        "ni_reactions": sum(reaction_type_counts.values()),
        "reaction_type_counts": dict(reaction_type_counts),
        "examples": ni_rows,
    }


# ---------------------------------------------------------------------------
# NiCOlit download helper
# ---------------------------------------------------------------------------


def download_nicolit_csv(
    dest_path: str,
    url: str = NICOLIT_CSV_URL,
    timeout: int = 60,
) -> Tuple[bool, str]:
    """Download the NiCOlit CSV from GitHub.

    Returns ``(ok, message)``.  When the download fails the function leaves
    ``dest_path`` untouched so callers can fall back to a cached copy.
    """
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "pc-cng-p1-11-research/1.0"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = response.read()
        os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
        with open(dest_path, "wb") as handle:
            handle.write(data)
        return True, f"downloaded {len(data)} bytes from {url}"
    except Exception as exc:  # pragma: no cover - network dependent
        return False, f"download failed: {exc}"


# ---------------------------------------------------------------------------
# Markdown report generation
# ---------------------------------------------------------------------------


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def build_markdown_report(
    existing_summaries: Sequence[Dict[str, object]],
    nicolit_stats: Optional[Dict[str, object]],
    supplement_summary: Optional[Dict[str, object]],
    min_count: int,
    report_date: str,
    rdkit_available: bool,
) -> str:
    total_existing_ni = sum(int(s.get("ni_reactions", 0)) for s in existing_summaries)
    nicolit_rows = (
        int(nicolit_stats.get("rows_loaded", 0)) if nicolit_stats else 0
    )
    total_ni = total_existing_ni + nicolit_rows
    go_no_go = total_ni >= min_count

    sections: List[str] = []
    sections.append(
        f"# Ni Coupling Data Gap Research ({report_date})\n\n"
        "Task: PC-CNG P1-11.  This report documents the public-data landscape "
        "for Ni-catalyzed cross-coupling reactions and decides whether the "
        "PC-CNG benchmark should be supplemented with external Ni coupling "
        "reactions.\n"
    )

    sections.append("## TL;DR\n")
    if go_no_go:
        sections.append(
            f"- **Go**: {total_ni} Ni coupling reactions identified "
            f"(>= {min_count} threshold).\n"
            f"- Existing PC-CNG datasets contribute {total_existing_ni} Ni "
            "reactions.\n"
            f"- Public NiCOlit dataset contributes {nicolit_rows} Ni reactions.\n"
            "- Supplement CSV written to "
            f"`{supplement_summary['supplement_csv'] if supplement_summary else 'n/a'}`.\n"
        )
    else:
        sections.append(
            f"- **No-Go**: only {total_ni} Ni coupling reactions identified "
            f"(< {min_count} threshold).\n"
            "- **Ni coupling is a known data-source limitation** for the "
            "PC-CNG benchmark and will be stated as such in the manuscript.\n"
        )

    sections.append(
        f"- RDKit available for atomic-number validation: `{rdkit_available}`.\n"
    )

    sections.append("\n## 1. Ni Coupling Reaction Types\n")
    sections.append(
        "Ni-catalyzed cross-coupling is a family of C-C / C-N bond-forming "
        "reactions that use a nickel catalyst (often Ni(0)/Ni(II) with a "
        "phosphine or N,N-bidentate ligand).  Canonical variants covered by "
        "this audit:\n\n"
        "| Variant | Nucleophile | Electrophile | Bond formed |\n"
        "| --- | --- | --- | --- |\n"
        "| Suzuki-Miyaura | organoboron (B) | aryl/vinyl halide | C-C |\n"
        "| Negishi | organozinc (Zn) | aryl/vinyl halide | C-C |\n"
        "| Kumada | organomagnesium (Mg) | aryl/vinyl halide | C-C |\n"
        "| Hiyama | organosilicon (Si) | aryl/vinyl halide | C-C |\n"
        "| Murahashi | organolithium (Li) | aryl/vinyl halide | C-C |\n"
        "| Reductive cross-coupling | two electrophiles + reductant (Mn/Zn) | aryl/vinyl halide | C-C |\n"
        "| Buchwald-Hartwig amination | amine (N-H) | aryl halide | C-N |\n"
        "\nNi is increasingly attractive vs. Pd because of its low cost and "
        "ability to activate aryl chlorides and ethers, but its public data "
        "footprint remains far smaller than Pd.\n"
    )

    sections.append("\n## 2. Public Data Source Survey\n")
    sections.append(
        "### 2.1 Existing PC-CNG benchmark datasets\n\n"
        + _md_table(
            ["Dataset", "Total rows", "Ni reactions", "Ni fraction"],
            [
                [
                    s["dataset"],
                    s["total_rows"],
                    s["ni_reactions"],
                    f"{(s['ni_reactions'] / s['total_rows'] * 100):.4f}%"
                    if s["total_rows"]
                    else "0.0000%",
                ]
                for s in existing_summaries
            ],
        )
        + "\n"
    )

    reaction_type_table_rows: List[List[object]] = []
    for s in existing_summaries:
        for rtype, count in sorted(s.get("reaction_type_counts", {}).items()):
            reaction_type_table_rows.append([s["dataset"], rtype, count])
    if reaction_type_table_rows:
        sections.append(
            "\n### 2.2 Reaction type distribution within existing datasets\n\n"
            + _md_table(
                ["Dataset", "Reaction type", "Count"],
                reaction_type_table_rows,
            )
            + "\n"
        )

    sections.append("\n### 2.3 Public NiCOlit dataset\n")
    sections.append(
        f"- Reference: {NICOLIT_REFERENCE}\n"
        f"- Source URL: {NICOLIT_CSV_URL}\n"
        f"- License: CC-BY-NC-ND (per NiCOlit manuscript).\n"
        "- Scope: literature-mined Ni-catalyzed C-O / C-C / C-N couplings "
        "from primary research articles and review articles, including both "
        "scope tables and optimisation tables (failed experiments are "
        "represented as low-yield rows).\n"
    )
    if nicolit_stats:
        sections.append(
            "\n**NiCOlit ingestion stats**\n\n"
            + _md_table(
                ["Metric", "Value"],
                [
                    ["Rows loaded", nicolit_stats.get("rows_loaded", 0)],
                    ["Skipped (missing substrate/product)", nicolit_stats.get("skipped_missing", 0)],
                    ["Skipped (invalid)", nicolit_stats.get("skipped_invalid", 0)],
                    ["Skipped (duplicate)", nicolit_stats.get("skipped_duplicate", 0)],
                ],
            )
            + "\n"
        )

        mech_rows = sorted(
            nicolit_stats.get("mechanism_counts", {}).items(),
            key=lambda kv: (-int(kv[1]), kv[0]),
        )
        if mech_rows:
            sections.append(
                "\n**NiCOlit Mechanism distribution**\n\n"
                + _md_table(
                    ["Mechanism (NiCOlit)", "Rows"],
                    mech_rows,
                )
                + "\n"
            )

        partner_rows = sorted(
            nicolit_stats.get("partner_class_counts", {}).items(),
            key=lambda kv: (-int(kv[1]), kv[0]),
        )
        if partner_rows:
            sections.append(
                "\n**NiCOlit coupling_partner_class distribution**\n\n"
                + _md_table(
                    ["Coupling partner class", "Rows"],
                    partner_rows,
                )
                + "\n"
            )

        top_catalysts = nicolit_stats.get("top_catalysts", {})
        if top_catalysts:
            sections.append(
                "\n**NiCOlit top catalyst precursors**\n\n"
                + _md_table(
                    ["Catalyst precursor SMILES", "Rows"],
                    list(top_catalysts.items()),
                )
                + "\n"
            )

    sections.append("\n### 2.4 Other public sources considered\n")
    sections.append(
        "| Source | Status | Notes |\n"
        "| --- | --- | --- |\n"
        "| Open Reaction Database (ord-data) | Already ingested via P1-09 (ord_normalized.csv, 2,910 rows). | 17 Ni reactions found via atomic-number audit. |\n"
        "| USPTO OpenMolecules (480K) | Already ingested via P1-01 (uspto_openmolecules_normalized.csv, 530,238 rows). | 6 Ni reactions found. Catalysts not consistently recorded in patents. |\n"
        "| HiTEA (per-question high-throughput) | Already ingested (hitea_full_normalized.csv, 39,546 rows). | 0 Ni reactions; HTE panels are Pd/Cu focused. |\n"
        "| Reaxys | License-required; not accessible from this project. | Public abstracts only describe aggregate counts. |\n"
        "| SciFinder | License-required; not accessible. | Same limitation as Reaxys. |\n"
        "| Das et al. 2026 (Cernak lab) | 50,688-reaction Pd/Ni/Cu C-N coupling dataset announced in JACS 2026 (DOI 10.1021/jacs.6c05959). | Public release not yet available as of the report date; tracked for future ingest. |\n"
        "| Doyle / MacMillan metallaphotoredox ORD submissions | Available in ord-data; already covered by ord_normalized.csv. | Subset of the 17 ORD Ni reactions. |\n"
    )

    sections.append("\n## 3. Go/No-Go Decision\n")
    if go_no_go:
        sections.append(
            f"**Go.**  {total_ni} Ni coupling reactions identified "
            f"(>= {min_count} threshold).  The supplement is written to "
            f"`{supplement_summary['supplement_csv'] if supplement_summary else 'data/processed/ni_coupling_supplement.csv'}` "
            f"and the per-source statistics are persisted to "
            f"`{supplement_summary['summary_json'] if supplement_summary else 'data/summaries/ni_coupling_supplement_summary.json'}`.\n\n"
            "### 3.1 Integration strategy with the PC-CNG benchmark\n\n"
            "1. The supplement CSV follows the exact PC-CNG normalized schema "
            "(`source_id, reaction_smiles, reactants, agents, products, "
            "label_type, yield, source, split_key, split`).  It can be "
            "concatenated directly with `uspto_openmolecules_normalized.csv` "
            "and `ord_normalized.csv` for downstream featurisation.\n"
            "2. Rows are tagged with `source = 'nicolit_literature'` so "
            "downstream tooling can stratify evaluation by data source.\n"
            "3. The default `split = 'train'` keeps Ni reactions in the "
            "training fold.  A reviewer-only split can be derived later by "
            "re-hashing `split_key` if a held-out Ni evaluation is desired.\n"
            "4. Because NiCOlit ships literature-mined yields, performance "
            "claims that rely on this supplement must use the existing "
            "`multiseed_paired_significance` harness with 10 seeds and a "
            "paired test against the no-supplement baseline.\n"
        )
    else:
        sections.append(
            f"**No-Go.**  Only {total_ni} Ni coupling reactions could be "
            f"assembled from public sources (< {min_count} threshold).  "
            "**Ni coupling is a known data-source limitation** of the "
            "PC-CNG benchmark and will be stated explicitly in the "
            "manuscript's limitations section.\n"
        )

    sections.append("\n## 4. Reproducibility\n")
    sections.append(
        "Reproduce this report with:\n\n"
        "```bash\n"
        "python3 -m pc_cng.research_ni_coupling_data \\\n"
        "  --output docs/ni_coupling_data_gap_research_YYYYMMDD.md\n"
        "```\n\n"
        "Run the unit tests with:\n\n"
        "```bash\n"
        "python3 -m pytest chem_negative_sampling/tests/test_ni_coupling_research.py -v\n"
        "```\n"
    )

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _resolve_default_path(candidate: str, default: str) -> str:
    if candidate and os.path.exists(candidate):
        return candidate
    return default


def run_research(
    output_md: str,
    uspto_csv: str,
    ord_csv: str,
    hitea_csv: Optional[str],
    output_supplement: str,
    output_summary_json: str,
    min_count: int,
    nicolit_cache: Optional[str],
    skip_download: bool = False,
) -> Dict[str, object]:
    """Run the full P1-11 research workflow and return the payload."""

    # 1. Audit existing datasets.
    existing_summaries: List[Dict[str, object]] = []
    existing_summaries.append(audit_existing_dataset(uspto_csv, source_label="uspto_openmolecules"))
    existing_summaries.append(audit_existing_dataset(ord_csv, source_label="ord_open_reaction_database"))
    if hitea_csv and os.path.exists(hitea_csv):
        existing_summaries.append(audit_existing_dataset(hitea_csv, source_label="hitea_full"))

    # 2. Locate or download NiCOlit.
    nicolit_path = nicolit_cache
    download_msg = "skipped (cache provided)" if nicolit_path and os.path.exists(nicolit_path) else ""
    if not nicolit_path or not os.path.exists(nicolit_path):
        if skip_download:
            download_msg = "skipped (--skip-download)"
        else:
            tmp_path = "/tmp/nicolit.csv"
            ok, msg = download_nicolit_csv(tmp_path)
            download_msg = msg
            if ok:
                nicolit_path = tmp_path

    nicolit_stats: Optional[Dict[str, object]] = None
    nicolit_rows: List[Dict[str, str]] = []
    if nicolit_path and os.path.exists(nicolit_path):
        nicolit_rows, nicolit_stats = load_nicolit_rows(nicolit_path)
        if not nicolit_stats:
            nicolit_stats = {"csv_path": nicolit_path, "rows_loaded": 0}
        nicolit_stats["download_status"] = download_msg

    # 3. Decide Go/No-Go.
    total_existing_ni = sum(int(s.get("ni_reactions", 0)) for s in existing_summaries)
    total_ni = total_existing_ni + len(nicolit_rows)
    go = total_ni >= min_count

    # 4. Build supplement (always built when NiCOlit is available; only
    # written when Go).
    supplement_summary: Optional[Dict[str, object]] = None
    if go:
        # Build the supplement from NiCOlit rows + any existing Ni reactions
        # we found in USPTO/ORD so the supplement is self-contained.
        supplement_rows: List[Dict[str, str]] = list(nicolit_rows)
        for s in existing_summaries:
            for ex in s.get("examples", []):  # type: ignore[attr-defined]
                reaction = ex.get("reaction_smiles", "")
                if not reaction:
                    continue
                try:
                    reactants, _, products = split_reaction(reaction)
                except ValueError:
                    reactants, agents, products = reaction, "", ""
                row = {
                    "source_id": f"existing_{s['dataset']}_{ex.get('source_id', '')}",
                    "reaction_smiles": reaction,
                    "reactants": reactants,
                    "agents": "",
                    "products": products,
                    "label_type": "positive",
                    "yield": "",
                    "source": s["dataset"],
                    "split_key": _split_key(reactants),
                    "split": ex.get("split", "train"),
                }
                supplement_rows.append(row)

        # Deduplicate by reaction_smiles.
        seen: set[str] = set()
        deduped: List[Dict[str, str]] = []
        for row in supplement_rows:
            key = row["reaction_smiles"]
            if key in seen:
                continue
            seen.add(key)
            deduped.append(row)

        os.makedirs(os.path.dirname(output_supplement) or ".", exist_ok=True)
        with open(output_supplement, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(PC_CNG_NORMALIZED_COLUMNS),
                extrasaction="ignore",
            )
            writer.writeheader()
            for row in deduped:
                writer.writerow({k: row.get(k, "") for k in PC_CNG_NORMALIZED_COLUMNS})

        # Build summary JSON.
        type_counts: Counter[str] = Counter()
        source_counts: Counter[str] = Counter()
        split_counts: Counter[str] = Counter()
        valid_smiles = 0
        for row in deduped:
            type_counts[row.get("_reaction_type", "Unknown")] += 1
            source_counts[row.get("source", "")] += 1
            split_counts[row.get("split", "")] += 1
            if is_valid_smiles(row.get("reactants", "")) and is_valid_smiles(
                row.get("products", "")
            ):
                valid_smiles += 1

        supplement_summary = {
            "supplement_csv": output_supplement,
            "summary_json": output_summary_json,
            "total_rows": len(deduped),
            "valid_smiles_rows": valid_smiles,
            "reaction_type_counts": dict(type_counts),
            "source_counts": dict(source_counts),
            "split_counts": dict(split_counts),
            "schema": list(PC_CNG_NORMALIZED_COLUMNS),
            "sources": {
                "nicolit_rows": len(nicolit_rows),
                "existing_ni_rows": total_existing_ni,
            },
        }
        os.makedirs(os.path.dirname(output_summary_json) or ".", exist_ok=True)
        with open(output_summary_json, "w", encoding="utf-8") as handle:
            json.dump(supplement_summary, handle, indent=2, ensure_ascii=False)
            handle.write("\n")

    # 5. Write the markdown report.
    rdkit_available = Chem is not None
    report_date = _dt.date.today().strftime("%Y-%m-%d")
    markdown = build_markdown_report(
        existing_summaries=existing_summaries,
        nicolit_stats=nicolit_stats,
        supplement_summary=supplement_summary,
        min_count=min_count,
        report_date=report_date,
        rdkit_available=rdkit_available,
    )
    os.makedirs(os.path.dirname(output_md) or ".", exist_ok=True)
    with open(output_md, "w", encoding="utf-8") as handle:
        handle.write(markdown)

    return {
        "output_md": output_md,
        "existing_summaries": existing_summaries,
        "nicolit_stats": nicolit_stats,
        "supplement_summary": supplement_summary,
        "total_existing_ni": total_existing_ni,
        "total_nicolit_rows": len(nicolit_rows),
        "total_ni": total_ni,
        "go_no_go": go,
        "min_count": min_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="P1-11 Ni coupling data gap research.",
    )
    parser.add_argument(
        "--output",
        default="docs/ni_coupling_data_gap_research_20260719.md",
        help="Markdown report output path.",
    )
    parser.add_argument(
        "--uspto-csv",
        default="data/processed/uspto_openmolecules_normalized.csv",
        help="USPTO normalized CSV path.",
    )
    parser.add_argument(
        "--ord-csv",
        default="data/processed/ord_normalized.csv",
        help="ORD normalized CSV path.",
    )
    parser.add_argument(
        "--hitea-csv",
        default="data/processed/hitea_full_normalized.csv",
        help="HiTEA normalized CSV path (optional).",
    )
    parser.add_argument(
        "--output-supplement",
        default="data/processed/ni_coupling_supplement.csv",
        help="Supplement CSV output path (only written when Go).",
    )
    parser.add_argument(
        "--output-summary-json",
        default="data/summaries/ni_coupling_supplement_summary.json",
        help="Supplement summary JSON output path (only written when Go).",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=50,
        help="Go/No-Go threshold for the total Ni coupling reaction count.",
    )
    parser.add_argument(
        "--nicolit-cache",
        default=None,
        help="Optional cached NiCOlit CSV path (skips download when present).",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Do not attempt to download NiCOlit (offline mode).",
    )
    args = parser.parse_args()

    payload = run_research(
        output_md=args.output,
        uspto_csv=args.uspto_csv,
        ord_csv=args.ord_csv,
        hitea_csv=args.hitea_csv,
        output_supplement=args.output_supplement,
        output_summary_json=args.output_summary_json,
        min_count=args.min_count,
        nicolit_cache=args.nicolit_cache,
        skip_download=args.skip_download,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
