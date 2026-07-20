"""Build a strict external product-prediction candidate set.

The output follows :mod:`pc_cng.reaction_lm_scorer` CSV fields so Chemformer,
Molecular Transformer, PC-CNG, and hybrid rerankers can be evaluated on the
same reactant contexts and candidate products.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import OrderedDict
from typing import Dict, Iterable, List, Sequence

from .chem_utils import join_reaction, split_reaction
from .reaction_lm_scorer import INPUT_FIELDS, canonical_smiles, chemformer_table_value


CONTEXT_FIELDS = [
    "row_index",
    "group_id",
    "source_id",
    "reactants",
    "agents",
    "observed_product",
    "split",
    "dataset",
    "reaction_class",
]

EXTRA_FIELDS = ["external_beam_rank", "external_beam_score", "external_beam_model"]


def sniff_dialect(path: str) -> csv.Dialect:
    with open(path, encoding="utf-8") as handle:
        sample = handle.read(4096)
    if "\t" in sample.splitlines()[0]:
        return csv.excel_tab
    return csv.excel


def reaction_parts(reaction_smiles: str) -> tuple[str, str, str] | None:
    try:
        return split_reaction(reaction_smiles)
    except ValueError:
        return None


def context_id(dataset: str, split: str, source_id: str) -> str:
    return f"external_product_prediction|{dataset}|{split}|{source_id}"


def read_contexts(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def read_positive_contexts(real_csvs: Sequence[str]) -> List[Dict[str, str]]:
    contexts: "OrderedDict[str, Dict[str, str]]" = OrderedDict()
    for path in real_csvs:
        dataset_from_path = os.path.splitext(os.path.basename(path))[0]
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for line_number, row in enumerate(reader, start=1):
                if row.get("label_type") != "positive":
                    continue
                reaction = row.get("reaction_smiles", "")
                parts = reaction_parts(reaction)
                if parts is None:
                    continue
                reactants, agents, product = parts
                source_id = row.get("source_id") or f"{dataset_from_path}_{line_number}"
                dataset = row.get("source") or dataset_from_path
                split = row.get("split") or "unknown"
                key = f"{dataset}|{split}|{source_id}"
                if key in contexts:
                    continue
                contexts[key] = {
                    "source_id": source_id,
                    "group_id": context_id(dataset, split, source_id),
                    "reactants": reactants,
                    "agents": agents,
                    "observed_product": product,
                    "split": split,
                    "dataset": dataset,
                    "reaction_class": row.get("reaction_class", ""),
                    "reaction_smiles": reaction,
                }
    return list(contexts.values())


def base_candidate_from_context(context: Dict[str, str]) -> Dict[str, str]:
    return {
        "group_id": context["group_id"],
        "source_id": context["source_id"],
        "reactants": context["reactants"],
        "agents": context["agents"],
        "candidate_product": context["observed_product"],
        "candidate_reaction": join_reaction(context["reactants"], context["observed_product"], context["agents"]),
        "label": "1",
        "split": context["split"],
        "dataset": context["dataset"],
        "candidate_source": "observed_positive",
        "candidate_family": "observed_positive",
        "reaction_class": context.get("reaction_class", ""),
    }


def product_key(product: str) -> str:
    return canonical_smiles(product) or product.strip()


def merge_tags(left: str, right: str) -> str:
    tags = [tag for tag in left.split("+") if tag] + [tag for tag in right.split("+") if tag]
    return "+".join(sorted(set(tags)))


def add_candidate(
    rows: "OrderedDict[tuple[str, str], Dict[str, str]]",
    row: Dict[str, str],
) -> None:
    key = (row["group_id"], product_key(row["candidate_product"]))
    existing = rows.get(key)
    if existing is None:
        rows[key] = row
        return
    existing["label"] = "1" if existing.get("label") == "1" or row.get("label") == "1" else "0"
    existing["candidate_source"] = merge_tags(existing.get("candidate_source", ""), row.get("candidate_source", ""))
    existing["candidate_family"] = merge_tags(existing.get("candidate_family", ""), row.get("candidate_family", ""))
    for field in EXTRA_FIELDS:
        if not existing.get(field) and row.get(field):
            existing[field] = row[field]


def read_synthetic_candidates(
    synthetic_csvs: Sequence[str],
    contexts_by_source: Dict[str, Dict[str, str]],
    review_statuses: Sequence[str],
) -> List[Dict[str, str]]:
    allowed_statuses = set(review_statuses)
    out: List[Dict[str, str]] = []
    for path in synthetic_csvs:
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                source_id = row.get("source_id", "")
                context = contexts_by_source.get(source_id)
                if context is None:
                    continue
                status = row.get("review_status", "keep_synthetic_negative")
                if allowed_statuses and status not in allowed_statuses:
                    continue
                reaction = row.get("candidate_reaction") or row.get("reaction_smiles") or ""
                parts = reaction_parts(reaction)
                if parts is None:
                    continue
                _, _, product = parts
                family = row.get("action_family") or row.get("failure_type") or "pc_cng_negative"
                out.append(
                    {
                        "group_id": context["group_id"],
                        "source_id": source_id,
                        "reactants": context["reactants"],
                        "agents": context["agents"],
                        "candidate_product": product,
                        "candidate_reaction": reaction,
                        "label": "0",
                        "split": context["split"],
                        "dataset": context["dataset"],
                        "candidate_source": "pc_cng",
                        "candidate_family": family,
                        "reaction_class": context.get("reaction_class", ""),
                    }
                )
    return out


def first_present(row: Dict[str, str], names: Sequence[str]) -> str:
    for name in names:
        value = row.get(name, "")
        if value:
            return value
    return ""


def score_for_rank(row: Dict[str, str], rank: int) -> str:
    return first_present(
        row,
        [
            f"loglikelihood_{rank}",
            f"log_likelihood_{rank}",
            f"score_{rank}",
            f"beam_score_{rank}",
            f"lm_score_{rank}",
        ],
    )


def product_for_rank(row: Dict[str, str], rank: int) -> str:
    return first_present(
        row,
        [
            f"sampled_smiles_{rank}",
            f"prediction_{rank}",
            f"beam_{rank}",
            f"product_{rank}",
            f"candidate_product_{rank}",
        ],
    )


def context_for_beam_row(
    row: Dict[str, str],
    row_index: int,
    contexts: Sequence[Dict[str, str]],
    contexts_by_source: Dict[str, Dict[str, str]],
    contexts_by_reactants: Dict[str, Dict[str, str]],
) -> Dict[str, str] | None:
    source_id = row.get("source_id", "")
    if source_id and source_id in contexts_by_source:
        return contexts_by_source[source_id]
    raw_index = row.get("row_index", "") or row.get("index", "")
    if raw_index:
        try:
            idx = int(raw_index)
            if 0 <= idx < len(contexts):
                return contexts[idx]
        except ValueError:
            pass
    reactants = first_present(row, ["reactants", "source", "input"])
    if reactants and reactants in contexts_by_reactants:
        return contexts_by_reactants[reactants]
    if 0 <= row_index < len(contexts):
        return contexts[row_index]
    return None


def read_external_beam_candidates(
    beam_csv: str,
    contexts: Sequence[Dict[str, str]],
    model_name: str,
    n_beams: int,
) -> List[Dict[str, str]]:
    contexts_by_source = {row["source_id"]: row for row in contexts}
    contexts_by_reactants = {}
    for context in contexts:
        contexts_by_reactants[context["reactants"]] = context
        if context.get("agents"):
            contexts_by_reactants[f"{context['reactants']}>{context['agents']}"] = context

    out: List[Dict[str, str]] = []
    dialect = sniff_dialect(beam_csv)
    with open(beam_csv, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, dialect=dialect)
        for row_index, row in enumerate(reader):
            context = context_for_beam_row(
                row=row,
                row_index=row_index,
                contexts=contexts,
                contexts_by_source=contexts_by_source,
                contexts_by_reactants=contexts_by_reactants,
            )
            if context is None:
                continue
            wide_products = [product_for_rank(row, rank) for rank in range(1, n_beams + 1)]
            if any(wide_products):
                for rank, product in enumerate(wide_products, start=1):
                    if not product:
                        continue
                    out.append(beam_candidate(context, product, model_name, str(rank), score_for_rank(row, rank)))
                continue
            product = first_present(row, ["candidate_product", "prediction", "sampled_smiles", "product", "beam_product"])
            if not product:
                continue
            rank = first_present(row, ["rank", "lm_rank", "beam_rank"]) or "1"
            score = first_present(row, ["lm_score", "score", "loglikelihood", "log_likelihood", "beam_score"])
            out.append(beam_candidate(context, product, model_name, rank, score))
    return out


def beam_candidate(context: Dict[str, str], product: str, model_name: str, rank: str, score: str) -> Dict[str, str]:
    return {
        "group_id": context["group_id"],
        "source_id": context["source_id"],
        "reactants": context["reactants"],
        "agents": context["agents"],
        "candidate_product": product,
        "candidate_reaction": join_reaction(context["reactants"], product, context.get("agents", "")),
        "label": "0",
        "split": context["split"],
        "dataset": context["dataset"],
        "candidate_source": f"{model_name}_beam",
        "candidate_family": f"{model_name}_beam",
        "reaction_class": context.get("reaction_class", ""),
        "external_beam_rank": rank,
        "external_beam_score": score,
        "external_beam_model": model_name,
    }


def write_contexts(path: str, contexts: Sequence[Dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CONTEXT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row_index, context in enumerate(contexts):
            out = dict(context)
            out["row_index"] = row_index
            writer.writerow(out)


def write_chemformer_input(path: str, contexts: Sequence[Dict[str, str]], include_agents: bool) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["reactants", "products", "set"], delimiter="\t")
        writer.writeheader()
        for context in contexts:
            source = context["reactants"]
            if include_agents and context.get("agents"):
                source = f"{source}>{context['agents']}"
            writer.writerow(
                {
                    "reactants": chemformer_table_value(source),
                    "products": chemformer_table_value(context["observed_product"]),
                    "set": context.get("split") or "test",
                }
            )


def write_candidates(path: str, rows: Iterable[Dict[str, str]]) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = INPUT_FIELDS + [field for field in EXTRA_FIELDS if field not in INPUT_FIELDS]
    count = 0
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def source_counts(rows: Sequence[Dict[str, str]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        for source in row.get("candidate_source", "unknown").split("+"):
            counts[source or "unknown"] = counts.get(source or "unknown", 0) + 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-csv", action="append", required=True)
    parser.add_argument("--synthetic-csv", action="append", default=[])
    parser.add_argument("--external-beam-csv", action="append", default=[])
    parser.add_argument("--beam-context-csv", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--contexts-output", default=None)
    parser.add_argument("--chemformer-input-output", default=None)
    parser.add_argument("--external-model-name", default="chemformer")
    parser.add_argument("--n-beams", type=int, default=10)
    parser.add_argument("--review-status", action="append", default=["keep_synthetic_negative"])
    parser.add_argument("--include-agents", dest="include_agents", action="store_true", default=True)
    parser.add_argument("--exclude-agents", dest="include_agents", action="store_false")
    args = parser.parse_args()

    contexts = read_positive_contexts(args.real_csv)
    if args.beam_context_csv:
        contexts = read_contexts(args.beam_context_csv)
    contexts_by_source = {row["source_id"]: row for row in contexts}

    merged: "OrderedDict[tuple[str, str], Dict[str, str]]" = OrderedDict()
    for context in contexts:
        add_candidate(merged, base_candidate_from_context(context))
    synthetic_candidates = read_synthetic_candidates(
        synthetic_csvs=args.synthetic_csv,
        contexts_by_source=contexts_by_source,
        review_statuses=args.review_status,
    )
    for row in synthetic_candidates:
        add_candidate(merged, row)
    beam_rows: List[Dict[str, str]] = []
    for beam_csv in args.external_beam_csv:
        beam_rows.extend(
            read_external_beam_candidates(
                beam_csv=beam_csv,
                contexts=contexts,
                model_name=args.external_model_name,
                n_beams=args.n_beams,
            )
        )
    for row in beam_rows:
        add_candidate(merged, row)

    candidates = list(merged.values())
    candidate_count = write_candidates(args.output, candidates)
    if args.contexts_output:
        write_contexts(args.contexts_output, contexts)
    if args.chemformer_input_output:
        write_chemformer_input(args.chemformer_input_output, contexts, include_agents=args.include_agents)

    summary = {
        "task": "strict_external_product_prediction_candidate_set",
        "real_csv": args.real_csv,
        "synthetic_csv": args.synthetic_csv,
        "external_beam_csv": args.external_beam_csv,
        "contexts": len(contexts),
        "candidate_rows": candidate_count,
        "candidate_source_counts": source_counts(candidates),
        "outputs": {
            "candidate_csv": args.output,
            "contexts_csv": args.contexts_output,
            "chemformer_input_csv": args.chemformer_input_output,
        },
        "schema": {
            "group_id": "one reactant context; all methods rank candidates within this group",
            "label": "1 for observed product, 0 for generated/negative alternatives",
            "candidate_source": "observed_positive, pc_cng, or external beam model tag",
        },
    }
    os.makedirs(os.path.dirname(args.summary), exist_ok=True)
    with open(args.summary, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
