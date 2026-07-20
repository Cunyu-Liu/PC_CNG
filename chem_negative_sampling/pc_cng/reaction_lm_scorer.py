"""Adapter interface for pretrained reaction language-model scorers.

This module intentionally keeps heavyweight model imports out of PC-CNG core.
It defines a common CSV scoring protocol so Chemformer, Molecular Transformer,
or other forward reaction models can be plugged in without changing downstream
reranking evaluation code.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import csv
from functools import lru_cache
import glob
import json
import os
import re
import subprocess
import tempfile
from typing import Dict, Iterable, List, Sequence, Tuple


INPUT_FIELDS = [
    "group_id",
    "source_id",
    "reactants",
    "agents",
    "candidate_product",
    "candidate_reaction",
    "label",
    "split",
    "dataset",
    "candidate_source",
    "candidate_family",
    "reaction_class",
]

OUTPUT_FIELDS = INPUT_FIELDS + [
    "lm_score",
    "lm_score_type",
    "lm_model",
    "lm_rank",
    "lm_prediction",
    "lm_status",
]


SMILES_TOKEN_PATTERN = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|Si?|Se?|Li?|Na?|Mg?|Al?|Ca?|Fe?|Zn?|"
    r"[BCNOPSFIbcnops]|\(|\)|\.|=|#|-|\+|\\\\|/|:|~|@|\?|>|\*|\$|%[0-9]{2}|[0-9])"
)


class CheckpointUnavailableError(RuntimeError):
    """Raised when a requested external reaction LM checkpoint is missing."""


try:
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
except Exception:
    pass


class ReactionLMScorer:
    """Minimal scorer interface for external reaction language models."""

    score_type = "unimplemented"

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def score_rows(self, rows: List[Dict[str, str]]) -> List[float]:
        raise NotImplementedError("Install a concrete scorer adapter, e.g. Chemformer or Molecular Transformer")

    def score_rows_with_metadata(self, rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
        scores = self.score_rows(rows)
        metadata: List[Dict[str, str]] = []
        for score in scores:
            metadata.append(
                {
                    "lm_score": f"{float(score):.8f}",
                    "lm_score_type": self.score_type,
                    "lm_model": self.model_name,
                    "lm_rank": "",
                    "lm_prediction": "",
                    "lm_status": "ok",
                }
            )
        return metadata


class IdentityLengthScorer(ReactionLMScorer):
    """Tiny deterministic smoke-test scorer with no chemistry claims."""

    score_type = "length_baseline"

    def score_rows(self, rows: List[Dict[str, str]]) -> List[float]:
        scores = []
        for row in rows:
            product = row.get("candidate_product", "")
            # Penalize very long candidates. This exists only to test the IO path.
            scores.append(-float(len(product)))
        return scores


def first_existing_path(paths_or_globs: Sequence[str]) -> str | None:
    for item in paths_or_globs:
        matches = sorted(glob.glob(item, recursive=True))
        if matches:
            return matches[0]
        if os.path.exists(item):
            return item
    return None


def require_file(path: str | None, label: str, guidance: str) -> str:
    if path and os.path.isfile(path):
        return path
    location = path or "<not provided>"
    raise CheckpointUnavailableError(
        f"{label} is unavailable at {location}. {guidance}"
    )


@lru_cache(maxsize=200_000)
def canonical_smiles(smiles: str) -> str:
    smiles = (smiles or "").strip()
    if not smiles:
        return ""
    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        pass
    return smiles


def context_input(row: Dict[str, str], include_agents: bool) -> str:
    reactants = row.get("reactants", "").strip()
    agents = row.get("agents", "").strip()
    if include_agents and agents:
        return f"{reactants}>{agents}"
    return reactants


def chemformer_table_value(value: str, fallback: str = "C") -> str:
    value = (value or "").strip()
    if not value or value.lower() == "nan":
        return fallback
    return value


def unique_context_targets(rows: Sequence[Dict[str, str]], include_agents: bool) -> List[Tuple[str, str]]:
    contexts: "OrderedDict[str, str]" = OrderedDict()
    for row in rows:
        source = context_input(row, include_agents=include_agents)
        product = row.get("candidate_product", "")
        if source not in contexts:
            contexts[source] = product
        if row.get("label") == "1":
            contexts[source] = product
    return list(contexts.items())


def with_pythonpath(env: Dict[str, str], path: str | None) -> Dict[str, str]:
    if not path:
        return env
    old_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = path if not old_path else f"{path}:{old_path}"
    return env


def run_subprocess(cmd: Sequence[str], cwd: str | None, env: Dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(cmd),
        cwd=cwd,
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


class ChemformerBeamScorer(ReactionLMScorer):
    """Score candidates by exact match against Chemformer forward-prediction beams."""

    score_type = "chemformer_beam_loglikelihood"

    def __init__(
        self,
        model_name: str,
        model_path: str | None,
        vocabulary_path: str | None,
        python_bin: str,
        chemformer_root: str | None,
        work_dir: str | None,
        batch_size: int,
        n_beams: int,
        device: str,
        include_agents: bool,
    ) -> None:
        super().__init__(model_name)
        self.model_path = require_file(
            model_path,
            "Chemformer checkpoint",
            "Download the USPTO-50K forward checkpoint and pass --model-path, or set CHEMFORMER_MODEL_PATH.",
        )
        self.vocabulary_path = require_file(
            vocabulary_path,
            "Chemformer vocabulary",
            "Pass --vocabulary-path, or set CHEMFORMER_VOCAB_PATH.",
        )
        self.python_bin = python_bin
        self.chemformer_root = chemformer_root
        self.work_dir = work_dir
        self.batch_size = batch_size
        self.n_beams = n_beams
        self.device = device
        self.include_agents = include_agents

    def _write_input(self, path: str, contexts: Sequence[Tuple[str, str]]) -> None:
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["reactants", "products", "set"], delimiter="\t")
            writer.writeheader()
            for source, target in contexts:
                writer.writerow(
                    {
                        "reactants": chemformer_table_value(source),
                        "products": chemformer_table_value(target),
                        "set": "test",
                    }
                )

    def _run_predict(self, input_path: str, output_path: str) -> None:
        n_gpus = "0" if self.device == "cpu" else "1"
        data_device = "cpu" if self.device == "cpu" else "cuda"
        cmd = [
            self.python_bin,
            "-m",
            "molbart.predict",
            f"data_path={input_path}",
            f"output_sampled_smiles={output_path}",
            f"model_path={self.model_path}",
            f"vocabulary_path={self.vocabulary_path}",
            f"batch_size={self.batch_size}",
            f"n_beams={self.n_beams}",
            f"n_gpus={n_gpus}",
            f"data_device={data_device}",
            "dataset_part=full",
            "task=forward_prediction",
            "model_type=bart",
            "train_mode=eval",
        ]
        env = with_pythonpath(dict(os.environ), self.chemformer_root)
        try:
            run_subprocess(cmd, cwd=self.chemformer_root, env=env)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Chemformer prediction failed:\n{exc.stdout}") from exc

    def _read_predictions(self, path: str) -> List[List[Tuple[str, float, int]]]:
        predictions: List[List[Tuple[str, float, int]]] = []
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                beams: List[Tuple[str, float, int]] = []
                for rank in range(1, self.n_beams + 1):
                    smi = row.get(f"sampled_smiles_{rank}", "")
                    if not smi:
                        continue
                    raw_llh = row.get(f"loglikelihood_{rank}", "")
                    try:
                        llh = float(raw_llh)
                    except (TypeError, ValueError):
                        llh = -float(rank)
                    beams.append((smi, llh, rank))
                predictions.append(beams)
        return predictions

    def score_rows_with_metadata(self, rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
        contexts = unique_context_targets(rows, include_agents=self.include_agents)
        context_sources = [source for source, _ in contexts]
        with tempfile.TemporaryDirectory(dir=self.work_dir) as tmp_dir:
            # Chemformer SynthesisDataModule uses tabular parsing only for .csv
            # paths, even though the expected delimiter is a tab.
            input_path = os.path.join(tmp_dir, "chemformer_input.csv")
            output_path = os.path.join(tmp_dir, "chemformer_predictions.tsv")
            self._write_input(input_path, contexts)
            self._run_predict(input_path, output_path)
            prediction_rows = self._read_predictions(output_path)

        context_predictions: Dict[str, List[Tuple[str, float, int]]] = {}
        for source, beams in zip(context_sources, prediction_rows):
            context_predictions[source] = beams

        output: List[Dict[str, str]] = []
        for row in rows:
            source = context_input(row, include_agents=self.include_agents)
            candidate = canonical_smiles(row.get("candidate_product", ""))
            matched_prediction = ""
            matched_rank = ""
            matched_score = -1_000_000.0 - len(candidate) * 1e-3
            status = "not_in_top_beams"
            for prediction, score, rank in context_predictions.get(source, []):
                if canonical_smiles(prediction) == candidate:
                    matched_prediction = prediction
                    matched_rank = str(rank)
                    matched_score = score
                    status = "matched_beam"
                    break
            output.append(
                {
                    "lm_score": f"{matched_score:.8f}",
                    "lm_score_type": self.score_type,
                    "lm_model": self.model_name,
                    "lm_rank": matched_rank,
                    "lm_prediction": matched_prediction,
                    "lm_status": status,
                }
            )
        return output


class ChemformerLogLikelihoodScorer(ChemformerBeamScorer):
    """Score each candidate product by conditional log-likelihood under Chemformer."""

    score_type = "chemformer_conditional_loglikelihood"

    def _write_candidate_input(self, path: str, rows: Sequence[Dict[str, str]]) -> None:
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["reactants", "products", "set"], delimiter="\t")
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "reactants": chemformer_table_value(
                            context_input(row, include_agents=self.include_agents)
                        ),
                        "products": chemformer_table_value(row.get("candidate_product", "")),
                        "set": "test",
                    }
                )

    def _run_log_likelihood(self, input_path: str, output_path: str, script_path: str) -> None:
        n_gpus = 0 if self.device == "cpu" else 1
        data_device = "cpu" if self.device == "cpu" else "cuda"
        config = {
            "input_path": input_path,
            "output_path": output_path,
            "model_path": self.model_path,
            "vocabulary_path": self.vocabulary_path,
            "batch_size": self.batch_size,
            "n_gpus": n_gpus,
            "data_device": data_device,
        }
        config_path = os.path.join(os.path.dirname(script_path), "chemformer_ll_config.json")
        with open(config_path, "w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)
        with open(script_path, "w", encoding="utf-8") as handle:
            handle.write(
                """
import csv
import json
import sys

from omegaconf import OmegaConf

from molbart.models import Chemformer


with open(sys.argv[1], encoding="utf-8") as handle:
    params = json.load(handle)

cfg = OmegaConf.create(
    {
        "batch_size": params["batch_size"],
        "n_beams": 1,
        "n_unique_beams": None,
        "n_gpus": params["n_gpus"],
        "data_path": params["input_path"],
        "output_sampled_smiles": None,
        "vocabulary_path": params["vocabulary_path"],
        "task": "forward_prediction",
        "i_chunk": 0,
        "n_chunks": 1,
        "data_device": params["data_device"],
        "model_path": params["model_path"],
        "model_type": "bart",
        "dataset_part": "full",
        "train_mode": "eval",
        "datamodule": ["SynthesisDataModule"],
    }
)

chemformer = Chemformer(cfg)
scores = chemformer.log_likelihood(dataset="full")
with open(params["output_path"], "w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(["row_index", "log_likelihood"])
    for idx, score in enumerate(scores):
        writer.writerow([idx, f"{float(score):.8f}"])
""".lstrip()
            )
        env = with_pythonpath(dict(os.environ), self.chemformer_root)
        cmd = [self.python_bin, script_path, config_path]
        try:
            run_subprocess(cmd, cwd=self.chemformer_root, env=env)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Chemformer log-likelihood scoring failed:\n{exc.stdout}") from exc

    def score_rows_with_metadata(self, rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
        with tempfile.TemporaryDirectory(dir=self.work_dir) as tmp_dir:
            input_path = os.path.join(tmp_dir, "chemformer_likelihood_input.csv")
            output_path = os.path.join(tmp_dir, "chemformer_likelihood_scores.csv")
            script_path = os.path.join(tmp_dir, "run_chemformer_log_likelihood.py")
            self._write_candidate_input(input_path, rows)
            self._run_log_likelihood(input_path, output_path, script_path)
            with open(output_path, newline="", encoding="utf-8") as handle:
                scores = [float(row["log_likelihood"]) for row in csv.DictReader(handle)]

        output: List[Dict[str, str]] = []
        for row, score in zip(rows, scores):
            status = "ok" if row.get("candidate_product", "").strip() else "fallback_product"
            output.append(
                {
                    "lm_score": f"{score:.8f}",
                    "lm_score_type": self.score_type,
                    "lm_model": self.model_name,
                    "lm_rank": "",
                    "lm_prediction": "",
                    "lm_status": status,
                }
            )
        return output


def tokenize_smiles(smiles: str) -> str:
    tokens = SMILES_TOKEN_PATTERN.findall(smiles)
    if tokens and "".join(tokens) == smiles:
        return " ".join(tokens)
    return " ".join(smiles.strip())


class MolecularTransformerBeamScorer(ReactionLMScorer):
    """Score candidates by rank in Molecular Transformer forward-prediction beams."""

    score_type = "molecular_transformer_beam_rank"

    def __init__(
        self,
        model_name: str,
        model_path: str | None,
        python_bin: str,
        mt_root: str | None,
        work_dir: str | None,
        batch_size: int,
        n_beams: int,
        device: str,
        include_agents: bool,
    ) -> None:
        super().__init__(model_name)
        self.model_path = require_file(
            model_path,
            "Molecular Transformer checkpoint",
            "Download the IBM MolecularTransformerModels checkpoint and pass --model-path, or set MOLECULAR_TRANSFORMER_MODEL_PATH.",
        )
        self.python_bin = python_bin
        self.mt_root = mt_root
        self.work_dir = work_dir
        self.batch_size = batch_size
        self.n_beams = n_beams
        self.device = device
        self.include_agents = include_agents

    def _run_translate(self, src_path: str, output_path: str) -> None:
        translate_py = os.path.join(self.mt_root or "", "translate.py")
        require_file(translate_py, "Molecular Transformer translate.py", "Set --molecular-transformer-root.")
        cmd = [
            self.python_bin,
            translate_py,
            "-model",
            self.model_path,
            "-src",
            src_path,
            "-output",
            output_path,
            "-batch_size",
            str(self.batch_size),
            "-beam_size",
            str(self.n_beams),
            "-n_best",
            str(self.n_beams),
            "-replace_unk",
            "-max_length",
            "200",
            "-fast",
        ]
        if self.device != "cpu":
            cmd.extend(["-gpu", "0"])
        env = with_pythonpath(dict(os.environ), self.mt_root)
        try:
            run_subprocess(cmd, cwd=self.mt_root, env=env)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Molecular Transformer prediction failed:\n{exc.stdout}") from exc

    def score_rows_with_metadata(self, rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
        contexts = unique_context_targets(rows, include_agents=self.include_agents)
        sources = [source for source, _ in contexts]
        with tempfile.TemporaryDirectory(dir=self.work_dir) as tmp_dir:
            src_path = os.path.join(tmp_dir, "mt_src.txt")
            output_path = os.path.join(tmp_dir, "mt_predictions.txt")
            with open(src_path, "w", encoding="utf-8") as handle:
                for source in sources:
                    handle.write(tokenize_smiles(source) + "\n")
            self._run_translate(src_path, output_path)
            with open(output_path, encoding="utf-8") as handle:
                flat_predictions = [line.strip().replace(" ", "") for line in handle]

        context_predictions: Dict[str, List[Tuple[str, float, int]]] = {}
        for idx, source in enumerate(sources):
            start = idx * self.n_beams
            preds = flat_predictions[start : start + self.n_beams]
            context_predictions[source] = [
                (pred, -float(rank), rank) for rank, pred in enumerate(preds, start=1) if pred
            ]

        output: List[Dict[str, str]] = []
        for row in rows:
            source = context_input(row, include_agents=self.include_agents)
            candidate = canonical_smiles(row.get("candidate_product", ""))
            matched_prediction = ""
            matched_rank = ""
            matched_score = -1_000_000.0 - len(candidate) * 1e-3
            status = "not_in_top_beams"
            for prediction, score, rank in context_predictions.get(source, []):
                if canonical_smiles(prediction) == candidate:
                    matched_prediction = prediction
                    matched_rank = str(rank)
                    matched_score = score
                    status = "matched_beam"
                    break
            output.append(
                {
                    "lm_score": f"{matched_score:.8f}",
                    "lm_score_type": self.score_type,
                    "lm_model": self.model_name,
                    "lm_rank": matched_rank,
                    "lm_prediction": matched_prediction,
                    "lm_status": status,
                }
            )
        return output


def read_rows(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_rows(path: str, rows: Iterable[Dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def scorer_from_args(args: argparse.Namespace) -> ReactionLMScorer:
    if args.scorer == "length_baseline":
        return IdentityLengthScorer(args.model_name)
    root = args.root
    if args.scorer == "chemformer_forward":
        chemformer_root = args.chemformer_root or os.path.join(root, "external/reaction_lm/Chemformer")
        model_path = args.model_path or os.environ.get("CHEMFORMER_MODEL_PATH") or first_existing_path(
            [
                os.path.join(root, "models/reaction_lm/chemformer_forward_uspto50k/**/*.ckpt"),
                os.path.join(root, "models/reaction_lm/chemformer/**/*.ckpt"),
                os.path.join(chemformer_root, "saved_models/**/*.ckpt"),
            ]
        )
        vocabulary_path = args.vocabulary_path or os.environ.get("CHEMFORMER_VOCAB_PATH") or first_existing_path(
            [
                os.path.join(chemformer_root, "bart_vocab_downstream.json"),
                os.path.join(chemformer_root, "bart_vocab.json"),
            ]
        )
        return ChemformerBeamScorer(
            model_name=args.model_name,
            model_path=model_path,
            vocabulary_path=vocabulary_path,
            python_bin=args.reaction_lm_python,
            chemformer_root=chemformer_root,
            work_dir=args.work_dir,
            batch_size=args.batch_size,
            n_beams=args.n_beams,
            device=args.device,
            include_agents=args.include_agents,
        )
    if args.scorer == "chemformer_log_likelihood":
        chemformer_root = args.chemformer_root or os.path.join(root, "external/reaction_lm/Chemformer")
        model_path = args.model_path or os.environ.get("CHEMFORMER_MODEL_PATH") or first_existing_path(
            [
                os.path.join(root, "models/reaction_lm/chemformer_forward_uspto50k/**/*.ckpt"),
                os.path.join(root, "models/reaction_lm/chemformer/**/*.ckpt"),
                os.path.join(chemformer_root, "saved_models/**/*.ckpt"),
            ]
        )
        vocabulary_path = args.vocabulary_path or os.environ.get("CHEMFORMER_VOCAB_PATH") or first_existing_path(
            [
                os.path.join(chemformer_root, "bart_vocab_downstream.json"),
                os.path.join(chemformer_root, "bart_vocab.json"),
            ]
        )
        return ChemformerLogLikelihoodScorer(
            model_name=args.model_name,
            model_path=model_path,
            vocabulary_path=vocabulary_path,
            python_bin=args.reaction_lm_python,
            chemformer_root=chemformer_root,
            work_dir=args.work_dir,
            batch_size=args.batch_size,
            n_beams=args.n_beams,
            device=args.device,
            include_agents=args.include_agents,
        )
    if args.scorer == "molecular_transformer_forward":
        mt_root = args.molecular_transformer_root or os.path.join(root, "external/reaction_lm/MolecularTransformer")
        model_path = args.model_path or os.environ.get("MOLECULAR_TRANSFORMER_MODEL_PATH") or first_existing_path(
            [
                os.path.join(root, "models/reaction_lm/molecular_transformer/**/*.pt"),
                os.path.join(mt_root, "experiments/models/**/*model*.pt"),
            ]
        )
        return MolecularTransformerBeamScorer(
            model_name=args.model_name,
            model_path=model_path,
            python_bin=args.reaction_lm_python,
            mt_root=mt_root,
            work_dir=args.work_dir,
            batch_size=args.batch_size,
            n_beams=args.n_beams,
            device=args.device,
            include_agents=args.include_agents,
        )
    raise ValueError(
        f"Unsupported scorer={args.scorer!r}. Add an adapter class for Chemformer/Molecular Transformer first."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Candidate CSV following INPUT_FIELDS")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument(
        "--scorer",
        choices=[
            "length_baseline",
            "chemformer_forward",
            "chemformer_log_likelihood",
            "molecular_transformer_forward",
        ],
        default="length_baseline",
    )
    parser.add_argument("--model-name", default="length_baseline")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--vocabulary-path", default=None)
    parser.add_argument(
        "--root",
        default=os.environ.get("PC_CNG_ROOT") or os.environ.get("ROOT") or "/home/cunyuliu/pc_cng_research",
    )
    parser.add_argument(
        "--reaction-lm-python",
        default=os.environ.get(
            "REACTION_LM_PYTHON",
            "/home/cunyuliu/pc_cng_research/envs/reaction_lm/bin/python",
        ),
    )
    parser.add_argument("--chemformer-root", default=os.environ.get("CHEMFORMER_ROOT"))
    parser.add_argument("--molecular-transformer-root", default=os.environ.get("MOLECULAR_TRANSFORMER_ROOT"))
    parser.add_argument("--work-dir", default=os.environ.get("REACTION_LM_WORK_DIR"))
    parser.add_argument("--n-beams", type=int, default=10)
    parser.add_argument("--device", choices=["cuda", "cpu"], default=os.environ.get("REACTION_LM_DEVICE", "cuda"))
    parser.add_argument("--include-agents", dest="include_agents", action="store_true", default=True)
    parser.add_argument("--exclude-agents", dest="include_agents", action="store_false")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    rows = read_rows(args.input)
    scorer = scorer_from_args(args)
    output_rows: List[Dict[str, str]] = []
    scored_rows = scorer.score_rows_with_metadata(rows)
    for row, scored in zip(rows, scored_rows):
        out = dict(row)
        out.update(scored)
        output_rows.append(out)

    write_rows(args.output, output_rows)
    summary = {
        "input": args.input,
        "output": args.output,
        "rows": len(rows),
        "scorer": args.scorer,
        "model_name": args.model_name,
        "score_type": scorer.score_type,
        "model_path": args.model_path,
        "n_beams": args.n_beams,
        "device": args.device,
        "include_agents": args.include_agents,
        "notes": [
            "length_baseline is only an IO smoke test.",
            "chemformer_forward and molecular_transformer_forward score candidates by exact match against top-k generated beams.",
            "chemformer_log_likelihood scores every candidate product by conditional log-likelihood.",
            "For candidates absent from top-k beams, lm_score is a large negative unmatched penalty.",
        ],
    }
    os.makedirs(os.path.dirname(args.summary), exist_ok=True)
    with open(args.summary, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
