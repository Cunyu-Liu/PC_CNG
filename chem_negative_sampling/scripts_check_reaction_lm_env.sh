#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/cunyuliu/pc_cng_research}
REACTION_LM_PY=${REACTION_LM_PY:-$ROOT/envs/reaction_lm/bin/python}
CHEMFORMER_DIR=${CHEMFORMER_DIR:-$ROOT/external/reaction_lm/Chemformer}
MOLECULAR_TRANSFORMER_DIR=${MOLECULAR_TRANSFORMER_DIR:-$ROOT/external/reaction_lm/MolecularTransformer}

"$REACTION_LM_PY" - <<'PY'
import importlib
import json
import sys

checks = [
    "torch",
    "rdkit",
    "transformers",
    "pytorch_lightning",
    "torchmetrics",
    "hydra",
    "omegaconf",
    "pysmilesutils",
    "torchtext",
    "tensorboard",
    "molbart",
    "molbart.models.chemformer",
    "molbart.predict",
    "onmt",
    "onmt.opts",
    "onmt.translate.translator",
]
results = {}
for name in checks:
    try:
        module = importlib.import_module(name)
        results[name] = {
            "ok": True,
            "version": getattr(module, "__version__", ""),
        }
    except Exception as exc:
        results[name] = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

print(json.dumps({"python": sys.executable, "checks": results}, indent=2))
failed = [name for name, result in results.items() if not result["ok"]]
if failed:
    raise SystemExit(f"Failed imports: {failed}")
PY

echo "==== Chemformer CLI ===="
cd "$CHEMFORMER_DIR"
"$REACTION_LM_PY" -m molbart.predict --help | head -35

echo "==== Molecular Transformer CLI ===="
cd "$MOLECULAR_TRANSFORMER_DIR"
"$REACTION_LM_PY" translate.py -h | head -35

echo "Reaction LM environment check passed."
