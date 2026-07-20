#!/usr/bin/env bash
set -euo pipefail

CONDA_BIN="${CONDA_BIN:-$HOME/miniconda3/bin/conda}"
ENV_NAME="${ENV_NAME:-pc_cng}"

if [[ ! -x "$CONDA_BIN" ]]; then
  echo "Conda not found at $CONDA_BIN" >&2
  exit 1
fi

if "$CONDA_BIN" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[setup] Conda env already exists: $ENV_NAME"
else
  "$CONDA_BIN" create -y -n "$ENV_NAME" --override-channels -c conda-forge python=3.10 pip
fi

source "$("$CONDA_BIN" info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

python -m pip install --upgrade pip "setuptools<81" wheel

"$CONDA_BIN" install -y -n "$ENV_NAME" --override-channels \
  -c pytorch -c nvidia -c conda-forge \
  pytorch torchvision torchaudio pytorch-cuda=12.4

python -m pip install \
  numpy pandas scikit-learn scipy tqdm pyarrow \
  rdkit rxnmapper transformers sentencepiece \
  networkx matplotlib seaborn pyyaml pytest

# rxnmapper imports pkg_resources; setuptools 81+ may no longer provide it.
python -m pip install "setuptools<81"

python - <<'PY'
import importlib
for name in ["torch", "rdkit", "pandas", "sklearn", "rxnmapper", "transformers"]:
    mod = importlib.import_module(name)
    print(name, getattr(mod, "__version__", "available"))

import torch
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("cuda_device_0", torch.cuda.get_device_name(0))
PY

echo "Environment ready: $ENV_NAME"
