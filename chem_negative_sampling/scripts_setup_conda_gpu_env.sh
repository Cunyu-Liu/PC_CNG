#!/usr/bin/env bash
set -euo pipefail

CONDA_BIN="${CONDA_BIN:-$HOME/miniconda3/bin/conda}"
ENV_NAME="${ENV_NAME:-pc_cng_gpu}"

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

# Keep the GPU environment minimal: PC-CNG training currently needs torch, not
# torchvision/torchaudio. This avoids pulling unrelated image/audio dependencies.
python -m pip install --force-reinstall \
  torch==2.6.0+cu124 \
  --index-url https://download.pytorch.org/whl/cu124

python -m pip install \
  numpy pandas scikit-learn scipy tqdm pyarrow \
  rdkit rxnmapper transformers sentencepiece \
  networkx matplotlib seaborn pyyaml pytest

python - <<'PY'
import importlib
for name in ["torch", "rdkit", "pandas", "sklearn", "rxnmapper", "transformers"]:
    mod = importlib.import_module(name)
    print(name, getattr(mod, "__version__", "available"))

import torch
print("torch_cuda", torch.version.cuda)
print("cuda_built", torch.backends.cuda.is_built())
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("cuda_device_0", torch.cuda.get_device_name(0))
PY

echo "GPU environment ready: $ENV_NAME"

