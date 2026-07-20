#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/pc_cng_research/chem_negative_sampling}"
VENV_DIR="${VENV_DIR:-$HOME/pc_cng_research/venv_pc_cng}"

if ! python3 -m venv "$VENV_DIR"; then
  echo "[setup] python3 -m venv failed; falling back to user-level virtualenv"
  python3 -m pip install --user --upgrade virtualenv
  python3 -m virtualenv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel

# CUDA 12.4 matches the remote driver. Keep PyTorch separated so failures are
# easier to diagnose and the system Python remains untouched.
python -m pip install \
  torch torchvision torchaudio \
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
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("cuda_device_0", torch.cuda.get_device_name(0))
PY

echo "Environment ready: $VENV_DIR"
echo "Project root: $PROJECT_ROOT"
