#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/cunyuliu/pc_cng_research}
REACTION_LM_PYTHON=${REACTION_LM_PYTHON:-$ROOT/envs/reaction_lm/bin/python}

MODEL_ROOT="$ROOT/models/reaction_lm"
CHEMFORMER_DIR="$MODEL_ROOT/chemformer_forward_uspto50k"
CHEMFORMER_HF_DIR="$MODEL_ROOT/chemformer_pretrained_hf"
MT_DIR="$MODEL_ROOT/molecular_transformer"
LOG_DIR="$ROOT/results/reaction_lm_checkpoint_download"
LOG_FILE="$LOG_DIR/download.log"

mkdir -p "$CHEMFORMER_DIR" "$CHEMFORMER_HF_DIR" "$MT_DIR" "$LOG_DIR"
: > "$LOG_FILE"

log() {
  printf '%s\n' "$*" | tee -a "$LOG_FILE"
}

try_download() {
  local url="$1"
  local output="$2"
  log "Trying: $url"
  if curl -fL --retry 3 --retry-delay 5 --connect-timeout 60 --max-time 7200 -C - -o "$output" "$url" >>"$LOG_FILE" 2>&1; then
    log "Downloaded: $output"
    return 0
  fi
  log "Failed: $url"
  return 1
}

sanitize_chemformer_checkpoint() {
  local src="$1"
  local dst="$2"
  "$REACTION_LM_PYTHON" - "$src" "$dst" <<'PY' >>"$LOG_FILE" 2>&1
import sys
import types
from collections import defaultdict
import os
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch.serialization import safe_globals
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

for name in ["deepspeed", "deepspeed.runtime", "deepspeed.runtime.fp16"]:
    sys.modules.setdefault(name, types.ModuleType(name))

loss_scaler_mod = types.ModuleType("deepspeed.runtime.fp16.loss_scaler")

class DynamicLossScaler:
    pass

DynamicLossScaler.__module__ = "deepspeed.runtime.fp16.loss_scaler"
loss_scaler_mod.DynamicLossScaler = DynamicLossScaler
sys.modules["deepspeed.runtime.fp16.loss_scaler"] = loss_scaler_mod

class _Dummy:
    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.state = state

    def __getstate__(self):
        return self.__dict__


class _DummyModule(types.ModuleType):
    def __getattr__(self, name):
        cls = type(name, (_Dummy,), {"__module__": self.__name__})
        setattr(self, name, cls)
        return cls


for mod_name in [
    "molbart.decoder",
    "molbart.tokeniser",
    "molbart.tokenizer",
    "molbart.data.dataset",
    "molbart.modules",
    "molbart.modules.transformer",
]:
    sys.modules.setdefault(mod_name, _DummyModule(mod_name))

src, dst = sys.argv[1:3]
try:
    with safe_globals([ModelCheckpoint, DynamicLossScaler, getattr, OneCycleLR, Adam, defaultdict, dict]):
        model = torch.load(src, map_location="cpu", weights_only=True)
    load_mode = "weights_only=True"
except Exception as exc:
    if os.environ.get("TRUST_EXTERNAL_CHECKPOINTS", "1") != "1":
        raise
    print(f"weights_only=True failed ({type(exc).__name__}: {exc}); falling back to weights_only=False for trusted checkpoint.")
    model = torch.load(src, map_location="cpu", weights_only=False)
    load_mode = "weights_only=False"

hyper_parameters = model.get("hyper_parameters", {})
if "vocab_size" in hyper_parameters and "vocabulary_size" not in hyper_parameters:
    hyper_parameters["vocabulary_size"] = hyper_parameters.pop("vocab_size")
hyper_parameters.pop("decode_sampler", None)

sanitized = {
    "state_dict": model["state_dict"],
    "hyper_parameters": hyper_parameters,
    "epoch": model.get("epoch", 0),
    "global_step": model.get("global_step", 0),
    "pytorch-lightning_version": model.get("pytorch-lightning_version", "1.5.10"),
}
torch.save(sanitized, dst)
print(f"Saved sanitized Chemformer inference checkpoint: {dst} (load_mode={load_mode})")
PY
}

log "=== Chemformer USPTO-50K forward checkpoint ==="
CHEMFORMER_ZIP="$CHEMFORMER_DIR/chemformer_forward.zip"
if [[ ! -s "$CHEMFORMER_ZIP" ]]; then
  try_download "https://figshare.com/ndownloader/files/42012708" "$CHEMFORMER_ZIP" || \
  try_download "https://ndownloader.figshare.com/files/42012708" "$CHEMFORMER_ZIP" || true
else
  log "Chemformer zip already exists: $CHEMFORMER_ZIP"
fi

if [[ -s "$CHEMFORMER_ZIP" ]]; then
  unzip -o "$CHEMFORMER_ZIP" -d "$CHEMFORMER_DIR" >>"$LOG_FILE" 2>&1 || true
  CHEMFORMER_CKPT=$(find "$CHEMFORMER_DIR" -type f -name '*.ckpt' | sort | head -1 || true)
  if [[ -n "$CHEMFORMER_CKPT" ]]; then
    sanitize_chemformer_checkpoint "$CHEMFORMER_CKPT" "$CHEMFORMER_DIR/model_sanitized.ckpt"
    log "Chemformer forward checkpoint ready: $CHEMFORMER_DIR/model_sanitized.ckpt"
  else
    log "Chemformer zip exists but no .ckpt file was found after unzip."
  fi
else
  log "Chemformer download did not complete. Current network returns 403 for Figshare download/API endpoints."
fi

log "=== Chemformer combined pretrained Hugging Face fallback ==="
CHEMFORMER_HF_CKPT="$CHEMFORMER_HF_DIR/pretrained.ckpt"
if [[ ! -s "$CHEMFORMER_HF_CKPT" ]]; then
  try_download "https://huggingface.co/Lytttttt/combined-pretrained-chemformer/resolve/main/pretrained.ckpt" "$CHEMFORMER_HF_CKPT" || true
else
  log "Chemformer HF pretrained checkpoint already exists: $CHEMFORMER_HF_CKPT"
fi

if [[ -s "$CHEMFORMER_HF_CKPT" ]]; then
  sanitize_chemformer_checkpoint "$CHEMFORMER_HF_CKPT" "$CHEMFORMER_HF_DIR/model_sanitized.ckpt"
  log "Chemformer HF pretrained checkpoint ready: $CHEMFORMER_HF_DIR/model_sanitized.ckpt"
else
  log "Chemformer HF fallback did not complete from this network."
fi

log "=== Molecular Transformer checkpoint ==="
log "Official model page: https://ibm.box.com/v/MolecularTransformerModels"
log "The public IBM Box page exposes model archives, but direct file-content endpoints currently require Box/IBM authentication from this environment."
log "Place a downloaded forward model .pt under: $MT_DIR"

log "=== Current checkpoint files ==="
find "$MODEL_ROOT" -type f \( -name '*.ckpt' -o -name '*.pt' -o -name '*.zip' \) | sort | tee -a "$LOG_FILE"
