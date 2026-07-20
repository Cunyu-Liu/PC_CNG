#!/usr/bin/env bash
set -euo pipefail

# Fetch the public code repository associated with
# "Negative chemical data boosts language models in reaction outcome prediction".
# This script only clones/pulls code by default. It does not download large data
# unless RUN_DOWNLOAD=1 is explicitly set.

EXTERNAL_DIR="${EXTERNAL_DIR:-$HOME/pc_cng_research/external}"
REPO_DIR="$EXTERNAL_DIR/negative_learning"
REPO_URL="${REPO_URL:-https://github.com/rxn4chemistry/negative_learning.git}"

mkdir -p "$EXTERNAL_DIR"

if [[ -d "$REPO_DIR/.git" ]]; then
  echo "[fetch] Updating $REPO_DIR"
  git -C "$REPO_DIR" pull --ff-only
else
  echo "[fetch] Cloning $REPO_URL"
  git clone "$REPO_URL" "$REPO_DIR"
fi

echo "[fetch] Repository ready: $REPO_DIR"

if [[ "${RUN_DOWNLOAD:-0}" == "1" ]]; then
  echo "[fetch] RUN_DOWNLOAD=1, invoking repository data download script"
  if [[ -f "$REPO_DIR/data/download_datasets.sh" ]]; then
    bash "$REPO_DIR/data/download_datasets.sh"
  else
    echo "[fetch] No data/download_datasets.sh found" >&2
    exit 1
  fi
else
  echo "[fetch] Skipping data download. Set RUN_DOWNLOAD=1 to enable it after checking size/licensing."
fi

