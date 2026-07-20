# A100 Remote Runbook

This runbook intentionally does not store passwords. Use SSH key authentication
or an already established secure SSH session before running remote experiments.

## 1. Remote Target

```text
host: 36.137.135.49
port: 22
user: cunyuliu
```

Current status from local automation:

- `ssh -o BatchMode=yes` is not available yet.
- Do not put passwords in commands, scripts, logs, or notebooks.

## 2. Safe Workspace

Use a user-owned workspace only:

```bash
mkdir -p "$HOME/pc_cng_research"/{data,logs,results}
```

Do not use `sudo`, do not modify system Python, and do not delete existing
directories.

## 3. Sync Code

From the local machine after SSH key authentication is available:

```bash
rsync -az --exclude '.DS_Store' --exclude '__pycache__' \
  code/chem_negative_sampling/ \
  -e 'ssh -p 22' \
  cunyuliu@36.137.135.49:$HOME/pc_cng_research/chem_negative_sampling/
```

## 4. Create User Venv

On the remote machine:

```bash
cd "$HOME/pc_cng_research/chem_negative_sampling"
bash scripts_setup_remote_env.sh
source "$HOME/pc_cng_research/venv_pc_cng/bin/activate"
```

## 5. Smoke Test

```bash
cd "$HOME/pc_cng_research/chem_negative_sampling"
python3 -m pc_cng.run_mvp \
  --output-dir "$HOME/pc_cng_research/results/remote_mvp" \
  --epochs 200
```

## 6. Prepare Data

Provide available public/local CSV paths:

```bash
export REGIOSQM_CSV=/path/to/regiosqm.csv
export HITEA_CSV=/path/to/hitea.csv
export USPTO_CSV=/path/to/uspto.csv
export ORD_CSV=/path/to/ord.csv

bash scripts_prepare_public_data.sh
```

## 7. Run Scale Pipeline

Example with a normalized USPTO positive CSV:

```bash
export INPUT_POSITIVES="$HOME/pc_cng_research/chem_negative_sampling/data/processed/uspto_normalized.csv"
export RESULTS_DIR="$HOME/pc_cng_research/results/uspto_pc_cng_run"
export LIMIT=100000
export EPOCHS=300

bash scripts_run_full_pipeline.sh
```

Remove `LIMIT` only after the limited run succeeds and output quality is
inspected.

## 8. Required Publishable Follow-ups

- Use RDKit/RXNMapper atom mapping before final claims.
- Run all baselines: random, template perturbation, DORA-style alternate center,
  PU reliable negative, and PC-CNG.
- Run OOD splits and false-negative review.
- Never report synthetic negatives as real failed experiments.

