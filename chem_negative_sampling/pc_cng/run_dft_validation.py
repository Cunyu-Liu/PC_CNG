"""P2-02: DFT (xTB) validation of chemoselectivity_error synthetic negatives.

Goal: Fix P1-10 partial support (0.48 support rate < 0.6 threshold).
P2-02 uses xTB (semi-empirical DFT) on 20-30 chemoselectivity_error
candidates to compute proper DeltaG and re-evaluate the support rate.

Usage::

    cd /home/cunyuliu/pc_cng_research/chem_negative_sampling
    PYTHONPATH=. /home/cunyuliu/miniconda3/envs/pc_cng_gpu/bin/python -m pc_cng.run_dft_validation \
        --candidates /home/cunyuliu/pc_cng_research/results/false_negative_three_layer_20260719/high_confidence_negatives.csv \
        --failure-type chemoselectivity_error \
        --limit 30 \
        --method xtb \
        --output-dir /home/cunyuliu/pc_cng_research/results/dft_validation_chemoselectivity_20260720

Method dispatch:
  * xtb   : subprocess to the ``dft`` conda env (``xtb-python`` / ASE), CPU-only
  * mmff94: RDKit force field fallback (run in pc_cng_gpu env directly)
  * orca  : dispatch stub (not implemented; returns ``not_implemented`` status)

Support criterion (P2-02, simpler than P1-10):
    DeltaG_reaction > 0 kcal/mol  =>  reaction thermodynamically unfavorable
                                    =>  supports chemoselectivity_error
                                    classification (negative sample is a true
                                    negative)

CPU-only constraint: ``CUDA_VISIBLE_DEVICES`` is cleared at the start of
``main()`` so xTB (which is CPU-only) is never accidentally routed to GPU 4.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdForceFieldHelpers import (
    MMFFGetMoleculeForceField,
    MMFFGetMoleculeProperties,
    MMFFHasAllMoleculeParams,
    MMFFOptimizeMolecule,
)

# --------------------------------------------------------------------------- #
# Constants (P2-02 judgment rule)
# --------------------------------------------------------------------------- #
# P2-02 spec: DeltaG > 0 kcal/mol => supports chemoselectivity_error
# (simpler than P1-10 which used DeltaG > +5 to give MMFF94 some slack)
DFT_SUPPORT_THRESHOLD = 0.0  # kcal/mol
BARRIER_SUPPORT_THRESHOLD = 25.0  # kcal/mol (kinetic fallback)
BARRIER_CONSTANT = 5.0  # kcal/mol (rough kinetic contribution)
HARTREE_TO_KCAL_MOL = 627.509474  # 1 Hartree in kcal/mol
DEFAULT_NUM_SEEDS = 10
DEFAULT_RANDOM_SEED = 20260720
GO_NO_GO_THRESHOLD = 0.60  # support rate >= 60% => GO
DEFAULT_DFT_PYTHON = "/home/cunyuliu/miniconda3/envs/dft/bin/python"
DEFAULT_XTB_METHOD = "GFN2-xTB"
DEFAULT_XTB_SOLVENT = "water"  # ALPB implicit solvation
DEFAULT_XTB_FMAX = 0.05  # eV/A
DEFAULT_XTB_MAX_STEPS = 50


# --------------------------------------------------------------------------- #
# SMILES parsing helpers (duplicated from run_xtb_validation to keep this
# module self-contained and testable in isolation)
# --------------------------------------------------------------------------- #
def parse_reaction_smiles(reaction_smiles: str) -> Tuple[str, str]:
    """Split a reaction SMILES ``"A.B>>C.D"`` into ``(reactants, products)``.

    Returns ``("", "")`` if the string does not contain exactly one ``>>``.
    """
    if not isinstance(reaction_smiles, str) or ">>" not in reaction_smiles:
        return "", ""
    parts = reaction_smiles.split(">>")
    if len(parts) != 2:
        return "", ""
    return parts[0], parts[1]


def strip_atom_maps(smiles: str) -> str:
    """Remove RDKit atom-map numbers (``[C:1]`` -> ``[C]``) from a SMILES.

    Returns the canonical SMILES without atom maps, or ``""`` if parsing
    fails.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return ""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol)


def split_components(smiles: str) -> List[str]:
    """Split a multi-component SMILES ``"A.B.C"`` into ``["A","B","C"]``.

    Empty components are dropped.
    """
    if not isinstance(smiles, str):
        return []
    return [s for s in smiles.split(".") if s.strip()]


def mol_to_xyz_block(mol: Chem.Mol) -> str:
    """Serialize an RDKit molecule (with a 3D conformer) to an XYZ string.

    Format:
        <n_atoms>
        <comment>
        <sym> <x> <y> <z>
        ...
    """
    conf = mol.GetConformer()
    n = mol.GetNumAtoms()
    lines = [str(n), "rdkit embedded"]
    for i in range(n):
        pos = conf.GetAtomPosition(i)
        sym = mol.GetAtomWithIdx(i).GetSymbol()
        lines.append(f"{sym} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}")
    return "\n".join(lines) + "\n"


def embed_smiles_to_xyz(
    smiles: str,
    seed: int = 42,
    max_iters: int = 200,
) -> Tuple[Optional[str], str]:
    """Strip atom maps, embed 3D coords, return ``(xyz_string, status)``.

    The 3D geometry is pre-optimized with MMFF94 to give xTB a reasonable
    starting point (xTB is sensitive to bad starting geometries).

    Returns ``(None, status)`` on failure where ``status`` is one of
    ``"empty"``, ``"parse_error"``, ``"embed_failed"``.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return None, "empty"
    clean = strip_atom_maps(smiles)
    if not clean:
        return None, "parse_error"
    mol = Chem.MolFromSmiles(clean)
    if mol is None:
        return None, "parse_error"
    mol = Chem.AddHs(mol)
    try:
        rid = AllChem.EmbedMolecule(mol, randomSeed=seed, useRandomCoords=True)
        if rid != 0:
            rid = AllChem.EmbedMolecule(mol, randomSeed=seed)
        if rid != 0:
            return None, "embed_failed"
    except Exception:
        return None, "embed_failed"
    # Pre-optimize with MMFF94 to clean up the geometry before xTB
    try:
        if MMFFHasAllMoleculeParams(mol):
            MMFFOptimizeMolecule(mol, maxIters=max_iters)
        else:
            AllChem.UFFOptimizeMolecule(mol, maxIters=max_iters)
    except Exception:
        pass  # xTB will refine whatever geometry we hand it
    return mol_to_xyz_block(mol), "ok"


# --------------------------------------------------------------------------- #
# MMFF94 energy (direct in pc_cng_gpu env — fallback when xTB unavailable)
# --------------------------------------------------------------------------- #
def compute_molecule_energy_mmff94(
    smiles: str,
    seed: int = 42,
    max_iters: int = 200,
) -> Dict:
    """MMFF94 single-point energy (kcal/mol) for one molecule SMILES."""
    if not isinstance(smiles, str) or not smiles.strip():
        return {"smiles": "", "energy_kcal_per_mol": None, "status": "empty", "method": "mmff94"}
    clean = strip_atom_maps(smiles)
    if not clean:
        return {"smiles": smiles, "energy_kcal_per_mol": None, "status": "parse_error", "method": "mmff94"}
    mol = Chem.MolFromSmiles(clean)
    if mol is None:
        return {"smiles": smiles, "energy_kcal_per_mol": None, "status": "parse_error", "method": "mmff94"}
    mol = Chem.AddHs(mol)
    try:
        rid = AllChem.EmbedMolecule(mol, randomSeed=seed, useRandomCoords=True)
        if rid != 0:
            rid = AllChem.EmbedMolecule(mol, randomSeed=seed)
        if rid != 0:
            return {"smiles": smiles, "energy_kcal_per_mol": None, "status": "embed_failed", "method": "mmff94"}
    except Exception:
        return {"smiles": smiles, "energy_kcal_per_mol": None, "status": "embed_failed", "method": "mmff94"}

    actual_method = "mmff94"
    energy: Optional[float] = None
    if not MMFFHasAllMoleculeParams(mol):
        actual_method = "uff"
    else:
        try:
            MMFFOptimizeMolecule(mol, maxIters=max_iters)
            props = MMFFGetMoleculeProperties(mol)
            ff = MMFFGetMoleculeForceField(mol, props)
            energy = float(ff.CalcEnergy())
        except Exception:
            actual_method = "uff"
    if actual_method == "uff" and energy is None:
        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=max_iters)
            ff = AllChem.UFFGetMoleculeForceField(mol)
            energy = float(ff.CalcEnergy())
        except Exception:
            return {"smiles": smiles, "energy_kcal_per_mol": None, "status": "uff_failed", "method": actual_method}
    if energy is None:
        return {"smiles": smiles, "energy_kcal_per_mol": None, "status": "no_energy", "method": actual_method}
    return {"smiles": smiles, "energy_kcal_per_mol": energy, "status": "ok", "method": actual_method}


# --------------------------------------------------------------------------- #
# xTB worker (run in dft conda env via subprocess)
# --------------------------------------------------------------------------- #
XTB_WORKER_SCRIPT = r'''#!/usr/bin/env python
"""xTB energy worker — spawned by run_dft_validation.py.

Runs in the ``dft`` conda env which has ``xtb-python`` and ``ase`` installed.
Reads a JSON request file, runs GFN2-xTB (or GFN1-xTB) optimization in ALPB
implicit solvation for each SMILES, and writes a JSON response file.

Input JSON schema:
    {
        "smiles_to_xyz": {"<smiles>": "<xyz_block>", ...},
        "method": "GFN2-xTB",
        "solvent": "water",
        "fmax": 0.05,
        "max_steps": 50
    }

Output JSON schema:
    {
        "results": {
            "<smiles>": {
                "energy_hartree": float | null,
                "energy_kcal_per_mol": float | null,
                "status": "ok" | "xtb_failed:<ErrType>" | "xyz_parse_failed",
                "method": "xtb"
            },
            ...
        },
        "worker_error": null | "<message>"
    }
"""
from __future__ import annotations

import io
import json
import sys
import traceback


def _compute_one(xyz_string, method="GFN2-xTB", solvent=None, fmax=0.05, max_steps=50):
    try:
        from ase.io import read
        from ase.optimize import BFGS
        from xtb.ase.calculator import XTB
    except Exception as e:  # pragma: no cover - environment issue
        return {
            "energy_hartree": None,
            "energy_kcal_per_mol": None,
            "status": f"xtb_import_failed:{type(e).__name__}",
            "method": "xtb",
        }
    try:
        atoms = read(io.StringIO(xyz_string), format="xyz")
    except Exception as e:
        return {
            "energy_hartree": None,
            "energy_kcal_per_mol": None,
            "status": f"xyz_parse_failed:{type(e).__name__}",
            "method": "xtb",
        }
    try:
        kwargs = {"method": method}
        if solvent:
            kwargs["solvent"] = solvent
        calc = XTB(atoms=atoms, **kwargs)
        atoms.calc = calc
        opt = BFGS(atoms, logfile=None)
        opt.run(fmax=fmax, steps=max_steps)
        e_hartree = float(atoms.get_potential_energy())
        e_kcal = e_hartree * 627.509474
        return {
            "energy_hartree": e_hartree,
            "energy_kcal_per_mol": e_kcal,
            "status": "ok",
            "method": "xtb",
        }
    except Exception as e:
        return {
            "energy_hartree": None,
            "energy_kcal_per_mol": None,
            "status": f"xtb_failed:{type(e).__name__}",
            "method": "xtb",
        }


def main():
    if len(sys.argv) != 3:
        print("Usage: dft_xtb_worker.py <input.json> <output.json>", file=sys.stderr)
        sys.exit(2)
    in_path, out_path = sys.argv[1], sys.argv[2]
    try:
        with open(in_path) as f:
            cfg = json.load(f)
    except Exception as e:
        with open(out_path, "w") as f:
            json.dump({"results": {}, "worker_error": f"read_input_failed:{e}"}, f)
        sys.exit(0)
    smiles_to_xyz = cfg.get("smiles_to_xyz", {})
    method = cfg.get("method", "GFN2-xTB")
    solvent = cfg.get("solvent", "water")
    fmax = float(cfg.get("fmax", 0.05))
    max_steps = int(cfg.get("max_steps", 50))
    results = {}
    for smiles, xyz in smiles_to_xyz.items():
        results[smiles] = _compute_one(xyz, method=method, solvent=solvent, fmax=fmax, max_steps=max_steps)
    try:
        with open(out_path, "w") as f:
            json.dump({"results": results, "worker_error": None}, f, indent=2)
    except Exception as e:
        print(f"[worker] failed to write output: {e}", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
'''


def write_xtb_worker_script(target_path: str) -> None:
    """Write the embedded xTB worker script to ``target_path``."""
    with open(target_path, "w") as f:
        f.write(XTB_WORKER_SCRIPT)


def _collect_unique_component_smiles(
    df: pd.DataFrame,
    reactant_col: str = "candidate_reactants",
    product_col: str = "candidate_product",
) -> List[str]:
    """Collect the set of unique component SMILES across all candidate rows.

    Splits multi-component SMILES (``"A.B"``) on ``"."`` and strips atom
    maps.  Duplicates are removed while preserving insertion order so the
    cache lookup is deterministic.
    """
    seen: set = set()
    ordered: List[str] = []
    for col in (reactant_col, product_col):
        if col not in df.columns:
            continue
        for val in df[col].astype(str):
            for comp in split_components(val):
                clean = strip_atom_maps(comp)
                if not clean or clean in seen:
                    continue
                seen.add(clean)
                ordered.append(clean)
    return ordered


def run_xtb_batch(
    smiles_list: List[str],
    dft_python: str = DEFAULT_DFT_PYTHON,
    method: str = DEFAULT_XTB_METHOD,
    solvent: Optional[str] = DEFAULT_XTB_SOLVENT,
    fmax: float = DEFAULT_XTB_FMAX,
    max_steps: int = DEFAULT_XTB_MAX_STEPS,
    seed: int = DEFAULT_RANDOM_SEED,
    worker_script_path: Optional[str] = None,
    log_dir: Optional[str] = None,
    timeout_per_molecule: float = 300.0,
) -> Dict[str, Dict]:
    """Run xTB on a batch of unique SMILES via a single subprocess call.

    Returns a dict ``{smiles: {energy_kcal_per_mol, status, method}}``.

    On failure (worker crash, timeout, missing dft python), every SMILES
    is mapped to ``{"energy_kcal_per_mol": None, "status": "xtb_batch_failed",
    "method": "xtb"}`` so the caller can degrade to MMFF94.
    """
    if not smiles_list:
        return {}

    # Embed 3D geometries in pc_cng_gpu env (where rdkit lives).
    # This step is independent of dft_python, so it runs first to surface
    # embed failures even when the dft env path is misconfigured.
    smiles_to_xyz: Dict[str, str] = {}
    embed_failures: Dict[str, Dict] = {}
    for smi in smiles_list:
        xyz, status = embed_smiles_to_xyz(smi, seed=seed)
        if xyz is None:
            embed_failures[smi] = {
                "energy_kcal_per_mol": None,
                "status": f"embed_failed:{status}",
                "method": "xtb",
            }
        else:
            smiles_to_xyz[smi] = xyz

    if not smiles_to_xyz:
        # All embeddings failed
        return {s: embed_failures[s] for s in smiles_list}

    # Write worker script + input/output JSON to a temp dir
    with tempfile.TemporaryDirectory(prefix="xtb_dft_") as tmpdir:
        worker_path = worker_script_path or os.path.join(tmpdir, "dft_xtb_worker.py")
        if worker_script_path is None:
            write_xtb_worker_script(worker_path)

        in_path = os.path.join(tmpdir, "xtb_input.json")
        out_path = os.path.join(tmpdir, "xtb_output.json")
        log_path = os.path.join(tmpdir, "xtb_worker.log") if log_dir is None else os.path.join(log_dir, "xtb_worker.log")

        request = {
            "smiles_to_xyz": smiles_to_xyz,
            "method": method,
            "solvent": solvent,
            "fmax": fmax,
            "max_steps": max_steps,
        }
        with open(in_path, "w") as f:
            json.dump(request, f)

        env = os.environ.copy()
        # CPU-only: xTB does not use GPU; ensure no GPU is reserved.
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "4")

        total_timeout = max(60.0, timeout_per_molecule * len(smiles_to_xyz))
        try:
            with open(log_path, "w") as logf:
                proc = subprocess.run(
                    [dft_python, worker_path, in_path, out_path],
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    timeout=total_timeout,
                    env=env,
                    check=False,
                )
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            return {
                s: embed_failures.get(s) or {
                    "energy_kcal_per_mol": None,
                    "status": "xtb_timeout",
                    "method": "xtb",
                }
                for s in smiles_list
            }
        except FileNotFoundError:
            return {
                s: embed_failures.get(s) or {
                    "energy_kcal_per_mol": None,
                    "status": "dft_python_missing",
                    "method": "xtb",
                }
                for s in smiles_list
            }

        # Parse output
        if not os.path.isfile(out_path):
            return {
                s: embed_failures.get(s) or {
                    "energy_kcal_per_mol": None,
                    "status": f"xtb_no_output:rc={rc}",
                    "method": "xtb",
                }
                for s in smiles_list
            }
        try:
            with open(out_path) as f:
                resp = json.load(f)
        except Exception as e:
            return {
                s: embed_failures.get(s) or {
                    "energy_kcal_per_mol": None,
                    "status": f"xtb_output_parse_failed:{e}",
                    "method": "xtb",
                }
                for s in smiles_list
            }

        results = resp.get("results", {})
        # Merge embed failures + xTB results
        out: Dict[str, Dict] = {}
        for s in smiles_list:
            if s in embed_failures:
                out[s] = embed_failures[s]
            elif s in results:
                out[s] = {
                    "energy_kcal_per_mol": results[s].get("energy_kcal_per_mol"),
                    "status": results[s].get("status", "unknown"),
                    "method": "xtb",
                    "energy_hartree": results[s].get("energy_hartree"),
                }
            else:
                out[s] = {
                    "energy_kcal_per_mol": None,
                    "status": "xtb_missing_from_results",
                    "method": "xtb",
                }
        return out


# --------------------------------------------------------------------------- #
# ORCA dispatch (stub — not implemented in this iteration)
# --------------------------------------------------------------------------- #
def compute_molecule_energy_orca(smiles: str, seed: int = 42) -> Dict:
    """ORCA DFT dispatch stub.

    ORCA is not yet wired up in the pc_cng_research environment.  This stub
    returns a ``not_implemented`` status so callers can degrade gracefully.
    """
    return {
        "smiles": smiles,
        "energy_kcal_per_mol": None,
        "status": "not_implemented",
        "method": "orca",
    }


# --------------------------------------------------------------------------- #
# Energy dispatch (single molecule)
# --------------------------------------------------------------------------- #
def compute_molecule_energy(
    smiles: str,
    method: str = "xtb",
    seed: int = 42,
    xtb_cache: Optional[Dict[str, Dict]] = None,
    dft_python: str = DEFAULT_DFT_PYTHON,
) -> Dict:
    """Dispatch a single-molecule energy calculation.

    For ``method="xtb"`` the result is read from ``xtb_cache`` (populated by
    :func:`run_xtb_batch`).  If the cache is missing or the entry is failed,
    the caller is responsible for degrading to MMFF94.

    For ``method="mmff94"`` RDKit MMFF94 is run directly in this process.

    For ``method="orca"`` the stub is returned.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return {"smiles": "", "energy_kcal_per_mol": None, "status": "empty", "method": method}

    clean = strip_atom_maps(smiles)

    if method == "xtb":
        if xtb_cache is None:
            return {"smiles": smiles, "energy_kcal_per_mol": None, "status": "no_xtb_cache", "method": "xtb"}
        entry = xtb_cache.get(clean)
        if entry is None:
            return {"smiles": smiles, "energy_kcal_per_mol": None, "status": "not_in_cache", "method": "xtb"}
        return {
            "smiles": smiles,
            "energy_kcal_per_mol": entry.get("energy_kcal_per_mol"),
            "status": entry.get("status", "unknown"),
            "method": "xtb",
        }
    if method == "mmff94":
        return compute_molecule_energy_mmff94(smiles, seed=seed)
    if method == "orca":
        return compute_molecule_energy_orca(smiles, seed=seed)
    return {"smiles": smiles, "energy_kcal_per_mol": None, "status": f"unknown_method:{method}", "method": method}


def compute_reaction_energy(
    reactants_smiles: str,
    products_smiles: str,
    method: str = "xtb",
    seed: int = 42,
    xtb_cache: Optional[Dict[str, Dict]] = None,
    dft_python: str = DEFAULT_DFT_PYTHON,
) -> Dict:
    """Compute ``DeltaG = E(products) - E(reactants)`` for a reaction.

    Multi-component SMILES (``"A.B.C"``) are split on ``"."`` and the
    component energies are summed.  The barrier estimate follows the
    P1-10 / P2-02 spec::

        barrier_estimate = |DeltaG| + BARRIER_CONSTANT kcal/mol

    Returns a dict with ``reactant_energy``, ``product_energy``,
    ``delta_g``, ``barrier_estimate``, ``status``, ``method``.
    """
    r_components = split_components(reactants_smiles)
    p_components = split_components(products_smiles)

    r_energy: Optional[float] = 0.0
    r_status = "ok"
    r_method_used = method
    for comp in r_components:
        res = compute_molecule_energy(
            comp, method=method, seed=seed, xtb_cache=xtb_cache, dft_python=dft_python
        )
        r_method_used = res["method"]
        if res["energy_kcal_per_mol"] is None:
            r_energy = None
            r_status = f"reactant_failed:{res['status']}"
            break
        r_energy += res["energy_kcal_per_mol"]

    p_energy: Optional[float] = 0.0
    p_status = "ok"
    p_method_used = method
    for comp in p_components:
        res = compute_molecule_energy(
            comp, method=method, seed=seed, xtb_cache=xtb_cache, dft_python=dft_python
        )
        p_method_used = res["method"]
        if res["energy_kcal_per_mol"] is None:
            p_energy = None
            p_status = f"product_failed:{res['status']}"
            break
        p_energy += res["energy_kcal_per_mol"]

    if r_energy is None or p_energy is None:
        return {
            "reactant_energy": r_energy,
            "product_energy": p_energy,
            "delta_g": None,
            "barrier_estimate": None,
            "status": f"{r_status};{p_status}",
            "method": r_method_used or p_method_used,
        }

    delta_g = float(p_energy - r_energy)
    barrier_estimate = abs(delta_g) + BARRIER_CONSTANT
    return {
        "reactant_energy": float(r_energy),
        "product_energy": float(p_energy),
        "delta_g": delta_g,
        "barrier_estimate": float(barrier_estimate),
        "status": "ok",
        "method": r_method_used or p_method_used,
    }


# --------------------------------------------------------------------------- #
# Judgment rule (P2-02)
# --------------------------------------------------------------------------- #
def judge_support(
    delta_g: Optional[float],
    barrier: Optional[float],
) -> str:
    """Apply the P2-02 chemoselectivity_error support rule.

    Rule (Section 26.1 P2-02):
      - ``DeltaG > 0 kcal/mol``  => supported (thermodynamically uphill
        => the proposed chemoselectivity_error is a true negative)

    The barrier-based fallback from P1-10 is intentionally NOT used here
    because the spec says DFT support is governed by the sign of
    ``DeltaG``.  ``barrier`` is kept as a parameter so the function
    signature stays compatible with P1-10 callers and tests.
    """
    if delta_g is None:
        return "inconclusive"
    if delta_g > DFT_SUPPORT_THRESHOLD:
        return "supported"
    return "not_supported"


def support_reason(
    delta_g: Optional[float],
    barrier: Optional[float],
) -> str:
    """Human-readable reason for the support verdict."""
    if delta_g is None:
        return "inconclusive: no DeltaG data"
    if delta_g > DFT_SUPPORT_THRESHOLD:
        return f"delta_g > 0 ({delta_g:+.2f} kcal/mol) -> unfavorable -> supports chemoselectivity_error"
    return f"not supported (delta_g={delta_g:+.2f} kcal/mol <= 0)"


# --------------------------------------------------------------------------- #
# Sampling / filtering
# --------------------------------------------------------------------------- #
def filter_by_failure_type(
    df: pd.DataFrame,
    failure_type: str,
) -> pd.DataFrame:
    """Return rows where ``failure_type == <failure_type>``.

    The CSV column ``failure_type`` is at column index 4 in
    ``high_confidence_negatives.csv``.  Case-sensitive exact match.
    """
    if "failure_type" not in df.columns:
        raise KeyError("candidates CSV is missing the 'failure_type' column")
    mask = df["failure_type"].astype(str) == failure_type
    return df[mask].copy().reset_index(drop=True)


def sample_candidates(
    df: pd.DataFrame,
    limit: int = 30,
    require_chemical_change: bool = True,
    deduplicate: bool = True,
) -> pd.DataFrame:
    """Sample high-confidence synthetic negatives.

    Sorted by ``hard_score`` descending (highest hard_score = most
    confident negative = least feasible synthetic = highest priority for
    DFT validation).  When ``require_chemical_change`` is True, rows
    with ``candidate_reactants == candidate_product`` are excluded.
    """
    df = df.copy()
    if "hard_score" in df.columns:
        df["hard_score"] = pd.to_numeric(df["hard_score"], errors="coerce")
        df = df.dropna(subset=["hard_score"])
    if require_chemical_change and "candidate_reactants" in df.columns and "candidate_product" in df.columns:
        mask = df["candidate_reactants"].astype(str) != df["candidate_product"].astype(str)
        df = df[mask].copy()
    if "hard_score" in df.columns:
        df = df.sort_values("hard_score", ascending=False, kind="mergesort")
    if deduplicate and "candidate_reactants" in df.columns and "candidate_product" in df.columns:
        df = df.drop_duplicates(subset=["candidate_reactants", "candidate_product"], keep="first")
    df = df.head(limit)
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "P2-02: DFT (xTB / MMFF94 / ORCA) validation of chemoselectivity_error "
            "synthetic negatives"
        ),
    )
    parser.add_argument(
        "--candidates",
        default="/home/cunyuliu/pc_cng_research/results/false_negative_three_layer_20260719/high_confidence_negatives.csv",
        help="Path to high_confidence_negatives.csv (P1-08 output)",
    )
    parser.add_argument(
        "--failure-type",
        default="chemoselectivity_error",
        help="Filter rows by this failure_type (default: chemoselectivity_error)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=30,
        help="Max number of chemoselectivity_error candidates to compute (default: 30)",
    )
    parser.add_argument(
        "--method",
        choices=["xtb", "mmff94", "orca"],
        default="xtb",
        help="Computational method (default: xtb; orca is a stub)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for per_candidate_results.csv + dft_validation_summary.json",
    )
    parser.add_argument(
        "--dft-python",
        default=DEFAULT_DFT_PYTHON,
        help=f"Path to dft conda env python (default: {DEFAULT_DFT_PYTHON})",
    )
    parser.add_argument(
        "--xtb-method",
        default=DEFAULT_XTB_METHOD,
        help="xTB Hamiltonian (default: GFN2-xTB)",
    )
    parser.add_argument(
        "--xtb-solvent",
        default=DEFAULT_XTB_SOLVENT,
        help="ALPB implicit solvent (default: water; empty string => gas phase)",
    )
    parser.add_argument(
        "--no-require-chemical-change",
        action="store_true",
        help="Do not require candidate_reactants != candidate_product",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=f"Random seed for RDKit embedding (default: {DEFAULT_RANDOM_SEED})",
    )
    return parser


def _build_summary(
    *,
    method_requested: str,
    method_actual: str,
    candidates_path: str,
    failure_type: str,
    total_loaded: int,
    total_filtered: int,
    n_computed: int,
    rows: List[Dict],
    go_no_go_threshold: float,
    seed: int,
    notes: Optional[List[str]] = None,
) -> Dict:
    # Accept either ``dG_reaction`` (the CSV column name written by main())
    # or ``delta_g`` (the internal compute_reaction_energy key) so this
    # function works for both synthetic test rows and the real CLI output.
    def _extract_dg(r: Dict) -> Optional[float]:
        v = r.get("dG_reaction", r.get("delta_g"))
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        return float(v)

    valid_deltas = [_extract_dg(r) for r in rows]
    valid_deltas = [d for d in valid_deltas if d is not None]
    n_supported = sum(1 for r in rows if r.get("support_verdict") == "supported")
    n_not_supported = sum(1 for r in rows if r.get("support_verdict") == "not_supported")
    n_inconclusive = sum(1 for r in rows if r.get("support_verdict") == "inconclusive")
    support_rate = n_supported / n_computed if n_computed > 0 else 0.0
    mean_dg = float(pd.Series(valid_deltas).mean()) if valid_deltas else None
    std_dg = float(pd.Series(valid_deltas).std()) if len(valid_deltas) > 1 else None
    go_no_go = "GO" if support_rate >= go_no_go_threshold else "NO_GO_partial_support"
    return {
        "task": "P2-02 DFT validation (chemoselectivity_error subset)",
        "method_requested": method_requested,
        "method_actual": method_actual,
        "degraded_from_requested": method_actual != method_requested,
        "candidates_path": os.path.abspath(candidates_path),
        "failure_type": failure_type,
        "total_candidates_loaded": int(total_loaded),
        "n_after_failure_type_filter": int(total_filtered),
        "n_computed": int(n_computed),
        "support_rule": f"delta_g > {DFT_SUPPORT_THRESHOLD:.1f} kcal/mol -> supports chemoselectivity_error",
        "n_supported": int(n_supported),
        "n_not_supported": int(n_not_supported),
        "n_inconclusive": int(n_inconclusive),
        "support_rate": float(support_rate),
        "mean_dg": mean_dg,
        "std_dg": std_dg,
        "min_dg": float(min(valid_deltas)) if valid_deltas else None,
        "max_dg": float(max(valid_deltas)) if valid_deltas else None,
        "median_dg": float(pd.Series(valid_deltas).median()) if valid_deltas else None,
        "go_no_go_threshold": float(go_no_go_threshold),
        "go_no_go_verdict": go_no_go,
        "xtb_method": DEFAULT_XTB_METHOD,
        "xtb_solvent": DEFAULT_XTB_SOLVENT,
        "seed": int(seed),
        "num_seeds_note": (
            "10-seed paired bootstrap is not required for DFT (deterministic); see "
            "dft_validation_protocol_20260720.md for details."
        ),
        "timestamp": pd.Timestamp.now().isoformat(),
        "notes": notes or [],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    # CPU-only: xTB does not use GPU.  Clear CUDA_VISIBLE_DEVICES so we
    # never accidentally reserve GPU 4.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    parser = _build_parser()
    args = parser.parse_args(argv)

    os.makedirs(args.output_dir, exist_ok=True)
    log_dir = os.path.join(args.output_dir, "detailed_logs")
    os.makedirs(log_dir, exist_ok=True)

    if not os.path.isfile(args.candidates):
        print(f"[error] candidates file not found: {args.candidates}", file=sys.stderr)
        return 2
    df = pd.read_csv(args.candidates)
    print(f"[info] loaded {len(df)} candidates from {args.candidates}")

    try:
        df_filtered = filter_by_failure_type(df, args.failure_type)
    except KeyError as e:
        print(f"[error] {e}", file=sys.stderr)
        return 3
    print(f"[info] after failure_type={args.failure_type!r} filter: {len(df_filtered)} rows")

    sampled = sample_candidates(
        df_filtered,
        limit=args.limit,
        require_chemical_change=not args.no_require_chemical_change,
    )
    print(f"[info] sampled {len(sampled)} candidates (limit={args.limit})")

    if len(sampled) == 0:
        print("[warn] no candidates to compute; writing empty outputs", file=sys.stderr)

    # Dispatch by method
    method_requested = args.method
    notes: List[str] = []
    xtb_cache: Optional[Dict[str, Dict]] = None

    if method_requested == "xtb":
        # Pre-populate the xTB cache by collecting all unique component SMILES
        # across sampled candidates and running one batch subprocess call.
        unique_smiles = _collect_unique_component_smiles(sampled)
        print(f"[info] xTB batch: {len(unique_smiles)} unique component SMILES")
        t0 = time.time()
        xtb_cache = run_xtb_batch(
            unique_smiles,
            dft_python=args.dft_python,
            method=args.xtb_method,
            solvent=args.xtb_solvent or None,
            seed=args.seed,
            log_dir=log_dir,
        )
        elapsed = time.time() - t0
        n_ok = sum(1 for v in xtb_cache.values() if v.get("status") == "ok")
        print(f"[info] xTB batch done in {elapsed:.1f}s ({n_ok}/{len(unique_smiles)} ok)")

        # Degrade to MMFF94 if every entry failed
        if xtb_cache and n_ok == 0:
            notes.append(
                "xtb batch returned 0/ok results; degrading per-molecule to MMFF94"
            )
            print("[warn] xTB batch all-failed; degrading to MMFF94 per-molecule", file=sys.stderr)
            method_actual = "mmff94"
        else:
            method_actual = "xtb"
    elif method_requested == "mmff94":
        method_actual = "mmff94"
    elif method_requested == "orca":
        method_actual = "orca"
        notes.append("ORCA dispatch is a stub (not implemented); energies will be None")
    else:
        method_actual = method_requested

    # Per-candidate computation
    rows: List[Dict] = []
    for idx, row in sampled.iterrows():
        source_id = str(row.get("source_id", ""))
        cand_r = str(row.get("candidate_reactants", ""))
        cand_p = str(row.get("candidate_product", ""))

        # Per-candidate xTB log (re-runs in mmff94 fallback mode are also logged)
        safe_id = source_id if source_id else f"{idx:04d}"
        cand_log = os.path.join(log_dir, f"{safe_id}.log")
        with open(cand_log, "w") as logf:
            logf.write(f"source_id: {source_id}\n")
            logf.write(f"candidate_reactants: {cand_r}\n")
            logf.write(f"candidate_product: {cand_p}\n")
            logf.write(f"method_requested: {method_requested}\n")
            logf.write(f"method_actual: {method_actual}\n")
            logf.write(f"failure_type: {row.get('failure_type', '')}\n")
            logf.write(f"edit_action: {row.get('edit_action', '')}\n")

        # If xtb batch failed wholesale, fall back per-molecule to MMFF94
        effective_method = method_actual if method_requested != "xtb" else (
            "xtb" if method_actual == "xtb" else "mmff94"
        )

        energy = compute_reaction_energy(
            cand_r,
            cand_p,
            method=effective_method,
            seed=args.seed,
            xtb_cache=xtb_cache,
            dft_python=args.dft_python,
        )
        verdict = judge_support(energy["delta_g"], energy["barrier_estimate"])
        rows.append({
            "source_id": source_id,
            "failure_type": str(row.get("failure_type", "")),
            "edit_action": str(row.get("edit_action", "")),
            "task": str(row.get("task", "")),
            "hard_score": float(row["hard_score"]) if pd.notna(row.get("hard_score")) else None,
            "false_negative_risk": float(row["false_negative_risk"]) if pd.notna(row.get("false_negative_risk")) else None,
            "candidate_reactants": cand_r,
            "candidate_product": cand_p,
            "dG_reactants": energy["reactant_energy"],
            "dG_products": energy["product_energy"],
            "dG_reaction": energy["delta_g"],
            "barrier_estimate": energy["barrier_estimate"],
            "method": energy["method"],
            "status": energy["status"],
            "supports_negative": bool(verdict == "supported"),
            "support_verdict": verdict,
            "support_reason": support_reason(energy["delta_g"], energy["barrier_estimate"]),
        })

        with open(cand_log, "a") as logf:
            logf.write(f"status: {energy['status']}\n")
            logf.write(f"dG_reactants: {energy['reactant_energy']}\n")
            logf.write(f"dG_products: {energy['product_energy']}\n")
            logf.write(f"dG_reaction: {energy['delta_g']}\n")
            logf.write(f"barrier_estimate: {energy['barrier_estimate']}\n")
            logf.write(f"verdict: {verdict}\n")
            logf.write(f"reason: {support_reason(energy['delta_g'], energy['barrier_estimate'])}\n")

    # Write per_candidate_results.csv
    res_df = pd.DataFrame(rows)
    res_csv = os.path.join(args.output_dir, "per_candidate_results.csv")
    res_df.to_csv(res_csv, index=False)
    print(f"[info] wrote per_candidate_results.csv ({len(res_df)} rows)")

    # Write summary.json
    summary = _build_summary(
        method_requested=method_requested,
        method_actual=method_actual,
        candidates_path=args.candidates,
        failure_type=args.failure_type,
        total_loaded=len(df),
        total_filtered=len(df_filtered),
        n_computed=len(rows),
        rows=rows,
        go_no_go_threshold=GO_NO_GO_THRESHOLD,
        seed=args.seed,
        notes=notes,
    )
    summary_path = os.path.join(args.output_dir, "dft_validation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[info] wrote dft_validation_summary.json")

    # Console summary
    print("\n" + "=" * 64)
    print("P2-02 DFT Validation (chemoselectivity_error subset) Summary")
    print("=" * 64)
    print(f"Method requested : {method_requested}")
    print(f"Method actual    : {method_actual}  (degraded: {method_actual != method_requested})")
    print(f"Failure type     : {args.failure_type}")
    print(f"Candidates       : loaded={len(df)} -> filtered={len(df_filtered)} -> computed={len(rows)}")
    if rows:
        n_sup = summary["n_supported"]
        rate = summary["support_rate"]
        print(f"Support          : {n_sup}/{len(rows)} = {rate:.1%}")
        print(f"Go/No-Go         : {summary['go_no_go_verdict']}  (threshold: {GO_NO_GO_THRESHOLD:.0%})")
        if summary["mean_dg"] is not None:
            print(
                f"DeltaG stats     : mean={summary['mean_dg']:+.2f}  median={summary['median_dg']:+.2f}  "
                f"min={summary['min_dg']:+.2f}  max={summary['max_dg']:+.2f}"
            )
    if notes:
        for n in notes:
            print(f"[note] {n}")
    print("=" * 64)

    return 0


if __name__ == "__main__":
    sys.exit(main())
