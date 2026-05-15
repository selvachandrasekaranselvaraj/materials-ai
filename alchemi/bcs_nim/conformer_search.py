#!/usr/bin/env python
"""
BCS NIM analogue — Batched Conformer Search.

Pipeline mirrors NVIDIA ALCHEMI BCS NIM:
  SMILES → RDKit 3-D embedding → AIMNet2/MACE geometry optimisation
  → Fmax threshold → energy ranking → CIF export.

Reference:
  https://developer.nvidia.com/blog/faster-chemistry-and-materials-discovery-with-ai-powered-simulations-using-nvidia-alchemi/
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

# RDKit
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

# ASE
from ase import Atoms
from ase.io import write as ase_write
from ase.optimize import LBFGS


@dataclass
class ConformerResult:
    smiles: str
    rank: int
    energy_eV: float
    fmax_eV_A: float
    converged: bool
    atoms: object  # ase.Atoms
    properties: dict = field(default_factory=dict)


def smiles_to_rdkit_conformers(smiles: str, n_confs: int = 10) -> List:
    """Generate 3-D starting structures with RDKit ETKDG."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    ids = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
    if len(ids) == 0:
        raise RuntimeError(f"RDKit failed to embed {smiles}")
    return mol, list(ids)


def rdkit_mol_conf_to_ase(mol, conf_id: int) -> Atoms:
    """Convert an RDKit conformer to an ASE Atoms object."""
    conf = mol.GetConformer(conf_id)
    positions = conf.GetPositions()        # Å
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    return Atoms(symbols=symbols, positions=positions)


def _get_calculator(model_name: str):
    """
    Return an ASE calculator for the requested MLIP backend.
    Supported: aimnet2, mace-mpa-0, mace-off-s/m/l
    Falls back to a zero-force stub when the model is not installed.
    """
    model_name = model_name.lower()

    if model_name == "aimnet2":
        try:
            from aimnet2calc import AIMNet2Calculator
            return AIMNet2Calculator("aimnet2")
        except ImportError:
            pass

    if model_name in ("mace-mpa-0", "mace_mpa_0"):
        try:
            from mace.calculators import mace_mp
            return mace_mp(model="medium", dispersion=False, default_dtype="float32")
        except ImportError:
            pass

    if model_name.startswith("mace-off"):
        size = model_name.split("-")[-1]      # s / m / l
        try:
            from mace.calculators import mace_off
            return mace_off(model=size, default_dtype="float32")
        except ImportError:
            pass

    # Stub — returns zero energies; useful for CI / unit tests
    from ase.calculators.emt import EMT
    print(f"[warn] {model_name} not found — falling back to EMT stub")
    return EMT()


def optimise_conformer(
    atoms: Atoms,
    calculator,
    fmax: float = 0.005,
    max_steps: int = 500,
) -> Tuple[Atoms, float, bool]:
    """Run LBFGS until Fmax < threshold or max_steps reached."""
    atoms.calc = calculator
    opt = LBFGS(atoms, logfile=None)
    converged = opt.run(fmax=fmax, steps=max_steps)
    energy = float(atoms.get_potential_energy())
    forces = atoms.get_forces()
    fmax_val = float(np.sqrt((forces ** 2).sum(axis=1).max()))
    return atoms, energy, fmax_val, converged


def batch_conformer_search(
    smiles: str,
    n_confs: int = 10,
    model: str = "aimnet2",
    fmax: float = 0.005,
    max_steps: int = 500,
    output_dir: str = "results/conformers",
) -> List[ConformerResult]:
    """
    Full BCS NIM pipeline for one SMILES string.

    Returns a list of ConformerResult sorted by energy (lowest first).
    """
    os.makedirs(output_dir, exist_ok=True)
    calc = _get_calculator(model)

    mol, conf_ids = smiles_to_rdkit_conformers(smiles, n_confs=n_confs)
    results: List[ConformerResult] = []

    for conf_id in conf_ids:
        atoms = rdkit_mol_conf_to_ase(mol, conf_id)
        atoms, energy, fmax_val, converged = optimise_conformer(
            atoms, calc, fmax=fmax, max_steps=max_steps
        )
        results.append(ConformerResult(
            smiles=smiles,
            rank=0,
            energy_eV=energy,
            fmax_eV_A=fmax_val,
            converged=converged,
            atoms=atoms,
            properties={
                "MW": Descriptors.MolWt(mol),
                "logP": Descriptors.MolLogP(mol),
            },
        ))

    # Sort by energy (lowest first) and assign ranks
    results.sort(key=lambda r: r.energy_eV)
    for i, r in enumerate(results):
        r.rank = i + 1

    # Save top-ranked structures as XYZ
    for r in results:
        out_path = os.path.join(output_dir, f"conf_rank{r.rank:03d}.xyz")
        ase_write(out_path, r.atoms)

    # Save summary JSON
    summary = [
        {
            "rank": r.rank,
            "energy_eV": r.energy_eV,
            "fmax_eV_A": r.fmax_eV_A,
            "converged": r.converged,
            "smiles": r.smiles,
            "properties": r.properties,
        }
        for r in results
    ]
    class _Enc(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.bool_,)): return bool(o)
            if isinstance(o, (np.integer,)): return int(o)
            if isinstance(o, (np.floating,)): return float(o)
            return super().default(o)

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, cls=_Enc)

    print(f"Optimised {len(results)} conformers for {smiles}")
    for r in results[:3]:
        print(f"  rank {r.rank}: E={r.energy_eV:.4f} eV  Fmax={r.fmax_eV_A:.4f} eV/Å"
              f"  converged={r.converged}")
    return results


def main():
    parser = argparse.ArgumentParser(description="BCS NIM — batched conformer search")
    parser.add_argument("--smiles", type=str, required=True,
                        help="SMILES string of the target molecule")
    parser.add_argument("--n_confs", type=int, default=10,
                        help="Number of RDKit starting conformers")
    parser.add_argument("--model", type=str, default="aimnet2",
                        choices=["aimnet2", "mace-mpa-0", "mace-off-s", "mace-off-m", "mace-off-l"],
                        help="MLIP backend (mirrors ALCHEMI BCS NIM supported models)")
    parser.add_argument("--fmax", type=float, default=0.005,
                        help="Force convergence threshold eV/Å (ALCHEMI BCS NIM default: 0.005)")
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--output_dir", type=str, default="results/conformers")
    args = parser.parse_args()

    results = batch_conformer_search(
        smiles=args.smiles,
        n_confs=args.n_confs,
        model=args.model,
        fmax=args.fmax,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
    )
    print(f"\nTop conformer energy: {results[0].energy_eV:.6f} eV")


if __name__ == "__main__":
    main()
