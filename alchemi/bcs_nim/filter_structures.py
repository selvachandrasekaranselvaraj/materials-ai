#!/usr/bin/env python
"""
BCS NIM — structure filtering step.

After geometry optimisation (conformer_search.py), this module:
  1. Validates molecular connectivity (no bond-breaking during opt)
  2. Removes duplicate structures (RMSD threshold)
  3. Applies energy window filter
  4. Writes the curated set as XYZ and a ranked summary CSV

Mirrors the ALCHEMI BCS NIM curation stage:
  connectivity validation → deduplication → energy filtering.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import read as ase_read, write as ase_write


# ---------------------------------------------------------------------------
# Connectivity validation
# ---------------------------------------------------------------------------

COVALENT_RADII = {
    "H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "S": 1.05,
    "F": 0.57, "Cl": 1.02, "Br": 1.20, "I": 1.39, "P": 1.07,
    "Li": 1.28, "Na": 1.66, "K": 2.03, "Ca": 1.76,
}
BOND_TOLERANCE = 0.25  # Å beyond sum of covalent radii


def build_connectivity(atoms: Atoms) -> set:
    """Return frozenset of bonded pairs (i, j) with i < j."""
    symbols = atoms.get_chemical_symbols()
    pos = atoms.get_positions()
    bonds = set()
    for i in range(len(atoms)):
        ri = COVALENT_RADII.get(symbols[i], 1.5)
        for j in range(i + 1, len(atoms)):
            rj = COVALENT_RADII.get(symbols[j], 1.5)
            d = np.linalg.norm(pos[i] - pos[j])
            if d < ri + rj + BOND_TOLERANCE:
                bonds.add((i, j))
    return bonds


def connectivity_ok(atoms_initial: Atoms, atoms_final: Atoms) -> bool:
    """Return True if topology is unchanged after optimisation."""
    return build_connectivity(atoms_initial) == build_connectivity(atoms_final)


# ---------------------------------------------------------------------------
# RMSD deduplication
# ---------------------------------------------------------------------------

def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """Minimum RMSD between two (N,3) coordinate arrays via Kabsch rotation."""
    P = P - P.mean(axis=0)
    Q = Q - Q.mean(axis=0)
    H = P.T @ Q
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    P_rot = P @ R.T
    return float(np.sqrt(((P_rot - Q) ** 2).sum() / len(P)))


def deduplicate(atoms_list: List[Atoms], rmsd_threshold: float = 0.3) -> List[int]:
    """Return indices of unique structures (first-occurrence kept)."""
    kept = []
    for i, ai in enumerate(atoms_list):
        is_dup = False
        pi = ai.get_positions()
        for k in kept:
            ak = atoms_list[k]
            if len(ai) != len(ak):
                continue
            rmsd = kabsch_rmsd(pi, ak.get_positions())
            if rmsd < rmsd_threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(i)
    return kept


# ---------------------------------------------------------------------------
# Energy window filter
# ---------------------------------------------------------------------------

def energy_filter(
    energies: np.ndarray,
    window_eV: float = 0.5,
) -> List[int]:
    """Keep structures within window_eV of the global minimum."""
    e_min = energies.min()
    return [i for i, e in enumerate(energies) if e - e_min <= window_eV]


# ---------------------------------------------------------------------------
# Full filtering pipeline
# ---------------------------------------------------------------------------

def filter_conformers(
    input_dir: str = "results/conformers",
    output_dir: str = "results/conformers_filtered",
    rmsd_threshold: float = 0.30,
    energy_window_eV: float = 0.50,
    check_connectivity: bool = False,
    original_xyz: str = None,
) -> pd.DataFrame:
    """
    Load optimised XYZ files from input_dir, apply filters, write curated set.

    Parameters
    ----------
    check_connectivity : bool
        If True, requires original_xyz to compare pre/post connectivity.
    """
    os.makedirs(output_dir, exist_ok=True)
    summary_path = Path(input_dir) / "summary.json"

    if not summary_path.exists():
        raise FileNotFoundError(f"No summary.json in {input_dir}")

    with open(summary_path) as f:
        records = json.load(f)

    # Load atoms
    all_atoms: List[Atoms] = []
    for rec in records:
        xyz = Path(input_dir) / f"conf_rank{rec['rank']:03d}.xyz"
        if xyz.exists():
            all_atoms.append(ase_read(str(xyz)))
        else:
            all_atoms.append(None)

    valid_idx = [i for i, a in enumerate(all_atoms) if a is not None]

    # 1. Connectivity check (optional)
    if check_connectivity and original_xyz:
        ref = ase_read(original_xyz)
        valid_idx = [i for i in valid_idx
                     if connectivity_ok(ref, all_atoms[i])]
        print(f"After connectivity filter: {len(valid_idx)} structures")

    # 2. Energy filter
    energies = np.array([records[i]["energy_eV"] for i in valid_idx])
    keep_e = energy_filter(energies, window_eV=energy_window_eV)
    valid_idx = [valid_idx[k] for k in keep_e]
    print(f"After energy filter ({energy_window_eV} eV window): {len(valid_idx)}")

    # 3. RMSD deduplication
    subset = [all_atoms[i] for i in valid_idx]
    unique_local = deduplicate(subset, rmsd_threshold=rmsd_threshold)
    valid_idx = [valid_idx[k] for k in unique_local]
    print(f"After RMSD deduplication ({rmsd_threshold} Å): {len(valid_idx)}")

    # 4. Write curated set
    rows = []
    for new_rank, orig_idx in enumerate(valid_idx, start=1):
        rec = records[orig_idx]
        atoms = all_atoms[orig_idx]
        out_xyz = Path(output_dir) / f"conf_curated_{new_rank:03d}.xyz"
        ase_write(str(out_xyz), atoms)
        rows.append({
            "curated_rank": new_rank,
            "original_rank": rec["rank"],
            "energy_eV": rec["energy_eV"],
            "fmax_eV_A": rec["fmax_eV_A"],
            "converged": rec["converged"],
            "smiles": rec["smiles"],
            **rec.get("properties", {}),
        })

    df = pd.DataFrame(rows)
    csv_path = Path(output_dir) / "curated_conformers.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} curated conformers → {csv_path}")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="BCS NIM filtering: connectivity + dedup + energy window"
    )
    parser.add_argument("--input_dir", default="results/conformers")
    parser.add_argument("--output_dir", default="results/conformers_filtered")
    parser.add_argument("--rmsd_threshold", type=float, default=0.30,
                        help="RMSD cutoff for deduplication (Å)")
    parser.add_argument("--energy_window", type=float, default=0.50,
                        help="Energy window above global min to keep (eV)")
    parser.add_argument("--check_connectivity", action="store_true",
                        help="Validate bond topology unchanged after optimisation")
    parser.add_argument("--original_xyz", type=str, default=None,
                        help="Original (pre-opt) XYZ for connectivity reference")
    args = parser.parse_args()

    df = filter_conformers(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        rmsd_threshold=args.rmsd_threshold,
        energy_window_eV=args.energy_window,
        check_connectivity=args.check_connectivity,
        original_xyz=args.original_xyz,
    )
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
