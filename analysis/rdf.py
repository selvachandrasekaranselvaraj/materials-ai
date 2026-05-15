#!/usr/bin/env python
"""
Radial distribution function (RDF) and partial RDF from MLMD trajectories.

Computes g(r) for all or selected species pairs using an efficient
histogram binning approach. Supports both wrapped (periodic) and
unwrapped trajectories.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from ase.io import read as ase_read


def minimum_image_distance(r: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Apply minimum image convention for orthorhombic cells."""
    for i in range(3):
        r[..., i] -= cell[i] * np.round(r[..., i] / cell[i])
    return r


def compute_rdf(
    traj_file: str,
    r_max: float = 8.0,
    dr: float = 0.05,
    species_pair: Optional[Tuple[str, str]] = None,
    n_frames_max: int = None,
    skip_first: int = 100,
) -> Dict[str, np.ndarray]:
    """
    Compute g(r) averaged over trajectory frames.

    Parameters
    ----------
    species_pair : (str, str) or None
        If None, compute total RDF. E.g. ("Li", "Cl") for partial RDF.
    skip_first : int
        Skip the first N frames (equilibration).
    """
    frames = ase_read(traj_file, index=":")
    frames = frames[skip_first:]
    if n_frames_max:
        frames = frames[:n_frames_max]

    n_bins = int(r_max / dr)
    r_edges = np.linspace(0, r_max, n_bins + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    histogram = np.zeros(n_bins)

    n_pairs_total = 0
    n_frames = len(frames)

    for atoms in frames:
        pos = atoms.get_positions()
        cell_diag = np.diag(atoms.get_cell())
        symbols = np.array(atoms.get_chemical_symbols())
        n = len(pos)

        if species_pair is not None:
            idx_A = np.where(symbols == species_pair[0])[0]
            idx_B = np.where(symbols == species_pair[1])[0]
        else:
            idx_A = np.arange(n)
            idx_B = np.arange(n)

        for i in idx_A:
            if species_pair is not None:
                partners = idx_B
            else:
                partners = np.arange(i + 1, n)

            if len(partners) == 0:
                continue
            rij = pos[partners] - pos[i]
            rij = minimum_image_distance(rij, cell_diag)
            dist = np.linalg.norm(rij, axis=1)
            mask = (dist > 0) & (dist < r_max)
            histogram += np.histogram(dist[mask], bins=r_edges)[0]
            n_pairs_total += mask.sum()

    # Normalise to g(r)
    volume = np.prod(np.diag(frames[0].get_cell()))
    n_total = len(frames[0])
    rho = n_total / volume     # average number density

    shell_volumes = (4.0 / 3.0) * np.pi * (r_edges[1:] ** 3 - r_edges[:-1] ** 3)
    normalisation = rho * n_frames * (n_pairs_total / n_frames + 1e-10) / n_total
    gr = histogram / (shell_volumes * normalisation + 1e-30)

    return {"r": r_centers, "gr": gr}


def compute_all_partial_rdfs(
    traj_file: str,
    species_list: List[str],
    r_max: float = 8.0,
    dr: float = 0.05,
    skip_first: int = 100,
    output_dir: str = "results/rdf",
) -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)

    # All unique pairs
    pairs = [(a, b) for i, a in enumerate(species_list)
             for b in species_list[i:]]

    all_data = {"r": None}
    for pair in pairs:
        label = f"g({pair[0]}-{pair[1]})"
        print(f"Computing {label}...")
        res = compute_rdf(traj_file, r_max=r_max, dr=dr,
                          species_pair=pair, skip_first=skip_first)
        all_data["r"] = res["r"]
        all_data[label] = res["gr"]

    df = pd.DataFrame(all_data)
    csv_path = Path(output_dir) / "partial_rdfs.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved partial RDFs → {csv_path}")
    return df


def find_coordination_number(r: np.ndarray, gr: np.ndarray,
                              rho: float, r_min: float, r_max: float) -> float:
    """Integrate g(r) to get coordination number N = 4π ρ ∫ g(r) r² dr."""
    mask = (r >= r_min) & (r <= r_max)
    dr = r[1] - r[0]
    return float(4 * np.pi * rho * np.trapz(gr[mask] * r[mask] ** 2, r[mask]))


def main():
    parser = argparse.ArgumentParser(description="RDF from MLMD trajectory")
    parser.add_argument("--traj", required=True)
    parser.add_argument("--species", nargs="+", default=["Li", "Cl"],
                        help="Species for partial RDFs")
    parser.add_argument("--r_max", type=float, default=8.0)
    parser.add_argument("--dr", type=float, default=0.05)
    parser.add_argument("--skip", type=int, default=100,
                        help="Equilibration frames to skip")
    parser.add_argument("--output_dir", default="results/rdf")
    args = parser.parse_args()

    df = compute_all_partial_rdfs(
        traj_file=args.traj,
        species_list=args.species,
        r_max=args.r_max,
        dr=args.dr,
        skip_first=args.skip,
        output_dir=args.output_dir,
    )
    print(df.head())


if __name__ == "__main__":
    main()
