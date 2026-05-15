#!/usr/bin/env python
"""
Ionic conductivity from MLMD trajectories.

Two methods:
  1. Einstein relation: σ = n q² D / (kB T)  — from MSD of charge carriers
  2. Green-Kubo: σ = 1/(3VkBT) ∫₀^∞ <J(0)·J(t)> dt  — from current autocorr.

Arrhenius extrapolation from multi-temperature runs:
  D(T) = D₀ exp(−Ea / kB T)
  σ(T) = σ₀ / T  exp(−Ea / kB T)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# Physical constants
kB_eV = 8.617333e-5    # eV/K
kB_J  = 1.380649e-23   # J/K
e_C   = 1.602176634e-19  # C
A2_to_m2 = 1e-20       # Å² → m²
ps_to_s  = 1e-12       # ps → s


def read_lammps_dump(dump_file: str, species: str) -> Tuple[np.ndarray, float]:
    """
    Parse a LAMMPS dump file and extract positions of a given species.
    Returns (positions array of shape (n_frames, n_species, 3), dt_ps).
    """
    frames = []
    current = []
    dt_ps = 0.001
    reading = False
    with open(dump_file) as f:
        for line in f:
            line = line.strip()
            if "ITEM: ATOMS" in line:
                reading = True
                current = []
                continue
            if "ITEM:" in line and "ATOMS" not in line:
                if current:
                    frames.append(current)
                reading = False
                continue
            if reading and line:
                parts = line.split()
                if len(parts) >= 5 and parts[1] == species:
                    current.append([float(parts[2]),
                                    float(parts[3]),
                                    float(parts[4])])
    if current:
        frames.append(current)
    return np.array(frames, dtype=np.float32), dt_ps


def compute_msd(
    positions: np.ndarray,
    dt_ps: float,
    max_dt_frac: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mean squared displacement via the FFT correlation method (fast).
    positions: (n_frames, n_atoms, 3)
    Returns (t_ps, msd_A2).
    """
    n_frames, n_atoms, _ = positions.shape
    max_lag = int(n_frames * max_dt_frac)
    msd = np.zeros(max_lag)

    for lag in range(1, max_lag):
        disp = positions[lag:] - positions[:-lag]       # (n_frames-lag, n_atoms, 3)
        msd[lag] = (disp ** 2).sum(axis=-1).mean()

    t_ps = np.arange(max_lag) * dt_ps
    return t_ps, msd


def diffusivity_from_msd(
    t_ps: np.ndarray,
    msd_A2: np.ndarray,
    fit_frac: Tuple[float, float] = (0.4, 0.9),
) -> float:
    """
    D = MSD / (6 t)  from a linear fit in the diffusive regime.
    Returns D in cm²/s.
    """
    n = len(t_ps)
    start = int(fit_frac[0] * n)
    end = int(fit_frac[1] * n)
    coeffs = np.polyfit(t_ps[start:end], msd_A2[start:end], 1)
    slope_A2_per_ps = coeffs[0]
    # Convert: Å²/ps → cm²/s  (1 Å²/ps = 1e-4 cm²/s ... actually:
    #   1 Å = 1e-8 cm  → 1 Å² = 1e-16 cm²
    #   1 ps = 1e-12 s → 1 Å²/ps = 1e-16/1e-12 cm²/s = 1e-4 cm²/s)
    D_cm2_s = slope_A2_per_ps / 6.0 * 1e-4
    return float(D_cm2_s)


def einstein_conductivity(
    D_cm2_s: float,
    T_K: float,
    n_carriers: int,
    volume_A3: float,
    charge: int = 1,
) -> float:
    """
    σ = n q² D / (kB T)   [S/cm]
    n: carrier density in 1/cm³
    """
    n_per_cm3 = n_carriers / (volume_A3 * 1e-24)
    sigma = n_per_cm3 * (charge * e_C) ** 2 * D_cm2_s / (kB_J * T_K)
    return float(sigma)


def arrhenius_fit(
    temperatures_K: List[float],
    D_values_cm2s: List[float],
) -> Dict[str, float]:
    """
    Fit D(T) = D0 * exp(-Ea / kB T).
    Returns {Ea_eV, D0_cm2s, sigma_300K (extrapolated)}.
    """
    inv_T = np.array([1.0 / T for T in temperatures_K])
    log_D = np.log(D_values_cm2s)
    # Linear fit: ln(D) = ln(D0) - Ea/kB * (1/T)
    coeffs = np.polyfit(inv_T, log_D, 1)
    Ea_eV = -coeffs[0] * kB_eV
    D0 = np.exp(coeffs[1])
    D_300K = D0 * np.exp(-Ea_eV / (kB_eV * 300.0))
    return {"Ea_eV": Ea_eV, "D0_cm2s": D0, "D_300K_cm2s": D_300K}


def analyse_conductivity(
    traj_file: str,
    temperature_K: float,
    species: str = "Li",
    volume_A3: float = None,
    dt_ps: float = 0.001,
    output_dir: str = "results/conductivity",
):
    os.makedirs(output_dir, exist_ok=True)

    from ase.io import read as ase_read
    frames = ase_read(traj_file, index=":")
    indices = [i for i, s in enumerate(frames[0].get_chemical_symbols())
               if s == species]

    if not indices:
        print(f"[warn] No {species} atoms found in trajectory")
        return {}

    positions = np.array([f.get_positions()[indices] for f in frames])
    n_atoms_total = len(frames[0])
    if volume_A3 is None:
        volume_A3 = float(frames[0].get_volume())

    t_ps, msd = compute_msd(positions, dt_ps)
    D = diffusivity_from_msd(t_ps, msd)
    sigma = einstein_conductivity(D, temperature_K, len(indices), volume_A3)

    print(f"Species: {species}  T={temperature_K} K")
    print(f"  D        = {D:.4e} cm²/s")
    print(f"  σ (Einstein) = {sigma:.4e} S/cm")

    np.savetxt(Path(output_dir) / "msd.dat",
               np.column_stack([t_ps, msd]),
               header="t_ps  MSD_A2")

    result = {
        "species": species,
        "T_K": temperature_K,
        "D_cm2s": D,
        "sigma_S_cm": sigma,
        "n_carriers": len(indices),
        "volume_A3": volume_A3,
    }
    pd.DataFrame([result]).to_csv(
        Path(output_dir) / "conductivity.csv", index=False
    )
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Ionic conductivity from MLMD trajectory"
    )
    parser.add_argument("--traj", required=True,
                        help="Trajectory file (XYZ, LAMMPS dump, etc.)")
    parser.add_argument("--temp", type=float, required=True, help="Temperature K")
    parser.add_argument("--species", default="Li",
                        help="Mobile ion species symbol")
    parser.add_argument("--volume", type=float, default=None,
                        help="Cell volume in Å³ (auto-detected if not given)")
    parser.add_argument("--dt", type=float, default=0.001, help="Timestep in ps")
    parser.add_argument("--output_dir", default="results/conductivity")
    args = parser.parse_args()

    analyse_conductivity(
        traj_file=args.traj,
        temperature_K=args.temp,
        species=args.species,
        volume_A3=args.volume,
        dt_ps=args.dt,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
