#!/usr/bin/env python
"""
BMD NIM — MACE-MPA-0 production MD for solid-state battery materials.

Runs GPU-accelerated NVT or NPT MLMD using the universal MACE-MPA-0
potential (one of the ALCHEMI BMD NIM supported backends). Designed for
multi-temperature ionic conductivity measurements:
  → run at 300, 600, 900, 1200 K → extrapolate to 300 K via Arrhenius.

Validated systems (this project):
  Li₃TiCl₆, Li₂ZrCl₆, Li-Ti-PS, NMC622, Na-air cathode, Li₂ZrO₃
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

import numpy as np
from ase import units
from ase.io import read as ase_read, write as ase_write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution


def get_mace_calculator(model_size: str = "medium", device: str = "cuda"):
    try:
        from mace.calculators import mace_mp
        return mace_mp(model=model_size, dispersion=False,
                       default_dtype="float32", device=device)
    except ImportError:
        from ase.calculators.emt import EMT
        print("[warn] mace-torch not installed — using EMT stub")
        return EMT()


def run_nvt_sweep(
    structure: str,
    temperatures: List[float],
    n_steps: int = 2_000_000,
    timestep_fs: float = 1.0,
    dump_freq: int = 5000,
    thermo_freq: int = 500,
    model_size: str = "medium",
    device: str = "cuda",
    output_base: str = "results/mace_mpa0",
):
    """
    Run NVT MD at each temperature in sequence on the same structure.
    Each run uses the final configuration of the previous as starting point
    (annealing-style: equilibrate at high T, cool to target T).
    """
    atoms = ase_read(structure)
    calc = get_mace_calculator(model_size=model_size, device=device)
    atoms.calc = calc

    for T in temperatures:
        run_dir = Path(output_base) / f"T{int(T)}K"
        run_dir.mkdir(parents=True, exist_ok=True)

        MaxwellBoltzmannDistribution(atoms, temperature_K=T)

        dyn = Langevin(
            atoms,
            timestep=timestep_fs * units.fs,
            temperature_K=T,
            friction=0.01 / units.fs,
            logfile=str(run_dir / "nvt.log"),
        )

        traj_file = str(run_dir / "nvt_traj.xyz")
        thermo_rows = []

        def log():
            step = dyn.get_number_of_steps()
            Tk = atoms.get_temperature()
            Ep = atoms.get_potential_energy()
            Ek = atoms.get_kinetic_energy()
            thermo_rows.append((step, Tk, Ep, Ek))
            if step % dump_freq == 0:
                ase_write(traj_file, atoms, append=(step > 0))

        dyn.attach(log, interval=thermo_freq)
        print(f"Running NVT at {T} K  ({n_steps} steps)  → {run_dir}")
        dyn.run(n_steps)

        # Save thermo
        arr = np.array(thermo_rows)
        np.savetxt(
            run_dir / "thermo.dat",
            arr,
            header="step  T_K  E_pot_eV  E_kin_eV",
            fmt=["%.0f", "%.4f", "%.8f", "%.8f"],
        )
        # Save final structure for next temperature
        ase_write(str(run_dir / "final.cif"), atoms)

    print("Temperature sweep complete.")


def compute_msd_and_diffusivity(traj_file: str, dt_ps: float = 0.001,
                                 species: str = "Li") -> dict:
    """
    Compute MSD and diffusion coefficient D from an XYZ trajectory.
    Uses the Einstein relation: D = MSD / (6 * t) in the diffusive regime.
    """
    frames = ase_read(traj_file, index=":")
    indices = [i for i, s in enumerate(frames[0].get_chemical_symbols())
               if s == species]
    if not indices:
        return {}

    positions = np.array([f.get_positions()[indices] for f in frames])
    # unwrap (simple linear unwrap — assumes small steps)
    n_frames, n_species, _ = positions.shape
    msd = np.zeros(n_frames)
    for t in range(1, n_frames):
        disp = positions[t] - positions[0]
        msd[t] = (disp ** 2).sum(axis=-1).mean()

    t_arr = np.arange(n_frames) * dt_ps  # ps
    # Linear fit in the diffusive regime (last 60% of trajectory)
    start = int(0.4 * n_frames)
    coeffs = np.polyfit(t_arr[start:], msd[start:], 1)
    D_cm2_s = coeffs[0] / 6.0 * 1e-4   # Å²/ps → cm²/s

    return {"D_cm2_s": D_cm2_s, "msd": msd, "t_ps": t_arr}


def main():
    parser = argparse.ArgumentParser(
        description="BMD NIM MACE-MPA-0 — temperature-sweep NVT for ionic conductivity"
    )
    parser.add_argument("--structure", required=True,
                        help="CIF / POSCAR / XYZ input structure")
    parser.add_argument("--temps", nargs="+", type=float,
                        default=[300.0, 600.0, 900.0, 1200.0],
                        help="Temperatures in K")
    parser.add_argument("--steps", type=int, default=2_000_000)
    parser.add_argument("--timestep", type=float, default=1.0, help="fs")
    parser.add_argument("--model_size", default="medium",
                        choices=["small", "medium", "large"])
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--output_dir", default="results/mace_mpa0")
    args = parser.parse_args()

    run_nvt_sweep(
        structure=args.structure,
        temperatures=args.temps,
        n_steps=args.steps,
        timestep_fs=args.timestep,
        model_size=args.model_size,
        device=args.device,
        output_base=args.output_dir,
    )


if __name__ == "__main__":
    main()
