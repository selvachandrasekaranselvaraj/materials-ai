#!/usr/bin/env python
"""
BMD NIM — TensorNet-MatPES MD for crystal systems.

TensorNet-MatPES-r2SCAN-v2025.1 and TensorNet-MatPES-PBE-v2025.1 are
two of the MLIPs natively supported by the NVIDIA ALCHEMI BMD NIM.

ALCHEMI benchmark: 1.4 μs/atom/step on HGX B200 with TensorNet
                   at 350,000+ atom scale.

This module provides:
  - NVT and NPT runs using TensorNet (with MACE-MPA-0 fallback)
  - Per-run performance reporting (ns/day, μs/atom/step)
  - Comparison table against ALCHEMI B200 benchmark
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List

import numpy as np
from ase import units
from ase.io import read as ase_read, write as ase_write
from ase.md.langevin import Langevin
from ase.md.nptberendsen import NPTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution


ALCHEMI_BENCHMARK = {
    "hardware": "NVIDIA HGX B200",
    "model": "TensorNet-MatPES-r2SCAN-v2025.1",
    "n_atoms": 350_000,
    "us_per_atom_per_step": 1.4,
}


def get_tensornet_calculator(variant: str = "r2SCAN", device: str = "cuda"):
    """
    Load TensorNet-MatPES calculator.
    Falls back to MACE-MPA-0 if tensornet package is not installed.
    """
    model_tag = f"MatPES-{variant}-v2025.1"
    try:
        from tensornet.ase_interface import TensorNetCalculator
        return TensorNetCalculator(model=model_tag, device=device)
    except ImportError:
        print(f"[warn] tensornet not installed — falling back to MACE-MPA-0")
        try:
            from mace.calculators import mace_mp
            return mace_mp(model="medium", default_dtype="float32", device=device)
        except ImportError:
            from ase.calculators.emt import EMT
            print("[warn] mace-torch not installed — using EMT stub")
            return EMT()


def run_tensornet_md(
    structure_path: str,
    temperature_K: float = 300.0,
    ensemble: str = "nvt",
    n_steps: int = 1_000_000,
    timestep_fs: float = 1.0,
    pressure_GPa: float = 0.0,
    dump_freq: int = 5000,
    thermo_freq: int = 500,
    variant: str = "r2SCAN",
    device: str = "cuda",
    output_dir: str = "results/tensornet",
) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    atoms = ase_read(structure_path)
    n_atoms = len(atoms)
    calc = get_tensornet_calculator(variant=variant, device=device)
    atoms.calc = calc

    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K)
    dt = timestep_fs * units.fs

    if ensemble == "nvt":
        dyn = Langevin(atoms, timestep=dt, temperature_K=temperature_K,
                       friction=0.01 / units.fs, logfile=str(out / "md.log"))
    else:
        dyn = NPTBerendsen(atoms, timestep=dt, temperature_K=temperature_K,
                           pressure_au=pressure_GPa * units.GPa,
                           taut=100 * units.fs, taup=1000 * units.fs,
                           compressibility_au=4.57e-5 / units.bar,
                           logfile=str(out / "md.log"))

    traj_file = str(out / "traj.xyz")
    thermo = []

    def log():
        step = dyn.get_number_of_steps()
        T = atoms.get_temperature()
        E = atoms.get_potential_energy()
        thermo.append((step, T, E))
        if step % dump_freq == 0:
            ase_write(traj_file, atoms, append=(step > 0))

    dyn.attach(log, interval=thermo_freq)

    t0 = time.perf_counter()
    dyn.run(n_steps)
    elapsed = time.perf_counter() - t0

    ns_per_day = (n_steps * timestep_fs * 1e-6) / (elapsed / 86400)
    us_per_atom_step = elapsed * 1e6 / (n_steps * n_atoms)

    np.savetxt(out / "thermo.dat",
               np.array(thermo), header="step T_K E_pot_eV",
               fmt=["%.0f", "%.4f", "%.8f"])

    perf = {
        "n_atoms": n_atoms,
        "n_steps": n_steps,
        "elapsed_s": elapsed,
        "ns_per_day": ns_per_day,
        "us_per_atom_per_step": us_per_atom_step,
        "model": f"TensorNet-MatPES-{variant}",
        "ensemble": ensemble,
        "T_K": temperature_K,
    }
    _print_performance(perf)
    return perf


def _print_performance(perf: dict):
    benchmark_us = ALCHEMI_BENCHMARK["us_per_atom_per_step"]
    ratio = perf["us_per_atom_per_step"] / benchmark_us

    print(f"\n{'='*55}")
    print(f"TensorNet-MatPES MD — Performance Report")
    print(f"{'='*55}")
    print(f"  Atoms           : {perf['n_atoms']}")
    print(f"  Steps           : {perf['n_steps']}")
    print(f"  Elapsed         : {perf['elapsed_s']:.1f} s")
    print(f"  Throughput      : {perf['ns_per_day']:.2f} ns/day")
    print(f"  μs/atom/step    : {perf['us_per_atom_per_step']:.4f}")
    print(f"  ALCHEMI B200 ref: {benchmark_us:.1f} μs/atom/step "
          f"({ALCHEMI_BENCHMARK['n_atoms']:,} atoms, {ALCHEMI_BENCHMARK['hardware']})")
    print(f"  Ratio vs B200   : {ratio:.2f}×")
    print(f"{'='*55}\n")


def main():
    parser = argparse.ArgumentParser(
        description="BMD NIM TensorNet-MatPES MD with performance benchmarking"
    )
    parser.add_argument("--structure", required=True)
    parser.add_argument("--temp", type=float, default=300.0, help="K")
    parser.add_argument("--ensemble", default="nvt", choices=["nvt", "npt"])
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--timestep", type=float, default=1.0, help="fs")
    parser.add_argument("--pressure", type=float, default=0.0, help="GPa")
    parser.add_argument("--variant", default="r2SCAN", choices=["r2SCAN", "PBE"])
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--output_dir", default="results/tensornet")
    args = parser.parse_args()

    run_tensornet_md(
        structure_path=args.structure,
        temperature_K=args.temp,
        ensemble=args.ensemble,
        n_steps=args.steps,
        timestep_fs=args.timestep,
        pressure_GPa=args.pressure,
        variant=args.variant,
        device=args.device,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
