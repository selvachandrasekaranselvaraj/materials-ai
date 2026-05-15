#!/usr/bin/env python
"""Arrhenius ionic conductivity analysis on real Li2ZrCl6 MLMD trajectories."""
import sys, numpy as np, pandas as pd, os
sys.path.insert(0, '.')
from analysis.conductivity import (compute_msd, diffusivity_from_msd,
                                   einstein_conductivity, arrhenius_fit)


def parse_dump(dump_file, species, max_frames=300, stride=20):
    """Read unwrapped LAMMPS dump; match by element name in last column."""
    frames, current = [], []
    reading, n_frame = False, 0
    with open(dump_file) as f:
        for line in f:
            line = line.strip()
            if "ITEM: ATOMS" in line:
                if current and n_frame % stride == 0:
                    frames.append(current)
                    if len(frames) >= max_frames:
                        break
                current = []
                reading = True
                n_frame += 1
                continue
            if "ITEM:" in line and "ATOMS" not in line:
                reading = False
                continue
            if reading and line:
                parts = line.split()
                if len(parts) >= 6 and parts[-1] == species:
                    current.append([float(parts[2]),
                                    float(parts[3]),
                                    float(parts[4])])
    if current and n_frame % stride == 0:
        frames.append(current)
    return np.array(frames, dtype=np.float64)


# Li2ZrCl6: 11232 atoms, cubic 64.38 Å box
vol_A3 = 64.38 ** 3
temps = [400, 500, 600, 700, 800]
base = "/projects/nmclps/LMZC/dlmd"

results, Ds, Ts = [], [], []
header = "{:>5}  {:>7}  {:>5}  {:>12}  {:>12}".format(
    "T(K)", "frames", "n_Li", "D(cm2/s)", "sigma(S/cm)")
print(header)
print("-" * len(header))

for T in temps:
    dump_file = os.path.join(base, str(T), "dump_unwrapped.lmp")
    if not os.path.exists(dump_file):
        continue
    pos = parse_dump(dump_file, "Li", max_frames=300, stride=20)
    if len(pos) < 10:
        print("  T={}K: only {} frames, skipping".format(T, len(pos)))
        continue
    n_Li = pos.shape[1]
    t_ps, msd = compute_msd(pos, dt_ps=0.02)   # stride 20 x 0.001 ps
    D = diffusivity_from_msd(t_ps, msd)
    sigma = einstein_conductivity(D, T, n_Li, vol_A3)
    print("{:>5}  {:>7}  {:>5}  {:>12.4e}  {:>12.4e}".format(
        T, len(pos), n_Li, D, sigma))
    results.append({"T_K": T, "D_cm2s": D, "sigma_S_cm": sigma})
    Ds.append(D)
    Ts.append(float(T))

os.makedirs("results/analysis/conductivity", exist_ok=True)
pd.DataFrame(results).to_csv(
    "results/analysis/conductivity/Li2ZrCl6_conductivity.csv", index=False)

if len(Ts) >= 3:
    arh = arrhenius_fit(Ts, Ds)
    print()
    print("--- Arrhenius fit (Li2ZrCl6) ---")
    print("  Ea      = {:.3f} eV".format(arh["Ea_eV"]))
    print("  D0      = {:.3e} cm2/s".format(arh["D0_cm2s"]))
    print("  D(300K) = {:.3e} cm2/s  (extrapolated to room temperature)".format(
        arh["D_300K_cm2s"]))
    pd.DataFrame([arh]).to_csv(
        "results/analysis/conductivity/Li2ZrCl6_arrhenius.csv", index=False)
    print("Saved -> results/analysis/conductivity/")
