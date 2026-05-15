#!/usr/bin/env python
"""
Active-learning snapshot selector for DeepMD-kit training set expansion.

Reads an MD trajectory (XYZ / VASP XDATCAR / LAMMPS dump), selects
structurally diverse frames via farthest-point sampling in descriptor
space, and writes them as DeepMD-compatible numpy sets ready for the
next training iteration.

Used after BCS NIM conformer search or after a BMD NIM production run
finds configurations outside the training set convex hull.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
from ase.io import read as ase_read


# ---------------------------------------------------------------------------
# Descriptor: lightweight Coulomb-matrix fingerprint (no external deps)
# ---------------------------------------------------------------------------

def coulomb_fingerprint(atoms, n_max: int = 20) -> np.ndarray:
    """
    Sorted Coulomb matrix eigenvalue fingerprint.
    Fast, environment-free descriptor sufficient for diversity sampling.
    """
    Z = np.array(atoms.get_atomic_numbers(), dtype=float)
    pos = atoms.get_positions()
    n = len(Z)
    CM = np.zeros((n, n))
    for i in range(n):
        CM[i, i] = 0.5 * Z[i] ** 2.4
        for j in range(i + 1, n):
            d = np.linalg.norm(pos[i] - pos[j]) + 1e-8
            CM[i, j] = CM[j, i] = Z[i] * Z[j] / d
    eigvals = np.sort(np.linalg.eigvalsh(CM))[::-1]
    # zero-pad to n_max dimensions
    fp = np.zeros(n_max)
    fp[:min(n, n_max)] = eigvals[:min(n, n_max)]
    return fp


def compute_fingerprints(frames: List, n_max: int = 20) -> np.ndarray:
    fps = np.array([coulomb_fingerprint(f, n_max) for f in frames])
    return fps


# ---------------------------------------------------------------------------
# Farthest-point sampling
# ---------------------------------------------------------------------------

def farthest_point_sampling(fps: np.ndarray, n_select: int, seed: int = 0) -> List[int]:
    """Select n_select indices that are maximally spread in descriptor space."""
    n = len(fps)
    if n_select >= n:
        return list(range(n))

    rng = np.random.default_rng(seed)
    selected = [int(rng.integers(n))]
    min_dists = np.full(n, np.inf)

    for _ in range(n_select - 1):
        last = fps[selected[-1]]
        dists = np.linalg.norm(fps - last, axis=1)
        min_dists = np.minimum(min_dists, dists)
        selected.append(int(np.argmax(min_dists)))

    return selected


# ---------------------------------------------------------------------------
# DeepMD numpy set writer
# ---------------------------------------------------------------------------

def write_deepmd_set(frames: List, output_dir: str, set_name: str = "set.000"):
    """
    Write ASE Atoms list to DeepMD-compatible numpy arrays:
      box.npy   shape (N, 9)
      coord.npy shape (N, 3*natoms)
      energy.npy shape (N,)
      force.npy  shape (N, 3*natoms)
    """
    set_dir = Path(output_dir) / set_name
    set_dir.mkdir(parents=True, exist_ok=True)

    boxes, coords, energies, forces = [], [], [], []

    for atoms in frames:
        cell = atoms.get_cell().array.flatten()
        pos = atoms.get_positions().flatten()
        boxes.append(cell)
        coords.append(pos)

        try:
            energies.append(atoms.get_potential_energy())
        except Exception:
            energies.append(0.0)

        try:
            f = atoms.get_forces().flatten()
        except Exception:
            f = np.zeros_like(pos)
        forces.append(f)

    np.save(set_dir / "box.npy",    np.array(boxes))
    np.save(set_dir / "coord.npy",  np.array(coords))
    np.save(set_dir / "energy.npy", np.array(energies))
    np.save(set_dir / "force.npy",  np.array(forces))
    print(f"Wrote {len(frames)} frames → {set_dir}")


# ---------------------------------------------------------------------------
# Uncertainty-based selector (committee disagreement)
# ---------------------------------------------------------------------------

def committee_disagreement(frames: List, model_paths: List[str]) -> np.ndarray:
    """
    Compute force-RMSD across a committee of DeepMD models.
    Returns per-frame disagreement score (higher = more uncertain).
    Requires deepmd-kit installed.
    """
    try:
        from deepmd.calculator import DP
    except ImportError:
        print("[warn] deepmd-kit not installed — returning uniform scores")
        return np.ones(len(frames))

    all_forces = []
    for mp in model_paths:
        calc = DP(model=mp)
        frame_forces = []
        for atoms in frames:
            atoms.calc = calc
            frame_forces.append(atoms.get_forces())
        all_forces.append(frame_forces)

    # shape: (n_models, n_frames, n_atoms, 3)
    all_forces = np.array(all_forces)
    mean_f = all_forces.mean(axis=0)
    disagreement = np.sqrt(((all_forces - mean_f) ** 2).mean(axis=(0, 2, 3)))
    return disagreement


# ---------------------------------------------------------------------------
# Main selection pipeline
# ---------------------------------------------------------------------------

def select_frames(
    traj_file: str,
    n_frames: int = 200,
    method: str = "fps",
    model_paths: List[str] = None,
    n_max_fp: int = 20,
    output_dir: str = "results/active_learning",
    seed: int = 42,
):
    print(f"Reading trajectory: {traj_file}")
    frames = ase_read(traj_file, index=":")
    print(f"  Total frames: {len(frames)}")

    if method == "fps":
        fps = compute_fingerprints(frames, n_max=n_max_fp)
        indices = farthest_point_sampling(fps, n_select=n_frames, seed=seed)
    elif method == "committee" and model_paths:
        scores = committee_disagreement(frames, model_paths)
        indices = np.argsort(scores)[-n_frames:].tolist()
    elif method == "random":
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(frames), size=min(n_frames, len(frames)),
                             replace=False).tolist()
    else:
        raise ValueError(f"Unknown method: {method}")

    selected = [frames[i] for i in sorted(indices)]
    print(f"  Selected {len(selected)} frames via '{method}'")

    write_deepmd_set(selected, output_dir=output_dir)
    idx_path = os.path.join(output_dir, "selected_indices.npy")
    np.save(idx_path, np.array(indices))
    print(f"  Index list saved → {idx_path}")
    return selected


def main():
    parser = argparse.ArgumentParser(
        description="Active-learning snapshot selector for DeepMD training set expansion"
    )
    parser.add_argument("--traj", required=True,
                        help="Trajectory file (XYZ, XDATCAR, LAMMPS dump)")
    parser.add_argument("--n_frames", type=int, default=200,
                        help="Number of diverse frames to select")
    parser.add_argument("--method", choices=["fps", "committee", "random"],
                        default="fps",
                        help="fps=farthest-point sampling, committee=model disagreement")
    parser.add_argument("--models", nargs="*", default=None,
                        help="DeepMD model paths for committee selection")
    parser.add_argument("--output_dir", default="results/active_learning")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    select_frames(
        traj_file=args.traj,
        n_frames=args.n_frames,
        method=args.method,
        model_paths=args.models,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
