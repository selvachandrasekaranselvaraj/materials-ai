#!/usr/bin/env python
"""
Generate deepmd_input.json for DeepMD-kit GPU training.

Supports DeePot-SE (default), DPA-1, and DPA-2 descriptors.
Automatically sets type_map, training data paths, and GPU settings.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List


def make_deepmd_input(
    type_map: List[str],
    train_dirs: List[str],
    val_dirs: List[str],
    descriptor: str = "se_e2_a",
    n_neuron: List[int] = None,
    rcut: float = 6.0,
    rcut_smth: float = 5.5,
    sel: List[int] = None,
    n_steps: int = 1_000_000,
    lr_start: float = 0.001,
    lr_stop: float = 1e-8,
    batch_size: int = 32,
    seed: int = 10,
    output_path: str = "deepmd_input.json",
):
    if n_neuron is None:
        n_neuron = [25, 50, 100]
    if sel is None:
        sel = [120] * len(type_map)

    config = {
        "model": {
            "type_map": type_map,
            "descriptor": {
                "type": descriptor,
                "sel": sel,
                "rcut_smth": rcut_smth,
                "rcut": rcut,
                "neuron": n_neuron,
                "resnet_dt": False,
                "axis_neuron": 16,
                "seed": seed,
            },
            "fitting_net": {
                "neuron": [240, 240, 240],
                "resnet_dt": True,
                "seed": seed,
            },
        },
        "learning_rate": {
            "type": "exp",
            "decay_steps": 5000,
            "start_lr": lr_start,
            "stop_lr": lr_stop,
        },
        "loss": {
            "type": "ener",
            "start_pref_e": 0.02,
            "limit_pref_e": 1,
            "start_pref_f": 1000,
            "limit_pref_f": 1,
            "start_pref_v": 0,
            "limit_pref_v": 0,
        },
        "training": {
            "training_data": {
                "systems": train_dirs,
                "batch_size": batch_size,
                "auto_prob": "prob_sys_size",
            },
            "validation_data": {
                "systems": val_dirs,
                "batch_size": batch_size,
                "numb_btch": 3,
            },
            "numb_steps": n_steps,
            "seed": seed,
            "disp_file": "lcurve.out",
            "disp_freq": 1000,
            "numb_test": 10,
            "save_freq": 50000,
            "save_ckpt": "model.ckpt",
            "disp_training": True,
            "time_training": True,
            "profiling": False,
            "profiling_file": "timeline.json",
        },
    }

    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"DeepMD input written → {output_path}")
    print(f"  type_map  : {type_map}")
    print(f"  descriptor: {descriptor}")
    print(f"  rcut      : {rcut} Å")
    print(f"  n_steps   : {n_steps:,}")
    print(f"  train sets: {len(train_dirs)}")
    return config


def auto_discover_training_sets(base_dir: str) -> List[str]:
    """Find all DeepMD set directories (contain box.npy, coord.npy, …)."""
    sets = []
    for root, dirs, files in os.walk(base_dir):
        if "box.npy" in files and "coord.npy" in files:
            sets.append(root)
    return sorted(sets)


def main():
    parser = argparse.ArgumentParser(
        description="Generate DeepMD-kit GPU training input JSON"
    )
    parser.add_argument("--type_map", nargs="+", required=True,
                        help="Element list, e.g. Li Ti Cl")
    parser.add_argument("--train_dir", nargs="+", default=None,
                        help="Training set directories (auto-discovered if omitted)")
    parser.add_argument("--val_dir", nargs="+", default=None,
                        help="Validation set directories")
    parser.add_argument("--data_root", default=".",
                        help="Root directory for auto-discovery")
    parser.add_argument("--descriptor",
                        choices=["se_e2_a", "se_e2_r", "dpa1", "dpa2"],
                        default="se_e2_a")
    parser.add_argument("--rcut", type=float, default=6.0)
    parser.add_argument("--n_steps", type=int, default=1_000_000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output", default="deepmd_input.json")
    args = parser.parse_args()

    if args.train_dir is None:
        all_sets = auto_discover_training_sets(args.data_root)
        n_val = max(1, len(all_sets) // 10)
        train_dirs = all_sets[n_val:]
        val_dirs = all_sets[:n_val]
        print(f"Auto-discovered: {len(all_sets)} sets → "
              f"{len(train_dirs)} train, {len(val_dirs)} val")
    else:
        train_dirs = args.train_dir
        val_dirs = args.val_dir or train_dirs[:1]

    make_deepmd_input(
        type_map=args.type_map,
        train_dirs=train_dirs,
        val_dirs=val_dirs,
        descriptor=args.descriptor,
        rcut=args.rcut,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
