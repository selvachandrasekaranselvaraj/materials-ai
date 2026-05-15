#!/usr/bin/env python
"""
Crystal → graph conversion for the multi-property GNN.

Node features (92-dim one-hot element encoding + 7 elemental properties):
  atomic number, period, group, electronegativity, atomic radius,
  ionisation energy, electron affinity

Edge features:
  interatomic distance (Å)

Handles pymatgen Structure objects and raw CIF strings.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import random_split

try:
    from torch_geometric.data import Data, DataLoader as PyGLoader
    from torch_geometric.loader import DataLoader as PyGLoader2
    HAS_PyG = True
except ImportError:
    from torch.utils.data import DataLoader
    HAS_PyG = False

try:
    from pymatgen.core import Structure, Element
    HAS_PYMATGEN = True
except ImportError:
    HAS_PYMATGEN = False


# Elemental feature lookup (Z=1..94)
ELEMENT_FEATURES: Dict[int, List[float]] = {}

_PERIOD_MAP = {**{z: 1 for z in range(1, 3)},
               **{z: 2 for z in range(3, 11)},
               **{z: 3 for z in range(11, 19)},
               **{z: 4 for z in range(19, 37)},
               **{z: 5 for z in range(37, 55)},
               **{z: 6 for z in range(55, 87)},
               **{z: 7 for z in range(87, 119)}}


def _build_element_features():
    if not HAS_PYMATGEN:
        return
    for z in range(1, 95):
        try:
            el = Element.from_Z(z)
            ELEMENT_FEATURES[z] = [
                z / 94.0,
                _PERIOD_MAP.get(z, 7) / 7.0,
                (el.group or 18) / 18.0,
                (el.X or 2.0) / 4.0,
                (el.atomic_radius or 1.5) / 3.0,
                (el.ionization_energy or 10.0) / 25.0,
                (el.electron_affinity or 0.0) / 4.0,
            ]
        except Exception:
            ELEMENT_FEATURES[z] = [z / 94.0] + [0.5] * 6


_build_element_features()


def z_to_feature(z: int, n_elements: int = 94) -> np.ndarray:
    """One-hot (94-dim) + 7 elemental properties = 101-dim node feature."""
    one_hot = np.zeros(n_elements, dtype=np.float32)
    if 1 <= z <= n_elements:
        one_hot[z - 1] = 1.0
    props = np.array(ELEMENT_FEATURES.get(z, [z / 94.0] + [0.5] * 6),
                     dtype=np.float32)
    return np.concatenate([one_hot, props])


NODE_FEATURE_DIM = 94 + 7   # = 101


def structure_to_graph(
    structure,
    target_dict: Dict[str, float],
    cutoff: float = 5.0,
) -> Optional["Data"]:
    """Convert a pymatgen Structure to a PyG Data object."""
    if not HAS_PYMATGEN:
        return None

    sites = structure.sites
    n = len(sites)
    node_features = np.array(
        [z_to_feature(site.specie.Z) for site in sites], dtype=np.float32
    )

    senders, receivers, dists = [], [], []
    for i, site in enumerate(sites):
        for nn in structure.get_neighbors(site, cutoff):
            senders.append(i)
            receivers.append(nn.index)
            dists.append(float(nn.nn_distance))

    if not senders:
        senders, receivers, dists = [0], [0], [0.0]

    edge_index = torch.tensor([senders, receivers], dtype=torch.long)
    edge_attr = torch.tensor(dists, dtype=torch.float32).unsqueeze(-1)
    x = torch.tensor(node_features, dtype=torch.float32)

    if HAS_PyG:
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        for k, v in target_dict.items():
            setattr(data, k, torch.tensor(float(v), dtype=torch.float32))
        return data
    else:
        return {"x": x, "edge_index": edge_index, "edge_attr": edge_attr,
                "targets": {k: torch.tensor(float(v)) for k, v in target_dict.items()}}


def build_graph_dataset(
    data_path: str,
    targets: List[str],
    batch_size: int = 64,
    test_size: float = 0.2,
    cutoff: float = 5.0,
    device: str = "cpu",
):
    """
    Load pickled dataset and build train/val dataloaders.

    data_path: pickle of List[dict] where each dict has:
        'structure': pymatgen Structure  or 'cif': str
        'Ef', 'EAH', 'Eg', 'VBM', 'CBM', 'FE', 'density': float
    """
    with open(data_path, "rb") as f:
        records = pickle.load(f)

    graphs = []
    for rec in records:
        if "structure" in rec:
            s = rec["structure"]
        elif "cif" in rec and HAS_PYMATGEN:
            s = Structure.from_str(rec["cif"], fmt="cif")
        else:
            continue
        tgt = {t: rec[t] for t in targets if t in rec}
        g = structure_to_graph(s, tgt, cutoff=cutoff)
        if g is not None:
            graphs.append(g)

    print(f"Built {len(graphs)} crystal graphs  (targets: {targets})")

    n_val = int(len(graphs) * test_size)
    n_train = len(graphs) - n_val

    if HAS_PyG:
        from torch_geometric.loader import DataLoader as PLoader
        train_ds, val_ds = random_split(graphs, [n_train, n_val])
        train_loader = PLoader(list(train_ds), batch_size=batch_size, shuffle=True)
        val_loader = PLoader(list(val_ds), batch_size=batch_size)
    else:
        from torch.utils.data import DataLoader
        train_ds, val_ds = random_split(graphs, [n_train, n_val])

        def collate(batch):
            # naive: process graph by graph (no batching for fallback)
            return batch

        train_loader = DataLoader(list(train_ds), batch_size=1,
                                  collate_fn=collate, shuffle=True)
        val_loader = DataLoader(list(val_ds), batch_size=1, collate_fn=collate)

    return train_loader, val_loader
