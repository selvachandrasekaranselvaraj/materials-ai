#!/usr/bin/env python
"""
Multi-property GNN for crystal property prediction.

Architecture: stacked GCNConv layers with global mean pooling,
multi-task regression heads for 8 DFT-derived properties.

Properties predicted (from Materials Project / DFT):
  Ef   — formation energy per atom   (eV/atom)
  VBM  — valence band maximum        (eV)
  CBM  — conduction band minimum     (eV)
  Eg   — band gap                    (eV)
  EAH  — energy above hull           (eV/atom)
  FE   — Fermi energy                (eV)
  ρ    — density                     (g/cm³)
  is_S — is stable (EAH < 0.1 eV)   (binary)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool
    from torch_geometric.data import Data, Batch
    HAS_PyG = True
except ImportError:
    HAS_PyG = False


REGRESSION_TARGETS = ["Ef", "VBM", "CBM", "Eg", "EAH", "FE", "ρ"]
BINARY_TARGETS = ["is_S"]
ALL_TARGETS = REGRESSION_TARGETS + BINARY_TARGETS


class ResidualGCNBlock(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        if not HAS_PyG:
            raise ImportError("torch_geometric required for GNN")
        self.conv = GCNConv(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x, edge_index):
        return F.relu(self.norm(self.conv(x, edge_index) + x))


class CrystalGNN(nn.Module):
    """
    Multi-property crystal GNN.
    Input: crystal graph (node features = one-hot element encoding)
    Output: dict of per-property scalar predictions.
    """

    def __init__(
        self,
        node_feature_dim: int,
        hidden: int = 256,
        n_layers: int = 4,
        dropout: float = 0.1,
        targets: List[str] = None,
    ):
        super().__init__()
        if not HAS_PyG:
            raise ImportError("torch_geometric required")

        self.targets = targets or ALL_TARGETS
        self.input_proj = nn.Linear(node_feature_dim, hidden)
        self.layers = nn.ModuleList(
            [ResidualGCNBlock(hidden) for _ in range(n_layers)]
        )
        self.dropout = nn.Dropout(dropout)

        self.heads = nn.ModuleDict({
            t: nn.Sequential(
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
                nn.Linear(hidden // 2, 1),
            )
            for t in self.targets
        })

    def forward(self, data: "Data") -> Dict[str, torch.Tensor]:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.input_proj(x))
        for layer in self.layers:
            x = layer(x, edge_index)
        x = self.dropout(x)
        g = global_mean_pool(x, batch)         # (B, hidden)

        out = {}
        for t, head in self.heads.items():
            logit = head(g).squeeze(-1)
            if t in BINARY_TARGETS:
                out[t] = torch.sigmoid(logit)
            else:
                out[t] = logit
        return out

    def loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        task_weights: Dict[str, float] = None,
    ) -> torch.Tensor:
        if task_weights is None:
            task_weights = {t: 1.0 for t in self.targets}
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        for t in self.targets:
            if t not in targets or targets[t] is None:
                continue
            w = task_weights.get(t, 1.0)
            if t in BINARY_TARGETS:
                total = total + w * F.binary_cross_entropy(
                    predictions[t], targets[t].float()
                )
            else:
                total = total + w * F.mse_loss(predictions[t], targets[t])
        return total


# ---------------------------------------------------------------------------
# Fallback message-passing GNN (no PyG dependency)
# ---------------------------------------------------------------------------

class SimpleMessagePassingGNN(nn.Module):
    """
    Drop-in replacement for CrystalGNN when torch_geometric is not available.
    Uses manual scatter-add message passing.
    """

    def __init__(
        self,
        node_feature_dim: int,
        hidden: int = 256,
        n_layers: int = 4,
        targets: List[str] = None,
    ):
        super().__init__()
        self.targets = targets or ALL_TARGETS
        self.node_emb = nn.Linear(node_feature_dim, hidden)
        self.msg_nets = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU())
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])
        self.heads = nn.ModuleDict({
            t: nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1))
            for t in self.targets
        })

    def forward(self, x, edge_index, batch):
        h = F.relu(self.node_emb(x))
        src, dst = edge_index
        for msg_net, norm in zip(self.msg_nets, self.norms):
            m = msg_net(torch.cat([h[src], h[dst]], dim=-1))
            agg = torch.zeros_like(h)
            agg.index_add_(0, dst, m)
            h = norm(h + agg)

        # global mean pool
        n_graphs = int(batch.max().item()) + 1
        g = torch.zeros(n_graphs, h.shape[-1], device=h.device)
        count = torch.zeros(n_graphs, device=h.device)
        g.index_add_(0, batch, h)
        count.index_add_(0, batch, torch.ones(h.shape[0], device=h.device))
        g = g / count.unsqueeze(-1).clamp(min=1)

        out = {}
        for t, head in self.heads.items():
            logit = head(g).squeeze(-1)
            out[t] = torch.sigmoid(logit) if t in BINARY_TARGETS else logit
        return out
