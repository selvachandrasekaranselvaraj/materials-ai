import math
from typing import Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleGraphConv(nn.Module):
    """
    Very simple message-passing layer:
    - Node embedding from atomic number.
    - Edge messages from concatenated node states + edge distance.
    - Sum aggregation.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        # h: (N, H)
        # edge_index: (2, E)
        # edge_attr: (E, 1)
        src, dst = edge_index  # (E,), (E,)
        h_src = h[src]
        h_dst = h[dst]
        m_in = torch.cat([h_src, h_dst, edge_attr], dim=-1)  # (E, 2H+1)
        m = self.edge_mlp(m_in)  # (E, H)

        # aggregate messages to destination nodes
        N = h.shape[0]
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, m)
        return h + agg  # residual


class CathodeOracle(nn.Module):
    """
    Tiny oracle: graph conv + pooling + MLP to predict a scalar
    (e.g., energy_above_hull).
    """

    def __init__(self, hidden_dim: int = 128, num_layers: int = 3, max_Z: int = 100):
        super().__init__()
        self.max_Z = max_Z
        # simple embedding for atomic number
        self.emb = nn.Embedding(max_Z + 1, hidden_dim)

        self.layers = nn.ModuleList(
            [SimpleGraphConv(hidden_dim) for _ in range(num_layers)]
        )

        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
        )

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch_graphs: List[Dict[str, Any]]) -> torch.Tensor:
        """
        batch_graphs: list of dicts with keys:
          - z: (Ni,)
          - edge_index: (2, Ei)
          - edge_attr: (Ei, 1)
        Returns: (B, 1)
        """
        device = next(self.parameters()).device

        # build a big batch graph
        all_z = []
        all_edge_index = []
        all_edge_attr = []
        graph_ids = []

        node_offset = 0
        for g_idx, g in enumerate(batch_graphs):
            z = g["z"].to(device)
            edge_index = g["edge_index"].to(device)
            edge_attr = g["edge_attr"].to(device)

            N = z.shape[0]
            E = edge_index.shape[1]

            all_z.append(z)
            all_edge_attr.append(edge_attr)

            # shift node indices for this graph
            all_edge_index.append(edge_index + node_offset)
            graph_ids.extend([g_idx] * N)

            node_offset += N

        z_cat = torch.cat(all_z, dim=0)  # (total_N,)
        edge_index_cat = torch.cat(all_edge_index, dim=1)  # (2, total_E)
        edge_attr_cat = torch.cat(all_edge_attr, dim=0)  # (total_E, 1)
        graph_ids = torch.tensor(graph_ids, device=device, dtype=torch.long)  # (total_N,)

        # embed nodes
        z_clamped = torch.clamp(z_cat, 0, self.max_Z)
        h = self.emb(z_clamped)

        # message passing
        for layer, norm in zip(self.layers, self.norms):
            h = layer(h, edge_index_cat, edge_attr_cat)
            h = norm(h)
            h = F.relu(h)

        # pool: mean per graph
        B = len(batch_graphs)
        pooled = torch.zeros(B, h.shape[-1], device=device)
        counts = torch.zeros(B, device=device)

        pooled.index_add_(0, graph_ids, h)
        counts.index_add_(0, graph_ids, torch.ones_like(graph_ids, dtype=torch.float32))
        counts = counts.clamp(min=1.0)
        pooled = pooled / counts.unsqueeze(-1)

        # final MLP
        out = self.readout(pooled)  # (B, 1)
        return out

