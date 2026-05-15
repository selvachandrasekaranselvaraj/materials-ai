from typing import Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (B,) in [0, 1] (float).
        """
        half_dim = self.dim // 2
        freqs = torch.exp(
            -torch.arange(half_dim, device=t.device, dtype=torch.float32)
            * (torch.log(torch.tensor(10000.0)) / (half_dim - 1))
        )
        args = t.unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb  # (B, dim)


class GraphDiffusionDenoiser(nn.Module):
    """
    Simple graph denoiser for diffusion on node embeddings.
    - Node input: noisy node embeddings (B, N, H)
    - Edge_index / edge_attr reused from oracle graphs
    - Conditioned on time embedding.
    """

    def __init__(self, hidden_dim: int = 128, edge_dim: int = 1, time_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.time_emb = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.ReLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim + time_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.norm_node = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h_noisy: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        t: torch.Tensor,
        batch_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        h_noisy: (total_N, H)
        edge_index: (2, E)
        edge_attr: (E, edge_dim)
        t: (B,) float in [0, 1]
        batch_ids: (total_N,) graph index for each node

        Returns predicted noise on h_noisy: (total_N, H)
        """
        device = h_noisy.device
        H = h_noisy.shape[-1]

        # time embedding per graph, then expand per node
        t_emb_graph = self.time_mlp(self.time_emb(t))  # (B, time_dim)
        t_emb = t_emb_graph[batch_ids]                 # (total_N, time_dim)

        src, dst = edge_index  # (E,), (E,)
        h_src = h_noisy[src]   # (E, H)
        h_dst = h_noisy[dst]   # (E, H)

        # gather time embedding per edge via destination node
        t_edge = t_emb[dst]    # (E, time_dim)

        m_in = torch.cat([h_src, h_dst, edge_attr, t_edge], dim=-1)  # (E, 2H+edge_dim+time_dim)
        m = self.edge_mlp(m_in)                                     # (E, H)

        # aggregate to destination nodes
        agg = torch.zeros_like(h_noisy)
        agg.index_add_(0, dst, m)

        h = h_noisy + self.node_mlp(agg)
        h = self.norm_node(h)
        return h  # predicted noise (same shape as h_noisy)

