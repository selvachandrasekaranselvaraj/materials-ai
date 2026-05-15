#!/usr/bin/env python
"""
Train graph diffusion model (DDPM) on 5,000 real MP cathode structures.
Uses the real GraphDiffusionDenoiser from GM/cathode_gen with a node projection layer.
"""
import sys, os, pickle, time
sys.path.insert(0, '/projects/nmclps/battery-materials-ai')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from pymatgen.core import Structure

DEVICE  = 'cuda' if torch.cuda.is_available() else 'cpu'
PARQUET = '/projects/nmclps/GM/cathode_gen/data/raw/cathodes_raw.parquet'
CACHE   = '/projects/nmclps/battery-materials-ai/data/processed/mp_cathode_graphs.pkl'
OUT_DIR = '/projects/nmclps/battery-materials-ai/results/generative'
os.makedirs(OUT_DIR, exist_ok=True)

HIDDEN   = 128
T_STEPS  = 1000

# ── Load graph data ───────────────────────────────────────────────────────────
print('Waiting for GNN cache (built by train_gnn.py)...')
while not os.path.exists(CACHE):
    time.sleep(10)
print(f'Loading {CACHE}')
with open(CACHE, 'rb') as f:
    graphs = pickle.load(f)
print(f'Loaded {len(graphs)} graphs  device={DEVICE}')

NODE_DIM = graphs[0].x.shape[1]   # 101
EDGE_DIM = 1                       # distance

# ── DDPM noise schedule ───────────────────────────────────────────────────────
betas     = torch.linspace(1e-4, 0.02, T_STEPS, device=DEVICE)
alphas    = 1.0 - betas
alpha_bar = torch.cumprod(alphas, dim=0)

def q_sample(h, t_node):
    """Add noise: h_t = sqrt(ab)*h0 + sqrt(1-ab)*eps"""
    noise = torch.randn_like(h)
    ab    = alpha_bar[t_node].unsqueeze(-1)   # (N, 1)
    return ab.sqrt() * h + (1 - ab).sqrt() * noise, noise

# ── Model with node projection ────────────────────────────────────────────────
from generative.energy_materials.models.graph_diffusion import GraphDiffusionDenoiser

class CathodeDenoiser(nn.Module):
    def __init__(self, node_dim, hidden, edge_dim, T):
        super().__init__()
        self.proj   = nn.Linear(node_dim, hidden)
        self.denoiser = GraphDiffusionDenoiser(hidden_dim=hidden,
                                               edge_dim=edge_dim,
                                               time_dim=64)
        self.out_proj = nn.Linear(hidden, node_dim)
        self.T = T

    def forward(self, x_noisy, edge_index, edge_attr, t_node, batch):
        h   = self.proj(x_noisy)
        t_b = t_node[torch.unique_consecutive(batch, return_inverse=False)[1]
                     ].float() / self.T if False else \
              torch.zeros(batch.max()+1, device=x_noisy.device)
        # use per-batch mean timestep
        from torch_scatter import scatter_mean
        t_batch = scatter_mean(t_node.float(), batch, dim=0) / self.T
        h_dn  = self.denoiser(h, edge_index, edge_attr, t_batch, batch)
        return self.out_proj(h_dn)

model = CathodeDenoiser(NODE_DIM, HIDDEN, EDGE_DIM, T_STEPS).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f'CathodeDenoiser params: {n_params:,}')

n_val   = max(1, int(len(graphs) * 0.1))
n_train = len(graphs) - n_val
train_ds, val_ds = random_split(graphs, [n_train, n_val],
                                generator=torch.Generator().manual_seed(0))
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=32)

opt   = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)

best_val, history = float('inf'), []
EPOCHS = 100
print(f'\nTraining DDPM {EPOCHS} epochs on {n_train} graphs...\n')

for epoch in range(1, EPOCHS + 1):
    model.train()
    tr_loss = 0.0
    for batch in train_loader:
        batch = batch.to(DEVICE)
        x0    = batch.x.float()
        t_g   = torch.randint(0, T_STEPS, (batch.num_graphs,), device=DEVICE)
        t_n   = t_g[batch.batch]
        xt, eps = q_sample(x0, t_n)

        opt.zero_grad()
        eps_pred = model(xt, batch.edge_index, batch.edge_attr, t_n, batch.batch)
        loss = nn.functional.mse_loss(eps_pred, eps)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tr_loss += loss.item()
    sched.step()

    if epoch % 10 == 0 or epoch == 1:
        model.eval()
        vl = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                x0    = batch.x.float()
                t_g   = torch.randint(0, T_STEPS, (batch.num_graphs,), device=DEVICE)
                t_n   = t_g[batch.batch]
                xt, eps = q_sample(x0, t_n)
                eps_p = model(xt, batch.edge_index, batch.edge_attr, t_n, batch.batch)
                vl   += nn.functional.mse_loss(eps_p, eps).item()
        vl /= len(val_loader)
        history.append({'epoch': epoch, 'train_loss': tr_loss/len(train_loader),
                        'val_loss': vl})
        print(f'Ep {epoch:>3}/{EPOCHS}  train={tr_loss/len(train_loader):.4f}  val={vl:.4f}')
        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), f'{OUT_DIR}/diffusion_best.pt')

pd.DataFrame(history).to_csv(f'{OUT_DIR}/diffusion_history.csv', index=False)
print(f'\nBest val loss: {best_val:.4f}  -> {OUT_DIR}/diffusion_best.pt')
