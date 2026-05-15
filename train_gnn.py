#!/usr/bin/env python
"""
Train multi-property GNN on 5,000 real Materials Project cathode structures.
Data: cathodes_raw.parquet — CIF strings + Ef, EAH, band_gap, volume
Targets: Ef (formation energy/atom), EAH (energy above hull), Eg (band gap)
"""
import sys, os, pickle, time
sys.path.insert(0, '/projects/nmclps/battery-materials-ai')

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split

from gnn.graphs import structure_to_graph, build_graph_dataset
from gnn.model import CrystalGNN
from pymatgen.core import Structure

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PARQUET = '/projects/nmclps/GM/cathode_gen/data/raw/cathodes_raw.parquet'
OUT_DIR = '/projects/nmclps/battery-materials-ai/results/gnn'
CACHE   = '/projects/nmclps/battery-materials-ai/data/processed/mp_cathode_graphs.pkl'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(CACHE), exist_ok=True)

# ── 1. Build graph dataset ────────────────────────────────────────────────────
if os.path.exists(CACHE):
    print(f'Loading cached graphs from {CACHE}')
    with open(CACHE, 'rb') as f:
        graphs = pickle.load(f)
else:
    print('Building graphs from parquet (5000 structures)...')
    df = pd.read_parquet(PARQUET)
    graphs = []
    t0 = time.time()
    for i, row in df.iterrows():
        try:
            s = Structure.from_str(row['cif'], fmt='cif')
            tgt = {
                'Ef':      float(row['formation_energy_per_atom']),
                'EAH':     float(row['energy_above_hull']),
                'Eg':      float(row['band_gap']),
                'density': float(s.density),
            }
            g = structure_to_graph(s, tgt, cutoff=5.0)
            if g is not None:
                graphs.append(g)
        except Exception:
            pass
        if (i + 1) % 500 == 0:
            print(f'  {i+1}/5000  ({len(graphs)} ok)  {time.time()-t0:.0f}s')
    print(f'Built {len(graphs)} graphs in {time.time()-t0:.0f}s')
    with open(CACHE, 'wb') as f:
        pickle.dump(graphs, f)
    print(f'Cached -> {CACHE}')

print(f'Dataset: {len(graphs)} graphs  device={DEVICE}')
print(f'Node features: {graphs[0].x.shape[1]}-dim')

# ── 2. Dataloaders ────────────────────────────────────────────────────────────
TARGETS = ['Ef', 'EAH', 'Eg', 'density']
n_val   = max(1, int(len(graphs) * 0.15))
n_train = len(graphs) - n_val
train_ds, val_ds = random_split(graphs, [n_train, n_val],
                                generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False)

# ── 3. Model ──────────────────────────────────────────────────────────────────
node_dim = graphs[0].x.shape[1]
model = CrystalGNN(node_feature_dim=node_dim, hidden=256, n_layers=4,
                   dropout=0.1, targets=TARGETS).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f'Model params: {n_params:,}')

opt   = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)

# ── 4. Training loop ──────────────────────────────────────────────────────────
from sklearn.metrics import r2_score, mean_absolute_error
BINARY = {'is_S'}

def loss_fn(preds, batch):
    total = 0.0
    for t in TARGETS:
        y = getattr(batch, t, None)
        if y is None: continue
        total += torch.nn.functional.mse_loss(preds[t], y.float().to(DEVICE))
    return total

best_val, history = float('inf'), []
EPOCHS = 200
print(f'\nTraining {EPOCHS} epochs on {n_train} / {n_val} graphs...\n')

for epoch in range(1, EPOCHS + 1):
    model.train()
    tr_loss = 0.0
    for batch in train_loader:
        batch = batch.to(DEVICE)
        opt.zero_grad()
        preds = model(batch)
        loss  = loss_fn(preds, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tr_loss += loss.item()
    sched.step()

    if epoch % 10 == 0 or epoch == 1:
        model.eval()
        all_p, all_t = {t: [] for t in TARGETS}, {t: [] for t in TARGETS}
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                preds = model(batch)
                for t in TARGETS:
                    y = getattr(batch, t, None)
                    if y is not None:
                        all_p[t].extend(preds[t].cpu().tolist())
                        all_t[t].extend(y.cpu().tolist())

        mae = {t: mean_absolute_error(all_t[t], all_p[t]) for t in TARGETS if all_t[t]}
        val_mae = np.mean(list(mae.values()))
        row = {'epoch': epoch, 'train_loss': tr_loss/len(train_loader),
               'val_MAE': val_mae, **{f'{t}_MAE': mae.get(t, 0) for t in TARGETS}}
        history.append(row)

        msg = f'Ep {epoch:>3}/{EPOCHS}  loss={row["train_loss"]:.4f}  val_MAE={val_mae:.4f}'
        msg += '  [' + '  '.join(f'{t}={mae.get(t,0):.4f}' for t in TARGETS) + ']'
        print(msg)

        if val_mae < best_val:
            best_val = val_mae
            torch.save(model.state_dict(), f'{OUT_DIR}/best_model.pt')

pd.DataFrame(history).to_csv(f'{OUT_DIR}/training_history.csv', index=False)
print(f'\nBest val MAE: {best_val:.4f}')
print(f'Model -> {OUT_DIR}/best_model.pt')
print(f'History -> {OUT_DIR}/training_history.csv')
