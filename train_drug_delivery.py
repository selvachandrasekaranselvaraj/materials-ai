#!/usr/bin/env python
"""
Train molecular VAE + ADMET predictor on ADC drug delivery molecules.
Data: generated from molecule_library.py (RDKit ECFP4 fingerprints)
"""
import sys, os
sys.path.insert(0, '/projects/nmclps/battery-materials-ai')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

DEVICE  = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT_DIR = '/projects/nmclps/battery-materials-ai/results/drug_delivery'
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Build training dataset from molecule library ───────────────────────────
print('Building drug delivery training dataset...')
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from generative.drug_delivery.data.molecule_library import build_library

lib_df = build_library(output_dir='/projects/nmclps/battery-materials-ai/data/drug_library')
mols = lib_df['smiles'].tolist()
print(f'Library size: {len(mols)} molecules')

PROP_NAMES = ['yield', 'logP', 'toxicity', 'log_solubility', 'bbb']
fps, props = [], []
for smi in mols:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        continue
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=256)
    fps.append(list(fp))
    mw   = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    rng  = np.random.default_rng(hash(smi) % (2**32))
    yield_  = float(np.clip(0.7 + rng.normal(0, 0.1), 0.1, 0.99))
    log_sol = float(np.clip(-2 + rng.normal(0, 1), -6, 2))
    toxicity= float(rng.random() > 0.7)
    bbb     = float(logp > 1 and mw < 500)
    props.append([yield_, logp, toxicity, log_sol, bbb])

X = torch.tensor(fps,  dtype=torch.float32)
Y = torch.tensor(props, dtype=torch.float32)
print(f'Dataset: {len(X)} molecules  fp_dim={X.shape[1]}  n_props={Y.shape[1]}')

n_val   = max(4, int(len(X) * 0.15))
n_train = len(X) - n_val
ds      = TensorDataset(X, Y)
tr_ds, val_ds = random_split(ds, [n_train, n_val],
                              generator=torch.Generator().manual_seed(42))
tr_loader  = DataLoader(tr_ds,  batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16)

def to_prop_dict(y_tensor, device):
    return {name: y_tensor[:, i].to(device) for i, name in enumerate(PROP_NAMES)}

# ── 2. Molecular β-VAE ────────────────────────────────────────────────────────
from generative.drug_delivery.models.mol_vae import MolVAE
LATENT = 32
model_vae = MolVAE(fp_dim=256, hidden=256, latent=LATENT, n_props=5, beta=1.0).to(DEVICE)
opt_vae = torch.optim.Adam(model_vae.parameters(), lr=1e-3)
print(f'\nVAE params: {sum(p.numel() for p in model_vae.parameters()):,}  device={DEVICE}')

EPOCHS_VAE = 150
best_val_vae, history_vae = float('inf'), []
print(f'Training VAE {EPOCHS_VAE} epochs...\n')

for epoch in range(1, EPOCHS_VAE + 1):
    model_vae.train()
    tr_loss = 0.0
    for xb, yb in tr_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt_vae.zero_grad()
        x_recon, mu, logvar, z, pred_props = model_vae(xb)
        loss = model_vae.elbo_loss(xb, x_recon, mu, logvar, pred_props,
                                   prop_targets=yb, prop_weight=1.0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_vae.parameters(), 1.0)
        opt_vae.step()
        tr_loss += loss.item()

    if epoch % 10 == 0 or epoch == 1:
        model_vae.eval()
        vl = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                x_recon, mu, logvar, z, pred_props = model_vae(xb)
                vl += model_vae.elbo_loss(xb, x_recon, mu, logvar, pred_props,
                                          prop_targets=yb, prop_weight=1.0).item()
        vl /= len(val_loader)
        history_vae.append({'epoch': epoch,
                             'train_loss': tr_loss/len(tr_loader),
                             'val_loss': vl})
        print(f'VAE Ep {epoch:>3}/{EPOCHS_VAE}  train={tr_loss/len(tr_loader):.4f}  val={vl:.4f}')
        if vl < best_val_vae:
            best_val_vae = vl
            torch.save(model_vae.state_dict(), f'{OUT_DIR}/mol_vae_best.pt')

pd.DataFrame(history_vae).to_csv(f'{OUT_DIR}/vae_history.csv', index=False)

# ── 3. ADMET predictor ────────────────────────────────────────────────────────
from generative.drug_delivery.models.property_predictor import ADMETPredictor
model_admet = ADMETPredictor(fp_dim=256, hidden=256, dropout=0.2).to(DEVICE)
opt_admet = torch.optim.Adam(model_admet.parameters(), lr=5e-4)
print(f'\nADMET params: {sum(p.numel() for p in model_admet.parameters()):,}')

EPOCHS_ADMET = 150
best_admet, history_admet = float('inf'), []
print(f'Training ADMET predictor {EPOCHS_ADMET} epochs...\n')

for epoch in range(1, EPOCHS_ADMET + 1):
    model_admet.train()
    tr_loss = 0.0
    for xb, yb in tr_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt_admet.zero_grad()
        preds = model_admet(xb)
        loss  = model_admet.loss(preds, to_prop_dict(yb, DEVICE))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_admet.parameters(), 1.0)
        opt_admet.step()
        tr_loss += loss.item()

    if epoch % 10 == 0 or epoch == 1:
        model_admet.eval()
        vl = 0.0
        all_p_d = {n: [] for n in PROP_NAMES}
        all_t_d = {n: [] for n in PROP_NAMES}
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                preds = model_admet(xb)
                tgts  = to_prop_dict(yb, DEVICE)
                vl   += model_admet.loss(preds, tgts).item()
                for i, n in enumerate(PROP_NAMES):
                    all_p_d[n].extend(preds[n].cpu().tolist())
                    all_t_d[n].extend(yb[:, i].cpu().tolist())
        vl /= len(val_loader)
        from sklearn.metrics import mean_absolute_error
        maes = {n: mean_absolute_error(all_t_d[n], all_p_d[n])
                for n in ['yield', 'logP', 'log_solubility']}
        history_admet.append({'epoch': epoch, 'train_loss': tr_loss/len(tr_loader),
                               'val_loss': vl, **maes})
        print(f'ADMET Ep {epoch:>3}/{EPOCHS_ADMET}  train={tr_loss/len(tr_loader):.4f}  '
              f'val={vl:.4f}  yield_MAE={maes["yield"]:.3f}  logP_MAE={maes["logP"]:.3f}')
        if vl < best_admet:
            best_admet = vl
            torch.save(model_admet.state_dict(), f'{OUT_DIR}/admet_best.pt')

pd.DataFrame(history_admet).to_csv(f'{OUT_DIR}/admet_history.csv', index=False)
print(f'\nVAE best val: {best_val_vae:.4f}  -> {OUT_DIR}/mol_vae_best.pt')
print(f'ADMET best val: {best_admet:.4f}  -> {OUT_DIR}/admet_best.pt')
