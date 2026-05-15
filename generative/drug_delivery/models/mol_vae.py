#!/usr/bin/env python
"""
Molecular VAE for drug candidate generation.

Encodes molecular fingerprints (ECFP4) into a continuous latent space,
then decodes latent vectors back to molecular fingerprints from which
nearest-neighbour molecules are retrieved. Latent space can be
optimised with a property predictor (Gaussian process / NN) to steer
generation toward high-yield, low-toxicity drug candidates.

Application: ADC (Antibody-Drug Conjugate) linker and payload design
using the EDC/NHS coupling chemistry from prepare_dataset.py.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class MolEncoder(nn.Module):
    def __init__(self, fp_dim: int = 2048, hidden: int = 512, latent: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(fp_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden, latent)
        self.logvar_head = nn.Linear(hidden, latent)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        return self.mu_head(h), self.logvar_head(h)


class MolDecoder(nn.Module):
    def __init__(self, latent: int = 128, hidden: int = 512, fp_dim: int = 2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, fp_dim),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class PropertyPredictor(nn.Module):
    """Predicts (yield, logP, toxicity) from latent vector."""
    def __init__(self, latent: int = 128, n_props: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, n_props),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class MolVAE(nn.Module):
    def __init__(
        self,
        fp_dim: int = 2048,
        hidden: int = 512,
        latent: int = 128,
        n_props: int = 3,
        beta: float = 1.0,
    ):
        super().__init__()
        self.latent = latent
        self.beta = beta
        self.encoder = MolEncoder(fp_dim, hidden, latent)
        self.decoder = MolDecoder(latent, hidden, fp_dim)
        self.predictor = PropertyPredictor(latent, n_props)

    def reparametrize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        return mu

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encoder(x)
        z = self.reparametrize(mu, logvar)
        x_recon = self.decoder(z)
        props = self.predictor(z)
        return x_recon, mu, logvar, z, props

    def elbo_loss(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        props: torch.Tensor,
        prop_targets: Optional[torch.Tensor] = None,
        prop_weight: float = 1.0,
    ) -> torch.Tensor:
        recon = F.binary_cross_entropy(x_recon, x, reduction="sum") / x.size(0)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()
        loss = recon + self.beta * kl
        if prop_targets is not None:
            prop_loss = F.mse_loss(props, prop_targets)
            loss = loss + prop_weight * prop_loss
        return loss

    @torch.no_grad()
    def sample(self, n: int, device: str = "cpu") -> torch.Tensor:
        """Sample n latent vectors from N(0,I) and decode."""
        z = torch.randn(n, self.latent, device=device)
        return self.decoder(z), z

    @torch.no_grad()
    def optimise_latent(
        self,
        z_init: torch.Tensor,
        target_props: torch.Tensor,
        lr: float = 0.05,
        steps: int = 200,
    ) -> torch.Tensor:
        """Gradient-based latent optimisation toward target properties."""
        z = z_init.clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([z], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            pred = self.predictor(z)
            loss = F.mse_loss(pred, target_props.expand_as(pred))
            loss.backward()
            opt.step()
        return z.detach()


def smiles_to_fp(smiles: str, n_bits: int = 2048) -> np.ndarray:
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(n_bits)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
        return np.array(fp)
    except ImportError:
        return np.random.randint(0, 2, n_bits).astype(float)


def train_vae(
    smiles_list: List[str],
    prop_list: Optional[List[List[float]]] = None,
    fp_dim: int = 2048,
    latent: int = 128,
    hidden: int = 512,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    beta: float = 1.0,
    output_dir: str = "results/mol_vae",
    device: str = "cpu",
):
    os.makedirs(output_dir, exist_ok=True)

    fps = np.array([smiles_to_fp(s, fp_dim) for s in smiles_list], dtype=np.float32)
    X = torch.tensor(fps, device=device)

    if prop_list is not None:
        P = torch.tensor(prop_list, dtype=torch.float32, device=device)
        dataset = TensorDataset(X, P)
    else:
        dataset = TensorDataset(X)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    n_props = len(prop_list[0]) if prop_list else 3
    model = MolVAE(fp_dim=fp_dim, hidden=hidden, latent=latent,
                   n_props=n_props, beta=beta).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in loader:
            if len(batch) == 2:
                x_b, p_b = batch
            else:
                x_b = batch[0]
                p_b = None
            x_recon, mu, logvar, z, props = model(x_b)
            loss = model.elbo_loss(x_b, x_recon, mu, logvar, props,
                                   prop_targets=p_b)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch:4d}/{epochs}  loss={total_loss/len(loader):.4f}")

    ckpt = Path(output_dir) / "mol_vae.pt"
    torch.save(model.state_dict(), ckpt)
    print(f"Saved VAE → {ckpt}")
    return model
