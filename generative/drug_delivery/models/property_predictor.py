#!/usr/bin/env python
"""
ADMET and reaction-yield property predictor for drug delivery candidates.

Predicts the following properties from molecular fingerprints (ECFP4):
  - Reaction yield  (0–1, EDC/NHS coupling)
  - Predicted logP  (lipophilicity)
  - Predicted toxicity score  (0=safe, 1=toxic; Tox21 proxy)
  - Aqueous solubility  (log mol/L)
  - Blood-brain barrier penetration  (binary)

Used as the oracle scorer in the generative drug delivery pipeline,
analogous to the ALCHEMI BCS NIM oracle and the CathodeOracle in the
energy materials branch.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


PROPERTY_NAMES = ["yield", "logP", "toxicity", "log_solubility", "bbb"]
N_PROPS = len(PROPERTY_NAMES)


class FingerprintBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ADMETPredictor(nn.Module):
    """
    Multi-task ADMET predictor.
    Shared backbone → per-property heads.
    Handles both regression (yield, logP, solubility) and
    classification (toxicity, BBB) heads.
    """

    REGRESSION_PROPS = ["yield", "logP", "log_solubility"]
    BINARY_PROPS = ["toxicity", "bbb"]

    def __init__(self, fp_dim: int = 2048, hidden: int = 512, dropout: float = 0.2):
        super().__init__()
        self.backbone = nn.Sequential(
            FingerprintBlock(fp_dim, hidden, dropout),
            FingerprintBlock(hidden, hidden, dropout),
            FingerprintBlock(hidden, 256, dropout),
        )
        self.heads = nn.ModuleDict({
            name: nn.Linear(256, 1) for name in PROPERTY_NAMES
        })

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.backbone(x)
        out = {}
        for name, head in self.heads.items():
            logit = head(h).squeeze(-1)
            if name in self.BINARY_PROPS:
                out[name] = torch.sigmoid(logit)
            else:
                out[name] = logit
        return out

    def loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        for name in PROPERTY_NAMES:
            if name not in targets:
                continue
            pred = predictions[name]
            tgt = targets[name]
            if name in self.BINARY_PROPS:
                total = total + F.binary_cross_entropy(pred, tgt)
            else:
                total = total + F.mse_loss(pred, tgt)
        return total

    @torch.no_grad()
    def score(self, fps: torch.Tensor) -> Dict[str, np.ndarray]:
        """Return numpy arrays of predictions for each property."""
        self.eval()
        preds = self(fps)
        return {k: v.cpu().numpy() for k, v in preds.items()}

    @torch.no_grad()
    def composite_score(
        self,
        fps: torch.Tensor,
        weights: Dict[str, float] = None,
    ) -> np.ndarray:
        """
        Single scalar score combining all properties.
        Higher is better (yield↑, logP moderate, toxicity↓, solubility↑, BBB flexible).
        """
        if weights is None:
            weights = {
                "yield": 1.0,
                "logP": -0.1,
                "toxicity": -2.0,
                "log_solubility": 0.3,
                "bbb": 0.0,
            }
        preds = self.score(fps)
        score = np.zeros(fps.shape[0])
        for name, w in weights.items():
            score += w * preds[name]
        return score


def smiles_to_fp(smiles: str, n_bits: int = 2048) -> np.ndarray:
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(n_bits, dtype=np.float32)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
        return np.array(fp, dtype=np.float32)
    except ImportError:
        return np.random.randint(0, 2, n_bits).astype(np.float32)


def train_predictor(
    smiles_list: List[str],
    property_dict: Dict[str, List[float]],
    fp_dim: int = 2048,
    hidden: int = 512,
    epochs: int = 200,
    batch_size: int = 32,
    lr: float = 1e-3,
    dropout: float = 0.2,
    output_dir: str = "results/admet_predictor",
    device: str = "cpu",
) -> ADMETPredictor:
    os.makedirs(output_dir, exist_ok=True)

    fps = np.array([smiles_to_fp(s, fp_dim) for s in smiles_list], dtype=np.float32)
    X = torch.tensor(fps, device=device)
    targets = {k: torch.tensor(v, dtype=torch.float32, device=device)
               for k, v in property_dict.items()}

    model = ADMETPredictor(fp_dim=fp_dim, hidden=hidden, dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    indices = list(range(len(smiles_list)))
    for epoch in range(1, epochs + 1):
        model.train()
        np.random.shuffle(indices)
        total_loss = 0.0
        for start in range(0, len(indices), batch_size):
            idx = indices[start:start + batch_size]
            x_b = X[idx]
            t_b = {k: v[idx] for k, v in targets.items()}
            preds = model(x_b)
            loss = model.loss(preds, t_b)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        scheduler.step()
        if epoch % 20 == 0:
            print(f"Epoch {epoch:4d}/{epochs}  loss={total_loss:.4f}")

    ckpt = Path(output_dir) / "admet_predictor.pt"
    torch.save(model.state_dict(), ckpt)
    print(f"Saved ADMET predictor → {ckpt}")
    return model
