#!/usr/bin/env python
"""
Train the multi-property crystal GNN on Materials Project data.

Usage:
    python gnn/train.py --data data/processed/structures.pkl \
                        --targets Ef EAH Eg --epochs 200 --device cuda
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

from gnn.model import CrystalGNN, SimpleMessagePassingGNN, ALL_TARGETS, HAS_PyG
from gnn.graphs import build_graph_dataset


def evaluate(model, loader, device: str) -> Dict[str, Dict[str, float]]:
    model.eval()
    all_preds: Dict[str, List] = {t: [] for t in model.targets}
    all_true: Dict[str, List] = {t: [] for t in model.targets}

    with torch.no_grad():
        for batch in loader:
            if HAS_PyG:
                batch = batch.to(device)
                preds = model(batch)
                for t in model.targets:
                    if hasattr(batch, t):
                        all_preds[t].extend(preds[t].cpu().tolist())
                        all_true[t].extend(getattr(batch, t).cpu().tolist())
            else:
                x, edge_index, batch_ids, targets = batch
                x = x.to(device)
                edge_index = edge_index.to(device)
                batch_ids = batch_ids.to(device)
                preds = model(x, edge_index, batch_ids)
                for t in model.targets:
                    if t in targets:
                        all_preds[t].extend(preds[t].cpu().tolist())
                        all_true[t].extend(targets[t].cpu().tolist())

    metrics = {}
    for t in model.targets:
        if not all_true[t]:
            continue
        y_true = np.array(all_true[t])
        y_pred = np.array(all_preds[t])
        metrics[t] = {
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "R2": float(r2_score(y_true, y_pred)),
        }
    return metrics


def train_gnn(
    data_path: str,
    targets: List[str] = None,
    node_feature_dim: int = 101,
    hidden: int = 256,
    n_layers: int = 4,
    dropout: float = 0.1,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 30,
    device: str = "cpu",
    output_dir: str = "results/gnn",
):
    os.makedirs(output_dir, exist_ok=True)
    targets = targets or ["Ef", "EAH", "Eg"]

    print(f"Loading dataset from {data_path}...")
    train_loader, val_loader = build_graph_dataset(
        data_path, targets=targets, batch_size=batch_size,
        test_size=0.2, device=device
    )

    if HAS_PyG:
        model = CrystalGNN(
            node_feature_dim=node_feature_dim,
            hidden=hidden, n_layers=n_layers,
            dropout=dropout, targets=targets,
        ).to(device)
    else:
        model = SimpleMessagePassingGNN(
            node_feature_dim=node_feature_dim,
            hidden=hidden, n_layers=n_layers,
            targets=targets,
        ).to(device)

    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            if HAS_PyG:
                batch = batch.to(device)
                preds = model(batch)
                tgt_dict = {t: getattr(batch, t) for t in targets
                            if hasattr(batch, t)}
            else:
                x, edge_index, batch_ids, tgt_dict = batch
                x, edge_index, batch_ids = (x.to(device), edge_index.to(device),
                                             batch_ids.to(device))
                tgt_dict = {k: v.to(device) for k, v in tgt_dict.items()}
                preds = model(x, edge_index, batch_ids)

            loss = model.loss(preds, tgt_dict)
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            train_loss += loss.item()

        scheduler.step()
        val_metrics = evaluate(model, val_loader, device)
        avg_val_mae = np.mean([v["MAE"] for v in val_metrics.values()])

        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_MAE": avg_val_mae, **{
                            f"{t}_MAE": val_metrics.get(t, {}).get("MAE", np.nan)
                            for t in targets
                        }})

        if epoch % 10 == 0:
            mae_str = "  ".join(
                f"{t}={val_metrics.get(t,{}).get('MAE', np.nan):.4f}"
                for t in targets
            )
            print(f"Epoch {epoch:4d}/{epochs}  loss={train_loss:.4f}  "
                  f"val_MAE={avg_val_mae:.4f}  [{mae_str}]")

        if avg_val_mae < best_val_loss:
            best_val_loss = avg_val_mae
            epochs_no_improve = 0
            torch.save(model.state_dict(), Path(output_dir) / "best_model.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    pd.DataFrame(history).to_csv(Path(output_dir) / "training_history.csv",
                                  index=False)
    print(f"\nBest model → {Path(output_dir)/'best_model.pt'}")
    print(f"Val MAE by property:")
    for t, m in val_metrics.items():
        print(f"  {t:8s}: MAE={m['MAE']:.4f}  R²={m['R2']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train multi-property crystal GNN")
    parser.add_argument("--data", required=True)
    parser.add_argument("--targets", nargs="+", default=["Ef", "EAH", "Eg"])
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--output_dir", default="results/gnn")
    args = parser.parse_args()

    train_gnn(
        data_path=args.data,
        targets=args.targets,
        hidden=args.hidden,
        n_layers=args.n_layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        device=args.device,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
