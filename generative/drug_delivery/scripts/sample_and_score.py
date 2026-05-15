#!/usr/bin/env python
"""
Drug delivery — sample from the molecular VAE and score with ADMET predictor.

Pipeline:
  1. Load trained MolVAE + ADMETPredictor checkpoints
  2. Sample N latent vectors from N(0,I)
  3. Decode to fingerprints → nearest-neighbour SMILES lookup
  4. Score candidates with ADMET oracle
  5. Optionally run BCS NIM conformer search on top-ranked candidates
  6. Save ranked hit list as CSV
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

from generative.drug_delivery.models.mol_vae import MolVAE, smiles_to_fp
from generative.drug_delivery.models.property_predictor import (
    ADMETPredictor, PROPERTY_NAMES, smiles_to_fp as admet_fp,
)


def load_reference_library(library_path: str) -> List[str]:
    """Load a SMILES library for nearest-neighbour decoding."""
    with open(library_path) as f:
        return [line.strip() for line in f if line.strip()]


def fps_to_nearest_smiles(
    decoded_fps: np.ndarray,
    library_fps: np.ndarray,
    library_smiles: List[str],
    top_k: int = 1,
) -> List[str]:
    """Retrieve nearest-neighbour SMILES via Tanimoto similarity."""
    hits = []
    for fp in decoded_fps:
        # Tanimoto: |A∩B| / |A∪B| for binary fps
        inter = (fp * library_fps).sum(axis=1)
        union = (fp + library_fps - fp * library_fps).sum(axis=1)
        sim = np.where(union > 0, inter / union, 0.0)
        best = sim.argsort()[-top_k:][::-1]
        hits.append(library_smiles[best[0]])
    return hits


def run_bcs_nim_on_hits(smiles_list: List[str], model: str, top_n: int = 5):
    """Optionally run BCS NIM conformer search on the top-ranked candidates."""
    try:
        from alchemi.bcs_nim.conformer_search import batch_conformer_search
        from alchemi.bcs_nim.filter_structures import filter_conformers
    except ImportError:
        print("[warn] alchemi.bcs_nim not importable — skipping conformer search")
        return

    for i, smi in enumerate(smiles_list[:top_n]):
        out_dir = f"results/drug_hits/candidate_{i+1}"
        print(f"\nBCS NIM conformer search for candidate {i+1}: {smi}")
        results = batch_conformer_search(
            smiles=smi,
            n_confs=10,
            model=model,
            fmax=0.005,
            output_dir=out_dir,
        )
        filter_conformers(input_dir=out_dir,
                          output_dir=out_dir + "_filtered")


def sample_and_score(
    vae_ckpt: str,
    admet_ckpt: str,
    library_smiles: List[str],
    n_samples: int = 1000,
    fp_dim: int = 2048,
    latent: int = 128,
    n_props: int = 5,
    bcs_model: str = "aimnet2",
    run_bcs: bool = False,
    output_dir: str = "results/drug_hits",
    device: str = "cpu",
) -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)

    # Load models
    vae = MolVAE(fp_dim=fp_dim, latent=latent, n_props=3).to(device)
    vae.load_state_dict(torch.load(vae_ckpt, map_location=device))
    vae.eval()

    admet = ADMETPredictor(fp_dim=fp_dim).to(device)
    admet.load_state_dict(torch.load(admet_ckpt, map_location=device))
    admet.eval()

    # Precompute library fingerprints
    print(f"Computing fingerprints for {len(library_smiles)} library molecules...")
    lib_fps = np.array([smiles_to_fp(s, fp_dim) for s in library_smiles],
                       dtype=np.float32)

    # Sample from VAE
    print(f"Sampling {n_samples} candidates from latent space...")
    decoded_fps, z_samples = vae.sample(n_samples, device=device)
    decoded_np = decoded_fps.cpu().numpy()

    # Nearest-neighbour SMILES retrieval
    hit_smiles = fps_to_nearest_smiles(decoded_np, lib_fps, library_smiles)

    # ADMET scoring
    hit_fps_t = torch.tensor(
        np.array([admet_fp(s, fp_dim) for s in hit_smiles], dtype=np.float32),
        device=device
    )
    scores = admet.score(hit_fps_t)
    composite = admet.composite_score(hit_fps_t)

    rows = []
    for i, smi in enumerate(hit_smiles):
        row = {"rank": i + 1, "smiles": smi, "composite_score": composite[i]}
        for prop in PROPERTY_NAMES:
            row[prop] = float(scores[prop][i])
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    csv_path = Path(output_dir) / "drug_candidates.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nTop 10 drug candidates:")
    print(df[["rank", "smiles", "composite_score", "yield",
              "toxicity"]].head(10).to_string(index=False))
    print(f"\nFull results → {csv_path}")

    if run_bcs:
        top_smiles = df["smiles"].head(5).tolist()
        run_bcs_nim_on_hits(top_smiles, model=bcs_model)

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Sample from MolVAE + score with ADMET oracle"
    )
    parser.add_argument("--vae_ckpt", required=True, help="Path to mol_vae.pt")
    parser.add_argument("--admet_ckpt", required=True,
                        help="Path to admet_predictor.pt")
    parser.add_argument("--library", required=True,
                        help="SMILES library file (one per line)")
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--fp_dim", type=int, default=2048)
    parser.add_argument("--latent", type=int, default=128)
    parser.add_argument("--run_bcs", action="store_true",
                        help="Run BCS NIM conformer search on top hits")
    parser.add_argument("--bcs_model", default="aimnet2",
                        choices=["aimnet2", "mace-mpa-0"])
    parser.add_argument("--output_dir", default="results/drug_hits")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    library_smiles = load_reference_library(args.library)
    sample_and_score(
        vae_ckpt=args.vae_ckpt,
        admet_ckpt=args.admet_ckpt,
        library_smiles=library_smiles,
        n_samples=args.n_samples,
        fp_dim=args.fp_dim,
        latent=args.latent,
        run_bcs=args.run_bcs,
        bcs_model=args.bcs_model,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
