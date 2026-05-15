#!/usr/bin/env python
"""
Build a curated small-molecule library for drug delivery generation.

Sources:
  - EDC/NHS coupling reagents and payloads (from prepare_dataset.py)
  - FDA-approved linker chemistries (PEG, maleimide, disulfide)
  - Common anticancer payloads (DM1, MMAE, SN-38 analogues as SMILES)
  - Drug-like ZINC fragments

Output: data/drug_library.smi  (one SMILES per line)
        data/drug_library_props.csv  (MW, logP, n_rotatable_bonds, …)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd

LINKER_SMILES: List[str] = [
    # EDC coupling reagents
    "CCN=C=NCCCN(C)C",                    # EDC
    "OC1=CC=CC=C1N1C(=O)CCC1=O",          # NHS
    # Maleimide linkers
    "O=C1C=CC(=O)N1CCCCCO",               # mal-PEG4-OH
    "O=C1C=CC(=O)N1CCC(=O)O",             # mal-propionic acid
    "N#CCCSC1=CC=CC=C1",                   # SATA
    # Disulfide linkers
    "SCCCOC(=O)OCCC",                      # SPDP analogue
    "OC(=O)CCSSCC(=O)O",                   # DTNB core
    # PEG linkers
    "OCCOCCOCCO",                           # PEG3-OH
    "OCCOCCOCCOCCO",                        # PEG4-OH
    "NCC(=O)OCCOCCOCCO",                    # NH2-PEG3-OH
    # Valine-citrulline dipeptide (vc-PABC for protease-cleavable ADCs)
    "CC(C)[C@@H](NC(=O)[C@@H](N)CCCNC(=N)N)C(=O)O",
]

PAYLOAD_SMILES: List[str] = [
    # DM1 analogue (simplified maytansinoid core)
    "COC1=CC=C(C=C1)C(=O)N[C@@H](C)C(=O)O",
    # MMAE (monomethyl auristatin E core fragment)
    "CC[C@H](C)[C@@H](OC(=O)[C@@H](NC(=O)[C@@H](CC(C)C)NC(C)=O)CC1=CC=CC=C1)C(=O)N",
    # SN-38 (camptothecin derivative)
    "CCC1(O)C(=O)OCC2=C1C=C1C(=O)N3CCCC3=NC1=C2",
    # Doxorubicin core
    "O=C1C2=C(O)C=CC=C2C(=O)C3=C1C=CC(O)=C3",
    # Mertansine (DM1) simplified
    "CC1=CC(=O)N(C(=O)C)C(=O)C1CC(=O)O",
]

DRUG_LIKE_FRAGMENTS: List[str] = [
    # Benzimidazole
    "C1=CC2=C(C=C1)NC=N2",
    # Pyrimidine
    "C1=CN=CN=C1",
    # Indole
    "C1=CC2=C(C=C1)C=CN2",
    # Quinoline
    "C1=CC2=CC=CC=C2N=C1",
    # Piperazine
    "C1CNCCN1",
    # Morpholine
    "C1COCCN1",
    # Thiophene
    "C1=CC=CS1",
    # Furan
    "C1=CC=CO1",
    # Oxazole
    "C1=COC=N1",
    # Triazole
    "C1=CN=NN1",
]

ALL_SMILES = LINKER_SMILES + PAYLOAD_SMILES + DRUG_LIKE_FRAGMENTS


def compute_properties(smiles_list: List[str]) -> pd.DataFrame:
    rows = []
    for smi in smiles_list:
        row: Dict = {"smiles": smi}
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, rdMolDescriptors
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                row.update({"MW": np.nan, "logP": np.nan,
                            "n_rot": np.nan, "n_hbd": np.nan,
                            "n_hba": np.nan, "tpsa": np.nan, "valid": False})
            else:
                row["MW"] = Descriptors.MolWt(mol)
                row["logP"] = Descriptors.MolLogP(mol)
                row["n_rot"] = rdMolDescriptors.CalcNumRotatableBonds(mol)
                row["n_hbd"] = rdMolDescriptors.CalcNumHBD(mol)
                row["n_hba"] = rdMolDescriptors.CalcNumHBA(mol)
                row["tpsa"] = Descriptors.TPSA(mol)
                row["valid"] = True
                # Lipinski rule-of-5 filter
                row["lipinski"] = (
                    row["MW"] <= 500
                    and row["logP"] <= 5
                    and row["n_hbd"] <= 5
                    and row["n_hba"] <= 10
                )
        except ImportError:
            row.update({"MW": np.nan, "logP": np.nan, "valid": None,
                        "lipinski": None})
        rows.append(row)
    return pd.DataFrame(rows)


def build_library(
    extra_smiles: List[str] = None,
    output_dir: str = "data",
    filter_lipinski: bool = False,
) -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)
    smiles = list(ALL_SMILES)
    if extra_smiles:
        smiles.extend(extra_smiles)

    # Deduplicate by canonical SMILES if RDKit available
    try:
        from rdkit import Chem
        canonical = {}
        for s in smiles:
            mol = Chem.MolFromSmiles(s)
            if mol:
                canonical[Chem.MolToSmiles(mol)] = s
        smiles = list(canonical.keys())
        print(f"After canonicalisation: {len(smiles)} unique SMILES")
    except ImportError:
        pass

    df = compute_properties(smiles)

    if filter_lipinski:
        df = df[df["lipinski"] == True].reset_index(drop=True)
        print(f"After Lipinski filter: {len(df)} molecules")

    smi_path = Path(output_dir) / "drug_library.smi"
    df["smiles"].to_csv(smi_path, index=False, header=False)

    csv_path = Path(output_dir) / "drug_library_props.csv"
    df.to_csv(csv_path, index=False)

    print(f"Library: {len(df)} molecules")
    print(f"  SMILES → {smi_path}")
    print(f"  Props  → {csv_path}")
    return df


if __name__ == "__main__":
    df = build_library(output_dir="data")
    print(df[["smiles", "MW", "logP", "lipinski"]].to_string(index=False))
