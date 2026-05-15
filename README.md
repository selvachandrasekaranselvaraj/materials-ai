# materials-ai

**GPU-Accelerated AI for Energy Materials and Drug Delivery**

End-to-end platform bridging NVIDIA ALCHEMI NIM microservices with research-grade
ML interatomic potentials, graph generative models, and ADMET-guided drug candidate
generation — all running on H100 GPUs (NREL Kestrel HPC).

> Reference: [NVIDIA ALCHEMI Blog](https://developer.nvidia.com/blog/faster-chemistry-and-materials-discovery-with-ai-powered-simulations-using-nvidia-alchemi/)

---

## Two application domains

### Energy materials (solid-state batteries)
`DFT (VASP AIMD) → DeepMD/MACE training → GPU MLMD (Kokkos H100) → GNN property prediction → graph diffusion cathode generation`

Systems studied: Li₃TiCl₆, Li₂ZrCl₆, Li-Ti-PS, NMC622, Na-air cathode, SrF₂, NaSnP

### Drug delivery (ADC linker/payload design)
`Molecular library → MolVAE generation → BCS NIM (AIMNet2 conformer search) → ADMET scoring → oracle-guided optimisation`

Chemistry: EDC/NHS coupling, maleimide linkers, vc-PABC cleavable payloads, anticancer payloads (DM1, MMAE, SN-38 analogues)

---

## NVIDIA ALCHEMI integration

| ALCHEMI NIM | This project | Supported MLIPs |
|---|---|---|
| **BCS NIM** | `alchemi/bcs_nim/` | AIMNet2, AIMNet2-NSE, AIMNet2-CPCM, MACE-MPA-0 |
| **BMD NIM** | `alchemi/bmd_nim/` | MACE-MPA-0, TensorNet-MatPES-r2SCAN/PBE, DeepMD |

Benchmark context: ALCHEMI BMD NIM achieves **1.4 μs/atom/step** on HGX B200 with
TensorNet at 350,000+ atom scale. BCS NIM achieves **10–100 ms/conformer** on H100.

---

## Project structure

```
battery-materials-ai/
├── alchemi/
│   ├── bcs_nim/
│   │   ├── conformer_search.py   # SMILES → AIMNet2/MACE opt → ranked conformers
│   │   ├── active_learning.py    # FPS / committee snapshot selection for DeepMD
│   │   └── filter_structures.py  # connectivity check, RMSD dedup, energy filter
│   └── bmd_nim/
│       ├── batch_md.py           # dynamic-batched NVT/NPT on GPU
│       ├── mace_mpa0_md.py       # temperature-sweep NVT with MACE-MPA-0
│       └── tensornet_md.py       # TensorNet-MatPES MD + ALCHEMI benchmark report
│
├── mlff/
│   ├── deepmd/
│   │   └── generate_input.py     # auto-generate deepmd_input.json
│   └── mace/
│       └── finetune_mace.py      # fine-tune MACE-MPA-0 on custom data
│
├── gnn/
│   ├── graphs.py                 # crystal → 101-dim node feature + PyG graph
│   ├── model.py                  # multi-property GCNConv + fallback MP-GNN
│   └── train.py                  # training loop, early stopping, Arrhenius
│
├── generative/
│   ├── energy_materials/         # graph diffusion (DDPM) for novel cathode structures
│   │   ├── models/graph_diffusion.py
│   │   ├── models/oracle_gnn.py
│   │   └── scripts/{train,sample,score}_diffusion.py
│   └── drug_delivery/
│       ├── data/
│       │   ├── prepare_dataset.py      # EDC/NHS coupling dataset (from experiment)
│       │   └── molecule_library.py     # linker + payload + drug-like fragment library
│       ├── models/
│       │   ├── mol_vae.py              # Molecular β-VAE + property predictor head
│       │   ├── property_predictor.py   # Multi-task ADMET (yield, logP, toxicity, BBB)
│       │   └── reaction_oracle.py      # ReactionOracle — predicts coupling yield
│       └── scripts/
│           └── sample_and_score.py     # VAE sampling → ADMET scoring → BCS NIM
│
├── analysis/
│   ├── rdf.py                    # partial RDF, coordination numbers
│   └── conductivity.py           # MSD, diffusivity D, Einstein conductivity σ, Arrhenius
│
├── workflows/
│   ├── dp_train_sub_gpu.sh       # Slurm: DeepMD GPU training (4× H100, 48 h)
│   └── lmp_sub_gpu.sh            # Slurm: LAMMPS Kokkos GPU MD (4 GPUs, MPI)
│
├── requirements.txt
└── README.md
```

---

## Quick start

### 1. BCS NIM — conformer search (energy materials or drug delivery)
```bash
# Battery cathode structure optimisation
python alchemi/bcs_nim/conformer_search.py \
    --smiles "CC(=O)[O-].[Li+]" \
    --model mace-mpa-0 --fmax 0.005 --n_confs 20

# Drug delivery candidate conformer search
python alchemi/bcs_nim/conformer_search.py \
    --smiles "O=C1C=CC(=O)N1CCCCCO" \
    --model aimnet2 --fmax 0.005 --n_confs 10

python alchemi/bcs_nim/filter_structures.py \
    --input_dir results/conformers \
    --energy_window 0.5 --rmsd_threshold 0.3
```

### 2. BMD NIM — batched GPU MD (energy materials)
```bash
# Run 3 systems in parallel on H100
python alchemi/bmd_nim/batch_md.py \
    --systems Li3TiCl6.cif Li2ZrCl6.cif NMC622.poscar \
    --model mace-mpa-0 --ensemble nvt --temp 600 --steps 2000000

# Temperature sweep for Arrhenius conductivity
python alchemi/bmd_nim/mace_mpa0_md.py \
    --structure Li3TiCl6.cif \
    --temps 300 600 900 1200 --steps 2000000

# TensorNet with ALCHEMI benchmark comparison
python alchemi/bmd_nim/tensornet_md.py \
    --structure NMC622.poscar --temp 600 --variant r2SCAN
```

### 3. DeepMD GPU training (4× H100)
```bash
# Auto-generate input.json from training sets
python mlff/deepmd/generate_input.py \
    --type_map Li Ti Cl \
    --data_root 00.data/training_data \
    --rcut 6.0 --n_steps 1000000

sbatch workflows/dp_train_sub_gpu.sh
```

### 4. GNN multi-property prediction
```bash
python gnn/train.py \
    --data data/processed/structures.pkl \
    --targets Ef EAH Eg VBM CBM \
    --epochs 200 --device cuda
```

### 5. Drug delivery — VAE generation + ADMET scoring
```bash
# Build molecule library
python generative/drug_delivery/data/molecule_library.py

# Sample and score candidates (VAE + ADMET oracle)
python generative/drug_delivery/scripts/sample_and_score.py \
    --vae_ckpt results/mol_vae/mol_vae.pt \
    --admet_ckpt results/admet_predictor/admet_predictor.pt \
    --library data/drug_library.smi \
    --n_samples 5000 \
    --run_bcs --bcs_model aimnet2

# Post-process: ionic conductivity of electrolyte
python analysis/conductivity.py --traj results/bmd/Li3TiCl6/trajectory.xyz \
    --temp 600 --species Li
python analysis/rdf.py --traj results/bmd/Li3TiCl6/trajectory.xyz \
    --species Li Cl
```

---

## HPC setup (NREL Kestrel)

```bash
salloc --account=lips --time=05:00:00 --partition=gpu-h100 --gpus=1

module purge
module load cuda/12.3 openmpi/4.1.6-gcc conda/2024.06.1
source activate /projects/lips/apps/deepmd-lammps-gpu/conda_env
export PATH=/projects/lips/apps/deepmd-lammps-gpu/lammps_install/bin:$PATH
export LD_LIBRARY_PATH=/projects/lips/apps/deepmd-lammps-gpu/conda_env/lib:$LD_LIBRARY_PATH
```

---

## Key publications (first author)

| Paper | Journal | IF |
|---|---|---|
| Li dynamics in Li-Ti-PS MLMD (2026) | *J. Energy Storage* 141 | 9.8 |
| Li₃TiCl₆ MLMD study (2024) | *J. Electrochem. Soc.* 171, 050544 | 3.1 |
| GNN for structural & electronic properties | arXiv:2411.02331 | — |

---

## Open-source packages used

- **AtomicAI** — `pip install AtomicAI` — custom MLFF, ACSF/LAAF descriptors, MD tools
- **deepatom** — github.com/selvachandrasekaranselvaraj/deepatom — GNN property prediction

**Author:** Dr. Selva Chandrasekaran Selvaraj
Research Scientist, UIC / Argonne National Laboratory
selvas@uic.edu
