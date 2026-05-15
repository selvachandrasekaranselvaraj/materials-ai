#!/bin/bash
#SBATCH --account=nmclps
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --job-name=deepmd_gpu
#SBATCH --mem=64G

module purge
module load cuda/12.3
module load openmpi/4.1.6-gcc
module load conda/2024.06.1

source /projects/lips/apps/deepmd-lammps-gpu/conda_env/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export DP_INTRA_OP_PARALLELISM_THREADS=$SLURM_CPUS_PER_TASK
export DP_INTER_OP_PARALLELISM_THREADS=2

# Optional: auto-generate input.json from training set directories
# python mlff/deepmd/generate_input.py \
#     --type_map Li Ti Cl \
#     --data_root 00.data/training_data \
#     --rcut 6.0 --n_steps 1000000

dp train deepmd_input.json && \
dp freeze -o pot.pb && \
dp compress -i pot.pb -o pot_com.pb && \
dp test -m pot_com.pb -s 00.data/validation_data -n 10000 -d results

deactivate
