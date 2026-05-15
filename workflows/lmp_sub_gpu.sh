#!/bin/bash
#SBATCH --account=nmclps
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks-per-node=4          # one MPI rank per GPU
#SBATCH --cpus-per-task=1
#SBATCH --time=72:00:00
#SBATCH --job-name=mlmd_gpu
#SBATCH --mem=300G

module purge
module load cuda/12.3
module load openmpi/4.1.6-gcc
module load conda/2024.06.1

source activate /projects/lips/apps/deepmd-lammps-gpu/conda_env
export PATH=/projects/lips/apps/deepmd-lammps-gpu/lammps_install/bin:$PATH
export LD_LIBRARY_PATH=/projects/lips/apps/deepmd-lammps-gpu/conda_env/lib:$LD_LIBRARY_PATH
export MPICH_GPU_SUPPORT_ENABLED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# BMD NIM analogue: Kokkos GPU-accelerated LAMMPS with DeepMD pair style
srun --gpus-per-task=1 lmp -k on gpus 4 -sf kk -in in.lammps
