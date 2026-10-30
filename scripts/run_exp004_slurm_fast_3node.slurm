#!/bin/bash
#SBATCH -J melspec-exp004-fast3
#SBATCH -w hepnode[1-3]
#SBATCH -N 3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=00:45:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -euo pipefail

ROOT="/home/wzt/melspec-eval-001"
PYTHON="/home/wzt/.venvs/melspec/bin/python"
LOG_FILE="${ROOT}/logs/004_slurm_fast_3node_slurm-${SLURM_JOB_ID}.log"

cd "${ROOT}"
export OMP_NUM_THREADS=64
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export MELSPEC_ROOT="${ROOT}"
export MELSPEC_MPI_RANKS=3
export MELSPEC_MPI_BINARY="${ROOT}/melspec_exp004_fast"

mkdir -p logs
{
    printf 'Experiment: 004 native Slurm three-node\n'
    printf 'Slurm job: %s\n' "${SLURM_JOB_ID}"
    printf 'Nodes: %s\n' "${SLURM_JOB_NODELIST}"

    g++ -std=c++17 -O3 -march=native -fopenmp -I./include \
        -DMELSPEC_DRIVER_MAIN -DMELSPEC_SLURM_MULTI \
        exp/004_mpi_multi_node.cpp -o melspec_exp004_fast

    "${PYTHON}" validate.py ./scripts/melspec_slurm_launcher.sh --runs 3
} 2>&1 | tee "${LOG_FILE}"
