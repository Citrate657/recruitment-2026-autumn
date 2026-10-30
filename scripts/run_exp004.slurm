#!/bin/bash
#SBATCH -J melspec-exp004
#SBATCH -w hepnode[1-3]
#SBATCH -N 3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -euo pipefail

ROOT="/home/wzt/melspec-eval-001"
MELSPEC_ROOT="${ROOT}"
MPI_ROOT="/cache/toolchain/openmpi-4.1.6"
UCX_ROOT="/cache/toolchain/ucx-1.18.1"
PYTHON="/home/wzt/.venvs/melspec/bin/python"
EXPERIMENT="004_mpi_multi_node"
LOG_FILE="${ROOT}/logs/${EXPERIMENT}_slurm-${SLURM_JOB_ID}.log"

cd "${ROOT}"
export OPAL_PREFIX="${MPI_ROOT}"
export PATH="/home/wzt/.venvs/melspec/bin:${MPI_ROOT}/bin:${UCX_ROOT}/bin:/usr/local/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="${MPI_ROOT}/lib:${UCX_ROOT}/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export UCX_NET_DEVICES="${UCX_NET_DEVICES:-mlx5_0:1}"
export OMPI_MCA_pml="${OMPI_MCA_pml:-ucx}"
export MELSPEC_ROOT="${ROOT}"
export MELSPEC_MPI_RANKS=3
export MELSPEC_MPI_BINARY="${ROOT}/melspec_exp004"

mkdir -p logs
{
    printf 'Experiment: %s\n' "${EXPERIMENT}"
    printf 'Slurm job: %s\n' "${SLURM_JOB_ID}"
    printf 'Nodes: %s\n' "${SLURM_JOB_NODELIST}"
    printf 'CPU per rank: %s\n' "${SLURM_CPUS_PER_TASK}"
    printf 'Compiler: %s\n' "$("${MPI_ROOT}/bin/mpicxx" --version | head -1)"

    "${MPI_ROOT}/bin/mpicxx" \
        -std=c++17 -Wall -Wextra -O3 -march=native -fopenmp \
        -I./include -DMELSPEC_DRIVER_MAIN exp/004_mpi_multi_node.cpp \
        -o melspec_exp004

    "${PYTHON}" validate.py ./scripts/melspec_mpi_launcher.sh --runs 3
} 2>&1 | tee "${LOG_FILE}"
