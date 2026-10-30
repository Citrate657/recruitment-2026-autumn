#!/bin/bash
#SBATCH -J melspec-exp004-debug
#SBATCH -w hepnode[2-3]
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:45:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -euo pipefail

ROOT="/home/wzt/melspec-eval-001"
MELSPEC_ROOT="${ROOT}"
MPI_ROOT="/cache/toolchain/openmpi-4.1.6"
UCX_ROOT="/cache/toolchain/ucx-1.18.1"
PYTHON="/home/wzt/.venvs/melspec/bin/python"
LOG_FILE="${ROOT}/logs/004_mpi_2node_debug_slurm-${SLURM_JOB_ID}.log"

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
export MELSPEC_MPI_RANKS=2
export MELSPEC_MPI_BINARY="${ROOT}/melspec_exp004"

mkdir -p logs
{
    printf 'Experiment: 004 two-node debug\n'
    printf 'Slurm job: %s\n' "${SLURM_JOB_ID}"
    printf 'Nodes: %s\n' "${SLURM_JOB_NODELIST}"
    printf 'MPI ranks: 2\n'
    printf 'CPU per rank: %s\n' "${SLURM_CPUS_PER_TASK}"

    "${MPI_ROOT}/bin/mpicxx" \
        -std=c++17 -Wall -Wextra -O3 -march=native -fopenmp \
        -I./include -DMELSPEC_DRIVER_MAIN exp/004_mpi_multi_node.cpp \
        -o melspec_exp004

    printf '\n[direct default debug]\n'
    ./scripts/melspec_mpi_launcher.sh data/audio.pcm /tmp/mpi_debug.bin \
        --sr 96000 --n_fft 8192 --hop_length 128 --n_mels 256 \
        --f_min 0 --f_max 22050
    ls -lh /tmp/mpi_debug.bin

    printf '\n[official validation]\n'
    "${PYTHON}" validate.py ./scripts/melspec_mpi_launcher.sh --runs 1
} 2>&1 | tee "${LOG_FILE}"
