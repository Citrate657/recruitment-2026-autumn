#!/bin/bash
#SBATCH -J melspec-exp005-fast1
#SBATCH -w hepnode[2-3]
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:L40:2
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -euo pipefail

ROOT="/home/wzt/melspec-eval-001"
PYTHON="/home/wzt/.venvs/melspec/bin/python"
CUDA_ROOT="/cache/spack/opt/spack/linux-ubuntu26.04-icelake/none-none/cuda-13.2.1-3oevt45te4n4wckbepliz3hkdfoj3uou"
CUDA_LIB="${CUDA_ROOT}/targets/x86_64-linux/lib"
LOG_FILE="${ROOT}/logs/005_slurm_fast_1node_slurm-${SLURM_JOB_ID}.log"

cd "${ROOT}"
export PATH="${CUDA_ROOT}/bin:/home/wzt/.venvs/melspec/bin:/usr/local/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="${CUDA_LIB}:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=64
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export MELSPEC_CPU_PERCENT=2
export MELSPEC_ROOT="${ROOT}"
export MELSPEC_MPI_RANKS=1
export MELSPEC_MPI_BINARY="${ROOT}/melspec_exp005_fast"

mkdir -p logs
{
    printf 'Experiment: 005 native Slurm one-node\n'
    printf 'Slurm job: %s\n' "${SLURM_JOB_ID}"
    nvidia-smi -L

    "${CUDA_ROOT}/bin/nvcc" \
        -std=c++17 -O3 -arch=sm_89 -Xcompiler -fopenmp \
        -I./include -DMELSPEC_DRIVER_MAIN -DMELSPEC_SLURM_MULTI \
        experiments/exp005_mpi_gpu_heterogeneous.cu \
        -L"${CUDA_LIB}" -lcufft -lcudart -o melspec_exp005_fast

    "${PYTHON}" validate.py ./scripts/launchers/slurm_native.sh --runs 3
} 2>&1 | tee "${LOG_FILE}"
