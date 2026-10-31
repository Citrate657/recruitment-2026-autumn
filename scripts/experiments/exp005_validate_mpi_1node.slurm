#!/bin/bash
#SBATCH -J melspec-exp005-1node
#SBATCH -w hepnode3
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
MPI_ROOT="/cache/toolchain/openmpi-4.1.6"
UCX_ROOT="/cache/toolchain/ucx-1.18.1"
CUDA_ROOT="/cache/spack/opt/spack/linux-ubuntu26.04-icelake/none-none/cuda-13.2.1-3oevt45te4n4wckbepliz3hkdfoj3uou"
CUDA_LIB="${CUDA_ROOT}/targets/x86_64-linux/lib"
LOG_FILE="${ROOT}/logs/005_gpu_1node_debug_slurm-${SLURM_JOB_ID}.log"

cd "${ROOT}"
export OPAL_PREFIX="${MPI_ROOT}"
export PATH="${CUDA_ROOT}/bin:/home/wzt/.venvs/melspec/bin:${MPI_ROOT}/bin:${UCX_ROOT}/bin:/usr/local/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="${CUDA_LIB}:${MPI_ROOT}/lib:${UCX_ROOT}/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=64
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export MELSPEC_CPU_PERCENT=0
export MELSPEC_ROOT="${ROOT}"
export MELSPEC_MPI_RANKS=1
export MELSPEC_MPI_BINARY="${ROOT}/melspec_exp005"

mkdir -p logs
{
    printf 'Experiment: 005 one-node driver debug\n'
    printf 'Slurm job: %s\n' "${SLURM_JOB_ID}"
    nvidia-smi -L

    "${CUDA_ROOT}/bin/nvcc" \
        -std=c++17 -O3 -arch=sm_89 -Xcompiler -fopenmp \
        -I./include -I"${MPI_ROOT}/include" -DMELSPEC_DRIVER_MAIN \
        -c experiments/exp005_mpi_gpu_heterogeneous.cu -o exp005_gpu.o
    "${MPI_ROOT}/bin/mpicxx" -fopenmp exp005_gpu.o \
        -L"${CUDA_LIB}" -lcufft -lcudart -o melspec_exp005

    "${PYTHON}" validate.py ./scripts/launchers/mpi_srun.sh --runs 1
} 2>&1 | tee "${LOG_FILE}"
