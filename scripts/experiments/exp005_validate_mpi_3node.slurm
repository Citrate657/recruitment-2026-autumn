#!/bin/bash
#SBATCH -J melspec-exp005
#SBATCH -w hepnode[1-3]
#SBATCH -N 3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:L40:2
#SBATCH --time=01:30:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -euo pipefail

ROOT="/home/wzt/melspec-eval-001"
PYTHON="/home/wzt/.venvs/melspec/bin/python"
MPI_ROOT="/cache/toolchain/openmpi-4.1.6"
UCX_ROOT="/cache/toolchain/ucx-1.18.1"
CUDA_ROOT="/cache/spack/opt/spack/linux-ubuntu26.04-icelake/none-none/cuda-13.2.1-3oevt45te4n4wckbepliz3hkdfoj3uou"
CUDA_LIB="${CUDA_ROOT}/targets/x86_64-linux/lib"
EXPERIMENT="exp005_mpi_gpu_heterogeneous"
LOG_FILE="${ROOT}/logs/005_mpi_gpu_heterogeneous_slurm-${SLURM_JOB_ID}.log"

cd "${ROOT}"
export OPAL_PREFIX="${MPI_ROOT}"
export PATH="${CUDA_ROOT}/bin:/home/wzt/.venvs/melspec/bin:${MPI_ROOT}/bin:${UCX_ROOT}/bin:/usr/local/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="${CUDA_LIB}:${MPI_ROOT}/lib:${UCX_ROOT}/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export UCX_NET_DEVICES="${UCX_NET_DEVICES:-mlx5_0:1}"
export OMPI_MCA_pml="${OMPI_MCA_pml:-ucx}"
export MELSPEC_CPU_PERCENT="${MELSPEC_CPU_PERCENT:-2}"
export MELSPEC_ROOT="${ROOT}"
export MELSPEC_MPI_RANKS=3
export MELSPEC_MPI_BINARY="${ROOT}/melspec_exp005"

mkdir -p logs
{
    printf 'Experiment: %s\n' "${EXPERIMENT}"
    printf 'Slurm job: %s\n' "${SLURM_JOB_ID}"
    printf 'Nodes: %s\n' "${SLURM_JOB_NODELIST}"
    printf 'GPUs per node: 2\n'
    printf 'CPU percent per rank: %s\n' "${MELSPEC_CPU_PERCENT}"
    printf 'OpenMP threads: %s\n' "${OMP_NUM_THREADS}"

    "${CUDA_ROOT}/bin/nvcc" \
        -std=c++17 -O3 -arch=sm_89 -Xcompiler -fopenmp \
        -I./include -I"${MPI_ROOT}/include" -DMELSPEC_DRIVER_MAIN \
        -c experiments/exp005_mpi_gpu_heterogeneous.cu -o exp005_gpu.o

    "${MPI_ROOT}/bin/mpicxx" -fopenmp \
        exp005_gpu.o \
        -L"${CUDA_LIB}" -lcufft -lcudart \
        -Wl,-rpath,"${CUDA_LIB}" -o melspec_exp005

    "${PYTHON}" validate.py ./scripts/launchers/mpi_mpirun.sh --runs 3
} 2>&1 | tee "${LOG_FILE}"
