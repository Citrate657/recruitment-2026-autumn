#!/bin/bash
#SBATCH -J melspec-exp002
#SBATCH -w hepnode[1-3]
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:L40:2
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -euo pipefail

ROOT="/home/wzt/melspec-eval-001"
PYTHON="/home/wzt/.venvs/melspec/bin/python"
CUDA_ROOT="/cache/spack/opt/spack/linux-ubuntu26.04-icelake/none-none/cuda-13.2.1-3oevt45te4n4wckbepliz3hkdfoj3uou"
CUDA_LIB="${CUDA_ROOT}/targets/x86_64-linux/lib"
EXPERIMENT="002_gpu_cufft"
LOG_FILE="${ROOT}/logs/${EXPERIMENT}_slurm-${SLURM_JOB_ID}.log"

cd "${ROOT}"
export PATH="${CUDA_ROOT}/bin:/home/wzt/.venvs/melspec/bin:/usr/local/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="${CUDA_LIB}:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores

mkdir -p logs
{
    printf 'Experiment: %s\n' "${EXPERIMENT}"
    printf 'Slurm job: %s\n' "${SLURM_JOB_ID}"
    printf 'Node: %s\n' "$(hostname)"
    printf 'GPUs: %s\n' "${CUDA_VISIBLE_DEVICES}"
    printf 'Compiler: %s\n' "$("${CUDA_ROOT}/bin/nvcc" --version | tail -1)"
    nvidia-smi -L

    "${CUDA_ROOT}/bin/nvcc" \
        -std=c++17 -O3 -arch=sm_89 -Xcompiler -fopenmp \
        -I./include src/main.cpp exp/002_gpu_cufft.cu \
        -L"${CUDA_LIB}" -lcufft -lcudart \
        -o melspec_exp002

    "${PYTHON}" validate.py ./melspec_exp002 --runs 3
} 2>&1 | tee "${LOG_FILE}"
