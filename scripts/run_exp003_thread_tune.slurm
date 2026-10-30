#!/bin/bash
#SBATCH -J melspec-exp003-thread-tune
#SBATCH -w hepnode[2-3]
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:L40:2
#SBATCH --time=00:45:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -euo pipefail

ROOT="/home/wzt/melspec-eval-001"
PYTHON="/home/wzt/.venvs/melspec/bin/python"
CUDA_ROOT="/cache/spack/opt/spack/linux-ubuntu26.04-icelake/none-none/cuda-13.2.1-3oevt45te4n4wckbepliz3hkdfoj3uou"
CUDA_LIB="${CUDA_ROOT}/targets/x86_64-linux/lib"
LOG_FILE="${ROOT}/logs/003_cpu_thread_tuning_slurm-${SLURM_JOB_ID}.log"

cd "${ROOT}"
export PATH="${CUDA_ROOT}/bin:/home/wzt/.venvs/melspec/bin:/usr/local/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="${CUDA_LIB}:${LD_LIBRARY_PATH:-}"
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

mkdir -p logs
{
    printf 'Experiment: 003 CPU thread/share tuning\n'
    printf 'Slurm job: %s\n' "${SLURM_JOB_ID}"
    printf 'Node: %s\n' "$(hostname)"
    printf 'GPUs: %s\n' "${CUDA_VISIBLE_DEVICES}"

    "${CUDA_ROOT}/bin/nvcc" \
        -std=c++17 -O3 -arch=sm_89 -Xcompiler -fopenmp \
        -I./include src/main.cpp exp/003_cpu_gpu_heterogeneous.cu \
        -L"${CUDA_LIB}" -lcufft -lcudart -o melspec_exp003

    for threads in 8 16 32 64; do
        "${PYTHON}" scripts/tune_gpu_heterogeneous.py \
            --binary ./melspec_exp003 \
            --percentages 4,8,12 \
            --threads "${threads}" \
            --runs 1
    done
} 2>&1 | tee "${LOG_FILE}"
