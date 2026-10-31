#!/bin/bash
#SBATCH -J melspec-cpu-scaling
#SBATCH -w hepnode[2-3]
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=00:45:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -euo pipefail

ROOT="/home/wzt/melspec-eval-001"
PYTHON="/home/wzt/.venvs/melspec/bin/python"
EXPERIMENT="exp001_packed_fft_sparse_mel_simd"
LOG_FILE="${ROOT}/logs/cpu-thread-scaling_slurm-${SLURM_JOB_ID}.log"

cd "${ROOT}"
export PATH="/home/wzt/.venvs/melspec/bin:/usr/local/bin:/usr/bin:/bin"

mkdir -p logs
{
    printf 'Experiment: %s\n' "${EXPERIMENT}"
    printf 'Slurm job: %s\n' "${SLURM_JOB_ID}"
    printf 'Node: %s\n' "$(hostname)"
    printf 'Allocated CPUs: %s\n' "${SLURM_CPUS_PER_TASK}"
    printf 'Compiler: %s\n' "$(g++ --version | head -1)"
    lscpu | grep -E '^(Architecture|CPU\(s\)|Thread|Core|Socket|NUMA|Model name)'

    make clean
    make EXPERIMENT="${EXPERIMENT}" -j64

    "${PYTHON}" scripts/tools/benchmark_cpu_scaling.py \
        --binary ./melspec \
        --threads 1,2,4,8,12,16,24,32,40,48,56,64 \
        --bindings close,spread \
        --runs 3
} 2>&1 | tee "${LOG_FILE}"
