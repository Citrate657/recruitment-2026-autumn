#!/bin/bash
#SBATCH -J melspec-exp001
#SBATCH -w hepnode[1-3]
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -euo pipefail

ROOT="/home/wzt/melspec-eval-001"
PYTHON="/home/wzt/.venvs/melspec/bin/python"
EXPERIMENT="exp001_packed_fft_sparse_mel_simd"
LOG_FILE="${ROOT}/logs/001_packed_fft_sparse_mel_simd_slurm-${SLURM_JOB_ID}.log"

cd "${ROOT}"
export PATH="/home/wzt/.venvs/melspec/bin:/usr/local/bin:/usr/bin:/bin"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-${SLURM_NTASKS:-1}}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"

mkdir -p logs
{
    printf 'Experiment: %s\n' "${EXPERIMENT}"
    printf 'Git revision: %s\n' "$(git rev-parse HEAD 2>/dev/null || printf 'not-a-git-checkout')"
    printf 'Slurm job: %s\n' "${SLURM_JOB_ID}"
    printf 'Node: %s\n' "$(hostname)"
    printf 'OpenMP threads: %s\n' "${OMP_NUM_THREADS}"
    printf 'Python: %s\n' "$("${PYTHON}" --version 2>&1)"
    printf 'Compiler: %s\n' "$(g++ --version | head -1)"

    make clean
    make -j"${OMP_NUM_THREADS}"
    cp melspec melspec_baseline

    make clean
    make EXPERIMENT="${EXPERIMENT}" -j"${OMP_NUM_THREADS}"
    cp melspec melspec_exp001

    printf '\n%s\n' '[OFFICIAL FULL VALIDATION]'
    "${PYTHON}" validate.py ./melspec_exp001 --runs 3

    printf '\n%s\n' '[ONE-MILLION-SAMPLE SPEEDUP]'
    "${PYTHON}" scripts/tools/benchmark_baseline.py \
        --baseline ./melspec_baseline \
        --experiment ./melspec_exp001 \
        --cases default \
        --runs 1 \
        --max-samples 1000000
} 2>&1 | tee "${LOG_FILE}"
