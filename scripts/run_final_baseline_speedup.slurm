#!/bin/bash
#SBATCH -J melspec-final-speedup
#SBATCH -w hepnode[1-3]
#SBATCH -N 3
#SBATCH --ntasks=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=00:20:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -euo pipefail

ROOT="/home/wzt/melspec-eval-001"
PYTHON="/home/wzt/.venvs/melspec/bin/python"
LOG_FILE="${ROOT}/logs/final_baseline_vs_exp009_slurm-${SLURM_JOB_ID}.log"
PREFIX="${ROOT}/shared_tmp/audio-1m.pcm"

cd "${ROOT}"
export OMP_NUM_THREADS=64
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export MELSPEC_ROOT="${ROOT}"
export MELSPEC_MPI_RANKS=3
export MELSPEC_MPI_BINARY="${ROOT}/melspec_exp009_fast3"

mkdir -p logs
mkdir -p "${ROOT}/shared_tmp"
dd if=data/audio.pcm of="${PREFIX}" bs=2 count=1000000 status=none
trap 'rm -f "${PREFIX}"' EXIT
{
    printf 'Experiment: original implementation vs exp009\n'
    printf 'Slurm job: %s\n' "${SLURM_JOB_ID}"
    printf 'Nodes: %s\n' "${SLURM_JOB_NODELIST}"
    printf 'Input: first 1,000,000 samples of data/audio.pcm\n'

    make clean
    make -j64
    cp melspec melspec_baseline

    g++ -std=c++17 -O3 -march=native -fopenmp -I./include \
        exp/009_three_node_cpu_pipelined.cpp -o melspec_exp009_fast3

    "${PYTHON}" scripts/benchmark_experiments.py \
        --baseline ./melspec_baseline \
        --experiment ./scripts/melspec_slurm_launcher.sh \
        --input "${PREFIX}" \
        --cases default \
        --runs 1 \
        --skip-correctness
} 2>&1 | tee "${LOG_FILE}"
