#!/bin/bash
#SBATCH -J melspec-exp007-profile
#SBATCH -w hepnode[1-3]
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=00:20:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -euo pipefail

ROOT="/home/wzt/melspec-eval-001"
PYTHON="/home/wzt/.venvs/melspec/bin/python"
BINARY="${ROOT}/melspec_exp007_profile"
LOG_FILE="${ROOT}/logs/007_single_node_profile_slurm-${SLURM_JOB_ID}.log"

cd "${ROOT}"
export OMP_NUM_THREADS=64
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

mkdir -p logs
{
    printf 'Experiment: 007 single-node profiling driver\n'
    printf 'Slurm job: %s\n' "${SLURM_JOB_ID}"
    printf 'Node: %s\n' "$(hostname)"
    printf 'OpenMP: %s/%s/%s\n' \
        "${OMP_NUM_THREADS}" "${OMP_PROC_BIND}" "${OMP_PLACES}"
    printf 'Compiler: %s\n' "$(g++ --version | head -1)"

    g++ -std=c++17 -O3 -march=native -fopenmp -I./include \
        exp/007_single_node_cpu_profiled.cpp -o "${BINARY}"

    printf '\n[stage profile: default]\n'
    for run in 1 2 3; do
        printf 'run=%s\n' "${run}"
        MELSPEC_PROFILE_CPU=1 /usr/bin/time -f \
            'wall=%e user=%U sys=%S maxrss_kb=%M' \
            "${BINARY}" data/audio.pcm /tmp/exp007_profile.bin \
            --sr 96000 --n_fft 8192 --hop_length 128 --n_mels 256 \
            --f_min 0 --f_max 22050
    done

    printf '\n[official correctness]\n'
    "${PYTHON}" validate.py "${BINARY}" --runs 1
} 2>&1 | tee "${LOG_FILE}"
