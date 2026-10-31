#!/bin/bash
#SBATCH -J melspec-exp009-fast3
#SBATCH -w hepnode[1-3]
#SBATCH -N 3
#SBATCH --ntasks=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -euo pipefail

ROOT="/home/wzt/melspec-eval-001"
PYTHON="/home/wzt/.venvs/melspec/bin/python"
BINARY="${ROOT}/melspec_exp009_fast3"
LOG_FILE="${ROOT}/logs/009_three_node_cpu_slurm-${SLURM_JOB_ID}.log"

cd "${ROOT}"
export OMP_NUM_THREADS=64
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export MELSPEC_ROOT="${ROOT}"
export MELSPEC_MPI_RANKS=3
export MELSPEC_MPI_BINARY="${BINARY}"

mkdir -p logs
{
    printf 'Experiment: 009 three-node CPU pipelined\n'
    printf 'Slurm job: %s\n' "${SLURM_JOB_ID}"
    printf 'Nodes: %s\n' "${SLURM_JOB_NODELIST}"
    printf 'OpenMP: %s/%s/%s\n' \
        "${OMP_NUM_THREADS}" "${OMP_PROC_BIND}" "${OMP_PLACES}"
    printf 'Compiler: %s\n' "$(g++ --version | head -1)"

    g++ -std=c++17 -O3 -march=native -fopenmp -I./include \
        experiments/exp009_three_node_cpu_pipelined.cpp -o "${BINARY}"

    printf '\n[default stage profile]\n'
    for run in 1 2 3; do
        printf 'run=%s\n' "${run}"
        MELSPEC_PROFILE_CPU=1 /usr/bin/time -f \
            'wall=%e user=%U sys=%S maxrss_kb=%M' \
            ./scripts/launchers/slurm_native.sh \
            data/audio.pcm /tmp/exp009_profile.bin \
            --sr 96000 --n_fft 8192 --hop_length 128 --n_mels 256 \
            --f_min 0 --f_max 22050
    done

    printf '\n[official validation]\n'
    "${PYTHON}" validate.py ./scripts/launchers/slurm_native.sh --runs 3
} 2>&1 | tee "${LOG_FILE}"
