#!/bin/bash
#SBATCH -J melspec-exp004-prof3
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
BINARY="${ROOT}/melspec_exp004_fast"
LOG_FILE="${ROOT}/logs/004_three_node_profile_slurm-${SLURM_JOB_ID}.log"

cd "${ROOT}"
export OMP_NUM_THREADS=64
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export MELSPEC_ROOT="${ROOT}"
export MELSPEC_MPI_RANKS=3
export MELSPEC_MPI_BINARY="${BINARY}"

mkdir -p logs
{
    printf 'Experiment: 004 three-node profile\n'
    printf 'Slurm job: %s\n' "${SLURM_JOB_ID}"
    printf 'Nodes: %s\n' "${SLURM_JOB_NODELIST}"
    printf 'OpenMP: %s/%s/%s\n' \
        "${OMP_NUM_THREADS}" "${OMP_PROC_BIND}" "${OMP_PLACES}"

    g++ -std=c++17 -O3 -march=native -fopenmp -I./include \
        -DMELSPEC_DRIVER_MAIN -DMELSPEC_SLURM_MULTI \
        experiments/exp004_mpi_multi_node.cpp -o "${BINARY}"

    printf '\n[default stage profile]\n'
    for run in 1 2 3; do
        printf 'run=%s\n' "${run}"
        MELSPEC_PROFILE_MPI=1 /usr/bin/time -f \
            'wall=%e user=%U sys=%S maxrss_kb=%M' \
            ./scripts/launchers/slurm_native.sh \
            data/audio.pcm /tmp/exp004_profile3.bin \
            --sr 96000 --n_fft 8192 --hop_length 128 --n_mels 256 \
            --f_min 0 --f_max 22050
    done
} 2>&1 | tee "${LOG_FILE}"
