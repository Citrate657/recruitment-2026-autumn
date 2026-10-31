#!/bin/bash
#SBATCH -J melspec-exp001-profile
#SBATCH -w hepnode[1-3]
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=00:20:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -euo pipefail

ROOT="/home/wzt/melspec-eval-001"
LOG_FILE="${ROOT}/logs/001_single_node_profile_slurm-${SLURM_JOB_ID}.log"

cd "${ROOT}"
export OMP_NUM_THREADS=64
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

mkdir -p logs
{
    printf 'Experiment: 001 single-node stage profile\n'
    printf 'Slurm job: %s\n' "${SLURM_JOB_ID}"
    printf 'Node: %s\n' "$(hostname)"
    printf 'OpenMP: %s/%s/%s\n' \
        "${OMP_NUM_THREADS}" "${OMP_PROC_BIND}" "${OMP_PLACES}"
    printf 'Compiler: %s\n' "$(g++ --version | head -1)"

    make clean
    make EXPERIMENT=exp001_packed_fft_sparse_mel_simd -j64

    printf '\n[full output]\n'
    for run in 1 2 3; do
        printf 'run=%s\n' "${run}"
        /usr/bin/time -f 'wall=%e user=%U sys=%S maxrss_kb=%M' \
            ./melspec data/audio.pcm /tmp/exp001_profile.bin \
            --sr 96000 --n_fft 8192 --hop_length 128 --n_mels 256 \
            --f_min 0 --f_max 22050
    done

    printf '\n[output conversion without disk payload: /dev/null]\n'
    for run in 1 2 3; do
        printf 'run=%s\n' "${run}"
        /usr/bin/time -f 'wall=%e user=%U sys=%S maxrss_kb=%M' \
            ./melspec data/audio.pcm /dev/null \
            --sr 96000 --n_fft 8192 --hop_length 128 --n_mels 256 \
            --f_min 0 --f_max 22050
    done

    printf '\n[perf counters]\n'
    if command -v perf >/dev/null 2>&1; then
        perf stat \
            -e task-clock,cycles,instructions,cache-references,cache-misses,page-faults \
            ./melspec data/audio.pcm /dev/null \
            --sr 96000 --n_fft 8192 --hop_length 128 --n_mels 256 \
            --f_min 0 --f_max 22050
    else
        printf 'perf is not available\n'
    fi
} 2>&1 | tee "${LOG_FILE}"
