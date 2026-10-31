#!/bin/bash
#SBATCH -J melspec-exp006
#SBATCH -w hepnode[1-3]
#SBATCH -N 2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:L40:2
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -euo pipefail

ROOT="/home/wzt/melspec-eval-001"
PYTHON="/home/wzt/.venvs/melspec/bin/python"
CUDA_ROOT="/cache/spack/opt/spack/linux-ubuntu26.04-icelake/none-none/cuda-13.2.1-3oevt45te4n4wckbepliz3hkdfoj3uou"
CUDA_LIB="${CUDA_ROOT}/targets/x86_64-linux/lib"
BINARY="${ROOT}/melspec_exp006"
LOG_FILE="${ROOT}/logs/006_multi_node_gpu_only_slurm-${SLURM_JOB_ID}.log"

cd "${ROOT}"
export PATH="${CUDA_ROOT}/bin:/home/wzt/.venvs/melspec/bin:/usr/local/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="${CUDA_LIB}:${LD_LIBRARY_PATH:-}"
export MELSPEC_ROOT="${ROOT}"
export MELSPEC_EXP006_BINARY="${BINARY}"
export MELSPEC_CPU_PERCENT=0
unset OMP_NUM_THREADS OMP_PROC_BIND OMP_PLACES

mkdir -p logs
{
    printf 'Experiment: 006 two-node GPU-only\n'
    printf 'Slurm job: %s\n' "${SLURM_JOB_ID}"
    printf 'Requested nodes: %s\n' "${SLURM_JOB_NODELIST}"
    printf 'Ranks: %s, tasks per node: %s\n' \
        "${SLURM_NTASKS}" "${SLURM_NTASKS_PER_NODE}"
    printf 'CPU frame percentage: 0\n'
    printf 'Compiled source: exp/006_multi_node_gpu_only.cu\n'

    printf '\n[allocated GPU resources]\n'
    srun \
        --jobid="${SLURM_JOB_ID}" \
        --overlap \
        --nodes=2 \
        --ntasks=2 \
        --ntasks-per-node=1 \
        --cpu-bind=none \
        --distribution=block:block \
        bash -c 'printf "node=%s\n" "$(hostname)"; nvidia-smi -L'

    printf '\n[compile]\n'
    "${CUDA_ROOT}/bin/nvcc" \
        -std=c++17 -O3 -arch=sm_89 \
        -I./include \
        exp/006_multi_node_gpu_only.cu \
        -L"${CUDA_LIB}" -lcufft -lcudart -o "${BINARY}"

    printf '\n[default profile]\n'
    MELSPEC_PROFILE_EXP006=1 /usr/bin/time -f 'wall=%e' \
        ./scripts/melspec_exp006_launcher.sh \
        data/audio.pcm /tmp/exp006_profile.bin \
        --sr 96000 --n_fft 8192 --hop_length 128 --n_mels 256 \
        --f_min 0 --f_max 22050
    ls -lh /tmp/exp006_profile.bin

    printf '\n[official validation]\n'
    "${PYTHON}" validate.py ./scripts/melspec_exp006_launcher.sh --runs 3
} 2>&1 | tee "${LOG_FILE}"
