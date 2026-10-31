#!/bin/bash
set -euo pipefail

ROOT="${MELSPEC_ROOT:-/home/wzt/melspec-eval-001}"
MPI_ROOT="${MELSPEC_MPI_ROOT:-/cache/toolchain/openmpi-4.1.6}"
UCX_ROOT="${MELSPEC_UCX_ROOT:-/cache/toolchain/ucx-1.18.1}"
RANKS="${MELSPEC_MPI_RANKS:-3}"
BINARY="${MELSPEC_MPI_BINARY:?MELSPEC_MPI_BINARY is required}"

export OPAL_PREFIX="${MPI_ROOT}"
export PATH="${MPI_ROOT}/bin:${UCX_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${MPI_ROOT}/lib:${UCX_ROOT}/lib:${LD_LIBRARY_PATH:-}"
export UCX_NET_DEVICES="${UCX_NET_DEVICES:-mlx5_0:1}"
export OMPI_MCA_pml="${OMPI_MCA_pml:-ucx}"
export PMIX_MCA_psec="${PMIX_MCA_psec:-^munge}"
export PMIX_MCA_gds="${PMIX_MCA_gds:-^shmem2}"

if [[ $# -lt 2 ]]; then
    echo "usage: $0 <input_pcm> <output_bin> [options]" >&2
    exit 1
fi

INPUT_PATH="$1"
REQUESTED_OUTPUT="$2"
shift 2
SHARED_DIR="${ROOT}/shared_tmp"
mkdir -p "${SHARED_DIR}"
SHARED_OUTPUT="${SHARED_DIR}/$(basename "${REQUESTED_OUTPUT}")"

if srun \
    --jobid="${SLURM_JOB_ID}" \
    --overlap \
    --mpi="${MELSPEC_SRUN_MPI:-pmix_v5}" \
    --ntasks="${RANKS}" \
    --ntasks-per-node=1 \
    --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" \
    --cpu-bind=none \
    --distribution=block:block \
    --kill-on-bad-exit=1 \
    "${BINARY}" "${INPUT_PATH}" "${SHARED_OUTPUT}" "$@"; then
    rm -f "${REQUESTED_OUTPUT}"
    ln -s "${SHARED_OUTPUT}" "${REQUESTED_OUTPUT}"
else
    status=$?
    exit "${status}"
fi
