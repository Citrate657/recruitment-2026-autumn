#!/bin/bash
set -euo pipefail

ROOT="${MELSPEC_ROOT:-/home/wzt/melspec-eval-001}"
BINARY="${MELSPEC_EXP006_BINARY:-${ROOT}/melspec_exp006}"

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
SHARED_READY="${SHARED_OUTPUT}.ready"
RUN_TOKEN="$(date +%s%N)"

truncate -s 0 "${SHARED_OUTPUT}" 2>/dev/null || true
truncate -s 8 "${SHARED_READY}" 2>/dev/null || true
export MELSPEC_RUN_TOKEN="${RUN_TOKEN}"
export MELSPEC_CPU_PERCENT=0

srun \
    --jobid="${SLURM_JOB_ID}" \
    --overlap \
    --mpi=none \
    --nodes=2 \
    --ntasks=2 \
    --ntasks-per-node=1 \
    --cpus-per-task="${SLURM_CPUS_PER_TASK:-4}" \
    --cpu-bind=none \
    --distribution=block:block \
    --kill-on-bad-exit=1 \
    "${BINARY}" "${INPUT_PATH}" "${SHARED_OUTPUT}" "$@"

rm -f "${REQUESTED_OUTPUT}"
ln -s "${SHARED_OUTPUT}" "${REQUESTED_OUTPUT}"
