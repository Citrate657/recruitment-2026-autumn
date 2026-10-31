#!/bin/bash
#SBATCH -J mpi-transport-probe
#SBATCH -w hepnode[2-3]
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -euo pipefail

ROOT="/home/wzt/melspec-eval-001"
MPI_ROOT="/cache/toolchain/openmpi-4.1.6"
UCX_ROOT="/cache/toolchain/ucx-1.18.1"
LOG_FILE="${ROOT}/logs/mpi_transport_probe_slurm-${SLURM_JOB_ID}.log"

cd "${ROOT}"
export OPAL_PREFIX="${MPI_ROOT}"
export PATH="${MPI_ROOT}/bin:${UCX_ROOT}/bin:/usr/local/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="${MPI_ROOT}/lib:${UCX_ROOT}/lib:${LD_LIBRARY_PATH:-}"
export UCX_NET_DEVICES="${UCX_NET_DEVICES:-mlx5_0:1}"
export OMPI_MCA_pml="${OMPI_MCA_pml:-ucx}"

mkdir -p logs
{
    printf 'MPI transport probe, job %s\n' "${SLURM_JOB_ID}"
    "${MPI_ROOT}/bin/mpicxx" -O2 -std=c++17 scripts/diagnostics/mpi_bandwidth_test.cpp \
        -o mpi_bandwidth_test

    "${MPI_ROOT}/bin/mpirun" --map-by ppr:1:node --bind-to core \
        --report-bindings -np 2 hostname

    "${MPI_ROOT}/bin/mpirun" --map-by ppr:1:node --bind-to core \
        -np 2 ibv_devinfo -l

    UCX_LOG_LEVEL=info "${MPI_ROOT}/bin/mpirun" \
        --map-by ppr:1:node --bind-to core -np 2 ./mpi_bandwidth_test
} 2>&1 | tee "${LOG_FILE}"
