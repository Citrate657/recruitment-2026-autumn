#!/bin/bash
#SBATCH -J mpi-ompi5-profile
#SBATCH -w hepnode[2-3]
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=00:20:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -euo pipefail

ROOT="/home/wzt/melspec-eval-001"
MPI_ROOT="/cache/spack/opt/spack/linux-ubuntu26.04-icelake/gcc-15.2.0/openmpi-5.0.10-jt2edt5rwr6mvcm3yiaoujlyiuvtsipm"
UCX_ROOT="/cache/spack/opt/spack/linux-ubuntu26.04-icelake/gcc-15.2.0/ucx-1.20.0-tu73yc75zvsqhrlqp2aflhjbaqrbdpen"
LOG_FILE="${ROOT}/logs/mpi_ompi5_profile_slurm-${SLURM_JOB_ID}.log"

cd "${ROOT}"
export OPAL_PREFIX="${MPI_ROOT}"
export PATH="${MPI_ROOT}/bin:${UCX_ROOT}/bin:/usr/local/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="${MPI_ROOT}/lib:${UCX_ROOT}/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=64
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export UCX_NET_DEVICES="${UCX_NET_DEVICES:-mlx5_0:1}"
export OMPI_MCA_pml=ucx
export MELSPEC_ROOT="${ROOT}"
export MELSPEC_MPI_RANKS=2
export MELSPEC_MPI_ROOT="${MPI_ROOT}"
export MELSPEC_UCX_ROOT="${UCX_ROOT}"
export MELSPEC_MPI_BINARY="${ROOT}/melspec_exp004"
export MELSPEC_PROFILE_MPI=1

mkdir -p logs
{
    printf 'OpenMPI 5 profile, job %s\n' "${SLURM_JOB_ID}"
    "${MPI_ROOT}/bin/mpicxx" \
        -std=c++17 -O3 -march=native -fopenmp -I./include \
        -DMELSPEC_DRIVER_MAIN exp/004_mpi_multi_node.cpp \
        -o melspec_exp004

    /usr/bin/time -f 'wall=%e user=%U maxrss=%MKB' \
        ./scripts/melspec_mpi_srun_launcher.sh \
        data/audio.pcm /tmp/ompi5.bin \
        --sr 96000 --n_fft 8192 --hop_length 128 --n_mels 256 \
        --f_min 0 --f_max 22050
} 2>&1 | tee "${LOG_FILE}"
