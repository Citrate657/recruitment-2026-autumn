#!/bin/bash
#SBATCH -J melspec-mpi-profile
#SBATCH -w hepnode[2-3]
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:L40:2
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -euo pipefail

ROOT="/home/wzt/melspec-eval-001"
MPI_ROOT="/cache/toolchain/openmpi-4.1.6"
UCX_ROOT="/cache/toolchain/ucx-1.18.1"
CUDA_ROOT="/cache/spack/opt/spack/linux-ubuntu26.04-icelake/none-none/cuda-13.2.1-3oevt45te4n4wckbepliz3hkdfoj3uou"
CUDA_LIB="${CUDA_ROOT}/targets/x86_64-linux/lib"
LOG_FILE="${ROOT}/logs/mpi_profile_slurm-${SLURM_JOB_ID}.log"

cd "${ROOT}"
export OPAL_PREFIX="${MPI_ROOT}"
export PATH="${CUDA_ROOT}/bin:${MPI_ROOT}/bin:${UCX_ROOT}/bin:/usr/local/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="${CUDA_LIB}:${MPI_ROOT}/lib:${UCX_ROOT}/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=64
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export UCX_NET_DEVICES="${UCX_NET_DEVICES:-mlx5_0:1}"
export OMPI_MCA_pml="${OMPI_MCA_pml:-ucx}"
export MELSPEC_ROOT="${ROOT}"
export MELSPEC_MPI_RANKS=2
export MELSPEC_PROFILE_MPI=1

mkdir -p logs
{
    printf 'MPI profile, job %s\n' "${SLURM_JOB_ID}"

    "${MPI_ROOT}/bin/mpicxx" \
        -std=c++17 -O3 -march=native -fopenmp -I./include \
        src/main.cpp experiments/exp004_mpi_multi_node.cpp -o melspec_exp004

    "${MPI_ROOT}/bin/mpicxx" -std=c++17 -O3 -march=native -fopenmp \
        -I./include -c src/main.cpp -o main_exp005.o
    "${CUDA_ROOT}/bin/nvcc" -std=c++17 -O3 -arch=sm_89 \
        -Xcompiler -fopenmp -I./include -I"${MPI_ROOT}/include" \
        -c experiments/exp005_mpi_gpu_heterogeneous.cu -o exp005_gpu.o
    "${MPI_ROOT}/bin/mpicxx" -fopenmp \
        main_exp005.o exp005_gpu.o \
        -L"${CUDA_LIB}" -lcufft -lcudart -o melspec_exp005

    COMMON_ARGS="data/audio.pcm /tmp/mpi_profile.bin --sr 96000 --n_fft 8192 --hop_length 128 --n_mels 256 --f_min 0 --f_max 22050"

    printf '\n[004 compute+gather profile]\n'
    export MELSPEC_MPI_BINARY="${ROOT}/melspec_exp004"
    /usr/bin/time -f 'wall=%e user=%U maxrss=%MKB' \
        ./scripts/launchers/mpi_mpirun.sh ${COMMON_ARGS}

    printf '\n[005 GPU-only profile]\n'
    export MELSPEC_MPI_BINARY="${ROOT}/melspec_exp005"
    export MELSPEC_CPU_PERCENT=0
    /usr/bin/time -f 'wall=%e user=%U maxrss=%MKB' \
        ./scripts/launchers/mpi_mpirun.sh ${COMMON_ARGS}

    printf '\n[005 heterogeneous 12%% profile]\n'
    export MELSPEC_CPU_PERCENT=12
    /usr/bin/time -f 'wall=%e user=%U maxrss=%MKB' \
        ./scripts/launchers/mpi_mpirun.sh ${COMMON_ARGS}
} 2>&1 | tee "${LOG_FILE}"
