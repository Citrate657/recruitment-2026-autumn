# Experiment Implementations

`src/melspec.cpp` is the untouched reference implementation from the original
repository. Experimental implementations live here as complete, compilable
translation units rather than patches.

## Naming

Use a monotonically increasing prefix and a short change description:

```text
NNN_short_description.cpp
```

Examples:

```text
001_packed_fft_sparse_mel_simd.cpp
002_avx512_fft.cpp
003_cuda_cufft.cpp
```

Do not rewrite an experiment after it has been benchmarked and logged. Create
the next numbered file instead, so source and measurements stay traceable.

## Building

Build the reference implementation:

```bash
make
```

Build a numbered experiment:

```bash
make EXPERIMENT=001_packed_fft_sparse_mel_simd
```

Run the validation target with the same selector:

```bash
make EXPERIMENT=001_packed_fft_sparse_mel_simd test
```

Experiments 002–006 require remote CUDA/MPI libraries and have reproducible
Slurm launchers:

```bash
sbatch scripts/run_exp002.slurm
sbatch scripts/run_exp003.slurm
sbatch scripts/run_exp004_slurm_fast.slurm
sbatch scripts/run_exp005_slurm_fast_2node.slurm
sbatch scripts/run_exp006_slurm_fast_2node.slurm
```

The `*_fast` launchers use native Slurm rank coordination and are the fastest
validated multi-node variants.

The executable is always `./melspec`, so the selected implementation should be
recorded together with each benchmark log.

## Current Experiments

| ID | File | Summary |
|---|---|---|
| 001 | `001_packed_fft_sparse_mel_simd.cpp` | Precomputed packed real FFT, sparse Mel projection, tiled memory access, SIMD and OpenMP |
| 002 | `002_gpu_cufft.cu` | Single-node dual-L40 execution with double-precision cuFFT and GPU Mel projection |
| 003 | `003_cpu_gpu_heterogeneous.cu` | Concurrent CPU OpenMP and dual-GPU execution on one node |
| 004 | `004_mpi_multi_node.cpp` | Multi-node CPU implementation with an MPI path and a faster native Slurm rank backend |
| 005 | `005_mpi_gpu_heterogeneous.cu` | Multi-node CPU + dual-GPU heterogeneous implementation with both MPI and native Slurm backends |
| 006 | `006_multi_node_gpu_only.cu` | Two-node GPU-only implementation with local PCM range reads and two L40 GPUs per node |
