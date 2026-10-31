# Scripts

## Final Reproduction

| Script | Purpose |
|---|---|
| `run_exp008_profile.slurm` | Build exp008, profile its data path, and validate on one node |
| `run_exp009_slurm_fast_3node.slurm` | Build exp009 and validate on three nodes |
| `run_exp006_slurm_fast_2node.slurm` | Build exp006 and validate on two GPU nodes |

## Launchers

| Script | Purpose |
|---|---|
| `melspec_exp006_launcher.sh` | Two-rank Slurm launcher for exp006 |
| `melspec_slurm_launcher.sh` | Generic native-Slurm launcher used by exp004 and exp009 |
| `melspec_mpi_launcher.sh` | MPI functional-path launcher |
| `melspec_mpi_srun_launcher.sh` | MPI launcher variant for troubleshooting |

## Profiling And Tuning

`run_exp001_profile.slurm`, `run_exp004_profile_3node.slurm`,
`cpu_thread_scaling.py`, `benchmark_experiments.py`,
`tune_gpu_heterogeneous.py` 以及 `run_mpi_*` 脚本保留为过程证据和复现工具。

所有应用计算必须通过 Slurm 提交到 `hepnode1~3`。
