# 集群脚本

所有正式计算都通过 Slurm 在 `hepnode1~3` 上执行。`hepnode0` 仅用于登录和
作业管理。

## 目录结构

```text
scripts/
  experiments/   按实验编号组织的 sbatch 脚本
  final/         原实现与最终方案速度对照
  launchers/     validate.py 使用的 Slurm/MPI 启动器
  tools/         基准、线程扩展和异构比例调优
  diagnostics/   MPI profile 和传输诊断
```

## 最终复现

| 用途 | 脚本 |
|---|---|
| exp008 单节点 profile 与验证 | `scripts/experiments/exp008_profile.slurm` |
| exp009 三节点正式验证 | `scripts/experiments/exp009_validate_3node.slurm` |
| exp006 两节点 GPU 验证 | `scripts/experiments/exp006_validate_2node.slurm` |
| 原实现与最终方案 1M 对照 | `scripts/final/baseline_speedup.slurm` |
| exp001 官方四用例验证 | `scripts/experiments/exp001_validate.slurm` |
| exp001 线程扩展性 | `scripts/experiments/exp001_cpu_scaling.slurm` |

示例：

```bash
sbatch scripts/experiments/exp009_validate_3node.slurm
sbatch scripts/experiments/exp008_profile.slurm
sbatch scripts/final/baseline_speedup.slurm
```

## 启动器

| 脚本 | 用途 |
|---|---|
| `scripts/launchers/slurm_native.sh` | 原生 Slurm rank 协调，exp004/exp009 使用 |
| `scripts/launchers/slurm_gpu_only.sh` | 两节点纯 GPU 启动器，exp006 使用 |
| `scripts/launchers/mpi_mpirun.sh` | `mpirun` MPI 功能路径 |
| `scripts/launchers/mpi_srun.sh` | Slurm 内 `srun --mpi` 诊断路径 |

## 工具脚本

| 脚本 | 用途 |
|---|---|
| `scripts/tools/benchmark_baseline.py` | 原实现与实验实现同输入测速 |
| `scripts/tools/benchmark_cpu_scaling.py` | CPU 线程数和绑核扫描 |
| `scripts/tools/tune_cpu_gpu_ratio.py` | CPU 帧比例与线程数联合调优 |

## 诊断脚本

`scripts/diagnostics/` 保留 MPI profile、OpenMPI/UCX 传输测试和带宽测量，
用于解释多节点路径为何选择原生 Slurm 和共享文件写入。

完整结果和日志对应关系见
[docs/optimization-report.md](../docs/optimization-report.md)。
