# 实验源码

`src/melspec.cpp` 是题目原始参考实现，不会被实验覆盖。九个实验作为完整、
可编译的源码保存在本目录。

## 命名规则

```text
expNNN_short_description.cpp
expNNN_short_description.cu
```

已完成的实验不重写。新的假设使用新的编号，保证源码、日志和性能数字能够对应。

## 实验索引

| ID | 文件 | 主题 |
|---|---|---|
| 001 | `exp001_packed_fft_sparse_mel_simd.cpp` | 打包实数 FFT、稀疏 Mel、分块、SIMD、OpenMP |
| 002 | `exp002_gpu_cufft.cu` | 单节点双 L40、cuFFT D2Z、GPU Mel |
| 003 | `exp003_cpu_gpu_heterogeneous.cu` | 单节点 CPU OpenMP + 双 GPU 异构 |
| 004 | `exp004_mpi_multi_node.cpp` | 多节点 CPU、MPI 功能版与原生 Slurm 性能版 |
| 005 | `exp005_mpi_gpu_heterogeneous.cu` | 多节点 CPU+GPU 异构 |
| 006 | `exp006_multi_node_gpu_only.cu` | 两节点纯 GPU |
| 007 | `exp007_single_node_cpu_profiled.cpp` | exp001 分阶段 profile |
| 008 | `exp008_single_node_cpu_pipelined.cpp` | 单节点最终方案 |
| 009 | `exp009_three_node_cpu_pipelined.cpp` | 三节点最终方案 |

## 构建

原始基准：

```bash
make baseline
```

单节点最终方案：

```bash
make clean
make
```

三节点最终方案：

```bash
make final-three-node
```

指定 CPU 实验：

```bash
make EXPERIMENT=exp001_packed_fft_sparse_mel_simd
```

GPU 实验需要使用集群上的 CUDA 编译器，并通过对应 Slurm 脚本提交，参见
`scripts/README.md`。

## 代码关系

- exp007 和 exp008 复用 exp001 的 FFT 与 Mel 数学。
- exp009 复用 exp008 的单节点数据路径。
- exp009 同时采用 exp004 的本地帧划分、token 协调和分布式写入设计。

完整优化过程见 [docs/optimization-report.md](../docs/optimization-report.md)。
