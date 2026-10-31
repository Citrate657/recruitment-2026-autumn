# 提交说明

本仓库完成了 2026 秋季招新题 Melspec 的性能优化，并保留九个实验、官方验证
日志、最终源码、汇报 PPT 和完整技术报告。

## 最终结果

| 方案 | 实现 | 四用例合计 | 相对 exp001 | 正确性 |
|---|---|---:|---:|---|
| 单节点最终 | `experiments/exp008_single_node_cpu_pipelined.cpp` | `2.733 s` | `1.382x` | `4/4 PASS` |
| 三节点最终 | `experiments/exp009_three_node_cpu_pipelined.cpp` | **`2.186 s`** | **`1.728x`** | `4/4 PASS` |

三节点 exp009 是总体最快方案。性能基线为 exp001 的 64 线程 spread 结果
`3.777 s`。

同一 1M 样本 default 前缀上，exp008 相对原实现为 `1427.391x`，exp009 为
`156.271x`。短前缀只用于速度对照，完整正确性由官方四用例负责。

机器可读结果：

- [results/final-metrics.csv](results/final-metrics.csv)
- [results/relative-to-original.csv](results/relative-to-original.csv)

## 推荐阅读

| 文件 | 内容 |
|---|---|
| [docs/README.md](docs/README.md) | 文档索引和推荐阅读顺序 |
| [docs/melspec-performance-review.pptx](docs/melspec-performance-review.pptx) | 10 页最终汇报 PPT |
| [docs/presentation-script.md](docs/presentation-script.md) | 与 PPT 对应的逐页讲稿、HPC 知识、优化推导和 Q&A |
| [docs/optimization-report.md](docs/optimization-report.md) | 合并后的完整技术报告 |
| [docs/reference/melspec-basics.md](docs/reference/melspec-basics.md) | PCM 到 Mel 输出的算法基础 |
| [experiments/README.md](experiments/README.md) | 九个实验源码和构建入口 |
| [scripts/README.md](scripts/README.md) | 最终验证、profile、调优和诊断脚本 |

## 推荐阅读顺序

1. 本文件：确认最终结果和交付物。
2. `docs/melspec-performance-review.pptx`：快速理解汇报主线。
3. `docs/presentation-script.md`：了解每页背后的优化思路和 HPC 知识。
4. `docs/optimization-report.md`：查看完整推导、实验数据和正确性证据。

## 仓库结构

```text
SUBMISSION.md                 提交入口
README.md                     原始题目说明
experiments/                  exp001–009 完整实验源码
scripts/
  experiments/                按实验编号组织的 Slurm 运行脚本
  launchers/                  原生 Slurm 和 MPI 启动器
  tools/                      基准、线程扩展和异构调优工具
  diagnostics/                MPI 和传输诊断
  final/                      原实现与最终方案速度对照
docs/
  README.md                   文档索引
  melspec-performance-review.pptx
  presentation-script.md      逐页讲稿和 Q&A
  optimization-report.md      完整技术报告
  reference/                  算法基础
  archive/                    开发计划和过程日志
logs/                         官方 Slurm 原始输出
results/                      机器可读结果
```

## 快速检查

默认构建单节点最终方案 exp008：

```bash
make clean
make -j64
OMP_NUM_THREADS=64 \
OMP_PROC_BIND=spread \
OMP_PLACES=cores \
python3 validate.py ./melspec --runs 3
```

构建三节点最终方案：

```bash
make final-three-node
```

在集群上验证三节点最终方案：

```bash
sbatch scripts/experiments/exp009_validate_3node.slurm
```

构建其他 CPU 实验：

```bash
make EXPERIMENT=exp001_packed_fft_sparse_mel_simd
```

GPU 和 MPI 实验使用 `experiments/` 中对应源码及 `scripts/experiments/`、
`scripts/launchers/` 中的专用脚本。

## 性能口径

- 所有耗时单位为秒，来自官方四个用例三次运行的最小值。
- 统一基线为 exp001 的 64 线程 spread 结果 `3.777 s`。
- 正式正确性门限为 `rtol=1e-3`、`atol=1e-4`。
- 原实现对照使用官方 PCM 的前 1,000,000 个样本。
- GPU 结果受 CUDA 驱动、cuFFT 和共享文件系统影响。
- 正式计算只在 `hepnode1~3` 上通过 Slurm 执行，`hepnode0` 仅用于登录和作业
  管理。

## 工具披露

使用 Codex 外接 DeepSeek-V4.1-Flash-1017，`reasoning_effort=max`，辅助源码
理解、方案讨论、实现草稿、实验编排和文档整理。最终代码与结论由本人理解并
复核，正式数据全部来自实际 Slurm 作业和保留日志。

## 相关提交

```text
b71de64 feat: add two-node GPU-only exp006
5549edd feat: add pipelined CPU optimization tracks
573cc6f chore: organize experiment sources and scripts
```
