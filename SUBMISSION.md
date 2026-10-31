# 提交说明

本仓库完成了 2026 秋季招新题的 melspec 性能优化，并保留 CPU、GPU 和
多节点方案的完整实现、Slurm 日志与实验报告。

## 最终结果

最终成果包含两个独立方案：

- **单节点最终方案**：`exp/008_single_node_cpu_pipelined.cpp`
- **三节点最终方案**：`exp/009_three_node_cpu_pipelined.cpp`

三节点方案是总体最快实现：

| 用例 | 耗时 |
|---|---:|
| default | 0.599 s |
| small_mels | 0.275 s |
| large_fft | 0.661 s |
| narrow_band | 0.651 s |
| 合计 | **2.186 s** |

job 415 在三节点 Slurm 环境通过官方 `4/4 PASS`。性能基线是 exp001 的
64 线程 spread 结果 `3.777 s`，exp009 相对基线的聚合加速为 `1.728x`。

在相同的 1M 样本 default 前缀上，exp008 相对原实现加速 `1427.391x`，
exp009 加速 `156.271x`。该短前缀测试只用于速度对照，完整正确性分别由
job 413 和 job 415 的 `4/4 PASS` 负责证明。job 418 同时验证了默认
`make` 和 `make final-three-node` 构建入口。

机器可读结果见 [results/final-metrics.csv](results/final-metrics.csv)，
相对原实现对照见
[results/relative-to-original.csv](results/relative-to-original.csv)。

## 推荐阅读

| 文件 | 内容 |
|---|---|
| [docs/README.md](docs/README.md) | 文档索引与推荐阅读顺序 |
| [docs/executive-summary.md](docs/executive-summary.md) | 结论、最终指标、关键发现和边界 |
| [docs/experiment-report.md](docs/experiment-report.md) | 001–009 的实现与完整实验记录 |
| [docs/presentation-script.md](docs/presentation-script.md) | 10 页逐页讲稿、汇报节奏与 Q&A 速查 |
| [docs/references-and-ai.md](docs/references-and-ai.md) | 参考资料、AI 工具使用和典型 bug |
| [exp/README.md](exp/README.md) | 实验源码与构建入口 |
| [output/melspec-review-20261031-template.pptx](output/melspec-review-20261031-template.pptx) | 使用学院模板的最终汇报 PPT，共 10 页 |

PPT 已通过 Artifact Tool finalizer 和重新导入渲染检查。由于当前环境没有
Microsoft PowerPoint 自动化，未进行本机 PowerPoint 人工打开验证。

## 仓库结构

```text
SUBMISSION.md                 提交入口
README.md                     原始题目说明，顶部增加提交入口
exp/                          按编号保存的完整实验实现
scripts/                      Slurm 启动、调优和 profiling 脚本
docs/                         实验报告、总结、讲稿和过程记录
logs/                         远程 Slurm 原始输出
results/                      最终结果的机器可读汇总
output/                       最终汇报 PPT
```

`src/main.cpp`、`include/melspec.h`、`validate.py` 和 `melspec.py` 保持题目
要求的接口与语义。

## 快速检查

本地检查题目源文件是否与 HEAD 一致：

```bash
git diff HEAD -- src/main.cpp include/melspec.h validate.py melspec.py
```

默认 `make` 和 `make test` 构建单节点最终方案 exp008：

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
sbatch scripts/run_exp009_slurm_fast_3node.slurm
```

历史 CPU 实验仍可通过 `make EXPERIMENT=<编号与名称>` 构建。GPU 实验必须
使用对应的 `nvcc` 和 Slurm 脚本。

题目要求计算只在 `hepnode1~3` 上通过 Slurm 执行，`hepnode0` 仅用于登录和
作业管理。

## 性能口径

- 所有耗时单位为秒，来自官方四个用例三次运行中的最小值。
- 统一基线为 exp001 的 64 线程 spread 结果，不使用早期 16 线程的
  `7.572 s` 作为优化加速基准。
- 正式正确性门限为 `rtol=1e-3`、`atol=1e-4`。
- 原实现对照统一使用官方 PCM 的前 1,000,000 个样本，避免运行完整原始
  benchmark 的高额耗时。
- GPU 结果受 CUDA 驱动、cuFFT 和共享文件系统影响，不保证与日志逐位一致。

## 关键提交

```text
b71de64 feat: add two-node GPU-only exp006
5549edd feat: add pipelined CPU optimization tracks
```

未提交的 `.codex-build/` 是 PPT 私有构建缓存，不属于评审交付内容。
