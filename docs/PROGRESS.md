# Long-Running Task Progress

> 本文件是开发过程记录，保留成功、失败和决策轨迹。提交评审的权威结论与
> 最终指标以 `SUBMISSION.md` 和 `docs/experiment-report.md` 为准。

## 当前目标

优化工作已按用户决定停止。当前只进行结果汇总、文档完善、PPT 制作和汇报
内容准备。最终状态以 exp009 的 `2.186 s` 和 exp008 的 `2.733 s` 为准。

## 已完成

- **001 全长度复数 FFT、稠密 Mel 和缓存问题已优化并远程通过。**
  - 验证命令（远程 Slurm 作业 187）：`python validate.py ./melspec_exp001 --runs 3`
  - 结果：`4/4 PASSED`；default 1.509 s，small_mels 1.128 s，large_fft 2.550 s，narrow_band 2.385 s。
  - 日志：`logs/001_packed_fft_sparse_mel_simd_slurm-187.log`

- **001 相对原始实现的 1M 样本加速比已实测。**
  - 命令：`scripts/benchmark_experiments.py --cases default --runs 1 --max-samples 1000000`
  - 结果：baseline 61.662986 s，exp001 0.084331 s，731.206x。
  - 硬件：`hepnode2`，16 OpenMP 线程。

- **CPU 线程数与绑核策略已实测。**
  - 命令（远程 Slurm 作业 275）：`scripts/run_cpu_scaling.slurm`
  - 结果：64 线程 spread 最优，四用例总耗时 3.776826 s；16 线程 spread 为 7.415062 s；32 线程 spread 为 5.248376 s。
  - 日志：`logs/cpu-thread-scaling_slurm-275.log`

- **实验目录与基准源码已重建并编译验证。**
  - 命令：`make clean && make -j16` 与 `make EXPERIMENT=001_packed_fft_sparse_mel_simd -j16`
  - 结果：两种构建均成功；`src/melspec.cpp` 与 HEAD 内容一致。
  - Commit：`93d3548 feat: add numbered melspec experiments`

- **远程 CUDA/MPI 工具链已确认存在。**
  - 只读检查：CUDA 13.2.1 位于 `/cache/spack/opt/spack/linux-ubuntu26.04-icelake/none-none/cuda-13.2.1-3oevt45te4n4wckbepliz3hkdfoj3uou`；OpenMPI 4.1.6 位于 `/cache/toolchain/openmpi-4.1.6`；UCX 位于 `/cache/toolchain/ucx-1.18.1`；IB 为 56 Gb/s FDR。

- **002 单节点双 GPU cuFFT 已通过官方四用例。**
  - 验证命令（远程 Slurm 作业 292）：`python validate.py ./melspec_exp002 --runs 3`
  - 结果：`4/4 PASSED`；default 1.341 s，small_mels 0.925 s，large_fft 1.444 s，narrow_band 1.454 s。
  - 相对 exp001 16-thread baseline 的加速：default 1.125x，small_mels 1.219x，large_fft 1.766x，narrow_band 1.640x。
  - 日志：`logs/002_gpu_cufft_slurm-292.log`

- **003 单节点 CPU+双 GPU 异构已通过官方四用例。**
  - 验证命令（远程 Slurm 作业 293）：`python validate.py ./melspec_exp003 --runs 3`
  - 配置：64 OpenMP 线程 + 2×L40，`MELSPEC_CPU_PERCENT=12`。
  - 结果：`4/4 PASSED`；default 1.355 s，small_mels 0.969 s，large_fft 1.524 s，narrow_band 1.523 s。
  - 每个用例都仍快于 exp001 的 16-thread baseline。
  - 日志：`logs/003_cpu_gpu_heterogeneous_slurm-293.log`

- **003 最终 4% CPU 分担配置已通过官方四用例。**
  - 验证命令（远程 Slurm 作业 347）：`python validate.py ./melspec_exp003 --runs 3`
  - 结果：`4/4 PASSED`；default 1.353 s，small_mels 0.929 s，large_fft 1.474 s，narrow_band 1.451 s。
  - 四个用例均快于 exp001；相比 12% 配置也有小幅改善。
  - 日志：`logs/003_cpu_gpu_heterogeneous_slurm-347.log`

- **003 CPU 帧比例已完成首轮调参。**
  - 验证/调参命令（远程 Slurm 作业 299）：`scripts/tune_gpu_heterogeneous.py --percentages 0,2,4,6,8,10,12 --threads 64 --runs 2`
  - 结果：0% 总耗时 5.102446 s，2% 为 5.178589 s，12% 为 5.284117 s；纯 GPU 最快。
  - 日志：`logs/003_cpu_gpu_tuning_slurm-299.log`

- **003 CPU 线程数 × 帧比例第二轮调参已完成。**
  - 验证命令（远程 Slurm 作业 301）：`8/16/32/64 threads × 4/8/12% CPU frames`
  - 结果：最好的 CPU+GPU 组合是 64 threads + 4%，四用例总计 5.207414 s；仍慢于同作业纯 GPU 的约 5.10 s。
  - 依据：降低 CPU 线程数没有消除竞争，反而降低 CPU 分担能力。
  - 日志：`logs/003_cpu_thread_tuning_slurm-301.log`

- **004 两节点 MPI 功能调试已通过正确性。**
  - 验证命令（远程 Slurm 作业 302）：`python validate.py ./scripts/melspec_mpi_launcher.sh --runs 1`
  - 配置：2 nodes × 1 rank × 32 OpenMP threads。
  - 结果：`4/4 PASSED`；default 2.474 s，small_mels 1.771 s，large_fft 2.914 s，narrow_band 2.945 s。
  - 当前版本比 exp001 慢，只证明 MPI 正确性，尚未达到性能目标。
  - 日志：`logs/004_mpi_2node_debug_slurm-302.log`

- **005 两节点 MPI+双 GPU 功能调试已通过正确性。**
  - 验证命令（远程 Slurm 作业 303）：`python validate.py ./scripts/melspec_mpi_launcher.sh --runs 1`
  - 配置：2 nodes × 1 rank × 64 CPU threads × 2 L40 per node，CPU 帧比例 12%。
  - 结果：`4/4 PASSED`；default 3.013 s，small_mels 1.993 s，large_fft 3.020 s，narrow_band 3.164 s。
  - 当前版本比 exp001 慢，功能通过但性能目标未达成。
  - 日志：`logs/005_mpi_gpu_2node_debug_slurm-303.log`

- **005 原生 Slurm 单节点 driver 已通过官方四用例。**
  - 验证命令（远程 Slurm 作业 328）：`python validate.py ./scripts/melspec_slurm_launcher.sh --runs 3`
  - 配置：1 node ×64 CPU threads ×2 L40，CPU 帧比例 2%。
  - 结果：`4/4 PASSED`；default 1.662 s，small_mels 1.038 s，large_fft 1.778 s，narrow_band 1.782 s。
  - 聚合 6.260 s，低于 exp001 的 7.572 s；default 单独仍慢于 exp001。
  - 日志：`logs/005_slurm_fast_1node_slurm-328.log`

- **005 两节点双 GPU/节点异构版本已通过官方四用例并达到性能目标。**
  - 验证命令（远程 Slurm 作业 352）：`python validate.py ./scripts/melspec_slurm_launcher.sh --runs 3`
  - 配置：2 nodes ×1 rank ×64 OpenMP threads ×2 L40/node；CPU 帧比例 2%。
  - 结果：`4/4 PASSED`；default 1.376 s，small_mels 0.904 s，large_fft 1.454 s，narrow_band 1.445 s。
  - 相比 exp001 的 1.509/1.128/2.550/2.385 s，四个用例全部更快。
  - 日志：`logs/005_slurm_fast_2node_slurm-352.log`

- **006 两节点纯 GPU 版本已通过官方四用例并达到全部单例性能目标。**
  - 验证命令（远程 Slurm 作业 410）：`python validate.py ./scripts/melspec_exp006_launcher.sh --runs 3`
  - 配置：2 nodes ×1 rank ×2 L40/node，4 CPU/rank；CPU 仅负责 I/O 和 GPU 编排，`MELSPEC_CPU_PERCENT=0`。
  - 实现：每 rank 只读取自己的 PCM 区间；节点内双 GPU 并行完成 Hann、cuFFT、功率谱、稀疏 Mel 和 float32 转换。
  - 结果：`4/4 PASSED`；default 1.068 s，small_mels 0.729 s，large_fft 1.113 s，narrow_band 1.114 s。
  - 相比 exp001 的 1.509/1.128/2.550/2.385 s，四个用例全部更快。
  - 源码和复现脚本 Commit：`b71de64 feat: add two-node GPU-only exp006`。
  - 日志：`logs/006_multi_node_gpu_only_slurm-410.log`

- **004 三节点 CPU 已通过官方四用例，成为阶段性总体最快方案。**
  - Slurm 作业 359，节点 `hepnode1~3`，每节点 64 OpenMP 线程。
  - 结果：`4/4 PASSED`；default 0.693 s，small_mels 0.305 s，large_fft 0.773 s，narrow_band 0.762 s，合计 2.533 s。
  - 相较两节点 CPU 的 3.083 s 快 1.217x；相较 64 线程 exp001 的 3.777 s 快 1.491x。
  - 日志：`logs/004_slurm_fast_3node_slurm-359.log`

- **单节点分阶段 profiling 已完成。**
  - job 411：完整写盘约 0.97–0.99 s，`/dev/null` 约 0.90–0.91 s。
  - job 412：default 拆分为读取/归一化约 0.20 s、计算约 0.50 s、转换/写入约 0.24 s。
  - 峰值 RSS 约 779 MB，主要优化点是串行归一化、完整 double 输出和最终转换。
  - job 411 末尾 `perf stat` 因 `perf_event_paranoid=4` 无权限而退出；此前所有计时有效。job 412 完整通过。

- **008 单节点 CPU 流水化已通过官方四用例。**
  - job 413：`4/4 PASSED`；default 0.629 s，small_mels 0.470 s，large_fft 0.827 s，narrow_band 0.807 s，合计 2.733 s。
  - 相较 exp001 T=64 快 1.382x；峰值 RSS 降至约 394 MB。
  - 优化：并行归一化、单 OpenMP 帧区域、直接 float32 输出，删除完整 double 输出和 tile 中间矩阵。
  - 日志：`logs/008_single_node_cpu_slurm-413.log`

- **009 三节点 CPU 流水化已通过官方四用例，成为当前总体最快方案。**
  - job 415：`4/4 PASSED`；default 0.599 s，small_mels 0.275 s，large_fft 0.661 s，narrow_band 0.651 s，合计 2.186 s。
  - 相较三节点 exp004 的 2.533 s 快 1.159x；相较 exp001 T=64 快 1.728x。
  - 优化：本地 PCM 区间读取、并行归一化、单 OpenMP 直接 float32 计算、token 协调和分布写入。
  - 源码与复现脚本 Commit：`5549edd feat: add pipelined CPU optimization tracks`。
  - 日志：`logs/009_three_node_cpu_slurm-415.log`

- **总结与汇报材料已完成。**
  - `docs/EXECUTIVE-SUMMARY.md`：完整结果、关键发现、可复现性和边界。
  - `docs/presentation-script.md`：11 页逐页讲稿与常见问答。
  - `output/melspec-review-20261031.pptx`：11 页可编辑 PPT，含 speaker notes。
  - finalizer 通过：11 页、4 个原生图表、2 个原生表格，字体与布局 finding 为 0。

- **004 分布写入 + 局部 PCM 读取已通过两节点正确性验证。**
  - 验证命令（远程 Slurm 作业 317）：`python validate.py ./scripts/melspec_mpi_launcher.sh --runs 1`
  - 配置：2 ranks ×32 threads，每 rank 只读自己帧所需的 PCM 区间，各 rank 直接 pwrite float32 输出。
  - 结果：`4/4 PASSED`；default 2.092 s，small_mels 1.666 s，large_fft 2.375 s，narrow_band 2.292 s。
  - 相对旧 Gather 版明显改善，但仍未全部快于 exp001。
  - 日志：`logs/004_mpi_2node_debug_slurm-317.log`

- **004 原生 Slurm 多进程后端已通过官方四用例并达到性能目标。**
  - 验证命令（远程 Slurm 作业 338）：`python validate.py ./scripts/melspec_slurm_launcher.sh --runs 3`
  - 编译：`g++ ... -DMELSPEC_DRIVER_MAIN -DMELSPEC_SLURM_MULTI exp/004_mpi_multi_node.cpp`
  - 配置：2 nodes ×64 OpenMP threads；每 rank 只读本区间 PCM；token 协调；float32 pwrite 输出。
  - 结果：`4/4 PASSED`；default 0.818 s，small_mels 0.398 s，large_fft 0.941 s，narrow_band 0.926 s。
  - 相比 exp001 的 1.509/1.128/2.550/2.385 s，四个用例全部更快。
  - 日志：`logs/004_slurm_fast_slurm-338.log`

- **GPU 与多节点实验源码/复现脚本已提交。**
  - 验证命令：`git diff --cached --check && git commit -m "feat: add GPU and multi-node melspec experiments" && git show --stat --oneline --summary HEAD`
  - 结果：commit `21d7669`，31 files changed，4687 insertions；不含 `docs/`、`logs/`、`.vscode/` 或构建产物。
  - 验证命令：`git status --short --branch`
  - 当时的提交检查结果：无未提交的非忽略文件；`master` 相对 `origin/master` 为 `ahead 2`。

## 进行中

- 无。优化工作、文档总结、PPT 和汇报内容均已完成。

## 失败记录

- **Conda 默认渠道首次创建失败**：未接受 Anaconda Terms of Service。用户随后明确接受；已创建 `/home/wzt/.venvs/melspec`，NumPy 2.5.4、SciPy 1.18.1、librosa 1.0.0。
- **远程 SSH 新执行环境首次失败**：未显式指定身份文件时认证失败。后续连接统一要求显式身份文件和 `IdentitiesOnly=yes`。
- **GPU 作业 288/289 编译失败**：现象为 `nvcc fatal: Unknown option '-Wl,-rpath,...'`。推测 NVCC 把该参数当作自身选项；已将脚本改为依赖 `LD_LIBRARY_PATH` 指定 CUDA 库，并重新提交 292/293。修复后 292 已通过 4/4；293 继续验证。
- **三节点作业 285/290 暂时阻塞**：不是程序失败；集群只有 3 台计算节点，`hepnode1` 被长作业占用，必须等待三个节点同时空闲。
- **旧作业 183 使用 UNLIMITED 导致 backfill 估计很差**：后续显式设置 `--time`，但多节点仍受独占节点限制。
- **003 调参作业 298 失败**：`tune_gpu_heterogeneous.py` 把 case 的内部字段 `name` 加进参数并传给程序，生成不存在的 `--name`。已修正为单独传 `case_name`，不修改 CLI 参数。
- **srun --mpi=pmi2 失败**：作业308报 OpenMPI 未使用 Slurm PMI2 构建，MPI_Init 失败；改回 PMIx，并最终采用原生 Slurm 多进程后端。
- **OpenMPI 5 UCX 不可达**：作业324中 UCX 只识别 `ibs25(tcp)`，未识别 `mlx5_0:1`，跨节点 endpoint 创建失败；未采用 OMPI5。
- **共享 ready 文件竞态**：固定输出路径重复运行时，`/home` 目录项和 mark 缓存导致 rank1 超时或抢跑。最终协议为 launcher 预创建固定输出/ready 文件，rank0 写入本轮 64-bit token，其他 rank `pread` token 后 pwrite；作业338稳定验证。

## 关键决策

- **最终取舍**：当前最快方案是三节点 CPU 流水化 exp009，四用例合计 2.186 s。GPU 路线的计算阶段很快，但完整 CLI 的搬运、同步和输出开销更大；CPU+GPU 异构也没有超过相应纯 GPU 配置。
- 001 作为后续优化比较基准；新实验报告必须计算 `T_exp001 / T_new`，不能继续拿原始朴素实现当分母。
- CPU 全节点优先使用 64 线程、`OMP_PROC_BIND=spread`、`OMP_PLACES=cores`；若 Slurm 只给 N 个 CPU，则 OpenMP 线程不超过 N。
- 002 先用 double 精度 `CUFFT_D2Z`，避免 float 在接近零的功率和隐藏数据上冒险；通过后再考虑 float。
- GPU 帧分块参数初始为 1024；双 GPU 各控制独立 CUDA context/stream 和连续帧区间。
- CPU+GPU 异构先以 12% 帧给 CPU，后续按实测吞吐调整 `MELSPEC_CPU_PERCENT`。
- MPI 按连续帧区间划分；每 rank 产生 `[n_mels, local_frames]`，最终用 `MPI_Gatherv` 交给 rank 0。
- `main.cpp` 不可修改，因此 MPI 非零 rank 在归并后必须 `MPI_Finalize()` 并 `_Exit(0)`，避免多个 rank 同时执行 `write_output`。
- 多节点启动使用包装器 `scripts/melspec_mpi_launcher.sh`，通过 Slurm allocation 内的 `mpirun` 启动 3 个 rank。
- 多节点最终性能路径使用 `scripts/melspec_slurm_launcher.sh` 和 `MELSPEC_SLURM_MULTI`：绕过 MPI_Init，用 Slurm rank、局部 PCM 读取、float32 pwrite 和 64-bit token 协调。MPI 功能路径保留作正确性和算法对照。
- exp006 使用独立纯 GPU 源码和 launcher；每个 rank 只读取本地连续帧所需的 PCM，节点内两个 GPU 完成全部逐帧计算，CPU 不保留 FFT/功率/Mel 帧分支。
- 后续方案分为单节点 CPU 与多节点 CPU 两条独立路线；统一使用 exp001 64 线程 spread 结果作为性能基线。
- 003 在 64 CPU 线程参与时，CPU 分担 2%~12% 都会轻微拖慢纯 GPU；下一步降低 CPU 线程数，检查是否由过度并行导致内存带宽竞争。
- 三节点正式作业在 `hepnode1` 长期忙碌时会占住调度优先级，导致两节点调试作业无法 backfill；暂时取消三节点 pending job，先完成 2-node 功能调试，再恢复三节点正式运行。

## 性能数据

| 日期/作业 | 平台 | 配置 | 指标 | 结果 |
|---|---|---|---|---|
| 2026-10-30 / 187 | `hepnode2` | exp001，16 OpenMP 线程 | 官方四用例 | 1.509 / 1.128 / 2.550 / 2.385 s，4/4 PASS |
| 2026-10-30 / 187 | `hepnode2` | 1M 官方 PCM 前缀 | exp001 vs baseline | 0.084331 vs 61.662986 s；731.206x |
| 2026-10-30 / 275 | `hepnode2` | 64 核，1~64 线程 | 四用例总耗时 | 1/16/32/64 spread：79.120 / 7.415 / 5.248 / 3.777 s |
| 2026-10-31 / 288,289 | `hepnode2`, `hepnode3` | 双 L40，CUDA 13.2 | 首轮构建 | FAIL：NVCC `-Wl,-rpath` 参数错误 |
| 2026-10-31 / 292 | `hepnode2` | exp002，2×L40，`CUFFT_D2Z` | 官方四用例 | 1.341 / 0.925 / 1.444 / 1.454 s，4/4 PASS |
| 2026-10-31 / 293 | `hepnode3` | exp003，64 CPU +2×L40，CPU 12% | 官方四用例 | 1.355 / 0.969 / 1.524 / 1.523 s，4/4 PASS |
| 2026-10-31 / 299 | `hepnode2` | exp003，64 CPU 线程，CPU 0~12% | 四用例总耗时 | 0% 5.102446 s；2% 5.178589 s；12% 5.284117 s |
| 2026-10-31 / 301 | `hepnode2` | exp003，8/16/32/64 线程 ×4/8/12% | 四用例总耗时 | 最优异构 64 线程+4%：5.207414 s；纯 GPU 仍约 5.10 s |
| 2026-10-31 / 347 | `hepnode3` | exp003 最终配置，64 线程 +2 L40，CPU 4% | 官方四用例 | 1.353 / 0.929 / 1.474 / 1.451 s，4/4 PASS |
| 2026-10-31 / 302 | `hepnode[2-3]` | exp004，2 ranks ×32 threads | 官方四用例，1 run | 2.474 / 1.771 / 2.914 / 2.945 s，4/4 PASS，慢于 exp001 |
| 2026-10-31 / 303 | `hepnode[2-3]` | exp005，2 ranks ×64 threads ×2 L40 | 官方四用例，1 run | 3.013 / 1.993 / 3.020 / 3.164 s，4/4 PASS，慢于 exp001 |
| 2026-10-31 / 317 | `hepnode[2-3]` | exp004 driver，2 ranks ×32 threads | 官方四用例，1 run | 2.092 / 1.666 / 2.375 / 2.292 s，4/4 PASS，仍慢于 exp001 |
| 2026-10-31 / 338 | `hepnode[2-3]` | exp004 native Slurm，2 ranks ×64 threads | 官方四用例 | 0.818 / 0.398 / 0.941 / 0.926 s，4/4 PASS，全部快于 exp001 |
| 2026-10-31 / 328 | `hepnode2` | exp005 native Slurm，1 rank ×64 threads ×2 L40 | 官方四用例 | 1.662 / 1.038 / 1.778 / 1.782 s，4/4 PASS；聚合快于 exp001 |
| 2026-10-31 / 352 | `hepnode[2-3]` | exp005 native Slurm，2 ranks ×64 threads ×2 L40/node | 官方四用例 | 1.376 / 0.904 / 1.454 / 1.445 s，4/4 PASS，全部快于 exp001 |
| 2026-10-31 / 410 | `hepnode[1-2]` | exp006 native Slurm，2 ranks ×2 L40/node，CPU 0% 帧 | 官方四用例 | **1.068 / 0.729 / 1.113 / 1.114 s，4/4 PASS，全部快于 exp001** |
| 2026-10-31 / 359 | `hepnode[1-3]` | exp004 native Slurm，3 ranks ×64 threads | 官方四用例 | **0.693 / 0.305 / 0.773 / 0.762 s，4/4 PASS，合计 2.533 s，当时最快** |
| 2026-10-31 / 411 | `hepnode1` | exp001 单节点 64 threads | 完整写盘 vs `/dev/null` | 0.97–0.99 / 0.90–0.91 s |
| 2026-10-31 / 412 | `hepnode1` | exp007 分阶段 driver | default profile | read+normalize 0.20 / compute 0.50 / convert+write 0.24 s |
| 2026-10-31 / 413 | `hepnode2` | exp008 单节点 CPU 流水化 | 官方四用例 | **0.629 / 0.470 / 0.827 / 0.807 s，4/4 PASS，合计 2.733 s** |
| 2026-10-31 / 414 | `hepnode[1-3]` | exp004 三节点 profile | default profile | read ~0.065 / compute ~0.17 / write ~0.28 s，wall 0.69–0.88 s |
| 2026-10-31 / 415 | `hepnode[1-3]` | exp009 三节点 CPU 流水化 | 官方四用例 | **0.599 / 0.275 / 0.661 / 0.651 s，4/4 PASS，合计 2.186 s，当前最快** |
