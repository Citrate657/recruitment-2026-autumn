# Melspec 实验报告

本文档记录每个编号实验的优化思路、代码实现、正确性验证和性能结果。

> 评测原则：正式计算只允许通过 Slurm 在 `hepnode1~3` 上执行。`hepnode0` 是登录节点，禁止运行程序或基准，即使 CPU 空闲也不能例外。

最终提交选择两个方案：单节点 exp008（`2.733 s`）和三节点 exp009
（`2.186 s`）。其余实验保留为路线探索、性能对照和 bug 证据。

## 1. 索引

| 实验 | 实现文件 | 主题 | 状态 |
|---|---|---|---|
| 基准 | `src/melspec.cpp` | 原始参考实现 | 已建立 |
| 001 | `exp/001_packed_fft_sparse_mel_simd.cpp` | 单节点 CPU，打包实时 FFT、稀疏 Mel、分块、SIMD、OpenMP | 远程验证通过 |
| 002 | `exp/002_gpu_cufft.cu` | 单节点双 L40、double cuFFT、GPU Mel 投影 | 远程验证通过 |
| 003 | `exp/003_cpu_gpu_heterogeneous.cu` | 单节点 CPU OpenMP + 双 GPU 异构 | 远程验证通过 |
| 004 | `exp/004_mpi_multi_node.cpp` | 多节点 CPU，MPI 功能版 + 原生 Slurm 性能版 | 三节点 4/4 PASS，全部快于 001 |
| 005 | `exp/005_mpi_gpu_heterogeneous.cu` | 多节点 GPU/CPU，MPI 功能版 + 原生 Slurm 性能版 | 两节点 4/4 PASS，全部快于 001 |
| 006 | `exp/006_multi_node_gpu_only.cu` | 两节点纯 GPU，每节点 2×L40 | 两节点 4/4 PASS，全部快于 001 |
| 007 | `exp/007_single_node_cpu_profiled.cpp` | exp001 分阶段 profiling driver | Slurm 阶段测量完成 |
| 008 | `exp/008_single_node_cpu_pipelined.cpp` | 单节点并行归一化、直接 float32 输出、单 OpenMP 区域 | 4/4 PASS，单节点新最快 |
| 009 | `exp/009_three_node_cpu_pipelined.cpp` | 三节点原生 Slurm + exp008 直接 float32 数据路径 | 4/4 PASS，当前总体最快 |

### 1.1 最终结果

| 实验 | 作业 | 部署配置 | default | small_mels | large_fft | narrow_band | 四用例合计 | 聚合加速 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| 001 CPU | 275 | 1 节点，64 CPU 线程，spread 绑核 | 0.931 | 0.569 | 1.155 | 1.122 | 3.777 | 1.000x |
| 002 单节点双 GPU | 292 | 1 节点，2×L40 | 1.341 | 0.925 | 1.444 | 1.454 | 5.164 | 0.731x |
| 003 CPU+双 GPU | 347 | 1 节点，64 CPU 线程 + 2×L40，CPU 帧 4% | 1.353 | 0.929 | 1.474 | 1.451 | 5.207 | 0.725x |
| 004 三节点 CPU | 359 | 3 节点，每节点 64 CPU 线程 | **0.693** | **0.305** | **0.773** | **0.762** | **2.533** | **1.491x** |
| 005 两节点 GPU/CPU | 352 | 2 节点，每节点 2×L40，CPU 帧 2% | 1.376 | 0.904 | 1.454 | 1.445 | 5.179 | 0.729x |
| 006 两节点纯 GPU | 410 | 2 节点，每节点 2×L40，CPU 仅负责 I/O 与 GPU 编排 | 1.068 | 0.729 | 1.113 | 1.114 | 4.024 | 0.939x |
| 008 单节点 CPU | 413 | 1 节点，64 CPU 线程，单并行区直接 float32 | 0.629 | 0.470 | 0.827 | 0.807 | 2.733 | 1.382x |
| 009 三节点 CPU | 415 | 3 节点，每节点 64 CPU 线程，exp008 数据路径 | **0.599** | **0.275** | **0.661** | **0.651** | **2.186** | **1.728x** |

所有结果均为官方四用例 `4/4 PASS`，表中单位是秒。exp001 使用 job 275 的
64 线程 spread 结果；每档线程数运行 3 次并取最小值，调优脚本还对每个用例
的输出执行了相同容差的 `allclose` 检查。exp004 使用三节点 job 359 的最新
正式结果；两节点 job 338 的对照保留在第 7.3 节。原 job 187 的 16 线程结果
保留在第 4.4 节，作为历史基准记录。

以 exp001 的 64 线程结果为基线后，exp004、exp008 和 exp009 的四个单用例
全部更快；exp006 仅在 `large_fft` 和 `narrow_band` 上更快，exp002、exp003
和 exp005 的四个单用例均慢于 exp001。

### 1.2 汇总对比

按四用例合计耗时排序：

```text
009 (2.186 s) < 004 (2.533 s) < 008 (2.733 s) < 001 (3.777 s)
             < 006 (4.024 s) < 002 (5.164 s) < 005 (5.179 s) < 003 (5.207 s)
```

- **三节点 exp009 是总体最快的实现**，四个单用例均为当前最低；相较 64 线程 exp001，四用例合计从 3.777 s 降至 2.186 s，聚合加速为 1.728x。
- **单节点 exp008 是单节点新最快实现**，合计 2.733 s，相较 exp001 提速 1.382x；相较三节点 exp009，单节点仍慢 1.250x。
- 三节点 exp009 相较原三节点 exp004 的 2.533 s 再快 1.159x，主要来自并行归一化、单 OpenMP 帧区域和去除 double 输出/中间 tile。
- **exp006 是 GPU 路线中的最快实现**，合计 4.024 s，但仍比 64 线程 exp001 慢约 6.5%，聚合加速为 0.939x。
- exp006 相比 exp005 合计缩短 1.155 s，约快 22.3%。主要收益来自每个 rank 只读取本地 PCM 区间，并彻底移除 CPU 帧计算。
- exp003 没有超过 exp002：合计 5.207 s 对 5.164 s。4% CPU 分担没有抵消 CPU/GPU 资源竞争。
- exp005 与 exp002 基本持平：5.179 s 对 5.164 s。增加第二节点后仍读取完整 PCM，分布式收益被启动和 I/O 开销抵消。

性能表中的聚合加速现在统一采用 exp001 的 **64 线程 spread 结果
3.777 s** 作为基线。该口径纠正了此前以 16 线程结果 7.572 s 比较造成的
高估，使所有实验与 exp001 的实际最优 CPU 配置对齐。

三节点作业已在全部节点空闲后完成：exp004 的 job 359 和 exp005 的 job 360
均为 `4/4 PASS`。三节点 exp004 比两节点版本合计快 1.217x，成为当前总冠军；
三节点 exp005 则仍慢于 64 线程 exp001。

### 1.3 相对原实现

为了避免运行完整原始 benchmark，原实现对照使用 default 用例的前
1,000,000 个 PCM 样本。基线和实验使用同一输入前缀与相同程序接口：

| 实验 | 原实现 | 实验实现 | 加速 | 口径 |
|---|---:|---:|---:|---|
| 001 | 61.662986 s | 0.084331 s | **731.206x** | 1M 样本前缀 |
| 008 | 62.461028 s | 0.043759 s | **1427.391x** | 1M 样本前缀，单节点 |
| 009 | 62.336486 s | 0.398901 s | **156.271x** | 1M 样本前缀，三节点 |

该表只比较速度，不替代官方完整四用例正确性。两个最终方案使用 job 418 的
同作业对照，日志为 `logs/final_baseline_vs_finalists_slurm-418.log`。
三节点方案在 1M 前缀上的绝对加速低于单节点，主要因为 `srun` 启动和协调
成本；完整官方四用例仍以三节点 exp009 的 1.728x 作为主性能结论。

## 2. 公共评测协议

### 2.1 基准定义

基准始终是仓库 HEAD 中的原始实现：

```text
src/melspec.cpp
```

每个实验不得覆盖该文件。实验代码作为完整实现保存在：

```text
exp/NNN_short_description.cpp
```

### 2.2 构建

构建基准：

```bash
make clean
make -j16
```

构建实验：

```bash
make clean
make EXPERIMENT=001_packed_fft_sparse_mel_simd -j16
```

### 2.3 正确性

使用官方：

```text
data/audio.pcm
data/expected/default.bin
data/expected/small_mels.bin
data/expected/large_fft.bin
data/expected/narrow_band.bin
```

验收条件：

```text
rtol = 1e-3
atol = 1e-4
```

### 2.4 加速比

对同一输入和相同参数分别测量基准与实验：

$$
\text{Speedup}
=\frac{T_{\text{baseline}}}{T_{\text{experiment}}}
$$

正式远程验证使用 3 次最小耗时。由于原始基准非常慢，附加的真实 PCM 前缀对比使用 1 次运行，并在报告中单独标注。

### 2.5 远程环境

| 项目 | 配置 |
|---|---|
| 登录节点 | `hepnode0`，只用于登录、编辑和 Slurm 管理 |
| 计算节点 | `hepnode1~3` |
| CPU | Intel Xeon Gold 6338，每节点 64 核 |
| GPU | 每节点 2 × NVIDIA L40 |
| OS | Ubuntu 26.04 LTS |
| GCC/G++ | 15.2.0 |
| Slurm | 25.11.2 |
| Python 验证环境 | `/home/wzt/.venvs/melspec` |
| NumPy/SciPy/librosa | 2.5.4 / 1.18.1 / 1.0.0 |

## 3. 实验 001：半长度 FFT、稀疏 Mel 与分块并行

### 3.1 文件

```text
exp/001_packed_fft_sparse_mel_simd.cpp
```

### 3.2 原始瓶颈

原始实现存在四类主要浪费。

#### 重复 FFT 准备工作

每一帧都会：

- 分配帧数组和复数 FFT 数组；
- 重复做位反转；
- 在每一级蝶形中重新计算 twiddle 的 `cos/sin`。

这些工作与音频内容无关。

#### 实数输入使用了全长度复数 FFT

输入帧是完全实数。原实现仍分配 `n_fft` 个复数并执行完整长度 FFT，而实数 FFT 可以通过打包把核心复数 FFT 长度降到 `n_fft/2`。

#### Mel 投影大量乘零

原始投影复杂度为：

$$
O(n\_mels\cdot n\_bins\cdot n\_frames)
$$

但三角滤波器只在局部频段非零。

#### 完整功率谱过大且缓存局部性差

正式 `default` 用例的功率谱形状是：

```text
[4097, 192397]
```

若用 double 保存，约需：

```text
6.3 GB
```

原始布局是 `power[k][t]`，大帧数下投影跨行访问，缓存局部性很差。

### 3.3 优化思路

优化顺序遵循：

1. 消除与输入内容无关的重复计算；
2. 利用实数 FFT 的共轭对称性；
3. 删除必然乘零的 Mel 计算；
4. 改变内存布局，避免完整功率谱；
5. 在良好数据布局上做 SIMD；
6. 最后做 OpenMP 帧级并行。

### 3.4 实现方案

#### FFT 计划复用

预先计算并复用：

- 位反转索引 `bit_reverse`；
- 每一级蝶形的 twiddle；
- 每一级 twiddle 的起始位置；
- 每个线程独立的 FFT 工作区。

读取 PCM、乘 Hann 窗和位反转写入合并为一次遍历。

#### 打包半长度实数 FFT

对偶数、奇数样本进行打包：

$$
z[i]=\frac{x[2i]}{2}+i\frac{x[2i+1]}{2}
$$

长度由 `n_fft` 降为 `n_fft/2`。完成复数 FFT 后，通过共轭对称关系恢复非负频率。

8192 点原 FFT 的核心复数 FFT 长度因此变为 4096。

#### 稀疏 Mel 滤波器

每个 Mel 滤波器只保存：

- 起始 FFT 箱；
- 非零权重数量；
- 权重在扁平数组中的偏移。

投影只遍历三角支撑区间，不执行 `0 * power`。

#### 256 帧分块融合

程序按 256 帧为一个 tile：

1. FFT；
2. 功率；
3. 稀疏 Mel 投影；
4. 写入小 `mel_tile`；
5. 按 Mel 行连续复制到最终输出。

完整功率谱不再物化。

#### SIMD 与 FMA

FFT 工作区采用实部、虚部 SoA 布局，并使用：

- `__restrict`；
- OpenMP `simd`；
- `-march=native`；
- FMA 指令。

Mel 点积也使用 SIMD reduction。

#### OpenMP

各线程处理 tile 内连续的帧段，避免相邻线程同时写最终矩阵的相邻列。每个 tile 只进入一次 `parallel` 区域，计算结束后 barrier，再并行复制 Mel 行。

### 3.5 复杂度变化

原始：

$$
O(F N\log N + M B F)
$$

001：

$$
O(F (N/2)\log(N/2) + F\cdot nnz(H))
$$

其中：

- `F`：帧数；
- `N`：`n_fft`；
- `M`：`n_mels`；
- `B`：FFT 箱数；
- `nnz(H)`：所有 Mel 三角滤波器的非零权重总数。

### 3.6 数值正确性设计

- 输入仍归一化为 double；
- Hann 窗使用周期形式，分母为 `n_fft`；
- FFT 不做归一化；
- 功率为 `|X[k]|^2`；
- Mel 使用 HTK 刻度；
- 保留 Slaney 面积归一化 `2/(f_r-f_l)`；
- 输出前才转换为 float32；
- 不根据 PCM 内容做分支优化。

## 4. 远程实验执行

### 4.1 远程工作目录

```text
/home/wzt/melspec-eval-001
```

官方数据通过符号链接访问：

```text
data -> /cache/recuritment/melspec/data
```

### 4.2 作业脚本

```text
scripts/run_exp001.slurm
```

流程：

1. 编译基准；
2. 编译 001；
3. 对四个官方用例执行 `validate.py --runs 3`；
4. 从官方 PCM 取前 1,000,000 个样本；
5. 基线与 001 各运行一次；
6. 计算加速比；
7. 保存合并日志。

### 4.3 Slurm 作业

| 项目 | 值 |
|---|---|
| Job ID | 187 |
| 请求节点 | 1 |
| 请求 CPU | 16 |
| OpenMP 线程 | 16 |
| 禁止节点 | `hepnode0` |
| 实际节点 | `hepnode2` |
| 状态 | `COMPLETED`，退出码 `0:0` |
| 开始时间 | 2026-10-30 18:31:06 HKT |
| 结束时间 | 2026-10-30 18:37:30 HKT |
| Slurm 总耗时 | 6 分 24 秒 |

日志已保存到本地：

```text
logs/001_packed_fft_sparse_mel_simd_slurm-187.log
```

### 4.4 远程结果

#### 官方完整数据正确性

`validate.py --runs 3` 的四用例结果：

| 用例 | n_fft | n_mels | 最大绝对误差 | 状态 | 001 最小耗时 |
|---|---:|---:|---:|---|---:|
| default | 8192 | 256 | 3.12e-2 | PASS | 1.509 s |
| small_mels | 8192 | 64 | 1.17e-2 | PASS | 1.128 s |
| large_fft | 16384 | 256 | 1.25e-1 | PASS | 2.550 s |
| narrow_band | 16384 | 256 | 6.25e-2 | PASS | 2.385 s |

结果为：

```text
4/4 PASSED
```

`validate.py` 中的 Speedup 列表示相对 librosa 的加速比，范围是 `8.925x~14.984x`。它不能与下表中的“相对原始 C++ 基准”混用。

#### 相对原始实现的加速比

为了控制原始实现的测试时间，从官方 `audio.pcm` 取前 1,000,000 个样本。基线和 001 使用完全相同的输入、参数、编译器和计算节点：

| 用例 | 基准耗时 | 001 耗时 | 加速比 |
|---|---:|---:|---:|
| default, 1M samples | 61.662986 s | 0.084331 s | **731.206x** |

该结果表明：原始实现的时间主要不是浪费在 I/O，而是全长度复数 FFT、稠密乘零 Mel 投影和跨行缓存访问；001 将这三部分同时消除后产生了约三个数量级的差距。

### 4.5 CPU 线程扩展性

在 `hepnode2` 上使用全部 64 个物理 CPU 核测试 `exp001`。每个线程点运行 3 次并取最小值。

| 线程数 | close 四用例总计 | spread 四用例总计 |
|---:|---:|---:|
| 1 | 78.929358 s | 79.119848 s |
| 8 | 12.179977 s | 12.580531 s |
| 16 | 7.898493 s | 7.415062 s |
| 32 | 5.247995 s | 5.248376 s |
| 48 | 4.309100 s | 4.306971 s |
| 56 | 4.222898 s | 4.041572 s |
| 64 | 3.962291 s | **3.776826 s** |

结论：

- 全用例最优配置是 64 线程、`OMP_PROC_BIND=spread`、`OMP_PLACES=cores`；
- 相对 16 线程约加速 `1.96x`；
- 64 线程相对单线程约加速 `20.95x`，并行效率约 `32.7%`；
- 32 线程已经达到 64 线程约 `72%` 的吞吐，适合作为资源较少时的折中配置；
- 线程数不应超过 Slurm 实际分配给进程的 CPU 数。

测试作业：

```text
Slurm job: 275
Node: hepnode2
State: COMPLETED
Elapsed: 21m39s
Log: logs/cpu-thread-scaling_slurm-275.log
```

## 5. 实验 002：单节点双 GPU cuFFT

### 5.1 计算路径

两个 L40 各自控制一个 CUDA context 和 stream，分别处理连续的帧区间：

```text
device 0 -> 帧 [0, F/2)
device 1 -> 帧 [F/2, F)
```

每个 device 内按 1024 帧一个 tile：

1. GPU kernel 从 host double 波形分帧并乘 Hann 窗；
2. `cufftPlanMany(..., CUFFT_D2Z, batch=1024)`；
3. 将复数谱平方并转置成 `[bin, tile]`；
4. 稀疏 Mel kernel 按连续 frame 维度做点积；
5. D2H 到 pinned host buffer，再写入最终 `[n_mels, n_frames]`。

先使用 double cuFFT，确保数值误差与 001 相同。

### 5.2 官方结果

| 用例 | exp001 | exp002 | 相对 exp001 |
|---|---:|---:|---:|
| default | 1.509 s | 1.341 s | 1.125x |
| small_mels | 1.128 s | 0.925 s | 1.219x |
| large_fft | 2.550 s | 1.444 s | 1.766x |
| narrow_band | 2.385 s | 1.454 s | 1.640x |

四用例均 `PASS`，最大绝对误差与 exp001 相同：

```text
3.12e-2, 1.17e-2, 1.25e-1, 6.25e-2
```

远程作业：

```text
Job: 292
Node: hepnode2
GPUs: 2 × NVIDIA L40
State: COMPLETED
Log: logs/002_gpu_cufft_slurm-292.log
```

## 6. 实验 003：单节点 CPU+双 GPU 异构

### 6.1 实现

把帧区间再分成 GPU 和 CPU 两部分：

```text
GPU 0 + GPU 1 -> 前 (100-p)% 帧
CPU OpenMP    -> 后 p% 帧
```

GPU thread 与 CPU OpenMP 同时工作，最终写回同一个输出矩阵的不同列。CPU 比例通过 `MELSPEC_CPU_PERCENT` 控制。

### 6.2 正式验证配置

```text
64 CPU threads
2 × NVIDIA L40
MELSPEC_CPU_PERCENT=4
OMP_PROC_BIND=spread
```

### 6.3 官方结果

| 用例 | exp001 | exp003 | 相对 exp001 |
|---|---:|---:|---:|
| default | 1.509 s | 1.353 s | 1.115x |
| small_mels | 1.128 s | 0.929 s | 1.214x |
| large_fft | 2.550 s | 1.474 s | 1.730x |
| narrow_band | 2.385 s | 1.451 s | 1.644x |

四用例均 `PASS`。

远程作业：

```text
Job: 347
Node: hepnode3
State: COMPLETED
Log: logs/003_cpu_gpu_heterogeneous_slurm-347.log
```

### 6.4 CPU 比例调参

在 64 CPU OpenMP 线程下，四用例总耗时：

| CPU 帧比例 | 四用例总耗时 |
|---:|---:|
| 0% | **5.102446 s** |
| 2% | 5.178589 s |
| 4% | 5.183178 s |
| 6% | 5.201430 s |
| 8% | 5.276062 s |
| 10% | 5.267444 s |
| 12% | 5.284117 s |

第二轮测试了 8/16/32/64 CPU 线程与 4/8/12% 的组合，最好的异构配置仍是 64 线程 + 4%，总耗时 5.207414 s。纯双 GPU 约 5.10 s 仍略快，说明 L40 已接近当前 I/O/内存瓶颈；003 保留 4% CPU 作为实际的 CPU+GPU 并发行程。

日志：

```text
logs/003_cpu_gpu_tuning_slurm-299.log
logs/003_cpu_thread_tuning_slurm-301.log
```

## 7. 实验 004：多节点 CPU

### 7.1 MPI 功能实现

实验保留了标准 MPI 路径：

- 3 个 rank 获取连续帧区间；
- 每 rank 运行 2/3 周期 Mel 计算；
- rank 局部结果转换为 float32；
- `MPI_Gatherv` 收集到 rank 0；
- 仅 rank 0 返回不可修改的 `main` 并调用 `write_output`。

两节点 MPI 调试作业 302 已通过 4/4，证明 rank 划分、输出归属和归并正确。但 OpenMPI 的 `MPI_Init`/启动成本使每个短用例增加约 1–2 秒，无法稳定超过 exp001。

### 7.2 原生 Slurm 多进程性能版

为保留多节点并行同时消除 MPI 启动固定开销，同一源码增加 `MELSPEC_SLURM_MULTI` 模式：

- rank 从 `SLURM_PROCID`、`SLURM_NTASKS` 获取；
- 每个 rank 只读取自己帧所需的 PCM 区间；
- 每 rank 直接计算局部 `[n_mels, local_frames]`；
- 局部 double 转 float32；
- 使用 `pwrite` 并行写最终矩阵的不重叠区域；
- launcher 预创建共享输出与 ready 文件；
- rank 0 写入本轮 64-bit token，其他 rank `pread` 确认 token 后写入；
- 完成后把 validate 的 `/tmp` 路径映射到共享结果文件。

该模式不依赖 MPI 库启动协议，但仍由 Slurm 在多个计算节点上并行运行独立进程。

### 7.3 官方结果

两节点、每节点 64 OpenMP 线程：

| 用例 | exp001 | exp004 | 相对 exp001 |
|---|---:|---:|---:|
| default | 1.509 s | 0.818 s | 1.845x |
| small_mels | 1.128 s | 0.398 s | 2.834x |
| large_fft | 2.550 s | 0.941 s | 2.710x |
| narrow_band | 2.385 s | 0.926 s | 2.576x |

四用例均 `PASS`，且每一个都显著快于 exp001。

```text
Job: 338
Nodes: hepnode2, hepnode3
State: COMPLETED
Log: logs/004_slurm_fast_slurm-338.log
```

### 7.4 三节点结果

三节点、每节点 64 OpenMP 线程的结果：

| 用例 | 64 线程 exp001 | 三节点 exp004 | 相对 exp001 |
|---|---:|---:|---:|
| default | 0.931 s | **0.693 s** | 1.344x |
| small_mels | 0.569 s | **0.305 s** | 1.865x |
| large_fft | 1.155 s | **0.773 s** | 1.494x |
| narrow_band | 1.122 s | **0.762 s** | 1.472x |
| 合计 | 3.777 s | **2.533 s** | **1.491x** |

四用例均 `PASS`。相较两节点合计 `3.083 s`，三节点版本再快 `1.217x`。

```text
Job: 359
Nodes: hepnode1, hepnode2, hepnode3
State: COMPLETED, exit code 0:0
Log: logs/004_slurm_fast_3node_slurm-359.log
```

### 7.5 性能诊断

相同节点上测得：

- 200 MiB MPI Sendrecv 有效带宽：4121.4 MiB/s；
- InfiniBand：56 Gb/s FDR，UCX 正确选择 `rc_mlx5/mlx5_0:1`；
- MPI 版内部 compute+gather+repack 仅约 0.7 s，但外部 wall 约 2.0 s；
- 原生 Slurm 版内部读取约 0.09 s/rank、compute 约 0.25 s、写文件约 0.28 s，总 wall 约 0.82–1.09 s。

结论：慢点不是网络，而是每次调用的 MPI 启动和 root 输出路径。

## 8. 实验 005：多节点 GPU/CPU 异构

### 8.1 实现

每个节点使用：

- 1 个 Slurm rank；
- 2 × NVIDIA L40；
- 64 CPU OpenMP 线程；
- CPU 分担 2% 帧，其余帧由双 GPU 处理。

节点内 GPU 计算继续使用 cuFFT double、GPU 功率转置和稀疏 Mel kernel；节点间使用 token 协调和 float32 分布写入，每个 rank 读取完整 PCM 并只计算自己的帧区间。

实验也保留了 MPI+CUDA 功能路径。两节点 MPI 调试作业 303 通过 4/4，但启动与 Gather 固定开销较大，最终性能版采用原生 Slurm 后端。

### 8.2 官方结果

两节点，每节点双 L40：

| 用例 | exp001 | exp005 | 相对 exp001 |
|---|---:|---:|---:|
| default | 1.509 s | 1.376 s | 1.097x |
| small_mels | 1.128 s | 0.904 s | 1.248x |
| large_fft | 2.550 s | 1.454 s | 1.754x |
| narrow_band | 2.385 s | 1.445 s | 1.650x |

四用例均 `PASS`，全部快于 exp001。

```text
Job: 352
Nodes: hepnode2, hepnode3
GPUs: 2 per node
State: COMPLETED
Log: logs/005_slurm_fast_2node_slurm-352.log
```

### 8.3 单节点对照

同一 driver 在单节点双 L40 上为：

| 用例 | 耗时 |
|---|---:|
| default | 1.662 s |
| small_mels | 1.038 s |
| large_fft | 1.778 s |
| narrow_band | 1.782 s |

单节点聚合耗时低于 exp001，但 default 单独仍略慢；增加第二节点后所有用例均达标。

## 9. 实验 006：两节点纯 GPU

### 9.1 实现

实验 006 使用独立源码和 launcher，不再通过 `MELSPEC_CPU_PERCENT` 启用 CPU
帧分支。每个节点运行一个 Slurm rank，并显式要求两块 L40：

- 两个 rank 各自只读取连续本地帧所需的 PCM 样本区间；
- 每节点将本地帧均分给两个独立 CUDA context；
- 周期性 Hann 窗、`CUFFT_D2Z`、功率谱、稀疏 Mel 和 float32 转换均由 GPU 完成；
- 只保留稀疏滤波器上界以内的功率箱，减少无效转置和显存占用；
- 两个 rank 通过 64-bit token 协调后，分别 `pwrite` 最终矩阵的不重叠帧列。

CPU 只负责命令行解析、输入读取、滤波器静态参数构造、线程/GPU 编排和最终
文件写入，不执行任何逐帧 FFT、功率或 Mel 计算。源码还拒绝非零
`MELSPEC_CPU_PERCENT` 和 `OMP_NUM_THREADS`，防止误启动 CPU 帧路径。

### 9.2 官方结果

| 用例 | exp001 | exp006 | 相对 exp001 |
|---|---:|---:|---:|
| default | 1.509 s | 1.068 s | 1.413x |
| small_mels | 1.128 s | 0.729 s | 1.547x |
| large_fft | 2.550 s | 1.113 s | 2.291x |
| narrow_band | 2.385 s | 1.114 s | 2.141x |

四用例均 `PASS`，最大绝对误差与 exp001 相同：

```text
3.12e-2, 1.17e-2, 1.25e-1, 6.25e-2
```

远程作业：

```text
Job: 410
Nodes: hepnode1, hepnode2
GPUs: 2 × NVIDIA L40 per node (4 total)
Command: python validate.py ./scripts/melspec_exp006_launcher.sh --runs 3
State: COMPLETED
Log: logs/006_multi_node_gpu_only_slurm-410.log
```

## 10. 实验 007：单节点分阶段 profiling

实验 007 包装未修改的 exp001 kernel，把单节点 default 拆成四个阶段：

| 阶段 | 耗时 |
|---|---:|
| PCM 读取与归一化 | 约 0.20 s |
| FFT、功率和 Mel 计算 | 约 0.49–0.51 s |
| double→float 与输出写入 | 约 0.23–0.25 s |
| 合计 | 约 0.93–0.96 s |

完整写盘与 `/dev/null` 输出只相差约 70–90 ms，说明主要时间并不是磁盘 payload，
而是串行归一化、完整 double 输出矩阵和最终转换。峰值 RSS 约 779 MB。

```text
Jobs: 411 (I/O comparison), 412 (stage driver)
Logs: logs/001_single_node_profile_slurm-411.log
      logs/007_single_node_profile_slurm-412.log
```

job 411 的完整写盘和 `/dev/null` 计时均正常完成，但末尾 `perf stat` 因集群
`perf_event_paranoid=4` 无法访问硬件计数器，使该作业最终退出码为 `1:0`。
本报告只采用失败前的有效计时；job 412 的 stage driver 和官方正确性检查
完整通过。

## 11. 实验 008：单节点 CPU 流水化

实验 008 保持 exp001 的 FFT/Mel 数学实现，但做三项数据路径优化：

1. PCM 归一化改用 64 线程 OpenMP；
2. 不再为每个 256 帧 tile 创建独立并行区，而是用单个并行区域覆盖全部帧；
3. 每个线程直接写出 float32 Mel，删除完整 double 输出矩阵、tile staging
   matrix 和最后一遍串行转换。

default profile 的分阶段耗时约为：

```text
read I/O 0.06 s + normalize 0.11 s + compute_float 0.36 s + write 0.08 s
```

峰值 RSS 从 exp001 的约 779 MB 降至约 394 MB。官方结果：

| 用例 | 64 线程 exp001 | exp008 | 相对 exp001 |
|---|---:|---:|---:|
| default | 0.931 s | **0.629 s** | 1.481x |
| small_mels | 0.569 s | **0.470 s** | 1.210x |
| large_fft | 1.155 s | **0.827 s** | 1.397x |
| narrow_band | 1.122 s | **0.807 s** | 1.390x |
| 合计 | 3.777 s | **2.733 s** | **1.382x** |

```text
Job: 413
Node: hepnode2
State: COMPLETED
Log: logs/008_single_node_cpu_slurm-413.log
```

## 12. 实验 009：三节点 CPU 流水化

实验 009 把 exp008 的数据路径与 exp004 的三节点原生 Slurm 后端结合：

- 每个 rank 只读取本地连续帧所需 PCM；
- 本地读取后并行归一化；
- 使用单 OpenMP 区域直接计算并输出 float32；
- rank 之间通过 64-bit token 协调，随后分别 `pwrite` 不重叠帧列。

default profile 中，每 rank 读取约 0.02 s、并行归一化约 0.04 s、计算约
0.13 s，分布式写入约 0.28 s。输出写入已经成为主要瓶颈。

| 用例 | 64 线程 exp001 | 三节点 exp004 | exp009 | 相对 exp001 |
|---|---:|---:|---:|---:|
| default | 0.931 s | 0.693 s | **0.599 s** | 1.555x |
| small_mels | 0.569 s | 0.305 s | **0.275 s** | 2.069x |
| large_fft | 1.155 s | 0.773 s | **0.661 s** | 1.747x |
| narrow_band | 1.122 s | 0.762 s | **0.651 s** | 1.723x |
| 合计 | 3.777 s | 2.533 s | **2.186 s** | **1.728x** |

四用例均 `PASS`。相比 exp004 三节点版本，exp009 再快 1.159x。

```text
Job: 415
Nodes: hepnode1, hepnode2, hepnode3
State: COMPLETED, exit code 0:0
Log: logs/009_three_node_cpu_slurm-415.log
```

## 13. 本地探索性结果

以下数据来自本地 Ubuntu 26.04 / GCC 15 开发环境，只用于实现调试，不作为正式远程成绩：

| 用例 | 基准 | 001 | 加速比 |
|---|---:|---:|---:|
| 32,768 样本，8192 FFT，256 Mel | 0.250991 s | 0.010863 s | 23.106x |

该小样本主要验证评测脚本和优化方向。正式结果必须来自 Slurm 计算节点。

## 14. 后续实验模板

新增实验时在本文档增加一节，至少包含：

```text
实验编号：
实现文件：
改动假设：
实现方案：
正确性：
基准耗时：
实验耗时：
加速比：
运行节点：
日志路径：
后续问题：
```

不覆盖旧实验，不复用旧日志，不把本地测量冒充远程成绩。

## 15. 参考资料、AI 工具与典型问题

### 15.1 参考资料

- 题目仓库的 `README.md`、`src/melspec.cpp`、`melspec.py`、
  `include/melspec.h` 和 `validate.py`；
- 官方 PCM 与 `data/expected` 四份期望输出；
- NVIDIA CUDA Runtime 与 cuFFT 官方 API 文档；
- OpenMP 和 GCC 优化选项文档；
- Slurm `sbatch`、`srun`、`sacct`、`squeue` 文档；
- 集群 `nvidia-smi`、`sinfo` 实测输出和第 4–15 节日志。

### 15.2 AI 工具

使用 OpenAI Codex（GPT-5）辅助代码阅读、优化方案讨论、实现草稿、Slurm
脚本、主要实验执行、结果整理和文档/PPT 制作。所有正式性能数据来自实际
Slurm 作业，没有使用生成数据冒充实测。最终实现均经过编译、官方用例、
日志复核和性能口径检查。

### 15.3 典型问题

- **跨节点临时文件不可见**：benchmark 将 1M 输入放在 batch 节点 `/tmp`，
  其他 rank 返回 143。改为共享目录输入后 job 418 通过。
- **共享 ready 文件竞态**：固定输出路径可能读到上一轮 ready 状态。最终使用
  每轮 64-bit token，由 rank 0 写入，其他 rank 确认后再写输出。
- **NVCC 不识别 `-Wl,-rpath`**：CUDA 编译阶段报未知选项。改为通过
  `LD_LIBRARY_PATH` 定位 CUDA 库，避免把链接器参数传给 NVCC。
- **perf 权限受限**：`perf_event_paranoid=4` 禁止硬件计数器。改用阶段计时
  driver，保留了有效 profile 数据。
