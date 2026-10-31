# 参考资料与 AI 使用说明

## 项目内资料

- `README.md`：题目要求、CLI、输出格式和提交规范。
- `src/main.cpp`、`src/melspec.cpp`：原始接口与参考实现。
- `include/melspec.h`：公开函数约束。
- `melspec.py`、`validate.py`：算法对照和官方评测流程。
- `data/audio.pcm`、`data/expected/*.bin`：评测输入和期望输出。
- `logs/`：所有保留的 Slurm 运行记录。

## 外部接口资料

- [NVIDIA CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/)：
  设备管理、host/device copy、stream 和同步。
- [NVIDIA cuFFT](https://docs.nvidia.com/cuda/cufft/)：`cufftPlanMany`、
  `cufftExecD2Z` 和 D2Z 输出布局。
- [OpenMP specifications](https://www.openmp.org/specifications/)：并行区域、
  线程绑定和 SIMD reduction。
- [GCC optimization options](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html)：
  `-O3`、`-march=native` 和自动向量化。
- [Slurm documentation](https://slurm.schedmd.com/documentation.html)：`sbatch`、
  `srun`、`sacct`、`squeue` 和多节点资源分配。
- [librosa melspectrogram](https://librosa.org/doc/main/generated/librosa.feature.melspectrogram.html)：
  官方评测语义和 Mel 输出参照。

这些资料用于确认 API 语义和实现边界。最终行为以本仓库实际编译和远程运行
结果为准。

## AI 工具

使用 OpenAI Codex（GPT-5）辅助：

- 阅读源文件、竞品实验和 Slurm 日志；
- 讨论 FFT、Mel、GPU、多节点和 I/O 优化方案；
- 草拟 C++/CUDA 实现、诊断脚本、报告和 PPT；
- 提交并监控 Slurm 作业，整理实测结果；
- 检查数值正确性、性能口径和文档一致性。

正式数据全部来自集群实际运行。AI 没有生成或估计任何替代性能数字。人工
审查的重点是算法语义、并行边界、错误处理结果和原始日志。

## 典型 bug

### 跨节点临时输入

`benchmark_experiments.py` 在 batch 节点创建 `/tmp` 临时 PCM。多节点实验
的另外两个 rank 无法看到该路径，作业以 143 退出。修复方式是先把相同输入
放到共享目录，再做 1M 样本速度对照。

失败日志：

```text
logs/final_baseline_vs_exp009_slurm-416_failed_nonshared_temp.log
```

成功日志：

```text
logs/final_baseline_vs_exp009_slurm-417.log
```

### ready token 竞态

固定输出路径和旧 ready 文件会让 rank 1 读到上一轮状态。最终协议为每轮生成
64-bit token，rank 0 准备输出后写入 token，其他 rank 通过 `pread` 确认后
再执行不重叠的 `pwrite`。

### NVCC 链接参数

`-Wl,-rpath,...` 被 NVCC 当作未知编译选项。修复为显式设置
`LD_LIBRARY_PATH`，避免将主机链接器参数混入 NVCC 命令行。

### perf 计数器权限

集群设置 `perf_event_paranoid=4`，普通用户无法读取硬件计数器。因此使用
exp007 的阶段计时 driver，并明确区分有效计时和 perf 权限失败。
