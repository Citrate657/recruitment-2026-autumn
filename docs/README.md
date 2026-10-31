# 文档索引

## 核心交付物

| 文件 | 用途 | 适合谁 |
|---|---|---|
| [melspec-performance-review.pptx](melspec-performance-review.pptx) | 10 页最终汇报 PPT | 评审、面试官、项目外部读者 |
| [presentation-script.md](presentation-script.md) | 与当前 PPT 逐页对应的讲稿、优化推导、HPC 知识说明和 Q&A | 汇报人 |
| [optimization-report.md](optimization-report.md) | 合并后的完整技术报告，包含优化笔记、exp001–009 实验、性能分析和总结 | 想了解完整实现与验证过程的读者 |

## 推荐阅读顺序

1. [SUBMISSION.md](../SUBMISSION.md)：提交入口和最终结果。
2. [presentation-script.md](presentation-script.md)：12–15 分钟汇报主线。
3. [optimization-report.md](optimization-report.md)：完整技术推导和实验证据。
4. [reference/melspec-basics.md](reference/melspec-basics.md)：从 PCM 到 Mel
   输出的算法基础。

## 参考与证据

| 位置 | 内容 |
|---|---|
| [assets/code-evolution.png](assets/code-evolution.png) | exp001–009 代码演进图 |
| [../experiments/README.md](../experiments/README.md) | 九个实验源码、构建入口和用途 |
| [../scripts/README.md](../scripts/README.md) | 复现、验证、profile 和诊断脚本 |
| [../results/README.md](../results/README.md) | 机器可读性能结果 |
| [../logs/README.md](../logs/README.md) | Slurm 原始日志索引 |

## 过程归档

- [archive/development-log.md](archive/development-log.md)：开发时间线、失败尝试和
  中间决策。
- [archive/project-plan.md](archive/project-plan.md)：执行计划与里程碑。

归档文件保留原始开发语境，其中的旧路径可能对应整理前的目录。正式路径、性能
口径和结论以本页列出的三个核心交付物为准。
