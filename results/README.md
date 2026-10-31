# Result Summary

`final-metrics.csv` 汇总最终采用的关键实验。数值来自
`docs/optimization-report.md` 和 `logs/` 下的 Slurm 原始日志。

统一性能基线为 exp001 的 64 线程 spread 结果 `3.776826 s`。表中
`speedup_vs_exp001_t64` 等于该基线除以实验四用例合计耗时。

`relative-to-original.csv` 记录与原实现的同输入前缀对照。该表只用于速度
比较，不替代官方完整四用例正确性验证。两个最终方案在 1M 样本前缀上的
原实现加速分别为 exp008 `1427.391x`、exp009 `156.271x`。
