# Long-Running Optimization Plan

## Goal

优化工作已按用户决定停止。当前阶段只做总结、文档收尾、PPT 制作和汇报准备。

已完成的两条路线：

1. **单节点 CPU**：exp008 通过官方四用例，合计 `2.733 s`。
2. **多节点 CPU**：exp009 通过官方四用例，合计 `2.186 s`。

汇报统一使用 exp001 的 64 线程 spread 结果 `3.777 s` 作为性能基线。

## Milestones

1. **001 baseline**: completed. Official validation and benchmark are in `docs/experiment-report.md`.
2. **002 dual-GPU cuFFT**: completed. Job 292 passed 4/4 and is faster than exp001 on every case.
3. **003 CPU+GPU heterogeneous**: completed. Job 347 passed 4/4 with final 4% CPU share and is faster than exp001 on every case.
4. **004 multi-node CPU**: two-node and three-node native Slurm runs are complete. Three-node job 359 passed 4/4 with `0.693 / 0.305 / 0.773 / 0.762 s` and `2.533 s` total, making it the current overall winner.
5. **005 multi-node GPU/CPU heterogeneous**: completed on two nodes with native Slurm ranks, dual GPU and CPU share per node; 4/4 PASS and every case faster than exp001. MPI+CUDA functional path also passed 4/4.
6. **Previous audit**: experiments through 005 were committed as `21d7669`. Submission cleanup now tracks `docs/`, final logs, results and the presentation under Git.
7. **006 two-node GPU-only**: completed and committed as `b71de64`. Job 410 used `hepnode1~2`, four L40 GPUs total, local PCM range reads per rank, and GPU-only frame processing. Official validation passed 4/4 with times `1.068 / 0.729 / 1.113 / 1.114 s`, all faster than exp001.
8. **Single-node track**: exp007 profiling is complete. exp008 passed 4/4 with `2.733 s`, down from exp001's `3.777 s`; peak memory fell from about 779 MB to 394 MB.
9. **Three-node CPU track**: exp004 job 359 passed in `2.533 s`. Profiling job 414 identified distributed output as the remaining bottleneck. exp009 combines that backend with the direct float32 path and passes 4/4 at `2.186 s`, currently the overall winner. Experiments 007–009 and reproducible scripts are committed as `5549edd`.
10. **Summary and presentation**: completed. The executive summary, 11-slide editable deck, per-slide speaker notes and presentation script are ready. Do not start another optimization experiment unless the user reopens the optimization phase.

## Execution constraints

- Never run application computation on `hepnode0`.
- Use only Slurm `hepnode1~3` for compiled programs and validation.
- `src/main.cpp`, `include/melspec.h`, `validate.py`, and `melspec.py` remain unchanged.
- Every claimed success needs a reproducible command and captured log.
- Update `docs/PROGRESS.md` after each checkpoint, failure, or key decision.
