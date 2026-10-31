# Remote Evaluation Logs

Store raw remote Slurm output in this directory. Submission-related logs are
tracked in Git so reviewers can inspect the measurements without cluster access.

Recommended file name:

```text
<experiment-id>_slurm-<job-id>.log
```

Example:

```text
001_packed_fft_sparse_mel_simd_slurm-123.log
```

Each log should identify:

- experiment source file;
- Git revision;
- Slurm job ID and node;
- Slurm resource request;
- OpenMP settings;
- validation command and complete output.
