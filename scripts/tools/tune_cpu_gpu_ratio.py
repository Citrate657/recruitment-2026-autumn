#!/usr/bin/env python3
"""Tune the CPU frame share of the CPU+dual-GPU melspec implementation."""

import argparse
import os
import pathlib
import subprocess
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from validate import TEST_CASES, build_args


def run_once(binary, input_path, case_name, params, shape, percent, threads, runs):
    best = float("inf")
    checked = False
    for _ in range(runs):
        fd, output_path = tempfile.mkstemp(suffix=".bin")
        os.close(fd)
        env = os.environ.copy()
        env["MELSPEC_CPU_PERCENT"] = str(percent)
        env["OMP_NUM_THREADS"] = str(threads)
        env["OMP_PROC_BIND"] = "spread"
        env["OMP_PLACES"] = "cores"
        command = [binary, input_path, output_path] + build_args(params)
        started = time.perf_counter()
        try:
            subprocess.run(command, check=True, capture_output=True, env=env)
            elapsed = time.perf_counter() - started
            best = min(best, elapsed)
            if not checked:
                result = np.fromfile(output_path, dtype="<f4").reshape(shape)
                expected = np.fromfile(
                    os.path.join("data/expected", f"{case_name}.bin"),
                    dtype="<f4",
                ).reshape(shape)
                if not np.allclose(result, expected, rtol=1e-3, atol=1e-4):
                    raise RuntimeError(
                        f"{case_name} failed at {percent}% CPU"
                    )
                checked = True
        finally:
            os.unlink(output_path)
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--percentages", default="0,2,4,6,8,10,12")
    parser.add_argument("--threads", type=int, default=64)
    parser.add_argument("--runs", type=int, default=2)
    args = parser.parse_args()

    percentages = [int(value) for value in args.percentages.split(",")]
    results = {}
    for percent in percentages:
        total = 0.0
        for case in TEST_CASES:
            elapsed = run_once(
                args.binary,
                "data/audio.pcm",
                case["name"],
                case["params"],
                case["shape"],
                percent,
                args.threads,
                args.runs,
            )
            results[(percent, case["name"])] = elapsed
            total += elapsed
            print(
                f"cpu_percent={percent:2d} case={case['name']:12} "
                f"time={elapsed:.6f}s",
                flush=True,
            )
        print(f"cpu_percent={percent:2d} total={total:.6f}s", flush=True)

    print()
    print("| CPU % | default | small_mels | large_fft | narrow_band | Total |")
    print("|---:|---:|---:|---:|---:|---:|")
    for percent in percentages:
        values = [
            results[(percent, case["name"])] for case in TEST_CASES
        ]
        print(
            f"| {percent} | {values[0]:.6f} | {values[1]:.6f} | "
            f"{values[2]:.6f} | {values[3]:.6f} | {sum(values):.6f} |"
        )


if __name__ == "__main__":
    main()
