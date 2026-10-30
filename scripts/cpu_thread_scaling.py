#!/usr/bin/env python3
"""Measure OpenMP thread scaling for melspec official test cases."""

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


def run_once(binary, input_path, params, threads, binding):
    fd, output_path = tempfile.mkstemp(suffix=".bin")
    os.close(fd)
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    env["OMP_PROC_BIND"] = binding
    env["OMP_PLACES"] = "cores"
    command = [binary, input_path, output_path] + build_args(params)

    started = time.perf_counter()
    try:
        subprocess.run(command, check=True, capture_output=True, env=env)
        elapsed = time.perf_counter() - started
        with open(output_path, "rb") as output_file:
            output_bytes = output_file.read()
    finally:
        os.unlink(output_path)
    return elapsed, output_bytes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--input", default="data/audio.pcm")
    parser.add_argument("--expected-dir", default="data/expected")
    parser.add_argument("--threads", default="1,2,4,8,12,16,24,32,40,48,56,64")
    parser.add_argument("--bindings", default="close,spread")
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    thread_counts = [int(value) for value in args.threads.split(",")]
    bindings = [value.strip() for value in args.bindings.split(",")]
    results = {}

    for case in TEST_CASES:
        name = case["name"]
        params = case["params"]
        shape = case["shape"]
        expected = np.fromfile(
            os.path.join(args.expected_dir, f"{name}.bin"), dtype="<f4"
        ).reshape(shape)

        for binding in bindings:
            for threads in thread_counts:
                best_time = float("inf")
                checked = False
                for _ in range(args.runs):
                    elapsed, output_bytes = run_once(
                        args.binary,
                        args.input,
                        params,
                        threads,
                        binding,
                    )
                    best_time = min(best_time, elapsed)
                    if not checked:
                        result = np.frombuffer(
                            output_bytes, dtype="<f4"
                        ).reshape(shape)
                        if not np.allclose(
                            result, expected, rtol=1e-3, atol=1e-4
                        ):
                            raise RuntimeError(
                                f"{name}, {threads} threads, {binding}: FAIL"
                            )
                        checked = True

                results[(name, binding, threads)] = best_time
                print(
                    f"{name:12} binding={binding:6} threads={threads:2d} "
                    f"time={best_time:.6f}s",
                    flush=True,
                )

    for case in TEST_CASES:
        name = case["name"]
        print()
        print(f"### {name}")
        print()
        print("| Threads | Close | Spread | Fastest binding |")
        print("|---:|---:|---:|---|")
        for threads in thread_counts:
            close_time = results[(name, "close", threads)]
            spread_time = results[(name, "spread", threads)]
            fastest = "close" if close_time <= spread_time else "spread"
            print(
                f"| {threads} | {close_time:.6f} | {spread_time:.6f} | "
                f"{fastest} |"
            )

    print()
    print("### Sum of four case minima")
    print()
    print("| Threads | Close | Spread |")
    print("|---:|---:|---:|")
    for threads in thread_counts:
        close_sum = sum(
            results[(case["name"], "close", threads)] for case in TEST_CASES
        )
        spread_sum = sum(
            results[(case["name"], "spread", threads)] for case in TEST_CASES
        )
        print(f"| {threads} | {close_sum:.6f} | {spread_sum:.6f} |")


if __name__ == "__main__":
    main()
