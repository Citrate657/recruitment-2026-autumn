#!/usr/bin/env python3
"""Benchmark two melspec executables on the official validation cases."""

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


def run_binary(binary, input_path, params, runs):
    cmd_prefix = [binary, input_path]
    args = build_args(params)
    best_time = float("inf")
    output_bytes = None
    output_path = None

    try:
        for run in range(runs):
            fd, current_path = tempfile.mkstemp(suffix=".bin")
            os.close(fd)
            if output_path is not None:
                os.unlink(output_path)
            output_path = current_path

            cmd = cmd_prefix + [output_path] + args
            started = time.perf_counter()
            subprocess.run(cmd, check=True, capture_output=True)
            elapsed = time.perf_counter() - started
            best_time = min(best_time, elapsed)

            if run == 0:
                with open(output_path, "rb") as output_file:
                    output_bytes = output_file.read()
    finally:
        if output_path is not None and os.path.exists(output_path):
            os.unlink(output_path)

    return best_time, output_bytes


def shape_for_input(input_path, params):
    sample_count = os.path.getsize(input_path) // 2
    n_frames = 1 + (sample_count - params["n_fft"]) // params["hop_length"]
    return params["n_mels"], n_frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--input", default="data/audio.pcm")
    parser.add_argument("--expected-dir", default="data/expected")
    parser.add_argument("--cases", default="all")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Benchmark a deterministic prefix of the PCM instead of all samples",
    )
    parser.add_argument(
        "--skip-correctness",
        action="store_true",
        help="Skip expected-output checks for a custom input file",
    )
    args = parser.parse_args()

    selected_names = (
        [case["name"] for case in TEST_CASES]
        if args.cases == "all"
        else [name.strip() for name in args.cases.split(",") if name.strip()]
    )
    case_map = {case["name"]: case for case in TEST_CASES}

    input_path = args.input
    temporary_input = None
    if args.max_samples > 0:
        samples = np.fromfile(args.input, dtype="<i2", count=args.max_samples)
        fd, temporary_input = tempfile.mkstemp(suffix=".pcm")
        with os.fdopen(fd, "wb") as output_file:
            samples.tofile(output_file)
        input_path = temporary_input

    rows = []
    try:
        for name in selected_names:
            case = case_map[name]
            params = case["params"]
            print(f"Benchmarking {name}...", flush=True)

            baseline_time, _ = run_binary(
                args.baseline, input_path, params, args.runs
            )
            experiment_time, experiment_bytes = run_binary(
                args.experiment, input_path, params, args.runs
            )
            speedup = baseline_time / experiment_time

            max_abs_error = None
            passed = None
            if args.max_samples == 0 and not args.skip_correctness:
                shape = shape_for_input(input_path, params)
                result = np.frombuffer(experiment_bytes, dtype="<f4").reshape(shape)
                expected_path = os.path.join(
                    args.expected_dir, f"{name}.bin"
                )
                expected = np.fromfile(expected_path, dtype="<f4").reshape(shape)
                max_abs_error = float(np.max(np.abs(result - expected)))
                passed = bool(
                    np.allclose(result, expected, rtol=1e-3, atol=1e-4)
                )

            rows.append(
                {
                    "name": name,
                    "baseline": baseline_time,
                    "experiment": experiment_time,
                    "speedup": speedup,
                    "max_abs_error": max_abs_error,
                    "passed": passed,
                }
            )
    finally:
        if temporary_input and os.path.exists(temporary_input):
            os.unlink(temporary_input)

    print()
    print("| Case | Baseline (s) | Experiment (s) | Speedup | Max abs error | Status |")
    print("|---|---:|---:|---:|---:|---|")
    for row in rows:
        error = (
            "N/A"
            if row["max_abs_error"] is None
            else f"{row['max_abs_error']:.9g}"
        )
        status = (
            "N/A" if row["passed"] is None else ("PASS" if row["passed"] else "FAIL")
        )
        print(
            f"| {row['name']} | {row['baseline']:.6f} | "
            f"{row['experiment']:.6f} | {row['speedup']:.3f}x | "
            f"{error} | {status} |"
        )


if __name__ == "__main__":
    main()
