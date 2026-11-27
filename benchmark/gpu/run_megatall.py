#!/usr/bin/env python3
"""Time and check the experimental Mojo GPU LDLT on MegaTall Model1."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / "benchmark/.tmp/megatall_model1_40steps/megatall_model1_40steps.json"
NUMERIC_TIMES = re.compile(r"LDLT numeric us:\s+(\d+)")
SYMBOLIC_FLOPS = re.compile(r"sparse LDLT analysis: .*flops=(\d+)")


def run_case(
    output: Path,
    backend: str,
    *,
    compute_only: bool,
    ordering: Path | None = None,
) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(STRUT_THREADS="8", STRUT_LDLT_TIMING="1", STRUT_VERBOSE="1")
    env.pop("STRUT_LDLT_NESDIS", None)
    if ordering:
        env["STRUT_LDLT_NESDIS"] = str(ordering)
    solver = ROOT / "build/strut" / ("strut_gpu" if backend == "gpu" else "strut")
    command = [
        str(solver),
        "--input",
        str(CASE),
        "--output",
        str(output),
        "--backend",
        backend,
    ]
    if compute_only:
        command.append("--compute-only")

    label = backend.upper()
    print(f"{label} {'timing' if compute_only else 'recorders'} → {output}", flush=True)
    log_path = output / "solver.log"
    with log_path.open("w") as log:
        result = subprocess.run(
            command, check=False, env=env, cwd=ROOT, stdout=log, stderr=log
        )
    log_text = log_path.read_text()
    if result.returncode:
        print("\n".join(log_text.splitlines()[-20:]), file=sys.stderr)
        result.check_returncode()
    analysis_us = int((output / "analysis_time_us.txt").read_text())
    factors = [int(match) for match in NUMERIC_TIMES.findall(log_text)]
    if len(factors) != 40:
        raise RuntimeError(
            f"expected 40 LDLT factorizations, saw {len(factors)} in {log_path}"
        )
    work = set(SYMBOLIC_FLOPS.findall(log_text))
    if len(work) != 1:
        raise RuntimeError(
            f"expected one symbolic flop count across both stages in {log_path}"
        )
    timing = {
        "analysis_s": analysis_us / 1e6,
        "numeric_ldlt_s": sum(factors) / 1e6,
        "symbolic_tflop": int(work.pop()) * len(factors) / 1e12,
    }

    phases = json.loads((output / "phase_times_us.json").read_text())
    expected = len(factors) if backend == "gpu" else 0
    if phases["gpu_factorizations"] != expected:
        raise RuntimeError(f"expected {expected} GPU factorizations. See {log_path}")

    print(json.dumps(timing), flush=True)
    return timing


def compare_recorders(cpu: Path, gpu: Path) -> dict:
    """Compare per-recorder columns relative to the largest response in each group."""
    names = {path.name for path in cpu.glob("*.out")}
    if not names or names != {path.name for path in gpu.glob("*.out")}:
        raise RuntimeError(
            "CPU and GPU must produce the same nonempty set of recorders"
        )
    peaks: dict[tuple[str, int], float] = {}
    errors: dict[tuple[str, int], float] = {}
    for name in sorted(names):
        expected = np.loadtxt(cpu / name, ndmin=2)
        actual = np.loadtxt(gpu / name, ndmin=2)
        if expected.shape != actual.shape or expected.shape[0] != 40:
            raise RuntimeError(
                f"{name} has CPU shape {expected.shape} and GPU shape {actual.shape}"
            )
        if not np.isfinite(expected).all() or not np.isfinite(actual).all():
            raise RuntimeError(f"nonfinite recorder value in {name}")
        group = re.sub(r"_(node|ele|i)\d+.*$", "", Path(name).stem)
        for column in range(expected.shape[1]):
            key = group, column
            peak = float(np.max(np.abs(expected[:, column])))
            error = float(np.max(np.abs(expected[:, column] - actual[:, column])))
            peaks[key] = max(peaks.get(key, 0.0), peak)
            errors[key] = max(errors.get(key, 0.0), error)
    group_peaks: dict[tuple[str, int], float] = defaultdict(float)
    for (group, column), peak in peaks.items():
        key = group, column // 3
        group_peaks[key] = max(group_peaks[key], peak)
    # Roundoff can dominate physically zero response columns.
    worst = max(
        error / max(peak, 1e-6 * group_peaks[key[0], key[1] // 3])
        for key, error in errors.items()
        if (peak := peaks[key]) or group_peaks[key[0], key[1] // 3]
    )
    for key, error in errors.items():
        if not group_peaks[key[0], key[1] // 3] and error:
            raise RuntimeError(
                f"nonzero GPU response in an exactly zero CPU group: {key}"
            )
    return {"files": len(names), "worst_scaled_error": worst}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_dir", type=Path, help="new directory for benchmark runs"
    )
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--timing-only", action="store_true", help="skip the paired recorder runs"
    )
    parser.add_argument(
        "--ordering-library",
        type=Path,
        help="use the CHOLMOD NESDIS ordering for both runs",
    )
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    output = args.output_dir.resolve()
    ordering = (
        args.ordering_library.resolve(strict=True) if args.ordering_library else None
    )
    output.mkdir(parents=True, exist_ok=False)

    for script, options in (
        ("setup_reference_data.py", ["install", "--fixtures-only"]),
        ("build_megatall_benchmark_case.py", []),
        ("build_mojo_solver.py", []),
        ("build_mojo_solver.py", ["--gpu"]),
    ):
        subprocess.run(
            ["uv", "run", "python", f"scripts/{script}", *options], cwd=ROOT, check=True
        )

    peak_probe = output / "fp64_peak"
    subprocess.run(
        [
            "uv",
            "run",
            "mojo",
            "build",
            str(Path(__file__).with_name("fp64_peak.mojo").resolve()),
            "-o",
            str(peak_probe),
        ],
        cwd=ROOT,
        check=True,
    )

    samples: dict[str, list[dict]] = {"cpu": [], "gpu": []}
    for trial in range(1, args.repeat + 1):
        for name in ("cpu", "gpu"):
            samples[name].append(
                run_case(
                    output / f"{name}-{trial}",
                    name,
                    compute_only=True,
                    ordering=ordering,
                )
            )
    cpu_s = median(sample["analysis_s"] for sample in samples["cpu"])
    gpu_s = median(sample["analysis_s"] for sample in samples["gpu"])
    cpu_rest = median(
        sample["analysis_s"] - sample["numeric_ldlt_s"] for sample in samples["cpu"]
    )
    tflops = float(subprocess.check_output([str(peak_probe)]))
    ideal_compute_s = samples["cpu"][0]["symbolic_tflop"] / tflops
    report = {
        "ordering": "NESDIS" if ordering else "Strut default",
        "samples": samples,
        "gpu_speedup": cpu_s / gpu_s,
        "numeric_ldlt_only_speedup_ceiling": cpu_s / cpu_rest,
        "register_fp64_tflop_per_s": tflops,
        "idealized_fp64_analysis_s": cpu_rest + ideal_compute_s,
        "idealized_fp64_speedup": cpu_s / (cpu_rest + ideal_compute_s),
    }
    if not args.timing_only:
        cpu, gpu = output / "cpu-recorders", output / "gpu-recorders"
        run_case(cpu, "cpu", compute_only=False, ordering=ordering)
        run_case(gpu, "gpu", compute_only=False, ordering=ordering)
        report["parity"] = compare_recorders(cpu, gpu)
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not args.timing_only and report["parity"]["worst_scaled_error"] > 1e-5:
        raise SystemExit("GPU recorder parity exceeded 1e-5")


if __name__ == "__main__":
    main()
