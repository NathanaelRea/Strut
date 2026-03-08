#!/usr/bin/env python3
import argparse
import json
import statistics
import subprocess
import tempfile
import time
from pathlib import Path


def _build_repeated_input(input_path: Path, path_repetitions: int) -> Path:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    path = data.get("deformation_path")
    if not isinstance(path, list) or not path:
        raise SystemExit("input deformation_path must be a non-empty list")
    if path_repetitions > 1:
        data["deformation_path"] = path * path_repetitions

    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        prefix="strut_fiber_microbench_",
        delete=False,
        encoding="utf-8",
    )
    try:
        json.dump(data, tmp)
        tmp.write("\n")
        tmp.flush()
    finally:
        tmp.close()
    return Path(tmp.name)


def _run_once(repo_root: Path, input_path: Path, output_path: Path) -> float:
    cmd = [
        "uv",
        "run",
        str(repo_root / "scripts" / "run_strut_section_path.py"),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
    ]
    started = time.perf_counter()
    subprocess.check_call(cmd, cwd=repo_root)
    return (time.perf_counter() - started) * 1_000_000.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a small fiber-section microbenchmark through section_path."
    )
    parser.add_argument(
        "--input",
        default="benchmark/microbench/fiber_section2d_rc_rect.json",
        help="Section-path JSON input.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Measured process launches after warmup.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Unmeasured warmup launches.",
    )
    parser.add_argument(
        "--path-repetitions",
        type=int,
        default=200,
        help="How many times to repeat the deformation_path inside one launch.",
    )
    parser.add_argument(
        "--output-json",
        help="Optional path for a JSON summary.",
    )
    args = parser.parse_args()

    if args.repeats <= 0:
        raise SystemExit("--repeats must be > 0")
    if args.warmup < 0:
        raise SystemExit("--warmup must be >= 0")
    if args.path_repetitions <= 0:
        raise SystemExit("--path-repetitions must be > 0")

    repo_root = Path(__file__).resolve().parents[1]
    input_path = (repo_root / args.input).resolve()
    bench_input = _build_repeated_input(input_path, args.path_repetitions)
    output_path = Path(tempfile.mkstemp(suffix=".csv", prefix="strut_fiber_microbench_")[1])

    try:
        for _ in range(args.warmup):
            _run_once(repo_root, bench_input, output_path)

        samples_us = []
        for _ in range(args.repeats):
            samples_us.append(_run_once(repo_root, bench_input, output_path))

        bench_data = json.loads(bench_input.read_text(encoding="utf-8"))
        deformation_steps = len(bench_data["deformation_path"])
        summary = {
            "input": str(input_path),
            "path_repetitions": args.path_repetitions,
            "deformation_steps": deformation_steps,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "samples_us": samples_us,
            "min_us": min(samples_us),
            "median_us": statistics.median(samples_us),
            "mean_us": statistics.fmean(samples_us),
            "max_us": max(samples_us),
            "median_us_per_step": statistics.median(samples_us) / float(deformation_steps),
        }

        if args.output_json:
            output_json = (repo_root / args.output_json).resolve()
            output_json.parent.mkdir(parents=True, exist_ok=True)
            output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

        print(json.dumps(summary, indent=2))
    finally:
        bench_input.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
