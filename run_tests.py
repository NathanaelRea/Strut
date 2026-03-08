#!/usr/bin/env python3
import argparse
import os
import subprocess
from pathlib import Path
from typing import Iterable, List


def run(cmd: List[str], env=None, verbose: bool = False) -> None:
    if verbose:
        print("+", " ".join(cmd))
    subprocess.check_call(cmd, env=env)


def _normalize_case_args(cases: Iterable[str]) -> List[str]:
    normalized: List[str] = []
    for case in cases:
        path = Path(case)
        if path.suffix == ".json":
            normalized.append(str(path))
        else:
            normalized.append(case)
    return normalized


def _benchmark_output_root(repo_root: Path, configured_root: str | None) -> Path:
    if configured_root:
        return Path(configured_root)
    return repo_root / "benchmark" / "results"


def _solver_binary_path(repo_root: Path, env: dict[str, str]) -> Path:
    if env.get("STRUT_PROFILE") == "1":
        return repo_root / "build" / "strut" / "strut_profile"
    return repo_root / "build" / "strut" / "strut"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the solver, run tests, and optionally run benchmark gates."
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Run parity for specific case names or JSON paths. Repeatable.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Include disabled parity cases.",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip the zero-warning Mojo build step.",
    )
    parser.add_argument(
        "--benchmark-suite",
        default=None,
        help="Optional benchmark suite name to run after tests.",
    )
    parser.add_argument(
        "--benchmark-engine",
        choices=("both", "opensees", "strut"),
        default="strut",
        help="Engine selection for --benchmark-suite.",
    )
    parser.add_argument(
        "--benchmark-output-root",
        default=None,
        help="Results root for benchmark runs (default: benchmark/results).",
    )
    parser.add_argument(
        "--benchmark-baseline",
        default=None,
        help="Baseline summary.json to compare against after running benchmarks.",
    )
    parser.add_argument(
        "--max-regression-pct",
        type=float,
        default=5.0,
        help="Failure threshold for benchmark regressions.",
    )
    parser.add_argument(
        "--min-regression-us",
        type=float,
        default=50.0,
        help="Ignore smaller absolute regressions during benchmark comparison.",
    )
    parser.add_argument(
        "--require-improvement",
        action="append",
        default=[],
        help="Required benchmark improvement in CASE=PCT form. Repeatable.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print commands as they run.",
    )
    args = parser.parse_args()

    if args.benchmark_baseline and not args.benchmark_suite:
        parser.error("--benchmark-baseline requires --benchmark-suite.")

    repo_root = Path(__file__).resolve().parent
    env = os.environ.copy()
    if args.verbose:
        env["STRUT_VERBOSE"] = "1"

    if not args.skip_build:
        run(
            [
                "uv",
                "run",
                "python",
                str(repo_root / "scripts" / "build_mojo_solver.py"),
            ],
            env=env,
            verbose=args.verbose,
        )
        env["STRUT_MOJO_BIN"] = str(_solver_binary_path(repo_root, env))
    elif "STRUT_MOJO_BIN" not in env:
        solver_path = _solver_binary_path(repo_root, env)
        if solver_path.exists():
            env["STRUT_MOJO_BIN"] = str(solver_path)

    run(
        ["uv", "run", "pytest", "-q", "tests/unit", "tests/validation/test_json_cases.py"],
        env=env,
        verbose=args.verbose,
    )

    parity_env = env.copy()
    if args.all:
        parity_env["STRUT_RUN_ALL_CASES"] = "1"
    if args.case:
        parity_env["STRUT_PARITY_CASES"] = ",".join(_normalize_case_args(args.case))
    run(
        ["uv", "run", "pytest", "-q", "tests/validation/test_parity_cases.py"],
        env=parity_env,
        verbose=args.verbose,
    )

    if args.benchmark_suite:
        output_root = _benchmark_output_root(repo_root, args.benchmark_output_root)
        benchmark_cmd = [
            "uv",
            "run",
            "scripts/run_benchmarks.py",
            "--benchmark-suite",
            args.benchmark_suite,
            "--engine",
            args.benchmark_engine,
            "--output-root",
            str(output_root),
            "--no-archive",
        ]
        run(benchmark_cmd, env=env, verbose=args.verbose)

        if args.benchmark_baseline:
            compare_cmd = [
                "uv",
                "run",
                "scripts/compare_benchmarks.py",
                args.benchmark_baseline,
                str(output_root / "summary.json"),
                "--max-regression-pct",
                str(args.max_regression_pct),
                "--min-regression-us",
                str(args.min_regression_us),
            ]
            for item in args.require_improvement:
                compare_cmd.extend(["--require-improvement", item])
            run(compare_cmd, env=env, verbose=args.verbose)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
