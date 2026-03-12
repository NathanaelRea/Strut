#!/usr/bin/env python3
import argparse
import os
import subprocess
from pathlib import Path
from typing import Iterable, List

INTERACTIVE_CASE_SENTINEL = "__interactive_case_selection__"


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


def _benchmark_output_root(
    repo_root: Path, configured_root: str | None, profile_root: str | None
) -> Path:
    if configured_root:
        return Path(configured_root)
    if profile_root:
        return repo_root / "benchmark" / "results-profile"
    return repo_root / "benchmark" / "results"


def _solver_binary_path(repo_root: Path, env: dict[str, str]) -> Path:
    if env.get("STRUT_PROFILE") == "1":
        return repo_root / "build" / "strut" / "strut_profile"
    return repo_root / "build" / "strut" / "strut"


def _add_pytest_workers(cmd: List[str], workers: int | str | None) -> List[str]:
    if workers is None:
        return cmd
    worker_value = str(workers).strip()
    if not worker_value or worker_value == "1":
        return cmd
    return [*cmd, "-n", worker_value]


def _append_benchmark_case_args(cmd: List[str], cases: list[str] | None) -> List[str]:
    if not cases:
        return cmd
    for case in cases:
        cmd.append("--cases")
        if case != INTERACTIVE_CASE_SENTINEL:
            cmd.append(case)
    return cmd


def build_argument_parser() -> argparse.ArgumentParser:
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
        "--cases",
        action="append",
        nargs="?",
        const=INTERACTIVE_CASE_SENTINEL,
        default=None,
        help="Run benchmark cases after tests. Repeatable. Pass with no value to pick one via fzf.",
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
        help="Engine selection for benchmark runs.",
    )
    parser.add_argument(
        "--benchmark-output-root",
        default=None,
        help="Results root for benchmark runs (default: benchmark/results, or benchmark/results-profile with --profile).",
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
        "--profile",
        default=None,
        metavar="DIR",
        help="Emit benchmark speedscope profiles to DIR.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print commands as they run.",
    )
    parser.add_argument(
        "--pytest-workers",
        default="8",
        help=(
            "Worker count for unit/json pytest runs. Use an integer or 'auto'. "
            "Default: 8."
        ),
    )
    parser.add_argument(
        "--parity-workers",
        default="8",
        help=(
            "Worker count for parity pytest runs. Use an integer or 'auto'. "
            "Default: 8."
        ),
    )
    return parser


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    benchmark_requested = bool(args.benchmark_suite or args.cases or args.profile)

    if args.benchmark_baseline and not benchmark_requested:
        parser.error(
            "--benchmark-baseline requires a benchmark run via --benchmark-suite, --cases, or --profile."
        )

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
        _add_pytest_workers(
            ["uv", "run", "pytest", "-q", "tests/unit", "tests/validation/test_json_cases.py"],
            args.pytest_workers,
        ),
        env=env,
        verbose=args.verbose,
    )

    parity_env = env.copy()
    if args.all:
        parity_env["STRUT_RUN_ALL_CASES"] = "1"
    if args.case:
        parity_env["STRUT_PARITY_CASES"] = ",".join(_normalize_case_args(args.case))
    run(
        _add_pytest_workers(
            ["uv", "run", "pytest", "-q", "tests/validation/test_parity_cases.py"],
            args.parity_workers,
        ),
        env=parity_env,
        verbose=args.verbose,
    )

    if benchmark_requested:
        output_root = _benchmark_output_root(
            repo_root, args.benchmark_output_root, args.profile
        )
        benchmark_cmd = [
            "uv",
            "run",
            "scripts/run_benchmarks.py",
            "--engine",
            args.benchmark_engine,
            "--no-archive",
        ]
        if args.benchmark_suite:
            benchmark_cmd.extend(["--benchmark-suite", args.benchmark_suite])
        benchmark_cmd = _append_benchmark_case_args(benchmark_cmd, args.cases)
        if args.benchmark_output_root or not args.profile:
            benchmark_cmd.extend(["--output-root", str(output_root)])
        if args.profile:
            benchmark_cmd.extend(["--profile", args.profile])
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
