import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_TESTS_PATH = REPO_ROOT / "run_tests.py"


def _load_run_tests_module():
    module_name = "strut_run_tests_test_module"
    spec = importlib.util.spec_from_file_location(module_name, RUN_TESTS_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


run_tests = _load_run_tests_module()


def test_main_runs_build_unit_and_parity(monkeypatch):
    calls = []

    def fake_check_call(cmd, env=None):
        calls.append((cmd, dict(env) if env is not None else None))

    monkeypatch.setattr(run_tests.subprocess, "check_call", fake_check_call)
    monkeypatch.setattr(sys, "argv", ["run_tests.py"])

    assert run_tests.main() == 0

    assert [cmd for cmd, _ in calls] == [
        [str(REPO_ROOT / "scripts" / "build_mojo_solver.sh")],
        ["uv", "run", "pytest", "-q", "tests/unit", "tests/validation/test_json_cases.py"],
        ["uv", "run", "pytest", "-q", "tests/validation/test_parity_cases.py"],
    ]


def test_main_passes_case_filters_and_all_flag(monkeypatch, tmp_path: Path):
    calls = []

    def fake_check_call(cmd, env=None):
        calls.append((cmd, dict(env) if env is not None else None))

    case_json = tmp_path / "case_a.json"
    monkeypatch.setattr(run_tests.subprocess, "check_call", fake_check_call)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_tests.py",
            "--case",
            "elastic_beam_cantilever",
            "--case",
            str(case_json),
            "--all",
            "--verbose",
        ],
    )

    assert run_tests.main() == 0

    parity_env = calls[-1][1]
    assert parity_env is not None
    assert parity_env["STRUT_RUN_ALL_CASES"] == "1"
    assert (
        parity_env["STRUT_PARITY_CASES"]
        == f"elastic_beam_cantilever,{case_json}"
    )
    assert parity_env["STRUT_VERBOSE"] == "1"


def test_main_runs_benchmark_and_compare_gate(monkeypatch, tmp_path: Path):
    calls = []

    def fake_check_call(cmd, env=None):
        calls.append((cmd, dict(env) if env is not None else None))

    baseline = tmp_path / "baseline.json"
    output_root = tmp_path / "candidate"
    monkeypatch.setattr(run_tests.subprocess, "check_call", fake_check_call)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_tests.py",
            "--skip-build",
            "--benchmark-suite",
            "opt_fast_v1",
            "--benchmark-baseline",
            str(baseline),
            "--benchmark-output-root",
            str(output_root),
            "--require-improvement",
            "opensees_example_rc_frame_earthquake=10",
        ],
    )

    assert run_tests.main() == 0

    assert [cmd for cmd, _ in calls] == [
        ["uv", "run", "pytest", "-q", "tests/unit", "tests/validation/test_json_cases.py"],
        ["uv", "run", "pytest", "-q", "tests/validation/test_parity_cases.py"],
        [
            "uv",
            "run",
            "scripts/run_benchmarks.py",
            "--benchmark-suite",
            "opt_fast_v1",
            "--engine",
            "strut",
            "--output-root",
            str(output_root),
            "--no-archive",
        ],
        [
            "uv",
            "run",
            "scripts/compare_benchmarks.py",
            str(baseline),
            str(output_root / "summary.json"),
            "--max-regression-pct",
            "5.0",
            "--min-regression-us",
            "50.0",
            "--require-improvement",
            "opensees_example_rc_frame_earthquake=10",
        ],
    ]


def test_main_requires_suite_when_baseline_is_set(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_tests.py",
            "--benchmark-baseline",
            "/tmp/baseline.json",
        ],
    )

    with pytest.raises(SystemExit):
        run_tests.main()
