import importlib.util
import sys
from pathlib import Path


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


def test_build_argument_parser_treats_bare_cases_as_interactive():
    parser = run_tests.build_argument_parser()

    args = parser.parse_args(["--cases"])

    assert args.cases == [run_tests.INTERACTIVE_CASE_SENTINEL]


def test_main_forwards_profile_and_bare_cases(monkeypatch):
    calls = []

    def fake_run(cmd, env=None, verbose=False):
        calls.append(cmd[:])

    monkeypatch.setattr(run_tests, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_tests.py", "--skip-build", "--profile", "asdf", "--cases"],
    )

    assert run_tests.main() == 0

    benchmark_cmd = next(cmd for cmd in calls if "scripts/run_benchmarks.py" in cmd)
    assert benchmark_cmd == [
        "uv",
        "run",
        "scripts/run_benchmarks.py",
        "--engine",
        "strut",
        "--no-archive",
        "--cases",
        "--profile",
        "asdf",
    ]


def test_main_forwards_explicit_benchmark_case(monkeypatch):
    calls = []

    def fake_run(cmd, env=None, verbose=False):
        calls.append(cmd[:])

    monkeypatch.setattr(run_tests, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_tests.py", "--skip-build", "--cases", "elastic_beam_cantilever"],
    )

    assert run_tests.main() == 0

    benchmark_cmd = next(cmd for cmd in calls if "scripts/run_benchmarks.py" in cmd)
    assert benchmark_cmd == [
        "uv",
        "run",
        "scripts/run_benchmarks.py",
        "--engine",
        "strut",
        "--no-archive",
        "--cases",
        "elastic_beam_cantilever",
        "--output-root",
        str(REPO_ROOT / "benchmark" / "results"),
    ]
