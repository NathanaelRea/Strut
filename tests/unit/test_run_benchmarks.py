import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_BENCHMARKS_PATH = REPO_ROOT / "scripts" / "run_benchmarks.py"


def _load_run_benchmarks_module():
    module_name = "strut_run_benchmarks_test_module"
    spec = importlib.util.spec_from_file_location(module_name, RUN_BENCHMARKS_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


run_benchmarks = _load_run_benchmarks_module()


def _write_case(path: Path, enabled=True, status="active") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"schema_version": "1.0", "enabled": enabled, "status": status}) + "\n",
        encoding="utf-8",
    )


def test_load_case_enabled_allows_benchmark_cases(tmp_path: Path):
    bench_case = tmp_path / "bench.json"
    normal_disabled_case = tmp_path / "disabled.json"
    _write_case(bench_case, enabled=False, status="benchmark")
    _write_case(normal_disabled_case, enabled=False, status="active")

    assert run_benchmarks.load_case_enabled(bench_case) is True
    assert run_benchmarks.load_case_enabled(normal_disabled_case) is False


def test_discover_default_cases_includes_disabled_benchmark(monkeypatch, tmp_path: Path):
    validation_root = tmp_path / "validation"
    _write_case(
        validation_root / "enabled_case" / "enabled_case.json", enabled=True, status="active"
    )
    _write_case(
        validation_root / "bench_case" / "bench_case.json", enabled=False, status="benchmark"
    )
    _write_case(
        validation_root / "disabled_case" / "disabled_case.json", enabled=False, status="active"
    )

    monkeypatch.delenv("STRUT_RUN_ALL_CASES", raising=False)
    case_names = [case.name for case in run_benchmarks.discover_default_cases(validation_root)]
    assert case_names == ["bench_case", "enabled_case"]


def test_expand_case_patterns_deduplicates_and_sorts(tmp_path: Path):
    validation_root = tmp_path / "validation"
    _write_case(validation_root / "beta_case" / "beta_case.json")
    _write_case(validation_root / "alpha_case" / "alpha_case.json")

    cases = run_benchmarks.expand_case_patterns(
        validation_root, ["*case", "alpha_case"]
    )
    assert [case.name for case in cases] == ["alpha_case", "beta_case"]


def test_case_free_dofs_counts_boolean_and_index_constraints(tmp_path: Path):
    case_path = tmp_path / "case.json"
    case_path.write_text(
        json.dumps(
            {
                "model": {"ndf": 3},
                "nodes": [
                    {"id": 1, "constraints": [True, False, True]},
                    {"id": 2, "constraints": [1]},
                    {"id": 3},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    # Total DOFs = 3 nodes * 3 ndf = 9. Constrained = 2 (bool) + 1 (index list) = 3.
    assert run_benchmarks._case_free_dofs(case_path) == 6


def test_inject_opensees_timing_wraps_first_analyze_call():
    lines = ["wipe", "analyze 10", "analyze 5"]
    timed = run_benchmarks._inject_opensees_timing(lines, "analysis_time_us.txt")

    first_analyze_index = timed.index("analyze 10")
    assert timed[first_analyze_index - 1] == "set __strut_t0 [clock microseconds]"
    assert timed[first_analyze_index + 1] == "set __strut_t1 [clock microseconds]"
    assert timed.count("set __strut_t0 [clock microseconds]") == 1
    assert "analyze 5" in timed


def test_inject_opensees_timing_requires_analyze_or_eigen():
    with pytest.raises(ValueError, match="failed to inject timing"):
        run_benchmarks._inject_opensees_timing(["wipe", "puts ok"], "analysis_time_us.txt")


def test_tcl_uses_eigen_detects_eigen_command(tmp_path: Path):
    tcl = tmp_path / "case.tcl"
    tcl.write_text("set vals [eigen 2]\n", encoding="utf-8")
    assert run_benchmarks._tcl_uses_eigen(tcl) is True


def test_tcl_uses_eigen_false_without_eigen(tmp_path: Path):
    tcl = tmp_path / "case.tcl"
    tcl.write_text("analyze 1\n", encoding="utf-8")
    assert run_benchmarks._tcl_uses_eigen(tcl) is False


def test_compare_mode_shape_vectors_tolerates_sign_flip():
    ok, errors = run_benchmarks._compare_mode_shape_vectors([1.0, 2.0], [-2.0, -4.0])
    assert ok is True
    assert errors == []
