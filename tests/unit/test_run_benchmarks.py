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


def test_load_case_enabled_uses_enabled_flag_only(tmp_path: Path):
    bench_case = tmp_path / "bench.json"
    normal_disabled_case = tmp_path / "disabled.json"
    enabled_bench_case = tmp_path / "enabled_bench.json"
    _write_case(bench_case, enabled=False, status="benchmark")
    _write_case(normal_disabled_case, enabled=False, status="active")
    _write_case(enabled_bench_case, enabled=True, status="benchmark")

    assert run_benchmarks.load_case_enabled(bench_case) is False
    assert run_benchmarks.load_case_enabled(normal_disabled_case) is False
    assert run_benchmarks.load_case_enabled(enabled_bench_case) is True


def test_discover_default_cases_excludes_disabled_cases(monkeypatch, tmp_path: Path):
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
    assert case_names == ["enabled_case"]


def test_discover_all_cases_includes_disabled_cases(tmp_path: Path):
    validation_root = tmp_path / "validation"
    _write_case(
        validation_root / "enabled_case" / "enabled_case.json", enabled=True, status="active"
    )
    _write_case(
        validation_root / "disabled_case" / "disabled_case.json", enabled=False, status="active"
    )
    _write_case(
        validation_root / "bench_case" / "bench_case.json", enabled=False, status="benchmark"
    )

    case_names = [case.name for case in run_benchmarks.discover_all_cases(validation_root)]
    assert case_names == ["bench_case", "disabled_case", "enabled_case"]


def test_expand_case_patterns_deduplicates_and_sorts(tmp_path: Path):
    validation_root = tmp_path / "validation"
    _write_case(validation_root / "beta_case" / "beta_case.json")
    _write_case(validation_root / "alpha_case" / "alpha_case.json")

    cases = run_benchmarks.expand_case_patterns(
        validation_root, ["*case", "alpha_case"]
    )
    assert [case.name for case in cases] == ["alpha_case", "beta_case"]


def test_filter_cases_by_enabled_counts_disabled_and_skipped(tmp_path: Path):
    validation_root = tmp_path / "validation"
    enabled_case = validation_root / "enabled_case" / "enabled_case.json"
    benchmark_disabled_case = validation_root / "benchmark_disabled" / "benchmark_disabled.json"
    disabled_case = validation_root / "disabled_case" / "disabled_case.json"
    _write_case(enabled_case, enabled=True, status="active")
    _write_case(benchmark_disabled_case, enabled=False, status="benchmark")
    _write_case(disabled_case, enabled=False, status="active")

    case_specs = [
        run_benchmarks.CaseSpec(name="enabled_case", json_path=enabled_case),
        run_benchmarks.CaseSpec(
            name="benchmark_disabled", json_path=benchmark_disabled_case
        ),
        run_benchmarks.CaseSpec(name="disabled_case", json_path=disabled_case),
    ]

    filtered, disabled_selected, skipped_disabled = run_benchmarks.filter_cases_by_enabled(
        case_specs, include_disabled=False
    )

    assert [case.name for case in filtered] == ["enabled_case"]
    assert disabled_selected == 2
    assert skipped_disabled == 2


def test_filter_cases_by_enabled_include_disabled_keeps_all_cases(tmp_path: Path):
    validation_root = tmp_path / "validation"
    enabled_case = validation_root / "enabled_case" / "enabled_case.json"
    disabled_case = validation_root / "disabled_case" / "disabled_case.json"
    _write_case(enabled_case, enabled=True, status="active")
    _write_case(disabled_case, enabled=False, status="active")

    case_specs = [
        run_benchmarks.CaseSpec(name="enabled_case", json_path=enabled_case),
        run_benchmarks.CaseSpec(name="disabled_case", json_path=disabled_case),
    ]

    filtered, disabled_selected, skipped_disabled = run_benchmarks.filter_cases_by_enabled(
        case_specs, include_disabled=True
    )

    assert [case.name for case in filtered] == ["enabled_case", "disabled_case"]
    assert disabled_selected == 1
    assert skipped_disabled == 0


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


def test_absolutize_time_series_paths_resolves_relative_values_path(tmp_path: Path):
    case_json = tmp_path / "validation" / "case_a" / "case_a.json"
    case_json.parent.mkdir(parents=True, exist_ok=True)
    motion = case_json.parent / "gm.acc"
    motion.write_text("0.0 1.0\n", encoding="utf-8")
    case_data = {"time_series": [{"type": "Path", "tag": 1, "values_path": "gm.acc"}]}

    run_benchmarks._absolutize_time_series_paths(case_data, case_json)

    resolved = Path(case_data["time_series"][0]["values_path"])
    assert resolved.is_absolute()
    assert resolved == motion.resolve()


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


def test_summarize_parity_failures_compacts_paths_and_groups_by_case():
    failures = [
        "beam_case: missing OpenSees output: /tmp/results/opensees/beam_case/node_disp_node1.out",
        "beam_case: missing Mojo output: /tmp/results/mojo/beam_case/node_disp_node1.out",
        "beam_case: node 1 mismatch at step 4",
        "  dof 1: ref=1.000000e+00 got=2.000000e+00 abs=1.000e+00 rel=1.000e+00",
        "beam_case: unsupported recorder type: weird_recorder",
    ]

    summary = run_benchmarks._summarize_parity_failures(failures)
    assert summary[0] == "Error: beam_case"
    assert 'Missing Opensees outputs: ["node_disp_node1.out"]' in summary
    assert 'Missing Mojo outputs: ["node_disp_node1.out"]' in summary
    assert (
        "Node Mismatch: "
        '[\"node 1 mismatch at step 4\", \"dof 1: ref=1.000000e+00 got=2.000000e+00 abs=1.000e+00 rel=1.000e+00\"]'
        in summary
    )
    assert 'Unsupported Recorder: ["unsupported recorder type: weird_recorder"]' in summary
    assert all("/tmp/results/" not in line for line in summary)


def test_summarize_parity_failures_reports_all_missing_opensees_outputs():
    failures = [
        "frame_case: missing OpenSees output: /tmp/results/opensees/frame_case/a.out",
        "frame_case: missing OpenSees output: /tmp/results/opensees/frame_case/b.out",
    ]

    summary = run_benchmarks._summarize_parity_failures(failures)
    assert summary == [
        "Error: frame_case",
        "Missing all Opensees Outputs",
    ]


def test_summarize_parity_failures_reports_all_missing_mojo_outputs():
    failures = [
        "frame_case: missing Mojo output: /tmp/results/mojo/frame_case/a.out",
        "frame_case: missing Mojo output: /tmp/results/mojo/frame_case/b.out",
    ]

    summary = run_benchmarks._summarize_parity_failures(failures)
    assert summary == [
        "Error: frame_case",
        "Missing all Mojo Outputs",
    ]
