import importlib.util
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

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
        json.dumps(
            {
                "schema_version": "1.0",
                "enabled": enabled,
                "status": status,
                "model": {"ndm": 2, "ndf": 3},
                "nodes": [],
                "elements": [],
                "recorders": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_direct_tcl_case(
    path: Path,
    entry_tcl: Path,
    enabled=True,
    status="active",
    benchmark_size=None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": path.parent.name,
        "entry_tcl": str(entry_tcl.resolve()),
        "enabled": enabled,
        "status": status,
    }
    if benchmark_size is not None:
        payload["benchmark_size"] = benchmark_size
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


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
        validation_root / "elastic_enabled_case" / "elastic_enabled_case.json",
        enabled=True,
        status="active",
    )
    _write_case(
        validation_root / "bench_case" / "bench_case.json",
        enabled=False,
        status="benchmark",
    )
    _write_case(
        validation_root / "disabled_case" / "disabled_case.json",
        enabled=False,
        status="active",
    )

    monkeypatch.delenv("STRUT_RUN_ALL_CASES", raising=False)
    case_names = [
        case.name for case in run_benchmarks.discover_default_cases(validation_root)
    ]
    assert case_names == ["elastic_enabled_case"]


def test_discover_default_cases_includes_enabled_direct_tcl_case(tmp_path: Path):
    validation_root = tmp_path / "validation"
    entry_tcl = tmp_path / "examples" / "ElasticExample.tcl"
    entry_tcl.parent.mkdir(parents=True, exist_ok=True)
    entry_tcl.write_text("puts ok\n", encoding="utf-8")
    _write_direct_tcl_case(
        validation_root / "opensees_example_2d_elastic_cantileaver_column" / "direct_tcl_case.json",
        entry_tcl=entry_tcl,
        enabled=True,
        status="active",
    )

    cases = run_benchmarks.discover_default_cases(validation_root)

    assert [case.name for case in cases] == ["opensees_example_2d_elastic_cantileaver_column"]
    assert cases[0].tcl_path == entry_tcl.resolve()
    assert cases[0].json_path is None


def test_discover_default_cases_ignores_non_case_json(tmp_path: Path):
    validation_root = tmp_path / "validation"
    case_dir = validation_root / "elastic_real_case"
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "elastic_real_case.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "enabled": True,
                "model": {"ndm": 2, "ndf": 3},
                "nodes": [],
                "elements": [],
                "recorders": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    artifact_dir = validation_root / "phase_times_us"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "phase_times_us.json").write_text(
        json.dumps({"analysis_us": 123.0}) + "\n",
        encoding="utf-8",
    )

    case_names = [
        case.name for case in run_benchmarks.discover_default_cases(validation_root)
    ]
    assert case_names == ["elastic_real_case"]


def test_discover_all_cases_includes_disabled_cases(tmp_path: Path):
    validation_root = tmp_path / "validation"
    _write_case(
        validation_root / "enabled_case" / "enabled_case.json",
        enabled=True,
        status="active",
    )
    _write_case(
        validation_root / "disabled_case" / "disabled_case.json",
        enabled=False,
        status="active",
    )
    _write_case(
        validation_root / "bench_case" / "bench_case.json",
        enabled=False,
        status="benchmark",
    )

    case_names = [
        case.name for case in run_benchmarks.discover_all_cases(validation_root)
    ]
    assert case_names == ["bench_case", "disabled_case", "enabled_case"]


def test_discover_all_cases_includes_direct_tcl_cases(tmp_path: Path):
    validation_root = tmp_path / "validation"
    entry_tcl = tmp_path / "examples" / "Ex1a.Canti2D.EQ.modif.tcl"
    entry_tcl.parent.mkdir(parents=True, exist_ok=True)
    entry_tcl.write_text("puts ok\n", encoding="utf-8")
    _write_direct_tcl_case(
        validation_root / "direct_case" / "direct_tcl_case.json",
        entry_tcl=entry_tcl,
        enabled=False,
        status="active",
    )

    cases = run_benchmarks.discover_all_cases(validation_root)

    assert [case.name for case in cases] == ["direct_case"]
    assert cases[0].tcl_path == entry_tcl.resolve()


def test_expand_case_patterns_deduplicates_and_sorts(tmp_path: Path):
    validation_root = tmp_path / "validation"
    _write_case(validation_root / "beta_case" / "beta_case.json")
    _write_case(validation_root / "alpha_case" / "alpha_case.json")

    cases = run_benchmarks.expand_case_patterns(
        validation_root, ["*case", "alpha_case"]
    )
    assert [case.name for case in cases] == ["alpha_case", "beta_case"]


def test_resolve_case_from_name_supports_direct_tcl_manifest(tmp_path: Path):
    validation_root = tmp_path / "validation"
    entry_tcl = tmp_path / "examples" / "case.tcl"
    entry_tcl.parent.mkdir(parents=True, exist_ok=True)
    entry_tcl.write_text("puts ok\n", encoding="utf-8")
    manifest = validation_root / "direct_case" / "direct_tcl_case.json"
    _write_direct_tcl_case(manifest, entry_tcl=entry_tcl, benchmark_size="medium")

    case = run_benchmarks.resolve_case_from_name(validation_root, "direct_case")

    assert case is not None
    assert case.name == "direct_case"
    assert case.tcl_path == entry_tcl.resolve()
    assert case.benchmark_size == "medium"


def test_opt_full_suite_includes_new_ex9_direct_tcl_benchmarks():
    suite = run_benchmarks.BENCHMARK_SUITES["opt_full_v1"]

    assert "opensees_example_ex9_moment_curvature_2d" in suite
    assert "opensees_example_ex9_analyze_moment_curvature_2d" in suite


def test_write_solver_input_pickle_round_trips(tmp_path: Path):
    payload = {"model": {"ndm": 2, "ndf": 3}, "recorders": []}
    out_path = tmp_path / "solver.pkl"

    result = run_benchmarks._write_solver_input_pickle(payload, out_path)

    assert result == out_path
    assert out_path.exists()
    assert pickle.loads(out_path.read_bytes()) == payload


def test_ensure_direct_tcl_case_artifacts_creates_canonical_reference(
    monkeypatch, tmp_path: Path
):
    repo_root = tmp_path / "repo"
    case_root = (
        repo_root
        / "tests"
        / "validation"
        / "opensees_example_2d_elastic_cantileaver_column"
    )
    manifest = case_root / "direct_tcl_case.json"
    entry_tcl = tmp_path / "examples" / "Ex1a.Canti2D.EQ.modif.tcl"
    entry_tcl.parent.mkdir(parents=True, exist_ok=True)
    entry_tcl.write_text("puts ok\n", encoding="utf-8")
    _write_direct_tcl_case(manifest, entry_tcl=entry_tcl)
    case = run_benchmarks._direct_tcl_case_spec(manifest)

    calls = []
    monkeypatch.setitem(
        sys.modules,
        "tcl_to_strut",
        SimpleNamespace(
            convert_tcl_to_solver_input=lambda entry, repo_root, compute_only=False: {
                "schema_version": "1.0",
                "metadata": {"name": "generated", "units": "unknown"},
                "model": {"ndm": 2, "ndf": 3},
                "nodes": [],
                "elements": [],
                "recorders": [
                    {
                        "type": "node_displacement",
                        "nodes": [1],
                        "output": "disp",
                        "raw_path": "Data/Disp.out",
                        "include_time": True,
                    }
                ],
            }
        ),
    )

    def fake_run(cmd, env=None, verbose=False, capture_on_error=False):
        calls.append(cmd)
        if "run_opensees_wine.sh" in cmd[0]:
            out_dir = Path(cmd[-1])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "Data").mkdir(parents=True, exist_ok=True)
            (out_dir / "Data" / "Disp.out").write_text(
                "0.0 1.0\n0.1 2.0\n", encoding="utf-8"
            )

    monkeypatch.setattr(run_benchmarks, "run", fake_run)

    case_data = run_benchmarks._ensure_direct_tcl_case_artifacts(
        case,
        repo_root=repo_root,
        env={},
        verbose=False,
    )

    assert case_data["recorders"][0]["output"] == "disp"
    assert (case_root / "reference" / "disp_node1.out").read_text(encoding="utf-8") == (
        "1.0\n2.0\n"
    )
    assert len(calls) == 1
    assert "run_opensees_wine.sh" in calls[0][0]


def test_ensure_direct_tcl_case_artifacts_reuses_existing_canonical_reference(
    monkeypatch, tmp_path: Path
):
    repo_root = tmp_path / "repo"
    case_root = repo_root / "tests" / "validation" / "direct_case"
    manifest = case_root / "direct_tcl_case.json"
    entry_tcl = tmp_path / "examples" / "case.tcl"
    entry_tcl.parent.mkdir(parents=True, exist_ok=True)
    entry_tcl.write_text("puts ok\n", encoding="utf-8")
    _write_direct_tcl_case(manifest, entry_tcl=entry_tcl)
    (case_root / "reference").mkdir(parents=True, exist_ok=True)
    (case_root / "reference" / "disp_node1.out").write_text("1.0\n", encoding="utf-8")
    case = run_benchmarks._direct_tcl_case_spec(manifest)

    calls = []
    monkeypatch.setitem(
        sys.modules,
        "tcl_to_strut",
        SimpleNamespace(
            convert_tcl_to_solver_input=lambda entry, repo_root, compute_only=False: {
                "schema_version": "1.0",
                "metadata": {"name": "generated", "units": "unknown"},
                "model": {"ndm": 2, "ndf": 3},
                "nodes": [],
                "elements": [],
                "recorders": [
                    {
                        "type": "node_displacement",
                        "nodes": [1],
                        "output": "disp",
                    }
                ],
            }
        ),
    )

    def fake_run(cmd, env=None, verbose=False, capture_on_error=False):
        calls.append(cmd)

    monkeypatch.setattr(run_benchmarks, "run", fake_run)

    case_data = run_benchmarks._ensure_direct_tcl_case_artifacts(
        case,
        repo_root=repo_root,
        env={},
        verbose=False,
    )

    assert case_data["recorders"][0]["output"] == "disp"
    assert calls == []


def test_filter_cases_by_enabled_counts_disabled_and_skipped(tmp_path: Path):
    validation_root = tmp_path / "validation"
    enabled_case = validation_root / "enabled_case" / "enabled_case.json"
    benchmark_disabled_case = (
        validation_root / "benchmark_disabled" / "benchmark_disabled.json"
    )
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

    filtered, disabled_selected, skipped_disabled = (
        run_benchmarks.filter_cases_by_enabled(case_specs, include_disabled=False)
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

    filtered, disabled_selected, skipped_disabled = (
        run_benchmarks.filter_cases_by_enabled(case_specs, include_disabled=True)
    )

    assert [case.name for case in filtered] == ["enabled_case", "disabled_case"]
    assert disabled_selected == 1
    assert skipped_disabled == 0


def test_case_free_dofs_counts_boolean_and_index_constraints(tmp_path: Path):
    case_data = {
        "model": {"ndf": 3},
        "nodes": [
            {"id": 1, "constraints": [True, False, True]},
            {"id": 2, "constraints": [1]},
            {"id": 3},
        ],
    }
    # Total DOFs = 3 nodes * 3 ndf = 9. Constrained = 2 (bool) + 1 (index list) = 3.
    assert run_benchmarks._case_free_dofs(case_data) == 6


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


def test_absolutize_time_series_paths_resolves_from_repo_root_when_needed(
    tmp_path: Path,
):
    case_json = tmp_path / "validation" / "case_b" / "case_b.json"
    case_json.parent.mkdir(parents=True, exist_ok=True)
    case_data = {
        "time_series": [
            {
                "type": "Path",
                "tag": 1,
                "values_path": "docs/agent-reference/OpenSeesExamplesBasic/time_history_analysis_of_a_2d_elastic_cantilever_column/A10000.tcl",
            }
        ]
    }

    run_benchmarks._absolutize_time_series_paths(case_data, case_json)

    resolved = Path(case_data["time_series"][0]["values_path"])
    expected = (
        REPO_ROOT
        / "docs/agent-reference/OpenSeesExamplesBasic/time_history_analysis_of_a_2d_elastic_cantilever_column/A10000.tcl"
    ).resolve()
    assert resolved.is_absolute()
    assert resolved == expected


def test_absolutize_time_series_paths_resolves_relative_to_source_example(
    tmp_path: Path,
):
    case_json = tmp_path / "validation" / "case_c" / "case_c.json"
    case_json.parent.mkdir(parents=True, exist_ok=True)
    example_dir = tmp_path / "examples"
    example_dir.mkdir(parents=True, exist_ok=True)
    example_tcl = example_dir / "example.tcl"
    example_tcl.write_text("# placeholder\n", encoding="utf-8")
    motion = example_dir / "gm.acc"
    motion.write_text("0.0 1.0\n", encoding="utf-8")
    case_data = {
        "source_example": str(example_tcl.resolve()),
        "time_series": [{"type": "Path", "tag": 1, "values_path": "gm.acc"}],
    }

    run_benchmarks._absolutize_time_series_paths(case_data, case_json)

    resolved = Path(case_data["time_series"][0]["values_path"])
    assert resolved.is_absolute()
    assert resolved == motion.resolve()


def test_inject_opensees_timing_wraps_all_analyze_calls_and_accumulates():
    lines = ["wipe", "analyze 10", "analyze 5"]
    timed = run_benchmarks._inject_opensees_timing(lines, "analysis_time_us.txt")

    assert timed[0] == "set __strut_analysis_us 0"
    assert timed.count("set __strut_t0 [clock microseconds]") == 2
    assert (
        timed.count("incr __strut_analysis_us [expr {$__strut_t1 - $__strut_t0}]") == 2
    )
    assert timed[-3] == 'set __strut_fp [open "analysis_time_us.txt" w]'
    assert timed[-2] == "puts $__strut_fp $__strut_analysis_us"
    assert timed[-1] == "close $__strut_fp"
    assert "analyze 10" in timed
    assert "analyze 5" in timed


def test_inject_opensees_timing_wraps_analyze_and_eigen_in_staged_patterns():
    lines = [
        "set strut_tr_ok [analyze 1 0.01]",
        "if {$strut_tr_ok != 0} {",
        "  set strut_tr_ok [analyze 1 0.01]",
        "}",
        "set modes [eigen 2]",
    ]
    timed = run_benchmarks._inject_opensees_timing(lines, "analysis_time_us.txt")

    assert timed.count("set __strut_t0 [clock microseconds]") == 3
    assert timed.count("set __strut_t1 [clock microseconds]") == 3
    assert (
        timed.count("incr __strut_analysis_us [expr {$__strut_t1 - $__strut_t0}]") == 3
    )


def test_inject_opensees_timing_requires_analyze_or_eigen():
    with pytest.raises(ValueError, match="failed to inject timing"):
        run_benchmarks._inject_opensees_timing(
            ["wipe", "puts ok"], "analysis_time_us.txt"
        )


def test_tcl_uses_eigen_detects_eigen_command(tmp_path: Path):
    tcl = tmp_path / "case.tcl"
    tcl.write_text("set vals [eigen 2]\n", encoding="utf-8")
    assert run_benchmarks._tcl_uses_eigen(tcl) is True


def test_tcl_uses_eigen_false_without_eigen(tmp_path: Path):
    tcl = tmp_path / "case.tcl"
    tcl.write_text("analyze 1\n", encoding="utf-8")
    assert run_benchmarks._tcl_uses_eigen(tcl) is False


def test_normalize_opensees_benchmark_outputs_writes_parity_filenames(tmp_path: Path):
    output_dir = tmp_path / "opensees_case"
    (output_dir / "Data").mkdir(parents=True, exist_ok=True)
    (output_dir / "Data" / "Disp.out").write_text(
        "0.0 1.0\n0.1 2.0\n", encoding="utf-8"
    )
    case_data = {
        "recorders": [
            {
                "type": "node_displacement",
                "nodes": [1],
                "output": "disp",
                "raw_path": "Data/Disp.out",
                "include_time": True,
            }
        ]
    }

    run_benchmarks._normalize_opensees_benchmark_outputs(case_data, output_dir)

    assert (output_dir / "disp_node1.out").read_text(encoding="utf-8") == "1.0\n2.0\n"


def test_compare_mode_shape_vectors_tolerates_sign_flip():
    ok, errors = run_benchmarks._compare_mode_shape_vectors([1.0, 2.0], [-2.0, -4.0])
    assert ok is True
    assert errors == []


def test_element_response_recorder_types_include_link_and_zero_length_outputs():
    assert run_benchmarks.ELEMENT_RESPONSE_RECORDER_TYPES == (
        "element_force",
        "element_local_force",
        "element_basic_force",
        "element_deformation",
    )


def test_summarize_parity_failures_compacts_paths_and_groups_by_case():
    failures = [
        "beam_case: missing OpenSees output: /tmp/results/opensees/beam_case/node_disp_node1.out",
        "beam_case: missing Mojo output: /tmp/results/strut/beam_case/node_disp_node1.out",
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
        '["node 1 mismatch at step 4", "dof 1: ref=1.000000e+00 got=2.000000e+00 abs=1.000e+00 rel=1.000e+00"]'
        in summary
    )
    assert (
        'Unsupported Recorder: ["unsupported recorder type: weird_recorder"]' in summary
    )
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


def test_summarize_parity_failures_reports_all_missing_strut_outputs():
    failures = [
        "frame_case: missing Mojo output: /tmp/results/strut/frame_case/a.out",
        "frame_case: missing Mojo output: /tmp/results/strut/frame_case/b.out",
    ]

    summary = run_benchmarks._summarize_parity_failures(failures)
    assert summary == [
        "Error: frame_case",
        "Missing all Mojo Outputs",
    ]


def test_analysis_is_transient_for_top_level_transient():
    analysis = {"type": "transient_nonlinear", "steps": 10, "dt": 0.01}
    assert run_benchmarks._analysis_is_transient(analysis) is True


def test_analysis_is_transient_for_staged_with_transient_stage():
    analysis = {
        "type": "staged",
        "stages": [
            {"analysis": {"type": "static_nonlinear", "steps": 5}},
            {"analysis": {"type": "transient_nonlinear", "steps": 10, "dt": 0.01}},
        ],
    }
    assert run_benchmarks._analysis_is_transient(analysis) is True


def test_analysis_is_transient_false_for_non_transient_staged():
    analysis = {
        "type": "staged",
        "stages": [
            {"analysis": {"type": "static_linear", "steps": 1}},
            {"analysis": {"type": "static_nonlinear", "steps": 5}},
        ],
    }
    assert run_benchmarks._analysis_is_transient(analysis) is False


def test_compare_transient_rows_max_abs_uses_peaks_not_first_mismatch():
    failures = []

    run_benchmarks._compare_transient_rows(
        ref_vals=[[0.0, 1.0], [1.0, 3.0]],
        strut_vals=[[0.0, 2.0], [1.0, 3.05]],
        label="case: node 1",
        failures=failures,
        rtol=0.05,
        atol=0.0,
        parity_mode="max_abs",
    )

    assert failures == []


def test_resolve_recorder_tolerance_prefers_per_recorder_override():
    rtol, atol = run_benchmarks._resolve_recorder_tolerance(
        "element_force",
        global_rtol=0.2,
        global_atol=0.001,
        has_global_override=True,
        per_recorder_overrides={"element_force": {"rtol": 0.3, "atol": 0.01}},
    )

    assert rtol == pytest.approx(0.3)
    assert atol == pytest.approx(0.01)
