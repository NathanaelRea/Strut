import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
import io

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_BENCHMARKS_PATH = REPO_ROOT / "scripts" / "run_benchmarks.py"
COMPARE_CASE_PATH = REPO_ROOT / "scripts" / "compare_case.py"


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


def _load_compare_case_module():
    module_name = "strut_compare_case_test_module"
    spec = importlib.util.spec_from_file_location(module_name, COMPARE_CASE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


compare_case = _load_compare_case_module()


def _write_case(path: Path, enabled=True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "enabled": enabled,
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
    benchmark_size=None,
    source_files=None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": path.parent.name,
        "entry_tcl": str(entry_tcl.resolve()),
        "enabled": enabled,
    }
    if benchmark_size is not None:
        payload["benchmark_size"] = benchmark_size
    if source_files is not None:
        payload["source_files"] = list(source_files)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_load_case_enabled_uses_enabled_flag_only(tmp_path: Path):
    bench_case = tmp_path / "bench.json"
    normal_disabled_case = tmp_path / "disabled.json"
    enabled_bench_case = tmp_path / "enabled_bench.json"
    _write_case(bench_case, enabled=False)
    _write_case(normal_disabled_case, enabled=False)
    _write_case(enabled_bench_case, enabled=True)

    assert run_benchmarks.load_case_enabled(bench_case) is False
    assert run_benchmarks.load_case_enabled(normal_disabled_case) is False
    assert run_benchmarks.load_case_enabled(enabled_bench_case) is True


def test_discover_default_cases_includes_all_cases_before_enabled_filter(
    monkeypatch, tmp_path: Path
):
    validation_root = tmp_path / "validation"
    _write_case(
        validation_root / "elastic_enabled_case" / "elastic_enabled_case.json",
        enabled=True,
    )
    _write_case(
        validation_root / "bench_case" / "bench_case.json",
        enabled=False,
    )
    _write_case(
        validation_root / "disabled_case" / "disabled_case.json",
        enabled=False,
    )

    monkeypatch.delenv("STRUT_RUN_ALL_CASES", raising=False)
    case_names = [
        case.name for case in run_benchmarks.discover_default_cases(validation_root)
    ]
    assert case_names == ["bench_case", "disabled_case", "elastic_enabled_case"]


def test_discover_default_cases_includes_enabled_direct_tcl_case(tmp_path: Path):
    validation_root = tmp_path / "validation"
    entry_tcl = tmp_path / "examples" / "ElasticExample.tcl"
    entry_tcl.parent.mkdir(parents=True, exist_ok=True)
    entry_tcl.write_text("puts ok\n", encoding="utf-8")
    _write_direct_tcl_case(
        validation_root / "opensees_example_2d_elastic_cantileaver_column" / "direct_tcl_case.json",
        entry_tcl=entry_tcl,
        enabled=True,
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
    )
    _write_case(
        validation_root / "disabled_case" / "disabled_case.json",
        enabled=False,
    )
    _write_case(
        validation_root / "bench_case" / "bench_case.json",
        enabled=False,
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
    )

    cases = run_benchmarks.discover_all_cases(validation_root)

    assert [case.name for case in cases] == ["direct_case"]
    assert cases[0].tcl_path == entry_tcl.resolve()


def test_resolve_engine_mode_defaults_to_both_without_mp():
    assert run_benchmarks._resolve_engine_mode(None, False, env={}) == "both"


def test_resolve_engine_mode_defaults_to_all_with_mp():
    assert run_benchmarks._resolve_engine_mode(None, True, env={}) == "all"


def test_resolve_engine_mode_uses_env_when_engine_unspecified():
    assert (
        run_benchmarks._resolve_engine_mode(
            None, False, env={"STRUT_BENCH_ENGINE": "strut"}
        )
        == "strut"
    )


def test_resolve_engine_mode_prefers_explicit_engine_over_mp_and_env():
    assert (
        run_benchmarks._resolve_engine_mode(
            "openseesmp", False, env={"STRUT_BENCH_ENGINE": "both"}
        )
        == "openseesmp"
    )


def test_engines_require_direct_tcl_parser_check_for_reference_engines():
    assert (
        run_benchmarks._engines_require_direct_tcl_parser_check(("opensees",))
        is True
    )
    assert (
        run_benchmarks._engines_require_direct_tcl_parser_check(("openseesmp",))
        is True
    )
    assert (
        run_benchmarks._engines_require_direct_tcl_parser_check(("strut",)) is False
    )
    assert (
        run_benchmarks._engines_require_direct_tcl_parser_check(("opensees", "strut"))
        is True
    )


def test_build_argument_parser_treats_bare_cases_as_interactive():
    parser = run_benchmarks.build_argument_parser()

    args = parser.parse_args(["--cases"])

    assert args.cases == [run_benchmarks.INTERACTIVE_CASE_SENTINEL]


def test_select_case_with_fzf_returns_selected_case_name(monkeypatch, tmp_path: Path):
    validation_root = tmp_path / "validation"
    selected_case = run_benchmarks.CaseSpec(
        name="alpha_case",
        json_path=validation_root / "alpha_case" / "alpha_case.json",
        metadata_path=validation_root / "alpha_case" / "alpha_case.json",
    )
    call = {}

    monkeypatch.setattr(run_benchmarks.shutil, "which", lambda name: "/usr/bin/fzf")
    monkeypatch.setattr(
        run_benchmarks,
        "_interactive_case_options",
        lambda root, include_disabled: [("alpha_case", selected_case)],
    )
    monkeypatch.setattr(run_benchmarks.Path, "open", lambda self, *args, **kwargs: io.StringIO())

    class FakePopen:
        returncode = 0

        def __init__(self, *args, **kwargs):
            call["kwargs"] = kwargs

        def communicate(self, text):
            call["input"] = text
            return ("alpha_case\n", "")

    def fake_popen(*args, **kwargs):
        call["kwargs"] = kwargs
        return FakePopen(*args, **kwargs)

    monkeypatch.setattr(run_benchmarks.subprocess, "Popen", fake_popen)

    selected = run_benchmarks._select_case_with_fzf(
        validation_root, include_disabled=False
    )

    assert selected == "alpha_case"
    assert call["kwargs"]["stdin"] == run_benchmarks.subprocess.PIPE
    assert call["kwargs"]["stdout"] == run_benchmarks.subprocess.PIPE
    assert call["kwargs"]["stderr"] is not None
    assert call["input"] == "alpha_case\n"


def test_resolve_case_args_expands_interactive_sentinel(monkeypatch, tmp_path: Path):
    validation_root = tmp_path / "validation"
    monkeypatch.setattr(
        run_benchmarks,
        "_select_case_with_fzf",
        lambda root, include_disabled: "picked_case",
    )

    resolved = run_benchmarks._resolve_case_args(
        [run_benchmarks.INTERACTIVE_CASE_SENTINEL, "explicit_case"],
        validation_root,
        include_disabled=True,
    )

    assert resolved == ["picked_case", "explicit_case"]


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


def test_strut_slower_than_opensees_lines_only_reports_slower_cases():
    lines = run_benchmarks._strut_slower_than_opensees_lines(
        [
            {
                "name": "faster_case",
                "opensees": {"analysis_us": 200},
                "strut": {"analysis_us": 100},
            },
            {
                "name": "slower_case",
                "opensees": {"analysis_us": 120},
                "strut": {"analysis_us": 150},
            },
            {
                "name": "tie_case",
                "opensees": {"analysis_us": 50},
                "strut": {"analysis_us": 50},
            },
        ]
    )

    assert lines == [
        "Strut slower than OpenSees (analysis_us):",
        "- slower_case: strut=150 us opensees=120 us (x1.2 slower)",
    ]


def test_strut_slower_than_opensees_lines_uses_batch_metrics_by_default():
    lines = run_benchmarks._strut_slower_than_opensees_lines(
        [
            {
                "name": "batch_case",
                "opensees_batch": {"analysis_us": 80},
                "strut": {"analysis_us": 120},
            }
        ]
    )

    assert lines == [
        "Strut slower than OpenSees (analysis_us):",
        "- batch_case: strut=120 us opensees=80 us (x1.5 slower)",
    ]


def test_write_solver_input_json_round_trips(tmp_path: Path):
    payload = {"model": {"ndm": 2, "ndf": 3}, "recorders": []}
    out_path = tmp_path / "solver.json"

    result = run_benchmarks._write_solver_input_json(payload, out_path)

    assert result == out_path
    assert out_path.exists()
    assert json.loads(out_path.read_text(encoding="utf-8")) == payload


def test_default_strut_solver_path_switches_with_profile_flag(tmp_path: Path):
    assert run_benchmarks.default_strut_solver_path(tmp_path, profile=False) == (
        tmp_path / "build" / "strut" / "strut"
    )
    assert run_benchmarks.default_strut_solver_path(tmp_path, profile=True) == (
        tmp_path / "build" / "strut" / "strut_profile"
    )


def test_ensure_strut_solver_builds_profile_binary_without_overwriting_default(
    monkeypatch, tmp_path: Path
):
    calls = []

    def fake_run(cmd, env=None, verbose=False):
        calls.append((cmd, dict(env) if env is not None else None, verbose))

    monkeypatch.setattr(run_benchmarks.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(run_benchmarks, "run", fake_run)
    monkeypatch.setattr(run_benchmarks, "log", lambda *args, **kwargs: None)
    monkeypatch.delenv("STRUT_MOJO_BIN", raising=False)

    solver_path = run_benchmarks.ensure_strut_solver(
        tmp_path, verbose=True, profile=True
    )

    assert solver_path == tmp_path / "build" / "strut" / "strut_profile"
    assert calls[0][0] == [
        "/usr/bin/uv",
        "run",
        "python",
        str(tmp_path / "scripts" / "build_mojo_solver.py"),
    ]
    assert calls[0][1] is not None
    assert calls[0][1]["STRUT_PROFILE"] == "1"
    assert calls[0][2] is True


def test_direct_tcl_raw_output_paths_collects_unique_recorder_outputs():
    case_data = {
        "recorders": [
            {"raw_path": "Data/Disp.out", "parity": True},
            {"raw_path": "example.out"},
            {"raw_path": "example.out"},
            {"raw_path": "skip.out", "parity": False},
        ]
    }

    assert run_benchmarks._direct_tcl_raw_output_paths(case_data) == [
        "Data/Disp.out",
        "example.out",
    ]


def test_write_direct_tcl_wrapper_copies_non_data_raw_outputs(tmp_path: Path):
    wrapper_path = tmp_path / "wrapper.tcl"

    run_benchmarks._write_direct_tcl_wrapper(
        original_script_name="case.tcl",
        wrapper_path=wrapper_path,
        compute_only=False,
        raw_output_paths=["Data/Disp.out", "example.out", "nested/out.dat"],
    )

    wrapper = wrapper_path.read_text(encoding="utf-8")

    assert 'file copy -force {Data/Disp.out} [file join $__strut_out_dir {Data/Disp.out}]' in wrapper
    assert 'file copy -force {example.out} [file join $__strut_out_dir {example.out}]' in wrapper
    assert 'file copy -force {nested/out.dat} [file join $__strut_out_dir {nested/out.dat}]' in wrapper


def test_write_direct_tcl_wrapper_compute_only_noops_display_commands(tmp_path: Path):
    wrapper_path = tmp_path / "wrapper_compute.tcl"

    run_benchmarks._write_direct_tcl_wrapper(
        original_script_name="case.tcl",
        wrapper_path=wrapper_path,
        compute_only=True,
        raw_output_paths=[],
    )

    wrapper = wrapper_path.read_text(encoding="utf-8")

    assert "rename source __strut_orig_source" in wrapper
    assert 'if {$tail == "DisplayModel2D.tcl"} {' in wrapper
    assert 'proc DisplayModel2D args { return {} }' in wrapper
    assert 'if {$tail == "DisplayPlane.tcl"} {' in wrapper
    assert 'proc DisplayPlane args { return {} }' in wrapper
    assert "rename exit __strut_orig_exit" in wrapper
    assert "proc exit args { return {} }" in wrapper
    assert "rename quit __strut_orig_quit" in wrapper
    assert "proc quit args { return {} }" in wrapper
    assert "foreach __strut_display_cmd {prp vup vpn viewWindow display} {" in wrapper
    assert 'proc $__strut_display_cmd args { return {} }' in wrapper


def test_prepare_direct_tcl_wrappers_mirror_shared_parent_assets(tmp_path: Path):
    bundle_root = tmp_path / "OpenSeesExamplesAdvanced"
    script_dir = bundle_root / "example"
    script_dir.mkdir(parents=True, exist_ok=True)
    entry_tcl = script_dir / "Ex1.Canti2D.EQ.tcl"
    entry_tcl.write_text("puts ok\n", encoding="utf-8")
    (script_dir / "ReadSMDFile.tcl").write_text("puts reader\n", encoding="utf-8")
    (bundle_root / "BM68elc.acc").write_text("0.0\n", encoding="utf-8")
    gmfiles = bundle_root / "GMfiles"
    gmfiles.mkdir()
    (gmfiles / "gm.dat").write_text("1.0\n", encoding="utf-8")
    sibling_example = bundle_root / "other_example"
    sibling_example.mkdir()
    (sibling_example / "other.tcl").write_text('puts "other"\n', encoding="utf-8")

    case = run_benchmarks.CaseSpec(name="direct_case", tcl_path=entry_tcl)
    timed_wrapper, compute_wrapper = run_benchmarks._prepare_direct_tcl_wrappers(
        case,
        {"recorders": []},
        tmp_path / "validation_root",
    )

    mirrored_script_dir = timed_wrapper.parent
    assert compute_wrapper.parent == mirrored_script_dir
    assert (mirrored_script_dir / "BM68elc.acc").read_text(encoding="utf-8") == "0.0\n"
    assert (mirrored_script_dir / "GMfiles" / "gm.dat").read_text(encoding="utf-8") == "1.0\n"
    assert (mirrored_script_dir / "ReadSMDfile.tcl").read_text(encoding="utf-8") == "puts reader\n"
    assert not (mirrored_script_dir / "other_example").exists()


def test_prepare_direct_tcl_wrappers_use_manifest_source_files_order(
    tmp_path: Path,
):
    bundle_root = tmp_path / "OpenSeesExamplesAdvanced"
    script_dir = bundle_root / "example"
    script_dir.mkdir(parents=True, exist_ok=True)
    analyze = script_dir / "Ex3.Canti2D.analyze.Dynamic.EQ.Uniform.tcl"
    analyze.write_text("puts analyze\n", encoding="utf-8")
    elastic = script_dir / "Ex3.Canti2D.build.ElasticElement.tcl"
    elastic.write_text("puts elastic\n", encoding="utf-8")
    inelastic = script_dir / "Ex3.Canti2D.build.InelasticSection.tcl"
    inelastic.write_text("puts inelastic\n", encoding="utf-8")
    fiber = script_dir / "Ex3.Canti2D.build.InelasticFiberSection.tcl"
    fiber.write_text("puts fiber\n", encoding="utf-8")

    manifest = tmp_path / "validation" / "direct_case" / "direct_tcl_case.json"
    _write_direct_tcl_case(
        manifest,
        entry_tcl=analyze,
        source_files=[
            "Ex3.Canti2D.build.InelasticFiberSection.tcl",
            "Ex3.Canti2D.analyze.Dynamic.EQ.Uniform.tcl",
        ],
    )
    case = run_benchmarks._direct_tcl_case_spec(manifest)
    timed_wrapper, compute_wrapper = run_benchmarks._prepare_direct_tcl_wrappers(
        case,
        {"recorders": []},
        tmp_path / "validation_root",
    )

    mirrored_script_dir = timed_wrapper.parent
    entry_wrapper = next(mirrored_script_dir.glob("__strut_*_entry.tcl"))
    assert entry_wrapper.read_text(encoding="utf-8").splitlines() == [
        "source {Ex3.Canti2D.build.InelasticFiberSection.tcl}",
        "source {Ex3.Canti2D.analyze.Dynamic.EQ.Uniform.tcl}",
    ]
    timed_text = timed_wrapper.read_text(encoding="utf-8")
    compute_text = compute_wrapper.read_text(encoding="utf-8")
    compat_wrapper = entry_wrapper.with_name(entry_wrapper.stem + "__opensees_compat.tcl")
    compute_entry = entry_wrapper.with_name(entry_wrapper.stem + "__compute.tcl")
    compute_compat = compute_entry.with_name(compute_entry.stem + "__opensees_compat.tcl")
    instrumented_fiber = mirrored_script_dir / "Ex3.Canti2D.build.InelasticFiberSection__strut_compute.tcl"
    assert compat_wrapper.exists()
    assert compute_entry.exists()
    assert compute_compat.exists()
    assert instrumented_fiber.exists()
    compat_text = compat_wrapper.read_text(encoding="utf-8")
    compute_entry_text = compute_entry.read_text(encoding="utf-8")
    instrumented_text = instrumented_fiber.read_text(encoding="utf-8")
    assert "rename section __strut_builtin_section" in compat_text
    assert 'set args [linsert $args 2 -GJ 1.0e-12]' in compat_text
    assert "rename nDMaterial __strut_builtin_nDMaterial" in compat_text
    assert 'if {[llength $args] == 4 && [lindex $args 0] eq "PlateFromPlaneStress"} {' in compat_text
    assert "rename mass __strut_builtin_mass" in compat_text
    assert f"source {{{entry_wrapper.name}}}" in compat_text
    assert f"source {{{compat_wrapper.name}}}" in timed_text
    assert f"source {{{compute_compat.name}}}" in compute_text
    assert "source {Ex3.Canti2D.build.InelasticFiberSection__strut_compute.tcl}" in compute_entry_text
    assert "puts fiber" in instrumented_text


def test_prepare_direct_tcl_entry_uses_explicit_source_files_order(
    tmp_path: Path,
):
    bundle_root = tmp_path / "OpenSeesExamplesAdvanced"
    script_dir = bundle_root / "example"
    script_dir.mkdir(parents=True, exist_ok=True)
    analyze = script_dir / "Ex3.Canti2D.analyze.Dynamic.EQ.Uniform.tcl"
    analyze.write_text("puts analyze\n", encoding="utf-8")
    elastic = script_dir / "Ex3.Canti2D.build.ElasticElement.tcl"
    elastic.write_text("puts elastic\n", encoding="utf-8")
    fiber = script_dir / "Ex3.Canti2D.build.InelasticFiberSection.tcl"
    fiber.write_text("puts fiber\n", encoding="utf-8")
    (bundle_root / "BM68elc.acc").write_text("0.0\n", encoding="utf-8")

    entry_wrapper = run_benchmarks._prepare_direct_tcl_entry(
        analyze,
        tmp_path / "mirror_root",
        source_files=[fiber, analyze],
    )

    assert entry_wrapper.read_text(encoding="utf-8").splitlines() == [
        "source {Ex3.Canti2D.build.InelasticFiberSection.tcl}",
        "source {Ex3.Canti2D.analyze.Dynamic.EQ.Uniform.tcl}",
    ]
    assert (entry_wrapper.parent / "BM68elc.acc").read_text(encoding="utf-8") == "0.0\n"


def test_ensure_direct_tcl_case_artifacts_parses_wrapped_entry_tcl(
    monkeypatch, tmp_path: Path
):
    repo_root = tmp_path / "repo"
    case_root = repo_root / "tests" / "validation" / "direct_case"
    manifest = case_root / "direct_tcl_case.json"
    entry_dir = repo_root / "docs" / "examples"
    entry_dir.mkdir(parents=True, exist_ok=True)
    analyze = entry_dir / "Ex3.Canti2D.analyze.Dynamic.EQ.Uniform.tcl"
    analyze.write_text("puts analyze\n", encoding="utf-8")
    build = entry_dir / "Ex3.Canti2D.build.InelasticFiberSection.tcl"
    build.write_text("puts build\n", encoding="utf-8")
    _write_direct_tcl_case(
        manifest,
        entry_tcl=analyze,
        source_files=[
            "Ex3.Canti2D.build.InelasticFiberSection.tcl",
            "Ex3.Canti2D.analyze.Dynamic.EQ.Uniform.tcl",
        ],
    )
    case = run_benchmarks._direct_tcl_case_spec(manifest)

    seen_entry = {}
    monkeypatch.setitem(
        sys.modules,
        "tcl_to_strut",
        SimpleNamespace(
            convert_tcl_to_solver_input=lambda entry, repo_root, compute_only=False: (
                seen_entry.setdefault("path", Path(entry)),
                {
                    "schema_version": "1.0",
                    "metadata": {"name": "generated", "units": "unknown"},
                    "model": {"ndm": 2, "ndf": 3},
                    "nodes": [],
                    "elements": [],
                    "recorders": [],
                },
            )[1]
        ),
    )

    def fake_run(cmd, env=None, verbose=False, capture_on_error=False):
        if "json_to_tcl.py" in str(cmd[3]):
            raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(run_benchmarks, "run", fake_run)

    case_data = run_benchmarks._ensure_direct_tcl_case_artifacts(
        case,
        repo_root=repo_root,
        env={},
        verbose=False,
    )

    assert case_data["metadata"]["name"] == "generated"
    assert seen_entry["path"].name.startswith("__strut_")
    assert seen_entry["path"].read_text(encoding="utf-8").splitlines() == [
        "source {Ex3.Canti2D.build.InelasticFiberSection.tcl}",
        "source {Ex3.Canti2D.analyze.Dynamic.EQ.Uniform.tcl}",
    ]


def test_ensure_direct_tcl_case_artifacts_skips_parser_check_when_not_required(
    monkeypatch, tmp_path: Path
):
    repo_root = tmp_path / "repo"
    case_root = repo_root / "tests" / "validation" / "direct_case"
    manifest = case_root / "direct_tcl_case.json"
    entry_tcl = tmp_path / "examples" / "case.tcl"
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
                "recorders": [],
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
        require_parser_check=False,
    )

    assert case_data["metadata"]["name"] == "generated"
    assert (case_root / "generated" / "case.json").exists()
    assert not (case_root / ".parser-check").exists()
    assert calls == []


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
        if "json_to_tcl.py" in str(cmd[3]):
            Path(cmd[-1]).write_text(
                "recorder Node -file Data/Disp.out -time -node 1 disp\n",
                encoding="utf-8",
            )
        elif "run_opensees.sh" in cmd[0]:
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
    assert (case_root / "reference-original" / "disp_node1.out").read_text(
        encoding="utf-8"
    ) == "1.0\n2.0\n"
    assert (case_root / ".parser-check").read_text(encoding="utf-8") == "ok\n"
    assert len(calls) == 3
    assert "json_to_tcl.py" in str(calls[0][3])
    assert "run_opensees.sh" in calls[1][0]
    assert "run_opensees.sh" in calls[2][0]


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
    (case_root / "generated").mkdir(parents=True, exist_ok=True)
    (case_root / "generated" / "model.tcl").write_text("puts generated\n", encoding="utf-8")
    (case_root / "generated" / "case.json").write_text("{}", encoding="utf-8")
    (case_root / "reference-original").mkdir(parents=True, exist_ok=True)
    (case_root / "reference-original" / "disp_node1.out").write_text("1.0\n", encoding="utf-8")
    (case_root / "reference").mkdir(parents=True, exist_ok=True)
    (case_root / "reference" / "disp_node1.out").write_text("1.0\n", encoding="utf-8")
    (case_root / ".parser-check").write_text("ok\n", encoding="utf-8")
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
    assert len(calls) == 1
    assert "json_to_tcl.py" in str(calls[0][3])


def test_ensure_direct_tcl_case_artifacts_falls_back_when_generated_tcl_fails(
    monkeypatch, tmp_path: Path
):
    repo_root = tmp_path / "repo"
    case_root = repo_root / "tests" / "validation" / "direct_case"
    manifest = case_root / "direct_tcl_case.json"
    entry_tcl = tmp_path / "examples" / "case.tcl"
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
                "recorders": [],
            }
        ),
    )

    def fake_run(cmd, env=None, verbose=False, capture_on_error=False):
        calls.append(cmd)
        if "json_to_tcl.py" in str(cmd[3]):
            raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(run_benchmarks, "run", fake_run)

    case_data = run_benchmarks._ensure_direct_tcl_case_artifacts(
        case,
        repo_root=repo_root,
        env={},
        verbose=False,
    )

    assert case_data["metadata"]["name"] == "generated"
    assert (case_root / "generated" / "case.json").exists()
    assert not (case_root / ".parser-check").exists()
    assert len(calls) == 1
    assert "json_to_tcl.py" in str(calls[0][3])


def test_require_direct_tcl_parser_check_accepts_validated_case(tmp_path: Path):
    repo_root = tmp_path / "repo"
    case_root = repo_root / "tests" / "validation" / "direct_case"
    manifest = case_root / "direct_tcl_case.json"
    entry_tcl = tmp_path / "examples" / "case.tcl"
    entry_tcl.parent.mkdir(parents=True, exist_ok=True)
    entry_tcl.write_text("puts ok\n", encoding="utf-8")
    _write_direct_tcl_case(manifest, entry_tcl=entry_tcl)
    parser_check = case_root / ".parser-check"
    parser_check.write_text("ok\n", encoding="utf-8")

    case = run_benchmarks._direct_tcl_case_spec(manifest)

    assert run_benchmarks._require_direct_tcl_parser_check(case) == parser_check


def test_require_direct_tcl_parser_check_rejects_missing_file(tmp_path: Path):
    repo_root = tmp_path / "repo"
    case_root = repo_root / "tests" / "validation" / "direct_case"
    manifest = case_root / "direct_tcl_case.json"
    entry_tcl = tmp_path / "examples" / "case.tcl"
    entry_tcl.parent.mkdir(parents=True, exist_ok=True)
    entry_tcl.write_text("puts ok\n", encoding="utf-8")
    _write_direct_tcl_case(manifest, entry_tcl=entry_tcl)

    case = run_benchmarks._direct_tcl_case_spec(manifest)

    with pytest.raises(SystemExit, match="missing parser-check for direct Tcl benchmark"):
        run_benchmarks._require_direct_tcl_parser_check(case)


def test_filter_cases_by_enabled_counts_disabled_and_skipped(tmp_path: Path):
    validation_root = tmp_path / "validation"
    enabled_case = validation_root / "enabled_case" / "enabled_case.json"
    benchmark_disabled_case = (
        validation_root / "benchmark_disabled" / "benchmark_disabled.json"
    )
    disabled_case = validation_root / "disabled_case" / "disabled_case.json"
    _write_case(enabled_case, enabled=True)
    _write_case(benchmark_disabled_case, enabled=False)
    _write_case(disabled_case, enabled=False)

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
    _write_case(enabled_case, enabled=True)
    _write_case(disabled_case, enabled=False)

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


def test_normalize_reference_outputs_splits_envelope_group_layout_with_time_pairs(
    tmp_path: Path,
):
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir(parents=True, exist_ok=True)
    (reference_dir / "ele32.out").write_text(
        "\n".join(
            [
                "7.0 -1 7.0 2 1.0 3 1.0 4 1.0 5 1.0 6 7.0 -7 1.0 8 1.0 9 1.0 10 7.0 11 1.0 12",
                "1.0 -13 1.0 14 7.0 15 7.0 16 7.0 17 7.0 18 1.0 -19 7.0 20 7.0 21 7.0 22 1.0 23 7.0 24",
                "7.0 -25 1.0 26 7.0 27 7.0 28 1.0 29 7.0 30 7.0 -31 7.0 32 7.0 33 7.0 34 7.0 35 7.0 36",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    case_data = {
        "recorders": [
            {
                "type": "envelope_element_force",
                "elements": [1],
                "output": "env",
                "raw_path": "ele32.out",
                "include_time": True,
                "group_layout": {
                    "type": "envelope_element_force",
                    "elements": [1, 2],
                    "values_per_element": [6, 6],
                },
            },
            {
                "type": "envelope_element_force",
                "elements": [2],
                "output": "env",
                "raw_path": "ele32.out",
                "include_time": True,
                "group_layout": {
                    "type": "envelope_element_force",
                    "elements": [1, 2],
                    "values_per_element": [6, 6],
                },
            },
        ]
    }

    run_benchmarks._normalize_reference_outputs(case_data, reference_dir)

    assert (reference_dir / "env_ele1.out").read_text(encoding="utf-8") == (
        "-1 2 3 4 5 6 -13 14 15 16 17 18 -25 26 27 28 29 30\n"
    )
    assert (reference_dir / "env_ele2.out").read_text(encoding="utf-8") == (
        "-7 8 9 10 11 12 -19 20 21 22 23 24 -31 32 33 34 35 36\n"
    )


def test_normalize_reference_outputs_splits_envelope_node_group_layout_with_time_pairs(
    tmp_path: Path,
):
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir(parents=True, exist_ok=True)
    (reference_dir / "disp.out").write_text(
        "\n".join(
            [
                "7.0 -1 7.0 2",
                "1.0 -3 1.0 4",
                "3.0 5 3.0 6",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    case_data = {
        "recorders": [
            {
                "type": "envelope_node_displacement",
                "nodes": [3],
                "dofs": [1],
                "output": "env_disp",
                "raw_path": "disp.out",
                "include_time": True,
                "group_layout": {
                    "type": "envelope_node_displacement",
                    "nodes": [3, 4],
                    "values_per_node": [1, 1],
                },
            },
            {
                "type": "envelope_node_displacement",
                "nodes": [4],
                "dofs": [1],
                "output": "env_disp",
                "raw_path": "disp.out",
                "include_time": True,
                "group_layout": {
                    "type": "envelope_node_displacement",
                    "nodes": [3, 4],
                    "values_per_node": [1, 1],
                },
            },
        ]
    }

    run_benchmarks._normalize_reference_outputs(case_data, reference_dir)

    assert (reference_dir / "env_disp_node3.out").read_text(encoding="utf-8") == (
        "-1 -3 5\n"
    )
    assert (reference_dir / "env_disp_node4.out").read_text(encoding="utf-8") == (
        "2 4 6\n"
    )


def test_normalize_reference_outputs_preserves_existing_grouped_targets_on_mismatch(
    tmp_path: Path,
):
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir(parents=True, exist_ok=True)
    (reference_dir / "ele32.out").write_text(
        "7.0 -1 7.0 2 1.0 3 1.0 4 1.0 5 1.0 6 7.0 -7 1.0 8 1.0 9 1.0 10 7.0 11 1.0\n",
        encoding="utf-8",
    )
    (reference_dir / "env_ele1.out").write_text("keep1\n", encoding="utf-8")
    (reference_dir / "env_ele2.out").write_text("keep2\n", encoding="utf-8")
    case_data = {
        "recorders": [
            {
                "type": "envelope_element_force",
                "elements": [1],
                "output": "env",
                "raw_path": "ele32.out",
                "include_time": True,
                "group_layout": {
                    "type": "envelope_element_force",
                    "elements": [1, 2],
                    "values_per_element": [6, 6],
                },
            },
            {
                "type": "envelope_element_force",
                "elements": [2],
                "output": "env",
                "raw_path": "ele32.out",
                "include_time": True,
                "group_layout": {
                    "type": "envelope_element_force",
                    "elements": [1, 2],
                    "values_per_element": [6, 6],
                },
            },
        ]
    }

    run_benchmarks._normalize_reference_outputs(case_data, reference_dir)

    assert (reference_dir / "env_ele1.out").read_text(encoding="utf-8") == "keep1\n"
    assert (reference_dir / "env_ele2.out").read_text(encoding="utf-8") == "keep2\n"


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


def test_summarize_benchmark_failures_includes_runtime_context():
    runtime_failures = [
        run_benchmarks.RuntimeFailure(
            case_name="beam_case",
            case_file="/tmp/cases/beam_case.json",
            engine="opensees",
            phase="total",
            detail="opensees total pass aborted (exit 1); stderr=bad recorder",
            error_file="/tmp/results/opensees/beam_case/case_error.txt",
        )
    ]
    parity_failures = [
        "beam_case: node 1 mismatch at step 4",
        "  dof 1: ref=1.000000e+00 got=2.000000e+00 abs=1.000e+00 rel=1.000e+00",
    ]

    summary = run_benchmarks._summarize_benchmark_failures(
        runtime_failures, parity_failures
    )

    assert summary[0] == "FAILED beam_case"
    assert summary[1] == "  Case File: /tmp/cases/beam_case.json"
    assert (
        '  Runtime Failures: ["opensees total: opensees total pass aborted (exit 1); stderr=bad recorder '
        '[error_file=/tmp/results/opensees/beam_case/case_error.txt]"]'
        in summary
    )
    assert (
        '  Node Mismatch: ["node 1 mismatch at step 4", "dof 1: ref=1.000000e+00 got=2.000000e+00 abs=1.000e+00 rel=1.000e+00"]'
        in summary
    )


def test_format_subprocess_failure_prefers_abort_marker_over_stack_tail():
    exc = subprocess.CalledProcessError(
        4,
        ["build/strut/strut", "--input", "case.json"],
        stderr="\n".join(
            [
                "ABORT: [precheck-fail] rigidDiaphragm perp_dirn must be in 1..3",
                "#0 stack frame",
                "strut.mojo:0:0",
            ]
        ),
    )

    message = run_benchmarks._format_subprocess_failure("strut compute-only pass aborted", exc)

    assert (
        "stderr=ABORT: [precheck-fail] rigidDiaphragm perp_dirn must be in 1..3"
        in message
    )


def test_format_subprocess_failure_prefers_load_fail_marker():
    exc = subprocess.CalledProcessError(
        4,
        ["build/strut/strut", "--input", "case.json"],
        stderr="\n".join(
            [
                "note: setup info",
                "[load-fail] rigidDiaphragm constrained_node not found",
                "tail line",
            ]
        ),
    )

    message = run_benchmarks._format_subprocess_failure("strut compute-only pass aborted", exc)

    assert "stderr=[load-fail] rigidDiaphragm constrained_node not found" in message


def test_read_runtime_failures_collects_case_error_files(tmp_path: Path):
    case_dir = tmp_path / "results" / "opensees" / "beam_case"
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "case_error.txt").write_text(
        "opensees compute-only pass aborted (exit 1)\ncmd=run\nstderr=bad\n",
        encoding="utf-8",
    )

    failures = run_benchmarks._read_runtime_failures(
        case_entries=[
            {"name": "beam_case", "case_file": "/tmp/cases/beam_case.json"},
        ],
        results_root=tmp_path / "results",
        run_opensees=True,
        run_strut=False,
    )

    assert failures == [
        run_benchmarks.RuntimeFailure(
            case_name="beam_case",
            case_file="/tmp/cases/beam_case.json",
            engine="opensees",
            phase="compute-only",
            detail="opensees compute-only pass aborted (exit 1) | cmd=run | stderr=bad",
            error_file=str(case_dir / "case_error.txt"),
        )
    ]


def test_read_runtime_failures_ignores_whitespace_only_case_error_files(tmp_path: Path):
    case_dir = tmp_path / "results" / "opensees" / "beam_case"
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "case_error.txt").write_text("\r\n", encoding="utf-8")

    failures = run_benchmarks._read_runtime_failures(
        case_entries=[
            {"name": "beam_case", "case_file": "/tmp/cases/beam_case.json"},
        ],
        results_root=tmp_path / "results",
        run_opensees=True,
        run_strut=False,
    )

    assert failures == []


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


def test_compare_case_max_abs_uses_shared_prefix_when_reference_is_truncated():
    failures = []

    compare_case._compare_transient_rows(
        ref_vals=[[0.0, 1.0], [1.0, 3.0]],
        strut_vals=[[0.0, 1.0], [1.0, 3.0], [2.0, 30.0]],
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
        per_category_overrides={},
        per_recorder_overrides={"element_force": {"rtol": 0.3, "atol": 0.01}},
    )

    assert rtol == pytest.approx(0.3)
    assert atol == pytest.approx(0.01)


def test_resolve_recorder_tolerance_applies_category_override():
    rtol, atol = run_benchmarks._resolve_recorder_tolerance(
        "node_reaction",
        global_rtol=run_benchmarks.REL_TOL,
        global_atol=run_benchmarks.ABS_TOL,
        has_global_override=False,
        per_category_overrides={"force": {"rtol": 0.25, "atol": 0.002}},
        per_recorder_overrides={},
    )

    assert rtol == pytest.approx(0.25)
    assert atol == pytest.approx(0.002)


def test_write_benchmark_plots_calls_plot_helper(monkeypatch, tmp_path: Path):
    calls = []

    class FakePlotBenchmarksModule:
        @staticmethod
        def write_plots_pdf(**kwargs):
            calls.append(kwargs)
            kwargs["output_path"].write_text("pdf", encoding="utf-8")
            return kwargs["output_path"]

    monkeypatch.setattr(
        run_benchmarks.importlib,
        "import_module",
        lambda name: (
            FakePlotBenchmarksModule
            if name == "plot_benchmarks"
            else (_ for _ in ()).throw(AssertionError(f"unexpected import {name}"))
        ),
    )

    summary_json = tmp_path / "summary.json"
    archive_dir = tmp_path / "archive"
    plots_pdf = tmp_path / "plots.pdf"
    summary_json.write_text('{"cases": []}\n', encoding="utf-8")

    result = run_benchmarks._write_benchmark_plots(
        results_path=summary_json,
        archive_dir=archive_dir,
        output_path=plots_pdf,
    )

    assert result == plots_pdf
    assert calls == [
        {
            "results_path": summary_json,
            "archive_dir": archive_dir,
            "output_path": plots_pdf,
        }
    ]


@pytest.mark.parametrize(
    (
        "engine",
        "no_batch",
        "expected_batch",
        "expected_opensees_batch",
        "expected_openseesmp_batch",
    ),
    [
        ("strut", False, False, False, False),
        ("both", False, True, True, False),
        ("all", False, True, True, False),
        ("opensees", True, False, False, False),
    ],
)
def test_collect_run_metadata_tracks_batch_mode_per_engine(
    monkeypatch,
    tmp_path: Path,
    engine: str,
    no_batch: bool,
    expected_batch: bool,
    expected_opensees_batch: bool,
    expected_openseesmp_batch: bool,
):
    monkeypatch.setattr(run_benchmarks, "git_rev", lambda repo_root: "deadbeef")
    monkeypatch.setattr(run_benchmarks, "_git_branch", lambda repo_root: "main")
    monkeypatch.setattr(run_benchmarks, "_read_cpu_model", lambda: "cpu")
    monkeypatch.setattr(
        run_benchmarks,
        "_safe_check_output",
        lambda cmd, cwd=None: "stubbed",
    )

    args = SimpleNamespace(
        benchmark_suite=None,
        engine=engine,
        no_batch=no_batch,
        repeat=2,
        warmup=1,
        profile=None,
    )

    metadata = run_benchmarks.collect_run_metadata(
        repo_root=tmp_path,
        args=args,
        results_root=tmp_path / "results",
        profile_root=None,
        strut_solver=tmp_path / "strut",
    )

    runner = metadata["runner"]
    assert runner["batch_mode"] is expected_batch
    assert runner["opensees_batch_mode"] is expected_opensees_batch
    assert runner["openseesmp_batch_mode"] is expected_openseesmp_batch
    assert runner["strut_batch_mode"] is False


def test_finalize_benchmark_outputs_skips_archive_when_disabled(
    monkeypatch, tmp_path: Path
):
    results_root = tmp_path / "results"
    archive_root = tmp_path / "archive"
    results_root.mkdir(parents=True, exist_ok=True)

    summary_json = results_root / "summary.json"
    summary_csv = results_root / "summary.csv"
    metadata_json = results_root / "metadata.json"
    phase_summary_csv = results_root / "phase_summary.csv"
    phase_rollup_csv = results_root / "phase_rollup.csv"
    for path in (
        summary_json,
        summary_csv,
        metadata_json,
        phase_summary_csv,
        phase_rollup_csv,
    ):
        path.write_text("data\n", encoding="utf-8")

    archive_calls = []
    plot_calls = []

    monkeypatch.setattr(
        run_benchmarks,
        "_archive_benchmark_artifacts",
        lambda *args, **kwargs: archive_calls.append((args, kwargs)) or [],
    )
    monkeypatch.setattr(
        run_benchmarks,
        "_write_benchmark_plots",
        lambda **kwargs: plot_calls.append(kwargs) or kwargs["output_path"],
    )

    result = run_benchmarks._finalize_benchmark_outputs(
        no_archive=True,
        archive_root=archive_root,
        results_root=results_root,
        summary_json=summary_json,
        summary_csv=summary_csv,
        metadata_json=metadata_json,
        phase_summary_csv=phase_summary_csv,
        phase_rollup_csv=phase_rollup_csv,
    )

    assert result == results_root / "plots.pdf"
    assert archive_calls == []
    assert plot_calls == [
        {
            "results_path": summary_json,
            "archive_dir": archive_root,
            "output_path": results_root / "plots.pdf",
        }
    ]


def test_finalize_benchmark_outputs_archives_summary_and_plot(
    monkeypatch, tmp_path: Path
):
    results_root = tmp_path / "results"
    archive_root = tmp_path / "archive"
    results_root.mkdir(parents=True, exist_ok=True)

    summary_json = results_root / "summary.json"
    summary_csv = results_root / "summary.csv"
    metadata_json = results_root / "metadata.json"
    phase_summary_csv = results_root / "phase_summary.csv"
    phase_rollup_csv = results_root / "phase_rollup.csv"
    for path in (
        summary_json,
        summary_csv,
        metadata_json,
        phase_summary_csv,
        phase_rollup_csv,
    ):
        path.write_text("data\n", encoding="utf-8")

    archive_calls = []
    plot_calls = []
    plots_pdf = results_root / "plots.pdf"

    monkeypatch.setattr(
        run_benchmarks,
        "_archive_benchmark_artifacts",
        lambda archive_root, stamp, artifacts: archive_calls.append(
            (archive_root, stamp, list(artifacts))
        )
        or [],
    )
    monkeypatch.setattr(
        run_benchmarks,
        "_write_benchmark_plots",
        lambda **kwargs: plot_calls.append(kwargs) or plots_pdf,
    )

    fixed_now = SimpleNamespace(
        strftime=lambda fmt: "20260303T120000Z",
    )
    monkeypatch.setattr(
        run_benchmarks,
        "datetime",
        SimpleNamespace(now=lambda tz: fixed_now),
    )

    result = run_benchmarks._finalize_benchmark_outputs(
        no_archive=False,
        archive_root=archive_root,
        results_root=results_root,
        summary_json=summary_json,
        summary_csv=summary_csv,
        metadata_json=metadata_json,
        phase_summary_csv=phase_summary_csv,
        phase_rollup_csv=phase_rollup_csv,
    )

    assert result == plots_pdf
    assert plot_calls == [
        {
            "results_path": summary_json,
            "archive_dir": archive_root,
            "output_path": plots_pdf,
        }
    ]
    assert archive_calls == [
        (
            archive_root,
            "20260303T120000Z",
            [
                summary_json,
                summary_csv,
                metadata_json,
                phase_summary_csv,
                phase_rollup_csv,
            ],
        ),
        (
            archive_root,
            "20260303T120000Z",
            [plots_pdf],
        ),
    ]


def test_build_phase_row_prefers_phase_json_splits_and_keeps_extra_metrics():
    row = run_benchmarks._build_phase_row(
        case_name="demo_case",
        dofs=42,
        phase_times={
            "case_load_parse_us": 10,
            "model_build_dof_map_us": 20,
            "output_write_us": 30,
            "solve_total_us": 40,
            "total_case_us": 50,
            "element_state_update_us": 60,
            "predictor_section_eval_us": 11,
            "corrector_section_eval_us": 12,
            "local_flexibility_accumulation_us": 13,
            "local_3x3_solve_us": 14,
            "local_commit_revert_us": 15,
            "global_nonlinear_iterations": 16,
            "tangent_factorizations": 17,
            "element_type_timing_us": {"forceBeamColumn2d": 123},
        },
        frame_totals={"nonlinear_iter": 999, "nonlinear_step": 1},
    )

    assert row["element_state_update_us"] == 60
    assert row["predictor_section_eval_us"] == 11
    assert row["corrector_section_eval_us"] == 12
    assert row["local_flexibility_accumulation_us"] == 13
    assert row["local_3x3_solve_us"] == 14
    assert row["local_commit_revert_us"] == 15
    assert row["global_nonlinear_iterations"] == 16
    assert row["tangent_factorizations"] == 17
    assert row["element_type_timing_us"] == {"forceBeamColumn2d": 123}
