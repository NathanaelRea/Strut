import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[2]
COMPARE_CASE_PATH = REPO_ROOT / "scripts" / "compare_case.py"


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


def test_analysis_is_transient_for_staged_with_transient_stage():
    analysis = {
        "type": "staged",
        "stages": [
            {"analysis": {"type": "static_nonlinear"}},
            {"analysis": {"type": "transient_nonlinear", "dt": 0.01}},
        ],
    }

    assert compare_case._analysis_is_transient(analysis) is True


def test_analysis_is_transient_false_for_non_transient_staged():
    analysis = {
        "type": "staged",
        "stages": [
            {"analysis": {"type": "static_linear"}},
            {"analysis": {"type": "static_nonlinear"}},
        ],
    }

    assert compare_case._analysis_is_transient(analysis) is False


def test_global_parity_tolerance_overrides_recorder_defaults():
    rtol, atol = compare_case._resolve_recorder_tolerance(
        "element_force",
        0.2,
        5.0e-4,
        True,
        {},
        {},
    )

    assert rtol == 0.2
    assert atol == 5.0e-4


def test_default_recorder_tolerance_used_without_global_override():
    rtol, atol = compare_case._resolve_recorder_tolerance(
        "element_force",
        compare_case.REL_TOL,
        compare_case.ABS_TOL,
        False,
        {},
        {},
    )

    assert rtol == compare_case.DEFAULT_RECORDER_TOLERANCES["element_force"]["rtol"]
    assert atol == compare_case.DEFAULT_RECORDER_TOLERANCES["element_force"]["atol"]


def test_category_override_applies_to_recorder_types():
    rtol, atol = compare_case._resolve_recorder_tolerance(
        "element_force",
        compare_case.REL_TOL,
        compare_case.ABS_TOL,
        False,
        {"force": {"rtol": 0.2, "atol": 1.0e-3}},
        {},
    )

    assert rtol == 0.2
    assert atol == 1.0e-3


def test_recorder_override_wins_over_category_override():
    rtol, atol = compare_case._resolve_recorder_tolerance(
        "element_force",
        compare_case.REL_TOL,
        compare_case.ABS_TOL,
        False,
        {"force": {"rtol": 0.2, "atol": 1.0e-3}},
        {"element_force": {"rtol": 0.3, "atol": 5.0e-3}},
    )

    assert rtol == 0.3
    assert atol == 5.0e-3


def test_compare_transient_rows_allows_different_lengths_for_max_abs():
    failures = []

    compare_case._compare_transient_rows(
        ref_vals=[[0.0, 1.0], [0.0, 2.0]],
        strut_vals=[[0.0, -2.0]],
        label="node 1",
        failures=failures,
        rtol=compare_case.REL_TOL,
        atol=compare_case.ABS_TOL,
        parity_mode="max_abs",
    )

    assert failures == []


def test_compare_case_accepts_direct_tcl_manifest(monkeypatch, tmp_path: Path):
    repo_root = tmp_path / "repo"
    scripts_dir = repo_root / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(compare_case, "__file__", str(scripts_dir / "compare_case.py"))

    case_root = repo_root / "tests" / "validation" / "direct_case"
    case_root.mkdir(parents=True, exist_ok=True)
    manifest = case_root / "direct_tcl_case.json"
    entry_tcl = repo_root / "docs" / "examples" / "case.tcl"
    entry_tcl.parent.mkdir(parents=True, exist_ok=True)
    entry_tcl.write_text("puts ok\n", encoding="utf-8")
    build_tcl = entry_tcl.parent / "build.tcl"
    build_tcl.write_text("puts build\n", encoding="utf-8")
    manifest.write_text(
        json.dumps(
            {
                "name": "direct_case",
                "entry_tcl": "docs/examples/case.tcl",
                "source_files": ["build.tcl", "case.tcl"],
            }
        ),
        encoding="utf-8",
    )
    (case_root / "reference").mkdir(parents=True, exist_ok=True)
    (case_root / "strut").mkdir(parents=True, exist_ok=True)
    (case_root / "reference" / "disp_node1.out").write_text("1.0\n", encoding="utf-8")
    (case_root / "strut" / "disp_node1.out").write_text("1.0\n", encoding="utf-8")

    seen_entry = {}
    monkeypatch.setitem(
        sys.modules,
        "tcl_to_strut",
        SimpleNamespace(
            convert_tcl_to_solver_input=lambda entry, repo_root, compute_only=False: (
                seen_entry.setdefault("path", Path(entry)),
                {
                    "analysis": {"type": "static_linear"},
                    "recorders": [
                        {"type": "node_displacement", "nodes": [1], "output": "disp"}
                    ],
                },
            )[1]
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["compare_case.py", "--case", "direct_case"],
    )

    compare_case.main()
    assert seen_entry["path"].name.startswith("__strut_")
    assert seen_entry["path"].read_text(encoding="utf-8").splitlines() == [
        "source {build.tcl}",
        "source {case.tcl}",
    ]


def test_load_direct_tcl_case_data_reuses_wrapped_input_without_self_deleting(
    monkeypatch, tmp_path: Path
):
    repo_root = tmp_path / "repo"
    scripts_dir = repo_root / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(compare_case, "__file__", str(scripts_dir / "compare_case.py"))

    case_root = repo_root / "tests" / "validation" / "direct_case"
    manifest = case_root / "direct_tcl_case.json"
    entry_dir = repo_root / "docs" / "examples"
    entry_dir.mkdir(parents=True, exist_ok=True)
    build_tcl = entry_dir / "build.tcl"
    build_tcl.write_text("puts build\n", encoding="utf-8")
    entry_tcl = entry_dir / "case.tcl"
    entry_tcl.write_text("puts ok\n", encoding="utf-8")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "name": "direct_case",
                "entry_tcl": "docs/examples/case.tcl",
                "source_files": ["build.tcl", "case.tcl"],
            }
        ),
        encoding="utf-8",
    )

    wrapped_input = (
        case_root
        / "generated"
        / "_direct_tcl_context"
        / "examples"
        / "__strut_case_entry.tcl"
    )
    wrapped_input.parent.mkdir(parents=True, exist_ok=True)
    wrapped_input.write_text("source {build.tcl}\nsource {case.tcl}\n", encoding="utf-8")

    seen_entry = {}
    monkeypatch.setitem(
        sys.modules,
        "tcl_to_strut",
        SimpleNamespace(
            convert_tcl_to_solver_input=lambda entry, repo_root, compute_only=False: (
                seen_entry.setdefault("path", Path(entry)),
                {"analysis": {"type": "static_linear"}, "recorders": []},
            )[1]
        ),
    )

    compare_case._load_direct_tcl_case_data(
        wrapped_input,
        repo_root,
        case_root,
        manifest,
    )

    assert seen_entry["path"].name.startswith("__strut_")
    assert seen_entry["path"].resolve() != wrapped_input.resolve()
