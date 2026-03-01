import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_CASE_PATH = REPO_ROOT / "scripts" / "run_case.py"
PARITY_CASES_PATH = REPO_ROOT / "tests" / "validation" / "test_parity_cases.py"


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


run_case = _load_module(RUN_CASE_PATH, "strut_run_case_tcl_test_module")
parity_cases = _load_module(PARITY_CASES_PATH, "strut_parity_cases_tcl_test_module")


def test_run_case_accepts_direct_tcl_manifest_and_normalizes_reference_outputs(
    monkeypatch, tmp_path: Path
):
    tmp_repo = tmp_path / "repo"
    (tmp_repo / "scripts").mkdir(parents=True, exist_ok=True)
    (tmp_repo / "tests" / "validation").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(run_case, "__file__", str(tmp_repo / "scripts" / "run_case.py"))
    monkeypatch.setattr(run_case.shutil, "which", lambda name: "/usr/bin/uv")

    entry_tcl = tmp_repo / "docs" / "examples" / "Ex1a.Canti2D.EQ.modif.tcl"
    entry_tcl.parent.mkdir(parents=True, exist_ok=True)
    entry_tcl.write_text('puts "placeholder"\n', encoding="utf-8")
    canonical_case = (
        tmp_repo
        / "tests"
        / "validation"
        / "opensees_example_2d_elastic_cantileaver_column"
    )
    canonical_case.mkdir(parents=True, exist_ok=True)
    (canonical_case / "direct_tcl_case.json").write_text(
        json.dumps(
            {
                "name": "opensees_example_2d_elastic_cantileaver_column",
                "enabled": True,
                "entry_tcl": "docs/examples/Ex1a.Canti2D.EQ.modif.tcl",
            }
        ),
        encoding="utf-8",
    )

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
                "analysis": {
                    "type": "staged",
                    "stages": [
                        {"analysis": {"type": "static_nonlinear"}},
                        {"analysis": {"type": "transient_linear", "dt": 0.01}},
                    ],
                },
                "recorders": [
                    {
                        "type": "node_displacement",
                        "nodes": [2],
                        "output": "DFree",
                        "raw_path": "Data/DFree.out",
                        "include_time": True,
                    }
                ],
            }
        ),
    )

    def fake_run(cmd, env=None, verbose=False):
        calls.append(cmd)
        if "run_opensees_wine.sh" in cmd[0]:
            reference_dir = Path(cmd[-1])
            (reference_dir / "Data").mkdir(parents=True, exist_ok=True)
            (reference_dir / "Data" / "DFree.out").write_text(
                "0.1 1.0 2.0 3.0\n0.2 4.0 5.0 6.0\n", encoding="utf-8"
            )

    manifest_path = canonical_case / "direct_tcl_case.json"

    monkeypatch.setattr(run_case, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["run_case.py", str(manifest_path)])

    run_case.main()

    compare_calls = [
        cmd for cmd in calls if any("compare_case.py" in part for part in cmd)
    ]
    strut_calls = [
        cmd for cmd in calls if any("run_strut_case.py" in part for part in cmd)
    ]
    assert len(strut_calls) == 1
    assert "--input-tcl" in strut_calls[0]
    assert "--input" not in strut_calls[0]
    assert len(compare_calls) == 1
    assert "--case-root" in compare_calls[0]
    assert "--input-tcl" in compare_calls[0]
    case_root = Path(compare_calls[0][compare_calls[0].index("--case-root") + 1])
    assert case_root == canonical_case
    assert (
        Path(compare_calls[0][compare_calls[0].index("--input-tcl") + 1])
        == entry_tcl.resolve()
    )

    normalized = canonical_case / "reference" / "DFree_node2.out"
    assert normalized.read_text(encoding="utf-8") == "1.0 2.0 3.0\n4.0 5.0 6.0\n"


def test_run_case_tcl_without_canonical_mapping_uses_slug_case_root(
    monkeypatch, tmp_path: Path
):
    tmp_repo = tmp_path / "repo"
    (tmp_repo / "scripts").mkdir(parents=True, exist_ok=True)
    (tmp_repo / "tests" / "validation").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(run_case, "__file__", str(tmp_repo / "scripts" / "run_case.py"))
    monkeypatch.setattr(run_case.shutil, "which", lambda name: "/usr/bin/uv")

    entry_tcl = tmp_repo / "docs" / "examples" / "example.tcl"
    entry_tcl.parent.mkdir(parents=True, exist_ok=True)
    entry_tcl.write_text('puts "placeholder"\n', encoding="utf-8")

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
                "analysis": {"type": "staged", "stages": []},
                "recorders": [],
            }
        ),
    )

    def fake_run(cmd, env=None, verbose=False):
        calls.append(cmd)

    monkeypatch.setattr(run_case, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["run_case.py", str(entry_tcl)])

    run_case.main()

    compare_calls = [
        cmd for cmd in calls if any("compare_case.py" in part for part in cmd)
    ]
    strut_calls = [
        cmd for cmd in calls if any("run_strut_case.py" in part for part in cmd)
    ]
    assert len(strut_calls) == 1
    assert "--input-tcl" in strut_calls[0]
    assert len(compare_calls) == 1
    generated_dirs = list((tmp_repo / "tests" / "validation").glob("tcl_example_*"))
    assert len(generated_dirs) == 1


def test_selected_case_paths_accepts_repo_relative_tcl(monkeypatch):
    monkeypatch.setenv(
        "STRUT_PARITY_CASES",
        "docs/agent-reference/OpenSeesExamplesBasic/time_history_analysis_of_a_2d_elastic_cantilever_column/Ex1a.Canti2D.EQ.modif.tcl",
    )

    paths = parity_cases._selected_case_paths()

    assert len(paths) == 1
    assert paths[0].suffix == ".tcl"
    assert paths[0].exists()


def test_selected_case_paths_include_todo_and_benchmark_direct_tcl_when_run_all(
    monkeypatch,
):
    monkeypatch.delenv("STRUT_PARITY_CASES", raising=False)
    monkeypatch.setenv("STRUT_RUN_ALL_CASES", "1")

    paths = parity_cases._selected_case_paths()

    assert any(
        path.parent.name == "opensees_example_rc_frame_pushover" for path in paths
    )
    assert any(
        path.parent.name == "opensees_example_ex5_frame2d_eq_uniform_rc_fiber"
        for path in paths
    )


def test_selected_case_paths_resolves_direct_tcl_case_name(monkeypatch):
    monkeypatch.setenv(
        "STRUT_PARITY_CASES", "opensees_example_2d_elastic_cantileaver_column"
    )

    paths = parity_cases._selected_case_paths()

    assert len(paths) == 1
    assert paths[0].name == "direct_tcl_case.json"
    assert paths[0].parent.name == "opensees_example_2d_elastic_cantileaver_column"


def test_selected_case_paths_explicit_filter_does_not_append_default_tcl(monkeypatch):
    monkeypatch.setenv("STRUT_PARITY_CASES", "elastic_beam_cantilever")

    paths = parity_cases._selected_case_paths()

    assert len(paths) == 1
    assert paths[0].name == "elastic_beam_cantilever.json"


def test_normalize_reference_outputs_splits_envelope_element_group_layout(
    tmp_path: Path,
):
    case_json = tmp_path / "case.json"
    reference_dir = tmp_path / "reference"
    (reference_dir / "Data").mkdir(parents=True, exist_ok=True)
    (reference_dir / "Data" / "ele32.out").write_text(
        "1 2 3 4 5\n6 7 8 9 10\n", encoding="utf-8"
    )
    case_json.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "metadata": {"name": "group_layout_case", "units": "SI"},
                "model": {"ndm": 2, "ndf": 3},
                "nodes": [],
                "elements": [],
                "analysis": {"type": "staged", "stages": []},
                "recorders": [
                    {
                        "type": "envelope_element_force",
                        "elements": [1],
                        "output": "env",
                        "raw_path": "Data/ele32.out",
                        "include_time": False,
                        "group_layout": {
                            "type": "envelope_element_force",
                            "elements": [1, 2],
                            "values_per_element": [2, 3],
                        },
                    },
                    {
                        "type": "envelope_element_force",
                        "elements": [2],
                        "output": "env",
                        "raw_path": "Data/ele32.out",
                        "include_time": False,
                        "group_layout": {
                            "type": "envelope_element_force",
                            "elements": [1, 2],
                            "values_per_element": [2, 3],
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    run_case._normalize_reference_outputs(case_json, reference_dir)

    assert (reference_dir / "env_ele1.out").read_text(encoding="utf-8") == "1 2\n6 7\n"
    assert (reference_dir / "env_ele2.out").read_text(
        encoding="utf-8"
    ) == "3 4 5\n8 9 10\n"


def test_normalize_reference_outputs_preserves_existing_grouped_targets_on_mismatch(
    tmp_path: Path,
):
    case_json = tmp_path / "case.json"
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir(parents=True, exist_ok=True)
    (reference_dir / "ele32.out").write_text(
        "7.0 -1 7.0 2 1.0 3 1.0 4 1.0 5 1.0 6 7.0 -7 1.0 8 1.0 9 1.0 10 7.0 11 1.0\n",
        encoding="utf-8",
    )
    (reference_dir / "env_ele1.out").write_text("keep1\n", encoding="utf-8")
    (reference_dir / "env_ele2.out").write_text("keep2\n", encoding="utf-8")
    case_json.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "metadata": {"name": "group_layout_case", "units": "SI"},
                "model": {"ndm": 2, "ndf": 3},
                "nodes": [],
                "elements": [],
                "analysis": {"type": "staged", "stages": []},
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
                ],
            }
        ),
        encoding="utf-8",
    )

    run_case._normalize_reference_outputs(case_json, reference_dir)

    assert (reference_dir / "env_ele1.out").read_text(encoding="utf-8") == "keep1\n"
    assert (reference_dir / "env_ele2.out").read_text(encoding="utf-8") == "keep2\n"


def test_normalize_reference_outputs_splits_envelope_group_layout_with_time_pairs(
    tmp_path: Path,
):
    case_json = tmp_path / "case.json"
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
    case_json.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "metadata": {"name": "group_layout_case", "units": "SI"},
                "model": {"ndm": 2, "ndf": 3},
                "nodes": [],
                "elements": [],
                "analysis": {"type": "staged", "stages": []},
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
                ],
            }
        ),
        encoding="utf-8",
    )

    run_case._normalize_reference_outputs(case_json, reference_dir)

    assert (reference_dir / "env_ele1.out").read_text(encoding="utf-8") == (
        "-1 2 3 4 5 6 -13 14 15 16 17 18 -25 26 27 28 29 30\n"
    )
    assert (reference_dir / "env_ele2.out").read_text(encoding="utf-8") == (
        "-7 8 9 10 11 12 -19 20 21 22 23 24 -31 32 33 34 35 36\n"
    )
