import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path


repo_root = Path(__file__).resolve().parents[2]


def _run_strut_case(case_data, out_dir: Path):
    input_path = out_dir / "input.json"
    input_path.write_text(json.dumps(case_data), encoding="utf-8")
    subprocess.check_call(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_strut_case.py"),
            "--input",
            str(input_path),
            "--output",
            str(out_dir),
        ]
    )


def _read_rows(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append([float(x) for x in line.split()])
    return rows


def test_zero_length_recorders_emit_global_local_basic_and_deformation():
    case_data = {
        "schema_version": "1.0",
        "metadata": {"name": "zero_length_recorders_unit", "units": "SI"},
        "model": {"ndm": 2, "ndf": 2},
        "nodes": [
            {"id": 1, "x": 0.0, "y": 0.0, "constraints": [1, 2]},
            {"id": 2, "x": 0.0, "y": 0.0, "constraints": [2]},
        ],
        "materials": [{"id": 1, "type": "Elastic", "params": {"E": 1000.0}}],
        "sections": [],
        "elements": [
            {
                "id": 1,
                "type": "zeroLength",
                "nodes": [1, 2],
                "materials": [1],
                "dirs": [1],
            }
        ],
        "loads": [{"node": 2, "dof": 1, "value": 1.0}],
        "analysis": {"type": "static_linear", "steps": 1},
        "recorders": [
            {"type": "element_force", "elements": [1], "output": "global"},
            {"type": "element_local_force", "elements": [1], "output": "local"},
            {"type": "element_basic_force", "elements": [1], "output": "basic"},
            {"type": "element_deformation", "elements": [1], "output": "defo"},
        ],
    }

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        _run_strut_case(case_data, out_dir)
        global_rows = _read_rows(out_dir / "global_ele1.out")
        local_rows = _read_rows(out_dir / "local_ele1.out")
        basic_rows = _read_rows(out_dir / "basic_ele1.out")
        defo_rows = _read_rows(out_dir / "defo_ele1.out")

    assert global_rows == [[-1.0, 0.0, 1.0, 0.0]]
    assert local_rows == [[1.0]]
    assert basic_rows == [[1.0]]
    assert defo_rows[0][0] == 0.001


def test_two_node_link_recorders_emit_global_local_basic_and_deformation():
    case_data = {
        "schema_version": "1.0",
        "metadata": {"name": "two_node_link_recorders_unit", "units": "SI"},
        "model": {"ndm": 2, "ndf": 2},
        "nodes": [
            {"id": 1, "x": 0.0, "y": 0.0, "constraints": [1, 2]},
            {"id": 2, "x": 2.0, "y": 0.0, "constraints": [2]},
        ],
        "materials": [{"id": 1, "type": "Elastic", "params": {"E": 1000.0}}],
        "sections": [],
        "elements": [
            {
                "id": 1,
                "type": "twoNodeLink",
                "nodes": [1, 2],
                "materials": [1],
                "dirs": [1],
            }
        ],
        "loads": [{"node": 2, "dof": 1, "value": 1.0}],
        "analysis": {"type": "static_linear", "steps": 1},
        "recorders": [
            {"type": "element_force", "elements": [1], "output": "global"},
            {"type": "element_local_force", "elements": [1], "output": "local"},
            {"type": "element_basic_force", "elements": [1], "output": "basic"},
            {"type": "element_deformation", "elements": [1], "output": "defo"},
        ],
    }

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        _run_strut_case(case_data, out_dir)
        global_rows = _read_rows(out_dir / "global_ele1.out")
        local_rows = _read_rows(out_dir / "local_ele1.out")
        basic_rows = _read_rows(out_dir / "basic_ele1.out")
        defo_rows = _read_rows(out_dir / "defo_ele1.out")

    assert global_rows == [[-1.0, 0.0, 1.0, 0.0]]
    assert local_rows == [[-1.0, 0.0, 1.0, 0.0]]
    assert basic_rows == [[1.0]]
    assert len(defo_rows) == 1
    assert len(defo_rows[0]) == 1
    assert math.isclose(defo_rows[0][0], 0.001, rel_tol=0.0, abs_tol=1.0e-12)


def test_two_node_link_recorders_work_in_static_nonlinear():
    case_data = {
        "schema_version": "1.0",
        "metadata": {"name": "two_node_link_recorders_static_nonlinear_unit", "units": "SI"},
        "model": {"ndm": 2, "ndf": 2},
        "nodes": [
            {"id": 1, "x": 0.0, "y": 0.0, "constraints": [1, 2]},
            {"id": 2, "x": 2.0, "y": 0.0, "constraints": [2]},
        ],
        "materials": [{"id": 1, "type": "Elastic", "params": {"E": 1000.0}}],
        "sections": [],
        "elements": [
            {
                "id": 1,
                "type": "twoNodeLink",
                "nodes": [1, 2],
                "materials": [1],
                "dirs": [1],
            }
        ],
        "loads": [{"node": 2, "dof": 1, "value": 1.0}],
        "analysis": {
            "type": "static_nonlinear",
            "steps": 1,
            "integrator": {"type": "LoadControl", "lambda": 1.0},
            "algorithm": "Newton",
        },
        "recorders": [
            {"type": "element_force", "elements": [1], "output": "global"},
            {"type": "element_local_force", "elements": [1], "output": "local"},
            {"type": "element_basic_force", "elements": [1], "output": "basic"},
            {"type": "element_deformation", "elements": [1], "output": "defo"},
        ],
    }

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        _run_strut_case(case_data, out_dir)
        global_rows = _read_rows(out_dir / "global_ele1.out")
        local_rows = _read_rows(out_dir / "local_ele1.out")
        basic_rows = _read_rows(out_dir / "basic_ele1.out")
        defo_rows = _read_rows(out_dir / "defo_ele1.out")

    assert global_rows == [[-1.0, 0.0, 1.0, 0.0]]
    assert local_rows == [[-1.0, 0.0, 1.0, 0.0]]
    assert basic_rows == [[1.0]]
    assert len(defo_rows) == 1
    assert len(defo_rows[0]) == 1
    assert math.isclose(defo_rows[0][0], 0.001, rel_tol=0.0, abs_tol=1.0e-12)
