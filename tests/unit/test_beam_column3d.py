import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


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


def _run_strut_case_proc(case_data, out_dir: Path):
    input_path = out_dir / "input.json"
    input_path.write_text(json.dumps(case_data), encoding="utf-8")
    return subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_strut_case.py"),
            "--input",
            str(input_path),
            "--output",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
    )


def _read_rows(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append([float(x) for x in line.split()])
    return rows


def _base_case(element_type: str):
    return {
        "schema_version": "1.0",
        "metadata": {"name": "beam_column3d_unit", "units": "SI"},
        "model": {"ndm": 3, "ndf": 6},
        "nodes": [
            {"id": 1, "x": 0.0, "y": 0.0, "z": 0.0, "constraints": [1, 2, 3, 4, 5, 6]},
            {"id": 2, "x": 0.0, "y": 0.0, "z": 3.0},
        ],
        "materials": [],
        "sections": [
            {
                "id": 1,
                "type": "ElasticSection3d",
                "params": {
                    "E": 200000000000.0,
                    "A": 0.01,
                    "Iy": 1.0e-5,
                    "Iz": 1.0e-5,
                    "G": 80000000000.0,
                    "J": 2.0e-5,
                },
            }
        ],
        "elements": [
            {
                "id": 1,
                "type": element_type,
                "nodes": [1, 2],
                "section": 1,
                "geomTransf": "Linear",
                "integration": "Lobatto",
                "num_int_pts": 3,
            }
        ],
        "loads": [{"node": 2, "dof": 1, "value": 1000.0}],
        "analysis": {"type": "static_linear", "steps": 1},
        "recorders": [
            {"type": "element_force", "elements": [1], "output": "element_force"}
        ],
    }


def _section_rows(case_data):
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        _run_strut_case(case_data, out_dir)
        sec_force_rows = _read_rows(out_dir / "sec_force_ele1_sec1.out")
        sec_defo_rows = _read_rows(out_dir / "sec_defo_ele1_sec1.out")
    return sec_force_rows, sec_defo_rows


@pytest.mark.parametrize(
    "element_type",
    ["forceBeamColumn3d", "dispBeamColumn3d"],
)
def test_beam_column3d_variants_static_linear_run(element_type: str):
    case_data = _base_case(element_type)

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        _run_strut_case(case_data, out_dir)
        rows = _read_rows(out_dir / "element_force_ele1.out")

    assert len(rows) == 1
    row = rows[0]
    assert len(row) == 12
    assert all(math.isfinite(v) for v in row)
    assert abs(row[0]) > 0.0
    assert row[0] == pytest.approx(-row[6], abs=1e-9)


@pytest.mark.parametrize("geom_transf", ["PDelta", "Corotational"])
def test_force_beam_column3d_non_linear_geom_transf_runs(geom_transf: str):
    case_data = _base_case("forceBeamColumn3d")
    case_data["elements"][0]["geomTransf"] = geom_transf

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        _run_strut_case(case_data, out_dir)
        rows = _read_rows(out_dir / "element_force_ele1.out")

    assert len(rows) == 1
    assert len(rows[0]) == 12
    assert all(math.isfinite(v) for v in rows[0])


def test_disp_beam_column3d_legendre_variable_points_runs():
    case_data = _base_case("dispBeamColumn3d")
    case_data["elements"][0]["geomTransf"] = "PDelta"
    case_data["elements"][0]["integration"] = "Legendre"
    case_data["elements"][0]["num_int_pts"] = 7

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        _run_strut_case(case_data, out_dir)
        rows = _read_rows(out_dir / "element_force_ele1.out")

    assert len(rows) == 1
    assert len(rows[0]) == 12
    assert all(math.isfinite(v) for v in rows[0])


@pytest.mark.parametrize(
    ("element_type", "analysis_type", "expected_rows"),
    [
        ("forceBeamColumn3d", "static_nonlinear", 2),
        ("dispBeamColumn3d", "static_linear", 1),
    ],
)
def test_beam_column3d_fiber_section3d_runtime_path_runs_with_section_recorders(
    element_type: str,
    analysis_type: str,
    expected_rows: int,
):
    case_data = _base_case(element_type)
    case_data["materials"] = [{"id": 1, "type": "Elastic", "params": {"E": 3.0e10}}]
    case_data["sections"] = [
        {
            "id": 1,
            "type": "FiberSection3d",
            "params": {
                "G": 1.2e10,
                "J": 2.0e-4,
                "patches": [
                    {
                        "type": "rect",
                        "material": 1,
                        "num_subdiv_y": 2,
                        "num_subdiv_z": 2,
                        "y_i": -0.2,
                        "z_i": -0.1,
                        "y_j": 0.2,
                        "z_j": 0.1,
                    }
                ],
                "layers": [],
            },
        }
    ]
    if analysis_type == "static_nonlinear":
        case_data["analysis"] = {
            "type": "static_nonlinear",
            "steps": expected_rows,
            "integrator": {"type": "LoadControl"},
        }
    else:
        case_data["analysis"] = {"type": "static_linear", "steps": expected_rows}
    case_data["recorders"] = [
        {"type": "element_force", "elements": [1], "output": "element_force"},
        {
            "type": "section_force",
            "elements": [1],
            "sections": [1, 3],
            "output": "sec_force",
        },
        {
            "type": "section_deformation",
            "elements": [1],
            "sections": [1, 3],
            "output": "sec_defo",
        },
    ]

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        _run_strut_case(case_data, out_dir)
        force_rows = _read_rows(out_dir / "element_force_ele1.out")
        sec_force_rows = _read_rows(out_dir / "sec_force_ele1_sec1.out")
        sec_defo_rows = _read_rows(out_dir / "sec_defo_ele1_sec3.out")

    assert len(force_rows) == expected_rows
    assert len(sec_force_rows) == expected_rows
    assert len(sec_defo_rows) == expected_rows
    assert all(len(row) == 12 for row in force_rows)
    assert all(len(row) == 4 for row in sec_force_rows)
    assert all(len(row) == 4 for row in sec_defo_rows)
    assert all(math.isfinite(v) for row in force_rows for v in row)
    assert all(math.isfinite(v) for row in sec_force_rows for v in row)
    assert all(math.isfinite(v) for row in sec_defo_rows for v in row)


@pytest.mark.parametrize("element_type", ["forceBeamColumn3d", "dispBeamColumn3d"])
def test_beam_column3d_section_recorders_use_opensees_3d_order(element_type: str):
    case_data = _base_case(element_type)
    case_data["sections"] = [
        {
            "id": 1,
            "type": "ElasticSection3d",
            "params": {
                "E": 3.0e10,
                "A": 0.08,
                "Iy": 2.666666666666667e-4,
                "Iz": 1.0666666666666669e-3,
                "G": 1.2e10,
                "J": 2.0e-4,
            },
        }
    ]
    case_data["loads"] = [{"node": 2, "dof": 1, "value": 50000.0}]
    case_data["recorders"] = [
        {"type": "section_force", "elements": [1], "section": 1, "output": "sec_force"},
        {
            "type": "section_deformation",
            "elements": [1],
            "section": 1,
            "output": "sec_defo",
        },
    ]

    sec_force_rows, sec_defo_rows = _section_rows(case_data)

    assert len(sec_force_rows) == 1
    assert len(sec_defo_rows) == 1
    assert sec_force_rows[0] == pytest.approx([0.0, 0.0, -150000.0, 0.0], abs=1e-9)
    assert sec_defo_rows[0] == pytest.approx([0.0, 0.0, -0.01875, 0.0], abs=1e-9)


@pytest.mark.parametrize("element_type", ["forceBeamColumn3d", "dispBeamColumn3d"])
def test_beam_column3d_section_recorders_include_torsion_and_twist(element_type: str):
    case_data = {
        "schema_version": "1.0",
        "metadata": {"name": "beam_column3d_torsion_unit", "units": "SI"},
        "model": {"ndm": 3, "ndf": 6},
        "nodes": [
            {"id": 1, "x": 0.0, "y": 0.0, "z": 0.0, "constraints": [1, 2, 3, 4, 5, 6]},
            {"id": 2, "x": 3.0, "y": 0.0, "z": 0.0},
        ],
        "materials": [],
        "sections": [
            {
                "id": 1,
                "type": "ElasticSection3d",
                "params": {
                    "E": 2.0e11,
                    "A": 0.02,
                    "Iy": 8.0e-5,
                    "Iz": 8.0e-5,
                    "G": 8.0e10,
                    "J": 2.0e-4,
                },
            }
        ],
        "elements": [
            {
                "id": 1,
                "type": element_type,
                "nodes": [1, 2],
                "section": 1,
                "geomTransf": "Linear",
                "integration": "Lobatto",
                "num_int_pts": 3,
            }
        ],
        "loads": [{"node": 2, "dof": 4, "value": 1200.0}],
        "analysis": {"type": "static_linear", "steps": 1},
        "recorders": [
            {"type": "section_force", "elements": [1], "section": 1, "output": "sec_force"},
            {
                "type": "section_deformation",
                "elements": [1],
                "section": 1,
                "output": "sec_defo",
            },
        ],
    }

    sec_force_rows, sec_defo_rows = _section_rows(case_data)

    assert len(sec_force_rows) == 1
    assert len(sec_defo_rows) == 1
    assert sec_force_rows[0] == pytest.approx([0.0, 0.0, 0.0, 1200.0], abs=1e-9)
    assert sec_defo_rows[0] == pytest.approx([0.0, 0.0, 0.0, 7.5e-05], abs=1e-12)


def test_force_beam_column3d_fiber_section3d_requires_positive_gj():
    case_data = _base_case("forceBeamColumn3d")
    case_data["materials"] = [{"id": 1, "type": "Elastic", "params": {"E": 3.0e10}}]
    case_data["sections"] = [
        {
            "id": 1,
            "type": "FiberSection3d",
            "params": {
                "patches": [
                    {
                        "type": "rect",
                        "material": 1,
                        "num_subdiv_y": 2,
                        "num_subdiv_z": 2,
                        "y_i": -0.2,
                        "z_i": -0.1,
                        "y_j": 0.2,
                        "z_j": 0.1,
                    }
                ],
                "layers": [],
            },
        }
    ]

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        proc = _run_strut_case_proc(case_data, out_dir)

    assert proc.returncode != 0


def test_loader_validates_unused_fiber_section3d_definitions():
    case_data = _base_case("forceBeamColumn3d")
    case_data["materials"] = [{"id": 1, "type": "Elastic", "params": {"E": 3.0e10}}]
    case_data["sections"].append(
        {
            "id": 99,
            "type": "FiberSection3d",
            "params": {
                "patches": [
                    {
                        "type": "triangle",
                        "material": 1,
                        "num_subdiv_y": 2,
                        "num_subdiv_z": 2,
                    }
                ],
                "layers": [],
            },
        }
    )

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        proc = _run_strut_case_proc(case_data, out_dir)

    assert proc.returncode != 0
    assert proc.stderr.strip() != ""
