import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


repo_root = Path(__file__).resolve().parents[2]


def _run_mojo_case(case_data, out_dir: Path):
    input_path = out_dir / "input.json"
    input_path.write_text(json.dumps(case_data), encoding="utf-8")
    subprocess.check_call(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_mojo_case.py"),
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


def _base_case(element_type: str):
    return {
        "schema_version": "1.0",
        "metadata": {"name": "disp_beam_column2d_unit", "units": "SI"},
        "model": {"ndm": 2, "ndf": 3},
        "nodes": [
            {"id": 1, "x": 0.0, "y": 0.0, "constraints": [1, 2, 3]},
            {"id": 2, "x": 3.0, "y": 0.0},
        ],
        "materials": [{"id": 1, "type": "Elastic", "params": {"E": 30000000000.0}}],
        "sections": [
            {
                "id": 1,
                "type": "ElasticSection2d",
                "params": {"E": 30000000000.0, "A": 0.04, "I": 1.0e-4},
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
                "num_int_pts": 5,
            }
        ],
        "loads": [{"node": 2, "dof": 2, "value": 1.0}],
        "analysis": {
            "type": "static_linear",
            "steps": 1,
            "force_beam_mode": "linear_if_elastic",
        },
        "recorders": [{"type": "element_force", "elements": [1], "output": "element_force"}],
    }


def test_disp_beam_column2d_static_linear_elastic_runs():
    element_type = "dispBeamColumn2d"
    case_data = _base_case(element_type)

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        _run_mojo_case(case_data, out_dir)
        rows = _read_rows(out_dir / "element_force_ele1.out")

    assert len(rows) == 1
    row = rows[0]
    assert len(row) == 6
    assert all(math.isfinite(value) for value in row)
    assert row[0] == pytest.approx(-row[3], abs=1e-9)
    assert row[1] == pytest.approx(-row[4], abs=1e-9)


def test_disp_beam_column2d_static_nonlinear_fiber_runs():
    case_data = _base_case("dispBeamColumn2d")
    case_data["materials"] = [
        {"id": 1, "type": "Elastic", "params": {"E": 30000000000.0}},
    ]
    case_data["sections"] = [
        {
            "id": 1,
            "type": "FiberSection2d",
            "params": {
                "patches": [
                    {
                        "type": "rect",
                        "material": 1,
                        "num_subdiv_y": 8,
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
    case_data["analysis"] = {
        "type": "static_nonlinear",
        "steps": 3,
        "max_iters": 20,
        "tol": 1.0e-10,
        "integrator": {"type": "LoadControl"},
    }

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        _run_mojo_case(case_data, out_dir)
        rows = _read_rows(out_dir / "element_force_ele1.out")

    assert len(rows) == 3
    assert all(len(row) == 6 for row in rows)
    assert all(math.isfinite(value) for row in rows for value in row)


def test_disp_beam_column2d_beam_uniform_reports_zero_free_end_forces():
    case_data = _base_case("dispBeamColumn2d")
    case_data["loads"] = []
    case_data["element_loads"] = [{"element": 1, "type": "beamUniform", "wy": -2.0}]

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        _run_mojo_case(case_data, out_dir)
        rows = _read_rows(out_dir / "element_force_ele1.out")

    assert len(rows) == 1
    row = rows[0]
    assert row[3] == pytest.approx(0.0, abs=1e-8)
    assert row[4] == pytest.approx(0.0, abs=1e-8)
    assert row[5] == pytest.approx(0.0, abs=1e-8)


def test_disp_beam_column2d_rejects_unsupported_geom_transf():
    case_data = _base_case("dispBeamColumn2d")
    case_data["elements"][0]["geomTransf"] = "Corotational"

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        with pytest.raises(subprocess.CalledProcessError):
            _run_mojo_case(case_data, out_dir)


def test_disp_beam_column2d_rejects_non_lobatto_integration():
    case_data = _base_case("dispBeamColumn2d")
    case_data["elements"][0]["integration"] = "Legendre"
    case_data["analysis"] = {
        "type": "static_nonlinear",
        "steps": 1,
        "max_iters": 20,
        "tol": 1.0e-10,
        "integrator": {"type": "LoadControl"},
    }

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        with pytest.raises(subprocess.CalledProcessError):
            _run_mojo_case(case_data, out_dir)


def test_disp_beam_column_alias_rejected():
    case_data = _base_case("dispBeamColumn")

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        with pytest.raises(subprocess.CalledProcessError):
            _run_mojo_case(case_data, out_dir)
