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


def _base_force_beam_case(material):
    return {
        "schema_version": "1.0",
        "metadata": {"name": "force_beam_column2d_unit", "units": "SI"},
        "model": {"ndm": 2, "ndf": 3},
        "nodes": [
            {"id": 1, "x": 0.0, "y": 0.0, "constraints": [1, 2, 3]},
            {"id": 2, "x": 3.0, "y": 0.0},
        ],
        "materials": [material],
        "sections": [
            {
                "id": 1,
                "type": "FiberSection2d",
                "params": {
                    "patches": [
                        {
                            "type": "rect",
                            "material": material["id"],
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
        ],
        "elements": [
            {
                "id": 1,
                "type": "forceBeamColumn2d",
                "nodes": [1, 2],
                "section": 1,
                "geomTransf": "Linear",
                "integration": "Lobatto",
                "num_int_pts": 3,
            }
        ],
        "loads": [{"node": 2, "dof": 2, "value": 1.0}],
        "recorders": [{"type": "element_force", "elements": [1], "output": "element_force"}],
    }


def _read_rows(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append([float(x) for x in line.split()])
    return rows


def test_force_beam_column2d_smoke_static_nonlinear_load_control():
    case_data = _base_force_beam_case(
        {"id": 1, "type": "Elastic", "params": {"E": 30000000000.0}}
    )
    case_data["analysis"] = {
        "type": "static_nonlinear",
        "steps": 3,
        "max_iters": 20,
        "tol": 1e-10,
        "integrator": {"type": "LoadControl"},
    }

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        _run_mojo_case(case_data, out_dir)
        rows = _read_rows(out_dir / "element_force_ele1.out")

    assert len(rows) == 3
    assert all(len(row) == 6 for row in rows)
    assert all(math.isfinite(value) for row in rows for value in row)


def test_force_beam_column2d_displacement_control_cyclic_sign_reversal():
    case_data = _base_force_beam_case(
        {
            "id": 1,
            "type": "Steel01",
            "params": {"Fy": 250000000.0, "E0": 200000000000.0, "b": 0.01},
        }
    )
    case_data["analysis"] = {
        "type": "static_nonlinear",
        "steps": 3,
        "max_iters": 30,
        "tol": 1e-9,
        "integrator": {
            "type": "DisplacementControl",
            "node": 2,
            "dof": 2,
            "targets": [0.01, -0.01, 0.0],
            "cutback": 0.5,
            "max_cutbacks": 8,
        },
    }

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        _run_mojo_case(case_data, out_dir)
        rows = _read_rows(out_dir / "element_force_ele1.out")

    assert len(rows) == 3
    # Global Fy at node i should reverse sign when control displacement reverses.
    assert rows[0][1] * rows[1][1] < 0.0


def test_force_beam_column2d_static_linear_elastic_runs():
    case_data = _base_force_beam_case(
        {"id": 1, "type": "Elastic", "params": {"E": 30000000000.0}}
    )
    case_data["analysis"] = {
        "type": "static_linear",
        "steps": 1,
        "force_beam_mode": "linear_if_elastic",
    }

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        _run_mojo_case(case_data, out_dir)
        rows = _read_rows(out_dir / "element_force_ele1.out")

    assert len(rows) == 1
    assert len(rows[0]) == 6
    assert all(math.isfinite(value) for value in rows[0])


def test_force_beam_column2d_static_linear_elastic_unit_tip_load_equilibrium():
    case_data = _base_force_beam_case(
        {"id": 1, "type": "Elastic", "params": {"E": 30000000000.0}}
    )
    case_data["analysis"] = {
        "type": "static_linear",
        "steps": 1,
        "force_beam_mode": "linear_if_elastic",
    }

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        _run_mojo_case(case_data, out_dir)
        rows = _read_rows(out_dir / "element_force_ele1.out")

    assert len(rows) == 1
    row = rows[0]
    assert len(row) == 6
    assert row[0] == pytest.approx(-row[3], abs=1e-9)
    assert row[1] == pytest.approx(-row[4], abs=1e-9)
    # Unit tip load should produce unit shear at the beam ends.
    assert abs(row[1]) == pytest.approx(1.0, rel=1e-7, abs=1e-9)


def test_force_beam_column2d_static_linear_nonelastic_rejected():
    case_data = _base_force_beam_case(
        {
            "id": 1,
            "type": "Steel01",
            "params": {"Fy": 250000000.0, "E0": 200000000000.0, "b": 0.01},
        }
    )
    case_data["analysis"] = {"type": "static_linear", "steps": 1}

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        with pytest.raises(subprocess.CalledProcessError):
            _run_mojo_case(case_data, out_dir)
