import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_strut_case(case_data, out_dir: Path):
    input_path = out_dir / "input.json"
    input_path.write_text(json.dumps(case_data), encoding="utf-8")
    subprocess.check_call(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_strut_case.py"),
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
            str(REPO_ROOT / "scripts" / "run_strut_case.py"),
            "--input",
            str(input_path),
            "--output",
            str(out_dir),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )


def _read_rows(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append([float(value) for value in line.split()])
    return rows


def test_rigid_diaphragm_static_linear_recovers_slave_dofs():
    case_data = {
        "model": {"ndm": 2, "ndf": 3},
        "nodes": [
            {"id": 1, "x": 0.0, "y": 0.0, "constraints": [1, 2, 3]},
            {"id": 2, "x": 0.0, "y": 0.0},
            {"id": 3, "x": 4.0, "y": 0.0},
        ],
        "materials": [
            {"id": 1, "type": "Elastic", "params": {"E": 1000.0}},
            {"id": 2, "type": "Elastic", "params": {"E": 2000.0}},
            {"id": 3, "type": "Elastic", "params": {"E": 500.0}},
        ],
        "elements": [
            {
                "id": 1,
                "type": "zeroLength",
                "nodes": [1, 2],
                "materials": [1, 2, 3],
                "dirs": [1, 2, 3],
            }
        ],
        "analysis": {"type": "static_linear", "constraints": "Transformation"},
        "mp_constraints": [
            {
                "type": "rigidDiaphragm",
                "retained_node": 2,
                "constrained_node": 3,
                "perp_dirn": 3,
                "constrained_dofs": [1, 2, 3],
                "retained_dofs": [1, 2, 3],
                "matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 4.0], [0.0, 0.0, 1.0]],
                "dx": 4.0,
                "dy": 0.0,
                "dz": 0.0,
            }
        ],
        "loads": [
            {"node": 2, "dof": 2, "value": 2000.0},
            {"node": 2, "dof": 3, "value": 500.0},
        ],
        "recorders": [
            {
                "type": "node_displacement",
                "nodes": [2, 3],
                "dofs": [1, 2, 3],
                "output": "disp",
            }
        ],
    }

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        _run_strut_case(case_data, out_dir)
        retained_rows = _read_rows(out_dir / "disp_node2.out")
        slave_rows = _read_rows(out_dir / "disp_node3.out")

    assert retained_rows == [pytest.approx([0.0, 1.0, 1.0], abs=1e-12)]
    assert slave_rows == [pytest.approx([0.0, 5.0, 1.0], abs=1e-12)]


def test_rigid_diaphragm_loader_reports_missing_constrained_node():
    case_data = {
        "model": {"ndm": 2, "ndf": 3},
        "nodes": [{"id": 1, "x": 0.0, "y": 0.0}, {"id": 2, "x": 0.0, "y": 0.0}],
        "elements": [],
        "analysis": {"type": "static_linear", "constraints": "Transformation"},
        "mp_constraints": [
            {
                "type": "rigidDiaphragm",
                "retained_node": 1,
                "constrained_node": 99,
                "perp_dirn": 3,
                "constrained_dofs": [1, 2, 3],
                "retained_dofs": [1, 2, 3],
                "matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]],
            }
        ],
    }

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        proc = _run_strut_case_proc(case_data, out_dir)

    assert proc.returncode != 0
    assert "[load-fail] rigidDiaphragm constrained_node not found" in proc.stdout


def test_rigid_diaphragm_static_linear_reaction_recorder_transfers_slave_loads():
    case_data = {
        "model": {"ndm": 2, "ndf": 3},
        "nodes": [
            {"id": 1, "x": 0.0, "y": 0.0, "constraints": [1, 2, 3]},
            {"id": 2, "x": 0.0, "y": 0.0, "constraints": [1]},
            {"id": 3, "x": 4.0, "y": 0.0},
        ],
        "materials": [
            {"id": 1, "type": "Elastic", "params": {"E": 20.0}},
            {"id": 2, "type": "Elastic", "params": {"E": 10.0}},
        ],
        "elements": [
            {
                "id": 1,
                "type": "zeroLength",
                "nodes": [1, 2],
                "materials": [1, 2],
                "dirs": [2, 3],
            }
        ],
        "analysis": {"type": "static_linear", "constraints": "Transformation"},
        "mp_constraints": [
            {
                "type": "rigidDiaphragm",
                "retained_node": 2,
                "constrained_node": 3,
                "perp_dirn": 3,
                "constrained_dofs": [1, 2, 3],
                "retained_dofs": [1, 2, 3],
                "matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 4.0], [0.0, 0.0, 1.0]],
                "dx": 4.0,
                "dy": 0.0,
                "dz": 0.0,
            }
        ],
        "loads": [{"node": 3, "dof": 2, "value": 20.0}],
        "recorders": [
            {"type": "node_displacement", "nodes": [3], "dofs": [2, 3], "output": "disp"},
            {"type": "node_reaction", "nodes": [1], "dofs": [2, 3], "output": "react"},
        ],
    }

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        _run_strut_case(case_data, out_dir)
        disp_rows = _read_rows(out_dir / "disp_node3.out")
        reaction_rows = _read_rows(out_dir / "react_node1.out")

    assert disp_rows == [pytest.approx([33.0, 8.0], abs=1e-12)]
    assert reaction_rows == [pytest.approx([-20.0, -80.0], abs=1e-12)]


def test_rigid_diaphragm_modal_eigen_uses_reduced_mass_matrix():
    case_data = {
        "model": {"ndm": 2, "ndf": 3},
        "nodes": [
            {"id": 1, "x": 0.0, "y": 0.0, "constraints": [1, 2, 3]},
            {"id": 2, "x": 0.0, "y": 0.0, "constraints": [1, 2]},
            {"id": 3, "x": 4.0, "y": 0.0},
        ],
        "materials": [{"id": 1, "type": "Elastic", "params": {"E": 8.0}}],
        "elements": [
            {
                "id": 1,
                "type": "zeroLength",
                "nodes": [1, 2],
                "materials": [1],
                "dirs": [3],
            }
        ],
        "masses": [{"node": 3, "dof": 2, "value": 2.0}],
        "analysis": {
            "type": "modal_eigen",
            "num_modes": 1,
            "constraints": "Transformation",
        },
        "mp_constraints": [
            {
                "type": "rigidDiaphragm",
                "retained_node": 2,
                "constrained_node": 3,
                "perp_dirn": 3,
                "constrained_dofs": [1, 2, 3],
                "retained_dofs": [1, 2, 3],
                "matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 4.0], [0.0, 0.0, 1.0]],
                "dx": 4.0,
                "dy": 0.0,
                "dz": 0.0,
            }
        ],
        "recorders": [
            {
                "type": "modal_eigen",
                "nodes": [2, 3],
                "dofs": [2, 3],
                "modes": [1],
                "output": "modal",
            }
        ],
    }

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        _run_strut_case(case_data, out_dir)
        eigenvalues = _read_rows(out_dir / "modal_eigenvalues.out")
        node2_rows = _read_rows(out_dir / "modal_mode1_node2.out")
        node3_rows = _read_rows(out_dir / "modal_mode1_node3.out")

    assert eigenvalues == [pytest.approx([0.25], abs=1e-12)]
    sign = 1.0 if node3_rows[0][0] >= 0.0 else -1.0
    expected_theta = 1.0 / (32.0 ** 0.5)
    assert node2_rows == [pytest.approx([0.0, sign * expected_theta], abs=1e-12)]
    assert node3_rows == [
        pytest.approx([sign * 4.0 * expected_theta, sign * expected_theta], abs=1e-12)
    ]


def test_rigid_diaphragm_transient_linear_uses_slave_mass_inertia():
    case_data = {
        "model": {"ndm": 2, "ndf": 3},
        "time_series": [{"type": "Constant", "tag": 1, "factor": 1.0}],
        "pattern": {"type": "UniformExcitation", "tag": 1, "direction": 2, "accel": 1},
        "nodes": [
            {"id": 1, "x": 0.0, "y": 0.0, "constraints": [1, 2, 3]},
            {"id": 2, "x": 0.0, "y": 0.0, "constraints": [1, 2]},
            {"id": 3, "x": 4.0, "y": 0.0},
        ],
        "materials": [{"id": 1, "type": "Elastic", "params": {"E": 8.0}}],
        "elements": [
            {
                "id": 1,
                "type": "zeroLength",
                "nodes": [1, 2],
                "materials": [1],
                "dirs": [3],
            }
        ],
        "masses": [{"node": 3, "dof": 2, "value": 2.0}],
        "analysis": {
            "type": "transient_linear",
            "steps": 1,
            "dt": 1.0,
            "constraints": "Transformation",
            "integrator": {"type": "Newmark", "gamma": 0.5, "beta": 0.25},
        },
        "mp_constraints": [
            {
                "type": "rigidDiaphragm",
                "retained_node": 2,
                "constrained_node": 3,
                "perp_dirn": 3,
                "constrained_dofs": [1, 2, 3],
                "retained_dofs": [1, 2, 3],
                "matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 4.0], [0.0, 0.0, 1.0]],
                "dx": 4.0,
                "dy": 0.0,
                "dz": 0.0,
            }
        ],
        "recorders": [
            {"type": "node_displacement", "nodes": [2, 3], "dofs": [2, 3], "output": "disp"}
        ],
    }

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        _run_strut_case(case_data, out_dir)
        node2_rows = _read_rows(out_dir / "disp_node2.out")
        node3_rows = _read_rows(out_dir / "disp_node3.out")

    expected_theta = -1.0 / 17.0
    assert node2_rows == [pytest.approx([0.0, expected_theta], abs=1e-12)]
    assert node3_rows == [pytest.approx([4.0 * expected_theta, expected_theta], abs=1e-12)]
