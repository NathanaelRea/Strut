import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


repo_root = Path(__file__).resolve().parents[2]


def _run_mojo_case(case_data, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
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


def _base_truss_dynamic_case(material):
    return {
        "schema_version": "1.0",
        "metadata": {"name": "earthquake_loading_unit", "units": "SI"},
        "model": {"ndm": 2, "ndf": 2},
        "nodes": [
            {"id": 1, "x": 0.0, "y": 0.0, "constraints": [1, 2]},
            {"id": 2, "x": 1.0, "y": 0.0, "constraints": [2]},
        ],
        "materials": [material],
        "sections": [],
        "elements": [{"id": 1, "type": "truss", "nodes": [1, 2], "area": 1.0, "material": 1}],
        "masses": [{"node": 2, "dof": 1, "value": 1.0}],
        "recorders": [
            {"type": "node_displacement", "nodes": [2], "dofs": [1], "output": "node_disp"},
            {"type": "element_force", "elements": [1], "output": "element_force"},
        ],
    }


def test_transient_linear_rayleigh_reduces_tail_response():
    base = _base_truss_dynamic_case({"id": 1, "type": "Elastic", "params": {"E": 100.0}})
    base["time_series"] = [
        {"type": "Path", "tag": 1, "dt": 0.02, "values": [0.0, 1.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0]}
    ]
    base["pattern"] = {"type": "Plain", "tag": 1, "time_series": 1}
    base["loads"] = [{"node": 2, "dof": 1, "value": 1.0}]
    base["analysis"] = {
        "type": "transient_linear",
        "steps": 8,
        "dt": 0.02,
        "integrator": {"type": "Newmark", "gamma": 0.5, "beta": 0.25},
    }

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        undamped = json.loads(json.dumps(base))
        _run_mojo_case(undamped, out_dir / "undamped")
        undamped_rows = _read_rows(out_dir / "undamped" / "node_disp_node2.out")

        damped = json.loads(json.dumps(base))
        damped["rayleigh"] = {"alphaM": 4.0}
        _run_mojo_case(damped, out_dir / "damped")
        damped_rows = _read_rows(out_dir / "damped" / "node_disp_node2.out")

    undamped_tail = sum(abs(row[0]) for row in undamped_rows[4:]) / 4.0
    damped_tail = sum(abs(row[0]) for row in damped_rows[4:]) / 4.0
    assert damped_tail < undamped_tail


def test_transient_linear_uniform_excitation_direction_sign():
    case_data = _base_truss_dynamic_case({"id": 1, "type": "Elastic", "params": {"E": 100.0}})
    case_data["time_series"] = [{"type": "Path", "tag": 2, "dt": 0.02, "values": [0.0, 1.0, 0.0, 0.0]}]
    case_data["pattern"] = {"type": "UniformExcitation", "tag": 2, "direction": 1, "accel": 2}
    case_data["analysis"] = {
        "type": "transient_linear",
        "steps": 3,
        "dt": 0.02,
        "integrator": {"type": "Newmark", "gamma": 0.5, "beta": 0.25},
    }

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        _run_mojo_case(case_data, out_dir)
        rows = _read_rows(out_dir / "node_disp_node2.out")

    assert len(rows) == 3
    assert min(row[0] for row in rows) < 0.0


def test_transient_nonlinear_newmark_newton_smoke():
    case_data = _base_truss_dynamic_case(
        {"id": 1, "type": "Steel01", "params": {"Fy": 10.0, "E0": 100.0, "b": 0.01}}
    )
    case_data["time_series"] = [
        {"type": "Path", "tag": 3, "dt": 0.02, "values": [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]}
    ]
    case_data["pattern"] = {"type": "UniformExcitation", "tag": 3, "direction": 1, "accel": 3}
    case_data["rayleigh"] = {"alphaM": 0.2, "betaKComm": 0.01}
    case_data["analysis"] = {
        "type": "transient_nonlinear",
        "steps": 6,
        "dt": 0.02,
        "integrator": {"type": "Newmark", "gamma": 0.5, "beta": 0.25},
        "algorithm": "Newton",
        "tol": 1.0e-8,
        "rel_tol": 1.0e-6,
        "max_iters": 30,
    }

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        _run_mojo_case(case_data, out_dir)
        disp_rows = _read_rows(out_dir / "node_disp_node2.out")
        force_rows = _read_rows(out_dir / "element_force_ele1.out")

    assert len(disp_rows) == 6
    assert len(force_rows) == 6
    assert all(math.isfinite(row[0]) for row in disp_rows)
    assert all(math.isfinite(v) for row in force_rows for v in row)


def test_transient_nonlinear_modified_newton_smoke():
    case_data = _base_truss_dynamic_case(
        {"id": 1, "type": "Steel01", "params": {"Fy": 10.0, "E0": 100.0, "b": 0.01}}
    )
    case_data["time_series"] = [
        {"type": "Path", "tag": 4, "dt": 0.02, "values": [1.0, 1.0, 0.5, 0.0, 0.0, 0.0]}
    ]
    case_data["pattern"] = {"type": "UniformExcitation", "tag": 4, "direction": 1, "accel": 4}
    case_data["analysis"] = {
        "type": "transient_nonlinear",
        "steps": 6,
        "dt": 0.02,
        "integrator": {"type": "Newmark", "gamma": 0.5, "beta": 0.25},
        "algorithm": "ModifiedNewton",
        "tol": 1.0e-8,
        "rel_tol": 1.0e-6,
        "max_iters": 30,
    }

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        _run_mojo_case(case_data, out_dir)
        disp_rows = _read_rows(out_dir / "node_disp_node2.out")

    assert len(disp_rows) == 6
    assert all(math.isfinite(row[0]) for row in disp_rows)


def test_transient_nonlinear_newton_without_fallback_can_fail():
    case_data = _base_truss_dynamic_case(
        {"id": 1, "type": "Steel01", "params": {"Fy": 10.0, "E0": 100.0, "b": 0.01}}
    )
    case_data["time_series"] = [
        {"type": "Path", "tag": 5, "dt": 0.02, "values": [1.0, 1.0, 0.5, 0.0, 0.0, 0.0]}
    ]
    case_data["pattern"] = {"type": "UniformExcitation", "tag": 5, "direction": 1, "accel": 5}
    case_data["analysis"] = {
        "type": "transient_nonlinear",
        "steps": 6,
        "dt": 0.02,
        "integrator": {"type": "Newmark", "gamma": 0.5, "beta": 0.25},
        "algorithm": "Newton",
        "test_type": "NormUnbalance",
        "tol": 0.0,
        "max_iters": 1,
    }

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        with pytest.raises(subprocess.CalledProcessError):
            _run_mojo_case(case_data, out_dir)


def test_transient_nonlinear_newton_fallback_modified_initial_smoke():
    case_data = _base_truss_dynamic_case(
        {"id": 1, "type": "Steel01", "params": {"Fy": 10.0, "E0": 100.0, "b": 0.01}}
    )
    case_data["time_series"] = [
        {"type": "Path", "tag": 6, "dt": 0.02, "values": [1.0, 1.0, 0.5, 0.0, 0.0, 0.0]}
    ]
    case_data["pattern"] = {"type": "UniformExcitation", "tag": 6, "direction": 1, "accel": 6}
    case_data["analysis"] = {
        "type": "transient_nonlinear",
        "steps": 6,
        "dt": 0.02,
        "integrator": {"type": "Newmark", "gamma": 0.5, "beta": 0.25},
        "algorithm": "Newton",
        "test_type": "NormUnbalance",
        "tol": 0.0,
        "max_iters": 1,
        "fallback_algorithm": "ModifiedNewtonInitial",
        "fallback_test_type": "MaxDispIncr",
        "fallback_tol": 1.0,
        "fallback_rel_tol": 0.0,
        "fallback_max_iters": 10,
    }

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        _run_mojo_case(case_data, out_dir)
        disp_rows = _read_rows(out_dir / "node_disp_node2.out")

    assert len(disp_rows) == 6
    assert all(math.isfinite(row[0]) for row in disp_rows)
