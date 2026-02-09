import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


repo_root = Path(__file__).resolve().parents[2]


def _run_json_to_tcl(case_data):
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        case_path = tmp_dir / "case.json"
        tcl_path = tmp_dir / "model.tcl"
        case_path.write_text(json.dumps(case_data), encoding="utf-8")
        subprocess.check_call(
            [
                sys.executable,
                str(repo_root / "scripts" / "json_to_tcl.py"),
                str(case_path),
                str(tcl_path),
            ]
        )
        return tcl_path.read_text(encoding="utf-8")


def _base_uniform_case():
    return {
        "schema_version": "1.0",
        "metadata": {"name": "uniform_excitation_tcl_unit", "units": "SI"},
        "model": {"ndm": 2, "ndf": 2},
        "nodes": [
            {"id": 1, "x": 0.0, "y": 0.0, "constraints": [1, 2]},
            {"id": 2, "x": 1.0, "y": 0.0},
        ],
        "materials": [{"id": 1, "type": "Elastic", "params": {"E": 100.0}}],
        "sections": [],
        "elements": [{"id": 1, "type": "truss", "nodes": [1, 2], "area": 1.0, "material": 1}],
        "masses": [{"node": 2, "dof": 1, "value": 1.0}],
        "time_series": [{"type": "Path", "tag": 2, "dt": 0.1, "values": [1.0, 0.0]}],
        "pattern": {"type": "UniformExcitation", "tag": 3, "direction": 1, "accel": 2},
        "analysis": {
            "type": "transient_nonlinear",
            "steps": 2,
            "dt": 0.1,
            "integrator": {"type": "Newmark", "gamma": 0.5, "beta": 0.25},
            "tol": 1.0e-9,
            "max_iters": 15,
        },
        "recorders": [{"type": "node_displacement", "nodes": [2], "dofs": [1], "output": "node_disp"}],
    }


def test_json_to_tcl_emits_uniform_excitation_rayleigh_and_transient_nonlinear():
    case = _base_uniform_case()
    case["rayleigh"] = {"alphaM": 0.1, "betaK": 0.2, "betaKInit": 0.3, "betaKComm": 0.4}

    text = _run_json_to_tcl(case)

    assert "pattern UniformExcitation 3 1 -accel 2\n" in text
    assert "rayleigh 0.1 0.2 0.3 0.4\n" in text
    assert "algorithm Newton\n" in text
    assert "integrator Newmark 0.5 0.25\n" in text
    assert "analysis Transient\n" in text
    assert "analyze 2 0.1\n" in text


def test_json_to_tcl_emits_modified_newton_algorithms():
    case = _base_uniform_case()
    case["analysis"]["algorithm"] = "ModifiedNewton"

    text = _run_json_to_tcl(case)
    assert "algorithm ModifiedNewton\n" in text

    case["analysis"]["algorithm"] = "ModifiedNewtonInitial"
    text_initial = _run_json_to_tcl(case)
    assert "algorithm ModifiedNewton -initial\n" in text_initial

    case["pattern"] = {"type": "Plain", "tag": 1, "time_series": 2}
    case["loads"] = [{"node": 2, "dof": 1, "value": 1.0}]
    case["analysis"] = {
        "type": "static_nonlinear",
        "steps": 2,
        "max_iters": 10,
        "tol": 1.0e-8,
        "algorithm": "ModifiedNewton",
        "integrator": {"type": "LoadControl"},
    }
    text_static = _run_json_to_tcl(case)
    assert "algorithm ModifiedNewton\n" in text_static
    assert "analysis Static\n" in text_static


def test_json_to_tcl_emits_transient_nonlinear_fallback_loop():
    case = _base_uniform_case()
    case["analysis"]["algorithm"] = "Newton"
    case["analysis"]["test_type"] = "NormUnbalance"
    case["analysis"]["fallback_algorithm"] = "ModifiedNewtonInitial"
    case["analysis"]["fallback_test_type"] = "NormDispIncr"
    case["analysis"]["fallback_tol"] = 1.0e-8
    case["analysis"]["fallback_max_iters"] = 25

    text = _run_json_to_tcl(case)

    assert "set strut_tr_ok 0\n" in text
    assert "set strut_tr_ok [analyze 1 0.1]\n" in text
    assert "test NormDispIncr 1e-08 25\n" in text
    assert "algorithm ModifiedNewton -initial\n" in text
    assert "if {$strut_tr_ok != 0} {\n" in text
    assert "analyze 2 0.1\n" not in text


def test_json_to_tcl_rejects_uniform_excitation_with_nodal_loads():
    case = _base_uniform_case()
    case["loads"] = [{"node": 2, "dof": 1, "value": 1.0}]

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        case_path = tmp_dir / "case.json"
        tcl_path = tmp_dir / "model.tcl"
        case_path.write_text(json.dumps(case), encoding="utf-8")
        with pytest.raises(subprocess.CalledProcessError):
            subprocess.check_call(
                [
                    sys.executable,
                    str(repo_root / "scripts" / "json_to_tcl.py"),
                    str(case_path),
                    str(tcl_path),
                ]
            )


def test_json_to_tcl_loads_path_values_from_file():
    case = _base_uniform_case()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        (tmp_dir / "A10000.tcl").write_text("1.0E+00 0.0D+00", encoding="utf-8")
        case["time_series"] = [
            {"type": "Path", "tag": 2, "dt": 0.1, "values_path": "A10000.tcl"}
        ]
        case_path = tmp_dir / "case.json"
        tcl_path = tmp_dir / "model.tcl"
        case_path.write_text(json.dumps(case), encoding="utf-8")
        subprocess.check_call(
            [
                sys.executable,
                str(repo_root / "scripts" / "json_to_tcl.py"),
                str(case_path),
                str(tcl_path),
            ]
        )
        text = tcl_path.read_text(encoding="utf-8")
    assert "timeSeries Path 2 -dt 0.1 -values {1.0 0.0} -factor 1.0\n" in text
