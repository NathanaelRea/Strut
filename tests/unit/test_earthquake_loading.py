import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


repo_root = Path(__file__).resolve().parents[2]


def _run_strut_case(case_data, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
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
    out_dir.mkdir(parents=True, exist_ok=True)
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
        "elements": [
            {"id": 1, "type": "truss", "nodes": [1, 2], "area": 1.0, "material": 1}
        ],
        "masses": [{"node": 2, "dof": 1, "value": 1.0}],
        "recorders": [
            {
                "type": "node_displacement",
                "nodes": [2],
                "dofs": [1],
                "output": "node_disp",
            },
            {"type": "element_force", "elements": [1], "output": "element_force"},
        ],
    }


def _base_zero_length_dynamic_case(do_rayleigh: bool):
    return {
        "schema_version": "1.0",
        "metadata": {"name": "zero_length_rayleigh_unit", "units": "SI"},
        "model": {"ndm": 2, "ndf": 2},
        "time_series": [
            {
                "type": "Path",
                "tag": 11,
                "dt": 0.02,
                "values": [0.0, 1.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0],
            }
        ],
        "pattern": {"type": "Plain", "tag": 11, "time_series": 11},
        "nodes": [
            {"id": 1, "x": 0.0, "y": 0.0, "constraints": [1, 2]},
            {"id": 2, "x": 0.0, "y": 0.0, "constraints": [2]},
        ],
        "materials": [{"id": 1, "type": "Elastic", "params": {"E": 100.0}}],
        "sections": [],
        "elements": [
            {
                "id": 1,
                "type": "zeroLength",
                "nodes": [1, 2],
                "materials": [1],
                "dirs": [1],
                "doRayleigh": do_rayleigh,
            }
        ],
        "loads": [{"node": 2, "dof": 1, "value": 1.0}],
        "masses": [{"node": 2, "dof": 1, "value": 1.0}],
        "analysis": {
            "type": "transient_linear",
            "steps": 8,
            "dt": 0.02,
            "integrator": {"type": "Newmark", "gamma": 0.5, "beta": 0.25},
        },
        "rayleigh": {"betaK": 0.25},
        "recorders": [
            {
                "type": "node_displacement",
                "nodes": [2],
                "dofs": [1],
                "output": "node_disp",
            }
        ],
    }


def _base_zero_length_damp_mats_dynamic_case(with_damp_mats: bool):
    case = _base_zero_length_dynamic_case(False)
    case["metadata"]["name"] = "zero_length_damp_mats_unit"
    case["rayleigh"] = {}
    if with_damp_mats:
        case["materials"].append({"id": 2, "type": "Elastic", "params": {"E": 25.0}})
        case["elements"][0]["dampMats"] = [2]
    return case


def _base_zero_length_damp_dynamic_case(with_damp: bool):
    case = _base_zero_length_dynamic_case(False)
    case["metadata"]["name"] = "zero_length_damp_unit"
    case["rayleigh"] = {}
    if with_damp:
        case["dampings"] = [{"id": 3, "type": "SecStif", "beta": 0.25}]
        case["elements"][0]["damp"] = 3
    return case


def _base_two_node_link_dynamic_case(do_rayleigh: bool):
    return {
        "schema_version": "1.0",
        "metadata": {"name": "two_node_link_rayleigh_unit", "units": "SI"},
        "model": {"ndm": 3, "ndf": 3},
        "time_series": [
            {
                "type": "Path",
                "tag": 12,
                "dt": 0.02,
                "values": [0.0, 1.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0],
            }
        ],
        "pattern": {"type": "Plain", "tag": 12, "time_series": 12},
        "nodes": [
            {"id": 1, "x": 0.0, "y": 0.0, "z": 0.0, "constraints": [1, 2, 3]},
            {"id": 2, "x": 1.0, "y": 0.0, "z": 0.0, "constraints": [1, 3]},
        ],
        "materials": [{"id": 1, "type": "Elastic", "params": {"E": 100.0}}],
        "sections": [],
        "elements": [
            {
                "id": 1,
                "type": "twoNodeLink",
                "nodes": [1, 2],
                "materials": [1],
                "dirs": [1],
                "orient": {"x": [0.0, 1.0, 0.0], "y": [0.0, 0.0, 1.0]},
                "mass": 2.0,
                "doRayleigh": do_rayleigh,
            }
        ],
        "analysis": {
            "type": "transient_linear",
            "steps": 8,
            "dt": 0.02,
            "integrator": {"type": "Newmark", "gamma": 0.5, "beta": 0.25},
        },
        "rayleigh": {"alphaM": 1.5},
        "loads": [{"node": 2, "dof": 2, "value": 1.0}],
        "recorders": [
            {
                "type": "node_displacement",
                "nodes": [2],
                "dofs": [2],
                "output": "node_disp",
            }
        ],
    }


def test_transient_linear_rayleigh_reduces_tail_response():
    base = _base_truss_dynamic_case(
        {"id": 1, "type": "Elastic", "params": {"E": 100.0}}
    )
    base["time_series"] = [
        {
            "type": "Path",
            "tag": 1,
            "dt": 0.02,
            "values": [0.0, 1.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0],
        }
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
        _run_strut_case(undamped, out_dir / "undamped")
        undamped_rows = _read_rows(out_dir / "undamped" / "node_disp_node2.out")

        damped = json.loads(json.dumps(base))
        damped["rayleigh"] = {"alphaM": 4.0}
        _run_strut_case(damped, out_dir / "damped")
        damped_rows = _read_rows(out_dir / "damped" / "node_disp_node2.out")

    undamped_tail = sum(abs(row[0]) for row in undamped_rows[4:]) / 4.0
    damped_tail = sum(abs(row[0]) for row in damped_rows[4:]) / 4.0
    assert damped_tail < undamped_tail


def test_zero_length_do_rayleigh_controls_stiffness_damping():
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        without_rayleigh = _base_zero_length_dynamic_case(False)
        _run_strut_case(without_rayleigh, out_dir / "without_rayleigh")
        without_rows = _read_rows(out_dir / "without_rayleigh" / "node_disp_node2.out")

        with_rayleigh = _base_zero_length_dynamic_case(True)
        _run_strut_case(with_rayleigh, out_dir / "with_rayleigh")
        with_rows = _read_rows(out_dir / "with_rayleigh" / "node_disp_node2.out")

    without_tail = sum(abs(row[0]) for row in without_rows[4:]) / 4.0
    with_tail = sum(abs(row[0]) for row in with_rows[4:]) / 4.0
    assert with_tail < without_tail


def test_two_node_link_do_rayleigh_controls_element_mass_damping():
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        without_rayleigh = _base_two_node_link_dynamic_case(False)
        _run_strut_case(without_rayleigh, out_dir / "without_rayleigh")
        without_rows = _read_rows(out_dir / "without_rayleigh" / "node_disp_node2.out")

        with_rayleigh = _base_two_node_link_dynamic_case(True)
        _run_strut_case(with_rayleigh, out_dir / "with_rayleigh")
        with_rows = _read_rows(out_dir / "with_rayleigh" / "node_disp_node2.out")

    without_tail = sum(abs(row[0]) for row in without_rows[4:]) / 4.0
    with_tail = sum(abs(row[0]) for row in with_rows[4:]) / 4.0
    assert with_tail < without_tail


def test_zero_length_damp_mats_add_element_damping():
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        without_damp_mats = _base_zero_length_damp_mats_dynamic_case(False)
        _run_strut_case(without_damp_mats, out_dir / "without_damp_mats")
        without_rows = _read_rows(out_dir / "without_damp_mats" / "node_disp_node2.out")

        with_damp_mats = _base_zero_length_damp_mats_dynamic_case(True)
        _run_strut_case(with_damp_mats, out_dir / "with_damp_mats")
        with_rows = _read_rows(out_dir / "with_damp_mats" / "node_disp_node2.out")

    without_tail = sum(abs(row[0]) for row in without_rows[4:]) / 4.0
    with_tail = sum(abs(row[0]) for row in with_rows[4:]) / 4.0
    assert with_tail < without_tail


def test_transient_nonlinear_newton_variants_match_on_elastic_truss():
    base = _base_truss_dynamic_case(
        {"id": 1, "type": "Elastic", "params": {"E": 100.0}}
    )
    base["time_series"] = [
        {
            "type": "Path",
            "tag": 1,
            "dt": 0.02,
            "values": [0.0, 1.0, 1.0, 0.5, 0.0, 0.0],
        }
    ]
    base["pattern"] = {"type": "Plain", "tag": 1, "time_series": 1}
    base["loads"] = [{"node": 2, "dof": 1, "value": 1.0}]
    base["analysis"] = {
        "type": "transient_nonlinear",
        "steps": 6,
        "dt": 0.02,
        "max_iters": 20,
        "tol": 1e-10,
        "system": "FullGeneral",
        "integrator": {"type": "Newmark", "gamma": 0.5, "beta": 0.25},
    }

    results = {}
    algorithm_options = {
        "Newton": {},
        "KrylovNewton": {"algorithm_options": {"maxDim": 2}},
        "Broyden": {"broyden_count": 3},
        "NewtonLineSearch": {"line_search_eta": 0.8},
    }
    for algorithm, extra in algorithm_options.items():
        case_data = json.loads(json.dumps(base))
        case_data["analysis"]["algorithm"] = algorithm
        case_data["analysis"].update(extra)

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            _run_strut_case(case_data, out_dir)
            results[algorithm] = {
                "disp": _read_rows(out_dir / "node_disp_node2.out"),
                "force": _read_rows(out_dir / "element_force_ele1.out"),
            }

    for algorithm in ("KrylovNewton", "Broyden", "NewtonLineSearch"):
        for got_row, ref_row in zip(
            results[algorithm]["disp"], results["Newton"]["disp"], strict=True
        ):
            assert got_row == pytest.approx(ref_row, abs=1e-10, rel=1e-10)
        for got_row, ref_row in zip(
            results[algorithm]["force"], results["Newton"]["force"], strict=True
        ):
            assert got_row == pytest.approx(ref_row, abs=1e-10, rel=1e-10)


def test_zero_length_damp_adds_element_damping():
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        without_damp = _base_zero_length_damp_dynamic_case(False)
        _run_strut_case(without_damp, out_dir / "without_damp")
        without_rows = _read_rows(out_dir / "without_damp" / "node_disp_node2.out")

        with_damp = _base_zero_length_damp_dynamic_case(True)
        _run_strut_case(with_damp, out_dir / "with_damp")
        with_rows = _read_rows(out_dir / "with_damp" / "node_disp_node2.out")

    without_tail = sum(abs(row[0]) for row in without_rows[4:]) / 4.0
    with_tail = sum(abs(row[0]) for row in with_rows[4:]) / 4.0
    assert with_tail < without_tail


def test_zero_length_damp_mats_require_matching_material_count():
    case_data = _base_zero_length_damp_mats_dynamic_case(True)
    case_data["elements"][0]["materials"] = [1, 2]
    case_data["elements"][0]["dirs"] = [1, 2]
    case_data["elements"][0]["dampMats"] = [2]
    case_data["nodes"][1]["constraints"] = []
    case_data["loads"] = [
        {"node": 2, "dof": 1, "value": 1.0},
        {"node": 2, "dof": 2, "value": 1.0},
    ]
    case_data["masses"] = [
        {"node": 2, "dof": 1, "value": 1.0},
        {"node": 2, "dof": 2, "value": 1.0},
    ]
    case_data["recorders"][0]["dofs"] = [1, 2]

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        with pytest.raises(subprocess.CalledProcessError):
            _run_strut_case(case_data, out_dir)


def test_two_node_link_rejects_damp_mats():
    case_data = _base_two_node_link_dynamic_case(False)
    case_data["materials"].append({"id": 2, "type": "Elastic", "params": {"E": 25.0}})
    case_data["elements"][0]["dampMats"] = [2]

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        with pytest.raises(subprocess.CalledProcessError):
            _run_strut_case(case_data, out_dir)


def test_zero_length_damp_requires_known_damping():
    case_data = _base_zero_length_damp_dynamic_case(True)
    case_data["elements"][0]["damp"] = 99

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        with pytest.raises(subprocess.CalledProcessError):
            _run_strut_case(case_data, out_dir)


def test_two_node_link_rejects_damp():
    case_data = _base_two_node_link_dynamic_case(False)
    case_data["dampings"] = [{"id": 3, "type": "SecStif", "beta": 0.25}]
    case_data["elements"][0]["damp"] = 3

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        with pytest.raises(subprocess.CalledProcessError):
            _run_strut_case(case_data, out_dir)


def test_run_strut_case_warns_for_noncoincident_zero_length(tmp_path: Path):
    case_data = _base_zero_length_dynamic_case(False)
    case_data["analysis"] = {"type": "static_linear", "steps": 1}
    case_data["time_series"] = [{"type": "Linear", "tag": 1, "factor": 1.0}]
    case_data["pattern"] = {"type": "Plain", "tag": 1, "time_series": 1}
    case_data["loads"] = [{"node": 2, "dof": 1, "value": 1.0}]
    case_data["nodes"][1]["x"] = 0.1
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "out"
    input_path.write_text(json.dumps(case_data), encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_strut_case.py"),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "WARNING ZeroLength::setDomain(): Element 1 has L=" in proc.stderr


def test_transient_linear_uniform_excitation_direction_sign():
    case_data = _base_truss_dynamic_case(
        {"id": 1, "type": "Elastic", "params": {"E": 100.0}}
    )
    case_data["time_series"] = [
        {"type": "Path", "tag": 2, "dt": 0.02, "values": [0.0, 1.0, 0.0, 0.0]}
    ]
    case_data["pattern"] = {
        "type": "UniformExcitation",
        "tag": 2,
        "direction": 1,
        "accel": 2,
    }
    case_data["analysis"] = {
        "type": "transient_linear",
        "steps": 3,
        "dt": 0.02,
        "integrator": {"type": "Newmark", "gamma": 0.5, "beta": 0.25},
    }

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        _run_strut_case(case_data, out_dir)
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
    case_data["pattern"] = {
        "type": "UniformExcitation",
        "tag": 3,
        "direction": 1,
        "accel": 3,
    }
    case_data["rayleigh"] = {"alphaM": 0.2, "betaKComm": 0.01}
    case_data["analysis"] = {
        "type": "transient_nonlinear",
        "steps": 6,
        "dt": 0.02,
        "integrator": {"type": "Newmark", "gamma": 0.5, "beta": 0.25},
        "algorithm": "Newton",
        "tol": 1.0e-8,
        "max_iters": 30,
    }

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        _run_strut_case(case_data, out_dir)
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
    case_data["pattern"] = {
        "type": "UniformExcitation",
        "tag": 4,
        "direction": 1,
        "accel": 4,
    }
    case_data["analysis"] = {
        "type": "transient_nonlinear",
        "steps": 6,
        "dt": 0.02,
        "integrator": {"type": "Newmark", "gamma": 0.5, "beta": 0.25},
        "algorithm": "ModifiedNewton",
        "tol": 1.0e-8,
        "max_iters": 30,
    }

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        _run_strut_case(case_data, out_dir)
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
    case_data["pattern"] = {
        "type": "UniformExcitation",
        "tag": 5,
        "direction": 1,
        "accel": 5,
    }
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
            _run_strut_case(case_data, out_dir)


def test_transient_nonlinear_rejects_rel_tol():
    case_data = _base_truss_dynamic_case(
        {"id": 1, "type": "Steel01", "params": {"Fy": 10.0, "E0": 100.0, "b": 0.01}}
    )
    case_data["time_series"] = [
        {"type": "Path", "tag": 5, "dt": 0.02, "values": [1.0, 1.0, 0.5, 0.0, 0.0, 0.0]}
    ]
    case_data["pattern"] = {
        "type": "UniformExcitation",
        "tag": 5,
        "direction": 1,
        "accel": 5,
    }
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
        proc = _run_strut_case_proc(case_data, out_dir)

    assert proc.returncode != 0


def test_transient_nonlinear_newton_fallback_modified_initial_smoke():
    case_data = _base_truss_dynamic_case(
        {"id": 1, "type": "Steel01", "params": {"Fy": 10.0, "E0": 100.0, "b": 0.01}}
    )
    case_data["time_series"] = [
        {"type": "Path", "tag": 6, "dt": 0.02, "values": [1.0, 1.0, 0.5, 0.0, 0.0, 0.0]}
    ]
    case_data["pattern"] = {
        "type": "UniformExcitation",
        "tag": 6,
        "direction": 1,
        "accel": 6,
    }
    case_data["analysis"] = {
        "type": "transient_nonlinear",
        "steps": 6,
        "dt": 0.02,
        "integrator": {"type": "Newmark", "gamma": 0.5, "beta": 0.25},
        "solver_chain": [
            {
                "algorithm": "Newton",
                "test_type": "NormUnbalance",
                "tol": 0.0,
                "max_iters": 1,
            },
            {
                "algorithm": "ModifiedNewtonInitial",
                "test_type": "NormDispIncr",
                "tol": 1.0,
                "max_iters": 10,
            },
        ],
    }

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        _run_strut_case(case_data, out_dir)
        disp_rows = _read_rows(out_dir / "node_disp_node2.out")

    assert len(disp_rows) == 6
    assert all(math.isfinite(row[0]) for row in disp_rows)


def test_transient_nonlinear_energy_incr_with_mapped_algorithms_smoke():
    case_data = _base_truss_dynamic_case(
        {"id": 1, "type": "Steel01", "params": {"Fy": 10.0, "E0": 100.0, "b": 0.01}}
    )
    case_data["time_series"] = [
        {"type": "Path", "tag": 7, "dt": 0.02, "values": [1.0, 0.5, 0.0, 0.0, 0.0]}
    ]
    case_data["pattern"] = {
        "type": "UniformExcitation",
        "tag": 7,
        "direction": 1,
        "accel": 7,
    }
    case_data["analysis"] = {
        "type": "transient_nonlinear",
        "steps": 5,
        "dt": 0.02,
        "integrator": {"type": "Newmark", "gamma": 0.5, "beta": 0.25},
        "solver_chain": [
            {
                "algorithm": "Broyden",
                "algorithm_options": {"max_iters": 8},
                "test_type": "EnergyIncr",
                "tol": 1.0,
                "max_iters": 10,
            },
            {
                "algorithm": "NewtonLineSearch",
                "algorithm_options": {"alpha": 0.8},
                "test_type": "EnergyIncr",
                "tol": 1.0,
                "max_iters": 10,
            },
        ],
    }

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        _run_strut_case(case_data, out_dir)
        disp_rows = _read_rows(out_dir / "node_disp_node2.out")

    assert len(disp_rows) == 5
    assert all(math.isfinite(row[0]) for row in disp_rows)


def test_staged_analysis_gravity_load_const_then_uniform_excitation_smoke():
    case_data = _base_truss_dynamic_case(
        {"id": 1, "type": "Elastic", "params": {"E": 100.0}}
    )
    case_data["time_series"] = [
        {"type": "Linear", "tag": 1, "factor": 1.0},
        {"type": "Path", "tag": 2, "dt": 0.02, "values": [0.0, 1.0, 0.0]},
    ]
    case_data["pattern"] = {"type": "Plain", "tag": 1, "time_series": 1}
    case_data["loads"] = [{"node": 2, "dof": 1, "value": 1.0}]
    case_data["analysis"] = {
        "type": "staged",
        "constraints": "Plain",
        "stages": [
            {
                "analysis": {
                    "type": "static_nonlinear",
                    "steps": 2,
                    "algorithm": "Newton",
                    "integrator": {"type": "LoadControl"},
                },
                "load_const": {"time": 0.0},
            },
            {
                "pattern": {
                    "type": "UniformExcitation",
                    "tag": 2,
                    "direction": 1,
                    "accel": 2,
                },
                "analysis": {
                    "type": "transient_linear",
                    "steps": 3,
                    "dt": 0.02,
                    "integrator": {"type": "Newmark", "gamma": 0.5, "beta": 0.25},
                },
            },
        ],
    }

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        _run_strut_case(case_data, out_dir)
        disp_rows = _read_rows(out_dir / "node_disp_node2.out")
        force_rows = _read_rows(out_dir / "element_force_ele1.out")

    assert len(disp_rows) == 5
    assert len(force_rows) == 5
    assert all(math.isfinite(row[0]) for row in disp_rows)
    assert all(math.isfinite(v) for row in force_rows for v in row)


def test_staged_transient_linear_load_const_preserves_element_gravity_state():
    case_data = {
        "schema_version": "1.0",
        "metadata": {"name": "staged_beam_gravity_hold_unit", "units": "SI"},
        "model": {"ndm": 2, "ndf": 3},
        "nodes": [
            {"id": 1, "x": 0.0, "y": 0.0, "constraints": [1, 2, 3]},
            {"id": 2, "x": 5.0, "y": 0.0},
        ],
        "materials": [{"id": 1, "type": "Elastic", "params": {"E": 1000.0}}],
        "sections": [
            {
                "id": 1,
                "type": "ElasticSection2d",
                "params": {"E": 1000.0, "A": 100.0, "I": 10.0},
            }
        ],
        "elements": [
            {
                "id": 1,
                "type": "elasticBeamColumn2d",
                "nodes": [1, 2],
                "section": 1,
                "geomTransf": "Linear",
            }
        ],
        "masses": [{"node": 2, "dof": 1, "value": 1.0}],
        "time_series": [
            {"type": "Linear", "tag": 1, "factor": 1.0},
            {"type": "Path", "tag": 2, "dt": 0.1, "values": [0.0, 0.0, 0.0]},
        ],
        "pattern": {"type": "Plain", "tag": 1, "time_series": 1},
        "element_loads": [{"element": 1, "type": "beamUniform", "wy": -2.0}],
        "analysis": {
            "type": "staged",
            "constraints": "Plain",
            "stages": [
                {
                    "analysis": {
                        "type": "static_nonlinear",
                        "steps": 1,
                        "algorithm": "Newton",
                        "integrator": {"type": "LoadControl", "step": 1.0},
                    },
                    "load_const": {"time": 0.0},
                },
                {
                    "pattern": {
                        "type": "UniformExcitation",
                        "tag": 2,
                        "direction": 1,
                        "accel": 2,
                    },
                    "analysis": {
                        "type": "transient_linear",
                        "steps": 3,
                        "dt": 0.1,
                        "integrator": {"type": "Newmark", "gamma": 0.5, "beta": 0.25},
                    },
                },
            ],
        },
        "recorders": [
            {
                "type": "node_displacement",
                "nodes": [2],
                "dofs": [2, 3],
                "output": "node_disp",
            }
        ],
    }

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        _run_strut_case(case_data, out_dir)
        disp_rows = _read_rows(out_dir / "node_disp_node2.out")

    assert len(disp_rows) == 4
    gravity_row = disp_rows[0]
    assert gravity_row[0] < 0.0
    assert gravity_row[1] < 0.0
    for row in disp_rows[1:]:
        assert row[0] == pytest.approx(gravity_row[0], abs=1.0e-9)
        assert row[1] == pytest.approx(gravity_row[1], abs=1.0e-9)


def test_staged_displacement_control_load_const_freezes_actual_load_factor():
    case_data = _base_truss_dynamic_case(
        {"id": 1, "type": "Elastic", "params": {"E": 100.0}}
    )
    case_data["time_series"] = [{"type": "Linear", "tag": 1, "factor": 1.0}]
    case_data["pattern"] = {"type": "Plain", "tag": 1, "time_series": 1}
    case_data["loads"] = [{"node": 2, "dof": 1, "value": 1.0}]
    case_data["analysis"] = {
        "type": "staged",
        "constraints": "Plain",
        "stages": [
            {
                "analysis": {
                    "type": "static_nonlinear",
                    "steps": 1,
                    "algorithm": "Newton",
                    "test_type": "NormDispIncr",
                    "tol": 1.0e-12,
                    "max_iters": 10,
                    "integrator": {
                        "type": "DisplacementControl",
                        "node": 2,
                        "dof": 1,
                        "du": 0.2,
                    },
                },
                "load_const": {"time": 0.0},
            },
            {"analysis": {"type": "static_linear", "steps": 1}},
        ],
    }

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        _run_strut_case(case_data, out_dir)
        disp_rows = _read_rows(out_dir / "node_disp_node2.out")

    assert len(disp_rows) == 1
    assert disp_rows[0][0] == pytest.approx(0.2, abs=1.0e-9)


def test_staged_displacement_control_does_not_skip_exact_min_du_step():
    case_data = _base_truss_dynamic_case(
        {"id": 1, "type": "Elastic", "params": {"E": 100.0}}
    )
    case_data["time_series"] = [{"type": "Linear", "tag": 1, "factor": 1.0}]
    case_data["pattern"] = {"type": "Plain", "tag": 1, "time_series": 1}
    case_data["loads"] = [{"node": 2, "dof": 1, "value": 1.0}]
    case_data["analysis"] = {
        "type": "static_nonlinear",
        "constraints": "Plain",
        "steps": 2,
        "algorithm": "Newton",
        "test_type": "NormDispIncr",
        "tol": 1.0e-12,
        "max_iters": 10,
        "integrator": {
            "type": "DisplacementControl",
            "node": 2,
            "dof": 1,
            "du": 0.1,
            "min_du": 0.1,
            "max_du": 0.1,
        },
    }

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        _run_strut_case(case_data, out_dir)
        disp_rows = _read_rows(out_dir / "node_disp_node2.out")

    assert [row[0] for row in disp_rows] == pytest.approx([0.1, 0.2], abs=1.0e-9)
