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
    _run_strut_case_path(input_path, out_dir)


def _run_strut_case_path(input_path: Path, out_dir: Path):
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
        "recorders": [
            {"type": "element_force", "elements": [1], "output": "element_force"}
        ],
    }


def _read_rows(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append([float(x) for x in line.split()])
    return rows


def test_force_beam_column2d_generated_rc_frame_gravity_case_converges():
    case_path = (
        repo_root
        / "tests"
        / "validation"
        / "opensees_example_rc_frame_gravity"
        / "generated"
        / "case.json"
    )

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        _run_strut_case_path(case_path, out_dir)
        analysis_time_us = int(
            (out_dir / "analysis_time_us.txt").read_text(encoding="utf-8").strip()
        )

    assert analysis_time_us > 0


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
        _run_strut_case(case_data, out_dir)
        rows = _read_rows(out_dir / "element_force_ele1.out")

    assert len(rows) == 3
    assert all(len(row) == 6 for row in rows)
    assert all(math.isfinite(value) for row in rows for value in row)


def test_force_beam_column2d_corotational_smoke_static_nonlinear_load_control():
    case_data = _base_force_beam_case(
        {"id": 1, "type": "Elastic", "params": {"E": 30000000000.0}}
    )
    case_data["elements"][0]["geomTransf"] = "Corotational"
    case_data["analysis"] = {
        "type": "static_nonlinear",
        "steps": 3,
        "max_iters": 20,
        "tol": 1e-10,
        "integrator": {"type": "LoadControl"},
    }

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        _run_strut_case(case_data, out_dir)
        rows = _read_rows(out_dir / "element_force_ele1.out")

    assert len(rows) == 3
    assert all(len(row) == 6 for row in rows)
    assert all(math.isfinite(value) for row in rows for value in row)


def test_force_beam_column2d_smoke_static_nonlinear_modified_newton():
    case_data = _base_force_beam_case(
        {"id": 1, "type": "Elastic", "params": {"E": 30000000000.0}}
    )
    case_data["analysis"] = {
        "type": "static_nonlinear",
        "steps": 3,
        "max_iters": 20,
        "tol": 1e-10,
        "algorithm": "ModifiedNewton",
        "integrator": {"type": "LoadControl"},
    }

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        _run_strut_case(case_data, out_dir)
        rows = _read_rows(out_dir / "element_force_ele1.out")

    assert len(rows) == 3
    assert all(len(row) == 6 for row in rows)
    assert all(math.isfinite(value) for row in rows for value in row)


def test_force_beam_column2d_dense_modified_newton_load_control_matches_newton():
    base_case = _base_force_beam_case(
        {
            "id": 1,
            "type": "Steel01",
            "params": {"Fy": 250000000.0, "E0": 200000000000.0, "b": 0.01},
        }
    )
    base_case["loads"] = [{"node": 2, "dof": 2, "value": 400000.0}]
    base_case["recorders"] = [
        {"type": "element_force", "elements": [1], "output": "element_force"},
        {"type": "node_displacement", "nodes": [2], "dofs": [2], "output": "disp"},
    ]

    results = {}
    for algorithm in ("Newton", "ModifiedNewton"):
        case_data = json.loads(json.dumps(base_case))
        case_data["analysis"] = {
            "type": "static_nonlinear",
            "steps": 2,
            "max_iters": 10,
            "tol": 1e-8,
            "rel_tol": 1e-8,
            "algorithm": algorithm,
            "solver": "dense",
            "integrator": {"type": "LoadControl"},
        }

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            _run_strut_case(case_data, out_dir)
            results[algorithm] = {
                "forces": _read_rows(out_dir / "element_force_ele1.out"),
                "disp": _read_rows(out_dir / "disp_node2.out"),
            }

    for modified_row, newton_row in zip(
        results["ModifiedNewton"]["forces"], results["Newton"]["forces"], strict=True
    ):
        assert modified_row == pytest.approx(newton_row, abs=2e-9, rel=1e-9)
    for modified_row, newton_row in zip(
        results["ModifiedNewton"]["disp"], results["Newton"]["disp"], strict=True
    ):
        assert modified_row == pytest.approx(newton_row, abs=1e-12, rel=1e-12)


def test_force_beam_column2d_displacement_control_uses_static_fallback_controls():
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
        "solver_chain": [
            {
                "algorithm": "Newton",
                "test_type": "MaxDispIncr",
                "tol": 1e-9,
                "max_iters": 1,
            },
            {
                "algorithm": "ModifiedNewtonInitial",
                "test_type": "NormDispIncr",
                "tol": 1e-9,
                "max_iters": 40,
            },
        ],
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
        _run_strut_case(case_data, out_dir)
        rows = _read_rows(out_dir / "element_force_ele1.out")

    assert len(rows) == 3
    assert rows[0][1] * rows[1][1] < 0.0


def test_force_beam_column2d_section_recorders_emit_force_and_deformation():
    case_data = _base_force_beam_case(
        {"id": 1, "type": "Elastic", "params": {"E": 30000000000.0}}
    )
    case_data["analysis"] = {
        "type": "static_nonlinear",
        "steps": 2,
        "max_iters": 20,
        "tol": 1e-10,
        "integrator": {"type": "LoadControl"},
    }
    case_data["recorders"] = [
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
        force_rows = _read_rows(out_dir / "sec_force_ele1_sec1.out")
        defo_rows = _read_rows(out_dir / "sec_defo_ele1_sec3.out")

    assert len(force_rows) == 2
    assert len(defo_rows) == 2
    assert all(len(row) == 2 for row in force_rows)
    assert all(len(row) == 2 for row in defo_rows)
    assert all(math.isfinite(value) for row in force_rows for value in row)
    assert all(math.isfinite(value) for row in defo_rows for value in row)


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
        _run_strut_case(case_data, out_dir)
        rows = _read_rows(out_dir / "element_force_ele1.out")

    assert len(rows) == 3
    # Global Fy at node i should reverse sign when control displacement reverses.
    assert rows[0][1] * rows[1][1] < 0.0


def test_force_beam_column2d_staged_displacement_control_commits_final_force_state():
    case_data = _base_force_beam_case(
        {"id": 1, "type": "Elastic", "params": {"E": 30000000000.0}}
    )
    case_data["time_series"] = [{"type": "Linear", "tag": 1, "factor": 1.0}]
    case_data["loads"] = []
    case_data["analysis"] = {
        "type": "staged",
        "stages": [
            {
                "analysis": {
                    "type": "static_nonlinear",
                    "steps": 1,
                    "solver_chain": [
                        {
                            "algorithm": "Newton",
                            "test_type": "MaxDispIncr",
                            "tol": 1e-10,
                            "max_iters": 20,
                        }
                    ],
                    "integrator": {"type": "LoadControl"},
                },
                "pattern": {"type": "Plain", "tag": 1, "time_series": 1},
                "loads": [{"node": 2, "dof": 2, "value": 1.0}],
                "load_const": {"time": 0.0},
            },
            {
                "analysis": {
                    "type": "static_nonlinear",
                    "steps": 1,
                    "solver_chain": [
                        {
                            "algorithm": "Newton",
                            "test_type": "MaxDispIncr",
                            "tol": 1e-10,
                            "max_iters": 20,
                        },
                        {
                            "algorithm": "ModifiedNewtonInitial",
                            "test_type": "NormDispIncr",
                            "tol": 1e-10,
                            "max_iters": 50,
                        },
                    ],
                    "integrator": {
                        "type": "DisplacementControl",
                        "node": 2,
                        "dof": 1,
                        "du": 0.01,
                    },
                },
                "pattern": {"type": "Plain", "tag": 2, "time_series": 1},
                "loads": [{"node": 2, "dof": 1, "value": 1.0}],
            },
        ],
    }
    case_data["recorders"] = [
        {"type": "element_force", "elements": [1], "output": "element_force"},
        {"type": "node_displacement", "nodes": [2], "dofs": [1], "output": "disp"},
    ]

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        _run_strut_case(case_data, out_dir)
        force_rows = _read_rows(out_dir / "element_force_ele1.out")
        disp_rows = _read_rows(out_dir / "disp_node2.out")

    assert len(force_rows) == 2
    assert len(disp_rows) == 2
    assert disp_rows[-1][0] > disp_rows[0][0]
    assert force_rows[-1] != pytest.approx(force_rows[0], abs=1e-9, rel=1e-9)


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
        _run_strut_case(case_data, out_dir)
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
        _run_strut_case(case_data, out_dir)
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
            _run_strut_case(case_data, out_dir)


def test_force_beam_column2d_with_elastic_section2d_static_linear_runs():
    case_data = {
        "schema_version": "1.0",
        "metadata": {
            "name": "force_beam_column2d_elastic_section2d_unit",
            "units": "SI",
        },
        "model": {"ndm": 2, "ndf": 3},
        "nodes": [
            {"id": 1, "x": 0.0, "y": 0.0, "constraints": [1, 2, 3]},
            {"id": 2, "x": 3.0, "y": 0.0},
        ],
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
                "type": "forceBeamColumn2d",
                "nodes": [1, 2],
                "section": 1,
                "geomTransf": "Linear",
                "integration": "Lobatto",
                "num_int_pts": 5,
            }
        ],
        "loads": [{"node": 2, "dof": 2, "value": 1.0}],
        "analysis": {"type": "static_linear", "steps": 1},
        "recorders": [
            {"type": "element_force", "elements": [1], "output": "element_force"}
        ],
    }

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        _run_strut_case(case_data, out_dir)
        rows = _read_rows(out_dir / "element_force_ele1.out")

    assert len(rows) == 1
    assert len(rows[0]) == 6
    assert all(math.isfinite(value) for value in rows[0])


def test_force_beam_column2d_beam_uniform_reports_zero_free_end_forces():
    case_data = {
        "schema_version": "1.0",
        "metadata": {"name": "force_beam_column2d_uniform_load_unit", "units": "SI"},
        "model": {"ndm": 2, "ndf": 3},
        "nodes": [
            {"id": 1, "x": 0.0, "y": 0.0, "constraints": [1, 2, 3]},
            {"id": 2, "x": 3.0, "y": 0.0},
        ],
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
                "type": "forceBeamColumn2d",
                "nodes": [1, 2],
                "section": 1,
                "geomTransf": "Linear",
                "integration": "Lobatto",
                "num_int_pts": 5,
            }
        ],
        "element_loads": [{"element": 1, "type": "beamUniform", "wy": -2.0}],
        "analysis": {
            "type": "static_linear",
            "steps": 1,
            "force_beam_mode": "linear_if_elastic",
        },
        "recorders": [
            {"type": "element_force", "elements": [1], "output": "element_force"}
        ],
    }

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        _run_strut_case(case_data, out_dir)
        rows = _read_rows(out_dir / "element_force_ele1.out")

    assert len(rows) == 1
    row = rows[0]
    # Cantilever free end should report near-zero end force under pure beamUniform loading.
    assert row[3] == pytest.approx(0.0, abs=1e-8)
    assert row[4] == pytest.approx(0.0, abs=1e-8)
    assert row[5] == pytest.approx(0.0, abs=1e-8)


def test_force_beam_column2d_elastic_section_static_nonlinear_applies_beam_uniform_load():
    case_data = {
        "schema_version": "1.0",
        "metadata": {
            "name": "force_beam_column2d_elastic_section_gravity_unit",
            "units": "SI",
        },
        "model": {"ndm": 2, "ndf": 3},
        "nodes": [
            {"id": 1, "x": 0.0, "y": 0.0, "constraints": [1, 2, 3]},
            {"id": 2, "x": 5.0, "y": 0.0},
        ],
        "sections": [
            {
                "id": 1,
                "type": "ElasticSection2d",
                "params": {"E": 30000.0, "A": 100.0, "I": 1000.0},
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
                "num_int_pts": 5,
            }
        ],
        "element_loads": [{"element": 1, "type": "beamUniform", "wy": -2.0}],
        "analysis": {
            "type": "static_nonlinear",
            "steps": 1,
            "algorithm": "Newton",
            "integrator": {"type": "LoadControl", "step": 1.0},
        },
        "recorders": [
            {"type": "node_displacement", "nodes": [2], "dofs": [2], "output": "disp"},
            {"type": "element_force", "elements": [1], "output": "element_force"},
        ],
    }

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        _run_strut_case(case_data, out_dir)
        disp_rows = _read_rows(out_dir / "disp_node2.out")
        force_rows = _read_rows(out_dir / "element_force_ele1.out")

    assert len(disp_rows) == 1
    assert len(force_rows) == 1
    assert disp_rows[0][0] < 0.0
    assert any(abs(value) > 1.0e-9 for value in force_rows[0])


def test_force_beam_column2d_accepts_corotational_geom_transf():
    case_data = _base_force_beam_case(
        {"id": 1, "type": "Elastic", "params": {"E": 30000000000.0}}
    )
    case_data["elements"][0]["geomTransf"] = "Corotational"
    case_data["analysis"] = {
        "type": "static_nonlinear",
        "steps": 1,
        "integrator": {"type": "LoadControl"},
    }

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        _run_strut_case(case_data, out_dir)
        rows = _read_rows(out_dir / "element_force_ele1.out")

    assert len(rows) == 1
    assert all(math.isfinite(value) for value in rows[0])


def test_force_beam_column2d_radau_variable_points_runs():
    case_data = _base_force_beam_case(
        {"id": 1, "type": "Elastic", "params": {"E": 30000000000.0}}
    )
    case_data["elements"][0]["integration"] = "Radau"
    case_data["elements"][0]["num_int_pts"] = 4
    case_data["recorders"] = [
        {"type": "element_force", "elements": [1], "output": "element_force"},
        {
            "type": "section_force",
            "elements": [1],
            "sections": [1, 4],
            "output": "sec_force",
        },
    ]
    case_data["analysis"] = {
        "type": "static_nonlinear",
        "steps": 2,
        "integrator": {"type": "LoadControl"},
    }

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        _run_strut_case(case_data, out_dir)
        rows = _read_rows(out_dir / "element_force_ele1.out")
        sec_rows = _read_rows(out_dir / "sec_force_ele1_sec4.out")

    assert len(rows) == 2
    assert len(sec_rows) == 2
    assert all(len(row) == 6 for row in rows)
    assert all(len(row) == 2 for row in sec_rows)
    assert all(math.isfinite(value) for row in rows for value in row)
    assert all(math.isfinite(value) for row in sec_rows for value in row)


def test_force_beam_column2d_fiber_pdelta_geom_changes_response():
    base_case = _base_force_beam_case(
        {"id": 1, "type": "Elastic", "params": {"E": 30000000000.0}}
    )
    base_case["nodes"] = [
        {"id": 1, "x": 0.0, "y": 0.0, "constraints": [1, 2, 3]},
        {"id": 2, "x": 0.0, "y": 3.0},
    ]
    base_case["loads"] = [
        {"node": 2, "dof": 1, "value": 1.0e5},
        {"node": 2, "dof": 2, "value": -1.0e7},
    ]
    base_case["analysis"] = {
        "type": "static_nonlinear",
        "steps": 3,
        "max_iters": 30,
        "tol": 1e-9,
        "integrator": {"type": "LoadControl"},
    }

    linear_case = json.loads(json.dumps(base_case))
    pdelta_case = json.loads(json.dumps(base_case))
    linear_case["elements"][0]["geomTransf"] = "Linear"
    pdelta_case["elements"][0]["geomTransf"] = "PDelta"

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        linear_out = tmp_path / "linear"
        pdelta_out = tmp_path / "pdelta"
        linear_out.mkdir()
        pdelta_out.mkdir()
        _run_strut_case(linear_case, linear_out)
        _run_strut_case(pdelta_case, pdelta_out)
        linear_rows = _read_rows(linear_out / "element_force_ele1.out")
        pdelta_rows = _read_rows(pdelta_out / "element_force_ele1.out")

    assert len(linear_rows) == 3
    assert len(pdelta_rows) == 3
    diff = sum(abs(a - b) for a, b in zip(linear_rows[-1], pdelta_rows[-1]))
    assert diff > 1.0e-3
