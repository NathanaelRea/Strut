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


def _read_last_row(path: Path) -> list[float]:
    lines = [
        ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()
    ]
    assert lines
    return [float(x) for x in lines[-1].split()]


def test_zero_length_section_2d_aggregator_axial_shear_and_flexure():
    e_p = 5000.0
    e_vy = 1200.0
    e_mz = 8000.0
    p_load = -150.0
    v_load = 60.0
    m_load = 240.0
    case = {
        "schema_version": "1.0",
        "metadata": {"name": "zls2d_agg_unit", "units": "SI"},
        "model": {"ndm": 2, "ndf": 3},
        "nodes": [
            {"id": 1, "x": 0.0, "y": 0.0, "constraints": [1, 2, 3]},
            {"id": 2, "x": 0.0, "y": 0.0},
        ],
        "materials": [
            {"id": 1, "type": "Elastic", "params": {"E": e_p}},
            {"id": 2, "type": "Elastic", "params": {"E": e_vy}},
            {"id": 3, "type": "Elastic", "params": {"E": e_mz}},
        ],
        "sections": [
            {
                "id": 10,
                "type": "AggregatorSection2d",
                "params": {
                    "axial_material": 1,
                    "shear_y_material": 2,
                    "flexural_material": 3,
                    "moment_y_material": -1,
                    "torsion_material": -1,
                    "shear_z_material": -1,
                    "base_section": -1,
                },
            }
        ],
        "elements": [
            {"id": 1, "type": "zeroLengthSection", "nodes": [1, 2], "section": 10}
        ],
        "loads": [
            {"node": 2, "dof": 1, "value": p_load},
            {"node": 2, "dof": 2, "value": v_load},
            {"node": 2, "dof": 3, "value": m_load},
        ],
        "analysis": {"type": "static_linear", "steps": 1},
        "recorders": [
            {
                "type": "node_displacement",
                "nodes": [2],
                "dofs": [1, 2, 3],
                "output": "disp",
            }
        ],
    }

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        _run_strut_case(case, out_dir)
        disp = _read_last_row(out_dir / "disp_node2.out")

    assert math.isclose(disp[0], p_load / e_p, rel_tol=1e-10, abs_tol=1e-12)
    assert math.isclose(disp[1], v_load / e_vy, rel_tol=1e-10, abs_tol=1e-12)
    assert math.isclose(disp[2], m_load / e_mz, rel_tol=1e-10, abs_tol=1e-12)


def test_zero_length_section_3d_elastic_and_aggregator_with_circ_fiber_base():
    e_val = 2.0e11
    a_val = 0.02
    iy_val = 3.0e-4
    iz_val = 5.0e-4
    g_val = 8.0e10
    j_val = 4.0e-4
    loads_3d = [
        {"node": 2, "dof": 1, "value": 1000.0},
        {"node": 2, "dof": 4, "value": 500.0},
        {"node": 2, "dof": 5, "value": -300.0},
        {"node": 2, "dof": 6, "value": 400.0},
    ]
    elastic_case = {
        "schema_version": "1.0",
        "metadata": {"name": "zls3d_elastic_unit", "units": "SI"},
        "model": {"ndm": 3, "ndf": 6},
        "nodes": [
            {"id": 1, "x": 0.0, "y": 0.0, "z": 0.0, "constraints": [1, 2, 3, 4, 5, 6]},
            {"id": 2, "x": 0.0, "y": 0.0, "z": 0.0, "constraints": [2, 3]},
        ],
        "materials": [],
        "sections": [
            {
                "id": 1,
                "type": "ElasticSection3d",
                "params": {
                    "E": e_val,
                    "A": a_val,
                    "Iy": iy_val,
                    "Iz": iz_val,
                    "G": g_val,
                    "J": j_val,
                },
            }
        ],
        "elements": [
            {"id": 1, "type": "zeroLengthSection", "nodes": [1, 2], "section": 1}
        ],
        "loads": loads_3d,
        "analysis": {"type": "static_linear", "steps": 1},
        "recorders": [
            {
                "type": "node_displacement",
                "nodes": [2],
                "dofs": [1, 4, 5, 6],
                "output": "disp",
            }
        ],
    }

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        _run_strut_case(elastic_case, out_dir)
        disp = _read_last_row(out_dir / "disp_node2.out")

    assert math.isclose(disp[0], 1000.0 / (e_val * a_val), rel_tol=1e-10, abs_tol=1e-15)
    assert math.isclose(disp[1], 500.0 / (g_val * j_val), rel_tol=1e-10, abs_tol=1e-15)
    assert math.isclose(
        disp[2], -300.0 / (e_val * iy_val), rel_tol=1e-10, abs_tol=1e-15
    )
    assert math.isclose(disp[3], 400.0 / (e_val * iz_val), rel_tol=1e-10, abs_tol=1e-15)

    # FiberSection3d with circular patch and elastic torsion (G*J) plus standalone AggregatorSection3d.
    r_ext = 0.5
    e_conc = 3.0e10
    gj_torsion = 1.0e9
    fiber3d_case = {
        "schema_version": "1.0",
        "metadata": {"name": "zls3d_fiber_circ_unit", "units": "SI"},
        "model": {"ndm": 3, "ndf": 6},
        "nodes": [
            {"id": 1, "x": 0.0, "y": 0.0, "z": 0.0, "constraints": [1, 2, 3, 4, 5, 6]},
            {"id": 2, "x": 0.0, "y": 0.0, "z": 0.0, "constraints": [2, 3]},
        ],
        "materials": [{"id": 1, "type": "Elastic", "params": {"E": e_conc}}],
        "sections": [
            {
                "id": 1,
                "type": "FiberSection3d",
                "params": {
                    "G": gj_torsion,
                    "J": 1.0,
                    "patches": [
                        {
                            "type": "circ",
                            "material": 1,
                            "num_subdiv_circ": 32,
                            "num_subdiv_rad": 16,
                            "y_center": 0.0,
                            "z_center": 0.0,
                            "int_rad": 0.0,
                            "ext_rad": r_ext,
                            "start_ang": 0.0,
                            "end_ang": 360.0,
                        }
                    ],
                    "layers": [],
                },
            }
        ],
        "elements": [
            {"id": 1, "type": "zeroLengthSection", "nodes": [1, 2], "section": 1}
        ],
        "loads": [
            {"node": 2, "dof": 1, "value": -1.0e6},
            {"node": 2, "dof": 4, "value": 2.0e5},
            {"node": 2, "dof": 6, "value": 5.0e5},
        ],
        "analysis": {"type": "static_linear", "steps": 1},
        "recorders": [
            {
                "type": "node_displacement",
                "nodes": [2],
                "dofs": [1, 4, 6],
                "output": "disp",
            }
        ],
    }

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        _run_strut_case(fiber3d_case, out_dir)
        disp = _read_last_row(out_dir / "disp_node2.out")

    area_exact = math.pi * r_ext * r_ext
    i_exact = 0.25 * math.pi * (r_ext**4)
    assert math.isclose(
        disp[0], -1.0e6 / (e_conc * area_exact), rel_tol=1e-6, abs_tol=1e-15
    )
    assert math.isclose(disp[1], 2.0e5 / gj_torsion, rel_tol=1e-10, abs_tol=1e-15)
    assert math.isclose(
        disp[2], 5.0e5 / (e_conc * i_exact), rel_tol=1e-2, abs_tol=1e-15
    )

    # Standalone AggregatorSection3d covering P, Vy, Vz, T, My, Mz.
    agg3d_case = {
        "schema_version": "1.0",
        "metadata": {"name": "zls3d_agg_unit", "units": "SI"},
        "model": {"ndm": 3, "ndf": 6},
        "nodes": [
            {"id": 1, "x": 0.0, "y": 0.0, "z": 0.0, "constraints": [1, 2, 3, 4, 5, 6]},
            {"id": 2, "x": 0.0, "y": 0.0, "z": 0.0},
        ],
        "materials": [
            {"id": 1, "type": "Elastic", "params": {"E": 1000.0}},
            {"id": 2, "type": "Elastic", "params": {"E": 2000.0}},
            {"id": 3, "type": "Elastic", "params": {"E": 3000.0}},
            {"id": 4, "type": "Elastic", "params": {"E": 4000.0}},
            {"id": 5, "type": "Elastic", "params": {"E": 5000.0}},
            {"id": 6, "type": "Elastic", "params": {"E": 6000.0}},
        ],
        "sections": [
            {
                "id": 2,
                "type": "AggregatorSection3d",
                "params": {
                    "axial_material": 1,
                    "shear_y_material": 2,
                    "shear_z_material": 3,
                    "torsion_material": 4,
                    "moment_y_material": 5,
                    "flexural_material": 6,
                    "base_section": -1,
                },
            }
        ],
        "elements": [
            {"id": 1, "type": "zeroLengthSection", "nodes": [1, 2], "section": 2}
        ],
        "loads": [
            {"node": 2, "dof": 1, "value": 10.0},
            {"node": 2, "dof": 2, "value": 20.0},
            {"node": 2, "dof": 3, "value": 30.0},
            {"node": 2, "dof": 4, "value": 40.0},
            {"node": 2, "dof": 5, "value": 50.0},
            {"node": 2, "dof": 6, "value": 60.0},
        ],
        "analysis": {"type": "static_linear", "steps": 1},
        "recorders": [
            {
                "type": "node_displacement",
                "nodes": [2],
                "dofs": [1, 2, 3, 4, 5, 6],
                "output": "disp",
            }
        ],
    }

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        _run_strut_case(agg3d_case, out_dir)
        disp = _read_last_row(out_dir / "disp_node2.out")

    for idx in range(6):
        assert math.isclose(disp[idx], 0.01, rel_tol=1e-10, abs_tol=1e-12)
