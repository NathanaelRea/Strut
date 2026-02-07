import math
import tempfile
from pathlib import Path
import json
import subprocess
import sys

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

def _run_mojo_case(case_data, out_dir: Path):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        tmp.write(json.dumps(case_data))
        tmp_path = Path(tmp.name)
    try:
        subprocess.check_call(
            [
                sys.executable,
                str(repo_root / "scripts" / "run_mojo_case.py"),
                "--input",
                str(tmp_path),
                "--output",
                str(out_dir),
            ]
        )
    finally:
        tmp_path.unlink(missing_ok=True)


def test_simply_supported_midspan_point_load():
    E = 200000000000.0
    A = 0.02
    I = 0.0001
    L = 2.0
    P = 1000.0

    case_data = {
        "schema_version": "1.0",
        "metadata": {"name": "unit_simply_supported_midspan", "units": "SI"},
        "model": {"ndm": 2, "ndf": 3},
        "nodes": [
            {"id": 1, "x": 0.0, "y": 0.0, "constraints": [1, 2]},
            {"id": 2, "x": L / 2.0, "y": 0.0},
            {"id": 3, "x": L, "y": 0.0, "constraints": [1, 2]},
        ],
        "materials": [{"id": 1, "type": "Elastic", "params": {"E": E}}],
        "sections": [
            {"id": 1, "type": "ElasticSection2d", "params": {"E": E, "A": A, "I": I}}
        ],
        "elements": [
            {"id": 1, "type": "elasticBeamColumn2d", "nodes": [1, 2], "section": 1, "geomTransf": "Linear"},
            {"id": 2, "type": "elasticBeamColumn2d", "nodes": [2, 3], "section": 1, "geomTransf": "Linear"},
        ],
        "loads": [{"node": 2, "dof": 2, "value": -P}],
        "analysis": {"type": "static_linear", "steps": 1},
        "recorders": [
            {"type": "node_displacement", "nodes": [2], "dofs": [1, 2, 3], "output": "node_disp"}
        ],
    }

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        _run_mojo_case(case_data, out_dir)
        disp_file = out_dir / "node_disp_node2.out"
        values = [float(v) for v in disp_file.read_text().split()]

    expected_v = -P * L**3 / (48 * E * I)

    assert math.isclose(values[1], expected_v, rel_tol=1e-6, abs_tol=1e-12)
    assert math.isclose(values[2], 0.0, abs_tol=1e-9)


def test_simply_supported_uniform_load():
    E = 200000000000.0
    A = 0.02
    I = 0.0001
    L = 2.0
    w = -1000.0

    case_data = {
        "schema_version": "1.0",
        "metadata": {"name": "unit_simply_supported_uniform", "units": "SI"},
        "model": {"ndm": 2, "ndf": 3},
        "nodes": [
            {"id": 1, "x": 0.0, "y": 0.0, "constraints": [1, 2]},
            {"id": 2, "x": L / 2.0, "y": 0.0},
            {"id": 3, "x": L, "y": 0.0, "constraints": [1, 2]},
        ],
        "materials": [{"id": 1, "type": "Elastic", "params": {"E": E}}],
        "sections": [
            {"id": 1, "type": "ElasticSection2d", "params": {"E": E, "A": A, "I": I}}
        ],
        "elements": [
            {"id": 1, "type": "elasticBeamColumn2d", "nodes": [1, 2], "section": 1, "geomTransf": "Linear"},
            {"id": 2, "type": "elasticBeamColumn2d", "nodes": [2, 3], "section": 1, "geomTransf": "Linear"},
        ],
        "element_loads": [
            {"element": 1, "type": "beamUniform", "w": w},
            {"element": 2, "type": "beamUniform", "w": w},
        ],
        "analysis": {"type": "static_linear", "steps": 1},
        "recorders": [
            {"type": "node_displacement", "nodes": [2], "dofs": [1, 2, 3], "output": "node_disp"}
        ],
    }

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        _run_mojo_case(case_data, out_dir)
        disp_file = out_dir / "node_disp_node2.out"
        values = [float(v) for v in disp_file.read_text().split()]

    expected_v = 5.0 * w * L**4 / (384 * E * I)

    assert math.isclose(values[1], expected_v, rel_tol=1e-6, abs_tol=1e-12)
    assert math.isclose(values[2], 0.0, abs_tol=1e-9)
