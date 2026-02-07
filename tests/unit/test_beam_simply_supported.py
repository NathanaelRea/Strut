import math
import tempfile
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from scripts.run_mojo_case_py import run_case


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
        run_case(case_data, out_dir)
        disp_file = out_dir / "node_disp_node2.out"
        values = [float(v) for v in disp_file.read_text().split()]

    expected_v = -P * L**3 / (48 * E * I)

    assert math.isclose(values[1], expected_v, rel_tol=1e-6, abs_tol=1e-12)
    assert math.isclose(values[2], 0.0, abs_tol=1e-9)
