import json
import math
import tempfile
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from scripts.run_mojo_case_py import run_case


def test_cantilever_tip_deflection():
    case_path = (
        Path(__file__).resolve().parents[1]
        / "validation"
        / "elastic_beam_cantilever"
        / "elastic_beam_cantilever.json"
    )
    case_data = json.loads(case_path.read_text())

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        run_case(case_data, out_dir)
        disp_file = out_dir / "node_disp_node2.out"
        values = [float(v) for v in disp_file.read_text().split()]

    # Expected: cantilever with tip load P
    E = 200000000000.0
    I = 0.0001
    L = 1.0
    P = 1000.0
    expected_v = -P * L**3 / (3 * E * I)
    expected_theta = -P * L**2 / (2 * E * I)

    # values: [ux, uy, rz]
    assert math.isclose(values[1], expected_v, rel_tol=1e-6, abs_tol=1e-12)
    assert math.isclose(values[2], expected_theta, rel_tol=1e-6, abs_tol=1e-12)
