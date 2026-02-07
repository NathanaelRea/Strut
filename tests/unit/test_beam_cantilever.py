import json
import math
import tempfile
from pathlib import Path
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
        _run_mojo_case(case_data, out_dir)
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
