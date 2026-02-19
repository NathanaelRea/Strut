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


def _read_rows(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append([float(x) for x in line.split()])
    return rows


def test_reaction_drift_and_envelope_recorders_static_linear():
    case_data = {
        "schema_version": "1.0",
        "metadata": {"name": "recorders_static_linear_unit", "units": "SI"},
        "model": {"ndm": 2, "ndf": 3},
        "nodes": [
            {"id": 1, "x": 0.0, "y": 0.0, "constraints": [1, 2, 3]},
            {"id": 2, "x": 0.0, "y": 3.0},
        ],
        "materials": [{"id": 1, "type": "Elastic", "params": {"E": 2.0e11}}],
        "sections": [
            {
                "id": 1,
                "type": "ElasticSection2d",
                "params": {"E": 2.0e11, "A": 0.02, "I": 8.0e-5},
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
        "loads": [{"node": 2, "dof": 1, "value": 1000.0}],
        "analysis": {"type": "static_linear", "steps": 1},
        "recorders": [
            {
                "type": "node_reaction",
                "nodes": [1],
                "dofs": [1, 2, 3],
                "output": "reaction",
            },
            {
                "type": "drift",
                "i_node": 1,
                "j_node": 2,
                "dof": 1,
                "perp_dirn": 2,
                "output": "drift",
            },
            {"type": "envelope_element_force", "elements": [1], "output": "env_force"},
        ],
    }

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        _run_strut_case(case_data, out_dir)

        reaction_rows = _read_rows(out_dir / "reaction_node1.out")
        drift_rows = _read_rows(out_dir / "drift_i1_j2.out")
        env_rows = _read_rows(out_dir / "env_force_ele1.out")

    assert len(reaction_rows) == 1
    assert len(reaction_rows[0]) == 3
    assert all(math.isfinite(v) for v in reaction_rows[0])

    assert len(drift_rows) == 1
    assert len(drift_rows[0]) == 1
    assert math.isfinite(drift_rows[0][0])

    assert len(env_rows) == 3
    assert all(len(row) == len(env_rows[0]) for row in env_rows)
    assert all(math.isfinite(v) for row in env_rows for v in row)
