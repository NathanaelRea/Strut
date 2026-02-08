import json
import subprocess
import sys
import tempfile
from pathlib import Path


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


def _base_case():
    return {
        "schema_version": "1.0",
        "metadata": {"name": "modal_constraints_tcl_unit", "units": "SI"},
        "model": {"ndm": 2, "ndf": 3},
        "nodes": [
            {"id": 1, "x": 0.0, "y": 0.0, "constraints": [1, 2, 3]},
            {"id": 2, "x": 0.0, "y": 3.0},
            {"id": 3, "x": 4.0, "y": 3.0},
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
            },
            {
                "id": 2,
                "type": "elasticBeamColumn2d",
                "nodes": [2, 3],
                "section": 1,
                "geomTransf": "Linear",
            },
        ],
        "masses": [
            {"node": 2, "dof": 1, "value": 1.0},
            {"node": 2, "dof": 2, "value": 1.0},
            {"node": 3, "dof": 1, "value": 1.0},
            {"node": 3, "dof": 2, "value": 1.0},
        ],
        "recorders": [],
    }


def test_json_to_tcl_emits_transformation_and_equaldof():
    case = _base_case()
    case["analysis"] = {"type": "static_linear", "steps": 1, "constraints": "Transformation"}
    case["mp_constraints"] = [
        {"type": "equalDOF", "retained_node": 2, "constrained_node": 3, "dofs": [1]}
    ]
    case["loads"] = [{"node": 2, "dof": 2, "value": -1000.0}]

    text = _run_json_to_tcl(case)

    assert "constraints Transformation\n" in text
    assert "equalDOF 2 3 1\n" in text


def test_json_to_tcl_emits_modal_eigen_outputs():
    case = _base_case()
    case["analysis"] = {"type": "modal_eigen", "num_modes": 2, "steps": 1}
    case["recorders"] = [
        {
            "type": "modal_eigen",
            "modes": [1, 2],
            "nodes": [2, 3],
            "dofs": [1, 2],
            "output": "modal",
        }
    ]

    text = _run_json_to_tcl(case)

    assert "set strut_modal_lambda [eigen 2]\n" in text
    assert "open \"modal_eigenvalues.out\" \"w\"" in text
    assert "nodeEigenvector 2 1 1" in text
    assert "nodeEigenvector 3 2 2" in text
