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


def test_json_to_tcl_emits_zero_length_orient_damp_mats_and_do_rayleigh():
    case = {
        "schema_version": "1.0",
        "metadata": {"name": "zero_length_tcl_unit", "units": "SI"},
        "model": {"ndm": 2, "ndf": 3},
        "nodes": [
            {"id": 1, "x": 0.0, "y": 0.0, "constraints": [1, 2, 3]},
            {"id": 2, "x": 0.0, "y": 0.0},
        ],
        "materials": [{"id": 1, "type": "Elastic", "params": {"E": 1000.0}}],
        "sections": [],
        "elements": [
            {
                "id": 1,
                "type": "zeroLength",
                "nodes": [1, 2],
                "materials": [1],
                "dirs": [1],
                "orient": {"x": [0.0, 1.0, 0.0]},
                "dampMats": [1],
                "doRayleigh": True,
            }
        ],
        "analysis": {"type": "static_linear", "steps": 1},
        "recorders": [{"type": "element_force", "elements": [1], "output": "force"}],
    }

    text = _run_json_to_tcl(case)

    assert (
        "element zeroLength 1 1 2 -mat 1 -dir 1 -orient 0.0 1.0 0.0 -dampMats 1 -doRayleigh 1\n"
        in text
    )


def test_json_to_tcl_emits_zero_length_damp_command():
    case = {
        "schema_version": "1.0",
        "metadata": {"name": "zero_length_damp_tcl_unit", "units": "SI"},
        "model": {"ndm": 2, "ndf": 2},
        "time_series": [{"type": "Linear", "tag": 9, "factor": 1.0}],
        "dampings": [
            {
                "id": 3,
                "type": "SecStif",
                "beta": 0.2,
                "activateTime": 0.1,
                "deactivateTime": 1.5,
                "factor": 9,
            }
        ],
        "nodes": [
            {"id": 1, "x": 0.0, "y": 0.0, "constraints": [1, 2]},
            {"id": 2, "x": 0.0, "y": 0.0},
        ],
        "materials": [{"id": 1, "type": "Elastic", "params": {"E": 1000.0}}],
        "sections": [],
        "elements": [
            {
                "id": 1,
                "type": "zeroLength",
                "nodes": [1, 2],
                "materials": [1],
                "dirs": [1],
                "damp": 3,
            }
        ],
        "analysis": {"type": "static_linear", "steps": 1},
        "recorders": [{"type": "element_force", "elements": [1], "output": "force"}],
    }

    text = _run_json_to_tcl(case)

    assert "timeSeries Linear 9 -factor 1.0\n" in text
    assert "damping SecStif 3 0.2 -activateTime 0.1 -deactivateTime 1.5 -factor 9\n" in text
    assert "element zeroLength 1 1 2 -mat 1 -dir 1 -damp 3\n" in text


def test_json_to_tcl_emits_two_node_link_full_option_set():
    case = {
        "schema_version": "1.0",
        "metadata": {"name": "two_node_link_tcl_unit", "units": "SI"},
        "model": {"ndm": 3, "ndf": 6},
        "nodes": [
            {"id": 1, "x": 0.0, "y": 0.0, "z": 0.0, "constraints": [1, 2, 3, 4, 5, 6]},
            {"id": 2, "x": 2.0, "y": 0.0, "z": 0.0},
        ],
        "materials": [
            {"id": 1, "type": "Elastic", "params": {"E": 1000.0}},
            {"id": 2, "type": "Elastic", "params": {"E": 500.0}},
        ],
        "sections": [],
        "elements": [
            {
                "id": 1,
                "type": "twoNodeLink",
                "nodes": [1, 2],
                "materials": [1, 2],
                "dirs": [2, 6],
                "orient": {"x": [0.0, 1.0, 0.0]},
                "pDelta": [0.0, 0.0, 0.1, 0.2],
                "shearDist": [0.25, 0.5],
                "doRayleigh": True,
                "mass": 2.0,
            }
        ],
        "analysis": {"type": "static_linear", "steps": 1},
        "recorders": [{"type": "element_force", "elements": [1], "output": "force"}],
    }

    text = _run_json_to_tcl(case)

    assert (
        "element twoNodeLink 1 1 2 -mat 1 2 -dir 2 6"
        " -orient 0.0 1.0 0.0 1.0 0.0 0.0"
        " -pDelta 0.0 0.0 0.1 0.2"
        " -shearDist 0.25 0.5"
        " -doRayleigh -mass 2.0\n"
        in text
    )


def test_json_to_tcl_emits_two_node_link_2d_pdelta_tail_values():
    case = {
        "schema_version": "1.0",
        "metadata": {"name": "two_node_link_2d_pdelta_tcl_unit", "units": "SI"},
        "model": {"ndm": 2, "ndf": 3},
        "nodes": [
            {"id": 1, "x": 0.0, "y": 0.0, "constraints": [1, 2, 3]},
            {"id": 2, "x": 2.0, "y": 0.0},
        ],
        "materials": [
            {"id": 1, "type": "Elastic", "params": {"E": 1000.0}},
            {"id": 2, "type": "Elastic", "params": {"E": 500.0}},
            {"id": 3, "type": "Elastic", "params": {"E": 250.0}},
        ],
        "sections": [],
        "elements": [
            {
                "id": 1,
                "type": "twoNodeLink",
                "nodes": [1, 2],
                "materials": [1, 2, 3],
                "dirs": [1, 2, 3],
                "pDelta": [0.0, 0.0, 0.3, 0.2],
            }
        ],
        "analysis": {"type": "static_linear", "steps": 1},
        "recorders": [{"type": "element_force", "elements": [1], "output": "force"}],
    }

    text = _run_json_to_tcl(case)

    assert "element twoNodeLink 1 1 2 -mat 1 2 3 -dir 1 2 3 -pDelta 0.3 0.2\n" in text
