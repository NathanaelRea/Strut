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


def test_json_to_tcl_emits_reaction_drift_and_envelope_recorders():
    case = {
        "schema_version": "1.0",
        "metadata": {"name": "recorder_tcl_unit", "units": "SI"},
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
        "loads": [{"node": 2, "dof": 1, "value": 1.0}],
        "analysis": {"type": "static_linear", "steps": 1},
        "recorders": [
            {"type": "node_reaction", "nodes": [1], "dofs": [1, 2, 3], "output": "reaction"},
            {"type": "drift", "i_node": 1, "j_node": 2, "dof": 1, "perp_dirn": 2, "output": "drift"},
            {"type": "envelope_element_force", "elements": [1], "output": "env_force"},
            {
                "type": "section_force",
                "elements": [1],
                "sections": [1],
                "output": "sec_force",
            },
            {
                "type": "section_deformation",
                "elements": [1],
                "section": 1,
                "output": "sec_defo",
            },
        ],
    }

    text = _run_json_to_tcl(case)

    assert "recorder Node -file reaction_node1.out -node 1 -dof 1 2 3 reaction\n" in text
    assert "recorder Drift -file drift_i1_j2.out -iNode 1 -jNode 2 -dof 1 -perpDirn 2\n" in text
    assert "recorder EnvelopeElement -file env_force_ele1.out -ele 1 forces\n" in text
    assert "recorder Element -file sec_force_ele1_sec1.out -ele 1 section 1 force\n" in text
    assert "recorder Element -file sec_defo_ele1_sec1.out -ele 1 section 1 deformation\n" in text


def test_json_to_tcl_emits_beam_uniform_with_optional_wx():
    case = {
        "schema_version": "1.0",
        "metadata": {"name": "beam_uniform_tcl_unit", "units": "SI"},
        "model": {"ndm": 2, "ndf": 3},
        "nodes": [
            {"id": 1, "x": 0.0, "y": 0.0, "constraints": [1, 2, 3]},
            {"id": 2, "x": 3.0, "y": 0.0},
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
        "pattern": {"type": "Plain", "tag": 1, "time_series": 1},
        "time_series": [{"type": "Linear", "tag": 1, "factor": 1.0}],
        "element_loads": [{"element": 1, "type": "beamUniform", "wy": -2.0, "wx": -0.5}],
        "analysis": {"type": "static_linear", "steps": 1},
        "recorders": [{"type": "element_force", "elements": [1], "output": "force"}],
    }

    text = _run_json_to_tcl(case)
    assert "eleLoad -ele 1 -type -beamUniform -2.0 -0.5\n" in text


def test_json_to_tcl_emits_legacy_beam_uniform_w_alias():
    case = {
        "schema_version": "1.0",
        "metadata": {"name": "beam_uniform_legacy_tcl_unit", "units": "SI"},
        "model": {"ndm": 2, "ndf": 3},
        "nodes": [
            {"id": 1, "x": 0.0, "y": 0.0, "constraints": [1, 2, 3]},
            {"id": 2, "x": 3.0, "y": 0.0},
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
        "pattern": {"type": "Plain", "tag": 1, "time_series": 1},
        "time_series": [{"type": "Linear", "tag": 1, "factor": 1.0}],
        "element_loads": [{"element": 1, "type": "beamUniform", "w": -2.0}],
        "analysis": {"type": "static_linear", "steps": 1},
        "recorders": [{"type": "element_force", "elements": [1], "output": "force"}],
    }

    text = _run_json_to_tcl(case)
    assert "eleLoad -ele 1 -type -beamUniform -2.0\n" in text


def test_json_to_tcl_emits_beam_point_2d_with_optional_px():
    case = {
        "schema_version": "1.0",
        "metadata": {"name": "beam_point_tcl_unit", "units": "SI"},
        "model": {"ndm": 2, "ndf": 3},
        "nodes": [
            {"id": 1, "x": 0.0, "y": 0.0, "constraints": [1, 2, 3]},
            {"id": 2, "x": 4.0, "y": 0.0},
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
        "pattern": {"type": "Plain", "tag": 1, "time_series": 1},
        "time_series": [{"type": "Linear", "tag": 1, "factor": 1.0}],
        "element_loads": [
            {"element": 1, "type": "beamPoint", "py": -3.0, "x": 0.4, "px": 1.25}
        ],
        "analysis": {"type": "static_linear", "steps": 1},
        "recorders": [{"type": "element_force", "elements": [1], "output": "force"}],
    }

    text = _run_json_to_tcl(case)
    assert "eleLoad -ele 1 -type -beamPoint -3.0 0.4 1.25\n" in text


def test_json_to_tcl_emits_3d_beam_uniform_and_point():
    case = {
        "schema_version": "1.0",
        "metadata": {"name": "beam_loads_3d_tcl_unit", "units": "SI"},
        "model": {"ndm": 3, "ndf": 6},
        "nodes": [
            {"id": 1, "x": 0.0, "y": 0.0, "z": 0.0, "constraints": [1, 2, 3, 4, 5, 6]},
            {"id": 2, "x": 0.0, "y": 0.0, "z": 4.0},
        ],
        "materials": [],
        "sections": [
            {
                "id": 1,
                "type": "ElasticSection3d",
                "params": {
                    "E": 2.0e11,
                    "A": 0.02,
                    "Iy": 8.0e-5,
                    "Iz": 6.0e-5,
                    "G": 8.0e10,
                    "J": 1.0e-4,
                },
            }
        ],
        "elements": [
            {
                "id": 1,
                "type": "elasticBeamColumn3d",
                "nodes": [1, 2],
                "section": 1,
                "geomTransf": "Linear",
            }
        ],
        "pattern": {"type": "Plain", "tag": 1, "time_series": 1},
        "time_series": [{"type": "Linear", "tag": 1, "factor": 1.0}],
        "element_loads": [
            {"element": 1, "type": "beamUniform", "wy": -2.0, "wz": 1.5, "wx": 0.25},
            {"element": 1, "type": "beamPoint", "py": -3.0, "pz": 4.0, "x": 0.35, "px": 0.5},
        ],
        "analysis": {"type": "static_linear", "steps": 1},
        "recorders": [{"type": "element_force", "elements": [1], "output": "force"}],
    }

    text = _run_json_to_tcl(case)
    assert "eleLoad -ele 1 -type -beamUniform -2.0 1.5 0.25\n" in text
    assert "eleLoad -ele 1 -type -beamPoint -3.0 4.0 0.35 0.5\n" in text
