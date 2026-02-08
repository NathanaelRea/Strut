import json
import subprocess
import sys
import tempfile
from pathlib import Path


repo_root = Path(__file__).resolve().parents[2]


def _run_json_to_tcl(case_data):
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        in_path = tmpdir / "case.json"
        out_path = tmpdir / "model.tcl"
        in_path.write_text(json.dumps(case_data), encoding="utf-8")
        proc = subprocess.run(
            [
                sys.executable,
                str(repo_root / "scripts" / "json_to_tcl.py"),
                str(in_path),
                str(out_path),
            ],
            capture_output=True,
            text=True,
        )
        text = out_path.read_text(encoding="utf-8") if out_path.exists() else ""
        return proc, text


def _base_case():
    return {
        "schema_version": "1.0",
        "metadata": {"name": "fiber_converter_unit", "units": "SI"},
        "model": {"ndm": 2, "ndf": 3},
        "nodes": [
            {"id": 1, "x": 0.0, "y": 0.0, "constraints": [1, 2, 3]},
            {"id": 2, "x": 1.0, "y": 0.0},
        ],
        "materials": [
            {"id": 1, "type": "Elastic", "params": {"E": 30000000000.0}},
            {"id": 2, "type": "Steel01", "params": {"Fy": 500000000.0, "E0": 200000000000.0, "b": 0.01}},
        ],
        "sections": [],
        "elements": [],
        "recorders": [],
        "analysis": {"type": "static_linear", "steps": 1},
    }


def test_json_to_tcl_emits_fiber_section_rect_and_straight():
    case_data = _base_case()
    case_data["sections"] = [
        {
            "id": 7,
            "type": "FiberSection2d",
            "params": {
                "patches": [
                    {
                        "type": "rect",
                        "material": 1,
                        "num_subdiv_y": 2,
                        "num_subdiv_z": 1,
                        "y_i": -0.2,
                        "z_i": -0.1,
                        "y_j": 0.2,
                        "z_j": 0.1,
                    }
                ],
                "layers": [
                    {
                        "type": "straight",
                        "material": 2,
                        "num_bars": 3,
                        "bar_area": 0.0002,
                        "y_start": -0.15,
                        "z_start": 0.08,
                        "y_end": 0.15,
                        "z_end": 0.08,
                    }
                ],
            },
        }
    ]

    proc, text = _run_json_to_tcl(case_data)
    assert proc.returncode == 0, proc.stderr
    fiber_block = (
        "section Fiber 7 {\n"
        "  patch rect 1 2 1 -0.2 -0.1 0.2 0.1\n"
        "  layer straight 2 3 0.0002 -0.15 0.08 0.15 0.08\n"
        "}\n"
    )
    assert fiber_block in text


def test_json_to_tcl_rejects_unsupported_fiber_patch_type():
    case_data = _base_case()
    case_data["sections"] = [
        {
            "id": 9,
            "type": "FiberSection2d",
            "params": {
                "patches": [
                    {
                        "type": "quad",
                        "material": 1,
                        "num_subdiv_y": 2,
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
    ]

    proc, _ = _run_json_to_tcl(case_data)
    assert proc.returncode != 0
    assert "unsupported FiberSection2d patch type: quad" in proc.stderr
