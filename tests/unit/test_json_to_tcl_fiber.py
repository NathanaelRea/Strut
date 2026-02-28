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
            {
                "id": 2,
                "type": "Steel01",
                "params": {"Fy": 500000000.0, "E0": 200000000000.0, "b": 0.01},
            },
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


def test_json_to_tcl_emits_fiber_section_quadr_patch():
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
                        "y_j": -0.2,
                        "z_j": 0.1,
                        "y_k": 0.2,
                        "z_k": 0.1,
                        "y_l": 0.2,
                        "z_l": -0.1,
                    }
                ],
                "layers": [],
            },
        }
    ]

    proc, text = _run_json_to_tcl(case_data)
    assert proc.returncode == 0, proc.stderr
    assert "patch quadr 1 2 2 -0.2 -0.1 -0.2 0.1 0.2 0.1 0.2 -0.1\n" in text


def test_json_to_tcl_rejects_unsupported_fiber_patch_type():
    case_data = _base_case()
    case_data["sections"] = [
        {
            "id": 10,
            "type": "FiberSection2d",
            "params": {
                "patches": [
                    {
                        "type": "triangle",
                        "material": 1,
                        "num_subdiv_y": 2,
                        "num_subdiv_z": 2,
                    }
                ],
                "layers": [],
            },
        }
    ]

    proc, _ = _run_json_to_tcl(case_data)
    assert proc.returncode != 0
    assert "unsupported FiberSection2d patch type: triangle" in proc.stderr


def test_json_to_tcl_emits_force_beam_column2d_with_lobatto():
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
    case_data["elements"] = [
        {
            "id": 4,
            "type": "forceBeamColumn2d",
            "nodes": [1, 2],
            "section": 7,
            "geomTransf": "Linear",
            "integration": "Lobatto",
            "num_int_pts": 3,
        }
    ]
    case_data["analysis"] = {"type": "static_nonlinear", "steps": 1}

    proc, text = _run_json_to_tcl(case_data)
    assert proc.returncode == 0, proc.stderr
    assert "beamIntegration Lobatto 1 7 3\n" in text
    assert "element forceBeamColumn 4 1 2 1 1\n" in text


def test_json_to_tcl_emits_disp_beam_column2d_with_lobatto():
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
    case_data["elements"] = [
        {
            "id": 4,
            "type": "dispBeamColumn2d",
            "nodes": [1, 2],
            "section": 7,
            "geomTransf": "Linear",
            "integration": "Lobatto",
            "num_int_pts": 3,
        }
    ]
    case_data["analysis"] = {"type": "static_nonlinear", "steps": 1}

    proc, text = _run_json_to_tcl(case_data)
    assert proc.returncode == 0, proc.stderr
    assert "beamIntegration Lobatto 1 7 3\n" in text
    assert "element dispBeamColumn 4 1 2 1 1\n" in text


def test_json_to_tcl_emits_force_beam_column2d_with_legendre_variable_points():
    case_data = _base_case()
    case_data["sections"] = [
        {
            "id": 7,
            "type": "ElasticSection2d",
            "params": {"E": 200000000000.0, "A": 0.02, "I": 0.00008},
        }
    ]
    case_data["elements"] = [
        {
            "id": 12,
            "type": "forceBeamColumn2d",
            "nodes": [1, 2],
            "section": 7,
            "geomTransf": "PDelta",
            "integration": "Legendre",
            "num_int_pts": 7,
        }
    ]
    case_data["analysis"] = {"type": "static_linear", "steps": 1}

    proc, text = _run_json_to_tcl(case_data)
    assert proc.returncode == 0, proc.stderr
    assert "geomTransf PDelta 1\n" in text
    assert "beamIntegration Legendre 1 7 7\n" in text
    assert "element forceBeamColumn 12 1 2 1 1\n" in text


def test_json_to_tcl_emits_force_beam_column3d_with_lobatto():
    case_data = {
        "schema_version": "1.0",
        "metadata": {"name": "fiber_converter_unit_3d", "units": "SI"},
        "model": {"ndm": 3, "ndf": 6},
        "nodes": [
            {"id": 1, "x": 0.0, "y": 0.0, "z": 0.0, "constraints": [1, 2, 3, 4, 5, 6]},
            {"id": 2, "x": 0.0, "y": 0.0, "z": 3.0},
        ],
        "materials": [],
        "sections": [
            {
                "id": 1,
                "type": "ElasticSection3d",
                "params": {
                    "E": 200000000000.0,
                    "A": 0.01,
                    "Iy": 1.0e-5,
                    "Iz": 1.0e-5,
                    "G": 80000000000.0,
                    "J": 2.0e-5,
                },
            }
        ],
        "elements": [
            {
                "id": 9,
                "type": "forceBeamColumn3d",
                "nodes": [1, 2],
                "section": 1,
                "geomTransf": "Linear",
                "integration": "Lobatto",
                "num_int_pts": 5,
            }
        ],
        "recorders": [],
        "analysis": {"type": "static_linear", "steps": 1},
    }

    proc, text = _run_json_to_tcl(case_data)
    assert proc.returncode == 0, proc.stderr
    assert "beamIntegration Lobatto 1 1 5\n" in text
    assert "element forceBeamColumn 9 1 2 1 1\n" in text


def test_json_to_tcl_emits_disp_beam_column3d_with_lobatto():
    case_data = {
        "schema_version": "1.0",
        "metadata": {"name": "fiber_converter_unit_3d_alias", "units": "SI"},
        "model": {"ndm": 3, "ndf": 6},
        "nodes": [
            {"id": 1, "x": 0.0, "y": 0.0, "z": 0.0, "constraints": [1, 2, 3, 4, 5, 6]},
            {"id": 2, "x": 0.0, "y": 0.0, "z": 3.0},
        ],
        "materials": [],
        "sections": [
            {
                "id": 2,
                "type": "ElasticSection3d",
                "params": {
                    "E": 200000000000.0,
                    "A": 0.01,
                    "Iy": 1.0e-5,
                    "Iz": 1.0e-5,
                    "G": 80000000000.0,
                    "J": 2.0e-5,
                },
            }
        ],
        "elements": [
            {
                "id": 10,
                "type": "dispBeamColumn3d",
                "nodes": [1, 2],
                "section": 2,
                "geomTransf": "Linear",
                "integration": "Lobatto",
                "num_int_pts": 3,
            }
        ],
        "recorders": [],
        "analysis": {"type": "static_linear", "steps": 1},
    }

    proc, text = _run_json_to_tcl(case_data)
    assert proc.returncode == 0, proc.stderr
    assert "beamIntegration Lobatto 1 2 3\n" in text
    assert "element dispBeamColumn 10 1 2 1 1\n" in text


def test_json_to_tcl_emits_force_beam_column3d_with_radau_pdelta():
    case_data = {
        "schema_version": "1.0",
        "metadata": {"name": "fiber_converter_unit_3d_radau", "units": "SI"},
        "model": {"ndm": 3, "ndf": 6},
        "nodes": [
            {"id": 1, "x": 0.0, "y": 0.0, "z": 0.0, "constraints": [1, 2, 3, 4, 5, 6]},
            {"id": 2, "x": 0.0, "y": 0.0, "z": 3.0},
        ],
        "materials": [],
        "sections": [
            {
                "id": 8,
                "type": "ElasticSection3d",
                "params": {
                    "E": 200000000000.0,
                    "A": 0.01,
                    "Iy": 1.0e-5,
                    "Iz": 1.0e-5,
                    "G": 80000000000.0,
                    "J": 2.0e-5,
                },
            }
        ],
        "elements": [
            {
                "id": 13,
                "type": "forceBeamColumn3d",
                "nodes": [1, 2],
                "section": 8,
                "geomTransf": "PDelta",
                "integration": "Radau",
                "num_int_pts": 4,
            }
        ],
        "recorders": [],
        "analysis": {"type": "static_linear", "steps": 1},
    }

    proc, text = _run_json_to_tcl(case_data)
    assert proc.returncode == 0, proc.stderr
    assert "geomTransf PDelta 1 1.0 0.0 0.0\n" in text
    assert "beamIntegration Radau 1 8 4\n" in text
    assert "element forceBeamColumn 13 1 2 1 1\n" in text


def test_json_to_tcl_emits_disp_beam_column3d_with_corotational():
    case_data = {
        "schema_version": "1.0",
        "metadata": {"name": "fiber_converter_unit_3d_corot", "units": "SI"},
        "model": {"ndm": 3, "ndf": 6},
        "nodes": [
            {"id": 1, "x": 0.0, "y": 0.0, "z": 0.0, "constraints": [1, 2, 3, 4, 5, 6]},
            {"id": 2, "x": 0.0, "y": 0.0, "z": 3.0},
        ],
        "materials": [],
        "sections": [
            {
                "id": 9,
                "type": "ElasticSection3d",
                "params": {
                    "E": 200000000000.0,
                    "A": 0.01,
                    "Iy": 1.0e-5,
                    "Iz": 1.0e-5,
                    "G": 80000000000.0,
                    "J": 2.0e-5,
                },
            }
        ],
        "elements": [
            {
                "id": 14,
                "type": "dispBeamColumn3d",
                "nodes": [1, 2],
                "section": 9,
                "geomTransf": "Corotational",
                "integration": "Legendre",
                "num_int_pts": 5,
            }
        ],
        "recorders": [],
        "analysis": {"type": "static_linear", "steps": 1},
    }

    proc, text = _run_json_to_tcl(case_data)
    assert proc.returncode == 0, proc.stderr
    assert "geomTransf Corotational 1 1.0 0.0 0.0\n" in text
    assert "beamIntegration Legendre 1 9 5\n" in text
    assert "element dispBeamColumn 14 1 2 1 1\n" in text


def test_json_to_tcl_emits_fiber_section3d_block():
    case_data = {
        "schema_version": "1.0",
        "metadata": {"name": "fiber_converter_section3d", "units": "SI"},
        "model": {"ndm": 3, "ndf": 6},
        "nodes": [
            {"id": 1, "x": 0.0, "y": 0.0, "z": 0.0},
            {"id": 2, "x": 0.0, "y": 0.0, "z": 1.0},
        ],
        "materials": [{"id": 1, "type": "Elastic", "params": {"E": 3.0e10}}],
        "sections": [
                {
                    "id": 14,
                    "type": "FiberSection3d",
                    "params": {
                        "G": 1.2e10,
                        "J": 2.0e-4,
                        "patches": [
                            {
                                "type": "rect",
                            "material": 1,
                            "num_subdiv_y": 2,
                            "num_subdiv_z": 2,
                            "y_i": -0.2,
                            "z_i": -0.1,
                            "y_j": 0.2,
                            "z_j": 0.1,
                        }
                    ],
                    "layers": [
                        {
                            "type": "straight",
                            "material": 1,
                            "num_bars": 2,
                            "bar_area": 0.0003,
                            "y_start": -0.15,
                            "z_start": 0.08,
                            "y_end": 0.15,
                            "z_end": 0.08,
                        }
                    ],
                },
            }
        ],
        "elements": [],
        "recorders": [],
        "analysis": {"type": "static_linear", "steps": 1},
    }

    proc, text = _run_json_to_tcl(case_data)
    assert proc.returncode == 0, proc.stderr
    fiber_block = (
        "section Fiber 14 -GJ 2400000.0 {\n"
        "  patch rect 1 2 2 -0.2 -0.1 0.2 0.1\n"
        "  layer straight 1 2 0.0003 -0.15 0.08 0.15 0.08\n"
        "}\n"
    )
    assert fiber_block in text


def test_json_to_tcl_emits_force_beam_column3d_with_fiber_section3d():
    case_data = {
        "schema_version": "1.0",
        "metadata": {"name": "fiber_converter_3d_section_emit", "units": "SI"},
        "model": {"ndm": 3, "ndf": 6},
        "nodes": [
            {"id": 1, "x": 0.0, "y": 0.0, "z": 0.0, "constraints": [1, 2, 3, 4, 5, 6]},
            {"id": 2, "x": 0.0, "y": 0.0, "z": 3.0},
        ],
        "materials": [{"id": 1, "type": "Elastic", "params": {"E": 3.0e10}}],
        "sections": [
            {
                "id": 1,
                "type": "FiberSection3d",
                "params": {
                    "G": 1.2e10,
                    "J": 2.0e-4,
                    "patches": [
                        {
                            "type": "rect",
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
        ],
        "elements": [
            {
                "id": 2,
                "type": "forceBeamColumn3d",
                "nodes": [1, 2],
                "section": 1,
                "geomTransf": "Linear",
                "integration": "Lobatto",
                "num_int_pts": 3,
            }
        ],
        "recorders": [],
        "analysis": {"type": "static_linear", "steps": 1},
    }

    proc, text = _run_json_to_tcl(case_data)
    assert proc.returncode == 0, proc.stderr
    assert "beamIntegration Lobatto 1 1 3\n" in text
    assert "element forceBeamColumn 2 1 2 1 1\n" in text


def test_json_to_tcl_rejects_force_beam_column3d_fiber_section3d_without_gj():
    case_data = {
        "schema_version": "1.0",
        "metadata": {"name": "fiber_converter_3d_section_missing_gj", "units": "SI"},
        "model": {"ndm": 3, "ndf": 6},
        "nodes": [
            {"id": 1, "x": 0.0, "y": 0.0, "z": 0.0, "constraints": [1, 2, 3, 4, 5, 6]},
            {"id": 2, "x": 0.0, "y": 0.0, "z": 3.0},
        ],
        "materials": [{"id": 1, "type": "Elastic", "params": {"E": 3.0e10}}],
        "sections": [
            {
                "id": 1,
                "type": "FiberSection3d",
                "params": {
                    "patches": [
                        {
                            "type": "rect",
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
        ],
        "elements": [
            {
                "id": 2,
                "type": "forceBeamColumn3d",
                "nodes": [1, 2],
                "section": 1,
                "geomTransf": "Linear",
                "integration": "Lobatto",
                "num_int_pts": 3,
            }
        ],
        "recorders": [],
        "analysis": {"type": "static_linear", "steps": 1},
    }

    proc, _ = _run_json_to_tcl(case_data)
    assert proc.returncode != 0
    assert "requires positive G and J" in proc.stderr


def test_json_to_tcl_rejects_disp_beam_column_alias():
    case_data = _base_case()
    case_data["sections"] = [
        {
            "id": 3,
            "type": "ElasticSection2d",
            "params": {"E": 200000000000.0, "A": 0.02, "I": 0.00008},
        }
    ]
    case_data["elements"] = [
        {
            "id": 5,
            "type": "dispBeamColumn",
            "nodes": [1, 2],
            "section": 3,
            "geomTransf": "Linear",
            "integration": "Lobatto",
            "num_int_pts": 5,
        }
    ]
    case_data["analysis"] = {"type": "static_linear", "steps": 1}

    proc, _ = _run_json_to_tcl(case_data)
    assert proc.returncode != 0
    assert "unsupported element type: dispBeamColumn" in proc.stderr


def test_json_to_tcl_emits_force_beam_column2d_with_elastic_section2d():
    case_data = _base_case()
    case_data["sections"] = [
        {
            "id": 3,
            "type": "ElasticSection2d",
            "params": {"E": 200000000000.0, "A": 0.02, "I": 0.00008},
        }
    ]
    case_data["elements"] = [
        {
            "id": 5,
            "type": "forceBeamColumn2d",
            "nodes": [1, 2],
            "section": 3,
            "geomTransf": "Linear",
            "integration": "Lobatto",
            "num_int_pts": 5,
        }
    ]
    case_data["analysis"] = {"type": "static_linear", "steps": 1}

    proc, text = _run_json_to_tcl(case_data)
    assert proc.returncode == 0, proc.stderr
    assert "beamIntegration Lobatto 1 3 5\n" in text
    assert "element forceBeamColumn 5 1 2 1 1\n" in text


def test_json_to_tcl_emits_zero_length_section():
    case_data = _base_case()
    case_data["nodes"][1]["x"] = 0.0
    case_data["sections"] = [
        {
            "id": 9,
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
                "layers": [],
            },
        }
    ]
    case_data["elements"] = [
        {
            "id": 6,
            "type": "zeroLengthSection",
            "nodes": [1, 2],
            "section": 9,
        }
    ]
    case_data["analysis"] = {"type": "static_nonlinear", "steps": 1}

    proc, text = _run_json_to_tcl(case_data)
    assert proc.returncode == 0, proc.stderr
    assert "element zeroLengthSection 6 1 2 9\n" in text
