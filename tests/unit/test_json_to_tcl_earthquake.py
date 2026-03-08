import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


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


def _base_uniform_case():
    return {
        "schema_version": "1.0",
        "metadata": {"name": "uniform_excitation_tcl_unit", "units": "SI"},
        "model": {"ndm": 2, "ndf": 2},
        "nodes": [
            {"id": 1, "x": 0.0, "y": 0.0, "constraints": [1, 2]},
            {"id": 2, "x": 1.0, "y": 0.0},
        ],
        "materials": [{"id": 1, "type": "Elastic", "params": {"E": 100.0}}],
        "sections": [],
        "elements": [{"id": 1, "type": "truss", "nodes": [1, 2], "area": 1.0, "material": 1}],
        "masses": [{"node": 2, "dof": 1, "value": 1.0}],
        "time_series": [{"type": "Path", "tag": 2, "dt": 0.1, "values": [1.0, 0.0]}],
        "pattern": {"type": "UniformExcitation", "tag": 3, "direction": 1, "accel": 2},
        "analysis": {
            "type": "transient_nonlinear",
            "steps": 2,
            "dt": 0.1,
            "integrator": {"type": "Newmark", "gamma": 0.5, "beta": 0.25},
            "tol": 1.0e-9,
            "max_iters": 15,
        },
        "recorders": [{"type": "node_displacement", "nodes": [2], "dofs": [1], "output": "node_disp"}],
    }


def test_json_to_tcl_emits_uniform_excitation_rayleigh_and_transient_nonlinear():
    case = _base_uniform_case()
    case["rayleigh"] = {"alphaM": 0.1, "betaK": 0.2, "betaKInit": 0.3, "betaKComm": 0.4}

    text = _run_json_to_tcl(case)

    assert "pattern UniformExcitation 3 1 -accel 2\n" in text
    assert "rayleigh 0.1 0.2 0.3 0.4\n" in text
    assert "algorithm Newton\n" in text
    assert "integrator Newmark 0.5 0.25\n" in text
    assert "analysis Transient\n" in text
    assert "analyze 2 0.1\n" in text


def test_json_to_tcl_emits_modified_newton_algorithms():
    case = _base_uniform_case()
    case["analysis"]["algorithm"] = "ModifiedNewton"

    text = _run_json_to_tcl(case)
    assert "algorithm ModifiedNewton\n" in text

    case["analysis"]["algorithm"] = "ModifiedNewtonInitial"
    text_initial = _run_json_to_tcl(case)
    assert "algorithm ModifiedNewton -initial\n" in text_initial

    case["pattern"] = {"type": "Plain", "tag": 1, "time_series": 2}
    case["loads"] = [{"node": 2, "dof": 1, "value": 1.0}]
    case["analysis"] = {
        "type": "static_nonlinear",
        "steps": 2,
        "max_iters": 10,
        "tol": 1.0e-8,
        "algorithm": "ModifiedNewton",
        "integrator": {"type": "LoadControl"},
    }
    text_static = _run_json_to_tcl(case)
    assert "algorithm ModifiedNewton\n" in text_static
    assert "analysis Static\n" in text_static


def test_json_to_tcl_emits_transient_nonlinear_fallback_loop():
    case = _base_uniform_case()
    case["analysis"]["solver_chain"] = [
        {
            "algorithm": "Newton",
            "test_type": "NormUnbalance",
            "tol": 1.0e-9,
            "max_iters": 15,
        },
        {
            "algorithm": "ModifiedNewtonInitial",
            "test_type": "NormDispIncr",
            "tol": 1.0e-8,
            "max_iters": 25,
        },
    ]

    text = _run_json_to_tcl(case)

    assert "set strut_tr_ok 0\n" in text
    assert "set strut_tr_ok [analyze 1 0.1]\n" in text
    assert "if {$strut_tr_ok != 0} {\n" in text
    assert "test NormDispIncr 1e-08 25\n" in text
    assert "algorithm ModifiedNewton -initial\n" in text
    assert "if {strut_tr_ok != 0} {\n" not in text
    assert "analyze 2 0.1\n" not in text


def test_json_to_tcl_emits_static_nonlinear_displacement_control_fallback_loop():
    case = _base_uniform_case()
    case["pattern"] = {"type": "Plain", "tag": 1, "time_series": 2}
    case["loads"] = [{"node": 2, "dof": 1, "value": 1.0}]
    case["analysis"] = {
        "type": "static_nonlinear",
        "steps": 2,
        "solver_chain": [
            {
                "algorithm": "Newton",
                "test_type": "NormDispIncr",
                "tol": 1.0e-8,
                "max_iters": 10,
            },
            {
                "algorithm": "ModifiedNewtonInitial",
                "test_type": "NormDispIncr",
                "tol": 1.0e-9,
                "max_iters": 25,
            },
        ],
        "integrator": {"type": "DisplacementControl", "node": 2, "dof": 1, "du": 0.1},
    }

    text = _run_json_to_tcl(case)

    assert "test NormDispIncr 1e-08 10\n" in text
    assert "algorithm Newton\n" in text
    assert "set strut_dc_targets {0.1 0.2}\n" in text
    assert "set strut_dc_cutback 0.5\n" in text
    assert "set strut_dc_max_cutbacks 8\n" in text
    assert "integrator DisplacementControl 2 1 $strut_dc_try_du\n" in text
    assert "set strut_dc_ok [analyze 1]\n" in text
    assert "if {$strut_dc_ok != 0} {\n" in text
    assert "test NormDispIncr 1e-09 25\n" in text
    assert "algorithm ModifiedNewton -initial\n" in text
    assert "if {strut_dc_ok != 0} {\n" not in text
    assert "\nset strut_dc_ok [analyze 2]\n" not in text


def test_json_to_tcl_emits_ordered_solver_chain_retry_algorithms():
    case = _base_uniform_case()
    case["analysis"]["solver_chain"] = [
        {
            "algorithm": "Newton",
            "test_type": "EnergyIncr",
            "tol": 1.0e-9,
            "max_iters": 15,
        },
        {
            "algorithm": "Broyden",
            "algorithm_options": {"max_iters": 6},
            "test_type": "EnergyIncr",
            "tol": 1.0e-9,
            "max_iters": 15,
        },
        {
            "algorithm": "NewtonLineSearch",
            "algorithm_options": {"alpha": 0.7},
            "test_type": "EnergyIncr",
            "tol": 1.0e-9,
            "max_iters": 15,
        },
    ]

    text = _run_json_to_tcl(case)

    assert "test EnergyIncr 1e-09 15\n" in text
    assert "algorithm Broyden 6\n" in text
    assert "algorithm NewtonLineSearch 0.7\n" in text
    assert text.index("algorithm Broyden 6\n") < text.index(
        "algorithm NewtonLineSearch 0.7\n"
    )


def test_json_to_tcl_preserves_numberer_system_and_print_commands():
    case = _base_uniform_case()
    case["pattern"] = {"type": "Plain", "tag": 1, "time_series": 2}
    case["loads"] = [{"node": 2, "dof": 1, "value": 1.0}]
    case["analysis"] = {
        "type": "static_linear",
        "steps": 1,
        "numberer": "Plain",
        "system": "BandSPD",
        "print_commands": [{"args": ["node", "2"]}, {"args": ["ele"]}],
    }

    text = _run_json_to_tcl(case)

    assert "numberer Plain\n" in text
    assert "system BandSPD\n" in text
    assert "print node 2\n" in text
    assert "print ele\n" in text


def test_json_to_tcl_emits_test_print_flag_and_extra_args():
    case = _base_uniform_case()
    case["pattern"] = {"type": "Plain", "tag": 1, "time_series": 2}
    case["loads"] = [{"node": 2, "dof": 1, "value": 1.0}]
    case["analysis"] = {
        "type": "static_nonlinear",
        "steps": 1,
        "algorithm": "Newton",
        "test_type": "NormDispIncr",
        "tol": 1.0e-12,
        "max_iters": 10,
        "test_print_flag": 3,
        "test_extra_args": ["7"],
        "integrator": {"type": "LoadControl", "step": 1.0},
    }

    text = _run_json_to_tcl(case)

    assert "test NormDispIncr 1e-12 10 3 7\n" in text


def test_json_to_tcl_rejects_rel_tol_in_nonlinear_analysis():
    case = _base_uniform_case()
    case["analysis"]["rel_tol"] = 1.0e-6

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        case_path = tmp_dir / "case.json"
        tcl_path = tmp_dir / "model.tcl"
        case_path.write_text(json.dumps(case), encoding="utf-8")
        proc = subprocess.run(
            [
                sys.executable,
                str(repo_root / "scripts" / "json_to_tcl.py"),
                str(case_path),
                str(tcl_path),
            ],
            capture_output=True,
            text=True,
        )

    assert proc.returncode != 0
    assert "rel_tol is not representable in Tcl/OpenSees" in proc.stderr


def test_json_to_tcl_emits_stage_initialize_before_analysis_setup():
    case = _base_uniform_case()
    case["pattern"] = {"type": "Plain", "tag": 1, "time_series": 2}
    case["loads"] = [{"node": 2, "dof": 1, "value": 1.0}]
    case["analysis"] = {
        "type": "staged",
        "constraints": "Transformation",
        "stages": [
            {
                "initialize": True,
                "analysis": {
                    "type": "static_nonlinear",
                    "steps": 1,
                    "numberer": "RCM",
                    "system": "ProfileSPD",
                    "algorithm": "Newton",
                    "test_type": "NormDispIncr",
                    "tol": 1.0e-12,
                    "max_iters": 10,
                    "integrator": {"type": "LoadControl", "step": 0.1},
                },
            }
        ],
    }

    text = _run_json_to_tcl(case)

    assert "initialize\nconstraints Transformation\nnumberer RCM\nsystem ProfileSPD\n" in text


def test_json_to_tcl_emits_staged_analysis_with_load_const_and_pattern_override():
    case = _base_uniform_case()
    case["time_series"] = [
        {"type": "Linear", "tag": 1, "factor": 1.0},
        {"type": "Path", "tag": 2, "dt": 0.1, "values": [0.0, 1.0, 0.0]},
    ]
    case["pattern"] = {"type": "Plain", "tag": 1, "time_series": 1}
    case["loads"] = [{"node": 2, "dof": 1, "value": 1.0}]
    case["analysis"] = {
        "type": "staged",
        "constraints": "Plain",
        "stages": [
            {
                "analysis": {"type": "static_nonlinear", "steps": 2, "algorithm": "Newton"},
                "load_const": {"time": 0.0},
            },
            {
                "pattern": {
                    "type": "UniformExcitation",
                    "tag": 3,
                    "direction": 1,
                    "accel": 2,
                },
                "rayleigh": {"betaKComm": 0.02},
                "analysis": {
                    "type": "transient_nonlinear",
                    "steps": 2,
                    "dt": 0.1,
                    "algorithm": "Newton",
                    "integrator": {"type": "Newmark", "gamma": 0.5, "beta": 0.25},
                },
            },
        ],
    }

    text = _run_json_to_tcl(case)
    assert "loadConst -time 0.0\n" in text
    assert text.count("pattern Plain 1 1 {\n") == 1
    assert "pattern UniformExcitation 3 1 -accel 2\n" in text
    assert "rayleigh 0.0 0.0 0.0 0.02\n" in text
    assert text.count("analysis Static\n") == 1
    assert text.count("analysis Transient\n") == 1


def test_json_to_tcl_emits_staged_plain_element_beam_point_load():
    case = {
        "schema_version": "1.0",
        "metadata": {"name": "staged_beam_point_tcl_unit", "units": "SI"},
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
        "time_series": [{"type": "Linear", "tag": 1, "factor": 1.0}],
        "analysis": {
            "type": "staged",
            "constraints": "Plain",
            "stages": [
                {
                    "pattern": {"type": "Plain", "tag": 1, "time_series": 1},
                    "element_loads": [
                        {"element": 1, "type": "beamPoint", "py": -3.0, "x": 0.25}
                    ],
                    "analysis": {"type": "static_linear", "steps": 1},
                }
            ],
        },
    }

    text = _run_json_to_tcl(case)
    assert "eleLoad -ele 1 -type -beamPoint -3.0 0.25\n" in text


def test_json_to_tcl_treats_empty_constraints_list_as_free_node():
    case = {
        "schema_version": "1.0",
        "metadata": {"name": "empty_constraints_tcl_unit", "units": "SI"},
        "model": {"ndm": 2, "ndf": 3},
        "nodes": [
            {"id": 1, "x": 0.0, "y": 0.0, "constraints": [1, 2, 3]},
            {"id": 2, "x": 1.0, "y": 0.0, "constraints": []},
        ],
        "materials": [{"id": 1, "type": "Elastic", "params": {"E": 100.0}}],
        "sections": [
            {
                "id": 1,
                "type": "ElasticSection2d",
                "params": {"E": 100.0, "A": 1.0, "I": 1.0},
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
        "analysis": {"type": "static_linear", "steps": 1},
        "recorders": [{"type": "node_displacement", "nodes": [2], "dofs": [1], "output": "disp"}],
    }

    text = _run_json_to_tcl(case)

    assert "fix 2 0 0 0\n" in text


def test_json_to_tcl_staged_transient_fallback_does_not_emit_extra_analyze():
    case = _base_uniform_case()
    case["time_series"] = [
        {"type": "Linear", "tag": 1, "factor": 1.0},
        {"type": "Path", "tag": 2, "dt": 0.1, "values": [0.0, 1.0, 0.0]},
    ]
    case["pattern"] = {"type": "Plain", "tag": 1, "time_series": 1}
    case["loads"] = [{"node": 2, "dof": 1, "value": 1.0}]
    case["analysis"] = {
        "type": "staged",
        "constraints": "Plain",
        "stages": [
            {
                "analysis": {"type": "static_nonlinear", "steps": 2, "algorithm": "Newton"},
                "load_const": {"time": 0.0},
            },
            {
                "pattern": {
                    "type": "UniformExcitation",
                    "tag": 3,
                    "direction": 1,
                    "accel": 2,
                },
                "analysis": {
                    "type": "transient_nonlinear",
                    "steps": 2,
                    "dt": 0.1,
                    "solver_chain": [
                        {
                            "algorithm": "Newton",
                            "test_type": "NormUnbalance",
                            "tol": 1.0e-10,
                            "max_iters": 20,
                        },
                        {
                            "algorithm": "ModifiedNewtonInitial",
                            "test_type": "NormUnbalance",
                            "tol": 1.0e-10,
                            "max_iters": 20,
                        },
                    ],
                    "integrator": {"type": "Newmark", "gamma": 0.5, "beta": 0.25},
                },
            },
        ],
    }

    text = _run_json_to_tcl(case)
    assert "for {set strut_tr_step 0} {$strut_tr_step < 2 && $strut_tr_ok == 0} {incr strut_tr_step} {\n" in text
    assert "set strut_tr_ok [analyze 1 0.1]\n" in text
    assert "if {$strut_tr_ok != 0} {\n" in text
    assert "if {strut_tr_ok != 0} {\n" not in text
    assert "\nanalyze 2\n" not in text
    assert "\nanalyze 2 0.1\n" not in text


def test_json_to_tcl_staged_static_fallback_does_not_emit_extra_analyze():
    case = _base_uniform_case()
    case["pattern"] = {"type": "Plain", "tag": 1, "time_series": 2}
    case["loads"] = [{"node": 2, "dof": 1, "value": 1.0}]
    case["analysis"] = {
        "type": "staged",
        "constraints": "Plain",
        "stages": [
            {
                "analysis": {
                    "type": "static_nonlinear",
                    "steps": 2,
                    "solver_chain": [
                        {
                            "algorithm": "Newton",
                            "test_type": "NormDispIncr",
                            "tol": 1.0e-8,
                            "max_iters": 8,
                        },
                        {
                            "algorithm": "ModifiedNewtonInitial",
                            "test_type": "NormDispIncr",
                            "tol": 1.0e-8,
                            "max_iters": 20,
                        },
                    ],
                    "integrator": {"type": "LoadControl"},
                }
            }
        ],
    }

    text = _run_json_to_tcl(case)

    assert "test NormDispIncr 1e-08 8\n" in text
    assert "set strut_nl_ok 0\n" in text
    assert "set strut_nl_ok [analyze 1]\n" in text
    assert "if {$strut_nl_ok != 0} {\n" in text
    assert "algorithm ModifiedNewton -initial\n" in text
    assert "if {strut_nl_ok != 0} {\n" not in text
    assert "\nset strut_nl_ok [analyze 2]\n" not in text


def test_json_to_tcl_rejects_uniform_excitation_with_nodal_loads():
    case = _base_uniform_case()
    case["loads"] = [{"node": 2, "dof": 1, "value": 1.0}]

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        case_path = tmp_dir / "case.json"
        tcl_path = tmp_dir / "model.tcl"
        case_path.write_text(json.dumps(case), encoding="utf-8")
        with pytest.raises(subprocess.CalledProcessError):
            subprocess.check_call(
                [
                    sys.executable,
                    str(repo_root / "scripts" / "json_to_tcl.py"),
                    str(case_path),
                    str(tcl_path),
                ]
            )


def test_json_to_tcl_loads_path_values_from_file():
    case = _base_uniform_case()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        (tmp_dir / "A10000.tcl").write_text("1.0E+00 0.0D+00", encoding="utf-8")
        case["time_series"] = [
            {"type": "Path", "tag": 2, "dt": 0.1, "values_path": "A10000.tcl"}
        ]
        case_path = tmp_dir / "case.json"
        tcl_path = tmp_dir / "model.tcl"
        case_path.write_text(json.dumps(case), encoding="utf-8")
        subprocess.check_call(
            [
                sys.executable,
                str(repo_root / "scripts" / "json_to_tcl.py"),
                str(case_path),
                str(tcl_path),
            ]
        )
        text = tcl_path.read_text(encoding="utf-8")
    assert "timeSeries Path 2 -dt 0.1 -values {1.0 0.0} -factor 1.0\n" in text


def test_json_to_tcl_loads_path_values_relative_to_source_example():
    case = _base_uniform_case()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        example_dir = tmp_dir / "examples"
        example_dir.mkdir(parents=True, exist_ok=True)
        example_tcl = example_dir / "example.tcl"
        example_tcl.write_text("# placeholder\n", encoding="utf-8")
        (example_dir / "A10000.tcl").write_text("1.0E+00 0.0D+00", encoding="utf-8")
        case["source_example"] = str(example_tcl)
        case["time_series"] = [
            {"type": "Path", "tag": 2, "dt": 0.1, "values_path": "A10000.tcl"}
        ]
        case_path = tmp_dir / "case.json"
        tcl_path = tmp_dir / "model.tcl"
        case_path.write_text(json.dumps(case), encoding="utf-8")
        subprocess.check_call(
            [
                sys.executable,
                str(repo_root / "scripts" / "json_to_tcl.py"),
                str(case_path),
                str(tcl_path),
            ]
        )
        text = tcl_path.read_text(encoding="utf-8")
    assert "timeSeries Path 2 -dt 0.1 -values {1.0 0.0} -factor 1.0\n" in text
