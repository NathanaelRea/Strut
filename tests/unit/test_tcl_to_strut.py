import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
TCL_TO_STRUT_PATH = REPO_ROOT / "scripts" / "tcl_to_strut.py"
RUN_CASE_PATH = REPO_ROOT / "scripts" / "run_case.py"


def _load_tcl_to_strut_module():
    module_name = "strut_tcl_to_strut_test_module"
    spec = importlib.util.spec_from_file_location(module_name, TCL_TO_STRUT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


tcl_to_strut = _load_tcl_to_strut_module()
run_case_spec = importlib.util.spec_from_file_location(
    "strut_run_case_test_module", RUN_CASE_PATH
)
assert run_case_spec is not None
assert run_case_spec.loader is not None
run_case = importlib.util.module_from_spec(run_case_spec)
sys.modules["strut_run_case_test_module"] = run_case
run_case_spec.loader.exec_module(run_case)


def _convert_direct_tcl_manifest(case_name: str) -> dict:
    manifest = REPO_ROOT / "tests" / "validation" / case_name / "direct_tcl_case.json"
    entry = run_case._resolve_entry_tcl_from_manifest(manifest, REPO_ROOT)
    source_files = run_case._resolve_direct_tcl_source_files(entry, manifest)
    runtime_entry, _ = run_case._prepare_direct_tcl_entry(entry, manifest.parent, source_files)
    return tcl_to_strut.convert_tcl_to_case(runtime_entry, REPO_ROOT)


def test_prepare_direct_tcl_entry_preserves_legacy_mass_arity_in_source(tmp_path: Path):
    bundle_root = tmp_path / "OpenSeesExamplesAdvanced"
    script_dir = bundle_root / "example"
    script_dir.mkdir(parents=True, exist_ok=True)
    entry = script_dir / "Example.tcl"
    entry.write_text(
        "\n".join(
            [
                "model BasicBuilder -ndm 2 -ndf 3",
                "node 21 0.0 0.0",
                "set nodeID 21",
                "set MassNode 1.0",
                "mass $nodeID $MassNode 0.0 0.0 0.0 0.0 0.0;",
                "",
            ]
        ),
        encoding="utf-8",
    )

    runtime_entry, _ = run_case._prepare_direct_tcl_entry(
        entry, tmp_path / "case_root", [entry]
    )

    mirrored_entry = runtime_entry.parent / "Example.tcl"
    mirrored_text = mirrored_entry.read_text(encoding="utf-8")
    assert "mass $nodeID $MassNode 0.0 0.0 0.0 0.0 0.0;" in mirrored_text

    opensees_wrapper = run_case._prepare_opensees_compat_entry(runtime_entry)
    wrapper_text = opensees_wrapper.read_text(encoding="utf-8")
    assert "rename mass __strut_builtin_mass" in wrapper_text
    assert f"source {{{runtime_entry.name}}}" in wrapper_text


def test_prepare_direct_tcl_entry_injects_missing_w_section_tags(tmp_path: Path):
    bundle_root = tmp_path / "OpenSeesExamplesAdvanced"
    script_dir = (
        bundle_root
        / "opensees_example_6_generic_2d_frame_n_story_n_bay_reinforced_concrete_section_steel_w_section"
    )
    script_dir.mkdir(parents=True, exist_ok=True)
    entry = script_dir / "Ex6.genericFrame2D.build.InelasticFiberWSection.tcl"
    entry.write_text(
        "\n".join(
            [
                "# define MATERIAL properties ----------------------------------------",
                "set matIDhard 1",
                "Wsection  $ColSecTag $matIDhard 1 1 1 1 1 1 1 1",
                "",
            ]
        ),
        encoding="utf-8",
    )

    runtime_entry, _ = run_case._prepare_direct_tcl_entry(
        entry, tmp_path / "case_root", [entry]
    )

    mirrored_text = (runtime_entry.parent / entry.name).read_text(encoding="utf-8")
    assert "set ColSecTag 1" in mirrored_text
    assert "set BeamSecTag 4" in mirrored_text


def test_convert_ex1a_builds_staged_case():
    entry = (
        REPO_ROOT
        / "docs/agent-reference/OpenSeesExamplesBasic"
        / "time_history_analysis_of_a_2d_elastic_cantilever_column"
        / "Ex1a.Canti2D.EQ.modif.tcl"
    )

    case = tcl_to_strut.convert_tcl_to_case(entry, REPO_ROOT)

    assert case["analysis"]["type"] == "staged"
    assert len(case["analysis"]["stages"]) == 2

    gravity = case["analysis"]["stages"][0]
    dynamic = case["analysis"]["stages"][1]

    assert gravity["analysis"]["type"] == "static_nonlinear"
    assert gravity["analysis"]["steps"] == 10
    assert gravity["load_const"] == {"time": 0.0}
    assert gravity["pattern"] == {"type": "Plain", "tag": 1, "time_series": 1}
    assert gravity["loads"] == [{"node": 2, "dof": 2, "value": -2000.0}]
    assert case["pattern"] == {"type": "Plain", "tag": 1, "time_series": 1}
    assert case["loads"] == [{"node": 2, "dof": 2, "value": -2000.0}]

    assert dynamic["analysis"]["type"] == "transient_linear"
    assert dynamic["analysis"]["steps"] == 3995
    assert dynamic["analysis"]["dt"] == 0.01
    assert dynamic["pattern"] == {
        "type": "UniformExcitation",
        "tag": 2,
        "direction": 1,
        "accel": 2,
    }
    assert dynamic["rayleigh"]["betaKComm"] == pytest.approx(0.007997, rel=5e-4)

    assert case["time_series"] == [
        {"type": "Linear", "tag": 1, "factor": 1.0},
        {
            "type": "Path",
            "tag": 2,
            "dt": 0.005,
            "values_path": str((entry.parent / "A10000.tcl").resolve()),
            "factor": 386.0,
        },
    ]

    recorders = {rec["type"]: rec for rec in case["recorders"]}
    assert recorders["node_displacement"]["raw_path"] == "Data/DFree.out"
    assert recorders["node_displacement"]["include_time"] is True
    assert recorders["drift"]["raw_path"] == "Data/Drift.out"
    assert recorders["element_force"]["raw_path"] == "Data/FCol.out"


def test_convert_example31_preserves_initialize_profilespd_and_rcm():
    entry = (
        REPO_ROOT
        / "docs/agent-reference/OpenSees/EXAMPLES/ExampleScripts"
        / "Example3.1.tcl"
    )

    case = tcl_to_strut.convert_tcl_to_case(entry, REPO_ROOT)

    assert case["analysis"]["type"] == "staged"
    assert len(case["analysis"]["stages"]) == 1
    stage = case["analysis"]["stages"][0]

    assert stage["initialize"] is True
    assert stage["analysis"]["type"] == "static_nonlinear"
    assert stage["analysis"]["steps"] == 10
    assert stage["analysis"]["constraints"] == "Transformation"
    assert stage["analysis"]["numberer"] == "RCM"
    assert stage["analysis"]["system"] == "ProfileSPD"
    assert stage["analysis"]["algorithm"] == "Newton"
    assert stage["analysis"]["test_type"] == "NormDispIncr"
    assert stage["analysis"]["integrator"]["type"] == "LoadControl"
    assert stage["analysis"]["integrator"]["step"] == pytest.approx(0.1)
    assert [element["type"] for element in case["elements"]] == [
        "forceBeamColumn2d",
        "forceBeamColumn2d",
        "elasticBeamColumn2d",
    ]
    assert case["elements"][0]["geomTransf"] == "Corotational"
    assert case["elements"][1]["geomTransf"] == "Corotational"
    assert case["elements"][2]["geomTransf"] == "Linear"


def test_convert_advanced_ex1a_eq_accepts_plural_element_deformations():
    entry = (
        REPO_ROOT
        / "docs/agent-reference/OpenSeesExamplesAdvanced"
        / "opensees_example_1a_2d_elastic_cantilever_column"
        / "Ex1a.Canti2D.EQ.tcl"
    )

    case = tcl_to_strut.convert_tcl_to_case(entry, REPO_ROOT)

    recorders = {rec["output"]: rec for rec in case["recorders"]}
    assert recorders["DCol"]["type"] == "element_deformation"
    assert recorders["DCol"]["raw_path"] == "Data/DCol.out"
    assert recorders["DCol"]["include_time"] is True
    assert recorders["DCol"]["parity"] is False


def test_convert_advanced_ex1b_eq_expands_grouped_drift_nodes():
    entry = (
        REPO_ROOT
        / "docs/agent-reference/OpenSeesExamplesAdvanced"
        / "opensees_example_1b_elastic_portal_frame"
        / "Ex1b.Portal2D.EQ.tcl"
    )

    case = tcl_to_strut.convert_tcl_to_case(entry, REPO_ROOT)

    drifts = [rec for rec in case["recorders"] if rec["type"] == "drift"]
    assert len(drifts) == 2
    assert {(rec["i_node"], rec["j_node"]) for rec in drifts} == {(1, 3), (2, 4)}
    assert all(rec["raw_path"] == "Data/Drift.out" for rec in drifts)
    assert all(rec["include_time"] is True for rec in drifts)


def test_convert_uniform_excitation_accepts_inline_sine_and_vel0(tmp_path: Path):
    script = tmp_path / "uniform_sine.tcl"
    script.write_text(
        "\n".join(
            (
                "model BasicBuilder -ndm 2 -ndf 2",
                "node 1 0 0",
                "node 2 1 0",
                "fix 1 1 1",
                "fix 2 0 1",
                'pattern UniformExcitation 400 1 -accel "Sine 0.0 3.0 0.35 -factor 193.2" -vel0 -10.76',
                "integrator Newmark 0.5 0.25",
                "analysis Transient",
                "analyze 3 0.01",
                "",
            )
        ),
        encoding="utf-8",
    )

    case = tcl_to_strut.convert_tcl_to_case(script, REPO_ROOT)

    assert case["time_series"] == [
        {
            "type": "Trig",
            "tag": 1,
            "t_start": 0.0,
            "t_finish": 3.0,
            "period": 0.35,
            "factor": pytest.approx(193.2),
        }
    ]
    stage = case["analysis"]["stages"][0]
    assert stage["pattern"] == {
        "type": "UniformExcitation",
        "tag": 400,
        "direction": 1,
        "accel": 1,
    }


def test_convert_multiple_support_pattern_downgrades_to_none_with_parity_disabled(
    tmp_path: Path,
):
    gm_path = tmp_path / "gm.g3"
    gm_path.write_text("0.0\n0.1\n", encoding="utf-8")
    script = tmp_path / "multiple_support.tcl"
    script.write_text(
        "\n".join(
            (
                "model BasicBuilder -ndm 2 -ndf 2",
                "node 1 0 0",
                "node 2 1 0",
                "fix 1 1 1",
                "fix 2 0 1",
                "pattern MultipleSupport 400 {",
                f'  groundMotion 500 Plain -disp "Series -dt 0.01 -filePath {gm_path} -factor 1.0"',
                "  imposedMotion 1 1 500",
                "}",
                "recorder Node -file Data/DFree.out -time -node 2 -dof 1 disp",
                "integrator Newmark 0.5 0.25",
                "analysis Transient",
                "analyze 2 0.01",
                "",
            )
        ),
        encoding="utf-8",
    )

    case = tcl_to_strut.convert_tcl_to_case(script, REPO_ROOT)

    stage = case["analysis"]["stages"][0]
    assert stage["pattern"] == {"type": "None"}
    assert all(recorder.get("parity") is False for recorder in case["recorders"])


def test_convert_hardening_uniaxial_material_maps_to_steel01(tmp_path: Path):
    script = tmp_path / "hardening_material.tcl"
    script.write_text(
        "\n".join(
            (
                "model BasicBuilder -ndm 2 -ndf 2",
                "node 1 0 0",
                "node 2 1 0",
                "fix 1 1 1",
                "uniaxialMaterial Hardening 1 29000.0 60.0 0.0 1000.0",
                "element truss 1 1 2 1.0 1",
                "timeSeries Linear 1",
                "pattern Plain 1 1 {",
                "  load 2 1.0 0.0",
                "}",
                "integrator LoadControl 1.0",
                "analysis Static",
                "analyze 1",
                "",
            )
        ),
        encoding="utf-8",
    )

    case = tcl_to_strut.convert_tcl_to_case(script, REPO_ROOT)

    assert case["materials"][0]["type"] == "Steel01"
    assert case["materials"][0]["params"]["Fy"] == pytest.approx(60.0)
    assert case["materials"][0]["params"]["E0"] == pytest.approx(29000.0)
    assert case["materials"][0]["params"]["b"] == pytest.approx(1000.0 / 29000.0)


def test_convert_load_ignores_oversized_numeric_tail_beyond_model_ndf(tmp_path: Path):
    script = tmp_path / "oversized_load.tcl"
    script.write_text(
        "\n".join(
            (
                "model BasicBuilder -ndm 2 -ndf 3",
                "node 1 0 0",
                "node 2 0 1",
                "fix 1 1 1 1",
                "timeSeries Linear 1",
                "pattern Plain 1 1 {",
                "  load 2 5.0 0.0 0.0 0.0 0.0 0.0",
                "}",
                "constraints Plain",
                "numberer Plain",
                "system BandGeneral",
                "integrator LoadControl 1.0",
                "analysis Static",
                "analyze 1",
                "",
            )
        ),
        encoding="utf-8",
    )

    case = tcl_to_strut.convert_tcl_to_case(script, REPO_ROOT)

    assert case.get("loads", []) == [{"node": 2, "dof": 1, "value": 5.0}]


def test_convert_displacement_control_with_empty_plain_pattern_is_noop(tmp_path: Path):
    script = tmp_path / "disp_control_empty_pattern.tcl"
    script.write_text(
        "\n".join(
            (
                "model BasicBuilder -ndm 2 -ndf 3",
                "node 1 0 0",
                "node 2 0 1",
                "fix 1 1 1 1",
                "timeSeries Linear 1",
                "pattern Plain 10 1 {",
                "  load 2 0.0 -1.0 0.0",
                "}",
                "constraints Plain",
                "numberer Plain",
                "system BandGeneral",
                "algorithm Newton",
                "integrator LoadControl 1.0",
                "analysis Static",
                "analyze 1",
                "loadConst -time 0.0",
                "pattern Plain 1 1 {",
                "  load 2 5.0 0.0 0.0 1.0 0.0 0.0",
                "}",
                "constraints Plain",
                "numberer Plain",
                "system BandGeneral",
                "algorithm Newton",
                "integrator DisplacementControl 2 1 0.1",
                "analysis Static",
                "analyze 5",
                "",
            )
        ),
        encoding="utf-8",
    )

    case = tcl_to_strut.convert_tcl_to_case(script, REPO_ROOT)

    assert len(case["analysis"]["stages"]) == 1
    assert case["analysis"]["stages"][0]["loads"] == [
        {"node": 2, "dof": 2, "value": -1.0}
    ]


def test_convert_displacement_control_zero_increment_is_noop(tmp_path: Path):
    script = tmp_path / "disp_control_zero_step.tcl"
    script.write_text(
        "\n".join(
            (
                "model BasicBuilder -ndm 2 -ndf 3",
                "node 1 0 0",
                "node 2 0 1",
                "fix 1 1 1 1",
                "geomTransf Linear 1",
                "element elasticBeamColumn 1 1 2 1.0 1000.0 1.0 1",
                "timeSeries Linear 1",
                "pattern Plain 1 1 {",
                "  load 2 1.0 0.0 0.0",
                "}",
                "constraints Plain",
                "numberer Plain",
                "system BandGeneral",
                "algorithm Newton",
                "integrator DisplacementControl 2 1 0.0",
                "analysis Static",
                "analyze 2",
                "integrator DisplacementControl 2 1 0.1",
                "analyze 1",
                "",
            )
        ),
        encoding="utf-8",
    )

    case = tcl_to_strut.convert_tcl_to_case(script, REPO_ROOT)

    assert len(case["analysis"]["stages"]) == 1
    stage = case["analysis"]["stages"][0]["analysis"]
    assert stage["steps"] == 1
    assert stage["integrator"]["du"] == pytest.approx(0.1)


def test_convert_ex2a_eq_resolves_ground_motion_from_example_root_when_available():
    entry = (
        REPO_ROOT
        / "docs/agent-reference/OpenSeesExamplesAdvanced"
        / "opensees_example_2a_elastic_cantilever_column_with_variables"
        / "Ex2a.Canti2D.ElasticElement.EQ.tcl"
    )

    case = tcl_to_strut.convert_tcl_to_case(entry, REPO_ROOT)

    assert case["time_series"][1]["values_path"] == str(
        (
            REPO_ROOT
            / "docs/agent-reference/OpenSeesExamplesAdvanced"
            / "opensees_example_2a_elastic_cantilever_column_with_variables"
            / "BM68elc.acc"
        ).resolve()
    )


def test_convert_reports_unknown_command_with_location(tmp_path: Path):
    script = tmp_path / "case.tcl"
    script.write_text("wipe\nbogusCommand 1 2\n", encoding="utf-8")

    with pytest.raises(tcl_to_strut.TclToStrutError) as exc_info:
        tcl_to_strut.convert_tcl_to_case(script, REPO_ROOT)

    message = str(exc_info.value)
    assert "bogusCommand" in message
    assert str(script) in message
    assert "line 2" in message


def test_convert_typed_case_returns_dataclass_model():
    entry = (
        REPO_ROOT
        / "docs/agent-reference/OpenSeesExamplesBasic"
        / "time_history_analysis_of_a_2d_elastic_cantilever_column"
        / "Ex1a.Canti2D.EQ.modif.tcl"
    )

    case = tcl_to_strut.convert_tcl_to_typed_case(entry, REPO_ROOT)

    assert case.model.ndm == 2
    assert case.model.ndf == 3
    assert case.analysis.constraints == "Plain"
    assert case.time_series[1].data["values_path"] == str(
        (entry.parent / "A10000.tcl").resolve()
    )


def test_convert_ex5_disables_transient_force_beam_local_force_parity():
    entry = (
        REPO_ROOT
        / "docs/agent-reference/OpenSeesExamplesAdvanced"
        / "Ex5Fram2DEQUniform"
        / "Ex5.Frame2D.complete.Dynamic.EQ.Uniform.tcl"
    )

    case = tcl_to_strut.convert_tcl_to_case(entry, REPO_ROOT)

    recorders = {}
    for recorder in case["recorders"]:
        key = (recorder["type"], recorder.get("output"))
        recorders[key] = recorder

    assert recorders[("node_reaction", "RBase")]["parity"] is False
    assert recorders[("element_local_force", "Fel1")]["parity"] is False
    assert recorders[("section_force", "ForceEle1sec1")]["parity"] is False
    assert recorders[("section_force", "ForceEle1sec5")]["parity"] is False
    assert (
        recorders[("section_deformation", "DefoEle1sec1")].get("parity", True) is True
    )
    assert (
        recorders[("section_deformation", "DefoEle1sec5")].get("parity", True) is True
    )
    assert case["parity_tolerance"] == {
        "rtol": pytest.approx(0.5),
        "atol": pytest.approx(5.0e-3),
    }
    assert case["parity_tolerance_by_category"]["force"] == {
        "rtol": pytest.approx(0.75),
        "atol": pytest.approx(1.0e-2),
    }
    assert case["parity_tolerance_by_recorder"]["element_local_force"] == {
        "rtol": pytest.approx(1.0),
        "atol": pytest.approx(3.0e-2),
    }
    assert case["parity_tolerance_by_recorder"]["section_force"] == {
        "rtol": pytest.approx(0.75),
        "atol": pytest.approx(3.0e-2),
    }


def test_convert_ex9_2d_aggregator_moment_curvature_wrapper(tmp_path: Path):
    example_dir = (
        REPO_ROOT
        / "docs/agent-reference/OpenSeesExamplesAdvanced"
        / "opensees_example_9_build_analyze_a_section_example"
    )
    entry = tmp_path / "Ex9.complete.Ex9AnalyzeMomentCurvature2D.tcl"
    entry.write_text(
        "\n".join(
            [
                f"source {{{example_dir / 'Ex9a.build.UniaxialSection2D.tcl'}}}",
                f"source {{{example_dir / 'Ex9.analyze.MomentCurvature2D.tcl'}}}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    case = tcl_to_strut.convert_tcl_to_case(entry, REPO_ROOT)

    assert case["model"] == {"ndm": 2, "ndf": 3}
    assert any(section["type"] == "AggregatorSection2d" for section in case["sections"])
    assert any(element["type"] == "zeroLengthSection" for element in case["elements"])
    assert case["analysis"]["type"] == "staged"
    recorders = {rec["type"]: rec for rec in case["recorders"]}
    assert recorders["node_displacement"]["raw_path"] == "data/Mphi.out"


def test_convert_rcframepushover_preserves_runtime_test_without_injected_fallback():
    entry = (
        REPO_ROOT
        / "docs/agent-reference/OpenSeesExamplesBasic"
        / "reinforced_concrete_frame_pushover_analysis"
        / "RCFramePushover.tcl"
    )

    case = tcl_to_strut.convert_tcl_to_case(entry, REPO_ROOT)

    stages = case["analysis"]["stages"]
    assert len(stages) == 2

    gravity = stages[0]["analysis"]
    assert "step_retry" not in gravity

    pushover = stages[1]["analysis"]
    assert pushover["type"] == "static_nonlinear"
    assert pushover["integrator"]["type"] == "DisplacementControl"
    assert pushover["algorithm"] == "Newton"
    assert "step_retry" not in pushover
    assert pushover["test_type"] == "NormDispIncr"
    assert pushover["tol"] == pytest.approx(1.0e-12)
    assert pushover["max_iters"] == 10
    assert pushover["test_print_flag"] == 3
    assert pushover["solver_chain"] == [
        {
            "algorithm": "Newton",
            "test_type": "NormDispIncr",
            "tol": pytest.approx(1.0e-12),
            "max_iters": 10,
        },
        {
            "algorithm": "ModifiedNewtonInitial",
            "test_type": "NormDispIncr",
            "tol": pytest.approx(1.0e-12),
            "max_iters": 1000,
        },
    ]


def test_convert_ex4_static_push_uses_solver_chain_and_continuation_policy():
    case = _convert_direct_tcl_manifest("opensees_example_ex4_portal2d_analyze_static_push")

    stages = case["analysis"]["stages"]
    pushover = stages[1]["analysis"]

    assert pushover["integrator"]["type"] == "DisplacementControl"
    assert len(pushover["solver_chain"]) == 4
    assert pushover["solver_chain"][0] == {
        "algorithm": "Newton",
        "test_type": "EnergyIncr",
        "tol": pytest.approx(1.0e-8),
        "max_iters": 6,
    }
    assert pushover["solver_chain"][1] == {
        "algorithm": "ModifiedNewtonInitial",
        "test_type": "NormDispIncr",
        "tol": pytest.approx(1.0e-8),
        "max_iters": 2000,
    }
    assert pushover["solver_chain"][2] == {
        "algorithm": "Broyden",
        "algorithm_options": {"max_iters": 8},
        "test_type": "EnergyIncr",
        "tol": pytest.approx(1.0e-8),
        "max_iters": 6,
    }
    assert pushover["solver_chain"][3] == {
        "algorithm": "NewtonLineSearch",
        "algorithm_options": {"alpha": pytest.approx(0.8)},
        "test_type": "EnergyIncr",
        "tol": pytest.approx(1.0e-8),
        "max_iters": 6,
    }
    assert pushover["step_retry"] == {
        "type": "continue_after_failure",
        "restore_primary_after_success": True,
        "continue_after_failure": "displacement_control_single_steps",
        "continue_target_disp": pytest.approx(43.2),
        "continue_max_steps": 102,
    }


def test_convert_ex5_static_push_uses_very_loose_parity_tolerances():
    case = _convert_direct_tcl_manifest("opensees_example_ex5_frame2d_analyze_static_push")

    assert case["parity_mode"] == "max_abs"
    assert case["parity_tolerance"] == {
        "rtol": pytest.approx(2.0),
        "atol": pytest.approx(1.0e-1),
    }
    assert case["parity_tolerance_by_category"]["deformation"] == {
        "rtol": pytest.approx(100.0),
        "atol": pytest.approx(1.0e-1),
    }
    assert case["parity_tolerance_by_recorder"]["drift"] == {
        "rtol": pytest.approx(100.0),
        "atol": pytest.approx(1.0e-1),
    }


def test_convert_ex4_static_cycle_uses_solver_chain_per_step():
    case = _convert_direct_tcl_manifest("opensees_example_ex4_portal2d_analyze_static_cycle")

    first_cycle = next(
        stage["analysis"]
        for stage in case["analysis"]["stages"]
        if stage["analysis"]["integrator"]["type"] == "DisplacementControl"
    )

    assert "step_retry" not in first_cycle
    assert len(first_cycle["solver_chain"]) == 4
    assert first_cycle["solver_chain"][0] == {
        "algorithm": "Newton",
        "test_type": "EnergyIncr",
        "tol": pytest.approx(1.0e-8),
        "max_iters": 6,
    }
    assert first_cycle["solver_chain"][1] == {
        "algorithm": "ModifiedNewtonInitial",
        "test_type": "NormDispIncr",
        "tol": pytest.approx(1.0e-8),
        "max_iters": 2000,
    }
    assert first_cycle["solver_chain"][2] == {
        "algorithm": "Broyden",
        "algorithm_options": {"max_iters": 8},
        "test_type": "EnergyIncr",
        "tol": pytest.approx(1.0e-8),
        "max_iters": 6,
    }
    assert first_cycle["solver_chain"][3] == {
        "algorithm": "NewtonLineSearch",
        "algorithm_options": {"alpha": pytest.approx(0.8)},
        "test_type": "EnergyIncr",
        "tol": pytest.approx(1.0e-8),
        "max_iters": 6,
    }


def test_solver_input_matches_json_adapter_for_tcl_case():
    entry = (
        REPO_ROOT
        / "docs/agent-reference/OpenSeesExamplesBasic"
        / "time_history_analysis_of_a_2d_elastic_cantilever_column"
        / "Ex1a.Canti2D.EQ.modif.tcl"
    )

    case = tcl_to_strut.convert_tcl_to_typed_case(entry, REPO_ROOT)
    json_case = case.to_json_dict()
    solver_input = case.to_solver_input(entry_path=entry)

    assert solver_input["model"] == json_case["model"]
    assert solver_input["nodes"] == json_case["nodes"]
    assert solver_input["materials"] == json_case["materials"]
    assert solver_input["sections"] == json_case["sections"]
    assert solver_input["elements"] == json_case["elements"]
    assert solver_input["time_series"] == json_case["time_series"]
    assert solver_input["analysis"] == json_case["analysis"]
    assert solver_input["recorders"] == json_case["recorders"]
    assert solver_input["__strut_case_dir"] == str(entry.parent.resolve())
    assert solver_input["__strut_case_json_path"] == str(entry.resolve())


def test_convert_rejects_cyclic_source(tmp_path: Path):
    script_a = tmp_path / "a.tcl"
    script_b = tmp_path / "b.tcl"
    script_a.write_text("source b.tcl\n", encoding="utf-8")
    script_b.write_text("source a.tcl\n", encoding="utf-8")

    with pytest.raises(tcl_to_strut.TclToStrutError) as exc_info:
        tcl_to_strut.convert_tcl_to_case(script_a, REPO_ROOT)

    assert "cyclic source detected" in str(exc_info.value)


def test_convert_evaluates_set_substitution_and_expr(tmp_path: Path):
    script = tmp_path / "case.tcl"
    script.write_text(
        "\n".join(
            [
                "wipe",
                "set height 144.0",
                "set tipLoad 2000.0",
                "model basic -ndm 2 -ndf 3",
                "node 1 0.0 0.0",
                "node 2 0.0 [expr $height]",
                "fix 1 1 1 1",
                "geomTransf Linear 1",
                "element elasticBeamColumn 1 1 2 3600 3225 1080000 1",
                "timeSeries Linear 1",
                "pattern Plain 1 1 {",
                "    load 2 0.0 [expr -$tipLoad] 0.0",
                "}",
                "constraints Plain",
                "numberer Plain",
                "system BandGeneral",
                "algorithm Linear",
                "integrator LoadControl 1.0",
                "analysis Static",
                "analyze 1",
                "",
            ]
        ),
        encoding="utf-8",
    )

    case = tcl_to_strut.convert_tcl_to_case(script, REPO_ROOT)

    assert case["nodes"][1]["y"] == pytest.approx(144.0)
    assert case["analysis"]["stages"][0]["loads"] == [
        {"node": 2, "dof": 2, "value": -2000.0}
    ]


def test_convert_rejects_builtin_outside_restricted_runtime(tmp_path: Path):
    script = tmp_path / "case.tcl"
    script.write_text("exec echo nope\n", encoding="utf-8")

    with pytest.raises(tcl_to_strut.TclToStrutError) as exc_info:
        tcl_to_strut.convert_tcl_to_case(script, REPO_ROOT)

    assert "unsupported Tcl/OpenSees command `exec`" in str(exc_info.value)


def test_convert_accepts_model_basicbuilder_alias(tmp_path: Path):
    script = tmp_path / "case.tcl"
    script.write_text(
        "\n".join(
            [
                "wipe",
                "model BasicBuilder -ndm 2 -ndf 3",
                "node 1 0.0 0.0",
                "node 2 0.0 144.0",
                "fix 1 1 1 1",
                "geomTransf Linear 1",
                "element elasticBeamColumn 1 1 2 3600 3225 1080000 1",
                "timeSeries Linear 1",
                "pattern Plain 1 1 {",
                "    load 2 0.0 -2000.0 0.0",
                "}",
                "constraints Plain",
                "numberer Plain",
                "system BandGeneral",
                "algorithm Linear",
                "integrator LoadControl 1.0",
                "analysis Static",
                "analyze 1",
                "",
            ]
        ),
        encoding="utf-8",
    )

    case = tcl_to_strut.convert_tcl_to_case(script, REPO_ROOT)

    assert case["model"] == {"ndm": 2, "ndf": 3}
    assert case["analysis"]["stages"][0]["analysis"]["type"] == "static_nonlinear"


def test_convert_normalizes_section_alias(tmp_path: Path):
    script = tmp_path / "case.tcl"
    script.write_text(
        "\n".join(
            [
                "wipe",
                "model basic -ndm 2 -ndf 3",
                "node 1 0.0 0.0",
                "node 2 0.0 144.0",
                "fix 1 1 1 1",
                "section fiberSec 1 {",
                "    patch rect 1 2 1 -0.2 -0.1 0.2 0.1",
                "}",
                "geomTransf Linear 1",
                "element elasticBeamColumn 1 1 2 10.0 3000.0 100.0 1",
                "timeSeries Linear 1",
                "pattern Plain 1 1 {",
                "    load 2 1.0 0.0 0.0",
                "}",
                "constraints Plain",
                "numberer Plain",
                "system BandGeneral",
                "algorithm Linear",
                "integrator LoadControl 1.0",
                "analysis Static",
                "analyze 1",
                "",
            ]
        ),
        encoding="utf-8",
    )

    case = tcl_to_strut.convert_tcl_to_case(script, REPO_ROOT)

    assert case["sections"][0]["type"] == "FiberSection2d"


def test_convert_2d_beam_uniform_preserves_opensees_wy_wx_order(tmp_path: Path):
    script = tmp_path / "case.tcl"
    script.write_text(
        "\n".join(
            [
                "wipe",
                "model basic -ndm 2 -ndf 3",
                "node 1 0.0 0.0",
                "node 2 0.0 144.0",
                "fix 1 1 1 1",
                "geomTransf Linear 1",
                "element elasticBeamColumn 1 1 2 3600 3225 1080000 1",
                "timeSeries Linear 1",
                "pattern Plain 1 1 {",
                "    eleLoad -ele 1 -type -beamUniform 0.0 -0.0095",
                "}",
                "constraints Plain",
                "numberer Plain",
                "system BandGeneral",
                "algorithm Linear",
                "integrator LoadControl 1.0",
                "analysis Static",
                "analyze 1",
                "",
            ]
        ),
        encoding="utf-8",
    )

    case = tcl_to_strut.convert_tcl_to_case(script, REPO_ROOT)

    assert case["analysis"]["stages"][0]["element_loads"] == [
        {"element": 1, "type": "beamUniform", "wy": 0.0, "wx": -0.0095}
    ]


def test_convert_normalizes_element_alias(tmp_path: Path):
    script = tmp_path / "case.tcl"
    script.write_text(
        "\n".join(
            [
                "wipe",
                "model basic -ndm 2 -ndf 3",
                "node 1 0.0 0.0",
                "node 2 0.0 144.0",
                "fix 1 1 1 1",
                "uniaxialMaterial Elastic 1 3000.0",
                "section Fiber 1 {",
                "    patch rect 1 2 1 -0.2 -0.1 0.2 0.1",
                "}",
                "geomTransf Linear 1",
                "element nonlinearBeamColumn 1 1 2 5 1 1",
                "timeSeries Linear 1",
                "pattern Plain 1 1 {",
                "    load 2 1.0 0.0 0.0",
                "}",
                "constraints Plain",
                "numberer Plain",
                "system BandGeneral",
                "algorithm Newton",
                "integrator LoadControl 1.0",
                "analysis Static",
                "analyze 1",
                "",
            ]
        ),
        encoding="utf-8",
    )

    case = tcl_to_strut.convert_tcl_to_case(script, REPO_ROOT)

    assert case["elements"][0]["type"] == "forceBeamColumn2d"


def test_convert_accepts_inline_node_mass(tmp_path: Path):
    script = tmp_path / "case.tcl"
    script.write_text(
        "\n".join(
            [
                "wipe",
                "model basic -ndm 2 -ndf 3",
                "node 1 0.0 0.0",
                "node 2 0.0 144.0 -mass 1.5 2.5 0.0",
                "fix 1 1 1 1",
                "geomTransf Linear 1",
                "element elasticBeamColumn 1 1 2 10.0 3000.0 100.0 1",
                "timeSeries Linear 1",
                "pattern Plain 1 1 {",
                "    load 2 0.0 -1.0 0.0",
                "}",
                "constraints Plain",
                "numberer Plain",
                "system BandGeneral",
                "algorithm Linear",
                "integrator LoadControl 1.0",
                "analysis Static",
                "analyze 1",
                "set rxn [nodeReaction 1]",
                "puts $rxn",
                "",
            ]
        ),
        encoding="utf-8",
    )

    case = tcl_to_strut.convert_tcl_to_case(script, REPO_ROOT)

    assert case["masses"] == [
        {"node": 2, "dof": 1, "value": pytest.approx(1.5)},
        {"node": 2, "dof": 2, "value": pytest.approx(2.5)},
    ]


def test_convert_imports_zero_length_section(tmp_path: Path):
    script = tmp_path / "case.tcl"
    script.write_text(
        "\n".join(
            [
                "wipe",
                "model basic -ndm 2 -ndf 3",
                "uniaxialMaterial Elastic 1 3000.0",
                "node 1 0.0 0.0",
                "node 2 0.0 0.0",
                "fix 1 1 1 1",
                "fix 2 0 1 0",
                "section Fiber 9 {",
                "    patch rect 1 2 1 -0.2 -0.1 0.2 0.1",
                "}",
                "element zeroLengthSection 6 1 2 9",
                "timeSeries Linear 1",
                "pattern Plain 1 1 {",
                "    load 2 0.0 0.0 1.0",
                "}",
                "constraints Plain",
                "numberer Plain",
                "system SparseGeneral -piv",
                "algorithm Newton",
                "integrator DisplacementControl 2 3 0.01 1 0.01 0.01",
                "analysis Static",
                "analyze 5",
                "",
            ]
        ),
        encoding="utf-8",
    )

    case = tcl_to_strut.convert_tcl_to_case(script, REPO_ROOT)

    assert case["elements"] == [{"id": 6, "type": "zeroLengthSection", "nodes": [1, 2], "section": 9}]


def test_convert_imports_block2d_quad_and_fallback_eigen(tmp_path: Path):
    script = tmp_path / "case.tcl"
    script.write_text(
        "\n".join(
            [
                "wipe",
                "model basic -ndm 2 -ndf 2",
                "nDMaterial ElasticIsotropic 1 1000.0 0.25 3.0",
                'block2D 1 1 1 1 quad "1 PlaneStress2D 1" {',
                "    1 0 0",
                "    2 40 0",
                "    3 40 10",
                "    4 0 10",
                "}",
                "fix 1 1 1",
                "fix 2 0 1",
                "timeSeries Linear 1",
                "pattern Plain 1 1 {",
                "    load 3 0.0 -1.0",
                "    load 4 0.0 -1.0",
                "}",
                "constraints Plain",
                "numberer RCM",
                "system ProfileSPD",
                "algorithm Newton",
                "integrator LoadControl 1.0",
                "analysis Static",
                "analyze 1",
                "rayleigh 0.0 0.0 0.0 [expr 2*0.02/sqrt([eigen 1])]",
                "wipeAnalysis",
                "setTime 0.0",
                "remove loadPattern 1",
                "test EnergyIncr 1.0e-12 10 0",
                "algorithm Newton",
                "numberer RCM",
                "constraints Plain",
                "integrator Newmark 0.5 0.25",
                "system BandGeneral",
                "analysis Transient",
                "analyze 2 0.5",
                "",
            ]
        ),
        encoding="utf-8",
    )

    case = tcl_to_strut.convert_tcl_to_case(script, REPO_ROOT)

    assert case["materials"] == [
        {"id": 1, "type": "ElasticIsotropic", "params": {"E": 1000.0, "nu": 0.25, "rho": 3.0}}
    ]
    assert case["elements"] == [
        {
            "id": 1,
            "type": "fourNodeQuad",
            "nodes": [1, 2, 4, 3],
            "material": 1,
            "thickness": 1.0,
            "formulation": "PlaneStress",
        }
    ]
    assert len(case["analysis"]["stages"]) == 2


def test_convert_imports_block2d_curved_quad_geometry(tmp_path: Path):
    script = tmp_path / "case.tcl"
    script.write_text(
        "\n".join(
            [
                "wipe",
                "model basic -ndm 2 -ndf 2",
                "nDMaterial ElasticIsotropic 1 1000.0 0.25 0.0",
                'block2D 1 1 10 20 quad "2.0 PlaneStress 1" {',
                "    1 0 0",
                "    2 10 0",
                "    3 12 8",
                "    4 -1 9",
                "    5 4 0",
                "    6 11 3",
                "    7 5 9",
                "    8 -1 4",
                "    9 5 5",
                "}",
                "fix 10 1 1",
                "fix 11 0 1",
                "timeSeries Linear 1",
                "pattern Plain 1 1 {",
                "    load 12 0.0 -1.0",
                "    load 13 0.0 -1.0",
                "}",
                "constraints Plain",
                "numberer RCM",
                "system ProfileSPD",
                "algorithm Newton",
                "integrator LoadControl 1.0",
                "analysis Static",
                "analyze 1",
                "",
            ]
        ),
        encoding="utf-8",
    )

    case = tcl_to_strut.convert_tcl_to_case(script, REPO_ROOT)

    assert case["nodes"] == [
        {"id": 10, "x": 0.0, "y": 0.0, "constraints": [1, 2]},
        {"id": 11, "x": 10.0, "y": 0.0, "constraints": [2]},
        {"id": 12, "x": -1.0, "y": 9.0},
        {"id": 13, "x": 12.0, "y": 8.0},
    ]
    assert case["elements"] == [
        {
            "id": 20,
            "type": "fourNodeQuad",
            "nodes": [10, 11, 13, 12],
            "material": 1,
            "thickness": 2.0,
            "formulation": "PlaneStress",
        }
    ]


def test_convert_imports_block2d_shell_geometry(tmp_path: Path):
    script = tmp_path / "case.tcl"
    script.write_text(
        "\n".join(
            [
                "wipe",
                "model basic -ndm 3 -ndf 6",
                "section ElasticMembranePlateSection 1 3000.0 0.25 1.175 1.27",
                'block2D 1 1 1 1 shell "1" {',
                "    1 -20 0 0",
                "    2 -20 0 40",
                "    3 20 0 40",
                "    4 20 0 0",
                "    9 0 10 20",
                "}",
                "fix 1 1 1 1 0 1 1",
                "fix 2 1 1 1 0 1 1",
                "timeSeries Linear 1",
                "pattern Plain 1 1 {",
                "    load 3 0.0 -0.5 0.0 0.0 0.0 0.0",
                "    load 4 0.0 -0.5 0.0 0.0 0.0 0.0",
                "}",
                "constraints Plain",
                "numberer RCM",
                "system SparseGeneral -piv",
                "algorithm Newton",
                "integrator LoadControl 1.0",
                "analysis Static",
                "analyze 1",
                "",
            ]
        ),
        encoding="utf-8",
    )

    case = tcl_to_strut.convert_tcl_to_case(script, REPO_ROOT)

    assert case["sections"] == [
        {
            "id": 1,
            "type": "ElasticMembranePlateSection",
            "params": {"E": 3000.0, "nu": 0.25, "h": 1.175, "rho": 1.27},
        }
    ]
    assert case["nodes"] == [
        {"id": 1, "x": -20.0, "y": 0.0, "z": 0.0, "constraints": [1, 2, 3, 5, 6]},
        {"id": 2, "x": -20.0, "y": 0.0, "z": 40.0, "constraints": [1, 2, 3, 5, 6]},
        {"id": 3, "x": 20.0, "y": 0.0, "z": 0.0},
        {"id": 4, "x": 20.0, "y": 0.0, "z": 40.0},
    ]
    assert case["elements"] == [{"id": 1, "type": "shell", "nodes": [1, 2, 4, 3], "section": 1}]


def test_convert_rejects_block2d_nine_node_elements_until_schema_supports_them(tmp_path: Path):
    script = tmp_path / "case.tcl"
    script.write_text(
        "\n".join(
            [
                "wipe",
                "model basic -ndm 2 -ndf 2",
                "nDMaterial ElasticIsotropic 1 1000.0 0.25 0.0",
                'block2D 2 2 1 1 quad "1.0 PlaneStress 1" -numEleNodes 9 {',
                "    1 0 0",
                "    2 10 0",
                "    3 10 10",
                "    4 0 10",
                "}",
                "constraints Plain",
                "numberer RCM",
                "system ProfileSPD",
                "algorithm Newton",
                "integrator LoadControl 1.0",
                "analysis Static",
                "analyze 1",
                "",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(tcl_to_strut.TclToStrutError, match="numEleNodes 9"):
        tcl_to_strut.convert_tcl_to_case(script, REPO_ROOT)


def test_convert_applies_fixy_to_matching_nodes(tmp_path: Path):
    script = tmp_path / "case.tcl"
    script.write_text(
        "\n".join(
            [
                "wipe",
                "model basic -ndm 2 -ndf 3",
                "node 1 0.0 0.0",
                "node 2 10.0 0.0",
                "node 3 0.0 5.0",
                "fixY 0.0 1 1 0",
                "timeSeries Linear 1",
                "pattern Plain 1 1 {",
                "    load 3 1.0 0.0 0.0",
                "}",
                "constraints Plain",
                "numberer RCM",
                "system BandGeneral",
                "algorithm Newton",
                "integrator LoadControl 1.0",
                "analysis Static",
                "analyze 1",
                "",
            ]
        ),
        encoding="utf-8",
    )

    case = tcl_to_strut.convert_tcl_to_case(script, REPO_ROOT)

    assert case["nodes"] == [
        {"id": 1, "x": 0.0, "y": 0.0, "constraints": [1, 2]},
        {"id": 2, "x": 10.0, "y": 0.0, "constraints": [1, 2]},
        {"id": 3, "x": 0.0, "y": 5.0},
    ]


def test_convert_mass_ignores_extra_trailing_dofs(tmp_path: Path):
    script = tmp_path / "case.tcl"
    script.write_text(
        "\n".join(
            [
                "wipe",
                "model basic -ndm 2 -ndf 3",
                "node 1 0.0 0.0",
                "mass 1 2.5 0.0 1.5 0.0 0.0 0.0",
                "timeSeries Linear 1",
                "pattern Plain 1 1 {",
                "    load 1 1.0 0.0 0.0",
                "}",
                "constraints Plain",
                "numberer RCM",
                "system BandGeneral",
                "algorithm Newton",
                "integrator LoadControl 1.0",
                "analysis Static",
                "analyze 1",
                "",
            ]
        ),
        encoding="utf-8",
    )

    case = tcl_to_strut.convert_tcl_to_case(script, REPO_ROOT)

    assert case["masses"] == [
        {"node": 1, "dof": 1, "value": 2.5},
        {"node": 1, "dof": 3, "value": 1.5},
    ]


def test_convert_allows_nodebounds_in_display_helper_proc(tmp_path: Path):
    helper = tmp_path / "helper.tcl"
    helper.write_text(
        "\n".join(
            [
                "proc ShowBounds {} {",
                "    set bounds [nodeBounds]",
                "    vrp 0 0 0",
                "    plane 1 -1",
                "    projection 1",
                "    fill 1",
                "    port -1 1 -1 1",
                "    return [llength $bounds]",
                "}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    script = tmp_path / "case.tcl"
    script.write_text(
        "\n".join(
            [
                "wipe",
                "model basic -ndm 2 -ndf 3",
                "node 1 0.0 0.0",
                "node 2 10.0 5.0",
                f"source {helper}",
                "ShowBounds",
                "timeSeries Linear 1",
                "pattern Plain 1 1 {",
                "    load 2 1.0 0.0 0.0",
                "}",
                "constraints Plain",
                "numberer RCM",
                "system BandGeneral",
                "algorithm Newton",
                "integrator LoadControl 1.0",
                "analysis Static",
                "analyze 1",
                "",
            ]
        ),
        encoding="utf-8",
    )

    case = tcl_to_strut.convert_tcl_to_case(script, REPO_ROOT)

    assert case["nodes"] == [{"id": 1, "x": 0.0, "y": 0.0}, {"id": 2, "x": 10.0, "y": 5.0}]


def test_convert_ignores_plot_recorder(tmp_path: Path):
    script = tmp_path / "case.tcl"
    script.write_text(
        "\n".join(
            [
                "wipe",
                "model basic -ndm 2 -ndf 3",
                "node 1 0.0 0.0",
                "node 2 1.0 0.0",
                "recorder plot Data/DFree.out ForceDisp 910 10 400 400 -columns 2 1",
                "timeSeries Linear 1",
                "pattern Plain 1 1 {",
                "    load 2 1.0 0.0 0.0",
                "}",
                "constraints Plain",
                "numberer RCM",
                "system BandGeneral",
                "algorithm Newton",
                "integrator LoadControl 1.0",
                "analysis Static",
                "analyze 1",
                "",
            ]
        ),
        encoding="utf-8",
    )

    case = tcl_to_strut.convert_tcl_to_case(script, REPO_ROOT)

    assert case["recorders"] == []


def test_convert_skips_truss_basic_force_recorders(tmp_path: Path):
    script = tmp_path / "case.tcl"
    script.write_text(
        "\n".join(
            [
                "wipe",
                "model basic -ndm 2 -ndf 2",
                "node 1 0.0 0.0",
                "node 2 144.0 0.0",
                "fix 1 1 1",
                "uniaxialMaterial Elastic 1 3000.0",
                "element truss 1 1 2 10.0 1",
                "timeSeries Linear 1",
                "pattern Plain 1 1 {",
                "    load 2 1.0 0.0",
                "}",
                "constraints Plain",
                "numberer Plain",
                "system BandGeneral",
                "algorithm Linear",
                "integrator LoadControl 1.0",
                "analysis Static",
                "recorder Element -file local.out -ele 1 basicForces",
                "analyze 1",
                "",
            ]
        ),
        encoding="utf-8",
    )

    case = tcl_to_strut.convert_tcl_to_case(script, REPO_ROOT)

    assert case["recorders"] == []


def test_convert_preserves_algorithm_integrator_test_and_system_options(tmp_path: Path):
    script = tmp_path / "case.tcl"
    script.write_text(
        "\n".join(
            [
                "wipe",
                "model basic -ndm 2 -ndf 3",
                "node 1 0.0 0.0",
                "node 2 0.0 144.0",
                "fix 1 1 1 1",
                "geomTransf Linear 1",
                "element elasticBeamColumn 1 1 2 3600 3225 1080000 1",
                "timeSeries Linear 1",
                "pattern Plain 1 1 {",
                "    load 2 0.0 -2000.0 0.0",
                "}",
                "constraints Plain",
                "numberer Plain",
                "system SparseGeneral -piv",
                "test NormDispIncr 1.0e-12 10 3 7 extraA extraB",
                "algorithm Broyden 8",
                "integrator LoadControl 0.1 5 0.01 0.2 extraLC",
                "analysis Static",
                "analyze 1",
                "algorithm NewtonLineSearch 0.8",
                "integrator DisplacementControl 2 1 0.01 6 1.0e-6 0.1 extraDC",
                "analyze 1",
            ]
        ),
        encoding="utf-8",
    )

    case = tcl_to_strut.convert_tcl_to_case(script, REPO_ROOT)
    stages = case["analysis"]["stages"]
    assert len(stages) == 2

    first = stages[0]["analysis"]
    assert first["algorithm"] == "Broyden"
    assert first["algorithm_options"]["max_iters"] == 8
    assert first["algorithm_options"]["raw_args"] == ["8"]
    assert first["integrator"]["type"] == "LoadControl"
    assert first["integrator"]["step"] == pytest.approx(0.1)
    assert first["integrator"]["num_iter"] == 5
    assert first["integrator"]["min_step"] == pytest.approx(0.01)
    assert first["integrator"]["max_step"] == pytest.approx(0.2)
    assert first["integrator"]["extra_args"] == ["extraLC"]
    assert first["test_type"] == "NormDispIncr"
    assert first["test_print_flag"] == 3
    assert first["test_extra_args"] == ["7", "extraA", "extraB"]
    assert first["numberer"] == "Plain"
    assert first["system"] == "SuperLU"
    assert first["system_options"] == ["-piv"]

    second = stages[1]["analysis"]
    assert second["algorithm"] == "NewtonLineSearch"
    assert second["algorithm_options"]["alpha"] == pytest.approx(0.8)
    assert second["integrator"]["type"] == "DisplacementControl"
    assert second["integrator"]["node"] == 2
    assert second["integrator"]["dof"] == 1
    assert second["integrator"]["du"] == pytest.approx(0.01)
    assert second["integrator"]["num_iter"] == 6
    assert second["integrator"]["min_du"] == pytest.approx(1.0e-6)
    assert second["integrator"]["max_du"] == pytest.approx(0.1)
    assert second["integrator"]["extra_args"] == ["extraDC"]
    assert second["numberer"] == "Plain"


def test_convert_maps_corot_truss_section_syntax_and_preserves_print_commands(
    tmp_path: Path,
):
    script = tmp_path / "case.tcl"
    script.write_text(
        "\n".join(
            [
                "wipe",
                "model BasicBuilder -ndm 2 -ndf 2",
                "node 1 0.0 0.0",
                "node 2 144.0 0.0",
                "fix 1 1 1",
                "fix 2 0 0",
                "uniaxialMaterial Elastic 1 3000.0",
                "section Elastic 1 3000.0 5.0 10000.0",
                "element corotTruss 1 1 2 1",
                "timeSeries Linear 1",
                "pattern Plain 1 1 {",
                "    load 2 1.0 0.0",
                "}",
                "constraints Plain",
                "numberer RCM",
                "system BandSPD",
                "algorithm Newton",
                "integrator LoadControl 1.0",
                "analysis Static",
                "analyze 1",
                "print node 2",
                "print ele",
                "",
            ]
        ),
        encoding="utf-8",
    )

    case = tcl_to_strut.convert_tcl_to_case(script, REPO_ROOT)

    assert case["materials"] == [{"id": 1, "type": "Elastic", "params": {"E": 3000.0}}]
    assert case["elements"] == [
        {"id": 1, "type": "truss", "nodes": [1, 2], "area": 5.0, "material": 1}
    ]
    stage = case["analysis"]["stages"][0]
    assert stage["analysis"]["numberer"] == "RCM"
    assert stage["analysis"]["system"] == "BandSPD"
    assert stage["print_commands"] == [
        {"args": ["node", "2"]},
        {"args": ["ele"]},
    ]


def test_convert_adds_envelope_element_group_layout(tmp_path: Path):
    script = tmp_path / "case.tcl"
    script.write_text(
        "\n".join(
            [
                "wipe",
                "model basic -ndm 2 -ndf 3",
                "node 1 0.0 0.0",
                "node 2 0.0 144.0",
                "node 3 0.0 288.0",
                "fix 1 1 1 1",
                "geomTransf Linear 1",
                "element elasticBeamColumn 1 1 2 3600 3225 1080000 1",
                "element elasticBeamColumn 2 2 3 3600 3225 1080000 1",
                "recorder EnvelopeElement -file ele32.out -ele 1 2 forces",
                "constraints Plain",
                "numberer Plain",
                "system BandGeneral",
                "algorithm Linear",
                "integrator LoadControl 1.0",
                "analysis Static",
                "analyze 1",
            ]
        ),
        encoding="utf-8",
    )

    case = tcl_to_strut.convert_tcl_to_case(script, REPO_ROOT)
    recorders = [rec for rec in case["recorders"] if rec["type"] == "envelope_element_force"]
    assert len(recorders) == 2
    for recorder in recorders:
        layout = recorder.get("group_layout")
        assert layout is not None
        assert layout["type"] == "envelope_element_force"
        assert layout["elements"] == [1, 2]
        assert layout["values_per_element"] == [6, 6]


def test_convert_adds_envelope_element_group_layout_with_time_without_doubling(
    tmp_path: Path,
):
    script = tmp_path / "case.tcl"
    script.write_text(
        "\n".join(
            [
                "wipe",
                "model basic -ndm 2 -ndf 3",
                "node 1 0.0 0.0",
                "node 2 0.0 144.0",
                "node 3 0.0 288.0",
                "fix 1 1 1 1",
                "geomTransf Linear 1",
                "element elasticBeamColumn 1 1 2 3600 3225 1080000 1",
                "element elasticBeamColumn 2 2 3 3600 3225 1080000 1",
                "recorder EnvelopeElement -file ele32.out -time -ele 1 2 forces",
                "constraints Plain",
                "numberer Plain",
                "system BandGeneral",
                "algorithm Linear",
                "integrator LoadControl 1.0",
                "analysis Static",
                "analyze 1",
            ]
        ),
        encoding="utf-8",
    )

    case = tcl_to_strut.convert_tcl_to_case(script, REPO_ROOT)
    recorders = [rec for rec in case["recorders"] if rec["type"] == "envelope_element_force"]
    assert len(recorders) == 2
    for recorder in recorders:
        layout = recorder.get("group_layout")
        assert layout is not None
        assert layout["values_per_element"] == [6, 6]


def test_convert_adds_envelope_local_force_and_node_recorders(tmp_path: Path):
    script = tmp_path / "case.tcl"
    script.write_text(
        "\n".join(
            [
                "wipe",
                "model basic -ndm 2 -ndf 3",
                "node 1 0.0 0.0",
                "node 2 0.0 144.0",
                "node 3 0.0 288.0",
                "fix 1 1 1 1",
                "geomTransf Linear 1",
                "element elasticBeamColumn 1 1 2 3600 3225 1080000 1",
                "element elasticBeamColumn 2 2 3 3600 3225 1080000 1",
                "recorder EnvelopeElement -file ele32.out -time -ele 1 2 localForce",
                "recorder EnvelopeNode -time -file disp.out -node 2 3 -dof 1 disp",
                "recorder EnvelopeNode -time -file accel.out -timeSeries 1 -node 2 3 -dof 1 accel",
                "constraints Plain",
                "numberer Plain",
                "system BandGeneral",
                "algorithm Linear",
                "integrator LoadControl 1.0",
                "analysis Static",
                "analyze 1",
            ]
        ),
        encoding="utf-8",
    )

    case = tcl_to_strut.convert_tcl_to_case(script, REPO_ROOT)

    local_force = [
        rec for rec in case["recorders"] if rec["type"] == "envelope_element_local_force"
    ]
    assert len(local_force) == 2
    for recorder in local_force:
        assert recorder["group_layout"]["type"] == "envelope_element_local_force"
        assert recorder["group_layout"]["values_per_element"] == [6, 6]

    disp = [rec for rec in case["recorders"] if rec["type"] == "envelope_node_displacement"]
    assert len(disp) == 2
    for recorder in disp:
        assert recorder["group_layout"]["type"] == "envelope_node_displacement"
        assert recorder["group_layout"]["nodes"] == [2, 3]
        assert recorder["group_layout"]["values_per_node"] == [1, 1]

    accel = [rec for rec in case["recorders"] if rec["type"] == "envelope_node_acceleration"]
    assert len(accel) == 2
    for recorder in accel:
        assert recorder["time_series"] == 1
        assert recorder["group_layout"]["type"] == "envelope_node_acceleration"
        assert recorder["group_layout"]["values_per_node"] == [1, 1]


def test_convert_accepts_xml_plastic_rotation_recorder_and_skips_it(tmp_path: Path):
    script = tmp_path / "case.tcl"
    script.write_text(
        "\n".join(
            [
                "wipe",
                "model basic -ndm 2 -ndf 3",
                "node 1 0.0 0.0",
                "node 2 0.0 144.0",
                "fix 1 1 1 1",
                "geomTransf Linear 1",
                "element elasticBeamColumn 1 1 2 3600 3225 1080000 1",
                "recorder Element -xml Data/PlasticRotation.out -time -ele 1 plasticRotation",
                "constraints Plain",
                "numberer Plain",
                "system BandGeneral",
                "algorithm Linear",
                "integrator LoadControl 1.0",
                "analysis Static",
                "analyze 1",
            ]
        ),
        encoding="utf-8",
    )

    case = tcl_to_strut.convert_tcl_to_case(script, REPO_ROOT)

    assert case["recorders"] == []


def test_convert_uniform_excitation_inline_series_resolves_values_path_case(tmp_path: Path):
    (tmp_path / "GMfiles").mkdir(parents=True, exist_ok=True)
    (tmp_path / "GMfiles" / "H-e12140.at2").write_text("0.0\n", encoding="utf-8")
    script = tmp_path / "case.tcl"
    script.write_text(
        "\n".join(
            [
                "wipe",
                "model basic -ndm 2 -ndf 3",
                "node 1 0.0 0.0",
                "fix 1 1 1 1",
                'set AccelSeries "Series -dt 0.02 -filePath GMfiles/H-E12140.at2 -factor 2.0"',
                "pattern UniformExcitation 1 1 -accel $AccelSeries",
                "constraints Plain",
                "numberer Plain",
                "system BandGeneral",
                "algorithm Linear",
                "integrator Newmark 0.5 0.25",
                "analysis Transient",
                "analyze 1 0.01",
            ]
        ),
        encoding="utf-8",
    )

    case = tcl_to_strut.convert_tcl_to_case(script, REPO_ROOT)
    path_series = [ts for ts in case["time_series"] if ts["type"] == "Path"]
    assert len(path_series) == 1
    assert path_series[0]["values_path"] == str(
        (tmp_path / "GMfiles" / "H-e12140.at2").resolve()
    )


def test_convert_uniform_excitation_time_series_path_resolves_values_path_case(
    tmp_path: Path,
):
    (tmp_path / "GMfiles").mkdir(parents=True, exist_ok=True)
    (tmp_path / "GMfiles" / "H-e12140.at2").write_text("0.0\n", encoding="utf-8")
    script = tmp_path / "case.tcl"
    script.write_text(
        "\n".join(
            [
                "wipe",
                "model basic -ndm 2 -ndf 3",
                "node 1 0.0 0.0",
                "fix 1 1 1 1",
                "timeSeries Path 3 -dt 0.01 -filePath GMfiles/h-e12140.at2 -factor 1.0",
                "pattern UniformExcitation 1 1 -accel 3",
                "constraints Plain",
                "numberer Plain",
                "system BandGeneral",
                "algorithm Linear",
                "integrator Newmark 0.5 0.25",
                "analysis Transient",
                "analyze 1 0.01",
            ]
        ),
        encoding="utf-8",
    )

    case = tcl_to_strut.convert_tcl_to_case(script, REPO_ROOT)
    path_series = [ts for ts in case["time_series"] if ts["type"] == "Path"]
    assert len(path_series) == 1
    assert path_series[0]["values_path"] == str(
        (tmp_path / "GMfiles" / "H-e12140.at2").resolve()
    )
