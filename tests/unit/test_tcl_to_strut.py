import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
TCL_TO_STRUT_PATH = REPO_ROOT / "scripts" / "tcl_to_strut.py"


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


def test_convert_load_drops_oversized_numeric_tail_like_benchmarked_opensees(tmp_path: Path):
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

    assert case.get("loads", []) == []


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
    assert recorders[("section_deformation", "DefoEle1sec1")].get("parity", True) is True
    assert recorders[("section_deformation", "DefoEle1sec5")].get("parity", True) is True


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


def test_convert_rcframepushover_preserves_explicit_step_retry():
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
    assert pushover["fallback_algorithm"] == "ModifiedNewtonInitial"
    assert pushover["fallback_test_type"] == "NormDispIncr"
    assert pushover["fallback_tol"] == pytest.approx(1.0e-12)
    assert pushover["fallback_max_iters"] == 1000
    assert pushover["step_retry"] == {
        "type": "on_failure_retry_once",
        "restore_primary_after_success": True,
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
    assert first["system"] == "SparseGeneral"
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
