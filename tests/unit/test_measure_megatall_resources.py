from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "measure_megatall_resources.py"
SMOKE_CASE = (
    REPO_ROOT
    / "tests"
    / "validation"
    / "opensees_megatall_building_model1_dynamiccpu"
    / "megatall_smoke.json"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "strut_measure_megatall_resources_test_module", SCRIPT_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


measure_megatall_resources = _load_module()


def test_summarize_case_input_reports_megatall_smoke_cost_centers():
    data = measure_megatall_resources._load_json(SMOKE_CASE)

    summary = measure_megatall_resources.summarize_case_input(data)

    assert summary["nodes"] == 6
    assert summary["elements"] == 3
    assert summary["element_types"] == {
        "elasticBeamColumn3d": 1,
        "shell": 1,
        "truss": 1,
    }
    assert summary["recorders"]["count"] == 3
    assert summary["recorders"]["output_targets"] == 3
    assert summary["mp_constraints"] == 1
    assert summary["shells"]["elements"] == 1
    assert summary["shells"]["layered_sections"] == 1
    assert summary["stages"]["root_type"] == "staged"
    assert summary["stages"]["stage_count"] == 2
    assert summary["stages"]["static_steps"] == 2
    assert summary["stages"]["transient_steps"] == 5


def test_build_report_skip_run_keeps_runtime_fields_empty():
    report = measure_megatall_resources.build_report(
        SMOKE_CASE,
        case_label="smoke",
        compute_only=False,
        skip_run=True,
        output_dir=None,
        timeout_seconds=None,
    )

    assert report["case_label"] == "smoke"
    assert report["run"] is None
    assert report["output"] is None
    assert report["failure"] is None
    assert report["input_summary"]["recorders"]["types"] == {
        "drift": 1,
        "node_displacement": 1,
        "node_reaction": 1,
    }


def test_summarize_failure_uses_progress_to_classify_timeout():
    failure = measure_megatall_resources._summarize_failure(
        {
            "returncode": -15,
            "timed_out": True,
            "stdout_tail": "",
            "stderr_tail": "",
            "progress": {
                "status": "running_step",
                "analysis_type": "static_nonlinear",
                "stage_number": 1,
                "stage_count": 2,
                "step_number": 1,
                "step_count": 10,
            },
        },
        {"file_count": 0, "total_bytes": 0},
    )

    assert failure == {
        "classification": "timeout",
        "likely_root_cause": "memory",
        "status": "running_step",
        "analysis_type": "static_nonlinear",
        "stage_number": 1,
        "stage_count": 2,
        "step_number": 1,
        "step_count": 10,
        "error_text": None,
        "output_file_count": 0,
    }


def test_prefer_progress_keeps_furthest_observed_stage():
    current = {
        "status": "running_stage",
        "analysis_type": "static_nonlinear",
        "stage_number": 1,
        "stage_count": 2,
        "step_number": 0,
        "step_count": 10,
    }
    candidate = {
        "status": "running_step",
        "analysis_type": "static_nonlinear",
        "stage_number": 1,
        "stage_count": 2,
        "step_number": 3,
        "step_count": 10,
    }

    preferred = measure_megatall_resources._prefer_progress(current, candidate)

    assert preferred == candidate


def test_summarize_failure_classifies_preload_sigsegv_as_crash():
    failure = measure_megatall_resources._summarize_failure(
        {
            "returncode": 1,
            "timed_out": False,
            "stdout_tail": "",
            "stderr_tail": (
                "subprocess.CalledProcessError: Command '['foo', '--output', '/tmp/x']' "
                "died with <Signals.SIGSEGV: 11>."
            ),
            "progress": None,
        },
        {"file_count": 0, "total_bytes": 0},
    )

    assert failure == {
        "classification": "crash",
        "likely_root_cause": "input_parse_or_load",
        "status": None,
        "analysis_type": None,
        "stage_number": None,
        "stage_count": None,
        "step_number": None,
        "step_count": None,
        "error_text": (
            "subprocess.CalledProcessError: Command '['foo', '--output', '/tmp/x']' "
            "died with <Signals.SIGSEGV: 11>."
        ),
        "output_file_count": 0,
    }


def test_default_timeout_is_ten_minutes():
    assert measure_megatall_resources.DEFAULT_TIMEOUT_SECONDS == 600.0
