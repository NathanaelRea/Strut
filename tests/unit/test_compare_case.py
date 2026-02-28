import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
COMPARE_CASE_PATH = REPO_ROOT / "scripts" / "compare_case.py"


def _load_compare_case_module():
    module_name = "strut_compare_case_test_module"
    spec = importlib.util.spec_from_file_location(module_name, COMPARE_CASE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


compare_case = _load_compare_case_module()


def test_analysis_is_transient_for_staged_with_transient_stage():
    analysis = {
        "type": "staged",
        "stages": [
            {"analysis": {"type": "static_nonlinear"}},
            {"analysis": {"type": "transient_nonlinear", "dt": 0.01}},
        ],
    }

    assert compare_case._analysis_is_transient(analysis) is True


def test_analysis_is_transient_false_for_non_transient_staged():
    analysis = {
        "type": "staged",
        "stages": [
            {"analysis": {"type": "static_linear"}},
            {"analysis": {"type": "static_nonlinear"}},
        ],
    }

    assert compare_case._analysis_is_transient(analysis) is False


def test_global_parity_tolerance_overrides_recorder_defaults():
    rtol, atol = compare_case._resolve_recorder_tolerance(
        "element_force",
        0.2,
        5.0e-4,
        True,
        {},
    )

    assert rtol == 0.2
    assert atol == 5.0e-4


def test_default_recorder_tolerance_used_without_global_override():
    rtol, atol = compare_case._resolve_recorder_tolerance(
        "element_force",
        compare_case.REL_TOL,
        compare_case.ABS_TOL,
        False,
        {},
    )

    assert rtol == compare_case.DEFAULT_RECORDER_TOLERANCES["element_force"]["rtol"]
    assert atol == compare_case.DEFAULT_RECORDER_TOLERANCES["element_force"]["atol"]
