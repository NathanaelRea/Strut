import json
from pathlib import Path

import pytest


def _case_paths():
    base = Path(__file__).resolve().parent
    return sorted(base.glob("*/**/*.json"))


@pytest.mark.parametrize("case_path", _case_paths())
def test_case_json(case_path: Path):
    data = json.loads(case_path.read_text())

    assert "schema_version" in data
    assert "metadata" in data
    assert "model" in data
    assert "nodes" in data
    assert "elements" in data
    assert "recorders" in data


@pytest.mark.parametrize("case_path", _case_paths())
def test_nonlinear_algorithm_values(case_path: Path):
    data = json.loads(case_path.read_text())
    analysis = data.get("analysis", {})
    if not isinstance(analysis, dict):
        return

    analysis_type = analysis.get("type")
    if analysis_type not in {"static_nonlinear", "transient_nonlinear"}:
        return

    algorithm = analysis.get("algorithm")
    if algorithm is None:
        return

    assert algorithm in {"Newton", "ModifiedNewton"}
