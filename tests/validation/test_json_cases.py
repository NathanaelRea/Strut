import json
from pathlib import Path

import pytest


_NONLINEAR_ALGORITHMS = {
    "Newton",
    "ModifiedNewton",
    "ModifiedNewtonInitial",
    "Broyden",
    "NewtonLineSearch",
}


def _case_paths():
    base = Path(__file__).resolve().parent
    paths = []
    for path in sorted(base.glob("*/*.json")):
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            continue
        if "model" not in data or "nodes" not in data:
            continue
        paths.append(path)
    return paths


def _iter_nonlinear_analyses(analysis):
    if not isinstance(analysis, dict):
        return
    analysis_type = analysis.get("type")
    if analysis_type == "staged":
        for stage in analysis.get("stages", []):
            if not isinstance(stage, dict):
                continue
            yield from _iter_nonlinear_analyses(stage.get("analysis"))
        return
    if analysis_type in {"static_nonlinear", "transient_nonlinear"}:
        yield analysis


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
    for analysis in _iter_nonlinear_analyses(data.get("analysis")):
        algorithm = analysis.get("algorithm")
        if algorithm is not None:
            assert algorithm in _NONLINEAR_ALGORITHMS

        solver_chain = analysis.get("solver_chain")
        if solver_chain is not None:
            assert isinstance(solver_chain, list)
            assert len(solver_chain) > 0
            for attempt in solver_chain:
                assert isinstance(attempt, dict)
                attempt_algorithm = attempt.get("algorithm", algorithm)
                assert attempt_algorithm in _NONLINEAR_ALGORITHMS
