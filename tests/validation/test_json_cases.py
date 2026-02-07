import json
import os
from pathlib import Path

import pytest


def _case_paths():
    base = Path(__file__).resolve().parent
    return sorted(base.glob("*/**/*.json"))


@pytest.mark.parametrize("case_path", _case_paths())
def test_case_json(case_path: Path):
    data = json.loads(case_path.read_text())
    enabled = data.get("enabled", True)
    if not enabled and os.getenv("STRUT_RUN_ALL_CASES") != "1":
        pytest.skip("case disabled (set STRUT_RUN_ALL_CASES=1 to run all)")

    assert "schema_version" in data
    assert "metadata" in data
    assert "model" in data
    assert "nodes" in data
    assert "elements" in data
    assert "recorders" in data
