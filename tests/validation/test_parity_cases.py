import os
import subprocess
from pathlib import Path

import pytest


def _case_paths():
    base = Path(__file__).resolve().parent
    return sorted(base.glob("*/**/*.json"))


def _enabled_case_paths():
    for case_path in _case_paths():
        data = case_path.read_text()
        if '"enabled": false' in data and os.getenv("STRUT_RUN_ALL_CASES") != "1":
            continue
        yield case_path


def _selected_case_paths():
    raw = os.getenv("STRUT_PARITY_CASES")
    if not raw:
        return list(_enabled_case_paths())
    items = [item.strip() for item in raw.split(",") if item.strip()]
    selected = []
    for item in items:
        path = Path(item)
        if path.exists():
            selected.append(path)
        else:
            selected.append(Path(__file__).resolve().parent / item / f"{item}.json")
    return selected


@pytest.mark.parametrize("case_path", _selected_case_paths())
def test_parity_case(case_path: Path):
    if os.getenv("STRUT_RUN_PARITY") != "1":
        pytest.skip("parity tests disabled (set STRUT_RUN_PARITY=1)")
    if not case_path.exists():
        pytest.fail(f"missing case JSON: {case_path}")

    try:
        subprocess.check_call(["scripts/run_case.py", str(case_path)])
    except subprocess.CalledProcessError as exc:
        pytest.fail(f"parity failed for {case_path.name} (exit {exc.returncode})")
