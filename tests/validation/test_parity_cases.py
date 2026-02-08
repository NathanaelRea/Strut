import subprocess
from pathlib import Path
import os

import pytest


def _case_paths():
    base = Path(__file__).resolve().parent
    return sorted(base.glob("*/**/*.json"))


def _enabled_case_paths():
    return _case_paths()


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
    if not case_path.exists():
        pytest.fail(f"missing case JSON: {case_path}")

    try:
        subprocess.check_call(["scripts/run_case.py", str(case_path)])
    except subprocess.CalledProcessError as exc:
        pytest.fail(f"parity failed for {case_path.name} (exit {exc.returncode})")
