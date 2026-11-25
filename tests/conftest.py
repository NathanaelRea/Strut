from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "setup_reference_data.py"


def _load_setup_reference_data():
    spec = importlib.util.spec_from_file_location("strut_setup_reference_data", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pytest_sessionstart(session: pytest.Session) -> None:
    # Fail fast with the fix instead of scattered "file not found" errors when
    # the pinned ground motions / mega-tall model are not installed here.
    setup = _load_setup_reference_data()
    problems = setup.collect_problems(setup.load_manifest(), include_agent_reference=False)
    if problems:
        listed = "\n  ".join(problems[:10])
        more = f"\n  ... and {len(problems) - 10} more" if len(problems) > 10 else ""
        raise pytest.UsageError(
            "pinned test fixtures are missing; run:\n"
            "  uv run scripts/setup_reference_data.py install --fixtures-only\n"
            f"  {listed}{more}"
        )
