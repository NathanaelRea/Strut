import json
import os
import subprocess
from pathlib import Path

import pytest


def _repo_root():
    return Path(__file__).resolve().parents[2]


def _validation_root():
    return Path(__file__).resolve().parent


def _load_case_data(path: Path):
    return json.loads(path.read_text())


def _case_paths():
    base = _validation_root()
    paths = []
    for path in sorted(base.glob("*/*.json")):
        data = _load_case_data(path)
        if not isinstance(data, dict):
            continue
        if "model" not in data or "nodes" not in data:
            continue
        paths.append(path)
    return paths


def _direct_tcl_manifest_paths():
    return sorted(_validation_root().glob("*/direct_tcl_case.json"))


def _include_direct_tcl_manifest_by_default(data: dict) -> bool:
    if not data.get("enabled", True):
        return False
    status = str(data.get("status", "")).strip().lower()
    return status not in {"todo", "benchmark"}


def _resolve_named_case_path(name: str):
    case_dir = _validation_root() / name
    case_json = case_dir / f"{name}.json"
    if case_json.exists():
        return case_json
    direct_manifest = case_dir / "direct_tcl_case.json"
    if direct_manifest.exists():
        return direct_manifest
    return case_json


def _enabled_case_paths():
    paths = []
    include_disabled = os.getenv("STRUT_RUN_ALL_CASES") == "1"
    for path in _case_paths():
        data = _load_case_data(path)
        if include_disabled:
            paths.append(path)
            continue
        if data.get("enabled", True):
            paths.append(path)
    for manifest in _direct_tcl_manifest_paths():
        data = _load_case_data(manifest)
        if include_disabled or _include_direct_tcl_manifest_by_default(data):
            paths.append(manifest)
    return paths


def _selected_case_paths():
    raw = os.getenv("STRUT_PARITY_CASES")
    if not raw:
        return list(_enabled_case_paths())
    repo_root = _repo_root()
    items = [item.strip() for item in raw.split(",") if item.strip()]
    selected = []
    for item in items:
        path = Path(item)
        if path.exists():
            selected.append(path)
        elif item.endswith(".tcl"):
            selected.append((repo_root / item).resolve())
        else:
            selected.append(_resolve_named_case_path(item))
    return selected


def _fixture_bundle(case_path: Path):
    """The fixture bundle a direct-Tcl case mirrors, relative to the fixture root.

    The harness copies the entry script's parent directory, so every pinned
    download under it (ground motions, the mega-tall model) is an input.
    """
    if case_path.name != "direct_tcl_case.json" or not case_path.exists():
        return None
    raw_entry = _load_case_data(case_path).get("entry_tcl")
    if not isinstance(raw_entry, str) or not raw_entry:
        return None
    entry = Path(raw_entry)
    if not entry.is_absolute():
        entry = _repo_root() / entry
    fixture_root = (_repo_root() / "tests" / "fixtures" / "opensees").resolve()
    try:
        return entry.resolve().parent.parent.relative_to(fixture_root).as_posix()
    except ValueError:
        return None


def _case_params():
    params = []
    for case_path in _selected_case_paths():
        bundle = _fixture_bundle(case_path)
        marks = [] if bundle is None else [pytest.mark.requires_pinned_fixtures(bundle)]
        params.append(pytest.param(case_path, marks=marks))
    return params


@pytest.mark.parametrize("case_path", _case_params())
def test_parity_case(case_path: Path):
    if not case_path.exists():
        pytest.fail(f"missing parity case: {case_path}")

    try:
        subprocess.check_call(["scripts/run_case.py", str(case_path)])
    except subprocess.CalledProcessError as exc:
        pytest.fail(f"parity failed for {case_path.name} (exit {exc.returncode})")
