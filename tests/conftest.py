from __future__ import annotations

import functools
import importlib.util
from pathlib import Path, PurePosixPath

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "setup_reference_data.py"
_INSTALL_COMMAND = "uv run scripts/setup_reference_data.py install --fixtures-only"


@functools.cache
def _setup_reference_data():
    spec = importlib.util.spec_from_file_location("strut_setup_reference_data", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _overlaps(entry_path: str, prefixes: tuple[str, ...]) -> bool:
    if not prefixes:
        return True
    path = PurePosixPath(entry_path)
    for raw in prefixes:
        prefix = PurePosixPath(raw)
        # A file under the prefix, or a tree that contains the prefix.
        if path.is_relative_to(prefix) or prefix.is_relative_to(path):
            return True
    return False


@functools.cache
def missing_pinned_fixtures(prefixes: tuple[str, ...] = ()) -> tuple[str, ...]:
    """Problems with pinned downloads under ``tests/fixtures/opensees/<prefix>``.

    Empty ``prefixes`` checks every pinned file and tree.
    """
    setup = _setup_reference_data()
    manifest = setup.load_manifest()
    problems = []
    for entry in manifest["files"]:
        if _overlaps(entry["path"], prefixes):
            status = setup.file_status(entry)
            if status:
                problems.append(f"tests/fixtures/opensees/{entry['path']}: {status}")
    for entry in manifest["trees"]:
        if _overlaps(entry["path"], prefixes):
            status = setup.tree_status(entry)
            if status:
                problems.append(f"tests/fixtures/opensees/{entry['path']}/: {status}")
    return tuple(problems)


def pytest_sessionstart(session: pytest.Session) -> None:
    # The mega-tall benchmark model is generated, never downloaded. Building it
    # takes about 2 s and is skipped when the stamp says it is current.
    _setup_reference_data().megatall_bundle().ensure_bundle()


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "requires_pinned_fixtures(*paths): skip unless the pinned downloads under "
        "tests/fixtures/opensees/<path> (all of them if no path is given) are "
        f"installed; install them with `{_INSTALL_COMMAND}`",
    )


def pytest_runtest_setup(item: pytest.Item) -> None:
    for marker in item.iter_markers("requires_pinned_fixtures"):
        problems = missing_pinned_fixtures(tuple(str(arg) for arg in marker.args))
        if problems:
            listed = "; ".join(problems[:3])
            more = f"; ... and {len(problems) - 3} more" if len(problems) > 3 else ""
            pytest.skip(
                f"pinned test fixtures not installed ({listed}{more}); "
                f"run: {_INSTALL_COMMAND}"
            )
