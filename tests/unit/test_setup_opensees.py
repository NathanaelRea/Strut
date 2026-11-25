from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "setup_opensees.py"
ENV_SH = REPO_ROOT / "scripts" / "opensees_env.sh"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "strut_setup_opensees_test_module", SCRIPT_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


setup_opensees = _load_module()


def _bash(snippet: str, env: dict[str, str]) -> str:
    result = subprocess.run(
        ["bash", "-c", f'source "{ENV_SH}"; {snippet}'],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _env(**overrides: str) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("STRUT_CACHE_DIR", "XDG_CACHE_HOME")
    }
    env.update(overrides)
    return env


def test_lockfile_is_explicit_and_hash_pinned():
    lines = setup_opensees.LOCK_FILE.read_text(encoding="utf-8").splitlines()
    assert "@EXPLICIT" in lines
    urls = lines[lines.index("@EXPLICIT") + 1 :]
    assert urls
    for url in urls:
        assert url.startswith("https://conda.anaconda.org/conda-forge/")
        _, sha = url.split("#sha256:")
        assert len(sha) == 64
    assert any("/opensees-" in url for url in urls)


@pytest.mark.parametrize(
    "overrides",
    [
        {"STRUT_CACHE_DIR": "/tmp/strut-cache-test"},
        {"XDG_CACHE_HOME": "/tmp/xdg-cache-test"},
        {"HOME": "/tmp/home-test"},
    ],
)
def test_bash_and_python_agree_on_prefix(monkeypatch, overrides):
    env = _env(**overrides)
    for key in ("STRUT_CACHE_DIR", "XDG_CACHE_HOME"):
        monkeypatch.delenv(key, raising=False)
    for key, value in overrides.items():
        monkeypatch.setenv(key, value)
    python_prefix = str(setup_opensees.env_prefix())
    bash_prefix = _bash("strut_opensees_prefix", env)
    assert bash_prefix == python_prefix
    assert python_prefix.endswith(f"/opensees/{setup_opensees.lock_hash()}")


def test_resolver_prefers_override_then_locked_install(tmp_path):
    env = _env(STRUT_CACHE_DIR=str(tmp_path))
    assert (
        _bash('strut_resolve_opensees_binary OpenSees "/custom/OpenSees"', env)
        == "/custom/OpenSees"
    )
    local_build = REPO_ROOT / ".build" / "opensees-linux" / "OpenSees"
    resolved = _bash("strut_resolve_opensees_binary OpenSees", env)
    if os.access(local_build, os.X_OK):
        assert resolved == str(local_build)
    else:
        prefix = _bash("strut_opensees_prefix", env)
        assert resolved == f"{prefix}/bin/OpenSees"


def test_ready_prefix_requires_marker(tmp_path):
    env = _env(STRUT_CACHE_DIR=str(tmp_path))
    prefix = Path(_bash("strut_opensees_prefix", env))
    check = "strut_opensees_ready_prefix || echo missing"
    (prefix / "bin").mkdir(parents=True)
    assert _bash(check, env) == "missing"
    (prefix / setup_opensees.READY_MARKER).write_text("{}\n", encoding="utf-8")
    assert _bash(check, env) == str(prefix)
