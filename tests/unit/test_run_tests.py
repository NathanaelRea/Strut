import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_TESTS_PATH = REPO_ROOT / "run_tests.py"


def _load_run_tests_module():
    module_name = "strut_run_tests_test_module"
    spec = importlib.util.spec_from_file_location(module_name, RUN_TESTS_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


run_tests = _load_run_tests_module()


def test_main_exports_solver_binary_after_build(monkeypatch):
    calls = []

    def fake_run(cmd, env=None, verbose=False):
        calls.append((cmd, dict(env) if env is not None else None, verbose))

    monkeypatch.setattr(run_tests, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["run_tests.py"])

    assert run_tests.main() == 0

    solver_path = str(REPO_ROOT / "build" / "strut" / "strut")
    assert calls[0][0] == [
        "uv",
        "run",
        "python",
        str(REPO_ROOT / "scripts" / "build_mojo_solver.py"),
    ]
    assert calls[1][1] is not None
    assert calls[1][1]["STRUT_MOJO_BIN"] == solver_path
    assert calls[2][1] is not None
    assert calls[2][1]["STRUT_MOJO_BIN"] == solver_path


def test_main_skip_build_exports_existing_solver_binary(monkeypatch, tmp_path: Path):
    solver_path = tmp_path / "build" / "strut" / "strut"
    solver_path.parent.mkdir(parents=True, exist_ok=True)
    solver_path.write_text("", encoding="utf-8")
    calls = []

    def fake_run(cmd, env=None, verbose=False):
        calls.append((cmd, dict(env) if env is not None else None, verbose))

    monkeypatch.setattr(run_tests, "run", fake_run)
    monkeypatch.setattr(run_tests.Path, "resolve", lambda self: tmp_path / self.name)
    monkeypatch.setattr(sys, "argv", ["run_tests.py", "--skip-build"])

    assert run_tests.main() == 0

    assert len(calls) == 2
    assert calls[0][1] is not None
    assert calls[0][1]["STRUT_MOJO_BIN"] == str(solver_path)
    assert calls[1][1] is not None
    assert calls[1][1]["STRUT_MOJO_BIN"] == str(solver_path)
