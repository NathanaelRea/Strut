import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_STRUT_CASE_PATH = REPO_ROOT / "scripts" / "run_strut_case.py"


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


run_strut_case = _load_module(
    RUN_STRUT_CASE_PATH, "strut_run_strut_case_direct_input_test_module"
)


def test_run_strut_case_parses_tcl_then_passes_solver_pickle(
    monkeypatch, tmp_path: Path
):
    repo_root = tmp_path / "repo"
    scripts_dir = repo_root / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(run_strut_case, "__file__", str(scripts_dir / "run_strut_case.py"))
    monkeypatch.setattr(run_strut_case.shutil, "which", lambda name: "/usr/bin/uv")

    solver_path = tmp_path / "fake_strut_solver"
    solver_path.write_text("", encoding="utf-8")
    monkeypatch.setenv("STRUT_MOJO_BIN", str(solver_path))

    entry_tcl = repo_root / "docs" / "examples" / "case.tcl"
    entry_tcl.parent.mkdir(parents=True, exist_ok=True)
    entry_tcl.write_text("puts ok\n", encoding="utf-8")
    output_dir = tmp_path / "out"

    calls = []

    def fake_run(cmd, env=None, verbose=False):
        calls.append(cmd)

    monkeypatch.setitem(
        sys.modules,
        "tcl_to_strut",
        SimpleNamespace(
            convert_tcl_to_solver_input=lambda entry, repo, compute_only: {
                "model": {"ndm": 2, "ndf": 3},
                "nodes": [],
                "elements": [],
                "analysis": {"type": "staged", "stages": []},
                "recorders": [] if compute_only else [{"type": "node_displacement"}],
            }
        ),
    )

    monkeypatch.setattr(run_strut_case, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_strut_case.py",
            "--input-tcl",
            str(entry_tcl),
            "--output",
            str(output_dir),
            "--compute-only",
        ],
    )

    run_strut_case.main()

    assert len(calls) == 1
    cmd = calls[0]
    assert cmd[0] == str(solver_path)
    assert "--input-pickle" in cmd
    assert "--input-tcl" not in cmd
    assert "--compute-only" in cmd


def test_run_strut_case_rebuilds_and_runs_profile_binary(
    monkeypatch, tmp_path: Path
):
    repo_root = tmp_path / "repo"
    scripts_dir = repo_root / "scripts"
    src_dir = repo_root / "src" / "mojo"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    src_dir.mkdir(parents=True, exist_ok=True)
    (src_dir / "strut.mojo").write_text("fn main():\n    pass\n", encoding="utf-8")
    monkeypatch.setattr(run_strut_case, "__file__", str(scripts_dir / "run_strut_case.py"))
    monkeypatch.setattr(run_strut_case.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.delenv("STRUT_MOJO_BIN", raising=False)
    monkeypatch.setenv("STRUT_PROFILE", "1")

    input_json = tmp_path / "case.json"
    input_json.write_text(
        '{"model":{"ndm":2,"ndf":3},"nodes":[],"elements":[],"analysis":{"type":"staged","stages":[]},"recorders":[]}\n',
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"

    calls = []

    def fake_run(cmd, env=None, verbose=False):
        calls.append((cmd, dict(env) if env is not None else None, verbose))

    monkeypatch.setattr(run_strut_case, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_strut_case.py",
            "--input",
            str(input_json),
            "--output",
            str(output_dir),
        ],
    )

    run_strut_case.main()

    assert calls[0][0] == [str(repo_root / "scripts" / "build_mojo_solver.sh")]
    assert calls[0][1] is not None
    assert calls[0][1]["STRUT_PROFILE"] == "1"
    assert calls[1][0][0] == str(repo_root / "build" / "strut" / "strut_profile")
