from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "ci_benchmark.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "strut_ci_benchmark_test_module", SCRIPT_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Dataclasses look their module up in sys.modules while it loads.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ci_benchmark = _load_module()


def _fake_solver(path: Path, analysis_us: int, value: float = 1.0, fail: bool = False):
    """A stand-in solver: writes one recorder file and a fixed analysis time."""
    body = (
        "import sys\n"
        "from pathlib import Path\n"
        f"if {fail!r}:\n"
        "    print('ABORT: boom')\n"
        "    sys.exit(3)\n"
        "out = Path(sys.argv[sys.argv.index('--output') + 1])\n"
        "out.mkdir(parents=True, exist_ok=True)\n"
        f"(out / 'disp.out').write_text('0.0 {value!r}\\n')\n"
        f"(out / 'analysis_time_us.txt').write_text('{analysis_us}\\n')\n"
    )
    path.write_text(f"#!{sys.executable}\n{body}", encoding="utf-8")
    path.chmod(0o755)
    return path


def _suite(tmp_path: Path) -> Path:
    suite = tmp_path / "suite"
    suite.mkdir()
    input_json = suite / "input.json"
    input_json.write_text("{}", encoding="utf-8")
    cases = [
        {
            "name": "timed_case",
            "input": str(input_json),
            "covers": "test",
            "threads": 1,
            "timed": True,
        },
        {
            "name": "short_case",
            "input": str(input_json),
            "covers": "",
            "threads": 1,
            "timed": False,
        },
    ]
    (suite / ci_benchmark.MANIFEST_NAME).write_text(
        json.dumps({"cases": cases, "skipped": []}), encoding="utf-8"
    )
    return suite


def _compare(monkeypatch, suite: Path, base: Path, cand: Path, *extra: str) -> int:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ci_benchmark.py",
            "compare",
            "--baseline",
            str(base),
            "--candidate",
            str(cand),
            "--suite",
            str(suite),
            "--repeat",
            "2",
            "--json-out",
            str(suite / "results.json"),
            *extra,
        ],
    )
    return ci_benchmark.main()


def _results(suite: Path) -> dict[str, dict]:
    rows = json.loads((suite / "results.json").read_text(encoding="utf-8"))
    return {row["case"]: row for row in rows}


def test_equal_builds_pass(tmp_path, monkeypatch):
    suite = _suite(tmp_path)
    base = _fake_solver(tmp_path / "base", 100_000)
    cand = _fake_solver(tmp_path / "cand", 100_000)

    assert _compare(monkeypatch, suite, base, cand) == 0
    rows = _results(suite)
    assert rows["timed_case"]["status"] == "OK"
    assert rows["timed_case"]["delta_pct"] == pytest.approx(0.0)
    assert rows["short_case"]["candidate_us"] == [100_000]


def test_confirmed_slowdown_fails(tmp_path, monkeypatch, capsys):
    suite = _suite(tmp_path)
    base = _fake_solver(tmp_path / "base", 100_000)
    cand = _fake_solver(tmp_path / "cand", 130_000)

    assert _compare(monkeypatch, suite, base, cand) == 1
    rows = _results(suite)
    assert rows["timed_case"]["regressed"] is True
    # Untimed cases never gate.
    assert rows["short_case"]["regressed"] is False
    assert "+30.0%" in capsys.readouterr().out


def test_slowdown_below_gate_floor_passes(tmp_path, monkeypatch):
    suite = _suite(tmp_path)
    base = _fake_solver(tmp_path / "base", 1_000)
    cand = _fake_solver(tmp_path / "cand", 2_000)

    assert _compare(monkeypatch, suite, base, cand) == 0


def test_candidate_crash_fails(tmp_path, monkeypatch):
    suite = _suite(tmp_path)
    base = _fake_solver(tmp_path / "base", 100_000)
    cand = _fake_solver(tmp_path / "cand", 100_000, fail=True)

    assert _compare(monkeypatch, suite, base, cand) == 1
    rows = _results(suite)
    assert rows["timed_case"]["status"] == "CAND-FAIL"
    assert "ABORT: boom" in rows["timed_case"]["detail"]


def test_output_difference_fails_only_when_asked(tmp_path, monkeypatch):
    suite = _suite(tmp_path)
    base = _fake_solver(tmp_path / "base", 100_000, value=1.0)
    cand = _fake_solver(tmp_path / "cand", 100_000, value=1.5)

    assert _compare(monkeypatch, suite, base, cand) == 0
    assert _results(suite)["timed_case"]["status"] == "DIFF"
    assert _compare(monkeypatch, suite, base, cand, "--fail-on-diff") == 1


def test_new_case_the_baseline_cannot_run_passes(tmp_path, monkeypatch):
    suite = _suite(tmp_path)
    base = _fake_solver(tmp_path / "base", 100_000, fail=True)
    cand = _fake_solver(tmp_path / "cand", 100_000)

    assert _compare(monkeypatch, suite, base, cand) == 0
    assert _results(suite)["timed_case"]["status"] == "NEW"


def test_missing_pinned_records_get_synthetic_stand_ins(tmp_path: Path):
    (tmp_path / "GM").mkdir()
    (tmp_path / "GM" / "present.at2").write_text("real\n", encoding="ascii")
    lock = {
        "files": [
            {"path": "GM/present.at2"},
            {"path": "GM/H-e12140.dt2"},
            {"path": "Ex/BM68elc.acc"},
            {"path": "Ex/tabasFN.txt"},
            {"path": "Ex/A10000.tcl"},
            {"path": "Ex/Model.tcl"},
            {"path": "Ex/run.m"},
        ]
    }
    lock_path = tmp_path / "reference-data.lock.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    written = ci_benchmark.write_missing_synthetic_records(lock_path)
    names = sorted(path.name for path in written)
    assert names == ["A10000.tcl", "BM68elc.acc", "H-e12140.dt2", "tabasFN.txt"]
    assert (tmp_path / "GM" / "present.at2").read_text(encoding="ascii") == "real\n"
    assert "NPTS=" in (tmp_path / "GM" / "H-e12140.dt2").read_text(encoding="ascii")
    values = (tmp_path / "Ex" / "tabasFN.txt").read_text(encoding="ascii").split()
    assert len(values) == 3000 and all(float(v) == float(v) for v in values)
    # Seeded: a second run writes the same bytes.
    first = (tmp_path / "Ex" / "BM68elc.acc").read_bytes()
    (tmp_path / "Ex" / "BM68elc.acc").unlink()
    ci_benchmark.write_missing_synthetic_records(lock_path)
    assert (tmp_path / "Ex" / "BM68elc.acc").read_bytes() == first
