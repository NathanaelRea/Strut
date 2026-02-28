import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
COMPARE_BENCHMARKS_PATH = REPO_ROOT / "scripts" / "compare_benchmarks.py"


def _load_compare_benchmarks_module():
    module_name = "strut_compare_benchmarks_test_module"
    spec = importlib.util.spec_from_file_location(module_name, COMPARE_BENCHMARKS_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


compare_benchmarks = _load_compare_benchmarks_module()


def _summary(case_rows, cpu_model="CPU-A", mojo_version="Mojo X", engine="strut"):
    cases = []
    for name, analysis_us in case_rows.items():
        cases.append({"name": name, engine: {"analysis_us": analysis_us}})
    return {
        "cases": cases,
        "metadata": {
            "platform": {
                "system": "Linux",
                "release": "6.x",
                "machine": "x86_64",
                "cpu_model": cpu_model,
            },
            "runner": {"engine": engine, "batch_mode": True},
            "build": {"mojo_version": mojo_version},
        },
    }


def test_compare_summaries_ignores_unshared_cases():
    baseline = _summary({"case_a": 100.0, "case_only_baseline": 50.0})
    candidate = _summary({"case_a": 90.0, "case_only_candidate": 60.0})

    rows, warnings = compare_benchmarks.compare_summaries(
        baseline, candidate, engine="strut"
    )

    assert [(row.name, row.delta_pct) for row in rows] == [("case_a", -10.0)]
    assert warnings == [
        "candidate-only cases ignored: case_only_candidate",
        "baseline-only cases ignored: case_only_baseline",
    ]


def test_metadata_mismatch_reports_differences():
    baseline = _summary({"case_a": 100.0}, cpu_model="CPU-A", mojo_version="Mojo A")
    candidate = _summary({"case_a": 95.0}, cpu_model="CPU-B", mojo_version="Mojo B")

    mismatches = compare_benchmarks._metadata_mismatch(baseline, candidate)

    assert "platform.cpu_model: baseline='CPU-A' candidate='CPU-B'" in mismatches
    assert "build.mojo_version: baseline='Mojo A' candidate='Mojo B'" in mismatches


def test_main_fails_when_regression_exceeds_threshold(
    tmp_path: Path, monkeypatch, capsys
):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text(
        json.dumps(_summary({"case_a": 100.0})) + "\n", encoding="utf-8"
    )
    candidate.write_text(
        json.dumps(_summary({"case_a": 108.0})) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_benchmarks.py",
            str(baseline),
            str(candidate),
            "--max-regression-pct",
            "5",
        ],
    )

    assert compare_benchmarks.main() == 1
    captured = capsys.readouterr()
    assert (
        "error: case_a: regression 8.00% exceeds 5.00% and absolute delta 8 us exceeds 0 us"
        in captured.out
    )


def test_main_ignores_small_regression_below_absolute_floor(
    tmp_path: Path, monkeypatch, capsys
):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text(
        json.dumps(_summary({"case_a": 12.0})) + "\n", encoding="utf-8"
    )
    candidate.write_text(
        json.dumps(_summary({"case_a": 13.0})) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_benchmarks.py",
            str(baseline),
            str(candidate),
            "--max-regression-pct",
            "5",
            "--min-regression-us",
            "5",
        ],
    )

    assert compare_benchmarks.main() == 0
    captured = capsys.readouterr()
    assert "warning: ignoring small regression for case_a: 1 us <= 5 us" in captured.out


def test_main_fails_when_required_improvement_is_not_met(
    tmp_path: Path, monkeypatch, capsys
):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text(
        json.dumps(_summary({"case_a": 100.0})) + "\n", encoding="utf-8"
    )
    candidate.write_text(
        json.dumps(_summary({"case_a": 95.0})) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_benchmarks.py",
            str(baseline),
            str(candidate),
            "--require-improvement",
            "case_a=10",
        ],
    )

    assert compare_benchmarks.main() == 1
    captured = capsys.readouterr()
    assert "error: case_a: improvement 5.00% is below required 10.00%" in captured.out


def test_main_passes_when_improvement_target_is_met(tmp_path: Path, monkeypatch):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text(
        json.dumps(_summary({"case_a": 100.0, "case_b": 200.0})) + "\n",
        encoding="utf-8",
    )
    candidate.write_text(
        json.dumps(_summary({"case_a": 85.0, "case_b": 204.0})) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_benchmarks.py",
            str(baseline),
            str(candidate),
            "--max-regression-pct",
            "5",
            "--require-improvement",
            "case_a=10",
        ],
    )

    assert compare_benchmarks.main() == 0
