import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PLOT_BENCHMARKS_PATH = REPO_ROOT / "scripts" / "plot_benchmarks.py"
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _load_plot_benchmarks_module():
    module_name = "strut_plot_benchmarks_test_module"
    sys.path.insert(0, str(SCRIPTS_DIR))
    spec = importlib.util.spec_from_file_location(module_name, PLOT_BENCHMARKS_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


plot_benchmarks = _load_plot_benchmarks_module()


def test_write_plots_pdf_writes_output_for_recent_case_summary(tmp_path: Path):
    results_path = tmp_path / "summary.json"
    archive_dir = tmp_path / "archive"
    output_path = tmp_path / "plots.pdf"
    results_path.write_text(
        (
            '{"generated_at":"2026-03-03T00:00:00Z","cases":['
            '{"name":"elastic_beam_cantilever","dofs":6,'
            '"opensees":{"analysis_us":20.0},"strut":{"analysis_us":10.0}}'
            "]}\n"
        ),
        encoding="utf-8",
    )

    result = plot_benchmarks.write_plots_pdf(
        results_path=results_path,
        archive_dir=archive_dir,
        output_path=output_path,
    )

    assert result == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_filter_enabled_cases_excludes_disabled_direct_tcl_case(
    monkeypatch, tmp_path: Path
):
    validation_root = tmp_path / "tests" / "validation" / "disabled_direct_case"
    validation_root.mkdir(parents=True, exist_ok=True)
    (validation_root / "direct_tcl_case.json").write_text(
        json.dumps(
            {
                "name": "disabled_direct_case",
                "entry_tcl": "examples/case.tcl",
                "enabled": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(plot_benchmarks, "_repo_root", lambda: tmp_path)

    filtered = plot_benchmarks._filter_enabled_cases(
        [
            {
                "name": "disabled_direct_case",
                "strut": {"analysis_us": 10.0},
            },
            {
                "name": "enabled_case",
                "strut": {"analysis_us": 5.0},
            },
        ]
    )

    assert [case["name"] for case in filtered] == ["enabled_case"]


def test_collect_archive_trend_skips_currently_disabled_cases(
    monkeypatch, tmp_path: Path
):
    validation_root = tmp_path / "tests" / "validation" / "disabled_direct_case"
    validation_root.mkdir(parents=True, exist_ok=True)
    (validation_root / "direct_tcl_case.json").write_text(
        json.dumps(
            {
                "name": "disabled_direct_case",
                "entry_tcl": "examples/case.tcl",
                "enabled": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    (archive_dir / "20260303T000000Z-summary.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-03-03T00:00:00Z",
                "cases": [
                    {
                        "name": "disabled_direct_case",
                        "strut": {"analysis_us": 10.0},
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(plot_benchmarks, "_repo_root", lambda: tmp_path)

    timestamps, means, stds = plot_benchmarks.collect_archive_trend(
        archive_dir,
        size_filter=None,
        medium_threshold=300,
        large_threshold=1000,
    )

    assert len(timestamps) == 1
    assert means["strut"][0] != means["strut"][0]
    assert means["opensees"][0] != means["opensees"][0]
    assert stds["strut"][0] != stds["strut"][0]


def test_case_size_bucket_defaults_to_small_and_promotes_from_timings():
    assert (
        plot_benchmarks._case_size_bucket(
            {
                "name": "fast_case",
                "opensees": {"analysis_us": 1000.0},
                "strut": {"analysis_us": 999.0},
            }
        )
        == "small"
    )
    assert (
        plot_benchmarks._case_size_bucket(
            {
                "name": "medium_case",
                "opensees": {"analysis_us": 1000.1},
                "strut": {"analysis_us": 400.0},
            }
        )
        == "medium"
    )
    assert (
        plot_benchmarks._case_size_bucket(
            {
                "name": "large_case",
                "opensees": {"analysis_us": 500.0},
                "strut": {"analysis_us": 1000000.1},
            }
        )
        == "large"
    )


def test_case_size_bucket_preserves_explicit_override():
    assert (
        plot_benchmarks._case_size_bucket(
            {
                "name": "forced_medium",
                "size": "medium",
                "opensees": {"analysis_us": 10.0},
                "strut": {"analysis_us": 10.0},
            }
        )
        == "medium"
    )
