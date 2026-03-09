import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PLOT_BENCHMARKS_PATH = REPO_ROOT / "scripts" / "plot_markdown_benchmarks.py"
SCRIPTS_DIR = str(REPO_ROOT / "scripts")


def _load_plot_benchmarks_module():
    module_name = "strut_plot_markdown_benchmarks_test_module"
    if SCRIPTS_DIR not in sys.path:
        sys.path.insert(0, SCRIPTS_DIR)
    spec = importlib.util.spec_from_file_location(module_name, PLOT_BENCHMARKS_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


plot_benchmarks = _load_plot_benchmarks_module()


def test_write_opensees_examples_markdown_emits_per_example_charts(tmp_path: Path):
    summary_path = tmp_path / "summary.json"
    output_path = tmp_path / "benchmark-large-cases.md"
    summary_path.write_text(
        json.dumps(
            {
                "generated_at": "2026-03-09T12:00:00Z",
                "cases": [
                    {
                        "name": "bench_ex5_frame2d_analyze_static_push",
                        "size": "large",
                        "opensees": {"analysis_us": 4_000_000},
                        "strut": {"analysis_us": 2_500_000},
                    },
                    {
                        "name": "opensees_example_ex50_frame2d_analyze_static_cycle",
                        "size": "medium",
                        "opensees": {"analysis_us": 3_000_000},
                        "strut": {"analysis_us": 1_500_000},
                    },
                    {
                        "name": "opensees_example_ex50_frame2d_analyze_static_push",
                        "size": "medium",
                        "opensees": {"analysis_us": 3_500_000},
                        "strut": {"analysis_us": 2_000_000},
                    },
                    {
                        "name": "cantilever_medium",
                        "size": "medium",
                        "opensees": {"analysis_us": 25_000},
                        "strut": {"analysis_us": 15_000},
                    },
                    {
                        "name": "opensees_example_ex60_genericframe2d_analyze_dynamic_eq_uniform",
                        "size": "small",
                        "opensees": {"analysis_us": 7.500},
                        "strut": {"analysis_us": 5.250},
                    },
                    {
                        "name": "opensees_example_ex70_microframe_run_fast",
                        "size": "small",
                        "opensees": {"analysis_us": 0.007},
                        "strut": {"analysis_us": 0.004},
                    },
                    {
                        "name": "opensees_example_ex80_midframe_run_balanced",
                        "size": "small",
                        "opensees": {"analysis_us": 2_500.0},
                        "strut": {"analysis_us": 1_750.0},
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = plot_benchmarks.write_opensees_examples_markdown(
        results_path=summary_path,
        output_path=output_path,
    )

    assert result == output_path
    text = output_path.read_text(encoding="utf-8")
    assert "# OpenSees Example Benchmarks" in text
    assert "plotColorPalette" in text
    assert 'title "Benchmark: Example 50"' in text
    assert 'title "Benchmark: Example 60"' in text
    assert 'title "Benchmark: Example 70"' in text
    assert 'title "Benchmark: Example 80"' in text
    assert "- Source:" not in text
    assert "- Generated at:" not in text
    assert "- Example groups:" not in text
    assert "## ex50" in text
    assert "## ex60" in text
    assert "## ex70" in text
    assert "## ex80" in text
    assert '"1O", "1S", "2O", "2S"' in text
    assert '"1O", "1S"' in text
    assert '"cantilever_medium"' not in text
    assert "- Unit:" not in text
    assert "- Cases:" not in text
    assert 'y-axis "Analysis time (s)" 0 --> 3.851' in text
    assert 'y-axis "Analysis time (us)" 0 --> 8.250' in text
    assert 'y-axis "Analysis time (ms)" 0 --> 2.750' in text
    assert 'y-axis "Analysis time (ns)" 0 --> 7.701' in text
    assert 'bar "mask" [0.0, 0.0, 0.0, 0.0]' in text
    assert 'bar "OS" [3.000, 0.0, 3.500, 0.0]' in text
    assert 'bar "STR" [0.0, 1.500, 0.0, 2.000]' in text
    assert 'bar "OS" [7.500, 0.0]' in text
    assert 'bar "STR" [0.0, 5.250]' in text
    assert 'bar "OS" [2.500, 0.0]' in text
    assert 'bar "STR" [0.0, 1.750]' in text
    assert 'bar "OS" [7.000, 0.0]' in text
    assert 'bar "STR" [0.0, 4.000]' in text
    assert "| # | Label | OpenSees (s) | Strut (s) |" in text
    assert "| # | Label | OpenSees (us) | Strut (us) |" in text
    assert "| # | Label | OpenSees (ms) | Strut (ms) |" in text
    assert "| # | Label | OpenSees (ns) | Strut (ns) |" in text
    assert "| 1 | `cycle` | 3.000 | 1.500 |" in text
    assert "| 2 | `push` | 3.500 | 2.000 |" in text
    assert "| 1 | `ex60_genericframe2d_analyze_dynamic_eq_uniform` | 7.500 | 5.250 |" in text
    assert "| 1 | `ex70_microframe_run_fast` | 7.000 | 4.000 |" in text
    assert "| 1 | `ex80_midframe_run_balanced` | 2.500 | 1.750 |" in text
