import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PLOT_BENCHMARKS_PATH = REPO_ROOT / "scripts" / "plot_markdown_benchmarks.py"
PLOT_PDF_BENCHMARKS_PATH = REPO_ROOT / "scripts" / "plot_benchmarks.py"
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


def _load_plot_pdf_benchmarks_module():
    module_name = "strut_plot_benchmarks_test_module"
    if SCRIPTS_DIR not in sys.path:
        sys.path.insert(0, SCRIPTS_DIR)
    spec = importlib.util.spec_from_file_location(module_name, PLOT_PDF_BENCHMARKS_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


plot_pdf_benchmarks = _load_plot_pdf_benchmarks_module()


def test_write_opensees_examples_markdown_groups_examples_by_opensees_time(
    tmp_path: Path,
):
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
                        "openseesmp": {"analysis_us": 3_250_000},
                        "strut": {"analysis_us": 2_500_000},
                    },
                    {
                        "name": "opensees_example_ex50_frame2d_analyze_static_cycle",
                        "size": "medium",
                        "opensees": {"analysis_us": 3_000_000},
                        "openseesmp": {"analysis_us": 2_250_000},
                        "strut": {"analysis_us": 1_500_000},
                    },
                    {
                        "name": "opensees_example_ex50_frame2d_analyze_static_push",
                        "size": "medium",
                        "opensees": {"analysis_us": 3_500_000},
                        "openseesmp": {"analysis_us": 2_750_000},
                        "strut": {"analysis_us": 2_000_000},
                    },
                    {
                        "name": "cantilever_medium",
                        "size": "medium",
                        "opensees": {"analysis_us": 25_000},
                        "openseesmp": {"analysis_us": 20_000},
                        "strut": {"analysis_us": 15_000},
                    },
                    {
                        "name": "opensees_example_ex60_genericframe2d_analyze_dynamic_eq_uniform",
                        "size": "small",
                        "opensees": {"analysis_us": 7.500},
                        "openseesmp": {"analysis_us": 6.125},
                        "strut": {"analysis_us": 5.250},
                    },
                    {
                        "name": "opensees_example_ex70_microframe_run_fast",
                        "size": "small",
                        "opensees": {"analysis_us": 0.007},
                        "openseesmp": {"analysis_us": 0.005},
                        "strut": {"analysis_us": 0.004},
                    },
                    {
                        "name": "opensees_example_ex80_midframe_run_balanced",
                        "size": "small",
                        "opensees": {"analysis_us": 2_500.0},
                        "openseesmp": {"analysis_us": 2_000.0},
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
    assert "# Benchmarks: Strut and OpenSees" in text
    assert "cantilever_medium" in text
    assert "bench_ex5" in text
    assert "7 of 7 cases have timings for both engines." in text
    assert "| Validation suite, 7 comparable cases |" in text
    assert "#### Over one second" in text
    assert "#### 10 ms to one second" in text
    assert "#### Under 10 ms" in text
    assert 'title "OpenSees examples manual: over one second"' in text
    assert 'title "OpenSees examples manual: under 10 ms"' not in text
    hidden = plot_benchmarks.HIDDEN_LABEL_CHAR
    assert f'x-axis ["{hidden}", "1", "{hidden * 2}", "{hidden * 3}", "2"]' in text
    assert 'y-axis "Analysis time (s)" 0 --> 3.85' in text
    assert 'bar "opensees" [3.500, 0, 0, 3.000, 0]' in text
    assert 'bar "strut" [0, 2.000, 0, 0, 1.500]' in text
    assert "bar [0, 0, 0, 0, 0]" in text
    assert 'bar "openseesmp"' not in text
    assert "| # | Case | opensees | strut | Strut is |" in text
    assert (
        "| 1 | `ex50_frame2d_analyze_static_push` | 3.50 s | 2.00 s | 1.8x faster |"
        in text
    )
    assert (
        "| 2 | `ex50_frame2d_analyze_static_cycle` | 3.00 s | 1.50 s | 2.0x faster |"
        in text
    )
    assert (
        "| 1 | `ex80_midframe_run_balanced` | 2.50 ms | 1.75 ms | 1.4x faster |" in text
    )
    assert (
        "| 2 | `ex60_genericframe2d_analyze_dynamic_eq_uniform` | 8 µs | 5 µs | 1.4x faster |"
        in text
    )


def test_write_opensees_examples_markdown_includes_mp_only_when_requested(
    tmp_path: Path,
):
    summary_path = tmp_path / "summary.json"
    output_path = tmp_path / "benchmark-large-cases.md"
    summary_path.write_text(
        json.dumps(
            {
                "generated_at": "2026-03-09T12:00:00Z",
                "cases": [
                    {
                        "name": "opensees_example_ex50_frame2d_analyze_static_cycle",
                        "size": "medium",
                        "opensees": {"analysis_us": 3_000_000},
                        "openseesmp": {"analysis_us": 2_250_000},
                        "strut": {"analysis_us": 1_500_000},
                    },
                    {
                        "name": "opensees_example_ex50_frame2d_analyze_static_push",
                        "size": "medium",
                        "opensees": {"analysis_us": 3_500_000},
                        "openseesmp": {"analysis_us": 2_750_000},
                        "strut": {"analysis_us": 2_000_000},
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    plot_benchmarks.write_opensees_examples_markdown(
        results_path=summary_path,
        output_path=output_path,
        include_mp=True,
    )

    text = output_path.read_text(encoding="utf-8")
    assert 'bar "opensees" [3.500, 0, 0, 0, 3.000, 0, 0]' in text
    assert 'bar "openseesmp" [0, 2.750, 0, 0, 0, 2.250, 0]' in text
    assert 'bar "strut" [0, 0, 2.000, 0, 0, 0, 1.500]' in text
    assert "| # | Case | opensees | openseesmp | strut | Strut is |" in text
    assert (
        "| 2 | `ex50_frame2d_analyze_static_cycle` | 3.00 s | 2.25 s | 1.50 s | 2.0x faster |"
        in text
    )


def test_write_plots_pdf_warns_when_summary_is_missing_openseesmp(
    tmp_path: Path, capsys
):
    results_path = tmp_path / "summary.json"
    archive_dir = tmp_path / "archive"
    output_path = tmp_path / "plots.pdf"
    results_path.write_text(
        json.dumps(
            {
                "generated_at": "2026-03-09T12:00:00Z",
                "cases": [
                    {
                        "name": "simple_case",
                        "size": "small",
                        "opensees": {"analysis_us": 12_000},
                        "strut": {"analysis_us": 8_000},
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = plot_pdf_benchmarks.write_plots_pdf(
        results_path=results_path,
        archive_dir=archive_dir,
        output_path=output_path,
        group_by="none",
    )

    captured = capsys.readouterr()
    assert result == output_path
    assert output_path.exists()
    assert (
        f"warning: no openseesmp analysis timings in {results_path}; "
        "summary.json is missing those engine results"
    ) in captured.out


def test_write_opensees_examples_markdown_adds_single_thread_bar_when_measured(
    tmp_path: Path,
):
    summary_path = tmp_path / "summary.json"
    output_path = tmp_path / "benchmarks.md"
    summary_path.write_text(
        json.dumps(
            {
                "generated_at": "2026-03-09T12:00:00Z",
                "metadata": {
                    "platform": {"cpu_model": "Test CPU", "physical_cores": 8},
                    "runner": {
                        "repeat": 3,
                        "strut_threads": "performance cores",
                        "strut_single_thread_pass": True,
                        "opensees_thread_env": {
                            "OPENBLAS_NUM_THREADS": "1",
                            "OMP_NUM_THREADS": "1",
                        },
                    },
                    "build": {"opensees_version": "3.7.1"},
                },
                "cases": [
                    {
                        "name": "opensees_example_ex50_frame2d_analyze_static_cycle",
                        "opensees": {"analysis_us": 3_000_000},
                        "strut_single_thread": {"analysis_us": 2_000_000},
                        "strut": {"analysis_us": 1_500_000},
                    },
                    {
                        "name": "opensees_example_ex50_frame2d_analyze_static_push",
                        "opensees": {"analysis_us": 3_500_000},
                        "strut_single_thread": {"analysis_us": 2_500_000},
                        "strut": {"analysis_us": 2_000_000},
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    plot_benchmarks.write_opensees_examples_markdown(
        results_path=summary_path,
        output_path=output_path,
    )

    text = output_path.read_text(encoding="utf-8")
    assert "Measured 2026-03-09 on Test CPU (8 cores), OpenSees 3.7.1." in text
    assert "OpenSees ran with `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1`." in text
    assert "Median of 3 runs per engine." in text
    assert (
        "`opensees` is OpenSees, `strut` is Strut on one thread, and "
        "`strut-8` is Strut on 8 threads."
    ) in text
    assert 'bar "opensees" [3.500, 0, 0, 0, 3.000, 0, 0]' in text
    assert 'bar "strut" [0, 2.500, 0, 0, 0, 2.000, 0]' in text
    assert 'bar "strut-8" [0, 0, 2.000, 0, 0, 0, 1.500]' in text
    assert "| # | Case | opensees | strut | strut-8 | Strut is |" in text
    assert (
        "| 2 | `ex50_frame2d_analyze_static_cycle` | 3.00 s | 2.00 s | 1.50 s | 2.0x faster |"
        in text
    )


def test_write_opensees_examples_markdown_charts_each_large_model(tmp_path: Path):
    summary_path = tmp_path / "summary.json"
    large_path = tmp_path / "large.json"
    output_path = tmp_path / "benchmarks.md"
    summary_path.write_text(json.dumps({"cases": []}) + "\n", encoding="utf-8")
    large_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "runner": {
                        "repeat": 1,
                        "strut_threads": "4",
                        "strut_single_thread_pass": True,
                    }
                },
                "cases": [
                    {
                        "name": "tower",
                        "dofs": 155130,
                        "opensees": {"analysis_us": 100_000_000},
                        "strut_single_thread": {"analysis_us": 20_000_000},
                        "strut": {"analysis_us": 10_000_000},
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    plot_benchmarks.write_opensees_examples_markdown(
        results_path=summary_path,
        output_path=output_path,
        large_results_paths=(large_path,),
        note="The machine was not idle.",
    )

    text = output_path.read_text(encoding="utf-8")
    assert "\nThe machine was not idle.\n\n| Model | Free DOFs |" in text
    assert "| Model | Free DOFs | opensees | strut | strut-4 |" in text
    assert (
        "| `tower` | 155,130 | 100.0 s | 20.0 s (**5.0x faster**) | 10.0 s (**10.0x faster**) |"
        in text
    )
    assert "`strut-4` is Strut on 4 threads." in text
    assert "## Large models" in text
    assert "### `tower`" in text
    assert "155,130 free DOFs. One run per engine." in text
    assert (
        'x-axis ["opensees", "strut (5.0x faster)", "strut-4 (10.0x faster)"]' in text
    )
    # One bar per engine, unnamed so no legend is drawn, then the mask.
    assert (
        "    bar [100.000, 0, 0]\n    bar [0, 20.000, 0]\n    bar [0, 0, 10.000]\n    bar [0, 0, 0]"
        in text
    )


def test_markdown_lists_all_families_and_absolute_losses_with_e2e(tmp_path):
    cases = [
        {
            "name": "gallery_slow",
            "opensees": {"analysis_us": 600},
            "strut": {"analysis_us": 900},
        },
        {
            "name": "opensees_source_example_99",
            "opensees": {"analysis_us": 100},
            "strut": {"analysis_us": 310},
        },
        {
            "name": "opensees_example_soil",
            "tcl_source": "OpenSeesExamplesPractical/soil.tcl",
            "opensees": {"analysis_us": 51},
            "strut": {"analysis_us": 52},
        },
        {"name": "strut_incomplete", "strut": {"analysis_us": 10}},
    ]
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"cases": cases}))
    e2e = tmp_path / "e2e.json"
    e2e.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "name": "gallery_live",
                        "replay_tcl": "live",
                        "opensees": {"e2e_median_s": 0.01},
                        "strut": {"e2e_median_s": 0.5},
                    }
                ]
            }
        )
    )
    output = tmp_path / "output.md"
    output.write_text("# Old\n\n## Large models\n\nKeep exactly.\n\n## Old cases\n")
    plot_benchmarks.write_opensees_examples_markdown(
        results_path=path,
        output_path=output,
        e2e_results_path=e2e,
        preserve_large_models=True,
    )
    text = output.read_text()
    assert "## Large models\n\nKeep exactly.\n\n## Analysis time" in text
    assert "### Practical and geotechnical examples" in text
    assert "### OpenSees source examples and verification" in text
    assert "### STAIRlab gallery" in text
    assert "### Strut validation cases" in text
    assert "strut_incomplete" in text
    loss_section, e2e_section = text.split("## Cases where Strut takes longer")[
        1
    ].split("## End-to-end time")
    assert loss_section.index("gallery_slow") < loss_section.index(
        "opensees_source_example_99"
    )
    assert "| `opensees_example_soil` | 51 µs | 52 µs | 1 µs |" in loss_section
    assert "| `gallery_live` | live | 10.0 ms | 500 ms |" in e2e_section
