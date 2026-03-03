import importlib.util
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
