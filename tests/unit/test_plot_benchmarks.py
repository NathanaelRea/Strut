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


def test_plot_recent_bar_uses_log_scale_when_requested():
    fig = plot_benchmarks.plot_recent_bar(
        ["case_a", "case_b"],
        {"opensees": [10.0, 1000.0], "strut": [5.0, 500.0]},
        {"opensees": [(0.0, 0.0), (0.0, 0.0)], "strut": [(0.0, 0.0), (0.0, 0.0)]},
        y_scale="log",
    )

    assert fig.axes[0].get_yscale() == "log"
    plot_benchmarks.plt.close(fig)


def test_plot_archive_trend_uses_log_scale():
    fig = plot_benchmarks.plot_archive_trend(
        [datetime(2026, 2, 1), datetime(2026, 2, 2)],
        {"opensees": [0.1, 0.2], "strut": [0.05, 0.1]},
        {"opensees": [0.0, 0.0], "strut": [0.0, 0.0]},
        "Archive trend",
        "s",
        1.0,
    )

    assert fig.axes[0].get_yscale() == "log"
    plot_benchmarks.plt.close(fig)
