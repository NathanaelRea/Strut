#!/usr/bin/env python3
import argparse
import json
import os
import sys
import webbrowser
from datetime import datetime
from pathlib import Path
from statistics import median, stdev
from typing import Dict, List, Tuple

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
except ModuleNotFoundError as exc:
    raise SystemExit(
        "matplotlib is required for plotting. Install it with "
        "`uv add --dev matplotlib` or `pip install matplotlib`."
    ) from exc


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text())


def parse_timestamp(path: Path, data: dict) -> datetime:
    ts = data.get("generated_at")
    if ts:
        try:
            return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            pass
    stem = path.stem
    # Expected format: YYYYmmddTHHMMSSZ-summary
    try:
        return datetime.strptime(stem.split("-")[0], "%Y%m%dT%H%M%SZ")
    except ValueError:
        return datetime.fromtimestamp(path.stat().st_mtime)


def _case_time_seconds(case: dict, engine: str) -> float:
    stats = case.get(engine)
    if isinstance(stats, dict):
        analysis_us = stats.get("analysis_us")
        if isinstance(analysis_us, (int, float)):
            return float(analysis_us) / 1e6
    if engine == "opensees":
        batch = case.get("opensees_batch")
        if isinstance(batch, dict):
            analysis_us = batch.get("analysis_us")
            if isinstance(analysis_us, (int, float)):
                return float(analysis_us) / 1e6
    return float("nan")


def _case_mean_std_seconds(case: dict, engine: str) -> Tuple[float, float]:
    value = _case_time_seconds(case, engine)
    if isinstance(value, (int, float)) and value == value:
        return (float(value), 0.0)
    return (float("nan"), 0.0)


def collect_recent_cases(
    summary: dict,
) -> Tuple[List[str], Dict[str, List[float]], Dict[str, List[Tuple[float, float]]]]:
    cases = summary.get("cases", [])
    names = [case["name"] for case in cases]
    engines = {}
    errors = {}
    for engine in ("opensees", "mojo"):
        values_us = []
        errs_us = []
        for case in cases:
            mean_s, std_s = _case_mean_std_seconds(case, engine)
            if isinstance(mean_s, float) and mean_s == mean_s:
                values_us.append(mean_s * 1e6)
                errs_us.append((std_s * 1e6, std_s * 1e6))
                continue
            fallback_s = _case_time_seconds(case, engine)
            values_us.append(
                fallback_s * 1e6 if isinstance(fallback_s, (int, float)) else fallback_s
            )
            errs_us.append((0.0, 0.0))
        engines[engine] = values_us
        errors[engine] = errs_us
    return names, engines, errors


def collect_archive_trend(archive_dir: Path) -> Tuple[List[datetime], Dict[str, List[float]]]:
    engine_series: Dict[str, List[float]] = {"opensees": [], "mojo": []}
    timestamps: List[datetime] = []
    files = sorted(archive_dir.glob("*-summary.json"))
    for path in files:
        data = load_summary(path)
        cases = data.get("cases", [])
        for engine in ("opensees", "mojo"):
            medians = [_case_time_seconds(case, engine) for case in cases]
            medians = [m for m in medians if isinstance(m, (int, float))]
            if medians:
                engine_series[engine].append(median(medians) * 1e6)
            else:
                engine_series[engine].append(float("nan"))
        timestamps.append(parse_timestamp(path, data))
    return timestamps, engine_series


MOJO_COLOR = "#FF5A1F"


def plot_recent_bar(
    names: List[str],
    engines: Dict[str, List[float]],
    errors: Dict[str, List[Tuple[float, float]]],
):
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(names))
    width = 0.38
    opensees_vals = engines.get("opensees", [])
    mojo_vals = engines.get("mojo", [])
    opensees_err = errors.get("opensees", [])
    mojo_err = errors.get("mojo", [])

    ax.bar(
        [i - width / 2 for i in x],
        opensees_vals,
        width,
        label="OpenSees",
        yerr=list(zip(*opensees_err)) if opensees_err else None,
        capsize=3,
    )
    ax.bar(
        [i + width / 2 for i in x],
        mojo_vals,
        width,
        label="Mojo",
        color=MOJO_COLOR,
        yerr=list(zip(*mojo_err)) if mojo_err else None,
        capsize=3,
    )

    ax.set_ylabel("Analysis time (us)")
    ax.set_title("Recent benchmark (analysis time per case)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    fig.tight_layout()
    return fig


def plot_archive_trend(timestamps: List[datetime], series: Dict[str, List[float]]):
    fig, ax = plt.subplots(figsize=(10, 4))
    if timestamps:
        ax.plot(timestamps, series.get("opensees", []), marker="o", label="OpenSees")
        ax.plot(
            timestamps,
            series.get("mojo", []),
            marker="o",
            label="Mojo",
            color=MOJO_COLOR,
        )

    ax.set_ylabel("Median analysis time across cases (us)")
    ax.set_title("Archive trend (median analysis across cases)")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot benchmark summaries")
    parser.add_argument(
        "--results",
        default=None,
        help="Path to benchmark/results/summary.json",
    )
    parser.add_argument(
        "--archive",
        default=None,
        help="Path to benchmark/archive directory",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PDF path (default: benchmark/results/plots.pdf)",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the output after writing (best effort).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    results_path = (
        Path(args.results)
        if args.results
        else repo_root / "benchmark" / "results" / "summary.json"
    )
    archive_dir = (
        Path(args.archive) if args.archive else repo_root / "benchmark" / "archive"
    )
    output_path = (
        Path(args.output)
        if args.output
        else repo_root / "benchmark" / "results" / "plots.pdf"
    )

    if not results_path.exists():
        raise SystemExit(f"Missing results summary: {results_path}")

    summary = load_summary(results_path)
    names, engines, errors = collect_recent_cases(summary)
    for engine, values in engines.items():
        if values and all(isinstance(v, float) and v != v for v in values):
            print(
                f"warning: no analysis timings for {engine} in {results_path}; "
                "rebuild and rerun benchmarks to populate analysis_us"
            )

    with PdfPages(output_path) as pdf:
        fig = plot_recent_bar(names, engines, errors)
        pdf.savefig(fig)
        plt.close(fig)

        if archive_dir.exists():
            timestamps, series = collect_archive_trend(archive_dir)
            if timestamps:
                fig = plot_archive_trend(timestamps, series)
                pdf.savefig(fig)
                plt.close(fig)

    print(f"Wrote {output_path}")

    if args.open:
        try:
            webbrowser.open(output_path.as_uri())
        except Exception as exc:
            print(f"warning: failed to open output: {exc}")


if __name__ == "__main__":
    main()
