#!/usr/bin/env python3
import argparse
import json
import os
import sys
import webbrowser
from datetime import datetime
from pathlib import Path
from statistics import stdev
import math
from typing import Dict, List, Optional, Tuple

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


def _case_time_map(summary: dict, engine: str) -> Dict[str, float]:
    cases = summary.get("cases", [])
    out: Dict[str, float] = {}
    for case in cases:
        name = case.get("name")
        if not isinstance(name, str):
            continue
        value = _case_time_seconds(case, engine)
        if isinstance(value, (int, float)) and math.isfinite(value):
            out[name] = float(value)
    return out


def collect_archive_trend(
    archive_dir: Path,
) -> Tuple[List[datetime], Dict[str, List[float]], Dict[str, List[float]]]:
    engine_means: Dict[str, List[float]] = {"opensees": [], "mojo": []}
    engine_stds: Dict[str, List[float]] = {"opensees": [], "mojo": []}
    timestamps: List[datetime] = []
    files = sorted(archive_dir.glob("*-summary.json"))
    if not files:
        return timestamps, engine_means, engine_stds

    baseline_data = load_summary(files[-1])
    baseline_maps = {
        engine: _case_time_map(baseline_data, engine)
        for engine in ("opensees", "mojo")
    }
    for path in files:
        data = load_summary(path)
        for engine in ("opensees", "mojo"):
            base = baseline_maps.get(engine, {})
            current = _case_time_map(data, engine)
            ratios: List[float] = []
            for name, cur in current.items():
                base_val = base.get(name)
                if base_val is None or base_val <= 0.0:
                    continue
                ratios.append(cur / base_val)
            if ratios:
                log_ratios = [math.log(r) for r in ratios if r > 0.0]
                if log_ratios:
                    mu = sum(log_ratios) / len(log_ratios)
                    mean_ratio = math.exp(mu)
                    engine_means[engine].append(mean_ratio)
                    if len(log_ratios) > 1:
                        sigma = stdev(log_ratios)
                        engine_stds[engine].append(sigma)
                    else:
                        engine_stds[engine].append(0.0)
                else:
                    engine_means[engine].append(float("nan"))
                    engine_stds[engine].append(0.0)
            else:
                engine_means[engine].append(float("nan"))
                engine_stds[engine].append(0.0)
        timestamps.append(parse_timestamp(path, data))
    return timestamps, engine_means, engine_stds


MOJO_COLOR = "#FF5A1F"


def _group_label_from_prefix(name: str) -> str:
    parts = name.split("_")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    if parts:
        return parts[0]
    return "other"


def _truncate_label(label: str, max_len: int) -> str:
    if max_len <= 0:
        return ""
    if len(label) <= max_len:
        return label
    if max_len <= 3:
        return label[:max_len]
    return label[: max_len - 3] + "..."


def _match_group(name: str, matchers: List[str]) -> bool:
    for matcher in matchers:
        if matcher == "*":
            return True
        if matcher.startswith("re:"):
            pattern = matcher[3:]
            try:
                if __import__("re").search(pattern, name):
                    return True
            except Exception:
                continue
        else:
            if name.startswith(matcher):
                return True
    return False


def _load_group_config(path: Path) -> dict:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise SystemExit(f"Invalid group config: {path}")
    groups = data.get("groups")
    if not isinstance(groups, list):
        raise SystemExit(f"Group config missing 'groups' list: {path}")
    return data


def group_cases(
    names: List[str],
    engines: Dict[str, List[float]],
    errors: Dict[str, List[Tuple[float, float]]],
    group_by: str,
    group_config: Optional[Path],
) -> Tuple[List[str], Dict[str, List[float]], Dict[str, List[Tuple[float, float]]], List[Tuple[str, int, int]]]:
    if group_by == "none":
        return names, engines, errors, []

    grouped: Dict[str, List[int]] = {}
    group_order: List[str] = []

    if group_by == "config":
        if not group_config:
            raise SystemExit("--group-config is required when --group-by=config")
        cfg = _load_group_config(group_config)
        cfg_groups = cfg.get("groups", [])
        for idx, name in enumerate(names):
            assigned = False
            for group in cfg_groups:
                label = group.get("label")
                matchers = group.get("match", [])
                if not isinstance(label, str) or not isinstance(matchers, list):
                    continue
                if _match_group(name, matchers):
                    grouped.setdefault(label, []).append(idx)
                    if label not in group_order:
                        group_order.append(label)
                    assigned = True
                    break
            if not assigned:
                grouped.setdefault("Other", []).append(idx)
                if "Other" not in group_order:
                    group_order.append("Other")
    else:
        for idx, name in enumerate(names):
            label = _group_label_from_prefix(name)
            grouped.setdefault(label, []).append(idx)
            if label not in group_order:
                group_order.append(label)

    ordered_indices: List[int] = []
    spans: List[Tuple[str, int, int]] = []
    cursor = 0
    for label in group_order:
        indices = grouped.get(label, [])
        if not indices:
            continue
        ordered_indices.extend(indices)
        span_start = cursor
        span_end = cursor + len(indices) - 1
        spans.append((label, span_start, span_end))
        cursor += len(indices)

    ordered_names = [names[i] for i in ordered_indices]
    ordered_engines = {
        engine: [values[i] for i in ordered_indices] for engine, values in engines.items()
    }
    ordered_errors = {
        engine: [errs[i] for i in ordered_indices] for engine, errs in errors.items()
    }
    return ordered_names, ordered_engines, ordered_errors, spans


def plot_recent_bar(
    names: List[str],
    engines: Dict[str, List[float]],
    errors: Dict[str, List[Tuple[float, float]]],
    group_spans: Optional[List[Tuple[str, int, int]]] = None,
    group_gap: float = 0.6,
    label_max_len: int = 28,
):
    fig, ax = plt.subplots(figsize=(10, 5))
    if group_spans:
        gap_slots = max(0, int(round(group_gap)))
        expanded_names: List[str] = []
        expanded_engines: Dict[str, List[float]] = {
            engine: [] for engine in engines
        }
        expanded_errors: Dict[str, List[Tuple[float, float]]] = {
            engine: [] for engine in errors
        }
        group_centers: List[Tuple[str, float]] = []
        for group_idx, (label, start, end) in enumerate(group_spans):
            group_len = end - start + 1
            group_start_pos = len(expanded_names)
            for idx in range(start, end + 1):
                expanded_names.append(names[idx])
                for engine, values in engines.items():
                    expanded_engines[engine].append(values[idx])
                for engine, errs in errors.items():
                    expanded_errors[engine].append(errs[idx])
            center = group_start_pos + (group_len - 1) / 2
            group_centers.append((label, center))
            if gap_slots and group_idx < len(group_spans) - 1:
                for _ in range(gap_slots):
                    expanded_names.append("")
                    for engine in expanded_engines:
                        expanded_engines[engine].append(float("nan"))
                    for engine in expanded_errors:
                        expanded_errors[engine].append((0.0, 0.0))

        names = expanded_names
        engines = expanded_engines
        errors = expanded_errors
        x = list(range(len(names)))
    else:
        x = list(range(len(names)))
        group_centers = []
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
    display_names = [_truncate_label(name, label_max_len) for name in names]
    ax.set_xticklabels(display_names, rotation=45, ha="right", fontsize=8)
    ax.tick_params(axis="x", pad=2)
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    if group_spans:
        group_axis = ax.secondary_xaxis("bottom")
        group_axis.set_xticks([c for _, c in group_centers])
        group_axis.set_xticklabels(
            [label.replace("_", " ").title() for label, _ in group_centers]
        )
        group_axis.spines["bottom"].set_position(("axes", -0.32))
        group_axis.spines["bottom"].set_visible(False)
        group_axis.tick_params(axis="x", length=0, pad=2, labelsize=9, colors="#555555")

    if group_spans:
        fig.tight_layout(rect=[0, 0.12, 1, 0.95])
    else:
        fig.tight_layout()
    return fig


def plot_archive_trend(
    timestamps: List[datetime],
    means: Dict[str, List[float]],
    stds: Dict[str, List[float]],
):
    fig, ax = plt.subplots(figsize=(10, 4))
    if timestamps:
        opensees_means = means.get("opensees", [])
        opensees_stds = stds.get("opensees", [])
        mojo_means = means.get("mojo", [])
        mojo_stds = stds.get("mojo", [])
        opensees_lower = []
        opensees_upper = []
        mojo_lower = []
        mojo_upper = []
        for mean_ratio, sigma in zip(opensees_means, opensees_stds):
            if not isinstance(mean_ratio, (int, float)) or not math.isfinite(mean_ratio):
                opensees_lower.append(0.0)
                opensees_upper.append(0.0)
                continue
            if sigma > 0.0:
                scale = math.exp(sigma)
                opensees_lower.append(mean_ratio - mean_ratio / scale)
                opensees_upper.append(mean_ratio * scale - mean_ratio)
            else:
                opensees_lower.append(0.0)
                opensees_upper.append(0.0)
        for mean_ratio, sigma in zip(mojo_means, mojo_stds):
            if not isinstance(mean_ratio, (int, float)) or not math.isfinite(mean_ratio):
                mojo_lower.append(0.0)
                mojo_upper.append(0.0)
                continue
            if sigma > 0.0:
                scale = math.exp(sigma)
                mojo_lower.append(mean_ratio - mean_ratio / scale)
                mojo_upper.append(mean_ratio * scale - mean_ratio)
            else:
                mojo_lower.append(0.0)
                mojo_upper.append(0.0)
        opensees_err = [opensees_lower, opensees_upper]
        mojo_err = [mojo_lower, mojo_upper]
        ax.errorbar(
            timestamps,
            opensees_means,
            yerr=opensees_err,
            marker="o",
            label="OpenSees",
            capsize=3,
        )
        ax.errorbar(
            timestamps,
            mojo_means,
            yerr=mojo_err,
            marker="o",
            label="Mojo",
            color=MOJO_COLOR,
            capsize=3,
        )

    ax.set_ylabel("Geometric mean ratio vs baseline")
    ax.set_title("Archive trend (normalized to baseline, std dev of log ratios)")
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
        "--group-by",
        choices=("prefix", "config", "none"),
        default="prefix",
        help="How to group and order cases in the bar chart (default: prefix).",
    )
    parser.add_argument(
        "--group-config",
        default=None,
        help="Path to grouping config JSON (required for --group-by=config).",
    )
    parser.add_argument(
        "--group-gap",
        type=float,
        default=0.6,
        help="Number of empty slots between case groups (default: 0.6 -> 1 slot).",
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
    names, engines, errors, group_spans = group_cases(
        names,
        engines,
        errors,
        args.group_by,
        Path(args.group_config) if args.group_config else None,
    )
    for engine, values in engines.items():
        if values and all(isinstance(v, float) and v != v for v in values):
            print(
                f"warning: no analysis timings for {engine} in {results_path}; "
                "rebuild and rerun benchmarks to populate analysis_us"
            )

    with PdfPages(output_path) as pdf:
        fig = plot_recent_bar(
            names, engines, errors, group_spans, group_gap=args.group_gap
        )
        pdf.savefig(fig)
        plt.close(fig)

        if archive_dir.exists():
            timestamps, means, stds = collect_archive_trend(archive_dir)
            if timestamps:
                fig = plot_archive_trend(timestamps, means, stds)
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
