#!/usr/bin/env python3
import argparse
import json
import os
import sys
import webbrowser
from datetime import datetime
from pathlib import Path
from statistics import median
import math
from typing import Dict, List, Optional, Tuple

try:
    from .plot_constants import MOJO_ORANGE, OPENSEES_BLUE
except ImportError:
    # Allow running as a standalone script.
    from plot_constants import MOJO_ORANGE, OPENSEES_BLUE

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
    indices: Optional[List[int]] = None,
) -> Tuple[List[str], Dict[str, List[float]], Dict[str, List[Tuple[float, float]]]]:
    cases = summary.get("cases", [])
    if indices is not None:
        cases = [cases[i] for i in indices]
    names = [case["name"] for case in cases]
    engines = {}
    errors = {}
    for engine in ("opensees", "strut"):
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


def _count_constrained_dofs(node: dict, ndf: int) -> int:
    constraints = node.get("constraints")
    if not constraints:
        return 0
    if isinstance(constraints, list):
        if all(isinstance(v, bool) for v in constraints):
            if len(constraints) != ndf:
                return 0
            return sum(1 for v in constraints if v)
        return len(constraints)
    return 0


def _case_free_dofs(case: dict) -> Optional[int]:
    dofs = case.get("dofs")
    if isinstance(dofs, int):
        return dofs
    json_path = case.get("json")
    if not json_path:
        return None
    path = Path(json_path)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    model = data.get("model", {})
    ndf = model.get("ndf")
    nodes = data.get("nodes")
    if not isinstance(ndf, int) or not isinstance(nodes, list):
        return None
    constrained = 0
    for node in nodes:
        if isinstance(node, dict):
            constrained += _count_constrained_dofs(node, ndf)
    total = len(nodes) * ndf
    return total - constrained


def _case_size_override(case: dict) -> Optional[str]:
    label = case.get("size")
    if isinstance(label, str):
        normalized = label.strip().lower()
        if normalized in {"small", "medium", "large"}:
            return normalized
    json_path = case.get("json")
    if not json_path:
        return None
    path = Path(json_path)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    label = data.get("benchmark_size")
    if not isinstance(label, str):
        return None
    normalized = label.strip().lower()
    if normalized in {"small", "medium", "large"}:
        return normalized
    return None


def _case_size_label(
    free_dofs: Optional[int], medium_threshold: int, large_threshold: int
) -> str:
    if free_dofs is None:
        return "small"
    if free_dofs >= large_threshold:
        return "large"
    if free_dofs >= medium_threshold:
        return "medium"
    return "small"


def collect_archive_trend(
    archive_dir: Path,
    size_filter: Optional[str],
    medium_threshold: int,
    large_threshold: int,
) -> Tuple[List[datetime], Dict[str, List[float]], Dict[str, List[float]]]:
    engine_means: Dict[str, List[float]] = {"opensees": [], "strut": []}
    engine_stds: Dict[str, List[float]] = {"opensees": [], "strut": []}
    timestamps: List[datetime] = []
    files = sorted(archive_dir.glob("*-summary.json"))
    if not files:
        return timestamps, engine_means, engine_stds

    per_hash_entries: Dict[str, List[Tuple[datetime, Dict[str, float]]]] = {}

    for path in files:
        data = load_summary(path)
        run_medians: Dict[str, float] = {}
        for engine in ("opensees", "strut"):
            values: List[float] = []
            for case in data.get("cases", []):
                if size_filter:
                    override = _case_size_override(case)
                    if override is not None:
                        label = override
                    else:
                        free_dofs = _case_free_dofs(case)
                        label = _case_size_label(
                            free_dofs, medium_threshold, large_threshold
                        )
                    if label != size_filter:
                        continue
                value = _case_time_seconds(case, engine)
                if isinstance(value, (int, float)) and math.isfinite(value):
                    values.append(float(value))
            run_medians[engine] = median(values) if values else float("nan")

        git_rev = data.get("git_rev")
        hash_key = (
            git_rev.strip()
            if isinstance(git_rev, str) and git_rev.strip()
            else "unknown"
        )
        per_hash_entries.setdefault(hash_key, []).append(
            (parse_timestamp(path, data), run_medians)
        )

    def _mean_std(values: List[float]) -> Tuple[float, float]:
        finite = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
        if not finite:
            return float("nan"), float("nan")
        mean = sum(finite) / len(finite)
        if len(finite) == 1:
            return mean, 0.0
        variance = sum((v - mean) ** 2 for v in finite) / len(finite)
        return mean, math.sqrt(variance)

    ordered_batches = sorted(
        (
            (max(ts for ts, _ in entries), entries)
            for entries in per_hash_entries.values()
            if entries
        ),
        key=lambda item: item[0],
    )

    for latest_ts, entries in ordered_batches:
        timestamps.append(latest_ts)
        for engine in ("opensees", "strut"):
            values = [run_vals.get(engine, float("nan")) for _, run_vals in entries]
            mean_val, std_val = _mean_std(values)
            engine_means[engine].append(mean_val)
            engine_stds[engine].append(std_val)

    return timestamps, engine_means, engine_stds


def archive_min_timestamp(archive_dir: Path) -> Optional[datetime]:
    files = sorted(archive_dir.glob("*-summary.json"))
    if not files:
        return None
    min_ts: Optional[datetime] = None
    for path in files:
        try:
            data = load_summary(path)
        except Exception:
            continue
        ts = parse_timestamp(path, data)
        if min_ts is None or ts < min_ts:
            min_ts = ts
    return min_ts


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
) -> Tuple[
    List[str],
    Dict[str, List[float]],
    Dict[str, List[Tuple[float, float]]],
    List[Tuple[str, int, int]],
]:
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
        engine: [values[i] for i in ordered_indices]
        for engine, values in engines.items()
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
    title: str = "Recent benchmark (analysis time per case)",
    unit_label: str = "us",
    scale: float = 1.0,
    y_scale: str = "linear",
):
    fig, ax = plt.subplots(figsize=(10, 5))
    if group_spans:
        gap_slots = max(0, int(round(group_gap)))
        expanded_names: List[str] = []
        expanded_engines: Dict[str, List[float]] = {engine: [] for engine in engines}
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
    opensees_vals = [v * scale for v in engines.get("opensees", [])]
    strut_vals = [v * scale for v in engines.get("strut", [])]

    ax.bar(
        [i - width / 2 for i in x],
        opensees_vals,
        width,
        label="OpenSees",
    )
    ax.bar(
        [i + width / 2 for i in x],
        strut_vals,
        width,
        label="Strut",
        color=MOJO_ORANGE,
    )

    ax.set_ylabel(f"Analysis time ({unit_label})")
    ax.set_title(title)
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
    title: str,
    unit_label: str,
    scale: float,
    min_timestamp: Optional[datetime] = None,
):
    fig, ax = plt.subplots(figsize=(10, 4))
    if timestamps:
        opensees_means = means.get("opensees", [])
        strut_means = means.get("strut", [])
        opensees_stds = stds.get("opensees", [])
        strut_stds = stds.get("strut", [])
        # Draw both series as filled circles and distinguish by color only.
        ax.errorbar(
            timestamps,
            [v * scale for v in strut_means],
            yerr=[v * scale for v in strut_stds],
            marker="o",
            linestyle="none",
            label="Strut",
            color=MOJO_ORANGE,
            capsize=3,
        )
        ax.errorbar(
            timestamps,
            [v * scale for v in opensees_means],
            yerr=[v * scale for v in opensees_stds],
            marker="o",
            linestyle="none",
            label="OpenSees",
            color=OPENSEES_BLUE,
            capsize=3,
        )

    ax.set_ylabel(f"Mean median analysis time ({unit_label})")
    ax.set_title(title)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend()
    if min_timestamp is not None:
        ax.set_xlim(left=min_timestamp)
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
        "--large-threshold",
        type=int,
        default=1000,
        help="Free-DOF threshold to split large benchmarks (default: 1000).",
    )
    parser.add_argument(
        "--medium-threshold",
        type=int,
        default=300,
        help="Free-DOF threshold to split medium benchmarks (default: 300).",
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
    cases = summary.get("cases", [])
    small_indices: List[int] = []
    medium_indices: List[int] = []
    large_indices: List[int] = []
    for idx, case in enumerate(cases):
        override = _case_size_override(case)
        if override == "large":
            large_indices.append(idx)
            continue
        if override == "medium":
            medium_indices.append(idx)
            continue
        if override == "small":
            small_indices.append(idx)
            continue
        free_dofs = _case_free_dofs(case)
        if free_dofs is not None:
            if free_dofs >= args.large_threshold:
                large_indices.append(idx)
            elif free_dofs >= args.medium_threshold:
                medium_indices.append(idx)
            else:
                small_indices.append(idx)
        else:
            small_indices.append(idx)

    def _prepare(indices: List[int]):
        names, engines, errors = collect_recent_cases(summary, indices)
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
        return names, engines, errors, group_spans

    small_names, small_engines, small_errors, small_spans = _prepare(small_indices)
    medium_names, medium_engines, medium_errors, medium_spans = _prepare(medium_indices)
    large_names, large_engines, large_errors, large_spans = _prepare(large_indices)

    with PdfPages(output_path) as pdf:
        if archive_dir.exists():
            archive_min_ts = archive_min_timestamp(archive_dir)
            archive_specs = [
                ("small", "Archive trend (small cases)", "us", 1e6),
                ("medium", "Archive trend (medium cases)", "ms", 1e3),
                ("large", "Archive trend (large cases)", "s", 1.0),
            ]
            for size_label, title, unit_label, scale in archive_specs:
                timestamps, means, stds = collect_archive_trend(
                    archive_dir,
                    size_label,
                    args.medium_threshold,
                    args.large_threshold,
                )
                if timestamps and any(
                    isinstance(v, float) and math.isfinite(v)
                    for vals in means.values()
                    for v in vals
                ):
                    fig = plot_archive_trend(
                        timestamps,
                        means,
                        stds,
                        title,
                        unit_label,
                        scale,
                        min_timestamp=archive_min_ts,
                    )
                    pdf.savefig(fig)
                    plt.close(fig)

        if small_names:
            fig = plot_recent_bar(
                small_names,
                small_engines,
                small_errors,
                small_spans,
                group_gap=args.group_gap,
                title="Recent benchmark (small cases)",
            )
            pdf.savefig(fig)
            plt.close(fig)

        if medium_names:
            fig = plot_recent_bar(
                medium_names,
                medium_engines,
                medium_errors,
                medium_spans,
                group_gap=args.group_gap,
                title="Recent benchmark (medium cases)",
                unit_label="ms",
                scale=1.0 / 1e3,
            )
            pdf.savefig(fig)
            plt.close(fig)

        if large_names:
            fig = plot_recent_bar(
                large_names,
                large_engines,
                large_errors,
                large_spans,
                group_gap=args.group_gap,
                title="Recent benchmark (large cases)",
            )
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
