#!/usr/bin/env python3
import argparse
import math
import re
from collections import OrderedDict
from pathlib import Path

import plot_benchmarks
from plot_constants import (
    MOJO_ORANGE,
    MOJO_ORANGE_SINGLE_THREAD,
    OPENSEES_BLUE,
    OPENSEESMP_GREEN,
)

SINGLE_THREAD_KEY = "strut_single_thread"
# (chart series name, summary key, x-axis slot suffix, color)
OPENSEES_SERIES = ("OS", "opensees", "O", OPENSEES_BLUE)
OPENSEESMP_SERIES = ("OMP", "openseesmp", "M", OPENSEESMP_GREEN)
STRUT_SINGLE_SERIES = ("STR1", SINGLE_THREAD_KEY, "S1", MOJO_ORANGE_SINGLE_THREAD)
STRUT_SERIES = ("STR", "strut", "S", MOJO_ORANGE)


def _format_scaled(value_us: float, unit_divisor_us: float) -> str:
    if isinstance(value_us, (int, float)) and math.isfinite(value_us):
        return f"{float(value_us) / unit_divisor_us:.3f}"
    return "n/a"


def _mermaid_scaled_value(value_us: float, unit_divisor_us: float) -> str:
    if isinstance(value_us, (int, float)) and math.isfinite(value_us):
        return f"{float(value_us) / unit_divisor_us:.3f}"
    return "0.0"


def _plot_unit(engines: dict[str, list[float]]) -> tuple[str, float, float]:
    max_us = 0.0
    for engine in ("opensees", "openseesmp", SINGLE_THREAD_KEY, "strut"):
        for value in engines.get(engine, []):
            if isinstance(value, (int, float)) and math.isfinite(value):
                max_us = max(max_us, float(value))
    max_ns = max_us * 1_000.0
    if max_ns < 1_000.0:
        return "ns", 0.001, max_ns
    if max_ns < 1_000_000.0:
        return "us", 1.0, max_us
    if max_ns < 1_000_000_000.0:
        return "ms", 1_000.0, max_us / 1_000.0
    return "s", 1_000_000.0, max_us / 1_000_000.0


def _mermaid_label(name: str, max_len: int = 24) -> str:
    return plot_benchmarks.json.dumps(_display_label(name))


def _display_label(name: str) -> str:
    match = re.search(r"(ex[0-9]+[a-z]?_.+)$", name)
    return match.group(1) if match else name


def _shared_prefix(labels: list[str]) -> str:
    if len(labels) < 2:
        return ""
    prefix = labels[0]
    for label in labels[1:]:
        limit = min(len(prefix), len(label))
        idx = 0
        while idx < limit and prefix[idx] == label[idx]:
            idx += 1
        prefix = prefix[:idx]
        if not prefix:
            return ""
    if "_" not in prefix:
        return ""
    return prefix[: prefix.rfind("_") + 1]


def _relative_labels(names: list[str]) -> list[str]:
    labels = [_display_label(name) for name in names]
    prefix = _shared_prefix(labels)
    if not prefix:
        return labels
    trimmed = [label[len(prefix) :] for label in labels]
    if any(not label for label in trimmed):
        return labels
    return trimmed


def _example_group_key(name: str) -> tuple[int, str] | None:
    if not name.startswith("opensees_example_"):
        return None
    match = re.search(r"ex(\d+)[a-z]?(?:_|$)", name)
    if match is None:
        return None
    example_num = int(match.group(1))
    return example_num, f"ex{example_num}"


def _active_series(
    engines: dict[str, list[float]], *, include_mp: bool
) -> list[tuple[str, str, str, str]]:
    series = [OPENSEES_SERIES]
    if include_mp:
        series.append(OPENSEESMP_SERIES)
    if _has_values(engines.get(SINGLE_THREAD_KEY, [])):
        series.append(STRUT_SINGLE_SERIES)
    series.append(STRUT_SERIES)
    return series


def _has_values(values: list[float]) -> bool:
    return any(isinstance(v, (int, float)) and math.isfinite(v) for v in values)


def _single_thread_values(cases: list[dict]) -> list[float]:
    return [
        plot_benchmarks._case_time_seconds(case, SINGLE_THREAD_KEY) * 1e6
        for case in cases
    ]


def _mermaid_chart(
    title: str,
    x_labels: list[str],
    series_values: list[tuple[str, list[str]]],
    palette: list[str],
    unit_label: str,
    axis_max: float,
) -> list[str]:
    return [
        "```mermaid",
        "---",
        "config:",
        "  themeVariables:",
        "    xyChart:",
        f"      plotColorPalette: '{', '.join(palette)}'",
        "---",
        "xychart-beta",
        f'    title "{title}"',
        f'    x-axis [{", ".join(x_labels)}]',
        f'    y-axis "Analysis time ({unit_label})" 0 --> {axis_max:.3f}',
        *[f'    bar "{name}" [{", ".join(values)}]' for name, values in series_values],
        "```",
    ]


def _build_chart_lines(
    example_label: str,
    names: list[str],
    engines: dict[str, list[float]],
    *,
    include_mp: bool,
) -> list[str]:
    base_labels = _relative_labels(names)
    unit_label, unit_divisor_us, max_value = _plot_unit(engines)
    axis_max = max(1.0, math.ceil(max_value * 1.1 * 1000.0) / 1000.0)
    example_title = example_label.replace("ex", "Example ", 1)
    active = _active_series(engines, include_mp=include_mp)
    x_labels: list[str] = []
    mask_slots: list[str] = []
    slots: dict[str, list[str]] = {key: [] for _, key, _, _ in active}
    # Each engine owns one x slot per case and is zero in the others, so every
    # bar takes its engine's color; a spacer slot separates the cases.
    for idx, _label in enumerate(base_labels):
        case_num = idx + 1
        for _, key, suffix, _ in active:
            x_labels.append(_mermaid_label(f"{case_num}{suffix}"))
            mask_slots.append("0.0")
            for _, other_key, _, _ in active:
                slots[other_key].append(
                    _mermaid_scaled_value(engines[key][idx], unit_divisor_us)
                    if other_key == key
                    else "0.0"
                )
        if idx < len(base_labels) - 1:
            x_labels.append(_mermaid_label(f"{case_num}-"))
            mask_slots.append("0.0")
            for key in slots:
                slots[key].append("0.0")
    return _mermaid_chart(
        f"Benchmark: {example_title}",
        x_labels,
        [(name, slots[key]) for name, key, _, _ in active] + [("mask", mask_slots)],
        [color for _, _, _, color in active] + ["#d0d0d0"],
        unit_label,
        axis_max,
    )


SERIES_TITLES = {
    "opensees": "OpenSees",
    "openseesmp": "OpenSeesMP",
    SINGLE_THREAD_KEY: "Strut, 1 thread",
    "strut": "Strut",
}


def _strut_title(summary: dict) -> str:
    """Name the default Strut pass by its thread count when the run recorded it."""
    metadata = summary.get("metadata") or {}
    runner = metadata.get("runner") or {}
    if not runner.get("strut_single_thread_pass"):
        return "Strut"
    threads = runner.get("strut_threads")
    if not (isinstance(threads, str) and threads.isdigit()):
        threads = (metadata.get("platform") or {}).get("physical_cores")
    return f"Strut, {threads} threads" if threads else "Strut, all cores"


def _environment_lines(summary: dict) -> list[str]:
    metadata = summary.get("metadata") or {}
    platform = metadata.get("platform") or {}
    cpu = platform.get("cpu_model")
    if not cpu:
        return []
    build = metadata.get("build") or {}
    parts = [f"{cpu} ({platform.get('physical_cores')} cores)"]
    if build.get("opensees_version"):
        parts.append(f"OpenSees {build['opensees_version']}")
    if build.get("mojo_version"):
        parts.append(str(build["mojo_version"]))
    generated_at = summary.get("generated_at")
    date = generated_at[:10] if isinstance(generated_at, str) else ""
    return [
        f"Measured {date} on {', '.join(parts)}.",
        "Times are the analysis only (OpenSees `analyze`/`eigen`, Strut solve phase), "
        "with recorders removed.",
        "",
    ]


def _runs_phrase(summary: dict) -> str:
    """How each time in this summary was taken, or "" when the run did not say."""
    repeat = ((summary.get("metadata") or {}).get("runner") or {}).get("repeat")
    if not isinstance(repeat, int):
        return ""
    return f"Median of {repeat} runs per engine." if repeat > 1 else "One run per engine."


def _key_lines(
    series: list[tuple[str, str, str, str]], strut_title: str, runs: str
) -> list[str]:
    names = {**SERIES_TITLES, "strut": strut_title}
    key = " · ".join(f"**{suffix}** {names[k]}" for _, k, suffix, _ in series)
    lines = [f"Bar labels are case number plus engine: {key}."]
    if runs:
        lines.append(runs)
    lines.append(
        "A case too small to reach the threaded code shows the same time for both Strut bars."
    )
    return [*lines, ""]


def _speedup(base_us: float, value_us: float) -> str:
    if all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in (base_us, value_us)):
        return f"{base_us / value_us:.1f}x"
    return "n/a"


def _large_model_lines(summary: dict) -> list[str]:
    """One chart and table per case: every case has its own time scale."""
    cases = summary.get("cases", [])
    strut_title = _strut_title(summary)
    titles = {**SERIES_TITLES, "strut": strut_title}
    runs = _runs_phrase(summary)
    lines: list[str] = []
    for case in cases:
        engines = {
            key: [plot_benchmarks._case_time_seconds(case, key) * 1e6]
            for key in ("opensees", "openseesmp", SINGLE_THREAD_KEY, "strut")
        }
        series = [
            spec
            for spec in (OPENSEES_SERIES, OPENSEESMP_SERIES, STRUT_SINGLE_SERIES, STRUT_SERIES)
            if _has_values(engines[spec[1]])
        ]
        if not series:
            continue
        unit_label, unit_divisor_us, max_value = _plot_unit(engines)
        axis_max = max(1.0, math.ceil(max_value * 1.1 * 1000.0) / 1000.0)
        slots = {
            key: [
                _mermaid_scaled_value(engines[key][0], unit_divisor_us) if key == other else "0.0"
                for _, other, _, _ in series
            ]
            for _, key, _, _ in series
        }
        name = str(case.get("name"))
        dofs = case.get("dofs")
        lines.extend([f"### `{name}`", ""])
        facts = [f"{dofs:,} free DOFs."] if isinstance(dofs, int) else []
        if runs:
            facts.append(runs)
        if facts:
            lines.extend([" ".join(facts), ""])
        notes = case.get("notes")
        if isinstance(notes, str) and notes:
            lines.extend([notes, ""])
        lines.extend(
            _mermaid_chart(
                name,
                [plot_benchmarks.json.dumps(titles[key]) for _, key, _, _ in series],
                [(chart_name, slots[key]) for chart_name, key, _, _ in series],
                [color for _, _, _, color in series],
                unit_label,
                axis_max,
            )
        )
        lines.extend(
            [
                "",
                f"| Engine | Analysis time ({unit_label}) | vs OpenSees |",
                "| --- | ---: | ---: |",
            ]
        )
        for _, key, _, _ in series:
            lines.append(
                f"| {titles[key]} | {_format_scaled(engines[key][0], unit_divisor_us)} | "
                f"{_speedup(engines['opensees'][0], engines[key][0])} |"
            )
        lines.append("")
    return lines


def write_opensees_examples_markdown(
    *,
    results_path: Path,
    output_path: Path,
    include_mp: bool = False,
    large_results_paths: tuple[Path, ...] = (),
    note: str = "",
) -> Path:
    if not results_path.exists():
        raise SystemExit(f"Missing results summary: {results_path}")
    for path in large_results_paths:
        if not path.exists():
            raise SystemExit(f"Missing results summary: {path}")

    summary = plot_benchmarks.load_summary(results_path)
    cases = plot_benchmarks._filter_enabled_cases(summary.get("cases", []))
    filtered_summary = dict(summary)
    filtered_summary["cases"] = cases
    grouped_pairs: list[tuple[int, str, int]] = []
    for idx, case in enumerate(cases):
        name = case.get("name")
        if not isinstance(name, str):
            continue
        group_key = _example_group_key(name)
        if group_key is None:
            continue
        example_num, label = group_key
        grouped_pairs.append((example_num, label, idx))

    grouped_indices: OrderedDict[str, list[int]] = OrderedDict()
    for _, label, idx in sorted(grouped_pairs):
        grouped_indices.setdefault(label, []).append(idx)

    lines = [
        "# OpenSees Example Benchmarks",
        "",
        "<!-- Auto-generated by scripts/plot_markdown_benchmarks.py -->",
        "",
    ]
    environment = _environment_lines(summary)
    lines.extend(environment)
    if note:
        lines.extend([note, ""])

    large_lines: list[str] = []
    for path in large_results_paths:
        large_summary = plot_benchmarks.load_summary(path)
        large_environment = _environment_lines(large_summary)
        # Only worth repeating when the large run used another machine or build.
        if large_environment[:1] != environment[:1]:
            large_lines.extend(large_environment)
        large_lines.extend(_large_model_lines(large_summary))
    if large_lines:
        lines.extend(["## Large models", "", *large_lines])

    strut_title = _strut_title(summary)
    key_written = False
    if grouped_indices:
        for group_label, indices in grouped_indices.items():
            names, engines, _ = plot_benchmarks.collect_recent_cases(
                filtered_summary, indices
            )
            engines[SINGLE_THREAD_KEY] = _single_thread_values(
                [cases[i] for i in indices]
            )
            series = _active_series(engines, include_mp=include_mp)
            has_single = STRUT_SINGLE_SERIES in series
            if has_single and not key_written:
                lines.extend(_key_lines(series, strut_title, _runs_phrase(summary)))
                key_written = True
            unit_label, unit_divisor_us, _ = _plot_unit(engines)
            lines.extend(
                [
                    f"## {group_label}",
                    "",
                ]
            )
            lines.extend(
                _build_chart_lines(
                    group_label,
                    names,
                    engines,
                    include_mp=include_mp,
                )
            )
            titles = {
                **SERIES_TITLES,
                "strut": strut_title if has_single else "Strut",
            }
            headers = [f"{titles[key]} ({unit_label})" for _, key, _, _ in series]
            lines.extend(
                [
                    "",
                    f"| # | Label | {' | '.join(headers)} |",
                    f"| ---: | --- | {' | '.join('---:' for _ in series)} |",
                ]
            )
            base_labels = _relative_labels(names)
            for idx, label in enumerate(base_labels):
                values = [
                    _format_scaled(engines[key][idx], unit_divisor_us)
                    for _, key, _, _ in series
                ]
                lines.append(f"| {idx + 1} | `{label}` | {' | '.join(values)} |")
            lines.append("")
    else:
        lines.append("No enabled `opensees_example_*` benchmark cases were found.")
        lines.append("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write Markdown Mermaid charts for enabled OpenSees example benchmarks."
    )
    parser.add_argument(
        "--results",
        default=None,
        help="Path to benchmark/results/summary.json",
    )
    parser.add_argument(
        "--large-results",
        action="append",
        default=[],
        metavar="SUMMARY_JSON",
        help=(
            "Summary from a separate large-model run. Every case in it gets its own "
            "chart under 'Large models'. Repeatable."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output Markdown path (default: docs/benchmark-opensees-examples.md)",
    )
    parser.add_argument(
        "--note",
        default="",
        help="A sentence to print under the header, such as what else the machine was doing.",
    )
    parser.add_argument(
        "--mp",
        action="store_true",
        help="Include OpenSeesMP bars and table columns in the Markdown charts.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    results_path = (
        Path(args.results)
        if args.results
        else repo_root / "benchmark" / "results" / "summary.json"
    )
    output_path = (
        Path(args.output)
        if args.output
        else repo_root / "docs" / "benchmark-opensees-examples.md"
    )

    write_opensees_examples_markdown(
        results_path=results_path,
        output_path=output_path,
        include_mp=args.mp,
        large_results_paths=tuple(Path(path) for path in args.large_results),
        note=args.note,
    )


if __name__ == "__main__":
    main()
