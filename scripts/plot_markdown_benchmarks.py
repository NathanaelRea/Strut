#!/usr/bin/env python3
"""Write docs/benchmark-opensees-examples.md from benchmark summaries.

The page has a summary table, one chart per large model, and the OpenSees
examples grouped by how long OpenSees takes, so every chart has one time scale.
"""

import argparse
import json
import math
from pathlib import Path

import plot_benchmarks
from plot_constants import (
    MOJO_ORANGE,
    MOJO_ORANGE_SINGLE_THREAD,
    OPENSEES_BLUE,
    OPENSEESMP_GREEN,
)

SINGLE_THREAD_KEY = "strut_single_thread"
ENGINE_KEYS = ("opensees", "openseesmp", SINGLE_THREAD_KEY, "strut")
ENGINE_COLORS = {
    "opensees": OPENSEES_BLUE,
    "openseesmp": OPENSEESMP_GREEN,
    SINGLE_THREAD_KEY: MOJO_ORANGE_SINGLE_THREAD,
    "strut": MOJO_ORANGE,
}
EXAMPLE_PREFIX = "opensees_example_"
# Mermaid draws a short stub for a zero bar. A last, unnamed series in this
# color covers the stubs, and an unnamed series is left out of the legend.
MASK_COLOR = "#808080"
# Mermaid needs every x label to be unique. Bars that carry no label get a
# run of zero-width spaces: unique, and invisible in the chart.
HIDDEN_LABEL_CHAR = chr(0x200B)
# (heading, lower bound on the OpenSees time in seconds, has a chart)
EXAMPLE_GROUPS = (
    ("Over one second", 1.0, True),
    ("10 ms to one second", 0.010, True),
    ("Under 10 ms", 0.0, False),
)


def _is_time(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value) and value > 0


def _case_times(case: dict) -> dict[str, float]:
    """Analysis seconds per engine, for the engines this case was run on."""
    times = {}
    for key in ENGINE_KEYS:
        seconds = plot_benchmarks._case_time_seconds(case, key)
        if _is_time(seconds):
            times[key] = seconds
    return times


TIME_UNITS = (("s", 1.0), ("ms", 1e-3), ("µs", 1e-6), ("ns", 1e-9))


def _time_unit(seconds: float) -> tuple[str, float]:
    """The largest unit in which `seconds` is at least 1."""
    for unit, scale in TIME_UNITS:
        if seconds >= scale:
            return unit, scale
    return TIME_UNITS[-1]


def _format_time(seconds, unit_of=None) -> str:
    """About three significant figures, in the unit that suits `unit_of`
    (default: the value itself). A table row passes its largest time, so
    the whole row reads in one unit."""
    if not _is_time(seconds):
        return "n/a"
    unit, scale = _time_unit(unit_of if _is_time(unit_of) else seconds)
    value = seconds / scale
    if unit in ("µs", "ns"):
        decimals = 0  # the summaries store whole microseconds
    elif value >= 100:
        decimals = 1 if unit == "s" else 0
    else:
        decimals = 1 if value >= 10 else 2 if value >= 1 else 3
    return f"{value:.{decimals}f} {unit}"


def _ratio(base, value) -> str:
    """`7.4x faster`, `1.6x slower`, or `same` within 5%: `value` against `base`."""
    if not (_is_time(base) and _is_time(value)):
        return "n/a"
    if max(base, value) / min(base, value) < 1.05:
        return "same"
    if value < base:
        return f"{base / value:.1f}x faster"
    return f"{value / base:.1f}x slower"


def _strut_threads(summary: dict) -> str:
    metadata = summary.get("metadata") or {}
    threads = (metadata.get("runner") or {}).get("strut_threads")
    if isinstance(threads, str) and threads.isdigit():
        return threads
    cores = (metadata.get("platform") or {}).get("physical_cores")
    return str(cores) if cores else ""


def _engine_names(summary: dict, keys: list[str]) -> dict[str, str]:
    """Chart and table names. The threaded pass is `strut-<threads>` when a
    single-thread pass was measured beside it, and plain `strut` otherwise."""
    names = {"opensees": "opensees", "openseesmp": "openseesmp", "strut": "strut"}
    if SINGLE_THREAD_KEY in keys:
        names[SINGLE_THREAD_KEY] = "strut"
        names["strut"] = f"strut-{_strut_threads(summary) or 'mt'}"
    return names


def _engine_key_sentence(summary: dict, keys: list[str]) -> str:
    if SINGLE_THREAD_KEY not in keys:
        return ""
    threads = _strut_threads(summary)
    names = _engine_names(summary, keys)
    threaded = f"on {threads} threads" if threads else "on all cores"
    return (
        f"`opensees` is OpenSees, `strut` is Strut on one thread, and "
        f"`{names['strut']}` is Strut {threaded}."
    )


def _axis_setup(max_seconds: float) -> tuple[str, float, str]:
    """Unit label, seconds per unit, and the y-axis upper bound as text."""
    unit, scale = _time_unit(max_seconds)
    top = max_seconds / scale * 1.1
    digits = max(0, 2 - int(math.floor(math.log10(top)))) if top > 0 else 0
    return unit, scale, f"{top:.{digits}f}"


def _mermaid_chart(
    *,
    title: str,
    x_labels: list[str],
    series: list[tuple[str, str, list[float]]],
    unit: str,
    axis_top: str,
    height: int,
) -> list[str]:
    """`series` is (legend name or "", color, values). The mask goes last."""
    palette = [color for _, color, _ in series] + [MASK_COLOR]
    lines = [
        "```mermaid",
        "---",
        "config:",
        "  xyChart:",
        f"    height: {height}",
        "  themeVariables:",
        "    xyChart:",
        f"      plotColorPalette: '{', '.join(palette)}'",
        "---",
        "xychart-beta",
        f"    title {json.dumps(title, ensure_ascii=False)}",
        "    x-axis ["
        + ", ".join(json.dumps(label, ensure_ascii=False) for label in x_labels)
        + "]",
        f'    y-axis "Analysis time ({unit})" 0 --> {axis_top}',
    ]
    for name, _, values in series:
        title_part = f"{json.dumps(name)} " if name else ""
        formatted = ", ".join(f"{v:.3f}" if v else "0" for v in values)
        lines.append(f"    bar {title_part}[{formatted}]")
    lines.append(f"    bar [{', '.join('0' for _ in x_labels)}]")
    lines.append("```")
    return lines


def _single_case_chart(title: str, times: dict[str, float], names: dict) -> list[str]:
    """One bar per engine, labelled with the engine and its speedup."""
    keys = [key for key in ENGINE_KEYS if key in times]
    unit, scale, axis_top = _axis_setup(max(times.values()))
    x_labels = []
    for key in keys:
        label = names[key]
        if key != "opensees" and "opensees" in times:
            label += f" ({_ratio(times['opensees'], times[key])})"
        x_labels.append(label)
    series = [
        (
            "",
            ENGINE_COLORS[key],
            [times[key] / scale if other == key else 0.0 for other in keys],
        )
        for key in keys
    ]
    return _mermaid_chart(
        title=title,
        x_labels=x_labels,
        series=series,
        unit=unit,
        axis_top=axis_top,
        height=320,
    )


def _group_chart(
    title: str, rows: list[dict[str, float]], keys: list[str], names: dict
) -> list[str]:
    """Every case gets one slot per engine, then a gap. The case number sits
    under the middle slot. A legend names the engines."""
    peak = max(max(times.get(key, 0.0) for key in keys) for times in rows)
    unit, scale, axis_top = _axis_setup(peak)
    x_labels: list[str] = []
    values: dict[str, list[float]] = {key: [] for key in keys}
    hidden = 0
    label_slot = len(keys) // 2
    for index, times in enumerate(rows):
        slots = len(keys) + (1 if index < len(rows) - 1 else 0)
        for slot in range(slots):
            if slot == label_slot:
                x_labels.append(str(index + 1))
            else:
                hidden += 1
                x_labels.append(HIDDEN_LABEL_CHAR * hidden)
            for key in keys:
                on = slot < len(keys) and keys[slot] == key
                values[key].append(times.get(key, 0.0) / scale if on else 0.0)
    return _mermaid_chart(
        title=title,
        x_labels=x_labels,
        series=[(names[key], ENGINE_COLORS[key], values[key]) for key in keys],
        unit=unit,
        axis_top=axis_top,
        height=380,
    )


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
    return (
        f"Median of {repeat} runs per engine." if repeat > 1 else "One run per engine."
    )


def _table(headers: list[str], rows: list[list[str]], left: int = 1) -> list[str]:
    """Markdown table; the first `left` columns are left-aligned."""
    align = ["---" if i < left else "---:" for i in range(len(headers))]
    return [
        f"| {' | '.join(headers)} |",
        f"| {' | '.join(align)} |",
        *[f"| {' | '.join(row)} |" for row in rows],
        "",
    ]


def _time_cells(times: dict[str, float], keys: list[str], ratios: bool) -> list[str]:
    """One cell per engine, all in the unit of the row's largest time. With
    `ratios`, the Strut cells also carry their speedup over OpenSees."""
    cells = []
    row_max = max(times.values(), default=None)
    for key in keys:
        text = _format_time(times.get(key), row_max)
        if ratios and key in (SINGLE_THREAD_KEY, "strut") and "opensees" in times:
            ratio = _ratio(times["opensees"], times.get(key))
            text += f" (**{ratio}**)" if ratio.endswith("faster") else f" ({ratio})"
        cells.append(text)
    return cells


def _large_cases(summary: dict) -> list[tuple[dict, dict[str, float]]]:
    return [
        (case, times)
        for case in summary.get("cases", [])
        if (times := _case_times(case))
    ]


def _large_model_lines(summary: dict) -> list[str]:
    """One chart per case: every large model has its own time scale."""
    runs = _runs_phrase(summary)
    lines: list[str] = []
    for case, times in _large_cases(summary):
        name = str(case.get("name"))
        names = _engine_names(summary, list(times))
        dofs = case.get("dofs")
        facts = [f"{dofs:,} free DOFs."] if isinstance(dofs, int) else []
        if runs:
            facts.append(runs)
        notes = case.get("notes")
        if isinstance(notes, str) and notes:
            facts.append(notes)
        lines.extend([f"### `{name}`", ""])
        if facts:
            lines.extend([" ".join(facts), ""])
        lines.extend([*_single_case_chart(name, times, names), ""])
    return lines


def _example_cases(summary: dict) -> list[tuple[str, dict[str, float], object]]:
    """(label, times, dofs) for the enabled OpenSees example cases, slowest first."""
    rows = []
    for case in plot_benchmarks._filter_enabled_cases(summary.get("cases", [])):
        name = case.get("name")
        if not (isinstance(name, str) and name.startswith(EXAMPLE_PREFIX)):
            continue
        times = _case_times(case)
        if "opensees" in times and "strut" in times:
            rows.append((name[len(EXAMPLE_PREFIX) :], times, case.get("dofs")))
    return sorted(rows, key=lambda row: -row[1]["opensees"])


def _example_lines(summary: dict, *, include_mp: bool) -> list[str]:
    rows = _example_cases(summary)
    if not rows:
        return ["No enabled `opensees_example_*` benchmark cases were found.", ""]
    keys = [
        key
        for key in ENGINE_KEYS
        if any(key in times for _, times, _ in rows)
        and (include_mp or key != "openseesmp")
    ]
    names = _engine_names(summary, keys)
    verdicts = [_ratio(times["opensees"], times["strut"]) for _, times, _ in rows]
    faster = sum(1 for verdict in verdicts if verdict.endswith("faster"))
    slower = sum(1 for verdict in verdicts if verdict.endswith("slower"))
    intro = [f"{len(rows)} cases, slowest first."]
    if runs := _runs_phrase(summary):
        intro.append(runs)
    intro.append(f"Strut is faster in {faster} and slower in {slower}.")
    if SINGLE_THREAD_KEY in keys:
        intro.append(
            f"None is large enough to reach the threaded code, so `strut` and "
            f"`{names['strut']}` take the same time."
        )
    lines = ["## OpenSees examples", "", " ".join(intro), ""]
    headers = ["#", "Case", *[names[key] for key in keys], "Strut is"]
    upper = math.inf
    for heading, lower, has_chart in EXAMPLE_GROUPS:
        group = [row for row in rows if lower <= row[1]["opensees"] < upper]
        upper = lower
        if not group:
            continue
        lines.extend([f"### {heading}", ""])
        if has_chart:
            title = f"OpenSees examples: {heading[0].lower()}{heading[1:]}"
            lines.extend(
                [*_group_chart(title, [row[1] for row in group], keys, names), ""]
            )
        lines.extend(
            _table(
                headers,
                [
                    [
                        str(index + 1),
                        f"`{label}`",
                        *_time_cells(times, keys, False),
                        _ratio(times["opensees"], times["strut"]),
                    ]
                    for index, (label, times, _) in enumerate(group)
                ],
                left=2,
            )
        )
    return lines


def _summary_lines(
    summary: dict, large_summaries: list[dict], *, include_mp: bool
) -> list[str]:
    """Headline table: every large model, then the example suite as one row."""
    entries: list[tuple[str, str, dict[str, float]]] = []
    for large in large_summaries:
        for case, times in _large_cases(large):
            dofs = case.get("dofs")
            entries.append(
                (
                    f"`{case.get('name')}`",
                    f"{dofs:,}" if isinstance(dofs, int) else "",
                    times,
                )
            )
    examples = _example_cases(summary)
    if examples:
        totals: dict[str, float] = {}
        for key in ENGINE_KEYS:
            if all(key in times for _, times, _ in examples):
                totals[key] = sum(times[key] for _, times, _ in examples)
        sizes = [dofs for _, _, dofs in examples if isinstance(dofs, int)]
        entries.append(
            (
                f"OpenSees examples, all {len(examples)} cases",
                f"up to {max(sizes):,}" if sizes else "",
                totals,
            )
        )
    if not entries:
        return []
    keys = [
        key
        for key in ENGINE_KEYS
        if any(key in times for _, _, times in entries)
        and (include_mp or key != "openseesmp")
    ]
    names = _engine_names(summary if examples else large_summaries[0], keys)
    return _table(
        ["Model", "Free DOFs", *[names[key] for key in keys]],
        [
            [model, dofs, *_time_cells(times, keys, True)]
            for model, dofs, times in entries
        ],
    )


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
    large_summaries = [plot_benchmarks.load_summary(p) for p in large_results_paths]

    lines = [
        "# Benchmarks: Strut and OpenSees",
        "",
        "<!-- Auto-generated by scripts/plot_markdown_benchmarks.py -->",
        "",
    ]
    environment = _environment_lines(summary)
    lines.extend(environment)
    # The engine names come from the first summary that timed both Strut passes.
    for source in (summary, *large_summaries):
        keys = {key for case in source.get("cases", []) for key in _case_times(case)}
        if key_sentence := _engine_key_sentence(source, sorted(keys)):
            lines.extend([key_sentence, ""])
            break
    if note:
        lines.extend([note, ""])
    lines.extend(_summary_lines(summary, large_summaries, include_mp=include_mp))

    large_lines: list[str] = []
    for large_summary in large_summaries:
        large_environment = _environment_lines(large_summary)
        # Only worth repeating when the large run used another machine or build.
        if large_environment[:1] != environment[:1]:
            large_lines.extend(large_environment)
        large_lines.extend(_large_model_lines(large_summary))
    if large_lines:
        lines.extend(["## Large models", "", *large_lines])

    lines.extend(_example_lines(summary, include_mp=include_mp))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write the Markdown benchmark page with Mermaid charts."
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
        help="Include OpenSeesMP bars and table columns.",
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
