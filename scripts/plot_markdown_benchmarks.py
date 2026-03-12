#!/usr/bin/env python3
import argparse
import math
import re
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import plot_benchmarks
from plot_constants import MOJO_ORANGE, OPENSEES_BLUE, OPENSEESMP_GREEN


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
    for engine in ("opensees", "openseesmp", "strut"):
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
    active_engines: list[tuple[str, str]] = [("OS", "opensees")]
    if include_mp:
        active_engines.append(("OMP", "openseesmp"))
    active_engines.append(("STR", "strut"))
    x_labels: list[str] = []
    mask_slots: list[str] = []
    engine_slots: dict[str, list[str]] = {
        engine_key: [] for _, engine_key in active_engines
    }
    for idx, _label in enumerate(base_labels):
        case_num = idx + 1
        slot_labels = [f"{case_num}O"]
        if include_mp:
            slot_labels.append(f"{case_num}M")
        slot_labels.append(f"{case_num}S")
        x_labels.extend(_mermaid_label(label) for label in slot_labels)
        mask_slots.extend("0.0" for _ in slot_labels)
        for slot_idx, (_, engine_key) in enumerate(active_engines):
            for other_idx, (_, other_engine_key) in enumerate(active_engines):
                if slot_idx == other_idx:
                    engine_slots[other_engine_key].append(
                        _mermaid_scaled_value(engines[other_engine_key][idx], unit_divisor_us)
                    )
                else:
                    engine_slots[other_engine_key].append("0.0")
        if idx < len(base_labels) - 1:
            x_labels.append(_mermaid_label(f"{case_num}-"))
            mask_slots.append("0.0")
            for engine_key in engine_slots:
                engine_slots[engine_key].append("0.0")
    x_axis = ", ".join(x_labels)
    mask_values = ", ".join(mask_slots)
    palette = [OPENSEES_BLUE]
    if include_mp:
        palette.append(OPENSEESMP_GREEN)
    palette.extend([MOJO_ORANGE, "#d0d0d0"])
    return [
        "```mermaid",
        "---",
        "config:",
        "  themeVariables:",
        "    xyChart:",
        f"      plotColorPalette: '{', '.join(palette)}'",
        "---",
        "xychart-beta",
        f'    title "Benchmark: {example_title}"',
        f'    x-axis [{x_axis}]',
        f'    y-axis "Analysis time ({unit_label})" 0 --> {axis_max:.3f}',
        *[
            f'    bar "{series_label}" [{", ".join(engine_slots[engine_key])}]'
            for series_label, engine_key in active_engines
        ],
        f'    bar "mask" [{mask_values}]',
        "```",
    ]


def write_opensees_examples_markdown(
    *,
    results_path: Path,
    output_path: Path,
    include_mp: bool = False,
) -> Path:
    if not results_path.exists():
        raise SystemExit(f"Missing results summary: {results_path}")

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

    generated_at = summary.get("generated_at")
    if not isinstance(generated_at, str) or not generated_at.strip():
        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    lines = [
        "# OpenSees Example Benchmarks",
        "",
        "<!-- Auto-generated by scripts/plot_markdown_benchmarks.py -->",
        "",
    ]

    if grouped_indices:
        for group_label, indices in grouped_indices.items():
            names, engines, _ = plot_benchmarks.collect_recent_cases(
                filtered_summary, indices
            )
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
            lines.extend(
                [
                    "",
                    (
                        f"| # | Label | OpenSees ({unit_label}) | OpenSeesMP ({unit_label}) | Strut ({unit_label}) |"
                        if include_mp
                        else f"| # | Label | OpenSees ({unit_label}) | Strut ({unit_label}) |"
                    ),
                    (
                        "| ---: | --- | ---: | ---: | ---: |"
                        if include_mp
                        else "| ---: | --- | ---: | ---: |"
                    ),
                ]
            )
            base_labels = _relative_labels(names)
            for idx, label in enumerate(base_labels):
                if include_mp:
                    lines.append(
                        f"| {idx + 1} | `{label}` | {_format_scaled(engines['opensees'][idx], unit_divisor_us)} | "
                        f"{_format_scaled(engines['openseesmp'][idx], unit_divisor_us)} | "
                        f"{_format_scaled(engines['strut'][idx], unit_divisor_us)} |"
                    )
                else:
                    lines.append(
                        f"| {idx + 1} | `{label}` | {_format_scaled(engines['opensees'][idx], unit_divisor_us)} | "
                        f"{_format_scaled(engines['strut'][idx], unit_divisor_us)} |"
                    )
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
        "--output",
        default=None,
        help="Output Markdown path (default: docs/benchmark-opensees-examples.md)",
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
    )


if __name__ == "__main__":
    main()
