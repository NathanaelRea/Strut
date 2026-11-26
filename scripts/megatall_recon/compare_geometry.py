"""Compare the parametric Model1 mesh with the pinned OpenSees Tcl fixture."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from math import ceil, dist
from pathlib import Path
from time import perf_counter

if __package__:
    from .geometry import ARITY, FAMILIES, PARAMS, Geometry, Point, generate
else:
    from geometry import ARITY, FAMILIES, PARAMS, Geometry, Point, generate

MODEL1 = (
    Path(__file__).resolve().parents[2]
    / "tests/fixtures/opensees/OpenSeesExamples/MegaTallBuildingBenchmark/Model1"
)
TOLERANCE = 0.01  # metres


@dataclass
class Reference:
    geometry: Geometry
    raw_command_bytes: int
    geometry_only_bytes: int


def read_reference(directory: Path) -> Reference:
    """Read just node positions, element endpoints, and diaphragm membership."""
    nodes = {}
    elements = {}
    diaphragms = {}
    order = {family: [] for family in FAMILIES}
    raw_bytes = 0
    geometry_bytes = 0
    sources = [*dict.fromkeys(file for file, _ in FAMILIES), "rigiddigram.tcl"]
    for filename in sources:
        path = directory / filename
        with path.open("rb") as source:
            for line in source:
                fields = line.split()
                if not fields:
                    continue
                command = fields[0]
                if command == b"node":
                    size = 5
                    tag = int(fields[1])
                    assert tag not in nodes, f"Duplicate node {tag} in {path}"
                    nodes[tag] = tuple(map(float, fields[2:size]))
                elif command == b"element":
                    kind = fields[1].decode()
                    size = 3 + ARITY[kind]
                    tag = int(fields[2])
                    assert tag not in elements, f"Duplicate element {tag} in {path}"
                    elements[tag] = (kind, tuple(map(int, fields[3:size])))
                    order[filename, kind].append(tag)
                elif command == b"rigidDiaphragm":
                    size = len(fields)
                    master = int(fields[2])
                    assert fields[1] == b"3", (
                        f"Unexpected diaphragm direction in {path}"
                    )
                    assert master not in diaphragms, (
                        f"Duplicate diaphragm master {master}"
                    )
                    diaphragms[master] = frozenset(map(int, fields[3:]))
                else:
                    continue
                raw_bytes += len(line)
                # One ASCII space between retained tokens, one LF per command.
                geometry_bytes += sum(map(len, fields[:size])) + size

    return Reference(
        Geometry(nodes, elements, diaphragms, {k: tuple(v) for k, v in order.items()}),
        raw_bytes,
        geometry_bytes,
    )


@dataclass
class NodeComparison:
    total: int
    missing: int
    extra: int
    max_error: float
    p99_error: float
    outside_tolerance: int


def compare_nodes(
    expected: dict[int, Point],
    actual: dict[int, Point],
    before: set[int],
    after: set[int],
) -> NodeComparison:
    errors = sorted(dist(expected[i], actual[i]) for i in before & after)
    return NodeComparison(
        len(before),
        len(before - after),
        len(after - before),
        max(errors, default=0.0),
        errors[ceil(0.99 * len(errors)) - 1] if errors else 0.0,
        sum(error > TOLERANCE for error in errors),
    )


def node_groups(geometry: Geometry, floors: set[float]) -> dict[str, set[int]]:
    groups = defaultdict(set)
    for tag, (x, y, z) in geometry.nodes.items():
        if z == 0:
            group = "Base"
        elif round(z, 4) not in floors:
            group = "Wall head"
        else:
            group = "Floor core" if max(abs(x), abs(y)) <= 15.005 else "Floor perimeter"
        groups[group].add(tag)
    return groups


def _row(values):
    print("| " + " | ".join(map(str, values)) + " |")


def report(reference: Reference, actual: Geometry) -> bool:
    expected = reference.geometry
    floors = {round(expected.nodes[m][2], 4) for m in expected.diaphragms}
    before = node_groups(expected, floors)
    after = node_groups(actual, floors)
    before["All"] = set(expected.nodes)
    after["All"] = set(actual.nodes)
    valid = True
    print("Node coordinate errors use 3D distance in metres.")
    print()
    _row(
        (
            "Node family",
            "Reference",
            "Missing",
            "Extra",
            "Max (m)",
            "p99 (m)",
            ">0.01 m",
        )
    )
    _row(("---", "---:", "---:", "---:", "---:", "---:", "---:"))
    for name in ("Base", "Wall head", "Floor core", "Floor perimeter", "All"):
        row = compare_nodes(expected.nodes, actual.nodes, before[name], after[name])
        _row(
            (
                name,
                row.total,
                row.missing,
                row.extra,
                f"{row.max_error:.6f}",
                f"{row.p99_error:.6f}",
                row.outside_tolerance,
            )
        )
        valid &= not (row.missing or row.extra or row.outside_tolerance)

    print()
    _row(
        (
            "Element family",
            "Reference",
            "Missing",
            "Extra",
            "Ordered connectivity",
            "Tcl order",
            "Node max (m)",
            "Node p99 (m)",
            "Nodes missing/extra",
        )
    )
    _row(("---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"))
    for family in FAMILIES:
        before_order = expected.element_order[family]
        after_order = actual.element_order[family]
        before_tags, after_tags = set(before_order), set(after_order)
        matching = sum(
            expected.elements[i] == actual.elements[i] for i in before_tags & after_tags
        )
        missing, extra = len(before_tags - after_tags), len(after_tags - before_tags)
        order = before_order == after_order
        nodes_before = {n for i in before_tags for n in expected.elements[i][1]}
        nodes_after = {n for i in after_tags for n in actual.elements[i][1]}
        node_row = compare_nodes(
            expected.nodes, actual.nodes, nodes_before, nodes_after
        )
        label = "/".join(family)
        _row(
            (
                label,
                len(before_tags),
                missing,
                extra,
                f"{matching}/{len(before_tags)} ({matching / len(before_tags):.2%})",
                "exact" if order else "different",
                f"{node_row.max_error:.6f}",
                f"{node_row.p99_error:.6f}",
                f"{node_row.missing}/{node_row.extra}",
            )
        )
        valid &= matching == len(before_tags) and not extra and order

    print()
    before_masters = set(expected.diaphragms)
    after_masters = set(actual.diaphragms)
    missing_masters = before_masters - after_masters
    extra_masters = after_masters - before_masters
    mismatched = {
        m
        for m in before_masters & after_masters
        if expected.diaphragms[m] != actual.diaphragms[m]
    }
    match_count = len(before_masters) - len(missing_masters) - len(mismatched)
    print(
        f"Diaphragms {match_count}/{len(before_masters)} exact master/slave sets "
        f"({len(missing_masters)} missing masters, {len(extra_masters)} extra masters, "
        f"{len(mismatched)} unequal slave sets)."
    )
    valid &= not (missing_masters or extra_masters or mismatched)
    return valid


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=MODEL1)
    parser.add_argument("--params", type=Path, default=PARAMS)
    args = parser.parse_args()
    try:
        reference = read_reference(args.model_dir)
    except FileNotFoundError as error:
        parser.exit(
            2,
            f"{error}\nInstall the fixture with uv run scripts/setup_reference_data.py.\n",
        )

    start = perf_counter()
    actual = generate(args.params)
    seconds = perf_counter() - start
    matched = report(reference, actual)
    print(f"Generator {seconds:.3f} s.")
    print(f"Original geometry commands {reference.raw_command_bytes:,} bytes.")
    print(
        f"Original geometry tokens with normalized whitespace {reference.geometry_only_bytes:,} bytes."
    )
    params_bytes = args.params.stat().st_size
    table_name = json.loads(args.params.read_text())["legacy_table"]
    table_bytes = args.params.with_name(table_name).stat().st_size
    total_bytes = params_bytes + table_bytes
    print(
        f"Parameter and stencil tables {total_bytes:,} bytes "
        f"({params_bytes:,} JSON + {table_bytes:,} binary)."
    )
    print(
        f"Compression {reference.raw_command_bytes / total_bytes:.1f}× against the "
        f"original commands and {reference.geometry_only_bytes / total_bytes:.1f}× "
        "against the geometry tokens."
    )
    return int(not matched)


if __name__ == "__main__":
    raise SystemExit(main())
