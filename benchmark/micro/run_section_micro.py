#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

from common import DEFAULT_CASE, REPO_ROOT, build_mojo_binary, prepare_case_json, resolve_case_input


DEFAULT_SWEEP_MATERIALS = (
    "elastic",
    "steel01",
    "concrete01",
    "steel02",
    "concrete02",
)

MATERIAL_LIBRARY: dict[str, dict] = {
    "elastic": {
        "label": "Elastic",
        "type": "Elastic",
        "params": {"E": 200_000_000_000.0},
    },
    "steel01": {
        "label": "Steel01",
        "type": "Steel01",
        "params": {"Fy": 350_000_000.0, "E0": 200_000_000_000.0, "b": 0.01},
    },
    "concrete01": {
        "label": "Concrete01",
        "type": "Concrete01",
        "params": {
            "fpc": -40_000_000.0,
            "epsc0": -0.002,
            "fpcu": -20_000_000.0,
            "epscu": -0.006,
        },
    },
    "steel02": {
        "label": "Steel02",
        "type": "Steel02",
        "params": {
            "Fy": 350_000_000.0,
            "E0": 200_000_000_000.0,
            "b": 0.01,
            "R0": 18.0,
            "cR1": 0.925,
            "cR2": 0.15,
        },
    },
    "concrete02": {
        "label": "Concrete02",
        "type": "Concrete02",
        "params": {
            "fpc": -40_000_000.0,
            "epsc0": -0.002,
            "fpcu": -20_000_000.0,
            "epscu": -0.006,
            "rat": 0.1,
            "ft": 3_500_000.0,
            "Ets": 200_000_000.0,
        },
    },
}

SCENARIO_LABELS = {
    "high_push": "High Push",
    "post_crushing_rebound": "Crushing Rebound",
    "post_12x_reversal": "12x Reversal",
}


def _parse_materials(raw: str | None) -> list[str]:
    if raw is None or raw.strip() == "":
        return list(DEFAULT_SWEEP_MATERIALS)
    names = [item.strip().lower() for item in raw.split(",") if item.strip()]
    if not names:
        raise SystemExit("--materials did not include any names")
    unknown = [name for name in names if name not in MATERIAL_LIBRARY]
    if unknown:
        raise SystemExit(f"unknown material(s): {', '.join(unknown)}")
    return names


def _build_material_sweep_case(
    material_names: list[str],
    num_subdiv_y: int,
    num_subdiv_z: int,
) -> tuple[Path, list[Path], dict[int, str]]:
    if num_subdiv_y <= 0 or num_subdiv_z <= 0:
        raise SystemExit("section subdivisions must be > 0")
    temp_root = Path(tempfile.mkdtemp(prefix="strut_section_micro_"))
    section_labels: dict[int, str] = {}
    materials = []
    sections = []
    nodes = []
    elements = []
    loads = []

    next_node_id = 1
    for index, material_name in enumerate(material_names, start=1):
        spec = MATERIAL_LIBRARY[material_name]
        materials.append(
            {
                "id": index,
                "type": spec["type"],
                "params": spec["params"],
            }
        )
        sections.append(
            {
                "id": index,
                "type": "FiberSection2d",
                "params": {
                    "patches": [
                        {
                            "type": "rect",
                            "material": index,
                            "num_subdiv_y": num_subdiv_y,
                            "num_subdiv_z": num_subdiv_z,
                            "y_i": -0.25,
                            "z_i": -0.15,
                            "y_j": 0.25,
                            "z_j": 0.15,
                        }
                    ],
                    "layers": [],
                },
            }
        )
        nodes.append(
            {
                "id": next_node_id,
                "x": 0.0,
                "y": float(index - 1) * 10.0,
                "constraints": [1, 2, 3],
            }
        )
        nodes.append(
            {
                "id": next_node_id + 1,
                "x": 4.0,
                "y": float(index - 1) * 10.0,
            }
        )
        elements.append(
            {
                "id": index,
                "type": "forceBeamColumn2d",
                "nodes": [next_node_id, next_node_id + 1],
                "section": index,
                "geomTransf": "Linear",
                "integration": "Lobatto",
                "num_int_pts": 3,
            }
        )
        loads.append({"node": next_node_id + 1, "dof": 2, "value": 1.0})
        section_labels[index] = spec["label"]
        next_node_id += 2

    case_data = {
        "schema_version": "1.0",
        "enabled": True,
        "metadata": {
            "name": "section_micro_material_sweep",
            "units": "SI",
        },
        "model": {"ndm": 2, "ndf": 3},
        "nodes": nodes,
        "materials": materials,
        "sections": sections,
        "elements": elements,
        "loads": loads,
        "analysis": {
            "type": "static_nonlinear",
            "steps": 1,
            "max_iters": 20,
            "tol": 1.0e-8,
            "integrator": {
                "type": "DisplacementControl",
                "node": 2,
                "dof": 2,
                "targets": [0.001],
                "cutback": 0.5,
                "max_cutbacks": 4,
            },
        },
        "recorders": [],
    }

    case_json = temp_root / "case.json"
    case_json.write_text(json.dumps(case_data, indent=2) + "\n", encoding="utf-8")
    return case_json, [temp_root], section_labels


def _load_section_labels(case_json: Path) -> dict[int, str]:
    data = json.loads(case_json.read_text(encoding="utf-8"))
    labels: dict[int, str] = {}
    materials = {
        int(item["id"]): item
        for item in data.get("materials", [])
        if isinstance(item, dict) and "id" in item
    }
    for section in data.get("sections", []):
        if not isinstance(section, dict):
            continue
        section_id = section.get("id")
        params = section.get("params", {})
        if not isinstance(section_id, int) or not isinstance(params, dict):
            continue
        material_ids: set[int] = set()
        for patch in params.get("patches", []):
            if isinstance(patch, dict) and isinstance(patch.get("material"), int):
                material_ids.add(int(patch["material"]))
        for layer in params.get("layers", []):
            if isinstance(layer, dict) and isinstance(layer.get("material"), int):
                material_ids.add(int(layer["material"]))
        if len(material_ids) == 1:
            material = materials.get(next(iter(material_ids)))
            if isinstance(material, dict) and isinstance(material.get("type"), str):
                labels[section_id] = str(material["type"])
                continue
        labels[section_id] = f"Section {section_id}"
    return labels


def _run_binary(
    bin_path: Path,
    case_json: Path,
    section_id: int | None,
    target_fibers: int,
    iterations: int,
    samples: int,
) -> list[dict]:
    cmd = [
        str(bin_path),
        "--input",
        str(case_json),
        "--target-fibers",
        str(target_fibers),
        "--iterations",
        str(iterations),
        "--samples",
        str(samples),
    ]
    if section_id is not None:
        cmd.extend(["--section-id", str(section_id)])
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    rows = list(csv.DictReader(proc.stdout.splitlines()))
    if not rows:
        raise SystemExit("benchmark binary produced no rows")
    return rows


def _summarize(
    rows: list[dict],
    section_labels: dict[int, str],
) -> list[dict]:
    grouped: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for row in rows:
        if row["benchmark"] != "full_section":
            continue
        section_id = int(row["section_id"])
        grouped[(section_id, row["scenario"])].append(row)

    summary = []
    for (section_id, scenario), sample_rows in grouped.items():
        ns_per_update = [
            int(item["elapsed_ns"]) / int(item["total_updates"]) for item in sample_rows
        ]
        best_ns_per_update = min(ns_per_update)
        fiber_count = int(sample_rows[0]["fiber_count"])
        summary.append(
            {
                "section_id": section_id,
                "label": section_labels.get(section_id, f"Section {section_id}"),
                "scenario": scenario,
                "scenario_label": SCENARIO_LABELS.get(scenario, scenario),
                "fiber_count": fiber_count,
                "instances": int(sample_rows[0]["instances"]),
                "samples": len(sample_rows),
                "best_ns_per_update": best_ns_per_update,
            }
        )

    summary.sort(key=lambda item: (item["label"], item["section_id"], item["scenario"]))
    return summary


def _build_table(summary: list[dict]) -> str:
    scenario_order = [name for name in SCENARIO_LABELS if any(item["scenario"] == name for item in summary)]
    by_label: dict[str, dict[str, dict]] = defaultdict(dict)
    row_meta: dict[str, dict] = {}
    for item in summary:
        by_label[item["label"]][item["scenario"]] = item
        row_meta[item["label"]] = item

    headers = ["Material", "Fibers", "Instances"] + [
        SCENARIO_LABELS[name] + " (ns/fiber)" for name in scenario_order
    ]
    widths = [len(header) for header in headers]
    rows: list[list[str]] = []
    ordered_labels = [
        label
        for label, _ in sorted(
            row_meta.items(),
            key=lambda item: (item[1]["section_id"], item[0]),
        )
    ]
    for label in ordered_labels:
        meta = row_meta[label]
        row = [label, str(meta["fiber_count"]), str(meta["instances"])]
        for scenario in scenario_order:
            item = by_label[label].get(scenario)
            row.append(f"{item['best_ns_per_update']:.2f}" if item else "-")
        rows.append(row)
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def format_row(values: list[str]) -> str:
        padded = []
        for index, value in enumerate(values):
            if index == 0:
                padded.append(value.ljust(widths[index]))
            else:
                padded.append(value.rjust(widths[index]))
        return " | ".join(padded)

    separator = "-+-".join("-" * width for width in widths)
    lines = [format_row(headers), separator]
    lines.extend(format_row(row) for row in rows)
    return "\n".join(lines)


def _print_summary(summary: list[dict], synthetic: bool) -> None:
    if not summary:
        raise SystemExit("no full_section benchmark rows were produced")
    heading = "Synthetic full-section material sweep" if synthetic else "Full-section benchmark summary"
    samples = summary[0]["samples"]
    instances = summary[0]["instances"]
    fibers = summary[0]["fiber_count"]
    print(heading)
    print(
        f"Metric: fastest of {samples} timed samples, reported as ns/fiber = elapsed_ns / total_updates."
    )
    print(
        f"Work per sample: {instances} section instances x {fibers} fibers/section x configured iterations."
    )
    print(_build_table(summary))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark full FiberSection2d updates with a compact material-by-scenario table."
    )
    parser.add_argument(
        "--case",
        default=None,
        help=(
            "Optional case JSON or direct_tcl_case.json manifest. "
            "Omit to generate one full-material section per supported material."
        ),
    )
    parser.add_argument(
        "--materials",
        help="Comma-separated synthetic material sweep list. Default: elastic,steel01,concrete01,steel02,concrete02.",
    )
    parser.add_argument(
        "--list-materials",
        action="store_true",
        help="List available synthetic material names and exit.",
    )
    parser.add_argument(
        "--section-id",
        type=int,
        default=None,
        help="Optional FiberSection2d section id to benchmark.",
    )
    parser.add_argument(
        "--target-fibers",
        type=int,
        default=None,
        help="Approximate total fibers per timed loop. Defaults to synthetic fibers-per-section x instances.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=32,
        help="Timed iterations per sample inside the benchmark binary.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Measured timing samples per benchmark/scenario.",
    )
    parser.add_argument(
        "--instances",
        type=int,
        default=10,
        help="Synthetic sweep section instances per timed sample. Default: 10.",
    )
    parser.add_argument(
        "--num-subdiv-y",
        type=int,
        default=25,
        help="Synthetic sweep patch subdivision count in local y.",
    )
    parser.add_argument(
        "--num-subdiv-z",
        type=int,
        default=20,
        help="Synthetic sweep patch subdivision count in local z.",
    )
    parser.add_argument("--output-json", help="Optional output path for summary JSON.")
    parser.add_argument("--output-csv", help="Optional output path for raw sample CSV.")
    args = parser.parse_args()

    if args.list_materials:
        for name in DEFAULT_SWEEP_MATERIALS:
            spec = MATERIAL_LIBRARY[name]
            print(f"{name}: {spec['label']}")
        return

    if args.iterations <= 0:
        raise SystemExit("--iterations must be > 0")
    if args.samples <= 0:
        raise SystemExit("--samples must be > 0")
    if args.instances <= 0:
        raise SystemExit("--instances must be > 0")

    synthetic = args.case is None
    if synthetic:
        material_names = _parse_materials(args.materials)
        case_json, temp_roots, section_labels = _build_material_sweep_case(
            material_names,
            args.num_subdiv_y,
            args.num_subdiv_z,
        )
        fibers_per_section = args.num_subdiv_y * args.num_subdiv_z
        target_fibers = args.target_fibers or (fibers_per_section * args.instances)
    else:
        case_path = resolve_case_input(args.case)
        case_json, temp_roots = prepare_case_json(case_path)
        section_labels = _load_section_labels(case_json)
        target_fibers = args.target_fibers or 5000

    if target_fibers <= 0:
        raise SystemExit("--target-fibers must be > 0")

    try:
        bin_path = build_mojo_binary("benchmarks_micro.mojo", "benchmarks_micro")
        rows = _run_binary(
            bin_path,
            case_json,
            args.section_id,
            target_fibers,
            args.iterations,
            args.samples,
        )
        summary = _summarize(rows, section_labels)
        _print_summary(summary, synthetic)

        if args.output_csv:
            out_csv = (REPO_ROOT / args.output_csv).resolve()
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            with out_csv.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

        if args.output_json:
            out_json = (REPO_ROOT / args.output_json).resolve()
            out_json.parent.mkdir(parents=True, exist_ok=True)
            out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    finally:
        for root in temp_roots:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
