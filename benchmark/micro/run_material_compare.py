#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

from common import DEFAULT_CASE, REPO_ROOT, build_mojo_binary, prepare_case_json, resolve_case_input


IMPLEMENTATIONS = {
    "elastic_uniaxial": "Baseline uniaxial Elastic implementation.",
    "elastic_fiber_scalar": "Fiber runtime-scalar Elastic implementation.",
    "steel01_uniaxial": "Baseline uniaxial Steel01 implementation.",
    "steel01_fiber_scalar": "Fiber-specialized scalar Steel01 implementation.",
    "concrete01_uniaxial": "Baseline uniaxial Concrete01 implementation.",
    "concrete01_fiber_scalar": "Fiber-specialized scalar Concrete01 implementation.",
    "concrete02_uniaxial": "Baseline uniaxial Concrete02 implementation.",
    "concrete02_fiber_scalar": "Fiber-specialized scalar Concrete02 implementation.",
    "steel02_uniaxial": "Baseline uniaxial Steel02 implementation.",
    "steel02_fiber_scalar": "Fiber-specialized scalar Steel02 implementation.",
}

DEFAULT_MATERIAL_COVERAGE_CASES = (
    "tests/validation/elastic_beam_cantilever/elastic_beam_cantilever.json",
    "tests/validation/steel01_truss_bilinear/steel01_truss_bilinear.json",
    "tests/validation/concrete01_truss_compression/concrete01_truss_compression.json",
    "tests/validation/opensees_example_ex6_genericframe2d_analyze_dynamic_eq_uniform/direct_tcl_case.json",
)


def _parse_compare(raw: str | None) -> tuple[str, str] | None:
    if raw is None or raw == "":
        return None
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise SystemExit("--compare must be impl_a,impl_b")
    for part in parts:
        if part not in IMPLEMENTATIONS:
            raise SystemExit(f"unknown implementation: {part}")
    return parts[0], parts[1]


def _build_default_material_case() -> tuple[Path, list[Path]]:
    temp_roots: list[Path] = []
    prepared_cases: list[Path] = []
    try:
        for case in DEFAULT_MATERIAL_COVERAGE_CASES:
            case_path = resolve_case_input(case)
            case_json, case_temp_roots = prepare_case_json(case_path)
            prepared_cases.append(case_json)
            temp_roots.extend(case_temp_roots)

        if not prepared_cases:
            raise SystemExit("default material coverage cases are empty")

        base_data = json.loads(prepared_cases[0].read_text(encoding="utf-8"))
        seen: set[tuple[str, str]] = set()
        merged_materials: list[dict] = []
        next_id = 1
        for case_json in prepared_cases:
            case_data = json.loads(case_json.read_text(encoding="utf-8"))
            for material in case_data.get("materials", []):
                material_type = material.get("type")
                if not isinstance(material_type, str):
                    continue
                params = material.get("params", {})
                params_key = json.dumps(params, sort_keys=True)
                key = (material_type, params_key)
                if key in seen:
                    continue
                seen.add(key)
                merged_materials.append(
                    {
                        "id": next_id,
                        "type": material_type,
                        "params": params,
                    }
                )
                next_id += 1

        if not merged_materials:
            raise SystemExit("default material coverage cases produced no materials")

        base_data["materials"] = merged_materials
        mirror_root = Path(tempfile.mkdtemp(prefix="strut_material_compare_"))
        temp_roots.append(mirror_root)
        json_path = mirror_root / "case.json"
        json_path.write_text(json.dumps(base_data, indent=2) + "\n", encoding="utf-8")
        return json_path, temp_roots
    except BaseException:
        for root in temp_roots:
            shutil.rmtree(root, ignore_errors=True)
        raise


def _run_binary(
    bin_path: Path,
    case_json: Path,
    compare: tuple[str, str] | None,
    states: int,
    steps: int,
    samples: int,
) -> list[dict]:
    cmd = [
        str(bin_path),
        "--input",
        str(case_json),
        "--states",
        str(states),
        "--steps",
        str(steps),
        "--samples",
        str(samples),
    ]
    if compare is not None:
        cmd.extend(["--left-impl", compare[0], "--right-impl", compare[1]])
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    rows = list(csv.DictReader(proc.stdout.splitlines()))
    if not rows:
        raise SystemExit("material compare binary produced no rows")
    return rows


def _summarize(rows: list[dict]) -> list[dict]:
    validation_rows = [row for row in rows if row["row_type"] == "validation"]
    bench_rows = [row for row in rows if row["row_type"] == "benchmark"]
    validation_by_key = {
        (row["compare"], row["scenario"], row["material_id"]): {
            "compare": row["compare"],
            "scenario": row["scenario"],
            "left_impl": row["left_impl"],
            "right_impl": row["right_impl"],
            "material_id": int(row["material_id"]),
            "material_type": row["material_type"],
            "states": int(row["states"]),
            "steps": int(row["steps"]),
            "max_abs_stress_diff": float(row["max_abs_stress_diff"]),
            "max_abs_tangent_diff": float(row["max_abs_tangent_diff"]),
            "max_abs_state_diff": float(row["max_abs_state_diff"]),
            "mismatch_count": int(row["mismatch_count"]),
        }
        for row in validation_rows
    }

    grouped: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    meta: dict[tuple[str, str, str, str], dict] = {}
    for row in bench_rows:
        key = (row["compare"], row["scenario"], row["material_id"], row["bench_impl"])
        updates = int(row["updates"])
        elapsed_ns = int(row["elapsed_ns"])
        grouped[key].append(elapsed_ns / updates)
        validation = validation_by_key[(row["compare"], row["scenario"], row["material_id"])]
        meta[key] = {
            **validation,
            "bench_impl": row["bench_impl"],
            "samples": 0,
        }

    summary = []
    for key, samples_ns_per_update in grouped.items():
        item = dict(meta[key])
        item["samples"] = len(samples_ns_per_update)
        item["mean_ns_per_update"] = statistics.fmean(samples_ns_per_update)
        item["median_ns_per_update"] = statistics.median(samples_ns_per_update)
        item["min_ns_per_update"] = min(samples_ns_per_update)
        item["max_ns_per_update"] = max(samples_ns_per_update)
        summary.append(item)

    summary.sort(
        key=lambda item: (item["compare"], item["material_id"], item["scenario"], item["bench_impl"])
    )
    return summary


def _print_summary(summary: list[dict]) -> None:
    grouped: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
    for item in summary:
        grouped[(item["compare"], item["material_id"], item["scenario"])].append(item)

    for (compare, material_id, scenario), items in grouped.items():
        head = items[0]
        print(
            f"Material {material_id} ({head['material_type']}): {compare} "
            f"scenario={scenario} states={head['states']} steps={head['steps']}"
        )
        print(
            "  validation: "
            f"max_stress_diff={head['max_abs_stress_diff']:.3e} "
            f"max_tangent_diff={head['max_abs_tangent_diff']:.3e} "
            f"max_state_diff={head['max_abs_state_diff']:.3e} "
            f"mismatches={head['mismatch_count']}"
        )
        by_impl = {item["bench_impl"]: item for item in items}
        for impl_name in (head["left_impl"], head["right_impl"]):
            if impl_name not in by_impl:
                continue
            item = by_impl[impl_name]
            print(
                f"  {impl_name}: "
                f"median={item['median_ns_per_update']:.2f} ns/update "
                f"mean={item['mean_ns_per_update']:.2f} ns/update"
            )
        if head["left_impl"] in by_impl and head["right_impl"] in by_impl:
            left = by_impl[head["left_impl"]]["median_ns_per_update"]
            right = by_impl[head["right_impl"]]["median_ns_per_update"]
            if right > 0.0:
                print(f"  speedup ({head['left_impl']} / {head['right_impl']}): {left / right:.3f}x")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare and benchmark material implementations across aggressive strain-history scenarios."
    )
    parser.add_argument(
        "--case",
        default=DEFAULT_CASE,
        help=(
            "Case JSON or direct_tcl_case.json manifest. "
            "Defaults to a curated multi-case material coverage bundle."
        ),
    )
    parser.add_argument(
        "--compare",
        help="Implementation pair as impl_a,impl_b. Default runs all supported pairs.",
    )
    parser.add_argument(
        "--states",
        type=int,
        default=1024,
        help="Independent randomized material states per material definition.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=128,
        help="Steps per scenario history.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Measured timing samples per implementation.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available implementation names and exit.",
    )
    parser.add_argument("--output-json", help="Optional output path for summary JSON.")
    parser.add_argument("--output-csv", help="Optional output path for raw CSV.")
    args = parser.parse_args()

    if args.list:
        for name, description in IMPLEMENTATIONS.items():
            print(f"{name}: {description}")
        return

    compare = _parse_compare(args.compare)
    if args.case == DEFAULT_CASE:
        case_json, temp_roots = _build_default_material_case()
    else:
        case_path = resolve_case_input(args.case)
        case_json, temp_roots = prepare_case_json(case_path)
    try:
        bin_path = build_mojo_binary("material_compare_micro.mojo", "material_compare_micro")
        rows = _run_binary(bin_path, case_json, compare, args.states, args.steps, args.samples)
        summary = _summarize(rows)
        _print_summary(summary)

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
