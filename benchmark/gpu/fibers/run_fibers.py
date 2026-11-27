#!/usr/bin/env python3
"""Benchmark Steel02 and Concrete01 updates with the MegaTall beam fiber mix."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CASE = ROOT / "benchmark/.tmp/megatall_model1_40steps/megatall_model1_40steps.json"
SOURCE = Path(__file__).with_name("sweep.mojo")
STEPS = 40
AUDIT_MAX_FIBERS = 16384


def fiber_materials(model: dict) -> tuple[list[dict], np.ndarray]:
    sections = {section["id"]: section for section in model["sections"]}
    beams = [
        element
        for element in model["elements"]
        if element["type"] in {"dispBeamColumn3d", "forceBeamColumn3d"}
        and sections[element["section"]]["type"] == "FiberSection3d"
    ]
    materials_by_section = {}
    for section_id in {element["section"] for element in beams}:
        params = sections[section_id]["params"]
        counts = [
            (item["material"], item["num_subdiv_y"] * item["num_subdiv_z"])
            for item in params.get("patches", [])
        ]
        counts.extend(
            (item["material"], item["num_bars"]) for item in params.get("layers", [])
        )
        counts.extend((item["material"], 1) for item in params.get("fibers", []))
        materials_by_section[section_id] = counts

    used_ids = {mid for counts in materials_by_section.values() for mid, _ in counts}
    materials = sorted(
        (material for material in model["materials"] if material["id"] in used_ids),
        key=lambda item: (item["type"] != "Steel02", item["id"]),
    )
    group_index = {material["id"]: i for i, material in enumerate(materials)}
    section_bytes = {
        key: b"".join(bytes((group_index[mid],)) * count for mid, count in counts)
        for key, counts in materials_by_section.items()
    }
    data = b"".join(
        section_bytes[element["section"]] * element["num_int_pts"] for element in beams
    )
    return materials, np.frombuffer(data, dtype=np.uint8)


def group_parameters(material: dict) -> tuple[list[float], float]:
    p = material["params"]
    if material["type"] == "Steel02":
        if any(p.get(key, 0.0) != 0 for key in ("a1", "a3", "sigInit")):
            raise ValueError(
                "the MegaTall sweep expects Steel02 with zero isotropic shift"
            )
        fy, modulus, b, r0, cr1, cr2 = (
            p[key] for key in ("Fy", "E0", "b", "R0", "cR1", "cR2")
        )
        yield_strain = fy / modulus
        return (
            [
                fy,
                modulus,
                b,
                r0,
                cr1,
                cr2,
                modulus * b,
                yield_strain,
                0.5 / yield_strain,
            ],
            yield_strain,
        )
    if material["type"] != "Concrete01":
        raise ValueError(f"unhandled fiber material {material['type']}")
    fpc, epsc0, fpcu, epscu = (p[key] for key in ("fpc", "epsc0", "fpcu", "epscu"))
    return (
        [
            fpc,
            epsc0,
            fpcu,
            epscu,
            2.0 * fpc / epsc0,
            (fpc - fpcu) / (epsc0 - epscu),
            0,
            0,
            0,
        ],
        -epsc0,
    )


def save_input(
    path: Path,
    ids: np.ndarray,
    strains: np.ndarray,
    offsets: np.ndarray,
    parameters: np.ndarray,
    steel_groups: int,
) -> None:
    with path.open("wb") as stream:
        np.asarray([len(ids), len(parameters), steel_groups], dtype="<i8").tofile(
            stream
        )
        offsets.astype("<i8").tofile(stream)
        parameters.astype("<f8").tofile(stream)
        ids.astype("<i4").tofile(stream)
        strains.astype("<f8").tofile(stream)


def check_parity(reference: np.ndarray, actual: np.ndarray) -> list[float]:
    # Normalize stress and tangent separately so one field cannot mask the other.
    if not np.isfinite(reference).all() or not np.isfinite(actual).all():
        raise ValueError("fiber sweep produced non-finite values")
    scale = np.maximum(np.max(np.abs(reference), axis=-1), 1.0)
    error = np.max(np.abs(reference - actual), axis=-1) / scale
    if np.any(error > 1e-10):
        raise ValueError(f"fiber response mismatch {error}")
    return error.reshape(-1, 2).max(axis=0).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, default=CASE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-fibers", type=int, default=0)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--strain-scale", type=float, default=1.0)
    args = parser.parse_args()
    if (
        args.samples <= 0
        or args.workers <= 0
        or args.max_fibers < 0
        or args.strain_scale <= 0
    ):
        parser.error("counts and strain scale must be positive, or omit --max-fibers")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    materials, tags = fiber_materials(json.loads(args.case.read_text()))
    if args.max_fibers and len(tags) > args.max_fibers:
        # Sample along the fixture order so the tagged case preserves locality.
        positions = (np.arange(args.max_fibers) + 0.5) * len(tags) / args.max_fibers
        tags = tags[positions.astype(np.int64)]
    n = len(tags)
    steel_groups = sum(mat["type"] == "Steel02" for mat in materials)
    param_rows, scales = zip(*(group_parameters(mat) for mat in materials), strict=True)
    parameters = np.asarray(param_rows)
    ref_strain = np.asarray(scales)[tags]
    rng = np.random.default_rng(1414)
    amplitude = rng.uniform(-1.9, 1.9, n) * ref_strain
    gravity_bias = (
        np.where(
            tags < steel_groups, rng.uniform(-0.4, 0.1, n), rng.uniform(-1.4, -0.05, n)
        )
        * ref_strain
    )
    strains = np.stack((amplitude, gravity_bias)) * args.strain_scale
    order = tags.argsort(kind="stable")
    offsets = np.append(0, np.bincount(tags, minlength=len(materials)).cumsum())
    save_input(output / "tagged.bin", tags, strains, offsets, parameters, steel_groups)
    save_input(
        output / "grouped.bin",
        tags[order],
        strains[:, order],
        offsets,
        parameters,
        steel_groups,
    )

    binary = output / "sweep"
    subprocess.run(
        [
            "uv",
            "run",
            "mojo",
            "build",
            "-I",
            str(ROOT / "src/mojo"),
            str(SOURCE),
            "-o",
            str(binary),
        ],
        cwd=ROOT,
        check=True,
    )
    counts = Counter()
    for group, material in enumerate(materials):
        counts[material["type"]] += int(offsets[group + 1] - offsets[group])
    results = {
        "fibers": n,
        "updates": n * STEPS,
        "counts": dict(counts),
        "strain_scale": args.strain_scale,
        "workers": args.workers,
        "modes": {},
    }
    for mode in ("cpu", "gpu_tagged", "gpu_grouped"):
        data = "tagged" if mode == "gpu_tagged" else "grouped"
        command = [
            "uv",
            "run",
            str(binary),
            str(output / f"{data}.bin"),
            mode,
            str(args.samples),
            str(args.workers),
            str(output / f"{mode}.bin"),
        ]
        run = subprocess.run(
            command,
            cwd=ROOT,
            env={"HIP_VISIBLE_DEVICES": "0", **os.environ},
            capture_output=True,
            text=True,
            check=False,
        )
        (output / f"{mode}.log").write_text(run.stdout + run.stderr)
        if run.returncode:
            raise RuntimeError(f"{mode} failed\n{run.stdout}\n{run.stderr}")
        samples = [
            float(line.split()[1])
            for line in run.stdout.splitlines()
            if line.startswith(("cpu_grouped ", "gpu_tagged ", "gpu_grouped "))
        ]
        if len(samples) != args.samples:
            raise RuntimeError(f"missing timing samples from {mode}: {run.stdout}")
        duration = statistics.median(samples)
        results["modes"][mode] = {
            "samples_s": samples,
            "median_s": duration,
            "million_updates_per_s": n * STEPS / duration / 1e6,
        }
    reference = np.fromfile(output / "cpu.bin", dtype="<f8").reshape(2, n)
    for mode in ("gpu_tagged", "gpu_grouped"):
        actual = np.fromfile(output / f"{mode}.bin", dtype="<f8").reshape(2, n)
        if mode == "gpu_tagged":
            actual = actual[:, order]
        results["modes"][mode]["peak_scaled_stress_tangent_difference"] = check_parity(
            reference, actual
        )
        results["modes"][mode]["speedup"] = (
            results["modes"]["cpu"]["median_s"] / results["modes"][mode]["median_s"]
        )
    if n <= AUDIT_MAX_FIBERS:
        oracle_path = output / "oracle.bin"
        subprocess.run(
            [
                "uv",
                "run",
                str(binary),
                str(output / "grouped.bin"),
                "oracle",
                "1",
                "1",
                str(oracle_path),
            ],
            cwd=ROOT,
            check=True,
        )
        oracle = np.fromfile(oracle_path, dtype="<f8").reshape(STEPS, 2, n)
        results["peak_scaled_history_difference_from_scalar"] = {}
        for mode in ("cpu", "gpu_tagged", "gpu_grouped"):
            history = np.fromfile(output / f"{mode}.bin.history", dtype="<f8").reshape(
                STEPS, 2, n
            )
            if mode == "gpu_tagged":
                history = history[:, :, order]
            results["peak_scaled_history_difference_from_scalar"][mode] = check_parity(
                oracle, history
            )
    (output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
