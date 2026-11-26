#!/usr/bin/env python3
"""Cut a small static model out of the mega-tall building for parity checks.

The full model takes OpenSees a quarter of an hour, and a mismatch in it says
nothing about which element is responsible. A slice keeps the real nodes,
sections, materials, orientations and diaphragm constraints of a few stories,
fixes its lowest level and loads it, so OpenSees answers in seconds and
`--types` can narrow a mismatch down to one element type.

What a slice is:

* Elements whose nodes all lie between `--z-min` and `--z-max`, optionally only
  some `--types`.
* The nodes they use, plus diaphragm master nodes. Nodes at the lowest level
  are fully fixed; the rest keep their original restraints.
* Rigid diaphragms whose two nodes both survive (drop them with
  `--no-diaphragm`). A constrained node that lost its master is left free.
* The model's gravity loads on the surviving nodes times `--load-scale`, plus
  `--lateral` times each node's load applied in +x and half that in +y, so
  lateral and torsional stiffness show up in the answer too. A slice carries
  only its own stories, so it stays nearly elastic at scale 1; raise the scale
  to crack the walls and yield the steel.
* One static stage solved to a tight tolerance with `UmfPack`. OpenSees'
  `SparseSYM` does not converge tightly on rigid-diaphragm systems, and the
  point here is to compare the models, not the solvers.
* With `--transient-steps`, a second stage shakes the slice with the model's
  ground motion, masses, Rayleigh damping and Newmark settings. The record
  builds up slowly, so the stage starts `--accel-start` seconds into it.
* Displacements of every free node (or every `--record-every`-th one, for the
  whole building) and reactions of every fixed node.

Examples:
  uv run scripts/megatall_slice.py --z-min 0 --z-max 12
  uv run scripts/megatall_slice.py --z-min 0 --z-max 12 --types shell --no-diaphragm
  uv run scripts/parity_report.py benchmark/.tmp/megatall_slice_z0_12/megatall_slice_z0_12.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

MEGATALL_CASE = "opensees_megatall_building_model1_dynamiccpu"
ELEVATION_TOLERANCE = 1.0e-6
ALL_DOFS = [1, 2, 3, 4, 5, 6]


def convert_megatall() -> dict:
    """Convert the installed mega-tall fixture the way the harness does.

    Always converts, so a slice can never come from a copy that predates the
    current fixture or converter. It takes about three seconds.
    """
    import run_benchmarks

    case = run_benchmarks.resolve_case_from_name(
        REPO_ROOT / "tests" / "validation", MEGATALL_CASE
    )
    if case is None:
        raise SystemExit("mega-tall validation case not found")
    return run_benchmarks._ensure_direct_tcl_case_artifacts(
        case, REPO_ROOT, None, False, require_parser_check=False
    )


def load_source(path: Path | None) -> dict:
    if path is None:
        return convert_megatall()
    if not path.is_file():
        raise SystemExit(f"source case not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _material_closure(materials_by_id: dict[int, dict], seeds: set[int]) -> set[int]:
    needed: set[int] = set()
    queue = list(seeds)
    while queue:
        material_id = queue.pop()
        if material_id in needed:
            continue
        needed.add(material_id)
        base = materials_by_id[material_id].get("params", {}).get("material")
        if isinstance(base, int):
            queue.append(base)
    return needed


def _section_materials(section: dict) -> set[int]:
    params = section.get("params", {})
    ids: set[int] = set()
    for key in ("fibers", "patches", "layers"):
        for item in params.get(key, []):
            if "material" in item:
                ids.add(int(item["material"]))
    return ids


def build_slice(
    source: dict,
    z_min: float,
    z_max: float,
    types: set[str] | None,
    keep_diaphragms: bool,
    lateral: float,
    load_scale: float,
    steps: int,
    name: str,
    record_every: int = 1,
    transient_steps: int = 0,
    accel_start: float = 6.0,
    accel_scale: float = 1.0,
) -> dict:
    node_by_id = {int(node["id"]): node for node in source["nodes"]}

    def inside(node_id: int) -> bool:
        z = node_by_id[node_id]["z"]
        return z_min - ELEVATION_TOLERANCE <= z <= z_max + ELEVATION_TOLERANCE

    elements = [
        element
        for element in source["elements"]
        if (types is None or element["type"] in types)
        and all(inside(node_id) for node_id in element["nodes"])
    ]
    if not elements:
        raise SystemExit("no elements in that range")
    kept_nodes = {node_id for element in elements for node_id in element["nodes"]}

    constraints = []
    if keep_diaphragms:
        for constraint in source.get("mp_constraints", []):
            constrained = int(constraint["constrained_node"])
            retained = int(constraint["retained_node"])
            # The master is often joined to the floor only through constraints.
            if constrained in kept_nodes and inside(retained):
                constraints.append(constraint)
        kept_nodes |= {int(c["retained_node"]) for c in constraints}

    base_z = min(node_by_id[node_id]["z"] for node_id in kept_nodes)
    nodes = []
    fixed_nodes = []
    for node_id in sorted(kept_nodes):
        node = dict(node_by_id[node_id])
        if node["z"] <= base_z + ELEVATION_TOLERANCE:
            node["constraints"] = list(ALL_DOFS)
        if node.get("constraints") == ALL_DOFS:
            fixed_nodes.append(node_id)
        nodes.append(node)
    fixed = set(fixed_nodes)
    # A constraint may not touch a fully fixed node.
    constraints = [
        c
        for c in constraints
        if int(c["constrained_node"]) not in fixed
        and int(c["retained_node"]) not in fixed
    ]

    gravity_stage = source["analysis"]["stages"][0]
    loads = []
    for load in gravity_stage["loads"]:
        node_id = int(load["node"])
        if node_id not in kept_nodes or node_id in fixed:
            continue
        scaled = {**load, "value": load_scale * float(load["value"])}
        loads.append(scaled)
        if lateral != 0.0 and int(load["dof"]) == 3:
            push = lateral * abs(scaled["value"])
            loads.append({"node": node_id, "dof": 1, "value": push})
            loads.append({"node": node_id, "dof": 2, "value": 0.5 * push})
    if not loads:
        raise SystemExit("no gravity loads act on the slice")

    section_by_id = {int(section["id"]): section for section in source["sections"]}
    materials_by_id = {
        int(material["id"]): material for material in source["materials"]
    }
    section_ids = sorted({int(e["section"]) for e in elements if "section" in e})
    material_seeds = {int(e["material"]) for e in elements if "material" in e}
    for section_id in section_ids:
        material_seeds |= _section_materials(section_by_id[section_id])
    material_ids = sorted(_material_closure(materials_by_id, material_seeds))

    pattern = dict(gravity_stage["pattern"])
    series_tag = pattern.get("time_series")
    free_nodes = [node_id for node_id in sorted(kept_nodes) if node_id not in fixed]
    time_series = [
        series
        for series in source.get("time_series", [])
        if series.get("tag") == series_tag
    ]
    static_analysis = {
        "type": "static_nonlinear",
        "constraints": "Transformation",
        "numberer": "RCM",
        "system": "UmfPack",
        "algorithm": "Newton",
        "integrator": {"type": "LoadControl", "step": 1.0 / steps},
        "steps": steps,
        "test_type": "NormDispIncr",
        "tol": 1.0e-8,
        "max_iters": 30,
    }
    analysis = static_analysis
    masses = []
    if transient_steps > 0:
        shaking = source["analysis"]["stages"][1]
        motion = next(
            series
            for series in source["time_series"]
            if series.get("tag") == shaking["pattern"]["accel"]
        )
        dt = float(shaking["analysis"]["dt"])
        record = [float(v) for v in Path(motion["values_path"]).read_text().split()]
        first = int(round(accel_start / float(motion["dt"])))
        count = int(round(transient_steps * dt / float(motion["dt"]))) + 2
        if first + count > len(record):
            raise SystemExit("the ground motion is shorter than that window")
        time_series.append(
            {
                "type": "Path",
                "tag": motion["tag"],
                "factor": accel_scale * float(motion.get("factor", 1.0)),
                "dt": motion["dt"],
                "values": record[first : first + count],
            }
        )
        masses = [
            mass
            for mass in source.get("masses", [])
            if int(mass["node"]) in kept_nodes and int(mass["node"]) not in fixed
        ]
        analysis = {
            "type": "staged",
            "constraints": "Transformation",
            "stages": [
                {
                    "analysis": static_analysis,
                    "pattern": pattern,
                    "loads": loads,
                    "load_const": {"time": 0.0},
                },
                {
                    "analysis": {
                        "type": "transient_nonlinear",
                        "constraints": "Transformation",
                        "numberer": "RCM",
                        "system": "UmfPack",
                        "algorithm": "Newton",
                        "integrator": shaking["analysis"]["integrator"],
                        "dt": dt,
                        "steps": transient_steps,
                        "test_type": "NormDispIncr",
                        "tol": 1.0e-8,
                        "max_iters": 30,
                    },
                    "pattern": shaking["pattern"],
                    "rayleigh": shaking["rayleigh"],
                },
            ],
        }
    return {
        "schema_version": source.get("schema_version", "1.0"),
        "enabled": True,
        "metadata": {
            "name": name,
            "units": (source.get("metadata") or {}).get("units", "SI"),
            "notes": (
                f"Mega-tall Model1 between z = {z_min:g} and {z_max:g} m"
                + (f", element types {sorted(types)}" if types else "")
                + ("" if keep_diaphragms else ", without rigid diaphragms")
                + f": {load_scale:g} times the gravity loads plus {lateral:g} of that "
                "sideways, base fixed. "
                "Built by scripts/megatall_slice.py."
            ),
        },
        "model": source["model"],
        "nodes": nodes,
        "materials": [materials_by_id[i] for i in material_ids],
        "sections": [section_by_id[i] for i in section_ids],
        "elements": elements,
        "mp_constraints": constraints,
        "masses": masses,
        "time_series": time_series,
        "pattern": pattern,
        "loads": loads,
        "analysis": analysis,
        "recorders": [
            {
                "type": "node_displacement",
                "nodes": free_nodes[::record_every],
                "dofs": ALL_DOFS,
                "output": "disp",
            },
            {
                "type": "node_reaction",
                "nodes": fixed_nodes,
                "dofs": [1, 2, 3],
                "output": "reaction",
            },
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="A converted mega-tall case JSON. Default: convert the installed fixture.",
    )
    parser.add_argument(
        "--write-source",
        type=Path,
        default=None,
        help="Only convert the fixture and write it here, for reuse with --source.",
    )
    parser.add_argument("--z-min", type=float, default=None)
    parser.add_argument("--z-max", type=float, default=None)
    parser.add_argument(
        "--types", default="", help="Comma-separated element types to keep."
    )
    parser.add_argument("--no-diaphragm", action="store_true")
    parser.add_argument("--lateral", type=float, default=0.05)
    parser.add_argument("--load-scale", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--record-every", type=int, default=1)
    parser.add_argument("--transient-steps", type=int, default=0)
    parser.add_argument(
        "--accel-start", type=float, default=6.0, help="Seconds into the record."
    )
    parser.add_argument("--accel-scale", type=float, default=1.0)
    parser.add_argument("--name", default="")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.write_source is not None:
        args.write_source.parent.mkdir(parents=True, exist_ok=True)
        args.write_source.write_text(json.dumps(convert_megatall()), encoding="utf-8")
        print(f"Wrote {args.write_source}")
        return 0
    if args.z_min is None or args.z_max is None:
        raise SystemExit("--z-min and --z-max are required")
    if args.z_max <= args.z_min:
        raise SystemExit("--z-max must be above --z-min")
    if args.steps <= 0:
        raise SystemExit("--steps must be positive")

    types = {item.strip() for item in args.types.split(",") if item.strip()} or None
    name = args.name or "megatall_slice_z{:g}_{:g}{}{}".format(
        args.z_min,
        args.z_max,
        "_" + "_".join(sorted(types)) if types else "",
        "_nodiaphragm" if args.no_diaphragm else "",
    ).replace(".", "p")
    if args.load_scale != 1.0:
        name += "_x{:g}".format(args.load_scale).replace(".", "p")
    if args.transient_steps > 0:
        name += f"_eq{args.transient_steps}"
    case = build_slice(
        load_source(args.source.resolve() if args.source else None),
        args.z_min,
        args.z_max,
        types,
        not args.no_diaphragm,
        args.lateral,
        args.load_scale,
        args.steps,
        name,
        max(1, args.record_every),
        args.transient_steps,
        args.accel_start,
        args.accel_scale,
    )
    # The benchmark runner accepts a case JSON only as <name>/<name>.json.
    output = args.output or REPO_ROOT / "benchmark" / ".tmp" / name / f"{name}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(case), encoding="utf-8")
    kinds: dict[str, int] = {}
    for element in case["elements"]:
        kinds[element["type"]] = kinds.get(element["type"], 0) + 1
    print(
        f"Wrote {output}: {len(case['nodes'])} nodes, {kinds}, "
        f"{len(case['mp_constraints'])} diaphragm constraints, {len(case['loads'])} loads"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
