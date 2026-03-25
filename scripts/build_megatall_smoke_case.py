#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = (
    REPO_ROOT / "tests" / "validation" / "opensees_megatall_building_model1_dynamiccpu"
)
DEFAULT_INPUT = CASE_DIR / "generated" / "case.json"
DEFAULT_OUTPUT = CASE_DIR / "megatall_smoke.json"

# Deterministic reduced megatall probe:
# - one extracted megatall beam carries the staged gravity/transient response
# - one extracted shell, one extracted truss, and one rigidDiaphragm remain present
# - the shell/truss/MPC probe is fully fixed so the smoke path stays stable
FREE_DYNAMIC_NODE_ID = 5404
FIXED_NODE_IDS = {1455, 2718, 3174, 17483, 17484}
SMOKE_NODE_IDS = FIXED_NODE_IDS | {FREE_DYNAMIC_NODE_ID}
SMOKE_ELEMENT_IDS = {743, 18754, 33085}
GRAVITY_STEPS = 2
TRANSIENT_STEPS = 5
GRAVITY_STEP_SIZE = 1.0 / GRAVITY_STEPS


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _collect_time_series_tags(data: dict) -> set[int]:
    tags: set[int] = set()
    pattern = data.get("pattern")
    if isinstance(pattern, dict):
        tag = pattern.get("time_series")
        if isinstance(tag, int):
            tags.add(tag)

    analysis = data.get("analysis", {})
    if isinstance(analysis, dict):
        for stage in analysis.get("stages", []):
            if not isinstance(stage, dict):
                continue
            stage_pattern = stage.get("pattern")
            if isinstance(stage_pattern, dict):
                ts_tag = stage_pattern.get("time_series")
                if isinstance(ts_tag, int):
                    tags.add(ts_tag)
                accel_tag = stage_pattern.get("accel")
                if isinstance(accel_tag, int):
                    tags.add(accel_tag)
    return tags


def _collect_material_ids_from_section(section: dict) -> set[int]:
    params = section.get("params", {})
    section_type = section.get("type")
    material_ids: set[int] = set()
    if section_type == "FiberSection3d":
        for fiber in params.get("fibers", []):
            material_ids.add(int(fiber["material"]))
        for patch in params.get("patches", []):
            if "material" in patch:
                material_ids.add(int(patch["material"]))
        for layer in params.get("layers", []):
            if "material" in layer:
                material_ids.add(int(layer["material"]))
        return material_ids
    if section_type == "LayeredShellSection":
        for layer in params.get("layers", []):
            material_ids.add(int(layer["material"]))
        return material_ids
    if section_type == "ElasticSection3d":
        return material_ids
    raise ValueError(f"unsupported megatall smoke section type: {section_type}")


def _plate_from_plane_stress_surrogate(
    material_id: int, material: dict, base_material: dict
) -> dict:
    props = list(base_material.get("params", {}).get("props", []))
    if not props:
        raise ValueError(
            f"PlaneStressUserMaterial {base_material['id']} is missing props for smoke surrogate"
        )
    nu = float(props[-1])
    gmod = float(material["params"]["gmod"])
    rho = float(base_material.get("params", {}).get("rho", 0.0))
    return {
        "id": material_id,
        "type": "ElasticIsotropic",
        "params": {
            "E": 2.0 * gmod * (1.0 + nu),
            "nu": nu,
            "rho": rho,
        },
    }


def _uniaxial_initial_modulus(material_by_id: dict[int, dict], material_id: int) -> float:
    material = material_by_id.get(material_id)
    if material is None:
        return 1.0
    params = material.get("params", {})
    material_type = material.get("type")
    if material_type == "Steel02":
        return float(params["E0"])
    if material_type == "Concrete01":
        epsc0 = float(params["epsc0"])
        if epsc0 == 0.0:
            return 1.0
        return abs(2.0 * float(params["fpc"]) / epsc0)
    if material_type in {"Elastic", "ElasticPP"}:
        return float(params["E"])
    return 1.0


def _fiber_section_to_elastic_surrogate(
    section: dict, material_by_id: dict[int, dict]
) -> dict:
    params = section.get("params", {})
    cells: list[tuple[float, float, float, int]] = []

    for patch in params.get("patches", []):
        patch_type = patch.get("type")
        if patch_type != "rect":
            raise ValueError(
                f"unsupported megatall smoke FiberSection3d patch type: {patch_type}"
            )
        num_subdiv_y = int(patch["num_subdiv_y"])
        num_subdiv_z = int(patch["num_subdiv_z"])
        y_i = float(patch["y_i"])
        z_i = float(patch["z_i"])
        y_j = float(patch["y_j"])
        z_j = float(patch["z_j"])
        dy = (y_j - y_i) / num_subdiv_y
        dz = (z_j - z_i) / num_subdiv_z
        area = abs(dy * dz)
        material_id = int(patch["material"])
        for iy in range(num_subdiv_y):
            y = y_i + (iy + 0.5) * dy
            for iz in range(num_subdiv_z):
                z = z_i + (iz + 0.5) * dz
                cells.append((y, z, area, material_id))

    for layer in params.get("layers", []):
        layer_type = layer.get("type")
        if layer_type != "straight":
            raise ValueError(
                f"unsupported megatall smoke FiberSection3d layer type: {layer_type}"
            )
        num_bars = int(layer["num_bars"])
        bar_area = float(layer["bar_area"])
        y_start = float(layer["y_start"])
        z_start = float(layer["z_start"])
        y_end = float(layer["y_end"])
        z_end = float(layer["z_end"])
        material_id = int(layer["material"])
        if num_bars == 1:
            cells.append((y_start, z_start, bar_area, material_id))
            continue
        for index in range(num_bars):
            ratio = index / (num_bars - 1)
            y = y_start + (y_end - y_start) * ratio
            z = z_start + (z_end - z_start) * ratio
            cells.append((y, z, bar_area, material_id))

    if not cells:
        raise ValueError(
            f"FiberSection3d {section['id']} produced no cells for smoke surrogate"
        )

    axial = 0.0
    moment_y = 0.0
    moment_z = 0.0
    for y, z, area, material_id in cells:
        modulus = _uniaxial_initial_modulus(material_by_id, material_id)
        axial += modulus * area
        moment_y += modulus * area * z * z
        moment_z += modulus * area * y * y

    gmod = float(params.get("G", 1.0))
    torsion_constant = float(params.get("J", 1.0))
    return {
        "id": int(section["id"]),
        "type": "ElasticSection3d",
        "params": {
            # Use stiffness-equivalent section products directly for the smoke surrogate.
            "E": 1.0,
            "A": max(axial, 1.0),
            "Iy": max(moment_y, 1.0),
            "Iz": max(moment_z, 1.0),
            "G": 1.0,
            "J": max(gmod * torsion_constant, 1.0),
        },
    }


def _collect_material_closure(
    data: dict, section_ids: set[int], seed_material_ids: set[int]
) -> tuple[list[dict], list[dict]]:
    section_by_id = {int(section["id"]): section for section in data["sections"]}
    material_by_id = {int(material["id"]): material for material in data["materials"]}

    needed_section_ids = sorted(section_ids)
    needed_material_ids: set[int] = set(seed_material_ids)
    for section_id in needed_section_ids:
        needed_material_ids.update(
            _collect_material_ids_from_section(section_by_id[section_id])
        )

    sections: list[dict] = []
    for section_id in needed_section_ids:
        section = copy.deepcopy(section_by_id[section_id])
        if section.get("type") == "FiberSection3d":
            section = _fiber_section_to_elastic_surrogate(section, material_by_id)
        sections.append(section)

    emitted_materials: dict[int, dict] = {}
    queue = list(needed_material_ids)
    while queue:
        material_id = queue.pop()
        if material_id in emitted_materials:
            continue
        material = material_by_id[material_id]
        ref = material.get("params", {}).get("material")
        if material.get("type") == "PlateFromPlaneStress" and isinstance(ref, int):
            base_material = material_by_id[ref]
            if base_material.get("type") == "PlaneStressUserMaterial":
                emitted_materials[material_id] = _plate_from_plane_stress_surrogate(
                    material_id, material, base_material
                )
                continue
        emitted_materials[material_id] = copy.deepcopy(material)
        if isinstance(ref, int):
            queue.append(ref)

    materials = [
        emitted_materials[material_id] for material_id in sorted(emitted_materials)
    ]
    return sections, materials


def _build_recorders() -> list[dict]:
    return [
        {
            "type": "node_reaction",
            "nodes": [3174],
            "dofs": [3],
            "output": "base_reaction",
            "raw_path": "base_reaction.txt",
            "include_time": True,
        },
        {
            "type": "node_displacement",
            "nodes": [FREE_DYNAMIC_NODE_ID],
            "dofs": [3],
            "output": "tip_disp",
            "raw_path": "tip_disp.txt",
            "include_time": True,
        },
        {
            "type": "drift",
            "i_node": 3174,
            "j_node": FREE_DYNAMIC_NODE_ID,
            "dof": 3,
            "perp_dirn": 2,
            "output": "tip_drift",
            "raw_path": "tip_drift.txt",
            "include_time": False,
        },
    ]


def _build_fixed_probe_rigid_diaphragm(node_by_id: dict[int, dict]) -> dict:
    retained = 1455
    constrained = 2718
    dx = float(node_by_id[constrained]["x"]) - float(node_by_id[retained]["x"])
    dy = float(node_by_id[constrained]["y"]) - float(node_by_id[retained]["y"])
    return {
        "type": "rigidDiaphragm",
        "retained_node": retained,
        "constrained_node": constrained,
        "perp_dirn": 3,
        "constrained_dofs": [1, 2, 6],
        "retained_dofs": [1, 2, 6],
        "matrix": [
            [1.0, 0.0, -dy],
            [0.0, 1.0, dx],
            [0.0, 0.0, 1.0],
        ],
        "dx": dx,
        "dy": dy,
        "dz": 0.0,
    }


def build_smoke_case(input_path: Path = DEFAULT_INPUT) -> dict:
    data = _load_json(input_path)
    node_by_id = {int(node["id"]): node for node in data["nodes"]}
    element_by_id = {int(element["id"]): element for element in data["elements"]}
    direct_material_ids = {179}
    section_ids = {
        int(element["section"])
        for element in element_by_id.values()
        if int(element["id"]) in SMOKE_ELEMENT_IDS and "section" in element
    }
    sections, materials = _collect_material_closure(data, section_ids, direct_material_ids)

    nodes = []
    for node_id in sorted(SMOKE_NODE_IDS):
        node = copy.deepcopy(node_by_id[node_id])
        if node_id == FREE_DYNAMIC_NODE_ID:
            node["constraints"] = [1, 2, 4, 5, 6]
        else:
            node["constraints"] = [1, 2, 3, 4, 5, 6]
        nodes.append(node)

    elements = [
        copy.deepcopy(element_by_id[element_id]) for element_id in sorted(SMOKE_ELEMENT_IDS)
    ]

    masses = [
        copy.deepcopy(mass)
        for mass in data.get("masses", [])
        if int(mass["node"]) == FREE_DYNAMIC_NODE_ID and int(mass["dof"]) == 3
    ]
    loads = [
        copy.deepcopy(load)
        for load in data.get("loads", [])
        if int(load["node"]) == FREE_DYNAMIC_NODE_ID and int(load["dof"]) == 3
    ]

    time_series_tags = _collect_time_series_tags(data)
    time_series = [
        copy.deepcopy(series)
        for series in data.get("time_series", [])
        if int(series["tag"]) in time_series_tags
    ]

    analysis = copy.deepcopy(data["analysis"])
    gravity_stage = analysis["stages"][0]
    gravity_stage["loads"] = [
        copy.deepcopy(load)
        for load in gravity_stage.get("loads", [])
        if int(load["node"]) == FREE_DYNAMIC_NODE_ID and int(load["dof"]) == 3
    ]
    gravity_stage["analysis"]["steps"] = GRAVITY_STEPS
    gravity_stage["analysis"]["integrator"]["step"] = GRAVITY_STEP_SIZE
    gravity_stage["analysis"]["algorithm"] = "Newton"
    gravity_stage["analysis"]["system"] = "SuperLU"
    transient_stage = analysis["stages"][1]
    transient_stage["analysis"]["steps"] = TRANSIENT_STEPS
    transient_stage["analysis"]["system"] = "SuperLU"
    if transient_stage.get("pattern", {}).get("type") == "UniformExcitation":
        transient_stage["pattern"]["direction"] = 3

    source_example = Path(data["source_example"])
    if source_example.is_absolute():
        source_rel = source_example.relative_to(REPO_ROOT)
    else:
        source_rel = source_example

    case = {
        "schema_version": data.get("schema_version", "1.0"),
        "enabled": False,
        "status": "generated_smoke",
        "notes": (
            "Deterministic extracted megatall smoke case derived from the generated native "
            "case. It keeps one real megatall frame member on the staged gravity/transient "
            "path, plus fixed shell, truss, and rigidDiaphragm probes. The extracted "
            "FiberSection3d frame member is converted to a stiffness-equivalent "
            "ElasticSection3d surrogate so the reduced smoke path remains mechanically "
            "stable while still exercising shells, staged analysis, UniformExcitation, "
            "and rigidDiaphragm loading."
        ),
        "metadata": {
            "name": "strut_dynamiccpu_entry_smoke",
            "units": data.get("metadata", {}).get("units", "unknown"),
            "source_case": str(input_path.relative_to(REPO_ROOT)),
            "free_dynamic_node": FREE_DYNAMIC_NODE_ID,
            "smoke_element_ids": sorted(SMOKE_ELEMENT_IDS),
        },
        "model": copy.deepcopy(data["model"]),
        "nodes": nodes,
        "materials": materials,
        "sections": sections,
        "elements": elements,
        "masses": masses,
        "time_series": time_series,
        "analysis": analysis,
        "recorders": _build_recorders(),
        "source_example": str(source_rel),
        "pattern": copy.deepcopy(data["pattern"]),
        "loads": loads,
        "mp_constraints": [_build_fixed_probe_rigid_diaphragm(node_by_id)],
    }

    for series in case["time_series"]:
        if series.get("type") in {"Path", "PathFile"}:
            values_path = Path(series["values_path"])
            if values_path.is_absolute():
                series["values_path"] = str(values_path.relative_to(input_path.parent))

    return case


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    case = build_smoke_case(args.input.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(case, indent=2) + "\n", encoding="utf-8")

    summary = {
        "nodes": len(case["nodes"]),
        "elements": len(case["elements"]),
        "sections": len(case["sections"]),
        "materials": len(case["materials"]),
        "masses": len(case["masses"]),
        "loads": len(case["loads"]),
        "mp_constraints": len(case["mp_constraints"]),
        "recorders": len(case["recorders"]),
        "gravity_steps": case["analysis"]["stages"][0]["analysis"]["steps"],
        "transient_steps": case["analysis"]["stages"][1]["analysis"]["steps"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
