#!/usr/bin/env python3
"""Generate synthetic Strut benchmark cases that isolate one cost center each.

Each family scales along the axis it is meant to stress so the cost of a single
part of the data flow (fiber state determination, global assembly, the linear
solve, constraint handling) can be measured independently of the others.

Families:
  rc_frame2d      N-bay x M-story RC fiber frame, gravity + earthquake.
                  Stresses fiber state determination; DOFs stay modest.
  rc_frame3d      3D RC fiber frame with optional rigid diaphragms.
                  Stresses biaxial fiber sections and MP constraints.
  elastic_frame2d Elastic frame under transient excitation.
                  Stresses global assembly and the linear solve only.
  shell_wall      Elastic shell wall under transient excitation.
                  Stresses sparse storage and factorization (24 DOF elements).

Usage:
  uv run python scripts/gen_synthetic_case.py rc_frame2d --bays 6 --stories 12 \
      --output .tmp/synthetic/rc_frame2d_6x12.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _ground_motion(steps: int, dt: float, peak: float) -> list[float]:
    """Deterministic two-frequency swept motion with a smooth envelope."""
    values = []
    duration = steps * dt
    for i in range(steps + 1):
        t = i * dt
        envelope = math.sin(math.pi * min(t / duration, 1.0)) ** 2
        values.append(
            peak
            * envelope
            * (
                0.65 * math.sin(2.0 * math.pi * 1.1 * t)
                + 0.35 * math.sin(2.0 * math.pi * 3.7 * t)
            )
        )
    return values


def _rc_materials() -> list[dict]:
    # Units: kip, inch, second (matches the OpenSees RC frame examples).
    return [
        {
            "id": 1,
            "type": "Concrete02",
            "params": {
                "fpc": -5.2,
                "epsc0": -0.0028848,
                "fpcu": -1.04,
                "epscu": -0.057697,
                "rat": 0.1,
                "ft": 0.728,
                "Ets": 280.0,
            },
        },
        {
            "id": 2,
            "type": "Concrete02",
            "params": {
                "fpc": -4.0,
                "epsc0": -0.003,
                "fpcu": -0.8,
                "epscu": -0.01,
                "rat": 0.1,
                "ft": 0.56,
                "Ets": 280.0,
            },
        },
        {
            "id": 3,
            "type": "Steel02",
            "params": {
                "Fy": 66.8,
                "E0": 29000.0,
                "b": 0.01,
                "R0": 18.0,
                "cR1": 0.925,
                "cR2": 0.15,
            },
        },
    ]


def _rc_rect_patches_layers(
    half_h: float,
    half_b: float,
    cover: float,
    subdiv: int,
    bars_face: int,
    bars_side: int,
    bar_area: float,
) -> tuple[list[dict], list[dict]]:
    ch = half_h - cover
    cb = half_b - cover
    patches = [
        {
            "type": "quadr",
            "material": 1,
            "num_subdiv_y": subdiv,
            "num_subdiv_z": subdiv,
            "y_i": -ch,
            "z_i": cb,
            "y_j": -ch,
            "z_j": -cb,
            "y_k": ch,
            "z_k": -cb,
            "y_l": ch,
            "z_l": cb,
        },
        {
            "type": "quadr",
            "material": 2,
            "num_subdiv_y": 2,
            "num_subdiv_z": subdiv,
            "y_i": -half_h,
            "z_i": half_b,
            "y_j": -ch,
            "z_j": cb,
            "y_k": ch,
            "z_k": cb,
            "y_l": half_h,
            "z_l": half_b,
        },
        {
            "type": "quadr",
            "material": 2,
            "num_subdiv_y": 2,
            "num_subdiv_z": subdiv,
            "y_i": -ch,
            "z_i": -cb,
            "y_j": -half_h,
            "z_j": -half_b,
            "y_k": half_h,
            "z_k": -half_b,
            "y_l": ch,
            "z_l": -cb,
        },
        {
            "type": "quadr",
            "material": 2,
            "num_subdiv_y": subdiv,
            "num_subdiv_z": 2,
            "y_i": -half_h,
            "z_i": half_b,
            "y_j": -half_h,
            "z_j": -half_b,
            "y_k": -ch,
            "z_k": -cb,
            "y_l": -ch,
            "z_l": cb,
        },
        {
            "type": "quadr",
            "material": 2,
            "num_subdiv_y": subdiv,
            "num_subdiv_z": 2,
            "y_i": ch,
            "z_i": cb,
            "y_j": ch,
            "z_j": -cb,
            "y_k": half_h,
            "z_k": -half_b,
            "y_l": half_h,
            "z_l": half_b,
        },
    ]
    layers = [
        {
            "type": "straight",
            "material": 3,
            "num_bars": bars_side,
            "bar_area": bar_area,
            "y_start": -ch,
            "z_start": cb,
            "y_end": ch,
            "z_end": cb,
        },
        {
            "type": "straight",
            "material": 3,
            "num_bars": bars_side,
            "bar_area": bar_area,
            "y_start": -ch,
            "z_start": -cb,
            "y_end": ch,
            "z_end": -cb,
        },
        {
            "type": "straight",
            "material": 3,
            "num_bars": bars_face,
            "bar_area": bar_area,
            "y_start": ch,
            "z_start": cb,
            "y_end": ch,
            "z_end": -cb,
        },
        {
            "type": "straight",
            "material": 3,
            "num_bars": bars_face,
            "bar_area": bar_area,
            "y_start": -ch,
            "z_start": cb,
            "y_end": -ch,
            "z_end": -cb,
        },
    ]
    return patches, layers


def _staged_gravity_then_quake(
    gravity_element_loads: list[dict],
    gravity_loads: list[dict],
    steps: int,
    dt: float,
    system: str,
    algorithm: str,
    constraints: str,
    rayleigh: dict | None,
    direction: int,
) -> dict:
    gravity_stage = {
        "analysis": {
            "type": "static_nonlinear",
            "constraints": constraints,
            "steps": 10,
            "integrator": {"type": "LoadControl", "step": 0.1},
            "algorithm": "Newton",
            "test_type": "NormDispIncr",
            "tol": 1.0e-8,
            "max_iters": 10,
            "system": system,
            "numberer": "RCM",
        },
        "pattern": {"type": "Plain", "tag": 101, "time_series": 101},
        "load_const": {"time": 0.0},
    }
    if gravity_element_loads:
        gravity_stage["element_loads"] = gravity_element_loads
    if gravity_loads:
        gravity_stage["loads"] = gravity_loads
    quake_stage = {
        "analysis": {
            "type": "transient_nonlinear",
            "constraints": constraints,
            "steps": steps,
            "dt": dt,
            "integrator": {"type": "Newmark", "gamma": 0.5, "beta": 0.25},
            "system": system,
            "numberer": "RCM",
            "algorithm": algorithm,
            "test_type": "EnergyIncr",
            "tol": 1.0e-8,
            "max_iters": 25,
        },
        "pattern": {
            "type": "UniformExcitation",
            "tag": 400,
            "direction": direction,
            "accel": 102,
        },
    }
    if rayleigh:
        quake_stage["rayleigh"] = rayleigh
    return {
        "type": "staged",
        "constraints": constraints,
        "stages": [gravity_stage, quake_stage],
    }


def build_rc_frame2d(args: argparse.Namespace) -> dict:
    bays, stories = args.bays, args.stories
    bay_w, story_h = 288.0, 144.0
    nodes = []

    def nid(i: int, j: int) -> int:
        return j * (bays + 1) + i + 1

    for j in range(stories + 1):
        for i in range(bays + 1):
            node = {"id": nid(i, j), "x": i * bay_w, "y": j * story_h}
            if j == 0:
                node["constraints"] = [1, 2, 3]
            nodes.append(node)

    col_patches, col_layers = _rc_rect_patches_layers(
        12.0, 12.0, 2.5, args.fiber_subdiv, 8, 3, 1.0
    )
    beam_patches, beam_layers = _rc_rect_patches_layers(
        12.0, 9.0, 2.5, args.fiber_subdiv, 6, 1, 1.0
    )
    sections = [
        {
            "id": 1,
            "type": "FiberSection2d",
            "params": {"patches": col_patches, "layers": col_layers},
        },
        {
            "id": 2,
            "type": "FiberSection2d",
            "params": {"patches": beam_patches, "layers": beam_layers},
        },
    ]

    elements = []
    eid = 1
    beam_ids = []
    for i in range(bays + 1):
        for j in range(stories):
            elements.append(
                {
                    "id": eid,
                    "type": args.element,
                    "nodes": [nid(i, j), nid(i, j + 1)],
                    "section": 1,
                    "geomTransf": args.geom_transf,
                    "integration": "Lobatto",
                    "num_int_pts": args.int_pts,
                }
            )
            eid += 1
    for j in range(1, stories + 1):
        for i in range(bays):
            elements.append(
                {
                    "id": eid,
                    "type": args.element,
                    "nodes": [nid(i, j), nid(i + 1, j)],
                    "section": 2,
                    "geomTransf": "Linear",
                    "integration": "Lobatto",
                    "num_int_pts": args.int_pts,
                }
            )
            beam_ids.append(eid)
            eid += 1

    floor_weight_per_len = 0.158
    masses = []
    g = 386.4
    for j in range(1, stories + 1):
        for i in range(bays + 1):
            tributary = bay_w if 0 < i < bays else bay_w / 2.0
            masses.append(
                {
                    "node": nid(i, j),
                    "dof": 1,
                    "value": floor_weight_per_len * tributary / g,
                }
            )

    gravity_element_loads = []
    gravity_loads = []
    if args.gravity == "element":
        gravity_element_loads = [
            {"element": beam_id, "type": "beamUniform", "wy": -floor_weight_per_len}
            for beam_id in beam_ids
        ]
    else:
        beam_set = set(beam_ids)
        nodal = {}
        for element in elements:
            if element["id"] in beam_set:
                for node_id in element["nodes"]:
                    nodal[node_id] = (
                        nodal.get(node_id, 0.0) - floor_weight_per_len * bay_w / 2.0
                    )
        gravity_loads = [
            {"node": node_id, "dof": 2, "value": value}
            for node_id, value in sorted(nodal.items())
        ]
    mp_constraints = []
    if args.rigid_diaphragm:
        # 2D rigid floor: tie every node's horizontal DOF to the first node of its floor.
        for j in range(1, stories + 1):
            for i in range(1, bays + 1):
                mp_constraints.append(
                    {
                        "type": "equalDOF",
                        "retained_node": nid(0, j),
                        "constrained_node": nid(i, j),
                        "dofs": [1],
                    }
                )
    constraints = "Transformation" if mp_constraints else "Plain"
    analysis = _staged_gravity_then_quake(
        gravity_element_loads,
        gravity_loads,
        args.steps,
        args.dt,
        args.system,
        args.algorithm,
        constraints,
        {"alphaM": 0.5, "betaK": 0.0, "betaKInit": 0.0, "betaKComm": 0.00035},
        1,
    )
    case = {
        "schema_version": "1.0",
        "enabled": True,
        "benchmark_size": "large",
        "metadata": {"name": args.name, "units": "kip-in-s"},
        "model": {"ndm": 2, "ndf": 3},
        "nodes": nodes,
        "materials": _rc_materials(),
        "sections": sections,
        "elements": elements,
        "masses": masses,
        "time_series": [
            {"type": "Linear", "tag": 101, "factor": 1.0},
            {
                "type": "Path",
                "tag": 102,
                "factor": 1.0,
                "dt": args.dt,
                "values": _ground_motion(args.steps, args.dt, args.peak_accel * g),
            },
        ],
        "pattern": {"type": "Plain", "tag": 101, "time_series": 101},
        "analysis": analysis,
        "recorders": [
            {
                "type": "node_displacement",
                "nodes": [nid(bays, stories)],
                "dofs": [1, 2, 3],
                "output": "roof_disp",
            },
            {
                "type": "node_displacement",
                "nodes": [nid(0, max(1, stories // 2))],
                "dofs": [1, 2, 3],
                "output": "mid_disp",
            },
        ],
    }
    if mp_constraints:
        case["mp_constraints"] = mp_constraints
    return case


def build_elastic_frame2d(args: argparse.Namespace) -> dict:
    bays, stories = args.bays, args.stories
    bay_w, story_h = 288.0, 144.0
    nodes = []

    def nid(i: int, j: int) -> int:
        return j * (bays + 1) + i + 1

    for j in range(stories + 1):
        for i in range(bays + 1):
            node = {"id": nid(i, j), "x": i * bay_w, "y": j * story_h}
            if j == 0:
                node["constraints"] = [1, 2, 3]
            nodes.append(node)
    elements = []
    eid = 1
    for i in range(bays + 1):
        for j in range(stories):
            elements.append(
                {
                    "id": eid,
                    "type": "elasticBeamColumn2d",
                    "nodes": [nid(i, j), nid(i, j + 1)],
                    "section": 1,
                    "geomTransf": "Linear",
                }
            )
            eid += 1
    for j in range(1, stories + 1):
        for i in range(bays):
            elements.append(
                {
                    "id": eid,
                    "type": "elasticBeamColumn2d",
                    "nodes": [nid(i, j), nid(i + 1, j)],
                    "section": 1,
                    "geomTransf": "Linear",
                }
            )
            eid += 1
    g = 386.4
    masses = []
    for j in range(1, stories + 1):
        for i in range(bays + 1):
            masses.append({"node": nid(i, j), "dof": 1, "value": 0.12})
            masses.append({"node": nid(i, j), "dof": 2, "value": 0.12})
    analysis_type = (
        "transient_nonlinear" if args.nonlinear_driver else "transient_linear"
    )
    analysis = {
        "type": analysis_type,
        "steps": args.steps,
        "dt": args.dt,
        "integrator": {"type": "Newmark", "gamma": 0.5, "beta": 0.25},
        "system": args.system,
        "numberer": "RCM",
    }
    if args.nonlinear_driver:
        analysis.update(
            {
                "algorithm": args.algorithm,
                "test_type": "EnergyIncr",
                "tol": 1.0e-8,
                "max_iters": 25,
            }
        )
    return {
        "schema_version": "1.0",
        "enabled": True,
        "benchmark_size": "large",
        "metadata": {"name": args.name, "units": "kip-in-s"},
        "model": {"ndm": 2, "ndf": 3},
        "nodes": nodes,
        "materials": [],
        "sections": [
            {
                "id": 1,
                "type": "ElasticSection2d",
                "params": {"E": 29000.0, "A": 30.0, "I": 1500.0},
            }
        ],
        "elements": elements,
        "masses": masses,
        "time_series": [
            {
                "type": "Path",
                "tag": 102,
                "factor": 1.0,
                "dt": args.dt,
                "values": _ground_motion(args.steps, args.dt, args.peak_accel * g),
            },
        ],
        "pattern": {
            "type": "UniformExcitation",
            "tag": 400,
            "direction": 1,
            "accel": 102,
        },
        "rayleigh": {
            "alphaM": 0.5,
            "betaK": 0.00035,
            "betaKInit": 0.0,
            "betaKComm": 0.0,
        },
        "analysis": analysis,
        "recorders": [
            {
                "type": "node_displacement",
                "nodes": [nid(bays, stories)],
                "dofs": [1, 2, 3],
                "output": "roof_disp",
            },
        ],
    }


def build_rc_frame3d(args: argparse.Namespace) -> dict:
    bx, bz, stories = args.bays, args.bays_z, args.stories
    bay_w, story_h = 288.0, 144.0
    nodes = []

    def nid(i: int, k: int, j: int) -> int:
        return j * (bx + 1) * (bz + 1) + k * (bx + 1) + i + 1

    for j in range(stories + 1):
        for k in range(bz + 1):
            for i in range(bx + 1):
                node = {
                    "id": nid(i, k, j),
                    "x": i * bay_w,
                    "y": j * story_h,
                    "z": k * bay_w,
                }
                if j == 0:
                    node["constraints"] = [1, 2, 3, 4, 5, 6]
                nodes.append(node)
    next_node = len(nodes) + 1

    col_patches, col_layers = _rc_rect_patches_layers(
        12.0, 12.0, 2.5, args.fiber_subdiv, 8, 3, 1.0
    )
    beam_patches, beam_layers = _rc_rect_patches_layers(
        12.0, 9.0, 2.5, args.fiber_subdiv, 6, 1, 1.0
    )
    shear_modulus, torsion_constant = 1500.0, 20000.0
    sections = [
        {
            "id": 1,
            "type": "FiberSection3d",
            "params": {
                "patches": col_patches,
                "layers": col_layers,
                "G": shear_modulus,
                "J": torsion_constant,
            },
        },
        {
            "id": 2,
            "type": "FiberSection3d",
            "params": {
                "patches": beam_patches,
                "layers": beam_layers,
                "G": shear_modulus,
                "J": torsion_constant,
            },
        },
    ]
    elements = []
    eid = 1
    girder_ids = []
    for k in range(bz + 1):
        for i in range(bx + 1):
            for j in range(stories):
                elements.append(
                    {
                        "id": eid,
                        "type": args.element,
                        "nodes": [nid(i, k, j), nid(i, k, j + 1)],
                        "section": 1,
                        "geomTransf": args.geom_transf,
                        "vecxz": [0.0, 0.0, 1.0],
                        "integration": "Lobatto",
                        "num_int_pts": args.int_pts,
                    }
                )
                eid += 1
    for j in range(1, stories + 1):
        for k in range(bz + 1):
            for i in range(bx):
                elements.append(
                    {
                        "id": eid,
                        "type": args.element,
                        "nodes": [nid(i, k, j), nid(i + 1, k, j)],
                        "section": 2,
                        "geomTransf": "Linear",
                        "vecxz": [0.0, 0.0, 1.0],
                        "integration": "Lobatto",
                        "num_int_pts": args.int_pts,
                    }
                )
                girder_ids.append(eid)
                eid += 1
        for k in range(bz):
            for i in range(bx + 1):
                elements.append(
                    {
                        "id": eid,
                        "type": args.element,
                        "nodes": [nid(i, k, j), nid(i, k + 1, j)],
                        "section": 2,
                        "geomTransf": "Linear",
                        "vecxz": [1.0, 0.0, 0.0],
                        "integration": "Lobatto",
                        "num_int_pts": args.int_pts,
                    }
                )
                girder_ids.append(eid)
                eid += 1

    g = 386.4
    floor_w = 0.158
    masses = []
    mp_constraints = []
    for j in range(1, stories + 1):
        floor_mass = floor_w * bay_w * (bx + bz + 1) / g
        if args.rigid_diaphragm:
            master = next_node
            next_node += 1
            nodes.append(
                {
                    "id": master,
                    "x": bx * bay_w / 2.0,
                    "y": j * story_h,
                    "z": bz * bay_w / 2.0,
                    "constraints": [2, 4, 6],
                }
            )
            masses.append({"node": master, "dof": 1, "value": floor_mass})
            masses.append({"node": master, "dof": 3, "value": floor_mass})
            masses.append(
                {
                    "node": master,
                    "dof": 5,
                    "value": floor_mass * (bx * bay_w) ** 2 / 6.0,
                }
            )
            for k in range(bz + 1):
                for i in range(bx + 1):
                    slave = nid(i, k, j)
                    dx = i * bay_w - bx * bay_w / 2.0
                    dz = k * bay_w - bz * bay_w / 2.0
                    mp_constraints.append(
                        {
                            "type": "rigidDiaphragm",
                            "retained_node": master,
                            "constrained_node": slave,
                            "perp_dirn": 2,
                            "constrained_dofs": [1, 3, 5],
                            "retained_dofs": [1, 3, 5],
                            "matrix": [
                                [1.0, 0.0, dz],
                                [0.0, 1.0, -dx],
                                [0.0, 0.0, 1.0],
                            ],
                            "dx": dx,
                            "dy": 0.0,
                            "dz": dz,
                        }
                    )
        else:
            per_node = floor_mass / ((bx + 1) * (bz + 1))
            for k in range(bz + 1):
                for i in range(bx + 1):
                    masses.append({"node": nid(i, k, j), "dof": 1, "value": per_node})
                    masses.append({"node": nid(i, k, j), "dof": 3, "value": per_node})

    gravity_element_loads = []
    gravity_loads = []
    if args.gravity == "element":
        gravity_element_loads = [
            {"element": girder_id, "type": "beamUniform", "wy": -floor_w, "wz": 0.0}
            for girder_id in girder_ids
        ]
    else:
        girder_set = set(girder_ids)
        nodal = {}
        for element in elements:
            if element["id"] in girder_set:
                for node_id in element["nodes"]:
                    nodal[node_id] = nodal.get(node_id, 0.0) - floor_w * bay_w / 2.0
        gravity_loads = [
            {"node": node_id, "dof": 2, "value": value}
            for node_id, value in sorted(nodal.items())
        ]
    constraints = "Transformation" if args.rigid_diaphragm else "Plain"
    analysis = _staged_gravity_then_quake(
        gravity_element_loads,
        gravity_loads,
        args.steps,
        args.dt,
        args.system,
        args.algorithm,
        constraints,
        {"alphaM": 0.5, "betaK": 0.0, "betaKInit": 0.0, "betaKComm": 0.00035},
        1,
    )
    case = {
        "schema_version": "1.0",
        "enabled": True,
        "benchmark_size": "large",
        "metadata": {"name": args.name, "units": "kip-in-s"},
        "model": {"ndm": 3, "ndf": 6},
        "nodes": nodes,
        "materials": _rc_materials(),
        "sections": sections,
        "elements": elements,
        "masses": masses,
        "time_series": [
            {"type": "Linear", "tag": 101, "factor": 1.0},
            {
                "type": "Path",
                "tag": 102,
                "factor": 1.0,
                "dt": args.dt,
                "values": _ground_motion(args.steps, args.dt, args.peak_accel * g),
            },
        ],
        "pattern": {"type": "Plain", "tag": 101, "time_series": 101},
        "analysis": analysis,
        "recorders": [
            {
                "type": "node_displacement",
                "nodes": [nid(bx, bz, stories)],
                "dofs": [1, 2, 3],
                "output": "roof_disp",
            },
        ],
    }
    if mp_constraints:
        case["mp_constraints"] = mp_constraints
    return case


def build_shell_wall(args: argparse.Namespace) -> dict:
    nx, ny = args.bays, args.stories
    size = 40.0
    nodes = []

    def nid(i: int, j: int) -> int:
        return j * (nx + 1) + i + 1

    for j in range(ny + 1):
        for i in range(nx + 1):
            node = {"id": nid(i, j), "x": i * size, "y": j * size, "z": 0.0}
            if j == 0:
                node["constraints"] = [1, 2, 3, 4, 5, 6]
            nodes.append(node)
    elements = []
    eid = 1
    for j in range(ny):
        for i in range(nx):
            elements.append(
                {
                    "id": eid,
                    "type": "shell",
                    "nodes": [
                        nid(i, j),
                        nid(i + 1, j),
                        nid(i + 1, j + 1),
                        nid(i, j + 1),
                    ],
                    "section": 1,
                }
            )
            eid += 1
    masses = []
    for j in range(1, ny + 1):
        for i in range(nx + 1):
            for dof in (1, 2, 3):
                masses.append({"node": nid(i, j), "dof": dof, "value": 0.01})
    g = 386.4
    return {
        "schema_version": "1.0",
        "enabled": True,
        "benchmark_size": "large",
        "metadata": {"name": args.name, "units": "kip-in-s"},
        "model": {"ndm": 3, "ndf": 6},
        "nodes": nodes,
        "materials": [],
        "sections": [
            {
                "id": 1,
                "type": "ElasticMembranePlateSection",
                "params": {"E": 3600.0, "nu": 0.2, "h": 12.0, "rho": 0.0},
            }
        ],
        "elements": elements,
        "masses": masses,
        "time_series": [
            {
                "type": "Path",
                "tag": 102,
                "factor": 1.0,
                "dt": args.dt,
                "values": _ground_motion(args.steps, args.dt, args.peak_accel * g),
            },
        ],
        "pattern": {
            "type": "UniformExcitation",
            "tag": 400,
            "direction": 3,
            "accel": 102,
        },
        "rayleigh": {
            "alphaM": 0.5,
            "betaK": 0.00035,
            "betaKInit": 0.0,
            "betaKComm": 0.0,
        },
        "analysis": {
            "type": "transient_linear",
            "steps": args.steps,
            "dt": args.dt,
            "integrator": {"type": "Newmark", "gamma": 0.5, "beta": 0.25},
            "system": args.system,
            "numberer": "RCM",
        },
        "recorders": [
            {
                "type": "node_displacement",
                "nodes": [nid(nx, ny)],
                "dofs": [1, 2, 3],
                "output": "top_disp",
            },
        ],
    }


BUILDERS = {
    "rc_frame2d": build_rc_frame2d,
    "rc_frame3d": build_rc_frame3d,
    "elastic_frame2d": build_elastic_frame2d,
    "shell_wall": build_shell_wall,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("family", choices=sorted(BUILDERS))
    parser.add_argument("--bays", type=int, default=3)
    parser.add_argument("--bays-z", type=int, default=3)
    parser.add_argument("--stories", type=int, default=3)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument(
        "--peak-accel", type=float, default=0.4, help="Peak ground acceleration in g."
    )
    parser.add_argument(
        "--fiber-subdiv",
        type=int,
        default=20,
        help="Core patch subdivisions per direction.",
    )
    parser.add_argument("--int-pts", type=int, default=5)
    parser.add_argument(
        "--element", default=None, help="Beam-column element type (default by family)."
    )
    parser.add_argument(
        "--geom-transf", default="Linear", choices=["Linear", "PDelta", "Corotational"]
    )
    parser.add_argument("--system", default="BandGeneral")
    parser.add_argument("--algorithm", default="Newton")
    parser.add_argument(
        "--rigid-diaphragm",
        action="store_true",
        help="Tie each floor with multi-point constraints (equalDOF in 2D, rigidDiaphragm in 3D).",
    )
    parser.add_argument(
        "--gravity",
        default="element",
        choices=["element", "nodal"],
        help="Apply floor gravity as beam element loads or as equivalent nodal loads.",
    )
    parser.add_argument(
        "--nonlinear-driver",
        action="store_true",
        help="elastic_frame2d only: run through the transient_nonlinear driver.",
    )
    parser.add_argument("--name", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.element is None:
        args.element = (
            "forceBeamColumn3d" if args.family == "rc_frame3d" else "forceBeamColumn2d"
        )
    if args.name is None:
        args.name = f"synthetic_{args.family}_{args.bays}x{args.stories}"
    case = BUILDERS[args.family](args)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(case) + "\n", encoding="utf-8")
    ndf = case["model"]["ndf"]
    print(
        f"{args.name}: nodes={len(case['nodes'])} elements={len(case['elements'])} dofs~{len(case['nodes']) * ndf}"
    )


if __name__ == "__main__":
    main()
