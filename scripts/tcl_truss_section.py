"""Reduce a fiber section in uniform axial strain to area-weighted materials."""

import math
from collections import defaultdict
from typing import Any


def fiber_material_areas(params: dict[str, Any]) -> dict[int, float]:
    areas: dict[int, float] = defaultdict(float)
    for patch in params.get("patches", []):
        kind = patch["type"]
        if kind == "rect":
            area = abs((patch["y_j"] - patch["y_i"]) * (patch["z_j"] - patch["z_i"]))
        elif kind == "circ":
            angle = math.radians(
                patch.get("end_ang", 360.0) - patch.get("start_ang", 0.0)
            )
            area = angle / 2.0 * (patch["ext_rad"] ** 2 - patch["int_rad"] ** 2)
        else:
            vertices = [(patch[f"y_{i}"], patch[f"z_{i}"]) for i in "ijkl"]
            area = 0.5 * sum(
                a[0] * b[1] - a[1] * b[0]
                for a, b in zip(vertices, vertices[1:] + vertices[:1])
            )
        areas[patch["material"]] += area
    for layer in params.get("layers", []):
        areas[layer["material"]] += layer["num_bars"] * layer["bar_area"]
    for fiber in params.get("fibers", []):
        areas[fiber["material"]] += fiber["area"]
    if not areas:
        raise ValueError("truss fiber section requires at least one fiber")
    return dict(areas)
