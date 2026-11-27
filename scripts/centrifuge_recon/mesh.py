"""GiD's warped quad mesh in original tag and command order."""

from __future__ import annotations

import lzma
import struct
from pathlib import Path
from typing import Iterator

MESH_FILE = Path(__file__).with_name("mesh.bin")
LOAM_TAGS = (*range(1, 417), *range(3357, 3773))
SAND_TAGS = (*range(417, 3357), *range(3773, 3933))
ELEMENT_TAGS = LOAM_TAGS + SAND_TAGS
Point = tuple[int, int]
Quad = tuple[int, int, int, int]


def quad_node_predictions(
    quads: list[Quad], points: dict[int, Point]
) -> Iterator[tuple[int, Point]]:
    """Visit each node once, predicting the missing corner of a parallelogram."""
    previous_point = (0, 0)
    for quad in quads:
        for index, node in enumerate(quad):
            if node in points:
                continue
            previous, opposite, following = (
                points.get(quad[(index + offset) % 4]) for offset in (3, 2, 1)
            )
            prediction = previous_point
            if previous is not None and opposite is not None and following is not None:
                prediction = (
                    previous[0] + following[0] - opposite[0],
                    previous[1] + following[1] - opposite[1],
                )
            yield node, prediction
            previous_point = points[node]


def read_mesh() -> tuple[dict[int, Point], list[Quad]]:
    data = lzma.decompress(MESH_FILE.read_bytes())
    connectivity_bytes = len(ELEMENT_TAGS) * struct.calcsize("<4h")
    quads = []
    first = 0
    for step, d1, d2, d3 in struct.iter_unpack("<4h", data[:connectivity_bytes]):
        first += step
        second = first + d1
        third = second + d2
        quads.append((first, second, third, third + d3))

    residuals = iter(struct.iter_unpack("<2i", data[connectivity_bytes:]))
    points = {}
    for node, (x, y) in quad_node_predictions(quads, points):
        dx, dy = next(residuals)
        points[node] = (x + dx, y + dy)
    if set(points) != set(range(1, 4205)) or next(residuals, None) is not None:
        raise ValueError("centrifuge mesh has inconsistent node or residual counts")
    return points, quads


def laminar_box_nodes() -> Iterator[tuple[int, Point]]:
    for offset in (0, 16):
        for side, x in enumerate((35114, 293110)):
            for ring in range(8):
                yield 4205 + offset + 8 * side + ring, (x, 83576 - 1250 * ring)
