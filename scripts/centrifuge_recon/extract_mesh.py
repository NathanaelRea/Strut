"""Import the irregular mesh table from an authenticated original without running Tcl."""

from __future__ import annotations

import argparse
import lzma
import struct
from decimal import Decimal
from pathlib import Path

from .mesh import ELEMENT_TAGS, quad_node_predictions
from .source import read_original


def extract_mesh(source: str) -> bytes:
    points = {}
    quads = []
    tags = []
    for line in source.splitlines():
        words = line.split()
        if words[:1] == ["node"]:
            _, tag, x, y = words
            points[int(tag)] = (int(Decimal(x) * 10000), int(Decimal(y) * 10000))
        if words[:2] in (["element", "quad"], ["element", "quadUP"]):
            tags.append(int(words[2]))
            quads.append(tuple(map(int, words[3:7])))
    if tuple(tags) != ELEMENT_TAGS:
        raise ValueError("unexpected element ordering")

    data = bytearray()
    first = 0
    for a, b, c, d in quads:
        data.extend(struct.pack("<4h", a - first, b - a, c - b, d - c))
        first = a

    decoded = {}
    for node, (x, y) in quad_node_predictions(quads, decoded):
        actual_x, actual_y = decoded[node] = points[node]
        data.extend(struct.pack("<2i", actual_x - x, actual_y - y))
    return lzma.compress(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_bytes(extract_mesh(read_original(args.original)))


if __name__ == "__main__":
    main()
