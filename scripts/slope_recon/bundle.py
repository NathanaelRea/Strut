"""Rebuild the OpenSees Wiki effective-stress slope example."""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
from pathlib import Path
from tempfile import NamedTemporaryFile

HERE = Path(__file__).resolve().parent
FIXTURE_DIR = (
    HERE.parents[1]
    / "tests/fixtures/opensees/OpenSeesExamplesPractical"
    / "dynamic_2d_effective_stress_analysis_of_slope"
)
ENTRY = "SlopeAnalysis2Deff.tcl"
UPSTREAM_SHA256 = "f7fb50b184803c7589d2de3d372198e5b244a23461679f4e070f8a250b18ff76"


def _node_row(node: tuple[int, float, float]) -> str:
    tag, x, y = node
    return f"{tag:8d}{x:12.3f}{y:11.3f}"


def model_bytes() -> bytes:
    with lzma.open(HERE / "mesh.json.xz", "rt", encoding="ascii") as file:
        mesh = json.load(file)
    pressure = [_node_row(node) for node in mesh["pressure"]]
    solid = [_node_row(node) for node in mesh["solid"]]
    element_lines = []
    element_info = []
    permeability = []
    for tag, *nodes, material in mesh["elements"]:
        connectivity = " ".join(map(str, nodes))
        element_lines.append(
            f"element 9_4_QuadUP {tag} {connectivity} "
            f"$thick{material} {material} $uBulk{material} 1.0 1.0 1.0 "
            f"$xWgt{material} $yWgt{material}"
        )
        corners = "".join(f"{node:10d}" for node in nodes[:4])
        element_info.append(f"{tag:8d}{corners}")
        parameter = 10000 + (tag - 1) * 2
        permeability.extend(
            f"updateParameter {parameter + offset} ${axis}Perm{material}"
            for offset, axis in enumerate(("v", "h"), start=1)
        )
    blocks = {
        "@PRESSURE_NODES@": [f"node {row}" for row in pressure],
        "@PORE_PRESSURE_INFO@": [f'puts $ppNodesInfo "{row}"' for row in pressure],
        "@SOLID_NODES@": [f"node {row}" for row in solid],
        "@ELEMENTS@": element_lines,
        "@MESH_NODES@": [f'puts $meshFile "{row}"' for row in pressure],
        "@MESH_ELEMENTS@": [f'puts $meshFile "{row}"' for row in element_info],
        "@ELEMENT_INFO@": [f'puts $eleFile "{row}"' for row in element_info],
        "@PERMEABILITY_UPDATES@": permeability,
    }
    template = (HERE / "model.tcl.in").read_text(encoding="ascii")
    for marker, rows in blocks.items():
        template = template.replace(marker, "\n".join(rows))
    return template.replace("\n", "\r\n").encode("ascii")


def bundle_is_current(target: Path = FIXTURE_DIR) -> bool:
    entry = target / ENTRY
    return (
        entry.is_file()
        and hashlib.sha256(entry.read_bytes()).hexdigest() == UPSTREAM_SHA256
    )


def ensure_bundle(target: Path = FIXTURE_DIR) -> bool:
    if bundle_is_current(target):
        return False
    script = model_bytes()
    if hashlib.sha256(script).hexdigest() != UPSTREAM_SHA256:
        raise ValueError("generated slope model differs from the pinned upstream Tcl")
    target.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=target, delete=False) as staged:
        staged.write(script)
    Path(staged.name).replace(target / ENTRY)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    location = parser.add_mutually_exclusive_group(required=True)
    location.add_argument("--install", action="store_true")
    location.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    ensure_bundle(args.out_dir or FIXTURE_DIR)


if __name__ == "__main__":
    main()
