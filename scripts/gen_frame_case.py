#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

def build_frame_case(
    bays: int,
    stories: int,
    bay_width: float,
    story_height: float,
    E: float,
    A: float,
    I: float,
    load_per_node: float,
    name: str,
    enabled: bool,
):
    nodes = []
    node_id = 1
    for j in range(stories + 1):
        y = j * story_height
        for i in range(bays + 1):
            x = i * bay_width
            node = {"id": node_id, "x": x, "y": y}
            if j == 0:
                node["constraints"] = [1, 2, 3]
            nodes.append(node)
            node_id += 1

    def nid(i: int, j: int) -> int:
        return j * (bays + 1) + i + 1

    elements = []
    elem_id = 1

    # Columns
    for i in range(bays + 1):
        for j in range(stories):
            elements.append(
                {
                    "id": elem_id,
                    "type": "elasticBeamColumn2d",
                    "nodes": [nid(i, j), nid(i, j + 1)],
                    "section": 1,
                    "geomTransf": "Linear",
                }
            )
            elem_id += 1

    # Beams (skip ground level)
    for j in range(1, stories + 1):
        for i in range(bays):
            elements.append(
                {
                    "id": elem_id,
                    "type": "elasticBeamColumn2d",
                    "nodes": [nid(i, j), nid(i + 1, j)],
                    "section": 1,
                    "geomTransf": "Linear",
                }
            )
            elem_id += 1

    loads = []
    top_j = stories
    for i in range(bays + 1):
        loads.append({"node": nid(i, top_j), "dof": 1, "value": load_per_node})

    top_nodes = [nid(i, top_j) for i in range(bays + 1)]

    data = {
        "schema_version": "1.0",
        "enabled": enabled,
        "status": "active",
        "metadata": {"name": name, "units": "SI"},
        "opensees": {
            "model_builder": {"type": "BasicBuilder", "ndm": 2, "ndf": 3},
            "time_series": {"type": "Linear", "tag": 1},
            "pattern": {"type": "Plain", "tag": 1, "time_series": 1},
            "analysis": {
                "constraints": "Plain",
                "numberer": "RCM",
                "system": "BandGeneral",
                "algorithm": "Linear",
                "integrator": "LoadControl 1.0",
                "analysis": "Static",
            },
        },
        "model": {"ndm": 2, "ndf": 3},
        "nodes": nodes,
        "materials": [{"id": 1, "type": "Elastic", "params": {"E": E}}],
        "sections": [
            {
                "id": 1,
                "type": "ElasticSection2d",
                "params": {"E": E, "A": A, "I": I},
            }
        ],
        "elements": elements,
        "loads": loads,
        "analysis": {"type": "static_linear", "steps": 1},
        "recorders": [
            {
                "type": "node_displacement",
                "nodes": top_nodes,
                "dofs": [1, 2, 3],
                "output": "node_disp",
            }
        ],
    }
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a 2D elastic frame JSON case")
    parser.add_argument("--bays", type=int, required=True)
    parser.add_argument("--stories", type=int, required=True)
    parser.add_argument("--bay-width", type=float, default=4.0)
    parser.add_argument("--story-height", type=float, default=3.0)
    parser.add_argument("--E", type=float, default=200000000000.0)
    parser.add_argument("--A", type=float, default=0.02)
    parser.add_argument("--I", type=float, default=0.00008)
    parser.add_argument("--load", type=float, default=6000.0)
    parser.add_argument("--name", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--disabled", action="store_true")
    args = parser.parse_args()

    name = args.name or f"elastic_frame_{args.bays}bay_{args.stories}story"
    enabled = not args.disabled
    data = build_frame_case(
        bays=args.bays,
        stories=args.stories,
        bay_width=args.bay_width,
        story_height=args.story_height,
        E=args.E,
        A=args.A,
        I=args.I,
        load_per_node=args.load,
        name=name,
        enabled=enabled,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
