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
    element_type: str,
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
                    "type": element_type,
                    "nodes": [nid(i, j), nid(i, j + 1)],
                    "section": 1,
                    "geomTransf": "Linear",
                }
            )
            if element_type == "forceBeamColumn2d":
                elements[-1]["integration"] = "Lobatto"
                elements[-1]["num_int_pts"] = 3
            elem_id += 1

    # Beams (skip ground level)
    for j in range(1, stories + 1):
        for i in range(bays):
            elements.append(
                {
                    "id": elem_id,
                    "type": element_type,
                    "nodes": [nid(i, j), nid(i + 1, j)],
                    "section": 1,
                    "geomTransf": "Linear",
                }
            )
            if element_type == "forceBeamColumn2d":
                elements[-1]["integration"] = "Lobatto"
                elements[-1]["num_int_pts"] = 3
            elem_id += 1

    loads = []
    top_j = stories
    for i in range(bays + 1):
        loads.append({"node": nid(i, top_j), "dof": 1, "value": load_per_node})

    top_nodes = [nid(i, top_j) for i in range(bays + 1)]

    sections = [
        {
            "id": 1,
            "type": "ElasticSection2d",
            "params": {"E": E, "A": A, "I": I},
        }
    ]
    analysis = {"type": "static_linear", "steps": 1}
    if element_type == "forceBeamColumn2d":
        sections = [
            {
                "id": 1,
                "type": "FiberSection2d",
                "params": {
                    "patches": [
                        {
                            "type": "rect",
                            "material": 1,
                            "num_subdiv_y": 4,
                            "num_subdiv_z": 2,
                            "y_i": -0.2,
                            "z_i": -0.1,
                            "y_j": 0.2,
                            "z_j": 0.1,
                        }
                    ],
                    "layers": [],
                },
            }
        ]
        analysis = {
            "type": "static_nonlinear",
            "steps": 1,
            "max_iters": 80,
            "tol": 1.0e-7,
            "rel_tol": 1.0e-8,
            "integrator": {"type": "LoadControl"},
        }

    data = {
        "schema_version": "1.0",
        "enabled": enabled,
        "status": "benchmark",
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
        "sections": sections,
        "elements": elements,
        "loads": loads,
        "analysis": analysis,
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
    parser.add_argument(
        "--element-type",
        choices=["elasticBeamColumn2d", "forceBeamColumn2d"],
        default="elasticBeamColumn2d",
    )
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
        element_type=args.element_type,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
