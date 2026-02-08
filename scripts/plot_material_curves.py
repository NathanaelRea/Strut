#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
except ModuleNotFoundError as exc:
    raise SystemExit(
        "matplotlib is required for plotting. Install it with "
        "`uv add --dev matplotlib` or `pip install matplotlib`."
    ) from exc


def run(cmd: List[str]) -> None:
    subprocess.check_call(cmd)


def material_definitions() -> List[dict]:
    return [
        {
            "name": "steel01",
            "type": "Steel01",
            "params": {"Fy": 250.0, "E0": 200000.0, "b": 0.01},
            "units": "SI",
            "steps": 100,
            "load_tension": 1000.0,
            "load_compression": -1000.0,
        }
    ]


def build_case(
    material: dict,
    direction: str,
    load_value: float,
) -> dict:
    name = f"{material['name']}_curve_{direction}"
    return {
        "schema_version": "1.0",
        "enabled": True,
        "metadata": {
            "name": name,
            "units": material.get("units", ""),
        },
        "model": {"ndm": 2, "ndf": 2},
        "nodes": [
            {"id": 1, "x": 0.0, "y": 0.0, "constraints": [1, 2]},
            {"id": 2, "x": 1.0, "y": 0.0, "constraints": [2]},
        ],
        "materials": [
            {"id": 1, "type": material["type"], "params": material["params"]}
        ],
        "elements": [
            {
                "id": 1,
                "type": "truss",
                "nodes": [1, 2],
                "area": 1.0,
                "material": 1,
            }
        ],
        "loads": [{"node": 2, "dof": 1, "value": load_value}],
        "analysis": {
            "type": "static_nonlinear",
            "steps": int(material.get("steps", 100)),
            "max_iters": 30,
            "tol": 1.0e-10,
            "rel_tol": 1.0e-8,
        },
        "recorders": [
            {
                "type": "node_displacement",
                "nodes": [2],
                "dofs": [1],
                "output": "node_disp",
            }
        ],
    }


def case_geometry(data: dict) -> Tuple[float, float]:
    nodes = {int(node["id"]): node for node in data.get("nodes", [])}
    elements = data.get("elements", [])
    if not elements:
        raise ValueError("case has no elements")
    elem = elements[0]
    node_ids = elem.get("nodes")
    if not node_ids or len(node_ids) < 2:
        raise ValueError("element missing nodes")
    n1 = nodes[int(node_ids[0])]
    n2 = nodes[int(node_ids[1])]
    x1 = float(n1["x"])
    y1 = float(n1["y"])
    x2 = float(n2["x"])
    y2 = float(n2["y"])
    length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    area = float(elem.get("area", 1.0))
    return length, area


def case_load_and_steps(data: dict) -> Tuple[float, int]:
    loads = data.get("loads", [])
    if not loads:
        raise ValueError("case has no loads")
    load = loads[0]
    load_value = float(load["value"])
    steps = int(data.get("analysis", {}).get("steps", 1))
    return load_value, steps


def recorder_output_file(data: dict) -> str:
    recorders = data.get("recorders", [])
    for rec in recorders:
        if rec.get("type") == "node_displacement":
            output = rec.get("output", "node_disp")
            nodes = rec.get("nodes", [])
            if not nodes:
                raise ValueError("node_displacement recorder missing nodes")
            node_id = int(nodes[0])
            return f"{output}_node{node_id}.out"
    raise ValueError("no node_displacement recorder found")


def read_displacements(path: Path) -> List[float]:
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"empty output file: {path}")
    values = []
    for line in lines:
        parts = line.replace(",", " ").split()
        values.append(float(parts[0]))
    return values


def compute_curve(
    displacements: List[float],
    steps: int,
    load_value: float,
    area: float,
    length: float,
) -> Tuple[List[float], List[float]]:
    count = min(len(displacements), steps)
    strains = []
    stresses = []
    for i in range(count):
        scale = float(i + 1) / float(steps)
        stress = load_value * scale / area
        strain = displacements[i] / length
        strains.append(strain)
        stresses.append(stress)
    return strains, stresses


def write_csv(path: Path, strains: List[float], stresses: List[float]) -> None:
    lines = ["strain,stress"]
    for strain, stress in zip(strains, stresses):
        lines.append(f"{strain:.12e},{stress:.12e}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_case(
    repo_root: Path,
    case_json: Path,
    out_dir: Path,
    engine: str,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    if engine == "opensees":
        tcl_path = out_dir / "model.tcl"
        run(
            [
                sys.executable,
                str(repo_root / "scripts" / "json_to_tcl.py"),
                str(case_json),
                str(tcl_path),
            ]
        )
        run(
            [
                str(repo_root / "scripts" / "run_opensees_wine.sh"),
                "--script",
                str(tcl_path),
                "--output",
                str(out_dir),
            ]
        )
    elif engine == "mojo":
        run(
            [
                sys.executable,
                str(repo_root / "scripts" / "run_mojo_case.py"),
                "--input",
                str(case_json),
                "--output",
                str(out_dir),
            ]
        )
    else:
        raise ValueError(f"unknown engine: {engine}")
    return out_dir


def plot_material(
    material: str,
    data: Dict[str, Dict[str, Tuple[List[float], List[float]]]],
    output_dir: Path,
    units: str,
) -> Tuple[Path, "plt.Figure"]:
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = {"tension": "#1f77b4", "compression": "#d62728"}
    styles = {"mojo": "-", "opensees": "--"}
    for engine, series in data.items():
        for direction, (strains, stresses) in series.items():
            label = f"{engine} {direction}"
            ax.plot(
                strains,
                stresses,
                linestyle=styles.get(engine, "-"),
                color=colors.get(direction, "#333333"),
                label=label,
            )
    ax.axhline(0.0, color="#777777", linewidth=0.8)
    ax.axvline(0.0, color="#777777", linewidth=0.8)
    ax.set_xlabel("strain (-)")
    ylabel = "stress"
    if units:
        ylabel = f"stress ({units})"
    ax.set_ylabel(ylabel)
    ax.set_title(f"{material} stress-strain")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{material}_stress_strain.png"
    fig.savefig(out_path, dpi=200)
    return out_path, fig


def _bbox(strains: List[float], stresses: List[float]) -> Tuple[float, float, float, float]:
    if not strains or not stresses:
        return (0.0, 0.0, 0.0, 0.0)
    return (
        min(strains),
        max(strains),
        min(stresses),
        max(stresses),
    )


def _isclose(a: float, b: float, rtol: float, atol: float) -> bool:
    return abs(a - b) <= (atol + rtol * abs(b))


def _color(text: str, code: str, use_color: bool) -> str:
    if not use_color:
        return text
    return f"\033[{code}m{text}\033[0m"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("build/material_curves"),
        help="Directory to write outputs and plots.",
    )
    parser.add_argument(
        "--engine",
        choices=["mojo", "opensees", "both"],
        default="both",
        help="Which solver(s) to run.",
    )
    parser.add_argument(
        "--materials",
        nargs="*",
        help="Optional list of materials to plot.",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Skip running solvers and only plot from existing outputs.",
    )
    parser.add_argument(
        "--bbox-rtol",
        type=float,
        default=1.0e-6,
        help="Relative tolerance for bbox comparison.",
    )
    parser.add_argument(
        "--bbox-atol",
        type=float,
        default=1.0e-9,
        help="Absolute tolerance for bbox comparison.",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI colors in bbox summary.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    requested = [m.lower() for m in args.materials] if args.materials else None

    engines = ["mojo", "opensees"] if args.engine == "both" else [args.engine]

    pdf_path = args.output_dir / "material_curves.pdf"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with PdfPages(pdf_path) as pdf:
        for material in material_definitions():
            material_name = material["name"]
            if requested and material_name.lower() not in requested:
                continue

            material_out = args.output_dir / material_name
            curve_data: Dict[str, Dict[str, Tuple[List[float], List[float]]]] = {}

            directions = {
                "tension": build_case(material, "tension", material["load_tension"]),
                "compression": build_case(
                    material, "compression", material["load_compression"]
                ),
            }

            for engine in engines:
                curve_data[engine] = {}
                for direction, case_data in directions.items():
                    out_dir = material_out / engine / direction
                    case_json_path = material_out / "tmp" / f"{direction}.json"
                    case_json_path.parent.mkdir(parents=True, exist_ok=True)
                    case_json_path.write_text(
                        json.dumps(case_data, indent=2) + "\n", encoding="utf-8"
                    )
                    if not args.skip_run:
                        run_case(repo_root, case_json_path, out_dir, engine)

                    length, area = case_geometry(case_data)
                    load_value, steps = case_load_and_steps(case_data)
                    output_file = recorder_output_file(case_data)
                    disp_path = out_dir / output_file
                    displacements = read_displacements(disp_path)

                    strains, stresses = compute_curve(
                        displacements, steps, load_value, area, length
                    )
                    curve_data[engine][direction] = (strains, stresses)

                    csv_path = material_out / f"{engine}_{direction}.csv"
                    write_csv(csv_path, strains, stresses)

            if "mojo" in curve_data and "opensees" in curve_data:
                print(f"{material_name} bbox:")
                for direction in ("tension", "compression"):
                    if direction not in curve_data["mojo"] or direction not in curve_data["opensees"]:
                        msg = f"  {direction}: missing data"
                        print(_color(msg, "33", not args.no_color))
                        continue
                    m_eps, m_sig = curve_data["mojo"][direction]
                    o_eps, o_sig = curve_data["opensees"][direction]
                    m_bbox = _bbox(m_eps, m_sig)
                    o_bbox = _bbox(o_eps, o_sig)
                    labels = ["min_eps", "max_eps", "min_sig", "max_sig"]
                    diffs = []
                    ok = True
                    for name, m_val, o_val in zip(labels, m_bbox, o_bbox):
                        if not _isclose(m_val, o_val, args.bbox_rtol, args.bbox_atol):
                            ok = False
                            diffs.append(
                                f"{name}: mojo={m_val:.6e} open={o_val:.6e}"
                            )
                    if ok:
                        msg = f"  {direction}: OK"
                        print(_color(msg, "32", not args.no_color))
                    else:
                        msg = f"  {direction}: MISMATCH"
                        print(_color(msg, "31", not args.no_color))
                        for diff in diffs:
                            print(f"    {diff}")
            else:
                print(f"{material_name} bbox: skip (need both mojo and opensees)")

            units = str(material.get("units", ""))

            plot_path, fig = plot_material(material_name, curve_data, material_out, units)
            print(f"wrote {plot_path}")
            pdf.savefig(fig)
            plt.close(fig)

    print(f"wrote {pdf_path}")


if __name__ == "__main__":
    main()
