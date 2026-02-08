#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
except ModuleNotFoundError as exc:
    raise SystemExit(
        "matplotlib is required for plotting. Install it with "
        "`uv add --dev matplotlib` or `pip install matplotlib`."
    ) from exc

try:
    from .plot_constants import MOJO_ORANGE, OPENSEES_BLUE
except ImportError:
    from plot_constants import MOJO_ORANGE, OPENSEES_BLUE


def run(cmd: List[str]) -> None:
    subprocess.check_call(cmd)


def material_definitions() -> List[dict]:
    return [
        {
            "name": "steel01",
            "type": "Steel01",
            "params": {"Fy": 250.0, "E0": 200000.0, "b": 0.01},
            "units": "SI",
            "steps": 120,
            "curve": {
                "strain_min": -0.02,
                "strain_max": 0.02,
            },
            "hysteresis": {
                "min_strain": -0.02,
                "max_strain": 0.02,
                "cycles": 3,
                "ramp": True,
                "steps_per_segment": 40,
            },
        },
        {
            "name": "steel02",
            "type": "Steel02",
            "params": {
                "Fy": 250.0,
                "E0": 200000.0,
                "b": 0.01,
                "R0": 18.0,
                "cR1": 0.925,
                "cR2": 0.15,
                "a1": 0.01,
                "a2": 1.0,
                "a3": 0.01,
                "a4": 1.0,
            },
            "units": "SI",
            "steps": 200,
            "curve": {
                "strain_min": -0.03,
                "strain_max": 0.03,
            },
            "hysteresis": {
                "min_strain": -0.03,
                "max_strain": 0.03,
                "cycles": 3,
                "ramp": True,
                "steps_per_segment": 40,
            },
        },
        {
            "name": "concrete01_unconfined",
            "type": "Concrete01",
            "params": {"fpc": -30.0, "epsc0": -0.002, "fpcu": -20.0, "epscu": -0.006},
            "units": "SI",
            "steps": 180,
            "curve": {
                "strain_min": -0.015,
                "strain_max": 0.005,
            },
            "hysteresis": {
                "min_strain": -0.015,
                "max_strain": 0.0,
                "cycles": 4,
                "ramp": True,
                "ramp_power": 2.0,
                "steps_per_segment": 40,
            },
        },
        {
            "name": "concrete02_unconfined",
            "type": "Concrete02",
            "params": {
                "fpc": -30.0,
                "epsc0": -0.002,
                "fpcu": -20.0,
                "epscu": -0.006,
                "rat": 0.1,
                "ft": 3.0,
                "Ets": 200.0,
            },
            "units": "SI",
            "steps": 240,
            "curve": {
                "strain_min": -0.015,
                "strain_max": 0.005,
            },
            "hysteresis": {
                "min_strain": -0.015,
                "max_strain": 0.0005,
                "cycles": 4,
                "ramp": True,
                "steps_per_segment": 40,
            },
        },
    ]


def _linspace_targets(end_value: float, steps: int) -> List[float]:
    if steps < 1:
        raise ValueError("steps must be >= 1")
    return [end_value * (float(i + 1) / float(steps)) for i in range(steps)]


def build_displacement_case(
    material: dict,
    case_name: str,
    displacement_targets: List[float],
    analysis_opts: Optional[dict] = None,
) -> dict:
    if not displacement_targets:
        raise ValueError("displacement_targets must not be empty")
    if analysis_opts is None:
        analysis_opts = {}
    return {
        "schema_version": "1.0",
        "enabled": True,
        "metadata": {
            "name": case_name,
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
        "loads": [{"node": 2, "dof": 1, "value": 1.0}],
        "analysis": {
            "type": "static_nonlinear",
            "steps": len(displacement_targets),
            "max_iters": int(analysis_opts.get("max_iters", 80)),
            "tol": float(analysis_opts.get("tol", 1.0e-10)),
            "rel_tol": float(analysis_opts.get("rel_tol", 1.0e-8)),
            "integrator": {
                "type": "DisplacementControl",
                "node": 2,
                "dof": 1,
                "targets": displacement_targets,
                "cutback": float(analysis_opts.get("cutback", 0.5)),
                "max_cutbacks": int(analysis_opts.get("max_cutbacks", 12)),
                "min_du": float(analysis_opts.get("min_du", 1.0e-8)),
            },
        },
        "recorders": [
            {
                "type": "node_displacement",
                "nodes": [2],
                "dofs": [1],
                "output": "node_disp",
            },
            {
                "type": "element_force",
                "elements": [1],
                "output": "element_force",
            },
        ],
    }


def _strain_path(
    strain_levels: List[float],
    steps_per_segment: int,
    min_strain: Optional[float] = None,
    max_strain: Optional[float] = None,
    cycles: int = 1,
    targets: Optional[List[float]] = None,
    ramp: bool = False,
    ramp_positive_only: bool = False,
    ramp_power: float = 1.0,
) -> List[float]:
    if steps_per_segment < 1:
        raise ValueError("steps_per_segment must be >= 1")
    if targets is not None:
        if len(targets) < 2:
            raise ValueError("targets must contain at least 2 entries")
        target_list = [float(v) for v in targets]
        if target_list[0] != 0.0:
            target_list = [0.0] + target_list
        strains: List[float] = []
        for start, end in zip(target_list[:-1], target_list[1:]):
            for i in range(1, steps_per_segment + 1):
                t = float(i) / float(steps_per_segment)
                strains.append(start + (end - start) * t)
        return strains
    if cycles < 1:
        raise ValueError("cycles must be >= 1")
    path_targets = [0.0]
    if min_strain is not None or max_strain is not None:
        if min_strain is None or max_strain is None:
            raise ValueError("min_strain and max_strain must both be set")
        min_val = float(min_strain)
        max_val = float(max_strain)
        if min_val > max_val:
            raise ValueError("min_strain must be <= max_strain")
        for i in range(1, cycles + 1):
            scale = float(i) / float(cycles) if ramp else 1.0
            if ramp and ramp_power != 1.0:
                scale = scale ** ramp_power
            path_targets.append(max_val * scale)
            if min_val != 0.0:
                if ramp_positive_only:
                    path_targets.append(min_val)
                else:
                    path_targets.append(min_val * scale)
            else:
                path_targets.append(0.0)
        path_targets.append(0.0)
    else:
        if not strain_levels:
            raise ValueError("strain_levels must not be empty")
        for _ in range(cycles):
            for level in strain_levels:
                path_targets.append(abs(float(level)))
                path_targets.append(-abs(float(level)))
        path_targets.append(0.0)
    strains: List[float] = []
    for start, end in zip(path_targets[:-1], path_targets[1:]):
        for i in range(1, steps_per_segment + 1):
            t = float(i) / float(steps_per_segment)
            strains.append(start + (end - start) * t)
    return strains


def case_geometry(data: dict) -> Tuple[float, float, List[float]]:
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
    dx = x2 - x1
    dy = y2 - y1
    length = (dx * dx + dy * dy) ** 0.5
    direction = [dx / length, dy / length]
    area = float(elem.get("area", 1.0))
    return length, area, direction


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


def recorder_element_force_file(data: dict) -> str:
    recorders = data.get("recorders", [])
    for rec in recorders:
        if rec.get("type") == "element_force":
            output = rec.get("output", "element_force")
            elements = rec.get("elements", [])
            if not elements:
                raise ValueError("element_force recorder missing elements")
            elem_id = int(elements[0])
            return f"{output}_ele{elem_id}.out"
    raise ValueError("no element_force recorder found")


def read_displacements(path: Path) -> List[float]:
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"empty output file: {path}")
    values = []
    for line in lines:
        parts = line.replace(",", " ").split()
        values.append(float(parts[0]))
    return values


def read_element_force_rows(path: Path) -> List[List[float]]:
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"empty output file: {path}")
    values: List[List[float]] = []
    for line in lines:
        parts = line.replace(",", " ").split()
        values.append([float(part) for part in parts])
    return values


def truss_axial_forces(rows: List[List[float]], direction: List[float]) -> List[float]:
    if not rows:
        return []
    c, s = direction
    forces = []
    for row in rows:
        if len(row) == 1:
            forces.append(row[0])
            continue
        if len(row) < 4:
            raise ValueError("truss force recorder expected 4 values for 2D")
        f1x, f1y = row[0], row[1]
        forces.append(-(f1x * c + f1y * s))
    return forces


def compute_curve(
    displacements: List[float],
    length: float,
    area: float,
    forces: List[float],
) -> Tuple[List[float], List[float]]:
    count = min(len(displacements), len(forces))
    strains = []
    stresses = []
    for i in range(count):
        strains.append(displacements[i] / length)
        stresses.append(forces[i] / area)
    return strains, stresses


def write_csv(path: Path, strains: List[float], stresses: List[float]) -> None:
    lines = ["strain,stress"]
    for strain, stress in zip(strains, stresses):
        lines.append(f"{strain:.12e},{stress:.12e}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_case(repo_root: Path, case_json: Path, out_dir: Path, engine: str) -> Path:
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


def run_curve_case(
    repo_root: Path,
    case_data: dict,
    material_out: Path,
    engine: str,
    label: str,
    skip_run: bool,
) -> Tuple[List[float], List[float]]:
    out_dir = material_out / engine / label
    case_json_path = material_out / "tmp" / f"{label}.json"
    case_json_path.parent.mkdir(parents=True, exist_ok=True)
    case_json_path.write_text(json.dumps(case_data, indent=2) + "\n", encoding="utf-8")

    if not skip_run:
        run_case(repo_root, case_json_path, out_dir, engine)

    length, area, direction_vec = case_geometry(case_data)
    disp_path = out_dir / recorder_output_file(case_data)
    force_path = out_dir / recorder_element_force_file(case_data)
    displacements = read_displacements(disp_path)
    force_rows = read_element_force_rows(force_path)
    forces = truss_axial_forces(force_rows, direction_vec)
    return compute_curve(displacements, length, area, forces)


def plot_material(
    material: str,
    data: Dict[str, Dict[str, Tuple[List[float], List[float]]]],
    output_dir: Path,
    units: str,
) -> Tuple[Path, "plt.Figure"]:
    fig, ax = plt.subplots(figsize=(7, 4))
    engine_colors = {
        "mojo": MOJO_ORANGE,
        "opensees": OPENSEES_BLUE,
    }
    for engine in ("opensees", "mojo"):
        series = data.get(engine)
        if not series:
            continue
        points: List[Tuple[float, float]] = []
        for strains, stresses in series.values():
            points.extend(zip(strains, stresses))
        if not points:
            continue
        points.sort(key=lambda pair: pair[0])
        xs = [pt[0] for pt in points]
        ys = [pt[1] for pt in points]
        line_color = engine_colors.get(engine, MOJO_ORANGE)
        ax.plot(xs, ys, linestyle="-", color=line_color, label=engine, zorder=2)
        if xs:
            ax.scatter([xs[0], xs[-1]], [ys[0], ys[-1]], s=18, color=line_color, edgecolors="none", zorder=1)
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


def plot_hysteresis(
    material: str,
    data: Dict[str, Tuple[List[float], List[float]]],
    output_dir: Path,
    units: str,
) -> Tuple[Path, "plt.Figure"]:
    fig, ax = plt.subplots(figsize=(7, 4))
    engine_colors = {
        "mojo": MOJO_ORANGE,
        "opensees": OPENSEES_BLUE,
    }
    for engine in ("opensees", "mojo"):
        series = data.get(engine)
        if not series:
            continue
        strains, stresses = series
        if not strains:
            continue
        ax.plot(strains, stresses, linestyle="-", color=engine_colors.get(engine, MOJO_ORANGE), label=engine)
    ax.axhline(0.0, color="#777777", linewidth=0.8)
    ax.axvline(0.0, color="#777777", linewidth=0.8)
    ax.set_xlabel("strain (-)")
    ylabel = "stress"
    if units:
        ylabel = f"stress ({units})"
    ax.set_ylabel(ylabel)
    ax.set_title(f"{material} hysteresis")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{material}_hysteresis.png"
    fig.savefig(out_path, dpi=200)
    return out_path, fig


def _bbox(strains: List[float], stresses: List[float]) -> Tuple[float, float, float, float]:
    if not strains or not stresses:
        return (0.0, 0.0, 0.0, 0.0)
    return (min(strains), max(strains), min(stresses), max(stresses))


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
        default=5.0e-6,
        help="Relative tolerance for bbox comparison.",
    )
    parser.add_argument(
        "--bbox-atol",
        type=float,
        default=1.0e-8,
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
            curve_cfg = material.get("curve", {})
            steps = int(material.get("steps", 120))
            min_strain = float(curve_cfg.get("strain_min", -0.01))
            max_strain = float(curve_cfg.get("strain_max", 0.01))
            analysis_cfg = material.get("analysis", {})

            direction_cases: Dict[str, dict] = {}
            if max_strain > 0.0:
                tension_targets = _linspace_targets(max_strain, steps)
                direction_cases["tension"] = build_displacement_case(
                    material,
                    f"{material_name}_curve_tension",
                    tension_targets,
                    analysis_cfg,
                )
            if min_strain < 0.0:
                compression_targets = _linspace_targets(min_strain, steps)
                direction_cases["compression"] = build_displacement_case(
                    material,
                    f"{material_name}_curve_compression",
                    compression_targets,
                    analysis_cfg,
                )

            for engine in engines:
                curve_data[engine] = {}
                for direction, case_data in direction_cases.items():
                    strains, stresses = run_curve_case(
                        repo_root,
                        case_data,
                        material_out,
                        engine,
                        direction,
                        args.skip_run,
                    )
                    curve_data[engine][direction] = (strains, stresses)
                    csv_path = material_out / f"{engine}_{direction}.csv"
                    write_csv(csv_path, strains, stresses)

            if "mojo" in curve_data and "opensees" in curve_data:
                print(f"{material_name} bbox:")
                directions = sorted(
                    set(curve_data["mojo"].keys()) | set(curve_data["opensees"].keys())
                )
                if "tension" in directions and "compression" in directions:
                    directions = ["tension", "compression"] + [
                        d for d in directions if d not in ("tension", "compression")
                    ]
                for direction in directions:
                    if (
                        direction not in curve_data["mojo"]
                        or direction not in curve_data["opensees"]
                    ):
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
                            diffs.append(f"{name}: mojo={m_val:.6e} open={o_val:.6e}")
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

            hysteresis_cfg = material.get("hysteresis")
            if hysteresis_cfg:
                steps_per_segment = int(hysteresis_cfg.get("steps_per_segment", 20))
                strain_levels = list(hysteresis_cfg.get("strain_levels", []))
                strain_history = _strain_path(
                    strain_levels,
                    steps_per_segment,
                    min_strain=hysteresis_cfg.get("min_strain"),
                    max_strain=hysteresis_cfg.get("max_strain"),
                    cycles=int(hysteresis_cfg.get("cycles", 1)),
                    targets=hysteresis_cfg.get("targets"),
                    ramp=bool(hysteresis_cfg.get("ramp", False)),
                    ramp_positive_only=bool(hysteresis_cfg.get("ramp_positive_only", False)),
                    ramp_power=float(hysteresis_cfg.get("ramp_power", 1.0)),
                )
                hyst_case = build_displacement_case(
                    material,
                    f"{material_name}_hysteresis",
                    strain_history,
                    {
                        "max_iters": hysteresis_cfg.get("max_iters", analysis_cfg.get("max_iters", 80)),
                        "tol": hysteresis_cfg.get("tol", analysis_cfg.get("tol", 1.0e-10)),
                        "rel_tol": hysteresis_cfg.get("rel_tol", analysis_cfg.get("rel_tol", 1.0e-8)),
                        "cutback": hysteresis_cfg.get("cutback", analysis_cfg.get("cutback", 0.5)),
                        "max_cutbacks": hysteresis_cfg.get("max_cutbacks", analysis_cfg.get("max_cutbacks", 12)),
                        "min_du": hysteresis_cfg.get("min_du", analysis_cfg.get("min_du", 1.0e-8)),
                    },
                )

                hyst_data: Dict[str, Tuple[List[float], List[float]]] = {}
                for engine in engines:
                    try:
                        strains, stresses = run_curve_case(
                            repo_root,
                            hyst_case,
                            material_out,
                            engine,
                            "hysteresis",
                            args.skip_run,
                        )
                    except subprocess.CalledProcessError as exc:
                        print(
                            f"{material_name} hysteresis: {engine} run failed ({exc.returncode}); skipping"
                        )
                        continue
                    except ValueError:
                        print(
                            f"{material_name} hysteresis: {engine} produced empty outputs; skipping"
                        )
                        continue
                    hyst_data[engine] = (strains, stresses)
                    csv_path = material_out / f"{engine}_hysteresis.csv"
                    write_csv(csv_path, strains, stresses)

                if hyst_data:
                    hyst_path, hyst_fig = plot_hysteresis(material_name, hyst_data, material_out, units)
                    print(f"wrote {hyst_path}")
                    pdf.savefig(hyst_fig)
                    plt.close(hyst_fig)

    print(f"wrote {pdf_path}")


if __name__ == "__main__":
    main()
