#!/usr/bin/env python3
import argparse
import json
import math
import os
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

try:
    from .plot_constants import MOJO_ORANGE, OPENSEES_BLUE
except ImportError:
    # Allow running as a standalone script.
    from plot_constants import MOJO_ORANGE, OPENSEES_BLUE


def _parse_rows(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if "," in line:
            parts = [p.strip() for p in line.split(",")]
        else:
            parts = line.split()
        rows.append([float(p) for p in parts])
    if not rows:
        raise ValueError(f"empty output file: {path}")
    width = len(rows[0])
    for i, row in enumerate(rows, start=1):
        if len(row) != width:
            raise ValueError(f"inconsistent row width in {path} at line {i}")
    return rows


def _series(rows, component_idx):
    return [row[component_idx] for row in rows]


def _max_abs(values):
    return max((abs(v) for v in values), default=0.0)


def _run_case(
    repo_root: Path,
    case_json: Path,
    refresh_reference: bool,
    force_case: bool,
):
    env = os.environ.copy()
    if refresh_reference:
        env["STRUT_REFRESH_REFERENCE"] = "1"
    if force_case:
        env["STRUT_FORCE_CASE"] = "1"
    cmd = ["uv", "run", str(repo_root / "scripts" / "run_case.py"), str(case_json)]
    print("+", " ".join(cmd))
    result = subprocess.run(cmd, env=env, check=False)
    if result.returncode != 0:
        print(
            f"warning: run_case exited with code {result.returncode}; plotting available outputs anyway"
        )


def _build_node_xy(case_data: dict) -> dict[int, tuple[float, float]]:
    node_xy: dict[int, tuple[float, float]] = {}
    for node in case_data.get("nodes", []):
        node_id = int(node["id"])
        node_xy[node_id] = (float(node.get("x", 0.0)), float(node.get("y", 0.0)))
    return node_xy


def _build_edges(
    case_data: dict, node_xy: dict[int, tuple[float, float]]
) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    for elem in case_data.get("elements", []):
        conn = elem.get("nodes", [])
        if not isinstance(conn, list) or len(conn) < 2:
            continue
        node_ids = [int(n) for n in conn if int(n) in node_xy]
        if len(node_ids) < 2:
            continue

        if len(node_ids) == 2:
            pairs = [(node_ids[0], node_ids[1])]
        else:
            pairs = [
                (node_ids[i], node_ids[(i + 1) % len(node_ids)])
                for i in range(len(node_ids))
            ]

        for n1, n2 in pairs:
            if n1 == n2:
                continue
            edge_key = (n1, n2) if n1 < n2 else (n2, n1)
            if edge_key in seen:
                continue
            seen.add(edge_key)
            edges.append((n1, n2))

    return edges


def _dof_series(rows, dofs: list[int], dof: int):
    if dof not in dofs:
        return None
    idx = dofs.index(dof)
    vals = []
    for row in rows:
        vals.append(row[idx] if idx < len(row) else 0.0)
    return vals


def _collect_node_displacements(
    case_data: dict, out_dir: Path, node_xy: dict[int, tuple[float, float]]
):
    raw: dict[int, dict[int, list[float]]] = {}
    for rec in case_data.get("recorders", []):
        if rec.get("type") != "node_displacement":
            continue
        dofs = [int(d) for d in rec.get("dofs", [])]
        output = rec.get("output", "node_disp")
        for node_id in rec.get("nodes", []):
            nid = int(node_id)
            path = out_dir / f"{output}_node{nid}.out"
            if not path.exists():
                continue
            try:
                rows = _parse_rows(path)
            except ValueError as exc:
                print(f"warning: skipping displacement file {path.name} ({exc})")
                continue
            per_node = raw.setdefault(nid, {})
            for target_dof in (1, 2):
                vals = _dof_series(rows, dofs, target_dof)
                if vals is None:
                    continue
                prev = per_node.get(target_dof)
                if prev is None or len(vals) > len(prev):
                    per_node[target_dof] = vals

    n = 0
    for dof_map in raw.values():
        for vals in dof_map.values():
            n = max(n, len(vals))

    node_disp: dict[int, list[tuple[float, float]]] = {}
    for node_id in node_xy:
        dof_map = raw.get(node_id, {})
        ux_vals = dof_map.get(1, [])
        uy_vals = dof_map.get(2, [])
        node_disp[node_id] = [
            (
                ux_vals[i] if i < len(ux_vals) else 0.0,
                uy_vals[i] if i < len(uy_vals) else 0.0,
            )
            for i in range(n)
        ]
    return node_disp, n


def _animation_frame_indices(n: int, max_frames: int) -> tuple[list[int], int]:
    if n <= 0:
        return [], 1
    if max_frames <= 0 or n <= max_frames:
        return list(range(n)), 1

    step = max(1, int(math.ceil(n / max_frames)))
    frame_indices = list(range(0, n, step))
    if frame_indices[-1] != n - 1:
        frame_indices.append(n - 1)
    return frame_indices, step


def _plot_structure_animation(
    case_name: str,
    case_data: dict,
    ref_dir: Path,
    strut_dir: Path,
    is_transient: bool,
    dt: float,
    out_path: Path,
    max_frames: int,
) -> bool:
    node_xy = _build_node_xy(case_data)
    edges = _build_edges(case_data, node_xy)
    if not node_xy or not edges:
        print("skip animation: missing model geometry in case JSON")
        return False

    ref_disp, n_ref = _collect_node_displacements(case_data, ref_dir, node_xy)
    strut_disp, n_strut = _collect_node_displacements(case_data, strut_dir, node_xy)
    n = min(n_ref, n_strut)
    if n <= 0:
        print("skip animation: no comparable node displacement history found")
        return False

    for node_id in node_xy:
        ref_disp[node_id] = ref_disp[node_id][:n]
        strut_disp[node_id] = strut_disp[node_id][:n]

    frame_indices, frame_step = _animation_frame_indices(n, max_frames)
    if not frame_indices:
        print("skip animation: no frames to render")
        return False

    xs = [xy[0] for xy in node_xy.values()]
    ys = [xy[1] for xy in node_xy.values()]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0e-9)

    max_disp = 0.0
    for node_disp in (ref_disp, strut_disp):
        for series in node_disp.values():
            for ux, uy in series:
                max_disp = max(max_disp, abs(ux), abs(uy))
    if max_disp <= 1.0e-15:
        disp_scale = 1.0
    else:
        disp_scale = min(max(0.15 * span / max_disp, 1.0), 1.0e6)

    def _node_pos(node_id: int, frame_idx: int, node_disp):
        x0, y0 = node_xy[node_id]
        ux, uy = node_disp[node_id][frame_idx]
        return (x0 + disp_scale * ux, y0 + disp_scale * uy)

    def _segments(node_disp, frame_idx: int):
        return [
            [_node_pos(n1, frame_idx, node_disp), _node_pos(n2, frame_idx, node_disp)]
            for n1, n2 in edges
        ]

    all_x = list(xs)
    all_y = list(ys)
    for frame_idx in frame_indices:
        for node_id in node_xy:
            rx, ry = _node_pos(node_id, frame_idx, ref_disp)
            mx, my = _node_pos(node_id, frame_idx, strut_disp)
            all_x.extend((rx, mx))
            all_y.extend((ry, my))
    pad = max(0.05 * span, 1.0e-8)
    x_min, x_max = min(all_x) - pad, max(all_x) + pad
    y_min, y_max = min(all_y) - pad, max(all_y) + pad

    base_segments = [[node_xy[n1], node_xy[n2]] for n1, n2 in edges]
    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    undeformed = LineCollection(
        base_segments,
        colors="#8f8f8f",
        linewidths=1.0,
        linestyles=":",
        alpha=0.55,
        zorder=1,
    )
    ref_lines = LineCollection(
        [],
        colors=OPENSEES_BLUE,
        linewidths=2.0,
        linestyles="-",
        zorder=2,
    )
    strut_lines = LineCollection(
        [],
        colors=MOJO_ORANGE,
        linewidths=1.8,
        linestyles="--",
        zorder=3,
    )
    ax.add_collection(undeformed)
    ax.add_collection(ref_lines)
    ax.add_collection(strut_lines)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, alpha=0.3)
    ax.set_title(f"{case_name} :: Structural Response")
    ax.legend(
        handles=[
            Line2D(
                [],
                [],
                color="#8f8f8f",
                linestyle=":",
                linewidth=1.0,
                label="Undeformed",
            ),
            Line2D(
                [],
                [],
                color=OPENSEES_BLUE,
                linestyle="-",
                linewidth=2.0,
                label="OpenSees reference",
            ),
            Line2D(
                [], [], color=MOJO_ORANGE, linestyle="--", linewidth=1.8, label="Strut"
            ),
        ],
        loc="best",
    )
    time_text = ax.text(
        0.015,
        0.985,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={
            "boxstyle": "round,pad=0.2",
            "facecolor": "white",
            "alpha": 0.85,
            "edgecolor": "none",
        },
    )
    ax.text(
        0.015,
        0.02,
        f"deformation scale = {disp_scale:.3e}",
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        bbox={
            "boxstyle": "round,pad=0.2",
            "facecolor": "white",
            "alpha": 0.85,
            "edgecolor": "none",
        },
    )

    def _frame_label(frame_idx: int) -> str:
        if is_transient:
            t = dt * (frame_idx + 1)
            return f"t = {t:.4g} s  (step {frame_idx + 1}/{n})"
        return f"step {frame_idx + 1}/{n}"

    def _draw(frame_idx: int):
        ref_lines.set_segments(_segments(ref_disp, frame_idx))
        strut_lines.set_segments(_segments(strut_disp, frame_idx))
        time_text.set_text(_frame_label(frame_idx))
        return ref_lines, strut_lines, time_text

    _draw(frame_indices[0])
    fig.tight_layout()

    if is_transient and dt > 0.0:
        effective_dt = dt * frame_step
        fps = int(max(2, min(30, round(1.0 / effective_dt))))
    else:
        fps = 4

    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not animation.writers.is_available("ffmpeg"):
            print(
                "warning: ffmpeg writer is not available; install ffmpeg to save MP4 animation"
            )
            plt.close(fig)
            return False
        anim = animation.FuncAnimation(
            fig,
            _draw,
            frames=frame_indices,
            interval=1000.0 / fps,
            blit=False,
            repeat=True,
        )
        writer = animation.FFMpegWriter(
            fps=fps,
            codec="libx264",
            extra_args=["-pix_fmt", "yuv420p"],
        )
        anim.save(out_path, writer=writer, dpi=140)
        print(f"saved movie: {out_path}")
        print(
            f"  frames={len(frame_indices)} from {n} step(s), frame_step={frame_step}, fps={fps}"
        )
    except Exception as exc:
        print(f"warning: failed to save animation {out_path} ({exc})")
        plt.close(fig)
        return False

    plt.close(fig)
    return True


def _plot_one(
    case_name: str,
    out_name: str,
    component: int,
    ref_vals,
    strut_vals,
    x_vals,
    x_label: str,
    out_path: Path | None,
    pdf: PdfPages | None,
):
    diffs = [g - r for r, g in zip(ref_vals, strut_vals)]
    rmse = (sum(d * d for d in diffs) / max(len(diffs), 1)) ** 0.5
    ref_peak = _max_abs(ref_vals)
    strut_peak = _max_abs(strut_vals)
    peak_abs_err = abs(strut_peak - ref_peak)
    peak_rel_err = peak_abs_err / max(ref_peak, 1.0e-30)

    single_sample = len(x_vals) == 1
    ref_style = {
        "label": "OpenSees reference",
        "linewidth": 2.0,
        "color": OPENSEES_BLUE,
    }
    strut_style = {
        "label": "Strut",
        "linewidth": 1.6,
        "linestyle": "--",
        "color": MOJO_ORANGE,
    }
    if single_sample:
        ref_style.update({"marker": "o", "markersize": 6.0})
        strut_style.update({"marker": "x", "markersize": 6.0})

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(x_vals, ref_vals, **ref_style)
    ax.plot(x_vals, strut_vals, **strut_style)
    ax.set_xlabel(x_label)
    ax.set_ylabel(f"Component {component}")
    ax.set_title(f"{case_name} :: {out_name} :: c{component}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=140)
        print(f"saved: {out_path}")
    if pdf is not None:
        pdf.savefig(fig, dpi=140)
    plt.close(fig)

    print(
        f"  peak(ref)={ref_peak:.6e} peak(strut)={strut_peak:.6e} peak_rel={peak_rel_err:.6e} rmse={rmse:.6e}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run a validation case and plot all comparable reference/strut output files."
    )
    parser.add_argument("case_json", help="path to case JSON")
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="skip running case; only plot from existing output files",
    )
    parser.add_argument(
        "--refresh-reference",
        action="store_true",
        help="refresh OpenSees reference outputs while running case",
    )
    parser.add_argument(
        "--force-case",
        action="store_true",
        help="force run even when case JSON has enabled=false",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="PNG output plot directory (default: build/plots/<case_name>)",
    )
    parser.add_argument(
        "--no-png",
        action="store_true",
        help="skip writing per-series PNG files",
    )
    parser.add_argument(
        "--pdf",
        default="",
        help="output PDF path (default: build/plots/<case_name>/<case_name>_all_recorders.pdf)",
    )
    parser.add_argument(
        "--no-animation",
        action="store_true",
        help="skip creating structural time-history GIF and preview page",
    )
    parser.add_argument(
        "--animation",
        default="",
        help="output MP4 path (default: build/plots/<case_name>/<case_name>_structure_time_history.mp4)",
    )
    parser.add_argument(
        "--animation-max-frames",
        type=int,
        default=400,
        help="max frames to render in structural GIF (default: 400)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    case_json = Path(args.case_json).resolve()
    if not case_json.exists():
        raise SystemExit(f"missing case json: {case_json}")
    case_data_input = json.loads(case_json.read_text(encoding="utf-8"))
    force_case = args.force_case or (not bool(case_data_input.get("enabled", True)))

    if not args.skip_run:
        _run_case(repo_root, case_json, args.refresh_reference, force_case)

    case_name = case_json.stem
    case_root = repo_root / "tests" / "validation" / case_name
    tgt_json = case_root / f"{case_name}.json"
    if not tgt_json.exists():
        tgt_json = case_json
    case_data = json.loads(tgt_json.read_text(encoding="utf-8"))
    analysis = case_data.get("analysis", {})
    is_transient = str(analysis.get("type", "")).startswith("transient")
    dt = float(analysis.get("dt", 1.0))

    ref_dir = case_root / "reference"
    strut_dir = case_root / "strut"
    if not ref_dir.exists() or not strut_dir.exists():
        raise SystemExit(
            f"missing output directories: reference={ref_dir.exists()} strut={strut_dir.exists()}"
        )

    if args.output_dir:
        plot_dir = Path(args.output_dir)
    else:
        plot_dir = repo_root / "build" / "plots" / case_name

    out_files = sorted(ref_dir.glob("*.out"))
    if not out_files:
        raise SystemExit(f"no .out files found in {ref_dir}")

    if args.pdf:
        pdf_path = Path(args.pdf)
    else:
        pdf_path = plot_dir / f"{case_name}_all_recorders.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    plotted = 0
    animation_saved = False
    with PdfPages(pdf_path) as pdf:
        if not args.no_animation:
            if args.animation:
                animation_path = Path(args.animation)
            else:
                animation_path = plot_dir / f"{case_name}_structure_time_history.mp4"
            animation_saved = _plot_structure_animation(
                case_name,
                case_data,
                ref_dir,
                strut_dir,
                is_transient,
                dt,
                animation_path,
                max_frames=max(1, int(args.animation_max_frames)),
            )

        for ref_file in out_files:
            strut_file = strut_dir / ref_file.name
            if not strut_file.exists():
                print(f"skip (missing strut): {strut_file}")
                continue
            try:
                ref_rows = _parse_rows(ref_file)
                strut_rows = _parse_rows(strut_file)
            except ValueError as exc:
                print(f"skip ({exc})")
                continue

            n = min(len(ref_rows), len(strut_rows))
            if n == 0:
                print(f"skip (no comparable rows): {ref_file.name}")
                continue
            if len(ref_rows) != len(strut_rows):
                print(
                    f"warning: row mismatch in {ref_file.name}; ref={len(ref_rows)} strut={len(strut_rows)} using first {n}"
                )
            ref_rows = ref_rows[:n]
            strut_rows = strut_rows[:n]

            width = min(len(ref_rows[0]), len(strut_rows[0]))
            if len(ref_rows[0]) != len(strut_rows[0]):
                print(
                    f"warning: column mismatch in {ref_file.name}; ref={len(ref_rows[0])} strut={len(strut_rows[0])} using first {width}"
                )

            if is_transient:
                x_vals = [dt * (i + 1) for i in range(n)]
                x_label = "Time (s)"
            else:
                x_vals = [i + 1 for i in range(n)]
                x_label = "Step"

            stem = ref_file.stem
            for comp_idx in range(width):
                component = comp_idx + 1
                ref_vals = _series(ref_rows, comp_idx)
                strut_vals = _series(strut_rows, comp_idx)
                out_path = (
                    None if args.no_png else (plot_dir / f"{stem}_c{component}.png")
                )
                _plot_one(
                    case_name,
                    ref_file.name,
                    component,
                    ref_vals,
                    strut_vals,
                    x_vals,
                    x_label,
                    out_path,
                    pdf,
                )
                plotted += 1

    print(f"saved: {pdf_path}")
    if animation_saved:
        print("done: generated structural animation + recorder plots")
    else:
        print(f"done: generated {plotted} plot(s)")


if __name__ == "__main__":
    main()
