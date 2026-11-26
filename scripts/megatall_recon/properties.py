"""Generate all non-geometric OpenSees Model1 properties from geometry + props.json.

Stdlib-only. Given a ``Geometry`` instance (loaded from the real Model1 files
via ``load_real_geometry`` or supplied by a separate geometry generator) and
``props.json``, this module generates all material, section, element-property,
mass, gravity-load, and ``dynamiccpu.tcl`` entry-script lines.
"""

from __future__ import annotations

import os
import argparse
import collections
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_MODEL_DIR = Path(
    os.environ.get(
        "STRUT_MEGATALL_REAL_DIR",
        Path(__file__).resolve().parents[2] / ".tmp/megatall-real/Model1",
    )
)
# A copy of the real benchmark model, used only to extract and verify the
# tables. Tests never need it: they use the generated bundle.


def require_real_model(model_dir: Path) -> Path:
    """Refuse the generated bundle: comparing it with itself proves nothing."""
    if (model_dir / ".strut-generated").exists():
        raise SystemExit(
            f"{model_dir} is the generated bundle, not the real model. Point "
            "STRUT_MEGATALL_REAL_DIR (or --model-dir) at a copy of the original Model1."
        )
    if not (model_dir / "opensees.tcl").is_file():
        raise SystemExit(
            f"real Model1 not found at {model_dir}; set STRUT_MEGATALL_REAL_DIR "
            "(see scripts/megatall_recon/README.md)"
        )
    return model_dir


DEFAULT_PROPS_PATH = Path(__file__).resolve().parent / "props.json"

GRID_VALS = (
    -25.0,
    -21.5,
    -18.5,
    -15.0,
    -11.5,
    -8.5,
    -5.0,
    -1.5,
    0.0,
    1.5,
    5.0,
    8.5,
    11.5,
    15.0,
    18.5,
    21.5,
    25.0,
)

D4_OPS = (
    (lambda x, y: (x, y), 0),
    (lambda x, y: (-y, x), 0),
    (lambda x, y: (-x, -y), 0),
    (lambda x, y: (y, -x), 0),
    (lambda x, y: (-x, y), 1),
    (lambda x, y: (x, -y), 1),
    (lambda x, y: (y, x), 1),
    (lambda x, y: (-y, -x), 1),
)
C4_OPS = D4_OPS[:4]
V4_OPS = (
    (lambda x, y: (x, y), 0),
    (lambda x, y: (-x, y), 1),
    (lambda x, y: (x, -y), 1),
    (lambda x, y: (-x, -y), 0),
)
C2_OPS = (
    (lambda x, y: (x, y), 0),
    (lambda x, y: (-x, -y), 0),
)


@dataclass
class Geometry:
    """Pure geometric input (coordinates, connectivity, numbering, diaphragms).

    Contains NO material, section, mass, or load data. Can be populated from
    the real Model1 files via ``load_real_geometry`` or from a synthetic
    geometry generator.
    """

    # node_id -> (x, y, z)
    nodes: dict[int, tuple[float, float, float]]
    # geomTransf lines: (tag, transf_type, vx, vy, vz)
    geom_transfs: list[tuple[int, str, float, float, float]]
    # opensees.tcl mega-column dispBeamColumn: (eid, n1, n2, transf_tag)
    megacol_elems: list[tuple[int, int, int, int]]
    # truss.tcl elasticBeamColumn rigid links: (eid, n1, n2, transf_tag)
    ebc_elems: list[tuple[int, int, int, int]]
    # truss.tcl coupling-beam longitudinal rebar trusses: (eid, n1, n2)
    truss_elems: list[tuple[int, int, int]]
    # Hbeam.tcl W-section dispBeamColumn: (eid, n1, n2, transf_tag)
    hbeam_elems: list[tuple[int, int, int, int]]
    # layershell.tcl ShellMITC4 wall/coupling-beam quads: (eid, n1, n2, n3, n4)
    shell_elems: list[tuple[int, int, int, int, int]]
    # rigiddigram.tcl rigidDiaphragm: (perp_dirn, master_node, [slave_nodes...])
    diaphragms: list[tuple[int, int, list[int]]]


def load_real_geometry(model_dir: Path = DEFAULT_MODEL_DIR) -> Geometry:
    """Read ONLY geometric entities (nodes, connectivity, transfTags, diaphragms)."""
    nodes: dict[int, tuple[float, float, float]] = {}
    geom_transfs: list[tuple[int, str, float, float, float]] = []
    megacol_elems: list[tuple[int, int, int, int]] = []

    with (model_dir / "opensees.tcl").open() as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            toks = s.split()
            if toks[0] == "node":
                nodes[int(toks[1])] = (float(toks[2]), float(toks[3]), float(toks[4]))
            elif toks[0] == "geomTransf":
                geom_transfs.append(
                    (
                        int(toks[2]),
                        toks[1],
                        float(toks[3]),
                        float(toks[4]),
                        float(toks[5]),
                    )
                )
            elif toks[0] == "element" and toks[1] == "dispBeamColumn":
                megacol_elems.append(
                    (int(toks[2]), int(toks[3]), int(toks[4]), int(toks[7]))
                )

    ebc_elems: list[tuple[int, int, int, int]] = []
    truss_elems: list[tuple[int, int, int]] = []
    with (model_dir / "truss.tcl").open() as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            toks = s.split()
            if toks[0] == "element" and toks[1] == "elasticBeamColumn":
                ebc_elems.append(
                    (int(toks[2]), int(toks[3]), int(toks[4]), int(toks[11]))
                )
            elif toks[0] == "element" and toks[1] == "truss":
                truss_elems.append((int(toks[2]), int(toks[3]), int(toks[4])))

    hbeam_elems: list[tuple[int, int, int, int]] = []
    with (model_dir / "Hbeam.tcl").open() as f:
        for line in f:
            s = line.strip()
            if s.startswith("element dispBeamColumn"):
                toks = s.split()
                hbeam_elems.append(
                    (int(toks[2]), int(toks[3]), int(toks[4]), int(toks[7]))
                )

    shell_elems: list[tuple[int, int, int, int, int]] = []
    with (model_dir / "layershell.tcl").open() as f:
        for line in f:
            s = line.strip()
            if s.startswith("element ShellMITC4"):
                toks = s.split()
                shell_elems.append(
                    (
                        int(toks[2]),
                        int(toks[3]),
                        int(toks[4]),
                        int(toks[5]),
                        int(toks[6]),
                    )
                )

    diaphragms: list[tuple[int, int, list[int]]] = []
    with (model_dir / "rigiddigram.tcl").open() as f:
        for line in f:
            s = line.strip()
            if s.startswith("rigidDiaphragm"):
                toks = s.split()
                diaphragms.append(
                    (int(toks[1]), int(toks[2]), [int(x) for x in toks[3:]])
                )

    return Geometry(
        nodes=nodes,
        geom_transfs=geom_transfs,
        megacol_elems=megacol_elems,
        ebc_elems=ebc_elems,
        truss_elems=truss_elems,
        hbeam_elems=hbeam_elems,
        shell_elems=shell_elems,
        diaphragms=diaphragms,
    )


def load_props(props_path: Path = DEFAULT_PROPS_PATH) -> dict[str, Any]:
    return json.loads(props_path.read_text())


def _expand_indices(spec: list[int | list[int]]) -> list[int]:
    out: list[int] = []
    for item in spec:
        if isinstance(item, list):
            out.extend(range(item[0], item[1] + 1))
        else:
            out.append(item)
    return out


def _clean_key(v: float, nd: int = 1) -> int | float:
    r = round(v, nd)
    if abs(r) < 1e-12:
        return 0
    if abs(r - round(r)) < 1e-9:
        return round(r)
    return r


def _expand_steel101_bars(
    bars: list[list[Any]],
) -> list[tuple[float, float, float, int]]:
    fibs: list[tuple[float, float, float, int]] = []
    for ax, fspec, lo, hi, n, area in bars:
        fixed_vals: list[float] = []
        for item in fspec:
            if isinstance(item, list):
                s0, s1, cnt = float(item[0]), float(item[1]), int(item[2])
                step = (s1 - s0) / (cnt - 1) if cnt > 1 else 0.0
                for j in range(cnt):
                    fixed_vals.append(s0 + j * step)
            else:
                fixed_vals.append(float(item))
        lo_f, hi_f, a_f = float(lo), float(hi), float(area)
        step_v = (hi_f - lo_f) / (n - 1) if n > 1 else 0.0
        for fv in fixed_vals:
            for m in range(n):
                vv = lo_f + m * step_v
                if ax == "z":
                    fibs.append((fv, vv, a_f, 101))
                else:
                    fibs.append((vv, fv, a_f, 101))
    return fibs


def _build_megacol_fibers(
    cmat: int,
    h_val: float,
    b_val: float,
    bars: list[list[Any]],
    cover: float,
    rebar_ratio: float,
) -> list[tuple[float, float, float, int]]:
    s101 = _expand_steel101_bars(bars)
    a_steel = sum(x[2] for x in s101)
    h_in = h_val - 2.0 * cover
    b_in = b_val - 2.0 * cover
    dy = h_in / 6.0
    dz = b_in / 6.0
    yc_edge = h_val / 2.0 - cover / 2.0
    zc_edge = b_val / 2.0 - cover / 2.0
    ys6 = [-(h_in / 2.0) + (i + 0.5) * dy for i in range(6)]
    zs6 = [-(b_in / 2.0) + (j + 0.5) * dz for j in range(6)]
    a_corner = cover * cover
    a_tb = cover * dz
    a_lr = dy * cover
    a_reb = rebar_ratio * h_val * b_val / 12.0
    a_core = (h_in * b_in - a_steel - 12.0 * a_reb) / 36.0

    conc: list[tuple[float, float, float, int]] = []
    for sy in (-1.0, 1.0):
        for sz in (-1.0, 1.0):
            conc.append((sy * yc_edge, sz * zc_edge, a_corner, cmat))
    for sy in (-1.0, 1.0):
        for z in zs6:
            conc.append((sy * yc_edge, z, a_tb, cmat))
    for sz in (-1.0, 1.0):
        for y in ys6:
            conc.append((y, sz * zc_edge, a_lr, cmat))
    for y in ys6:
        for z in zs6:
            conc.append((y, z, a_core, cmat))

    yr = h_val / 2.0 - cover
    zr = b_val / 2.0 - cover
    rebs: list[tuple[float, float, float, int]] = []
    for z in (-zr, -zr / 3.0, zr / 3.0, zr):
        for y in (yr, -yr):
            rebs.append((y, z, a_reb, 102))
    for y in (yr / 3.0, -yr / 3.0):
        for z in (-zr, zr):
            rebs.append((y, z, a_reb, 102))

    return conc + s101 + rebs


def generate_opensees(geom: Geometry, props: dict[str, Any]) -> dict[str, Any]:
    os_p = props["opensees"]
    ec_gpa: list[float] = os_p["ec_gpa"]
    gc_by_cmat = {i + 1: ec_gpa[i] * 1e9 / 2.4 for i in range(len(ec_gpa))}
    cover = float(os_p["cover"])
    rebar_ratio = float(os_p["rebar_ratio"])
    rho = float(os_p["rho"])
    g = float(os_p["g"])
    n_ip = int(os_p["nIP"])

    # 39 Fiber sections (13 identical triplets)
    fiber_secs: dict[int, list[tuple[float, float, float, int]]] = {}
    triplet_meta: dict[int, tuple[int, float, float]] = {}
    for k, (cmat, h_val, b_val, bars) in enumerate(os_p["triplets"]):
        fibs = _build_megacol_fibers(
            int(cmat), float(h_val), float(b_val), bars, cover, rebar_ratio
        )
        for off in (1, 2, 3):
            sid = 3 * k + off
            fiber_secs[sid] = fibs
            triplet_meta[sid] = (int(cmat), float(h_val), float(b_val))

    assign_rules: dict[str, list[list[Any]]] = os_p["assign"]

    def assign_fsec(n1: int, n2: int) -> int:
        p1, p2 = geom.nodes[n1], geom.nodes[n2]
        mx, my = 0.5 * (p1[0] + p2[0]), 0.5 * (p1[1] + p2[1])
        z1, z2 = min(p1[2], p2[2]), max(p1[2], p2[2])
        zm = 0.5 * (z1 + z2)
        if abs(abs(mx) - abs(my)) < 1.0:
            sub = "C_POS" if mx * my > 0 else "C_NEG"
        else:
            sub = "F_X" if abs(mx) > abs(my) else "F_Y"
        for a, b, fsec in assign_rules[sub]:
            if a - 0.05 <= z1 and z2 <= b + 0.05:
                return int(fsec)
        # Fallback to nearest interval midpoint
        best = min(assign_rules[sub], key=lambda seg: abs(0.5 * (seg[0] + seg[1]) - zm))
        return int(best[2])

    elastics: dict[int, float] = {}
    aggregators: dict[int, tuple[int, int]] = {}
    elem_props: list[tuple[int, int, int, int, int, int, float]] = []
    loads: list[tuple[int, float]] = []

    for eid, n1, n2, transf in sorted(geom.megacol_elems, key=lambda x: x[0]):
        fsec = assign_fsec(n1, n2)
        cmat, h_val, b_val = triplet_meta[fsec]
        length = math.dist(geom.nodes[n1], geom.nodes[n2])
        el_tag = 10000000 + eid
        agg_tag = 20000000 + eid
        j_rect = (h_val * (b_val**3) + b_val * (h_val**3)) / 12.0
        e_tors = gc_by_cmat[cmat] * j_rect / length
        mass_per_len = rho * h_val * b_val
        elastics[el_tag] = e_tors
        aggregators[agg_tag] = (el_tag, fsec)
        elem_props.append((eid, n1, n2, n_ip, agg_tag, transf, mass_per_len))
        fz = -0.5 * mass_per_len * length * g
        loads.append((n1, fz))
        loads.append((n2, fz))

    return {
        "base_mats": os_p["mats"],
        "fiber_secs": fiber_secs,
        "elastics": elastics,
        "aggregators": aggregators,
        "elems": elem_props,
        "loads": loads,
    }


def _truss_key(
    p1: tuple[float, float, float], p2: tuple[float, float, float]
) -> tuple[tuple[str, int | float, int | float], float]:
    dx, dy = abs(p2[0] - p1[0]), abs(p2[1] - p1[1])
    mx = 0.5 * (p1[0] + p2[0])
    my = 0.5 * (p1[1] + p2[1])
    mz = 0.5 * (p1[2] + p2[2])
    if dx > 1e-3 and dy > 1e-3:
        return ("D", _clean_key(abs(mx), 1), _clean_key(abs(my), 1)), mz
    if dx > dy:
        return ("X", _clean_key(abs(my), 1), _clean_key(abs(mx), 2)), mz
    return ("Y", _clean_key(abs(mx), 1), _clean_key(abs(my), 2)), mz


def generate_truss_and_loads(geom: Geometry, props: dict[str, Any]) -> dict[str, Any]:
    tp = props["truss"]
    mats: dict[int, list[float]] = {}
    for tag_spec, vals in tp["mats"]:
        for tag in _expand_indices(tag_spec):
            mats[tag] = [float(x) for x in vals]

    ebc_default = [float(x) for x in tp["ebc_default"]]
    ebc_ex = [(float(x), float(y), float(z), float(a)) for x, y, z, a in tp["ebc_ex"]]
    ebc_lpl = float(tp["ebc_load_per_len"])
    t_mat = int(tp["truss_mat"])
    t_rho = float(tp["truss_rho"])
    g = float(tp["g"])

    rule_map: dict[tuple[str, float, float], list[list[float]]] = {}
    for key_list, segs in tp["truss_rules"]:
        k = (str(key_list[0]), float(key_list[1]), float(key_list[2]))
        rule_map[k] = [[float(a), float(b), float(v)] for a, b, v in segs]

    ebcs: list[tuple[int, int, int, list[float], int]] = []
    truss_loads: list[tuple[int, float]] = []

    for eid, n1, n2, transf in geom.ebc_elems:
        p1, p2 = geom.nodes[n1], geom.nodes[n2]
        mx, my, mz = 0.5 * (p1[0] + p2[0]), 0.5 * (p1[1] + p2[1]), 0.5 * (p1[2] + p2[2])
        ep = list(ebc_default)
        for ex, ey, ez, ea in ebc_ex:
            if abs(mx - ex) < 0.05 and abs(my - ey) < 0.05 and abs(mz - ez) < 0.05:
                ep[0] = ea
                break
        ebcs.append((eid, n1, n2, ep, transf))
        length = math.dist(p1, p2)
        fz = ebc_lpl * length
        truss_loads.append((n1, fz))
        truss_loads.append((n2, fz))

    trusses: list[tuple[int, int, int, float, int]] = []
    for eid, n1, n2 in geom.truss_elems:
        p1, p2 = geom.nodes[n1], geom.nodes[n2]
        k_raw, mz = _truss_key(p1, p2)
        k = (k_raw[0], float(k_raw[1]), float(k_raw[2]))
        segs = rule_map.get(k)
        if segs is None:
            # Nearest geometric key of same orientation
            cand = [rk for rk in rule_map if rk[0] == k[0]]
            best_k = min(cand, key=lambda rk: (rk[1] - k[1]) ** 2 + (rk[2] - k[2]) ** 2)
            segs = rule_map[best_k]
        area = None
        for a, b, val in segs:
            if a - 0.05 <= mz <= b + 0.05:
                area = val
                break
        if area is None:
            area = min(segs, key=lambda s: abs(0.5 * (s[0] + s[1]) - mz))[2]
        trusses.append((eid, n1, n2, area, t_mat))
        length = math.dist(p1, p2)
        fz = -0.5 * t_rho * area * length * g
        truss_loads.append((n1, fz))
        truss_loads.append((n2, fz))

    return {
        "mats": mats,
        "ebcs": ebcs,
        "trusses": trusses,
        "loads": truss_loads,
    }


def _canonical_octant(
    p1: tuple[float, float, float], p2: tuple[float, float, float]
) -> tuple[float, float, float, float]:
    mx = 0.5 * (p1[0] + p2[0])
    my = 0.5 * (p1[1] + p2[1])
    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]
    if mx < -1e-6:
        x1, x2, mx = -x1, -x2, -mx
    if my < -1e-6:
        y1, y2, my = -y1, -y2, -my
    if my > mx + 1e-6:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    if (round(x1, 3), round(y1, 3)) > (round(x2, 3), round(y2, 3)):
        x1, y1, x2, y2 = x2, y2, x1, y1
    return (x1, y1, x2, y2)


def _norm_bay(x: float, y: float, xz: float, yz: float) -> tuple[float, float]:
    nx = (
        x
        if (abs(x) <= 15.05 or xz <= 15.5)
        else (15.0 + (x - 15.0) * 20.0 / (xz - 15.0))
    )
    ny = (
        y
        if (abs(y) <= 15.05 or yz <= 15.5)
        else (15.0 + (y - 15.0) * 20.0 / (yz - 15.0))
    )
    return (nx, ny)


def _floor_bounds(
    nodes: dict[int, tuple[float, float, float]],
) -> tuple[
    dict[float, list[tuple[int, float, float]]], dict[float, float], dict[float, float]
]:
    nodes_by_z: dict[float, list[tuple[int, float, float]]] = collections.defaultdict(
        list
    )
    for n, (x, y, z) in nodes.items():
        nodes_by_z[round(z, 2)].append((n, x, y))
    x_max_z = {z: max(abs(x) for _, x, _ in lst) for z, lst in nodes_by_z.items()}
    y_max_z = {z: max(abs(y) for _, _, y in lst) for z, lst in nodes_by_z.items()}
    return nodes_by_z, x_max_z, y_max_z


def generate_hbeam(geom: Geometry, props: dict[str, Any]) -> dict[str, Any]:
    hp = props["hbeam"]
    mat_tag = int(hp["mat"])
    g_steel = float(hp["E"]) / (2.0 * (1.0 + float(hp["nu"])))
    rho = float(hp["rho"])
    g = float(hp["g"])
    n_ip = int(hp["nIP"])
    split_from = int(hp["split_from"])

    # 148 parametric W-sections -> patch rect lines, GJ elastics, Aggregators, patch areas
    fiber_secs: dict[int, list[tuple[int, int, int, float, float, float, float]]] = {}
    elastics: dict[int, float] = {}
    aggregators: dict[int, tuple[int, int]] = {}
    patch_area: dict[int, float] = {}  # sidx (1..148) -> sum_patch(A_p)

    for i0, (d, bf, tf, tw) in enumerate(hp["secs"]):
        sidx = i0 + 1
        d_f, bf_f, tf_f, tw_f = float(d), float(bf), float(tf), float(tw)
        fsec = 1200 + sidx
        el_tag = 1000 * sidx
        agg_tag = 10000 * sidx
        if sidx < split_from:
            patches = [
                (mat_tag, 8, 2, -bf_f / 2.0, -d_f / 2.0, bf_f / 2.0, -d_f / 2.0 + tf_f),
                (
                    mat_tag,
                    2,
                    8,
                    -tw_f / 2.0,
                    -(d_f - tf_f) / 2.0,
                    tw_f / 2.0,
                    (d_f - tf_f) / 2.0,
                ),
                (mat_tag, 8, 2, -bf_f / 2.0, d_f / 2.0 - tf_f, bf_f / 2.0, d_f / 2.0),
            ]
        else:
            patches = [
                (mat_tag, 4, 2, 0.0, -d_f / 2.0, bf_f / 2.0, -d_f / 2.0 + tf_f),
                (
                    mat_tag,
                    2,
                    8,
                    -tw_f / 2.0,
                    -(d_f - tf_f) / 2.0,
                    tw_f / 2.0,
                    (d_f - tf_f) / 2.0,
                ),
                (mat_tag, 4, 2, 0.0, d_f / 2.0 - tf_f, bf_f / 2.0, d_f / 2.0),
                (mat_tag, 4, 2, -bf_f / 2.0, d_f / 2.0 - tf_f, 0.0, d_f / 2.0),
                (mat_tag, 4, 2, -bf_f / 2.0, -d_f / 2.0, 0.0, -d_f / 2.0 + tf_f),
            ]
        fiber_secs[fsec] = patches
        a_tot = 0.0
        j_val = 0.0
        for _, _, _, y1, z1, y2, z2 in patches:
            ap = (y2 - y1) * (z2 - z1)
            yc = 0.5 * (y1 + y2)
            zc = 0.5 * (z1 + z2)
            a_tot += ap
            j_val += ap * (yc * yc + zc * zc)
        patch_area[sidx] = a_tot
        elastics[el_tag] = g_steel * j_val
        aggregators[agg_tag] = (el_tag, fsec)

    _, x_max_z, y_max_z = _floor_bounds(geom.nodes)

    # Decode vertical, sloped-column, brace, and horizontal rules
    def decode_key_z_rules(
        raw: list[list[Any]],
    ) -> dict[tuple[float, float], list[tuple[float, float, int]]]:
        out: dict[tuple[float, float], list[tuple[float, float, int]]] = {}
        for row in raw:
            k = (float(row[0]), float(row[1]))
            segs = [
                (float(row[j]), float(row[j + 1]), int(row[j + 2]))
                for j in range(2, len(row), 3)
            ]
            out[k] = segs
        return out

    v_rules = decode_key_z_rules(hp["v"])
    sc_rules = decode_key_z_rules(hp["sc"])
    br_rules: dict[float, list[tuple[float, float, int]]] = {}
    for row in hp["br"]:
        mz_k = float(row[0])
        br_rules[mz_k] = [
            (float(row[j]), float(row[j + 1]), int(row[j + 2]))
            for j in range(1, len(row), 3)
        ]

    # Floor index map for horizontal beams (aligned with sorted horizontal floor z's)
    h_zs = sorted(
        {
            round(geom.nodes[n1][2], 2)
            for _, n1, n2, _ in geom.hbeam_elems
            if abs(geom.nodes[n2][2] - geom.nodes[n1][2]) < 1e-3
        }
    )
    z_to_fidx = {z: i for i, z in enumerate(h_zs)}
    n_floors = len(h_zs)

    h_active: list[list[tuple[float, float, int]]] = [[] for _ in range(n_floors)]
    for row in hp["h"]:
        cx, cy = float(row[0]), float(row[1])
        for j in range(2, len(row), 3):
            a, b, sidx = int(row[j]), int(row[j + 1]), int(row[j + 2])
            for f in range(max(0, a), min(n_floors - 1, b) + 1):
                h_active[f].append((cx, cy, sidx))

    h_ex: dict[tuple[int, float, float], int] = {
        (int(f), float(mx), float(my)): int(s) for f, mx, my, s in hp["h_ex"]
    }

    def lookup_key_z(
        rmap: dict[tuple[float, float], list[tuple[float, float, int]]],
        k: tuple[float, float],
        mz: float,
    ) -> int:
        segs = rmap.get(k)
        if segs is None:
            best_k = min(rmap, key=lambda rk: (rk[0] - k[0]) ** 2 + (rk[1] - k[1]) ** 2)
            segs = rmap[best_k]
        for a, b, sidx in segs:
            if a - 0.05 <= mz <= b + 0.05:
                return sidx
        return min(segs, key=lambda s: abs(0.5 * (s[0] + s[1]) - mz))[2]

    elems_out: list[tuple[int, int, int, int, int, int, float]] = []
    node_fz: dict[int, float] = collections.defaultdict(float)

    for eid, n1, n2, transf in geom.hbeam_elems:
        p1, p2 = geom.nodes[n1], geom.nodes[n2]
        dx = abs(p2[0] - p1[0])
        dy = abs(p2[1] - p1[1])
        dz = abs(p2[2] - p1[2])

        if dz < 1e-3:
            z = round(p1[2], 2)
            fidx = z_to_fidx[z]
            amx = float(_clean_key(0.5 * (p1[0] + p2[0]), 1))
            amy = float(_clean_key(0.5 * (p1[1] + p2[1]), 1))
            ex_s = h_ex.get((fidx, amx, amy))
            if ex_s is not None:
                sidx = ex_s
            else:
                xz, yz = x_max_z[z], y_max_z[z]
                c1, c2, c3, c4 = _canonical_octant(p1, p2)
                q1 = _norm_bay(c1, c2, xz, yz)
                q2 = _norm_bay(c3, c4, xz, yz)
                mx, my = 0.5 * (q1[0] + q2[0]), 0.5 * (q1[1] + q2[1])
                sidx = min(
                    h_active[fidx],
                    key=lambda r: (r[0] - mx) ** 2 + (r[1] - my) ** 2,
                )[2]
        elif dx < 1e-3 and dy < 1e-3:
            ax, ay = abs(p1[0]), abs(p1[1])
            if ay > ax:
                ax, ay = ay, ax
            mz = 0.5 * (p1[2] + p2[2])
            sidx = lookup_key_z(
                v_rules, (float(_clean_key(ax, 1)), float(_clean_key(ay, 1))), mz
            )
        elif math.hypot(dx, dy) < 0.5 * dz:
            z1, z2 = round(min(p1[2], p2[2]), 2), round(max(p1[2], p2[2]), 2)
            mz = 0.5 * (z1 + z2)
            xm = 0.5 * (x_max_z[z1] + x_max_z[z2])
            ym = 0.5 * (y_max_z[z1] + y_max_z[z2])
            ax = abs(0.5 * (p1[0] + p2[0])) / xm
            ay = abs(0.5 * (p1[1] + p2[1])) / ym
            if ay > ax:
                ax, ay = ay, ax
            sidx = lookup_key_z(
                sc_rules, (float(_clean_key(ax, 1)), float(_clean_key(ay, 1))), mz
            )
        else:
            mx = abs(0.5 * (p1[0] + p2[0]))
            my = abs(0.5 * (p1[1] + p2[1]))
            if my > mx:
                mx, my = my, mx
            mz = 0.5 * (p1[2] + p2[2])
            mz_k = min(br_rules, key=lambda zk: abs(zk - mz))
            sidx = min(
                br_rules[mz_k], key=lambda r: (r[0] - mx) ** 2 + (r[1] - my) ** 2
            )[2]

        agg_tag = 10000 * sidx
        mass_per_len = rho * patch_area[sidx]
        elems_out.append((eid, n1, n2, n_ip, agg_tag, transf, mass_per_len))
        length = math.dist(p1, p2)
        half_w = -0.5 * mass_per_len * length * g
        node_fz[n1] += half_w
        node_fz[n2] += half_w

    loads_out = [(n, node_fz[n]) for n in sorted(node_fz)]
    return {
        "fiber_secs": fiber_secs,
        "elastics": elastics,
        "aggregators": aggregators,
        "elems": elems_out,
        "loads": loads_out,
    }


def _poly_area_2d(pts: list[tuple[float, float]]) -> float:
    s = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return 0.5 * abs(s)


def _map_pt(
    px: float,
    py: float,
    xt: float,
    yt: float,
    xz: float,
    yz: float,
    zn_list: list[tuple[int, float, float]],
) -> int:
    best_n = -1
    best_d = 1e18
    for n, nx, ny in zn_list:
        d = math.hypot(nx - px, ny - py)
        if d < best_d:
            best_d = d
            best_n = n
    if best_d < 0.03:
        return best_n
    sx = (
        px
        if (any(abs(px - g) < 0.02 for g in GRID_VALS) and abs(px) < abs(py))
        else px * (xz / xt)
    )
    sy = (
        py
        if (any(abs(py - g) < 0.02 for g in GRID_VALS) and abs(py) < abs(px))
        else py * (yz / yt)
    )
    best_n = -1
    best_d = 1e18
    for n, nx, ny in zn_list:
        d = math.hypot(nx - sx, ny - sy)
        if d < best_d:
            best_d = d
            best_n = n
    return best_n


def _sym_ops(code: int) -> tuple[tuple[tuple[Any, int], ...], int]:
    if code == 0:
        return ((lambda x, y: (x, y), 0),), 0
    if code == 1:
        return C4_OPS, 0
    if code == 2:
        return C2_OPS, 0
    if code in (3, 4, 5):
        return D4_OPS, code - 3
    if code in (6, 7, 8):
        return V4_OPS, code - 6
    raise ValueError(f"Unknown symmetry code {code}")


def generate_rigiddigram(geom: Geometry, props: dict[str, Any]) -> dict[str, Any]:
    rp = props["rigiddigram"]
    core_cutoff_z = float(rp["core_cutoff_z"])
    q_segs = [(float(a), float(b), float(cq), float(nq)) for a, b, cq, nq in rp["q"]]
    g_pts = [float(x) for x in rp["p"]]
    v0_ov_map = {(int(f), int(o), int(op)): int(v) for f, o, op, v in rp["v0_ov"]}

    nodes_by_z, x_max_z, y_max_z = _floor_bounds(geom.nodes)
    # 128 diaphragm floor elevations sorted ascending
    floor_zs = sorted(round(geom.nodes[m][2], 2) for _, m, _ in geom.diaphragms)

    # Map floor index -> template entry [xt, yt, orbits]
    fidx_to_tpl: dict[int, tuple[float, float, list[list[int]]]] = {}
    for xt, yt, fspec, orbits in rp["t"]:
        for fidx in _expand_indices(fspec):
            fidx_to_tpl[fidx] = (float(xt), float(yt), orbits)

    def lookup_q(z: float) -> tuple[float, float]:
        for a, b, cq, nq in q_segs:
            if a - 0.05 <= z <= b + 0.05:
                return cq, nq
        best = min(q_segs, key=lambda s: abs(0.5 * (s[0] + s[1]) - z))
        return best[2], best[3]

    panels: list[tuple[list[int], float]] = []
    loads: list[tuple[int, float]] = []
    node_fz: dict[int, float] = collections.defaultdict(float)

    for fidx, z in enumerate(floor_zs):
        cq, nq = lookup_q(z)
        zn = nodes_by_z[z]
        xz, yz = x_max_z[z], y_max_z[z]
        # Exact point lookup on floor z for active core bays
        by_xy = {(round(nx, 1), round(ny, 1)): n for n, nx, ny in zn}

        # 1. Deterministic active core bays (9 bays for z < 270, 5 '+' bays for z >= 270)
        for x0 in (-15.0, -5.0, 5.0):
            for y0 in (-15.0, -5.0, 5.0):
                if z >= core_cutoff_z and x0 != -5.0 and y0 != -5.0:
                    continue
                corners = [
                    by_xy[(x0, y0)],
                    by_xy[(x0 + 10.0, y0)],
                    by_xy[(x0 + 10.0, y0 + 10.0)],
                    by_xy[(x0, y0 + 10.0)],
                ]
                area = _poly_area_2d(
                    [(geom.nodes[n][0], geom.nodes[n][1]) for n in corners]
                )
                fz = -0.25 * cq * area
                panels.append((corners, fz))
                for n in corners:
                    loads.append((n, fz))
                    node_fz[n] += fz

        # 2. Non-core panels from compact D4/V4/C4/C2 floor templates
        xt, yt, orbits = fidx_to_tpl[fidx]
        seen: set[frozenset[int]] = set()
        for oidx, orb in enumerate(orbits):
            code = orb[0]
            pidxs = orb[1:]
            is_tri = len(pidxs) == 3
            ops, ref_idx = _sym_ops(code)
            for opidx, (op, is_ref) in enumerate(ops):
                m_ns = []
                for pidx in pidxs:
                    px, py = g_pts[2 * pidx], g_pts[2 * pidx + 1]
                    ox, oy = op(px, py)
                    m_ns.append(_map_pt(ox, oy, xt, yt, xz, yz, zn))
                gk = frozenset(m_ns)
                if gk in seen:
                    continue
                seen.add(gk)
                area = _poly_area_2d(
                    [(geom.nodes[n][0], geom.nodes[n][1]) for n in m_ns]
                )
                fz = -0.25 * nq * area
                if is_tri:
                    ov = v0_ov_map.get((fidx, oidx, opidx))
                    if ov is not None:
                        v0_i = ov
                    else:
                        v0_i = ref_idx if is_ref else 0
                    rot = m_ns[v0_i:] + m_ns[:v0_i]
                    p_nodes = [rot[0], rot[1], rot[2], rot[0]]
                else:
                    p_nodes = m_ns
                panels.append((p_nodes, fz))
                for n in p_nodes:
                    loads.append((n, fz))
                    node_fz[n] += fz

    return {
        "panels": panels,
        "loads": loads,
        "node_fz": dict(node_fz),
    }


def _shell_family(pts: list[tuple[float, float, float]]) -> str:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    zs = [p[2] for p in pts]
    dx = max(xs) - min(xs)
    dy = max(ys) - min(ys)
    dz = max(zs) - min(zs)
    cx = sum(xs) / 4.0
    cy = sum(ys) / 4.0
    if dy >= 1e-3 and dx >= 1e-3:
        return "CB_DIAG"
    if dy < 1e-3:
        plane = "Y"
        pcoord = round(abs(cy), 1)
        along = round(abs(cx), 2)
    else:
        plane = "X"
        pcoord = round(abs(cx), 1)
        along = round(abs(cy), 2)
    if pcoord == 15.0:
        if dz < 1.5 and (along < 1.0 or abs(along - 10.0) < 0.05):
            if abs(along - 10.0) < 0.05 and plane == "Y":
                return "W_Y15_FAR"
            return "CB_OUT"
        if along > 13.0:
            return "W_OUT_CORNER"
        if plane == "X":
            return "W_X15_MID"
        return "W_Y15_NEAR" if along < 5.0 else "W_Y15_FAR"
    if dz < 1.5 and along < 1.0:
        return "CB_IN"
    return "W_IN_NEAR" if along < 5.0 else "W_IN_FAR"


def _quad_area_3d(
    p1: tuple[float, float, float],
    p2: tuple[float, float, float],
    p3: tuple[float, float, float],
    p4: tuple[float, float, float],
) -> float:
    def tri(
        a: tuple[float, float, float],
        b: tuple[float, float, float],
        c: tuple[float, float, float],
    ) -> float:
        ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
        cx = uy * vz - uz * vy
        cy = uz * vx - ux * vz
        cz = ux * vy - uy * vx
        return 0.5 * math.sqrt(cx * cx + cy * cy + cz * cz)

    return tri(p1, p2, p3) + tri(p1, p3, p4)


def generate_layershell(
    geom: Geometry,
    props: dict[str, Any],
    rd_node_fz: dict[int, float] | None = None,
) -> dict[str, Any]:
    lp = props["layershell"]
    rho_c = float(lp["rho_conc"])
    rho_r = float(lp["rho_rebar"])
    rho_p = float(lp["rho_plate_layer"])
    g = float(lp["g"])

    if rd_node_fz is None:
        rd_node_fz = generate_rigiddigram(geom, props)["node_fz"]

    # Expand 38 profiles -> 91 LayeredShell sections
    secs: dict[int, list[tuple[int, float]]] = {}
    mass_per_area: dict[int, float] = {}
    for prof, tags in lp["secs"]:
        cmat = int(prof[0])
        t_cov, t_r90, t_r0, t_core = (
            float(prof[1]),
            float(prof[2]),
            float(prof[3]),
            float(prof[4]),
        )
        t_plate = float(prof[5]) if len(prof) > 5 else 0.0
        if t_plate == 0.0:
            lays = (
                [(cmat, t_cov), (77998, t_r90), (7798, t_r0)]
                + [(cmat, t_core)] * 10
                + [(7798, t_r0), (77998, t_r90), (cmat, t_cov)]
            )
        else:
            lays = (
                [(cmat, t_cov), (77998, t_r90), (7798, t_r0)]
                + [(cmat, t_core)] * 5
                + [(7698, t_plate), (76998, t_plate)]
                + [(cmat, t_core)] * 5
                + [(7798, t_r0), (77998, t_r90), (cmat, t_cov)]
            )
        mpa = (
            rho_c * (2.0 * t_cov + 10.0 * t_core)
            + rho_r * (2.0 * t_r90 + 2.0 * t_r0)
            + rho_p * (2.0 * t_plate)
        )
        for stag in tags:
            secs[int(stag)] = lays
            mass_per_area[int(stag)] = mpa

    assign_rules: dict[str, list[tuple[float, float, int]]] = {}
    for fam, flat in lp["assign"].items():
        assign_rules[fam] = [
            (float(flat[j]), float(flat[j + 1]), int(flat[j + 2]))
            for j in range(0, len(flat), 3)
        ]

    elems_out: list[tuple[int, int, int, int, int, int]] = []
    elem_sec_by_eid: dict[int, int] = {}
    for eid, n1, n2, n3, n4 in geom.shell_elems:
        pts = [geom.nodes[n] for n in (n1, n2, n3, n4)]
        fam = _shell_family(pts)
        cz = sum(p[2] for p in pts) / 4.0
        segs = assign_rules[fam]
        stag = None
        for a, b, s in segs:
            if a - 0.05 <= cz <= b + 0.05:
                stag = s
                break
        if stag is None:
            stag = min(segs, key=lambda s: abs(0.5 * (s[0] + s[1]) - cz))[2]
        elems_out.append((eid, n1, n2, n3, n4, stag))
        elem_sec_by_eid[eid] = stag

    # Loads are ordered by eid ascending, 4 identical nodal loads per element
    loads_out: list[tuple[int, float]] = []
    shell_node_fz: dict[int, float] = collections.defaultdict(float)
    for eid, n1, n2, n3, n4, stag in sorted(elems_out, key=lambda x: x[0]):
        area = _quad_area_3d(
            geom.nodes[n1], geom.nodes[n2], geom.nodes[n3], geom.nodes[n4]
        )
        fz = -0.25 * g * area * mass_per_area[stag]
        for n in (n1, n2, n3, n4):
            loads_out.append((n, fz))
            shell_node_fz[n] += fz

    # Nodal masses = -(Fz_shell + Fz_rigiddigram) / g for every node with non-zero sum
    masses: dict[int, float] = {}
    for n in sorted(set(shell_node_fz) | set(rd_node_fz)):
        tot_fz = shell_node_fz.get(n, 0.0) + rd_node_fz.get(n, 0.0)
        if abs(tot_fz) > 1e-12:
            masses[n] = -tot_fz / g

    return {
        "mats": lp["mats"],
        "secs": secs,
        "elems": elems_out,
        "masses": masses,
        "loads": loads_out,
    }


def generate_dynamiccpu(geom: Geometry, props: dict[str, Any]) -> str:
    dp = props["dynamiccpu"]
    base_nodes = sorted(n for n, p in geom.nodes.items() if abs(p[2]) < 1e-6)
    sx, sy = float(dp["story_xy"][0]), float(dp["story_xy"][1])
    floor_zs = sorted(round(geom.nodes[m][2], 2) for _, m, _ in geom.diaphragms)
    nodes_by_z, _, _ = _floor_bounds(geom.nodes)
    story_nodes: list[int] = []
    for z in floor_zs:
        best_n = min(
            nodes_by_z[z], key=lambda item: (item[1] - sx) ** 2 + (item[2] - sy) ** 2
        )[0]
        story_nodes.append(best_n)
    max_node = max(geom.nodes)

    grav = dp["gravity"]
    damp = dp["damping"]
    gm = dp["gm"]
    tr = dp["transient"]

    reac_str = " ".join(str(n) for n in base_nodes)
    story_str = "\t".join(str(n) for n in story_nodes)
    drift_i = "\t".join(str(n) for n in story_nodes[:-1])
    drift_j = "\t".join(str(n) for n in story_nodes[1:])

    return (
        "wipe\n"
        "#unit: N, m, Pa\n"
        f"model BasicBuilder -ndm {dp['ndm']} -ndf {dp['ndf']}\n\n"
        "source opensees.tcl\n"
        'puts "fiber element ok"\n\n'
        "source truss.tcl\n"
        'puts "truss ok"\n\n'
        "source trussload.tcl\n"
        'puts "trussload ok"\n\n'
        "source Hbeam.tcl\n"
        'puts "Hbeam ok"\n\n'
        "source layershell.tcl\n"
        'puts "layershell ok"\n\n'
        "source rigiddigram.tcl\n"
        'puts "rigiddigram ok"\n\n'
        f"recorder Node -file reaction.txt -time -node  {reac_str} -dof 3 reaction;\n\n"
        f"recorder Node -file storyx.txt -time -node {story_str} -dof 1 disp;\n\n"
        f"recorder Drift   -file drift1.txt  -iNode {drift_i}\t  -jNode  \t{drift_j} -dof 1 -perpDirn 3\n\n"
        f"recorder Node -file dispx.txt -time -nodeRange 1 {max_node} -dof 1 disp;\n"
        f"recorder Node -file dispy.txt -time -nodeRange 1 {max_node} -dof 2 disp;\n"
        f"recorder Node -file dispz.txt -time -nodeRange 1 {max_node} -dof 3 disp;\n\n"
        "fixZ 0 1 1 1 1 1 1\n\n"
        f"constraints {grav['constraints']};\n"
        f"numberer {grav['numberer']};\n"
        f"system {grav['system']};\n"
        f"test {' '.join(str(x) for x in grav['test'])};\n"
        f"algorithm {grav['algorithm']}\n"
        f"integrator {' '.join(str(x) for x in grav['integrator'])};\n"
        "analysis Static\n"
        f"analyze {grav['steps']}\n"
        'puts "gravity ok"\n'
        "loadConst -time 0.0;\n"
        f"set xDamp {damp['xDamp']};\n"
        f"set MpropSwitch {damp['MpropSwitch']};\n"
        f"set KcurrSwitch {damp['KcurrSwitch']};\n"
        f"set KcommSwitch {damp['KcommSwitch']};\n"
        f"set KinitSwitch {damp['KinitSwitch']};\n"
        f"set nEigenI {damp['nEigenI']};\n"
        f"set nEigenJ {damp['nEigenJ']};\n\n"
        f"set omegaI {damp['omegaI']}\n"
        f"set omegaJ {damp['omegaJ']}\n"
        "set alphaM [expr $MpropSwitch*$xDamp*(2*$omegaI*$omegaJ)/($omegaI+$omegaJ)];\n"
        "set betaKcurr [expr $KcurrSwitch*2.*$xDamp/($omegaI+$omegaJ)];\n"
        "set betaKcomm [expr $KcommSwitch*2.*$xDamp/($omegaI+$omegaJ)];\n"
        "set betaKinit [expr $KinitSwitch*2.*$xDamp/($omegaI+$omegaJ)];\n"
        "rayleigh $alphaM $betaKcurr $betaKinit $betaKcomm;\n"
        f"set IDloadTag {gm['loadTag']};\n"
        f'set iGMfile "{gm["file"]}";\n'
        f'set iGMdirection "{gm["direction"]}";\n'
        f'set iGMfact "{gm["factor"]}";\n'
        f"set dt {gm['dt']};\n"
        "set GMfatt [expr $iGMfact];\n"
        'set AccelSeries "Series -dt $dt -filePath $iGMfile -factor $GMfatt";\n'
        "pattern UniformExcitation $IDloadTag $iGMdirection -accel $AccelSeries;\n\n"
        f"constraints {tr['constraints']};\n"
        f"numberer {tr['numberer']};\n"
        f"system {tr['system']}\n"
        f"test {' '.join(str(x) for x in tr['test'])}\n"
        f"algorithm {' '.join(str(x) for x in tr['algorithm'])}\n"
        f"integrator {' '.join(str(x) for x in tr['integrator'])}\n"
        "analysis Transient\n"
        'puts "ok"\n\n'
        f"set Nsteps {tr['Nsteps']}\n"
        f"set DtAnalysis {tr['DtAnalysis']}\n"
        f"set TmaxAnalysis {tr['TmaxAnalysis']}\n"
        "set ok [analyze $Nsteps $DtAnalysis];\n\n"
        "if {$ok != 0} {\n"
        "\tset ok 0;\n"
        "\tset controlTime [getTime];\n"
        "\twhile {$controlTime < $TmaxAnalysis && $ok == 0} {\n"
        "\t\tset controlTime [getTime]\n"
        "\t\tset ok [analyze 1 $DtAnalysis]\n"
        "\t\tif {$ok != 0} {\n"
        '\t\t\tputs "try krylov and change test"\n'
        "\t\t\ttest NormDispIncr 5e-2 100 2\n"
        "\t\t\talgorithm KrylovNewton\n"
        "\t\t\tset ok [analyze 1 $DtAnalysis]\n"
        "\t\t\ttest NormDispIncr 5e-2 10 2\n"
        "\t\t\talgorithm NewtonLineSearch 0.75\n"
        "\t\t}\n"
        "\t\tif {$ok != 0} {\n"
        '\t\t\tputs "try relative"\n'
        "\t\t\ttest RelativeNormDispIncr 0.1 10 2\n"
        "\t\t\tset ok [analyze 1 $DtAnalysis]\n"
        "\t\t\ttest NormDispIncr 5e-2 10 2\n"
        "\t\t}\n"
        "\t\tif {$ok != 0} {\n"
        '\t\t\tputs "try relative and krylov"\n'
        "\t\t\ttest RelativeNormDispIncr 0.1 10 2\n"
        "\t\t\talgorithm KrylovNewton\n"
        "\t\t\tset ok [analyze 1 $DtAnalysis]\n"
        "\t\t\ttest NormDispIncr 5e-2 10 2\n"
        "\t\t\talgorithm NewtonLineSearch 0.75\n"
        "\t\t}\n"
        "\t\tif {$ok != 0} {\n"
        '\t\t\tputs "Trying Broyden .."\n'
        "\t\t\talgorithm Broyden 8\n"
        "\t\t\tset ok [analyze 1 $DtAnalysis]\n"
        "\t\t\talgorithm NewtonLineSearch 0.75\n"
        "\t\t}\n"
        "\t\tif {$ok != 0} {\n"
        '\t\t\tputs "Trying NewtonWithLineSearch .."\n'
        "\t\t\talgorithm NewtonLineSearch .8\n"
        "\t\t\tset ok [analyze 1 $DtAnalysis]\n"
        "\t\t\talgorithm NewtonLineSearch 0.75\n"
        "\t\t}\n"
        "\t}\n"
        "};\n"
    )


def generate_all(geom: Geometry, props: dict[str, Any]) -> dict[str, Any]:
    """Generate all non-geometric model structures for all 7 Model1 Tcl files."""
    opensees = generate_opensees(geom, props)
    truss_all = generate_truss_and_loads(geom, props)
    hbeam = generate_hbeam(geom, props)
    rigiddigram = generate_rigiddigram(geom, props)
    layershell = generate_layershell(geom, props, rd_node_fz=rigiddigram["node_fz"])
    dynamiccpu = generate_dynamiccpu(geom, props)
    return {
        "opensees": opensees,
        "truss": truss_all,
        "hbeam": hbeam,
        "rigiddigram": rigiddigram,
        "layershell": layershell,
        "dynamiccpu": dynamiccpu,
    }


def format_tcl_files(geom: Geometry, gen: dict[str, Any]) -> dict[str, str]:
    """Render all 7 OpenSees Tcl files combining ``geom`` and generated properties."""
    files: dict[str, str] = {}

    # 1. opensees.tcl
    os_d = gen["opensees"]
    lines: list[str] = []
    for nid, (x, y, z) in sorted(geom.nodes.items()):
        lines.append(f"node {nid} {x:.6g} {y:.6g} {z:.6g}")
    for mtype, tag, vals in os_d["base_mats"]:
        vstr = " ".join(f"{v:.10g}" for v in vals)
        lines.append(f"uniaxialMaterial {mtype} {tag} {vstr}")
    for sid, fibs in sorted(os_d["fiber_secs"].items()):
        lines.append(f"section Fiber {sid} {{")
        for y, z, a, m in fibs:
            lines.append(f"  fiber {y:.5g} {z:.5g} {a:.5f} {m}")
        lines.append("}")
    for tid, ttype, vx, vy, vz in geom.geom_transfs:
        lines.append(f"geomTransf {ttype} {tid} {vx:.6g} {vy:.6g} {vz:.6g}")
    for el_tag, e_val in sorted(os_d["elastics"].items()):
        lines.append(f"uniaxialMaterial Elastic {el_tag} {e_val:.10g}")
    for agg_tag, (el_tag, fsec) in sorted(os_d["aggregators"].items()):
        lines.append(f"section Aggregator {agg_tag} {el_tag} T -section {fsec}")
    for eid, n1, n2, n_ip, agg_tag, transf, mass in os_d["elems"]:
        lines.append(
            f"element dispBeamColumn {eid} {n1} {n2} {n_ip} {agg_tag} {transf} -mass {mass:.10f}"
        )
    lines.append("pattern Plain 1 Linear {")
    for n, fz in os_d["loads"]:
        lines.append(f"  load {n} 0 0 {fz:.10f} 0 0 0")
    lines.append("}")
    files["opensees.tcl"] = "\n".join(lines) + "\n"

    # 2. truss.tcl
    tr_d = gen["truss"]
    lines = []
    for tag, vals in sorted(tr_d["mats"].items()):
        vstr = " ".join(f"{v:.15g}" for v in vals)
        lines.append(f"uniaxialMaterial Steel02 {tag} {vstr}")
    for eid, n1, n2, ep, transf in tr_d["ebcs"]:
        pstr = " ".join(f"{v:.10g}" for v in ep)
        lines.append(f"element elasticBeamColumn {eid} {n1} {n2} {pstr} {transf}")
    for eid, n1, n2, area, mtag in tr_d["trusses"]:
        lines.append(f"element truss {eid} {n1} {n2} {area:.6g} {mtag}")
    files["truss.tcl"] = "\n".join(lines) + "\n"

    # 3. trussload.tcl
    lines = ["pattern Plain 68 Linear {"]
    for n, fz in tr_d["loads"]:
        lines.append(f"  load {n} 0 0 {fz:.10f} 0 0 0")
    lines.append("}")
    files["trussload.tcl"] = "\n".join(lines) + "\n"

    # 4. Hbeam.tcl
    hb_d = gen["hbeam"]
    lines = []
    for sidx in range(1, len(hb_d["fiber_secs"]) + 1):
        fsec = 1200 + sidx
        el_tag = 1000 * sidx
        agg_tag = 10000 * sidx
        lines.append(f"section Fiber {fsec} {{")
        for m, ny, nz, y1, z1, y2, z2 in hb_d["fiber_secs"][fsec]:
            lines.append(
                f"  patch rect {m} {ny} {nz} {y1:.7g} {z1:.7g} {y2:.7g} {z2:.7g}"
            )
        lines.append("}")
        lines.append(
            f"uniaxialMaterial Elastic {el_tag} {hb_d['elastics'][el_tag]:.10g}"
        )
        lines.append(f"section Aggregator {agg_tag} {el_tag} T -section {fsec}")
    for eid, n1, n2, n_ip, agg_tag, transf, mass in hb_d["elems"]:
        lines.append(
            f"element dispBeamColumn {eid} {n1} {n2} {n_ip} {agg_tag} {transf} -mass {mass:.6f}"
        )
    lines.append("pattern Plain 10 Linear {")
    for n, fz in hb_d["loads"]:
        lines.append(f"  load {n} 0 0 {fz:.10f} 0 0 0")
    lines.append("}")
    files["Hbeam.tcl"] = "\n".join(lines) + "\n"

    # 5. layershell.tcl
    ls_d = gen["layershell"]
    lines = []
    for cmd, mtype, tag, vals in ls_d["mats"]:
        vstr = " ".join(f"{v:.10g}" for v in vals)
        lines.append(f"{cmd} {mtype} {tag} {vstr}")
    for stag, lays in sorted(ls_d["secs"].items()):
        lstr = " ".join(f"{m} {t:.6g}" for m, t in lays)
        lines.append(f"section LayeredShell {stag} {len(lays)} {lstr}")
    for eid, n1, n2, n3, n4, stag in ls_d["elems"]:
        lines.append(f"element ShellMITC4 {eid} {n1} {n2} {n3} {n4} {stag}")
    for n, m in sorted(ls_d["masses"].items()):
        lines.append(f"mass {n} {m:.10f} {m:.10f} {m:.10f} 0 0 0")
    lines.append("pattern Plain 2 Linear {")
    for n, fz in ls_d["loads"]:
        lines.append(f"  load {n} 0 0 {fz:.10f} 0 0 0")
    lines.append("}")
    files["layershell.tcl"] = "\n".join(lines) + "\n"

    # 6. rigiddigram.tcl
    rd_d = gen["rigiddigram"]
    lines = ["pattern Plain 6 Linear {"]
    for n, fz in rd_d["loads"]:
        lines.append(f"  load {n} 0 0 {fz:.10f} 0 0 0")
    lines.append("}")
    for perp, master, slaves in geom.diaphragms:
        sstr = " ".join(str(x) for x in slaves)
        lines.append(f"rigidDiaphragm {perp} {master} {sstr}")
    files["rigiddigram.tcl"] = "\n".join(lines) + "\n"

    # 7. dynamiccpu.tcl
    files["dynamiccpu.tcl"] = gen["dynamiccpu"]
    return files


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--props", type=Path, default=DEFAULT_PROPS_PATH)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Optional directory to write reconstructed Tcl files",
    )
    args = ap.parse_args()

    geom = load_real_geometry(args.model_dir)
    props = load_props(args.props)
    gen = generate_all(geom, props)
    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        rendered = format_tcl_files(geom, gen)
        for name, content in rendered.items():
            (args.out_dir / name).write_text(content)
            print(f"Wrote {args.out_dir / name} ({len(content):,} bytes)")
    else:
        print(
            f"Generated Model1 properties from {args.props} "
            f"({args.props.stat().st_size:,} bytes):"
        )
        print(
            f"  opensees   : {len(gen['opensees']['fiber_secs'])} Fiber secs, "
            f"{len(gen['opensees']['elems'])} elems, {len(gen['opensees']['loads'])} loads"
        )
        print(
            f"  truss      : {len(gen['truss']['mats'])} mats, "
            f"{len(gen['truss']['ebcs'])} EBCs, {len(gen['truss']['trusses'])} trusses, "
            f"{len(gen['truss']['loads'])} loads"
        )
        print(
            f"  Hbeam      : {len(gen['hbeam']['fiber_secs'])} Fiber secs, "
            f"{len(gen['hbeam']['elems'])} elems, {len(gen['hbeam']['loads'])} loads"
        )
        print(
            f"  layershell : {len(gen['layershell']['mats'])} mats, "
            f"{len(gen['layershell']['secs'])} secs, {len(gen['layershell']['elems'])} elems, "
            f"{len(gen['layershell']['masses'])} masses, {len(gen['layershell']['loads'])} loads"
        )
        print(
            f"  rigiddigram: {len(gen['rigiddigram']['panels'])} panels, "
            f"{len(gen['rigiddigram']['loads'])} loads"
        )
        print(f"  dynamiccpu : {len(gen['dynamiccpu'])} bytes")


if __name__ == "__main__":
    main()
