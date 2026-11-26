"""Extract compact non-geometric property tables (props.json) from Model1.

Stdlib-only. Reads the real OpenSees MegaTallBuildingBenchmark Model1 files
and extracts minimal parametric rules + small tables into ``props.json``
(< 50 KB total including exceptions).
"""

from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path
from typing import Any

DEFAULT_MODEL_DIR = (
    Path(__file__).resolve().parents[2]
    / "tests/fixtures/opensees/OpenSeesExamples/MegaTallBuildingBenchmark/Model1"
)
DEFAULT_OUT_PATH = Path(__file__).resolve().parent / "props.json"

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

# D4 / V4 / C4 / C2 planar symmetry operations: (fn(x, y), is_reflection)
D4_OPS = [
    (lambda x, y: (x, y), 0),
    (lambda x, y: (-y, x), 0),
    (lambda x, y: (-x, -y), 0),
    (lambda x, y: (y, -x), 0),
    (lambda x, y: (-x, y), 1),
    (lambda x, y: (x, -y), 1),
    (lambda x, y: (y, x), 1),
    (lambda x, y: (-y, -x), 1),
]
C4_OPS = D4_OPS[:4]
V4_OPS = [
    (lambda x, y: (x, y), 0),
    (lambda x, y: (-x, y), 1),
    (lambda x, y: (x, -y), 1),
    (lambda x, y: (-x, -y), 0),
]
C2_OPS = [
    (lambda x, y: (x, y), 0),
    (lambda x, y: (-x, -y), 0),
]


def clean_num(v: float, nd: int = 6) -> int | float:
    r = round(v, nd)
    if abs(r) < 1e-12:
        return 0
    if abs(r - round(r)) < 1e-9:
        return round(r)
    return r


def compress_indices(idxs: list[int]) -> list[int | list[int]]:
    if not idxs:
        return []
    out: list[int | list[int]] = []
    a = b = idxs[0]
    for x in idxs[1:]:
        if x == b + 1:
            b = x
        else:
            out.append(a if a == b else [a, b])
            a = b = x
    out.append(a if a == b else [a, b])
    return out


def parse_nodes(model_dir: Path) -> dict[int, tuple[float, float, float]]:
    nodes: dict[int, tuple[float, float, float]] = {}
    with (model_dir / "opensees.tcl").open() as f:
        for line in f:
            toks = line.split()
            if toks and toks[0] == "node":
                nodes[int(toks[1])] = (float(toks[2]), float(toks[3]), float(toks[4]))
    return nodes


def canonical_octant(
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


def norm_bay(x: float, y: float, xz: float, yz: float) -> tuple[float, float]:
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


def _compress_steel101_bars(
    s101: list[tuple[float, float, float, int]],
) -> list[list[Any]]:
    """Compress ordered steel-101 fibers into compact uniform bar groups.

    Each returned item is ``[axis, fixed_spec, lo, hi, n, A]`` where:
    - ``axis`` is ``"z"`` (fixed y, uniform z from lo..hi) or ``"y"`` (fixed z, uniform y)
    - ``fixed_spec`` is either a list of numbers, or a list of ``[start, end, count]``
      uniform arithmetic progressions / scalars.
    """
    runs: list[tuple[str, float, float, float, int, float]] = []
    i = 0
    while i < len(s101):
        y0, z0, a0, _ = s101[i]
        jz = i + 1
        while jz < len(s101) and s101[jz][2] == a0 and abs(s101[jz][1] - z0) < 1e-6:
            jz += 1
        jy = i + 1
        while jy < len(s101) and s101[jy][2] == a0 and abs(s101[jy][0] - y0) < 1e-6:
            jy += 1
        if jy >= jz:
            grp = s101[i:jy]
            zs = [x[1] for x in grp]
            runs.append(("z", y0, zs[0], zs[-1], len(grp), a0))
            i = jy
        else:
            grp = s101[i:jz]
            ys = [x[0] for x in grp]
            runs.append(("y", z0, ys[0], ys[-1], len(grp), a0))
            i = jz

    # Group consecutive runs that share (axis, lo, hi, n, A)
    grouped: list[list[Any]] = []
    k = 0
    while k < len(runs):
        ax, f0, lo, hi, n, a0 = runs[k]
        fixed = [f0]
        m = k + 1
        while m < len(runs):
            ax2, fm, lo2, hi2, n2, a2 = runs[m]
            if (
                ax2 == ax
                and n2 == n
                and a2 == a0
                and abs(lo2 - lo) < 1e-6
                and abs(hi2 - hi) < 1e-6
            ):
                fixed.append(fm)
                m += 1
            else:
                break
        # Compress runs of >= 4 evenly spaced values in `fixed` into [start, end, count]
        fspec: list[Any] = []
        p = 0
        while p < len(fixed):
            if p + 3 < len(fixed):
                step = fixed[p + 1] - fixed[p]
                q = p + 2
                while q < len(fixed) and abs((fixed[q] - fixed[q - 1]) - step) < 2e-4:
                    q += 1
                if q - p >= 4:
                    fspec.append(
                        [clean_num(fixed[p], 5), clean_num(fixed[q - 1], 5), q - p]
                    )
                    p = q
                    continue
            fspec.append(clean_num(fixed[p], 5))
            p += 1
        grouped.append(
            [
                ax,
                fspec,
                clean_num(lo, 5),
                clean_num(hi, 5),
                n,
                clean_num(a0, 6),
            ]
        )
        k = m
    return grouped


def extract_opensees(
    model_dir: Path, nodes: dict[int, tuple[float, float, float]]
) -> dict[str, Any]:
    base_mats: list[list[Any]] = []
    fiber_secs: dict[int, list[tuple[float, float, float, int]]] = {}
    agg_secs: dict[int, tuple[int, int]] = {}
    elems: list[tuple[int, int, int, int, int]] = []
    cur_sec: int | None = None

    with (model_dir / "opensees.tcl").open() as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            toks = s.split()
            if toks[0] == "uniaxialMaterial" and toks[1] in ("Concrete01", "Steel02"):
                vals = [clean_num(float(x), 8) for x in toks[3:]]
                base_mats.append([toks[1], int(toks[2]), vals])
            elif toks[0] == "section" and toks[1] == "Fiber":
                cur_sec = int(toks[2])
                fiber_secs[cur_sec] = []
            elif toks[0] == "fiber" and cur_sec is not None:
                fiber_secs[cur_sec].append(
                    (float(toks[1]), float(toks[2]), float(toks[3]), int(toks[4]))
                )
            elif toks[0] == "}":
                cur_sec = None
            elif toks[0] == "section" and toks[1] == "Aggregator":
                agg_secs[int(toks[2])] = (int(toks[3]), int(toks[6]))
            elif toks[0] == "element" and toks[1] == "dispBeamColumn":
                elems.append(
                    (
                        int(toks[2]),
                        int(toks[3]),
                        int(toks[4]),
                        int(toks[6]),
                        int(toks[7]),
                    )
                )

    # 13 mega-column triplets: sections (3k+1, 3k+2, 3k+3) are identical
    triplets: list[list[Any]] = []
    for k in range(13):
        sid = 3 * k + 1
        fibs = fiber_secs[sid]
        c_fibs = [x for x in fibs if x[3] < 100]
        s_fibs = [x for x in fibs if x[3] == 101]
        cmat = c_fibs[0][3]
        h_val = clean_num(2.0 * (max(x[0] for x in c_fibs) + 0.05), 3)
        b_val = clean_num(2.0 * (max(x[1] for x in c_fibs) + 0.05), 3)
        bars = _compress_steel101_bars(s_fibs)
        triplets.append([cmat, h_val, b_val, bars])

    # Column subfamily -> contiguous (z1, z2, fsec) intervals
    by_sub: dict[str, dict[tuple[float, float], set[int]]] = collections.defaultdict(
        lambda: collections.defaultdict(set)
    )
    for eid, n1, n2, agg, _ in elems:
        fsec = agg_secs[agg][1]
        p1, p2 = nodes[n1], nodes[n2]
        mx, my = 0.5 * (p1[0] + p2[0]), 0.5 * (p1[1] + p2[1])
        z1, z2 = round(min(p1[2], p2[2]), 2), round(max(p1[2], p2[2]), 2)
        if abs(abs(mx) - abs(my)) < 1.0:
            sub = "C_POS" if mx * my > 0 else "C_NEG"
        else:
            sub = "F_X" if abs(mx) > abs(my) else "F_Y"
        by_sub[sub][(z1, z2)].add(fsec)

    sub_rules: dict[str, list[list[Any]]] = {}
    for sub, zmap in sorted(by_sub.items()):
        segs: list[list[Any]] = []
        for (z1, z2), fset in sorted(zmap.items()):
            assert len(fset) == 1, f"opensees conflict at {sub} {(z1, z2)}: {fset}"
            fsec = next(iter(fset))
            if not segs or segs[-1][2] != fsec:
                segs.append([clean_num(z1, 2), clean_num(z2, 2), fsec])
            else:
                segs[-1][1] = clean_num(z2, 2)
        sub_rules[sub] = segs

    return {
        "mats": base_mats,
        "ec_gpa": [40, 39, 38, 37, 36],
        "cover": 0.1,
        "rebar_ratio": 0.01,
        "rho": 2765.0,
        "g": 9.8,
        "nIP": 3,
        "triplets": triplets,
        "assign": sub_rules,
    }


def _truss_key(
    p1: tuple[float, float, float], p2: tuple[float, float, float]
) -> tuple[tuple[str, int | float, int | float], float]:
    dx, dy = abs(p2[0] - p1[0]), abs(p2[1] - p1[1])
    mx = 0.5 * (p1[0] + p2[0])
    my = 0.5 * (p1[1] + p2[1])
    mz = round(0.5 * (p1[2] + p2[2]), 2)
    if dx > 1e-3 and dy > 1e-3:
        return ("D", clean_num(abs(mx), 1), clean_num(abs(my), 1)), mz
    if dx > dy:
        return ("X", clean_num(abs(my), 1), clean_num(abs(mx), 2)), mz
    return ("Y", clean_num(abs(mx), 1), clean_num(abs(my), 2)), mz


def extract_truss(
    model_dir: Path, nodes: dict[int, tuple[float, float, float]]
) -> dict[str, Any]:
    mat_groups: dict[tuple[Any, ...], list[int]] = collections.defaultdict(list)
    ebc_counts: collections.Counter[tuple[float, ...]] = collections.Counter()
    ebc_elems: list[tuple[int, int, int, tuple[float, ...]]] = []
    truss_elems: list[tuple[int, int, int, float, int]] = []

    with (model_dir / "truss.tcl").open() as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            toks = s.split()
            if toks[0] == "uniaxialMaterial" and toks[1] == "Steel02":
                tag = int(toks[2])
                vals = tuple(clean_num(float(x), 8) for x in toks[3:])
                mat_groups[vals].append(tag)
            elif toks[0] == "element" and toks[1] == "elasticBeamColumn":
                props = tuple(float(x) for x in toks[5:11])
                ebc_counts[props] += 1
                ebc_elems.append((int(toks[2]), int(toks[3]), int(toks[4]), props))
            elif toks[0] == "element" and toks[1] == "truss":
                truss_elems.append(
                    (
                        int(toks[2]),
                        int(toks[3]),
                        int(toks[4]),
                        float(toks[5]),
                        int(toks[6]),
                    )
                )

    mats_out = []
    for vals, tags in mat_groups.items():
        mats_out.append([compress_indices(sorted(tags)), list(vals)])

    default_ebc = ebc_counts.most_common(1)[0][0]
    ebc_ex = []
    for _, n1, n2, props in ebc_elems:
        if props != default_ebc:
            p1, p2 = nodes[n1], nodes[n2]
            ebc_ex.append(
                [
                    clean_num(0.5 * (p1[0] + p2[0]), 2),
                    clean_num(0.5 * (p1[1] + p2[1]), 2),
                    clean_num(0.5 * (p1[2] + p2[2]), 2),
                    clean_num(props[0], 4),
                ]
            )

    by_k: dict[tuple[str, int | float, int | float], dict[float, set[float]]] = (
        collections.defaultdict(lambda: collections.defaultdict(set))
    )
    for _, n1, n2, area, _ in truss_elems:
        k, mz = _truss_key(nodes[n1], nodes[n2])
        by_k[k][mz].add(area)

    truss_rules = []
    for k, zmap in sorted(by_k.items()):
        segs: list[list[Any]] = []
        for mz, aset in sorted(zmap.items()):
            assert len(aset) == 1, f"truss area conflict at {k} z={mz}: {aset}"
            a = clean_num(next(iter(aset)), 6)
            if not segs or segs[-1][2] != a:
                segs.append([clean_num(mz, 2), clean_num(mz, 2), a])
            else:
                segs[-1][1] = clean_num(mz, 2)
        truss_rules.append([list(k), segs])

    return {
        "mats": mats_out,
        "ebc_default": [clean_num(x, 10) for x in default_ebc],
        "ebc_ex": ebc_ex,
        "ebc_load_per_len": -4.9,
        "truss_mat": 179,
        "truss_rho": 7800.0,
        "g": 9.8,
        "truss_rules": truss_rules,
    }


def extract_hbeam(
    model_dir: Path,
    nodes: dict[int, tuple[float, float, float]],
    x_max_z: dict[float, float],
    y_max_z: dict[float, float],
) -> dict[str, Any]:
    h_fibers: dict[int, list[tuple[float, float, float, float]]] = {}
    h_elems: list[tuple[int, int, int, int, int]] = []
    cur_sec: int | None = None

    with (model_dir / "Hbeam.tcl").open() as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            toks = s.split()
            if toks[0] == "section" and toks[1] == "Fiber":
                cur_sec = int(toks[2])
                h_fibers[cur_sec] = []
            elif toks[0] == "patch" and toks[1] == "rect" and cur_sec is not None:
                h_fibers[cur_sec].append(
                    (float(toks[5]), float(toks[6]), float(toks[7]), float(toks[8]))
                )
            elif toks[0] == "}":
                cur_sec = None
            elif toks[0] == "element" and toks[1] == "dispBeamColumn":
                h_elems.append(
                    (
                        int(toks[2]),
                        int(toks[3]),
                        int(toks[4]),
                        int(toks[6]) // 10000,
                        int(toks[7]),
                    )
                )

    # 148 parametric W-sections [d, bf, tf, tw]
    w_secs: list[list[int | float]] = []
    for idx in range(1, 149):
        ps = h_fibers[1200 + idx]
        if len(ps) == 3:
            bf = ps[0][2] - ps[0][0]
            tf = ps[0][3] - ps[0][1]
            d = -2.0 * ps[0][1]
            tw = ps[1][2] - ps[1][0]
        else:
            bf = 2.0 * (ps[0][2] - ps[0][0])
            tf = ps[0][3] - ps[0][1]
            d = -2.0 * ps[0][1]
            tw = ps[1][2] - ps[1][0]
        w_secs.append(
            [clean_num(d, 7), clean_num(bf, 7), clean_num(tf, 7), clean_num(tw, 7)]
        )

    # Partition elements into vertical, sloped cols, braces, horizontal
    v_map: dict[tuple[int | float, int | float], dict[float, set[int]]] = (
        collections.defaultdict(lambda: collections.defaultdict(set))
    )
    sc_map: dict[tuple[int | float, int | float], dict[float, set[int]]] = (
        collections.defaultdict(lambda: collections.defaultdict(set))
    )
    br_map: dict[float, dict[tuple[int | float, int | float], set[int]]] = (
        collections.defaultdict(lambda: collections.defaultdict(set))
    )
    h_by_z: dict[
        float,
        list[
            tuple[
                int,
                int,
                int,
                int,
                tuple[float, float, float],
                tuple[float, float, float],
            ]
        ],
    ] = collections.defaultdict(list)

    for eid, n1, n2, sidx, _ in h_elems:
        p1, p2 = nodes[n1], nodes[n2]
        dx = abs(p2[0] - p1[0])
        dy = abs(p2[1] - p1[1])
        dz = abs(p2[2] - p1[2])
        if dz < 1e-3:
            h_by_z[round(p1[2], 2)].append((eid, n1, n2, sidx, p1, p2))
        elif dx < 1e-3 and dy < 1e-3:
            ax, ay = abs(p1[0]), abs(p1[1])
            if ay > ax:
                ax, ay = ay, ax
            mz = round(0.5 * (p1[2] + p2[2]), 2)
            v_map[(clean_num(ax, 1), clean_num(ay, 1))][mz].add(sidx)
        elif math.hypot(dx, dy) < 0.5 * dz:
            z1, z2 = round(min(p1[2], p2[2]), 2), round(max(p1[2], p2[2]), 2)
            mz = round(0.5 * (z1 + z2), 2)
            xm = 0.5 * (x_max_z[z1] + x_max_z[z2])
            ym = 0.5 * (y_max_z[z1] + y_max_z[z2])
            ax = abs(0.5 * (p1[0] + p2[0])) / xm
            ay = abs(0.5 * (p1[1] + p2[1])) / ym
            if ay > ax:
                ax, ay = ay, ax
            sc_map[(clean_num(ax, 1), clean_num(ay, 1))][mz].add(sidx)
        else:
            mx = abs(0.5 * (p1[0] + p2[0]))
            my = abs(0.5 * (p1[1] + p2[1]))
            if my > mx:
                mx, my = my, mx
            mz = round(0.5 * (p1[2] + p2[2]), 2)
            br_map[mz][(clean_num(mx, 1), clean_num(my, 1))].add(sidx)

    def compress_z_map(
        zdict: dict[tuple[int | float, int | float], dict[float, set[int]]],
    ) -> list[list[Any]]:
        out = []
        for k, d in sorted(zdict.items()):
            flat: list[Any] = [k[0], k[1]]
            segs: list[list[Any]] = []
            for mz, sset in sorted(d.items()):
                assert len(sset) == 1, f"Hbeam conflict at {k} z={mz}: {sset}"
                s = next(iter(sset))
                if not segs or segs[-1][2] != s:
                    segs.append([clean_num(mz, 2), clean_num(mz, 2), s])
                else:
                    segs[-1][1] = clean_num(mz, 2)
            for a, b, s in segs:
                flat.extend([a, b, s])
            out.append(flat)
        return out

    v_rules = compress_z_map(v_map)
    sc_rules = compress_z_map(sc_map)

    br_rules: list[list[Any]] = []
    for mz, kmap in sorted(br_map.items()):
        entry: list[Any] = [clean_num(mz, 2)]
        for (mx, my), sset in sorted(kmap.items()):
            assert len(sset) == 1, f"Hbeam brace conflict at z={mz} {(mx, my)}: {sset}"
            entry.extend([mx, my, next(iter(sset))])
        br_rules.append(entry)

    # Horizontal floor beams: 8-fold octant + outer-bay normalization + Voronoi pruning
    all_h_zs = sorted(h_by_z)
    z_to_fidx = {z: i for i, z in enumerate(all_h_zs)}

    floor_beams: dict[int, list[tuple[float, float, int]]] = {}
    sym_ex: list[list[Any]] = []
    for z in all_h_zs:
        fidx = z_to_fidx[z]
        xz, yz = x_max_z[z], y_max_z[z]
        orbs: dict[tuple[float, float], list[Any]] = collections.defaultdict(list)
        for eid, n1, n2, sidx, p1, p2 in h_by_z[z]:
            c1, c2, c3, c4 = canonical_octant(p1, p2)
            q1 = norm_bay(c1, c2, xz, yz)
            q2 = norm_bay(c3, c4, xz, yz)
            mx, my = 0.5 * (q1[0] + q2[0]), 0.5 * (q1[1] + q2[1])
            orbs[(round(mx, 1), round(my, 1))].append((sidx, mx, my, p1, p2))
        fb: list[tuple[float, float, int]] = []
        for items in orbs.values():
            maj = collections.Counter(i[0] for i in items).most_common(1)[0][0]
            mx = sum(i[1] for i in items) / len(items)
            my = sum(i[2] for i in items) / len(items)
            fb.append((mx, my, maj))
            for sidx, _, _, p1, p2 in items:
                if sidx != maj:
                    amx = clean_num(0.5 * (p1[0] + p2[0]), 1)
                    amy = clean_num(0.5 * (p1[1] + p2[1]), 1)
                    sym_ex.append([fidx, amx, amy, sidx])
        floor_beams[fidx] = fb

    clusters: list[list[Any]] = []
    for fidx in range(len(all_h_zs)):
        for mx, my, sec in floor_beams[fidx]:
            best_i: int | None = None
            best_d = 1e18
            for idx, (sx, sy, cnt, fdict) in enumerate(clusters):
                if fidx in fdict:
                    continue
                d = math.hypot(sx / cnt - mx, sy / cnt - my)
                if d < best_d:
                    best_d = d
                    best_i = idx
            if best_i is not None and best_d < 0.4:
                clusters[best_i][0] += mx
                clusters[best_i][1] += my
                clusters[best_i][2] += 1
                clusters[best_i][3][fidx] = sec
            else:
                clusters.append([mx, my, 1, {fidx: sec}])

    raw_rules: list[list[Any]] = []
    for sx, sy, cnt, fdict in clusters:
        cx, cy = clean_num(sx / cnt, 1), clean_num(sy / cnt, 1)
        segs: list[list[int]] = []
        for fidx in sorted(fdict):
            sec = fdict[fidx]
            if not segs or segs[-1][2] != sec:
                segs.append([fidx, fidx, sec])
            else:
                segs[-1][1] = fidx
        raw_rules.append([cx, cy, segs])

    # Voronoi prototype pruning: drop any segment whose removal leaves every floor 100% exact
    fa: list[list[tuple[int, int, float, float, int]]] = [
        [] for _ in range(len(all_h_zs))
    ]
    for ri, (cx, cy, segs) in enumerate(raw_rules):
        for si, (a, b, sec) in enumerate(segs):
            for f in range(a, b + 1):
                fa[f].append((ri, si, float(cx), float(cy), sec))

    pruned_h: list[list[Any]] = []
    for ri, (cx, cy, segs) in enumerate(raw_rules):
        keep: list[list[int]] = []
        for si, (a, b, sec) in enumerate(segs):
            ok = True
            for f in range(a, b + 1):
                rem = [
                    (x, y, s)
                    for r2, s2, x, y, s in fa[f]
                    if not (r2 == ri and s2 == si)
                ]
                if not rem:
                    ok = False
                    break
                for mx, my, tsec in floor_beams[f]:
                    best = min(rem, key=lambda r: (r[0] - mx) ** 2 + (r[1] - my) ** 2)
                    if best[2] != tsec:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                for f in range(a, b + 1):
                    fa[f] = [x for x in fa[f] if not (x[0] == ri and x[1] == si)]
            else:
                keep.append([a, b, sec])
        if keep:
            flat: list[Any] = [cx, cy]
            for a, b, sec in keep:
                flat.extend([a, b, sec])
            pruned_h.append(flat)

    return {
        "mat": 176,
        "E": 200e9,
        "nu": 0.3,
        "rho": 7800.0,
        "g": 9.8,
        "nIP": 3,
        "split_from": 145,
        "secs": w_secs,
        "v": v_rules,
        "sc": sc_rules,
        "br": br_rules,
        "h": pruned_h,
        "h_ex": sym_ex,
    }


def shell_family(
    pts: list[tuple[float, float, float]],
) -> str:
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


def extract_layershell(
    model_dir: Path, nodes: dict[int, tuple[float, float, float]]
) -> dict[str, Any]:
    mats: list[list[Any]] = []
    prof_to_tags: dict[tuple[Any, ...], list[int]] = collections.defaultdict(list)
    fam_z: dict[str, dict[float, set[int]]] = collections.defaultdict(
        lambda: collections.defaultdict(set)
    )

    with (model_dir / "layershell.tcl").open() as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            toks = s.split()
            if toks[0] in ("nDMaterial", "uniaxialMaterial"):
                vals = [clean_num(float(x), 10) for x in toks[3:]]
                mats.append([toks[0], toks[1], int(toks[2]), vals])
            elif toks[0] == "section" and toks[1] == "LayeredShell":
                stag = int(toks[2])
                nlay = int(toks[3])
                lays = [
                    (int(toks[4 + 2 * k]), float(toks[5 + 2 * k])) for k in range(nlay)
                ]
                cmat = lays[0][0]
                t_cov = clean_num(lays[0][1], 6)
                t_r90 = clean_num(lays[1][1], 6)
                t_r0 = clean_num(lays[2][1], 6)
                t_core = clean_num(lays[3][1], 6)
                t_plate = clean_num(lays[8][1], 6) if nlay == 18 else 0
                prof = (
                    (cmat, t_cov, t_r90, t_r0, t_core)
                    if t_plate == 0
                    else (cmat, t_cov, t_r90, t_r0, t_core, t_plate)
                )
                prof_to_tags[prof].append(stag)
            elif toks[0] == "element" and toks[1] == "ShellMITC4":
                ns = [int(toks[3]), int(toks[4]), int(toks[5]), int(toks[6])]
                stag = int(toks[7])
                pts = [nodes[n] for n in ns]
                fam = shell_family(pts)
                cz = round(sum(p[2] for p in pts) / 4.0, 2)
                fam_z[fam][cz].add(stag)

    secs: list[list[Any]] = [
        [list(prof), sorted(tags)] for prof, tags in prof_to_tags.items()
    ]

    assign: dict[str, list[Any]] = {}
    for fam, zmap in sorted(fam_z.items()):
        segs: list[list[Any]] = []
        for cz, sset in sorted(zmap.items()):
            assert len(sset) == 1, f"layershell conflict at {fam} z={cz}: {sset}"
            st = next(iter(sset))
            if not segs or segs[-1][2] != st:
                segs.append([clean_num(cz, 2), clean_num(cz, 2), st])
            else:
                segs[-1][1] = clean_num(cz, 2)
        flat: list[Any] = []
        for a, b, st in segs:
            flat.extend([a, b, st])
        assign[fam] = flat

    return {
        "mats": mats,
        "secs": secs,
        "rho_conc": 2500.0,
        "rho_rebar": 7800.0,
        "rho_plate_layer": 3900.0,
        "g": 9.8,
        "assign": assign,
    }


def poly_area_2d(pts: list[tuple[float, float]]) -> float:
    s = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return 0.5 * abs(s)


def is_active_core_bay(z: float, pts: list[tuple[float, float, float]]) -> bool:
    xs = sorted({round(p[0], 1) for p in pts})
    ys = sorted({round(p[1], 1) for p in pts})
    if not (
        len(xs) == 2
        and len(ys) == 2
        and abs(xs[1] - xs[0] - 10.0) < 0.05
        and abs(ys[1] - ys[0] - 10.0) < 0.05
    ):
        return False
    if xs[0] not in (-15.0, -5.0, 5.0) or ys[0] not in (-15.0, -5.0, 5.0):
        return False
    return not (z >= 270.0 and xs[0] != -5.0 and ys[0] != -5.0)


def map_pt(
    px: float,
    py: float,
    xt: float,
    yt: float,
    xz: float,
    yz: float,
    zn_list: list[tuple[int, float, float]],
) -> tuple[int, float]:
    best_n = -1
    best_d = 1e18
    for n, nx, ny in zn_list:
        d = math.hypot(nx - px, ny - py)
        if d < best_d:
            best_d = d
            best_n = n
    if best_d < 0.03:
        return best_n, best_d
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
    return best_n, best_d


def _sym_ops(code: int) -> tuple[list[tuple[Any, int]], int]:
    if code == 0:
        return [(lambda x, y: (x, y), 0)], 0
    if code == 1:
        return C4_OPS, 0
    if code == 2:
        return C2_OPS, 0
    if code in (3, 4, 5):
        return D4_OPS, code - 3
    if code in (6, 7, 8):
        return V4_OPS, code - 6
    raise ValueError(f"Unknown symmetry code {code}")


def extract_rigiddigram(
    model_dir: Path,
    nodes: dict[int, tuple[float, float, float]],
    nodes_by_z: dict[float, list[tuple[int, float, float]]],
    x_max_z: dict[float, float],
    y_max_z: dict[float, float],
) -> dict[str, Any]:
    rd_loads: list[tuple[int, float]] = []
    with (model_dir / "rigiddigram.tcl").open() as f:
        for line in f:
            s = line.strip()
            if s.startswith("load"):
                toks = s.split()
                rd_loads.append((int(toks[1]), float(toks[4])))

    panels_by_z: dict[float, list[tuple[list[int], float]]] = collections.defaultdict(
        list
    )
    for i in range(len(rd_loads) // 4):
        ch = rd_loads[4 * i : 4 * i + 4]
        ns = [x[0] for x in ch]
        z = round(nodes[ns[0]][2], 2)
        panels_by_z[z].append((ns, ch[0][1]))

    all_zs = sorted(panels_by_z)
    z_to_fidx = {z: i for i, z in enumerate(all_zs)}

    # 1. Exact canonical pressures per floor (core vs non-core)
    canon_q = (5822.5, 7572.5, 7940.0, 9410.0, 9787.5, 9910.0, 13437.5, 13560.0)
    q_segs: list[list[Any]] = []
    for z in all_zs:
        cqs: set[float] = set()
        nqs: set[float] = set()
        for ns, fz in panels_by_z[z]:
            vs = ns[:3] if ns[0] == ns[3] else ns
            area = poly_area_2d([(nodes[n][0], nodes[n][1]) for n in vs])
            q_raw = -4.0 * fz / area
            qc = min(canon_q, key=lambda c: abs(c - q_raw))
            if is_active_core_bay(z, [nodes[n] for n in ns]):
                cqs.add(qc)
            else:
                nqs.add(qc)
        assert len(cqs) == 1 and len(nqs) == 1, (
            f"Multiple pressures at z={z}: {cqs}, {nqs}"
        )
        cq = clean_num(next(iter(cqs)), 2)
        nq = clean_num(next(iter(nqs)), 2)
        if not q_segs or (q_segs[-1][2], q_segs[-1][3]) != (cq, nq):
            q_segs.append([clean_num(z, 2), clean_num(z, 2), cq, nq])
        else:
            q_segs[-1][1] = clean_num(z, 2)

    # 2. Match non-core panel meshes into floor templates
    templates: list[
        tuple[float, float, float, list[tuple[list[tuple[float, float]], bool]]]
    ] = []
    floor_to_tpl: dict[float, int] = {}
    for z in all_zs:
        noncore = [
            ns
            for ns, _ in panels_by_z[z]
            if not is_active_core_bay(z, [nodes[n] for n in ns])
        ]
        act_map = {frozenset(ns): (ns[0] if ns[0] == ns[3] else None) for ns in noncore}
        zn = nodes_by_z[z]
        xz, yz = x_max_z[z], y_max_z[z]
        matched: int | None = None
        for tidx, (tz, xt, yt, tpanels) in enumerate(templates):
            if len(tpanels) != len(noncore):
                continue
            ok = True
            seen: set[frozenset[int]] = set()
            for tpts, _ in tpanels:
                m_ns = [map_pt(px, py, xt, yt, xz, yz, zn)[0] for px, py in tpts]
                gk = frozenset(m_ns)
                if len(gk) != len(tpts) or gk not in act_map or gk in seen:
                    ok = False
                    break
                seen.add(gk)
            if ok and len(seen) == len(act_map):
                matched = tidx
                break
        if matched is None:
            tpanels = []
            for ns in noncore:
                is_tri = ns[0] == ns[3]
                vs = ns[:3] if is_tri else ns
                tpanels.append(
                    (
                        [
                            (clean_num(nodes[n][0], 2), clean_num(nodes[n][1], 2))
                            for n in vs
                        ],
                        is_tri,
                    )
                )
            matched = len(templates)
            templates.append(
                (z, float(clean_num(xz, 2)), float(clean_num(yz, 2)), tpanels)
            )
        floor_to_tpl[z] = matched

    # 3. Compress each template via D4/V4/C4/C2 orbits + global deduped point table
    g_pts: list[int | float] = []
    g_pt_idx: dict[tuple[int | float, int | float], int] = {}

    def get_pt(x: float, y: float) -> int:
        k = (clean_num(x, 2), clean_num(y, 2))
        if k not in g_pt_idx:
            g_pt_idx[k] = len(g_pts) // 2
            g_pts.extend([k[0], k[1]])
        return g_pt_idx[k]

    sym_defs = [
        (D4_OPS, (3, 4, 5)),
        (V4_OPS, (6, 7, 8)),
        (C4_OPS, (1,)),
        (C2_OPS, (2,)),
    ]

    g_tpls: list[list[Any]] = []
    v0_ovs: list[list[int]] = []

    def _octant_rank(
        item: tuple[list[tuple[int | float, int | float]], bool],
    ) -> tuple[int, int, int, float, float]:
        pts_i = item[0]
        cx = sum(p[0] for p in pts_i) / len(pts_i)
        cy = sum(p[1] for p in pts_i) / len(pts_i)
        # Prefer x >= 0, y >= 0, and cx >= cy (fundamental positive octant)
        neg_x = 0 if all(p[0] >= -1e-6 for p in pts_i) else (1 if cx >= 0 else 2)
        neg_y = 0 if all(p[1] >= -1e-6 for p in pts_i) else (1 if cy >= 0 else 2)
        above_diag = 0 if cx >= cy - 1e-6 else 1
        return (neg_x, neg_y, above_diag, abs(cx), abs(cy))

    for tidx, (tz, xt, yt, tpanels) in enumerate(templates):
        zn = nodes_by_z[tz]
        act_map = {}
        for tpts, is_tri in tpanels:
            ns = [map_pt(px, py, xt, yt, xt, yt, zn)[0] for px, py in tpts]
            act_map[frozenset(ns)] = ns[0] if is_tri else None

        # Collect actual non-core panels for all floors assigned to this template
        fidxs = [z_to_fidx[z] for z in all_zs if floor_to_tpl[z] == tidx]
        xt_c, yt_c = clean_num(xt, 2), clean_num(yt, 2)
        floor_acts: list[
            tuple[
                int,
                float,
                list[tuple[int, float, float]],
                float,
                float,
                dict[frozenset[int], int | None],
            ]
        ] = []
        for fidx in fidxs:
            z_f = all_zs[fidx]
            noncore_f = [
                ns
                for ns, _ in panels_by_z[z_f]
                if not is_active_core_bay(z_f, [nodes[n] for n in ns])
            ]
            act_f = {
                frozenset(ns): (ns[0] if ns[0] == ns[3] else None) for ns in noncore_f
            }
            floor_acts.append(
                (fidx, z_f, nodes_by_z[z_f], x_max_z[z_f], y_max_z[z_f], act_f)
            )

        covered: set[frozenset[int]] = set()
        orbits: list[list[int]] = []
        for tpts, is_tri in sorted(tpanels, key=_octant_rank):
            ns = [map_pt(px, py, xt, yt, xt, yt, zn)[0] for px, py in tpts]
            gk = frozenset(ns)
            if gk in covered:
                continue
            rpts = [(clean_num(px, 2), clean_num(py, 2)) for px, py in tpts]
            best_code = 0
            best_gen = {gk}
            for ops, codes in sym_defs:
                found = False
                for r_i, code in enumerate(codes):
                    gen: set[frozenset[int]] = set()
                    ok = True
                    # Verify on tz AND on every floor assigned to this template!
                    for fi_idx, (fidx, z_f, zn_f, xz_f, yz_f, act_f) in enumerate(
                        floor_acts
                    ):
                        for op, is_ref in ops:
                            m_ns = []
                            for px, py in rpts:
                                ox, oy = op(px, py)
                                sn, sd = map_pt(
                                    ox, oy, float(xt_c), float(yt_c), xz_f, yz_f, zn_f
                                )
                                if fi_idx == 0 and sd > 0.05:
                                    ok = False
                                    break
                                m_ns.append(sn)
                            if not ok:
                                break
                            mgk = frozenset(m_ns)
                            if len(mgk) != len(rpts) or mgk not in act_f:
                                ok = False
                                break
                            if fi_idx == 0:
                                if is_tri:
                                    exp_v0 = m_ns[r_i] if is_ref else m_ns[0]
                                    if act_f[mgk] != exp_v0:
                                        ok = False
                                        break
                                gen.add(mgk)
                        if not ok:
                            break
                    if ok and len(gen - covered) >= 2:
                        best_code = code
                        best_gen = gen
                        found = True
                        break
                if found:
                    break
            orbits.append([best_code] + [get_pt(px, py) for px, py in rpts])
            covered.update(best_gen)

        # Compute exact tri_v0 overrides for every floor assigned to this template
        for fidx, z_f, zn_f, xz_f, yz_f, act_f in floor_acts:
            seen_f: set[frozenset[int]] = set()
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
                        sn, _ = map_pt(
                            ox, oy, float(xt_c), float(yt_c), xz_f, yz_f, zn_f
                        )
                        m_ns.append(sn)
                    gk = frozenset(m_ns)
                    if gk in seen_f:
                        continue
                    seen_f.add(gk)
                    if is_tri:
                        exp_v0 = m_ns[ref_idx] if is_ref else m_ns[0]
                        act_v0 = act_f[gk]
                        if act_v0 != exp_v0:
                            v0_ovs.append([fidx, oidx, opidx, m_ns.index(act_v0)])

        g_tpls.append([xt_c, yt_c, compress_indices(fidxs), orbits])

    return {
        "core_cutoff_z": 270.0,
        "q": q_segs,
        "p": g_pts,
        "t": g_tpls,
        "v0_ov": v0_ovs,
    }


def extract_dynamiccpu() -> dict[str, Any]:
    return {
        "ndm": 3,
        "ndf": 6,
        "story_xy": [-5.0, 5.0],
        "gravity": {
            "constraints": "Transformation",
            "numberer": "RCM",
            "system": "SparseSYM",
            "test": ["NormDispIncr", 0.05, 10, 1],
            "algorithm": "KrylovNewton",
            "integrator": ["LoadControl", 0.1],
            "steps": 10,
        },
        "damping": {
            "xDamp": 0.05,
            "MpropSwitch": 1.0,
            "KcurrSwitch": 0.0,
            "KcommSwitch": 1.0,
            "KinitSwitch": 0.0,
            "nEigenI": 1,
            "nEigenJ": 10,
            "omegaI": 0.669862891,
            "omegaJ": 6.804704678,
        },
        "gm": {
            "loadTag": 1001,
            "file": "GMX.txt",
            "direction": 1,
            "factor": 2.2,
            "dt": 0.01,
        },
        "transient": {
            "constraints": "Transformation",
            "numberer": "RCM",
            "system": "SparseSYM",
            "test": ["NormDispIncr", 0.05, 10, 2],
            "algorithm": ["NewtonLineSearch", 0.75],
            "integrator": ["Newmark", 0.55, 0.2765625],
            "Nsteps": 2000,
            "DtAnalysis": 0.02,
            "TmaxAnalysis": 40,
        },
    }


def extract_all(model_dir: Path) -> dict[str, Any]:
    nodes = parse_nodes(model_dir)
    nodes_by_z: dict[float, list[tuple[int, float, float]]] = collections.defaultdict(
        list
    )
    for n, (x, y, z) in nodes.items():
        nodes_by_z[round(z, 2)].append((n, x, y))
    x_max_z = {z: max(abs(x) for _, x, _ in lst) for z, lst in nodes_by_z.items()}
    y_max_z = {z: max(abs(y) for _, _, y in lst) for z, lst in nodes_by_z.items()}

    return {
        "opensees": extract_opensees(model_dir, nodes),
        "truss": extract_truss(model_dir, nodes),
        "hbeam": extract_hbeam(model_dir, nodes, x_max_z, y_max_z),
        "layershell": extract_layershell(model_dir, nodes),
        "rigiddigram": extract_rigiddigram(
            model_dir, nodes, nodes_by_z, x_max_z, y_max_z
        ),
        "dynamiccpu": extract_dynamiccpu(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    args = ap.parse_args()

    props = extract_all(args.model_dir)
    blob = json.dumps(props, separators=(",", ":"))
    args.out.write_text(blob + "\n")

    print(f"Wrote {args.out} ({len(blob):,} bytes = {len(blob) / 1024:.2f} KB)")
    for k, v in props.items():
        sb = len(json.dumps(v, separators=(",", ":")))
        print(f"  {k:12s}: {sb:6,d} B ({sb / 1024:5.2f} KB)")


if __name__ == "__main__":
    main()
