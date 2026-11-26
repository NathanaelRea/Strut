"""Compare generated Model1 properties against the real OpenSees Model1 files.

Stdlib-only. Reports per-file and per-family exact-match rates (at 1e-6 and
1e-4 relative tolerance), max relative error, unexplained lines, and
compression ratios (props.json bytes vs original non-geometric line bytes).
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.megatall_recon.properties import (
    DEFAULT_MODEL_DIR,
    DEFAULT_PROPS_PATH,
    generate_all,
    load_props,
    load_real_geometry,
)


@dataclass
class FamilyResult:
    file_name: str
    family: str
    count: int
    match_1e6: int
    match_1e4: int
    max_rel_err: float
    unexplained: int
    note: str = ""


def rel_err(actual: float, pred: float) -> float:
    denom = max(abs(actual), abs(pred), 1e-12)
    return abs(actual - pred) / denom


def _eval_vec_errors(errs: list[float]) -> tuple[int, int, float]:
    m6 = sum(1 for e in errs if e <= 1e-6)
    m4 = sum(1 for e in errs if e <= 1e-4)
    mx = max(errs) if errs else 0.0
    return m6, m4, mx


def compare_opensees(model_dir: Path, gen_os: dict[str, Any]) -> list[FamilyResult]:
    act_mats: dict[tuple[str, int], list[float]] = {}
    act_fibers: dict[int, list[tuple[float, float, float, int]]] = {}
    act_elasts: dict[int, float] = {}
    act_aggs: dict[int, tuple[int, int]] = {}
    act_elems: dict[int, tuple[int, int, float]] = {}  # eid -> (nIP, agg, mass)
    act_loads: list[tuple[int, float]] = []
    cur_sec: int | None = None
    unexp_lines = 0

    with (model_dir / "opensees.tcl").open() as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#") or s == "}":
                continue
            toks = s.split()
            if toks[0] in ("node", "geomTransf", "pattern"):
                continue
            if toks[0] == "uniaxialMaterial" and toks[1] in ("Concrete01", "Steel02"):
                act_mats[(toks[1], int(toks[2]))] = [float(x) for x in toks[3:]]
            elif toks[0] == "uniaxialMaterial" and toks[1] == "Elastic":
                act_elasts[int(toks[2])] = float(toks[3])
            elif toks[0] == "section" and toks[1] == "Fiber":
                cur_sec = int(toks[2])
                act_fibers[cur_sec] = []
            elif toks[0] == "fiber" and cur_sec is not None:
                act_fibers[cur_sec].append(
                    (float(toks[1]), float(toks[2]), float(toks[3]), int(toks[4]))
                )
            elif toks[0] == "section" and toks[1] == "Aggregator":
                act_aggs[int(toks[2])] = (int(toks[3]), int(toks[6]))
            elif toks[0] == "element" and toks[1] == "dispBeamColumn":
                act_elems[int(toks[2])] = (int(toks[5]), int(toks[6]), float(toks[9]))
            elif toks[0] == "load":
                act_loads.append((int(toks[1]), float(toks[4])))
            else:
                unexp_lines += 1

    res: list[FamilyResult] = []

    # 1. Base materials
    gen_mats = {(m, t): [float(x) for x in vs] for m, t, vs in gen_os["base_mats"]}
    errs = []
    miss = len(set(act_mats) ^ set(gen_mats))
    for k, av in act_mats.items():
        if k in gen_mats:
            errs.append(max(rel_err(a, b) for a, b in zip(av, gen_mats[k])))
    m6, m4, mx = _eval_vec_errors(errs)
    res.append(
        FamilyResult("opensees.tcl", "base materials", len(act_mats), m6, m4, mx, miss)
    )

    # 2. Fiber lines (5,070 across 39 Fiber sections)
    # Compare sorted sets per section (section Fiber is unordered in OpenSees)
    # Note: original opensees.tcl prints fiber y, z, A to %.5g (5 sig figs)
    fib_errs: list[float] = []
    fib_miss = 0
    for sid, afibs in sorted(act_fibers.items()):
        gfibs = gen_os["fiber_secs"].get(sid, [])
        if len(afibs) != len(gfibs):
            fib_miss += abs(len(afibs) - len(gfibs))
            continue
        asrt = sorted(afibs, key=lambda x: (x[3], round(x[0], 3), round(x[1], 3)))
        gsrt = sorted(gfibs, key=lambda x: (x[3], round(x[0], 3), round(x[1], 3)))
        for (ay, az, aa, am), (gy, gz, ga, gm) in zip(asrt, gsrt):
            if am != gm:
                fib_miss += 1
                continue
            # Original opensees.tcl prints fiber (y, z) to %.5g and area to %.5f
            gy5 = float(f"{gy:.5g}")
            gz5 = float(f"{gz:.5g}")
            ga5 = float(f"{ga:.5f}")
            e_pos = max(abs(ay - gy5), abs(az - gz5)) / max(1.0, abs(ay), abs(az))
            e_area = rel_err(aa, ga5)
            fib_errs.append(max(e_pos, e_area))
    m6, m4, mx = _eval_vec_errors(fib_errs)
    res.append(
        FamilyResult(
            "opensees.tcl",
            "fiber lines (%.5g/%.5f)",
            sum(len(v) for v in act_fibers.values()),
            m6,
            m4,
            mx,
            fib_miss,
            note="orig printed %.5g / %.5f",
        )
    )

    # 3. Torsional Elastic materials (1,240)
    errs = []
    miss = len(set(act_elasts) ^ set(gen_os["elastics"]))
    for k, av in act_elasts.items():
        if k in gen_os["elastics"]:
            errs.append(rel_err(av, gen_os["elastics"][k]))
    m6, m4, mx = _eval_vec_errors(errs)
    res.append(
        FamilyResult(
            "opensees.tcl", "Elastic torsion", len(act_elasts), m6, m4, mx, miss
        )
    )

    # 4. Aggregator sections (1,240)
    ag_ok = sum(1 for k, v in act_aggs.items() if gen_os["aggregators"].get(k) == v)
    ag_miss = len(act_aggs) - ag_ok
    res.append(
        FamilyResult(
            "opensees.tcl",
            "Aggregator sec assign",
            len(act_aggs),
            ag_ok,
            ag_ok,
            0.0 if ag_miss == 0 else 1.0,
            ag_miss,
        )
    )

    # 5. Element -mass & nIP/agg (1,240)
    gen_em = {eid: (nip, agg, m) for eid, _, _, nip, agg, _, m in gen_os["elems"]}
    errs = []
    miss = 0
    for eid, (anip, aagg, am) in act_elems.items():
        g = gen_em.get(eid)
        if g is None or g[0] != anip or g[1] != aagg:
            miss += 1
        else:
            errs.append(rel_err(am, g[2]))
    m6, m4, mx = _eval_vec_errors(errs)
    res.append(
        FamilyResult(
            "opensees.tcl", "dispBeamColumn -mass", len(act_elems), m6, m4, mx, miss
        )
    )

    # 6. Loads (2,480)
    errs = []
    miss = abs(len(act_loads) - len(gen_os["loads"]))
    for (an, afz), (gn, gfz) in zip(act_loads, gen_os["loads"]):
        if an != gn:
            miss += 1
        else:
            errs.append(rel_err(afz, gfz))
    m6, m4, mx = _eval_vec_errors(errs)
    res.append(
        FamilyResult(
            "opensees.tcl",
            "loads (pattern 1)",
            len(act_loads),
            m6,
            m4,
            mx,
            miss + unexp_lines,
        )
    )
    return res


def compare_truss(model_dir: Path, gen_tr: dict[str, Any]) -> list[FamilyResult]:
    act_mats: dict[int, list[float]] = {}
    act_ebcs: dict[int, list[float]] = {}
    act_trusses: dict[int, tuple[float, int]] = {}
    unexp_t = 0
    with (model_dir / "truss.tcl").open() as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            toks = s.split()
            if toks[0] == "uniaxialMaterial" and toks[1] == "Steel02":
                act_mats[int(toks[2])] = [float(x) for x in toks[3:]]
            elif toks[0] == "element" and toks[1] == "elasticBeamColumn":
                act_ebcs[int(toks[2])] = [float(x) for x in toks[5:11]]
            elif toks[0] == "element" and toks[1] == "truss":
                act_trusses[int(toks[2])] = (float(toks[5]), int(toks[6]))
            else:
                unexp_t += 1

    act_loads: list[tuple[int, float]] = []
    unexp_l = 0
    with (model_dir / "trussload.tcl").open() as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith(("#", "pattern")) or s == "}":
                continue
            toks = s.split()
            if toks[0] == "load":
                act_loads.append((int(toks[1]), float(toks[4])))
            else:
                unexp_l += 1

    res: list[FamilyResult] = []
    errs = []
    miss = len(set(act_mats) ^ set(gen_tr["mats"]))
    for k, av in act_mats.items():
        if k in gen_tr["mats"]:
            errs.append(max(rel_err(a, b) for a, b in zip(av, gen_tr["mats"][k])))
    m6, m4, mx = _eval_vec_errors(errs)
    res.append(
        FamilyResult("truss.tcl", "Steel02 materials", len(act_mats), m6, m4, mx, miss)
    )

    gen_ebc_map = {eid: props for eid, _, _, props, _ in gen_tr["ebcs"]}
    errs = []
    miss = len(set(act_ebcs) ^ set(gen_ebc_map))
    for k, av in act_ebcs.items():
        if k in gen_ebc_map:
            errs.append(max(rel_err(a, b) for a, b in zip(av, gen_ebc_map[k])))
    m6, m4, mx = _eval_vec_errors(errs)
    res.append(
        FamilyResult("truss.tcl", "elasticBeamColumn", len(act_ebcs), m6, m4, mx, miss)
    )

    gen_tr_map = {eid: (a, m) for eid, _, _, a, m in gen_tr["trusses"]}
    errs = []
    miss = len(set(act_trusses) ^ set(gen_tr_map))
    for k, (aa, am) in act_trusses.items():
        g = gen_tr_map.get(k)
        if g is None or g[1] != am:
            miss += 1
        else:
            errs.append(rel_err(aa, g[0]))
    m6, m4, mx = _eval_vec_errors(errs)
    res.append(
        FamilyResult(
            "truss.tcl", "truss area+mat", len(act_trusses), m6, m4, mx, miss + unexp_t
        )
    )

    # trussload.tcl (EBC loads + truss loads)
    n_ebc = 2 * len(act_ebcs)
    errs_e: list[float] = []
    miss_e = 0
    for (an, afz), (gn, gfz) in zip(act_loads[:n_ebc], gen_tr["loads"][:n_ebc]):
        if an != gn:
            miss_e += 1
        else:
            errs_e.append(rel_err(afz, gfz))
    m6, m4, mx = _eval_vec_errors(errs_e)
    res.append(
        FamilyResult(
            "trussload.tcl",
            "rigid-link loads",
            n_ebc,
            m6,
            m4,
            mx,
            miss_e,
            note="4-dec coord rounding",
        )
    )

    errs_t: list[float] = []
    miss_t = abs(len(act_loads) - len(gen_tr["loads"]))
    for (an, afz), (gn, gfz) in zip(act_loads[n_ebc:], gen_tr["loads"][n_ebc:]):
        if an != gn:
            miss_t += 1
        else:
            errs_t.append(rel_err(afz, gfz))
    m6, m4, mx = _eval_vec_errors(errs_t)
    res.append(
        FamilyResult(
            "trussload.tcl",
            "truss loads",
            len(act_loads) - n_ebc,
            m6,
            m4,
            mx,
            miss_t + unexp_l,
            note="4-dec coord rounding",
        )
    )
    return res


def compare_hbeam(model_dir: Path, gen_hb: dict[str, Any]) -> list[FamilyResult]:
    act_patches: dict[int, list[tuple[int, int, int, float, float, float, float]]] = {}
    act_elasts: dict[int, float] = {}
    act_aggs: dict[int, tuple[int, int]] = {}
    act_elems: dict[int, tuple[int, int, float]] = {}
    act_loads: dict[int, float] = {}
    cur_sec: int | None = None
    unexp = 0

    with (model_dir / "Hbeam.tcl").open() as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith(("#", "pattern")) or s == "}":
                continue
            toks = s.split()
            if toks[0] == "section" and toks[1] == "Fiber":
                cur_sec = int(toks[2])
                act_patches[cur_sec] = []
            elif toks[0] == "patch" and toks[1] == "rect" and cur_sec is not None:
                act_patches[cur_sec].append(
                    (
                        int(toks[2]),
                        int(toks[3]),
                        int(toks[4]),
                        float(toks[5]),
                        float(toks[6]),
                        float(toks[7]),
                        float(toks[8]),
                    )
                )
            elif toks[0] == "uniaxialMaterial" and toks[1] == "Elastic":
                act_elasts[int(toks[2])] = float(toks[3])
            elif toks[0] == "section" and toks[1] == "Aggregator":
                act_aggs[int(toks[2])] = (int(toks[3]), int(toks[6]))
            elif toks[0] == "element" and toks[1] == "dispBeamColumn":
                act_elems[int(toks[2])] = (int(toks[5]), int(toks[6]), float(toks[9]))
            elif toks[0] == "load":
                act_loads[int(toks[1])] = float(toks[4])
            else:
                unexp += 1

    res: list[FamilyResult] = []
    # 1. patch rect (452 patches across 148 sections)
    errs: list[float] = []
    miss = 0
    for fsec, aps in sorted(act_patches.items()):
        gps = gen_hb["fiber_secs"].get(fsec, [])
        if len(aps) != len(gps):
            miss += abs(len(aps) - len(gps))
            continue
        for ap, gp in zip(aps, gps):
            if ap[:3] != gp[:3]:
                miss += 1
            else:
                errs.append(max(rel_err(a, b) for a, b in zip(ap[3:], gp[3:])))
    m6, m4, mx = _eval_vec_errors(errs)
    res.append(
        FamilyResult(
            "Hbeam.tcl",
            "patch rect (148 secs)",
            sum(len(v) for v in act_patches.values()),
            m6,
            m4,
            mx,
            miss,
        )
    )

    # 2. Elastic GJ (148)
    errs = []
    miss = len(set(act_elasts) ^ set(gen_hb["elastics"]))
    for k, av in act_elasts.items():
        if k in gen_hb["elastics"]:
            errs.append(rel_err(av, gen_hb["elastics"][k]))
    m6, m4, mx = _eval_vec_errors(errs)
    res.append(
        FamilyResult("Hbeam.tcl", "Elastic GJ", len(act_elasts), m6, m4, mx, miss)
    )

    # 3. Aggregator (148)
    ok = sum(1 for k, v in act_aggs.items() if gen_hb["aggregators"].get(k) == v)
    res.append(
        FamilyResult(
            "Hbeam.tcl",
            "Aggregator defs",
            len(act_aggs),
            ok,
            ok,
            0.0 if ok == len(act_aggs) else 1.0,
            len(act_aggs) - ok,
        )
    )

    # 4. Element section assignment (28,310) & -mass (28,310)
    gen_em = {eid: (nip, agg, m) for eid, _, _, nip, agg, _, m in gen_hb["elems"]}
    sec_ok = 0
    mass_errs: list[float] = []
    sec_miss = 0
    for eid, (anip, aagg, am) in act_elems.items():
        g = gen_em.get(eid)
        if g is None or g[0] != anip or g[1] != aagg:
            sec_miss += 1
        else:
            sec_ok += 1
            mass_errs.append(rel_err(am, g[2]))
    res.append(
        FamilyResult(
            "Hbeam.tcl",
            "dispBeamColumn sec",
            len(act_elems),
            sec_ok,
            sec_ok,
            0.0 if sec_miss == 0 else 1.0,
            sec_miss,
        )
    )
    m6, m4, mx = _eval_vec_errors(mass_errs)
    res.append(
        FamilyResult(
            "Hbeam.tcl",
            "dispBeamColumn -mass",
            len(act_elems),
            m6,
            m4,
            mx,
            sec_miss,
            note="orig printed %.4f",
        )
    )

    # 5. Loads (18,612)
    gen_lm = dict(gen_hb["loads"])
    errs = []
    miss = len(set(act_loads) ^ set(gen_lm))
    for n, afz in act_loads.items():
        if n in gen_lm:
            errs.append(rel_err(afz, gen_lm[n]))
    m6, m4, mx = _eval_vec_errors(errs)
    res.append(
        FamilyResult(
            "Hbeam.tcl",
            "loads (pattern 10)",
            len(act_loads),
            m6,
            m4,
            mx,
            miss + unexp,
            note="4-dec coord rounding",
        )
    )
    return res


def compare_layershell(model_dir: Path, gen_ls: dict[str, Any]) -> list[FamilyResult]:
    act_mats: dict[tuple[str, str, int], list[float]] = {}
    act_secs: dict[int, list[tuple[int, float]]] = {}
    act_elems: dict[int, int] = {}
    act_masses: dict[int, float] = {}
    act_loads: list[tuple[int, float]] = []
    unexp = 0

    with (model_dir / "layershell.tcl").open() as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith(("#", "pattern")) or s == "}":
                continue
            toks = s.split()
            if toks[0] in ("nDMaterial", "uniaxialMaterial"):
                act_mats[(toks[0], toks[1], int(toks[2]))] = [
                    float(x) for x in toks[3:]
                ]
            elif toks[0] == "section" and toks[1] == "LayeredShell":
                stag = int(toks[2])
                nlay = int(toks[3])
                act_secs[stag] = [
                    (int(toks[4 + 2 * k]), float(toks[5 + 2 * k])) for k in range(nlay)
                ]
            elif toks[0] == "element" and toks[1] == "ShellMITC4":
                act_elems[int(toks[2])] = int(toks[7])
            elif toks[0] == "mass":
                act_masses[int(toks[1])] = float(toks[2])
            elif toks[0] == "load":
                act_loads.append((int(toks[1]), float(toks[4])))
            else:
                unexp += 1

    res: list[FamilyResult] = []
    # 1. Materials (16)
    gen_mats = {(c, m, t): [float(x) for x in vs] for c, m, t, vs in gen_ls["mats"]}
    errs = []
    miss = len(set(act_mats) ^ set(gen_mats))
    for k, av in act_mats.items():
        if k in gen_mats:
            errs.append(max(rel_err(a, b) for a, b in zip(av, gen_mats[k])))
    m6, m4, mx = _eval_vec_errors(errs)
    res.append(
        FamilyResult("layershell.tcl", "materials", len(act_mats), m6, m4, mx, miss)
    )

    # 2. LayeredShell sections (91)
    errs = []
    miss = len(set(act_secs) ^ set(gen_ls["secs"]))
    for stag, alays in act_secs.items():
        glays = gen_ls["secs"].get(stag)
        if (
            glays is None
            or len(alays) != len(glays)
            or any(a[0] != g[0] for a, g in zip(alays, glays))
        ):
            miss += 1
        else:
            errs.append(max(rel_err(a[1], g[1]) for a, g in zip(alays, glays)))
    m6, m4, mx = _eval_vec_errors(errs)
    res.append(
        FamilyResult(
            "layershell.tcl", "LayeredShell secs", len(act_secs), m6, m4, mx, miss
        )
    )

    # 3. ShellMITC4 section assignment (16,616)
    gen_em = {eid: stag for eid, _, _, _, _, stag in gen_ls["elems"]}
    ok = sum(1 for eid, stag in act_elems.items() if gen_em.get(eid) == stag)
    res.append(
        FamilyResult(
            "layershell.tcl",
            "ShellMITC4 sec assign",
            len(act_elems),
            ok,
            ok,
            0.0 if ok == len(act_elems) else 1.0,
            len(act_elems) - ok,
        )
    )

    # 4. Shell loads (66,464)
    errs = []
    miss = abs(len(act_loads) - len(gen_ls["loads"]))
    for (an, afz), (gn, gfz) in zip(act_loads, gen_ls["loads"]):
        if an != gn:
            miss += 1
        else:
            errs.append(rel_err(afz, gfz))
    m6, m4, mx = _eval_vec_errors(errs)
    res.append(
        FamilyResult(
            "layershell.tcl",
            "loads (pattern 2)",
            len(act_loads),
            m6,
            m4,
            mx,
            miss,
            note="CB_DIAG 4-dec rounding",
        )
    )

    # 5. Nodal masses (26,380)
    errs = []
    miss = len(set(act_masses) ^ set(gen_ls["masses"]))
    for n, am in act_masses.items():
        if n in gen_ls["masses"]:
            errs.append(rel_err(am, gen_ls["masses"][n]))
    m6, m4, mx = _eval_vec_errors(errs)
    res.append(
        FamilyResult(
            "layershell.tcl",
            "nodal masses",
            len(act_masses),
            m6,
            m4,
            mx,
            miss + unexp,
            note="= -(Fz_shell+Fz_slab)/g",
        )
    )
    return res


def compare_rigiddigram(model_dir: Path, gen_rd: dict[str, Any]) -> list[FamilyResult]:
    act_loads: list[tuple[int, float]] = []
    unexp = 0
    with (model_dir / "rigiddigram.tcl").open() as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith(("#", "pattern")) or s == "}":
                continue
            toks = s.split()
            if toks[0] == "rigidDiaphragm":
                continue
            if toks[0] == "load":
                act_loads.append((int(toks[1]), float(toks[4])))
            else:
                unexp += 1

    # Compare per-panel (vertex-set + repeated tri v0 + fz) AND summed per-node Fz
    act_panels: dict[frozenset[int], tuple[int | None, float]] = {}
    act_node_fz: dict[int, float] = collections.defaultdict(float)
    for i in range(len(act_loads) // 4):
        ch = act_loads[4 * i : 4 * i + 4]
        ns = [x[0] for x in ch]
        v0 = ns[0] if ns[0] == ns[3] else None
        act_panels[frozenset(ns)] = (v0, ch[0][1])
        for n, fz in ch:
            act_node_fz[n] += fz

    gen_panels: dict[frozenset[int], tuple[int | None, float]] = {}
    for p_nodes, fz in gen_rd["panels"]:
        v0 = p_nodes[0] if p_nodes[0] == p_nodes[3] else None
        gen_panels[frozenset(p_nodes)] = (v0, fz)

    errs: list[float] = []
    miss = len(set(act_panels) ^ set(gen_panels))
    v0_miss = 0
    for gk, (av0, afz) in act_panels.items():
        gp = gen_panels.get(gk)
        if gp is None:
            continue
        if gp[0] != av0:
            v0_miss += 1
        errs.append(rel_err(afz, gp[1]))
    m6, m4, mx = _eval_vec_errors(errs)

    # Summed per-node Fz comparison (what OpenSees actually assembles into RHS!)
    gen_node_fz = gen_rd["node_fz"]
    n_errs: list[float] = []
    n_miss = len(set(act_node_fz) ^ set(gen_node_fz))
    for n, afz in act_node_fz.items():
        if n in gen_node_fz:
            n_errs.append(rel_err(afz, gen_node_fz[n]))
    nm6, nm4, nmx = _eval_vec_errors(n_errs)

    return [
        FamilyResult(
            "rigiddigram.tcl",
            "slab panels (11,190)",
            len(act_panels),
            m6,
            m4,
            mx,
            miss + v0_miss + unexp,
            note="4x loads/panel = 44,760",
        ),
        FamilyResult(
            "rigiddigram.tcl",
            "nodal slab Fz sum",
            len(act_node_fz),
            nm6,
            nm4,
            nmx,
            n_miss,
            note="assembled RHS per node",
        ),
    ]


def compare_dynamiccpu(model_dir: Path, gen_dyn: str) -> list[FamilyResult]:
    act_txt = (model_dir / "dynamiccpu.tcl").read_text()

    def _norm_lines(txt: str) -> list[str]:
        out: list[str] = []
        for raw in txt.splitlines():
            # Strip comments (full-line and inline # ...)
            s = raw.split("#", 1)[0].strip()
            if not s:
                continue
            # Normalize whitespace and trailing semicolons for semantic comparison
            s = re.sub(r"\s+", " ", s)
            s = re.sub(r"\s*;\s*$", "", s)
            # Normalize numeric literal formatting like 5e-2 vs 0.05 or .8 vs 0.8
            toks = []
            for t in s.split():
                try:
                    if re.fullmatch(r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?", t):
                        v = float(t)
                        toks.append(f"{v:.12g}")
                    else:
                        toks.append(t)
                except ValueError:
                    toks.append(t)
            out.append(" ".join(toks))
        return out

    act_lines = _norm_lines(act_txt)
    gen_lines = _norm_lines(gen_dyn)
    n = max(len(act_lines), len(gen_lines))
    matched = sum(1 for a, g in zip(act_lines, gen_lines) if a == g)
    unexp = n - matched
    return [
        FamilyResult(
            "dynamiccpu.tcl",
            "script commands",
            len(act_lines),
            matched,
            matched,
            0.0 if unexp == 0 else 1.0,
            unexp,
            note="incl. 116+128+127 rec nodes",
        )
    ]


def compute_non_geom_bytes(model_dir: Path) -> dict[str, int]:
    """Measure original bytes of non-geometric content in each Model1 file.

    For element lines that carry BOTH geometry (connectivity/transf) and
    properties (section/area/mass), we count only the property tokens (or full
    non-geometric lines for materials, sections, masses, loads, and script).
    """
    counts: dict[str, int] = {}

    def _prop_bytes(fname: str) -> int:
        tot = 0
        with (model_dir / fname).open() as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                toks = s.split()
                cmd = toks[0]
                if fname == "dynamiccpu.tcl":
                    tot += len(line.encode())
                elif cmd in ("node", "geomTransf", "rigidDiaphragm"):
                    continue
                elif cmd == "element":
                    # Count only the non-geometric property substring of element lines
                    etype = toks[1]
                    if etype == "dispBeamColumn":
                        # tokens: nIP secTag ... -mass val
                        prop_part = " ".join([toks[5], toks[6], toks[8], toks[9]])
                        tot += len(prop_part) + 1
                    elif etype == "elasticBeamColumn":
                        prop_part = " ".join(toks[5:11])
                        tot += len(prop_part) + 1
                    elif etype == "truss":
                        prop_part = " ".join(toks[5:7])
                        tot += len(prop_part) + 1
                    elif etype == "ShellMITC4":
                        tot += len(toks[7]) + 1
                else:
                    # Purely non-geometric lines: materials, sections, fibers,
                    # patches, masses, patterns, loads, closing braces
                    tot += len(line.encode())
        return tot

    for fn in (
        "opensees.tcl",
        "truss.tcl",
        "trussload.tcl",
        "Hbeam.tcl",
        "layershell.tcl",
        "rigiddigram.tcl",
        "dynamiccpu.tcl",
    ):
        counts[fn] = _prop_bytes(fn)
    return counts


def run_comparison(
    model_dir: Path, props_path: Path, generated_geometry: bool = False
) -> None:
    if generated_geometry:
        from scripts.megatall_recon.bundle import generated_geometry as _generated

        geom = _generated()
        print("Geometry: generated by geometry.py (fixture used only as reference)")
    else:
        geom = load_real_geometry(model_dir)
    props = load_props(props_path)
    gen = generate_all(geom, props)

    results: list[FamilyResult] = []
    results.extend(compare_opensees(model_dir, gen["opensees"]))
    results.extend(compare_truss(model_dir, gen["truss"]))
    results.extend(compare_hbeam(model_dir, gen["hbeam"]))
    results.extend(compare_layershell(model_dir, gen["layershell"]))
    results.extend(compare_rigiddigram(model_dir, gen["rigiddigram"]))
    results.extend(compare_dynamiccpu(model_dir, gen["dynamiccpu"]))

    print("=== PROPERTY RECONSTRUCTION RESIDUALS (Generated vs Real Model1) ===")
    hdr = (
        f"{'File':16s} {'Family':23s} {'Count':>7s} "
        f"{'<=1e-6':>9s} {'<=1e-4':>9s} {'MaxRelErr':>10s} {'Unexp':>6s}  Note"
    )
    print(hdr)
    print("-" * len(hdr) + "-" * 24)
    tot_count = 0
    tot_m6 = 0
    tot_m4 = 0
    tot_unexp = 0
    for r in results:
        p6 = 100.0 * r.match_1e6 / r.count if r.count else 100.0
        p4 = 100.0 * r.match_1e4 / r.count if r.count else 100.0
        print(
            f"{r.file_name:16s} {r.family:23s} {r.count:7,d} "
            f"{p6:8.2f}% {p4:8.2f}% {r.max_rel_err:10.3e} {r.unexplained:6d}  {r.note}"
        )
        # Avoid double-counting nodal slab Fz sum vs slab panels in grand total
        if r.family != "nodal slab Fz sum":
            tot_count += r.count
            tot_m6 += r.match_1e6
            tot_m4 += r.match_1e4
            tot_unexp += r.unexplained

    print("-" * len(hdr) + "-" * 24)
    print(
        f"{'TOTAL':16s} {'(all non-geom items)':23s} {tot_count:7,d} "
        f"{100.0 * tot_m6 / tot_count:8.2f}% {100.0 * tot_m4 / tot_count:8.2f}% "
        f"{max(r.max_rel_err for r in results):10.3e} {tot_unexp:6d}"
    )

    # Compression report
    orig_bytes = compute_non_geom_bytes(model_dir)
    props_size = props_path.stat().st_size
    sec_sizes = {k: len(json.dumps(v, separators=(",", ":"))) for k, v in props.items()}
    file_to_sec = {
        "opensees.tcl": sec_sizes["opensees"],
        "truss.tcl": sec_sizes["truss"],
        "trussload.tcl": 0,
        "Hbeam.tcl": sec_sizes["hbeam"],
        "layershell.tcl": sec_sizes["layershell"],
        "rigiddigram.tcl": sec_sizes["rigiddigram"],
        "dynamiccpu.tcl": sec_sizes["dynamiccpu"],
    }

    print("\n=== COMPRESSION REPORT (Original Non-Geometric Bytes vs props.json) ===")
    chdr = f"{'File':16s} {'Orig Non-Geom':>14s} {'props.json Sec':>15s} {'Ratio':>9s}"
    print(chdr)
    print("-" * len(chdr))
    tot_orig = 0
    for fn, ob in orig_bytes.items():
        tot_orig += ob
        sb = file_to_sec[fn]
        ratio = f"{ob / sb:7.1f}x" if sb > 0 else "  (0 B!)"
        print(f"{fn:16s} {ob:12,d} B {sb:13,d} B {ratio:>9s}")
    print("-" * len(chdr))
    print(
        f"{'TOTAL':16s} {tot_orig:12,d} B {props_size:13,d} B {tot_orig / props_size:8.1f}x"
    )
    print(
        f"\nprops.json total size: {props_size:,d} bytes ({props_size / 1024:.2f} KB, budget < 50 KB)"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--props", type=Path, default=DEFAULT_PROPS_PATH)
    ap.add_argument(
        "--generated-geometry",
        action="store_true",
        help="Generate properties from geometry.py output instead of the real nodes",
    )
    args = ap.parse_args()
    run_comparison(args.model_dir, args.props, args.generated_geometry)


if __name__ == "__main__":
    main()
