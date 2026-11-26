"""Compute all Model1 ``geomTransf PDelta`` definitions and per-beam tags from geometry.

Derived rules (all 30,166 beam elements across ``opensees.tcl``, ``Hbeam.tcl`` and
``truss.tcl``, producing all 348 ``geomTransf PDelta`` definitions on tags 1..371):

1. **Mega-columns** (``opensees.tcl``, 1,240 ``dispBeamColumn`` -> tags 346..371):
   100% exact from elevation zone (8 belt-truss zones, upper bounds
   58.5..571.7 m) and plan octant: ``|x| > |y|`` -> ``(1, 0, 0)``,
   ``|y| > |x|`` -> ``(0, 1, 0)``, diagonal ``x*y > 0`` -> ``(0.7071, 0.7071, 0)``,
   diagonal ``x*y < 0`` -> ``(0.7071, -0.7071, 0)``. Zones 1..5 have 4
   orientations (tags 346..365); zones 6..8 drop the corner mega-columns and
   have 2 orientations (tags 366..371).

2. **Rigid links** (``truss.tcl``, 616 ``elasticBeamColumn`` -> tag 277):
   100% exact: all use tag 277 with ``vecxz = (0, 0, 1)``.

3. **Non-vertical Hbeams** (``Hbeam.tcl``, 23,170 ``dispBeamColumn``: 21,982
   horizontal beams + 1,188 inclined braces):
   100% exact: all use ``vecxz = (0, 0, 1)``, one tag per Hbeam section.

4. **Vertical Hbeams** (``Hbeam.tcl``, 5,140 ``dispBeamColumn`` across 54
   sections):
   - Direction mod 180° (5,110/5,140 = 99.42% exact before exceptions):
     * Inner core (``r < 8 m``, ``(±5, ±5)``): ``0°``.
     * Outer core (``8 <= r < 17 m``, ``(±15, ±5)`` / ``(±5, ±15)``):
       ``90°`` if ``|x| > |y|`` else ``0°`` (parallel to the core face).
     * Core corners (``z < 270 m``, ``|x| ≈ 15, |y| ≈ 15``): radial
       (``45°`` if ``x*y > 0`` else ``135°``).
     * Perimeter on ``±5 m`` grid or podium ``|x| = 25`` / ``|y| = 25``
       (``z < 47 m``): cardinal tangent (``90°`` if ``|x| > |y|`` else ``0°``).
     * Perimeter on 45° diagonals (``||x| - |y|| < 0.5``): diagonal tangent
       (``135°`` if ``x*y > 0`` else ``45°``).
     * Perimeter facade below belt truss 2 (``z < 113 m``, 16 columns/ring):
       15° two-spoke tangent separated at folded polar angle 29°.
     * Perimeter facade from belt truss 2 upward (``z >= 113 m``, 8 columns/ring):
       tangent quantized to 22.5°.
   - Sign convention (``±vecxz``) follows the tower's story-zone replication
     (upper half-plane, right half-plane, clockwise tangent, or ``[67.5°, 247.5°)``),
     matching 5,044/5,140 (98.13%) vertical columns exactly before the 10-entry
     geometric exception table (96 elements = 0.32% of all 30,166 beams: model
     sign flips at core/belt-truss transitions, two 90° rotations, and two ~1°
     angle typos at 66.5° and 66.7°).

5. **Tag allocation** (tags 1..371):
   Every Hbeam section (148/148) owns a single contiguous block of tags ordered
   by the exporter's structural section sequence (with gaps at 222..233, 276,
   and 336..345 corresponding to truss/unused groups). Within each multi-vector
   section block (40/40 sections, 321/321 tags = 100%), tags are ordered by
   Windows string collation of ``f"{angle:g}_"`` (ignoring leading ``-`` on the
   primary pass and breaking ties with positive before negative).

Run ``uv run python -m scripts.megatall_recon.transf`` to verify exact tag and
definition equality against the installed Model1 fixture.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from scripts.megatall_recon import geometry as geo
from scripts.megatall_recon import properties as prop

MEGACOL_FAMILY = ("opensees.tcl", "dispBeamColumn")
HBEAM_FAMILY = ("Hbeam.tcl", "dispBeamColumn")
EBC_FAMILY = ("truss.tcl", "elasticBeamColumn")

EBC_TAG = 277
MEGACOL_FIRST_TAG = 346
MEGACOL_ZONE_TOPS = (58.5, 124.5, 195.0, 265.5, 340.5, 415.5, 495.0, 571.7)
MEGACOL_VECS = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.7071, 0.7071, 0.0),
    (0.7071, -0.7071, 0.0),
)

# Exporter structural order of the 148 Hbeam section indices (1..148).
# Negative entries encode skipped tag ranges: -12 skips unused tags 222..233,
# -2 skips unused tag 276 and rigid-link tag 277. (Tags 336..345 are skipped
# automatically because mega-columns begin at tag 346.)
_HBEAM_SECTION_ORDER: tuple[int, ...] = (
    # Typical floor W-beams (W14..W36)
    35, 87, 36, 37, 39, 32, 33, 34,
    # Belt-truss chords, outer diagonals, outer verticals (zones 1..8)
    88, 89, 113, 51, 53, 55, 94, 95,
    57, 59, 58, 61, 60, 63, 62, 64,
    68, 66, 67, 71, 72, 69, 70, 73,
    # Belt-truss inner diagonals, inner chords, radial braces, inner verticals
    24, 23, 22, 29, 28, 27, 26, 17,
    9, 11, 10, 13, 12, 15, 14, 8,
    105, 106, 107, 108, 109, 110, 111, 104,
    123, 122, 121, 120, 119, 118, 117, 116,
    # Perimeter & crown gravity / transfer columns
    125, 142, 102, 84, 54, 93,
    -12,  # tags 222..233 (unused / non-beam groups)
    # Outrigger truss members (7 groups x 6 outrigger zones)
    101, 100, 99, 98, 97, 103,
    50, 48, 49, 46, 47, 52,
    137, 134, 133, 136, 135, 140,
    45, 41, 42, 43, 44, 40,
    16, 21, 20, 19, 18, 25,
    1, 4, 5, 2, 3, 6,
    90, 92, 91, 56, 115, 96,
    -2,  # tag 276 (unused) and tag 277 (elasticBeamColumn rigid links)
    # Core corner columns (zones 1..4)
    139, 82, 138, 141,
    # Outer-core embedded columns (zones 1..8, regular & belt-truss)
    79, 80, 143, 81, 75, 38, 76, 30, 77, 74, 78, 7, 112, 86,
    # Inner-core embedded columns (zones 1..8, regular & belt-truss)
    124, 127, 85, 126, 129, 83, 128, 144, 131, 114, 130, 31, 132, 65,
    # Transition / transfer beams
    146, 147, 145, 148,
)  # fmt: skip


def _ang_to_vec(ang_deg: float) -> tuple[float, float, float]:
    """Convert a plan orientation angle in degrees to 4-decimal ``(vx, vy, 0.0)``."""
    rad = math.radians(ang_deg)
    vx = round(math.cos(rad), 4)
    vy = round(math.sin(rad), 4)
    return (0.0 if vx == 0.0 else vx, 0.0 if vy == 0.0 else vy, 0.0)


def _win_angle_sort_key(ang_deg: float) -> tuple[str, int]:
    """Emulate Windows ``CompareString`` collation on ``f'{ang:g}_'``.

    Windows word-sort ignores leading ``-`` during the primary pass and sorts
    positive before negative on ties; trailing ``_`` (ASCII 95 > digits) matches
    the separator between angle and suffix in the exporter's group key.
    """
    if abs(ang_deg - round(ang_deg)) < 1e-6:
        s = f"{round(ang_deg)}_"
    else:
        s = f"{ang_deg:.1f}_"
    return (s.lstrip("-"), 1 if ang_deg < 0 else 0)


def _megacol_tag(p1: tuple[float, float, float], p2: tuple[float, float, float]) -> int:
    """Return the exact mega-column ``geomTransf`` tag (346..371) from endpoints."""
    xm = 0.5 * (p1[0] + p2[0])
    ym = 0.5 * (p1[1] + p2[1])
    zm = 0.5 * (p1[2] + p2[2])
    z_idx = next(i for i, top in enumerate(MEGACOL_ZONE_TOPS) if zm <= top + 1e-6)
    if abs(abs(xm) - abs(ym)) < 1.0:
        orient = 2 if xm * ym > 0 else 3
    elif abs(xm) > abs(ym):
        orient = 0
    else:
        orient = 1
    if z_idx < 5:
        return MEGACOL_FIRST_TAG + 4 * z_idx + orient
    return MEGACOL_FIRST_TAG + 20 + 2 * (z_idx - 5) + orient


def _vertical_hbeam_base_angle(x: float, y: float, z1: float) -> float:
    """Predict a vertical Hbeam's orientation angle (deg, signed for collation).

    Covers 5,044 / 5,140 (98.13%) vertical Hbeams exactly before the 10-entry
    exception table below (5,110 / 5,140 = 99.42% mod 180°).
    """
    r = math.hypot(x, y)

    # 1. Inner core columns at (±5, ±5)
    if r < 8.0:
        return 0.0

    # 2. Outer core boundary columns at (±15, ±5) and (±5, ±15)
    if r < 17.0:
        return 90.0 if abs(x) > abs(y) else 0.0

    # 3. Core corner columns at (±15, ±15), zones 1..4 (radial)
    if z1 < 270.0 and abs(abs(x) - 15.0) < 0.2 and abs(abs(y) - 15.0) < 0.2:
        if z1 < 58.0:
            return 45.0 if x * y > 0 else 135.0
        return 45.0 if x * y > 0 else -45.0

    # 4. Perimeter / belt-truss columns: direction mod 180° in [0, 180)
    if min(abs(x), abs(y)) < 5.1 or (
        z1 < 47.0 and (abs(abs(x) - 25.0) < 0.2 or abs(abs(y) - 25.0) < 0.2)
    ):
        base = 90.0 if abs(x) > abs(y) else 0.0
    elif abs(abs(x) - abs(y)) < 0.5:
        base = 135.0 if x * y > 0 else 45.0
    elif z1 < 113.0:
        # 16-column lower ring: two 15° spokes separated at folded polar angle 29°
        xf, yf = max(abs(x), abs(y)), min(abs(x), abs(y))
        ap = math.degrees(math.atan2(yf, xf))
        spoke = 15.0 if ap < 29.0 else 30.0
        if abs(x) >= abs(y):
            rad = spoke if x * y >= 0 else 180.0 - spoke
        else:
            rad = 90.0 - spoke if x * y >= 0 else 90.0 + spoke
        base = (rad + 90.0) % 180.0
    else:
        # 8-column upper ring: facade tangent quantized to 22.5°
        ang_tan = (math.degrees(math.atan2(y, x)) + 90.0) % 180.0
        base = (round(ang_tan / 22.5) * 22.5) % 180.0

    # Sign convention by story zone & role (UHP / RHP / CW / [67.5°, 247.5°))
    is_bt_transfer = (47.0 <= z1 <= 54.0 and r < 34.0) or (
        113.0 <= z1 <= 120.0 and r < 31.0
    )
    if is_bt_transfer:
        mode = "RHP"
    elif 47.0 <= z1 <= 109.0:
        mode = "CW"
    elif (
        329.0 <= z1 <= 336.0
        or (340.0 <= z1 <= 395.0 and abs(abs(x) - abs(y)) < 0.5)
        or 404.0 <= z1 <= 490.0
    ):
        mode = "RHP"
    elif 560.0 <= z1 <= 567.0:
        mode = "L67"
    else:
        mode = "UHP"

    if mode == "UHP":
        return base
    if mode == "RHP":
        return base - 180.0 if base > 90.0 else base
    if mode == "CW":
        rad = math.radians(base)
        vx, vy = math.cos(rad), math.sin(rad)
        return base + 180.0 if vx * y - vy * x < 0 else base
    return base + 180.0 if base < 67.0 else base


def _vertical_hbeam_exception(
    x: float, y: float, z1: float, base_ang: float
) -> float | None:
    """Return an override angle for the 10 model-specific vertical column quirks.

    Covers 96 / 5,140 vertical Hbeams (0.32% of all 30,166 beams):
    - 32 outer-core columns in belt truss 2 (z=113.5..123.5): +180° flip
    - 8 outer-core columns below belt truss 2 (z=108.0..112.5, |x|>|y|): 180° (90° rot)
    - 8 outer-core columns at top of zone 3 (z=189.5..194.0, |x|>|y|): 270° flip
    - 12 core-corner columns at z=108.0..123.5 (x*y<0): 135° flip
    - 8 core-corner columns at z=184.0..194.0 (x*y>0): 225° flip
    - 4 belt-truss 1 transfer posts at z=53.0 (|x|≈25, |y|≈5): 0° (90° rot)
    - 4 belt-truss 6 outer posts at z=404.5..410.0 (|x|>|y|, x*y>0): 112.5° flip
    - 2 belt-truss 3 inner posts at z=184.0..189.5 (-13.93, -26.57): -22.5° flip
    - 4 belt-truss 6 inner posts at z=404.5..410.0 (|x|>|y|, x*y<0): 66.5° (~1° typo)
    - 14 facade columns in zone 7 at z=495.0..550.9 (-20.07, 11.24): 66.7° (~0.8° typo)
    """
    r = math.hypot(x, y)
    # 1. Outer core belt-truss 2 sign flip (32 elems)
    if 8.0 <= r < 17.0 and 113.0 <= z1 <= 124.0:
        return base_ang + 180.0
    # 2. Outer core below belt-truss 2 90° rotation (8 elems)
    if 8.0 <= r < 17.0 and 107.5 <= z1 <= 113.0 and abs(x) > abs(y):
        return 180.0
    # 3. Outer core top of zone 3 sign flip (8 elems)
    if 8.0 <= r < 17.0 and 189.0 <= z1 <= 194.5 and abs(x) > abs(y):
        return 270.0
    # 4. Core corner z=108..123.5 sign flip (12 elems)
    if (
        107.5 <= z1 <= 124.0
        and abs(abs(x) - 15.0) < 0.2
        and abs(abs(y) - 15.0) < 0.2
        and x * y < 0
    ):
        return 135.0
    # 5. Core corner z=184..194 sign flip (8 elems)
    if (
        183.5 <= z1 <= 194.5
        and abs(abs(x) - 15.0) < 0.2
        and abs(abs(y) - 15.0) < 0.2
        and x * y > 0
    ):
        return 225.0
    # 6. Belt-truss 1 transfer post z=53.0 90° rotation (4 elems)
    if 52.5 <= z1 <= 53.5 and abs(abs(x) - 25.0) < 0.2 and abs(abs(y) - 5.0) < 0.2:
        return 0.0
    # 7. Belt-truss 6 outer post z=404.5..410.0 sign flip (4 elems)
    if 404.0 <= z1 <= 410.5 and r > 23.0 and abs(x) > abs(y) and x * y > 0:
        return 112.5
    # 8. Belt-truss 3 inner post z=184.0..189.5 single-stack sign flip (2 elems)
    if 183.5 <= z1 <= 190.0 and 20.0 < r < 31.0 and x < 0 and y < 0 and abs(x) < abs(y):
        return -22.5
    # 9. Belt-truss 6 inner post z=404.5..410.0 66.5° typo (4 elems)
    if 404.0 <= z1 <= 410.5 and 17.0 < r < 23.0 and abs(x) > abs(y) and x * y < 0:
        return 66.5
    # 10. Zone 7 perimeter stack (-20.07, 11.24) z=495.0..550.9 66.7° typo (14 elems)
    if 494.5 <= z1 <= 551.5 and x < 0 and y > 0 and abs(x) > abs(y) and r > 20.0:
        return 66.7
    return None


def vertical_hbeam_angle(x: float, y: float, z1: float) -> float:
    """Return the exact orientation angle (deg, signed for collation) of a vertical Hbeam."""
    base = _vertical_hbeam_base_angle(x, y, z1)
    override = _vertical_hbeam_exception(x, y, z1, base)
    return base if override is None else override


def generate_transf(
    g: geo.Geometry,
    props: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute all 348 ``geomTransf PDelta`` definitions and per-beam tags.

    Returns a dictionary matching ``bundle.to_property_geometry``'s input:
    - ``geom_transfs``: list of ``(tag, "PDelta", vx, vy, vz)`` sorted by tag
    - ``megacol``: list of 1,240 tags in ``opensees.tcl`` element order
    - ``hbeam``: list of 28,310 tags in ``Hbeam.tcl`` element order
    - ``ebc``: list of 616 tags in ``truss.tcl`` element order
    """
    if props is None:
        props = prop.load_props()

    nodes = g.nodes

    # 1. Mega-columns (opensees.tcl)
    mc_eids = g.element_order[MEGACOL_FAMILY]
    mc_tags: list[int] = []
    for eid in mc_eids:
        n1, n2 = g.elements[eid][1]
        mc_tags.append(_megacol_tag(nodes[n1], nodes[n2]))

    # 2. Rigid links (truss.tcl elasticBeamColumn)
    ebc_eids = g.element_order[EBC_FAMILY]
    ebc_tags = [EBC_TAG] * len(ebc_eids)

    # 3. Hbeams (Hbeam.tcl): classify sections, compute per-element orientation key
    hb_eids = g.element_order[HBEAM_FAMILY]
    hb_edges = [
        (int(g.elements[eid][1][0]), int(g.elements[eid][1][1])) for eid in hb_eids
    ]
    hb_sidxs = prop.classify_hbeam_sections(nodes, hb_edges, props)

    # For each Hbeam element, compute orientation key: None for (0,0,1), else angle (deg)
    hb_keys: list[tuple[int, float | None]] = []
    sec_angles: dict[int, set[float | None]] = defaultdict(set)
    for (n1, n2), sidx in zip(hb_edges, hb_sidxs, strict=True):
        p1, p2 = nodes[n1], nodes[n2]
        if abs(p2[0] - p1[0]) + abs(p2[1] - p1[1]) < 0.01:
            ang: float | None = vertical_hbeam_angle(p1[0], p1[1], p1[2])
        else:
            ang = None
        hb_keys.append((sidx, ang))
        sec_angles[sidx].add(ang)

    # 4. Allocate tags 1..335 for Hbeams in exporter section order + Windows angle collation
    pair_to_tag: dict[tuple[int, float | None], int] = {}
    geom_transfs: list[tuple[int, str, float, float, float]] = []
    next_tag = 1
    for entry in _HBEAM_SECTION_ORDER:
        if entry < 0:
            next_tag += -entry
            continue
        angs = sec_angles[entry]
        if angs == {None}:
            pair_to_tag[(entry, None)] = next_tag
            geom_transfs.append((next_tag, "PDelta", 0.0, 0.0, 1.0))
            next_tag += 1
        else:
            # All-vertical section: sort angles by Windows CompareString collation
            ordered_angs = sorted(
                (a for a in angs if a is not None), key=_win_angle_sort_key
            )
            for a in ordered_angs:
                vx, vy, vz = _ang_to_vec(a)
                pair_to_tag[(entry, a)] = next_tag
                geom_transfs.append((next_tag, "PDelta", vx, vy, vz))
                next_tag += 1

    hb_tags = [pair_to_tag[k] for k in hb_keys]

    # 5. Insert rigid-link definition (tag 277) and mega-column definitions (346..371)
    geom_transfs.append((EBC_TAG, "PDelta", 0.0, 0.0, 1.0))
    for z_idx in range(len(MEGACOL_ZONE_TOPS)):
        n_orients = 4 if z_idx < 5 else 2
        base_tag = (
            MEGACOL_FIRST_TAG + 4 * z_idx
            if z_idx < 5
            else MEGACOL_FIRST_TAG + 20 + 2 * (z_idx - 5)
        )
        for o in range(n_orients):
            vx, vy, vz = MEGACOL_VECS[o]
            geom_transfs.append((base_tag + o, "PDelta", vx, vy, vz))

    geom_transfs.sort(key=lambda row: row[0])
    return {
        "geom_transfs": geom_transfs,
        "megacol": mc_tags,
        "hbeam": hb_tags,
        "ebc": ebc_tags,
    }


def verify_against_real(
    model_dir: Path = prop.DEFAULT_MODEL_DIR,
    use_generated_geometry: bool = True,
) -> bool:
    """Compare generated geomTransf definitions and per-beam tags against Model1."""
    real = prop.load_real_geometry(prop.require_real_model(model_dir))
    if use_generated_geometry:
        g = geo.generate()
    else:
        # Wrap real geometry into geo.Geometry
        elements: dict[int, geo.Element] = {}
        order: dict[geo.Family, tuple[int, ...]] = {}
        for fam, kind, rows in (
            (MEGACOL_FAMILY, "dispBeamColumn", real.megacol_elems),
            (HBEAM_FAMILY, "dispBeamColumn", real.hbeam_elems),
            (EBC_FAMILY, "elasticBeamColumn", real.ebc_elems),
        ):
            eids = []
            for eid, n1, n2, _ in rows:
                elements[eid] = (kind, (n1, n2))
                eids.append(eid)
            order[fam] = tuple(eids)
        g = geo.Geometry(
            nodes=real.nodes,
            elements=elements,
            diaphragms={},
            element_order=order,
        )

    gen = generate_transf(g)
    ok = True

    def_match = gen["geom_transfs"] == real.geom_transfs
    print(
        f"geomTransf definitions: {len(gen['geom_transfs'])}/{len(real.geom_transfs)} "
        f"exact={def_match}"
    )
    if not def_match:
        ok = False
        real_map = {r[0]: r for r in real.geom_transfs}
        gen_map = {r[0]: r for r in gen["geom_transfs"]}
        for t in sorted(set(real_map) | set(gen_map)):
            if real_map.get(t) != gen_map.get(t):
                print(
                    f"  DIFF def tag={t}: gen={gen_map.get(t)} real={real_map.get(t)}"
                )

    for label, gen_tags, real_rows in (
        ("megacol", gen["megacol"], real.megacol_elems),
        ("hbeam", gen["hbeam"], real.hbeam_elems),
        ("ebc", gen["ebc"], real.ebc_elems),
    ):
        real_tags = [r[3] for r in real_rows]
        mismatches = sum(1 for a, b in zip(gen_tags, real_tags, strict=True) if a != b)
        print(
            f"{label:8s} per-beam tags: {len(real_tags) - mismatches}/{len(real_tags)} "
            f"exact (mismatches={mismatches})"
        )
        if mismatches:
            ok = False
            for (eid, _, _, rt), gt in zip(real_rows, gen_tags, strict=True):
                if gt != rt:
                    print(f"  DIFF {label} eid={eid}: gen={gt} real={rt}")
                    break

    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", type=Path, default=prop.DEFAULT_MODEL_DIR)
    ap.add_argument(
        "--real-geometry",
        action="store_true",
        help="Verify on fixture geometry in addition to generated geometry",
    )
    args = ap.parse_args()
    print("=== Verification on GENERATED geometry ===")
    ok_gen = verify_against_real(args.model_dir, use_generated_geometry=True)
    ok_real = True
    if args.real_geometry:
        print("=== Verification on REAL fixture geometry ===")
        ok_real = verify_against_real(args.model_dir, use_generated_geometry=False)
    if not (ok_gen and ok_real):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
