import csv
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path


repo_root = Path(__file__).resolve().parents[2]


def _run_section_path(case_data):
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        input_path = tmpdir / "section_input.json"
        output_path = tmpdir / "section_output.csv"
        input_path.write_text(json.dumps(case_data), encoding="utf-8")
        subprocess.check_call(
            [
                sys.executable,
                str(repo_root / "scripts" / "run_strut_section_path.py"),
                "--input",
                str(input_path),
                "--output",
                str(output_path),
            ]
        )
        with output_path.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    return [{k: float(v) for k, v in row.items()} for row in rows]


def _discretize_fibers(section_params):
    fibers = []
    for patch in section_params.get("patches", []):
        patch_type = patch["type"]
        if patch_type == "quad":
            patch_type = "quadr"
        if patch_type == "rect":
            ny = patch["num_subdiv_y"]
            nz = patch["num_subdiv_z"]
            y0, y1 = sorted([patch["y_i"], patch["y_j"]])
            z0, z1 = sorted([patch["z_i"], patch["z_j"]])
            dy = (y1 - y0) / ny
            dz = (z1 - z0) / nz
            area = dy * dz
            for iy in range(ny):
                y = y0 + (iy + 0.5) * dy
                for iz in range(nz):
                    z = z0 + (iz + 0.5) * dz
                    fibers.append((y, z, area, patch["material"]))
        elif patch_type == "quadr":
            ny = patch["num_subdiv_y"]
            nz = patch["num_subdiv_z"]
            yi, zi = patch["y_i"], patch["z_i"]
            yj, zj = patch["y_j"], patch["z_j"]
            yk, zk = patch["y_k"], patch["z_k"]
            yl, zl = patch["y_l"], patch["z_l"]
            for i in range(ny):
                u0 = i / ny
                u1 = (i + 1) / ny
                uc = 0.5 * (u0 + u1)
                for j in range(nz):
                    v0 = j / nz
                    v1 = (j + 1) / nz
                    vc = 0.5 * (v0 + v1)
                    y00 = (
                        (1 - u0) * (1 - v0) * yi
                        + u0 * (1 - v0) * yj
                        + u0 * v0 * yk
                        + (1 - u0) * v0 * yl
                    )
                    z00 = (
                        (1 - u0) * (1 - v0) * zi
                        + u0 * (1 - v0) * zj
                        + u0 * v0 * zk
                        + (1 - u0) * v0 * zl
                    )
                    y10 = (
                        (1 - u1) * (1 - v0) * yi
                        + u1 * (1 - v0) * yj
                        + u1 * v0 * yk
                        + (1 - u1) * v0 * yl
                    )
                    z10 = (
                        (1 - u1) * (1 - v0) * zi
                        + u1 * (1 - v0) * zj
                        + u1 * v0 * zk
                        + (1 - u1) * v0 * zl
                    )
                    y11 = (
                        (1 - u1) * (1 - v1) * yi
                        + u1 * (1 - v1) * yj
                        + u1 * v1 * yk
                        + (1 - u1) * v1 * yl
                    )
                    z11 = (
                        (1 - u1) * (1 - v1) * zi
                        + u1 * (1 - v1) * zj
                        + u1 * v1 * zk
                        + (1 - u1) * v1 * zl
                    )
                    y01 = (
                        (1 - u0) * (1 - v1) * yi
                        + u0 * (1 - v1) * yj
                        + u0 * v1 * yk
                        + (1 - u0) * v1 * yl
                    )
                    z01 = (
                        (1 - u0) * (1 - v1) * zi
                        + u0 * (1 - v1) * zj
                        + u0 * v1 * zk
                        + (1 - u0) * v1 * zl
                    )
                    area = 0.5 * abs(
                        y00 * z10
                        + y10 * z11
                        + y11 * z01
                        + y01 * z00
                        - z00 * y10
                        - z10 * y11
                        - z11 * y01
                        - z01 * y00
                    )
                    y = (
                        (1 - uc) * (1 - vc) * yi
                        + uc * (1 - vc) * yj
                        + uc * vc * yk
                        + (1 - uc) * vc * yl
                    )
                    z = (
                        (1 - uc) * (1 - vc) * zi
                        + uc * (1 - vc) * zj
                        + uc * vc * zk
                        + (1 - uc) * vc * zl
                    )
                    fibers.append((y, z, area, patch["material"]))
        else:
            raise AssertionError(f"unsupported patch type {patch['type']}")
    for layer in section_params.get("layers", []):
        assert layer["type"] == "straight"
        n = layer["num_bars"]
        area = layer["bar_area"]
        ys = layer["y_start"]
        zs = layer["z_start"]
        ye = layer["y_end"]
        ze = layer["z_end"]
        if n == 1:
            fibers.append((0.5 * (ys + ye), 0.5 * (zs + ze), area, layer["material"]))
        else:
            for i in range(n):
                t = i / (n - 1)
                fibers.append(
                    (ys + (ye - ys) * t, zs + (ze - zs) * t, area, layer["material"])
                )
    for fiber in section_params.get("fibers", []):
        fibers.append((fiber["y"], fiber["z"], fiber["area"], fiber["material"]))
    return fibers


def _expected_elastic_response(case_data, eps0, kappa):
    e_by_mat = {
        m["id"]: m["params"]["E"]
        for m in case_data["materials"]
        if m["type"] == "Elastic"
    }
    fibers = _discretize_fibers(case_data["section"]["params"])
    a_sum = sum(area for y, z, area, mat in fibers)
    y_bar = sum(area * y for y, z, area, mat in fibers) / a_sum
    n = 0.0
    mz = 0.0
    k11 = 0.0
    k12 = 0.0
    k22 = 0.0
    for y, z, area, mat in fibers:
        e = e_by_mat[mat]
        y_rel = y - y_bar
        strain = eps0 - y_rel * kappa
        stress = e * strain
        fs = stress * area
        ks = e * area
        n += fs
        mz += -fs * y_rel
        k11 += ks
        k12 += -ks * y_rel
        k22 += ks * y_rel * y_rel
    return {"N": n, "Mz": mz, "k11": k11, "k12": k12, "k22": k22}


def test_fiber_section_rect_patch_elastic_aggregation():
    case_data = {
        "materials": [{"id": 1, "type": "Elastic", "params": {"E": 30000000000.0}}],
        "section": {
            "id": 1,
            "type": "FiberSection2d",
            "params": {
                "patches": [
                    {
                        "type": "rect",
                        "material": 1,
                        "num_subdiv_y": 6,
                        "num_subdiv_z": 4,
                        "y_i": -0.3,
                        "z_i": -0.2,
                        "y_j": 0.3,
                        "z_j": 0.2,
                    }
                ],
                "layers": [],
            },
        },
        "deformation_path": [{"eps0": 1.2e-4, "kappa": 2.5e-3}],
    }
    rows = _run_section_path(case_data)
    assert len(rows) == 1
    got = rows[0]
    expected = _expected_elastic_response(case_data, eps0=1.2e-4, kappa=2.5e-3)
    assert math.isclose(got["N"], expected["N"], rel_tol=1e-10, abs_tol=1e-10)
    assert math.isclose(got["Mz"], expected["Mz"], rel_tol=1e-10, abs_tol=1e-10)
    assert math.isclose(got["k11"], expected["k11"], rel_tol=1e-10, abs_tol=1e-10)
    assert math.isclose(got["k12"], expected["k12"], rel_tol=1e-8, abs_tol=1e-6)
    assert math.isclose(got["k22"], expected["k22"], rel_tol=1e-10, abs_tol=1e-10)


def test_fiber_section_straight_layer_single_bar_midpoint():
    case_data = {
        "materials": [{"id": 1, "type": "Elastic", "params": {"E": 200000000000.0}}],
        "section": {
            "id": 1,
            "type": "FiberSection2d",
            "params": {
                "patches": [],
                "layers": [
                    {
                        "type": "straight",
                        "material": 1,
                        "num_bars": 1,
                        "bar_area": 0.0005,
                        "y_start": -0.1,
                        "z_start": 0.0,
                        "y_end": 0.3,
                        "z_end": 0.0,
                    }
                ],
            },
        },
        "deformation_path": [{"eps0": 0.0, "kappa": 0.02}],
    }
    rows = _run_section_path(case_data)
    assert len(rows) == 1
    got = rows[0]
    assert math.isclose(got["N"], 0.0, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(got["Mz"], 0.0, rel_tol=0.0, abs_tol=1e-12)
    assert got["k11"] > 0.0
    assert math.isclose(got["k12"], 0.0, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(got["k22"], 0.0, rel_tol=0.0, abs_tol=1e-12)


def test_fiber_section_quadr_patch_elastic_aggregation():
    case_data = {
        "materials": [{"id": 1, "type": "Elastic", "params": {"E": 3.0e10}}],
        "section": {
            "id": 2,
            "type": "FiberSection2d",
            "params": {
                "patches": [
                    {
                        "type": "quadr",
                        "material": 1,
                        "num_subdiv_y": 4,
                        "num_subdiv_z": 3,
                        "y_i": -0.4,
                        "z_i": 0.2,
                        "y_j": -0.2,
                        "z_j": -0.2,
                        "y_k": 0.4,
                        "z_k": -0.1,
                        "y_l": 0.2,
                        "z_l": 0.3,
                    }
                ],
                "layers": [],
            },
        },
        "deformation_path": [{"eps0": 8.0e-5, "kappa": -1.5e-3}],
    }
    rows = _run_section_path(case_data)
    assert len(rows) == 1
    got = rows[0]
    expected = _expected_elastic_response(case_data, eps0=8.0e-5, kappa=-1.5e-3)
    assert math.isclose(got["N"], expected["N"], rel_tol=1e-10, abs_tol=1e-10)
    assert math.isclose(got["Mz"], expected["Mz"], rel_tol=1e-10, abs_tol=1e-10)
    assert math.isclose(got["k11"], expected["k11"], rel_tol=1e-10, abs_tol=1e-10)
    assert math.isclose(got["k12"], expected["k12"], rel_tol=1e-10, abs_tol=1e-10)
    assert math.isclose(got["k22"], expected["k22"], rel_tol=1e-10, abs_tol=1e-10)


def _mixed_family_section_case(layer_order):
    layers = []
    for material, y, area in layer_order:
        layers.append(
            {
                "type": "straight",
                "material": material,
                "num_bars": 1,
                "bar_area": area,
                "y_start": y,
                "z_start": 0.0,
                "y_end": y,
                "z_end": 0.0,
            }
        )
    return {
        "materials": [
            {
                "id": 1,
                "type": "Concrete02",
                "params": {
                    "fpc": -32.0,
                    "epsc0": -0.0021,
                    "fpcu": -18.0,
                    "epscu": -0.0058,
                    "rat": 0.12,
                    "ft": 2.8,
                    "Ets": 180.0,
                },
            },
            {
                "id": 2,
                "type": "Concrete02",
                "params": {
                    "fpc": -36.0,
                    "epsc0": -0.0024,
                    "fpcu": -21.0,
                    "epscu": -0.0064,
                    "rat": 0.1,
                    "ft": 3.1,
                    "Ets": 220.0,
                },
            },
            {
                "id": 3,
                "type": "Steel02",
                "params": {
                    "Fy": 420.0,
                    "E0": 200000.0,
                    "b": 0.012,
                    "R0": 18.0,
                    "cR1": 0.925,
                    "cR2": 0.15,
                    "a1": 0.01,
                    "a2": 1.0,
                    "a3": 0.01,
                    "a4": 1.0,
                },
            },
            {
                "id": 4,
                "type": "Steel02",
                "params": {
                    "Fy": 460.0,
                    "E0": 200000.0,
                    "b": 0.01,
                    "R0": 20.0,
                    "cR1": 0.92,
                    "cR2": 0.18,
                    "a1": 0.015,
                    "a2": 1.0,
                    "a3": 0.015,
                    "a4": 1.0,
                },
            },
        ],
        "section": {
            "id": 9,
            "type": "FiberSection2d",
            "params": {"patches": [], "layers": layers},
        },
        "deformation_path": [
            {"eps0": -8.0e-4, "kappa": 6.0e-3},
            {"eps0": -1.6e-3, "kappa": 1.0e-2},
            {"eps0": -2.0e-4, "kappa": -4.0e-3},
            {"eps0": 4.0e-4, "kappa": -8.0e-3},
            {"eps0": -6.0e-4, "kappa": 5.0e-3},
        ],
    }


def test_fiber_section_mixed_material_order_is_response_invariant():
    alternating_order = [
        (1, -0.30, 0.0042),
        (3, -0.24, 0.00045),
        (2, -0.10, 0.0040),
        (4, -0.04, 0.00055),
        (3, 0.04, 0.00045),
        (1, 0.10, 0.0042),
        (4, 0.24, 0.00055),
        (2, 0.30, 0.0040),
    ]
    grouped_order = [
        (1, -0.30, 0.0042),
        (1, 0.10, 0.0042),
        (2, -0.10, 0.0040),
        (2, 0.30, 0.0040),
        (3, -0.24, 0.00045),
        (3, 0.04, 0.00045),
        (4, -0.04, 0.00055),
        (4, 0.24, 0.00055),
    ]

    alternating_rows = _run_section_path(_mixed_family_section_case(alternating_order))
    grouped_rows = _run_section_path(_mixed_family_section_case(grouped_order))

    assert len(alternating_rows) == len(grouped_rows)
    for alternating_row, grouped_row in zip(alternating_rows, grouped_rows):
        for key in alternating_row:
            assert math.isclose(
                alternating_row[key],
                grouped_row[key],
                rel_tol=1e-10,
                abs_tol=1e-8,
            )


def test_fiber_section2d_condenses_same_y_fibers_without_changing_response():
    """A 2D section merges same-material fibers that share a y coordinate.

    The z-refined section must respond exactly like a section that lists one
    fiber per (material, y) with the summed area, through a cyclic nonlinear
    path. Steel bars sit at the same y as concrete fibers, so merging across
    materials would show up as a different response.
    """
    base = _mixed_family_section_case([])
    materials = base["materials"]
    deformation_path = base["deformation_path"]
    half_depth = 0.3
    half_width = 0.2
    rows_y = 8
    refined_params = {
        "patches": [
            {
                "type": "rect",
                "material": 1,
                "num_subdiv_y": rows_y,
                "num_subdiv_z": 12,
                "y_i": -half_depth,
                "z_i": -half_width,
                "y_j": half_depth,
                "z_j": half_width,
            }
        ],
        "layers": [
            # Bars spread along z at one y: these collapse to a single fiber.
            {
                "type": "straight",
                "material": 3,
                "num_bars": 4,
                "bar_area": 0.0004,
                "y_start": -0.2625,
                "z_start": -0.15,
                "y_end": -0.2625,
                "z_end": 0.15,
            },
            {
                "type": "straight",
                "material": 3,
                "num_bars": 3,
                "bar_area": 0.0004,
                "y_start": 0.2625,
                "z_start": -0.15,
                "y_end": 0.2625,
                "z_end": 0.15,
            },
        ],
    }
    row_height = 2.0 * half_depth / rows_y
    condensed_fibers = [
        {
            "y": -half_depth + (i + 0.5) * row_height,
            "z": 0.0,
            "area": row_height * 2.0 * half_width,
            "material": 1,
        }
        for i in range(rows_y)
    ]
    # -0.2625 and 0.2625 are also concrete row centroids (first and last row).
    condensed_fibers.append({"y": -0.2625, "z": 0.0, "area": 4 * 0.0004, "material": 3})
    condensed_fibers.append({"y": 0.2625, "z": 0.0, "area": 3 * 0.0004, "material": 3})

    refined_rows = _run_section_path(
        {
            "materials": materials,
            "section": {"id": 1, "type": "FiberSection2d", "params": refined_params},
            "deformation_path": deformation_path,
        }
    )
    condensed_rows = _run_section_path(
        {
            "materials": materials,
            "section": {
                "id": 1,
                "type": "FiberSection2d",
                "params": {"patches": [], "layers": [], "fibers": condensed_fibers},
            },
            "deformation_path": deformation_path,
        }
    )

    assert len(refined_rows) == len(condensed_rows) == len(deformation_path)
    for refined_row, condensed_row in zip(refined_rows, condensed_rows):
        for key in refined_row:
            assert math.isclose(
                refined_row[key],
                condensed_row[key],
                rel_tol=1e-10,
                abs_tol=1e-8,
            ), key


def _expected_elastic_response_3d(case_data, eps0, ky, kz):
    e_by_mat = {
        m["id"]: m["params"]["E"]
        for m in case_data["materials"]
        if m["type"] == "Elastic"
    }
    fibers = _discretize_fibers(case_data["section"]["params"])
    a_sum = sum(area for y, z, area, mat in fibers)
    y_bar = sum(area * y for y, z, area, mat in fibers) / a_sum
    z_bar = sum(area * z for y, z, area, mat in fibers) / a_sum
    n = 0.0
    my = 0.0
    mz = 0.0
    k11 = 0.0
    k12 = 0.0
    k13 = 0.0
    k22 = 0.0
    k23 = 0.0
    k33 = 0.0
    for y, z, area, mat in fibers:
        e = e_by_mat[mat]
        y_rel = y - y_bar
        z_rel = z - z_bar
        strain = eps0 + z_rel * ky - y_rel * kz
        stress = e * strain
        fs = stress * area
        ks = e * area
        n += fs
        my += fs * z_rel
        mz += -fs * y_rel
        k11 += ks
        k12 += ks * z_rel
        k13 += -ks * y_rel
        k22 += ks * z_rel * z_rel
        k23 += -ks * z_rel * y_rel
        k33 += ks * y_rel * y_rel
    return {
        "N": n,
        "My": my,
        "Mz": mz,
        "k11": k11,
        "k12": k12,
        "k13": k13,
        "k22": k22,
        "k23": k23,
        "k33": k33,
    }


def test_fiber_section3d_rect_patch_elastic_aggregation():
    case_data = {
        "materials": [{"id": 1, "type": "Elastic", "params": {"E": 2.8e10}}],
        "section": {
            "id": 1,
            "type": "FiberSection3d",
            "params": {
                "patches": [
                    {
                        "type": "rect",
                        "material": 1,
                        "num_subdiv_y": 6,
                        "num_subdiv_z": 5,
                        "y_i": -0.3,
                        "z_i": -0.2,
                        "y_j": 0.3,
                        "z_j": 0.2,
                    }
                ],
                "layers": [],
            },
        },
        "deformation_path": [{"eps0": 1.2e-4, "kappa_y": -7.5e-4, "kappa_z": 2.5e-3}],
    }
    rows = _run_section_path(case_data)
    assert len(rows) == 1
    got = rows[0]
    expected = _expected_elastic_response_3d(
        case_data, eps0=1.2e-4, ky=-7.5e-4, kz=2.5e-3
    )
    assert math.isclose(got["N"], expected["N"], rel_tol=1e-10, abs_tol=1e-10)
    assert math.isclose(got["My"], expected["My"], rel_tol=1e-10, abs_tol=1e-10)
    assert math.isclose(got["Mz"], expected["Mz"], rel_tol=1e-10, abs_tol=1e-10)
    assert math.isclose(got["k11"], expected["k11"], rel_tol=1e-10, abs_tol=1e-10)
    assert math.isclose(got["k12"], expected["k12"], rel_tol=1e-8, abs_tol=1e-6)
    assert math.isclose(got["k13"], expected["k13"], rel_tol=1e-8, abs_tol=1e-6)
    assert math.isclose(got["k22"], expected["k22"], rel_tol=1e-10, abs_tol=1e-10)
    assert math.isclose(got["k23"], expected["k23"], rel_tol=1e-8, abs_tol=1e-6)
    assert math.isclose(got["k33"], expected["k33"], rel_tol=1e-10, abs_tol=1e-10)


def test_fiber_section3d_inline_fibers_elastic_aggregation():
    case_data = {
        "materials": [
            {"id": 1, "type": "Elastic", "params": {"E": 2.8e10}},
            {"id": 2, "type": "Elastic", "params": {"E": 2.0e11}},
        ],
        "section": {
            "id": 2,
            "type": "FiberSection3d",
            "params": {
                "patches": [],
                "layers": [],
                "fibers": [
                    {"y": -0.3, "z": -0.15, "area": 0.012, "material": 1},
                    {"y": -0.1, "z": 0.12, "area": 0.01, "material": 1},
                    {"y": 0.2, "z": -0.08, "area": 0.009, "material": 2},
                    {"y": 0.35, "z": 0.18, "area": 0.011, "material": 2},
                ],
            },
        },
        "deformation_path": [{"eps0": 9.0e-5, "kappa_y": -6.5e-4, "kappa_z": 1.8e-3}],
    }
    rows = _run_section_path(case_data)
    assert len(rows) == 1
    got = rows[0]
    expected = _expected_elastic_response_3d(
        case_data, eps0=9.0e-5, ky=-6.5e-4, kz=1.8e-3
    )
    assert math.isclose(got["N"], expected["N"], rel_tol=1e-10, abs_tol=1e-10)
    assert math.isclose(got["My"], expected["My"], rel_tol=1e-10, abs_tol=1e-10)
    assert math.isclose(got["Mz"], expected["Mz"], rel_tol=1e-10, abs_tol=1e-10)
    assert math.isclose(got["k11"], expected["k11"], rel_tol=1e-10, abs_tol=1e-10)
    assert math.isclose(got["k12"], expected["k12"], rel_tol=1e-10, abs_tol=1e-10)
    assert math.isclose(got["k13"], expected["k13"], rel_tol=1e-10, abs_tol=1e-10)
    assert math.isclose(got["k22"], expected["k22"], rel_tol=1e-10, abs_tol=1e-10)
    assert math.isclose(got["k23"], expected["k23"], rel_tol=1e-10, abs_tol=1e-10)
    assert math.isclose(got["k33"], expected["k33"], rel_tol=1e-10, abs_tol=1e-10)


def _single_step_section_path(case_data, eps0, ky, kz):
    run_case = json.loads(json.dumps(case_data))
    run_case["deformation_path"] = [{"eps0": eps0, "kappa_y": ky, "kappa_z": kz}]
    rows = _run_section_path(run_case)
    assert len(rows) == 1
    return rows[0]


def test_fiber_section3d_tangent_matches_finite_difference():
    case_data = {
        "materials": [
            {"id": 1, "type": "Elastic", "params": {"E": 3.0e10}},
            {"id": 2, "type": "Elastic", "params": {"E": 2.2e11}},
        ],
        "section": {
            "id": 7,
            "type": "FiberSection3d",
            "params": {
                "patches": [
                    {
                        "type": "quadr",
                        "material": 1,
                        "num_subdiv_y": 4,
                        "num_subdiv_z": 3,
                        "y_i": -0.35,
                        "z_i": -0.2,
                        "y_j": -0.3,
                        "z_j": 0.22,
                        "y_k": 0.28,
                        "z_k": 0.18,
                        "y_l": 0.33,
                        "z_l": -0.25,
                    }
                ],
                "layers": [
                    {
                        "type": "straight",
                        "material": 2,
                        "num_bars": 3,
                        "bar_area": 0.0004,
                        "y_start": -0.25,
                        "z_start": 0.16,
                        "y_end": 0.1,
                        "z_end": 0.2,
                    }
                ],
            },
        },
        "deformation_path": [],
    }
    eps0 = 8.0e-5
    ky = -1.2e-3
    kz = 1.7e-3
    d = 1.0e-7
    base = _single_step_section_path(case_data, eps0, ky, kz)
    plus_eps = _single_step_section_path(case_data, eps0 + d, ky, kz)
    plus_ky = _single_step_section_path(case_data, eps0, ky + d, kz)
    plus_kz = _single_step_section_path(case_data, eps0, ky, kz + d)

    d_n_deps = (plus_eps["N"] - base["N"]) / d
    d_n_dky = (plus_ky["N"] - base["N"]) / d
    d_n_dkz = (plus_kz["N"] - base["N"]) / d
    d_my_deps = (plus_eps["My"] - base["My"]) / d
    d_my_dky = (plus_ky["My"] - base["My"]) / d
    d_my_dkz = (plus_kz["My"] - base["My"]) / d
    d_mz_deps = (plus_eps["Mz"] - base["Mz"]) / d
    d_mz_dky = (plus_ky["Mz"] - base["Mz"]) / d
    d_mz_dkz = (plus_kz["Mz"] - base["Mz"]) / d

    assert math.isclose(base["k11"], d_n_deps, rel_tol=1e-5, abs_tol=1e-2)
    assert math.isclose(base["k12"], d_n_dky, rel_tol=1e-5, abs_tol=1e-2)
    assert math.isclose(base["k13"], d_n_dkz, rel_tol=1e-5, abs_tol=1e-2)
    assert math.isclose(base["k12"], d_my_deps, rel_tol=1e-5, abs_tol=1e-2)
    assert math.isclose(base["k22"], d_my_dky, rel_tol=1e-5, abs_tol=1e-2)
    assert math.isclose(base["k23"], d_my_dkz, rel_tol=1e-5, abs_tol=1e-2)
    assert math.isclose(base["k13"], d_mz_deps, rel_tol=1e-5, abs_tol=1e-2)
    assert math.isclose(base["k23"], d_mz_dky, rel_tol=1e-5, abs_tol=1e-2)
    assert math.isclose(base["k33"], d_mz_dkz, rel_tol=1e-5, abs_tol=1e-2)


def _discretize_circ_fibers(section_params):
    fibers = []
    for patch in section_params.get("patches", []):
        assert patch["type"] == "circ"
        nc = patch["num_subdiv_circ"]
        nr = patch["num_subdiv_rad"]
        yc = patch["y_center"]
        zc = patch["z_center"]
        ri = patch["int_rad"]
        re = patch["ext_rad"]
        start_rad = math.radians(patch.get("start_ang", 0.0))
        end_rad = math.radians(patch.get("end_ang", 360.0))
        d_rad = (re - ri) / nr
        d_theta = (end_rad - start_rad) / nc
        for j in range(nr):
            r1 = ri + j * d_rad
            r2 = r1 + d_rad
            for i in range(nc):
                t1 = start_rad + i * d_theta
                t2 = t1 + d_theta
                area = 0.5 * abs(t2 - t1) * (r2 * r2 - r1 * r1)
                if abs(t2 - t1) > 1.0e-12 and (r2 * r2 - r1 * r1) > 0.0:
                    r_bar = (
                        (4.0 / 3.0)
                        * (math.sin(0.5 * (t2 - t1)) / (0.5 * (t2 - t1)))
                        * (r2 * r2 * r2 - r1 * r1 * r1)
                        / (r2 * r2 - r1 * r1)
                    )
                    # OpenSees CircSectionCell::getCentroidPosition uses 4/3 * sin(dtheta/2)/dtheta * ...
                    # Wait: check exact factor 2/3 * sin(dtheta/2)/(dtheta/2) = 4/3 * sin(dtheta/2)/dtheta!
                    r_bar *= 0.5
                else:
                    r_bar = 0.5 * (r1 + r2)
                theta_mid = 0.5 * (t1 + t2)
                y = yc + r_bar * math.cos(theta_mid)
                z = zc + r_bar * math.sin(theta_mid)
                fibers.append((y, z, area, patch["material"]))
    for layer in section_params.get("layers", []):
        assert layer["type"] == "circ"
        n = layer["num_bars"]
        area = layer["bar_area"]
        yc = layer["y_center"]
        zc = layer["z_center"]
        rad = layer["radius"]
        has_angles = "start_ang" in layer and "end_ang" in layer
        if has_angles:
            a0 = layer["start_ang"]
            a1 = layer["end_ang"]
        else:
            a0 = 0.0
            a1 = 360.0 - 360.0 / n
        if n == 1:
            angles = [math.radians(a0)]
        else:
            d_ang = math.radians(a1 - a0) / (n - 1)
            angles = [math.radians(a0) + i * d_ang for i in range(n)]
        for ang in angles:
            fibers.append(
                (
                    yc + rad * math.cos(ang),
                    zc + rad * math.sin(ang),
                    area,
                    layer["material"],
                )
            )
    return fibers


def test_fiber_section2d_circ_patch_and_layer_elastic_aggregation():
    case_data = {
        "materials": [
            {"id": 1, "type": "Elastic", "params": {"E": 3.0e10}},
            {"id": 2, "type": "Elastic", "params": {"E": 2.0e11}},
        ],
        "section": {
            "id": 1,
            "type": "FiberSection2d",
            "params": {
                "patches": [
                    {
                        "type": "circ",
                        "material": 1,
                        "num_subdiv_circ": 8,
                        "num_subdiv_rad": 4,
                        "y_center": 0.05,
                        "z_center": -0.02,
                        "int_rad": 0.1,
                        "ext_rad": 0.3,
                        "start_ang": 0.0,
                        "end_ang": 180.0,
                    }
                ],
                "layers": [
                    {
                        "type": "circ",
                        "material": 2,
                        "num_bars": 8,
                        "bar_area": 0.0003,
                        "y_center": 0.05,
                        "z_center": -0.02,
                        "radius": 0.25,
                        "start_ang": 22.5,
                        "end_ang": 337.5,
                    }
                ],
            },
        },
        "deformation_path": [{"eps0": 1.5e-4, "kappa": 2.0e-3}],
    }
    rows = _run_section_path(case_data)
    assert len(rows) == 1
    got = rows[0]

    e_by_mat = {m["id"]: m["params"]["E"] for m in case_data["materials"]}
    fibers = _discretize_circ_fibers(case_data["section"]["params"])
    a_sum = sum(area for _y, _z, area, _mat in fibers)
    y_bar = sum(area * y for y, _z, area, _mat in fibers) / a_sum
    eps0 = 1.5e-4
    kappa = 2.0e-3
    n = mz = k11 = k12 = k22 = 0.0
    for y, _z, area, mat in fibers:
        e = e_by_mat[mat]
        y_rel = y - y_bar
        strain = eps0 - y_rel * kappa
        fs = e * strain * area
        ks = e * area
        n += fs
        mz += -fs * y_rel
        k11 += ks
        k12 += -ks * y_rel
        k22 += ks * y_rel * y_rel

    assert math.isclose(got["N"], n, rel_tol=1e-10, abs_tol=1e-8)
    assert math.isclose(got["Mz"], mz, rel_tol=1e-10, abs_tol=1e-8)
    assert math.isclose(got["k11"], k11, rel_tol=1e-10, abs_tol=1e-8)
    assert math.isclose(got["k12"], k12, rel_tol=1e-10, abs_tol=1e-6)
    assert math.isclose(got["k22"], k22, rel_tol=1e-10, abs_tol=1e-8)


def test_fiber_section3d_circ_patch_and_default_angle_layer_elastic_aggregation():
    case_data = {
        "materials": [
            {"id": 1, "type": "Elastic", "params": {"E": 2.8e10}},
            {"id": 2, "type": "Elastic", "params": {"E": 2.0e11}},
        ],
        "section": {
            "id": 1,
            "type": "FiberSection3d",
            "params": {
                "patches": [
                    {
                        "type": "circ",
                        "material": 1,
                        "num_subdiv_circ": 12,
                        "num_subdiv_rad": 3,
                        "y_center": 0.0,
                        "z_center": 0.0,
                        "int_rad": 0.0,
                        "ext_rad": 0.25,
                        "start_ang": 0.0,
                        "end_ang": 360.0,
                    }
                ],
                "layers": [
                    {
                        "type": "circ",
                        "material": 2,
                        "num_bars": 6,
                        "bar_area": 0.0002,
                        "y_center": 0.0,
                        "z_center": 0.0,
                        "radius": 0.2,
                    }
                ],
            },
        },
        "deformation_path": [{"eps0": -1.0e-4, "kappa_y": 8.0e-4, "kappa_z": -1.2e-3}],
    }
    rows = _run_section_path(case_data)
    assert len(rows) == 1
    got = rows[0]

    e_by_mat = {m["id"]: m["params"]["E"] for m in case_data["materials"]}
    fibers = _discretize_circ_fibers(case_data["section"]["params"])
    eps0, ky, kz = -1.0e-4, 8.0e-4, -1.2e-3
    n = my = mz = k11 = k12 = k13 = k22 = k23 = k33 = 0.0
    for y, z, area, mat in fibers:
        e = e_by_mat[mat]
        strain = eps0 + z * ky - y * kz
        fs = e * strain * area
        ks = e * area
        n += fs
        my += fs * z
        mz += -fs * y
        k11 += ks
        k12 += ks * z
        k13 += -ks * y
        k22 += ks * z * z
        k23 += -ks * z * y
        k33 += ks * y * y

    assert math.isclose(got["N"], n, rel_tol=1e-10, abs_tol=1e-8)
    assert math.isclose(got["My"], my, rel_tol=1e-10, abs_tol=1e-8)
    assert math.isclose(got["Mz"], mz, rel_tol=1e-10, abs_tol=1e-8)
    assert math.isclose(got["k11"], k11, rel_tol=1e-10, abs_tol=1e-8)
    assert math.isclose(got["k12"], k12, rel_tol=1e-8, abs_tol=1e-6)
    assert math.isclose(got["k13"], k13, rel_tol=1e-8, abs_tol=1e-6)
    assert math.isclose(got["k22"], k22, rel_tol=1e-10, abs_tol=1e-8)
    assert math.isclose(got["k23"], k23, rel_tol=1e-8, abs_tol=1e-6)
    assert math.isclose(got["k33"], k33, rel_tol=1e-10, abs_tol=1e-8)
