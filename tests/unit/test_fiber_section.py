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
