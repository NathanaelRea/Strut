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
                str(repo_root / "scripts" / "run_mojo_section_path.py"),
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
        assert patch["type"] == "rect"
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
                fibers.append((ys + (ye - ys) * t, zs + (ze - zs) * t, area, layer["material"]))
    return fibers


def _expected_elastic_response(case_data, eps0, kappa):
    e_by_mat = {
        m["id"]: m["params"]["E"] for m in case_data["materials"] if m["type"] == "Elastic"
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
