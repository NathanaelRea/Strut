import os
import subprocess
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def material_engines(tmp_path_factory):
    opensees = os.environ.get("OPENSEES_BIN")
    if not opensees:
        result = subprocess.run(
            ["uv", "run", "scripts/setup_opensees.py", "path"],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        if result.returncode:
            pytest.skip("The pinned OpenSees binary is unavailable.")
        opensees = result.stdout.strip()
    driver = tmp_path_factory.mktemp("uniaxial_path") / "damper_fatigue_path"
    subprocess.run(
        [
            "uv",
            "run",
            "mojo",
            "build",
            "-I",
            "src/mojo",
            "tests/mojo/damper_fatigue_path.mojo",
            "-o",
            driver,
        ],
        check=True,
        cwd=ROOT,
    )
    return opensees, driver


def _response(cmd, npoints):
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    rows = [
        [float(word) for word in line.split()[1:]]
        for line in (result.stdout + result.stderr).splitlines()
        if line.startswith("ROW ")
    ]
    assert len(rows) == npoints, result.stdout + result.stderr
    return np.asarray(rows)


@pytest.mark.parametrize("method,gap", [(1, 0.02), (2, 0.0), (3, 0.0)])
def test_viscous_damper_material_matches_opensees(
    material_engines, tmp_path, method, gap
):
    opensees, driver = material_engines
    oracle = tmp_path / "damper.tcl"
    oracle.write_text(
        f"""model basic -ndm 2 -ndf 2
node 1 0 0
node 2 1 0
fix 1 1 1
fix 2 0 1
mass 2 1 0
uniaxialMaterial Elastic 1 200
uniaxialMaterial ViscousDamper 2 120 8 0.55 {gap} {method} 1e-6 1e-10 15
element truss 1 1 2 1 1
element twoNodeLink 2 1 2 -mat 2 -dir 1
timeSeries Path 1 -dt 0.05 -values {{0 4 7 3 -8 -7 -4 0 6 7 3 -8 -7 -4 0 0}}
pattern Plain 1 1 {{load 2 1 0}}
constraints Plain
numberer Plain
system BandGeneral
test NormDispIncr 1e-12 100
algorithm Newton
integrator Newmark 0.5 0.25
analysis Transient
for {{set i 0}} {{$i < 15}} {{incr i}} {{
    if {{[analyze 1 0.05]}} {{error "analysis failed"}}
    puts "ROW [nodeDisp 2 1] [nodeVel 2 1] [eleResponse 2 basicForce]"
}}
exit
"""
    )
    expected = _response([opensees, oracle], 15)
    observed = _response(
        [driver, "damper", str(gap), str(method), *map(str, expected[:, :2].flat)], 15
    )
    np.testing.assert_allclose(observed[:, 0], expected[:, 2], rtol=1e-8, atol=1e-10)
    np.testing.assert_array_equal(observed[:, 1], np.zeros(15))


@pytest.mark.parametrize(
    "dmax,strain_max,strains",
    [
        (1.0, 0.04, [0.0, 0.01, -0.01, 0.04, 0.02, -0.015]),
        (0.25, 1.0, [0.0, 0.035, -0.025, 0.005, -0.02, 0.03, -0.035, 0.025, -0.03]),
    ],
)
def test_fatigue_material_matches_opensees(
    material_engines, tmp_path, dmax, strain_max, strains
):
    opensees, driver = material_engines
    oracle = tmp_path / "fatigue.tcl"
    oracle.write_text(
        f"""model UniaxialMaterialTest 1
uniaxialMaterial Steel02 1 50 29000 0.003
uniaxialMaterial Fatigue 2 1 -Dmax {dmax} -E0 0.095 -m -0.5 -min -1 -max {strain_max}
uniaxialTest 2
foreach strain {{{" ".join(map(str, strains))}}} {{
    strainUniaxialTest $strain
    puts "ROW [stressUniaxialTest] [tangUniaxialTest]"
}}
exit
"""
    )
    expected = _response([opensees, oracle], len(strains))
    observed = _response(
        [
            driver,
            "fatigue",
            str(dmax),
            str(strain_max),
            *(word for strain in strains for word in (str(strain), "0")),
        ],
        len(strains),
    )
    assert expected[-1, 1] == pytest.approx(29000 * 1e-8)
    np.testing.assert_allclose(observed, expected, rtol=1e-8, atol=1e-10)
