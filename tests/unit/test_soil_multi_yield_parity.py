import subprocess
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
STRAINS = [
    (0, -0.0001, 0),
    (0, -0.0003, 0),
    (-0.00005, -0.0005, 0.0002),
    (-0.0001, -0.0007, 0.001),
    (-0.0003, -0.0012, 0.006),
    (-0.0003, -0.0012, 0.018),
    (-0.0002, -0.001, 0.003),
    (-0.0001, -0.0014, -0.007),
    (-0.0001, -0.0016, -0.025),
    (-0.0002, -0.0014, 0.009),
]


@pytest.fixture(scope="module", params=["independ", "depend", "depend02"])
def soil_path_engines(tmp_path_factory, request):
    variant = request.param
    opensees = subprocess.check_output(
        [
            "bash",
            "-c",
            'source "$1"; strut_resolve_opensees_binary OpenSees "${OPENSEES_BIN:-}"',
            "bash",
            str(ROOT / "scripts/opensees_env.sh"),
        ],
        text=True,
    ).strip()
    if not Path(opensees).is_file():
        pytest.skip("OpenSees is unavailable. Run uv run scripts/setup_opensees.py.")
    driver = tmp_path_factory.mktemp("soil_path") / variant
    subprocess.run(
        [
            "uv",
            "run",
            "mojo",
            "build",
            "-I",
            str(ROOT / "src/mojo"),
            str(ROOT / f"tests/mojo/pressure_{variant}_multi_yield_path.mojo"),
            "-o",
            str(driver),
        ],
        check=True,
        cwd=ROOT,
    )
    return opensees, driver, variant


def _run_path(command, cwd):
    result = subprocess.run(
        command, cwd=cwd, capture_output=True, text=True, check=True
    )
    rows = [
        [float(value) for value in line.split()[1:]]
        for line in (result.stdout + result.stderr).splitlines()
        if line.startswith("ROW ")
    ]
    assert len(rows) == len(STRAINS), result.stdout + result.stderr
    assert all(len(row) == 14 for row in rows)
    return np.asarray(rows)


@pytest.mark.parametrize("option", [False, True])
def test_soil_multi_yield_path(soil_path_engines, tmp_path, option):
    opensees, driver, variant = soil_path_engines
    if variant == "independ":
        friction_angle = 28 if option else 0
        material = f"PressureIndependMultiYield 1 2 1.8 50000 100000 25 0.1 {friction_angle} 100 0.5 20"
        args = [str(friction_angle)]
    elif variant == "depend02":
        liquefaction = 1.0 if option else 0.0
        material = (
            "PressureDependMultiYield02 1 2 1.8 50000 100000 35 0.1 100 0.5 27 "
            f"0.06 0.23 0.5 0.27 20 5.0 3.0 {liquefaction} 0.7"
        )
        args = [str(liquefaction)]
    else:
        exponent, liquefaction_pressure, bulk = (
            (0.5, 150, 100000.0) if option else (0.0, 0, 100000.0 / 3.0)
        )
        material = (
            f"PressureDependMultiYield 1 2 1.8 50000 {bulk} 35 0.1 100 {exponent} "
            f"27 0.06 0.5 2.5 {liquefaction_pressure} 0.02 1.0 20"
        )
        args = list(map(str, (exponent, liquefaction_pressure, bulk)))
    # An affine displacement field subjects all four integration points to the
    # same plane strain history. A separate truss supplies one free equation.
    oracle = tmp_path / "soil.tcl"
    script = [
        "model BasicBuilder -ndm 2 -ndf 2",
        "node 1 0 0",
        "node 2 1 0",
        "node 3 1 1",
        "node 4 0 1",
        "node 5 2 0",
        "node 6 3 0",
        "fix 1 1 1",
        "fix 2 0 1",
        "fix 5 1 1",
        "fix 6 0 1",
        f"nDMaterial {material}",
        "element quad 1 1 2 3 4 1 PlaneStrain 1",
        "uniaxialMaterial Elastic 1 1",
        "element truss 2 5 6 1 1",
    ]
    histories = [
        (2, 1, [ex for ex, _, _ in STRAINS]),
        (3, 1, [ex + shear for ex, _, shear in STRAINS]),
        (3, 2, [ey for _, ey, _ in STRAINS]),
        (4, 1, [shear for _, _, shear in STRAINS]),
        (4, 2, [ey for _, ey, _ in STRAINS]),
    ]
    for tag, (node, dof, values) in enumerate(histories, 1):
        samples = " ".join(map(str, [0.0, *values]))
        script.extend(
            [
                f"timeSeries Path {tag} -dt 1 -values {{{samples}}} -useLast",
                f"pattern Plain {tag} {tag} {{sp {node} {dof} 1}}",
            ]
        )
    script.extend(
        [
            "constraints Transformation",
            "numberer Plain",
            "system BandGeneral",
            "test NormUnbalance 1e-12 5",
            "algorithm Newton",
            "integrator LoadControl 1",
            "analysis Static",
        ]
    )
    previous_stage = 0
    for i, strain in enumerate(STRAINS):
        stage = int(i >= 2)
        if variant == "depend" and i == 1:
            stage = 2
        if stage != previous_stage:
            script.append(f"updateMaterialStage -material 1 -stage {stage}")
        previous_stage = stage
        script.extend(
            [
                'if {[analyze 1] != 0} {error "strain step failed"}',
                'puts "ROW [eleResponse 1 material 1 stress] [eleResponse 1 material 1 tangent]"',
            ]
        )
        args.extend(map(str, (stage, *strain)))
    oracle.write_text("\n".join([*script, "exit", ""]))

    expected = _run_path([opensees, oracle], tmp_path)
    observed = _run_path([driver, *args], tmp_path)
    np.testing.assert_allclose(observed, expected, rtol=1e-9, atol=1e-8)
