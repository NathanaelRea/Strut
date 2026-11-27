import subprocess
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
MATERIALS = {
    "ModIMKPeakOriented": (
        "ModIMKPeakOriented 1 15000.0 0.01 0.02 17.0 -14.0 0.3 0.4 0.5 0.6 "
        "1.0 1.0 1.0 1.0 0.006 0.008 0.02 0.03 0.2 0.3 0.04 0.05 1.0 0.9 2.0"
    ),
    "ConcreteCM": "ConcreteCM 1 -5.6 -0.002 3900.0 7.0 1.1 0.5 0.00013 1.2 2.0",
    "ConcreteCM-gap": "ConcreteCM 1 -5.6 -0.002 3900.0 7.0 1.1 0.5 0.00013 1.2 2.0 -GapClose 1",
    "SteelMPF": "SteelMPF 1 71.0 64.0 29000.0 0.012 0.017 20.0 0.925 0.15 0.03 1.2 0.02 1.3",
}
STRAINS = (
    0.0,
    -0.0001,
    -0.0015,
    -0.0025,
    -0.0013,
    -0.0008,
    -0.001,
    0.0,
    0.0001,
    0.0003,
    0.001,
    0.004,
    0.009,
    0.002,
    -0.003,
    -0.006,
    -0.001,
    0.002,
    0.001,
    0.0009,
    0.0015,
    -0.0033,
    -0.003,
    -0.002,
    -0.004,
    0.0,
    0.0001,
    0.012,
    -0.01,
)


@pytest.fixture(scope="module")
def material_engines(tmp_path_factory):
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
    driver = tmp_path_factory.mktemp("wall_materials") / "wall_material_path"
    subprocess.run(
        [
            "uv",
            "run",
            "mojo",
            "build",
            "-I",
            str(ROOT / "src/mojo"),
            str(ROOT / "tests/mojo/wall_material_path.mojo"),
            "-o",
            str(driver),
        ],
        cwd=ROOT,
        check=True,
    )
    return opensees, driver


def _response(command):
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    rows = [
        [float(value) for value in line.split()[1:]]
        for line in (result.stdout + result.stderr).splitlines()
        if line.startswith("ROW ")
    ]
    assert len(rows) == len(STRAINS), result.stdout + result.stderr
    return np.asarray(rows)


@pytest.mark.parametrize("material", MATERIALS)
@pytest.mark.parametrize("commit_interval", (1, 2))
def test_wall_material_cyclic_path(
    material_engines, tmp_path, material, commit_interval
):
    opensees, driver = material_engines
    oracle = tmp_path / "oracle.tcl"
    commands = [
        f"model UniaxialMaterialTest {commit_interval}",
        f"uniaxialMaterial {MATERIALS[material]}",
        "uniaxialTest 1",
    ]
    for strain in STRAINS:
        commands.extend(
            [
                f"strainUniaxialTest {strain}",
                'puts "ROW [stressUniaxialTest] [tangUniaxialTest]"',
            ]
        )
    oracle.write_text("\n".join(commands + ["exit", ""]))
    expected = _response([opensees, oracle])
    observed = _response([driver, material, str(commit_interval), *map(str, STRAINS)])
    np.testing.assert_allclose(observed, expected, rtol=2.0e-10, atol=1.0e-8)


@pytest.mark.parametrize(
    "source",
    (
        "mvlem_response",
        "sfi_mvlem_response",
        "catenary_cable_response",
        "hs_constraint",
    ),
)
def test_specialised_response(source):
    subprocess.run(
        [
            "uv",
            "run",
            "mojo",
            "-I",
            str(ROOT / "src/mojo"),
            str(ROOT / "tests/mojo" / f"{source}.mojo"),
        ],
        cwd=ROOT,
        check=True,
    )
