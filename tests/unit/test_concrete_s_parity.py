import subprocess
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
PATHS = {
    "tension": [
        (x, -0.2 * x, 0)
        for x in (0, 4e-5, 8e-5, 2e-4, 8e-4, 3e-4, 0, -2e-4, 5e-4, 1e-3)
    ],
    "compression": [
        (x, -0.2 * x, 0) for x in (0, -5e-4, -1.2e-3, -4e-3, -2e-3, -4e-4, -2e-3, -5e-3)
    ],
    "shear": [(0, 0, xy) for xy in (0, 1e-4, 3e-4, 1e-3, 0, -3e-4, -2e-3, 3e-4)],
    "mixed": [
        (0, 0, 0),
        (3e-4, 4e-4, 2e-4),
        (2e-5, 3e-5, 0),
        (-1e-3, -8e-4, 2e-3),
        (-2e-3, -2.7e-3, -1.6e-3),
        (-2e-4, 8e-4, -3e-4),
        (0, 0, 0),
        (9e-4, -3e-3, 3e-3),
    ],
}


@pytest.fixture(scope="module")
def concrete_s_engines(tmp_path_factory):
    engine_dir = tmp_path_factory.mktemp("concrete_s")
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
    driver = engine_dir / "concrete_s_path"
    subprocess.run(
        [
            "uv",
            "run",
            "mojo",
            "build",
            "-I",
            str(ROOT / "src/mojo"),
            str(ROOT / "tests/mojo/concrete_s_path.mojo"),
            "-o",
            str(driver),
        ],
        check=True,
        cwd=ROOT,
    )
    return opensees, driver


def _response(command, npoints):
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    rows = [
        [float(value) for value in line.split()[1:]]
        for line in (result.stdout + result.stderr).splitlines()
        if line.startswith("ROW ")
    ]
    assert len(rows) == npoints, result.stdout + result.stderr
    assert all(len(row) == 12 for row in rows)
    return np.asarray(rows)


@pytest.mark.parametrize(
    "path_name,commit_interval",
    [(name, 1) for name in PATHS] + [("mixed", 2)],
)
def test_concrete_s_material_matches_opensees(
    concrete_s_engines, tmp_path, path_name, commit_interval
):
    opensees, driver = concrete_s_engines
    strains = PATHS[path_name]
    oracle = tmp_path / "oracle.tcl"
    commands = [
        f"model PlaneStressMaterialTest {commit_interval}",
        "nDMaterial ConcreteS 1 36e9 0.2 -38.5e6 2.85e6 -3.6e9",
        "setMaterial 1",
    ]
    for point in strains:
        commands += [
            "setStrain " + " ".join(map(str, point)),
            'puts "ROW [getStress] [getTangent]"',
        ]
    oracle.write_text("\n".join(commands + ["exit", ""]))
    expected = _response([opensees, oracle], len(strains))
    observed = _response(
        [driver, str(commit_interval), *(str(v) for point in strains for v in point)],
        len(strains),
    )
    np.testing.assert_allclose(observed, expected, rtol=2e-10, atol=1e-6)
