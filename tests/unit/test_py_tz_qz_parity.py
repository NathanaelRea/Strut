import subprocess
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]

PY_STRAINS = [
    0.0,
    0.002,
    0.005,
    0.015,
    0.04,
    0.08,
    0.03,
    0.0,
    -0.005,
    -0.02,
    -0.06,
    -0.01,
    0.02,
    0.10,
    -0.08,
    0.0,
]

TZ_STRAINS = [
    0.0,
    0.001,
    0.005,
    0.02,
    0.05,
    0.01,
    -0.005,
    -0.03,
    -0.06,
    -0.01,
    0.04,
    0.0,
]

QZ_STRAINS = [
    0.0,
    0.002,
    0.01,
    0.05,
    0.15,
    0.08,
    0.02,
    0.0,
    -0.01,
    -0.05,
    -0.02,
    0.01,
    0.06,
    0.20,
    -0.03,
    0.10,
]

CASES = [
    ("PySimple1", "py_clay_drag01", [1, 100.0, 0.01, 0.1, 0.0], PY_STRAINS, 1),
    ("PySimple1", "py_clay_drag03", [1, 120.0, 0.015, 0.3, 0.0], PY_STRAINS, 1),
    ("PySimple1", "py_sand_drag00", [2, 250.0, 0.008, 0.0, 0.0], PY_STRAINS, 1),
    ("PySimple1", "py_sand_drag02", [2, 250.0, 0.008, 0.2, 0.0], PY_STRAINS, 1),
    ("PySimple1", "py_clay_commit2", [1, 100.0, 0.01, 0.1, 0.0], PY_STRAINS, 2),
    ("TzSimple1", "tz_clay", [1, 80.0, 0.005, 0.0], TZ_STRAINS, 1),
    ("TzSimple1", "tz_sand", [2, 150.0, 0.01, 0.0], TZ_STRAINS, 1),
    ("TzSimple1", "tz_sand_commit2", [2, 150.0, 0.01, 0.0], TZ_STRAINS, 2),
    ("QzSimple1", "qz_clay_suc00", [1, 500.0, 0.02, 0.0, 0.0], QZ_STRAINS, 1),
    ("QzSimple1", "qz_clay_suc05", [1, 500.0, 0.02, 0.05, 0.0], QZ_STRAINS, 1),
    ("QzSimple1", "qz_sand_suc00", [2, 800.0, 0.025, 0.0, 0.0], QZ_STRAINS, 1),
    ("QzSimple1", "qz_sand_suc08", [2, 800.0, 0.025, 0.08, 0.0], QZ_STRAINS, 1),
    ("QzSimple1", "qz_sand_commit2", [2, 800.0, 0.025, 0.05, 0.0], QZ_STRAINS, 2),
]


@pytest.fixture(scope="module")
def py_tz_qz_engines(tmp_path_factory):
    engine_dir = tmp_path_factory.mktemp("py_tz_qz")
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
    driver = engine_dir / "py_tz_qz_path"
    subprocess.run(
        [
            "uv",
            "run",
            "mojo",
            "build",
            "-I",
            str(ROOT / "src/mojo"),
            str(ROOT / "tests/mojo/py_tz_qz_path.mojo"),
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
    assert all(len(row) == 2 for row in rows)
    return np.asarray(rows)


@pytest.mark.parametrize(
    "mat_name,case_id,params,strains,commit_interval",
    CASES,
    ids=[c[1] for c in CASES],
)
def test_py_tz_qz_material_matches_opensees(
    py_tz_qz_engines, tmp_path, mat_name, case_id, params, strains, commit_interval
):
    opensees, driver = py_tz_qz_engines
    oracle = tmp_path / f"{case_id}.tcl"
    param_str = " ".join(str(v) for v in params)
    commands = [
        f"model testUniaxial {commit_interval}",
        f"uniaxialMaterial {mat_name} 1 {param_str}",
        "uniaxialTest 1",
    ]
    for eps in strains:
        commands += [
            f"strainUniaxialTest {eps}",
            'puts "ROW [stressUniaxialTest] [tangUniaxialTest]"',
        ]
    oracle.write_text("\n".join(commands + ["exit", ""]))
    expected = _response([opensees, str(oracle)], len(strains))

    padded_params = list(params) + [0.0] * (5 - len(params))
    observed = _response(
        [
            str(driver),
            mat_name,
            str(commit_interval),
            *(str(v) for v in padded_params),
            *(str(v) for v in strains),
        ],
        len(strains),
    )
    np.testing.assert_allclose(observed, expected, rtol=1e-8, atol=1e-7)


@pytest.mark.parametrize(
    "mat_name,params,committed_prefix,rejected_strain,final_strain",
    [
        ("PySimple1", [1, 100.0, 0.01, 0.1, 0.0], [0.005, 0.02], 0.08, 0.015),
        ("TzSimple1", [2, 150.0, 0.01, 0.0, 0.0], [0.004, 0.015], -0.04, 0.01),
        ("QzSimple1", [2, 800.0, 0.025, 0.05, 0.0], [0.01, 0.04], -0.05, 0.06),
    ],
)
def test_py_tz_qz_revert_trial_restores_committed_state(
    py_tz_qz_engines, mat_name, params, committed_prefix, rejected_strain, final_strain
):
    _, driver = py_tz_qz_engines
    # Commit the prefix, then revert one uncommitted strain increment.
    prefix = [
        str(driver),
        mat_name,
        str(len(committed_prefix)),
        *(str(v) for v in params),
        *(str(v) for v in committed_prefix),
    ]
    clean = _response([*prefix, str(final_strain)], len(committed_prefix) + 1)
    with_revert = _response(
        [*prefix, str(rejected_strain), "R", str(final_strain)],
        len(committed_prefix) + 2,
    )
    np.testing.assert_allclose(with_revert[-1], clean[-1], rtol=1e-12, atol=1e-12)
