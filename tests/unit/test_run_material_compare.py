from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_MATERIAL_COMPARE_PATH = (
    REPO_ROOT / "benchmark" / "micro" / "run_material_compare.py"
)
RUN_MATERIAL_COMPARE_DIR = RUN_MATERIAL_COMPARE_PATH.parent

if str(RUN_MATERIAL_COMPARE_DIR) not in sys.path:
    sys.path.insert(0, str(RUN_MATERIAL_COMPARE_DIR))


def _load_module():
    module_name = "strut_run_material_compare_test_module"
    spec = importlib.util.spec_from_file_location(
        module_name, RUN_MATERIAL_COMPARE_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_material_compare = _load_module()

HINGE_PATHS = {
    "ElasticPP": [1000, 0.0105, -0.025, 0.002],
    "ElasticBilin": [1000, 25, 0.023, 300, 40, -0.025],
    "MinMax": [1000, -0.065, 0.060],
    "Ratchet": [1000, 0.015, 0.006, 1],
    "RatchetContinuous": [1000, 0.015, 0.006, 2],
    "Bilin": [
        1000,
        0.035,
        0.025,
        12,
        -10,
        0.6,
        0.8,
        0.9,
        1.2,
        1.1,
        0.9,
        1,
        1.2,
        0.025,
        0.020,
        0.08,
        0.06,
        0.2,
        0.15,
        0.15,
        0.12,
        0.9,
        0.8,
        2,
    ],
    "Hysteretic": [
        12,
        0.01,
        16,
        0.04,
        4,
        0.08,
        -10,
        -0.008,
        -13,
        -0.03,
        -3,
        -0.09,
        0.55,
        0.35,
        0.018,
        0.045,
        0.4,
    ],
}


@pytest.fixture(scope="module")
def hinge_engines(tmp_path_factory):
    opensees = subprocess.check_output(
        [
            "bash",
            "-c",
            'source "$1"; strut_resolve_opensees_binary OpenSees "${OPENSEES_BIN:-}"',
            "bash",
            str(REPO_ROOT / "scripts/opensees_env.sh"),
        ],
        text=True,
    ).strip()
    if not Path(opensees).is_file():
        pytest.skip("OpenSees is unavailable. Run uv run scripts/setup_opensees.py.")
    driver = tmp_path_factory.mktemp("uniaxial_hinges") / "path"
    subprocess.run(
        [
            "uv",
            "run",
            "mojo",
            "build",
            "-I",
            "src/mojo",
            "tests/mojo/uniaxial_hinge_path.mojo",
            "-o",
            str(driver),
        ],
        check=True,
        cwd=REPO_ROOT,
    )
    return opensees, driver


@pytest.mark.parametrize("material", HINGE_PATHS)
@pytest.mark.parametrize("commit_interval", [1, 2])
def test_hinge_stress_and_tangent_match_opensees(
    hinge_engines, tmp_path, material, commit_interval
):
    opensees, driver = hinge_engines
    parameters = list(map(str, HINGE_PATHS[material]))
    strains = list(
        map(
            str,
            [
                0,
                0.003,
                -0.006,
                0.0105,
                0.0115,
                0.023,
                0.045,
                0.060,
                0.027,
                0.014,
                0.002,
                -0.004,
                -0.025,
                -0.041,
                -0.065,
                -0.008,
                0.007,
                0.04,
                0.09,
                0.031,
                -0.018,
                -0.1,
                0.002,
                0.038,
                0.17,
                0.02,
                -0.18,
            ],
        )
    )
    script = tmp_path / "material.tcl"
    if material == "MinMax":
        stiffness, minimum, maximum = parameters
        definitions = [
            f"uniaxialMaterial Elastic 2 {stiffness}",
            f"uniaxialMaterial MinMax 1 2 -min {minimum} -max {maximum}",
        ]
    else:
        opensees_material = "Ratchet" if material == "RatchetContinuous" else material
        definitions = [f"uniaxialMaterial {opensees_material} 1 {' '.join(parameters)}"]
    commands = [
        f"model testUniaxial {commit_interval}",
        *definitions,
        "uniaxialTest 1",
        f"foreach strain {{{' '.join(strains)}}} {{",
        "    strainUniaxialTest $strain",
        '    puts "ROW [stressUniaxialTest] [tangUniaxialTest]"',
        "}",
    ]
    script.write_text("\n".join(commands) + "\n", encoding="utf-8")

    expected, observed = [
        [
            tuple(map(float, line.split()[1:]))
            for line in subprocess.check_output(
                command, text=True, stderr=subprocess.STDOUT
            ).splitlines()
            if line.startswith("ROW ")
        ]
        for command in (
            [opensees, str(script)],
            [
                str(driver),
                material,
                str(commit_interval),
                str(len(parameters)),
                *parameters,
                *strains,
            ],
        )
    ]
    assert len(expected) == len(observed) == len(strains)
    for strain, actual, reference in zip(strains, observed, expected):
        assert actual == pytest.approx(reference, rel=2e-10, abs=1e-10), (
            f"strain {strain}"
        )


def test_implementations_include_all_supported_uniaxial_materials():
    assert set(run_material_compare.IMPLEMENTATIONS) == {
        "elastic_uniaxial",
        "elastic_fiber_scalar",
        "steel01_uniaxial",
        "steel01_fiber_scalar",
        "concrete01_uniaxial",
        "concrete01_fiber_scalar",
        "steel02_uniaxial",
        "steel02_fiber_scalar",
        "concrete02_uniaxial",
        "concrete02_fiber_scalar",
    }


def test_build_default_material_case_merges_and_deduplicates_materials(
    monkeypatch, tmp_path: Path
):
    base_case = tmp_path / "base.json"
    base_case.write_text(
        json.dumps(
            {
                "metadata": {"name": "base"},
                "materials": [],
                "nodes": [],
                "elements": [],
                "constraints": [],
                "loads": [],
                "recorders": [],
                "analysis": {"type": "static_linear"},
            }
        ),
        encoding="utf-8",
    )
    steel_case = tmp_path / "steel.json"
    steel_case.write_text(
        json.dumps(
            {
                "materials": [
                    {
                        "id": 10,
                        "type": "Steel01",
                        "params": {"Fy": 50.0, "E0": 200.0, "b": 0.01},
                    },
                    {"id": 11, "type": "Elastic", "params": {"E": 300.0}},
                ]
            }
        ),
        encoding="utf-8",
    )
    concrete_case = tmp_path / "concrete.json"
    concrete_case.write_text(
        json.dumps(
            {
                "materials": [
                    {
                        "id": 20,
                        "type": "Concrete01",
                        "params": {
                            "fpc": -6.0,
                            "epsc0": -0.002,
                            "fpcu": -5.0,
                            "epscu": -0.006,
                        },
                    },
                    {"id": 21, "type": "Elastic", "params": {"E": 300.0}},
                ]
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        run_material_compare,
        "DEFAULT_MATERIAL_COVERAGE_CASES",
        ("base", "steel", "concrete"),
    )
    monkeypatch.setattr(
        run_material_compare,
        "resolve_case_input",
        lambda raw: {"base": base_case, "steel": steel_case, "concrete": concrete_case}[
            raw
        ],
    )
    monkeypatch.setattr(
        run_material_compare,
        "prepare_case_json",
        lambda path: (path, []),
    )

    case_json, temp_roots = run_material_compare._build_default_material_case()
    try:
        data = json.loads(case_json.read_text(encoding="utf-8"))
    finally:
        for root in temp_roots:
            shutil.rmtree(root, ignore_errors=True)

    assert [material["type"] for material in data["materials"]] == [
        "Steel01",
        "Elastic",
        "Concrete01",
    ]
    assert [material["id"] for material in data["materials"]] == [1, 2, 3]
