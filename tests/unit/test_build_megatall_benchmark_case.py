import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = str(REPO_ROOT / "scripts")
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_megatall_benchmark_case.py"


def _load_module():
    if SCRIPTS_DIR not in sys.path:
        sys.path.insert(0, SCRIPTS_DIR)
    spec = importlib.util.spec_from_file_location(
        "strut_build_megatall_benchmark_case_test_module", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = _load_module()


def test_elastic_surrogates_replace_plates_over_user_materials_only():
    materials = [
        {"id": 1, "type": "Steel02", "params": {"Fy": 4.0e8}},
        {
            "id": 7,
            "type": "PlaneStressUserMaterial",
            "params": {
                "nstatevs": 40,
                "nprops": 7,
                "props": [1.0, 2.0, 0.2],
                "rho": 2.5,
            },
        },
        {
            "id": 8,
            "type": "PlateFromPlaneStress",
            "params": {"material": 7, "gmod": 1.0e10},
        },
        {"id": 9, "type": "ElasticIsotropic", "params": {"E": 3.0e10, "nu": 0.2}},
        {
            "id": 10,
            "type": "PlateFromPlaneStress",
            "params": {"material": 9, "gmod": 1.25e10},
        },
    ]

    result = builder._elastic_surrogates(materials)

    assert [material["id"] for material in result] == [1, 8, 9, 10]
    assert result[1] == {
        "id": 8,
        "type": "ElasticIsotropic",
        "params": {"E": 2.0 * 1.0e10 * 1.2, "nu": 0.2, "rho": 2.5},
    }
    assert result[0] is materials[0]
    assert result[3] is materials[4]
