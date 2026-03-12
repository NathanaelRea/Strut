from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_MATERIAL_COMPARE_PATH = REPO_ROOT / "benchmark" / "micro" / "run_material_compare.py"
RUN_MATERIAL_COMPARE_DIR = RUN_MATERIAL_COMPARE_PATH.parent

if str(RUN_MATERIAL_COMPARE_DIR) not in sys.path:
    sys.path.insert(0, str(RUN_MATERIAL_COMPARE_DIR))


def _load_module():
    module_name = "strut_run_material_compare_test_module"
    spec = importlib.util.spec_from_file_location(module_name, RUN_MATERIAL_COMPARE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_material_compare = _load_module()


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
                    {"id": 10, "type": "Steel01", "params": {"Fy": 50.0, "E0": 200.0, "b": 0.01}},
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
                    {"id": 20, "type": "Concrete01", "params": {"fpc": -6.0, "epsc0": -0.002, "fpcu": -5.0, "epscu": -0.006}},
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
        lambda raw: {"base": base_case, "steel": steel_case, "concrete": concrete_case}[raw],
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
