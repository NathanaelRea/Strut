from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_SECTION_MICRO_PATH = REPO_ROOT / "benchmark" / "micro" / "run_section_micro.py"
RUN_SECTION_MICRO_DIR = RUN_SECTION_MICRO_PATH.parent

if str(RUN_SECTION_MICRO_DIR) not in sys.path:
    sys.path.insert(0, str(RUN_SECTION_MICRO_DIR))


def _load_module():
    module_name = "strut_run_section_micro_test_module"
    spec = importlib.util.spec_from_file_location(module_name, RUN_SECTION_MICRO_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_section_micro = _load_module()


def test_build_material_sweep_case_creates_one_section_per_material():
    case_json, temp_roots, section_labels = run_section_micro._build_material_sweep_case(
        ["elastic", "steel01", "concrete01"],
        num_subdiv_y=8,
        num_subdiv_z=2,
    )
    try:
        data = json.loads(case_json.read_text(encoding="utf-8"))
    finally:
        for root in temp_roots:
            shutil.rmtree(root, ignore_errors=True)

    assert [material["type"] for material in data["materials"]] == [
        "Elastic",
        "Steel01",
        "Concrete01",
    ]
    assert [section["id"] for section in data["sections"]] == [1, 2, 3]
    assert section_labels == {
        1: "Elastic",
        2: "Steel01",
        3: "Concrete01",
    }
    assert all(
        section["params"]["patches"][0]["material"] == section["id"] for section in data["sections"]
    )


def test_build_table_pivots_scenarios_per_material():
    summary = [
        {
            "section_id": 1,
            "label": "Elastic",
            "scenario": "high_push",
            "scenario_label": "High Push",
            "fiber_count": 64,
            "instances": 100,
            "samples": 3,
            "best_ns_per_update": 20.0,
        },
        {
            "section_id": 1,
            "label": "Elastic",
            "scenario": "post_crushing_rebound",
            "scenario_label": "Crushing Rebound",
            "fiber_count": 64,
            "instances": 100,
            "samples": 3,
            "best_ns_per_update": 22.0,
        },
        {
            "section_id": 2,
            "label": "Steel01",
            "scenario": "high_push",
            "scenario_label": "High Push",
            "fiber_count": 64,
            "instances": 100,
            "samples": 3,
            "best_ns_per_update": 30.0,
        },
    ]

    table = run_section_micro._build_table(summary)

    assert "Material" in table
    assert "High Push (ns/fiber)" in table
    assert "Crushing Rebound (ns/fiber)" in table
    assert "Elastic" in table
    assert "Steel01" in table
    assert "20.00" in table
    assert "30.00" in table
