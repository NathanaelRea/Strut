import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CASE_DIR = (
    REPO_ROOT / "tests" / "validation" / "opensees_megatall_building_model1_dynamiccpu"
)
BUILD_SCRIPT = REPO_ROOT / "scripts" / "build_megatall_smoke_case.py"
CHECKED_IN_SMOKE_CASE = CASE_DIR / "megatall_smoke.json"
RUN_STRUT_CASE = REPO_ROOT / "scripts" / "run_strut_case.py"


def test_megatall_smoke_case_builds_and_runs_natively(tmp_path: Path):
    generated_smoke_case = tmp_path / "smoke_case.json"
    subprocess.check_call(
        [sys.executable, str(BUILD_SCRIPT), "--output", str(generated_smoke_case)],
        cwd=REPO_ROOT,
    )

    built_data = json.loads(generated_smoke_case.read_text(encoding="utf-8"))
    checked_in_data = json.loads(CHECKED_IN_SMOKE_CASE.read_text(encoding="utf-8"))
    assert built_data == checked_in_data

    assert built_data["analysis"]["type"] == "staged"
    assert len(built_data["analysis"]["stages"]) == 2
    assert built_data["analysis"]["stages"][0]["analysis"]["type"] == "static_nonlinear"
    assert built_data["analysis"]["stages"][0]["analysis"]["steps"] == 2
    assert built_data["analysis"]["stages"][1]["analysis"]["type"] == "transient_nonlinear"
    assert built_data["analysis"]["stages"][1]["analysis"]["steps"] == 5
    assert {element["type"] for element in built_data["elements"]} == {
        "elasticBeamColumn3d",
        "shell",
        "truss",
    }
    assert len(built_data["mp_constraints"]) == 1
    assert {recorder["type"] for recorder in built_data["recorders"]} == {
        "node_reaction",
        "node_displacement",
        "drift",
    }
    assert built_data["analysis"]["stages"][1]["pattern"]["type"] == "UniformExcitation"
    assert built_data["analysis"]["stages"][1]["pattern"]["direction"] == 3

    out_dir = tmp_path / "out"
    subprocess.check_call(
        [
            sys.executable,
            str(RUN_STRUT_CASE),
            "--input",
            str(generated_smoke_case),
            "--output",
            str(out_dir),
        ],
        cwd=REPO_ROOT,
    )
    output_names = {path.name for path in out_dir.iterdir()}
    assert any(name.startswith("base_reaction") for name in output_names)
    assert any(name.startswith("tip_disp") for name in output_names)
    assert any("tip_drift" in name for name in output_names)
