import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from scripts.centrifuge_recon.verify import opensees_executable
from scripts.tcl_to_strut import convert_tcl_to_solver_input

ROOT = Path(__file__).resolve().parents[2]
STRAINS = (0, -0.003, 0, 0.0025, 0.015, -0.004, -0.004, 0.0002)


def read_rows(command: list, expected_rows: int) -> np.ndarray:
    result = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, check=True
    )
    rows = [
        [float(value) for value in line.split()[1:]]
        for line in (result.stdout + result.stderr).splitlines()
        if line.startswith("ROW ")
    ]
    assert len(rows) == expected_rows, result.stdout + result.stderr
    return np.asarray(rows)


def test_ent_and_weighted_parallel_material_paths(tmp_path: Path):
    opensees = opensees_executable()
    if not opensees.is_file():
        pytest.skip("OpenSees is unavailable. Run uv run scripts/setup_opensees.py.")
    oracle = tmp_path / "oracle.tcl"
    oracle.write_text(
        """model UniaxialMaterialTest 1
uniaxialMaterial ENT 1 1500
uniaxialMaterial ENT 2 1500 0.02 8
uniaxialMaterial Elastic 3 10
uniaxialMaterial Steel01 4 5 2000 0.02
uniaxialMaterial Parallel 5 1 3
uniaxialMaterial Parallel 6 5 2 4 4 -factors 1 0.75 1 -0.25
"""
        + "foreach tag {1 2 5 6} {\n"
        + "    uniaxialTest $tag\n"
        + "    foreach strain {"
        + " ".join(map(str, STRAINS))
        + "} {\n"
        + '        strainUniaxialTest $strain\n        puts "ROW [stressUniaxialTest] [tangUniaxialTest]"\n'
        + "    }\n}\n"
    )
    expected = read_rows([opensees, oracle], 4 * len(STRAINS))
    actual = read_rows(
        [
            "uv",
            "run",
            "mojo",
            "-I",
            "src/mojo",
            "tests/mojo/ent_parallel_path.mojo",
            *map(str, STRAINS),
        ],
        4 * len(STRAINS),
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize(("load", "displacement"), [(-1.51, -0.001), (0.02, 0.002)])
def test_parallel_zero_length_from_tcl(
    tmp_path: Path, load: float, displacement: float
):
    # The aggregate tag precedes its children when the parser sorts by tag.
    # The Tcl converter can import the spring material through truss. The
    # zeroLength assembly path takes the same materials in its JSON form.
    entry = tmp_path / "model.tcl"
    entry.write_text(
        f"""model BasicBuilder -ndm 2 -ndf 2
node 1 0 0
node 2 1 0
fix 1 1 1
fix 2 0 1
uniaxialMaterial ENT 8 1500
uniaxialMaterial Elastic 9 10
uniaxialMaterial Parallel 7 8 9
element truss 1 1 2 1 7
timeSeries Linear 1
pattern Plain 1 1 {{load 2 {load} 0}}
recorder Node -file displacement.out -node 2 -dof 1 disp
constraints Plain
numberer RCM
system ProfileSPD
algorithm Newton
test NormUnbalance 1e-9 10
integrator LoadControl 1
analysis Static
analyze 1
"""
    )
    solver_input = tmp_path / "input.json"
    data = convert_tcl_to_solver_input(entry, ROOT)
    assert data["materials"][0]["params"] == {"materials": [8, 9]}
    data["nodes"][1]["x"] = 0
    data["elements"] = [
        {"id": 1, "type": "zeroLength", "nodes": [1, 2], "materials": [7], "dirs": [1]}
    ]
    solver_input.write_text(json.dumps(data))
    roundtrip = tmp_path / "roundtrip.tcl"
    subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/json_to_tcl.py",
            str(solver_input),
            str(roundtrip),
        ],
        cwd=ROOT,
        check=True,
    )
    output = roundtrip.read_text()
    assert output.index("uniaxialMaterial ENT 8") < output.index(
        "uniaxialMaterial Parallel 7"
    )
    out_dir = tmp_path / "results"
    subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/run_strut_case.py",
            "--input",
            str(solver_input),
            "--output",
            str(out_dir),
        ],
        cwd=ROOT,
        check=True,
    )
    assert float(
        (out_dir / "displacement_node2.out").read_text().split()[-1]
    ) == pytest.approx(displacement)
