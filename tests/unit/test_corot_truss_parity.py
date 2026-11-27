import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]


def test_3d_corot_truss_matches_opensees(tmp_path):
    opensees = os.environ.get("OPENSEES_BIN")
    if not opensees:
        result = subprocess.run(
            [sys.executable, ROOT / "scripts/setup_opensees.py", "path"],
            capture_output=True,
            text=True,
        )
        if result.returncode:
            pytest.skip("The pinned OpenSees binary is unavailable.")
        opensees = result.stdout.strip()
    script = tmp_path / "truss.tcl"
    script.write_text(
        """model basic -ndm 3 -ndf 6
node 1 0 0 0
node 2 3 4 1
fix 1 1 1 1 1 1 1
fix 2 0 1 1 1 1 1
uniaxialMaterial Elastic 1 100
element corotTruss 1 1 2 2 1
timeSeries Linear 1
pattern Plain 1 1 {load 2 75 0 0 0 0 0}
recorder Node -file disp.out -node 2 -dof 1 disp
recorder Element -file force.out -ele 1 force
constraints Plain
numberer Plain
system BandGeneral
test NormDispIncr 1e-12 20
algorithm Newton
integrator LoadControl 0.1
analysis Static
analyze 10
wipe
"""
    )
    oracle = tmp_path / "oracle.tcl"
    oracle.write_text(script.read_text().replace(" -file ", " -precision 17 -file "))
    subprocess.run([opensees, oracle], cwd=tmp_path, capture_output=True, check=True)
    output = tmp_path / "strut"
    subprocess.run(
        [
            sys.executable,
            ROOT / "scripts/run_strut_case.py",
            "--input-tcl",
            script,
            "--output",
            output,
        ],
        check=True,
    )
    for raw, normalized in [
        ("disp.out", "disp_node2.out"),
        ("force.out", "force_ele1.out"),
    ]:
        reference = np.loadtxt(tmp_path / raw)
        assert len(reference) == 10
        np.testing.assert_allclose(
            np.loadtxt(output / normalized), reference, rtol=1e-10, atol=1e-11
        )
