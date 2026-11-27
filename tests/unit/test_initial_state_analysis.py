import subprocess
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def test_initial_state_reset_keeps_quad_material_strain_and_stress(tmp_path: Path):
    entry = tmp_path / "initial_state.tcl"
    entry.write_text(
        """model BasicBuilder -ndm 2 -ndf 2
node 1 0 0
node 2 1 0
node 3 1 1
node 4 0 1
fix 1 1 1
fix 2 1 1
fix 3 1 0
fix 4 1 0
nDMaterial PressureDependMultiYield02 5 2 1.8 90000 270000 36 0.1 100 0.5 26 0.067 0.23 0.06 0.27
nDMaterial InitialStateAnalysisWrapper 1 5 2
element quad 1 1 2 3 4 1 PlaneStrain 1 0 0 0 -18
recorder Node -file disp.out -node 3 -dof 2 disp
recorder Element -file stress.out -ele 1 material 1 stress
recorder Element -file strain.out -ele 1 material 1 strain
constraints Transformation
system BandGeneral
test NormDispIncr 1e-10 20
algorithm Newton
integrator LoadControl 1
analysis Static
InitialStateAnalysis on
updateMaterialStage -material 1 -stage 0
analyze 1
InitialStateAnalysis off
analyze 1
""",
        encoding="utf-8",
    )
    output = tmp_path / "strut"
    subprocess.run(
        [
            "uv",
            "run",
            "python",
            ROOT / "scripts/run_strut_case.py",
            "--input-tcl",
            entry,
            "--output",
            output,
        ],
        check=True,
        cwd=ROOT,
    )
    # The top nodes each carry a vertical body load of -4.5. The Gauss
    # points see the same vertical strain and a stress of -9.
    strain = -9.0 / (270000.0 + 4.0 / 3.0 * 90000.0)
    np.testing.assert_allclose(
        np.loadtxt(output / "disp_node3.out"), [strain, 0.0], atol=1e-12
    )
    np.testing.assert_allclose(
        np.loadtxt(output / "strain_ele1.out")[:, 1], [strain, strain], atol=1e-12
    )
    np.testing.assert_allclose(
        np.loadtxt(output / "stress_ele1.out")[:, 1], [-9.0, -9.0], atol=1e-12
    )


def test_quad_removal_rebuilds_the_active_equations(tmp_path: Path):
    entry = tmp_path / "remove_quad.tcl"
    entry.write_text(
        """model BasicBuilder -ndm 2 -ndf 2
node 1 0 0
node 2 1 0
node 3 1 1
node 4 0 1
node 5 1 2
node 6 0 2
fix 1 1 1
fix 2 1 1
fix 3 1 0
fix 4 1 0
fix 5 1 0
fix 6 1 0
nDMaterial ElasticIsotropic 1 1000 0
element quad 1 1 2 3 4 1 PlaneStrain 1 0 0 0 -18
element quad 2 4 3 5 6 1 PlaneStrain 1 0 0 0 -18
recorder Node -file retained.out -node 3 -dof 2 disp
set removed [recorder Node -file removed.out -node 5 6 -dof 2 disp]
constraints Transformation
system BandGeneral
test NormDispIncr 1e-10 20
algorithm Newton
integrator LoadControl 1
analysis Static
analyze 1
remove recorder $removed
remove element 2
remove node 5
remove node 6
analyze 2
""",
        encoding="utf-8",
    )
    output = tmp_path / "strut"
    subprocess.run(
        [
            "uv",
            "run",
            "python",
            ROOT / "scripts/run_strut_case.py",
            "--input-tcl",
            entry,
            "--output",
            output,
        ],
        check=True,
        cwd=ROOT,
    )
    np.testing.assert_allclose(
        np.loadtxt(output / "retained_node3.out"), [-0.027, -0.009, -0.009], atol=1e-12
    )
    np.testing.assert_allclose(
        np.loadtxt(output / "removed_node5.out"), -0.036, atol=1e-12
    )
