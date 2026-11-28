import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _run_live(tmp_path, script):
    out = tmp_path / "output"
    subprocess.run(
        [
            sys.executable,
            ROOT / "scripts/run_strut_case.py",
            "--interactive-tcl",
            script,
            "--replay-mode",
            "live",
            "--output",
            out,
        ],
        env={**os.environ, "STRUT_TCL_SESSION": "required"},
        check=True,
    )
    return out


def test_session_keeps_successful_prefix_and_retries_failed_load_step(tmp_path):
    script = tmp_path / "model.tcl"
    script.write_text(
        """model basic -ndm 1 -ndf 1
node 1 0
node 2 0
fix 1 1
uniaxialMaterial Elastic 1 10
element zeroLength 1 1 2 -mat 1 -dir 1
timeSeries Path 1 -dt 1 -values {0 0.05 0.1 1 0} -useLast
pattern Plain 1 1 {load 2 1}
recorder Node -file disp.out -node 2 -dof 1 disp
constraints Plain
numberer Plain
system BandGeneral
test NormDispIncr 0.05 1
algorithm Newton
integrator LoadControl 1
analysis Static
if {[analyze 1] != 0} {error "first command failed"}
if {abs([nodeDisp 2 1] - 0.005) > 1e-12} {error "wrong committed displacement"}
if {abs([eleForce 1 2] - 0.05) > 1e-12} {error "wrong committed element force"}
reactions
if {abs([nodeReaction 1 1] + 0.05) > 1e-12} {error "wrong committed reaction"}
if {[analyze 2] != -3} {error "expected failure after one step in the continued command"}
if {[getTime] != 2} {error "failed step changed the clock"}
if {abs([nodeDisp 2 1] - 0.01) > 1e-12} {error "failed step changed displacement"}
test NormDispIncr 1e-12 3
algorithm ModifiedNewton -initial
if {[analyze 1] != 0} {error "retry failed"}
if {[getTime] != 3} {error "retry did not advance the clock"}
if {abs([nodeDisp 2 1] - 0.1) > 1e-12} {error "wrong displacement after retry"}
if {abs([eleForce 1 2] - 1) > 1e-12} {error "wrong element force after retry"}
reactions
if {abs([nodeReaction 1 1] + 1) > 1e-12} {error "wrong reaction after retry"}
""",
        encoding="utf-8",
    )
    out = _run_live(tmp_path, script)
    rows = (out / "__strut_domains/1/disp.out").read_text().split()
    assert list(map(float, rows)) == pytest.approx([0.005, 0.01, 0.1])
    assert (out / "__strut_query_analyze.out").read_text().split() == ["0", "-3", "0"]


def test_session_reverts_failed_transient_step_before_retry(tmp_path):
    script = tmp_path / "model.tcl"
    script.write_text(
        """model basic -ndm 1 -ndf 1
node 1 0
node 2 0 -mass 1
fix 1 1
uniaxialMaterial Steel01 1 100 10 0.1
element zeroLength 1 1 2 -mat 1 -dir 1
timeSeries Path 1 -dt 1 -values {0 0.1 0.15 1 0} -useLast
pattern Plain 1 1 {load 2 1}
constraints Plain
numberer Plain
system BandGeneral
test NormDispIncr 0.05 1
algorithm Newton
integrator Newmark 0.5 0.25
analysis Transient
if {[analyze 1 1] != 0} {error "first step failed"}
nodeDisp 2 1
nodeVel 2 1
nodeAccel 2 1
if {[analyze 1 1] != 0} {error "second step failed"}
set displacement [nodeDisp 2 1]
set velocity [nodeVel 2 1]
set acceleration [nodeAccel 2 1]
if {[analyze 1 1] != -3} {error "expected third-step failure"}
if {[getTime] != 2} {error "failed step changed the clock"}
if {[nodeDisp 2 1] != $displacement || [nodeVel 2 1] != $velocity ||
    [nodeAccel 2 1] != $acceleration} {error "failed step changed node state"}
test NormDispIncr 1e-12 3
if {[analyze 1 1] != 0} {error "retry failed"}
if {[getTime] != 3} {error "retry did not advance the clock"}
""",
        encoding="utf-8",
    )
    out = _run_live(tmp_path, script)
    assert (out / "__strut_query_analyze.out").read_text().split() == [
        "0",
        "0",
        "-3",
        "0",
    ]


def test_session_reuses_process_and_isolates_domains(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    monkeypatch.setenv("STRUT_TCL_SESSION", "required")
    from interactive_tcl_runner import run_interactive_tcl_case

    script = tmp_path / "model.tcl"
    script.write_text(
        """foreach stiffness {10 40} {
    model basic -ndm 1 -ndf 1
    node 1 0
    node 2 0
    fix 1 1
    uniaxialMaterial Elastic 1 $stiffness
    element zeroLength 1 1 2 -mat 1 -dir 1
    if {abs([eigen standard 1] - $stiffness) > 1e-12} {error "wrong domain eigenvalue"}
    pattern Plain 1 Linear {load 2 1}
    recorder Node -file disp.out -node 2 -dof 1 disp
    constraints Plain
    numberer Plain
    system BandGeneral
    test NormDispIncr 1e-12 3
    algorithm Newton
    integrator LoadControl 1
    analysis Static
    if {[analyze 1] != 0 || [getTime] != 1} {error "domain did not advance from zero"}
    wipe
}
""",
        encoding="utf-8",
    )
    commands = []
    popen = subprocess.Popen

    def record_spawn(command, **kwargs):
        commands.append(command)
        return popen(command, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", record_spawn)
    solver = Path(os.environ.get("STRUT_MOJO_BIN") or ROOT / "build/strut/strut")
    out = tmp_path / "output"
    run_interactive_tcl_case(script, out, ROOT, solver, "live")

    assert commands == [[str(solver.resolve()), "--session"]]
    for domain, expected in enumerate((0.1, 0.025), start=1):
        values = (out / f"__strut_domains/{domain}/disp.out").read_text().split()
        assert list(map(float, values)) == pytest.approx([expected])
