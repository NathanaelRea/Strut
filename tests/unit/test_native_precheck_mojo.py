import subprocess
from pathlib import Path


def _run_precheck_script(tmp_path: Path, case_json: str, include_recorders: bool):
    repo_root = Path(__file__).resolve().parents[2]
    script_path = tmp_path / "native_precheck_case.mojo"
    include_literal = "True" if include_recorders else "False"
    script_path.write_text(
        "\n".join(
            [
                "from json_native import parse_json_native",
                "from solver.run_case.precheck import precheck_case_input_native",
                "",
                "def main():",
                '    var doc = parse_json_native("""' + case_json + '""")',
                f"    precheck_case_input_native(doc, {include_literal})",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return subprocess.run(
        [
            "uv",
            "run",
            "mojo",
            "-I",
            str(repo_root / "src" / "mojo"),
            str(script_path),
        ],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )


def test_native_precheck_rejects_unsupported_mp_constraint_type(tmp_path: Path):
    proc = _run_precheck_script(
        tmp_path,
        '{"model":{"ndm":3,"ndf":6},"nodes":[],"elements":[],"analysis":{"type":"static_linear","constraints":"Transformation"},"mp_constraints":[{"type":"rigidDiaphragm"}]}',
        include_recorders=True,
    )

    assert proc.returncode != 0
    assert "unsupported mp constraint type: rigidDiaphragm" in proc.stdout


def test_native_precheck_rejects_unsupported_recorders_when_enabled(tmp_path: Path):
    proc = _run_precheck_script(
        tmp_path,
        '{"model":{"ndm":2,"ndf":3},"nodes":[],"elements":[],"analysis":{"type":"static_linear"},"recorders":[{"type":"unsupported_recorder"}]}',
        include_recorders=True,
    )

    assert proc.returncode != 0
    assert "unsupported recorder type" in proc.stdout


def test_native_precheck_skips_recorders_for_compute_only(tmp_path: Path):
    proc = _run_precheck_script(
        tmp_path,
        '{"model":{"ndm":2,"ndf":3},"nodes":[],"elements":[],"analysis":{"type":"static_linear"},"recorders":[{"type":"unsupported_recorder"}]}',
        include_recorders=False,
    )

    assert proc.returncode == 0, proc.stdout
