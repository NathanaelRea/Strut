#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import direct_tcl_support
from compare_case import _load_direct_tcl_case_data


DEFAULT_CASE = (
    "tests/validation/"
    "opensees_example_ex6_genericframe2d_analyze_dynamic_eq_uniform/"
    "direct_tcl_case.json"
)


def resolve_case_input(case_arg: str) -> Path:
    raw_path = Path(case_arg)
    path = raw_path.resolve() if raw_path.is_absolute() else (REPO_ROOT / raw_path).resolve()
    if not path.exists():
        raise SystemExit(f"case input not found: {case_arg}")
    return path


def prepare_case_json(case_path: Path) -> tuple[Path, list[Path]]:
    if case_path.suffix == ".json" and case_path.name != "direct_tcl_case.json":
        return case_path, []

    entry_tcl = direct_tcl_support.resolve_entry_tcl_from_manifest(case_path, REPO_ROOT)
    case_root = case_path.parent
    case_data = _load_direct_tcl_case_data(entry_tcl, REPO_ROOT, case_root, case_path)
    mirror_root = Path(tempfile.mkdtemp(prefix="strut_microbench_case_"))
    json_path = mirror_root / "case.json"
    json_path.write_text(json.dumps(case_data, indent=2) + "\n", encoding="utf-8")
    return json_path, [mirror_root]


def build_mojo_binary(source_name: str, binary_name: str) -> Path:
    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit("uv executable not found on PATH")

    bin_path = REPO_ROOT / "build" / "strut" / binary_name
    src_roots = [
        REPO_ROOT / "src" / "mojo" / source_name,
        REPO_ROOT / "src" / "mojo" / "sections",
        REPO_ROOT / "src" / "mojo" / "materials",
        REPO_ROOT / "src" / "mojo" / "solver",
    ]
    rebuild = not bin_path.exists()
    latest_src = 0.0
    for src in src_roots:
        if src.is_dir():
            latest_src = max(
                latest_src,
                max((p.stat().st_mtime for p in src.rglob("*.mojo")), default=0.0),
            )
        else:
            latest_src = max(latest_src, src.stat().st_mtime)
    if not rebuild:
        rebuild = bin_path.stat().st_mtime < latest_src
    if rebuild:
        bin_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(
            [
                uv,
                "run",
                "mojo",
                "build",
                str(REPO_ROOT / "src" / "mojo" / source_name),
                "-o",
                str(bin_path),
            ],
            cwd=REPO_ROOT,
        )
    return bin_path

