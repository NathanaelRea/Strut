#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path


def run(cmd, env=None, verbose=False):
    if verbose:
        print("+", " ".join(cmd))
    subprocess.check_call(cmd, env=env)


def _case_enabled(case_path: Path) -> bool:
    data = json.loads(case_path.read_text())
    return bool(data.get("enabled", True))


def _compute_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case_json")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    case_json = Path(args.case_json).resolve()
    case_name = case_json.stem
    case_root = repo_root / "tests" / "validation" / case_name

    case_root.mkdir(parents=True, exist_ok=True)
    (case_root / "generated").mkdir(parents=True, exist_ok=True)
    (case_root / "reference").mkdir(parents=True, exist_ok=True)
    (case_root / "strut").mkdir(parents=True, exist_ok=True)

    tgt_json = case_root / f"{case_name}.json"
    if case_json != tgt_json:
        shutil.copy2(case_json, tgt_json)

    if not _case_enabled(tgt_json) and os.getenv("STRUT_FORCE_CASE") != "1":
        return

    verbose = os.getenv("STRUT_VERBOSE") == "1"
    env = os.environ.copy()

    tcl_out = case_root / "generated" / "model.tcl"
    run(
        [
            "python",
            str(repo_root / "scripts" / "json_to_tcl.py"),
            str(tgt_json),
            str(tcl_out),
        ],
        env=env,
        verbose=verbose,
    )

    refresh_reference = os.getenv("STRUT_REFRESH_REFERENCE") == "1"
    ref_hash_file = case_root / "reference" / ".ref_hash"
    if not refresh_reference:
        current_hash = _compute_hash(tgt_json)
        stored_hash = (
            ref_hash_file.read_text().strip() if ref_hash_file.exists() else None
        )
        if stored_hash != current_hash:
            refresh_reference = True

    if refresh_reference:
        run(
            [
                str(repo_root / "scripts" / "run_opensees_wine.sh"),
                "--script",
                str(tcl_out),
                "--output",
                str(case_root / "reference"),
            ],
            env=env,
            verbose=verbose,
        )
        ref_hash_file.parent.mkdir(parents=True, exist_ok=True)
        ref_hash_file.write_text(_compute_hash(tgt_json) + "\n", encoding="utf-8")

    run(
        [
            "python",
            str(repo_root / "scripts" / "run_strut_case.py"),
            "--input",
            str(tgt_json),
            "--output",
            str(case_root / "strut"),
        ],
        env=env,
        verbose=verbose,
    )

    run(
        [
            "python",
            str(repo_root / "scripts" / "compare_case.py"),
            "--case",
            case_name,
        ],
        env=env,
        verbose=verbose,
    )


if __name__ == "__main__":
    main()
