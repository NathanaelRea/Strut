#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
from pathlib import Path


def run(cmd, env=None, verbose=False):
    if verbose:
        print("+", " ".join(cmd))
    subprocess.check_call(cmd, env=env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    mojo = shutil.which("mojo")
    if mojo is None:
        raise SystemExit("mojo executable not found on PATH; required to run solver.")

    verbose = os.getenv("STRUT_VERBOSE") == "1"
    solver_bin = os.getenv("STRUT_MOJO_BIN")
    if solver_bin:
        solver_path = Path(solver_bin)
    else:
        solver_path = repo_root / "build" / "mojo" / "strut"

    if not solver_path.exists():
        solver_path.parent.mkdir(parents=True, exist_ok=True)
        run(
            [
                mojo,
                "build",
                str(repo_root / "src" / "mojo" / "strut.mojo"),
                "-o",
                str(solver_path),
            ],
            verbose=verbose,
        )

    run(
        [
            str(solver_path),
            "--input",
            args.input,
            "--output",
            args.output,
        ],
        verbose=verbose,
    )


if __name__ == "__main__":
    main()
