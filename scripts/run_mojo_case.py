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

    verbose = os.getenv("STRUT_VERBOSE") == "1"

    if os.getenv("STRUT_MOJO_SOLVER") != "1" or mojo is None:
        run(
            [
                "python",
                str(repo_root / "scripts" / "run_mojo_case_py.py"),
                "--input",
                args.input,
                "--output",
                args.output,
            ],
            verbose=verbose,
        )
        return

    run(
        [
            "mojo",
            "run",
            str(repo_root / "src" / "mojo" / "strut.mojo"),
            "--",
            "--input",
            args.input,
            "--output",
            args.output,
        ],
        verbose=verbose,
    )


if __name__ == "__main__":
    main()
