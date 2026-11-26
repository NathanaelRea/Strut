#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
from pathlib import Path

from mojo_tool_build import ensure_built


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
    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit(
            "uv executable not found on PATH; required to run section path."
        )

    verbose = os.getenv("STRUT_VERBOSE") == "1"
    bin_path = repo_root / "build" / "strut" / "section_path"
    entry = repo_root / "src" / "mojo" / "section_path.mojo"
    ensure_built(
        uv,
        entry,
        bin_path,
        [
            entry,
            repo_root / "src" / "mojo" / "sections",
            repo_root / "src" / "mojo" / "materials",
        ],
        verbose=verbose,
    )

    run(
        [
            str(bin_path),
            "--input",
            args.input,
            "--output",
            args.output,
        ],
        verbose=verbose,
    )


if __name__ == "__main__":
    main()
