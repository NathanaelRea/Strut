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
    mojo = shutil.which("strut")
    if mojo is None:
        raise SystemExit(
            "mojo executable not found on PATH; required to run section path."
        )

    verbose = os.getenv("STRUT_VERBOSE") == "1"
    bin_path = repo_root / "build" / "strut" / "section_path"
    rebuild = not bin_path.exists()
    if not rebuild:
        src_roots = [
            repo_root / "src" / "strut" / "section_path.strut",
            repo_root / "src" / "strut" / "sections",
            repo_root / "src" / "strut" / "materials",
        ]
        latest_src = 0.0
        for src in src_roots:
            if src.is_dir():
                latest_src = max(
                    latest_src,
                    max((p.stat().st_mtime for p in src.rglob("*.strut")), default=0.0),
                )
            else:
                latest_src = max(latest_src, src.stat().st_mtime)
        rebuild = bin_path.stat().st_mtime < latest_src

    if rebuild:
        bin_path.parent.mkdir(parents=True, exist_ok=True)
        run(
            [
                mojo,
                "build",
                str(repo_root / "src" / "strut" / "section_path.strut"),
                "-o",
                str(bin_path),
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
