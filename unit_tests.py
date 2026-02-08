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
    parser.add_argument("--case", action="append", default=[], help="run specific validation case(s)")
    parser.add_argument("--verbose", action="store_true", help="print commands as they run")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    env = os.environ.copy()

    # Unit + schema tests
    if args.verbose:
        env["STRUT_VERBOSE"] = "1"
    pytest_cmd = [
        "pytest",
        "-q",
        "tests/unit",
        "tests/validation/test_json_cases.py",
    ]
    if shutil.which("pytest") is None:
        if shutil.which("uv") is None:
            raise SystemExit("pytest not found (install with uv or add pytest to PATH).")
        pytest_cmd = ["uv", "run"] + pytest_cmd
    run(pytest_cmd, env=env, verbose=args.verbose)

    if args.case:
        env["STRUT_PARITY_CASES"] = ",".join(args.case)
    parity_cmd = ["pytest", "-q", "tests/validation/test_parity_cases.py"]
    if shutil.which("pytest") is None:
        parity_cmd = ["uv", "run"] + parity_cmd
    run(parity_cmd, env=env, verbose=args.verbose)


if __name__ == "__main__":
    main()
