#!/usr/bin/env python3
import argparse
import os
import subprocess
from pathlib import Path


def run(cmd, env=None, verbose=False):
    if verbose:
        print("+", " ".join(cmd))
    subprocess.check_call(cmd, env=env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="run all validation cases (even disabled)")
    parser.add_argument("--case", action="append", default=[], help="run specific validation case(s)")
    parser.add_argument("--no-parity", action="store_true", help="skip running OpenSees/Mojo parity")
    parser.add_argument("--verbose", action="store_true", help="print commands as they run")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    env = os.environ.copy()

    # Unit + schema tests
    if args.all:
        env["STRUT_RUN_ALL_CASES"] = "1"
    if args.verbose:
        env["STRUT_VERBOSE"] = "1"
    run(["pytest", "-q", "tests/unit", "tests/validation/test_json_cases.py"], env=env, verbose=args.verbose)

    if args.no_parity:
        return

    run_cases = []
    if args.case:
        run_cases = args.case
    else:
        # By default, run only enabled cases
        validation_root = repo_root / "tests" / "validation"
        for case_dir in sorted(validation_root.iterdir()):
            if not case_dir.is_dir():
                continue
            case_json = case_dir / f"{case_dir.name}.json"
            if not case_json.exists():
                continue
            run_cases.append(str(case_json))

    for case_path in run_cases:
        run(["scripts/run_case.py", case_path], verbose=args.verbose)


if __name__ == "__main__":
    main()
