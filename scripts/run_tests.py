#!/usr/bin/env python3
import os
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
run_tests = repo_root / "run_tests.py"

os.execv(sys.executable, [sys.executable, str(run_tests), *sys.argv[1:]])
