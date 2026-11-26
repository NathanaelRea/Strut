#!/usr/bin/env python3
"""Sampling profile of the Strut solver on one case, attributed to source lines.

Works without `perf` or root: it uses a small perf_event_open sampler
(scripts/profiling/usample.c) and a solver built with line tables (-g1), so
samples resolve through inlining to the Mojo function and line that spent the
time. The speedscope instrumentation (`--profile`) answers "which phase";
this answers "which code".

Examples:
  uv run scripts/profile_sample.py --case opensees_example_ex5_frame2d_eq_uniform_rc_fiber
  uv run scripts/profile_sample.py --input .tmp/synthetic/rc_frame3d_2x2x3.json --top 30
"""
from __future__ import annotations

import argparse
import bisect
import collections
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

BUILD_DIR = REPO_ROOT / "build" / "strut"
SAMPLER_SRC = SCRIPT_DIR / "profiling" / "usample.c"
SAMPLER_BIN = BUILD_DIR / "usample"
SOLVER_BIN = BUILD_DIR / "strut_g1"
# Addresses below this belong to the non-PIE solver executable itself.
MAIN_IMAGE_LIMIT = 0x10000000


def _newer_than(target: Path, sources: list[Path]) -> bool:
    if not target.exists():
        return False
    target_mtime = target.stat().st_mtime
    return all(src.stat().st_mtime <= target_mtime for src in sources)


def _ensure_sampler() -> Path:
    if not _newer_than(SAMPLER_BIN, [SAMPLER_SRC]):
        compiler = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
        if compiler is None:
            raise SystemExit("a C compiler (cc, gcc, or clang) is required to build the sampler")
        BUILD_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.check_call([compiler, "-O2", "-o", str(SAMPLER_BIN), str(SAMPLER_SRC)])
    return SAMPLER_BIN


def _ensure_line_table_solver() -> Path:
    sources = sorted((REPO_ROOT / "src" / "mojo").rglob("*.mojo"))
    if not _newer_than(SOLVER_BIN, sources):
        BUILD_DIR.mkdir(parents=True, exist_ok=True)
        print("building line-table solver (about a minute)...", file=sys.stderr)
        subprocess.check_call(
            ["uv", "run", "mojo", "build", "-g1", str(REPO_ROOT / "src" / "mojo" / "strut.mojo"),
             "-o", str(SOLVER_BIN)],
            cwd=REPO_ROOT,
        )
    return SOLVER_BIN


def _short_path(location: str) -> str:
    location = location.split(" (discriminator")[0]
    for marker in ("/src/mojo/", "/stdlib/stdlib/"):
        if marker in location:
            prefix = "" if marker == "/src/mojo/" else "stdlib/"
            return prefix + location.split(marker, 1)[1]
    return location


def _report(samples_path: Path, binary: Path, top: int) -> None:
    maps: list[tuple[int, int, str]] = []
    ips: collections.Counter[int] = collections.Counter()
    total = 0
    for line in samples_path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "M":
            maps.append((int(parts[1], 16), int(parts[2], 16), " ".join(parts[4:])))
            continue
        ips[int(parts[0], 16)] += 1
        total += 1
    if total == 0:
        raise SystemExit("no samples collected (run too short? raise --freq)")

    symbols: list[tuple[int, str]] = []
    nm = subprocess.run(["nm", "-n", "--defined-only", str(binary)], capture_output=True, text=True)
    for line in nm.stdout.splitlines():
        fields = line.split(None, 2)
        if len(fields) == 3 and fields[1].lower() in "tw":
            symbols.append((int(fields[0], 16), fields[2]))
    addresses = [address for address, _ in symbols]

    def symbol_of(ip: int) -> str:
        if ip < MAIN_IMAGE_LIMIT:
            index = bisect.bisect_right(addresses, ip) - 1
            if index >= 0:
                return symbols[index][1].split("(")[0]
        for start, length, path in maps:
            if start <= ip < start + length:
                return "[" + path.rsplit("/", 1)[-1] + "]"
        return "[unknown]"

    by_symbol: collections.Counter[str] = collections.Counter()
    main_ips = []
    for ip, count in ips.items():
        by_symbol[symbol_of(ip)] += count
        if ip < MAIN_IMAGE_LIMIT:
            main_ips.append(ip)

    def show(title: str, counter: collections.Counter[str], limit: int) -> None:
        print(f"\n== {title} ==")
        for name, count in counter.most_common(limit):
            print(f"{100.0 * count / total:6.2f}%  {count:7d}  {name[:140]}")

    print(f"total samples: {total}")
    show("by symbol, self time (inlined callees are folded into their caller)", by_symbol, top)

    by_function: collections.Counter[str] = collections.Counter()
    by_file: collections.Counter[str] = collections.Counter()
    by_line: collections.Counter[str] = collections.Counter()
    if main_ips:
        resolved = subprocess.run(
            ["addr2line", "-e", str(binary), "-f", "-i", "-a"] + [hex(ip) for ip in main_ips],
            capture_output=True,
            text=True,
        )
        current = None
        pending = None
        innermost: dict[int, tuple[str, str]] = {}
        for line in resolved.stdout.splitlines():
            if line.startswith("0x"):
                current = int(line, 16)
                pending = None
            elif current is not None and current not in innermost:
                if pending is None:
                    pending = line
                else:
                    innermost[current] = (pending, line)
        for ip in main_ips:
            count = ips[ip]
            function, location = innermost.get(ip, ("??", "??:0"))
            location = _short_path(location)
            by_function[function.split("(")[0]] += count
            by_file[location.rsplit(":", 1)[0]] += count
            by_line[location] += count
        show("by innermost function (after inlining)", by_function, top)
        show("by source file", by_file, 20)
        show("by source line", by_line, top)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--case", help="Validation case name (JSON or direct-Tcl).")
    source.add_argument("--input", help="Solver-ready JSON input.")
    parser.add_argument("--freq", type=int, default=4000, help="Sampling frequency in Hz.")
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--with-recorders", action="store_true",
                        help="Keep recorder output in the profile (default profiles compute only).")
    parser.add_argument("--keep-samples", default="", help="Write the raw sample file here.")
    args = parser.parse_args()

    sampler = _ensure_sampler()
    solver = _ensure_line_table_solver()

    scratch_root = REPO_ROOT / ".tmp" / "profile_sample"
    scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="run_", dir=scratch_root) as tmp:
        work_dir = Path(tmp)
        if args.case:
            import ab_compare

            _, input_json = ab_compare._resolved_input(ab_compare._resolve_named_case(args.case), work_dir)
        else:
            input_json = Path(args.input).resolve()
        samples_path = work_dir / "samples.txt"
        command = [str(sampler), str(samples_path), str(args.freq), "--", str(solver),
                   "--input", str(input_json), "--output", str(work_dir / "out")]
        if not args.with_recorders:
            command.append("--compute-only")
        proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            sys.stderr.write(proc.stdout + proc.stderr)
            raise SystemExit(f"solver failed under the sampler (exit {proc.returncode})")
        analysis_path = work_dir / "out" / "analysis_time_us.txt"
        if analysis_path.exists():
            print(f"analysis time: {int(analysis_path.read_text().strip()) / 1000.0:.3f} ms")
        _report(samples_path, solver, args.top)
        if args.keep_samples:
            shutil.copy2(samples_path, args.keep_samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
