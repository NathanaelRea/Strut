#!/usr/bin/env python3
"""Sweep OpenSees/Strut parity over synthetic ground motions (local tool, not CI).

For a direct-Tcl parity case that reads ground motions, every pinned motion
file the case can see is replaced by a seeded synthetic record from
``synthetic_ground_motion.py`` (same file name, format, dt and length; PGA is
the real record's PGA times ``--pga-scale``). Each seed/intensity runs the
normal harness, ``scripts/run_case.py``, against a scratch copy of the
example bundle. OpenSees and Strut both run, and ``compare_case.py``'s
tolerance machinery judges the result.

Per run this reports parity pass/fail, the worst tolerance ratio
``|strut - ref| / (atol + rtol |ref|)`` seen by the comparison (``> 1`` is
outside tolerance; step-mode comparisons stop at the first failing step, so a
failing ratio is a lower bound), the PGA of each synthetic motion, and each
engine's convergence status. Nonlinear models can legitimately diverge or lose
convergence near instability, so non-convergence is reported separately from
parity mismatches.

Needs the locked OpenSees binary (``uv run scripts/setup_opensees.py``) and a
built solver (``uv run python scripts/build_mojo_solver.py``). The real pinned
motions are not needed.

Example:
    uv run scripts/sweep_synthetic_parity.py opensees_example_ex1a_canti2d_eq \\
        --seeds 1-5 --pga-scale 1,2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import compare_case
import direct_tcl_support
import synthetic_ground_motion as sgm

REPO_ROOT = SCRIPT_DIR.parent
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "opensees"
LOCK_PATH = FIXTURE_ROOT / "reference-data.lock.json"
DEFAULT_WORK_ROOT = REPO_ROOT / ".tmp" / "synthetic_parity_sweep"


@dataclass(frozen=True)
class RecordShape:
    """dt, length and PGA of a real pinned record (PGA in the file's units)."""

    dt: float
    npts: int
    pga: float


# Keyed by the lower-case file stem; .AT2/.DT2 pairs share an entry and one
# synthetic acceleration. Values are from the pinned files; values-only files
# (.acc, A10000.tcl) take dt from the -dt their example scripts pass.
REAL_RECORDS = {
    "h-e12140": RecordShape(dt=0.005, npts=7802, pga=0.1433283),
    "h-e01140": RecordShape(dt=0.005, npts=7807, pga=0.1394886),
    "bm68elc": RecordShape(dt=0.01, npts=4000, pga=0.056658),
    "a10000": RecordShape(dt=0.005, npts=7990, pga=0.102657),
}

_STRUT_NONCONVERGED_RE = re.compile(r"[^\n]*did not converge[^\n]*")
_OPENSEES_STEP_FAILED_RE = re.compile(r"Analysis::analyze\(\) - the Algorithm failed")


@dataclass
class MotionInfo:
    files: list[str]
    seed: int
    target_pga: float
    measured_pga: float
    referenced: bool


@dataclass
class RunResult:
    case: str
    seed: int
    pga_scale: float
    motions: list[MotionInfo]
    status: str
    parity: str
    worst_ratio: float | None
    opensees: str
    strut: str
    first_failure: str
    seconds: float
    run_dir: str
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Case and bundle preparation
# --------------------------------------------------------------------------- #


def resolve_manifest(case: str) -> Path:
    candidate = Path(case)
    if candidate.is_file():
        return candidate.resolve()
    manifest = REPO_ROOT / "tests" / "validation" / case / "direct_tcl_case.json"
    if not manifest.is_file():
        raise SystemExit(
            f"no direct-Tcl parity case named {case!r} ({manifest} missing)"
        )
    return manifest


def pinned_files() -> list[str]:
    data = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    return [entry["path"] for entry in data["files"]]


def _has_tcl(directory: Path) -> bool:
    return any(p.is_file() and p.suffix.lower() == ".tcl" for p in directory.iterdir())


def copy_bundle(entry: Path, source_files: list[Path], target_bundle: Path) -> None:
    """Copy the committed files the harness would mirror for ``entry``.

    The harness mirrors the entry's parent ("bundle") directory. Only the
    example directories the case uses are copied, plus shared assets (bundle
    root files and directories without Tcl, e.g. ``GMfiles``). Pinned
    downloads are left out; synthetic motions replace them.
    """
    bundle = entry.parent.parent
    pinned = set(pinned_files())
    bundle_rel = bundle.relative_to(FIXTURE_ROOT).as_posix()
    needed_dirs = {entry.parent} | {path.parent for path in source_files}

    def _ignore(dirpath: str, names: list[str]) -> set[str]:
        base = Path(dirpath)
        ignored = set()
        for name in names:
            rel = (base / name).relative_to(FIXTURE_ROOT).as_posix()
            if rel in pinned:
                ignored.add(name)
        return ignored

    target_bundle.mkdir(parents=True, exist_ok=True)
    for child in sorted(bundle.iterdir()):
        rel = f"{bundle_rel}/{child.name}"
        if child.is_file():
            if rel not in pinned:
                shutil.copy2(child, target_bundle / child.name)
            continue
        if not child.is_dir():
            continue
        if child in needed_dirs or not _has_tcl(child):
            shutil.copytree(child, target_bundle / child.name, ignore=_ignore)


def motion_targets(entry: Path, source_files: list[Path]) -> dict[str, list[str]]:
    """Pinned motion files visible to the case, grouped by lower-case stem.

    Paths are relative to the bundle directory.
    """
    bundle = entry.parent.parent
    bundle_rel = PurePosixPath(bundle.relative_to(FIXTURE_ROOT).as_posix())
    used_dirs = {entry.parent.name} | {path.parent.name for path in source_files}
    groups: dict[str, list[str]] = {}
    for raw in pinned_files():
        path = PurePosixPath(raw)
        if not path.is_relative_to(bundle_rel):
            continue
        rel = path.relative_to(bundle_rel)
        if len(rel.parts) > 1:
            top = bundle / rel.parts[0]
            if rel.parts[0] not in used_dirs and _has_tcl(top):
                continue  # belongs to another example
        groups.setdefault(path.stem.lower(), []).append(rel.as_posix())
    return groups


def referenced_stems(
    entry: Path, source_files: list[Path], stems: list[str]
) -> set[str]:
    """Stems named in the Tcl the case sources (others merely sit in the bundle)."""
    text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace").lower()
        for path in sorted({entry, *source_files})
    )
    return {stem for stem in stems if stem in text}


def _motion_seed(seed: int, stem: str) -> int:
    digest = hashlib.sha256(f"{seed}:{stem}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def write_synthetic_motions(
    bundle_copy: Path,
    groups: dict[str, list[str]],
    seed: int,
    pga_scale: float,
    overrides: dict,
    referenced: set[str],
) -> list[MotionInfo]:
    infos = []
    for stem in sorted(groups):
        shape = REAL_RECORDS.get(stem)
        if shape is None:
            raise SystemExit(
                f"no dt/npts/PGA known for pinned motion {stem!r}; add it to "
                "REAL_RECORDS in scripts/sweep_synthetic_parity.py"
            )
        params = sgm.MotionParams(
            seed=_motion_seed(seed, stem),
            dt=shape.dt,
            npts=shape.npts,
            pga=shape.pga * pga_scale,
            **overrides,
        )
        acc = sgm.generate_acceleration(params)
        files = sorted(groups[stem])
        for rel in files:
            sgm.write_motion(bundle_copy / rel, params, acceleration=acc)
        # PGA as written (after rounding to the file's precision).
        accel_files = [f for f in files if sgm.format_for_path(Path(f)) != "dt2"]
        written = (
            sgm.read_motion(bundle_copy / accel_files[0]).values if accel_files else acc
        )
        measured = max(abs(v) for v in written)
        infos.append(
            MotionInfo(
                files=files,
                seed=params.seed,
                target_pga=params.pga,
                measured_pga=measured,
                referenced=stem in referenced,
            )
        )
    return infos


def resolve_opensees_binary() -> Path:
    proc = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; strut_resolve_opensees_binary OpenSees "${OPENSEES_BIN:-}"',
            "bash",
            str(SCRIPT_DIR / "opensees_env.sh"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    binary = Path(proc.stdout.strip())
    if not binary.is_file():
        raise SystemExit(
            f"OpenSees binary not found at {binary}; run: uv run scripts/setup_opensees.py"
        )
    return binary


def write_opensees_wrapper(run_dir: Path, binary: Path) -> tuple[Path, Path]:
    """A wrapper that keeps OpenSees' console output (run_opensees.sh drops it)."""
    log = run_dir / "opensees_console.log"
    wrapper = run_dir / "opensees_logged.sh"
    wrapper.write_text(
        "#!/usr/bin/env bash\n"
        f'exec {shlex.quote(str(binary))} "$@" >>{shlex.quote(str(log))} 2>&1\n',
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    return wrapper, log


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


def compare_with_ratio(
    data: dict, ref_dir: Path, strut_dir: Path
) -> tuple[list[str], float]:
    """``compare_case._compare_output_dirs`` plus the worst tolerance ratio."""
    worst = 0.0
    original = compare_case._isclose

    def _tracking_isclose(a, b, rtol=compare_case.REL_TOL, atol=compare_case.ABS_TOL):
        nonlocal worst
        ratio = abs(a - b) / (atol + rtol * abs(b))
        worst = max(worst, ratio)
        return original(a, b, rtol=rtol, atol=atol)

    compare_case._isclose = _tracking_isclose
    try:
        failures = compare_case._compare_output_dirs(data, ref_dir, strut_dir)
    finally:
        compare_case._isclose = original
    return failures, worst


def _line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def recorded_rows(output_dir: Path) -> int:
    """Most rows in any recorder output: how many steps an engine got through."""
    if not output_dir.is_dir():
        return 0
    return max((_line_count(path) for path in output_dir.rglob("*.out")), default=0)


def transient_steps(case_data: dict) -> int:
    """Requested transient steps; a complete run records at least this many rows."""
    analysis = case_data.get("analysis", {})
    stages = analysis.get("stages") if analysis.get("type") == "staged" else [analysis]
    total = 0
    for stage in stages or []:
        stage_analysis = stage.get("analysis", stage) if isinstance(stage, dict) else {}
        if str(stage_analysis.get("type", "")).startswith("transient"):
            total += int(stage_analysis.get("steps", 0))
    return total


def _engine_progress(rows: int, full: int) -> str | None:
    if rows >= full:
        return None
    if rows == 0:
        return "no output"
    return f"stopped at row {rows}/{full}"


def run_one(
    manifest: Path,
    work_root: Path,
    seed: int,
    pga_scale: float,
    overrides: dict,
    opensees_binary: Path,
    timeout: float,
    memory_limit_kb: int | None,
) -> RunResult:
    data = direct_tcl_support.load_direct_tcl_manifest(manifest)
    case_name = str(data.get("name") or manifest.parent.name)
    entry = direct_tcl_support.resolve_entry_tcl_from_manifest(manifest, REPO_ROOT)
    source_files = direct_tcl_support.resolve_direct_tcl_source_files(entry, manifest)
    try:
        entry.parent.parent.relative_to(FIXTURE_ROOT)
    except ValueError as exc:
        raise SystemExit(f"{case_name}: entry Tcl is not under {FIXTURE_ROOT}") from exc
    groups = motion_targets(entry, source_files)
    if not groups:
        raise SystemExit(f"{case_name}: the case reads no pinned ground motions")

    run_dir = work_root / case_name / f"seed{seed}_pga{pga_scale:g}"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    bundle_copy = run_dir / "bundle" / entry.parent.parent.name
    copy_bundle(entry, source_files, bundle_copy)
    referenced = referenced_stems(entry, source_files, sorted(groups))
    motions = write_synthetic_motions(
        bundle_copy, groups, seed, pga_scale, overrides, referenced
    )

    case_root = run_dir / "case"
    case_root.mkdir(parents=True)
    run_manifest = dict(data)
    run_manifest["entry_tcl"] = str(bundle_copy / entry.parent.name / entry.name)
    manifest_copy = case_root / "direct_tcl_case.json"
    manifest_copy.write_text(
        json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8"
    )

    wrapper, opensees_log = write_opensees_wrapper(run_dir, opensees_binary)
    env = os.environ.copy()
    env["OPENSEES_BIN"] = str(wrapper)
    env["STRUT_VERBOSE"] = "1"
    env["STRUT_REFRESH_REFERENCE"] = "1"
    cmd = ["uv", "run", "python", str(SCRIPT_DIR / "run_case.py"), str(manifest_copy)]
    if memory_limit_kb is not None:
        cmd = [
            "bash",
            "-c",
            f'ulimit -v {int(memory_limit_kb)} && exec "$@"',
            "bash",
            *cmd,
        ]

    started = time.monotonic()
    notes = []
    try:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        returncode = proc.returncode
        log_text = proc.stdout + proc.stderr
    except subprocess.TimeoutExpired as exc:
        returncode = None
        log_text = (exc.stdout or "") + (exc.stderr or "")
        if isinstance(log_text, bytes):
            log_text = log_text.decode(errors="replace")
        notes.append(f"timed out after {timeout:g}s")
    seconds = time.monotonic() - started
    (run_dir / "run_case.log").write_text(log_text, encoding="utf-8")

    opensees_text = (
        opensees_log.read_text(encoding="utf-8", errors="replace")
        if opensees_log.exists()
        else ""
    )
    ref_dir = case_root / "reference"
    strut_dir = case_root / "strut"
    case_json = case_root / "generated" / "case.json"
    case_data = (
        json.loads(case_json.read_text(encoding="utf-8"))
        if case_json.is_file()
        else None
    )
    ref_rows = recorded_rows(ref_dir)
    strut_rows = recorded_rows(strut_dir)
    full_rows = max(ref_rows, strut_rows, transient_steps(case_data or {}))

    strut_failure = _STRUT_NONCONVERGED_RE.search(log_text)
    strut_progress = _engine_progress(strut_rows, full_rows)
    if strut_failure:
        strut = "nonconverged: " + strut_failure.group(0).replace("ABORT:", "").strip()
    elif returncode is None:
        strut = "timeout"
    elif strut_progress:
        strut = strut_progress
    else:
        strut = "ok"
    strut_complete = strut == "ok"

    step_failures = len(_OPENSEES_STEP_FAILED_RE.findall(opensees_text))
    opensees_progress = _engine_progress(ref_rows, full_rows)
    opensees_complete = opensees_progress is None
    opensees = opensees_progress or "ok"
    if step_failures:
        # The example scripts retry failed steps with other algorithms.
        opensees += (
            f" ({step_failures} failed steps{' retried' if opensees_complete else ''})"
        )

    failures: list[str] = []
    worst: float | None = None
    parity = "n/a"
    if case_data is not None and ref_rows and strut_rows:
        try:
            failures, worst = compare_with_ratio(case_data, ref_dir, strut_dir)
        except (OSError, ValueError) as exc:
            notes.append(f"compare error: {exc}")
        else:
            parity = "fail" if failures else "pass"

    if not (strut_complete and opensees_complete) and returncode is not None:
        status = "NONCONVERGED"
    elif parity == "pass" and returncode == 0:
        status = "PASS"
    elif parity == "fail":
        status = "FAIL"
    else:
        status = "ERROR"
        tail = [line for line in log_text.strip().splitlines() if line.strip()][-3:]
        notes.extend(tail)

    first_failure = next((f.strip() for f in failures if not f.startswith(" ")), "")
    detail = next((f.strip() for f in failures if f.startswith(" ")), "")
    if detail:
        first_failure = f"{first_failure}: {detail}"
    return RunResult(
        case=case_name,
        seed=seed,
        pga_scale=pga_scale,
        motions=motions,
        status=status,
        parity=parity,
        worst_ratio=worst,
        opensees=opensees,
        strut=strut,
        first_failure=first_failure,
        seconds=seconds,
        run_dir=str(run_dir),
        notes=notes,
    )


# --------------------------------------------------------------------------- #
# Reporting and CLI
# --------------------------------------------------------------------------- #


def parse_seeds(text: str) -> list[int]:
    seeds: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_text, hi_text = part.split("-", 1)
            lo, hi = int(lo_text), int(hi_text)
            if lo < 0 or hi < lo:
                raise argparse.ArgumentTypeError(f"bad seed range {part!r}")
            seeds.extend(range(lo, hi + 1))
        else:
            seeds.append(int(part))
    if not seeds:
        raise argparse.ArgumentTypeError("no seeds given")
    return seeds


def parse_floats(text: str) -> list[float]:
    values = [float(part) for part in text.split(",") if part.strip()]
    if not values or any(v <= 0.0 for v in values):
        raise argparse.ArgumentTypeError(f"expected positive numbers, got {text!r}")
    return values


def _pga_text(motions: list[MotionInfo]) -> str:
    shown = [m for m in motions if m.referenced] or motions
    return ", ".join(f"{Path(m.files[0]).stem}={m.measured_pga:.4g}" for m in shown)


def markdown_table(results: list[RunResult]) -> str:
    lines = [
        "| case | seed | PGA scale | PGA | status | worst tol. ratio | OpenSees | Strut | first mismatch |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        pga = _pga_text(r.motions)
        ratio = "-" if r.worst_ratio is None else f"{r.worst_ratio:.3g}"
        mismatch = (r.first_failure or "; ".join(r.notes) or "").replace("|", "\\|")
        lines.append(
            f"| {r.case} | {r.seed} | {r.pga_scale:g} | {pga} | {r.status} | {ratio} "
            f"| {r.opensees} | {r.strut} | {mismatch[:160]} |"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("\n\n", 1)[1],
    )
    parser.add_argument(
        "cases",
        nargs="+",
        help="direct-Tcl parity case names (tests/validation/<name>) or manifest paths",
    )
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds("1-3"))
    parser.add_argument(
        "--pga-scale",
        type=parse_floats,
        default=[1.0],
        help="comma-separated multiples of each real record's PGA (default 1)",
    )
    parser.add_argument("--site-freq-hz", type=float)
    parser.add_argument("--site-damping", type=float)
    parser.add_argument("--envelope", choices=sgm.ENVELOPES)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=DEFAULT_WORK_ROOT,
        help="scratch root inside the repo (the Tcl converter only reads files "
        "under the repo); default .tmp/synthetic_parity_sweep",
    )
    parser.add_argument("--timeout", type=float, default=1800.0, help="seconds per run")
    parser.add_argument(
        "--memory-limit-kb",
        type=int,
        help="run each case under `ulimit -v` with this many KiB",
    )
    parser.add_argument("--json-out", type=Path, help="write all results as JSON")
    parser.add_argument(
        "--markdown-out", type=Path, help="write a Markdown results table"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    work_root = args.work_dir.resolve()
    try:
        work_root.relative_to(REPO_ROOT)
    except ValueError:
        raise SystemExit(
            f"--work-dir must be inside the repo ({REPO_ROOT}): the Tcl converter "
            "refuses file reads outside the repo"
        ) from None
    overrides = {}
    if args.site_freq_hz is not None:
        overrides["site_freq_hz"] = args.site_freq_hz
    if args.site_damping is not None:
        overrides["site_damping"] = args.site_damping
    if args.envelope is not None:
        overrides["envelope"] = args.envelope

    manifests = [resolve_manifest(case) for case in args.cases]
    opensees_binary = resolve_opensees_binary()
    results: list[RunResult] = []
    for manifest in manifests:
        for pga_scale in args.pga_scale:
            for seed in args.seeds:
                result = run_one(
                    manifest,
                    work_root,
                    seed,
                    pga_scale,
                    overrides,
                    opensees_binary,
                    args.timeout,
                    args.memory_limit_kb,
                )
                results.append(result)
                ratio = (
                    "-" if result.worst_ratio is None else f"{result.worst_ratio:.3g}"
                )
                pga = _pga_text(result.motions)
                print(
                    f"[{result.case}] seed={seed} pga_scale={pga_scale:g} pga={pga} "
                    f"{result.status} ratio={ratio} opensees={result.opensees} "
                    f"strut={result.strut} ({result.seconds:.0f}s)",
                    flush=True,
                )
                if result.first_failure:
                    print(f"    first mismatch: {result.first_failure}", flush=True)
                for note in result.notes:
                    print(f"    {note}", flush=True)

    table = markdown_table(results)
    print()
    print(table)
    if args.markdown_out:
        args.markdown_out.write_text(table + "\n", encoding="utf-8")
    if args.json_out:
        args.json_out.write_text(
            json.dumps([asdict(r) for r in results], indent=2) + "\n", encoding="utf-8"
        )
    counts: dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    print("\n" + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
    return 0 if all(r.status == "PASS" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
