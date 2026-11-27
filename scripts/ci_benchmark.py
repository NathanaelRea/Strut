#!/usr/bin/env python3
"""Benchmark a pull request against main on the same machine.

Timings from different GitHub runners differ by tens of percent, so stored
numbers cannot serve as a baseline. This runs both solver builds on one
machine instead, alternating them run by run, and compares their medians.

`prepare` writes the suite: generated models sized to take about 0.1-2 s each
and chosen to cover the element, section, transformation, constraint, solver
and analysis paths, plus the validation cases that run without pinned ground
motions. `compare` times the timed cases on both binaries, checks that every
case's outputs match, and writes a markdown table.

`compare` fails when the candidate crashes on a case the baseline runs, or when
a timed case is slower than `--max-regression-pct` both times it is measured. A
flagged case is measured again with twice the repeats first, so one noisy batch
does not fail the build. Output differences are reported but fail only with
`--fail-on-diff`, because fixes change results on purpose; parity against
OpenSees is checked by the test suite.

Examples:
  uv run scripts/ci_benchmark.py prepare
  uv run scripts/ci_benchmark.py compare --baseline build/baseline/strut --candidate build/strut/strut
  uv run scripts/ci_benchmark.py compare --baseline A --candidate B --only megatall_slice_sparse_t4
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import ab_compare  # noqa: E402

# The direct-Tcl converter only reads inside the repo, so the suite lives there.
DEFAULT_SUITE_DIR = REPO_ROOT / ".tmp" / "ci_benchmark" / "suite"
MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True)
class SyntheticCase:
    name: str
    covers: str
    args: tuple[str, ...]
    threads: int = 1


@dataclass(frozen=True)
class MegatallSliceCase:
    name: str
    covers: str
    z_min: float
    z_max: float
    transient_steps: int
    system: str
    threads: int = 1
    # Several cases can time one input at different thread counts.
    input_name: str = ""


@dataclass(frozen=True)
class ValidationCase:
    name: str
    covers: str
    threads: int = 1


SYNTHETIC_CASES = (
    SyntheticCase(
        "rc2d_force_fiber",
        "forceBeamColumn2d fiber, Linear, gravity + Newmark, Newton, BandGeneral",
        ("rc_frame2d", "--bays", "4", "--stories", "8", "--steps", "150"),
    ),
    SyntheticCase(
        "rc2d_force_corotational",
        "forceBeamColumn2d fiber, Corotational",
        (
            "rc_frame2d",
            "--bays",
            "4",
            "--stories",
            "8",
            "--steps",
            "100",
            "--geom-transf",
            "Corotational",
        ),
    ),
    SyntheticCase(
        "rc2d_disp_pdelta_modnewton",
        "dispBeamColumn2d fiber, PDelta, nodal gravity, ModifiedNewton",
        (
            "rc_frame2d",
            "--bays",
            "4",
            "--stories",
            "8",
            "--steps",
            "150",
            "--element",
            "dispBeamColumn2d",
            "--gravity",
            "nodal",
            "--geom-transf",
            "PDelta",
            "--algorithm",
            "ModifiedNewton",
        ),
    ),
    SyntheticCase(
        "rc3d_force_diaphragm",
        "forceBeamColumn3d biaxial fiber, rigidDiaphragm, Transformation",
        (
            "rc_frame3d",
            "--bays",
            "2",
            "--bays-z",
            "2",
            "--stories",
            "4",
            "--steps",
            "50",
            "--rigid-diaphragm",
        ),
    ),
    SyntheticCase(
        "rc3d_disp_fiber",
        "dispBeamColumn3d biaxial fiber, nodal gravity",
        (
            "rc_frame3d",
            "--bays",
            "2",
            "--bays-z",
            "2",
            "--stories",
            "4",
            "--steps",
            "50",
            "--element",
            "dispBeamColumn3d",
            "--gravity",
            "nodal",
        ),
    ),
    SyntheticCase(
        "elastic2d_transient_linear_band",
        "elasticBeamColumn2d, transient_linear, BandGeneral (assembly + banded solve)",
        ("elastic_frame2d", "--bays", "30", "--stories", "30", "--steps", "500"),
    ),
    SyntheticCase(
        "elastic2d_transient_nonlinear_sparse",
        "elasticBeamColumn2d, transient_nonlinear driver, SparseSYM",
        (
            "elastic_frame2d",
            "--bays",
            "40",
            "--stories",
            "40",
            "--steps",
            "200",
            "--nonlinear-driver",
            "--system",
            "SparseSYM",
        ),
    ),
    SyntheticCase(
        "shell_wall_sparse",
        "shell (ElasticMembranePlate), transient_linear, SparseSYM",
        (
            "shell_wall",
            "--bays",
            "40",
            "--stories",
            "40",
            "--steps",
            "100",
            "--system",
            "SparseSYM",
        ),
    ),
)

MEGATALL_CASES = (
    MegatallSliceCase(
        "megatall_slice_umfpack",
        "mega-tall stories: dispBeamColumn3d, LayeredShell ConcreteS, truss, "
        "rigidDiaphragm; static + transient nonlinear, UmfPack",
        z_min=0.0,
        z_max=12.0,
        transient_steps=20,
        system="UmfPack",
    ),
    MegatallSliceCase(
        "megatall_slice_sparse_t1",
        "mega-tall stories, SparseSYM, 1 thread",
        z_min=0.0,
        z_max=30.0,
        transient_steps=10,
        system="SparseSYM",
        input_name="megatall_slice_sparse",
    ),
    MegatallSliceCase(
        "megatall_slice_sparse_t4",
        "mega-tall stories, SparseSYM, 4 threads (threaded assembly + factorization)",
        z_min=0.0,
        z_max=30.0,
        transient_steps=10,
        system="SparseSYM",
        threads=4,
        input_name="megatall_slice_sparse",
    ),
)

TIMED_VALIDATION_CASES = (
    ValidationCase(
        "opensees_example_ex5_frame2d_analyze_dynamic_sine_uniform",
        "OpenSees Ex5 RC/W fiber frame, sine transient",
    ),
    ValidationCase(
        "opensees_example_ex6_genericframe2d_analyze_dynamic_eq_multiplesupport",
        "OpenSees Ex6 frame, multiple-support excitation",
    ),
    ValidationCase(
        "opensees_example_simply_supported_beam_solid",
        "quad solid elements, ProfileSPD static + transient",
    ),
    ValidationCase(
        "opensees_example_ex1a_canti2d_push",
        "static pushover, DisplacementControl",
    ),
)


# ---------------------------------------------------------------- prepare


def _write_case(suite_dir: Path, name: str, data: dict) -> Path:
    path = suite_dir / name / "input.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _prepare_synthetic(case: SyntheticCase, suite_dir: Path) -> Path:
    path = suite_dir / case.name / "input.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_DIR / "gen_synthetic_case.py"),
            *case.args,
            "--name",
            case.name,
            "--output",
            str(path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    return path


def _megatall_source() -> dict:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "strut_setup_reference_data", SCRIPT_DIR / "setup_reference_data.py"
    )
    assert spec is not None and spec.loader is not None
    setup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(setup)
    # The mega-tall model is generated from tables in the repo, never downloaded.
    setup.megatall_bundle().ensure_bundle()
    import megatall_slice

    return megatall_slice.load_source(None)


def _prepare_megatall(
    case: MegatallSliceCase, suite_dir: Path, source: dict, written: dict[str, Path]
) -> Path:
    input_name = case.input_name or case.name
    if input_name in written:
        return written[input_name]
    import megatall_slice

    data = megatall_slice.build_slice(
        source,
        case.z_min,
        case.z_max,
        types=None,
        keep_diaphragms=True,
        lateral=0.05,
        load_scale=1.0,
        steps=4,
        name=input_name,
        record_every=50,
        transient_steps=case.transient_steps,
    )
    stages = data["analysis"].get("stages") or [{"analysis": data["analysis"]}]
    for stage in stages:
        stage["analysis"]["system"] = case.system
    path = _write_case(suite_dir, input_name, data)
    written[input_name] = path
    return path


def _validation_names() -> list[str]:
    names = []
    for path in ab_compare._validation_case_paths(include_disabled=False):
        data = json.loads(path.read_text(encoding="utf-8"))
        # The full mega-tall model needs far more time and memory than a runner has.
        if data.get("benchmark_size") == "large":
            continue
        names.append(path.parent.name)
    return names


def cmd_prepare(args: argparse.Namespace) -> int:
    suite_dir: Path = args.suite.resolve()
    if suite_dir.exists():
        shutil.rmtree(suite_dir)
    suite_dir.mkdir(parents=True)
    entries = []

    for case in SYNTHETIC_CASES:
        path = _prepare_synthetic(case, suite_dir)
        entries.append(_entry(case.name, path, case.covers, case.threads, True))
        print(f"  synthetic   {case.name}")

    source = _megatall_source()
    written: dict[str, Path] = {}
    for case in MEGATALL_CASES:
        path = _prepare_megatall(case, suite_dir, source, written)
        entries.append(_entry(case.name, path, case.covers, case.threads, True))
        print(f"  megatall    {case.name}")

    timed = {case.name: case for case in TIMED_VALIDATION_CASES}
    skipped = []
    validation_dir = suite_dir / "_validation"
    names = _validation_names()
    # Capped subprocesses, as in ab_compare: converting some live-replay scripts
    # statically needs more memory than a runner has.
    with ThreadPoolExecutor(max_workers=4) as pool:
        outcomes = list(
            pool.map(
                lambda name: ab_compare._resolve_in_subprocess(
                    ab_compare._resolve_named_case(name), validation_dir
                ),
                names,
            )
        )
    for name, outcome in zip(names, outcomes):
        if isinstance(outcome, str):
            reason = outcome
            # Name the missing file, not the scratch path it was looked up in.
            reason = re.sub(r'"[^"]*/([^"/]+)"', r'"\1"', reason).split(" at /")[0]
            skipped.append({"name": name, "reason": reason})
            continue
        _, path = outcome
        spec = timed.get(name)
        entries.append(
            _entry(
                name,
                path,
                spec.covers if spec else "",
                spec.threads if spec else 1,
                spec is not None,
            )
        )
    missing = sorted(set(timed) - {entry["name"] for entry in entries})
    if missing:
        raise SystemExit(f"timed validation cases did not resolve: {missing}")
    print(f"  validation  {len(entries) - len(SYNTHETIC_CASES) - len(MEGATALL_CASES)}")

    manifest = {"cases": entries, "skipped": skipped}
    (suite_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Wrote {suite_dir / MANIFEST_NAME}")
    return 0


def _entry(name: str, path: Path, covers: str, threads: int, timed: bool) -> dict:
    return {
        "name": name,
        "input": str(path),
        "covers": covers,
        "threads": threads,
        "timed": timed,
    }


# ---------------------------------------------------------------- compare


@dataclass
class Side:
    times_us: list[int] = field(default_factory=list)
    error: str = ""

    @property
    def median_us(self) -> float | None:
        return statistics.median(self.times_us) if self.times_us else None


@dataclass
class Result:
    entry: dict
    base: Side = field(default_factory=Side)
    cand: Side = field(default_factory=Side)
    status: str = "OK"
    detail: str = ""
    max_rel_diff: float | None = None
    regressed: bool = False

    @property
    def delta_pct(self) -> float | None:
        base, cand = self.base.median_us, self.cand.median_us
        if not base or cand is None:
            return None
        return (cand / base - 1.0) * 100.0


def _run_once(
    binary: Path, entry: dict, out_dir: Path, timeout: float
) -> tuple[int | None, str]:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    env = dict(os.environ, STRUT_THREADS=str(entry["threads"]))
    try:
        proc = subprocess.run(
            [str(binary), "--input", entry["input"], "--output", str(out_dir)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return None, f"timeout after {timeout:.0f}s"
    if proc.returncode != 0:
        lines = [line for line in proc.stdout.splitlines() if "ABORT" in line]
        lines = lines or proc.stdout.strip().splitlines()[-1:]
        return None, (lines[0] if lines else f"exit {proc.returncode}")[:160]
    timing = out_dir / "analysis_time_us.txt"
    if not timing.exists():
        return None, "solver wrote no analysis_time_us.txt"
    return int(timing.read_text().strip()), ""


def _measure(
    result: Result,
    baseline: Path,
    candidate: Path,
    work: Path,
    repeat: int,
    warmup: int,
    timeout: float,
) -> None:
    """Alternate the two binaries so drift on the machine hits both alike."""
    sides = (
        (baseline, result.base, work / "base"),
        (candidate, result.cand, work / "cand"),
    )
    result.base.times_us.clear()
    result.cand.times_us.clear()
    for index in range(warmup + repeat):
        order = sides if index % 2 == 0 else sides[::-1]
        for binary, side, out_dir in order:
            if side.error:
                continue
            elapsed, error = _run_once(binary, result.entry, out_dir, timeout)
            if error:
                side.error = error
            elif index >= warmup:
                side.times_us.append(elapsed)


def _classify(result: Result, work: Path, rtol: float, atol: float) -> None:
    base_err, cand_err = result.base.error, result.cand.error
    if base_err and cand_err:
        result.status, result.detail = "SKIP", f"fails on both: {base_err}"
    elif cand_err:
        result.status, result.detail = "CAND-FAIL", cand_err
    elif base_err:
        result.status, result.detail = "NEW", f"baseline fails: {base_err}"
    else:
        ok, worst, detail = ab_compare._compare_outputs(
            work / "base", work / "cand", rtol, atol
        )
        result.max_rel_diff = worst
        result.status, result.detail = ("OK", "") if ok else ("DIFF", detail)


def _is_regression(result: Result, args: argparse.Namespace) -> bool:
    if not result.entry["timed"] or result.status not in ("OK", "DIFF"):
        return False
    base, delta = result.base.median_us, result.delta_pct
    if base is None or delta is None:
        return False
    return base / 1000.0 >= args.gate_min_ms and delta > args.max_regression_pct


def _fmt_ms(value_us: float | None) -> str:
    if value_us is None:
        return "-"
    ms = value_us / 1000.0
    return f"{ms:.0f}" if ms >= 100 else f"{ms:.1f}" if ms >= 10 else f"{ms:.2f}"


def _fmt_delta(result: Result, args: argparse.Namespace) -> str:
    delta = result.delta_pct
    if delta is None:
        return "-"
    text = f"{delta:+.1f}%"
    if result.regressed:
        return f"**{text}** 🔴"
    if delta < -args.max_regression_pct:
        return f"{text} 🟢"
    return text


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def _markdown(
    results: list[Result], skipped: list[dict], args: argparse.Namespace
) -> str:
    timed = [r for r in results if r.entry["timed"]]
    checked = [r for r in results if not r.entry["timed"]]
    failing = [r for r in results if r.regressed or r.status == "CAND-FAIL"]
    diffs = [r for r in results if r.status == "DIFF"]

    ratios = [
        r.cand.median_us / r.base.median_us
        for r in timed
        if r.base.median_us and r.cand.median_us
    ]
    geomean = statistics.geometric_mean(ratios) if ratios else None

    lines = [args.title, ""]
    verdict = "❌ failing" if failing else "✅ no regressions"
    summary = f"**{verdict}**"
    if geomean is not None:
        summary += f" · geometric mean {(geomean - 1.0) * 100.0:+.1f}% across {len(ratios)} timed cases"
    lines += [summary, ""]
    lines += [
        "| case | covers | threads | base ms | PR ms | Δ | outputs |",
        "|---|---|--:|--:|--:|--:|---|",
    ]
    for r in timed:
        outputs = {
            "OK": "same",
            "DIFF": "⚠️ differ",
            "CAND-FAIL": f"❌ {r.detail}",
            "NEW": "new in PR",
            "SKIP": f"skipped: {r.detail}",
        }[r.status].replace("|", "\\|")
        lines.append(
            f"| `{r.entry['name']}` | {r.entry['covers']} | {r.entry['threads']} "
            f"| {_fmt_ms(r.base.median_us)} | {_fmt_ms(r.cand.median_us)} "
            f"| {_fmt_delta(r, args)} | {outputs} |"
        )
    lines.append("")

    same = sum(1 for r in checked if r.status == "OK")
    lines.append(
        f"Outputs also compared on {len(checked)} short validation cases: {same} identical."
    )
    problem_rows = [r for r in checked if r.status != "OK"]
    skipped_rows = [r for r in problem_rows if r.status == "SKIP"]
    other_rows = [r for r in problem_rows if r.status != "SKIP"]
    for r in other_rows:
        lines.append(f"- `{r.entry['name']}`: **{r.status}** {r.detail}")
    if diffs:
        lines.append("")
        lines.append(
            "⚠️ Output differences are expected when a PR changes results on purpose; "
            "otherwise they are worth a look."
        )
    if skipped_rows or skipped:
        lines.append("")
        lines.append("<details><summary>Skipped cases</summary>")
        lines.append("")
        for r in skipped_rows:
            lines.append(f"- `{r.entry['name']}`: {r.detail}")
        for item in skipped:
            lines.append(f"- `{item['name']}`: {item['reason']}")
        lines.append("")
        lines.append("</details>")
    lines += [
        "",
        (
            f"<sub>Median of {args.repeat} alternating runs per binary of the solver's analysis "
            f"time, on {_cpu_model()} ({os.cpu_count()} CPUs). Fails when a case of at least "
            f"{args.gate_min_ms:g} ms is more than {args.max_regression_pct:g}% slower twice in a "
            f"row. Shared runners vary by a few percent between runs, so small deltas are "
            f"noise.</sub>"
        ),
    ]
    return "\n".join(lines) + "\n"


def cmd_compare(args: argparse.Namespace) -> int:
    baseline = Path(args.baseline).resolve()
    candidate = Path(args.candidate).resolve()
    for binary in (baseline, candidate):
        if not binary.exists():
            raise SystemExit(f"solver binary not found: {binary}")
    manifest_path = args.suite.resolve() / MANIFEST_NAME
    if not manifest_path.exists():
        raise SystemExit(f"{manifest_path} not found; run `prepare` first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest["cases"]
    if args.only:
        wanted = {name.strip() for name in args.only.split(",") if name.strip()}
        entries = [e for e in entries if e["name"] in wanted]

    results: list[Result] = []
    scratch = REPO_ROOT / ".tmp" / "ci_benchmark"
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="run_", dir=scratch) as tmp:
        work_root = Path(tmp)
        for entry in entries:
            result = Result(entry=entry)
            work = work_root / entry["name"]
            repeat = args.repeat if entry["timed"] else 1
            warmup = args.warmup if entry["timed"] else 0
            _measure(result, baseline, candidate, work, repeat, warmup, args.timeout)
            _classify(result, work, args.rtol, args.atol)
            if _is_regression(result, args):
                first = result.delta_pct
                # Confirm with more runs before failing on a noisy batch.
                _measure(
                    result, baseline, candidate, work, 2 * args.repeat, 1, args.timeout
                )
                _classify(result, work, args.rtol, args.atol)
                result.regressed = _is_regression(result, args)
                sys.stderr.write(
                    f"  recheck   {entry['name']}: {first:+.1f}% then {result.delta_pct:+.1f}%\n"
                )
            results.append(result)
            if entry["timed"] or result.status != "OK":
                delta = result.delta_pct
                delta_text = f"{delta:+6.1f}%" if delta is not None else "      -"
                sys.stderr.write(
                    f"  {'SLOWER' if result.regressed else result.status:10s} {_fmt_ms(result.base.median_us):>8s} -> "
                    f"{_fmt_ms(result.cand.median_us):>8s} ms {delta_text}  {entry['name']}\n"
                )

    markdown = _markdown(results, manifest.get("skipped", []), args)
    print(markdown)
    if args.markdown_out:
        Path(args.markdown_out).write_text(markdown, encoding="utf-8")
    if args.json_out:
        payload = [
            {
                "case": r.entry["name"],
                "timed": r.entry["timed"],
                "threads": r.entry["threads"],
                "status": r.status,
                "regressed": r.regressed,
                "detail": r.detail,
                "baseline_us": r.base.times_us,
                "candidate_us": r.cand.times_us,
                "baseline_median_us": r.base.median_us,
                "candidate_median_us": r.cand.median_us,
                "delta_pct": r.delta_pct,
                "max_rel_diff": r.max_rel_diff,
            }
            for r in results
        ]
        Path(args.json_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    failing = {"CAND-FAIL"} | ({"DIFF"} if args.fail_on_diff else set())
    return 1 if any(r.regressed or r.status in failing for r in results) else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="Write the benchmark suite inputs.")
    prepare.add_argument("--suite", type=Path, default=DEFAULT_SUITE_DIR)
    prepare.set_defaults(func=cmd_prepare)

    compare = sub.add_parser("compare", help="Time a baseline and a candidate binary.")
    compare.add_argument("--baseline", required=True, help="Baseline solver binary.")
    compare.add_argument("--candidate", required=True, help="Candidate solver binary.")
    compare.add_argument("--suite", type=Path, default=DEFAULT_SUITE_DIR)
    compare.add_argument("--only", default="", help="Comma-separated case names.")
    compare.add_argument("--repeat", type=int, default=5, help="Timed runs per binary.")
    compare.add_argument(
        "--warmup", type=int, default=1, help="Untimed runs per binary."
    )
    compare.add_argument("--max-regression-pct", type=float, default=15.0)
    compare.add_argument(
        "--gate-min-ms",
        type=float,
        default=20.0,
        help="Only fail on cases whose baseline takes at least this long.",
    )
    compare.add_argument("--fail-on-diff", action="store_true")
    compare.add_argument("--rtol", type=float, default=1.0e-6)
    compare.add_argument("--atol", type=float, default=1.0e-9)
    compare.add_argument("--timeout", type=float, default=600.0)
    compare.add_argument("--title", default="### Benchmark: PR vs main")
    compare.add_argument("--markdown-out", default="")
    compare.add_argument("--json-out", default="")
    compare.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
