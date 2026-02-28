#!/usr/bin/env python3
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class CaseTiming:
    name: str
    analysis_us: float


@dataclass(frozen=True)
class ComparisonRow:
    name: str
    baseline_us: float
    candidate_us: float
    delta_pct: float


def _load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_case_timings(summary: dict, engine: str) -> Dict[str, CaseTiming]:
    rows: Dict[str, CaseTiming] = {}
    for case in summary.get("cases", []):
        if not isinstance(case, dict):
            continue
        stats = case.get(engine)
        if not isinstance(stats, dict):
            continue
        analysis_us = stats.get("analysis_us")
        if not isinstance(analysis_us, (int, float)):
            continue
        name = case.get("name")
        if not isinstance(name, str) or not name:
            continue
        rows[name] = CaseTiming(name=name, analysis_us=float(analysis_us))
    return rows


def _metadata_mismatch(baseline: dict, candidate: dict) -> List[str]:
    mismatches: List[str] = []
    baseline_meta = baseline.get("metadata", {})
    candidate_meta = candidate.get("metadata", {})
    checks = [
        ("platform.system", baseline_meta.get("platform", {}).get("system"), candidate_meta.get("platform", {}).get("system")),
        ("platform.release", baseline_meta.get("platform", {}).get("release"), candidate_meta.get("platform", {}).get("release")),
        ("platform.machine", baseline_meta.get("platform", {}).get("machine"), candidate_meta.get("platform", {}).get("machine")),
        ("platform.cpu_model", baseline_meta.get("platform", {}).get("cpu_model"), candidate_meta.get("platform", {}).get("cpu_model")),
        ("build.mojo_version", baseline_meta.get("build", {}).get("mojo_version"), candidate_meta.get("build", {}).get("mojo_version")),
        ("runner.engine", baseline_meta.get("runner", {}).get("engine"), candidate_meta.get("runner", {}).get("engine")),
        ("runner.batch_mode", baseline_meta.get("runner", {}).get("batch_mode"), candidate_meta.get("runner", {}).get("batch_mode")),
    ]
    for key, lhs, rhs in checks:
        if lhs != rhs:
            mismatches.append(f"{key}: baseline={lhs!r} candidate={rhs!r}")
    return mismatches


def _compare_cases(
    baseline: Dict[str, CaseTiming], candidate: Dict[str, CaseTiming]
) -> List[ComparisonRow]:
    rows: List[ComparisonRow] = []
    for name in sorted(set(baseline.keys()) & set(candidate.keys())):
        baseline_us = baseline[name].analysis_us
        candidate_us = candidate[name].analysis_us
        if baseline_us == 0.0:
            continue
        delta_pct = ((candidate_us - baseline_us) / baseline_us) * 100.0
        rows.append(
            ComparisonRow(
                name=name,
                baseline_us=baseline_us,
                candidate_us=candidate_us,
                delta_pct=delta_pct,
            )
        )
    return rows


def _parse_required_improvement(items: List[str]) -> Dict[str, float]:
    targets: Dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(
                f"invalid improvement target '{item}', expected CASE=PERCENT"
            )
        case_name, pct_text = item.split("=", 1)
        case_name = case_name.strip()
        if not case_name:
            raise ValueError(f"invalid improvement target '{item}', empty case name")
        pct = float(pct_text)
        if pct < 0.0:
            raise ValueError(
                f"invalid improvement target '{item}', percent must be non-negative"
            )
        targets[case_name] = pct
    return targets


def compare_summaries(
    baseline_summary: dict,
    candidate_summary: dict,
    engine: str,
) -> Tuple[List[ComparisonRow], List[str]]:
    baseline_rows = _extract_case_timings(baseline_summary, engine)
    candidate_rows = _extract_case_timings(candidate_summary, engine)
    rows = _compare_cases(baseline_rows, candidate_rows)
    shared_names = {row.name for row in rows}
    warnings: List[str] = []
    missing_baseline = sorted(set(candidate_rows.keys()) - set(baseline_rows.keys()))
    missing_candidate = sorted(set(baseline_rows.keys()) - set(candidate_rows.keys()))
    if missing_baseline:
        warnings.append(
            "candidate-only cases ignored: " + ", ".join(missing_baseline)
        )
    if missing_candidate:
        warnings.append(
            "baseline-only cases ignored: " + ", ".join(missing_candidate)
        )
    if not shared_names:
        warnings.append("no shared benchmark cases with analysis_us were found")
    return rows, warnings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare benchmark summary.json files and fail on regressions."
    )
    parser.add_argument("baseline", type=Path, help="Baseline benchmark summary.json")
    parser.add_argument("candidate", type=Path, help="Candidate benchmark summary.json")
    parser.add_argument(
        "--engine",
        default="strut",
        choices=("strut", "opensees"),
        help="Engine timing column to compare.",
    )
    parser.add_argument(
        "--max-regression-pct",
        type=float,
        default=5.0,
        help="Maximum allowed regression percentage before failing.",
    )
    parser.add_argument(
        "--min-regression-us",
        type=float,
        default=0.0,
        help="Ignore regressions smaller than this absolute analysis_us delta.",
    )
    parser.add_argument(
        "--require-improvement",
        action="append",
        default=[],
        help="Require a case to improve by at least this percentage: CASE=PERCENT",
    )
    args = parser.parse_args()

    baseline_summary = _load_summary(args.baseline)
    candidate_summary = _load_summary(args.candidate)
    rows, comparison_warnings = compare_summaries(
        baseline_summary, candidate_summary, engine=args.engine
    )
    required_improvement = _parse_required_improvement(args.require_improvement)
    metadata_warnings = _metadata_mismatch(baseline_summary, candidate_summary)

    failures: List[str] = []
    for row in rows:
        delta_us = row.candidate_us - row.baseline_us
        if (
            row.delta_pct > args.max_regression_pct
            and delta_us > args.min_regression_us
        ):
            failures.append(
                "{name}: regression {delta_pct:.2f}% exceeds {pct:.2f}% "
                "and absolute delta {delta_us:.0f} us exceeds {min_us:.0f} us".format(
                    name=row.name,
                    delta_pct=row.delta_pct,
                    pct=args.max_regression_pct,
                    delta_us=delta_us,
                    min_us=args.min_regression_us,
                )
            )
        required_gain = required_improvement.get(row.name)
        if required_gain is None:
            continue
        actual_gain = -row.delta_pct
        if actual_gain + 1.0e-12 < required_gain:
            failures.append(
                f"{row.name}: improvement {actual_gain:.2f}% is below required {required_gain:.2f}%"
            )

    print("Benchmark comparison")
    print(
        "Shared cases: {count} | engine={engine} | max_regression_pct={pct:.2f} | "
        "min_regression_us={delta:.2f}".format(
            count=len(rows),
            engine=args.engine,
            pct=args.max_regression_pct,
            delta=args.min_regression_us,
        )
    )
    for row in sorted(rows, key=lambda item: item.delta_pct, reverse=True):
        delta_us = row.candidate_us - row.baseline_us
        print(
            "{name}: baseline={baseline:.0f} us candidate={candidate:.0f} us "
            "delta_us={delta_us:+.0f} delta_pct={delta_pct:+.2f}%".format(
                name=row.name,
                baseline=row.baseline_us,
                candidate=row.candidate_us,
                delta_us=delta_us,
                delta_pct=row.delta_pct,
            )
        )

    for warning in comparison_warnings:
        print(f"warning: {warning}")
    for warning in metadata_warnings:
        print(f"warning: metadata mismatch {warning}")
    for row in rows:
        delta_us = row.candidate_us - row.baseline_us
        if row.delta_pct > args.max_regression_pct and delta_us <= args.min_regression_us:
            print(
                "warning: ignoring small regression for {name}: {delta_us:.0f} us <= {floor:.0f} us".format(
                    name=row.name,
                    delta_us=delta_us,
                    floor=args.min_regression_us,
                )
            )
    for failure in failures:
        print(f"error: {failure}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
