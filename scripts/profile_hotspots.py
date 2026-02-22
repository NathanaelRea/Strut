#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


WRAPPER_FRAMES = {
    "total",
    "solve",
    "assemble",
    "output",
    "nonlinear_step",
    "transient_step",
}

MEMORY_BOUND_HINTS = {
    "assemble_stiffness",
    "kff_extract",
    "recorders",
    "constraints",
    "model_build_dof_map",
}

COMPUTE_BOUND_HINTS = {
    "nonlinear_iter",
    "solve_linear",
    "solve_nonlinear",
    "factorize",
    "time_series_eval",
}

DEFAULT_TARGET_SPEEDUP = {
    "memory_bound": 1.35,
    "compute_bound": 1.2,
    "mixed": 1.15,
}


def _classify_frame(name: str) -> str:
    if name in MEMORY_BOUND_HINTS:
        return "memory_bound"
    if name in COMPUTE_BOUND_HINTS:
        return "compute_bound"
    if "assemble" in name or "record" in name or "extract" in name:
        return "memory_bound"
    if "solve" in name or "iter" in name or "factor" in name:
        return "compute_bound"
    return "mixed"


def _parse_target_overrides(items: List[str]) -> Dict[str, float]:
    overrides: Dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"invalid target override '{item}', expected frame=speedup")
        frame, speedup_raw = item.split("=", 1)
        frame = frame.strip()
        if not frame:
            raise ValueError(f"invalid target override '{item}', empty frame name")
        speedup = float(speedup_raw)
        if speedup <= 1.0:
            raise ValueError(f"target speedup for '{frame}' must be > 1.0")
        overrides[frame] = speedup
    return overrides


def _compute_frame_times(profile: dict) -> Tuple[Dict[str, int], Dict[str, int], int]:
    frames = profile.get("shared", {}).get("frames", [])
    events = profile.get("profiles", [{}])[0].get("events", [])
    end_value = int(profile.get("profiles", [{}])[0].get("endValue", 0))

    frame_names: Dict[int, str] = {}
    for i, frame in enumerate(frames):
        if isinstance(frame, dict) and isinstance(frame.get("name"), str):
            frame_names[i] = frame["name"]

    stack: List[int] = []
    inclusive_by_id: Dict[int, int] = defaultdict(int)
    exclusive_by_id: Dict[int, int] = defaultdict(int)
    last_at = 0

    for event in events:
        if not isinstance(event, dict):
            continue
        event_type = event.get("type")
        frame_id = event.get("frame")
        at_raw = event.get("at")
        if event_type not in {"O", "C"}:
            continue
        if not isinstance(frame_id, int) or not isinstance(at_raw, (int, float)):
            continue
        at = int(at_raw)
        delta = at - last_at
        if delta < 0:
            delta = 0

        if stack and delta > 0:
            for open_id in stack:
                inclusive_by_id[open_id] += delta
            exclusive_by_id[stack[-1]] += delta

        if event_type == "O":
            stack.append(frame_id)
        else:
            if stack and stack[-1] == frame_id:
                stack.pop()
            else:
                for idx in range(len(stack) - 1, -1, -1):
                    if stack[idx] == frame_id:
                        del stack[idx]
                        break
        last_at = at

    if end_value <= 0:
        end_value = last_at

    inclusive: Dict[str, int] = {}
    exclusive: Dict[str, int] = {}
    for frame_id, value in inclusive_by_id.items():
        name = frame_names.get(frame_id)
        if name:
            inclusive[name] = inclusive.get(name, 0) + value
    for frame_id, value in exclusive_by_id.items():
        name = frame_names.get(frame_id)
        if name:
            exclusive[name] = exclusive.get(name, 0) + value
    return inclusive, exclusive, end_value


def _build_report(
    inclusive: Dict[str, int],
    exclusive: Dict[str, int],
    total_us: int,
    top: int,
    target_overrides: Dict[str, float],
) -> dict:
    rows = []
    names = set(inclusive.keys()) | set(exclusive.keys())
    for name in names:
        exc_us = int(exclusive.get(name, 0))
        inc_us = int(inclusive.get(name, 0))
        share = (exc_us / total_us) if total_us > 0 else 0.0
        frame_class = _classify_frame(name)
        rows.append(
            {
                "frame": name,
                "exclusive_us": exc_us,
                "inclusive_us": inc_us,
                "exclusive_share": share,
                "class": frame_class,
            }
        )

    rows.sort(key=lambda row: row["exclusive_us"], reverse=True)

    targets = []
    for row in rows:
        if row["frame"] in WRAPPER_FRAMES:
            continue
        if row["exclusive_us"] <= 0:
            continue
        speedup = target_overrides.get(
            row["frame"], DEFAULT_TARGET_SPEEDUP[row["class"]]
        )
        end_to_end_gain = row["exclusive_share"] * (1.0 - (1.0 / speedup))
        target = dict(row)
        target["target_speedup"] = speedup
        target["projected_end_to_end_gain"] = end_to_end_gain
        targets.append(target)
        if len(targets) >= top:
            break

    modeled_gain = sum(t["projected_end_to_end_gain"] for t in targets)
    modeled_total_speedup = (
        1.0 / (1.0 - modeled_gain) if modeled_gain < 0.999999 else float("inf")
    )

    return {
        "total_us": total_us,
        "ranked_frames": rows,
        "targets": targets,
        "modeled_gain_fraction": modeled_gain,
        "modeled_total_speedup": modeled_total_speedup,
    }


def _to_markdown(report: dict) -> str:
    total_ms = report["total_us"] / 1000.0
    lines = [
        "# Hotspot Report",
        "",
        f"- Total profile time: **{total_ms:.3f} ms**",
        f"- Modeled end-to-end gain (targets): **{report['modeled_gain_fraction'] * 100.0:.2f}%**",
        f"- Modeled total speedup (targets): **{report['modeled_total_speedup']:.3f}x**",
        "",
        "## Top Targets",
        "",
        "| Rank | Frame | Class | Exclusive (ms) | Inclusive (ms) | Exclusive Share | Target Speedup | Modeled E2E Gain |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for i, row in enumerate(report["targets"], start=1):
        lines.append(
            "| {rank} | `{frame}` | {klass} | {exc:.3f} | {inc:.3f} | {share:.2f}% | {spd:.2f}x | {gain:.2f}% |".format(
                rank=i,
                frame=row["frame"],
                klass=row["class"],
                exc=row["exclusive_us"] / 1000.0,
                inc=row["inclusive_us"] / 1000.0,
                share=row["exclusive_share"] * 100.0,
                spd=row["target_speedup"],
                gain=row["projected_end_to_end_gain"] * 100.0,
            )
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank Speedscope hotspots and produce optimization targets."
    )
    parser.add_argument("profile", type=Path, help="Path to .speedscope.json")
    parser.add_argument("--top", type=int, default=5, help="Target hotspot count")
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        help="Override target speedup as frame=speedup; repeatable",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Optional output JSON report path",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=None,
        help="Optional output Markdown report path",
    )
    args = parser.parse_args()

    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    target_overrides = _parse_target_overrides(args.target)
    inclusive, exclusive, total_us = _compute_frame_times(profile)
    report = _build_report(
        inclusive=inclusive,
        exclusive=exclusive,
        total_us=total_us,
        top=args.top,
        target_overrides=target_overrides,
    )

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.out_md is not None:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(_to_markdown(report), encoding="utf-8")

    print(f"Profile: {args.profile}")
    print(f"Total: {report['total_us'] / 1000.0:.3f} ms")
    print("Top targets:")
    for idx, row in enumerate(report["targets"], start=1):
        print(
            f"  {idx}. {row['frame']} | class={row['class']} | "
            f"exc={row['exclusive_us'] / 1000.0:.3f} ms ({row['exclusive_share'] * 100.0:.2f}%) | "
            f"target={row['target_speedup']:.2f}x | "
            f"modeled_e2e_gain={row['projected_end_to_end_gain'] * 100.0:.2f}%"
        )
    print(
        "Modeled combined speedup: "
        f"{report['modeled_total_speedup']:.3f}x "
        f"({report['modeled_gain_fraction'] * 100.0:.2f}% total gain)"
    )


if __name__ == "__main__":
    main()
