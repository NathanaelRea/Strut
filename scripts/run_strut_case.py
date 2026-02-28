#!/usr/bin/env python3
import argparse
import json
import os
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
import sys


def run(cmd, env=None, verbose=False):
    if verbose:
        print("+", " ".join(cmd))
    subprocess.check_call(cmd, env=env)


def _resolve_optional_repo_path(path_text, repo_root: Path):
    if not isinstance(path_text, str) or not path_text:
        return None
    path_obj = Path(path_text)
    if path_obj.is_absolute():
        return path_obj
    return (repo_root / path_obj).resolve()


def _resolve_values_path(
    values_path: str, case_json_path: Path, case_data: dict, repo_root: Path
):
    src = Path(values_path)
    candidates = []
    if src.is_absolute():
        candidates.append(src)
    else:
        case_dir = case_json_path.parent
        candidates.append((case_dir / src).resolve())
        candidates.append((repo_root / src).resolve())

        source_example = _resolve_optional_repo_path(
            case_data.get("source_example"), repo_root
        )
        if source_example is not None:
            candidates.append((source_example.parent / src).resolve())

        source_doc = _resolve_optional_repo_path(case_data.get("source_doc"), repo_root)
        if source_doc is not None:
            candidates.append((source_doc.parent / src).resolve())

        migration = case_data.get("migration", {})
        ground_motion = (
            migration.get("ground_motion", {}) if isinstance(migration, dict) else {}
        )
        source_file = _resolve_optional_repo_path(
            ground_motion.get("source_file"), repo_root
        )
        if source_file is not None:
            candidates.append((source_file.parent / src).resolve())

        candidates.append(src)

    resolved = next((p for p in candidates if p.exists()), None)
    if resolved is None:
        checked = ", ".join(str(p) for p in candidates)
        raise SystemExit(
            f"Path time_series values_path not found for Mojo solver: {values_path}; checked: {checked}"
        )
    return resolved


def _normalize_time_series_paths(
    case_data: dict, case_json_path: Path, repo_root: Path
):
    changed = False

    def _normalize_list(ts_list):
        nonlocal changed
        if not isinstance(ts_list, list):
            return
        for ts in ts_list:
            if not isinstance(ts, dict):
                continue
            ts_type = ts.get("type")
            if ts_type not in ("Path", "PathFile"):
                continue
            if "values" in ts:
                continue
            values_path = ts.get("values_path", ts.get("path"))
            if not isinstance(values_path, str) or not values_path:
                continue
            resolved = _resolve_values_path(
                values_path, case_json_path, case_data, repo_root
            )
            resolved_text = str(resolved)
            if ts.get("values_path") != resolved_text:
                ts["values_path"] = resolved_text
                changed = True
            if "path" in ts:
                del ts["path"]
                changed = True

    _normalize_list(case_data.get("time_series"))

    analysis = case_data.get("analysis", {})
    if isinstance(analysis, dict):
        stages = analysis.get("stages", [])
        if isinstance(stages, list):
            for stage in stages:
                if isinstance(stage, dict):
                    _normalize_list(stage.get("time_series"))

    return changed


def _warn_zero_length_node_separation(case_data: dict) -> None:
    model = case_data.get("model", {})
    ndm = int(model.get("ndm", 0))
    if ndm not in (2, 3):
        return

    node_by_id = {}
    for node in case_data.get("nodes", []):
        if not isinstance(node, dict) or "id" not in node:
            continue
        node_by_id[int(node["id"])] = node

    for elem in case_data.get("elements", []):
        if not isinstance(elem, dict) or elem.get("type") != "zeroLength":
            continue
        nodes = elem.get("nodes", [])
        if not isinstance(nodes, list) or len(nodes) != 2:
            continue
        node_i = node_by_id.get(int(nodes[0]))
        node_j = node_by_id.get(int(nodes[1]))
        if node_i is None or node_j is None:
            continue
        x1 = float(node_i.get("x", 0.0))
        y1 = float(node_i.get("y", 0.0))
        z1 = float(node_i.get("z", 0.0))
        x2 = float(node_j.get("x", 0.0))
        y2 = float(node_j.get("y", 0.0))
        z2 = float(node_j.get("z", 0.0))
        dx = x1 - x2
        dy = y1 - y2
        dz = z1 - z2 if ndm == 3 else 0.0
        length = math.sqrt(dx * dx + dy * dy + dz * dz)
        v1 = math.sqrt(x1 * x1 + y1 * y1 + z1 * z1)
        v2 = math.sqrt(x2 * x2 + y2 * y2 + z2 * z2)
        if length > 1.0e-6 * max(v1, v2):
            print(
                f"WARNING ZeroLength::setDomain(): Element {elem['id']} has L= {length}, which is greater than the tolerance",
                file=sys.stderr,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit("uv executable not found on PATH; required to run solver.")

    verbose = os.getenv("STRUT_VERBOSE") == "1"
    solver_bin = os.getenv("STRUT_MOJO_BIN")
    if solver_bin:
        solver_path = Path(solver_bin)
    else:
        solver_path = repo_root / "build" / "strut" / "strut"

    rebuild = not solver_path.exists()
    if not rebuild and not solver_bin:
        src_dir = repo_root / "src" / "mojo"
        latest_src = max(
            (p.stat().st_mtime for p in src_dir.rglob("*.mojo")),
            default=0.0,
        )
        if solver_path.stat().st_mtime < latest_src:
            rebuild = True

    if rebuild and not solver_bin:
        solver_path.parent.mkdir(parents=True, exist_ok=True)
        run(
            [
                uv,
                "run",
                "mojo",
                "build",
                str(repo_root / "src" / "mojo" / "strut.mojo"),
                "-o",
                str(solver_path),
            ],
            verbose=verbose,
        )

    input_path = Path(args.input).resolve()
    case_data = json.loads(input_path.read_text(encoding="utf-8"))
    _warn_zero_length_node_separation(case_data)
    normalized_input = input_path
    tmp_path = None
    if _normalize_time_series_paths(case_data, input_path, repo_root):
        tmp = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            prefix="strut_case_resolved_",
            delete=False,
            encoding="utf-8",
        )
        try:
            tmp.write(json.dumps(case_data))
            tmp.flush()
        finally:
            tmp.close()
        tmp_path = Path(tmp.name)
        normalized_input = tmp_path
        if verbose:
            print(f"+ normalized time_series values_path(s) in {tmp_path}")

    try:
        run(
            [
                str(solver_path),
                "--input",
                str(normalized_input),
                "--output",
                args.output,
            ],
            verbose=verbose,
        )
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
