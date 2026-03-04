#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def run(cmd, env=None, verbose=False):
    if verbose:
        print("+", " ".join(cmd))
    subprocess.check_call(cmd, env=env)


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _case_enabled(case_path: Path) -> bool:
    data = json.loads(case_path.read_text())
    return bool(data.get("enabled", True))


def _compute_combined_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        resolved = path.resolve()
        digest.update(str(resolved).encode("utf-8"))
        digest.update(b"\0")
        digest.update(resolved.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _slugify_tcl_path(path: Path, repo_root: Path) -> str:
    try:
        rel = path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        rel = path.resolve()
    stem = "".join(ch.lower() if ch.isalnum() else "_" for ch in path.stem).strip("_")
    while "__" in stem:
        stem = stem.replace("__", "_")
    digest = hashlib.sha256(str(rel).encode("utf-8")).hexdigest()[:10]
    return f"tcl_{stem or 'case'}_{digest}"


def _direct_tcl_manifest_paths(repo_root: Path) -> list[Path]:
    return sorted(
        (repo_root / "tests" / "validation").glob("*/direct_tcl_case.json")
    )


def _load_direct_tcl_manifest(manifest_path: Path) -> dict:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"invalid direct Tcl manifest: {manifest_path}")
    return data


def _resolve_entry_tcl_from_manifest(manifest_path: Path, repo_root: Path) -> Path:
    data = _load_direct_tcl_manifest(manifest_path)
    raw_path = data.get("entry_tcl")
    if not isinstance(raw_path, str) or not raw_path:
        raise SystemExit(f"direct Tcl manifest missing entry_tcl: {manifest_path}")
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def _resolve_canonical_tcl_case(entry_tcl: Path, repo_root: Path):
    entry_path = entry_tcl.resolve()
    matches = []
    for manifest_path in _direct_tcl_manifest_paths(repo_root):
        if _resolve_entry_tcl_from_manifest(manifest_path, repo_root) != entry_path:
            continue
        matches.append((manifest_path.parent.name, manifest_path.parent))

    if not matches:
        return None
    if len(matches) > 1:
        return None
    return matches[0]


def _is_direct_tcl_manifest(path: Path) -> bool:
    if path.name != "direct_tcl_case.json":
        return False
    data = _load_direct_tcl_manifest(path)
    raw_path = data.get("entry_tcl")
    return isinstance(raw_path, str) and bool(raw_path)


def _reference_output_path(reference_dir: Path, raw_path: str):
    path = Path(raw_path)
    current = reference_dir
    for part in path.parts:
        candidate = current / part
        if candidate.exists():
            current = candidate
            continue
        if not current.exists() or not current.is_dir():
            return None
        matches = [child for child in current.iterdir() if child.name.lower() == part.lower()]
        if len(matches) != 1:
            return None
        current = matches[0]
    return current


def _normalized_recorder_outputs(recorder: dict):
    if recorder.get("parity", True) is False:
        return []
    rec_type = recorder["type"]
    output = recorder.get("output", rec_type)
    if rec_type in ("node_displacement", "node_reaction"):
        nodes = recorder.get("nodes", [])
        if len(nodes) != 1:
            raise SystemExit(f"direct Tcl parity requires single-node recorder: {recorder}")
        return [f"{output}_node{int(nodes[0])}.out"]
    if rec_type == "element_force":
        elements = recorder.get("elements", [])
        if len(elements) != 1:
            raise SystemExit(f"direct Tcl parity requires single-element recorder: {recorder}")
        return [f"{output}_ele{int(elements[0])}.out"]
    if rec_type in ("element_local_force", "element_basic_force", "element_deformation"):
        elements = recorder.get("elements", [])
        if len(elements) != 1:
            raise SystemExit(f"direct Tcl parity requires single-element recorder: {recorder}")
        return [f"{output}_ele{int(elements[0])}.out"]
    if rec_type == "envelope_element_force":
        elements = recorder.get("elements", [])
        if len(elements) != 1:
            raise SystemExit(f"direct Tcl parity requires single-element recorder: {recorder}")
        return [f"{output}_ele{int(elements[0])}.out"]
    if rec_type in ("section_force", "section_deformation"):
        elements = recorder.get("elements", [])
        if len(elements) != 1:
            raise SystemExit(f"direct Tcl parity requires single-element recorder: {recorder}")
        section = recorder.get("section")
        if section is None:
            sections = recorder.get("sections") or []
            if len(sections) != 1:
                raise SystemExit(f"direct Tcl parity requires single section: {recorder}")
            section = sections[0]
        return [f"{output}_ele{int(elements[0])}_sec{int(section)}.out"]
    if rec_type == "drift":
        return [f"{output}_i{int(recorder['i_node'])}_j{int(recorder['j_node'])}.out"]
    raise SystemExit(f"unsupported direct Tcl recorder normalization: {rec_type}")


def _normalize_reference_outputs(case_data: dict | Path, reference_dir: Path):
    if isinstance(case_data, Path):
        data = json.loads(case_data.read_text(encoding="utf-8"))
    else:
        data = case_data
    grouped_recorders: dict[str, list[dict]] = {}
    for recorder in data.get("recorders", []):
        raw_path = recorder.get("raw_path")
        if not raw_path or recorder.get("parity", True) is False:
            continue
        grouped_recorders.setdefault(raw_path, []).append(recorder)

    for raw_path, recorders in grouped_recorders.items():
        targets = []
        for recorder in recorders:
            targets.extend(_normalized_recorder_outputs(recorder))
        if not targets:
            continue

        payloads = [[] for _ in targets]
        group_layout = recorders[0].get("group_layout")
        strip_time = bool(recorders[0].get("include_time"))
        preserve_existing_targets = all(
            (reference_dir / rel_name).exists() for rel_name in targets
        )
        source = _reference_output_path(reference_dir, raw_path)
        if source is None or not source.exists():
            raise SystemExit(f"missing raw OpenSees recorder output: {raw_path}")
        normalized_rows = []
        for line in source.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            parts = line.replace(",", " ").split()
            if (
                group_layout
                and group_layout.get("type") == "envelope_element_force"
                and strip_time
            ):
                normalized_rows.append(parts)
                continue
            if strip_time and parts:
                parts = parts[1:]
            normalized_rows.append(parts)

        if group_layout and group_layout.get("type") == "envelope_element_force":
            layout_elements = group_layout.get("elements") or []
            layout_widths = group_layout.get("values_per_element") or []
            width_by_element = dict(zip(layout_elements, layout_widths))
            widths = []
            for recorder in recorders:
                elements = recorder.get("elements") or []
                if len(elements) != 1 or elements[0] not in width_by_element:
                    widths = []
                    break
                widths.append(int(width_by_element[elements[0]]))
            if widths:
                expected = sum(widths)
                paired_payloads = [[] for _ in targets]
                used_paired_layout = False
                for parts in normalized_rows:
                    if len(parts) == expected:
                        if used_paired_layout:
                            raise SystemExit(
                                f"mixed grouped envelope layouts in {raw_path}"
                            )
                        offset = 0
                        for idx, width in enumerate(widths):
                            payloads[idx].append(" ".join(parts[offset : offset + width]))
                            offset += width
                        continue
                    if strip_time and len(parts) == 2 * expected:
                        used_paired_layout = True
                        offset = 0
                        for idx, width in enumerate(widths):
                            segment = parts[offset : offset + 2 * width]
                            paired_payloads[idx].extend(segment[1::2])
                            offset += 2 * width
                        continue
                    if expected != len(parts):
                        if preserve_existing_targets:
                            payloads = []
                            break
                        raise SystemExit(
                            "cannot split grouped recorder output "
                            f"{raw_path}: expected {expected} values, got {len(parts)}"
                        )
                if used_paired_layout and payloads:
                    for idx in range(len(targets)):
                        payloads[idx] = [" ".join(paired_payloads[idx])]
            else:
                group_layout = None
        if not group_layout:
            if len(targets) == 1:
                for parts in normalized_rows:
                    payloads[0].append(" ".join(parts))
            else:
                for parts in normalized_rows:
                    if len(parts) % len(targets) != 0:
                        raise SystemExit(
                            "cannot evenly split grouped recorder output "
                            f"{raw_path}: {len(parts)} values across {len(targets)} targets"
                        )
                    width = len(parts) // len(targets)
                    for idx, _ in enumerate(targets):
                        start = idx * width
                        payloads[idx].append(" ".join(parts[start : start + width]))

        if not payloads:
            continue

        for rel_name, rows in zip(targets, payloads):
            payload = "\n".join(rows)
            if payload:
                payload += "\n"
            (reference_dir / rel_name).write_text(payload, encoding="utf-8")


def _merge_direct_tcl_manifest_metadata(case_data: dict, manifest_path: Path | None) -> dict:
    if manifest_path is None or not manifest_path.exists():
        return case_data
    manifest = _load_direct_tcl_manifest(manifest_path)
    merged = dict(case_data)
    for key in (
        "enabled",
        "status",
        "benchmark_size",
        "parity_tolerance",
        "parity_tolerance_by_recorder",
        "parity_mode",
    ):
        if key in manifest:
            merged[key] = manifest[key]
    return merged


def _write_generated_direct_tcl_case(case_data: dict, case_root: Path) -> Path:
    generated_case_json = case_root / "generated" / "case.json"
    generated_case_json.parent.mkdir(parents=True, exist_ok=True)
    generated_case_json.write_text(
        json.dumps(case_data, indent=2) + "\n", encoding="utf-8"
    )
    return generated_case_json


def _emit_case_tcl(
    repo_root: Path, uv: str, case_json: Path, tcl_out: Path, env: dict, verbose: bool
) -> Path:
    tcl_out.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            uv,
            "run",
            "python",
            str(repo_root / "scripts" / "json_to_tcl.py"),
            str(case_json),
            str(tcl_out),
        ],
        env=env,
        verbose=verbose,
    )
    return tcl_out


def _write_parser_check(case_root: Path) -> None:
    (case_root / ".parser-check").write_text("ok\n", encoding="utf-8")


def _clear_parser_check(case_root: Path) -> None:
    (case_root / ".parser-check").unlink(missing_ok=True)


def _validate_generated_tcl_matches_original(
    *,
    repo_root: Path,
    case_root: Path,
    case_data: dict,
    entry_tcl: Path,
    generated_tcl: Path,
    env: dict,
    verbose: bool,
    refresh_reference: bool,
) -> None:
    import compare_case

    original_reference_dir = case_root / "reference-original"
    generated_reference_dir = case_root / "reference"
    parser_check_path = case_root / ".parser-check"
    needs_original = refresh_reference or not original_reference_dir.exists()
    needs_generated = refresh_reference or not generated_reference_dir.exists()
    needs_compare = refresh_reference or not parser_check_path.exists()
    if not needs_original and not needs_generated and not needs_compare:
        return

    if needs_original:
        ensure_clean_dir(original_reference_dir)
        run(
            [
                str(repo_root / "scripts" / "run_opensees_wine.sh"),
                "--script",
                str(entry_tcl),
                "--output",
                str(original_reference_dir),
            ],
            env=env,
            verbose=verbose,
        )
        _normalize_reference_outputs(case_data, original_reference_dir)

    if needs_generated:
        ensure_clean_dir(generated_reference_dir)
        run(
            [
                str(repo_root / "scripts" / "run_opensees_wine.sh"),
                "--script",
                str(generated_tcl),
                "--output",
                str(generated_reference_dir),
            ],
            env=env,
            verbose=verbose,
        )
        _normalize_reference_outputs(case_data, generated_reference_dir)

    failures = compare_case._compare_output_dirs(
        case_data, original_reference_dir, generated_reference_dir
    )
    if failures:
        _clear_parser_check(case_root)
        raise SystemExit(
            "generated Tcl does not match original Tcl:\n" + "\n".join(failures)
        )
    _write_parser_check(case_root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case_input")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit("uv executable not found on PATH; required to run harness.")

    case_input = Path(args.case_input).resolve()
    entry_tcl = None
    hash_inputs = [case_input]
    if case_input.suffix == ".json":
        if _is_direct_tcl_manifest(case_input):
            data = _load_direct_tcl_manifest(case_input)
            case_name = str(data.get("name") or case_input.parent.name)
            case_root = case_input.parent
            entry_tcl = _resolve_entry_tcl_from_manifest(case_input, repo_root)
            hash_inputs.append(entry_tcl)
            is_tcl = True
        else:
            case_name = case_input.stem
            case_root = repo_root / "tests" / "validation" / case_name
            case_json = case_root / f"{case_name}.json"
            is_tcl = False
    elif case_input.suffix == ".tcl":
        canonical_case = _resolve_canonical_tcl_case(case_input, repo_root)
        if canonical_case is None:
            case_name = _slugify_tcl_path(case_input, repo_root)
            case_root = repo_root / "tests" / "validation" / case_name
        else:
            case_name, case_root = canonical_case
        entry_tcl = case_input
        is_tcl = True
    else:
        raise SystemExit(f"unsupported case input: {case_input}")

    case_root.mkdir(parents=True, exist_ok=True)
    (case_root / "reference").mkdir(parents=True, exist_ok=True)
    (case_root / "strut").mkdir(parents=True, exist_ok=True)

    verbose = os.getenv("STRUT_VERBOSE") == "1"
    env = os.environ.copy()
    case_data = None
    manifest_path = None
    use_generated_direct_tcl = False

    if is_tcl:
        assert entry_tcl is not None
        import tcl_to_strut

        if case_input.suffix == ".json" and _is_direct_tcl_manifest(case_input):
            manifest_path = case_input
        case_data = _merge_direct_tcl_manifest_metadata(
            tcl_to_strut.convert_tcl_to_solver_input(entry_tcl, repo_root),
            manifest_path,
        )
        try:
            case_json = _write_generated_direct_tcl_case(case_data, case_root)
            hash_inputs.append(case_json)
            tcl_out = _emit_case_tcl(
                repo_root,
                uv,
                case_json,
                case_root / "generated" / "model.tcl",
                env,
                verbose,
            )
            use_generated_direct_tcl = True
        except (OSError, ValueError, subprocess.CalledProcessError, SystemExit) as exc:
            if verbose:
                print(f"direct Tcl fallback to original script: {exc}", file=sys.stderr)
            tcl_out = entry_tcl
    else:
        tgt_json = case_root / f"{case_name}.json"
        if case_input != tgt_json:
            shutil.copy2(case_input, tgt_json)
        case_json = tgt_json
        if not _case_enabled(case_json) and os.getenv("STRUT_FORCE_CASE") != "1":
            return

        tcl_out = _emit_case_tcl(
            repo_root,
            uv,
            case_json,
            case_root / "generated" / "model.tcl",
            env,
            verbose,
        )
        case_data = json.loads(case_json.read_text(encoding="utf-8"))

    refresh_reference = os.getenv("STRUT_REFRESH_REFERENCE") == "1"
    ref_hash_file = case_root / "reference" / ".ref_hash"
    if not refresh_reference:
        current_hash = _compute_combined_hash(hash_inputs)
        stored_hash = (
            ref_hash_file.read_text().strip() if ref_hash_file.exists() else None
        )
        if stored_hash != current_hash:
            refresh_reference = True

    if is_tcl and use_generated_direct_tcl:
        assert entry_tcl is not None
        try:
            _validate_generated_tcl_matches_original(
                repo_root=repo_root,
                case_root=case_root,
                case_data=case_data,
                entry_tcl=entry_tcl,
                generated_tcl=tcl_out,
                env=env,
                verbose=verbose,
                refresh_reference=refresh_reference,
            )
        except (OSError, ValueError, subprocess.CalledProcessError, SystemExit) as exc:
            use_generated_direct_tcl = False
            _clear_parser_check(case_root)
            if verbose:
                print(
                    f"direct Tcl fallback to original script after parser-check failure: {exc}",
                    file=sys.stderr,
                )
            tcl_out = entry_tcl
        else:
            ref_hash_file.parent.mkdir(parents=True, exist_ok=True)
            ref_hash_file.write_text(
                _compute_combined_hash(hash_inputs) + "\n", encoding="utf-8"
            )

    if is_tcl and not use_generated_direct_tcl:
        if refresh_reference:
            run(
                [
                    str(repo_root / "scripts" / "run_opensees_wine.sh"),
                    "--script",
                    str(entry_tcl),
                    "--output",
                    str(case_root / "reference"),
                ],
                env=env,
                verbose=verbose,
            )
            _normalize_reference_outputs(case_data, case_root / "reference")
        ref_hash_file.parent.mkdir(parents=True, exist_ok=True)
        ref_hash_file.write_text(
            _compute_combined_hash(hash_inputs) + "\n", encoding="utf-8"
        )
    elif refresh_reference:
        run(
            [
                str(repo_root / "scripts" / "run_opensees_wine.sh"),
                "--script",
                str(tcl_out),
                "--output",
                str(case_root / "reference"),
            ],
            env=env,
            verbose=verbose,
        )
        ref_hash_file.parent.mkdir(parents=True, exist_ok=True)
        ref_hash_file.write_text(
            _compute_combined_hash(hash_inputs) + "\n", encoding="utf-8"
        )
    else:
        ref_hash_file.parent.mkdir(parents=True, exist_ok=True)
        ref_hash_file.write_text(
            _compute_combined_hash(hash_inputs) + "\n", encoding="utf-8"
        )

    strut_cmd = [
        uv,
        "run",
        "python",
        str(repo_root / "scripts" / "run_strut_case.py"),
        "--output",
        str(case_root / "strut"),
    ]
    if is_tcl and not use_generated_direct_tcl:
        strut_cmd += ["--input-tcl", str(entry_tcl)]
    else:
        strut_cmd += ["--input", str(case_json)]
    run(strut_cmd, env=env, verbose=verbose)

    run(
        [
            uv,
            "run",
            "python",
            str(repo_root / "scripts" / "compare_case.py"),
            "--case-root",
            str(case_root),
            *(
                ["--input-tcl", str(entry_tcl)]
                if is_tcl and not use_generated_direct_tcl
                else ["--case-json", str(case_json)]
            ),
        ],
        env=env,
        verbose=verbose,
    )


if __name__ == "__main__":
    main()
