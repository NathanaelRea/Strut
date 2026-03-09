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

import direct_tcl_support


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
    return direct_tcl_support.direct_tcl_manifest_paths(repo_root)


def _load_direct_tcl_manifest(manifest_path: Path) -> dict:
    return direct_tcl_support.load_direct_tcl_manifest(manifest_path)


def _resolve_entry_tcl_from_manifest(manifest_path: Path, repo_root: Path) -> Path:
    return direct_tcl_support.resolve_entry_tcl_from_manifest(manifest_path, repo_root)


def _resolve_canonical_tcl_case(entry_tcl: Path, repo_root: Path):
    manifest_path = direct_tcl_support.find_direct_tcl_manifest_for_entry(
        entry_tcl, repo_root
    )
    if manifest_path is None:
        return None
    return manifest_path.parent.name, manifest_path.parent, manifest_path


def _is_direct_tcl_manifest(path: Path) -> bool:
    if path.name != "direct_tcl_case.json":
        return False
    data = _load_direct_tcl_manifest(path)
    raw_path = data.get("entry_tcl")
    return isinstance(raw_path, str) and bool(raw_path)


def _resolve_direct_tcl_source_files(
    entry_tcl: Path, manifest_path: Path | None = None
) -> list[Path]:
    return direct_tcl_support.resolve_direct_tcl_source_files(entry_tcl, manifest_path)


def _prepare_direct_tcl_entry(
    entry_tcl: Path, case_root: Path, source_files: list[Path] | None = None
) -> tuple[Path, list[Path]]:
    return direct_tcl_support.prepare_direct_tcl_entry(
        entry_tcl,
        source_files or [entry_tcl],
        case_root / "generated" / "_direct_tcl_context",
        excluded_roots=[case_root],
    )


def _prepare_opensees_compat_entry(entry_tcl: Path) -> Path:
    wrapper_path = entry_tcl.parent / f"__strut_opensees_{entry_tcl.stem}_entry.tcl"
    wrapper_path.write_text(
        "\n".join(
            [
                "set __strut_ndf 0",
                "if {[llength [info commands model]] > 0 && ![llength [info commands __strut_builtin_model]]} {",
                "    rename model __strut_builtin_model",
                "    proc model {args} {",
                "        global __strut_ndf",
                "        set idx [lsearch -exact $args -ndf]",
                "        if {$idx >= 0 && ($idx + 1) < [llength $args]} {",
                "            set __strut_ndf [lindex $args [expr {$idx + 1}]]",
                "        }",
                "        return [uplevel 1 [linsert $args 0 __strut_builtin_model]]",
                "    }",
                "}",
                "if {[llength [info commands mass]] > 0 && ![llength [info commands __strut_builtin_mass]]} {",
                "    rename mass __strut_builtin_mass",
                "    proc mass {args} {",
                "        global __strut_ndf",
                "        if {[llength $args] < 2} {",
                "            return [uplevel 1 [linsert $args 0 __strut_builtin_mass]]",
                "        }",
                "        set nodeTag [lindex $args 0]",
                "        set values [lrange $args 1 end]",
                "        if {$__strut_ndf > 0 && [llength $values] > $__strut_ndf} {",
                "            set values [lrange $values 0 [expr {$__strut_ndf - 1}]]",
                "        }",
                "        set attempt_lengths [list [llength $values]]",
                "        foreach expected {6 3 1} {",
                "            if {$expected < [llength $values] && [lsearch -exact $attempt_lengths $expected] < 0} {",
                "                lappend attempt_lengths $expected",
                "            }",
                "        }",
                "        set result {}",
                "        set opts {}",
                "        foreach expected $attempt_lengths {",
                "            set trimmed [lrange $values 0 [expr {$expected - 1}]]",
                "            set call [linsert $trimmed 0 $nodeTag]",
                "            set rc [catch {uplevel 1 [linsert $call 0 __strut_builtin_mass]} result opts]",
                "            if {$rc == 0} {",
                "                return $result",
                "            }",
                "        }",
                "        return -options $opts $result",
                "    }",
                "}",
                f"source {{{entry_tcl.name}}}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return wrapper_path


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
    if rec_type in (
        "node_displacement",
        "node_reaction",
        "envelope_node_displacement",
        "envelope_node_acceleration",
    ):
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
    if rec_type in ("envelope_element_force", "envelope_element_local_force"):
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
            if preserve_existing_targets:
                continue
            raise SystemExit(f"missing raw OpenSees recorder output: {raw_path}")
        normalized_rows = []
        for line in source.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            parts = line.replace(",", " ").split()
            if group_layout and group_layout.get("type", "").startswith("envelope_") and strip_time:
                normalized_rows.append(parts)
                continue
            if strip_time and parts:
                parts = parts[1:]
            normalized_rows.append(parts)

        if group_layout and group_layout.get("type", "").startswith("envelope_"):
            layout_items = group_layout.get("elements")
            layout_widths = group_layout.get("values_per_element")
            item_key = "elements"
            if layout_items is None:
                layout_items = group_layout.get("nodes") or []
                layout_widths = group_layout.get("values_per_node") or []
                item_key = "nodes"
            width_by_item = dict(zip(layout_items, layout_widths))
            widths = []
            for recorder in recorders:
                items = recorder.get(item_key) or []
                if len(items) != 1 or items[0] not in width_by_item:
                    widths = []
                    break
                widths.append(int(width_by_item[items[0]]))
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
        "parity_tolerance_by_category",
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


def _canonical_reference_ready(case_data: dict, reference_dir: Path) -> bool:
    if not reference_dir.exists():
        return False
    for recorder in case_data.get("recorders", []):
        if recorder.get("parity", True) is False:
            continue
        rec_type = recorder.get("type")
        output = recorder.get("output", rec_type)
        if rec_type in {
            "node_displacement",
            "node_reaction",
            "envelope_node_displacement",
            "envelope_node_acceleration",
        }:
            nodes = recorder.get("nodes", [])
            if len(nodes) != 1:
                return False
            targets = [f"{output}_node{int(nodes[0])}.out"]
        elif rec_type in {
            "element_force",
            "element_local_force",
            "element_basic_force",
            "element_deformation",
            "envelope_element_force",
            "envelope_element_local_force",
        }:
            elements = recorder.get("elements", [])
            if len(elements) != 1:
                return False
            targets = [f"{output}_ele{int(elements[0])}.out"]
        elif rec_type in {"section_force", "section_deformation"}:
            elements = recorder.get("elements", [])
            if len(elements) != 1:
                return False
            section = recorder.get("section")
            if section is None:
                sections = recorder.get("sections") or []
                if len(sections) != 1:
                    return False
                section = sections[0]
            targets = [f"{output}_ele{int(elements[0])}_sec{int(section)}.out"]
        elif rec_type == "drift":
            targets = [
                f"{output}_i{int(recorder['i_node'])}_j{int(recorder['j_node'])}.out"
            ]
        else:
            return False
        for rel_name in targets:
            if not (reference_dir / rel_name).exists():
                return False
    return True


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
    needs_original = refresh_reference or not _canonical_reference_ready(
        case_data, original_reference_dir
    )
    needs_generated = refresh_reference or not _canonical_reference_ready(
        case_data, generated_reference_dir
    )
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
    manifest_path = None
    hash_inputs = [case_input]
    if case_input.suffix == ".json":
        if _is_direct_tcl_manifest(case_input):
            data = _load_direct_tcl_manifest(case_input)
            case_name = str(data.get("name") or case_input.parent.name)
            case_root = case_input.parent
            manifest_path = case_input
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
            case_name, case_root, manifest_path = canonical_case
        entry_tcl = case_input
        is_tcl = True
    else:
        raise SystemExit(f"unsupported case input: {case_input}")

    if manifest_path is not None and manifest_path != case_input:
        hash_inputs.append(manifest_path)

    case_root.mkdir(parents=True, exist_ok=True)
    (case_root / "reference").mkdir(parents=True, exist_ok=True)
    (case_root / "strut").mkdir(parents=True, exist_ok=True)
    runtime_entry_tcl = entry_tcl
    opensees_entry_tcl = entry_tcl
    if is_tcl:
        assert entry_tcl is not None
        source_files = _resolve_direct_tcl_source_files(entry_tcl, manifest_path)
        runtime_entry_tcl, runtime_hash_inputs = _prepare_direct_tcl_entry(
            entry_tcl, case_root, source_files
        )
        opensees_entry_tcl = _prepare_opensees_compat_entry(runtime_entry_tcl)
        hash_inputs.extend(runtime_hash_inputs)

    verbose = os.getenv("STRUT_VERBOSE") == "1"
    env = os.environ.copy()
    case_data = None
    use_generated_direct_tcl = False

    if is_tcl:
        assert runtime_entry_tcl is not None
        import tcl_to_strut

        case_data = _merge_direct_tcl_manifest_metadata(
            tcl_to_strut.convert_tcl_to_solver_input(runtime_entry_tcl, repo_root),
            manifest_path,
        )
        opensees_entry_tcl = _prepare_opensees_compat_entry(runtime_entry_tcl)
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
            tcl_out = runtime_entry_tcl
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
        assert runtime_entry_tcl is not None
        assert opensees_entry_tcl is not None
        try:
            _validate_generated_tcl_matches_original(
                repo_root=repo_root,
                case_root=case_root,
                case_data=case_data,
                entry_tcl=opensees_entry_tcl,
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
            tcl_out = runtime_entry_tcl
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
                    str(opensees_entry_tcl),
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
        strut_cmd += ["--input-tcl", str(runtime_entry_tcl)]
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
                ["--input-tcl", str(runtime_entry_tcl)]
                if is_tcl and not use_generated_direct_tcl
                else ["--case-json", str(case_json)]
            ),
        ],
        env=env,
        verbose=verbose,
    )


if __name__ == "__main__":
    main()
