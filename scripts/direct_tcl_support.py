from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Sequence


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def load_direct_tcl_manifest(manifest_path: Path) -> dict:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"invalid direct Tcl manifest: {manifest_path}")
    return data


def direct_tcl_manifest_paths(repo_root: Path) -> list[Path]:
    return sorted(
        (repo_root / "tests" / "validation").glob("*/direct_tcl_case.json")
    )


def resolve_repo_relative_path(raw_path: str | Path, repo_root: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def resolve_entry_tcl_from_manifest(manifest_path: Path, repo_root: Path) -> Path:
    data = load_direct_tcl_manifest(manifest_path)
    raw_path = data.get("entry_tcl")
    if not isinstance(raw_path, str) or not raw_path:
        raise SystemExit(f"direct Tcl manifest missing entry_tcl: {manifest_path}")
    return resolve_repo_relative_path(raw_path, repo_root)


def resolve_direct_tcl_source_files(
    entry_tcl: Path, manifest_path: Path | None = None
) -> list[Path]:
    entry_path = entry_tcl.resolve()
    if manifest_path is None or not manifest_path.exists():
        return [entry_path]

    manifest = load_direct_tcl_manifest(manifest_path)
    raw_source_files = manifest.get("source_files")
    if raw_source_files is None:
        return [entry_path]
    if not isinstance(raw_source_files, list) or not raw_source_files:
        raise SystemExit(
            f"direct Tcl manifest source_files must be a non-empty list: {manifest_path}"
        )

    source_files: list[Path] = []
    for raw_path in raw_source_files:
        if not isinstance(raw_path, str) or not raw_path:
            raise SystemExit(
                f"direct Tcl manifest source_files entries must be non-empty strings: {manifest_path}"
            )
        source_path = Path(raw_path)
        if not source_path.is_absolute():
            source_path = (entry_path.parent / source_path).resolve()
        else:
            source_path = source_path.resolve()
        if not source_path.exists():
            raise SystemExit(
                f"direct Tcl manifest source_files entry does not exist: {manifest_path}: {raw_path}"
            )
        if source_path.suffix.lower() != ".tcl":
            raise SystemExit(
                f"direct Tcl manifest source_files entry is not a Tcl file: {manifest_path}: {raw_path}"
            )
        source_files.append(source_path)

    if source_files[-1] != entry_path:
        raise SystemExit(
            "direct Tcl manifest source_files must end with entry_tcl: "
            f"{manifest_path}"
        )

    return source_files


def find_direct_tcl_manifest_for_entry(entry_tcl: Path, repo_root: Path) -> Path | None:
    entry_path = entry_tcl.resolve()
    matches = [
        manifest_path
        for manifest_path in direct_tcl_manifest_paths(repo_root)
        if resolve_entry_tcl_from_manifest(manifest_path, repo_root) == entry_path
    ]
    if len(matches) != 1:
        return None
    return matches[0]


def _should_mirror_direct_tcl_sibling(path: Path) -> bool:
    if path.name.lower() == "data":
        return False
    if path.is_file():
        return True
    if not path.is_dir():
        return False
    for child in path.iterdir():
        if child.is_file() and child.suffix.lower() == ".tcl":
            return False
    return True


def _copy_direct_tcl_shared_assets(script_dir: Path, mirrored_parent: Path) -> list[Path]:
    mirrored_script_dir = mirrored_parent / script_dir.name
    mirrored_inputs: list[Path] = []
    for child in sorted(script_dir.parent.iterdir()):
        if child == script_dir or not _should_mirror_direct_tcl_sibling(child):
            continue
        source = mirrored_parent / child.name
        target = mirrored_script_dir / child.name
        if source.is_file():
            shutil.copy2(source, target)
            mirrored_inputs.append(child.resolve())
            continue
        if not source.is_dir():
            continue
        shutil.copytree(source, target, dirs_exist_ok=True)
        mirrored_inputs.extend(
            sorted(path.resolve() for path in child.rglob("*") if path.is_file())
        )
    return mirrored_inputs


def _ensure_direct_tcl_compat_aliases(
    mirrored_parent: Path, mirrored_script_dir: Path
) -> list[Path]:
    mirrored_inputs: list[Path] = []

    # Some OpenSees bundles source GeneratePeaks.tcl even when only
    # LibGeneratePeaks.tcl is present in the directory.
    lib_generate_peaks = mirrored_script_dir / "LibGeneratePeaks.tcl"
    generate_peaks = mirrored_script_dir / "GeneratePeaks.tcl"
    if lib_generate_peaks.exists() and not generate_peaks.exists():
        shutil.copy2(lib_generate_peaks, generate_peaks)
        mirrored_inputs.append(lib_generate_peaks.resolve())

    canonical_read_smd = mirrored_script_dir / "ReadSMDFile.tcl"
    legacy_read_smd = mirrored_script_dir / "ReadSMDfile.tcl"
    if canonical_read_smd.exists() and not legacy_read_smd.exists():
        shutil.copy2(canonical_read_smd, legacy_read_smd)
        mirrored_inputs.append(canonical_read_smd.resolve())

    gm_dirs = [mirrored_parent / "GMfiles", mirrored_script_dir / "GMfiles"]
    seen_dirs: set[Path] = set()
    for gm_dir in gm_dirs:
        gm_dir = gm_dir.resolve()
        if gm_dir in seen_dirs or not gm_dir.is_dir():
            continue
        seen_dirs.add(gm_dir)

        at2_files = sorted(
            path for path in gm_dir.iterdir() if path.is_file() and path.suffix.lower() == ".at2"
        )
        for at2_path in at2_files:
            dt2_path = at2_path.with_suffix(".dt2")
            if not dt2_path.exists():
                shutil.copy2(at2_path, dt2_path)
                mirrored_inputs.append(at2_path.resolve())

        canonical_peer = gm_dir / "H-e12140.at2"
        legacy_peer = gm_dir / "H-E01140.at2"
        if canonical_peer.exists() and not legacy_peer.exists():
            shutil.copy2(canonical_peer, legacy_peer)
            mirrored_inputs.append(canonical_peer.resolve())
        if legacy_peer.exists():
            legacy_dt2 = gm_dir / "H-E01140.dt2"
            if not legacy_dt2.exists():
                shutil.copy2(legacy_peer, legacy_dt2)
                mirrored_inputs.append(legacy_peer.resolve())

    return mirrored_inputs


def _patch_direct_tcl_compatibility(mirrored_tcl: Path, original_tcl: Path) -> None:
    text = mirrored_tcl.read_text(encoding="utf-8")

    if original_tcl.name == "Ex6.genericFrame2D.build.InelasticFiberWSection.tcl":
        if "set ColSecTag" not in text:
            marker = "# define MATERIAL properties ----------------------------------------\n"
            insert = "set ColSecTag 1\nset BeamSecTag 4\n"
            if marker in text:
                text = text.replace(marker, insert + marker, 1)
            else:
                text = insert + text

    if original_tcl.name in {
        "Ex6.genericFrame2D.build.ElasticSection.tcl",
        "Ex6.genericFrame2D.build.InelasticSection.tcl",
        "Ex6.genericFrame2D.build.InelasticFiberRCSection.tcl",
        "Ex6.genericFrame2D.build.InelasticFiberWSection.tcl",
    }:
        text = text.replace(
            "mass $nodeID $MassNode 0.0 0.0 0.0 0.0 0.0;",
            "mass $nodeID $MassNode 0.0 0.0;",
        )
        text = text.replace(
            "mass $nodeID $MassNode 0. 0. 0. 0. 0.;",
            "mass $nodeID $MassNode 0. 0.;",
        )

    mirrored_tcl.write_text(text, encoding="utf-8")


def prepare_direct_tcl_entry(
    entry_tcl: Path,
    source_files: Sequence[Path],
    mirror_root: Path,
    *,
    excluded_roots: Sequence[Path] = (),
) -> tuple[Path, list[Path]]:
    entry_path = entry_tcl.resolve()
    resolved_sources = [Path(path).resolve() for path in source_files]
    if not resolved_sources:
        raise SystemExit(f"direct Tcl entry requires at least one source file: {entry_tcl}")

    script_dir = entry_path.parent
    script_parent = script_dir.parent
    for source_path in resolved_sources:
        try:
            source_path.relative_to(script_parent)
        except ValueError as exc:
            raise SystemExit(
                "direct Tcl source_files entries must stay within the entry Tcl bundle: "
                f"{source_path}"
            ) from exc

    ensure_clean_dir(mirror_root)
    mirrored_parent = mirror_root / script_parent.name
    excluded = [Path(path).resolve() for path in excluded_roots]
    mirror_root_resolved = mirror_root.resolve()
    if mirror_root_resolved not in excluded:
        excluded.append(mirror_root_resolved)

    def _ignore(dirpath: str, names: list[str]) -> set[str]:
        resolved_dir = Path(dirpath).resolve()
        ignored: set[str] = set()
        for excluded_root in excluded:
            if excluded_root.parent == resolved_dir:
                ignored.add(excluded_root.name)
        return ignored

    shutil.copytree(
        script_parent, mirrored_parent, dirs_exist_ok=True, ignore=_ignore
    )
    mirrored_script_dir = mirrored_parent / script_dir.name
    shared_asset_inputs = _copy_direct_tcl_shared_assets(script_dir, mirrored_parent)
    compat_alias_inputs = _ensure_direct_tcl_compat_aliases(
        mirrored_parent, mirrored_script_dir
    )
    for source_path in resolved_sources:
        relative_path = source_path.relative_to(script_dir)
        _patch_direct_tcl_compatibility(mirrored_script_dir / relative_path, source_path)

    wrapper_path = mirrored_script_dir / f"__strut_{entry_path.stem}_entry.tcl"
    wrapper_lines = []
    for source_path in resolved_sources:
        relative_path = os.path.relpath(source_path, script_dir)
        wrapper_lines.append(f"source {{{relative_path}}}")
    wrapper_path.write_text("\n".join(wrapper_lines) + "\n", encoding="utf-8")
    return wrapper_path, [*resolved_sources, *shared_asset_inputs, *compat_alias_inputs]
