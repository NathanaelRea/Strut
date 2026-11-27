#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from mojo_target import mojo_target_args


def _run_capture_output(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        text=True,
    )
    return proc.returncode, proc.stdout


def _source_fingerprint(src_root: Path, out_name: str, target_args: list[str]) -> str:
    digest = hashlib.sha256()
    digest.update(f"out_name={out_name}\n".encode("utf-8"))
    for arg in target_args:
        digest.update(f"{arg}\n".encode("utf-8"))
    for src_path in sorted(src_root.rglob("*.mojo")):
        rel_path = src_path.relative_to(src_root).as_posix()
        digest.update(rel_path.encode("utf-8"))
        digest.update(b"\0")
        with src_path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    return digest.hexdigest()


def _normalized_diagnostics_json(text: str) -> str:
    decoder = json.JSONDecoder()
    pos = 0
    docs = []
    text_len = len(text)
    while pos < text_len:
        while pos < text_len and text[pos] not in "{[":
            pos += 1
        if pos >= text_len:
            break
        try:
            doc, end = decoder.raw_decode(text, pos)
        except json.JSONDecodeError:
            pos += 1
            continue
        docs.append(doc)
        pos = end
    if not docs:
        return ""
    return "".join(f"{json.dumps(doc)}\n" for doc in docs)


def _build_mojo(repo_root: Path, *, gpu: bool) -> int:
    src_root = repo_root / "src" / "mojo"
    out_dir = repo_root / "build" / "strut"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_name = "strut_gpu" if gpu else "strut"
    target_args = mojo_target_args()
    build_cmd = [
        "uv",
        "run",
        "mojo",
        "build",
        *target_args,
        "--diagnostic-format",
        "json",
        str(src_root / "strut.mojo"),
    ]
    if os.getenv("STRUT_PROFILE", "0") == "1":
        out_name += "_profile"
        build_cmd.extend(["-D", "STRUT_PROFILE=1"])

    if gpu:
        build_cmd.extend(["-D", "STRUT_GPU=1"])

    out_bin = out_dir / out_name
    stamp_file = Path(f"{out_bin}.src-sha256")
    fingerprint_args = target_args
    if gpu:
        fingerprint_args = [
            *target_args,
            *(
                f"{name}={os.environ.get(name)}"
                for name in (
                    "CUDA_VISIBLE_DEVICES",
                    "HIP_VISIBLE_DEVICES",
                    "ROCR_VISIBLE_DEVICES",
                )
            ),
        ]
    source_fingerprint = _source_fingerprint(src_root, out_name, fingerprint_args)

    if out_bin.exists() and stamp_file.exists():
        cached_fingerprint = stamp_file.read_text(encoding="utf-8").strip()
        if cached_fingerprint == source_fingerprint:
            print(f"Build cache hit {out_bin}")
            if not gpu:
                return 0
            return subprocess.call([str(out_bin), "--check-gpu"])

    stamp_file.unlink(missing_ok=True)
    build_cmd.extend(["-o", str(out_bin)])

    tmp_handle = tempfile.NamedTemporaryFile(
        mode="w", suffix=".diag", delete=False, encoding="utf-8"
    )
    try:
        tmp_handle.close()
        diag_path = Path(tmp_handle.name)

        build_status, diagnostics = _run_capture_output(build_cmd)
        diag_path.write_text(
            _normalized_diagnostics_json(diagnostics), encoding="utf-8"
        )
        if build_status != 0:
            sys.stderr.write(diagnostics)
            return build_status

        warning_cmd = [
            "uv",
            "run",
            "python",
            str(repo_root / "scripts" / "check_mojo_warnings.py"),
            "--quiet",
            str(diag_path),
        ]
        warning_status = subprocess.run(warning_cmd, check=False).returncode
        if warning_status != 0:
            sys.stderr.write(diagnostics)
            sys.stderr.write("Mojo compiler warnings detected.\n")
            return 1

        if diagnostics:
            sys.stdout.write(diagnostics)
        if gpu:
            status = subprocess.call([str(out_bin), "--check-gpu"])
            if status != 0:
                out_bin.unlink(missing_ok=True)
                return status
        stamp_file.write_text(f"{source_fingerprint}\n", encoding="utf-8")
        print(f"Built {out_bin}")
        return 0
    finally:
        Path(tmp_handle.name).unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the Strut solver.")
    parser.add_argument(
        "--gpu", action="store_true", help="build for Mojo's detected GPU target"
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    return _build_mojo(repo_root, gpu=args.gpu)


if __name__ == "__main__":
    raise SystemExit(main())
