#!/usr/bin/env python3
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def _run_capture_output(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        text=True,
    )
    return proc.returncode, proc.stdout


def _source_fingerprint(src_root: Path, out_name: str) -> str:
    digest = hashlib.sha256()
    digest.update(f"out_name={out_name}\n".encode("utf-8"))
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


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    src_root = repo_root / "src" / "mojo"
    out_dir = repo_root / "build" / "strut"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_name = "strut"
    build_cmd = [
        "uv",
        "run",
        "mojo",
        "build",
        "--diagnostic-format",
        "json",
        str(src_root / "strut.mojo"),
    ]
    if os.getenv("STRUT_PROFILE", "0") == "1":
        out_name = "strut_profile"
        build_cmd.extend(["-D", "STRUT_PROFILE=1"])

    out_bin = out_dir / out_name
    stamp_file = Path(f"{out_bin}.src-sha256")
    source_fingerprint = _source_fingerprint(src_root, out_name)

    if out_bin.exists() and stamp_file.exists():
        cached_fingerprint = stamp_file.read_text(encoding="utf-8").strip()
        if cached_fingerprint == source_fingerprint:
            print(f"Build cache hit {out_bin}")
            return 0

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
        stamp_file.write_text(f"{source_fingerprint}\n", encoding="utf-8")
        print(f"Built {out_bin}")
        return 0
    finally:
        Path(tmp_handle.name).unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
