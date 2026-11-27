"""Build small Mojo tool binaries safely under concurrent callers.

pytest-xdist workers call the path runners in parallel. Building straight to
the output path lets one worker exec the binary while another is still writing
it, which fails with ETXTBSY ("Text file busy"). Builds here are serialized by
a file lock and land in place with an atomic rename.
"""

from __future__ import annotations

import fcntl
import os
import subprocess
from pathlib import Path

from mojo_target import mojo_target_args


def _latest_mtime(sources: list[Path]) -> float:
    latest = 0.0
    for src in sources:
        if src.is_dir():
            latest = max(
                latest,
                max((p.stat().st_mtime for p in src.rglob("*.mojo")), default=0.0),
            )
        else:
            latest = max(latest, src.stat().st_mtime)
    return latest


def _is_stale(bin_path: Path, sources: list[Path]) -> bool:
    if not bin_path.exists():
        return True
    return bin_path.stat().st_mtime < _latest_mtime(sources)


def ensure_built(
    uv: str,
    entry: Path,
    bin_path: Path,
    sources: list[Path],
    verbose: bool = False,
) -> Path:
    """Return the tool for the selected architecture, rebuilding if stale."""
    target_args = mojo_target_args()
    if target_args:
        arch = target_args[0].removeprefix("--march=")
        bin_path = bin_path.with_name(f"{bin_path.name}-{arch}")
    if not _is_stale(bin_path, sources):
        return bin_path
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = bin_path.with_name(bin_path.name + ".lock")
    with lock_path.open("w") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        # Another caller may have finished the build while we waited.
        if not _is_stale(bin_path, sources):
            return bin_path
        tmp_path = bin_path.with_name(f"{bin_path.name}.tmp.{os.getpid()}")
        cmd = [
            uv,
            "run",
            "mojo",
            "build",
            *target_args,
            str(entry),
            "-o",
            str(tmp_path),
        ]
        if verbose:
            print("+", " ".join(cmd))
        try:
            subprocess.check_call(cmd)
            os.replace(tmp_path, bin_path)
        finally:
            tmp_path.unlink(missing_ok=True)
    return bin_path
