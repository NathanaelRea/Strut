"""Install the reconstructed centrifuge model and its OpenSees 3.7.1 entries."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import tempfile
from pathlib import Path

from .analysis import cap_shaking, modern_newmark
from .model import model_text

HERE = Path(__file__).parent
REPO_ROOT = HERE.parents[1]
FIXTURE_DIR = REPO_ROOT / "tests/fixtures/opensees/OpenSeesManual/Centrifuge"
STAMP_NAME = ".strut-generated"
ORIGINAL_ENTRY = "CentrifugeExample.tcl"
BENCHMARK_ENTRY = "CentrifugeFull.tcl"
PARITY_ENTRY = "CentrifugeCapped.tcl"
ENTRY_NAMES = (ORIGINAL_ENTRY, BENCHMARK_ENTRY, PARITY_ENTRY)


def build_bundle(target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    original = model_text()
    benchmark = modern_newmark(original)
    for name, script in zip(
        ENTRY_NAMES, (original, benchmark, cap_shaking(benchmark)), strict=True
    ):
        (target / name).write_text(script, encoding="ascii")


def generator_fingerprint() -> str:
    digest = hashlib.sha256()
    for name in (
        "analysis.py",
        "bundle.py",
        "mesh.py",
        "mesh.bin",
        "model.py",
        "pore_pressures.tsv",
    ):
        digest.update(name.encode() + b"\0" + (HERE / name).read_bytes() + b"\0")
    return digest.hexdigest()


def bundle_is_current(target: Path = FIXTURE_DIR) -> bool:
    stamp = target / STAMP_NAME
    return (
        stamp.is_file()
        and all((target / name).is_file() for name in ENTRY_NAMES)
        and stamp.read_text().strip() == generator_fingerprint()
    )


def ensure_bundle(target: Path = FIXTURE_DIR) -> bool:
    """Install the deterministic bundle once across parallel setup callers."""
    if bundle_is_current(target):
        return False
    target.mkdir(parents=True, exist_ok=True)
    with (target / ".generate.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if bundle_is_current(target):
            return False
        with tempfile.TemporaryDirectory(dir=target) as directory:
            staging = Path(directory)
            build_bundle(staging)
            (staging / STAMP_NAME).write_text(generator_fingerprint() + "\n")
            for name in (*ENTRY_NAMES, STAMP_NAME):
                (staging / name).replace(target / name)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--install", action="store_true")
    target.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    if args.install:
        ensure_bundle()
    else:
        build_bundle(args.out_dir)


if __name__ == "__main__":
    main()
