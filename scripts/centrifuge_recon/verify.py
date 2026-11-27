"""Compare the original GiD export against the model and capped OpenSees runs."""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
from decimal import Decimal
from itertools import zip_longest
from pathlib import Path
from tempfile import TemporaryDirectory

from .analysis import cap_shaking, modern_newmark
from .bundle import PARITY_ENTRY, REPO_ROOT, build_bundle
from .model import model_text
from .source import read_original

# The export writes one command per line (including lines inside Tcl bodies),
# quotes only double-quoted words, and continues only the second Newmark call.
WORDS = re.compile(r'"[^"]*"|\S+')
NUMBER = re.compile(r"[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")


def parse_commands(script: str) -> list[tuple[str, ...]]:
    commands = []
    for line in re.sub(r"\\\r?\n[ \t]*", " ", script).splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        words = WORDS.findall(stripped)
        if words[0] in {"node", "load"}:
            words = [
                str(Decimal(word).normalize()) if NUMBER.fullmatch(word) else word
                for word in words
            ]
        if words[0] == "set" and re.fullmatch(r"Timeseries\d+", words[1]):
            words[2] = " ".join(words[2].split())
        commands.append(tuple(words))
    return commands


def command_digest(script: str) -> str:
    data = "\n".join("\0".join(command) for command in parse_commands(script))
    return hashlib.sha256(data.encode()).hexdigest()


def compare_commands(original: str, reconstructed: str) -> int:
    expected, actual = parse_commands(original), parse_commands(reconstructed)
    for index, (left, right) in enumerate(zip_longest(expected, actual), 1):
        if left != right:
            raise ValueError(f"command {index}: original {left!r}, generated {right!r}")
    return len(expected)


def opensees_executable() -> Path:
    result = subprocess.check_output(
        [
            "bash",
            "-c",
            'source "$1" && strut_resolve_opensees_binary OpenSees "${OPENSEES_BIN:-}"',
            "opensees",
            REPO_ROOT / "scripts/opensees_env.sh",
        ],
        text=True,
    )
    return Path(result.strip()).resolve()


def run_capped(opensees: Path, directory: Path, entry: str) -> dict[str, bytes]:
    # The reviewed source has a pinned digest. Isolate its execution as well.
    run = subprocess.run(
        [
            "bwrap",
            "--ro-bind",
            "/",
            "/",
            "--dev",
            "/dev",
            "--proc",
            "/proc",
            "--bind",
            directory,
            directory,
            "--chdir",
            directory,
            "--unshare-net",
            "--unshare-pid",
            "--die-with-parent",
            opensees,
            entry,
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    expected_markers = [
        "Centrifuge gravity time: 1500000.000000",
        "Centrifuge gravity time: 4000000.000000",
        "Centrifuge shaking time: 0.150000",
    ]
    if not all(marker in run.stdout + run.stderr for marker in expected_markers):
        raise ValueError(
            f"{entry} did not complete the capped analysis:\n{run.stdout}\n{run.stderr}"
        )
    if "failed" in run.stderr.lower():
        raise ValueError(f"{entry} reported an OpenSees failure:\n{run.stderr}")
    return {path.name: path.read_bytes() for path in directory.glob("*.txt")}


def verify_runs(original: str, opensees: Path) -> None:
    with TemporaryDirectory(prefix="strut-centrifuge-") as root:
        original_dir, generated_dir = (
            Path(root) / name for name in ("original", "generated")
        )
        original_dir.mkdir()
        (original_dir / PARITY_ENTRY).write_text(cap_shaking(modern_newmark(original)))
        build_bundle(generated_dir)
        expected = run_capped(opensees, original_dir, PARITY_ENTRY)
        actual = run_capped(opensees, generated_dir, PARITY_ENTRY)

    if len(expected) != 11 or any(
        len(rows.splitlines()) != 3 for rows in expected.values()
    ):
        raise ValueError("expected three samples in each of the 11 recorder outputs")
    if expected.keys() != actual.keys():
        raise ValueError(f"recorder files differ: {expected.keys() ^ actual.keys()}")
    for name in expected:
        if expected[name] != actual[name]:
            raise ValueError(f"recorder output differs: {name}")
    print(
        "OpenSees: all eight gravity steps, 0.15 s of shaking, and 11 byte-identical recorder files"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path)
    parser.add_argument(
        "--run-opensees", action="store_true", help="requires bubblewrap"
    )
    args = parser.parse_args()
    original = read_original(args.original)
    commands = compare_commands(original, model_text())
    print(
        f"Original and generated model match: {commands} command lines ({command_digest(original)})"
    )
    if args.run_opensees:
        verify_runs(original, opensees_executable())


if __name__ == "__main__":
    main()
