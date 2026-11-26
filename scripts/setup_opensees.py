#!/usr/bin/env python3
"""Install the locked OpenSees reference binaries into a shared per-user cache.

The OpenSees, OpenSeesMP and OpenSeesSP binaries come from the conda-forge
``opensees`` package. ``scripts/opensees/opensees-linux-64.lock`` pins every
package by URL and sha256. A pinned, checksum-verified micromamba installs the
lock, so this needs no sudo, cmake, compilers or system Tcl/MPI.

The environment goes in ``<cache>/opensees/<lock-hash>/``. ``<cache>`` is
``$STRUT_CACHE_DIR``, else ``$XDG_CACHE_HOME/strut``, else ``~/.cache/strut``.
Every checkout and worktree with the same lockfile shares one install. A branch
that changes the lock gets its own prefix and leaves the others untouched.
``scripts/opensees_env.sh`` resolves the same path for the run scripts.

Subcommands:
  install (default)  Install the locked environment if missing, then smoke test.
  path               Print the resolved binary path (or the prefix with --prefix).
  verify             Run the smoke tests against the installed environment.
  lock               Re-resolve the spec below and rewrite the lockfile.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCK_FILE = REPO_ROOT / "scripts" / "opensees" / "opensees-linux-64.lock"

# Spec that `lock` resolves. Edit here, then run `setup_opensees.py lock`.
# Pinned to 3.7.1 on purpose: from OpenSees commit a1a506a (Feb 2025, first
# released in 3.8.0) Tcl `Path`/`Series` require a tag, which breaks the
# inline `-accel "Series -dt ... -filePath ..."` form the wiki examples use.
# Keep the OpenSees source in tests/fixtures/opensees/reference-data.lock.json
# on the same tag.
CONDA_CHANNEL = "conda-forge"
CONDA_SPECS = ("opensees=3.7.1=*_6", "python=3.12")

MICROMAMBA_VERSION = "2.10.0-0"
MICROMAMBA_URL = (
    "https://github.com/mamba-org/micromamba-releases/releases/download/"
    f"{MICROMAMBA_VERSION}/micromamba-linux-64"
)
MICROMAMBA_SHA256 = "701f39c06f8e6a670a0e186c8479b9e5c0502a2f0cda6cef65d83bdd2875e8bd"

READY_MARKER = ".strut-ready"
BINARIES = ("OpenSees", "OpenSeesMP", "OpenSeesSP")

SMOKE_TCL = """\
model basic -ndm 2 -ndf 3
node 1 0.0 0.0
node 2 0.0 10.0
fix 1 1 1 1
geomTransf Linear 1
element elasticBeamColumn 1 1 2 10.0 3000.0 100.0 1
timeSeries Linear 1
pattern Plain 1 1 { load 2 1.0 0.0 0.0 }
system BandGeneral
numberer Plain
constraints Plain
integrator LoadControl 1.0
algorithm Linear
analysis Static
analyze 1
puts "STRUT_SMOKE [nodeDisp 2 1]"
"""
# Cantilever tip deflection P L^3 / (3 E I) = 1000 / 900000.
SMOKE_EXPECTED = 1000.0 / 900000.0

SMOKE_MP_TCL = 'puts "STRUT_SMOKE_MP [getPID] [getNP]"\n'


def log(message: str) -> None:
    print(f"[setup_opensees] {message}", file=sys.stderr, flush=True)


def die(message: str) -> None:
    print(f"[setup_opensees] error: {message}", file=sys.stderr, flush=True)
    raise SystemExit(1)


def cache_root() -> Path:
    explicit = os.environ.get("STRUT_CACHE_DIR")
    if explicit:
        return Path(explicit).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "strut"


def lock_hash(lock_file: Path = LOCK_FILE) -> str:
    # Keep in sync with strut_opensees_prefix in scripts/opensees_env.sh.
    return hashlib.sha256(lock_file.read_bytes()).hexdigest()[:16]


def env_prefix() -> Path:
    return cache_root() / "opensees" / lock_hash()


def is_ready(prefix: Path) -> bool:
    return (prefix / READY_MARKER).is_file() and all(
        (prefix / "bin" / name).is_file() for name in BINARIES
    )


def check_platform() -> None:
    if sys.platform != "linux" or platform.machine() not in ("x86_64", "AMD64"):
        die(
            f"only linux-64 is locked (found {sys.platform}/{platform.machine()}). "
            "Set OPENSEES_BIN/OPENSEESMP_BIN to your own build instead."
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_micromamba() -> Path:
    tool_dir = cache_root() / "tools" / f"micromamba-{MICROMAMBA_VERSION}"
    exe = tool_dir / "micromamba"
    if exe.is_file() and sha256_file(exe) == MICROMAMBA_SHA256:
        return exe
    tool_dir.mkdir(parents=True, exist_ok=True)
    log(f"downloading micromamba {MICROMAMBA_VERSION}")
    fd, tmp_name = tempfile.mkstemp(dir=tool_dir, prefix=".micromamba-")
    tmp = Path(tmp_name)
    try:
        with (
            os.fdopen(fd, "wb") as out,
            urllib.request.urlopen(MICROMAMBA_URL, timeout=120) as response,
        ):
            shutil.copyfileobj(response, out)
        actual = sha256_file(tmp)
        if actual != MICROMAMBA_SHA256:
            die(
                f"micromamba checksum mismatch: expected {MICROMAMBA_SHA256}, got {actual}"
            )
        tmp.chmod(0o755)
        tmp.replace(exe)
    finally:
        tmp.unlink(missing_ok=True)
    return exe


def micromamba_env() -> dict[str, str]:
    env = os.environ.copy()
    # Keep micromamba away from any user/system conda configuration.
    env["MAMBA_ROOT_PREFIX"] = str(cache_root() / "conda-root")
    env["CONDARC"] = os.devnull
    env["MAMBARC"] = os.devnull
    env.pop("CONDA_PREFIX", None)
    return env


def run_micromamba(exe: Path, args: list[str], capture: bool = False) -> str:
    cmd = [str(exe), *args, "--root-prefix", str(cache_root() / "conda-root")]
    result = subprocess.run(
        cmd,
        env=micromamba_env(),
        text=True,
        stdout=subprocess.PIPE if capture else None,
    )
    if result.returncode != 0:
        die(f"micromamba failed (exit {result.returncode}): {' '.join(cmd)}")
    return result.stdout or ""


def runtime_env() -> dict[str, str]:
    # Run the binaries the way the harness does: no conda activation.
    env = os.environ.copy()
    for key in ("CONDA_PREFIX", "TCL_LIBRARY", "TK_LIBRARY", "LD_LIBRARY_PATH"):
        env.pop(key, None)
    return env


def smoke_test(prefix: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="strut-opensees-smoke-") as tmp:
        work = Path(tmp)
        (work / "smoke.tcl").write_text(SMOKE_TCL, encoding="utf-8")
        result = subprocess.run(
            [str(prefix / "bin" / "OpenSees"), "smoke.tcl"],
            cwd=work,
            env=runtime_env(),
            capture_output=True,
            text=True,
            timeout=120,
        )
        output = result.stdout + result.stderr
        values = [
            line.split()[1]
            for line in output.splitlines()
            if line.startswith("STRUT_SMOKE ")
        ]
        if result.returncode != 0 or not values:
            die(f"OpenSees smoke test failed (exit {result.returncode}):\n{output}")
        disp = float(values[0])
        if abs(disp - SMOKE_EXPECTED) > 1e-12 * SMOKE_EXPECTED:
            die(
                f"OpenSees smoke test: tip displacement {disp}, expected {SMOKE_EXPECTED}"
            )
        log(f"OpenSees smoke test OK (tip displacement {disp:.12g})")

        (work / "smoke_mp.tcl").write_text(SMOKE_MP_TCL, encoding="utf-8")
        result = subprocess.run(
            [
                str(prefix / "bin" / "mpiexec"),
                "-n",
                "2",
                "--mca",
                "accelerator",
                "null",
                str(prefix / "bin" / "OpenSeesMP"),
                "smoke_mp.tcl",
            ],
            cwd=work,
            env=runtime_env(),
            capture_output=True,
            text=True,
            timeout=120,
        )
        output = result.stdout + result.stderr
        ranks = sorted(
            line.split()[1]
            for line in output.splitlines()
            if line.startswith("STRUT_SMOKE_MP ")
        )
        if result.returncode != 0 or ranks != ["0", "1"]:
            die(f"OpenSeesMP smoke test failed (exit {result.returncode}):\n{output}")
        log("OpenSeesMP smoke test OK (2 MPI ranks)")


def cmd_install(force: bool, skip_verify: bool) -> None:
    check_platform()
    if not LOCK_FILE.is_file():
        die(f"lockfile not found: {LOCK_FILE}")
    prefix = env_prefix()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    # Serialize installs across worktrees sharing this cache.
    with open(prefix.parent / ".install.lock", "w") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        if is_ready(prefix) and not force:
            log(f"already installed: {prefix}")
        else:
            if prefix.exists():
                log(f"removing incomplete or forced install: {prefix}")
                shutil.rmtree(prefix)
            exe = ensure_micromamba()
            log(f"installing {LOCK_FILE.relative_to(REPO_ROOT)} into {prefix}")
            # Conda prefixes are not relocatable, so install in place and mark
            # completion last; is_ready() rejects a half-written prefix.
            run_micromamba(
                exe,
                [
                    "create",
                    "--yes",
                    "--quiet",
                    "--prefix",
                    str(prefix),
                    "--file",
                    str(LOCK_FILE),
                ],
            )
            if not skip_verify:
                smoke_test(prefix)
            (prefix / READY_MARKER).write_text(
                json.dumps({"lock_sha256_16": lock_hash()}) + "\n", encoding="utf-8"
            )
    for name in BINARIES:
        print(prefix / "bin" / name)


def cmd_path(prefix_only: bool, binary: str) -> None:
    prefix = env_prefix()
    if not is_ready(prefix):
        log(f"not installed at {prefix}; run: uv run scripts/setup_opensees.py")
        raise SystemExit(1)
    print(prefix if prefix_only else prefix / "bin" / binary)


def cmd_verify() -> None:
    prefix = env_prefix()
    if not is_ready(prefix):
        die(f"not installed at {prefix}; run: uv run scripts/setup_opensees.py")
    smoke_test(prefix)


def cmd_lock() -> None:
    check_platform()
    exe = ensure_micromamba()
    with tempfile.TemporaryDirectory(prefix="strut-opensees-lock-") as tmp:
        raw = run_micromamba(
            exe,
            [
                "create",
                "--dry-run",
                "--json",
                "--yes",
                "--prefix",
                str(Path(tmp) / "env"),
                "--override-channels",
                "--channel",
                CONDA_CHANNEL,
                "--platform",
                "linux-64",
                *CONDA_SPECS,
            ],
            capture=True,
        )
    actions = json.loads(raw).get("actions", {})
    packages = actions.get("FETCH") or actions.get("LINK") or []
    if not any(pkg["name"] == "opensees" for pkg in packages):
        die("resolution did not include the opensees package")
    lines = [
        "# Generated by scripts/setup_opensees.py lock. Do not edit by hand.",
        f"# channel: {CONDA_CHANNEL}",
        f"# specs: {' '.join(CONDA_SPECS)}",
        "# platform: linux-64",
        "@EXPLICIT",
    ]
    for pkg in sorted(packages, key=lambda item: item["name"]):
        lines.append(f"{pkg['url']}#sha256:{pkg['sha256']}")
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    LOCK_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    opensees = next(pkg for pkg in packages if pkg["name"] == "opensees")
    log(
        f"wrote {LOCK_FILE.relative_to(REPO_ROOT)}: {len(packages)} packages, "
        f"opensees {opensees['version']} {opensees['build_string']}, hash {lock_hash()}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command")
    install = sub.add_parser("install", help="install the locked environment (default)")
    install.add_argument(
        "--force", action="store_true", help="reinstall even if present"
    )
    install.add_argument("--skip-verify", action="store_true", help="skip smoke tests")
    path = sub.add_parser("path", help="print the installed binary path")
    path.add_argument(
        "--prefix", action="store_true", help="print the environment prefix"
    )
    path.add_argument("--binary", default="OpenSees", choices=(*BINARIES, "mpiexec"))
    sub.add_parser("verify", help="smoke test the installed binaries")
    sub.add_parser("lock", help="re-resolve the spec and rewrite the lockfile")
    args = parser.parse_args()

    if args.command in (None, "install"):
        cmd_install(
            force=getattr(args, "force", False),
            skip_verify=getattr(args, "skip_verify", False),
        )
    elif args.command == "path":
        cmd_path(args.prefix, args.binary)
    elif args.command == "verify":
        cmd_verify()
    elif args.command == "lock":
        cmd_lock()


if __name__ == "__main__":
    main()
