#!/usr/bin/env python3
"""Fetch the reference data that tests and agents need, once per machine.

The committed manifest ``tests/fixtures/opensees/reference-data.lock.json``
pins every input by sha256 or git commit. Each one downloads once into
``<cache>/reference/`` and every checkout and worktree reuses it. ``<cache>``
is ``$STRUT_CACHE_DIR``, else ``$XDG_CACHE_HOME/strut``, else ``~/.cache/strut``.

What gets installed:

* ``files``: data files (ground motions and other non-Tcl inputs) that the
  committed example Tcl reads. Each one is pinned by its own sha256 and by the
  sha256 of the wiki download (a single file or a zip) it came from. Each is
  copied into ``tests/fixtures/opensees/<path>``.
* ``trees``: whole archives extracted into ``tests/fixtures/opensees/<path>``.
  The mega-tall benchmark model is one.
* ``git``: source trees cloned at a pinned commit into the cache and linked
  from ``docs/agent-reference/<path>``. These are OpenSees (matching the locked
  binary) and Modular (matching the Mojo version). An entry's ``links`` map
  adds extra ``docs/agent-reference/<name>`` links to files inside the tree.

Fixture data is copied, not linked, because the harness writes recorder
output next to the scripts it mirrors and must never write into the cache.

Subcommands:
  install (default)  Fetch anything missing and materialize it. Safe to re-run.
  check              Exit 1 and list what is missing or wrong in this checkout.
  vendor-wiki        Maintainers only. Rebuild the committed example Tcl and the
                     manifest's ``files`` from a fresh scrape made with
                     ``scripts/setup-opensees-docs.py --root DIR``.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parent))
from setup_opensees import cache_root  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "opensees"
MANIFEST = FIXTURE_ROOT / "reference-data.lock.json"
FIXTURE_GITIGNORE = FIXTURE_ROOT / ".gitignore"
AGENT_REFERENCE = REPO_ROOT / "docs" / "agent-reference"

# Wiki example directories that vendor-wiki syncs from a scrape.
WIKI_EXAMPLE_DIRS = ("OpenSeesExamplesBasic", "OpenSeesExamplesAdvanced")
WIKI_PROVENANCE = "opensees-wiki-provenance.json"
# Committed as source: example scripts. Everything else in a scrape is data
# and goes through the manifest.
VENDORED_SUFFIXES = {".tcl"}
# Tcl files that are really ground-motion data (value lists read by
# `timeSeries Path -values`), above this size, go through the manifest too.
MAX_VENDORED_TCL_BYTES = 64 * 1024

# Extra placements for scraped data files. The example Tcl reads
# `$GMdir/$GMfile.at2` with GMdir "../GMfiles" and lower-case names such as
# H-e12140, but the wiki zips extract H-E12140.AT2 at the manual root. On a
# case-sensitive filesystem the scripts only find these copies.
WIKI_ALIASES = {
    "OpenSeesExamplesAdvanced/GMfiles/H-e12140.at2": "OpenSeesExamplesAdvanced/H-E12140.AT2",
    "OpenSeesExamplesAdvanced/GMfiles/H-e12140.dt2": "OpenSeesExamplesAdvanced/H-E12140.DT2",
    "OpenSeesExamplesAdvanced/GMfiles/H-E01140.at2": "OpenSeesExamplesAdvanced/H-E01140.AT2",
}

USER_AGENT = "strut-reference-data/1.0"
REQUEST_INTERVAL_S = 1.0
TREE_MARKER = ".strut-tree-source"
GIT_MARKER = ".strut-git-commit"


def log(message: str) -> None:
    print(f"[setup_reference_data] {message}", file=sys.stderr, flush=True)


def die(message: str) -> None:
    print(f"[setup_reference_data] error: {message}", file=sys.stderr, flush=True)
    raise SystemExit(1)


def reference_cache() -> Path:
    return cache_root() / "reference"


def blob_path(sha256: str) -> Path:
    return reference_cache() / "blobs" / "sha256" / sha256


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest() -> dict:
    if not MANIFEST.is_file():
        die(f"manifest not found: {MANIFEST}")
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if data.get("version") != 1:
        die(f"unsupported manifest version in {MANIFEST}")
    for key in ("sources", "files", "trees", "git"):
        data.setdefault(key, [] if key != "sources" else {})
    return data


def write_manifest(data: dict) -> None:
    data["files"] = sorted(data["files"], key=lambda item: item["path"])
    data["sources"] = dict(sorted(data["sources"].items()))
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def safe_relative(raw: str) -> PurePosixPath:
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        die(f"unsafe path in manifest: {raw!r}")
    return path


def atomic_write(target: Path, data: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        # mkstemp creates 0600; give installed files normal permissions.
        os.chmod(tmp_name, 0o644)
        os.replace(tmp_name, target)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


class Downloader:
    """Polite sequential downloads: one request per REQUEST_INTERVAL_S."""

    def __init__(self) -> None:
        self._last = 0.0
        self.count = 0

    def fetch(self, url: str) -> bytes:
        wait = REQUEST_INTERVAL_S - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()
        self.count += 1
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.read()


def ensure_blob(sha256: str, source: dict, downloader: Downloader) -> Path:
    path = blob_path(sha256)
    if path.is_file():
        return path
    errors = []
    for url in source["urls"]:
        log(f"downloading {url}")
        try:
            data = downloader.fetch(url)
        except OSError as exc:
            errors.append(f"{url}: {exc}")
            continue
        actual = sha256_bytes(data)
        if actual != sha256:
            errors.append(f"{url}: sha256 {actual}, expected {sha256}")
            continue
        atomic_write(path, data)
        return path
    die("could not fetch a verified copy:\n  " + "\n  ".join(errors))
    raise AssertionError("unreachable")


def read_member(archive: Path, kind: str, member: str) -> bytes:
    if kind == "zip":
        with zipfile.ZipFile(archive) as zf:
            return zf.read(member)
    for name, data in iter_archive(archive, kind):
        if name == member:
            return data
    die(f"member {member!r} not found in {archive}")
    raise AssertionError("unreachable")


def iter_archive(archive: Path, kind: str):
    """Yield (member name, bytes) for every regular file in the archive."""
    if kind == "zip":
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                if not info.is_dir():
                    yield info.filename, zf.read(info)
        return
    if kind == "rar":
        try:
            import libarchive  # type: ignore[import-not-found]
        except ImportError:
            die("extracting RAR needs libarchive-c; run via `uv run` after `uv sync`")
        with libarchive.file_reader(str(archive)) as reader:
            for entry in reader:
                if not entry.isfile:
                    continue
                yield entry.pathname, b"".join(entry.get_blocks())
        return
    die(f"unsupported archive kind: {kind}")


def file_status(entry: dict) -> str | None:
    target = FIXTURE_ROOT / safe_relative(entry["path"])
    if not target.is_file():
        return "missing"
    if sha256_file(target) != entry["sha256"]:
        return "sha256 mismatch"
    return None


def install_file(entry: dict, sources: dict, downloader: Downloader) -> bool:
    if file_status(entry) is None:
        return False
    source = sources[entry["source"]]
    blob = ensure_blob(entry["source"], source, downloader)
    member = entry.get("member")
    data = blob.read_bytes() if member is None else read_member(blob, source["kind"], member)
    if sha256_bytes(data) != entry["sha256"]:
        die(f"{entry['path']}: extracted sha256 does not match the manifest")
    atomic_write(FIXTURE_ROOT / safe_relative(entry["path"]), data)
    return True


def tree_status(entry: dict) -> str | None:
    target = FIXTURE_ROOT / safe_relative(entry["path"])
    marker = target / TREE_MARKER
    if not marker.is_file():
        return "missing"
    if marker.read_text(encoding="utf-8").strip() != entry["source"]:
        return "stale"
    return None


def install_tree(entry: dict, sources: dict, downloader: Downloader) -> bool:
    if tree_status(entry) is None:
        return False
    source = sources[entry["source"]]
    blob = ensure_blob(entry["source"], source, downloader)
    target = FIXTURE_ROOT / safe_relative(entry["path"])
    exclude = set(entry.get("exclude", []))
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=target.parent, prefix=f".{target.name}."))
    try:
        count = 0
        for name, data in iter_archive(blob, source["kind"]):
            if name in exclude:
                continue
            rel = safe_relative(name.replace("\\", "/"))
            out = staging / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(data)
            count += 1
        (staging / TREE_MARKER).write_text(entry["source"] + "\n", encoding="utf-8")
        if target.exists():
            shutil.rmtree(target)
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    log(f"extracted {count} files into {target.relative_to(REPO_ROOT)}")
    return True


def git_cache_dir(entry: dict) -> Path:
    return reference_cache() / "git" / f"{entry['name']}-{entry['commit'][:12]}"


def ensure_git(entry: dict) -> Path:
    dest = git_cache_dir(entry)
    marker = dest / GIT_MARKER
    if marker.is_file() and marker.read_text(encoding="utf-8").strip() == entry["commit"]:
        return dest
    if dest.exists():
        shutil.rmtree(dest)
    staging = dest.with_name(dest.name + ".partial")
    shutil.rmtree(staging, ignore_errors=True)
    log(f"cloning {entry['url']} at {entry['ref']} (shallow)")
    subprocess.run(
        ["git", "-c", "advice.detachedHead=false", "clone", "--quiet",
         "--depth", "1", "--branch", entry["ref"],
         entry["url"], str(staging)],
        check=True,
    )
    head = subprocess.run(
        ["git", "-C", str(staging), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if head != entry["commit"]:
        shutil.rmtree(staging, ignore_errors=True)
        die(f"{entry['url']} {entry['ref']} is {head}, manifest pins {entry['commit']}")
    # Read-only reference copy: drop history so agents cannot commit into it.
    shutil.rmtree(staging / ".git")
    (staging / GIT_MARKER).write_text(entry["commit"] + "\n", encoding="utf-8")
    staging.rename(dest)
    return dest


def link_agent_reference(rel: str, target: Path) -> None:
    link = AGENT_REFERENCE / safe_relative(rel)
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if Path(os.readlink(link)) == target:
            return
        link.unlink()
    elif link.exists():
        log(f"leaving existing {link.relative_to(REPO_ROOT)} in place (not a link); "
            "delete it to use the shared copy")
        return
    link.symlink_to(target, target_is_directory=target.is_dir())


def cmd_install(skip_agent_reference: bool) -> None:
    manifest = load_manifest()
    sources = manifest["sources"]
    downloader = Downloader()
    reference_cache().mkdir(parents=True, exist_ok=True)
    with open(reference_cache() / ".install.lock", "w") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        written = sum(install_file(entry, sources, downloader) for entry in manifest["files"])
        trees = sum(install_tree(entry, sources, downloader) for entry in manifest["trees"])
        if not skip_agent_reference:
            for entry in manifest["git"]:
                tree = ensure_git(entry)
                link_agent_reference(entry["path"], tree)
                for name, inner in entry.get("links", {}).items():
                    target = tree / safe_relative(inner)
                    if not target.exists():
                        die(f"{entry['name']} has no {inner} to link as {name}")
                    link_agent_reference(name, target)
    log(
        f"fixtures: {len(manifest['files'])} files ({written} written), "
        f"{len(manifest['trees'])} trees ({trees} extracted); "
        f"{downloader.count} download(s); cache {reference_cache()}"
    )
    problems = collect_problems(manifest, include_agent_reference=not skip_agent_reference)
    if problems:
        die("install incomplete:\n  " + "\n  ".join(problems))


def collect_problems(manifest: dict, include_agent_reference: bool) -> list[str]:
    problems = []
    for entry in manifest["files"]:
        status = file_status(entry)
        if status:
            problems.append(f"tests/fixtures/opensees/{entry['path']}: {status}")
    for entry in manifest["trees"]:
        status = tree_status(entry)
        if status:
            problems.append(f"tests/fixtures/opensees/{entry['path']}/: {status}")
    if include_agent_reference:
        for entry in manifest["git"]:
            for name in [entry["path"], *entry.get("links", {})]:
                if not (AGENT_REFERENCE / name).exists():
                    problems.append(f"docs/agent-reference/{name}: missing")
    return problems


def cmd_check(include_agent_reference: bool) -> None:
    problems = collect_problems(load_manifest(), include_agent_reference)
    if problems:
        print("reference data incomplete; run: uv run scripts/setup_reference_data.py")
        for problem in problems:
            print(f"  {problem}")
        raise SystemExit(1)
    print("reference data OK")


def is_vendored(rel: PurePosixPath, size: int) -> bool:
    return rel.suffix.lower() in VENDORED_SUFFIXES and size <= MAX_VENDORED_TCL_BYTES


def cmd_vendor_wiki(scrape_root: Path) -> None:
    provenance_path = scrape_root / WIKI_PROVENANCE
    if not provenance_path.is_file():
        die(f"{provenance_path} not found; scrape with "
            "`uv run python scripts/setup-opensees-docs.py --root DIR` first")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    manifest = load_manifest() if MANIFEST.is_file() else {
        "version": 1, "sources": {}, "files": [], "trees": [], "git": []}

    # Replace everything previously vendored from the wiki.
    for name in WIKI_EXAMPLE_DIRS:
        shutil.rmtree(FIXTURE_ROOT / name, ignore_errors=True)
    wiki_prefixes = tuple(f"{name}/" for name in WIKI_EXAMPLE_DIRS)
    manifest["files"] = [e for e in manifest["files"] if not e["path"].startswith(wiki_prefixes)]

    vendored = 0
    data_entries: dict[str, dict] = {}
    for name in WIKI_EXAMPLE_DIRS:
        base = scrape_root / name
        if not base.is_dir():
            die(f"scrape is missing {base}")
        for path in sorted(p for p in base.rglob("*") if p.is_file()):
            rel = PurePosixPath(path.relative_to(scrape_root).as_posix())
            data = path.read_bytes()
            if is_vendored(rel, len(data)):
                atomic_write(FIXTURE_ROOT / rel, data)
                vendored += 1
                continue
            origin = provenance.get(rel.as_posix())
            if origin is None:
                die(f"no provenance for {rel}; it cannot be pinned to a download")
            source_sha = origin["url_sha256"]
            kind = "file" if origin["member"] is None else "zip"
            source = manifest["sources"].setdefault(
                source_sha, {"kind": kind, "urls": [origin["url"]]})
            if origin["url"] not in source["urls"]:
                source["urls"].append(origin["url"])
            entry = {
                "path": rel.as_posix(),
                "sha256": sha256_bytes(data),
                "size": len(data),
                "source": source_sha,
                "member": origin["member"],
            }
            manifest["files"].append(entry)
            data_entries[entry["path"]] = entry
            # Seed the cache so install does not re-download what we have.
            if not blob_path(source_sha).is_file() and kind == "file":
                atomic_write(blob_path(source_sha), data)
            atomic_write(FIXTURE_ROOT / rel, data)

    for alias, original in WIKI_ALIASES.items():
        if original not in data_entries:
            die(f"alias {alias} points at {original}, which the scrape did not produce")
        entry = {**data_entries[original], "path": alias}
        manifest["files"].append(entry)
        atomic_write(FIXTURE_ROOT / alias, (FIXTURE_ROOT / original).read_bytes())

    used = {e["source"] for e in [*manifest["files"], *manifest["trees"]]}
    manifest["sources"] = {k: v for k, v in manifest["sources"].items() if k in used}
    write_manifest(manifest)
    write_fixture_gitignore(manifest)
    log(f"vendored {vendored} Tcl files; manifest lists {len(manifest['files'])} data files "
        f"from {len(manifest['sources'])} sources")


def write_fixture_gitignore(manifest: dict) -> None:
    lines = [
        "# Generated by scripts/setup_reference_data.py vendor-wiki.",
        "# Pinned data that setup_reference_data.py installs; not committed.",
    ]
    lines += sorted(f"/{e['path']}" for e in manifest["files"])
    lines += sorted(f"/{e['path']}/" for e in manifest["trees"])
    FIXTURE_GITIGNORE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command")
    install = sub.add_parser("install", help="fetch and materialize everything (default)")
    install.add_argument("--fixtures-only", action="store_true",
                         help="skip docs/agent-reference (pinned source clones)")
    check = sub.add_parser("check", help="report missing or mismatched data")
    check.add_argument("--fixtures-only", action="store_true")
    vendor = sub.add_parser("vendor-wiki", help="maintainers: re-vendor from a wiki scrape")
    vendor.add_argument("--from", dest="scrape_root", type=Path, required=True)
    args = parser.parse_args()

    if args.command in (None, "install"):
        cmd_install(skip_agent_reference=getattr(args, "fixtures_only", False))
    elif args.command == "check":
        cmd_check(include_agent_reference=not args.fixtures_only)
    elif args.command == "vendor-wiki":
        cmd_vendor_wiki(args.scrape_root.resolve())


if __name__ == "__main__":
    main()
