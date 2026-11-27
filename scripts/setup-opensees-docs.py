#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import io
import re
import shutil
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable
from urllib.error import HTTPError
from urllib.parse import parse_qs, unquote, urlencode, urljoin, urlparse
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup

BASE_URL = "https://opensees.berkeley.edu"
BASIC_MANUAL_URL = f"{BASE_URL}/wiki/index.php?title=Basic_Examples_Manual"
ADVANCED_MANUAL_URL = f"{BASE_URL}/wiki/index.php?title=Examples_Manual"
# The Examples index itself lists the practical structural, geotechnical, and
# parallel examples; the sensitivity examples hang off their own index.
PRACTICAL_INDEX_URL = f"{BASE_URL}/wiki/index.php?title=Examples"
SENSITIVITY_INDEX_URL = f"{BASE_URL}/wiki/index.php?title=Demonstration_Examples:"
REQUEST_INTERVAL_S = 1.0

DEFAULT_ROOT = Path("docs/agent-reference")
DOCS_DIR = DEFAULT_ROOT / "OpenSeesDocs"
BASIC_EXAMPLES_DIR = DEFAULT_ROOT / "OpenSeesExamplesBasic"
ADVANCED_EXAMPLES_DIR = DEFAULT_ROOT / "OpenSeesExamplesAdvanced"
PRACTICAL_EXAMPLES_DIR = DEFAULT_ROOT / "OpenSeesExamplesPractical"
SENSITIVITY_EXAMPLES_DIR = DEFAULT_ROOT / "OpenSeesExamplesSensitivity"
PROVENANCE_FILENAME = "opensees-wiki-provenance.json"

# Where each materialized file came from, keyed by resolved path:
# {"url": <downloaded URL>, "url_sha256": <sha256 of the downloaded bytes>,
#  "member": <zip member name or None>}.
# scripts/setup_reference_data.py vendor-wiki reads the written copy.
PROVENANCE: dict[Path, dict[str, str | None]] = {}

USER_AGENT = "strut-opensees-docs/1.0"
DIRECT_FILE_SUFFIXES = {
    ".acc",
    ".at2",
    ".dat",
    ".dt2",
    ".g3",
    ".out",
    ".tcl",
    ".txt",
    ".zip",
}

SHARED_EXAMPLE_FILE_SUFFIXES = {
    ".acc",
    ".at2",
    ".dat",
    ".dt2",
    ".g3",
    ".out",
    ".txt",
}

# GiD post-processing output shipped inside some geotechnical zips. Results,
# not analysis inputs, and tens of MB each.
SKIPPED_ZIP_MEMBER_SUFFIXES = (".flavia.res", ".flavia.msh", ".post.res", ".post.msh")

# Pages whose script is only printed inline, in <pre> blocks, rather than
# uploaded. Title -> (filename to write, <pre> block indices to join; None = all).
INLINE_SCRIPTS: dict[str, tuple[tuple[str, tuple[int, ...] | None], ...]] = {
    # The RCFrameEarthquake.tcl upload is a redlink; the page prints it in parts.
    "Reinforced_Concrete_Frame_Earthquake_Analysis": (("RCFrameEarthquake.tcl", None),),
    "Rigid_Diaphragm_Consequences": (("RigidDiaphragmConsequences.tcl", (0,)),),
    "Calling_Matlab_from_a_Scipt": (
        ("CallingMatlab.tcl", (0,)),
        ("getA.m", (1,)),
    ),
}

KNOWN_TITLE_REWRITES = {
    "OpenSees_Example_3._Cantilever_Column_with_unit": "OpenSees_Example_3._Cantilever_Column_with_units",
}


@dataclass(frozen=True)
class Page:
    url: str
    html: str


@dataclass(frozen=True)
class ManualSpec:
    name: str
    manual_url: str
    output_dir: Path
    link_collector: Callable[[BeautifulSoup], list[str]]
    extract_zips_to_root: bool = False


@dataclass(frozen=True)
class DownloadSummary:
    title: str
    zip_count: int
    file_count: int
    extracted_count: int


class Logger:
    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose

    def info(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}", file=sys.stderr, flush=True)

    def detail(self, message: str) -> None:
        if self.verbose:
            self.info(message)


class RequestThrottle:
    def __init__(self, min_interval_s: float = REQUEST_INTERVAL_S) -> None:
        self.min_interval_s = min_interval_s
        self._last_request_started_at: float | None = None

    def wait(self, logger: Logger | None = None) -> None:
        if self.min_interval_s <= 0:
            self._last_request_started_at = time.monotonic()
            return
        now = time.monotonic()
        if self._last_request_started_at is None:
            self._last_request_started_at = now
            return
        elapsed = now - self._last_request_started_at
        remaining = self.min_interval_s - elapsed
        if remaining > 0:
            if logger is not None and logger.verbose:
                logger.detail(f"Throttling requests for {remaining:.2f}s")
            time.sleep(remaining)
            now = time.monotonic()
        self._last_request_started_at = now


REQUEST_THROTTLE = RequestThrottle()


def fetch_html(url: str, logger: Logger | None = None) -> Page:
    if logger is not None:
        logger.detail(f"Fetching HTML: {url}")
    REQUEST_THROTTLE.wait(logger=logger)
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request) as response:
        final_url = response.geturl()
        raw = response.read()
    html = raw.decode("utf-8", errors="replace")
    if logger is not None:
        logger.detail(f"Fetched HTML: {final_url} ({len(raw)} bytes)")
    return Page(url=final_url, html=html)


def content_root(soup: BeautifulSoup) -> BeautifulSoup:
    for selector in ("#mw-content-text", "#bodyContent", "main", "body"):
        found = soup.select_one(selector)
        if found is not None:
            return found
    return soup


def extract_title(url: str) -> str | None:
    parsed = urlparse(url)
    if parsed.query:
        params = parse_qs(parsed.query)
        title = params.get("title", [None])[0]
        if title:
            return title
    if parsed.path.startswith("/wiki/"):
        tail = parsed.path.split("/wiki/", 1)[1]
        if tail.startswith("index.php/"):
            title = tail.split("index.php/", 1)[1]
            return unquote(title) or None
        if tail and "/" not in tail:
            return unquote(tail) or None
    if parsed.path.startswith("/wiki/index.php/"):
        tail = parsed.path.split("/wiki/index.php/", 1)[1]
        return unquote(tail) or None
    return None


def slugify(title: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", title).strip("_").lower()
    return slug or "example"


def iter_wiki_links(root: BeautifulSoup) -> Iterable[str]:
    for anchor in root.find_all("a", href=True):
        href = anchor["href"].strip()
        if not href or href.startswith("#"):
            continue
        if "action=edit" in href or "redlink=1" in href:
            continue
        if href.startswith("mailto:"):
            continue
        yield urljoin(BASE_URL, href)


def ensure_dirs() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    BASIC_EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    ADVANCED_EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    PRACTICAL_EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    SENSITIVITY_EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)


def save_html(slug: str, html: str) -> Path:
    path = DOCS_DIR / f"{slug}.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path


def safe_extract_zip(
    data: bytes,
    target_dir: Path,
    logger: Logger | None = None,
    source_url: str | None = None,
) -> int:
    extracted = 0
    source_sha256 = hashlib.sha256(data).hexdigest()
    resolved_target_dir = target_dir.resolve()
    with zipfile.ZipFile(io.BytesIO(data)) as zip_file:
        for member in zip_file.infolist():
            if member.is_dir():
                continue
            member_path = Path(member.filename)
            if member.filename.lower().endswith(SKIPPED_ZIP_MEMBER_SUFFIXES):
                continue
            if member_path.is_absolute():
                continue
            resolved = (target_dir / member_path).resolve()
            if not str(resolved).startswith(str(resolved_target_dir)):
                continue
            resolved.parent.mkdir(parents=True, exist_ok=True)
            with zip_file.open(member) as source, open(resolved, "wb") as output:
                output.write(source.read())
            PROVENANCE[resolved] = {
                "url": source_url,
                "url_sha256": source_sha256,
                "member": member.filename,
            }
            extracted += 1
            if logger is not None:
                logger.detail(f"Extracted {resolved.relative_to(resolved_target_dir)}")
    return extracted


def download_file(url: str, logger: Logger | None = None) -> bytes:
    if logger is not None:
        logger.detail(f"Downloading file: {url}")
    REQUEST_THROTTLE.wait(logger=logger)
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request) as response:
        data = response.read()
    if logger is not None:
        logger.detail(f"Downloaded file: {url} ({len(data)} bytes)")
    return data


def looks_like_html(data: bytes) -> bool:
    head = data[:512].lstrip().lower()
    return head.startswith((b"<!doctype html", b"<html"))


def file_page_upload_url(data: bytes) -> str | None:
    """Return the raw upload a MediaWiki ``File:`` description page links to."""
    soup = BeautifulSoup(data.decode("utf-8", errors="replace"), "html.parser")
    for link in iter_wiki_links(content_root(soup)):
        if "/wiki/images/" in urlparse(link).path:
            return link
    return None


def write_inline_scripts(
    title: str, soup: BeautifulSoup, example_dir: Path, page_url: str
) -> int:
    specs = INLINE_SCRIPTS.get(title, ())
    if not specs:
        return 0
    blocks = [pre.get_text() for pre in content_root(soup).select("pre")]
    written = 0
    for filename, indices in specs:
        chosen = blocks if indices is None else [blocks[i] for i in indices]
        text = "\n".join(block.rstrip() + "\n" for block in chosen)
        data = text.encode("utf-8")
        example_dir.mkdir(parents=True, exist_ok=True)
        (example_dir / filename).write_bytes(data)
        PROVENANCE[(example_dir / filename).resolve()] = {
            "url": page_url,
            "url_sha256": hashlib.sha256(data).hexdigest(),
            "member": None,
        }
        written += 1
    return written


def _iter_shared_file_reference_texts(text: str) -> Iterable[str]:
    pattern = r'["\']([^"\']+\.(?:acc|at2|dat|dt2|g3|out|txt))["\']'
    for match in re.finditer(pattern, text, flags=re.IGNORECASE):
        yield match.group(1).strip()


def _resolve_shared_example_source(
    reference: str, example_dir: Path, shared_root: Path
) -> tuple[Path | None, Path]:
    ref_path = Path(reference)
    target = example_dir / ref_path
    candidates: list[Path] = []
    if ref_path.parts:
        candidates.append(shared_root / ref_path)
    candidates.append(shared_root / ref_path.name)
    candidates.append(shared_root / "GMfiles" / ref_path.name)

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved, target
    return None, target


def copy_referenced_shared_files(
    example_dir: Path, shared_root: Path, logger: Logger | None = None
) -> int:
    copied = 0
    seen_targets: set[Path] = set()
    for tcl_path in sorted(example_dir.glob("*.tcl")):
        text = tcl_path.read_text(encoding="utf-8", errors="replace")
        for reference in _iter_shared_file_reference_texts(text):
            source, target = _resolve_shared_example_source(
                reference, example_dir, shared_root
            )
            if source is None:
                continue
            target_resolved = target.resolve()
            if target_resolved in seen_targets or target_resolved.exists():
                seen_targets.add(target_resolved)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            if source in PROVENANCE:
                PROVENANCE[target_resolved] = dict(PROVENANCE[source])
            seen_targets.add(target_resolved)
            copied += 1
            if logger is not None:
                logger.detail(f"Copied shared file {source.name} -> {target}")
    return copied


def unique_links(links: Iterable[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for link in links:
        if link in seen:
            continue
        seen.add(link)
        ordered.append(link)
    return ordered


def rewrite_known_title(link: str) -> str:
    parsed = urlparse(link)
    title = extract_title(link)
    if not title:
        return link
    rewritten_title = KNOWN_TITLE_REWRITES.get(title)
    if not rewritten_title:
        return link

    if parsed.query:
        params = parse_qs(parsed.query)
        if "title" in params:
            params["title"] = [rewritten_title]
            return parsed._replace(query=urlencode(params, doseq=True)).geturl()

    if parsed.path.startswith("/wiki/index.php/"):
        return parsed._replace(path=f"/wiki/index.php/{rewritten_title}").geturl()
    if parsed.path.startswith("/wiki/"):
        return parsed._replace(path=f"/wiki/{rewritten_title}").geturl()
    return link


def collect_basic_example_links(root: BeautifulSoup) -> list[str]:
    links: list[str] = []
    for link in iter_wiki_links(root):
        title = extract_title(link)
        if not title:
            continue
        if title in {"Examples", "Basic_Examples_Manual"}:
            continue
        if title.startswith("File:") or title.startswith("Help:"):
            continue
        links.append(link)
    return unique_links(links)


def collect_advanced_example_links(root: BeautifulSoup) -> list[str]:
    links: list[str] = []
    for link in iter_wiki_links(root):
        link = rewrite_known_title(link)
        title = extract_title(link)
        if not title or not title.startswith("OpenSees_Example_"):
            continue
        links.append(link)
    return unique_links(links)


# Pages on the Examples index that are not practical examples: the other
# indexes, and the mega-tall model, which Strut generates (its .rar is external).
PRACTICAL_SKIP_TITLES = {
    "Examples",
    "Basic_Examples_Manual",
    "Examples_Manual",
    "Sensitivity_Analysis",
    "Mega-Tall_Building_Benchmark_Models",
}


def collect_practical_example_links(root: BeautifulSoup) -> list[str]:
    links: list[str] = []
    for link in iter_wiki_links(root):
        if urlparse(link).netloc != urlparse(BASE_URL).netloc:
            continue
        title = extract_title(link)
        if not title or title in PRACTICAL_SKIP_TITLES or ":" in title:
            continue
        links.append(link)
    return unique_links(links)


def collect_sensitivity_example_links(root: BeautifulSoup) -> list[str]:
    links: list[str] = []
    for link in iter_wiki_links(root):
        title = extract_title(link)
        if title and title.startswith("Example_"):
            links.append(link)
    return unique_links(links)


def link_filename(url: str) -> str | None:
    title = extract_title(url)
    if title:
        title_name = title.removeprefix("File:").removeprefix("Image:")
        title_filename = Path(title_name).name
        if Path(title_filename).suffix:
            return unquote(title_filename)

    path_name = unquote(Path(urlparse(url).path).name)
    if path_name.startswith("File:") or path_name.startswith("Image:"):
        path_name = path_name.split(":", 1)[1]
    if Path(path_name).suffix:
        return path_name
    return None


def collect_manual_example_links(
    spec: ManualSpec, logger: Logger | None = None
) -> list[str]:
    if logger is not None:
        logger.info(f"Loading {spec.name} manual: {spec.manual_url}")
    manual_page = fetch_html(spec.manual_url, logger=logger)
    manual_soup = BeautifulSoup(manual_page.html, "html.parser")
    manual_root = content_root(manual_soup)
    links = spec.link_collector(manual_root)
    if logger is not None:
        logger.info(f"Found {len(links)} {spec.name} example pages")
    return links


def direct_file_links(root: BeautifulSoup) -> tuple[list[str], list[str]]:
    zip_links: list[str] = []
    file_links: list[str] = []
    for link in iter_wiki_links(root):
        filename = link_filename(link)
        if not filename:
            continue
        suffix = Path(filename).suffix.lower()
        if suffix not in DIRECT_FILE_SUFFIXES:
            continue
        if suffix == ".zip":
            zip_links.append(link)
        else:
            file_links.append(link)
    return unique_links(zip_links), unique_links(file_links)


def download_example(
    url: str,
    spec: ManualSpec,
    logger: Logger | None = None,
    example_index: int | None = None,
    example_total: int | None = None,
) -> DownloadSummary:
    prefix = ""
    if example_index is not None and example_total is not None:
        prefix = f"[{example_index}/{example_total}] "
    if logger is not None:
        logger.info(f"{prefix}Fetching example page: {url}")
    page = fetch_html(url, logger=logger)
    soup = BeautifulSoup(page.html, "html.parser")
    page_title = extract_title(page.url)
    if not page_title and soup.title:
        page_title = soup.title.string
    title = page_title or "example"
    title = title.strip() if isinstance(title, str) else "example"
    slug = slugify(f"{spec.name}_{title}")

    save_html(slug, page.html)
    zip_links, file_links = direct_file_links(content_root(soup))
    if logger is not None:
        logger.info(
            f"{prefix}Resolved {title}: {len(zip_links)} zip link(s), {len(file_links)} direct file link(s)"
        )
    extracted_count = 0

    if zip_links:
        zip_target_dir = (
            spec.output_dir
            if spec.extract_zips_to_root
            else spec.output_dir / slugify(title)
        )
        zip_target_dir.mkdir(parents=True, exist_ok=True)
        for zip_index, link in enumerate(zip_links, start=1):
            if logger is not None:
                logger.info(
                    f"{prefix}Downloading archive {zip_index}/{len(zip_links)} for {title}: {link}"
                )
            data = download_file(link, logger=logger)
            extracted_count += safe_extract_zip(
                data, zip_target_dir, logger=logger, source_url=link
            )

    if file_links:
        example_dir = spec.output_dir / slugify(title)
        example_dir.mkdir(parents=True, exist_ok=True)
        for file_index, link in enumerate(file_links, start=1):
            filename = link_filename(link) or "example.tcl"
            if logger is not None:
                logger.info(
                    f"{prefix}Downloading file {file_index}/{len(file_links)} for {title}: {filename}"
                )
            data = download_file(link, logger=logger)
            if looks_like_html(data):
                # A link to the wiki's File: page, not the upload itself.
                upload = file_page_upload_url(data)
                if upload is None:
                    if logger is not None:
                        logger.info(
                            f"{prefix}Skipping {filename}: wiki page, no upload"
                        )
                    continue
                link = upload
                filename = link_filename(link) or filename
                existing = {p.name.lower() for p in example_dir.iterdir()}
                if filename.lower() in existing:
                    # The example's zip already has the original-case file.
                    continue
                data = download_file(link, logger=logger)
            (example_dir / filename).write_bytes(data)
            PROVENANCE[(example_dir / filename).resolve()] = {
                "url": link,
                "url_sha256": hashlib.sha256(data).hexdigest(),
                "member": None,
            }
            extracted_count += 1
            if logger is not None:
                logger.detail(f"Wrote {example_dir / filename}")
        copy_referenced_shared_files(example_dir, spec.output_dir, logger=logger)
    extracted_count += write_inline_scripts(
        page_title or "", soup, spec.output_dir / slugify(title), page.url
    )
    if logger is not None:
        logger.info(
            f"{prefix}Completed {title}: {extracted_count} file(s) materialized"
        )
    return DownloadSummary(
        title=title,
        zip_count=len(zip_links),
        file_count=len(file_links),
        extracted_count=extracted_count,
    )


def build_manual_specs() -> dict[str, ManualSpec]:
    return {
        "basic": ManualSpec(
            name="basic",
            manual_url=BASIC_MANUAL_URL,
            output_dir=BASIC_EXAMPLES_DIR,
            link_collector=collect_basic_example_links,
        ),
        "advanced": ManualSpec(
            name="advanced",
            manual_url=ADVANCED_MANUAL_URL,
            output_dir=ADVANCED_EXAMPLES_DIR,
            link_collector=collect_advanced_example_links,
            extract_zips_to_root=True,
        ),
        "practical": ManualSpec(
            name="practical",
            manual_url=PRACTICAL_INDEX_URL,
            output_dir=PRACTICAL_EXAMPLES_DIR,
            link_collector=collect_practical_example_links,
        ),
        "sensitivity": ManualSpec(
            name="sensitivity",
            manual_url=SENSITIVITY_INDEX_URL,
            output_dir=SENSITIVITY_EXAMPLES_DIR,
            link_collector=collect_sensitivity_example_links,
        ),
    }


def configure_root(root: Path) -> None:
    global DOCS_DIR, BASIC_EXAMPLES_DIR, ADVANCED_EXAMPLES_DIR
    global PRACTICAL_EXAMPLES_DIR, SENSITIVITY_EXAMPLES_DIR
    DOCS_DIR = root / "OpenSeesDocs"
    BASIC_EXAMPLES_DIR = root / "OpenSeesExamplesBasic"
    ADVANCED_EXAMPLES_DIR = root / "OpenSeesExamplesAdvanced"
    PRACTICAL_EXAMPLES_DIR = root / "OpenSeesExamplesPractical"
    SENSITIVITY_EXAMPLES_DIR = root / "OpenSeesExamplesSensitivity"


def write_provenance(root: Path) -> Path:
    resolved_root = root.resolve()
    entries = {}
    for path, source in sorted(PROVENANCE.items()):
        if not path.is_file():
            continue
        entries[path.relative_to(resolved_root).as_posix()] = {
            **source,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size": path.stat().st_size,
        }
    out = root / PROVENANCE_FILENAME
    out.write_text(
        json.dumps(entries, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return out


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download OpenSees example manuals.")
    parser.add_argument(
        "--manual",
        choices=("all", "basic", "advanced", "practical", "sensitivity"),
        default="all",
        help="Which example manual to download.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Output root for docs and examples. Default: {DEFAULT_ROOT}",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log individual network fetches and extracted files.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    logger = Logger(verbose=args.verbose)
    configure_root(args.root)
    ensure_dirs()

    manual_specs = build_manual_specs()
    selected_specs = (
        list(manual_specs.values())
        if args.manual == "all"
        else [manual_specs[args.manual]]
    )
    any_links = False
    for spec in selected_specs:
        links = collect_manual_example_links(spec, logger=logger)
        if not links:
            print(f"No {spec.name} example links found.")
            continue
        any_links = True
        total = len(links)
        for index, link in enumerate(links, start=1):
            try:
                summary = download_example(
                    link,
                    spec,
                    logger=logger,
                    example_index=index,
                    example_total=total,
                )
            except HTTPError as exc:
                logger.info(
                    f"[{index}/{total}] Skipping {spec.name} example due to HTTP {exc.code}: {link}"
                )
                continue
            logger.info(
                f"[{index}/{total}] Fetched {spec.name} {summary.title} from {link}"
            )
    if not any_links:
        return 1
    logger.info(f"Wrote provenance: {write_provenance(args.root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
