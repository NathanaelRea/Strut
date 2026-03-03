#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import re
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import parse_qs, urljoin, urlparse
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup

BASE_URL = "https://opensees.berkeley.edu"
BASIC_MANUAL_URL = f"{BASE_URL}/wiki/index.php?title=Basic_Examples_Manual"
ADVANCED_MANUAL_URL = f"{BASE_URL}/wiki/index.php?title=Examples_Manual"

DOCS_DIR = Path("docs/agent-reference/OpenSeesDocs")
BASIC_EXAMPLES_DIR = Path("docs/agent-reference/OpenSeesExamplesBasic")
ADVANCED_EXAMPLES_DIR = Path("docs/agent-reference/OpenSeesExamplesAdvanced")

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


def fetch_html(url: str, logger: Logger | None = None) -> Page:
    if logger is not None:
        logger.detail(f"Fetching HTML: {url}")
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
    if parsed.path.startswith("/wiki/index.php/"):
        tail = parsed.path.split("/wiki/index.php/", 1)[1]
        return tail or None
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


def save_html(slug: str, html: str) -> Path:
    path = DOCS_DIR / f"{slug}.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path


def safe_extract_zip(
    data: bytes, target_dir: Path, logger: Logger | None = None
) -> int:
    extracted = 0
    resolved_target_dir = target_dir.resolve()
    with zipfile.ZipFile(io.BytesIO(data)) as zip_file:
        for member in zip_file.infolist():
            if member.is_dir():
                continue
            member_path = Path(member.filename)
            if member_path.is_absolute():
                continue
            resolved = (target_dir / member_path).resolve()
            if not str(resolved).startswith(str(resolved_target_dir)):
                continue
            resolved.parent.mkdir(parents=True, exist_ok=True)
            with zip_file.open(member) as source, open(resolved, "wb") as output:
                output.write(source.read())
            extracted += 1
            if logger is not None:
                logger.detail(f"Extracted {resolved.relative_to(resolved_target_dir)}")
    return extracted


def download_file(url: str, logger: Logger | None = None) -> bytes:
    if logger is not None:
        logger.detail(f"Downloading file: {url}")
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request) as response:
        data = response.read()
    if logger is not None:
        logger.detail(f"Downloaded file: {url} ({len(data)} bytes)")
    return data


def unique_links(links: Iterable[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for link in links:
        if link in seen:
            continue
        seen.add(link)
        ordered.append(link)
    return ordered


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
    for anchor in root.select("h2 span.mw-headline > a[href]"):
        href = anchor["href"].strip()
        if not href:
            continue
        link = urljoin(BASE_URL, href)
        title = extract_title(link)
        if not title or not title.startswith("OpenSees_Example_"):
            continue
        links.append(link)
    return unique_links(links)


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
        suffix = Path(urlparse(link).path).suffix.lower()
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
        zip_target_dir = spec.output_dir if spec.extract_zips_to_root else spec.output_dir / slugify(title)
        zip_target_dir.mkdir(parents=True, exist_ok=True)
        for zip_index, link in enumerate(zip_links, start=1):
            if logger is not None:
                logger.info(
                    f"{prefix}Downloading archive {zip_index}/{len(zip_links)} for {title}: {link}"
                )
            data = download_file(link, logger=logger)
            extracted_count += safe_extract_zip(data, zip_target_dir, logger=logger)
    else:
        example_dir = spec.output_dir / slugify(title)
        example_dir.mkdir(parents=True, exist_ok=True)
        for file_index, link in enumerate(file_links, start=1):
            filename = Path(urlparse(link).path).name or "example.tcl"
            if logger is not None:
                logger.info(
                    f"{prefix}Downloading file {file_index}/{len(file_links)} for {title}: {filename}"
                )
            data = download_file(link, logger=logger)
            (example_dir / filename).write_bytes(data)
            extracted_count += 1
            if logger is not None:
                logger.detail(f"Wrote {example_dir / filename}")
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


MANUAL_SPECS = {
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
}


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download OpenSees example manuals.")
    parser.add_argument(
        "--manual",
        choices=("all", "basic", "advanced"),
        default="all",
        help="Which example manual to download.",
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
    ensure_dirs()

    selected_specs = (
        list(MANUAL_SPECS.values())
        if args.manual == "all"
        else [MANUAL_SPECS[args.manual]]
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
            summary = download_example(
                link,
                spec,
                logger=logger,
                example_index=index,
                example_total=total,
            )
            logger.info(
                f"[{index}/{total}] Fetched {spec.name} {summary.title} from {link}"
            )
    if not any_links:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
