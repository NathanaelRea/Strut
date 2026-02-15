#!/usr/bin/env python3
from __future__ import annotations

import io
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qs, urljoin, urlparse
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup

BASE_URL = "https://opensees.berkeley.edu"
EXAMPLES_URL = f"{BASE_URL}/wiki/index.php/Examples"

DOCS_DIR = Path("docs/agent-reference/OpenSeesDocs")
EXAMPLES_DIR = Path("docs/agent-reference/OpenSeesExamplesBasic")

USER_AGENT = "strut-opensees-docs/1.0"


@dataclass(frozen=True)
class Page:
    url: str
    html: str


def fetch_html(url: str) -> Page:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request) as response:
        final_url = response.geturl()
        raw = response.read()
    html = raw.decode("utf-8", errors="replace")
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
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)


def save_html(slug: str, html: str) -> Path:
    path = DOCS_DIR / f"{slug}.html"
    path.write_text(html, encoding="utf-8")
    return path


def safe_extract_zip(data: bytes, target_dir: Path) -> None:
    with zipfile.ZipFile(io.BytesIO(data)) as zip_file:
        for member in zip_file.infolist():
            if member.is_dir():
                continue
            member_path = Path(member.filename)
            if member_path.is_absolute():
                continue
            resolved = (target_dir / member_path).resolve()
            if not str(resolved).startswith(str(target_dir.resolve())):
                continue
            resolved.parent.mkdir(parents=True, exist_ok=True)
            with zip_file.open(member) as source, open(resolved, "wb") as output:
                output.write(source.read())


def download_file(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request) as response:
        return response.read()


def collect_example_links() -> list[str]:
    examples_page = fetch_html(EXAMPLES_URL)
    examples_soup = BeautifulSoup(examples_page.html, "html.parser")
    root = content_root(examples_soup)

    manual_url = None
    for link in iter_wiki_links(root):
        title = extract_title(link)
        if title == "Basic_Examples_Manual":
            manual_url = link
            break

    if manual_url is None:
        raise RuntimeError("Could not find Basic_Examples_Manual link.")

    manual_page = fetch_html(manual_url)
    manual_soup = BeautifulSoup(manual_page.html, "html.parser")
    manual_root = content_root(manual_soup)

    links: dict[str, str] = {}
    for link in iter_wiki_links(manual_root):
        title = extract_title(link)
        if not title:
            continue
        if title in {"Examples", "Basic_Examples_Manual"}:
            continue
        if title.startswith("File:") or title.startswith("Help:"):
            continue
        links[title] = link

    return sorted(links.values())


def download_example(url: str) -> None:
    page = fetch_html(url)
    soup = BeautifulSoup(page.html, "html.parser")
    title = extract_title(page.url) or soup.title.string if soup.title else "example"
    title = title.strip() if isinstance(title, str) else "example"
    slug = slugify(title)

    save_html(slug, page.html)

    example_dir = EXAMPLES_DIR / slug
    example_dir.mkdir(parents=True, exist_ok=True)

    links = list(iter_wiki_links(content_root(soup)))
    zip_links = []
    tcl_links = []
    for link in links:
        path = urlparse(link).path.lower()
        if path.endswith(".zip"):
            zip_links.append(link)
        elif path.endswith(".tcl"):
            tcl_links.append(link)

    if zip_links:
        for link in sorted(set(zip_links)):
            data = download_file(link)
            safe_extract_zip(data, example_dir)
    else:
        for link in sorted(set(tcl_links)):
            filename = Path(urlparse(link).path).name or "example.tcl"
            data = download_file(link)
            (example_dir / filename).write_bytes(data)


def main() -> int:
    ensure_dirs()
    links = collect_example_links()
    if not links:
        print("No example links found.")
        return 1

    for link in links:
        download_example(link)
        print(f"Fetched {link}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
