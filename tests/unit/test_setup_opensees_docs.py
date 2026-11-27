import importlib.util
import sys
from pathlib import Path

from bs4 import BeautifulSoup

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_scraper():
    path = REPO_ROOT / "scripts" / "setup-opensees-docs.py"
    spec = importlib.util.spec_from_file_location("setup_opensees_docs", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


scraper = _load_scraper()

FILE_PAGE = b"""<!DOCTYPE html><html><body><div id="mw-content-text">
<a href="/wiki/index.php/Main_Page">home</a>
<a href="/wiki/images/9/9c/Pushover_concentrated.tcl">Pushover_concentrated.tcl</a>
</div></body></html>"""


def test_file_page_resolves_to_upload():
    assert scraper.looks_like_html(FILE_PAGE)
    assert not scraper.looks_like_html(b"# a tcl script\nwipe\n")
    assert (
        scraper.file_page_upload_url(FILE_PAGE)
        == "https://opensees.berkeley.edu/wiki/images/9/9c/Pushover_concentrated.tcl"
    )


def test_file_page_without_upload():
    page = b"<html><body><div id='mw-content-text'>no file</div></body></html>"
    assert scraper.file_page_upload_url(page) is None


def test_inline_scripts_join_selected_blocks(tmp_path):
    html = """<html><body><div id="mw-content-text">
<pre>tcl block one
</pre><p>prose</p><pre>matlab block</pre></div></body></html>"""
    soup = BeautifulSoup(html, "html.parser")
    written = scraper.write_inline_scripts(
        "Calling_Matlab_from_a_Scipt", soup, tmp_path, "https://example.invalid"
    )
    assert written == 2
    assert (tmp_path / "CallingMatlab.tcl").read_text() == "tcl block one\n"
    assert (tmp_path / "getA.m").read_text() == "matlab block\n"
    assert scraper.write_inline_scripts("Other_Page", soup, tmp_path, "x") == 0


def test_practical_collector_skips_indexes_and_megatall():
    html = """<div>
<a href="/wiki/index.php?title=Basic_Examples_Manual">b</a>
<a href="/wiki/index.php?title=Mega-Tall_Building_Benchmark_Models">m</a>
<a href="/wiki/index.php?title=File:x.tcl">f</a>
<a href="/wiki/index.php?title=SCBF_Model">s</a>
<a href="https://opensees.ist.berkeley.edu/wiki/index.php?title=Examples&amp;oldid=1">o</a>
</div>"""
    links = scraper.collect_practical_example_links(BeautifulSoup(html, "html.parser"))
    assert [scraper.extract_title(link) for link in links] == ["SCBF_Model"]
