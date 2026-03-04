import importlib.util
import io
import sys
import zipfile
from pathlib import Path
from urllib.error import HTTPError


REPO_ROOT = Path(__file__).resolve().parents[2]
SETUP_OPENSEES_DOCS_PATH = REPO_ROOT / "scripts" / "setup-opensees-docs.py"


def _load_setup_opensees_docs_module():
    module_name = "strut_setup_opensees_docs_test_module"
    spec = importlib.util.spec_from_file_location(module_name, SETUP_OPENSEES_DOCS_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


setup_opensees_docs = _load_setup_opensees_docs_module()


def _zip_bytes(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def test_request_throttle_limits_to_one_request_per_second(monkeypatch):
    current = 100.0
    sleeps: list[float] = []

    def fake_monotonic():
        return current

    def fake_sleep(duration: float):
        nonlocal current
        sleeps.append(duration)
        current += duration

    monkeypatch.setattr(setup_opensees_docs.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(setup_opensees_docs.time, "sleep", fake_sleep)

    throttle = setup_opensees_docs.RequestThrottle(min_interval_s=1.0)
    throttle.wait()
    current += 0.25
    throttle.wait()
    current += 1.5
    throttle.wait()

    assert sleeps == [0.75]


def test_collect_advanced_example_links_filters_non_examples_and_dedupes():
    root = setup_opensees_docs.BeautifulSoup(
        """
        <div id="mw-content-text">
          <h2>
            <span class="mw-headline" id="OpenSees_Example_5">
              <a href="/wiki/index.php?title=OpenSees_Example_5._2D_Frame">Example 5</a>
            </span>
          </h2>
          <p><a href="/wiki/index.php?title=Elastic_Beam_Column_Element">Elastic Beam Column Element</a></p>
          <h2>
            <span class="mw-headline" id="OpenSees_Example_6">
              <a href="/wiki/index.php?title=OpenSees_Example_6._generic_2D_Frame">Example 6</a>
            </span>
          </h2>
          <h2>
            <span class="mw-headline" id="OpenSees_Example_5_duplicate">
              <a href="/wiki/index.php?title=OpenSees_Example_5._2D_Frame">Example 5 duplicate</a>
            </span>
          </h2>
        </div>
        """,
        "html.parser",
    )

    links = setup_opensees_docs.collect_advanced_example_links(root)

    assert links == [
        "https://opensees.berkeley.edu/wiki/index.php?title=OpenSees_Example_5._2D_Frame",
        "https://opensees.berkeley.edu/wiki/index.php?title=OpenSees_Example_6._generic_2D_Frame",
    ]


def test_collect_advanced_example_links_supports_pretty_wiki_paths():
    root = setup_opensees_docs.BeautifulSoup(
        """
        <div id="mw-content-text">
          <p><a href="/wiki/OpenSees_Example_1a._2D_Elastic_Cantilever_Column">Example 1a</a></p>
          <p><a href="/wiki/OpenSees_Example_2a._Elastic_Cantilever_Column_with_Variables">Example 2a</a></p>
          <p><a href="/wiki/Examples_Manual">Examples Manual</a></p>
        </div>
        """,
        "html.parser",
    )

    links = setup_opensees_docs.collect_advanced_example_links(root)

    assert links == [
        "https://opensees.berkeley.edu/wiki/OpenSees_Example_1a._2D_Elastic_Cantilever_Column",
        "https://opensees.berkeley.edu/wiki/OpenSees_Example_2a._Elastic_Cantilever_Column_with_Variables",
    ]


def test_collect_advanced_example_links_rewrites_known_broken_titles():
    root = setup_opensees_docs.BeautifulSoup(
        """
        <div id="mw-content-text">
          <p><a href="/wiki/index.php?title=OpenSees_Example_3._Cantilever_Column_with_unit">Example 3</a></p>
        </div>
        """,
        "html.parser",
    )

    links = setup_opensees_docs.collect_advanced_example_links(root)

    assert links == [
        "https://opensees.berkeley.edu/wiki/index.php?title=OpenSees_Example_3._Cantilever_Column_with_units"
    ]


def test_direct_file_links_support_mediawiki_file_titles():
    root = setup_opensees_docs.BeautifulSoup(
        """
        <div id="mw-content-text">
          <a href="/wiki/index.php?title=File:Ex5Fram2DEQ.zip">Archive</a>
          <a href="/wiki/File:Ex5.Frame2D.build.ElasticSection.tcl">Tcl file</a>
          <a href="/wiki/index.php?title=File:Figure.gif">Figure</a>
        </div>
        """,
        "html.parser",
    )

    zip_links, file_links = setup_opensees_docs.direct_file_links(root)

    assert zip_links == [
        "https://opensees.berkeley.edu/wiki/index.php?title=File:Ex5Fram2DEQ.zip"
    ]
    assert file_links == [
        "https://opensees.berkeley.edu/wiki/File:Ex5.Frame2D.build.ElasticSection.tcl"
    ]


def test_download_advanced_example_extracts_zip_into_shared_root(
    monkeypatch, tmp_path: Path
):
    spec = setup_opensees_docs.ManualSpec(
        name="advanced",
        manual_url="https://example.test/manual",
        output_dir=tmp_path / "OpenSeesExamplesAdvanced",
        link_collector=setup_opensees_docs.collect_advanced_example_links,
        extract_zips_to_root=True,
    )
    spec.output_dir.mkdir(parents=True, exist_ok=True)

    example_url = "https://opensees.berkeley.edu/wiki/index.php?title=OpenSees_Example_5._2D_Frame"
    page_html = """
        <div id="mw-content-text">
          <a href="/wiki/images/archive/Ex5Fram2DEQ.zip">Download all files in zip format</a>
          <a href="/wiki/images/scripts/Ex5.Frame2D.analyze.Dynamic.EQ.Uniform.tcl">Direct Tcl</a>
        </div>
    """
    monkeypatch.setattr(
        setup_opensees_docs,
        "fetch_html",
        lambda url, logger=None: setup_opensees_docs.Page(url=url, html=page_html),
    )

    def fake_download_file(url, logger=None):
        if url.endswith("Ex5Fram2DEQ.zip"):
            return _zip_bytes(
                {
                    "Ex5Fram2DEQUniform/Ex5.Frame2D.analyze.Dynamic.EQ.Uniform.tcl": b"puts test\n",
                    "GMfiles/H-e12140.g3": b"0.0 1.0\n",
                }
            )
        if url.endswith("Ex5.Frame2D.analyze.Dynamic.EQ.Uniform.tcl"):
            return b"puts direct\n"
        raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr(
        setup_opensees_docs,
        "download_file",
        fake_download_file,
    )

    setup_opensees_docs.download_example(example_url, spec)

    assert (
        spec.output_dir
        / "Ex5Fram2DEQUniform"
        / "Ex5.Frame2D.analyze.Dynamic.EQ.Uniform.tcl"
    ).read_text(encoding="utf-8") == "puts test\n"
    assert (spec.output_dir / "GMfiles" / "H-e12140.g3").read_text(
        encoding="utf-8"
    ) == "0.0 1.0\n"
    assert (
        spec.output_dir / "opensees_example_5_2d_frame"
    ).exists()
    assert (
        spec.output_dir
        / "opensees_example_5_2d_frame"
        / "Ex5.Frame2D.analyze.Dynamic.EQ.Uniform.tcl"
    ).read_text(encoding="utf-8") == "puts direct\n"


def test_download_advanced_example_materializes_direct_files_even_when_zip_links_exist(
    monkeypatch, tmp_path: Path
):
    spec = setup_opensees_docs.ManualSpec(
        name="advanced",
        manual_url="https://example.test/manual",
        output_dir=tmp_path / "OpenSeesExamplesAdvanced",
        link_collector=setup_opensees_docs.collect_advanced_example_links,
        extract_zips_to_root=True,
    )
    spec.output_dir.mkdir(parents=True, exist_ok=True)

    example_url = "https://opensees.berkeley.edu/wiki/index.php?title=OpenSees_Example_1a._2D_Elastic_Cantilever_Column"
    page_html = """
        <div id="mw-content-text">
          <a href="/wiki/images/6/63/BM68elc.zip">BM68elc.acc</a>
          <a href="/wiki/images/f/f6/Ex1a.Canti2D.Push.tcl">Push Tcl</a>
          <a href="/wiki/images/4/45/Ex1a.Canti2D.EQ.tcl">EQ Tcl</a>
        </div>
    """
    monkeypatch.setattr(
        setup_opensees_docs,
        "fetch_html",
        lambda url, logger=None: setup_opensees_docs.Page(url=url, html=page_html),
    )

    def fake_download_file(url, logger=None):
        if url.endswith("BM68elc.zip"):
            return _zip_bytes({"BM68elc.acc": b"0.0 1.0\n"})
        if url.endswith("Ex1a.Canti2D.Push.tcl"):
            return b"puts push\n"
        if url.endswith("Ex1a.Canti2D.EQ.tcl"):
            return b'set GMfile "BM68elc.acc"\nputs eq\n'
        raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr(
        setup_opensees_docs,
        "download_file",
        fake_download_file,
    )

    summary = setup_opensees_docs.download_example(example_url, spec)

    assert summary.extracted_count == 3
    assert (spec.output_dir / "BM68elc.acc").read_text(encoding="utf-8") == "0.0 1.0\n"
    assert (
        spec.output_dir
        / "opensees_example_1a_2d_elastic_cantilever_column"
        / "Ex1a.Canti2D.Push.tcl"
    ).read_text(encoding="utf-8") == "puts push\n"
    assert (
        spec.output_dir
        / "opensees_example_1a_2d_elastic_cantilever_column"
        / "Ex1a.Canti2D.EQ.tcl"
    ).read_text(encoding="utf-8") == 'set GMfile "BM68elc.acc"\nputs eq\n'
    assert (
        spec.output_dir
        / "opensees_example_1a_2d_elastic_cantilever_column"
        / "BM68elc.acc"
    ).read_text(encoding="utf-8") == "0.0 1.0\n"


def test_copy_referenced_shared_files_copies_root_and_gmfiles_assets(
    tmp_path: Path,
):
    shared_root = tmp_path / "OpenSeesExamplesAdvanced"
    example_dir = shared_root / "opensees_example_3_cantilever_column_with_units"
    gm_dir = shared_root / "GMfiles"
    example_dir.mkdir(parents=True, exist_ok=True)
    gm_dir.mkdir(parents=True, exist_ok=True)
    (shared_root / "BM68elc.acc").write_text("root gm\n", encoding="utf-8")
    (gm_dir / "H-e12140.g3").write_text("gmfiles gm\n", encoding="utf-8")
    (example_dir / "case.tcl").write_text(
        '\n'.join(
            [
                'set GMfile "BM68elc.acc"',
                'set AltGM "GMfiles/H-e12140.g3"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    copied = setup_opensees_docs.copy_referenced_shared_files(example_dir, shared_root)

    assert copied == 2
    assert (example_dir / "BM68elc.acc").read_text(encoding="utf-8") == "root gm\n"
    assert (example_dir / "GMfiles" / "H-e12140.g3").read_text(
        encoding="utf-8"
    ) == "gmfiles gm\n"


def test_download_basic_example_writes_direct_files_into_page_slug_directory(
    monkeypatch, tmp_path: Path
):
    spec = setup_opensees_docs.ManualSpec(
        name="basic",
        manual_url="https://example.test/manual",
        output_dir=tmp_path / "OpenSeesExamplesBasic",
        link_collector=setup_opensees_docs.collect_basic_example_links,
    )
    spec.output_dir.mkdir(parents=True, exist_ok=True)

    example_url = "https://opensees.berkeley.edu/wiki/index.php?title=Elastic_Frame_Example"
    page_html = """
        <div id="mw-content-text">
          <a href="/wiki/images/scripts/ElasticFrame.tcl">ElasticFrame.tcl</a>
          <a href="/wiki/images/figures/ElasticFrame.gif">Figure</a>
        </div>
    """
    monkeypatch.setattr(
        setup_opensees_docs,
        "fetch_html",
        lambda url, logger=None: setup_opensees_docs.Page(url=url, html=page_html),
    )
    monkeypatch.setattr(
        setup_opensees_docs,
        "download_file",
        lambda url, logger=None: b'model basic -ndm 2 -ndf 3\nputs "ok"\n',
    )

    setup_opensees_docs.download_example(example_url, spec)

    output = spec.output_dir / "elastic_frame_example" / "ElasticFrame.tcl"
    assert output.read_text(encoding="utf-8") == 'model basic -ndm 2 -ndf 3\nputs "ok"\n'


def test_download_basic_example_uses_mediawiki_file_title_as_output_name(
    monkeypatch, tmp_path: Path
):
    spec = setup_opensees_docs.ManualSpec(
        name="basic",
        manual_url="https://example.test/manual",
        output_dir=tmp_path / "OpenSeesExamplesBasic",
        link_collector=setup_opensees_docs.collect_basic_example_links,
    )
    spec.output_dir.mkdir(parents=True, exist_ok=True)

    example_url = "https://opensees.berkeley.edu/wiki/Elastic_Frame_Example"
    page_html = """
        <div id="mw-content-text">
          <a href="/wiki/index.php?title=File:ElasticFrame.tcl">ElasticFrame.tcl</a>
        </div>
    """
    monkeypatch.setattr(
        setup_opensees_docs,
        "fetch_html",
        lambda url, logger=None: setup_opensees_docs.Page(url=url, html=page_html),
    )
    monkeypatch.setattr(
        setup_opensees_docs,
        "download_file",
        lambda url, logger=None: b'puts "file page"\n',
    )

    setup_opensees_docs.download_example(example_url, spec)

    output = spec.output_dir / "elastic_frame_example" / "ElasticFrame.tcl"
    assert output.read_text(encoding="utf-8") == 'puts "file page"\n'


def test_download_example_logs_progress(monkeypatch, tmp_path: Path, capsys):
    spec = setup_opensees_docs.ManualSpec(
        name="advanced",
        manual_url="https://example.test/manual",
        output_dir=tmp_path / "OpenSeesExamplesAdvanced",
        link_collector=setup_opensees_docs.collect_advanced_example_links,
        extract_zips_to_root=True,
    )
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_opensees_docs.Logger(verbose=False)

    example_url = "https://opensees.berkeley.edu/wiki/index.php?title=OpenSees_Example_5._2D_Frame"
    page_html = """
        <div id="mw-content-text">
          <a href="/wiki/images/archive/Ex5Fram2DEQ.zip">Download all files in zip format</a>
        </div>
    """
    monkeypatch.setattr(
        setup_opensees_docs,
        "fetch_html",
        lambda url, logger=None: setup_opensees_docs.Page(url=url, html=page_html),
    )
    monkeypatch.setattr(
        setup_opensees_docs,
        "download_file",
        lambda url, logger=None: _zip_bytes(
            {"Ex5Fram2DEQUniform/Example.tcl": b'puts "ok"\n'}
        ),
    )

    summary = setup_opensees_docs.download_example(
        example_url,
        spec,
        logger=logger,
        example_index=2,
        example_total=9,
    )

    captured = capsys.readouterr()
    assert "Fetching example page" in captured.err
    assert "Resolved OpenSees_Example_5._2D_Frame" in captured.err
    assert "Downloading archive 1/1" in captured.err
    assert "Completed OpenSees_Example_5._2D_Frame: 1 file(s) materialized" in captured.err
    assert summary.extracted_count == 1


def test_main_skips_http_errors_for_individual_examples(monkeypatch, capsys):
    spec = setup_opensees_docs.ManualSpec(
        name="advanced",
        manual_url="https://example.test/manual",
        output_dir=Path("docs/agent-reference/OpenSeesExamplesAdvanced"),
        link_collector=setup_opensees_docs.collect_advanced_example_links,
        extract_zips_to_root=True,
    )
    monkeypatch.setattr(setup_opensees_docs, "ensure_dirs", lambda: None)
    monkeypatch.setattr(
        setup_opensees_docs,
        "MANUAL_SPECS",
        {"advanced": spec},
    )
    monkeypatch.setattr(
        setup_opensees_docs,
        "collect_manual_example_links",
        lambda spec, logger=None: ["https://example.test/missing", "https://example.test/ok"],
    )

    calls: list[str] = []

    def fake_download_example(url, spec, logger=None, example_index=None, example_total=None):
        calls.append(url)
        if url.endswith("/missing"):
            raise HTTPError(url, 404, "Not Found", hdrs=None, fp=None)
        return setup_opensees_docs.DownloadSummary(
            title="ok",
            zip_count=0,
            file_count=1,
            extracted_count=1,
        )

    monkeypatch.setattr(
        setup_opensees_docs,
        "download_example",
        fake_download_example,
    )

    result = setup_opensees_docs.main(["--manual", "advanced"])

    captured = capsys.readouterr()
    assert result == 0
    assert calls == ["https://example.test/missing", "https://example.test/ok"]
    assert "Skipping advanced example due to HTTP 404: https://example.test/missing" in captured.err
    assert "Fetched advanced ok from https://example.test/ok" in captured.err


def test_safe_extract_zip_verbose_handles_relative_target_dir(
    tmp_path: Path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)
    target_dir = Path("docs/agent-reference/OpenSeesExamplesAdvanced")
    target_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_opensees_docs.Logger(verbose=True)

    extracted = setup_opensees_docs.safe_extract_zip(
        _zip_bytes({"BM68elc.acc": b"0.0 1.0\n"}),
        target_dir,
        logger=logger,
    )

    captured = capsys.readouterr()
    assert extracted == 1
    assert "Extracted BM68elc.acc" in captured.err
    assert (target_dir / "BM68elc.acc").read_text(encoding="utf-8") == "0.0 1.0\n"
