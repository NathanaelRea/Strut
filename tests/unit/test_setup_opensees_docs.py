import importlib.util
import io
import sys
import zipfile
from pathlib import Path


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
    monkeypatch.setattr(
        setup_opensees_docs,
        "download_file",
        lambda url, logger=None: _zip_bytes(
            {
                "Ex5Fram2DEQUniform/Ex5.Frame2D.analyze.Dynamic.EQ.Uniform.tcl": b"puts test\n",
                "GMfiles/H-e12140.g3": b"0.0 1.0\n",
            }
        ),
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
    assert not (
        spec.output_dir / "opensees_example_5_2d_frame"
    ).exists()


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
