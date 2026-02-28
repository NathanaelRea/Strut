import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CHECK_WARNINGS_PATH = REPO_ROOT / "scripts" / "check_mojo_warnings.py"


def _load_check_warnings_module():
    module_name = "strut_check_mojo_warnings_test_module"
    spec = importlib.util.spec_from_file_location(module_name, CHECK_WARNINGS_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


check_mojo_warnings = _load_check_warnings_module()


def test_warning_diagnostics_ignores_notes_and_remarks():
    text = json.dumps(
        {
            "diagnostics": [
                {"severity": "warning", "message": "warn-a"},
                {"severity": "note", "message": "note-a"},
                {"severity": "remark", "message": "remark-a"},
            ]
        }
    )

    warnings = check_mojo_warnings.warning_diagnostics(text)

    assert [diag["message"] for diag in warnings] == ["warn-a"]


def test_warning_diagnostics_supports_json_lines():
    text = "\n".join(
        [
            json.dumps({"kind": "warning", "message": "warn-a"}),
            json.dumps({"kind": "warning", "message": "warn-b"}),
        ]
    )

    warnings = check_mojo_warnings.warning_diagnostics(text)

    assert [diag["message"] for diag in warnings] == ["warn-a", "warn-b"]


def test_main_returns_nonzero_when_warnings_exist(tmp_path: Path, monkeypatch):
    diagnostics = tmp_path / "diagnostics.json"
    diagnostics.write_text(
        json.dumps({"diagnostics": [{"severity": "warning", "message": "warn-a"}]})
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["check_mojo_warnings.py", "--quiet", str(diagnostics)],
    )

    assert check_mojo_warnings.main() == 1


def test_main_returns_zero_when_warning_free(tmp_path: Path, monkeypatch):
    diagnostics = tmp_path / "diagnostics.json"
    diagnostics.write_text(
        json.dumps({"diagnostics": [{"severity": "note", "message": "note-a"}]})
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["check_mojo_warnings.py", "--quiet", str(diagnostics)],
    )

    assert check_mojo_warnings.main() == 0
