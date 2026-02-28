#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, List


def _iter_json_documents(text: str) -> Iterable[Any]:
    stripped = text.strip()
    if not stripped:
        return []

    docs: List[Any] = []
    try:
        docs.append(json.loads(stripped))
        return docs
    except json.JSONDecodeError:
        pass

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        docs.append(json.loads(line))
    return docs


def _collect_diagnostics(node: Any, out: List[dict]) -> None:
    if isinstance(node, dict):
        severity = node.get("severity")
        kind = node.get("kind")
        if isinstance(severity, str) or isinstance(kind, str):
            out.append(node)
        for value in node.values():
            _collect_diagnostics(value, out)
        return
    if isinstance(node, list):
        for item in node:
            _collect_diagnostics(item, out)


def load_diagnostics(text: str) -> List[dict]:
    diagnostics: List[dict] = []
    for doc in _iter_json_documents(text):
        _collect_diagnostics(doc, diagnostics)
    return diagnostics


def warning_diagnostics(text: str) -> List[dict]:
    return [
        diag
        for diag in load_diagnostics(text)
        if (
            str(diag.get("severity", "")).lower() == "warning"
            or str(diag.get("kind", "")).lower() == "warning"
        )
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail when Mojo JSON diagnostics contain compiler warnings."
    )
    parser.add_argument(
        "diagnostics",
        nargs="?",
        type=Path,
        default=None,
        help="Path to a Mojo JSON diagnostic log. Reads stdin when omitted.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress summary output; use exit status only.",
    )
    args = parser.parse_args()

    if args.diagnostics is None:
        text = sys.stdin.read()
    else:
        text = args.diagnostics.read_text(encoding="utf-8")

    try:
        warnings = warning_diagnostics(text)
    except json.JSONDecodeError as exc:
        print(f"failed to parse Mojo diagnostics as JSON: {exc}", file=sys.stderr)
        return 2

    if not args.quiet:
        count = len(warnings)
        noun = "warning" if count == 1 else "warnings"
        print(f"Mojo compiler {noun}: {count}")
    return 1 if warnings else 0


if __name__ == "__main__":
    raise SystemExit(main())
