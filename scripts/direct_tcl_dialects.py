"""Upgrade legacy direct-case Tcl to the pinned OpenSees parser's dialect."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable


PINNED_TCL_VERSION = "3.8.0"

# Keep Tcl words intact, including variable and command substitutions used as
# arguments. A continuation is whitespace between words, not a command end.
_WORD = (
    rb'(?:"(?:\\[^\r\n]|[^"\\\r\n])*"|\[[^\[\]\r\n]+\]|\{[^{}\r\n]*\}|[^\s;"\[\]{}\\]+)'
)
_SPACE = rb"(?:[ \t]|\\\r?\n)+"
_END = rb"(?=[ \t]*(?:[;\r\n]|\Z))"

_LEGACY_FIBER = re.compile(
    rb"^([ \t]*section[ \t]+(?:Fiber|fiberSec)"
    + _SPACE
    + _WORD
    + rb")"
    + _SPACE
    + rb"(?=\{)",
    re.MULTILINE,
)


def _upgrade_legacy_fiber(text: bytes) -> bytes:
    # V3.0.2 buildSection leaves torsion unset for this form. The null-torsion
    # branch of FiberSection3d supplies an ElasticMaterial with GJ = 1.0e-10.
    # Query the active builder when Tcl executes the section, including inside
    # procedures sourced before the calling script creates a model.
    return _LEGACY_FIBER.sub(
        rb"\1 {*}[expr {[getNDM] == 3 ? {-GJ 1.0e-10} : {}}] ", text
    )


_RT_MODEL = re.compile(
    rb"^([ \t]*model)[ \t]+(?:[Bb]asic(?:Builder)?)[ \t]+"
    rb"(" + _WORD + rb")(?:" + _SPACE + rb"(" + _WORD + rb"))?" + _END,
    re.MULTILINE,
)
_RT_FIBER_ALIAS = re.compile(rb"^([ \t]*section[ \t]+)FiberSec(?=[ \t])", re.MULTILINE)
_RT_DRUCKER_PRAGER = re.compile(
    rb"^([ \t]*nDMaterial)[ \t]+(?:DruckerPrager|DP)[ \t]+(" + _WORD + rb")"
    rb"((?:" + _SPACE + rb"-[A-Za-z][A-Za-z0-9]*" + _SPACE + _WORD + rb")+)" + _END,
    re.MULTILINE,
)
_RT_OPTION_PAIR = re.compile(
    rb"(-[A-Za-z][A-Za-z0-9]*)" + _SPACE + rb"(" + _WORD + rb")"
)
_DRUCKER_PRAGER_REQUIRED = (
    b"-K",
    b"-G",
    b"-Fy",
    b"-Rvol",
    b"-Rbar",
    b"-Fs",
    b"-Fo",
    b"-Hsat",
    b"-delta2",
    b"-H",
    b"-theta",
)
_DRUCKER_PRAGER_OPTIONAL = {b"-density": b"0.0", b"-atm": b"101.0"}


def _upgrade_rt_model(match: re.Match[bytes]) -> bytes:
    command, ndm, ndf = match.groups()
    if ndm.startswith(b"-") or (ndf is not None and ndf.startswith(b"-")):
        return match[0]
    if ndf is not None:
        return b" ".join((command, b"BasicBuilder", b"-ndm", ndm, b"-ndf", ndf))
    return b" ".join((command, b"BasicBuilder", b"-ndm", ndm))


def _upgrade_rt_drucker_prager(match: re.Match[bytes]) -> bytes:
    command, tag, raw_options = match.groups()
    pairs = _RT_OPTION_PAIR.findall(raw_options)
    options = dict(pairs)
    missing = set(_DRUCKER_PRAGER_REQUIRED) - options.keys()
    extra = (
        options.keys() - set(_DRUCKER_PRAGER_REQUIRED) - _DRUCKER_PRAGER_OPTIONAL.keys()
    )
    if missing or extra or len(pairs) != len(options):
        return match[0]
    return b" ".join(
        (
            command,
            b"DruckerPrager",
            tag,
            *(options[key] for key in _DRUCKER_PRAGER_REQUIRED),
            *(
                options.get(key, default)
                for key, default in _DRUCKER_PRAGER_OPTIONAL.items()
            ),
        )
    )


_RT_SSP_QUAD = re.compile(
    rb"^(?P<element>[ \t]*element[ \t]+SSPquad(?:"
    + _SPACE
    + _WORD
    + rb"){5})"
    + _SPACE
    + rb"(?P<thickness>"
    + _WORD
    + rb")"
    + _SPACE
    + rb"(?P<formulation>\"Plane(?:Stress|Strain)\"|Plane(?:Stress|Strain))"
    + _SPACE
    + rb"(?P<material>"
    + _WORD
    + rb")"
    + rb"(?P<optional>(?:"
    + _SPACE
    + _WORD
    + rb"){0,4})"
    + _END,
    re.MULTILINE,
)
_ARGUMENT = re.compile(_SPACE + rb"(" + _WORD + rb")")


def _upgrade_rt_ssp_quad(match: re.Match[bytes]) -> bytes:
    # RT's common quad parser consumes pressure and density before body forces.
    # SSPquad ignores those two slots and derives its mass from the material.
    # Keep argument substitutions for ignored slots so Tcl evaluates them too.
    optional = _ARGUMENT.findall(match["optional"])
    if len(optional) > 2:
        body = (optional[2], optional[3] if len(optional) == 4 else b"0.0")
    else:
        body = ()
    if any(re.search(rb"[$\[\\]", arg) for arg in optional[:2]):
        omitted = b" {*}[lrange [list " + b" ".join(optional[:2]) + b"] 2 end]"
    else:
        omitted = b""
    return (
        b" ".join(
            (
                match["element"],
                match["material"],
                match["formulation"],
                match["thickness"],
            )
        )
        + omitted
        + (b" " + b" ".join(body) if body else b"")
    )


def _upgrade_gallery_rt(text: bytes) -> bytes:
    text = _RT_MODEL.sub(_upgrade_rt_model, text)
    text = _RT_FIBER_ALIAS.sub(rb"\1Fiber", text)
    text = _RT_DRUCKER_PRAGER.sub(_upgrade_rt_drucker_prager, text)
    return _RT_SSP_QUAD.sub(_upgrade_rt_ssp_quad, text)


# OpenSees 3.8.0 dropped the inline `"Series -dt ..."` string form for
# UniformExcitation/MultipleSupport. Convert each assignment to a tagged
# `timeSeries Path`; high tags avoid collisions with the fixture's own tags.
_LEGACY_PATH_ASSIGNMENT = re.compile(
    rb"^(?P<indent>[ \t]*)set[ \t]+(?P<variable>\w+)[ \t]+"
    rb'"(?:Series|Path)[ \t]+-d[tT](?P<options>[ \t]+[^"\r\n]+)"'
    rb"(?=[ \t]*(?:;|\r?$))",
    re.MULTILINE,
)


def _upgrade_legacy_path_series_match(match: re.Match[bytes]) -> bytes:
    indent, variable, options = match.group("indent", "variable", "options")
    return (
        b"%sset %s [expr {1000000000 + [incr ::__strut_legacy_path_tag]}]\n%stimeSeries Path $%s -dt%s"
        % (indent, variable, indent, variable, options)
    )


def _upgrade_legacy_path_series(text: bytes) -> bytes:
    return _LEGACY_PATH_ASSIGNMENT.sub(_upgrade_legacy_path_series_match, text)


# Each edge applies before any edge out of its target.
TCL_UPGRADES: dict[str, tuple[str, Callable[[bytes], bytes]]] = {
    "3.0.2": ("3.7.1", _upgrade_legacy_fiber),
    "openseesrt-fd3958f": ("3.7.1", _upgrade_gallery_rt),
    "3.7.1": ("3.8.0", _upgrade_legacy_path_series),
}


def upgrade_mirrored_tcl(source_version: str, mirrored_dir: Path) -> list[Path]:
    """Upgrade mirrored files and return inputs for the parity cache hash."""
    passes: dict[str, Callable[[bytes], bytes]] = {}
    version = source_version
    while version != PINNED_TCL_VERSION:
        if version not in TCL_UPGRADES or version in passes:
            raise ValueError(
                f"no Tcl upgrade path from {source_version!r} to {PINNED_TCL_VERSION}"
            )
        target, upgrade = TCL_UPGRADES[version]
        passes[version] = upgrade
        version = target

    if not passes:
        return []

    hash_inputs = []
    for script in sorted(mirrored_dir.rglob("*.tcl")):
        original = script.read_bytes()
        upgraded = original
        for upgrade in passes.values():
            upgraded = upgrade(upgraded)
        if upgraded != original:
            script.write_bytes(upgraded)
        # Include helpers that were sourced from Tcl rather than listed in the
        # manifest. Hash the staged bytes so changes to the passes also count.
        hash_inputs.append(script)
    return hash_inputs
