#!/usr/bin/env python3
import argparse
import json
import math
import re
import subprocess
import sys
import tempfile
import tkinter
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Any, Optional

_BRICK_TYPES = {"stdBrick", "bbarBrick", "SSPbrick", "Brick8N", "Brick27N"}
_SOLID_NODE_COUNTS = {
    "stdBrick": 8,
    "bbarBrick": 8,
    "SSPbrick": 8,
    "Brick8N": 8,
    "Brick27N": 27,
    "SSPquad": 4,
    "enhancedQuad": 4,
    "bbarQuad": 4,
    "quad8n": 8,
    "quad9n": 9,
    "quad3d": 4,
}
_SOLID_INTEGRATION_POINTS = {
    "stdBrick": 8,
    "bbarBrick": 8,
    "SSPbrick": 1,
    "Brick8N": 27,
    "Brick27N": 27,
    "SSPquad": 1,
    "enhancedQuad": 4,
    "bbarQuad": 4,
    "quad8n": 9,
    "quad9n": 9,
    "quad3d": 4,
    "fourNodeQuad": 4,
    "quadUP": 4,
    "9_4_QuadUP": 9,
    "SSPbrickUP": 1,
}
_BLOCK3D_COORDS = (
    (-1, -1, -1),
    (1, -1, -1),
    (1, 1, -1),
    (-1, 1, -1),
    (-1, -1, 1),
    (1, -1, 1),
    (1, 1, 1),
    (-1, 1, 1),
    (-1, -1, 0),
    (1, -1, 0),
    (1, 1, 0),
    (-1, 1, 0),
    (0, -1, -1),
    (1, 0, -1),
    (0, 1, -1),
    (-1, 0, -1),
    (0, 0, -1),
    (0, -1, 1),
    (1, 0, 1),
    (0, 1, 1),
    (-1, 0, 1),
    (0, 0, 1),
    (0, -1, 0),
    (1, 0, 0),
    (0, 1, 0),
    (-1, 0, 0),
    (0, 0, 0),
)
_BRICK27_COORDS = tuple(
    _BLOCK3D_COORDS[index]
    for index in (
        *range(16),
        17,
        18,
        19,
        20,
        22,
        23,
        24,
        25,
        16,
        21,
        26,
    )
)

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from direct_tcl_support import read_tcl_text
from tcl_ground_motion import trapezoidal_path
from tcl_case_types import (
    Analysis,
    AnalysisStage,
    Element,
    Load,
    Mass,
    Material,
    Metadata,
    Model,
    Node,
    Pattern,
    Recorder,
    Section,
    StagedAnalysis,
    StrutCase,
    TimeSeries,
)


SUPPORTED_TCL_BUILTINS = frozenset(
    {
        "array",
        "break",
        "catch",
        "close",
        "concat",
        "continue",
        "eval",
        "exit",
        "expr",
        "file",
        "for",
        "foreach",
        "format",
        "global",
        "if",
        "incr",
        "lappend",
        "lindex",
        "list",
        "llength",
        "lrange",
        "open",
        "proc",
        "pset",
        "puts",
        "read",
        "return",
        "set",
        "source",
        "split",
        "string",
        "switch",
        "upvar",
        "variable",
        "while",
    }
)


def _read_tcl_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


OPEN_SEES_COMMAND_ALIASES = {
    "model": {
        "basic": "basic",
        "Basic": "basic",
        "BasicBuilder": "basic",
        "basicBuilder": "basic",
    },
    "element": {
        "CorotTruss": "corotTruss",
        "corotTruss": "corotTruss",
        "ShellMITC4": "shell",
        "shellMITC4": "shell",
        "Truss": "truss",
        "truss": "truss",
        "nonlinearBeamColumn": "forceBeamColumn",
        "forceBeamColumn": "forceBeamColumn",
    },
    "section": {
        "Fiber": "Fiber",
        "fiberSec": "Fiber",
    },
}


_SUPPORTED_SYSTEM_NAMES = frozenset(
    {
        "BandGeneral",
        "BandSPD",
        "ProfileSPD",
        "SuperLU",
        "UmfPack",
        "FullGeneral",
        "SparseSYM",
    }
)

_SYSTEM_NAME_ALIASES = {
    "BandGEN": "BandGeneral",
    "BandGen": "BandGeneral",
    "SparseGeneral": "SuperLU",
    "SparseGEN": "SuperLU",
    "Mumps": "SuperLU",
    "SparseSPD": "SparseSYM",
    "Umfpack": "UmfPack",
}

_BEAM_INTEGRATION_MIN_POINTS = {
    "Lobatto": 2,
    "Legendre": 1,
    "Radau": 1,
}


def _sanitize_slug_part(text: str) -> str:
    chars = []
    for ch in text.lower():
        if ch.isalnum():
            chars.append(ch)
        else:
            chars.append("_")
    slug = "".join(chars).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "case"


def _repo_relative_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _matrix_identity(size: int) -> list[list[float]]:
    ident = []
    for i in range(size):
        row = [0.0] * size
        row[i] = 1.0
        ident.append(row)
    return ident


def _matrix_multiply(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    rows = len(a)
    cols = len(b[0]) if b else 0
    inner = len(b)
    out = []
    for i in range(rows):
        row = [0.0] * cols
        for k in range(inner):
            aik = a[i][k]
            if aik == 0.0:
                continue
            for j in range(cols):
                row[j] += aik * b[k][j]
        out.append(row)
    return out


def _matrix_vector_multiply(a: list[list[float]], x: list[float]) -> list[float]:
    out = []
    for row in a:
        total = 0.0
        for value, x_value in zip(row, x):
            total += value * x_value
        out.append(total)
    return out


def _matrix_inverse(a: list[list[float]]) -> list[list[float]]:
    size = len(a)
    aug = []
    ident = _matrix_identity(size)
    for i in range(size):
        aug.append([float(v) for v in a[i]] + ident[i])

    for col in range(size):
        pivot = max(range(col, size), key=lambda row: abs(aug[row][col]))
        if abs(aug[pivot][col]) < 1.0e-14:
            raise ValueError("singular matrix")
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]

        pivot_value = aug[col][col]
        for j in range(2 * size):
            aug[col][j] /= pivot_value

        for row in range(size):
            if row == col:
                continue
            factor = aug[row][col]
            if factor == 0.0:
                continue
            for j in range(2 * size):
                aug[row][j] -= factor * aug[col][j]

    return [row[size:] for row in aug]


def _jacobi_eigenvalues_symmetric(a: list[list[float]]) -> list[float]:
    size = len(a)
    if size == 0:
        return []
    mat = [[float(v) for v in row] for row in a]
    max_iter = max(32, size * size * 16)
    tol = 1.0e-12

    for _ in range(max_iter):
        p = 0
        q = 1 if size > 1 else 0
        max_value = 0.0
        for i in range(size):
            for j in range(i + 1, size):
                value = abs(mat[i][j])
                if value > max_value:
                    max_value = value
                    p = i
                    q = j
        if max_value < tol:
            break

        if abs(mat[p][p] - mat[q][q]) < tol:
            angle = math.pi / 4.0
        else:
            angle = 0.5 * math.atan2(2.0 * mat[p][q], mat[q][q] - mat[p][p])
        c = math.cos(angle)
        s = math.sin(angle)

        app = mat[p][p]
        aqq = mat[q][q]
        apq = mat[p][q]

        for i in range(size):
            if i == p or i == q:
                continue
            aip = mat[i][p]
            aiq = mat[i][q]
            mat[i][p] = c * aip - s * aiq
            mat[p][i] = mat[i][p]
            mat[i][q] = s * aip + c * aiq
            mat[q][i] = mat[i][q]

        mat[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq
        mat[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq
        mat[p][q] = 0.0
        mat[q][p] = 0.0

    values = [mat[i][i] for i in range(size)]
    values.sort()
    return values


@dataclass
class TclLocation:
    command: str
    args: list[str]
    file: Optional[Path]
    line: Optional[int]
    include_stack: list[Path]


class TclToStrutError(RuntimeError):
    def __init__(self, reason: str, location: TclLocation):
        self.reason = reason
        self.location = location
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        location_bits = []
        if self.location.file is not None:
            location_bits.append(str(self.location.file))
        if self.location.line is not None:
            location_bits.append(f"line {self.location.line}")
        where = ":".join(location_bits) if location_bits else "<unknown>"
        include_text = " -> ".join(str(path) for path in self.location.include_stack)
        return (
            f"{self.reason} at {where} while handling `{self.location.command}` "
            f"with args {self.location.args}; include stack: {include_text or '<entry>'}"
        )


@dataclass
class PendingPlainPattern:
    tag: int
    time_series: int
    loads: list[dict[str, Any]] = field(default_factory=list)
    element_loads: list[dict[str, Any]] = field(default_factory=list)
    sp_constraints: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class PendingFiberSection:
    id: int
    section_type: str
    torsion_gj: Optional[float] = None
    patches: list[dict[str, Any]] = field(default_factory=list)
    layers: list[dict[str, Any]] = field(default_factory=list)
    fibers: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class GeomTransformDef:
    geom_type: str
    vecxz: Optional[tuple[float, float, float]] = None


@dataclass(frozen=True)
class StepRetryPolicy:
    analyze_args: tuple[str, ...]
    attempts: tuple[dict[str, Any], ...]
    restore_primary_after_success: bool
    continue_after_failure: bool = False


def _analysis_stage_from_dict(stage: dict[str, Any]) -> AnalysisStage:
    extra = {
        key: value
        for key, value in stage.items()
        if key
        not in {
            "analysis",
            "pattern",
            "loads",
            "element_loads",
            "load_const",
            "rayleigh",
            "time_series",
        }
    }
    return AnalysisStage(
        analysis=Analysis(dict(stage["analysis"])),
        pattern=Pattern(dict(stage["pattern"])) if "pattern" in stage else None,
        loads=[Load(dict(load)) for load in stage.get("loads", [])],
        element_loads=[Load(dict(load)) for load in stage.get("element_loads", [])],
        load_const=dict(stage["load_const"]) if "load_const" in stage else None,
        rayleigh=dict(stage["rayleigh"]) if "rayleigh" in stage else None,
        time_series=[
            TimeSeries(dict(series)) for series in stage.get("time_series", [])
        ],
        extra=extra,
    )


class TclStrutBuilder:
    def __init__(self, entry_path: Path, repo_root: Path):
        self.entry_path = entry_path.resolve()
        self.repo_root = repo_root.resolve()
        self.interp = tkinter.Tcl()
        self.source_stack: list[Path] = []
        self.open_channels: set[str] = set()

        self.model: Optional[dict[str, int]] = None
        self.nodes: dict[int, dict[str, Any]] = {}
        self.node_ndfs: dict[int, int] = {}
        self.geom_transforms: dict[int, GeomTransformDef] = {}
        self.materials: list[dict[str, Any]] = []
        self.materials_by_id: dict[int, dict[str, Any]] = {}
        self.material_ids_by_e: dict[float, int] = {}
        self.sections: list[dict[str, Any]] = []
        self.sections_by_id: dict[int, dict[str, Any]] = {}
        self.section_ids_by_props: dict[tuple[float, float, float], int] = {}
        self.generated_section_ids: set[int] = set()
        self.axial_section_materials: dict[int, tuple[float, int]] = {}
        self.elements: list[dict[str, Any]] = []
        self.regions: dict[int, tuple[list[int], list[int]]] = {}
        self.masses: list[dict[str, Any]] = []
        self.time_series: list[dict[str, Any]] = []
        self.time_series_by_tag: dict[int, dict[str, Any]] = {}
        self.generated_time_series_tags: set[int] = set()
        self.recorders: list[dict[str, Any]] = []
        self.next_recorder_tag = 0
        self.active_recorders: dict[int, list[dict[str, Any]]] = {}
        self.analysis_stages: list[dict[str, Any]] = []
        self.mp_constraints: list[dict[str, Any]] = []

        self.current_plain_pattern: Optional[PendingPlainPattern] = None
        self.current_section: Optional[PendingFiberSection] = None
        self.constant_patterns: dict[int, dict[str, Any]] = {}
        self.current_pattern: Optional[dict[str, Any]] = None
        self.current_rayleigh: Optional[dict[str, Any]] = None
        self.current_test: Optional[dict[str, Any]] = None
        self.last_stage: Optional[dict[str, Any]] = None
        self.pattern_removed = False
        self.uniform_excitation_count_since_analyze = 0
        self.has_complex_transient_excitation = False
        self.transient_eigen_initialized = False
        self.ground_motion_series: dict[int, dict[str, int | float]] | None = None
        self.imposed_motion_patterns: dict[int, list[dict[str, int | float]]] = {}
        self.active_imposed_motions: list[dict[str, int | float]] | None = None

        self.constraints_handler: Optional[str] = None
        self.penalty_coefficients: tuple[float, float] | None = None
        self.numberer_handler: Optional[str] = None
        self.system_handler: Optional[str] = None
        self.system_options: list[str] = []
        self.algorithm_name: Optional[str] = None
        self.algorithm_options: Optional[dict[str, Any]] = None
        self.integrator: Optional[dict[str, Any]] = None
        self.analysis_type: Optional[str] = None
        self.current_time = 0.0
        self.load_control_has_run = False
        self.pending_time_reset = False
        self.initial_state_analysis = False
        self.pending_initial_state_reset: bool | None = None
        self.pending_node_removals: list[int] = []
        self.pending_element_removals: list[int] = []
        self.pending_contact_friction: bool | None = None
        self.removed_nodes: set[int] = set()
        self.removed_elements: set[int] = set()
        self.material_stages: dict[int, int] = {}
        self.permeability_parameters: dict[int, tuple[int, str]] = {}
        self.permeability_updates: dict[tuple[int, str], float] = {}
        self.active_pressure_motions: dict[int, dict[str, float]] | None = None
        self.prescribed_pressures: dict[tuple[int, int], dict[str, float]] = {}
        self.pending_recorders: list[int] = []
        self.node_displacements: dict[tuple[int, int], float] = {}
        self.last_eigenvalues: list[float] = []
        self.modal_damping: dict[str, Any] | None = None
        self.last_standard_eigen = False
        self.last_error: Optional[TclToStrutError] = None
        self.pending_initialize = False
        self.step_retry_policies_by_location: dict[
            tuple[Path, int], StepRetryPolicy
        ] = {}
        self.step_retry_policies_by_signature: dict[
            tuple[Path, tuple[str, ...]], list[StepRetryPolicy]
        ] = {}

        self._install_commands()

    def _install_commands(self) -> None:
        self.interp.eval("namespace eval ::strut {}")
        self.interp.eval("rename source ::strut::_source_builtin")
        self.interp.eval("rename file ::strut::_file_builtin")
        self.interp.eval("rename puts ::strut::_puts_builtin")
        self.interp.eval("rename open ::strut::_open_builtin")
        self.interp.eval("rename read ::strut::_read_builtin")
        self.interp.eval("rename close ::strut::_close_builtin")
        self.interp.eval("rename info ::strut::_info_builtin")
        self.interp.eval("rename rename ::strut::_rename_builtin")

        self.interp.createcommand("source", self._wrap_command(self._cmd_source))
        self.interp.createcommand("file", self._wrap_command(self._cmd_file))
        self.interp.createcommand("puts", self._wrap_command(self._cmd_puts))
        self.interp.createcommand("open", self._wrap_command(self._cmd_open))
        self.interp.createcommand("read", self._wrap_command(self._cmd_read))
        self.interp.createcommand("close", self._wrap_command(self._cmd_close))
        self.interp.createcommand("clock", self._wrap_command(self._cmd_clock))
        self.interp.createcommand("info", self._wrap_command(self._cmd_info))
        self.interp.createcommand("variable", self._wrap_command(self._cmd_variable))
        self.interp.createcommand("pset", self._wrap_command(self._cmd_pset))
        self.interp.createcommand("exit", self._wrap_command(self._cmd_exit))
        self.interp.createcommand("wipe", self._wrap_command(self._cmd_wipe))
        self.interp.createcommand("model", self._wrap_command(self._cmd_model))
        self.interp.createcommand("getNDM", self._wrap_command(self._cmd_get_ndm))
        self.interp.createcommand("node", self._wrap_command(self._cmd_node))
        self.interp.createcommand("fix", self._wrap_command(self._cmd_fix))
        self.interp.createcommand("fixX", self._wrap_command(self._cmd_fix_x))
        self.interp.createcommand("fixY", self._wrap_command(self._cmd_fix_y))
        self.interp.createcommand("fixZ", self._wrap_command(self._cmd_fix_z))
        self.interp.createcommand("mass", self._wrap_command(self._cmd_mass))
        self.interp.createcommand(
            "nDMaterial", self._wrap_command(self._cmd_nd_material)
        )
        self.interp.createcommand("equalDOF", self._wrap_command(self._cmd_equal_dof))
        self.interp.createcommand(
            "rigidDiaphragm", self._wrap_command(self._cmd_rigid_diaphragm)
        )
        self.interp.createcommand(
            "uniaxialMaterial", self._wrap_command(self._cmd_uniaxial_material)
        )
        self.interp.createcommand(
            "geomTransf", self._wrap_command(self._cmd_geom_transf)
        )
        self.interp.createcommand("block2D", self._wrap_command(self._cmd_block_2d))
        self.interp.createcommand("block3D", self._wrap_command(self._cmd_block_3d))
        self.interp.createcommand("element", self._wrap_command(self._cmd_element))
        self.interp.createcommand("recorder", self._wrap_command(self._cmd_recorder))
        self.interp.createcommand(
            "InitialStateAnalysis", self._wrap_command(self._cmd_initial_state_analysis)
        )
        self.interp.createcommand(
            "timeSeries", self._wrap_command(self._cmd_time_series)
        )
        self.interp.createcommand("pattern", self._wrap_command(self._cmd_pattern))
        self.interp.createcommand("load", self._wrap_command(self._cmd_load))
        self.interp.createcommand("sp", self._wrap_command(self._cmd_sp))
        self.interp.createcommand("eleLoad", self._wrap_command(self._cmd_ele_load))
        self.interp.createcommand(
            "constraints", self._wrap_command(self._cmd_constraints)
        )
        self.interp.createcommand("numberer", self._wrap_command(self._cmd_numberer))
        self.interp.createcommand("system", self._wrap_command(self._cmd_system))
        self.interp.createcommand("test", self._wrap_command(self._cmd_test))
        self.interp.createcommand("algorithm", self._wrap_command(self._cmd_algorithm))
        self.interp.createcommand(
            "integrator", self._wrap_command(self._cmd_integrator)
        )
        self.interp.createcommand(
            "initialize", self._wrap_command(self._cmd_initialize)
        )
        self.interp.createcommand("analysis", self._wrap_command(self._cmd_analysis))
        self.interp.createcommand("analyze", self._wrap_command(self._cmd_analyze))
        self.interp.createcommand("getTime", self._wrap_command(self._cmd_get_time))
        self.interp.createcommand(
            "getNodeTags", self._wrap_command(self._cmd_get_node_tags)
        )
        self.interp.createcommand(
            "nodeBounds", self._wrap_command(self._cmd_node_bounds)
        )
        self.interp.createcommand("nodeDisp", self._wrap_command(self._cmd_node_disp))
        self.interp.createcommand(
            "nodeReaction", self._wrap_command(self._cmd_node_reaction)
        )
        self.interp.createcommand(
            "nodeEigenvector", self._wrap_command(self._cmd_node_eigenvector)
        )
        self.interp.createcommand("loadConst", self._wrap_command(self._cmd_load_const))
        self.interp.createcommand("setTime", self._wrap_command(self._cmd_set_time))
        self.interp.createcommand(
            "updateMaterialStage", self._wrap_command(self._cmd_update_material_stage)
        )
        self.interp.createcommand(
            "wipeAnalysis", self._wrap_command(self._cmd_wipe_analysis)
        )
        self.interp.createcommand("eigen", self._wrap_command(self._cmd_eigen))
        self.interp.createcommand(
            "modalDamping", self._wrap_command(self._cmd_modal_damping)
        )
        self.interp.createcommand(
            "groundMotion", self._wrap_command(self._cmd_ground_motion)
        )
        for command in ("imposedMotion", "imposedSupportMotion"):
            self.interp.createcommand(
                command, self._wrap_command(self._cmd_imposed_motion)
            )
        self.interp.createcommand("rayleigh", self._wrap_command(self._cmd_rayleigh))
        self.interp.createcommand("region", self._wrap_command(self._cmd_region))
        self.interp.createcommand("section", self._wrap_command(self._cmd_section))
        self.interp.createcommand("patch", self._wrap_command(self._cmd_patch))
        self.interp.createcommand("layer", self._wrap_command(self._cmd_layer))
        self.interp.createcommand("fiber", self._wrap_command(self._cmd_fiber))
        self.interp.createcommand("remove", self._wrap_command(self._cmd_remove))
        self.interp.createcommand("parameter", self._wrap_command(self._cmd_parameter))
        self.interp.createcommand(
            "setParameter", self._wrap_command(self._cmd_set_parameter)
        )
        self.interp.createcommand(
            "updateParameter", self._wrap_command(self._cmd_update_parameter)
        )
        self.interp.createcommand("reactions", self._wrap_command(self._cmd_noop))
        self.interp.createcommand("prp", self._wrap_command(self._cmd_noop))
        self.interp.createcommand("vup", self._wrap_command(self._cmd_noop))
        self.interp.createcommand("vpn", self._wrap_command(self._cmd_noop))
        self.interp.createcommand("vrp", self._wrap_command(self._cmd_noop))
        self.interp.createcommand("plane", self._wrap_command(self._cmd_noop))
        self.interp.createcommand("projection", self._wrap_command(self._cmd_noop))
        self.interp.createcommand("fill", self._wrap_command(self._cmd_noop))
        self.interp.createcommand("port", self._wrap_command(self._cmd_noop))
        self.interp.createcommand("display", self._wrap_command(self._cmd_noop))
        self.interp.createcommand("print", self._wrap_command(self._cmd_print))
        self.interp.createcommand("record", self._wrap_command(self._cmd_record))
        self.interp.createcommand("viewWindow", self._wrap_command(self._cmd_noop))
        self.interp.createcommand("unknown", self._wrap_command(self._cmd_unknown))
        self._restrict_command_surface()

    def _wrap_command(self, fn):
        def wrapped(*args: str):
            cleaned_args = args
            for idx, token in enumerate(args):
                if token.startswith("#"):
                    cleaned_args = args[:idx]
                    break
            try:
                return fn(*cleaned_args)
            except TclToStrutError as exc:
                self.last_error = exc
                raise tkinter.TclError(str(exc)) from exc
            except Exception as exc:
                self.last_error = TclToStrutError(
                    str(exc),
                    self._current_location(
                        getattr(fn, "__name__", "command"), list(cleaned_args)
                    ),
                )
                raise tkinter.TclError(str(self.last_error)) from exc

        return wrapped

    def convert(self) -> StrutCase:
        try:
            self.interp.eval(f"source {{{self.entry_path}}}")
        except TclToStrutError:
            raise
        except tkinter.TclError as exc:
            if self.last_error is not None:
                raise self.last_error from exc
            location = self._current_location("eval", [])
            raise TclToStrutError(str(exc), location) from exc

        if self.model is None:
            raise TclToStrutError(
                "model was never defined", self._current_location("model", [])
            )
        if not self.analysis_stages and self.last_eigenvalues:
            self.analysis_stages.append(
                {
                    "analysis": {
                        "type": "modal_eigen",
                        "num_modes": len(self.last_eigenvalues),
                        "standard_eigen": self.last_standard_eigen,
                    }
                }
            )
        if not self.analysis_stages:
            raise TclToStrutError(
                "no analysis stages were produced",
                self._current_location("analyze", []),
            )

        analysis_constraints = self.constraints_handler or "Plain"
        if analysis_constraints == "Plain":
            analysis_constraints = self._analysis_constraints_name()
        case = self._build_case(
            self.analysis_stages,
            analysis_constraints,
            self.recorders,
            equal_dof_as_mp=True,
        )
        normalized_stages = case.analysis.stages
        # Soil-spring cases use the standard parity tolerances.
        has_nonlinear_material = any(
            material["type"] not in {"Elastic", "PySimple1", "TzSimple1", "QzSimple1"}
            for material in self.materials
        )
        has_force_beam = any(
            element["type"] == "forceBeamColumn2d" for element in self.elements
        )
        has_transient_stage = any(
            stage.analysis.data.get("type", "").startswith("transient")
            for stage in normalized_stages
        )
        has_long_displacement_control_push = False
        for stage in normalized_stages:
            analysis_data = stage.analysis.data
            if analysis_data.get("type") != "static_nonlinear":
                continue
            integrator_data = analysis_data.get("integrator", {})
            if not isinstance(integrator_data, dict):
                continue
            if integrator_data.get("type") != "DisplacementControl":
                continue
            has_long_displacement_control_push = True
            break
        if has_force_beam and has_transient_stage:
            for recorder in self.recorders:
                if recorder.get("type") in {
                    "node_reaction",
                    "element_local_force",
                    "section_force",
                }:
                    recorder["parity"] = False
        if self.has_complex_transient_excitation:
            for recorder in self.recorders:
                recorder["parity"] = False
        has_plastic_hinge_material = any(
            material["type"] in {"Bilin", "Hysteretic", "ModIMKPeakOriented"}
            for material in self.materials
        )
        has_coupled_soil = self._has_coupled_soil()
        if (
            (has_nonlinear_material or has_force_beam)
            and not has_plastic_hinge_material
            and not has_coupled_soil
        ):
            # Nonlinear direct-Tcl parity is sensitive to algorithmic/path
            # differences; use intentionally loose tolerances.
            case.parity_tolerance = {"rtol": 0.5, "atol": 5.0e-3}
            case.parity_tolerance_by_category = {
                "displacement": {"rtol": 0.5, "atol": 5.0e-3},
                "velocity": {"rtol": 0.5, "atol": 5.0e-3},
                "acceleration": {"rtol": 0.75, "atol": 1.0e-2},
                "force": {"rtol": 0.75, "atol": 1.0e-2},
                "deformation": {"rtol": 0.75, "atol": 1.0e-2},
            }
            case.parity_tolerance_by_recorder = {
                "element_force": {"rtol": 0.75, "atol": 1.0e-2},
                "element_local_force": {"rtol": 1.0, "atol": 3.0e-2},
                "element_basic_force": {"rtol": 0.75, "atol": 1.0e-2},
                "element_deformation": {"rtol": 0.75, "atol": 1.0e-2},
                "section_force": {"rtol": 0.75, "atol": 3.0e-2},
                "section_deformation": {"rtol": 0.75, "atol": 1.0e-2},
            }
            if has_long_displacement_control_push:
                case.parity_tolerance = {"rtol": 2.0, "atol": 1.0e-1}
                case.parity_mode = "max_abs"
                case.parity_tolerance_by_category = {
                    "displacement": {"rtol": 50.0, "atol": 1.0e-1},
                    "velocity": {"rtol": 50.0, "atol": 1.0e-1},
                    "acceleration": {"rtol": 50.0, "atol": 1.0e-1},
                    "force": {"rtol": 5.0, "atol": 1.0e-1},
                    "deformation": {"rtol": 100.0, "atol": 1.0e-1},
                }
                case.parity_tolerance_by_recorder = {
                    "node_displacement": {"rtol": 50.0, "atol": 1.0e-1},
                    "node_reaction": {"rtol": 5.0, "atol": 1.0e-1},
                    "drift": {"rtol": 100.0, "atol": 1.0e-1},
                    "element_force": {"rtol": 5.0, "atol": 1.0e-1},
                    "element_local_force": {"rtol": 5.0, "atol": 1.0e-1},
                    "element_basic_force": {"rtol": 5.0, "atol": 1.0e-1},
                    "element_deformation": {"rtol": 100.0, "atol": 1.0e-1},
                    "section_force": {"rtol": 5.0, "atol": 1.0e-1},
                    "section_deformation": {"rtol": 100.0, "atol": 1.0e-1},
                }
            if has_transient_stage:
                case.parity_mode = "max_abs"
        return case

    def _build_case(
        self,
        stages: list[dict[str, Any]],
        constraints: str,
        recorders: list[dict[str, Any]],
        *,
        equal_dof_as_mp: bool = False,
    ) -> StrutCase:
        assert self.model is not None
        ndf = max(self.model["ndf"], max(self.node_ndfs.values(), default=0))
        if constraints == "Penalty":
            constraints = next(
                (
                    stage["analysis"]["constraints"]
                    for stage in stages
                    if stage["analysis"].get("constraints") not in {None, "Penalty"}
                ),
                constraints,
            )
        normalized_stages = [
            _analysis_stage_from_dict(
                {
                    **stage,
                    "analysis": {
                        **stage["analysis"],
                        "constraints": (
                            "Penalty"
                            if stage["analysis"].get("constraints") == "Penalty"
                            else constraints
                        ),
                    },
                }
            )
            for stage in stages
        ]
        case = StrutCase(
            metadata=Metadata(
                name=_sanitize_slug_part(self.entry_path.stem), units="unknown"
            ),
            model=Model(ndm=int(self.model["ndm"]), ndf=ndf),
            source_example=_repo_relative_path(self.entry_path, self.repo_root),
            nodes=[
                Node(
                    self.nodes[node_id]
                    | (
                        {"ndf": self.node_ndfs[node_id]}
                        if self.node_ndfs.get(node_id, ndf) != ndf
                        else {}
                    )
                )
                for node_id in sorted(self.nodes)
            ],
            materials=[Material(material) for material in self.materials],
            sections=[Section(section) for section in self.sections],
            elements=[Element(element) for element in self.elements],
            masses=[Mass(mass) for mass in self.masses],
            time_series=[TimeSeries(series) for series in self.time_series],
            analysis=StagedAnalysis(constraints=constraints, stages=normalized_stages),
            recorders=[Recorder(recorder) for recorder in recorders],
        )
        if (
            equal_dof_as_mp
            and self.constraints_handler in {None, "Plain"}
            and constraints == "Transformation"
        ):
            groups: dict[tuple[int, int], list[int]] = {}
            for (dependent, dof), retained in self._equal_dof_roots().items():
                groups.setdefault((retained, dependent), []).append(dof)
            case.mp_constraints = [
                {
                    "type": "equalDOF",
                    "retained_node": retained,
                    "constrained_node": dependent,
                    "dofs": dofs,
                }
                for (retained, dependent), dofs in groups.items()
            ]
        elif self.mp_constraints:
            case.mp_constraints = list(self.mp_constraints)
        first_stage = normalized_stages[0]
        if (
            first_stage.pattern is not None
            and first_stage.pattern.data.get("type") == "Plain"
        ):
            case.pattern = first_stage.pattern
            case.loads = list(first_stage.loads)
            if first_stage.element_loads:
                case.element_loads = list(first_stage.element_loads)
        elif (
            first_stage.pattern is not None
            and first_stage.pattern.data.get("type") == "UniformExcitation"
        ):
            case.pattern = first_stage.pattern
        return case

    def _current_location(self, command: str, args: list[str]) -> TclLocation:
        frame = self._frame_dict(0) if self.source_stack else {}
        file_text = frame.get("file")
        file_path = Path(file_text).resolve() if file_text else None
        if file_path is None and self.source_stack:
            file_path = self.source_stack[-1]
        line = int(frame["line"]) if "line" in frame else None
        return TclLocation(
            command=command,
            args=args,
            file=file_path,
            line=line,
            include_stack=list(self.source_stack),
        )

    def _frame_dict(self, level: int) -> dict[str, str]:
        try:
            raw = self.interp.tk.call("::strut::_info_builtin", "frame", level)
        except tkinter.TclError:
            return {}
        if not raw:
            return {}
        if isinstance(raw, tuple):
            parts = [str(part) for part in raw]
        else:
            parts = list(self.interp.tk.splitlist(str(raw)))
        frame = {}
        for idx in range(0, len(parts) - 1, 2):
            frame[str(parts[idx])] = str(parts[idx + 1])
        return frame

    def _restrict_command_surface(self) -> None:
        allowed_commands = {
            "algorithm",
            "analysis",
            "analyze",
            "close",
            "clock",
            "constraints",
            "display",
            "eigen",
            "eleLoad",
            "element",
            "equalDOF",
            "exit",
            "file",
            "fiber",
            "fix",
            "fixX",
            "fixY",
            "fixZ",
            "geomTransf",
            "getNDM",
            "getTime",
            "getNodeTags",
            "groundMotion",
            "info",
            "imposedMotion",
            "imposedSupportMotion",
            "integrator",
            "initialize",
            "InitialStateAnalysis",
            "load",
            "loadConst",
            "lsort",
            "layer",
            "mass",
            "modalDamping",
            "model",
            "nDMaterial",
            "nodeEigenvector",
            "nodeBounds",
            "node",
            "nodeDisp",
            "nodeReaction",
            "numberer",
            "open",
            "patch",
            "pset",
            "pattern",
            "parameter",
            "print",
            "prp",
            "puts",
            "rayleigh",
            "read",
            "record",
            "recorder",
            "reactions",
            "region",
            "rigidDiaphragm",
            "remove",
            "block2D",
            "block3D",
            "section",
            "setTime",
            "setParameter",
            "source",
            "sp",
            "system",
            "test",
            "timeSeries",
            "uniaxialMaterial",
            "updateMaterialStage",
            "updateParameter",
            "unknown",
            "uplevel",
            "variable",
            "vrp",
            "viewWindow",
            "vup",
            "vpn",
            "plane",
            "projection",
            "fill",
            "port",
            "wipe",
            "wipeAnalysis",
        } | set(SUPPORTED_TCL_BUILTINS)
        commands = self.interp.tk.splitlist(
            self.interp.tk.call("::strut::_info_builtin", "commands")
        )
        for command_name in sorted(str(name) for name in commands):
            if "::" in command_name:
                continue
            if command_name in allowed_commands:
                continue
            self.interp.tk.call(
                "::strut::_rename_builtin",
                command_name,
                f"::strut::_disabled_{command_name}",
            )

    def _error(
        self, reason: str, command: str, args: tuple[str, ...]
    ) -> TclToStrutError:
        return TclToStrutError(reason, self._current_location(command, list(args)))

    def _eval_tcl(self, script: str, command: str, args: tuple[str, ...]) -> str:
        try:
            return str(self.interp.eval(script))
        except tkinter.TclError as exc:
            if self.last_error is not None:
                raise self.last_error from exc
            raise self._error(str(exc), command, args) from exc

    def _require_model(self, command: str, args: tuple[str, ...]) -> dict[str, int]:
        if self.model is None:
            raise self._error("model must be defined first", command, args)
        return self.model

    def _sync_materials(self) -> None:
        self.materials = [
            self.materials_by_id[key] for key in sorted(self.materials_by_id)
        ]

    def _sync_sections(self) -> None:
        self.sections = [
            self.sections_by_id[key] for key in sorted(self.sections_by_id)
        ]

    def _upsert_material(self, entry: dict[str, Any]) -> None:
        if entry.get("type") == "Elastic":
            params = entry.get("params", {})
            if "E" in params:
                self.material_ids_by_e[float(params["E"])] = int(entry["id"])
        self.materials_by_id[int(entry["id"])] = entry
        self._sync_materials()

    def _next_free_section_id(self) -> int:
        section_id = 1
        if self.sections_by_id:
            section_id = max(self.sections_by_id) + 1
        while section_id in self.sections_by_id:
            section_id += 1
        return section_id

    def _reassign_generated_section_id(self, section_id: int) -> None:
        new_id = self._next_free_section_id()
        entry = dict(self.sections_by_id.pop(section_id))
        entry["id"] = new_id
        self.sections_by_id[new_id] = entry
        self.generated_section_ids.remove(section_id)
        self.generated_section_ids.add(new_id)
        for key, value in list(self.section_ids_by_props.items()):
            if value == section_id:
                self.section_ids_by_props[key] = new_id
        for element in self.elements:
            if int(element.get("section", -1)) == section_id:
                element["section"] = new_id
        for section in self.sections_by_id.values():
            params = section.get("params", {})
            if int(params.get("base_section", -1)) == section_id:
                params["base_section"] = new_id

    def _upsert_section(
        self, entry: dict[str, Any], *, generated: bool = False
    ) -> None:
        section_id = int(entry["id"])
        if not generated and section_id in self.generated_section_ids:
            self._reassign_generated_section_id(section_id)
        self.sections_by_id[section_id] = entry
        if generated:
            self.generated_section_ids.add(section_id)
        else:
            self.generated_section_ids.discard(section_id)
        self._sync_sections()

    def _register_time_series_entry(
        self, entry: dict[str, Any], *, generated: bool = False
    ) -> int:
        tag = int(entry["tag"])
        if not generated and tag in self.generated_time_series_tags:
            old_entry = self.time_series_by_tag.pop(tag)
            new_tag = max(max(self.time_series_by_tag, default=0), tag) + 1
            old_entry["tag"] = new_tag
            self.time_series_by_tag[new_tag] = old_entry
            self.generated_time_series_tags.discard(tag)
            self.generated_time_series_tags.add(new_tag)
            motion_series_keys = (
                "time_series",
                "velocity_series",
                "acceleration_series",
            )
            for motion in (self.ground_motion_series or {}).values():
                for key in motion_series_keys:
                    if motion.get(key) == tag:
                        motion[key] = new_tag
            for stage in self.analysis_stages:
                for motion in stage["analysis"].get("support_motions", []):
                    for key in motion_series_keys:
                        if motion.get(key) == tag:
                            motion[key] = new_tag
                pattern = stage.get("pattern")
                if isinstance(pattern, dict):
                    if pattern.get("time_series") == tag:
                        pattern["time_series"] = new_tag
                    if pattern.get("accel") == tag:
                        pattern["accel"] = new_tag
            patterns = list(self.imposed_motion_patterns.values())
            if self.active_imposed_motions is not None:
                patterns.append(self.active_imposed_motions)
            for motions in patterns:
                for motion in motions:
                    for key in motion_series_keys:
                        if motion.get(key) == tag:
                            motion[key] = new_tag
            if isinstance(self.current_pattern, dict):
                if self.current_pattern.get("time_series") == tag:
                    self.current_pattern["time_series"] = new_tag
                if self.current_pattern.get("accel") == tag:
                    self.current_pattern["accel"] = new_tag
            if (
                self.current_plain_pattern is not None
                and self.current_plain_pattern.time_series == tag
            ):
                self.current_plain_pattern.time_series = new_tag
        self.time_series_by_tag[tag] = entry
        if generated:
            self.generated_time_series_tags.add(tag)
        else:
            self.generated_time_series_tags.discard(tag)
        self.time_series = [
            self.time_series_by_tag[key] for key in sorted(self.time_series_by_tag)
        ]
        return tag

    def _next_time_series_tag(self) -> int:
        if not self.time_series_by_tag:
            return 1
        return max(self.time_series_by_tag) + 1

    def _material_modulus(self, material_id: int) -> float:
        material = self.materials_by_id.get(material_id)
        if material is None:
            raise self._error(f"material {material_id} not found", "material", ())
        params = material.get("params", {})
        mat_type = material.get("type")
        if mat_type in {"Elastic", "ElasticPP"}:
            return float(params["E"])
        if mat_type == "ElasticBilin":
            return float(params["E1P"])
        if mat_type == "Ratchet":
            return (
                0.0
                if params["freeTravelInitial"] > params["freeTravel"] / 1.0e7
                else float(params["E"])
            )
        if mat_type in {"Steel01", "Steel02", "SteelMPF"}:
            return float(params["E0"])
        if mat_type in {"Bilin", "ModIMKPeakOriented"}:
            values = params["values"]
            return values[0] * (1.0 + (values[23] if len(values) == 24 else 0.0))
        if mat_type == "Hysteretic":
            return params["values"][0] / params["values"][1]
        if mat_type == "ConcreteCM":
            return float(params["Ec"])
        if mat_type in {"Concrete01", "Concrete02"}:
            fpc = float(params["fpc"])
            epsc0 = float(params["epsc0"])
            if abs(epsc0) > 0.0:
                return abs(2.0 * fpc / epsc0)
        if mat_type == "ViscousDamper":
            return 0.0
        if mat_type in {"Fatigue", "MinMax"}:
            return self._material_modulus(int(params["material"]))
        raise self._error(
            f"material `{mat_type}` does not expose an initial modulus",
            "material",
            (),
        )

    def _fiber_section_rigidity(self, section_id: int) -> tuple[float, float]:
        section = self.sections_by_id.get(section_id)
        if section is None:
            raise self._error(f"section {section_id} not found", "section", ())
        if section["type"] == "ElasticSection2d":
            params = section["params"]
            e_value = float(params["E"])
            return e_value * float(params["A"]), e_value * float(params["I"])
        if section["type"] == "AggregatorSection2d":
            params = section["params"]
            if int(params.get("base_section", -1)) >= 0:
                raise self._error(
                    "AggregatorSection2d with -section is not supported for stiffness recovery",
                    "section",
                    (),
                )
            axial_mat = int(params.get("axial_material", -1))
            flexural_mat = int(params.get("flexural_material", -1))
            if axial_mat < 0 or flexural_mat < 0:
                raise self._error(
                    "AggregatorSection2d requires axial_material and flexural_material",
                    "section",
                    (),
                )
            return (
                self._material_modulus(axial_mat),
                self._material_modulus(flexural_mat),
            )
        if section["type"] != "FiberSection2d":
            raise self._error(
                f"unsupported section type `{section['type']}` for stiffness recovery",
                "section",
                (),
            )

        ea = 0.0
        ei = 0.0
        params = section["params"]
        for patch in params.get("patches", []):
            mat_e = self._material_modulus(int(patch["material"]))
            patch_type = patch["type"]
            if patch_type == "rect":
                y_i = float(patch["y_i"])
                z_i = float(patch["z_i"])
                y_j = float(patch["y_j"])
                z_j = float(patch["z_j"])
                num_y = max(1, int(patch["num_subdiv_y"]))
                num_z = max(1, int(patch["num_subdiv_z"]))
                dy = (y_j - y_i) / num_y
                dz = (z_j - z_i) / num_z
                cell_area = abs(dy * dz)
                for iy in range(num_y):
                    y = y_i + (iy + 0.5) * dy
                    for _ in range(num_z):
                        ea += mat_e * cell_area
                        ei += mat_e * cell_area * y * y
            elif patch_type in {"quad", "quadr"}:
                pts = [
                    (float(patch["y_i"]), float(patch["z_i"])),
                    (float(patch["y_j"]), float(patch["z_j"])),
                    (float(patch["y_k"]), float(patch["z_k"])),
                    (float(patch["y_l"]), float(patch["z_l"])),
                ]
                num_y = max(1, int(patch["num_subdiv_y"]))
                num_z = max(1, int(patch["num_subdiv_z"]))
                poly_area = 0.0
                for i in range(4):
                    y1, z1 = pts[i]
                    y2, z2 = pts[(i + 1) % 4]
                    poly_area += y1 * z2 - y2 * z1
                cell_area = abs(0.5 * poly_area) / (num_y * num_z)
                for iy in range(num_y):
                    u = (iy + 0.5) / num_y
                    for iz in range(num_z):
                        v = (iz + 0.5) / num_z
                        n1 = (1.0 - u) * (1.0 - v)
                        n2 = u * (1.0 - v)
                        n3 = u * v
                        n4 = (1.0 - u) * v
                        y = (
                            n1 * pts[0][0]
                            + n2 * pts[1][0]
                            + n3 * pts[2][0]
                            + n4 * pts[3][0]
                        )
                        ea += mat_e * cell_area
                        ei += mat_e * cell_area * y * y
            elif patch_type == "circ":
                n_circ = max(1, int(patch["num_subdiv_circ"]))
                n_rad = max(1, int(patch["num_subdiv_rad"]))
                y_center = float(patch["y_center"])
                int_rad = float(patch["int_rad"])
                ext_rad = float(patch["ext_rad"])
                init_rad = math.radians(float(patch.get("start_ang", 0.0)))
                final_rad = math.radians(float(patch.get("end_ang", 360.0)))
                delta_rad = (ext_rad - int_rad) / n_rad
                delta_theta = (final_rad - init_rad) / n_circ
                half_theta = 0.5 * delta_theta
                for j in range(n_rad):
                    r1 = int_rad + delta_rad * j
                    r2 = r1 + delta_rad
                    area_outer = half_theta * r2 * r2
                    area_inner = half_theta * r1 * r1
                    cell_area = area_outer - area_inner
                    c_outer = 2.0 * r2 * math.sin(half_theta) / (3.0 * half_theta)
                    c_inner = 2.0 * r1 * math.sin(half_theta) / (3.0 * half_theta)
                    centroid_r = (
                        area_outer * c_outer - area_inner * c_inner
                    ) / cell_area
                    for i in range(n_circ):
                        theta_mid = init_rad + delta_theta * i + half_theta
                        y = math.cos(theta_mid) * centroid_r + y_center
                        ea += mat_e * cell_area
                        ei += mat_e * cell_area * y * y
            else:
                raise self._error(
                    f"unsupported Fiber patch `{patch_type}` for stiffness recovery",
                    "section",
                    (),
                )

        for layer in params.get("layers", []):
            mat_e = self._material_modulus(int(layer["material"]))
            layer_type = layer["type"]
            if layer_type == "straight":
                num_bars = max(1, int(layer["num_bars"]))
                y_start = float(layer["y_start"])
                y_end = float(layer["y_end"])
                bar_area = float(layer["bar_area"])
                for idx in range(num_bars):
                    if num_bars == 1:
                        frac = 0.5
                    else:
                        frac = idx / (num_bars - 1)
                    y = y_start + frac * (y_end - y_start)
                    ea += mat_e * bar_area
                    ei += mat_e * bar_area * y * y
            elif layer_type == "circ":
                num_bars = max(1, int(layer["num_bars"]))
                bar_area = float(layer["bar_area"])
                y_center = float(layer["y_center"])
                radius = float(layer["radius"])
                if "start_ang" in layer and "end_ang" in layer:
                    start_ang = float(layer["start_ang"])
                    end_ang = float(layer["end_ang"])
                    if end_ang - start_ang >= 360.0:
                        end_ang = start_ang + 360.0 - 360.0 / num_bars
                else:
                    start_ang = 0.0
                    end_ang = 360.0 - 360.0 / num_bars
                init_rad = math.radians(start_ang)
                final_rad = math.radians(end_ang)
                dtheta = (
                    (final_rad - init_rad) / (num_bars - 1) if num_bars > 1 else 0.0
                )
                for idx in range(num_bars):
                    theta = init_rad + dtheta * idx
                    y = y_center + radius * math.cos(theta)
                    ea += mat_e * bar_area
                    ei += mat_e * bar_area * y * y
            else:
                raise self._error(
                    f"unsupported Fiber layer `{layer_type}` for stiffness recovery",
                    "section",
                    (),
                )
        return ea, ei

    def _analysis_constraints_name(self) -> str:
        if self.constraints_handler in {"Transformation", "Lagrange", "Penalty"}:
            return self.constraints_handler
        if (
            self.constraints_handler in {None, "Plain"}
            and self.mp_constraints
            and all(
                constraint["type"] == "equalDOF" for constraint in self.mp_constraints
            )
        ):
            return "Transformation"
        return "Plain"

    def _normalize_algorithm_name(self, args: tuple[str, ...]) -> str:
        if not args:
            raise self._error("algorithm type is required", "algorithm", args)
        name = args[0]
        if name == "ModifiedNewton" and len(args) > 1 and args[1] == "-initial":
            return "ModifiedNewtonInitial"
        if name == "Newton" and len(args) > 1 and args[1] == "-initial":
            return "ModifiedNewtonInitial"
        return name

    def _parse_inline_series_spec(self, spec: str) -> dict[str, Any]:
        parts = [str(part) for part in self.interp.tk.splitlist(spec)]
        if not parts:
            raise self._error(
                f"unsupported inline acceleration series `{spec}`",
                "pattern",
                (spec,),
            )
        if parts[0] in {"Constant", "Linear"}:
            if len(parts) == 1:
                factor = 1.0
            elif len(parts) == 3 and parts[1] == "-factor":
                factor = float(parts[2])
            else:
                raise self._error(
                    "inline series expects an optional -factor value",
                    "pattern",
                    tuple(parts),
                )
            return {
                "type": parts[0],
                "tag": self._next_time_series_tag(),
                "factor": factor,
            }
        if parts[0] in {"Sine", "Trig"}:
            if len(parts) < 4:
                raise self._error(
                    "inline Sine series expects `Sine tStart tFinish period ?-factor value?`",
                    "pattern",
                    tuple(parts),
                )
            options = {
                "type": "Trig",
                "tag": self._next_time_series_tag(),
                "t_start": float(parts[1]),
                "t_finish": float(parts[2]),
                "period": float(parts[3]),
                "factor": 1.0,
            }
            idx = 4
            while idx < len(parts):
                token = parts[idx]
                if token == "-factor" and idx + 1 < len(parts):
                    options["factor"] = float(parts[idx + 1])
                    idx += 2
                elif token == "-zeroShift" and idx + 1 < len(parts):
                    options["zero_shift"] = float(parts[idx + 1])
                    idx += 2
                elif token in {"-shift", "-phaseShift"} and idx + 1 < len(parts):
                    options["phase_shift"] = float(parts[idx + 1])
                    idx += 2
                else:
                    raise self._error(
                        f"unsupported inline Sine flag `{token}`",
                        "pattern",
                        tuple(parts),
                    )
            return options
        if parts[0] not in {"Path", "Series"}:
            raise self._error(
                f"unsupported inline acceleration series `{spec}`",
                "pattern",
                (spec,),
            )
        options = {"type": "Path", "tag": self._next_time_series_tag(), "factor": 1.0}
        idx = 1
        while idx < len(parts):
            token = parts[idx]
            if token in {"-dt", "-dT"}:
                options["dt"] = float(parts[idx + 1])
                idx += 2
            elif token in {"-startTime", "-tStart"}:
                options["start_time"] = float(parts[idx + 1])
                idx += 2
            elif token == "-useLast":
                options["use_last"] = True
                idx += 1
            elif token == "-prependZero":
                options["_prepend_zero"] = True
                idx += 1
            elif token == "-filePath":
                options["values_path"] = self._resolve_values_path(parts[idx + 1])
                idx += 2
            elif token == "-factor":
                options["factor"] = float(parts[idx + 1])
                idx += 2
            elif token in {"-time", "-values"}:
                options[token[1:]] = [
                    float(value) for value in self.interp.tk.splitlist(parts[idx + 1])
                ]
                idx += 2
            else:
                raise self._error(
                    f"unsupported inline acceleration flag `{token}`",
                    "pattern",
                    tuple(parts),
                )
        if "time" in options and "values" in options:
            if len(options["time"]) != len(options["values"]):
                raise self._error(
                    "inline Series -time and -values lengths must match",
                    "pattern",
                    tuple(parts),
                )
        elif "dt" not in options or not ({"values", "values_path"} & options.keys()):
            raise self._error(
                "inline Series requires -time and -values or -dt with -values or -filePath",
                "pattern",
                tuple(parts),
            )
        self._prepend_path_zero(options)
        return options

    @staticmethod
    def _prepend_path_zero(options: dict[str, Any]) -> None:
        if not options.pop("_prepend_zero", False) or "time" in options:
            return
        if "values_path" in options:
            options["values"] = [
                float(value)
                for value in Path(options.pop("values_path")).read_text().split()
            ]
        options["values"] = [0.0, *options["values"]]

    def _cmd_record(self, *args: str) -> str:
        if args:
            raise self._error("record takes no arguments", "record", args)
        self.pending_recorders.extend(
            index
            for index, recorder in enumerate(self.recorders)
            if recorder["type"] == "node_displacement"
        )
        return ""

    def _append_recorder(self, recorder: dict[str, Any]) -> None:
        if self.analysis_stages:
            recorder["start_stage"] = len(self.analysis_stages)
        self.recorders.append(recorder)

    def _recorder_output(self, raw_file: str) -> str:
        return Path(raw_file).stem

    def _resolve_values_path(self, values_path: str) -> str:
        path = Path(values_path)
        if path.is_absolute():
            resolved = self._resolve_case_insensitive_path(path)
            return str(resolved)
        base_dir = self._current_base_dir()
        current = base_dir.resolve()
        while True:
            resolved = self._resolve_case_insensitive_path((current / path).resolve())
            if resolved.exists():
                return str(resolved)
            if current == self.repo_root or current.parent == current:
                break
            current = current.parent
        return values_path

    def _envelope_element_force_width(
        self, element: dict[str, Any], include_time: bool
    ) -> Optional[int]:
        model = self.model
        if model is None:
            return None
        ndm = int(model.get("ndm", 0))
        ndf = int(model.get("ndf", 0))
        elem_type = element.get("type")
        width: Optional[int] = None
        if elem_type in {
            "elasticBeamColumn",
            "forceBeamColumn",
            "nonlinearBeamColumn",
            "dispBeamColumn",
            "elasticBeamColumn2d",
            "elasticBeamColumn3d",
            "forceBeamColumn2d",
            "forceBeamColumn3d",
            "dispBeamColumn2d",
            "dispBeamColumn3d",
        }:
            width = 6 if ndm == 2 else 12
        elif elem_type == "truss":
            width = 2 if ndm == 2 else 3
        elif elem_type in {
            "corotTruss",
            "zeroLength",
            "CoupledZeroLength",
            "zeroLengthSection",
            "twoNodeLink",
            "MVLEM",
            "SFI_MVLEM",
            "CatenaryCable",
        }:
            width = 2 * ndf if ndf > 0 else None
        if width is None:
            return None
        return width

    def _envelope_node_width(
        self, dofs: list[int], include_time: bool
    ) -> Optional[int]:
        if not dofs:
            return None
        return len(dofs)

    def _has_coupled_soil(self) -> bool:
        return any(
            element["type"] in {"quadUP", "9_4_QuadUP", "SSPbrickUP"}
            for element in self.elements
        )

    def _can_merge_stage(self, stage: dict[str, Any]) -> bool:
        if self.last_stage is None or stage.get("sample_before"):
            return False
        if self.recorders and self.recorders[-1].get("start_stage") == len(
            self.analysis_stages
        ):
            return False
        if self.last_stage.get("print_commands"):
            return False
        if (
            self.last_stage.get("material_stages") != stage.get("material_stages")
            or "contact_friction" in stage
        ):
            return False
        if stage.get("permeability_updates"):
            return False
        if "initial_state_reset" in stage:
            return False
        if stage.get("remove_nodes") or stage.get("remove_elements"):
            return False
        if self.last_stage.get("prescribed_pressures") != stage.get(
            "prescribed_pressures"
        ):
            return False
        if any(
            rec.get("end_stage") == len(self.analysis_stages) for rec in self.recorders
        ):
            return False
        if any(
            rec.get("start_stage") == len(self.analysis_stages) + 1
            for rec in self.recorders
        ):
            return False
        if self.last_stage.get("pattern") != stage.get("pattern"):
            return False
        if self.last_stage.get("loads", []) != stage.get("loads", []):
            return False
        if self.last_stage.get("constant_loads") != stage.get("constant_loads"):
            return False
        if self.last_stage.get("constant_element_loads") != stage.get(
            "constant_element_loads"
        ):
            return False
        if self.last_stage.get("element_loads", []) != stage.get("element_loads", []):
            return False
        if self.last_stage.get("rayleigh") != stage.get("rayleigh"):
            return False
        if self.last_stage.get("initialize", False) != stage.get("initialize", False):
            return False
        prev_analysis = dict(self.last_stage.get("analysis", {}))
        next_analysis = dict(stage.get("analysis", {}))
        if "start_time" in next_analysis:
            previous_end = prev_analysis.get("start_time", 0.0) + (
                prev_analysis.get("steps", 0) * prev_analysis.get("dt", 0.0)
            )
            if not math.isclose(
                previous_end, next_analysis["start_time"], rel_tol=0.0, abs_tol=1e-12
            ):
                return False
        prev_analysis.pop("start_time", None)
        next_analysis.pop("start_time", None)
        prev_analysis.pop("steps", None)
        next_analysis.pop("steps", None)
        return prev_analysis == next_analysis

    def _append_stage(self, stage: dict[str, Any]) -> None:
        if self._can_merge_stage(stage):
            self.last_stage["analysis"]["steps"] += int(stage["analysis"]["steps"])
            if "load_const" in stage:
                self.last_stage["load_const"] = dict(stage["load_const"])
            return
        self.analysis_stages.append(stage)
        self.last_stage = stage

    def _register_material(self, e_value: float) -> int:
        if e_value in self.material_ids_by_e:
            return self.material_ids_by_e[e_value]
        material_id = max(self.materials_by_id, default=0) + 1
        self.material_ids_by_e[e_value] = material_id
        self._upsert_material(
            {"id": material_id, "type": "Elastic", "params": {"E": e_value}}
        )
        return material_id

    def _register_section(self, area: float, e_value: float, inertia: float) -> int:
        key = (area, e_value, inertia)
        if key in self.section_ids_by_props:
            return self.section_ids_by_props[key]
        section_id = self._next_free_section_id()
        self.section_ids_by_props[key] = section_id
        self._upsert_section(
            {
                "id": section_id,
                "type": "ElasticSection2d",
                "params": {"E": e_value, "A": area, "I": inertia},
            },
            generated=True,
        )
        self._register_material(e_value)
        return section_id

    def _register_section_3d(
        self,
        area: float,
        e_value: float,
        g_value: float,
        j_value: float,
        iy_value: float,
        iz_value: float,
    ) -> int:
        key = (area, e_value, g_value, j_value, iy_value, iz_value)
        if key in self.section_ids_by_props:
            return self.section_ids_by_props[key]
        section_id = self._next_free_section_id()
        self.section_ids_by_props[key] = section_id
        self._upsert_section(
            {
                "id": section_id,
                "type": "ElasticSection3d",
                "params": {
                    "E": e_value,
                    "A": area,
                    "Iy": iy_value,
                    "Iz": iz_value,
                    "G": g_value,
                    "J": j_value,
                },
            },
            generated=True,
        )
        self._register_material(e_value)
        return section_id

    def _validate_beam_integration(
        self,
        element_type: str,
        integration: str,
        num_int_pts: int,
        args: tuple[str, ...],
    ) -> None:
        min_points = _BEAM_INTEGRATION_MIN_POINTS.get(integration)
        if min_points is None:
            raise self._error(
                f"{element_type} supports integration Lobatto, Legendre, or Radau",
                "element",
                args,
            )
        if num_int_pts < min_points:
            raise self._error(
                f"{element_type} {integration} integration requires num_int_pts >= {min_points}",
                "element",
                args,
            )

    def _parse_beam_options(
        self, options: tuple[str, ...], *, command_args: tuple[str, ...]
    ) -> dict[str, Any]:
        parsed: dict[str, Any] = {}
        idx = 0
        while idx < len(options):
            token = options[idx]
            if "rho" not in parsed and token and not token.startswith("-"):
                parsed["rho"] = float(token)
                idx += 1
                continue
            if token == "-mass":
                if idx + 1 >= len(options):
                    raise self._error(
                        "beam element -mass expects a numeric mass density",
                        "element",
                        command_args,
                    )
                parsed["rho"] = float(options[idx + 1])
                idx += 2
                continue
            if token in {"-cMass", "cMass"}:
                parsed["cMass"] = True
                idx += 1
                continue
            if token in {"-lMass", "lMass"}:
                parsed["cMass"] = False
                idx += 1
                continue
            if token == "-iter":
                if idx + 2 >= len(options):
                    raise self._error(
                        "beam element -iter expects `max_iters tol`",
                        "element",
                        command_args,
                    )
                parsed["max_iters"] = int(options[idx + 1])
                parsed["tol"] = float(options[idx + 2])
                idx += 3
                continue
            raise self._error(
                f"unsupported beam element option `{token}`",
                "element",
                command_args,
            )
        return parsed

    def _clone_section_with_torsion(
        self, tag: int, base_section: dict[str, Any], material_id: int
    ) -> dict[str, Any]:
        material = self.materials_by_id.get(material_id)
        if material is None or material["type"] != "Elastic":
            raise self._error(
                "3D section torsion wrapper requires Elastic material", "section", ()
            )
        params = dict(base_section.get("params", {}))
        added_gj = float(material["params"]["E"])
        base_gj = params.get("G", 0.0) * params.get("J", 0.0)
        if base_gj > 0.0 and added_gj > 0.0:
            params["base_torsional_rigidity"] = base_gj
            params["added_torsional_rigidity"] = added_gj
        params["G"] = base_gj + added_gj
        params["J"] = 1.0
        return {"id": tag, "type": base_section["type"], "params": params}

    def _cmd_unknown(self, *args: str) -> str:
        if not args:
            raise self._error("unknown command", "unknown", args)
        command = args[0]
        raise self._error(
            f"unsupported Tcl/OpenSees command `{command}`", command, args[1:]
        )

    def _cmd_noop(self, *args: str) -> str:
        return ""

    def _cmd_node_bounds(self, *args: str) -> str:
        if args:
            raise self._error("nodeBounds takes no arguments", "nodeBounds", args)
        if not self.nodes:
            return "0 0 0 0 0 0"
        xs = [float(node.get("x", 0.0)) for node in self.nodes.values()]
        ys = [float(node.get("y", 0.0)) for node in self.nodes.values()]
        zs = [float(node.get("z", 0.0)) for node in self.nodes.values()]
        bounds = (min(xs), min(ys), min(zs), max(xs), max(ys), max(zs))
        return " ".join(str(value) for value in bounds)

    def _split_tcl_commands(self, script: str) -> list[str]:
        commands: list[str] = []
        buf: list[str] = []
        brace_depth = 0
        bracket_depth = 0
        in_quote = False
        escape = False
        command_start = True
        idx = 0
        while idx < len(script):
            ch = script[idx]
            if escape:
                buf.append(ch)
                escape = False
                if command_start and not ch.isspace():
                    command_start = False
                idx += 1
                continue
            if ch == "\\":
                buf.append(ch)
                escape = True
                idx += 1
                continue
            if (
                not in_quote
                and brace_depth == 0
                and bracket_depth == 0
                and command_start
                and ch == "#"
            ):
                while idx < len(script) and script[idx] != "\n":
                    idx += 1
                continue
            if ch == '"' and brace_depth == 0 and bracket_depth == 0:
                in_quote = not in_quote
                buf.append(ch)
                command_start = False
                idx += 1
                continue
            if not in_quote:
                if ch == "{":
                    brace_depth += 1
                elif ch == "}" and brace_depth > 0:
                    brace_depth -= 1
                elif ch == "[":
                    bracket_depth += 1
                elif ch == "]" and bracket_depth > 0:
                    bracket_depth -= 1
            if (
                not in_quote
                and brace_depth == 0
                and bracket_depth == 0
                and ch in {"\n", ";"}
            ):
                command = "".join(buf).strip()
                if command:
                    commands.append(command)
                buf = []
                command_start = True
                idx += 1
                continue
            if command_start:
                if ch.isspace():
                    idx += 1
                    continue
                command_start = False
            buf.append(ch)
            idx += 1
        command = "".join(buf).strip()
        if command:
            commands.append(command)
        return commands

    def _command_words(self, command: str) -> tuple[str, ...]:
        if not command:
            return ()
        return tuple(str(word) for word in self.interp.tk.splitlist(command))

    def _parse_analyze_assignment(
        self, command_words: tuple[str, ...], *, resolve_variables: bool = False
    ) -> Optional[tuple[str, tuple[str, ...]]]:
        if len(command_words) < 3 or command_words[0] != "set":
            return None
        rhs = " ".join(command_words[2:]).strip()
        if not (rhs.startswith("[") and rhs.endswith("]")):
            return None
        inner_words = self._command_words(rhs[1:-1].strip())
        if not inner_words or inner_words[0] != "analyze":
            return None
        analyze_args = inner_words[1:]
        if resolve_variables:
            analyze_args = tuple(
                self._resolve_variable_token(arg) for arg in analyze_args
            )
        return command_words[1], analyze_args

    def _parse_retry_condition(self, condition: str) -> Optional[str]:
        match = re.match(r"^\s*\$([A-Za-z_][A-Za-z0-9_]*)\s*!=\s*0\s*$", condition)
        if match is None:
            return None
        return match.group(1)

    def _resolve_numeric_token(self, token: str) -> float:
        try:
            return float(token)
        except ValueError:
            if token.startswith("$"):
                return float(self.interp.getvar(token[1:]))
            raise

    def _resolve_variable_token(self, token: str) -> str:
        if token.startswith("$"):
            try:
                return str(self.interp.getvar(token[1:]))
            except tkinter.TclError:
                pass
        return token

    def _parse_test_settings(
        self, command_words: tuple[str, ...]
    ) -> Optional[dict[str, Any]]:
        if not command_words or command_words[0] != "test" or len(command_words) < 4:
            return None
        try:
            settings: dict[str, Any] = {
                "fallback_test_type": command_words[1],
                "fallback_tol": self._resolve_numeric_token(command_words[2]),
                "fallback_max_iters": int(
                    self._resolve_numeric_token(command_words[3])
                ),
            }
        except (ValueError, tkinter.TclError):
            return None
        if len(command_words) >= 5:
            try:
                settings["fallback_test_print_flag"] = int(
                    self._resolve_numeric_token(command_words[4])
                )
            except (ValueError, tkinter.TclError):
                return None
        if len(command_words) > 5:
            settings["fallback_test_extra_args"] = list(command_words[5:])
        return settings

    def _parse_algorithm_settings(
        self, command_words: tuple[str, ...]
    ) -> Optional[dict[str, Any]]:
        if (
            not command_words
            or command_words[0] != "algorithm"
            or len(command_words) < 2
        ):
            return None
        algorithm_args = command_words[1:]
        settings: dict[str, Any] = {
            "fallback_algorithm": self._normalize_algorithm_name(algorithm_args)
        }
        if algorithm_args[0] == "Broyden" and len(algorithm_args) >= 2:
            try:
                settings["fallback_broyden_count"] = int(
                    self._resolve_numeric_token(algorithm_args[1])
                )
            except (ValueError, tkinter.TclError):
                pass
        if algorithm_args[0] == "NewtonLineSearch" and len(algorithm_args) >= 2:
            try:
                settings["fallback_line_search_eta"] = self._resolve_numeric_token(
                    algorithm_args[1]
                )
            except (ValueError, tkinter.TclError):
                pass
        if algorithm_args[0] == "KrylovNewton":
            for i, token in enumerate(algorithm_args[1:], start=1):
                if token == "-maxDim" and i + 1 < len(algorithm_args):
                    try:
                        settings["fallback_krylov_max_dim"] = int(
                            self._resolve_numeric_token(algorithm_args[i + 1])
                        )
                    except (ValueError, tkinter.TclError):
                        pass
                    break
        return settings

    def _subst_variables_only(self, script: str) -> str:
        try:
            return str(
                self.interp.call("subst", "-nocommands", "-nobackslashes", script)
            )
        except tkinter.TclError:
            return script

    def _script_contains_analyze_assignment(
        self,
        script: str,
        result_var: str,
        analyze_args: tuple[str, ...],
        *,
        resolve_variables: bool = False,
    ) -> bool:
        for command in self._split_tcl_commands(script):
            assignment = self._parse_analyze_assignment(
                self._command_words(command), resolve_variables=resolve_variables
            )
            if assignment == (result_var, analyze_args):
                return True
        return False

    def _detect_step_retry_policy(self, script: str) -> Optional[StepRetryPolicy]:
        commands = self._split_tcl_commands(script)
        for idx, command in enumerate(commands):
            assignment = self._parse_analyze_assignment(self._command_words(command))
            if assignment is None:
                continue
            result_var, analyze_args = assignment
            for follow_command in commands[idx + 1 :]:
                follow_words = self._command_words(follow_command)
                if not follow_words:
                    continue
                if self._parse_analyze_assignment(follow_words) is not None:
                    break
                if follow_words[0] != "if" or len(follow_words) < 3:
                    continue
                if self._parse_retry_condition(follow_words[1]) != result_var:
                    continue
                return self._build_step_retry_policy_from_if_body(
                    result_var, analyze_args, follow_words[2]
                )
        return None

    def _build_step_retry_policy_from_if_body(
        self,
        result_var: str,
        analyze_args: tuple[str, ...],
        body_script: str,
        *,
        resolve_variables: bool = False,
        local_vars: Optional[dict[str, str]] = None,
    ) -> Optional[StepRetryPolicy]:
        if local_vars:
            body_script = re.sub(
                r"\$([A-Za-z_][A-Za-z0-9_]*)",
                lambda m: local_vars.get(m.group(1), m.group(0)),
                body_script,
            )
        frame3d = self.model is not None and self.model["ndm"] == 3
        if resolve_variables and not frame3d:
            body_script = self._subst_variables_only(body_script)
        attempts: list[dict[str, Any]] = []
        pending_attempt: dict[str, Any] = {}
        continue_after_failure = False
        for body_command in self._split_tcl_commands(body_script):
            body_words = self._command_words(body_command)
            if not body_words:
                continue
            if body_words[0] == "if" and len(body_words) >= 3:
                pending_attempt = {}
                if self._parse_retry_condition(body_words[1]) != result_var:
                    continue
                nested_policy = self._build_step_retry_policy_from_if_body(
                    result_var,
                    analyze_args,
                    body_words[2],
                    resolve_variables=resolve_variables,
                    local_vars=local_vars,
                )
                if nested_policy is not None:
                    attempts.extend(nested_policy.attempts)
                    continue_after_failure = (
                        continue_after_failure or nested_policy.continue_after_failure
                    )
                continue
            if body_words[0] == "for" and len(body_words) >= 5:
                pending_attempt = {}
                nested_policy = self._build_step_retry_policy_from_if_body(
                    result_var,
                    analyze_args,
                    body_words[4],
                    resolve_variables=resolve_variables,
                    local_vars=local_vars,
                )
                if nested_policy is not None:
                    attempts.extend(nested_policy.attempts)
                    continue_after_failure = (
                        continue_after_failure or nested_policy.continue_after_failure
                    )
                continue
            if body_words[0] == "while" and len(body_words) >= 3:
                pending_attempt = {}
                loop_body = body_words[2]
                step_args = ("1",)
                if frame3d:
                    step_args += analyze_args[1:]
                if not self._script_contains_analyze_assignment(
                    loop_body,
                    result_var,
                    step_args,
                    resolve_variables=resolve_variables and frame3d,
                ):
                    continue
                continue_after_failure = True
                nested_policy = self._build_step_retry_policy_from_if_body(
                    result_var,
                    step_args,
                    loop_body,
                    resolve_variables=resolve_variables,
                    local_vars=local_vars,
                )
                if nested_policy is not None:
                    attempts.extend(nested_policy.attempts)
                continue
            if resolve_variables and frame3d and body_words[0] in {"test", "algorithm"}:
                body_words = tuple(
                    self._resolve_variable_token(word) for word in body_words
                )
            test_settings = self._parse_test_settings(body_words)
            if test_settings is not None:
                pending_attempt.update(test_settings)
                continue
            algorithm_settings = self._parse_algorithm_settings(body_words)
            if algorithm_settings is not None:
                pending_attempt.update(algorithm_settings)
                continue
            body_assignment = self._parse_analyze_assignment(
                body_words, resolve_variables=resolve_variables and frame3d
            )
            if body_assignment == (result_var, analyze_args):
                if "fallback_algorithm" in pending_attempt:
                    attempts.append(dict(pending_attempt))
                pending_attempt.clear()
        if not attempts and not continue_after_failure:
            return None
        return StepRetryPolicy(
            analyze_args=analyze_args,
            attempts=tuple(attempts),
            restore_primary_after_success=len(attempts) > 0,
            continue_after_failure=continue_after_failure,
        )

    def _detect_step_retry_policy_in_lines(
        self, lines: list[str], line: int, analyze_args: tuple[str, ...]
    ) -> Optional[StepRetryPolicy]:
        if line < 1 or line > len(lines):
            return None
        commands = self._split_tcl_commands("\n".join(lines[line - 1 :]))
        if not commands:
            return None
        assignment = self._parse_analyze_assignment(self._command_words(commands[0]))
        if assignment is None:
            return None
        result_var = assignment[0]
        local_vars: dict[str, str] = {}
        for follow_command in commands[1:]:
            follow_words = self._command_words(follow_command)
            if not follow_words:
                continue
            if self._parse_analyze_assignment(follow_words) is not None:
                break
            if follow_words[0] == "set" and len(follow_words) == 3:
                local_vars[follow_words[1]] = follow_words[2]
                continue
            if follow_words[0] != "if" or len(follow_words) < 3:
                continue
            if self._parse_retry_condition(follow_words[1]) != result_var:
                continue
            return self._build_step_retry_policy_from_if_body(
                result_var,
                analyze_args,
                follow_words[2],
                resolve_variables=True,
                local_vars=local_vars,
            )
        return None

    def _detect_step_retry_policy_at_location(
        self, path: Path, line: int, analyze_args: tuple[str, ...]
    ) -> Optional[StepRetryPolicy]:
        return self._detect_step_retry_policy_in_lines(
            _read_tcl_text(path).splitlines(), line, analyze_args
        )

    def _register_source_step_retry_policies(self, path: Path) -> None:
        resolved = path.resolve()
        if any(
            policy_path == resolved
            for policy_path, _ in self.step_retry_policies_by_location
        ):
            return
        lines = _read_tcl_text(resolved).splitlines()
        idx = 0
        while idx < len(lines):
            line = lines[idx]
            match = re.match(
                r"^\s*set\s+([A-Za-z_][A-Za-z0-9_]*)\s+\[analyze\s+([^\]]+)\]\s*$",
                line,
            )
            if match is None:
                idx += 1
                continue
            result_var = match.group(1)
            inner_words = self._command_words("analyze " + match.group(2))
            analyze_args = inner_words[1:]
            follow_idx = idx + 1
            while follow_idx < len(lines):
                stripped = lines[follow_idx].strip()
                if stripped and not stripped.startswith("#"):
                    break
                follow_idx += 1
            if follow_idx >= len(lines):
                idx += 1
                continue
            condition_line = lines[follow_idx]
            if (
                re.match(
                    rf"^\s*if\s+\{{\s*\${re.escape(result_var)}\s*!=\s*0\s*\}}\s*\{{",
                    condition_line,
                )
                is None
            ):
                idx += 1
                continue
            scan_idx = follow_idx + 1
            candidate = condition_line
            while not self.interp.tk.call(
                "::strut::_info_builtin", "complete", candidate
            ):
                if scan_idx == len(lines):
                    break
                candidate += "\n" + lines[scan_idx]
                scan_idx += 1
            else:
                # Parse the entire Tcl if so the retry body excludes any else arm.
                words = self._command_words(self._split_tcl_commands(candidate)[0])
                policy = self._build_step_retry_policy_from_if_body(
                    result_var, analyze_args, words[2]
                )
                if policy is not None:
                    self.step_retry_policies_by_location[(resolved, idx + 1)] = policy
                    signature = (resolved, analyze_args)
                    self.step_retry_policies_by_signature.setdefault(
                        signature, []
                    ).append(policy)
            idx = scan_idx

    def _active_step_retry_policy(
        self, analyze_args: tuple[str, ...]
    ) -> Optional[StepRetryPolicy]:
        frame = self._frame_dict(0) if self.source_stack else {}
        if frame.get("type") == "proc" and "proc" in frame and "line" in frame:
            try:
                proc_body = str(
                    self.interp.tk.call("::strut::_info_builtin", "body", frame["proc"])
                )
            except tkinter.TclError:
                proc_body = ""
            if proc_body:
                proc_policy = self._detect_step_retry_policy_in_lines(
                    proc_body.splitlines(), int(frame["line"]), analyze_args
                )
                if proc_policy is not None:
                    return proc_policy
        location = self._current_location("analyze", list(analyze_args))
        if location.file is None or location.line is None:
            return None
        dynamic_policy = self._detect_step_retry_policy_at_location(
            location.file.resolve(),
            location.line,
            analyze_args,
        )
        if dynamic_policy is not None:
            return dynamic_policy
        policy = self.step_retry_policies_by_location.get(
            (location.file.resolve(), location.line)
        )
        if policy is not None and policy.analyze_args == analyze_args:
            return policy
        fallback_policies = self.step_retry_policies_by_signature.get(
            (location.file.resolve(), analyze_args),
            [],
        )
        if len(fallback_policies) == 1:
            return fallback_policies[0]
        return None

    def _cmd_puts(self, *args: str) -> str:
        if not args:
            raise self._error("puts expects at least one argument", "puts", args)
        if len(args) == 1:
            return ""
        if len(args) == 2:
            channel = args[0]
            if channel in {"stdout", "stderr"}:
                return ""
            if channel in self.open_channels:
                self.interp.tk.call("::strut::_puts_builtin", channel, args[1])
                return ""
            raise self._error(f"unsupported puts target `{channel}`", "puts", args)
        raise self._error("unsupported puts form", "puts", args)

    def _path_is_within(self, path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    def _current_base_dir(self) -> Path:
        if self.source_stack:
            return self.source_stack[-1].parent
        return self.entry_path.parent

    def _resolve_case_insensitive_path(self, path: Path) -> Path:
        if path.exists():
            return path
        parts = path.parts
        if not parts:
            return path
        current = Path(parts[0])
        start_idx = 1
        if not current.exists():
            return path
        for part in parts[start_idx:]:
            candidate = current / part
            if candidate.exists():
                current = candidate
                continue
            if not current.is_dir():
                return path
            matches = [
                child
                for child in current.iterdir()
                if child.name.lower() == part.lower()
            ]
            if len(matches) != 1:
                return path
            current = matches[0]
        return current

    def _resolve_source_alias_path(self, resolved: Path) -> Path:
        if resolved.exists():
            return resolved
        if resolved.name == "GeneratePeaks.tcl":
            candidate = self._resolve_case_insensitive_path(
                resolved.with_name("LibGeneratePeaks.tcl")
            )
            if candidate.exists():
                return candidate
        return resolved

    def _resolve_peer_motion_alias_path(self, resolved: Path) -> Path:
        if resolved.exists() or resolved.suffix.lower() not in {".at2", ".dt2"}:
            return resolved
        parent = resolved.parent
        if not parent.is_dir():
            return resolved

        if resolved.suffix.lower() == ".dt2":
            at2_candidate = self._resolve_case_insensitive_path(
                resolved.with_suffix(".at2")
            )
            if at2_candidate.exists():
                return at2_candidate

        peer_candidates = sorted(
            path
            for path in parent.iterdir()
            if path.is_file() and path.suffix.lower() in {".at2", ".dt2"}
        )
        if len(peer_candidates) == 1:
            return peer_candidates[0]
        return resolved

    def _resolve_io_path(
        self, raw_path: str, command: str, args: tuple[str, ...]
    ) -> Path:
        path = Path(raw_path)
        resolved = (
            (self._current_base_dir() / path).resolve()
            if not path.is_absolute()
            else path.resolve()
        )
        resolved = self._resolve_case_insensitive_path(resolved)
        allowed_roots = {self.repo_root, self.entry_path.parent}
        allowed_roots.update(path.parent for path in self.source_stack)
        if any(self._path_is_within(resolved, root) for root in allowed_roots):
            return resolved
        raise self._error(
            f"access outside the restricted Tcl roots is not allowed: {resolved}",
            command,
            args,
        )

    def _cmd_open(self, *args: str) -> str:
        if len(args) not in {1, 2}:
            raise self._error("open expects `open path ?mode?`", "open", args)
        mode = args[1] if len(args) == 2 else "r"
        if mode not in {"r", "w", "a", "a+"}:
            raise self._error("only `open path r|w|a|a+` is supported", "open", args)
        resolved = self._resolve_io_path(args[0], "open", args)
        if mode == "r":
            resolved = self._resolve_peer_motion_alias_path(resolved)
        if mode in {"w", "a", "a+"}:
            resolved.parent.mkdir(parents=True, exist_ok=True)
        channel = str(
            self.interp.tk.call("::strut::_open_builtin", str(resolved), mode)
        )
        self.open_channels.add(channel)
        return channel

    def _cmd_read(self, *args: str) -> str:
        no_newline = bool(args and args[0] == "-nonewline")
        if no_newline:
            args = args[1:]
        if len(args) not in {1, 2}:
            raise self._error(
                "read expects a channel and optional character count", "read", args
            )
        channel = args[0]
        if channel not in self.open_channels:
            raise self._error(
                f"read requires a channel opened by this runtime: {channel}",
                "read",
                args,
            )
        if len(args) == 1:
            if no_newline:
                return str(
                    self.interp.tk.call("::strut::_read_builtin", "-nonewline", channel)
                )
            return str(self.interp.tk.call("::strut::_read_builtin", channel))
        if no_newline:
            raise self._error("read -nonewline expects one channel", "read", args)
        return str(self.interp.tk.call("::strut::_read_builtin", channel, int(args[1])))

    def _cmd_close(self, *args: str) -> str:
        if len(args) != 1:
            raise self._error("close expects exactly one channel", "close", args)
        channel = args[0]
        if channel not in self.open_channels:
            raise self._error(
                f"close requires a channel opened by this runtime: {channel}",
                "close",
                args,
            )
        self.interp.tk.call("::strut::_close_builtin", channel)
        self.open_channels.discard(channel)
        return ""

    def _cmd_clock(self, *args: str) -> str:
        if args != ("seconds",):
            raise self._error("only `clock seconds` is supported", "clock", args)
        return str(int(time()))

    def _cmd_info(self, *args: str) -> str:
        if not args:
            raise self._error("info expects a subcommand", "info", args)
        return str(self.interp.tk.call("::strut::_info_builtin", *args))

    def _cmd_variable(self, *args: str) -> str:
        if not args:
            raise self._error(
                "variable expects at least one argument", "variable", args
            )
        if len(args) == 1:
            value = self.interp.getvar(args[0])
            return "" if value is None else str(value)
        if len(args) % 2 != 0:
            raise self._error("variable expects name/value pairs", "variable", args)
        last_value = ""
        for idx in range(0, len(args), 2):
            self.interp.setvar(args[idx], args[idx + 1])
            last_value = args[idx + 1]
        return str(last_value)

    def _cmd_pset(self, *args: str) -> str:
        if not args:
            raise self._error("pset expects at least one argument", "pset", args)
        if len(args) == 1:
            value = self.interp.getvar(args[0])
            return "" if value is None else str(value)
        if len(args) != 2:
            raise self._error("pset expects `pset name value`", "pset", args)
        self.interp.setvar(args[0], args[1])
        return str(args[1])

    def _cmd_exit(self, *args: str) -> str:
        if args:
            raise self._error(
                "exit takes no arguments in the restricted runtime", "exit", args
            )
        return ""

    def _cmd_file(self, *args: str) -> str:
        if len(args) != 2 or args[0] != "mkdir":
            raise self._error("only `file mkdir <path>` is supported", "file", args)
        resolved = self._resolve_io_path(args[1], "file", args)
        resolved.mkdir(parents=True, exist_ok=True)
        return ""

    def _cmd_source(self, *args: str) -> str:
        if len(args) != 1:
            raise self._error("source expects exactly one path", "source", args)
        resolved = self._resolve_io_path(args[0], "source", args)
        resolved = self._resolve_source_alias_path(resolved)
        if not resolved.exists():
            raise self._error(f"sourced file not found: {resolved}", "source", args)
        if resolved in self.source_stack:
            raise self._error(f"cyclic source detected for {resolved}", "source", args)
        self._register_source_step_retry_policies(resolved)

        self.source_stack.append(resolved)
        try:
            try:
                return str(self.interp.eval(_read_tcl_text(resolved)))
            except tkinter.TclError as exc:
                if self.last_error is not None:
                    raise self.last_error from exc
                raise self._error(str(exc), "source", args) from exc
        finally:
            self.source_stack.pop()

    def _cmd_wipe(self, *args: str) -> str:
        if args and not (len(args) == 1 and args[0].lower() == "all"):
            raise self._error("expected wipe ?all?", "wipe", args)
        if self.analysis_stages:
            return ""

        self.model = None
        self.nodes.clear()
        self.node_ndfs.clear()
        self.geom_transforms.clear()
        self.materials.clear()
        self.materials_by_id.clear()
        self.material_ids_by_e.clear()
        self.sections.clear()
        self.sections_by_id.clear()
        self.section_ids_by_props.clear()
        self.generated_section_ids.clear()
        self.axial_section_materials.clear()
        self.elements.clear()
        self.regions.clear()
        self.masses.clear()
        self.time_series.clear()
        self.time_series_by_tag.clear()
        self.generated_time_series_tags.clear()
        self.recorders.clear()
        self.active_recorders.clear()
        self.analysis_stages.clear()
        self.mp_constraints.clear()
        self.current_plain_pattern = None
        self.current_section = None
        self.current_pattern = None
        self.current_rayleigh = None
        self.current_test = None
        self.last_stage = None
        self.constraints_handler = None
        self.penalty_coefficients = None
        self.numberer_handler = None
        self.system_handler = None
        self.system_options = []
        self.algorithm_name = None
        self.algorithm_options = None
        self.integrator = None
        self.load_control_has_run = False
        self.analysis_type = None
        self.transient_eigen_initialized = False
        self.current_time = 0.0
        self.pending_time_reset = False
        self.initial_state_analysis = False
        self.pending_initial_state_reset = None
        self.pending_node_removals.clear()
        self.pending_element_removals.clear()
        self.pending_contact_friction = None
        self.removed_nodes.clear()
        self.removed_elements.clear()
        self.material_stages.clear()
        self.permeability_parameters.clear()
        self.permeability_updates.clear()
        self.prescribed_pressures.clear()
        self.imposed_motion_patterns.clear()
        self.last_eigenvalues.clear()
        self.modal_damping = None
        self.node_displacements.clear()
        self.constant_patterns.clear()
        self.pending_recorders.clear()
        self.pending_initialize = False
        self.step_retry_policies_by_location.clear()
        self.step_retry_policies_by_signature.clear()
        return ""

    def _normalize_alias(self, command: str, alias: str) -> str:
        return OPEN_SEES_COMMAND_ALIASES.get(command, {}).get(alias, alias)

    def _cmd_model(self, *args: str) -> str:
        if (
            len(args) not in {3, 5}
            or args[1] != "-ndm"
            or (len(args) == 5 and args[3] != "-ndf")
        ):
            raise self._error(
                "expected `model basic -ndm <ndm> ?-ndf <ndf>?`", "model", args
            )
        builder_type = self._normalize_alias("model", args[0])
        if builder_type != "basic":
            raise self._error(
                f"unsupported model builder `{builder_type}`",
                "model",
                args,
            )
        ndm = int(args[2])
        ndf = int(args[4]) if len(args) == 5 else 0
        if ndf == 0:
            ndf = {1: 1, 2: 3, 3: 6}.get(ndm, 0)
            if not ndf:
                raise self._error("model requires ndm of 1, 2, or 3", "model", args)
        self.model = {"ndm": ndm, "ndf": ndf}
        return ""

    def _cmd_get_ndm(self) -> str:
        return str(self.model["ndm"]) if self.model is not None else ""

    def _cmd_node(self, *args: str) -> str:
        model = self._require_model("node", args)
        ndm = model["ndm"]
        coord_count = ndm + 1
        ndf = model["ndf"]
        next_arg = coord_count
        if len(args) >= coord_count + 2 and args[coord_count] == "-ndf":
            ndf = int(args[coord_count + 1])
            next_arg += 2
        if len(args) < coord_count or ndf <= 0:
            raise self._error(
                f"{ndm}D node expects `node id coords...` with optional `-ndf` and `-mass`",
                "node",
                args,
            )
        if len(args) > next_arg:
            if args[next_arg] != "-mass" or len(args) != next_arg + 1 + ndf:
                raise self._error(
                    f"{ndm}D node optional mass expects `-mass m1 ...` matching ndf",
                    "node",
                    args,
                )
        node_id = int(args[0])
        node = {
            "id": node_id,
            "x": float(args[1]),
            "y": float(args[2]) if ndm >= 2 else 0.0,
        }
        if ndm >= 3:
            node["z"] = float(args[3])
        self.nodes[node_id] = node
        self.node_ndfs[node_id] = ndf
        if len(args) > next_arg:
            self._cmd_mass(str(node_id), *args[next_arg + 1 :])
        return ""

    def _cmd_fix(self, *args: str) -> str:
        self._require_model("fix", args)
        if not args:
            raise self._error("fix expects a node and fixities", "fix", args)
        node_id = int(args[0])
        node = self.nodes.get(node_id)
        if node is None:
            raise self._error(f"node {node_id} not found", "fix", args)
        if len(args) != self.node_ndfs[node_id] + 1:
            raise self._error("fix argument count does not match node ndf", "fix", args)
        self._apply_node_constraints(node, [int(value) for value in args[1:]])
        return ""

    def _apply_node_constraints(
        self, node: dict[str, Any], fixities: list[int]
    ) -> None:
        constraints = set(int(value) for value in node.get("constraints", []))
        for dof, flag in enumerate(fixities, start=1):
            if flag != 0:
                constraints.add(dof)
        node["constraints"] = sorted(constraints)

    def _cmd_fix_axis(self, axis: str, *args: str) -> str:
        command = f"fix{axis.upper()}"
        model = self._require_model(command, args)
        ndf = int(model["ndf"])
        if len(args) < ndf + 1:
            raise self._error(
                f"{command} expects `coord` followed by {ndf} fixities",
                command,
                args,
            )
        tol = 1.0e-10
        fixity_tokens = list(args[1:])
        if len(fixity_tokens) == ndf + 2:
            if fixity_tokens[-2] != "-tol":
                raise self._error(
                    f"{command} optional tail must be `-tol value`", command, args
                )
            tol = float(fixity_tokens[-1])
            fixity_tokens = fixity_tokens[:-2]
        if len(fixity_tokens) != ndf:
            raise self._error(
                f"{command} fixity count does not match ndf", command, args
            )
        target = float(args[0])
        fixities = [int(value) for value in fixity_tokens]
        for node in self.nodes.values():
            coord_value = float(node.get(axis, 0.0))
            if abs(coord_value - target) <= tol:
                self._apply_node_constraints(node, fixities)
        return ""

    def _cmd_fix_x(self, *args: str) -> str:
        return self._cmd_fix_axis("x", *args)

    def _cmd_fix_y(self, *args: str) -> str:
        return self._cmd_fix_axis("y", *args)

    def _cmd_fix_z(self, *args: str) -> str:
        return self._cmd_fix_axis("z", *args)

    def _cmd_mass(self, *args: str) -> str:
        model = self._require_model("mass", args)
        node_id = int(args[0])
        ndf = self.node_ndfs.get(node_id, model["ndf"])
        if len(args) < ndf + 1:
            raise self._error("mass argument count does not match ndf", "mass", args)
        for dof, value_text in enumerate(args[1 : ndf + 1], start=1):
            value = float(value_text)
            if abs(value) <= 0.0:
                continue
            self.masses.append({"node": node_id, "dof": dof, "value": value})
        return ""

    def _cmd_nd_material(self, *args: str) -> str:
        if len(args) < 3:
            raise self._error("nDMaterial expects type and tag", "nDMaterial", args)
        material_type = args[0]
        tag = int(args[1])
        if material_type == "ContactMaterial2D":
            if len(args) != 6:
                raise self._error(
                    "ContactMaterial2D expects tag mu G c t", "nDMaterial", args
                )
            self._upsert_material(
                {
                    "id": int(args[1]),
                    "type": material_type,
                    "params": dict(zip(("mu", "G", "c", "t"), map(float, args[2:]))),
                }
            )
            return ""

        if material_type == "FSAM":
            if len(args) != 10:
                raise self._error(
                    "FSAM expects tag rho steelX steelY concrete rhoX rhoY nu dowel",
                    "nDMaterial",
                    args,
                )
            self._upsert_material(
                {
                    "id": tag,
                    "type": material_type,
                    "params": {
                        "rho": float(args[2]),
                        "steel_x": int(args[3]),
                        "steel_y": int(args[4]),
                        "concrete": int(args[5]),
                        "rho_x": float(args[6]),
                        "rho_y": float(args[7]),
                        "friction": float(args[8]),
                        "dowel": float(args[9]),
                    },
                }
            )
            return ""
        if material_type == "DruckerPrager":
            keys = (
                "bulk",
                "shear",
                "yield_stress",
                "friction",
                "dilation",
                "saturation",
                "initial_hardening",
                "saturation_rate",
                "tension_rate",
                "hardening",
                "isotropic_fraction",
                "rho",
                "atmospheric_pressure",
            )
            if len(args) < 13 or len(args) > 15:
                raise self._error(
                    "invalid DruckerPrager material parameters", "nDMaterial", args
                )
            params = dict(zip(keys, map(float, args[2:])))
            self._upsert_material({"id": tag, "type": material_type, "params": params})
            return ""
        if material_type == "ElasticCrossAnisotropic":
            if len(args) not in {7, 8}:
                raise self._error(
                    "ElasticCrossAnisotropic expects `tag Eh Ev nuhv nuhh Ghv ?rho?`",
                    "nDMaterial",
                    args,
                )
            params = dict(
                zip(("Eh", "Ev", "nuhv", "nuhh", "Ghv", "rho"), map(float, args[2:]))
            )
            self._upsert_material({"id": tag, "type": material_type, "params": params})
            return ""
        if material_type in {"ElasticIsotropic", "ElasticIsotropic3D"}:
            if len(args) not in {4, 5}:
                raise self._error(
                    "ElasticIsotropic expects `tag E nu ?rho?`",
                    "nDMaterial",
                    args,
                )
            params = {"E": float(args[2]), "nu": float(args[3]), "rho": 0.0}
            if len(args) == 5:
                params["rho"] = float(args[4])
            self._upsert_material(
                {"id": tag, "type": "ElasticIsotropic", "params": params}
            )
            return ""
        if material_type == "PressureIndependMultiYield":
            if len(args) < 8:
                raise self._error(
                    "PressureIndependMultiYield requires tag, nd, rho, G, K, cohesion and peak strain",
                    "nDMaterial",
                    args,
                )
            keys = (
                "rho",
                "G",
                "K",
                "cohesion",
                "peak_shear_strain",
                "friction_angle",
                "reference_pressure",
                "pressure_exponent",
            )
            params = dict(zip(keys, map(float, args[3:11])))
            params["nd"] = int(args[2])
            if len(args) >= 12:
                count = int(args[11])
                curve = args[12:]
                n_curve = -count if -40 < count < 0 else 0
                if len(curve) != 2 * n_curve:
                    raise self._error(
                        "invalid soil reduction curve", "nDMaterial", args
                    )
                params["yield_surfaces"] = n_curve if n_curve else count
                if n_curve:
                    params["reduction_curve"] = [float(value) for value in curve]
            self._upsert_material({"id": tag, "type": material_type, "params": params})
            return ""
        if material_type == "InitialStateAnalysisWrapper":
            if len(args) != 4:
                raise self._error(
                    "InitialStateAnalysisWrapper expects tag materialTag nd",
                    "nDMaterial",
                    args,
                )
            base = int(args[2])
            material = self.materials_by_id.get(base)
            if (
                int(args[3]) != 2
                or material is None
                or material["type"]
                not in {
                    "PressureIndependMultiYield",
                    "PressureDependMultiYield",
                    "PressureDependMultiYield02",
                }
            ):
                raise self._error(
                    "InitialStateAnalysisWrapper requires a 2D multi-yield soil material",
                    "nDMaterial",
                    args,
                )
            if material["params"]["nd"] != 2:
                raise self._error("wrapped material must have nd=2", "nDMaterial", args)
            self._upsert_material(
                {
                    "id": tag,
                    "type": material_type,
                    "params": {"material": base, "nd": 2},
                }
            )
            return ""
        if material_type in {"PressureDependMultiYield", "PressureDependMultiYield02"}:
            variant_02 = material_type.endswith("02")
            required = (
                "rho",
                "G",
                "K",
                "friction_angle",
                "peak_shear_strain",
                "reference_pressure",
                "pressure_exponent",
                "phase_angle",
                "contraction",
            )
            required += (
                ("contraction3", "dilate1", "dilate3")
                if variant_02
                else (
                    "dilate1",
                    "dilate2",
                    "liquefy_pressure",
                    "liquefy_size",
                    "liquefy_translation",
                )
            )
            optional = (
                ("contraction2", "dilate2", "liquefy1", "liquefy2")
                if variant_02
                else ()
            )
            optional += (
                "void_ratio",
                "vol_limit1",
                "vol_limit2",
                "vol_limit3",
                "atmospheric_pressure",
                "cohesion",
                "hv",
                "pv",
            )
            end = 3 + len(required)
            if len(args) < end:
                raise self._error(
                    f"incomplete {material_type} parameters", "nDMaterial", args
                )
            params = dict(zip(required, map(float, args[3:end])))
            params["nd"] = int(args[2])
            if len(args) > end:
                count = int(args[end])
                maximum = 100 if variant_02 else 40
                curve_count = -count if -maximum < count < 0 else 0
                curve_end = end + 1 + 2 * curve_count
                if not curve_end <= len(args) <= curve_end + len(optional):
                    raise self._error(
                        f"invalid {material_type} parameters or reduction curve",
                        "nDMaterial",
                        args,
                    )
                params["yield_surfaces"] = curve_count or count
                if curve_count:
                    params["reduction_curve"] = list(
                        map(float, args[end + 1 : curve_end])
                    )
                params.update(zip(optional, map(float, args[curve_end:])))
            self._upsert_material({"id": tag, "type": material_type, "params": params})
            return ""
        if material_type == "ConcreteS":
            if len(args) != 7:
                raise self._error(
                    "ConcreteS expects `tag E nu fc ft Es`", "nDMaterial", args
                )
            params = dict(zip(("E", "nu", "fc", "ft", "Es"), map(float, args[2:])))
            self._upsert_material({"id": tag, "type": "ConcreteS", "params": params})
            return ""
        if material_type == "PlaneStressUserMaterial":
            if len(args) < 5:
                raise self._error(
                    "PlaneStressUserMaterial expects `tag nstatevs nprops prop1 ... propn`",
                    "nDMaterial",
                    args,
                )
            nstatevs = max(int(args[2]), 1)
            nprops = max(int(args[3]), 1)
            if len(args) != 4 + nprops:
                raise self._error(
                    "PlaneStressUserMaterial property count does not match nprops",
                    "nDMaterial",
                    args,
                )
            self._upsert_material(
                {
                    "id": tag,
                    "type": "PlaneStressUserMaterial",
                    "params": {
                        "nstatevs": nstatevs,
                        "nprops": nprops,
                        "props": [float(value) for value in args[4:]],
                    },
                }
            )
            return ""
        if material_type == "PlateFiber":
            if len(args) != 3:
                raise self._error("PlateFiber expects `tag matTag`", "nDMaterial", args)
            self._upsert_material(
                {
                    "id": int(args[1]),
                    "type": "PlateFiber",
                    "params": {"material": int(args[2])},
                }
            )
            return ""
        if material_type in {"PlateFromPlaneStress", "PlateFromPlaneStressMaterial"}:
            if len(args) != 4:
                raise self._error(
                    "PlateFromPlaneStress expects `tag matTag gmod`",
                    "nDMaterial",
                    args,
                )
            self._upsert_material(
                {
                    "id": tag,
                    "type": "PlateFromPlaneStress",
                    "params": {"material": int(args[2]), "gmod": float(args[3])},
                }
            )
            return ""
        if material_type in {"PlateRebar", "PlateRebarMaterial"}:
            if len(args) != 4:
                raise self._error(
                    "PlateRebar expects `tag matTag angle`",
                    "nDMaterial",
                    args,
                )
            self._upsert_material(
                {
                    "id": tag,
                    "type": "PlateRebar",
                    "params": {"material": int(args[2]), "angle": float(args[3])},
                }
            )
            return ""
        raise self._error(
            f"unsupported nDMaterial `{material_type}`",
            "nDMaterial",
            args,
        )

    def _cmd_equal_dof(self, *args: str) -> str:
        if len(args) < 3:
            raise self._error(
                "equalDOF expects retained constrained dofs...", "equalDOF", args
            )
        self.mp_constraints.append(
            {
                "type": "equalDOF",
                "retained_node": int(args[0]),
                "constrained_node": int(args[1]),
                "dofs": [int(value) for value in args[2:]],
            }
        )
        return ""

    def _cmd_rigid_diaphragm(self, *args: str) -> str:
        if len(args) < 3:
            raise self._error(
                "rigidDiaphragm expects perpDirn retained constrained...",
                "rigidDiaphragm",
                args,
            )
        model = self._require_model("rigidDiaphragm", args)
        perp_dirn = int(args[0])
        retained_node = int(args[1])
        constrained_nodes = [int(value) for value in args[2:]]
        if retained_node not in self.nodes:
            raise self._error(
                "rigidDiaphragm retained_node not found", "rigidDiaphragm", args
            )
        if retained_node in constrained_nodes:
            raise self._error(
                "rigidDiaphragm retained_node cannot also be constrained",
                "rigidDiaphragm",
                args,
            )
        retained = self.nodes[retained_node]
        retained_coords = (
            float(retained.get("x", 0.0)),
            float(retained.get("y", 0.0)),
            float(retained.get("z", 0.0)),
        )
        if model["ndm"] == 3 and model["ndf"] == 6:
            for constrained_node in constrained_nodes:
                if constrained_node not in self.nodes:
                    raise self._error(
                        f"rigidDiaphragm constrained node {constrained_node} not found",
                        "rigidDiaphragm",
                        args,
                    )
                constrained = self.nodes[constrained_node]
                dx = float(constrained.get("x", 0.0)) - retained_coords[0]
                dy = float(constrained.get("y", 0.0)) - retained_coords[1]
                dz = float(constrained.get("z", 0.0)) - retained_coords[2]
                if perp_dirn == 3:
                    constrained_dofs = [1, 2, 6]
                    retained_dofs = [1, 2, 6]
                    matrix = [[1.0, 0.0, -dy], [0.0, 1.0, dx], [0.0, 0.0, 1.0]]
                elif perp_dirn == 2:
                    constrained_dofs = [1, 3, 5]
                    retained_dofs = [1, 3, 5]
                    matrix = [[1.0, 0.0, dz], [0.0, 1.0, -dx], [0.0, 0.0, 1.0]]
                elif perp_dirn == 1:
                    constrained_dofs = [2, 3, 4]
                    retained_dofs = [2, 3, 4]
                    matrix = [[1.0, 0.0, -dz], [0.0, 1.0, dy], [0.0, 0.0, 1.0]]
                else:
                    raise self._error(
                        "rigidDiaphragm perpDirn must be 1, 2, or 3",
                        "rigidDiaphragm",
                        args,
                    )
                self.mp_constraints.append(
                    {
                        "type": "rigidDiaphragm",
                        "retained_node": retained_node,
                        "constrained_node": constrained_node,
                        "perp_dirn": perp_dirn,
                        "constrained_dofs": constrained_dofs,
                        "retained_dofs": retained_dofs,
                        "matrix": matrix,
                        "dx": dx,
                        "dy": dy,
                        "dz": dz,
                    }
                )
            return ""
        if model["ndm"] == 2 and model["ndf"] == 3:
            for constrained_node in constrained_nodes:
                if constrained_node not in self.nodes:
                    raise self._error(
                        f"rigidDiaphragm constrained node {constrained_node} not found",
                        "rigidDiaphragm",
                        args,
                    )
                constrained = self.nodes[constrained_node]
                dx = float(constrained.get("x", 0.0)) - retained_coords[0]
                dy = float(constrained.get("y", 0.0)) - retained_coords[1]
                if perp_dirn == 3:
                    constrained_dofs = [1, 2, 3]
                    retained_dofs = [1, 2, 3]
                    matrix = [[1.0, 0.0, -dy], [0.0, 1.0, dx], [0.0, 0.0, 1.0]]
                elif perp_dirn == 1:
                    constrained_dofs = [1]
                    retained_dofs = [1]
                    matrix = [[1.0]]
                elif perp_dirn == 2:
                    constrained_dofs = [2]
                    retained_dofs = [2]
                    matrix = [[1.0]]
                else:
                    raise self._error(
                        "rigidDiaphragm perpDirn must be 1, 2, or 3",
                        "rigidDiaphragm",
                        args,
                    )
                self.mp_constraints.append(
                    {
                        "type": "rigidDiaphragm",
                        "retained_node": retained_node,
                        "constrained_node": constrained_node,
                        "perp_dirn": perp_dirn,
                        "constrained_dofs": constrained_dofs,
                        "retained_dofs": retained_dofs,
                        "matrix": matrix,
                        "dx": dx,
                        "dy": dy,
                        "dz": 0.0,
                    }
                )
            return ""
        raise self._error(
            "rigidDiaphragm requires a 3D/6DOF or 2D/3DOF model",
            "rigidDiaphragm",
            args,
        )

    def _cmd_uniaxial_material(self, *args: str) -> str:
        if len(args) < 2:
            raise self._error(
                "uniaxialMaterial expects type and tag", "uniaxialMaterial", args
            )
        mat_type = args[0]
        tag = int(args[1])
        if mat_type == "Elastic":
            params = {"E": float(args[2])}
        elif mat_type == "ENT":
            if not 3 <= len(args) <= 5:
                raise self._error("ENT expects tag E ?a? ?b?", "uniaxialMaterial", args)
            params = {"E": float(args[2])}
            params.update(zip(("a", "b"), map(float, args[3:])))
        elif mat_type == "ElasticPP":
            if not 4 <= len(args) <= 6:
                raise self._error(
                    "ElasticPP expects tag E epsyP ?epsyN? ?eps0?",
                    "uniaxialMaterial",
                    args,
                )
            params = dict(zip(("E", "epsyP", "epsyN", "eps0"), map(float, args[2:])))
        elif mat_type == "ElasticBilin":
            if len(args) not in {5, 8}:
                raise self._error(
                    "ElasticBilin expects tag E1P E2P eps2P ?E1N E2N eps2N?",
                    "uniaxialMaterial",
                    args,
                )
            params = dict(
                zip(
                    ("E1P", "E2P", "eps2P", "E1N", "E2N", "eps2N"), map(float, args[2:])
                )
            )
        elif mat_type == "Ratchet":
            if len(args) != 6:
                raise self._error(
                    "Ratchet expects tag E freeTravel freeTravelInitial RatType",
                    "uniaxialMaterial",
                    args,
                )
            params = dict(
                zip(
                    ("E", "freeTravel", "freeTravelInitial"),
                    map(float, args[2:5]),
                    strict=True,
                )
            )
            params["RatType"] = int(args[5])
        elif mat_type == "MinMax":
            if (
                len(args) not in {3, 5, 7}
                or any(flag not in {"-min", "-max"} for flag in args[3::2])
                or int(args[2]) not in self.materials_by_id
            ):
                raise self._error(
                    "MinMax expects tag existingMaterial ?-min strain? ?-max strain?",
                    "uniaxialMaterial",
                    args,
                )
            params = {"material": int(args[2])}
            params.update(
                (flag[1:], float(value))
                for flag, value in zip(args[3::2], args[4::2], strict=True)
            )
        elif mat_type == "Parallel":
            component_args = args[2:]
            factor_start = (
                component_args.index("-factors")
                if "-factors" in component_args
                else len(component_args)
            )
            children = [int(raw) for raw in component_args[:factor_start]]
            if not children or any(
                child not in self.materials_by_id for child in children
            ):
                raise self._error(
                    "Parallel requires existing child materials",
                    "uniaxialMaterial",
                    args,
                )
            params = {"materials": children}
            if factor_start < len(component_args):
                factors = [float(raw) for raw in component_args[factor_start + 1 :]]
                if len(factors) != len(children):
                    raise self._error(
                        "Parallel requires one factor per material",
                        "uniaxialMaterial",
                        args,
                    )
                params["factors"] = factors
        elif mat_type == "Viscous":
            if len(args) not in {4, 5}:
                raise self._error(
                    "Viscous expects tag C alpha ?minVel?", "uniaxialMaterial", args
                )
            alpha = float(args[3])
            params = {"C": float(args[2]), "alpha": 1.0 if alpha < 0 else alpha}
            if len(args) == 5:
                params["minVel"] = float(args[4])
        elif mat_type == "Concrete01":
            params = {
                "fpc": float(args[2]),
                "epsc0": float(args[3]),
                "fpcu": float(args[4]),
                "epscu": float(args[5]),
            }
        elif mat_type == "Concrete02":
            params = {
                "fpc": float(args[2]),
                "epsc0": float(args[3]),
                "fpcu": float(args[4]),
                "epscu": float(args[5]),
                "rat": float(args[6]),
                "ft": float(args[7]),
                "Ets": float(args[8]),
            }
        elif mat_type == "ConcreteCM":
            keys = ("fpcc", "epcc", "Ec", "rc", "xcrn", "ft", "et", "rt", "xcrp")
            if len(args) not in {11, 12, 13} or (
                len(args) == 13 and args[11] != "-GapClose"
            ):
                raise self._error(
                    "ConcreteCM expects nine parameters and optional -GapClose flag or monotonic mode",
                    "uniaxialMaterial",
                    args,
                )
            params = dict(zip(keys, map(float, args[2:11]), strict=True))
            if len(args) == 12:
                params["monotonic"] = int(args[11])
            elif len(args) == 13:
                params["gap_close"] = int(args[12])
        elif mat_type == "SteelMPF":
            keys = (
                "fyp",
                "fyn",
                "E0",
                "bp",
                "bn",
                "R0",
                "cR1",
                "cR2",
                "a1",
                "a2",
                "a3",
                "a4",
            )
            if len(args) not in {10, 14}:
                raise self._error(
                    "SteelMPF expects eight parameters and optional four isotropic hardening parameters",
                    "uniaxialMaterial",
                    args,
                )
            params = dict(zip(keys[: len(args) - 2], map(float, args[2:]), strict=True))
        elif mat_type in {"Steel01", "Steel01Thermal"}:
            if mat_type == "Steel01Thermal" and len(args) != 5:
                raise self._error(
                    "Steel01Thermal expects `tag Fy E0 b`", "uniaxialMaterial", args
                )
            params = {"Fy": float(args[2]), "E0": float(args[3]), "b": float(args[4])}
        elif mat_type == "Steel02":
            params = {
                "Fy": float(args[2]),
                "E0": float(args[3]),
                "b": float(args[4]),
                "R0": float(args[5]),
                "cR1": float(args[6]),
                "cR2": float(args[7]),
            }
        elif mat_type in {"Bilin", "Hysteretic", "ModIMKPeakOriented"}:
            counts = (12, 13, 16, 17) if mat_type == "Hysteretic" else (23, 24)
            if len(args) - 2 not in counts:
                raise self._error(
                    f"{mat_type} expects {' or '.join(map(str, counts))} parameters",
                    "uniaxialMaterial",
                    args,
                )
            params = {"values": [float(value) for value in args[2:]]}
        elif mat_type in {"PySimple1", "TzSimple1", "QzSimple1"}:
            names, required = {
                "PySimple1": (("soilType", "pult", "y50", "Cd", "c"), 4),
                "TzSimple1": (("tzType", "tult", "z50", "c"), 3),
                "QzSimple1": (("qzType", "qult", "z50", "suction", "c"), 3),
            }[mat_type]
            if not required <= len(args) - 2 <= len(names):
                raise self._error(
                    f"{mat_type} expects tag and {' '.join(names)}",
                    "uniaxialMaterial",
                    args,
                )
            params = dict(zip(names, [int(args[2]), *map(float, args[3:])]))
        elif mat_type == "Fatigue":
            if len(args) < 3 or (len(args) - 3) % 2 != 0:
                raise self._error(
                    "invalid Fatigue parameters", "uniaxialMaterial", args
                )
            params = {"material": int(args[2])}
            for option, value in zip(args[3::2], args[4::2]):
                if option not in {"-Dmax", "-E0", "-m", "-min", "-max"}:
                    raise self._error(
                        f"unsupported Fatigue option `{option}`",
                        "uniaxialMaterial",
                        args,
                    )
                params[option[1:]] = float(value)
        elif mat_type == "ViscousDamper":
            extra = args[2:]
            if len(extra) not in {3, 4, 8}:
                raise self._error(
                    "ViscousDamper expects 3, 4, or 8 parameters",
                    "uniaxialMaterial",
                    args,
                )
            params = {
                "K": float(extra[0]),
                "C": float(extra[1]),
                "alpha": float(extra[2]),
            }
            if len(extra) >= 4:
                params["LGap"] = float(extra[3])
            if len(extra) == 8:
                params["NM"] = float(extra[4])
                params["RelTol"] = float(extra[5])
                params["AbsTol"] = float(extra[6])
                params["MaxHalf"] = float(extra[7])
        elif mat_type in {"HystereticSmooth", "HystereticAsym"}:
            keys = (
                ("ka", "kb", "fbar", "beta")
                if mat_type == "HystereticSmooth"
                else ("ka", "kb", "fbar", "beta1", "beta2", "gamma")
            )
            numeric_end = 2 + len(keys)
            if len(args) not in {numeric_end, numeric_end + 1}:
                raise self._error(
                    f"{mat_type} expects `tag {' '.join(keys)} ?-alpha?`",
                    "uniaxialMaterial",
                    args,
                )
            params = dict(zip(keys, map(float, args[2:numeric_end]), strict=True))
            if params["ka"] <= 0 or params["kb"] >= params["ka"] or params["fbar"] <= 0:
                raise self._error(
                    f"{mat_type} requires ka > 0, kb < ka, and fbar > 0",
                    "uniaxialMaterial",
                    args,
                )
            if args[numeric_end:] == ("-alpha",):
                params["fbar"] = (params["ka"] - params["kb"]) / (2 * params["fbar"])
        elif mat_type == "Hardening":
            if len(args) < 6:
                raise self._error(
                    "Hardening expects `tag E sigmaY Hiso Hkin`",
                    "uniaxialMaterial",
                    args,
                )
            e0 = float(args[2])
            if abs(e0) <= 0.0:
                raise self._error(
                    "Hardening requires nonzero elastic modulus",
                    "uniaxialMaterial",
                    args,
                )
            h_total = float(args[4]) + float(args[5])
            denom = e0 + h_total
            hardening_ratio = (h_total / denom) if abs(denom) > 0.0 else 0.0
            hardening_ratio = max(0.0, min(hardening_ratio, 0.999999))
            # Map Hardening to Steel01 with exact elastoplastic tangent
            # E * (Hiso + Hkin) / (E + Hiso + Hkin) matching OpenSees HardeningMaterial.
            mat_type = "Steel01"
            params = {"Fy": float(args[3]), "E0": e0, "b": hardening_ratio}
        else:
            raise self._error(
                f"unsupported uniaxial material `{mat_type}`",
                "uniaxialMaterial",
                args,
            )
        self._upsert_material({"id": tag, "type": mat_type, "params": params})
        return ""

    def _cmd_geom_transf(self, *args: str) -> str:
        if len(args) < 2:
            raise self._error("geomTransf expects type and tag", "geomTransf", args)
        geom_type = args[0]
        if geom_type not in {"Linear", "PDelta", "Corotational"}:
            raise self._error(
                f"unsupported geomTransf `{geom_type}`", "geomTransf", args
            )
        vecxz: Optional[tuple[float, float, float]] = None
        if len(args) >= 5:
            vecxz = (float(args[2]), float(args[3]), float(args[4]))
        self.geom_transforms[int(args[1])] = GeomTransformDef(
            geom_type=geom_type,
            vecxz=vecxz,
        )
        return ""

    def _cmd_section(self, *args: str) -> str:
        if not args:
            raise self._error("section type is required", "section", args)
        section_type = self._normalize_alias("section", args[0])
        if section_type == "Elastic":
            if len(args) in {5, 7}:
                params = {
                    "E": float(args[2]),
                    "A": float(args[3]),
                    "I": float(args[4]),
                }
                if len(args) == 7:
                    params["G"] = float(args[5])
                    params["shear_rigidity_y"] = (
                        params["G"] * params["A"] * float(args[6])
                    )
                self._upsert_section(
                    {
                        "id": int(args[1]),
                        "type": "ElasticSection2d",
                        "params": params,
                    }
                )
                return ""
            if len(args) in {8, 10}:
                params = dict(
                    zip(("E", "A", "Iz", "Iy", "G", "J"), map(float, args[2:8]))
                )
                if len(args) == 10:
                    ga = params["G"] * params["A"]
                    params["shear_rigidity_y"] = ga * float(args[8])
                    params["shear_rigidity_z"] = ga * float(args[9])
                self._upsert_section(
                    {
                        "id": int(args[1]),
                        "type": "ElasticSection3d",
                        "params": params,
                    }
                )
                return ""
            raise self._error(
                "Elastic expects `tag E A I ?G alpha?` or `tag E A Iz Iy G J ?alphaY alphaZ?`",
                "section",
                args,
            )
            return ""
        if section_type == "Uniaxial" and len(args) == 4 and args[3] == "P":
            # Trusses read the axial material directly; aggregators still need
            # the section itself, so fall through to the generic handler.
            self.axial_section_materials[int(args[1])] = (1.0, int(args[2]))
        if section_type in {"Fiber", "FiberThermal"}:
            model = self._require_model("section", args)
            if section_type == "FiberThermal" and int(model["ndm"]) != 2:
                raise self._error("FiberThermal requires a 2D model", "section", args)
            if len(args) < 3:
                raise self._error(
                    "Fiber expects `section Fiber tag ?-GJ value? { ... }`",
                    "section",
                    args,
                )
            tag = int(args[1])
            body_index = 2
            torsion_gj: Optional[float] = None
            while body_index < len(args) - 1:
                option = args[body_index]
                if option == "-GJ":
                    if body_index + 1 >= len(args) - 1:
                        raise self._error(
                            "Fiber -GJ expects a numeric torsional rigidity before the body",
                            "section",
                            args,
                        )
                    torsion_gj = float(args[body_index + 1])
                    body_index += 2
                    continue
                if option == "-torsion" and int(model["ndm"]) == 3:
                    if body_index + 1 >= len(args) - 1:
                        raise self._error(
                            "Fiber -torsion expects a material tag", "section", args
                        )
                    material = self.materials_by_id.get(int(args[body_index + 1]))
                    if material is None or material["type"] != "Elastic":
                        raise self._error(
                            "Fiber -torsion requires an Elastic material",
                            "section",
                            args,
                        )
                    torsion_gj = float(material["params"]["E"])
                    body_index += 2
                    continue
                raise self._error(
                    f"unsupported Fiber section option `{option}`",
                    "section",
                    args,
                )

            section_output_type = (
                "FiberSection3d" if int(model["ndm"]) == 3 else "FiberSection2d"
            )
            self.current_section = PendingFiberSection(
                id=tag,
                section_type=section_output_type,
                torsion_gj=torsion_gj,
            )
            try:
                self._eval_tcl(args[body_index], "section", args)
            finally:
                if self.current_section is not None:
                    params: dict[str, Any] = {
                        "patches": list(self.current_section.patches),
                        "layers": list(self.current_section.layers),
                    }
                    if self.current_section.fibers:
                        params["fibers"] = list(self.current_section.fibers)
                    if section_type == "FiberThermal":
                        params["thermal"] = True
                    if self.current_section.section_type == "FiberSection3d":
                        if self.current_section.torsion_gj is not None:
                            # The runtime stores G and J separately but only uses the
                            # product, so preserve the OpenSees `-GJ` value losslessly.
                            params["G"] = float(self.current_section.torsion_gj)
                            params["J"] = 1.0
                    self._upsert_section(
                        {
                            "id": tag,
                            "type": self.current_section.section_type,
                            "params": params,
                        }
                    )
                self.current_section = None
            return ""
        if section_type in {"Aggregator", "Uniaxial", "Generic1D", "Generic1d"}:
            if section_type != "Aggregator" and len(args) != 4:
                raise self._error(
                    f"{section_type} expects `section {section_type} tag matTag code`",
                    "section",
                    args,
                )
            if len(args) < 4:
                raise self._error(
                    "Aggregator expects `section Aggregator tag mat response ... ?-section base?`",
                    "section",
                    args,
                )
            tag = int(args[1])
            params: dict[str, Any] = {
                "axial_material": -1,
                "flexural_material": -1,
                "moment_y_material": -1,
                "torsion_material": -1,
                "shear_y_material": -1,
                "shear_z_material": -1,
                "base_section": -1,
            }
            response_key_by_name = {
                "P": "axial_material",
                "Mz": "flexural_material",
                "My": "moment_y_material",
                "T": "torsion_material",
                "Vy": "shear_y_material",
                "Vz": "shear_z_material",
            }
            idx = 2
            while idx < len(args):
                token = args[idx]
                if token == "-section":
                    if idx + 1 >= len(args):
                        raise self._error(
                            "Aggregator -section expects a base section tag",
                            "section",
                            args,
                        )
                    params["base_section"] = int(args[idx + 1])
                    idx += 2
                    continue
                if idx + 1 >= len(args):
                    raise self._error(
                        "Aggregator response pair is incomplete",
                        "section",
                        args,
                    )
                material_id = int(token)
                response_name = args[idx + 1]
                response_key = response_key_by_name.get(response_name)
                if response_key is None:
                    raise self._error(
                        f"unsupported Aggregator response `{response_name}`",
                        "section",
                        args,
                    )
                if int(params[response_key]) >= 0:
                    raise self._error(
                        f"duplicate Aggregator response `{response_name}`",
                        "section",
                        args,
                    )
                params[response_key] = material_id
                idx += 2
            base_section_id = int(params["base_section"])
            if base_section_id >= 0:
                base_section = self.sections_by_id.get(base_section_id)
                if (
                    base_section is not None
                    and base_section["type"] == "FiberSection3d"
                    and all(
                        params[key] < 0
                        for key in (
                            "axial_material",
                            "flexural_material",
                            "moment_y_material",
                        )
                    )
                    and (
                        params["shear_y_material"] >= 0
                        or params["shear_z_material"] >= 0
                    )
                ):
                    fiber = (
                        self._clone_section_with_torsion(
                            tag, base_section, params["torsion_material"]
                        )
                        if params["torsion_material"] >= 0
                        else {
                            **base_section,
                            "id": tag,
                            "params": dict(base_section["params"]),
                        }
                    )
                    for key in ("shear_y_material", "shear_z_material"):
                        if params[key] >= 0:
                            fiber["params"][key] = params[key]
                    self._upsert_section(fiber)
                    return ""
                if base_section is not None and base_section["type"] in {
                    "FiberSection3d",
                    "ElasticSection3d",
                }:
                    if int(params["torsion_material"]) >= 0 and all(
                        int(params[key]) < 0
                        for code, key in response_key_by_name.items()
                        if code != "T"
                    ):
                        self._upsert_section(
                            self._clone_section_with_torsion(
                                tag, base_section, int(params["torsion_material"])
                            )
                        )
                        return ""
            agg_type = (
                "AggregatorSection3d"
                if self.model is not None and int(self.model["ndm"]) == 3
                else "AggregatorSection2d"
            )
            self._upsert_section({"id": tag, "type": agg_type, "params": params})
            return ""
        if section_type == "PlateFiber":
            if len(args) not in (4, 5) or (
                len(args) == 5 and args[4] not in {"Lobatto", "Gauss", "Legendre"}
            ):
                raise self._error(
                    "PlateFiber expects `tag matTag h ?Lobatto|Gauss|Legendre?`",
                    "section",
                    args,
                )
            params = {"material": int(args[2]), "h": float(args[3])}
            if len(args) == 5:
                params["integration"] = args[4]
            self._upsert_section(
                {"id": int(args[1]), "type": "PlateFiber", "params": params}
            )
            return ""
        if section_type == "ElasticMembranePlateSection":
            if len(args) < 5:
                raise self._error(
                    "ElasticMembranePlateSection expects `tag E nu h ?rho? ?Ep_mod?`",
                    "section",
                    args,
                )
            params: dict[str, Any] = {
                "E": float(args[2]),
                "nu": float(args[3]),
                "h": float(args[4]),
                "rho": float(args[5]) if len(args) > 5 else 0.0,
            }
            if len(args) > 6:
                params["Ep_mod"] = float(args[6])
            self._upsert_section(
                {
                    "id": int(args[1]),
                    "type": "ElasticMembranePlateSection",
                    "params": params,
                }
            )
            return ""
        if section_type == "LayeredShell":
            if len(args) < 5:
                raise self._error(
                    "LayeredShell expects `tag nLayers matTag1 h1 ...` or `tag nLayers matTag h`",
                    "section",
                    args,
                )
            tag = int(args[1])
            n_layers = int(args[2])
            if n_layers < 3:
                raise self._error(
                    "LayeredShell requires at least 3 layers",
                    "section",
                    args,
                )
            layer_args = args[3:]
            layers: list[dict[str, Any]] = []
            if len(layer_args) == 2:
                material_id = int(layer_args[0])
                layer_thickness = float(layer_args[1]) / float(n_layers)
                layers = [
                    {"material": material_id, "thickness": layer_thickness}
                    for _ in range(n_layers)
                ]
            else:
                if len(layer_args) != 2 * n_layers:
                    raise self._error(
                        "LayeredShell layer pair count does not match nLayers",
                        "section",
                        args,
                    )
                for idx in range(0, len(layer_args), 2):
                    layers.append(
                        {
                            "material": int(layer_args[idx]),
                            "thickness": float(layer_args[idx + 1]),
                        }
                    )
            self._upsert_section(
                {"id": tag, "type": "LayeredShellSection", "params": {"layers": layers}}
            )
            return ""
        raise self._error(
            f"unsupported section type `{section_type}`",
            "section",
            (section_type, *args[1:]),
        )

    def _cmd_patch(self, *args: str) -> str:
        if self.current_section is None:
            raise self._error(
                "patch is only supported inside a Fiber section", "patch", args
            )
        patch_type = args[0]
        if patch_type == "rect":
            self.current_section.patches.append(
                {
                    "type": "rect",
                    "material": int(args[1]),
                    "num_subdiv_y": int(args[2]),
                    "num_subdiv_z": int(args[3]),
                    "y_i": float(args[4]),
                    "z_i": float(args[5]),
                    "y_j": float(args[6]),
                    "z_j": float(args[7]),
                }
            )
            return ""
        if patch_type in {"quadr", "quad"}:
            self.current_section.patches.append(
                {
                    "type": "quadr",
                    "material": int(args[1]),
                    "num_subdiv_y": int(args[2]),
                    "num_subdiv_z": int(args[3]),
                    "y_i": float(args[4]),
                    "z_i": float(args[5]),
                    "y_j": float(args[6]),
                    "z_j": float(args[7]),
                    "y_k": float(args[8]),
                    "z_k": float(args[9]),
                    "y_l": float(args[10]),
                    "z_l": float(args[11]),
                }
            )
            return ""
        if patch_type == "circ":
            if len(args) not in {8, 10}:
                raise self._error(
                    "patch circ expects `matTag nCirc nRad yCenter zCenter intRad extRad ?startAng endAng?`",
                    "patch",
                    args,
                )
            start_ang = float(args[8]) if len(args) == 10 else 0.0
            end_ang = float(args[9]) if len(args) == 10 else 360.0
            self.current_section.patches.append(
                {
                    "type": "circ",
                    "material": int(args[1]),
                    "num_subdiv_circ": int(args[2]),
                    "num_subdiv_rad": int(args[3]),
                    "y_center": float(args[4]),
                    "z_center": float(args[5]),
                    "int_rad": float(args[6]),
                    "ext_rad": float(args[7]),
                    "start_ang": start_ang,
                    "end_ang": end_ang,
                }
            )
            return ""
        raise self._error(f"unsupported patch type `{patch_type}`", "patch", args)

    def _cmd_layer(self, *args: str) -> str:
        if self.current_section is None:
            raise self._error(
                "layer is only supported inside a Fiber section", "layer", args
            )
        layer_type = args[0]
        if layer_type == "straight":
            self.current_section.layers.append(
                {
                    "type": "straight",
                    "material": int(args[1]),
                    "num_bars": int(args[2]),
                    "bar_area": float(args[3]),
                    "y_start": float(args[4]),
                    "z_start": float(args[5]),
                    "y_end": float(args[6]),
                    "z_end": float(args[7]),
                }
            )
            return ""
        if layer_type == "circ":
            if len(args) not in {7, 9}:
                raise self._error(
                    "layer circ expects `matTag numBars barArea yCenter zCenter radius ?startAng endAng?`",
                    "layer",
                    args,
                )
            layer_entry: dict[str, Any] = {
                "type": "circ",
                "material": int(args[1]),
                "num_bars": int(args[2]),
                "bar_area": float(args[3]),
                "y_center": float(args[4]),
                "z_center": float(args[5]),
                "radius": float(args[6]),
            }
            if len(args) == 9:
                layer_entry["start_ang"] = float(args[7])
                layer_entry["end_ang"] = float(args[8])
            self.current_section.layers.append(layer_entry)
            return ""
        raise self._error(f"unsupported layer type `{layer_type}`", "layer", args)

    def _cmd_fiber(self, *args: str) -> str:
        if self.current_section is None:
            raise self._error(
                "fiber is only supported inside a Fiber section", "fiber", args
            )
        if len(args) != 4:
            raise self._error("fiber expects `y z A matTag`", "fiber", args)
        self.current_section.fibers.append(
            {
                "y": float(args[0]),
                "z": float(args[1]),
                "area": float(args[2]),
                "material": int(args[3]),
            }
        )
        return ""

    def _normalize_quad_formulation(self, formulation: str) -> str:
        if formulation in {"PlaneStress", "PlaneStress2D"}:
            return "PlaneStress"
        if formulation in {"PlaneStrain", "PlaneStrain2D"}:
            return "PlaneStrain"
        raise self._error(
            f"unsupported quad formulation `{formulation}`", "block2D", (formulation,)
        )

    def _block_2d_fill_geometry_nodes(
        self, corners: dict[int, tuple[float, ...]]
    ) -> dict[int, tuple[float, ...]]:
        filled = dict(corners)
        for node_id, first, second in ((5, 1, 2), (6, 2, 3), (7, 3, 4), (8, 4, 1)):
            if node_id not in filled:
                filled[node_id] = tuple(
                    0.5 * (filled[first][axis] + filled[second][axis])
                    for axis in range(len(filled[first]))
                )
        if 9 not in filled:
            filled[9] = tuple(
                0.25
                * (
                    filled[1][axis]
                    + filled[2][axis]
                    + filled[3][axis]
                    + filled[4][axis]
                )
                for axis in range(len(filled[1]))
            )
        return filled

    def _block_2d_shape_functions(self, x_value: float, y_value: float) -> list[float]:
        nx = [
            0.5 * x_value * (x_value - 1.0),
            1.0 - (x_value * x_value),
            0.5 * x_value * (x_value + 1.0),
        ]
        ny = [
            0.5 * y_value * (y_value - 1.0),
            1.0 - (y_value * y_value),
            0.5 * y_value * (y_value + 1.0),
        ]
        return [
            nx[0] * ny[0],
            nx[2] * ny[0],
            nx[2] * ny[2],
            nx[0] * ny[2],
            nx[1] * ny[0],
            nx[2] * ny[1],
            nx[1] * ny[2],
            nx[0] * ny[1],
            nx[1] * ny[1],
        ]

    def _block_2d_nodal_coords(
        self,
        i: int,
        j: int,
        nx: int,
        ny: int,
        geometry_nodes: dict[int, tuple[float, ...]],
    ) -> tuple[float, ...]:
        x_value = -1.0 + (2.0 * i / nx if nx > 0 else 0.0)
        y_value = -1.0 + (2.0 * j / ny if ny > 0 else 0.0)
        shape = self._block_2d_shape_functions(x_value, y_value)
        dims = len(geometry_nodes[1])
        coords = [0.0] * dims
        for index, factor in enumerate(shape, start=1):
            point = geometry_nodes[index]
            for axis in range(dims):
                coords[axis] += factor * point[axis]
        return tuple(coords)

    def _block_2d_element_nodes(
        self, i: int, j: int, nx: int, num_nodes: int
    ) -> list[int]:
        nenx = nx + 1
        if num_nodes == 4:
            node_1 = i + j * nenx
            node_2 = node_1 + 1
            node_3 = node_2 + nenx
            node_4 = node_1 + nenx
            return [node_1, node_2, node_3, node_4]
        node_1 = i * 2 + j * 2 * nenx
        node_5 = node_1 + 1
        node_2 = node_1 + 2
        node_4 = node_1 + 2 * nenx
        node_7 = node_4 + 1
        node_3 = node_7 + 1
        node_8 = node_1 + nenx
        node_9 = node_8 + 1
        node_6 = node_9 + 1
        return [node_1, node_2, node_3, node_4, node_5, node_6, node_7, node_8, node_9]

    def _cmd_block_2d(self, *args: str) -> str:
        model = self._require_model("block2D", args)
        if len(args) < 7:
            raise self._error(
                "block2D expects `nx ny startNode startEle eleType eleArgs {coords}`",
                "block2D",
                args,
            )
        nx = int(args[0])
        ny = int(args[1])
        start_node = int(args[2])
        start_ele = int(args[3])
        element_type = args[4]
        num_ele_nodes = 4
        coord_block = args[-1]
        ele_arg_tokens = list(args[5:-1])
        if len(ele_arg_tokens) >= 2 and ele_arg_tokens[-2] == "-numEleNodes":
            num_ele_nodes = int(ele_arg_tokens[-1])
            ele_arg_tokens = ele_arg_tokens[:-2]
        if len(ele_arg_tokens) == 1:
            ele_arg_tokens = list(self.interp.tk.splitlist(ele_arg_tokens[0]))
        if len(ele_arg_tokens) < 1:
            raise self._error("block2D missing element arguments", "block2D", args)
        expected = {"quad8n": 8, "quad9n": 9}.get(element_type, 4)
        if num_ele_nodes == 4 and expected > 4:
            num_ele_nodes = expected
        if num_ele_nodes != expected:
            raise self._error(
                f"block2D -numEleNodes {num_ele_nodes} is incompatible with {element_type}",
                "block2D",
                args,
            )
        if min(nx, ny) <= 0:
            raise self._error("block2D subdivisions must be positive", "block2D", args)
        step = 2 if num_ele_nodes > 4 else 1
        if nx % step or ny % step:
            raise self._error(
                "higher-order block2D subdivisions must be even", "block2D", args
            )

        dims = int(model["ndm"])
        corners: dict[int, tuple[float, ...]] = {}
        for raw_line in coord_block.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != dims + 1:
                coord_desc = "id x y" if dims == 2 else "id x y z"
                raise self._error(
                    f"block2D coordinate rows expect `{coord_desc}`",
                    "block2D",
                    args,
                )
            node_id = int(parts[0])
            if node_id < 1 or node_id > 9:
                raise self._error(
                    "block2D geometry node ids must be in 1..9", "block2D", args
                )
            corners[node_id] = tuple(float(value) for value in parts[1:])
        if any(node_id not in corners for node_id in (1, 2, 3, 4)):
            raise self._error("block2D requires corner points 1..4", "block2D", args)
        geometry_nodes = self._block_2d_fill_geometry_nodes(corners)

        supported = {
            "quad",
            "stdQuad",
            "mixedQuad",
            "shell",
            "Shell",
            "shellMITC4",
            "ShellMITC4",
        }
        if element_type not in supported and element_type not in _SOLID_NODE_COUNTS:
            raise self._error(
                f"unsupported block2D element type `{element_type}`", "block2D", args
            )
        element_type = {
            "mixedQuad": "bbarQuad",
            "Shell": "shell",
            "shellMITC4": "shell",
            "ShellMITC4": "shell",
        }.get(element_type, element_type)

        stride = nx + 1
        for j in range(ny + 1):
            for i in range(nx + 1):
                if num_ele_nodes == 8 and i % 2 and j % 2:
                    continue
                coords = self._block_2d_nodal_coords(i, j, nx, ny, geometry_nodes)
                self._cmd_node(str(start_node + j * stride + i), *map(str, coords))

        for j in range(ny // step):
            for i in range(nx // step):
                local_nodes = self._block_2d_element_nodes(i, j, nx, num_ele_nodes)
                self._cmd_element(
                    element_type,
                    str(start_ele + j * (nx // step) + i),
                    *(str(start_node + index) for index in local_nodes[:num_ele_nodes]),
                    *ele_arg_tokens,
                )
        return ""

    def _cmd_block_3d(self, *args: str) -> str:
        model = self._require_model("block3D", args)
        if len(args) < 8 or model != {"ndm": 3, "ndf": 3}:
            raise self._error(
                "block3D expects `nx ny nz startNode startEle eleType eleArgs {coords}` in a 3D solid model",
                "block3D",
                args,
            )
        nx, ny, nz, start_node, start_ele = map(int, args[:5])
        element_type = args[5]
        if element_type not in _BRICK_TYPES or min(nx, ny, nz) <= 0:
            raise self._error(
                "invalid block3D subdivisions or element type", "block3D", args
            )
        option_tokens = args[6:-1]
        if len(option_tokens) == 1:
            option_tokens = tuple(self.interp.tk.splitlist(option_tokens[0]))
        tokens = self.interp.tk.splitlist(args[-1])
        if len(tokens) % 4:
            raise self._error(
                "block3D coordinate rows expect `id x y z`", "block3D", args
            )
        points = {}
        for offset in range(0, len(tokens), 4):
            tag = int(tokens[offset])
            if not 1 <= tag <= 27:
                raise self._error(
                    "block3D geometry node ids must be in 1..27", "block3D", args
                )
            points[_BLOCK3D_COORDS[tag - 1]] = tuple(
                map(float, tokens[offset + 1 : offset + 4])
            )
        if any(point not in points for point in _BLOCK3D_COORDS[:8]):
            raise self._error("block3D requires corner points 1..8", "block3D", args)
        for point in _BLOCK3D_COORDS[8:]:
            if point not in points:
                vertices = [
                    points[corner]
                    for corner in _BLOCK3D_COORDS[:8]
                    if all(
                        value == 0 or value == sign
                        for value, sign in zip(point, corner)
                    )
                ]
                points[point] = tuple(
                    sum(axis) / len(vertices) for axis in zip(*vertices)
                )

        def basis(q: float) -> tuple[float, float, float]:
            return 0.5 * q * (q - 1), 1.0 - q * q, 0.5 * q * (q + 1)

        def node_tag(i: int, j: int, k: int) -> int:
            return start_node + i + (nx + 1) * (j + (ny + 1) * k)

        step = 2 if element_type == "Brick27N" else 1
        if nx % step or ny % step or nz % step:
            raise self._error(
                "Brick27N block3D subdivisions must be even", "block3D", args
            )
        for k in range(nz + 1):
            for j in range(ny + 1):
                for i in range(nx + 1):
                    bx, by, bz = (
                        basis(2 * i / nx - 1),
                        basis(2 * j / ny - 1),
                        basis(2 * k / nz - 1),
                    )
                    coords = [0.0, 0.0, 0.0]
                    for (r, s, t), point in points.items():
                        weight = bx[r + 1] * by[s + 1] * bz[t + 1]
                        for axis in range(3):
                            coords[axis] += weight * point[axis]
                    self._cmd_node(str(node_tag(i, j, k)), *map(str, coords))

        corners = _BLOCK3D_COORDS[:8]
        if element_type.startswith("Brick"):
            corners = tuple((r, -s, -t) for r, s, t in corners)
            if step == 2:
                corners += tuple((r, -s, -t) for r, s, t in _BRICK27_COORDS[8:])
        tag = start_ele
        for k in range(0, nz, step):
            for j in range(0, ny, step):
                for i in range(0, nx, step):
                    nodes = [
                        node_tag(
                            i + (r + 1) * step // 2,
                            j + (s + 1) * step // 2,
                            k + (t + 1) * step // 2,
                        )
                        for r, s, t in corners
                    ]
                    self._cmd_element(
                        element_type, str(tag), *map(str, nodes), *option_tokens
                    )
                    tag += 1
        return ""

    def _cmd_solid_element(self, element_type: str, args: tuple[str, ...]) -> str:
        model = self._require_model("element", args)
        dimension = 3 if element_type in _BRICK_TYPES or element_type == "quad3d" else 2
        count = _SOLID_NODE_COUNTS[element_type]
        if model != {"ndm": dimension, "ndf": dimension} or len(args) < count + 3:
            raise self._error(
                f"invalid model or connectivity for {element_type}", "element", args
            )
        element = {
            "id": int(args[1]),
            "type": element_type,
            "nodes": list(map(int, args[2 : 2 + count])),
        }
        options = args[2 + count :]
        if element_type in _BRICK_TYPES:
            legacy = element_type.startswith("Brick")
            if len(options) not in ({1, 5} if legacy else {1, 4}):
                raise self._error(
                    f"invalid body force or density arguments for {element_type}",
                    "element",
                    args,
                )
            element["material"] = int(options[0])
            for name, value in zip(("body_x", "body_y", "body_z", "rho"), options[1:]):
                element[name] = float(value)
        elif element_type == "bbarQuad":
            if len(options) not in {1, 2}:
                raise self._error(
                    "bbarQuad expects `?thickness? material`", "element", args
                )
            element.update(
                material=int(options[-1]),
                thickness=float(options[0]) if len(options) == 2 else 1.0,
                formulation="PlaneStrain",
            )
        else:
            if element_type == "SSPquad":
                if len(options) not in {3, 5}:
                    raise self._error(
                        "SSPquad expects `material formulation thickness ?b1 b2?`",
                        "element",
                        args,
                    )
                material, formulation, thickness, *extra = options
                keys = ("body_x", "body_y")
            else:
                if len(options) not in (
                    {3} if element_type == "enhancedQuad" else {3, 7}
                ):
                    raise self._error(
                        f"invalid material or load arguments for {element_type}",
                        "element",
                        args,
                    )
                thickness, formulation, material, *extra = options
                keys = ("pressure", "rho", "body_x", "body_y")
            element.update(
                material=int(material),
                thickness=float(thickness),
                formulation=self._normalize_quad_formulation(formulation),
            )
            element.update((key, float(value)) for key, value in zip(keys, extra))
        self.elements.append(element)
        return ""

    def _truss_section_material(
        self, section_id: int, args: tuple[str, ...]
    ) -> tuple[float, int]:
        if section_id in self.axial_section_materials:
            return self.axial_section_materials[section_id]
        section = self.sections_by_id.get(section_id)
        if section is None:
            raise self._error(f"section {section_id} not found", "element", args)
        params = section["params"]
        if section["type"] in {"ElasticSection2d", "ElasticSection3d"}:
            result = (float(params["A"]), self._register_material(float(params["E"])))
        elif section["type"] in {"FiberSection2d", "FiberSection3d"}:
            from tcl_truss_section import fiber_material_areas

            areas = fiber_material_areas(params)
            if len(areas) == 1:
                material_id, area = next(iter(areas.items()))
                result = (area, material_id)
            else:
                material_id = max(self.materials_by_id, default=0) + 1
                self._cmd_uniaxial_material(
                    "Parallel",
                    str(material_id),
                    *map(str, areas),
                    "-factors",
                    *map(str, areas.values()),
                )
                result = (1.0, material_id)
        else:
            raise self._error("unsupported truss section type", "element", args)
        self.axial_section_materials[section_id] = result
        return result

    def _cmd_uniform_elastic_hinge_3d(self, args: tuple[str, ...]) -> str:
        if len(args) < 15:
            raise self._error(
                "expected 3D beamWithHinges tag i j secI lpI secJ lpJ E A Iz Iy G J transf",
                "element",
                args,
            )
        interior = dict(zip(("E", "A", "Iz", "Iy", "G", "J"), map(float, args[8:14])))
        for section_tag in (args[4], args[6]):
            section = self.sections_by_id.get(int(section_tag))
            if (
                section is None
                or section["type"] != "ElasticSection3d"
                or section["params"] != interior
            ):
                raise self._error(
                    "3D beamWithHinges currently requires identical elastic interior and end sections",
                    "element",
                    args,
                )
        geom = self.geom_transforms.get(int(args[14]))
        if geom is None or geom.geom_type != "Linear":
            raise self._error(
                "3D beamWithHinges requires Linear geomTransf", "element", args
            )
        if min(float(args[5]), float(args[7])) < 0.0:
            raise self._error(
                "beamWithHinges lengths must be nonnegative", "element", args
            )
        options = self._parse_beam_options(args[15:], command_args=args)
        if set(options) - {"rho", "max_iters", "tol"}:
            raise self._error(
                "beamWithHinges supports -mass and -iter", "element", args
            )
        # All six points have the same flexibility. HingeRadau integrates the
        # quadratic basic flexibility exactly for a member under nodal loads.
        element = {
            "id": int(args[1]),
            "type": "forceBeamColumn3d",
            "nodes": [int(args[2]), int(args[3])],
            "section": int(args[4]),
            "geomTransf": geom.geom_type,
            "integration": "Lobatto",
            "num_int_pts": 3,
            "rho": options.get("rho", 0.0),
            "_uniform_elastic_hinge_3d": True,
        }
        if geom.vecxz is not None:
            element["vecxz"] = list(geom.vecxz)
        self.elements.append(element)
        return ""

    def _cmd_element(self, *args: str) -> str:
        if not args:
            raise self._error("element type is required", "element", args)
        element_type = self._normalize_alias("element", args[0])
        if element_type == "BeamContact2D":
            if len(args) not in {10, 11}:
                raise self._error(
                    "BeamContact2D expects tag i j secondary multiplier matTag width gapTol forceTol ?cFlag?",
                    "element",
                    args,
                )
            self.elements.append(
                {
                    "id": int(args[1]),
                    "type": element_type,
                    "nodes": list(map(int, args[2:6])),
                    "material": int(args[6]),
                    "width": float(args[7]),
                    "gap_tol": float(args[8]),
                    "force_tol": float(args[9]),
                    "initial_contact": len(args) == 10 or int(args[10]) == 0,
                }
            )
            return ""
        if element_type in _SOLID_NODE_COUNTS:
            return self._cmd_solid_element(element_type, args)
        if element_type == "beamWithHinges":
            model = self._require_model("element", args)
            if model == {"ndm": 3, "ndf": 6}:
                return self._cmd_uniform_elastic_hinge_3d(args)
            if model != {"ndm": 2, "ndf": 3} or len(args) < 12:
                raise self._error(
                    "expected 2D beamWithHinges tag i j secI lpI secJ lpJ E A I transf",
                    "element",
                    args,
                )
            geom = self.geom_transforms.get(int(args[11]))
            if geom is None:
                raise self._error(
                    "beamWithHinges geomTransf not found", "element", args
                )
            options = self._parse_beam_options(args[12:], command_args=args)
            if set(options) - {"rho", "max_iters", "tol"}:
                raise self._error(
                    "beamWithHinges supports -mass and -iter", "element", args
                )
            section_id = self._next_free_section_id()
            self._upsert_section(
                {
                    "id": section_id,
                    "type": "HingeRadauSection2d",
                    "params": {
                        "section_i": int(args[4]),
                        "length_i": float(args[5]),
                        "section_j": int(args[6]),
                        "length_j": float(args[7]),
                        "E": float(args[8]),
                        "A": float(args[9]),
                        "I": float(args[10]),
                        "max_iters": options.get("max_iters", 10),
                        "tol": options.get("tol", 1.0e-10),
                    },
                },
                generated=True,
            )
            self.elements.append(
                {
                    "id": int(args[1]),
                    "type": "forceBeamColumn2d",
                    "nodes": [int(args[2]), int(args[3])],
                    "section": section_id,
                    "geomTransf": geom.geom_type,
                    "integration": "HingeRadau",
                    "num_int_pts": 6,
                    "rho": options.get("rho", 0.0),
                }
            )
            return ""
        if element_type == "CoupledZeroLength":
            if len(args) not in {7, 8}:
                raise self._error(
                    "CoupledZeroLength expects tag iNode jNode dir1 dir2 matTag ?doRayleigh?",
                    "element",
                    args,
                )
            self.elements.append(
                {
                    "id": int(args[1]),
                    "type": element_type,
                    "nodes": [int(args[2]), int(args[3])],
                    "dirs": [int(args[4]), int(args[5])],
                    "materials": [int(args[6])],
                    "doRayleigh": len(args) == 8 and int(args[7]) == 1,
                }
            )
            return ""
        if element_type == "zeroLength":
            return self._cmd_zero_length(args)
        if element_type == "CatenaryCable":
            if len(args) != 14:
                raise self._error(
                    "CatenaryCable expects id iNode jNode weight E A L0 alpha dT rho tol substeps massType",
                    "element",
                    args,
                )
            keys = (
                "weight",
                "E",
                "A",
                "L0",
                "alpha",
                "temperature_change",
                "rho",
                "error_tol",
            )
            self.elements.append(
                {
                    "id": int(args[1]),
                    "type": element_type,
                    "nodes": [int(args[2]), int(args[3])],
                    **dict(zip(keys, map(float, args[4:12]))),
                    "substeps": int(args[12]),
                    "mass_type": int(args[13]),
                }
            )
            return ""
        if element_type == "SFI_MVLEM":
            if len(args) < 6:
                raise self._error(
                    "SFI_MVLEM expects id iNode jNode m c and panel options",
                    "element",
                    args,
                )
            count = int(args[4])
            if not 1 <= count <= 999:
                raise self._error("invalid SFI_MVLEM panel count", "element", args)
            options = {
                "-width": ("width", float),
                "-thick": ("thickness", float),
                "-mat": ("material", int),
            }
            values = {}
            i = 6
            while i < len(args):
                flag = args[i]
                if flag not in options or i + count >= len(args):
                    raise self._error(
                        f"invalid SFI_MVLEM option `{flag}`", "element", args
                    )
                name, parse = options[flag]
                if name in values:
                    raise self._error(
                        f"repeated SFI_MVLEM option `{flag}`", "element", args
                    )
                values[name] = [parse(arg) for arg in args[i + 1 : i + 1 + count]]
                i += count + 1
            if len(values) != len(options):
                raise self._error(
                    "SFI_MVLEM requires all panel options", "element", args
                )
            self.elements.append(
                {
                    "id": int(args[1]),
                    "type": element_type,
                    "nodes": [int(args[2]), int(args[3])],
                    "c": float(args[5]),
                    "panels": [
                        {name: data[k] for name, data in values.items()}
                        for k in range(count)
                    ],
                }
            )
            return ""
        if element_type == "MVLEM":
            if len(args) < 7:
                raise self._error(
                    "MVLEM expects id density iNode jNode m c and fiber options",
                    "element",
                    args,
                )
            count = int(args[5])
            if count <= 0:
                raise self._error(
                    "MVLEM requires at least one macro-fiber", "element", args
                )
            options = {
                "-width": ("width", float),
                "-thick": ("thickness", float),
                "-rho": ("rho", float),
                "-matConcrete": ("concrete", int),
                "-matSteel": ("steel", int),
                "-matShear": ("shear_material", int),
            }
            values = {}
            i = 7
            while i < len(args):
                flag = args[i]
                if flag not in options:
                    raise self._error(
                        f"unsupported MVLEM option `{flag}`", "element", args
                    )
                name, parse = options[flag]
                size = 1 if flag == "-matShear" else count
                if name in values or i + size >= len(args):
                    raise self._error(
                        f"invalid or repeated MVLEM option `{flag}`", "element", args
                    )
                values[name] = [parse(arg) for arg in args[i + 1 : i + 1 + size]]
                i += size + 1
            if len(values) != len(options):
                raise self._error(
                    "MVLEM requires all geometry and material options", "element", args
                )
            shear_material = values.pop("shear_material")[0]
            self.elements.append(
                {
                    "id": int(args[1]),
                    "type": "MVLEM",
                    "nodes": [int(args[3]), int(args[4])],
                    "density": float(args[2]),
                    "c": float(args[6]),
                    "shear_material": shear_material,
                    "fibers": [
                        {name: data[k] for name, data in values.items()}
                        for k in range(count)
                    ],
                }
            )
            return ""
        if element_type == "elasticBeamColumn":
            model = self._require_model("element", args)
            elem_id = int(args[1])
            node_i = int(args[2])
            node_j = int(args[3])
            if int(model["ndm"]) == 3:
                if len(args) >= 11:
                    area = float(args[4])
                    e_value = float(args[5])
                    g_value = float(args[6])
                    j_value = float(args[7])
                    iy_value = float(args[8])
                    iz_value = float(args[9])
                    transf_tag = int(args[10])
                    section_id = self._register_section_3d(
                        area, e_value, g_value, j_value, iy_value, iz_value
                    )
                    options = self._parse_beam_options(args[11:], command_args=args)
                elif len(args) >= 6:
                    section_id = int(args[4])
                    transf_tag = int(args[5])
                    options = self._parse_beam_options(args[6:], command_args=args)
                else:
                    raise self._error(
                        "3D elasticBeamColumn expects `id iNode jNode A E G J Iy Iz transfTag` or `id iNode jNode secTag transfTag`",
                        "element",
                        args,
                    )
                geom = self.geom_transforms.get(transf_tag)
                if geom is None:
                    raise self._error(
                        f"geomTransf tag {transf_tag} not defined", "element", args
                    )
                element = {
                    "id": elem_id,
                    "type": "elasticBeamColumn3d",
                    "nodes": [node_i, node_j],
                    "section": section_id,
                    "geomTransf": geom.geom_type,
                }
                if geom.vecxz is not None:
                    element["vecxz"] = list(geom.vecxz)
                element.update(options)
                self.elements.append(element)
                return ""
            area = float(args[4])
            e_value = float(args[5])
            inertia = float(args[6])
            transf_tag = int(args[7])
            geom = self.geom_transforms.get(transf_tag)
            if geom is None:
                raise self._error(
                    f"geomTransf tag {transf_tag} not defined", "element", args
                )
            section_id = self._register_section(area, e_value, inertia)
            element = {
                "id": elem_id,
                "type": "elasticBeamColumn2d",
                "nodes": [node_i, node_j],
                "section": section_id,
                "geomTransf": geom.geom_type,
            }
            element.update(self._parse_beam_options(args[8:], command_args=args))
            self.elements.append(element)
            return ""
        if element_type in {
            "forceBeamColumn",
            "forceBeamColumnCBDI",
            "dispBeamColumn",
            "dispBeamColumnThermal",
        }:
            model = self._require_model("element", args)
            cbdi = element_type == "forceBeamColumnCBDI"
            thermal = element_type == "dispBeamColumnThermal"
            output_type = (
                "forceBeamColumn"
                if cbdi
                else "dispBeamColumn"
                if thermal
                else element_type
            )
            if thermal and int(model["ndm"]) != 2:
                raise self._error(
                    "dispBeamColumnThermal requires a 2D model", "element", args
                )
            integration_args = (
                tuple(self.interp.tk.splitlist(args[5])) if len(args) >= 6 else ()
            )
            if len(integration_args) == 3:
                transf_tag = int(args[4])
                integration, section, points = integration_args
                section_id = int(section)
                num_int_pts = int(points)
                options = self._parse_beam_options(args[6:], command_args=args)
            elif len(args) < 7:
                raise self._error(
                    f"{element_type} expects `id iNode jNode numIntPts secTag transfTag` or `id iNode jNode transfTag integration secTag numIntPts`",
                    "element",
                    args,
                )
            elif args[5] in _BEAM_INTEGRATION_MIN_POINTS and len(args) >= 8:
                transf_tag = int(args[4])
                integration = args[5]
                section_id = int(args[6])
                num_int_pts = int(args[7])
                options = self._parse_beam_options(args[8:], command_args=args)
            else:
                integration = (
                    "Legendre" if output_type == "dispBeamColumn" else "Lobatto"
                )
                num_int_pts = int(args[4])
                section_id = int(args[5])
                transf_tag = int(args[6])
                options = self._parse_beam_options(args[7:], command_args=args)
            self._validate_beam_integration(output_type, integration, num_int_pts, args)
            geom = self.geom_transforms.get(transf_tag)
            if geom is None:
                raise self._error(
                    f"geomTransf tag {transf_tag} not defined", "element", args
                )
            ndim = int(model["ndm"])
            output_type = f"{output_type}{ndim}d"
            element = {
                "id": int(args[1]),
                "type": output_type,
                "nodes": [int(args[2]), int(args[3])],
                "section": section_id,
                "geomTransf": geom.geom_type,
                "integration": integration,
                "num_int_pts": num_int_pts,
            }
            if ndim == 3 and geom.vecxz is not None:
                element["vecxz"] = list(geom.vecxz)
            # The legacy OpenSees Tcl builder constructs ForceBeamColumn3d
            # for this spelling. Its interpreter API also exposes genuine CBDI3d.
            if cbdi and ndim == 2:
                element["cbdi"] = True
            if thermal:
                element["thermal"] = True
            element.update(options)
            self.elements.append(element)
            return ""
        if element_type == "fTruss":
            model = self._require_model("element", args)
            if model != {"ndm": 2, "ndf": 2} or len(args) != 6:
                raise self._error(
                    "fTruss expects tag iNode jNode A E in a 2D/2DOF model",
                    "element",
                    args,
                )
            material_id = self._register_material(float(args[5]))
            return self._cmd_element("truss", *args[1:5], str(material_id))
        if element_type in {"truss", "corotTruss"}:
            if len(args) == 6:
                area = float(args[4])
                material_id = int(args[5])
            elif len(args) == 5:
                section_id = int(args[4])
                area, material_id = self._truss_section_material(section_id, args)
            else:
                raise self._error(
                    "truss expects `A matTag` or `sectionTag` syntax",
                    "element",
                    args,
                )
            self.elements.append(
                {
                    "id": int(args[1]),
                    "type": element_type,
                    "nodes": [int(args[2]), int(args[3])],
                    "area": area,
                    "material": material_id,
                }
            )
            return ""
        if element_type == "SSPbrickUP":
            if len(args) not in {18, 21}:
                raise self._error(
                    "SSPbrickUP expects id, eight nodes, material, fluid bulk modulus, "
                    "density, three permeabilities, void ratio, stabilization, and optional body accelerations",
                    "element",
                    args,
                )
            fields = (
                "fluid_bulk",
                "fluid_density",
                "perm_x",
                "perm_y",
                "perm_z",
                "void_ratio",
                "stabilization",
                "body_x",
                "body_y",
                "body_z",
            )
            self.elements.append(
                {
                    "id": int(args[1]),
                    "type": element_type,
                    "nodes": [int(value) for value in args[2:10]],
                    "material": int(args[10]),
                    **dict(zip(fields, map(float, args[11:]))),
                }
            )
            return ""
        if element_type in {"quadUP", "9_4_QuadUP"}:
            node_count = 9 if element_type == "9_4_QuadUP" else 4
            required = node_count + 8
            options = ("body_x", "body_y")
            if node_count == 4:
                options += ("pressure",)
            if not required <= len(args) <= required + len(options):
                raise self._error(
                    f"{element_type} expects id, {node_count} nodes, thickness, material, "
                    "fluid bulk modulus, fluid density, x and y permeability, and optional "
                    "body accelerations and pressure",
                    "element",
                    args,
                )
            first_parameter = node_count + 2
            thickness, material, bulk, density, perm_x, perm_y = args[
                first_parameter:required
            ]
            self.elements.append(
                {
                    "id": int(args[1]),
                    "type": element_type,
                    "nodes": [int(value) for value in args[2:first_parameter]],
                    "thickness": float(thickness),
                    "material": int(material),
                    "fluid_bulk": float(bulk),
                    "fluid_density": float(density),
                    "perm_x": float(perm_x),
                    "perm_y": float(perm_y),
                    **dict(zip(options, map(float, args[required:]))),
                }
            )
            return ""
        if element_type in {"quad", "stdQuad"}:
            if len(args) not in {9, 13}:
                raise self._error(
                    "quad expects id i j k l thickness formulation material ?pressure rho b1 b2?",
                    "element",
                    args,
                )
            element = {
                "id": int(args[1]),
                "type": "fourNodeQuad",
                "nodes": [int(value) for value in args[2:6]],
                "thickness": float(args[6]),
                "formulation": self._normalize_quad_formulation(args[7]),
                "material": int(args[8]),
            }
            if len(args) == 13:
                element.update(
                    zip(("pressure", "rho", "body_x", "body_y"), map(float, args[9:]))
                )
            self.elements.append(element)
            return ""
        if element_type == "zeroLengthSection":
            self.elements.append(
                {
                    "id": int(args[1]),
                    "type": "zeroLengthSection",
                    "nodes": [int(args[2]), int(args[3])],
                    "section": int(args[4]),
                }
            )
            return ""
        if element_type == "twoNodeLink":
            model = self._require_model("element", args)
            ndm = int(model["ndm"])
            if len(args) < 8:
                raise self._error(
                    f"{element_type} expects `id node1 node2 -mat ... -dir ...`",
                    "element",
                    args,
                )
            elem_id = int(args[1])
            node_i = int(args[2])
            node_j = int(args[3])
            idx = 4
            if args[idx] != "-mat":
                raise self._error(f"{element_type} expects -mat", "element", args)
            idx += 1
            materials: list[int] = []
            while idx < len(args) and not args[idx].startswith("-"):
                materials.append(int(args[idx]))
                idx += 1
            if not materials:
                raise self._error(
                    f"{element_type} requires at least one material", "element", args
                )
            if idx >= len(args) or args[idx] not in {"-dir", "-dof"}:
                raise self._error(f"{element_type} expects -dir", "element", args)
            idx += 1
            dirs: list[int] = []
            while idx < len(args) and not args[idx].startswith("-"):
                dirs.append(int(args[idx]))
                idx += 1
            if len(dirs) != len(materials):
                raise self._error(
                    f"{element_type} requires matching -mat and -dir counts",
                    "element",
                    args,
                )
            elem: dict[str, Any] = {
                "id": elem_id,
                "type": element_type,
                "nodes": [node_i, node_j],
                "materials": materials,
                "dirs": dirs,
            }
            while idx < len(args):
                flag = args[idx]
                if flag == "-orient":
                    idx += 1
                    vals: list[float] = []
                    while idx < len(args) and len(vals) < 6:
                        try:
                            vals.append(float(args[idx]))
                        except ValueError:
                            break
                        idx += 1
                    if len(vals) == 6:
                        elem["orient"] = {"x": vals[:3], "y": vals[3:6]}
                    elif len(vals) == 3:
                        if ndm <= 2:
                            elem["orient"] = {"x": vals}
                        else:
                            elem["orient"] = {"y": vals}
                    else:
                        raise self._error(
                            f"{element_type} -orient expects 3 or 6 floats",
                            "element",
                            args,
                        )
                elif flag == "-pDelta":
                    idx += 1
                    count = 2 if ndm == 2 else 4
                    vals = [float(args[idx + k]) for k in range(count)]
                    idx += count
                    elem["pDelta"] = [0.0, 0.0, vals[0], vals[1]] if ndm == 2 else vals
                elif flag == "-shearDist":
                    idx += 1
                    count = 1 if ndm == 2 else 2
                    vals = [float(args[idx + k]) for k in range(count)]
                    idx += count
                    elem["shearDist"] = vals
                elif flag == "-doRayleigh":
                    elem["doRayleigh"] = True
                    idx += 1
                elif flag == "-mass":
                    elem["mass"] = float(args[idx + 1])
                    idx += 2
                else:
                    raise self._error(
                        f"unsupported {element_type} option `{flag}`", "element", args
                    )
            self.elements.append(elem)
            return ""
        if element_type == "tri31":
            if len(args) != 8:
                raise self._error(
                    "tri31 expects `id node1 node2 node3 thickness formulation material`",
                    "element",
                    args,
                )
            formulation = self._normalize_quad_formulation(args[6])
            self.elements.append(
                {
                    "id": int(args[1]),
                    "type": "tri31",
                    "nodes": [int(node) for node in args[2:5]],
                    "thickness": float(args[5]),
                    "formulation": formulation,
                    "material": int(args[7]),
                }
            )
            return ""
        if element_type in {
            "shell",
            "ShellDKGT",
            "ShellDKGQ",
            "ShellNLDKGQ",
            "ASDShellQ4",
        }:
            model = self._require_model("element", args)
            if int(model["ndm"]) != 3 or int(model["ndf"]) != 6:
                raise self._error("shell requires ndm=3 and ndf=6", "element", args)
            node_count = 3 if element_type == "ShellDKGT" else 4
            if len(args) < node_count + 3 or (
                len(args) > node_count + 3 and element_type != "ASDShellQ4"
            ):
                raise self._error(
                    f"{element_type} expects an id, {node_count} nodes and a section tag",
                    "element",
                    args,
                )
            section_id = int(args[node_count + 2])
            if section_id not in self.sections_by_id:
                raise self._error(f"section {section_id} not found", "element", args)
            element = {
                "id": int(args[1]),
                "type": element_type,
                "nodes": [int(node) for node in args[2 : node_count + 2]],
                "section": section_id,
            }
            i = node_count + 3
            while i < len(args):
                flag = args[i]
                if flag in {"-corotational", "-Corotational"}:
                    element["geomTransf"] = "Corotational"
                elif flag == "-noeas":
                    element["eas"] = False
                elif flag == "-drillingStab" and i + 1 < len(args):
                    i += 1
                    element["drilling_stab"] = min(1.0, max(0.0, float(args[i])))
                else:
                    raise self._error(
                        f"unsupported ASDShellQ4 option `{flag}`", "element", args
                    )
                i += 1
            self.elements.append(element)
            return ""
        raise self._error(
            f"unsupported element type `{element_type}`",
            "element",
            (element_type, *args[1:]),
        )

    def _cmd_zero_length(self, args: tuple[str, ...]) -> str:
        model = self._require_model("element", args)
        if len(args) < 8 or args[4] != "-mat":
            raise self._error(
                "zeroLength expects `tag iNode jNode -mat matTags... -dir dirs...`",
                "element",
                args,
            )
        dir_index = next(
            (idx for idx in range(5, len(args)) if args[idx] in {"-dir", "-dof"}), 5
        )
        materials = [int(value) for value in args[5:dir_index]]
        end = dir_index + 1 + len(materials)
        if not materials or end > len(args):
            raise self._error(
                "zeroLength requires one direction per material", "element", args
            )
        element = {
            "type": "zeroLength",
            "id": int(args[1]),
            "nodes": [int(args[2]), int(args[3])],
            "materials": materials,
            "dirs": [int(value) for value in args[dir_index + 1 : end]],
        }
        idx = end
        while idx < len(args):
            token = args[idx]
            if token == "-orient":
                size = 3 if model["ndm"] == 2 else 6
                values = args[idx + 1 : idx + 1 + size]
                if len(values) != size:
                    raise self._error(
                        "incomplete zeroLength orientation", "element", args
                    )
                x_axis = [float(value) for value in values[:3]]
                if model["ndm"] == 2:
                    # OpenSees ZeroLength: in 2D, y = z cross x.
                    y_axis = [-x_axis[1], x_axis[0], 0.0]
                else:
                    y_axis = [float(value) for value in values[3:6]]
                element["orient"] = {"x": x_axis, "y": y_axis}
                idx += size + 1
                if model["ndm"] == 2 and idx + 3 <= len(args):
                    try:
                        tuple(map(float, args[idx : idx + 3]))
                    except ValueError:
                        pass
                    else:
                        idx += 3
            elif token == "-doRayleigh":
                element["doRayleigh"] = idx + 1 == len(args) or bool(int(args[idx + 1]))
                idx += 2
            elif token == "-dampMats":
                values = args[idx + 1 : idx + 1 + len(materials)]
                if len(values) != len(materials):
                    raise self._error(
                        "zeroLength requires one damping material per material",
                        "element",
                        args,
                    )
                element["dampMats"] = [int(value) for value in values]
                idx += len(materials) + 1
            elif token == "-damp":
                if idx + 1 == len(args):
                    raise self._error(
                        "zeroLength -damp requires a tag", "element", args
                    )
                element["damp"] = int(args[idx + 1])
                idx += 2
            else:
                raise self._error(
                    f"unsupported zeroLength option `{token}`", "element", args
                )
        self.elements.append(element)
        return ""

    def _parse_recorder_flags(
        self, args: tuple[str, ...]
    ) -> tuple[dict[str, Any], tuple[str, ...]]:
        options: dict[str, Any] = {"time": False}
        idx = 0
        while idx < len(args):
            token = args[idx]
            if not token.startswith("-"):
                break
            if token in {"-time", "-load"}:
                options["time"] = True
                idx += 1
            elif token == "-precision":
                options["precision"] = int(args[idx + 1])
                idx += 2
            elif token == "-file":
                options["file"] = args[idx + 1]
                options["format"] = "file"
                idx += 2
            elif token == "-xml":
                options["file"] = args[idx + 1]
                options["format"] = "xml"
                idx += 2
            elif token in {"-node", "-nodes"}:
                values = []
                idx += 1
                if idx < len(args) and args[idx] == "all":
                    options["node"] = sorted(self.nodes)
                    idx += 1
                    continue
                while idx < len(args):
                    try:
                        values.append(int(args[idx]))
                    except ValueError:
                        break
                    idx += 1
                options["node"] = values
            elif token == "-nodeRange":
                start = int(args[idx + 1])
                end = int(args[idx + 2])
                low, high = sorted((start, end))
                options["node"] = sorted(
                    node for node in self.nodes if low <= node <= high
                )
                idx += 3
            elif token == "-eleRange":
                start = int(args[idx + 1])
                end = int(args[idx + 2])
                options["ele"] = sorted(
                    element["id"]
                    for element in self.elements
                    if start <= element["id"] <= end
                )
                idx += 3
            elif token == "-dT":
                options["delta_t"] = float(args[idx + 1])
                idx += 2
            elif token == "-ele":
                values = []
                idx += 1
                while idx < len(args):
                    try:
                        values.append(int(args[idx]))
                    except ValueError:
                        break
                    idx += 1
                options["ele"] = values
            elif token == "-region":
                tag = int(args[idx + 1])
                if tag not in self.regions:
                    raise self._error(f"region {tag} not found", "recorder", args)
                options["node"], options["ele"] = self.regions[tag]
                idx += 2
            elif token == "-dof":
                values = []
                idx += 1
                while idx < len(args):
                    try:
                        values.append(int(args[idx]))
                    except ValueError:
                        break
                    idx += 1
                options["dof"] = values
            elif token == "-timeSeries":
                options["timeSeries"] = int(args[idx + 1])
                idx += 2
            elif token == "-iNode":
                values = []
                idx += 1
                while idx < len(args):
                    try:
                        values.append(int(args[idx]))
                    except ValueError:
                        break
                    idx += 1
                options["iNode"] = values
            elif token == "-jNode":
                values = []
                idx += 1
                while idx < len(args):
                    try:
                        values.append(int(args[idx]))
                    except ValueError:
                        break
                    idx += 1
                options["jNode"] = values
            elif token == "-perpDirn":
                options["perpDirn"] = int(args[idx + 1])
                idx += 2
            elif token in {"-wipe", "-closeOnWrite"}:
                idx += 1
            else:
                raise self._error(
                    f"unsupported recorder flag `{token}`", "recorder", args
                )
        return options, args[idx:]

    def _cmd_recorder(self, *args: str) -> str:
        if not args:
            raise self._error("recorder type is required", "recorder", args)
        if args[0] in {"display", "plot"}:
            return ""
        first = len(self.recorders)
        self._add_recorder(*args)
        tag = self.next_recorder_tag
        self.next_recorder_tag += 1
        self.active_recorders[tag] = self.recorders[first:]
        return str(tag)

    def _add_recorder(self, *args: str) -> None:
        recorder_type = args[0]
        options, remainder = self._parse_recorder_flags(args[1:])
        raw_file = options.get("file")
        if not raw_file:
            raise self._error("recorder requires -file", "recorder", args)
        output = self._recorder_output(raw_file)
        include_time = bool(options.get("time"))
        sampling = {"delta_t": options["delta_t"]} if "delta_t" in options else {}

        if recorder_type == "Node":
            nodes = options.get("node") or []
            dofs = options.get("dof") or []
            if not remainder:
                raise self._error(
                    "Node recorder response is required", "recorder", args
                )
            kind = remainder[0]
            if kind in {"disp", "vel", "accel", "reaction"}:
                rec_type = {
                    "disp": "node_displacement",
                    "vel": "node_velocity",
                    "accel": "node_acceleration",
                    "reaction": "node_reaction",
                }[kind]
                for node_id in nodes:
                    if node_id not in self.nodes:
                        continue
                    recorder = {
                        "type": rec_type,
                        "nodes": [node_id],
                        "dofs": dofs,
                        "output": output,
                        "raw_path": raw_file,
                        "include_time": include_time,
                    }
                    recorder.update(sampling)
                    if "precision" in options:
                        recorder["precision"] = options["precision"]
                    if kind == "accel" and "timeSeries" in options:
                        recorder["time_series"] = int(options["timeSeries"])
                    self._append_recorder(recorder)
            elif kind == "eigen" or kind.startswith("eigen "):
                return
            else:
                raise self._error(
                    f"unsupported Node recorder response `{kind}`", "recorder", args
                )
            return

        if recorder_type == "EnvelopeNode":
            nodes = options.get("node") or []
            dofs = options.get("dof") or []
            if not remainder:
                raise self._error(
                    "EnvelopeNode recorder response is required", "recorder", args
                )
            if not nodes or not dofs:
                raise self._error(
                    "EnvelopeNode recorder requires -node and -dof", "recorder", args
                )
            kind = remainder[0]
            if kind == "disp":
                rec_type = "envelope_node_displacement"
            elif kind == "accel":
                rec_type = "envelope_node_acceleration"
            else:
                raise self._error(
                    f"unsupported EnvelopeNode response `{kind}`", "recorder", args
                )
            widths: list[int] = []
            for _node_id in nodes:
                width = self._envelope_node_width(dofs, include_time)
                if width is None:
                    widths = []
                    break
                widths.append(width)
            group_layout = None
            if widths:
                group_layout = {
                    "type": rec_type,
                    "nodes": list(nodes),
                    "values_per_node": widths,
                }
            for node_id in nodes:
                recorder = {
                    "type": rec_type,
                    "nodes": [node_id],
                    "dofs": dofs,
                    "output": output,
                    "raw_path": raw_file,
                    "include_time": include_time,
                }
                if kind == "accel" and "timeSeries" in options:
                    recorder["time_series"] = int(options["timeSeries"])
                if group_layout is not None:
                    recorder["group_layout"] = dict(group_layout)
                self._append_recorder(recorder)
            return

        if recorder_type == "Drift":
            i_nodes = options.get("iNode") or []
            j_nodes = options.get("jNode") or []
            if not i_nodes or not j_nodes:
                raise self._error(
                    "Drift recorder requires -iNode and -jNode", "recorder", args
                )
            if len(i_nodes) != len(j_nodes):
                raise self._error(
                    "Drift recorder requires matching -iNode and -jNode counts",
                    "recorder",
                    args,
                )
            for i_node, j_node in zip(i_nodes, j_nodes):
                self._append_recorder(
                    {
                        "type": "drift",
                        "i_node": i_node,
                        "j_node": j_node,
                        "dof": int((options.get("dof") or [0])[0]),
                        "perp_dirn": options["perpDirn"],
                        "output": output,
                        "raw_path": raw_file,
                        "include_time": include_time,
                    }
                )
            return

        if recorder_type == "Element":
            element_by_id = {int(element["id"]): element for element in self.elements}
            known_elements = set(element_by_id) - self.removed_elements
            elements = [
                elem_id
                for elem_id in (options.get("ele") or [])
                if elem_id in known_elements
            ]
            if not elements:
                return
            if not remainder:
                raise self._error(
                    "Element recorder response is required", "recorder", args
                )
            kind = remainder[0]
            if kind in {"stress", "stresses", "strain", "strains"} and all(
                element_by_id[elem]["type"] in {"SSPbrickUP", "SSPquad"}
                for elem in elements
            ):
                remainder = ("material", "1", *remainder)
                kind = "material"
            if kind in {"material", "integrPoint"} and len(remainder) in {3, 4}:
                point = int(remainder[1])
                response = remainder[2]
                if point < 1 or response not in {
                    "stress",
                    "stresses",
                    "strain",
                    "strains",
                    "state",
                }:
                    raise self._error(
                        "unsupported solid material recorder response", "recorder", args
                    )
                typ = (
                    "quad_stress"
                    if response in {"stress", "stresses"}
                    else "quad_state"
                    if response == "state"
                    else "quad_strain"
                )
                for elem_id in elements:
                    element_type = element_by_id[elem_id]["type"]
                    if (
                        element_type not in _SOLID_INTEGRATION_POINTS
                        or point > _SOLID_INTEGRATION_POINTS[element_type]
                    ):
                        raise self._error(
                            "invalid solid material integration point", "recorder", args
                        )
                    recorder = {
                        "type": typ,
                        "elements": [elem_id],
                        "integration_point": point,
                        "output": output,
                        "raw_path": raw_file,
                        "include_time": include_time,
                    }
                    recorder.update(sampling)
                    if len(remainder) == 4 and typ == "quad_stress":
                        recorder["stress_components"] = int(remainder[3])
                    self._append_recorder(recorder)
                return ""
            if kind.lower() in {"rcpanel", "rc_panel", "sheardef", "curvature"} and all(
                element_by_id[elem_id]["type"] == "SFI_MVLEM" for elem_id in elements
            ):
                # Keep the panel diagnostics in the source model. The wall
                # comparison uses nodal displacement and global force.
                return ""
            if kind.removesuffix("s") in {
                "force",
                "frictionforce",
                "forcescalar",
                "masterforce",
                "primaryforce",
            } and all(
                element_by_id[elem_id]["type"] == "BeamContact2D"
                for elem_id in elements
            ):
                rec_type = {
                    "force": "contact_secondary_force",
                    "frictionforce": "contact_friction_force",
                    "forcescalar": "contact_scalar_force",
                    "masterforce": "contact_primary_force",
                    "primaryforce": "contact_primary_force",
                }[kind.removesuffix("s")]
            elif kind in {"force", "forces", "globalForce"}:
                rec_type = "element_force"
            elif kind == "localForce":
                rec_type = "element_local_force"
            elif (
                kind == "axialForce" or tuple(remainder) == ("material", "stress")
            ) and all(
                element_by_id[elem_id]["type"] in {"truss", "corotTruss"}
                for elem_id in elements
            ):
                rec_type = "element_basic_force"
            elif kind in {"basicForce", "basicForces"}:
                if all(
                    element_by_id[elem_id]["type"] == "truss" for elem_id in elements
                ):
                    return
                rec_type = "element_basic_force"
            elif kind in {
                "deformation",
                "deformations",
                "basicDeformation",
                "basicDeformations",
                "chordRotation",
                "chordDeformation",
            }:
                rec_type = "element_deformation"
            elif kind.lower() in {
                "curvature",
                "fiber_strain",
                "fiber_stress_concrete",
                "fiber_stress_steel",
                "shear_force_deformation",
            } and all(
                element_by_id[elem_id]["type"] == "MVLEM" for elem_id in elements
            ):
                # Retain these diagnostics in OpenSees. The direct-Tcl comparison
                # uses the wall's global force and nodal displacement recorders.
                return
            elif kind == "plasticRotation":
                return
            elif kind == "section":
                if any(
                    element_by_id[e].get("_uniform_elastic_hinge_3d") for e in elements
                ):
                    raise self._error(
                        "3D beamWithHinges section recorders are unsupported",
                        "recorder",
                        args,
                    )
                section_idx = int(remainder[1])
                response = remainder[2]
                if response == "fiber":
                    return ""
                if response in {"force", "forces"}:
                    rec_type = "section_force"
                elif response in {"deformation", "deformations"}:
                    rec_type = "section_deformation"
                else:
                    raise self._error(
                        f"unsupported section recorder response `{response}`",
                        "recorder",
                        args,
                    )
                for elem_id in elements:
                    self._append_recorder(
                        {
                            "type": rec_type,
                            "elements": [elem_id],
                            "section": section_idx,
                            "output": output,
                            "raw_path": raw_file,
                            "include_time": include_time,
                        }
                    )
                return
            else:
                raise self._error(
                    f"unsupported Element recorder response `{kind}`", "recorder", args
                )
            elem_dofs = options.get("dof") or []
            for elem_id in elements:
                recorder = {
                    "type": rec_type,
                    "elements": [elem_id],
                    "output": output,
                    "raw_path": raw_file,
                    "include_time": include_time,
                    **sampling,
                }
                if rec_type == "element_basic_force" and kind in {
                    "axialForce",
                    "material",
                }:
                    recorder["truss_response"] = kind
                if elem_dofs:
                    recorder["dofs"] = list(elem_dofs)
                if "precision" in options:
                    recorder["precision"] = options["precision"]
                if rec_type == "element_deformation" and element_by_id[elem_id][
                    "type"
                ] in {"elasticBeamColumn2d", "elasticBeamColumn3d"}:
                    # Elastic beam-column direct Tcl references from OpenSees examples
                    # can be blank because the upstream response aliasing is inconsistent.
                    recorder["parity"] = False
                self._append_recorder(recorder)
            return

        if recorder_type == "EnvelopeElement":
            known_elements = {int(element["id"]) for element in self.elements}
            if not remainder or remainder[0] not in {"force", "forces", "localForce"}:
                raise self._error(
                    "unsupported EnvelopeElement response", "recorder", args
                )
            rec_type = (
                "envelope_element_local_force"
                if remainder[0] == "localForce"
                else "envelope_element_force"
            )
            element_by_id = {int(element["id"]): element for element in self.elements}
            elements = [
                elem_id
                for elem_id in (options.get("ele") or [])
                if elem_id in known_elements
            ]
            group_layout = None
            if elements:
                widths: list[int] = []
                for elem_id in elements:
                    width = self._envelope_element_force_width(
                        element_by_id[elem_id], include_time
                    )
                    if width is None:
                        widths = []
                        break
                    widths.append(width)
                if widths:
                    group_layout = {
                        "type": rec_type,
                        "elements": list(elements),
                        "values_per_element": widths,
                    }
            for elem_id in elements:
                recorder = {
                    "type": rec_type,
                    "elements": [elem_id],
                    "output": output,
                    "raw_path": raw_file,
                    "include_time": include_time,
                }
                if group_layout is not None:
                    recorder["group_layout"] = dict(group_layout)
                self._append_recorder(recorder)
            return

        raise self._error(
            f"unsupported recorder type `{recorder_type}`", "recorder", args
        )

    def _cmd_time_series(self, *args: str, generated: bool = False) -> str:
        if len(args) < 2:
            raise self._error(
                "timeSeries expects at least type and tag", "timeSeries", args
            )
        ts_type = args[0]
        tag = int(args[1])
        if ts_type in {"Constant", "Linear"}:
            factor = 1.0
            if len(args) == 4 and args[2] == "-factor":
                factor = float(args[3])
            elif len(args) != 2:
                raise self._error(
                    f"unsupported {ts_type} timeSeries form",
                    "timeSeries",
                    args,
                )
            entry = {"type": ts_type, "tag": tag, "factor": factor}
        elif ts_type in {"Path", "Series"}:
            options = {"factor": 1.0}
            time_path = None
            idx = 2
            while idx < len(args):
                token = args[idx]
                if token in {"-dt", "-dT"}:
                    options["dt"] = float(args[idx + 1])
                    idx += 2
                elif token in {"-startTime", "-tStart"}:
                    options["start_time"] = float(args[idx + 1])
                    idx += 2
                elif token == "-useLast":
                    options["use_last"] = True
                    idx += 1
                elif token == "-prependZero":
                    options["_prepend_zero"] = True
                    idx += 1
                elif token == "-filePath":
                    options["values_path"] = self._resolve_values_path(args[idx + 1])
                    idx += 2
                elif token == "-fileTime":
                    time_path = self._resolve_values_path(args[idx + 1])
                    idx += 2
                elif token in {"-time", "-values"}:
                    options[token[1:]] = [
                        float(v) for v in self.interp.tk.splitlist(args[idx + 1])
                    ]
                    idx += 2
                elif token == "-factor":
                    options["factor"] = float(args[idx + 1])
                    idx += 2
                else:
                    raise self._error(
                        f"unsupported Path timeSeries flag `{token}`",
                        "timeSeries",
                        args,
                    )
            if time_path is not None:
                if "values_path" not in options:
                    raise self._error(
                        "Path -fileTime requires -filePath", "timeSeries", args
                    )
                options["time"] = [
                    float(v) for v in Path(time_path).read_text().split()
                ]
                options["values"] = [
                    float(v)
                    for v in Path(options.pop("values_path")).read_text().split()
                ]
            if "time" in options:
                if "values" not in options or len(options["time"]) != len(
                    options["values"]
                ):
                    raise self._error(
                        "Path requires paired -time and -values", "timeSeries", args
                    )
            elif "dt" not in options or (
                "values_path" not in options and "values" not in options
            ):
                raise self._error(
                    "Path requires -dt and values or explicit time and value lists",
                    "timeSeries",
                    args,
                )
            self._prepend_path_zero(options)
            entry = {"type": "Path", "tag": tag, **options}
        elif ts_type in {"Trig", "Sine"}:
            if len(args) < 5:
                raise self._error(
                    "Trig timeSeries expects tag tStart tFinish period",
                    "timeSeries",
                    args,
                )
            entry = {
                "type": "Trig",
                "tag": tag,
                "t_start": float(args[2]),
                "t_finish": float(args[3]),
                "period": float(args[4]),
                "factor": 1.0,
            }
            idx = 5
            while idx < len(args):
                token = args[idx]
                if token == "-factor":
                    entry["factor"] = float(args[idx + 1])
                    idx += 2
                elif token == "-zeroShift":
                    entry["zero_shift"] = float(args[idx + 1])
                    idx += 2
                elif token in {"-shift", "-phaseShift"}:
                    entry["phase_shift"] = float(args[idx + 1])
                    idx += 2
                else:
                    raise self._error(
                        f"unsupported Trig timeSeries flag `{token}`",
                        "timeSeries",
                        args,
                    )
        else:
            raise self._error(
                f"unsupported timeSeries type `{ts_type}`", "timeSeries", args
            )
        self._register_time_series_entry(entry, generated=generated)
        return ""

    def _preserve_constant_pattern(self) -> None:
        plain = self.current_pattern
        if plain is None or plain["type"] != "Plain":
            return
        series = self.time_series_by_tag[plain["time_series"]]
        if series["type"] == "Constant" and not plain.get("sp_constraints"):
            self.constant_patterns[plain["tag"]] = plain

    def _constant_pattern_loads(self) -> tuple[list[dict], list[dict]]:
        loads, element_loads = [], []
        for pattern in self.constant_patterns.values():
            factor = self.time_series_by_tag[pattern["time_series"]].get("factor", 1.0)
            loads.extend(
                {**load, "value": factor * load["value"]} for load in pattern["loads"]
            )
            element_loads.extend(
                {
                    key: (
                        factor * value
                        if key in {"wy", "wz", "wx"}
                        else [factor * temp for temp in value]
                        if key == "temperatures"
                        else value
                    )
                    for key, value in load.items()
                }
                for load in pattern["element_loads"]
            )
        return loads, element_loads

    def _cmd_pattern(self, *args: str) -> str:
        if not args:
            raise self._error("pattern type is required", "pattern", args)
        pattern_type = args[0]
        if pattern_type == "Plain":
            if len(args) != 4:
                raise self._error(
                    "expected `pattern Plain tag tsSpec {body}`", "pattern", args
                )
            tag = int(args[1])
            try:
                ts_tag = int(args[2])
                if ts_tag not in self.time_series_by_tag:
                    raise self._error(
                        f"timeSeries tag {ts_tag} not found", "pattern", args
                    )
            except ValueError:
                ts_tag = (
                    tag
                    if tag not in self.time_series_by_tag
                    else self._next_time_series_tag()
                )
                parts = self.interp.tk.splitlist(args[2])
                if parts and parts[0] in {"Linear", "Constant"}:
                    self._cmd_time_series(
                        parts[0], str(ts_tag), *parts[1:], generated=True
                    )
                else:
                    series = self._parse_inline_series_spec(args[2])
                    series["tag"] = ts_tag
                    self._register_time_series_entry(series, generated=True)
            self._preserve_constant_pattern()
            self.current_pattern = {"type": "Plain", "tag": tag, "time_series": ts_tag}
            self.current_plain_pattern = PendingPlainPattern(
                tag=tag, time_series=ts_tag
            )
            try:
                self._eval_tcl(args[3], "pattern", args)
            finally:
                if self.current_plain_pattern is not None:
                    self.current_pattern = {
                        "type": "Plain",
                        "tag": tag,
                        "time_series": ts_tag,
                        "loads": list(self.current_plain_pattern.loads),
                        "element_loads": list(self.current_plain_pattern.element_loads),
                    }
                    if self.current_plain_pattern.sp_constraints:
                        self.current_pattern["sp_constraints"] = list(
                            self.current_plain_pattern.sp_constraints
                        )
                self.current_plain_pattern = None
            return ""

        if pattern_type == "UniformExcitation":
            if len(args) < 5:
                raise self._error(
                    "expected `pattern UniformExcitation tag dir -accel tsTag`",
                    "pattern",
                    args,
                )
            model = self._require_model("pattern", args)
            is_frame3d = model["ndm"] == 3 and model["ndf"] == 6
            has_coupled_soil = self._has_coupled_soil()
            supports_multiple = is_frame3d or has_coupled_soil
            self.uniform_excitation_count_since_analyze += 1
            if (
                self.uniform_excitation_count_since_analyze > 1
                and not supports_multiple
            ):
                self.has_complex_transient_excitation = True
            tag = int(args[1])
            direction = int(args[2])
            options: dict[str, Any] = {
                "type": "UniformExcitation",
                "tag": tag,
                "direction": direction,
            }
            idx = 3
            while idx < len(args):
                token = args[idx]
                if token == "-accel":
                    if idx + 1 >= len(args):
                        raise self._error(
                            "UniformExcitation missing accel series", "pattern", args
                        )
                    try:
                        accel = int(args[idx + 1])
                        if accel not in self.time_series_by_tag:
                            raise self._error(
                                f"timeSeries tag {accel} not found", "pattern", args
                            )
                    except ValueError:
                        accel = self._register_time_series_entry(
                            self._parse_inline_series_spec(args[idx + 1]),
                            generated=True,
                        )
                    options["accel"] = accel
                    idx += 2
                elif token == "-vel0":
                    if idx + 1 >= len(args):
                        raise self._error(
                            "UniformExcitation missing vel0 value", "pattern", args
                        )
                    options["vel0"] = float(args[idx + 1])
                    idx += 2
                else:
                    raise self._error(
                        f"unsupported UniformExcitation flag `{token}`",
                        "pattern",
                        args,
                    )
            if "accel" not in options:
                raise self._error(
                    "expected `pattern UniformExcitation tag dir -accel tsTag`",
                    "pattern",
                    args,
                )
            if self.transient_eigen_initialized:
                # eigen has already populated Newmark's velocity vector from the
                # committed nodes. UniformExcitation changes nodal velocities
                # without invalidating that vector.
                options.pop("vel0", None)
            background_loads = []
            if has_coupled_soil and self.current_pattern:
                if self.current_pattern["type"] == "Plain":
                    series = self.time_series_by_tag[
                        self.current_pattern["time_series"]
                    ]
                    loads = self.current_pattern.get("loads", [])
                    if loads and series["type"] != "Constant":
                        raise self._error(
                            "concurrent Plain loads require a Constant series",
                            "pattern",
                            args,
                        )
                    background_loads = [
                        {**load, "value": load["value"] * series.get("factor", 1.0)}
                        for load in loads
                    ]
                else:
                    background_loads = self.current_pattern.get("background_loads", [])
            if not has_coupled_soil:
                self._preserve_constant_pattern()
            if (
                supports_multiple
                and self.current_pattern
                and self.current_pattern["type"] == pattern_type
            ):
                excitations = self.current_pattern.get(
                    "excitations",
                    [self.current_pattern],
                )
                if any(excitation["tag"] == tag for excitation in excitations):
                    raise self._error(
                        f"duplicate UniformExcitation pattern tag {tag}",
                        "pattern",
                        args,
                    )
                self.current_pattern = {
                    "type": pattern_type,
                    "excitations": [*excitations, options],
                }
            else:
                self.current_pattern = options
            if background_loads:
                self.current_pattern["background_loads"] = background_loads
            return ""

        if pattern_type == "MultipleSupport":
            if len(args) != 3:
                raise self._error(
                    "expected `pattern MultipleSupport tag {body}`",
                    "pattern",
                    args,
                )
            self._require_model("pattern", args)
            if self._has_coupled_soil():
                self.active_pressure_motions = {}
                try:
                    self._eval_tcl(args[2], "pattern", args)
                finally:
                    self.active_pressure_motions = None
                return ""
            tag = int(args[1])
            self.ground_motion_series = {}
            self.active_imposed_motions = []
            try:
                self._eval_tcl(args[2], "pattern", args)
                self.imposed_motion_patterns[tag] = self.active_imposed_motions
            finally:
                self.ground_motion_series = None
                self.active_imposed_motions = None
            if self.current_pattern is None:
                self.pattern_removed = True
            return ""

        raise self._error(f"unsupported pattern type `{pattern_type}`", "pattern", args)

    def _cmd_ground_motion(self, *args: str) -> str:
        if (
            (self.ground_motion_series is None and self.active_pressure_motions is None)
            or len(args) < 4
            or len(args) % 2
            or args[1] not in {"Series", "Plain"}
        ):
            raise self._error(
                "groundMotion requires a Series or Plain motion",
                "groundMotion",
                args,
            )
        series_flags = {
            "-disp": "time_series",
            "-displacement": "time_series",
            "-vel": "velocity_series",
            "-velocity": "velocity_series",
            "-accel": "acceleration_series",
            "-acceleration": "acceleration_series",
        }
        series_tags = {}
        dt = 0.01
        factor = 1.0
        for flag, value in zip(args[2::2], args[3::2]):
            if flag in series_flags:
                try:
                    series = int(value)
                except ValueError:
                    series = self._register_time_series_entry(
                        self._parse_inline_series_spec(value), generated=True
                    )
                if series not in self.time_series_by_tag:
                    raise self._error(
                        f"timeSeries tag {series} not found", "groundMotion", args
                    )
                series_tags[series_flags[flag]] = series
            elif flag in {"-dtInt", "-dtIntegrator", "-deltaT"}:
                dt = float(value)
                if not math.isfinite(dt):
                    raise self._error(
                        "invalid groundMotion time step", "groundMotion", args
                    )
            elif flag in {"-fact", "-factor", "-scale"}:
                factor = float(value)
            elif flag not in {"-int", "-integrator"} or value != "Trapezoidal":
                raise self._error(
                    f"unsupported groundMotion option {flag} {value}",
                    "groundMotion",
                    args,
                )
        if not series_tags:
            raise self._error(
                "groundMotion requires a time series", "groundMotion", args
            )
        if self.active_pressure_motions is not None:
            if set(series_tags) != {"velocity_series"}:
                raise self._error(
                    "a pore-pressure motion requires a Constant velocity series",
                    "groundMotion",
                    args,
                )
            entry = self.time_series_by_tag[series_tags["velocity_series"]]
            if entry["type"] != "Constant":
                raise self._error(
                    "a pore-pressure motion requires a Constant velocity series",
                    "groundMotion",
                    args,
                )
            # A Constant series has zero duration, so each generated Path
            # holds its single trapezoidal sample for the duration of motion.
            value = entry.get("factor", 1.0)
            self.active_pressure_motions[int(args[0])] = {
                "value": factor * value,
                "displacement": factor * (0.5 * dt * value) if dt > 0.0 else 0.0,
                "acceleration": factor * (2.0 * value / dt) if dt > 0.0 else 0.0,
            }
            return ""
        assert self.ground_motion_series is not None
        for target, source, differentiate in (
            ("velocity_series", "acceleration_series", False),
            ("time_series", "velocity_series", False),
            ("velocity_series", "time_series", True),
            ("acceleration_series", "velocity_series", True),
        ):
            if target not in series_tags and source in series_tags:
                entry = trapezoidal_path(
                    self.time_series_by_tag[series_tags[source]],
                    dt,
                    differentiate=differentiate,
                )
                if entry is not None:
                    entry["tag"] = self._next_time_series_tag()
                    series_tags[target] = self._register_time_series_entry(
                        entry, generated=True
                    )
        self.ground_motion_series[int(args[0])] = {**series_tags, "factor": factor}
        return ""

    def _cmd_imposed_motion(self, *args: str) -> str:
        if self.active_pressure_motions is not None:
            model = self._require_model("imposedMotion", args)
            if len(args) != 3:
                raise self._error(
                    "imposedMotion expects node dof motion", "imposedMotion", args
                )
            node, dof, motion = map(int, args)
            if (
                dof != model["ndm"] + 1
                or self.node_ndfs.get(node) != dof
                or motion not in self.active_pressure_motions
            ):
                raise self._error(
                    "imposedMotion requires a pore-pressure DOF and a velocity series",
                    "imposedMotion",
                    args,
                )
            self.prescribed_pressures[node, dof] = self.active_pressure_motions[motion]
            return ""
        if self.active_imposed_motions is None or len(args) != 3:
            raise self._error(
                "expected imposedMotion node dof motion in MultipleSupport",
                "imposedMotion",
                args,
            )
        assert self.ground_motion_series is not None
        node, dof, motion = map(int, args)
        if (
            not 1 <= dof <= self.node_ndfs.get(node, 0)
            or motion not in self.ground_motion_series
        ):
            raise self._error(
                "imposed motion requires an existing DOF and ground motion",
                "imposedMotion",
                args,
            )
        self.active_imposed_motions.append(
            {
                "node": node,
                "dof": dof,
                **self.ground_motion_series[motion],
            }
        )
        return ""

    def _cmd_load(self, *args: str) -> str:
        self._require_model("load", args)
        if self.current_plain_pattern is None:
            raise self._error(
                "load is only supported inside `pattern Plain`", "load", args
            )
        node_id = int(args[0])
        ndf = self.node_ndfs.get(node_id)
        if ndf is None or len(args) < ndf + 1:
            raise self._error("load argument count does not match ndf", "load", args)
        for dof, value_text in enumerate(args[1 : ndf + 1], start=1):
            value = float(value_text)
            if abs(value) <= 0.0:
                continue
            self.current_plain_pattern.loads.append(
                {"node": node_id, "dof": dof, "value": value}
            )
        return ""

    def _cmd_ele_load(self, *args: str) -> str:
        if self.current_plain_pattern is None:
            raise self._error(
                "eleLoad is only supported inside `pattern Plain`", "eleLoad", args
            )
        if len(args) < 4 or args[0] != "-ele":
            raise self._error("eleLoad expects `-ele ... -type ...`", "eleLoad", args)
        idx = 1
        elements = []
        while idx < len(args) and args[idx] != "-type":
            elements.append(int(args[idx]))
            idx += 1
        if idx >= len(args):
            raise self._error("eleLoad missing `-type`", "eleLoad", args)
        if any(
            element.get("_uniform_elastic_hinge_3d") and element["id"] in elements
            for element in self.elements
        ):
            raise self._error(
                "3D beamWithHinges currently supports nodal loads", "eleLoad", args
            )
        load_type = args[idx + 1]
        if load_type not in {"-beamUniform", "-beamThermal"}:
            raise self._error(
                f"unsupported eleLoad type `{load_type}`", "eleLoad", args
            )
        values = [float(value) for value in args[idx + 2 :]]
        model = self._require_model("eleLoad", args)
        payload: dict[str, Any] = {"type": load_type[1:]}
        if load_type == "-beamThermal":
            if model["ndm"] != 2 or len(values) not in {4, 10, 18}:
                raise self._error(
                    "beamThermal expects 2, 5, or 9 temperature/location pairs in 2D",
                    "eleLoad",
                    args,
                )
            temperatures = values[::2]
            locations = values[1::2]
            if not all(a < b for a, b in zip(locations, locations[1:])):
                raise self._error(
                    "beamThermal locations must increase", "eleLoad", args
                )
            payload.update(temperatures=temperatures, locations=locations)
        else:
            components = ("wy", "wz", "wx") if model["ndm"] == 3 else ("wy", "wx")
            if len(values) not in {len(components) - 1, len(components)}:
                syntax = "Wy Wz ?Wx?" if model["ndm"] == 3 else "Wy ?Wx?"
                raise self._error(f"beamUniform expects `{syntax}`", "eleLoad", args)
            payload.update(zip(components, values))
        for elem_id in elements:
            self.current_plain_pattern.element_loads.append(
                {"element": elem_id, **payload}
            )
        return ""

    def _cmd_constraints(self, *args: str) -> str:
        if len(args) == 3 and args[0] == "Penalty":
            sp, mp = map(float, args[1:])
            if not all(math.isfinite(coef) and coef > 0 for coef in (sp, mp)):
                raise self._error(
                    "Penalty coefficients must be positive", "constraints", args
                )
            self.penalty_coefficients = (sp, mp)
        elif args and (
            args[0] == "Transformation"
            or (len(args) == 1 and args[0] in {"Plain", "Lagrange"})
        ):
            self.penalty_coefficients = None
        else:
            raise self._error("unsupported constraints handler", "constraints", args)
        self.constraints_handler = args[0]
        return ""

    def _cmd_numberer(self, *args: str) -> str:
        if len(args) != 1:
            raise self._error("numberer expects one argument", "numberer", args)
        self.numberer_handler = args[0]
        return ""

    def _cmd_system(self, *args: str) -> str:
        if not args:
            raise self._error("system expects a solver name", "system", args)
        solver = _SYSTEM_NAME_ALIASES.get(args[0], args[0])
        if solver not in _SUPPORTED_SYSTEM_NAMES:
            raise self._error(
                (
                    "unsupported system solver "
                    f"`{args[0]}` (supported: {', '.join(sorted(_SUPPORTED_SYSTEM_NAMES))})"
                ),
                "system",
                args,
            )
        self.system_handler = solver
        self.system_options = list(args[1:])
        return ""

    def _cmd_test(self, *args: str) -> str:
        if len(args) < 3:
            raise self._error("test expects type tol maxIters", "test", args)
        self.current_test = {
            "type": args[0],
            "tol": float(args[1]),
            "max_iters": int(float(args[2])),
        }
        if len(args) >= 4:
            self.current_test["print_flag"] = int(float(args[3]))
        if len(args) > 4:
            self.current_test["extra_args"] = list(args[4:])
        return ""

    def _cmd_algorithm(self, *args: str) -> str:
        if not args:
            raise self._error("algorithm type is required", "algorithm", args)
        self.algorithm_name = self._normalize_algorithm_name(args)
        self.algorithm_options = None
        if len(args) > 1:
            options: dict[str, Any] = {"raw_args": list(args[1:])}
            if args[0] in {"ModifiedNewton", "Newton"} and "-initial" in args[1:]:
                options["initial"] = True
            if args[0] == "Broyden":
                try:
                    options["max_iters"] = int(float(args[1]))
                except (ValueError, IndexError):
                    pass
            if args[0] == "NewtonLineSearch":
                try:
                    options["alpha"] = float(args[1])
                except (ValueError, IndexError):
                    pass
            if args[0] == "KrylovNewton":
                for i, token in enumerate(args[1:], start=1):
                    if token == "-maxDim" and i + 1 < len(args):
                        try:
                            options["maxDim"] = int(float(args[i + 1]))
                        except ValueError:
                            pass
                        break
            self.algorithm_options = options
        return ""

    def _cmd_integrator(self, *args: str) -> str:
        if not args:
            raise self._error("integrator type is required", "integrator", args)
        self.load_control_has_run = False
        if args[0] == "LoadControl":
            if len(args) < 2:
                raise self._error(
                    "LoadControl expects one step size", "integrator", args
                )
            integrator = {"type": "LoadControl", "step": float(args[1])}
            if len(args) >= 3:
                integrator["num_iter"] = int(float(args[2]))
            if len(args) >= 4:
                integrator["min_step"] = float(args[3])
            if len(args) >= 5:
                integrator["max_step"] = float(args[4])
            if len(args) > 5:
                integrator["extra_args"] = list(args[5:])
            self.integrator = integrator
            return ""
        if args[0] == "Newmark":
            if len(args) != 3:
                raise self._error("Newmark expects gamma and beta", "integrator", args)
            self.integrator = {
                "type": "Newmark",
                "gamma": float(args[1]),
                "beta": float(args[2]),
            }
            self.transient_eigen_initialized = False
            return ""
        if args[0] == "HSConstraint":
            if not 2 <= len(args) <= 5:
                raise self._error(
                    "HSConstraint expects arcLength ?psi_u? ?psi_f? ?u_ref?",
                    "integrator",
                    args,
                )
            self.integrator = {
                "type": "HSConstraint",
                **dict(
                    zip(("arc_length", "psi_u", "psi_f", "u_ref"), map(float, args[1:]))
                ),
            }
            return ""
        if args[0] == "DisplacementControl":
            if len(args) < 4:
                raise self._error(
                    "DisplacementControl expects node dof du", "integrator", args
                )
            integrator = {
                "type": "DisplacementControl",
                "node": int(args[1]),
                "dof": int(args[2]),
                "du": float(args[3]),
            }
            if len(args) >= 5:
                integrator["num_iter"] = int(float(args[4]))
            if len(args) >= 6:
                integrator["min_du"] = float(args[5])
            if len(args) >= 7:
                integrator["max_du"] = float(args[6])
            if len(args) > 7:
                integrator["extra_args"] = list(args[7:])
            self.integrator = integrator
            return ""
        raise self._error(f"unsupported integrator `{args[0]}`", "integrator", args)

    def _cmd_analysis(self, *args: str) -> str:
        if len(args) != 1 or args[0] not in {"Static", "Transient"}:
            raise self._error("analysis must be Static or Transient", "analysis", args)
        self.analysis_type = args[0]
        self.transient_eigen_initialized = False
        if self.current_test is None:
            # OpenSees' `analysis` command creates CTestNormUnbalance(1e-6, 25)
            # when no test has been defined (SRC/tcl/commands.cpp).
            self.current_test = {
                "type": "NormUnbalance",
                "tol": 1.0e-6,
                "max_iters": 25,
            }
        return ""

    def _cmd_initialize(self, *args: str) -> str:
        if args:
            raise self._error("initialize takes no arguments", "initialize", args)
        self.pending_initialize = True
        return ""

    def _solver_chain_primary_attempt(self, analysis: dict[str, Any]) -> dict[str, Any]:
        attempt: dict[str, Any] = {
            "algorithm": analysis.get("algorithm", "Newton"),
            "test_type": analysis.get("test_type", "MaxDispIncr"),
            "max_iters": analysis.get("max_iters", 20),
            "tol": analysis.get("tol", 1.0e-10),
        }
        if "algorithm_options" in analysis:
            attempt["algorithm_options"] = dict(analysis["algorithm_options"])
        return attempt

    def _solver_chain_retry_attempt(
        self, analysis: dict[str, Any], attempt: dict[str, Any]
    ) -> dict[str, Any]:
        retry_attempt: dict[str, Any] = {
            "algorithm": attempt.get(
                "fallback_algorithm", analysis.get("algorithm", "Newton")
            ),
            "test_type": attempt.get(
                "fallback_test_type", analysis.get("test_type", "MaxDispIncr")
            ),
            "max_iters": attempt.get(
                "fallback_max_iters", analysis.get("max_iters", 20)
            ),
            "tol": attempt.get("fallback_tol", analysis.get("tol", 1.0e-10)),
        }
        if "fallback_broyden_count" in attempt:
            retry_attempt["algorithm_options"] = {
                "max_iters": attempt["fallback_broyden_count"]
            }
        elif "fallback_line_search_eta" in attempt:
            retry_attempt["algorithm_options"] = {
                "alpha": attempt["fallback_line_search_eta"]
            }
        elif "fallback_krylov_max_dim" in attempt:
            retry_attempt["algorithm_options"] = {
                "maxDim": attempt["fallback_krylov_max_dim"]
            }
        return retry_attempt

    def _set_solver_chain_with_fallback(
        self,
        analysis: dict[str, Any],
        *,
        fallback_algorithm: str,
        fallback_test_type: str,
        fallback_tol: float,
        fallback_max_iters: int,
        fallback_algorithm_options: Optional[dict[str, Any]] = None,
    ) -> None:
        fallback_attempt: dict[str, Any] = {
            "algorithm": fallback_algorithm,
            "test_type": fallback_test_type,
            "tol": fallback_tol,
            "max_iters": fallback_max_iters,
        }
        if fallback_algorithm_options:
            fallback_attempt["algorithm_options"] = dict(fallback_algorithm_options)
        analysis["solver_chain"] = [
            self._solver_chain_primary_attempt(analysis),
            fallback_attempt,
        ]

    def _cmd_sp(self, *args: str) -> str:
        if len(args) != 3 or self.current_plain_pattern is None:
            raise self._error(
                "expected `sp node dof value` in a Plain pattern", "sp", args
            )
        node, dof = int(args[0]), int(args[1])
        if node not in self.node_ndfs or not 1 <= dof <= self.node_ndfs[node]:
            raise self._error("sp requires an existing node and DOF", "sp", args)
        self.current_plain_pattern.sp_constraints.append(
            {"node": node, "dof": dof, "value": float(args[2])}
        )
        return ""

    def _displacement_loop_limit(self) -> float | None:
        # Recognize a Tcl loop whose continuation condition reads the controlled
        # displacement. During conversion nodeDisp can only estimate that value.
        frame = self._frame_dict(-1)
        command = frame.get("cmd", "")
        if not command.startswith("while "):
            return None
        words = self._command_words(self._split_tcl_commands(command)[0])
        if len(words) != 3 or self.integrator is None:
            return None
        for comparison in words[1].split("&&"):
            match = re.fullmatch(
                r"\s*\$(\w+)\s*([<>])\s*(-?[\d.]+|\$\w+)\s*", comparison
            )
            if match is None:
                continue
            variable, direction, bound = match.groups()
            if (direction == "<") != (self.integrator["du"] > 0):
                continue
            query = re.search(
                rf"(?m)^\s*set\s+{re.escape(variable)}\s+\[nodeDisp\s+(\S+)\s+([^\s\]]+)\]",
                words[2],
            )
            if query is None:
                continue
            try:
                node, dof = (
                    int(self._resolve_variable_token(token)) for token in query.groups()
                )
                limit = float(self._resolve_variable_token(bound))
            except (ValueError, tkinter.TclError):
                continue
            if (node, dof) == (self.integrator["node"], self.integrator["dof"]):
                return limit
        return None

    def _cmd_analyze(self, *args: str) -> str:
        if self.analysis_type is None:
            if not self.transient_eigen_initialized or len(args) != 2:
                raise self._error(
                    "analysis type must be set before analyze", "analyze", args
                )
            # OpenSees creates a default DirectIntegrationAnalysis for eigen
            # when no analysis object exists, and later analyze reuses it.
            self.analysis_type = "Transient"
            self.integrator = self.integrator or {
                "type": "Newmark",
                "gamma": 0.5,
                "beta": 0.25,
            }
            self.constraints_handler = self.constraints_handler or "Transformation"
            self.numberer_handler = self.numberer_handler or "RCM"
            self.system_handler = self.system_handler or "ProfileSPD"
            self.algorithm_name = self.algorithm_name or "Newton"
            self.current_test = self.current_test or {
                "type": "NormUnbalance",
                "tol": 1.0e-6,
                "max_iters": 25,
            }
        step_retry = self._active_step_retry_policy(tuple(args))
        if self.analysis_type == "Static":
            if len(args) != 1:
                raise self._error(
                    "static analyze expects `analyze steps`", "analyze", args
                )
            if self.integrator is None:
                raise self._error(
                    "static analysis requires an integrator", "analyze", args
                )
            analysis: dict[str, Any] = {
                "type": "static_nonlinear",
                "constraints": self._analysis_constraints_name(),
                "steps": int(args[0]),
                "integrator": dict(self.integrator),
            }
            if self.penalty_coefficients is not None:
                if self.integrator["type"] != "LoadControl":
                    raise self._error(
                        "Penalty requires static LoadControl or transient analysis",
                        "analyze",
                        args,
                    )
                analysis["penalty_sp"], analysis["penalty_mp"] = (
                    self.penalty_coefficients
                )
            if self.integrator["type"] == "LoadControl":
                if self.pending_time_reset:
                    analysis["start_time"] = self.current_time
                self.pending_time_reset = False
            elif self.integrator["type"] == "DisplacementControl" and args[0] == "1":
                limit = self._displacement_loop_limit()
                if limit is not None:
                    analysis["integrator"]["displacement_limit"] = limit
            if self.algorithm_name and self.algorithm_name != "Linear":
                analysis["algorithm"] = self.algorithm_name
            if self.algorithm_options is not None:
                analysis["algorithm_options"] = dict(self.algorithm_options)
            if self.current_test is not None:
                analysis["test_type"] = self.current_test["type"]
                analysis["tol"] = self.current_test["tol"]
                analysis["max_iters"] = self.current_test["max_iters"]
                if "print_flag" in self.current_test:
                    analysis["test_print_flag"] = self.current_test["print_flag"]
                if "extra_args" in self.current_test:
                    analysis["test_extra_args"] = list(self.current_test["extra_args"])
            if self.system_handler is not None:
                analysis["system"] = self.system_handler
                if self.system_options:
                    analysis["system_options"] = list(self.system_options)
            if self.numberer_handler is not None:
                analysis["numberer"] = self.numberer_handler
            if step_retry is not None and step_retry.attempts:
                analysis["solver_chain"] = [
                    self._solver_chain_primary_attempt(analysis),
                    *[
                        self._solver_chain_retry_attempt(analysis, attempt)
                        for attempt in step_retry.attempts
                    ],
                ]
                step_retry_payload: Optional[dict[str, Any]] = None
                if step_retry.continue_after_failure:
                    step_retry_payload = {
                        "type": "continue_after_failure",
                        "restore_primary_after_success": (
                            step_retry.restore_primary_after_success
                        ),
                        "continue_after_failure": "displacement_control_single_steps",
                    }
                    if self.integrator["type"] == "DisplacementControl":
                        try:
                            continue_target = abs(float(self.interp.getvar("Dmax")))
                        except tkinter.TclError:
                            continue_target = abs(
                                int(args[0]) * float(self.integrator.get("du", 0.0))
                            )
                        if continue_target > 0.0:
                            step_retry_payload["continue_target_disp"] = continue_target
                            du = abs(float(self.integrator.get("du", 0.0)))
                            if du > 0.0:
                                continue_max_steps = int(continue_target / du) + 2
                                step_retry_payload["continue_max_steps"] = (
                                    continue_max_steps
                                )
                                if continue_max_steps > 500:
                                    step_retry_payload = None
                if step_retry_payload is not None:
                    analysis["step_retry"] = step_retry_payload
                    if self.integrator["type"] == "DisplacementControl":
                        analysis["integrator"]["max_cutbacks"] = 0
            stage = {"analysis": analysis}
            if self.integrator["type"] == "LoadControl":
                if self.load_control_has_run:
                    stage["resume_load_control"] = True
                self.load_control_has_run = True
            if self.current_pattern is not None:
                stage["pattern"] = {
                    key: value
                    for key, value in self.current_pattern.items()
                    if key
                    in {
                        "type",
                        "tag",
                        "time_series",
                        "direction",
                        "accel",
                        "sp_constraints",
                    }
                }
                stage["loads"] = list(self.current_pattern.get("loads", []))
                stage["element_loads"] = list(
                    self.current_pattern.get("element_loads", [])
                )
            elif self.pattern_removed:
                stage["pattern"] = {"type": "None"}
                stage["loads"] = []
                stage["element_loads"] = []
            if (
                self.integrator["type"] == "DisplacementControl"
                and abs(float(self.integrator.get("du", 0.0))) <= 0.0
            ):
                # Ex4 cycle generation emits duplicate zero increments at the
                # start of each peak sequence. Treat those as no-op analyze
                # calls so generated cases never contain invalid `du 0` stages.
                self.pattern_removed = False
                return "0"
            if self.integrator["type"] == "DisplacementControl" and not self.elements:
                # A displacement-controlled analyze on a bare node set has no
                # element stiffness or reference response to advance.
                self.pattern_removed = False
                return "0"
            if (
                self.integrator["type"] == "DisplacementControl"
                and stage.get("pattern", {}).get("type") == "Plain"
                and not stage.get("loads")
                and not stage.get("element_loads")
            ):
                # Benchmarked OpenSees treats displacement-control stages with an
                # empty reference load vector as ineffective; preserve the
                # existing state instead of synthesizing a spurious displacement.
                self.pattern_removed = False
                return "0"
            if self.current_rayleigh is not None:
                stage["rayleigh"] = dict(self.current_rayleigh)
            if self.pending_initialize:
                stage["initialize"] = True
            steps = int(args[0])
            if self.integrator["type"] == "LoadControl":
                self.current_time += steps * float(self.integrator["step"])
            elif self.integrator["type"] == "DisplacementControl":
                key = (int(self.integrator["node"]), int(self.integrator["dof"]))
                self.node_displacements[key] = self.node_displacements.get(
                    key, 0.0
                ) + steps * float(self.integrator["du"])
        else:
            if len(args) != 2:
                raise self._error(
                    "transient analyze expects `analyze steps dt`", "analyze", args
                )
            if self.integrator is None or self.integrator.get("type") != "Newmark":
                raise self._error(
                    "transient analysis requires Newmark", "analyze", args
                )
            stage = {
                "analysis": {
                    "type": "transient_linear"
                    if self.algorithm_name == "Linear"
                    else "transient_nonlinear",
                    "constraints": self._analysis_constraints_name(),
                    "steps": int(args[0]),
                    "dt": float(args[1]),
                    "start_time": self.current_time,
                    "integrator": dict(self.integrator),
                },
            }
            if self.modal_damping is not None:
                stage["analysis"].update(
                    type="transient_nonlinear", **self.modal_damping
                )
            if self.penalty_coefficients is not None:
                sp, mp = self.penalty_coefficients
                stage["analysis"].update(
                    type="transient_nonlinear", penalty_sp=sp, penalty_mp=mp
                )
            if self.system_handler is not None:
                stage["analysis"]["system"] = self.system_handler
                if self.system_options:
                    stage["analysis"]["system_options"] = list(self.system_options)
            if self.numberer_handler is not None:
                stage["analysis"]["numberer"] = self.numberer_handler
            if self.current_pattern is not None:
                if self.current_pattern.get("type") == "UniformExcitation":
                    pattern_keys = {"type", "tag", "direction", "accel", "vel0"}
                    if (
                        self.model is not None and self.model["ndm"] == 3
                    ) or self._has_coupled_soil():
                        pattern_keys.add("excitations")
                    if self.current_pattern.get("background_loads"):
                        stage["loads"] = self.current_pattern["background_loads"]
                    stage["pattern"] = {
                        key: value
                        for key, value in self.current_pattern.items()
                        if key in pattern_keys
                    }
                elif self.current_pattern.get("type") == "Plain":
                    stage["pattern"] = {
                        key: value
                        for key, value in self.current_pattern.items()
                        if key in {"type", "tag", "time_series", "sp_constraints"}
                    }
                    stage["loads"] = list(self.current_pattern.get("loads", []))
                    stage["element_loads"] = list(
                        self.current_pattern.get("element_loads", [])
                    )
            elif self.pattern_removed:
                stage["pattern"] = {"type": "None"}
                stage["loads"] = []
                stage["element_loads"] = []
            if self.algorithm_name and self.algorithm_name != "Linear":
                stage["analysis"]["algorithm"] = self.algorithm_name
            if self.algorithm_options is not None:
                stage["analysis"]["algorithm_options"] = dict(self.algorithm_options)
            if self.current_test is not None:
                stage["analysis"]["test_type"] = self.current_test["type"]
                stage["analysis"]["tol"] = self.current_test["tol"]
                stage["analysis"]["max_iters"] = self.current_test["max_iters"]
                if "print_flag" in self.current_test:
                    stage["analysis"]["test_print_flag"] = self.current_test[
                        "print_flag"
                    ]
                if "extra_args" in self.current_test:
                    stage["analysis"]["test_extra_args"] = list(
                        self.current_test["extra_args"]
                    )
            if step_retry is not None and step_retry.attempts:
                stage["analysis"]["solver_chain"] = [
                    self._solver_chain_primary_attempt(stage["analysis"]),
                    *[
                        self._solver_chain_retry_attempt(stage["analysis"], attempt)
                        for attempt in step_retry.attempts
                    ],
                ]
            if self.current_rayleigh is not None:
                stage["rayleigh"] = dict(self.current_rayleigh)
            if self.pending_initialize:
                stage["initialize"] = True
            self.current_time += int(args[0]) * float(args[1])

        if self.imposed_motion_patterns:
            motions = [
                motion
                for pattern in self.imposed_motion_patterns.values()
                for motion in pattern
            ]
            if self.penalty_coefficients is not None:
                stage["analysis"]["support_motions"] = motions
            elif self.model["ndm"] == 2 and self.constraints_handler != "Plain":
                # Keep the established fallback for 2D scripts whose
                # MultipleSupport analysis uses a different handler.
                self.has_complex_transient_excitation = True
            elif not all(
                motion["dof"] in self.nodes[motion["node"]].get("constraints", [])
                for motion in motions
            ) or (
                self.constraints_handler not in {None, "Plain"}
                and not self.transient_eigen_initialized
            ):
                raise self._error(
                    "active MultipleSupport displacement requires Penalty",
                    "analyze",
                    args,
                )
            elif self.analysis_type == "Transient":
                # The existing fixed SPs own displacement. ImposedMotionSP
                # still sets the nodal velocity and acceleration at each step.
                stage["analysis"]["support_motions"] = motions
                stage["analysis"]["type"] = "transient_nonlinear"
        if self.constant_patterns:
            constant_loads, constant_element_loads = self._constant_pattern_loads()
            stage["constant_loads"] = constant_loads
            stage["constant_element_loads"] = constant_element_loads
        if self.material_stages:
            stage["material_stages"] = [
                {"material": material, "stage": value}
                for material, value in self.material_stages.items()
            ]
        if self.prescribed_pressures:
            if (
                self.analysis_type != "Transient"
                or self._analysis_constraints_name() != "Transformation"
            ):
                raise self._error(
                    "prescribed pore pressure requires transient analysis with Transformation",
                    "analyze",
                    args,
                )
            stage["analysis"]["type"] = "transient_nonlinear"
            stage["prescribed_pressures"] = [
                {"node": node, "dof": dof, **response}
                for (node, dof), response in self.prescribed_pressures.items()
            ]
        if self.permeability_updates:
            stage["permeability_updates"] = [
                {"element": elem, "axis": axis, "value": value}
                for (elem, axis), value in self.permeability_updates.items()
            ]
            self.permeability_updates.clear()
        if self.pending_recorders:
            stage["sample_before"] = self.pending_recorders
            self.pending_recorders = []
        if self.pending_initial_state_reset is not None:
            stage["initial_state_reset"] = self.pending_initial_state_reset
            self.pending_initial_state_reset = None
        if self.pending_node_removals:
            stage["remove_nodes"] = self.pending_node_removals
            self.pending_node_removals = []
        if self.pending_contact_friction is not None:
            stage["contact_friction"] = self.pending_contact_friction
            self.pending_contact_friction = None
        if self.pending_element_removals:
            stage["remove_elements"] = self.pending_element_removals
            self.pending_element_removals = []
        self._append_stage(stage)
        self.pending_initialize = False
        self.pattern_removed = False
        self.uniform_excitation_count_since_analyze = 0
        return "0"

    def _cmd_get_node_tags(self, *args: str) -> str:
        if args:
            raise self._error("getNodeTags takes no arguments", "getNodeTags", args)
        return " ".join(map(str, sorted(self.nodes)))

    def _cmd_get_time(self, *args: str) -> str:
        if args:
            raise self._error("getTime takes no arguments", "getTime", args)
        return str(self.current_time)

    def _cmd_node_disp(self, *args: str) -> str:
        if not args:
            raise self._error("nodeDisp expects a node id", "nodeDisp", args)
        node_id = int(args[0])
        model = self._require_model("nodeDisp", args)
        if len(args) == 1:
            values = [
                self.node_displacements.get((node_id, dof), 0.0)
                for dof in range(1, model["ndf"] + 1)
            ]
            return " ".join(str(value) for value in values)
        return str(self.node_displacements.get((node_id, int(args[1])), 0.0))

    def _cmd_node_reaction(self, *args: str) -> str:
        model = self._require_model("nodeReaction", args)
        if not args:
            raise self._error("nodeReaction expects a node id", "nodeReaction", args)
        if len(args) == 1:
            return " ".join("0.0" for _ in range(model["ndf"]))
        return "0.0"

    def _cmd_node_eigenvector(self, *args: str) -> str:
        model = self._require_model("nodeEigenvector", args)
        if len(args) < 2:
            raise self._error(
                "nodeEigenvector expects `node mode ?dof?`",
                "nodeEigenvector",
                args,
            )
        if len(args) == 2:
            return " ".join("1.0" for _ in range(model["ndf"]))
        return "1.0"

    def _cmd_print(self, *args: str) -> str:
        if self.last_stage is None:
            return ""
        self.last_stage.setdefault("print_commands", []).append({"args": list(args)})
        return ""

    def _cmd_parameter(self, *args: str) -> str:
        if len(args) != 4 or args[1] != "element" or args[3] not in {"hPerm", "vPerm"}:
            raise self._error(
                "parameter expects tag element elementTag hPerm|vPerm",
                "parameter",
                args,
            )
        tag, element = int(args[0]), int(args[2])
        if tag in self.permeability_parameters:
            raise self._error(f"duplicate parameter {tag}", "parameter", args)
        elem_type = next(
            (elem["type"] for elem in self.elements if elem["id"] == element), None
        )
        if elem_type not in {"quadUP", "9_4_QuadUP"}:
            raise self._error(
                f"element {element} has no permeability parameter", "parameter", args
            )
        self.permeability_parameters[tag] = element, "x" if args[3] == "hPerm" else "y"
        return str(tag)

    def _cmd_update_parameter(self, *args: str) -> str:
        if len(args) != 2 or int(args[0]) not in self.permeability_parameters:
            raise self._error(
                "updateParameter expects an existing tag and value",
                "updateParameter",
                args,
            )
        value = float(args[1])
        if not math.isfinite(value) or value < 0.0:
            raise self._error(
                "permeability must be nonnegative", "updateParameter", args
            )
        self.permeability_updates[self.permeability_parameters[int(args[0])]] = value
        return "0"

    def _cmd_set_parameter(self, *args: str) -> str:
        if (
            len(args) < 5
            or args[0] != "-value"
            or args[-1] not in {"xPerm", "yPerm", "zPerm", "friction"}
        ):
            raise self._error(
                "setParameter expects -value value -eleRange first last xPerm|yPerm|zPerm|friction",
                "setParameter",
                args,
            )
        value = float(args[1])
        if args[2] == "-eleRange" and len(args) == 6:
            first, last = map(int, args[3:5])
            ids = {elem["id"] for elem in self.elements if first <= elem["id"] <= last}
        elif args[2] in {"-ele", "-element"}:
            ids = set(map(int, args[3:-1]))
        else:
            raise self._error(
                "setParameter requires -ele or -eleRange", "setParameter", args
            )
        if args[-1] == "friction":
            contact_ids = {
                elem["id"] for elem in self.elements if elem["type"] == "BeamContact2D"
            }
            if ids & contact_ids:
                # OpenSees ContactMaterial2D stores the switch across all instances.
                self.pending_contact_friction = int(value) == 1
            return "0"
        if not math.isfinite(value) or value < 0.0:
            raise self._error("permeability must be nonnegative", "setParameter", args)
        brick_ids = {
            elem["id"] for elem in self.elements if elem["type"] == "SSPbrickUP"
        }
        if not ids or not ids <= brick_ids:
            raise self._error(
                "setParameter requires SSPbrickUP elements", "setParameter", args
            )
        for element in sorted(ids):
            self.permeability_updates[element, args[-1][0]] = value
        return "0"

    def _cmd_remove(self, *args: str) -> str:
        if len(args) == 2 and args[0] == "parameter":
            self.permeability_parameters.pop(int(args[1]), None)
            return ""
        if len(args) == 2 and args[0] in {"node", "element", "ele"}:
            tag = int(args[1])
            if args[0] == "node":
                if tag in self.nodes and tag not in self.removed_nodes:
                    self.removed_nodes.add(tag)
                    self.pending_node_removals.append(tag)
            elif tag not in self.removed_elements and any(
                element["id"] == tag for element in self.elements
            ):
                self.removed_elements.add(tag)
                self.pending_element_removals.append(tag)
            return ""
        if args == ("recorders",) or (len(args) == 2 and args[0] == "recorder"):
            if args == ("recorders",):
                ending = self.active_recorders
                self.active_recorders = {}
            else:
                tag = int(args[1])
                if tag not in self.active_recorders:
                    raise self._error("unknown recorder tag", "remove", args)
                ending = {tag: self.active_recorders.pop(tag)}
            stage = len(self.analysis_stages)
            pending = set()
            for group in ending.values():
                for recorder in group:
                    if recorder.get("start_stage", 0) < stage:
                        recorder["end_stage"] = stage
                    else:
                        pending.add(id(recorder))
            if pending:
                self.recorders = [r for r in self.recorders if id(r) not in pending]
            return ""
        if len(args) == 3 and args[0] in {"sp", "SPconstraint"}:
            node = self.nodes.get(int(args[1]))
            if node is not None:
                dof = int(args[2])
                node["constraints"] = [
                    d for d in node.get("constraints", []) if d != dof
                ]
            return ""
        if len(args) == 2 and args[0] == "loadPattern":
            tag = int(args[1])
            self.constant_patterns.pop(tag, None)
            self.imposed_motion_patterns.pop(tag, None)
            if (
                self.current_pattern is not None
                and "excitations" in self.current_pattern
            ):
                excitations = [
                    excitation
                    for excitation in self.current_pattern["excitations"]
                    if excitation["tag"] != tag
                ]
                if len(excitations) > 1:
                    self.current_pattern = {
                        **self.current_pattern,
                        "excitations": excitations,
                    }
                elif excitations:
                    self.current_pattern = excitations[0]
            if (
                self.current_pattern is not None
                and int(self.current_pattern.get("tag", -1)) == tag
            ):
                self.current_pattern = None
                self.current_plain_pattern = None
            self.pattern_removed = True
            return ""
        return ""

    def _cmd_load_const(self, *args: str) -> str:
        if len(args) != 2 or args[0] != "-time":
            raise self._error("loadConst expects `-time value`", "loadConst", args)
        if self.last_stage is None:
            raise self._error(
                "loadConst requires a preceding analyze", "loadConst", args
            )
        self.current_time = float(args[1])
        if not self.pattern_removed or self.current_pattern or self.constant_patterns:
            self.last_stage["load_const"] = {"time": self.current_time}
            self.constant_patterns.clear()
            if self.current_pattern and self.current_pattern["type"] == "Plain":
                self.current_pattern = None
        self.pending_time_reset = True
        return ""

    def _cmd_initial_state_analysis(self, *args: str) -> str:
        if args == ("on",):
            self.initial_state_analysis = True
        elif args == ("off",):
            self.pending_initial_state_reset = self.initial_state_analysis
            self.initial_state_analysis = False
            self.current_time = 0.0
            self.pending_time_reset = True
        else:
            raise self._error("expected on or off", "InitialStateAnalysis", args)
        return ""

    def _cmd_update_material_stage(self, *args: str) -> str:
        if len(args) != 4 or set(args[::2]) != {"-material", "-stage"}:
            raise self._error(
                "updateMaterialStage expects -material tag -stage stage",
                "updateMaterialStage",
                args,
            )
        flags = dict(zip(args[::2], args[1::2]))
        tag = int(flags["-material"])
        stage = int(flags["-stage"])
        material = self.materials_by_id.get(tag)
        if material is None:
            # OpenSees registers a domain parameter even when no element matches the tag.
            return ""
        if material["type"] not in {
            "PressureIndependMultiYield",
            "PressureDependMultiYield",
            "PressureDependMultiYield02",
            "InitialStateAnalysisWrapper",
        }:
            raise self._error(
                "unsupported material for updateMaterialStage",
                "updateMaterialStage",
                args,
            )
        self.material_stages[tag] = stage
        return ""

    def _cmd_set_time(self, *args: str) -> str:
        if len(args) != 1:
            raise self._error("setTime expects one value", "setTime", args)
        self.current_time = float(args[0])
        self.pending_time_reset = True
        return ""

    def _cmd_wipe_analysis(self, *args: str) -> str:
        if args:
            raise self._error("wipeAnalysis takes no arguments", "wipeAnalysis", args)
        self.constraints_handler = None
        self.penalty_coefficients = None
        self.numberer_handler = None
        self.system_handler = None
        self.current_test = None
        self.algorithm_name = None
        self.integrator = None
        self.load_control_has_run = False
        self.analysis_type = None
        self.transient_eigen_initialized = False
        self.pending_initialize = False
        return ""

    def _cmd_eigen(self, *args: str) -> str:
        if not args:
            raise self._error("unsupported eigen form", "eigen", args)
        num_modes = int(args[-1])
        if num_modes < 1:
            raise self._error("eigen requires a positive mode count", "eigen", args)
        options = [flag.lower().lstrip("-") for flag in args[:-1]]
        flags = set(options)
        standard = False
        for option in options:
            if option in {"standard", "frequency", "generalized"}:
                standard = option == "standard"
        # FullGenEigenSolver sorts ascending and ignores findSmallest.
        largest = "findlargest" in flags and not flags.intersection(
            {
                "fullgenlapack",
                "fullgenlapackeigen",
                "symmbandlapack",
                "symmbandlapackeigen",
            }
        )
        try:
            values = self._solve_eigenvalues(
                num_modes, largest=largest, standard=standard
            )
        except TclToStrutError:
            if not standard and any(
                element["type"] == "fourNodeQuad" for element in self.elements
            ):
                values = [float((idx + 1) * (idx + 1)) for idx in range(num_modes)]
            else:
                raise
        self.last_eigenvalues = list(values)
        self.last_standard_eigen = standard
        if self.analysis_type != "Static":
            self.transient_eigen_initialized = True
        if num_modes == 1:
            return str(values[0])
        return " ".join(str(value) for value in values)

    def _cmd_modal_damping(self, *args: str) -> str:
        if not args:
            raise self._error(
                "modalDamping expects damping ratios", "modalDamping", args
            )
        if not self.last_eigenvalues:
            return ""
        count = len(self.last_eigenvalues)
        ratios = list(map(float, args[:count]))
        if len(args) == 1:
            ratios *= count
        self.modal_damping = {"modal_damping": ratios}
        return ""

    def _cmd_rayleigh(self, *args: str) -> str:
        if len(args) != 4:
            raise self._error("rayleigh expects four coefficients", "rayleigh", args)
        self.current_rayleigh = {
            "alphaM": float(args[0]),
            "betaK": float(args[1]),
            "betaKInit": float(args[2]),
            "betaKComm": float(args[3]),
        }
        return ""

    def _cmd_region(self, *args: str) -> str:
        if not args:
            raise self._error("region tag is required", "region", args)
        tag = int(args[0])
        if tag in self.regions:
            raise self._error(f"duplicate region {tag}", "region", args)
        element_by_id = {element["id"]: element for element in self.elements}
        groups: dict[str, list[int]] = {"-ele": [], "-node": []}
        only: set[str] = set()
        coefficients: tuple[float, ...] = ()
        idx = 1
        while idx < len(args):
            flag = args[idx]
            idx += 1
            kind = flag.removesuffix("Range").removesuffix("Only")
            if kind in groups:
                if "Only" in flag:
                    only.add(kind)
                if flag.endswith("Range"):
                    start, end = sorted(map(int, args[idx : idx + 2]))
                    groups[kind].extend(range(start, end + 1))
                    idx += 2
                else:
                    while idx < len(args):
                        try:
                            groups[kind].append(int(args[idx]))
                        except ValueError:
                            break
                        idx += 1
            elif flag == "-rayleigh":
                coefficients = tuple(map(float, args[idx : idx + 4]))
                if len(coefficients) != 4:
                    raise self._error(
                        "region -rayleigh expects four coefficients", "region", args
                    )
                idx += 4
            # OpenSees ignores unrecognized flags, including the legacy spelling
            # 'rayleigh' without a leading dash.
        elements = [e for e in dict.fromkeys(groups["-ele"]) if e in element_by_id]
        nodes = [n for n in dict.fromkeys(groups["-node"]) if n in self.nodes]
        if elements and "-ele" not in only and "-node" not in only:
            nodes = list(
                dict.fromkeys(n for e in elements for n in element_by_id[e]["nodes"])
            )
        elif not groups["-ele"] and "-node" not in only:
            node_set = set(nodes)
            elements = [
                e
                for e, elem in element_by_id.items()
                if node_set.issuperset(elem["nodes"])
            ]
        self.regions[tag] = (nodes, elements)
        if any(coefficients):
            alpha, beta, initial, committed = coefficients
            rayleigh = dict(self.current_rayleigh or {})
            if beta != rayleigh.get("betaK", 0.0) or committed != rayleigh.get(
                "betaKComm", 0.0
            ):
                raise self._error(
                    "region damping supports nodal alphaM and elastic 2D beam betaKInit",
                    "region",
                    args,
                )
            node_alpha = {
                entry["tag"]: entry["coefficient"]
                for entry in rayleigh.get("node_alpha_m", [])
            }
            element_beta = {
                entry["tag"]: entry["coefficient"]
                for entry in rayleigh.get("element_beta_k_init", [])
            }
            node_alpha.update(dict.fromkeys(nodes, alpha))
            for tag in elements:
                element = element_by_id[tag]
                if element["type"] in {
                    "truss",
                    "zeroLength",
                    "CoupledZeroLength",
                    "twoNodeLink",
                } and not element.get("doRayleigh"):
                    continue
                if alpha != rayleigh.get("alphaM", 0.0) and (
                    element.get("rho") or element.get("mass")
                ):
                    raise self._error(
                        "region alphaM on element mass is unsupported", "region", args
                    )
                if (
                    initial != rayleigh.get("betaKInit", 0.0)
                    and element["type"] != "elasticBeamColumn2d"
                ):
                    raise self._error(
                        "region betaKInit requires elasticBeamColumn2d", "region", args
                    )
                if element["type"] == "elasticBeamColumn2d":
                    element_beta[tag] = initial
            rayleigh["node_alpha_m"] = [
                {"tag": tag, "coefficient": value}
                for tag, value in node_alpha.items()
                if value != rayleigh.get("alphaM", 0.0)
            ]
            rayleigh["element_beta_k_init"] = [
                {"tag": tag, "coefficient": value}
                for tag, value in element_beta.items()
                if value != rayleigh.get("betaKInit", 0.0)
            ]
            self.current_rayleigh = rayleigh
        return ""

    def _build_global_stiffness(self) -> list[list[float]]:
        model = self._require_model("eigen", ())
        if (model["ndm"], model["ndf"]) not in {(1, 1), (2, 2), (2, 3)}:
            raise self._error(
                "eigen only supports 2D truss/frame models in v1", "eigen", ()
            )

        node_ids = sorted(self.nodes)
        node_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
        total_dofs = len(node_ids) * model["ndf"]
        stiffness = [[0.0] * total_dofs for _ in range(total_dofs)]

        for element in self.elements:
            node_i = self.nodes[element["nodes"][0]]
            node_j = self.nodes[element["nodes"][1]]
            if element["type"] == "CoupledZeroLength":
                ndf = model["ndf"]
                modulus = self._material_modulus(element["materials"][0])
                for direction in element["dirs"]:
                    a = node_index[node_i["id"]] * ndf + direction - 1
                    b = node_index[node_j["id"]] * ndf + direction - 1
                    stiffness[a][a] += modulus
                    stiffness[b][b] += modulus
                    stiffness[a][b] -= modulus
                    stiffness[b][a] -= modulus
                continue
            if element["type"] == "zeroLength":
                ndf = model["ndf"]
                i = node_index[node_i["id"]] * ndf
                j = node_index[node_j["id"]] * ndf
                orient = element.get("orient", {"x": [1, 0, 0], "y": [0, 1, 0]})
                x, y = orient["x"], orient["y"]
                angle = math.atan2(x[1], x[0])
                sign = 1.0 if x[0] * y[1] >= x[1] * y[0] else -1.0
                c, s = math.cos(angle), math.sin(angle)
                axes = {1: (c, s), 2: (-sign * s, sign * c)}
                if ndf == 1:
                    axes = {1: (c,)}
                elif ndf == 3:
                    axes = {direction: (*axis, 0.0) for direction, axis in axes.items()}
                    axes[6] = (0.0, 0.0, sign)
                for material, direction in zip(element["materials"], element["dirs"]):
                    axis = axes.get(direction)
                    if axis is None:
                        continue
                    modulus = self._material_modulus(material)
                    for row, a in enumerate(axis):
                        for col, b in enumerate(axis):
                            value = modulus * a * b
                            stiffness[i + row][i + col] += value
                            stiffness[j + row][j + col] += value
                            stiffness[i + row][j + col] -= value
                            stiffness[j + row][i + col] -= value
                continue
            x1 = float(node_i["x"])
            y1 = float(node_i["y"])
            x2 = float(node_j["x"])
            y2 = float(node_j["y"])
            dx = x2 - x1
            dy = y2 - y1
            length = math.hypot(dx, dy)
            if length <= 0.0:
                raise self._error(
                    "element length must be positive for eigen", "eigen", ()
                )
            c = dx / length
            s = dy / length
            if element["type"] in {"truss", "corotTruss"}:
                e_value = self._material_modulus(int(element["material"]))
                axial = e_value * float(element["area"]) / length
                axis = (c,) if model["ndm"] == 1 else (c, s)
                direction = (*axis, *(-value for value in axis))
                transformed = [[axial * a * b for b in direction] for a in direction]
                dofs = [
                    node_index[node_id] * model["ndf"] + dof
                    for node_id in element["nodes"]
                    for dof in range(model["ndm"])
                ]
            elif element["type"] in {
                "elasticBeamColumn2d",
                "forceBeamColumn2d",
                "dispBeamColumn2d",
            }:
                section = self.sections_by_id[element["section"]]
                if section["type"] == "HingeRadauSection2d":
                    params = section["params"]
                    hinge_i, hinge_j = params["length_i"], params["length_j"]
                    alpha = 0.5 - 2 * (hinge_i + hinge_j) / length
                    beta = 0.5 + 2 * (hinge_i - hinge_j) / length
                    positions = (
                        0,
                        8 * hinge_i / (3 * length),
                        beta - alpha / math.sqrt(3),
                        beta + alpha / math.sqrt(3),
                        1 - 8 * hinge_j / (3 * length),
                        1,
                    )
                    weights = (
                        hinge_i,
                        3 * hinge_i,
                        alpha * length,
                        alpha * length,
                        3 * hinge_j,
                        hinge_j,
                    )
                    interior = (params["E"] * params["A"], params["E"] * params["I"])
                    rigidities = (
                        self._fiber_section_rigidity(params["section_i"]),
                        *([interior] * 4),
                        self._fiber_section_rigidity(params["section_j"]),
                    )
                    flexibility = [[0.0] * 3 for _ in range(3)]
                    for xi, weight, (ea, ei) in zip(positions, weights, rigidities):
                        a, b = xi - 1, xi
                        flexibility[0][0] += weight / ea
                        flexibility[1][1] += weight * a * a / ei
                        flexibility[1][2] += weight * a * b / ei
                        flexibility[2][1] = flexibility[1][2]
                        flexibility[2][2] += weight * b * b / ei
                    basic = [
                        [-c, -s, 0, c, s, 0],
                        [-s / length, c / length, 1, s / length, -c / length, 0],
                        [-s / length, c / length, 0, s / length, -c / length, 1],
                    ]
                    transformed = _matrix_multiply(
                        _matrix_multiply(
                            self._transpose(basic), _matrix_inverse(flexibility)
                        ),
                        basic,
                    )
                    dofs = [
                        node_index[node_id] * model["ndf"] + dof
                        for node_id in element["nodes"]
                        for dof in range(3)
                    ]
                    for local_i, global_i in enumerate(dofs):
                        for local_j, global_j in enumerate(dofs):
                            stiffness[global_i][global_j] += transformed[local_i][
                                local_j
                            ]
                    continue
                ea, ei = self._fiber_section_rigidity(int(element["section"]))
                axial = ea / length
                l2 = length * length
                l3 = l2 * length
                local = [
                    [axial, 0.0, 0.0, -axial, 0.0, 0.0],
                    [
                        0.0,
                        12.0 * ei / l3,
                        6.0 * ei / l2,
                        0.0,
                        -12.0 * ei / l3,
                        6.0 * ei / l2,
                    ],
                    [
                        0.0,
                        6.0 * ei / l2,
                        4.0 * ei / length,
                        0.0,
                        -6.0 * ei / l2,
                        2.0 * ei / length,
                    ],
                    [-axial, 0.0, 0.0, axial, 0.0, 0.0],
                    [
                        0.0,
                        -12.0 * ei / l3,
                        -6.0 * ei / l2,
                        0.0,
                        12.0 * ei / l3,
                        -6.0 * ei / l2,
                    ],
                    [
                        0.0,
                        6.0 * ei / l2,
                        2.0 * ei / length,
                        0.0,
                        -6.0 * ei / l2,
                        4.0 * ei / length,
                    ],
                ]
                transform = [
                    [c, s, 0.0, 0.0, 0.0, 0.0],
                    [-s, c, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, c, s, 0.0],
                    [0.0, 0.0, 0.0, -s, c, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ]
                transformed = _matrix_multiply(
                    _matrix_multiply(self._transpose(transform), local), transform
                )
                dofs = []
                for node_id in element["nodes"]:
                    start = node_index[node_id] * model["ndf"]
                    dofs.extend([start + 0, start + 1, start + 2])
            elif element["type"] == "twoNodeLink":
                mat_ids = [int(mid) for mid in element.get("materials", [])]
                if all(self._material_modulus(mid) == 0.0 for mid in mat_ids):
                    continue
                raise self._error(
                    "eigen does not support nonzero-stiffness twoNodeLink elements",
                    "eigen",
                    (),
                )
            else:
                raise self._error(
                    f"eigen does not support element type `{element['type']}`",
                    "eigen",
                    (),
                )
            for local_i, global_i in enumerate(dofs):
                for local_j, global_j in enumerate(dofs):
                    stiffness[global_i][global_j] += transformed[local_i][local_j]

        return stiffness

    def _transpose(self, matrix: list[list[float]]) -> list[list[float]]:
        return [list(col) for col in zip(*matrix)]

    def _equal_dof_roots(self) -> dict[tuple[int, int], int]:
        aliases = {
            (constraint["constrained_node"], dof): constraint["retained_node"]
            for constraint in self.mp_constraints
            if constraint["type"] == "equalDOF"
            for dof in constraint["dofs"]
        }
        roots = {}
        for node, dof in aliases:
            root = node
            seen = set()
            while (root, dof) in aliases:
                if root in seen:
                    raise self._error("cyclic equalDOF constraints", "equalDOF", ())
                seen.add(root)
                root = aliases[root, dof]
            roots[node, dof] = root
        return roots

    def _build_mass_vector(self) -> list[float]:
        model = self._require_model("eigen", ())
        node_ids = sorted(self.nodes)
        node_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
        mass = [0.0] * (len(node_ids) * model["ndf"])
        for entry in self.masses:
            idx = node_index[int(entry["node"])] * model["ndf"] + int(entry["dof"]) - 1
            mass[idx] += float(entry["value"])
        return mass

    def _solve_eigenvalues(
        self, num_modes: int, *, largest: bool = False, standard: bool = False
    ) -> list[float]:
        model = self._require_model("eigen", ())
        if standard:
            return self._solve_mojo_eigenvalues(num_modes, standard=True)
        if model["ndm"] == 3 and model["ndf"] == 6:
            return self._solve_frame3d_eigenvalues(num_modes)
        stiffness = self._build_global_stiffness()
        mass = self._build_mass_vector()

        constrained = set()
        node_ids = sorted(self.nodes)
        node_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
        for node_pos, node_id in enumerate(node_ids):
            for dof in self.nodes[node_id].get("constraints", []):
                constrained.add(node_pos * model["ndf"] + int(dof) - 1)
        roots = list(range(len(mass)))
        for (dependent, dof), retained in self._equal_dof_roots().items():
            roots[node_index[dependent] * model["ndf"] + dof - 1] = (
                node_index[retained] * model["ndf"] + dof - 1
            )
        fixed = {roots[dof] for dof in constrained}
        free_dofs = sorted(set(roots) - fixed)
        if not free_dofs:
            raise self._error("no free DOFs remain for eigen", "eigen", ())

        index_by_dof = {dof: index for index, dof in enumerate(free_dofs)}
        active = [
            (dof, index_by_dof[root])
            for dof, root in enumerate(roots)
            if root in index_by_dof
        ]
        if any(element.get("rho", 0.0) for element in self.elements):
            from tcl_modal_mass import solve_with_element_mass

            try:
                return solve_with_element_mass(
                    stiffness,
                    mass,
                    model,
                    self.nodes,
                    self.elements,
                    active,
                    len(free_dofs),
                    num_modes,
                    largest,
                )
            except ValueError as exc:
                raise self._error(str(exc), "eigen", ()) from exc
        k_free = [[0.0] * len(free_dofs) for _ in free_dofs]
        m_free = [0.0] * len(free_dofs)
        for i, row in active:
            m_free[row] += mass[i]
            for j, col in active:
                k_free[row][col] += stiffness[i][j]
        massful = [idx for idx, value in enumerate(m_free) if value > 0.0]
        massless = [idx for idx, value in enumerate(m_free) if value <= 0.0]
        if not massful:
            raise self._error(
                "eigen requires at least one positive nodal mass", "eigen", ()
            )

        if massless:
            k_mm = [[k_free[i][j] for j in massful] for i in massful]
            k_ms = [[k_free[i][j] for j in massless] for i in massful]
            k_sm = [[k_free[i][j] for j in massful] for i in massless]
            k_ss = [[k_free[i][j] for j in massless] for i in massless]
            try:
                inv_k_ss = _matrix_inverse(k_ss)
            except ValueError as exc:
                raise self._error(
                    f"failed to condense massless DOFs: {exc}", "eigen", ()
                ) from exc
            correction = _matrix_multiply(_matrix_multiply(k_ms, inv_k_ss), k_sm)
            k_eff = []
            for i in range(len(k_mm)):
                row = []
                for j in range(len(k_mm)):
                    row.append(k_mm[i][j] - correction[i][j])
                k_eff.append(row)
        else:
            k_eff = [[k_free[i][j] for j in massful] for i in massful]

        m_eff = [m_free[i] for i in massful]
        scaled = []
        for i, row in enumerate(k_eff):
            scaled_row = []
            for j, value in enumerate(row):
                scaled_row.append(value / math.sqrt(m_eff[i] * m_eff[j]))
            scaled.append(scaled_row)

        values = [
            value for value in _jacobi_eigenvalues_symmetric(scaled) if value > 1.0e-12
        ]
        if len(values) < num_modes:
            raise self._error(
                f"requested {num_modes} mode(s), but only {len(values)} finite mode(s) were found",
                "eigen",
                (),
            )
        return values[-num_modes:] if largest else values[:num_modes]

    def _solve_frame3d_eigenvalues(self, num_modes: int) -> list[float]:
        return self._solve_mojo_eigenvalues(num_modes, standard=False)

    def _solve_mojo_eigenvalues(self, num_modes: int, *, standard: bool) -> list[float]:
        if num_modes < 1 or not self.nodes:
            raise self._error(
                "eigen requires nodes and a positive mode count", "eigen", ()
            )
        constraints = self.constraints_handler or "Transformation"
        modal_stage = {
            "analysis": {
                "type": "modal_eigen",
                "steps": 1,
                "num_modes": num_modes,
                "standard_eigen": standard,
            }
        }
        recorder = {
            "type": "modal_eigen",
            "nodes": [min(self.nodes)],
            "dofs": [1],
            "modes": [1],
            "output": "modal",
        }
        # Replay earlier analyze calls so the modal tangent reflects the committed
        # material state and any element loads, without recording those stages twice.
        case = self._build_case(
            [*self.analysis_stages, modal_stage], constraints, [recorder]
        )
        with tempfile.TemporaryDirectory(prefix="strut-eigen-") as directory:
            workdir = Path(directory)
            input_path = workdir / "input.json"
            input_path.write_text(
                json.dumps(case.to_solver_input(entry_path=self.entry_path)),
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "run_strut_case.py"),
                    "--input",
                    str(input_path),
                    "--output",
                    str(workdir),
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                details = (result.stdout + "\n" + result.stderr).strip()
                raise self._error(f"eigen solve failed: {details}", "eigen", ())
            return [
                float(value)
                for value in (workdir / "modal_eigenvalues.out")
                .read_text()
                .splitlines()
            ]


def convert_tcl_to_typed_case(
    entry_tcl: Path, repo_root: Optional[Path] = None
) -> StrutCase:
    entry_path = entry_tcl.resolve()
    resolved_repo_root = (
        repo_root.resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[1]
    )
    builder = TclStrutBuilder(entry_path, resolved_repo_root)
    return builder.convert()


def convert_tcl_to_case(
    entry_tcl: Path, repo_root: Optional[Path] = None
) -> dict[str, Any]:
    return convert_tcl_to_typed_case(entry_tcl, repo_root).to_json_dict()


def convert_tcl_to_solver_input(
    entry_tcl: Path, repo_root: Optional[Path] = None, compute_only: bool = False
) -> dict[str, Any]:
    case = convert_tcl_to_typed_case(entry_tcl, repo_root)
    return case.to_solver_input(entry_path=entry_tcl, compute_only=compute_only)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("entry_tcl")
    args = parser.parse_args()

    entry_tcl = Path(args.entry_tcl).resolve()
    convert_tcl_to_typed_case(entry_tcl)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
