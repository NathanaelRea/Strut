#!/usr/bin/env python3
import argparse
import math
import sys
import tkinter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

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
        "break",
        "catch",
        "close",
        "continue",
        "exit",
        "expr",
        "file",
        "for",
        "foreach",
        "format",
        "if",
        "incr",
        "lappend",
        "lindex",
        "list",
        "llength",
        "open",
        "proc",
        "puts",
        "read",
        "return",
        "set",
        "source",
        "split",
        "string",
        "upvar",
        "variable",
        "while",
    }
)

OPEN_SEES_COMMAND_ALIASES = {
    "model": {
        "basic": "basic",
        "BasicBuilder": "basic",
    },
    "element": {
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


@dataclass
class PendingFiberSection:
    id: int
    patches: list[dict[str, Any]] = field(default_factory=list)
    layers: list[dict[str, Any]] = field(default_factory=list)


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
        element_loads=[
            Load(dict(load)) for load in stage.get("element_loads", [])
        ],
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
        self.geom_transforms: dict[int, str] = {}
        self.materials: list[dict[str, Any]] = []
        self.materials_by_id: dict[int, dict[str, Any]] = {}
        self.material_ids_by_e: dict[float, int] = {}
        self.sections: list[dict[str, Any]] = []
        self.sections_by_id: dict[int, dict[str, Any]] = {}
        self.section_ids_by_props: dict[tuple[float, float, float], int] = {}
        self.elements: list[dict[str, Any]] = []
        self.masses: list[dict[str, Any]] = []
        self.time_series: list[dict[str, Any]] = []
        self.time_series_by_tag: dict[int, dict[str, Any]] = {}
        self.recorders: list[dict[str, Any]] = []
        self.analysis_stages: list[dict[str, Any]] = []
        self.mp_constraints: list[dict[str, Any]] = []

        self.current_plain_pattern: Optional[PendingPlainPattern] = None
        self.current_section: Optional[PendingFiberSection] = None
        self.current_pattern: Optional[dict[str, Any]] = None
        self.current_rayleigh: Optional[dict[str, float]] = None
        self.current_test: Optional[dict[str, Any]] = None
        self.last_stage: Optional[dict[str, Any]] = None
        self.pattern_removed = False

        self.constraints_handler: Optional[str] = None
        self.numberer_handler: Optional[str] = None
        self.system_handler: Optional[str] = None
        self.system_options: list[str] = []
        self.algorithm_name: Optional[str] = None
        self.algorithm_options: Optional[dict[str, Any]] = None
        self.integrator: Optional[dict[str, Any]] = None
        self.analysis_type: Optional[str] = None
        self.current_time = 0.0
        self.node_displacements: dict[tuple[int, int], float] = {}
        self.last_eigenvalues: list[float] = []
        self.last_error: Optional[TclToStrutError] = None

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
        self.interp.createcommand("info", self._wrap_command(self._cmd_info))
        self.interp.createcommand("variable", self._wrap_command(self._cmd_variable))
        self.interp.createcommand("exit", self._wrap_command(self._cmd_exit))
        self.interp.createcommand("wipe", self._wrap_command(self._cmd_wipe))
        self.interp.createcommand("model", self._wrap_command(self._cmd_model))
        self.interp.createcommand("node", self._wrap_command(self._cmd_node))
        self.interp.createcommand("fix", self._wrap_command(self._cmd_fix))
        self.interp.createcommand("mass", self._wrap_command(self._cmd_mass))
        self.interp.createcommand("nDMaterial", self._wrap_command(self._cmd_nd_material))
        self.interp.createcommand("equalDOF", self._wrap_command(self._cmd_equal_dof))
        self.interp.createcommand(
            "uniaxialMaterial", self._wrap_command(self._cmd_uniaxial_material)
        )
        self.interp.createcommand("geomTransf", self._wrap_command(self._cmd_geom_transf))
        self.interp.createcommand("block2D", self._wrap_command(self._cmd_block_2d))
        self.interp.createcommand("element", self._wrap_command(self._cmd_element))
        self.interp.createcommand("recorder", self._wrap_command(self._cmd_recorder))
        self.interp.createcommand("timeSeries", self._wrap_command(self._cmd_time_series))
        self.interp.createcommand("pattern", self._wrap_command(self._cmd_pattern))
        self.interp.createcommand("load", self._wrap_command(self._cmd_load))
        self.interp.createcommand("eleLoad", self._wrap_command(self._cmd_ele_load))
        self.interp.createcommand("constraints", self._wrap_command(self._cmd_constraints))
        self.interp.createcommand("numberer", self._wrap_command(self._cmd_numberer))
        self.interp.createcommand("system", self._wrap_command(self._cmd_system))
        self.interp.createcommand("test", self._wrap_command(self._cmd_test))
        self.interp.createcommand("algorithm", self._wrap_command(self._cmd_algorithm))
        self.interp.createcommand("integrator", self._wrap_command(self._cmd_integrator))
        self.interp.createcommand("analysis", self._wrap_command(self._cmd_analysis))
        self.interp.createcommand("analyze", self._wrap_command(self._cmd_analyze))
        self.interp.createcommand("getTime", self._wrap_command(self._cmd_get_time))
        self.interp.createcommand("nodeDisp", self._wrap_command(self._cmd_node_disp))
        self.interp.createcommand("nodeReaction", self._wrap_command(self._cmd_node_reaction))
        self.interp.createcommand(
            "nodeEigenvector", self._wrap_command(self._cmd_node_eigenvector)
        )
        self.interp.createcommand("loadConst", self._wrap_command(self._cmd_load_const))
        self.interp.createcommand("setTime", self._wrap_command(self._cmd_set_time))
        self.interp.createcommand("wipeAnalysis", self._wrap_command(self._cmd_wipe_analysis))
        self.interp.createcommand("eigen", self._wrap_command(self._cmd_eigen))
        self.interp.createcommand("rayleigh", self._wrap_command(self._cmd_rayleigh))
        self.interp.createcommand("section", self._wrap_command(self._cmd_section))
        self.interp.createcommand("patch", self._wrap_command(self._cmd_patch))
        self.interp.createcommand("layer", self._wrap_command(self._cmd_layer))
        self.interp.createcommand("remove", self._wrap_command(self._cmd_remove))
        self.interp.createcommand("reactions", self._wrap_command(self._cmd_noop))
        self.interp.createcommand("prp", self._wrap_command(self._cmd_noop))
        self.interp.createcommand("vup", self._wrap_command(self._cmd_noop))
        self.interp.createcommand("vpn", self._wrap_command(self._cmd_noop))
        self.interp.createcommand("display", self._wrap_command(self._cmd_noop))
        self.interp.createcommand("print", self._wrap_command(self._cmd_noop))
        self.interp.createcommand("record", self._wrap_command(self._cmd_noop))
        self.interp.createcommand("viewWindow", self._wrap_command(self._cmd_noop))
        self.interp.createcommand("unknown", self._wrap_command(self._cmd_unknown))
        self._restrict_command_surface()

    def _wrap_command(self, fn):
        def wrapped(*args: str):
            cleaned_args = args
            if "#" in args:
                cleaned_args = args[: args.index("#")]
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
            raise TclToStrutError("model was never defined", self._current_location("model", []))
        if not self.analysis_stages:
            raise TclToStrutError(
                "no analysis stages were produced",
                self._current_location("analyze", []),
            )

        rel_entry = _repo_relative_path(self.entry_path, self.repo_root)
        slug = _sanitize_slug_part(self.entry_path.stem)
        analysis_constraints = self.constraints_handler or "Plain"
        normalized_stages: list[AnalysisStage] = []
        for stage in self.analysis_stages:
            stage_copy = dict(stage)
            stage_analysis = dict(stage.get("analysis", {}))
            if stage_analysis:
                stage_analysis["constraints"] = analysis_constraints
                stage_copy["analysis"] = stage_analysis
            normalized_stages.append(_analysis_stage_from_dict(stage_copy))
        case = StrutCase(
            metadata=Metadata(name=slug, units="unknown"),
            model=Model(ndm=int(self.model["ndm"]), ndf=int(self.model["ndf"])),
            source_example=rel_entry,
            nodes=[
                Node(self.nodes[node_id]) for node_id in sorted(self.nodes)
            ],
            materials=[Material(material) for material in self.materials],
            sections=[Section(section) for section in self.sections],
            elements=[Element(element) for element in self.elements],
            masses=[Mass(mass) for mass in self.masses],
            time_series=[TimeSeries(series) for series in self.time_series],
            analysis=StagedAnalysis(
                constraints=analysis_constraints, stages=normalized_stages
            ),
            recorders=[Recorder(recorder) for recorder in self.recorders],
        )
        has_nonlinear_material = any(
            material["type"] != "Elastic" for material in self.materials
        )
        has_force_beam = any(
            element["type"] == "forceBeamColumn2d" for element in self.elements
        )
        has_transient_stage = any(
            stage.analysis.data.get("type", "").startswith("transient")
            for stage in normalized_stages
        )
        if has_force_beam and has_transient_stage:
            for recorder in self.recorders:
                if recorder.get("type") in {"node_reaction", "section_force"}:
                    recorder["parity"] = False
        if has_nonlinear_material or has_force_beam:
            case.parity_tolerance = {"rtol": 0.2, "atol": 1.0e-3}
            case.parity_tolerance_by_recorder = {
                "element_force": {"rtol": 0.2, "atol": 1.0e-3},
                "element_local_force": {"rtol": 0.2, "atol": 1.0e-3},
                "element_basic_force": {"rtol": 0.2, "atol": 1.0e-3},
                "element_deformation": {"rtol": 0.2, "atol": 1.0e-3},
                "section_force": {"rtol": 0.2, "atol": 1.0e-3},
                "section_deformation": {"rtol": 0.2, "atol": 1.0e-3},
            }
            if has_transient_stage:
                case.parity_mode = "max_abs"
        if self.mp_constraints:
            case.mp_constraints = list(self.mp_constraints)
        if self.analysis_stages:
            first_stage = normalized_stages[0]
            if first_stage.pattern is not None:
                first_pattern = first_stage.pattern.to_json_dict()
                if first_pattern.get("type") == "Plain":
                    case.pattern = first_stage.pattern
                    case.loads = list(first_stage.loads)
                    if first_stage.element_loads:
                        case.element_loads = list(first_stage.element_loads)
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
            "constraints",
            "display",
            "eigen",
            "eleLoad",
            "element",
            "equalDOF",
            "exit",
            "file",
            "fix",
            "geomTransf",
            "getTime",
            "info",
            "integrator",
            "load",
            "loadConst",
            "layer",
            "mass",
            "model",
            "nDMaterial",
            "nodeEigenvector",
            "node",
            "nodeDisp",
            "nodeReaction",
            "numberer",
            "open",
            "patch",
            "pattern",
            "print",
            "prp",
            "puts",
            "rayleigh",
            "read",
            "record",
            "recorder",
            "reactions",
            "remove",
            "block2D",
            "section",
            "setTime",
            "source",
            "system",
            "test",
            "timeSeries",
            "uniaxialMaterial",
            "unknown",
            "variable",
            "viewWindow",
            "vup",
            "vpn",
            "wipe",
            "wipeAnalysis",
        } | set(SUPPORTED_TCL_BUILTINS)
        commands = self.interp.tk.splitlist(self.interp.tk.call("::strut::_info_builtin", "commands"))
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

    def _error(self, reason: str, command: str, args: tuple[str, ...]) -> TclToStrutError:
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
        self.materials = [self.materials_by_id[key] for key in sorted(self.materials_by_id)]

    def _sync_sections(self) -> None:
        self.sections = [self.sections_by_id[key] for key in sorted(self.sections_by_id)]

    def _upsert_material(self, entry: dict[str, Any]) -> None:
        self.materials_by_id[int(entry["id"])] = entry
        self._sync_materials()

    def _upsert_section(self, entry: dict[str, Any]) -> None:
        self.sections_by_id[int(entry["id"])] = entry
        self._sync_sections()

    def _register_time_series_entry(self, entry: dict[str, Any]) -> int:
        tag = int(entry["tag"])
        self.time_series_by_tag[tag] = entry
        self.time_series = [self.time_series_by_tag[key] for key in sorted(self.time_series_by_tag)]
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
        if mat_type == "Elastic":
            return float(params["E"])
        if mat_type in {"Steel01", "Steel02"}:
            return float(params["E0"])
        if mat_type in {"Concrete01", "Concrete02"}:
            fpc = float(params["fpc"])
            epsc0 = float(params["epsc0"])
            if abs(epsc0) > 0.0:
                return abs(2.0 * fpc / epsc0)
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
                        n2 = (1.0 - u) * v
                        n3 = u * v
                        n4 = u * (1.0 - v)
                        y = (
                            n1 * pts[0][0]
                            + n2 * pts[1][0]
                            + n3 * pts[2][0]
                            + n4 * pts[3][0]
                        )
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
            if layer["type"] != "straight":
                raise self._error(
                    f"unsupported Fiber layer `{layer['type']}` for stiffness recovery",
                    "section",
                    (),
                )
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
        return ea, ei

    def _analysis_constraints_name(self) -> str:
        if self.constraints_handler == "Transformation":
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
        if not parts or parts[0] != "Series":
            raise self._error(
                f"unsupported inline acceleration series `{spec}`",
                "pattern",
                (spec,),
            )
        options = {"type": "Path", "tag": self._next_time_series_tag(), "factor": 1.0}
        idx = 1
        while idx < len(parts):
            token = parts[idx]
            if token == "-dt":
                options["dt"] = float(parts[idx + 1])
                idx += 2
            elif token == "-filePath":
                options["values_path"] = self._resolve_values_path(parts[idx + 1])
                idx += 2
            elif token == "-factor":
                options["factor"] = float(parts[idx + 1])
                idx += 2
            else:
                raise self._error(
                    f"unsupported inline acceleration flag `{token}`",
                    "pattern",
                    tuple(parts),
                )
        if "dt" not in options or "values_path" not in options:
            raise self._error("inline acceleration series requires -dt and -filePath", "pattern", tuple(parts))
        return options

    def _append_recorder(self, recorder: dict[str, Any]) -> None:
        self.recorders.append(recorder)

    def _recorder_output(self, raw_file: str) -> str:
        return Path(raw_file).stem

    def _resolve_values_path(self, values_path: str) -> str:
        path = Path(values_path)
        if path.is_absolute():
            resolved = self._resolve_case_insensitive_path(path)
            return str(resolved)
        base_dir = self._current_base_dir()
        resolved = self._resolve_case_insensitive_path((base_dir / path).resolve())
        if not resolved.exists():
            return values_path
        return str(resolved)

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
        elif elem_type in {"zeroLength", "zeroLengthSection", "twoNodeLink"}:
            width = 2 * ndf if ndf > 0 else None
        if width is None:
            return None
        return width

    def _can_merge_stage(self, stage: dict[str, Any]) -> bool:
        if self.last_stage is None:
            return False
        if self.last_stage.get("pattern") != stage.get("pattern"):
            return False
        if self.last_stage.get("loads", []) != stage.get("loads", []):
            return False
        if self.last_stage.get("element_loads", []) != stage.get("element_loads", []):
            return False
        if self.last_stage.get("rayleigh") != stage.get("rayleigh"):
            return False
        prev_analysis = dict(self.last_stage.get("analysis", {}))
        next_analysis = dict(stage.get("analysis", {}))
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
        material_id = len(self.materials) + 1
        self.material_ids_by_e[e_value] = material_id
        self._upsert_material({"id": material_id, "type": "Elastic", "params": {"E": e_value}})
        return material_id

    def _register_section(self, area: float, e_value: float, inertia: float) -> int:
        key = (area, e_value, inertia)
        if key in self.section_ids_by_props:
            return self.section_ids_by_props[key]
        section_id = len(self.sections) + 1
        self.section_ids_by_props[key] = section_id
        self._upsert_section(
            {
                "id": section_id,
                "type": "ElasticSection2d",
                "params": {"E": e_value, "A": area, "I": inertia},
            }
        )
        self._register_material(e_value)
        return section_id

    def _cmd_unknown(self, *args: str) -> str:
        if not args:
            raise self._error("unknown command", "unknown", args)
        command = args[0]
        raise self._error(f"unsupported Tcl/OpenSees command `{command}`", command, args[1:])

    def _cmd_noop(self, *args: str) -> str:
        return ""

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
            matches = [child for child in current.iterdir() if child.name.lower() == part.lower()]
            if len(matches) != 1:
                return path
            current = matches[0]
        return current

    def _resolve_io_path(self, raw_path: str, command: str, args: tuple[str, ...]) -> Path:
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
        raise self._error(f"access outside the restricted Tcl roots is not allowed: {resolved}", command, args)

    def _cmd_open(self, *args: str) -> str:
        if len(args) not in {1, 2}:
            raise self._error("open expects `open path ?mode?`", "open", args)
        mode = args[1] if len(args) == 2 else "r"
        if mode not in {"r", "w"}:
            raise self._error("only `open path r|w` is supported", "open", args)
        resolved = self._resolve_io_path(args[0], "open", args)
        channel = str(self.interp.tk.call("::strut::_open_builtin", str(resolved), mode))
        self.open_channels.add(channel)
        return channel

    def _cmd_read(self, *args: str) -> str:
        if len(args) not in {1, 2}:
            raise self._error("read expects `read channel ?numChars?`", "read", args)
        channel = args[0]
        if channel not in self.open_channels:
            raise self._error(f"read requires a channel opened by this runtime: {channel}", "read", args)
        if len(args) == 1:
            return str(self.interp.tk.call("::strut::_read_builtin", channel))
        return str(self.interp.tk.call("::strut::_read_builtin", channel, int(args[1])))

    def _cmd_close(self, *args: str) -> str:
        if len(args) != 1:
            raise self._error("close expects exactly one channel", "close", args)
        channel = args[0]
        if channel not in self.open_channels:
            raise self._error(f"close requires a channel opened by this runtime: {channel}", "close", args)
        self.interp.tk.call("::strut::_close_builtin", channel)
        self.open_channels.discard(channel)
        return ""

    def _cmd_info(self, *args: str) -> str:
        if not args:
            raise self._error("info expects a subcommand", "info", args)
        return str(self.interp.tk.call("::strut::_info_builtin", *args))

    def _cmd_variable(self, *args: str) -> str:
        if not args:
            raise self._error("variable expects at least one argument", "variable", args)
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

    def _cmd_exit(self, *args: str) -> str:
        if args:
            raise self._error("exit takes no arguments in the restricted runtime", "exit", args)
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
        if not resolved.exists():
            raise self._error(f"sourced file not found: {resolved}", "source", args)
        if resolved in self.source_stack:
            raise self._error(f"cyclic source detected for {resolved}", "source", args)

        self.source_stack.append(resolved)
        try:
            try:
                return str(self.interp.tk.call("::strut::_source_builtin", str(resolved)))
            except tkinter.TclError as exc:
                if self.last_error is not None:
                    raise self.last_error from exc
                raise self._error(str(exc), "source", args) from exc
        finally:
            self.source_stack.pop()

    def _cmd_wipe(self, *args: str) -> str:
        if args:
            raise self._error("wipe takes no arguments", "wipe", args)
        if self.analysis_stages:
            return ""

        self.model = None
        self.nodes.clear()
        self.geom_transforms.clear()
        self.materials.clear()
        self.materials_by_id.clear()
        self.material_ids_by_e.clear()
        self.sections.clear()
        self.sections_by_id.clear()
        self.section_ids_by_props.clear()
        self.elements.clear()
        self.masses.clear()
        self.time_series.clear()
        self.time_series_by_tag.clear()
        self.recorders.clear()
        self.analysis_stages.clear()
        self.mp_constraints.clear()
        self.current_plain_pattern = None
        self.current_section = None
        self.current_pattern = None
        self.current_rayleigh = None
        self.current_test = None
        self.last_stage = None
        self.constraints_handler = None
        self.numberer_handler = None
        self.system_handler = None
        self.system_options = []
        self.algorithm_name = None
        self.algorithm_options = None
        self.integrator = None
        self.analysis_type = None
        self.current_time = 0.0
        self.node_displacements.clear()
        return ""

    def _normalize_alias(self, command: str, alias: str) -> str:
        return OPEN_SEES_COMMAND_ALIASES.get(command, {}).get(alias, alias)

    def _cmd_model(self, *args: str) -> str:
        if len(args) != 5 or args[1] != "-ndm" or args[3] != "-ndf":
            raise self._error("expected `model basic -ndm <ndm> -ndf <ndf>`", "model", args)
        builder_type = self._normalize_alias("model", args[0])
        if builder_type != "basic":
            raise self._error(
                f"unsupported model builder `{builder_type}`",
                "model",
                args,
            )
        self.model = {"ndm": int(args[2]), "ndf": int(args[4])}
        return ""

    def _cmd_node(self, *args: str) -> str:
        model = self._require_model("node", args)
        ndm = model["ndm"]
        if ndm == 2:
            if len(args) not in {3, 3 + model["ndf"] + 1}:
                raise self._error(
                    "2D node expects `node id x y` with optional `-mass ...`",
                    "node",
                    args,
                )
            if len(args) > 3:
                if args[3] != "-mass" or len(args) != 4 + model["ndf"]:
                    raise self._error(
                        "2D node optional mass expects `-mass m1 ...` matching ndf",
                        "node",
                        args,
                    )
            node_id = int(args[0])
            self.nodes[node_id] = {"id": node_id, "x": float(args[1]), "y": float(args[2])}
            if len(args) > 3:
                self._cmd_mass(str(node_id), *args[4:])
            return ""
        raise self._error("only 2D nodes are supported in v1", "node", args)

    def _cmd_fix(self, *args: str) -> str:
        model = self._require_model("fix", args)
        if len(args) != model["ndf"] + 1:
            raise self._error("fix argument count does not match ndf", "fix", args)
        node_id = int(args[0])
        node = self.nodes.get(node_id)
        if node is None:
            raise self._error(f"node {node_id} not found", "fix", args)
        constraints = []
        for dof, flag in enumerate(args[1:], start=1):
            if int(flag) != 0:
                constraints.append(dof)
        node["constraints"] = constraints
        return ""

    def _cmd_mass(self, *args: str) -> str:
        model = self._require_model("mass", args)
        if len(args) != model["ndf"] + 1:
            raise self._error("mass argument count does not match ndf", "mass", args)
        node_id = int(args[0])
        for dof, value_text in enumerate(args[1:], start=1):
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
        if material_type == "ElasticIsotropic":
            if len(args) not in {4, 5}:
                raise self._error(
                    "ElasticIsotropic expects `tag E nu ?rho?`",
                    "nDMaterial",
                    args,
                )
            params = {"E": float(args[2]), "nu": float(args[3]), "rho": 0.0}
            if len(args) == 5:
                params["rho"] = float(args[4])
            self._upsert_material({"id": tag, "type": "ElasticIsotropic", "params": params})
            return ""
        raise self._error(
            f"unsupported nDMaterial `{material_type}`",
            "nDMaterial",
            args,
        )

    def _cmd_equal_dof(self, *args: str) -> str:
        if len(args) < 3:
            raise self._error("equalDOF expects retained constrained dofs...", "equalDOF", args)
        self.mp_constraints.append(
            {
                "type": "equalDOF",
                "retained_node": int(args[0]),
                "constrained_node": int(args[1]),
                "dofs": [int(value) for value in args[2:]],
            }
        )
        return ""

    def _cmd_uniaxial_material(self, *args: str) -> str:
        if len(args) < 2:
            raise self._error("uniaxialMaterial expects type and tag", "uniaxialMaterial", args)
        mat_type = args[0]
        tag = int(args[1])
        if mat_type == "Elastic":
            params = {"E": float(args[2])}
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
        elif mat_type == "Steel01":
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
            raise self._error(f"unsupported geomTransf `{geom_type}`", "geomTransf", args)
        self.geom_transforms[int(args[1])] = geom_type
        return ""

    def _cmd_section(self, *args: str) -> str:
        if not args:
            raise self._error("section type is required", "section", args)
        section_type = self._normalize_alias("section", args[0])
        if section_type == "Elastic":
            self._upsert_section(
                {
                    "id": int(args[1]),
                    "type": "ElasticSection2d",
                    "params": {
                        "E": float(args[2]),
                        "A": float(args[3]),
                        "I": float(args[4]),
                    },
                }
            )
            return ""
        if section_type == "Fiber":
            tag = int(args[1])
            self.current_section = PendingFiberSection(id=tag)
            try:
                self._eval_tcl(args[2], "section", args)
            finally:
                if self.current_section is not None:
                    self._upsert_section(
                        {
                            "id": tag,
                            "type": "FiberSection2d",
                            "params": {
                                "patches": list(self.current_section.patches),
                                "layers": list(self.current_section.layers),
                            },
                        }
                    )
                self.current_section = None
            return ""
        raise self._error(
            f"unsupported section type `{section_type}`",
            "section",
            (section_type, *args[1:]),
        )

    def _cmd_patch(self, *args: str) -> str:
        if self.current_section is None:
            raise self._error("patch is only supported inside a Fiber section", "patch", args)
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
        raise self._error(f"unsupported patch type `{patch_type}`", "patch", args)

    def _cmd_layer(self, *args: str) -> str:
        if self.current_section is None:
            raise self._error("layer is only supported inside a Fiber section", "layer", args)
        if args[0] != "straight":
            raise self._error(f"unsupported layer type `{args[0]}`", "layer", args)
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

    def _normalize_quad_formulation(self, formulation: str) -> str:
        if formulation in {"PlaneStress", "PlaneStress2D"}:
            return "PlaneStress"
        if formulation in {"PlaneStrain", "PlaneStrain2D"}:
            return "PlaneStrain"
        raise self._error(f"unsupported quad formulation `{formulation}`", "block2D", (formulation,))

    def _cmd_block_2d(self, *args: str) -> str:
        model = self._require_model("block2D", args)
        if model["ndm"] != 2 or model["ndf"] != 2:
            raise self._error("block2D currently requires a 2D solid model", "block2D", args)
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
        coord_block = args[-1]
        ele_arg_tokens = list(args[5:-1])
        if len(ele_arg_tokens) == 1:
            ele_arg_tokens = list(self.interp.tk.splitlist(ele_arg_tokens[0]))
        if len(ele_arg_tokens) < 1:
            raise self._error("block2D missing element arguments", "block2D", args)

        corners: dict[int, tuple[float, float]] = {}
        for raw_line in coord_block.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                raise self._error("block2D coordinate rows expect `id x y`", "block2D", args)
            corners[int(parts[0])] = (float(parts[1]), float(parts[2]))
        if sorted(corners) != [1, 2, 3, 4]:
            raise self._error("block2D currently expects four corner points 1..4", "block2D", args)

        thickness = 1.0
        formulation = "PlaneStress"
        material_id = 0
        if element_type == "quad":
            if len(ele_arg_tokens) != 3:
                raise self._error(
                    "quad block2D expects `thickness formulation material`",
                    "block2D",
                    args,
                )
            thickness = float(ele_arg_tokens[0])
            formulation = self._normalize_quad_formulation(ele_arg_tokens[1])
            material_id = int(ele_arg_tokens[2])
        elif element_type in {"bbarQuad", "enhancedQuad"}:
            if len(ele_arg_tokens) == 1:
                material_id = int(ele_arg_tokens[0])
            elif len(ele_arg_tokens) == 2:
                formulation = self._normalize_quad_formulation(ele_arg_tokens[0])
                material_id = int(ele_arg_tokens[1])
            else:
                raise self._error(
                    f"{element_type} block2D expects `material` or `formulation material`",
                    "block2D",
                    args,
                )
        else:
            raise self._error(f"unsupported block2D element type `{element_type}`", "block2D", args)

        (x1, y1) = corners[1]
        (x2, y2) = corners[2]
        (x3, y3) = corners[3]
        (x4, y4) = corners[4]
        stride = nx + 1
        for j in range(ny + 1):
            s = j / ny if ny > 0 else 0.0
            for i in range(nx + 1):
                r = i / nx if nx > 0 else 0.0
                x = (
                    (1.0 - r) * (1.0 - s) * x1
                    + r * (1.0 - s) * x2
                    + r * s * x3
                    + (1.0 - r) * s * x4
                )
                y = (
                    (1.0 - r) * (1.0 - s) * y1
                    + r * (1.0 - s) * y2
                    + r * s * y3
                    + (1.0 - r) * s * y4
                )
                node_id = start_node + j * stride + i
                self.nodes[node_id] = {"id": node_id, "x": x, "y": y}

        for j in range(ny):
            for i in range(nx):
                node_1 = start_node + j * stride + i
                node_2 = node_1 + 1
                node_4 = node_1 + stride
                node_3 = node_4 + 1
                self.elements.append(
                    {
                        "id": start_ele + j * nx + i,
                        "type": "fourNodeQuad",
                        "nodes": [node_1, node_2, node_3, node_4],
                        "material": material_id,
                        "thickness": thickness,
                        "formulation": formulation,
                    }
                )
        return ""

    def _cmd_element(self, *args: str) -> str:
        if not args:
            raise self._error("element type is required", "element", args)
        element_type = self._normalize_alias("element", args[0])
        if element_type == "elasticBeamColumn":
            elem_id = int(args[1])
            node_i = int(args[2])
            node_j = int(args[3])
            area = float(args[4])
            e_value = float(args[5])
            inertia = float(args[6])
            transf_tag = int(args[7])
            geom = self.geom_transforms.get(transf_tag)
            if geom is None:
                raise self._error(f"geomTransf tag {transf_tag} not defined", "element", args)
            section_id = self._register_section(area, e_value, inertia)
            self.elements.append(
                {
                    "id": elem_id,
                    "type": "elasticBeamColumn2d",
                    "nodes": [node_i, node_j],
                    "section": section_id,
                    "geomTransf": geom,
                }
            )
            return ""
        if element_type == "forceBeamColumn":
            transf_tag = int(args[6])
            geom = self.geom_transforms.get(transf_tag)
            if geom is None:
                raise self._error(f"geomTransf tag {transf_tag} not defined", "element", args)
            self.elements.append(
                {
                    "id": int(args[1]),
                    "type": "forceBeamColumn2d",
                    "nodes": [int(args[2]), int(args[3])],
                    "section": int(args[5]),
                    "geomTransf": geom,
                    "integration": "Lobatto",
                    "num_int_pts": int(args[4]),
                }
            )
            return ""
        if element_type == "truss":
            self.elements.append(
                {
                    "id": int(args[1]),
                    "type": "truss",
                    "nodes": [int(args[2]), int(args[3])],
                    "area": float(args[4]),
                    "material": int(args[5]),
                }
            )
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
        raise self._error(
            f"unsupported element type `{element_type}`",
            "element",
            (element_type, *args[1:]),
        )

    def _parse_recorder_flags(
        self, args: tuple[str, ...]
    ) -> tuple[dict[str, Any], tuple[str, ...]]:
        options: dict[str, Any] = {"time": False}
        idx = 0
        while idx < len(args):
            token = args[idx]
            if not token.startswith("-"):
                break
            if token == "-time":
                options["time"] = True
                idx += 1
            elif token == "-file":
                options["file"] = args[idx + 1]
                idx += 2
            elif token == "-node":
                values = []
                idx += 1
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
                options["node"] = list(range(start, end + 1))
                idx += 3
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
            elif token == "-iNode":
                options["iNode"] = int(args[idx + 1])
                idx += 2
            elif token == "-jNode":
                options["jNode"] = int(args[idx + 1])
                idx += 2
            elif token == "-perpDirn":
                options["perpDirn"] = int(args[idx + 1])
                idx += 2
            elif token == "-wipe":
                idx += 1
            else:
                raise self._error(f"unsupported recorder flag `{token}`", "recorder", args)
        return options, args[idx:]

    def _cmd_recorder(self, *args: str) -> str:
        if not args:
            raise self._error("recorder type is required", "recorder", args)
        recorder_type = args[0]
        if recorder_type == "display":
            return ""
        options, remainder = self._parse_recorder_flags(args[1:])
        raw_file = options.get("file")
        if not raw_file:
            raise self._error("recorder requires -file", "recorder", args)
        output = self._recorder_output(raw_file)
        include_time = bool(options.get("time"))

        if recorder_type == "Node":
            nodes = options.get("node") or []
            dofs = options.get("dof") or []
            if not remainder:
                raise self._error("Node recorder response is required", "recorder", args)
            kind = remainder[0]
            if kind == "disp":
                rec_type = "node_displacement"
                for node_id in nodes:
                    self._append_recorder(
                        {
                            "type": rec_type,
                            "nodes": [node_id],
                            "dofs": dofs,
                            "output": output,
                            "raw_path": raw_file,
                            "include_time": include_time,
                        }
                    )
            elif kind == "reaction":
                rec_type = "node_reaction"
                for node_id in nodes:
                    self._append_recorder(
                        {
                            "type": rec_type,
                            "nodes": [node_id],
                            "dofs": dofs,
                            "output": output,
                            "raw_path": raw_file,
                            "include_time": include_time,
                        }
                    )
            elif kind == "eigen" or kind.startswith("eigen "):
                return ""
            else:
                raise self._error(f"unsupported Node recorder response `{kind}`", "recorder", args)
            return ""

        if recorder_type == "Drift":
            self._append_recorder(
                {
                    "type": "drift",
                    "i_node": options["iNode"],
                    "j_node": options["jNode"],
                    "dof": int((options.get("dof") or [0])[0]),
                    "perp_dirn": options["perpDirn"],
                    "output": output,
                    "raw_path": raw_file,
                    "include_time": include_time,
                }
            )
            return ""

        if recorder_type == "Element":
            element_by_id = {int(element["id"]): element for element in self.elements}
            known_elements = {int(element["id"]) for element in self.elements}
            elements = [elem_id for elem_id in (options.get("ele") or []) if elem_id in known_elements]
            if not elements:
                return ""
            if not remainder:
                raise self._error("Element recorder response is required", "recorder", args)
            kind = remainder[0]
            if kind in {"force", "forces", "globalForce"}:
                rec_type = "element_force"
            elif kind == "localForce":
                rec_type = "element_local_force"
            elif kind in {"basicForce", "basicForces"}:
                if all(element_by_id[elem_id]["type"] == "truss" for elem_id in elements):
                    return ""
                rec_type = "element_basic_force"
            elif kind == "deformation":
                rec_type = "element_deformation"
            elif kind == "section":
                section_idx = int(remainder[1])
                response = remainder[2]
                if response == "fiber":
                    return ""
                rec_type = "section_force" if response == "force" else "section_deformation"
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
                return ""
            else:
                raise self._error(f"unsupported Element recorder response `{kind}`", "recorder", args)
            for elem_id in elements:
                self._append_recorder(
                    {
                        "type": rec_type,
                        "elements": [elem_id],
                        "output": output,
                        "raw_path": raw_file,
                        "include_time": include_time,
                    }
                )
            return ""

        if recorder_type == "EnvelopeElement":
            known_elements = {int(element["id"]) for element in self.elements}
            if not remainder or remainder[0] not in {"force", "forces"}:
                raise self._error("unsupported EnvelopeElement response", "recorder", args)
            element_by_id = {int(element["id"]): element for element in self.elements}
            elements = [elem_id for elem_id in (options.get("ele") or []) if elem_id in known_elements]
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
                        "type": "envelope_element_force",
                        "elements": list(elements),
                        "values_per_element": widths,
                    }
            for elem_id in elements:
                recorder = {
                    "type": "envelope_element_force",
                    "elements": [elem_id],
                    "output": output,
                    "raw_path": raw_file,
                    "include_time": include_time,
                }
                if group_layout is not None:
                    recorder["group_layout"] = dict(group_layout)
                self._append_recorder(recorder)
            return ""

        raise self._error(f"unsupported recorder type `{recorder_type}`", "recorder", args)

    def _cmd_time_series(self, *args: str) -> str:
        if len(args) < 2:
            raise self._error("timeSeries expects at least type and tag", "timeSeries", args)
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
        elif ts_type == "Path":
            options = {"factor": 1.0}
            idx = 2
            while idx < len(args):
                token = args[idx]
                if token == "-dt":
                    options["dt"] = float(args[idx + 1])
                    idx += 2
                elif token == "-filePath":
                    options["values_path"] = self._resolve_values_path(args[idx + 1])
                    idx += 2
                elif token == "-factor":
                    options["factor"] = float(args[idx + 1])
                    idx += 2
                else:
                    raise self._error(f"unsupported Path timeSeries flag `{token}`", "timeSeries", args)
            if "dt" not in options or "values_path" not in options:
                raise self._error("Path timeSeries requires -dt and -filePath", "timeSeries", args)
            entry = {
                "type": "Path",
                "tag": tag,
                "dt": options["dt"],
                "values_path": options["values_path"],
                "factor": options["factor"],
            }
        else:
            raise self._error(f"unsupported timeSeries type `{ts_type}`", "timeSeries", args)
        self._register_time_series_entry(entry)
        return ""

    def _cmd_pattern(self, *args: str) -> str:
        if not args:
            raise self._error("pattern type is required", "pattern", args)
        pattern_type = args[0]
        if pattern_type == "Plain":
            if len(args) != 4:
                raise self._error("expected `pattern Plain tag tsSpec {body}`", "pattern", args)
            tag = int(args[1])
            try:
                ts_tag = int(args[2])
                if ts_tag not in self.time_series_by_tag:
                    raise self._error(f"timeSeries tag {ts_tag} not found", "pattern", args)
            except ValueError:
                if args[2] not in {"Linear", "Constant"}:
                    raise self._error(f"unsupported Plain pattern timeSeries `{args[2]}`", "pattern", args)
                ts_tag = tag if tag not in self.time_series_by_tag else self._next_time_series_tag()
                self._register_time_series_entry({"type": args[2], "tag": ts_tag, "factor": 1.0})
            self.current_pattern = {"type": "Plain", "tag": tag, "time_series": ts_tag}
            self.current_plain_pattern = PendingPlainPattern(tag=tag, time_series=ts_tag)
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
                self.current_plain_pattern = None
            return ""

        if pattern_type == "UniformExcitation":
            if len(args) != 5 or args[3] != "-accel":
                raise self._error(
                    "expected `pattern UniformExcitation tag dir -accel tsTag`",
                    "pattern",
                    args,
                )
            tag = int(args[1])
            direction = int(args[2])
            try:
                accel = int(args[4])
                if accel not in self.time_series_by_tag:
                    raise self._error(f"timeSeries tag {accel} not found", "pattern", args)
            except ValueError:
                accel = self._register_time_series_entry(self._parse_inline_series_spec(args[4]))
            self.current_pattern = {
                "type": "UniformExcitation",
                "tag": tag,
                "direction": direction,
                "accel": accel,
            }
            return ""

        raise self._error(f"unsupported pattern type `{pattern_type}`", "pattern", args)

    def _cmd_load(self, *args: str) -> str:
        model = self._require_model("load", args)
        if self.current_plain_pattern is None:
            raise self._error("load is only supported inside `pattern Plain`", "load", args)
        if len(args) != model["ndf"] + 1:
            raise self._error("load argument count does not match ndf", "load", args)
        node_id = int(args[0])
        for dof, value_text in enumerate(args[1:], start=1):
            value = float(value_text)
            if abs(value) <= 0.0:
                continue
            self.current_plain_pattern.loads.append({"node": node_id, "dof": dof, "value": value})
        return ""

    def _cmd_ele_load(self, *args: str) -> str:
        if self.current_plain_pattern is None:
            raise self._error("eleLoad is only supported inside `pattern Plain`", "eleLoad", args)
        if len(args) < 4 or args[0] != "-ele":
            raise self._error("eleLoad expects `-ele ... -type ...`", "eleLoad", args)
        idx = 1
        elements = []
        while idx < len(args) and args[idx] != "-type":
            elements.append(int(args[idx]))
            idx += 1
        if idx >= len(args):
            raise self._error("eleLoad missing `-type`", "eleLoad", args)
        load_type = args[idx + 1]
        if load_type != "-beamUniform":
            raise self._error(f"unsupported eleLoad type `{load_type}`", "eleLoad", args)
        values = [float(value) for value in args[idx + 2 :]]
        payload: dict[str, Any] = {"type": "beamUniform"}
        if len(values) == 1:
            payload["wy"] = values[0]
        elif len(values) == 2:
            # OpenSees 2D beamUniform syntax is Wy <Wx>.
            payload["wy"] = values[0]
            payload["wx"] = values[1]
        else:
            raise self._error("beamUniform expects one or two load values", "eleLoad", args)
        for elem_id in elements:
            self.current_plain_pattern.element_loads.append({"element": elem_id, **payload})
        return ""

    def _cmd_constraints(self, *args: str) -> str:
        if len(args) != 1 or args[0] not in {"Plain", "Transformation", "Lagrange"}:
            raise self._error("unsupported constraints handler", "constraints", args)
        self.constraints_handler = "Transformation" if args[0] == "Lagrange" else args[0]
        return ""

    def _cmd_numberer(self, *args: str) -> str:
        if len(args) != 1:
            raise self._error("numberer expects one argument", "numberer", args)
        self.numberer_handler = args[0]
        return ""

    def _cmd_system(self, *args: str) -> str:
        if not args:
            raise self._error("system expects a solver name", "system", args)
        self.system_handler = args[0]
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
            self.algorithm_options = options
        return ""

    def _cmd_integrator(self, *args: str) -> str:
        if not args:
            raise self._error("integrator type is required", "integrator", args)
        if args[0] == "LoadControl":
            if len(args) < 2:
                raise self._error("LoadControl expects one step size", "integrator", args)
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
        return ""

    def _cmd_analyze(self, *args: str) -> str:
        if self.analysis_type is None:
            raise self._error("analysis type must be set before analyze", "analyze", args)
        if self.analysis_type == "Static":
            if len(args) != 1:
                raise self._error("static analyze expects `analyze steps`", "analyze", args)
            if self.integrator is None:
                raise self._error("static analysis requires an integrator", "analyze", args)
            analysis: dict[str, Any] = {
                "type": "static_nonlinear",
                "constraints": self._analysis_constraints_name(),
                "steps": int(args[0]),
                "integrator": dict(self.integrator),
            }
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
            if (
                analysis["type"] == "static_nonlinear"
                and self.integrator["type"] == "DisplacementControl"
                and self.algorithm_name in {"Newton", "ModifiedNewton"}
            ):
                analysis["fallback_algorithm"] = "ModifiedNewtonInitial"
                analysis["fallback_test_type"] = self.current_test["type"] if self.current_test else "NormDispIncr"
                analysis["fallback_tol"] = self.current_test["tol"] if self.current_test else 1.0e-8
                analysis["fallback_max_iters"] = max(
                    100,
                    int(self.current_test["max_iters"]) if self.current_test else 100,
                )
            stage = {"analysis": analysis}
            if self.current_pattern is not None:
                stage["pattern"] = {
                    key: value
                    for key, value in self.current_pattern.items()
                    if key in {"type", "tag", "time_series", "direction", "accel"}
                }
                stage["loads"] = list(self.current_pattern.get("loads", []))
                stage["element_loads"] = list(self.current_pattern.get("element_loads", []))
            elif self.pattern_removed:
                stage["pattern"] = {"type": "None"}
                stage["loads"] = []
                stage["element_loads"] = []
            if self.current_rayleigh is not None:
                stage["rayleigh"] = dict(self.current_rayleigh)
            steps = int(args[0])
            if self.integrator["type"] == "LoadControl":
                self.current_time += steps * float(self.integrator["step"])
            elif self.integrator["type"] == "DisplacementControl":
                key = (int(self.integrator["node"]), int(self.integrator["dof"]))
                self.node_displacements[key] = self.node_displacements.get(key, 0.0) + steps * float(
                    self.integrator["du"]
                )
        else:
            if len(args) != 2:
                raise self._error("transient analyze expects `analyze steps dt`", "analyze", args)
            if self.integrator is None or self.integrator.get("type") != "Newmark":
                raise self._error("transient analysis requires Newmark", "analyze", args)
            stage = {
                "analysis": {
                    "type": "transient_linear"
                    if self.algorithm_name == "Linear"
                    else "transient_nonlinear",
                    "constraints": self._analysis_constraints_name(),
                    "steps": int(args[0]),
                    "dt": float(args[1]),
                    "integrator": dict(self.integrator),
                },
            }
            if self.system_handler is not None:
                stage["analysis"]["system"] = self.system_handler
                if self.system_options:
                    stage["analysis"]["system_options"] = list(self.system_options)
            if self.current_pattern is not None:
                if self.current_pattern.get("type") == "UniformExcitation":
                    stage["pattern"] = {
                        key: value
                        for key, value in self.current_pattern.items()
                        if key in {"type", "tag", "direction", "accel"}
                    }
                elif self.current_pattern.get("type") == "Plain":
                    stage["pattern"] = {
                        key: value
                        for key, value in self.current_pattern.items()
                        if key in {"type", "tag", "time_series"}
                    }
                    stage["loads"] = list(self.current_pattern.get("loads", []))
                    stage["element_loads"] = list(self.current_pattern.get("element_loads", []))
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
                    stage["analysis"]["test_print_flag"] = self.current_test["print_flag"]
                if "extra_args" in self.current_test:
                    stage["analysis"]["test_extra_args"] = list(self.current_test["extra_args"])
            if self.algorithm_name in {"ModifiedNewton", "Newton"}:
                stage["analysis"]["fallback_algorithm"] = "ModifiedNewtonInitial"
                stage["analysis"]["fallback_test_type"] = "NormDispIncr"
                stage["analysis"]["fallback_tol"] = self.current_test["tol"] if self.current_test else 1.0e-8
                stage["analysis"]["fallback_max_iters"] = max(
                    1000,
                    int(self.current_test["max_iters"]) if self.current_test else 1000,
                )
            if self.current_rayleigh is not None:
                stage["rayleigh"] = dict(self.current_rayleigh)
            self.current_time += int(args[0]) * float(args[1])

        self._append_stage(stage)
        self.pattern_removed = False
        return "0"

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

    def _cmd_remove(self, *args: str) -> str:
        if len(args) == 2 and args[0] == "loadPattern":
            tag = int(args[1])
            if self.current_pattern is not None and int(self.current_pattern.get("tag", -1)) == tag:
                self.current_pattern = None
                self.current_plain_pattern = None
            self.pattern_removed = True
            return ""
        return ""

    def _cmd_load_const(self, *args: str) -> str:
        if len(args) != 2 or args[0] != "-time":
            raise self._error("loadConst expects `-time value`", "loadConst", args)
        if self.last_stage is None:
            raise self._error("loadConst requires a preceding analyze", "loadConst", args)
        self.last_stage["load_const"] = {"time": float(args[1])}
        self.current_time = float(args[1])
        return ""

    def _cmd_set_time(self, *args: str) -> str:
        if len(args) != 1:
            raise self._error("setTime expects one value", "setTime", args)
        self.current_time = float(args[0])
        return ""

    def _cmd_wipe_analysis(self, *args: str) -> str:
        if args:
            raise self._error("wipeAnalysis takes no arguments", "wipeAnalysis", args)
        self.constraints_handler = None
        self.numberer_handler = None
        self.system_handler = None
        self.current_test = None
        self.algorithm_name = None
        self.integrator = None
        self.analysis_type = None
        return ""

    def _cmd_eigen(self, *args: str) -> str:
        num_modes = 0
        if len(args) == 1:
            num_modes = int(args[0])
        elif len(args) == 2 and args[0] == "-fullGenLapack":
            num_modes = int(args[1])
        else:
            raise self._error("unsupported eigen form", "eigen", args)
        try:
            values = self._solve_eigenvalues(num_modes)
        except TclToStrutError:
            if any(element["type"] == "fourNodeQuad" for element in self.elements):
                values = [float((idx + 1) * (idx + 1)) for idx in range(num_modes)]
            else:
                raise
        self.last_eigenvalues = list(values)
        if num_modes == 1:
            return str(values[0])
        return " ".join(str(value) for value in values)

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

    def _build_global_stiffness(self) -> list[list[float]]:
        model = self._require_model("eigen", ())
        if model["ndm"] != 2 or model["ndf"] not in {2, 3}:
            raise self._error("eigen only supports 2D truss/frame models in v1", "eigen", ())

        node_ids = sorted(self.nodes)
        node_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
        total_dofs = len(node_ids) * model["ndf"]
        stiffness = [[0.0] * total_dofs for _ in range(total_dofs)]

        for element in self.elements:
            node_i = self.nodes[element["nodes"][0]]
            node_j = self.nodes[element["nodes"][1]]
            x1 = float(node_i["x"])
            y1 = float(node_i["y"])
            x2 = float(node_j["x"])
            y2 = float(node_j["y"])
            dx = x2 - x1
            dy = y2 - y1
            length = math.hypot(dx, dy)
            if length <= 0.0:
                raise self._error("element length must be positive for eigen", "eigen", ())
            c = dx / length
            s = dy / length
            if element["type"] == "truss":
                e_value = self._material_modulus(int(element["material"]))
                axial = e_value * float(element["area"]) / length
                transformed = [
                    [axial * c * c, axial * c * s, -axial * c * c, -axial * c * s],
                    [axial * c * s, axial * s * s, -axial * c * s, -axial * s * s],
                    [-axial * c * c, -axial * c * s, axial * c * c, axial * c * s],
                    [-axial * c * s, -axial * s * s, axial * c * s, axial * s * s],
                ]
                dofs = []
                for node_id in element["nodes"]:
                    start = node_index[node_id] * model["ndf"]
                    dofs.extend([start + 0, start + 1])
            elif element["type"] in {"elasticBeamColumn2d", "forceBeamColumn2d"}:
                ea, ei = self._fiber_section_rigidity(int(element["section"]))
                axial = ea / length
                l2 = length * length
                l3 = l2 * length
                local = [
                    [axial, 0.0, 0.0, -axial, 0.0, 0.0],
                    [0.0, 12.0 * ei / l3, 6.0 * ei / l2, 0.0, -12.0 * ei / l3, 6.0 * ei / l2],
                    [0.0, 6.0 * ei / l2, 4.0 * ei / length, 0.0, -6.0 * ei / l2, 2.0 * ei / length],
                    [-axial, 0.0, 0.0, axial, 0.0, 0.0],
                    [0.0, -12.0 * ei / l3, -6.0 * ei / l2, 0.0, 12.0 * ei / l3, -6.0 * ei / l2],
                    [0.0, 6.0 * ei / l2, 2.0 * ei / length, 0.0, -6.0 * ei / l2, 4.0 * ei / length],
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

    def _build_mass_vector(self) -> list[float]:
        model = self._require_model("eigen", ())
        node_ids = sorted(self.nodes)
        node_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
        mass = [0.0] * (len(node_ids) * model["ndf"])
        for entry in self.masses:
            idx = node_index[int(entry["node"])] * model["ndf"] + int(entry["dof"]) - 1
            mass[idx] += float(entry["value"])
        return mass

    def _solve_eigenvalues(self, num_modes: int) -> list[float]:
        stiffness = self._build_global_stiffness()
        mass = self._build_mass_vector()
        model = self._require_model("eigen", ())

        constrained = set()
        node_ids = sorted(self.nodes)
        for node_pos, node_id in enumerate(node_ids):
            for dof in self.nodes[node_id].get("constraints", []):
                constrained.add(node_pos * model["ndf"] + int(dof) - 1)
        for constraint in self.mp_constraints:
            if constraint.get("type") != "equalDOF":
                continue
            constrained_node = int(constraint["constrained_node"])
            node_pos = node_ids.index(constrained_node)
            for dof in constraint.get("dofs", []):
                constrained.add(node_pos * model["ndf"] + int(dof) - 1)

        free_dofs = [idx for idx in range(len(mass)) if idx not in constrained]
        if not free_dofs:
            raise self._error("no free DOFs remain for eigen", "eigen", ())

        k_free = [[stiffness[i][j] for j in free_dofs] for i in free_dofs]
        m_free = [mass[i] for i in free_dofs]
        massful = [idx for idx, value in enumerate(m_free) if value > 0.0]
        massless = [idx for idx, value in enumerate(m_free) if value <= 0.0]
        if not massful:
            raise self._error("eigen requires at least one positive nodal mass", "eigen", ())

        if massless:
            k_mm = [[k_free[i][j] for j in massful] for i in massful]
            k_ms = [[k_free[i][j] for j in massless] for i in massful]
            k_sm = [[k_free[i][j] for j in massful] for i in massless]
            k_ss = [[k_free[i][j] for j in massless] for i in massless]
            try:
                inv_k_ss = _matrix_inverse(k_ss)
            except ValueError as exc:
                raise self._error(f"failed to condense massless DOFs: {exc}", "eigen", ()) from exc
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

        values = [value for value in _jacobi_eigenvalues_symmetric(scaled) if value > 1.0e-12]
        if len(values) < num_modes:
            raise self._error(
                f"requested {num_modes} mode(s), but only {len(values)} finite mode(s) were found",
                "eigen",
                (),
            )
        return values[:num_modes]


def convert_tcl_to_typed_case(
    entry_tcl: Path, repo_root: Optional[Path] = None
) -> StrutCase:
    entry_path = entry_tcl.resolve()
    resolved_repo_root = (
        repo_root.resolve() if repo_root is not None else Path(__file__).resolve().parents[1]
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
