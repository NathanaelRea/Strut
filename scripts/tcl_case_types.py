from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


def _clone(value: Any) -> Any:
    return deepcopy(value)


@dataclass(frozen=True)
class Metadata:
    name: str
    units: str = "unknown"

    def to_json_dict(self) -> dict[str, Any]:
        return {"name": self.name, "units": self.units}


@dataclass(frozen=True)
class Model:
    ndm: int
    ndf: int

    def to_json_dict(self) -> dict[str, Any]:
        return {"ndm": self.ndm, "ndf": self.ndf}


@dataclass(frozen=True)
class Node:
    data: dict[str, Any]

    def to_json_dict(self) -> dict[str, Any]:
        return _clone(self.data)


@dataclass(frozen=True)
class Material:
    data: dict[str, Any]

    def to_json_dict(self) -> dict[str, Any]:
        return _clone(self.data)


@dataclass(frozen=True)
class Section:
    data: dict[str, Any]

    def to_json_dict(self) -> dict[str, Any]:
        return _clone(self.data)


@dataclass(frozen=True)
class Element:
    data: dict[str, Any]

    def to_json_dict(self) -> dict[str, Any]:
        return _clone(self.data)


@dataclass(frozen=True)
class Mass:
    data: dict[str, Any]

    def to_json_dict(self) -> dict[str, Any]:
        return _clone(self.data)


@dataclass(frozen=True)
class TimeSeries:
    data: dict[str, Any]

    def to_json_dict(self) -> dict[str, Any]:
        return _clone(self.data)


@dataclass(frozen=True)
class Pattern:
    data: dict[str, Any]

    def to_json_dict(self) -> dict[str, Any]:
        return _clone(self.data)


@dataclass(frozen=True)
class Load:
    data: dict[str, Any]

    def to_json_dict(self) -> dict[str, Any]:
        return _clone(self.data)


@dataclass(frozen=True)
class Recorder:
    data: dict[str, Any]

    def to_json_dict(self) -> dict[str, Any]:
        return _clone(self.data)


@dataclass(frozen=True)
class Analysis:
    data: dict[str, Any]

    def to_json_dict(self) -> dict[str, Any]:
        return _clone(self.data)


@dataclass(frozen=True)
class AnalysisStage:
    analysis: Analysis
    pattern: Optional[Pattern] = None
    loads: list[Load] = field(default_factory=list)
    element_loads: list[Load] = field(default_factory=list)
    load_const: Optional[dict[str, Any]] = None
    rayleigh: Optional[dict[str, Any]] = None
    time_series: list[TimeSeries] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        data = _clone(self.extra)
        data["analysis"] = self.analysis.to_json_dict()
        if self.pattern is not None:
            data["pattern"] = self.pattern.to_json_dict()
        if self.loads:
            data["loads"] = [item.to_json_dict() for item in self.loads]
        if self.element_loads:
            data["element_loads"] = [item.to_json_dict() for item in self.element_loads]
        if self.load_const is not None:
            data["load_const"] = _clone(self.load_const)
        if self.rayleigh is not None:
            data["rayleigh"] = _clone(self.rayleigh)
        if self.time_series:
            data["time_series"] = [item.to_json_dict() for item in self.time_series]
        return data


@dataclass(frozen=True)
class StagedAnalysis:
    constraints: str
    stages: list[AnalysisStage]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "type": "staged",
            "constraints": self.constraints,
            "stages": [stage.to_json_dict() for stage in self.stages],
        }


@dataclass
class StrutCase:
    metadata: Metadata
    model: Model
    nodes: list[Node]
    materials: list[Material]
    sections: list[Section]
    elements: list[Element]
    masses: list[Mass]
    time_series: list[TimeSeries]
    analysis: StagedAnalysis
    recorders: list[Recorder]
    source_example: Optional[str] = None
    schema_version: str = "1.0"
    enabled: bool = True
    status: str = "generated"
    parity_tolerance: Optional[dict[str, Any]] = None
    parity_tolerance_by_recorder: Optional[dict[str, Any]] = None
    parity_mode: Optional[str] = None
    mp_constraints: list[dict[str, Any]] = field(default_factory=list)
    pattern: Optional[Pattern] = None
    loads: list[Load] = field(default_factory=list)
    element_loads: list[Load] = field(default_factory=list)

    def to_json_dict(self) -> dict[str, Any]:
        data = {
            "schema_version": self.schema_version,
            "enabled": self.enabled,
            "status": self.status,
            "metadata": self.metadata.to_json_dict(),
            "model": self.model.to_json_dict(),
            "nodes": [node.to_json_dict() for node in self.nodes],
            "materials": [material.to_json_dict() for material in self.materials],
            "sections": [section.to_json_dict() for section in self.sections],
            "elements": [element.to_json_dict() for element in self.elements],
            "masses": [mass.to_json_dict() for mass in self.masses],
            "time_series": [series.to_json_dict() for series in self.time_series],
            "analysis": self.analysis.to_json_dict(),
            "recorders": [recorder.to_json_dict() for recorder in self.recorders],
        }
        if self.source_example is not None:
            data["source_example"] = self.source_example
        if self.parity_tolerance is not None:
            data["parity_tolerance"] = _clone(self.parity_tolerance)
        if self.parity_tolerance_by_recorder is not None:
            data["parity_tolerance_by_recorder"] = _clone(
                self.parity_tolerance_by_recorder
            )
        if self.parity_mode is not None:
            data["parity_mode"] = self.parity_mode
        if self.mp_constraints:
            data["mp_constraints"] = _clone(self.mp_constraints)
        if self.pattern is not None:
            data["pattern"] = self.pattern.to_json_dict()
        if self.loads:
            data["loads"] = [load.to_json_dict() for load in self.loads]
        if self.element_loads:
            data["element_loads"] = [load.to_json_dict() for load in self.element_loads]
        return data

    def to_solver_input(
        self,
        *,
        entry_path: Optional[Path] = None,
        compute_only: bool = False,
    ) -> dict[str, Any]:
        data = self.to_json_dict()
        if compute_only:
            data["recorders"] = []
        if entry_path is not None:
            resolved_entry = entry_path.resolve()
            data["__strut_case_dir"] = str(resolved_entry.parent)
            data["__strut_case_json_path"] = str(resolved_entry)
        return data
