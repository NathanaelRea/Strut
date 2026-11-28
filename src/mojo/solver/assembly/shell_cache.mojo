"""Geometry and workspace owned by one loaded model."""

from elements.shell_asd_q4 import asd_shell_elastic_geometry
from elements.shell_asd_transform import (
    ASDFrame,
    ASDPoint,
    ASDPoints,
    ASDCorotational,
    asd_linear_stiffness,
)
from elements.shell_discrete_kirchhoff import discrete_kirchhoff_geometry
from sections.plate_fiber import elastic_plate_fiber_section_tangent
from collections import InlineArray, List
from elements.shell4 import (
    SHELL4_DOFS,
    SHELL4_GEOMETRY_SIZE,
    SHELL4_MATRIX_SIZE,
    SHELL4_SCRATCH_SIZE,
    compact_shell_matrix,
    elastic_membrane_plate_tangent,
    shell4_integrate_tangent,
    shell4_mindlin_stiffness,
    shell4_precompute_geometry,
)
from elements.shell_mass import ShellMass
from elements.shell_nldkgq import ShellNLDKGQState
from materials.concrete_s import ConcreteSState
from os import abort
from sections import LayeredShellSectionDef
from solver.assembly.parallel_elements import ParallelElementWork
from solver.run_case.input_types import ElementInput, NodeInput, SectionInput
from tag_types import GeomTransfTag


# Bound result storage to about 2.5 MB, even for very large models.
alias _BATCH_SIZE = 512
# The estimate charges for each Gauss point and its nonlinear layers, plus
# the force and tangent work. Cached elastic shells only apply a matrix.
alias _MINIMUM_WORK = 1024


@fieldwise_init
struct ShellAssemblyEntry(ImplicitlyCopyable, Movable):
    var section: Int
    var offset: Int
    var state_offset: Int
    var concrete_offset: Int
    var dof_count: Int
    var nonlinear_index: Int
    var corotational_index: Int

    @always_inline
    fn uses_cached_matrix(self) -> Bool:
        return self.section < 0 and self.nonlinear_index < 0 and self.corotational_index < 0


struct ShellAssemblyCache(Movable):
    """Element-major layered Gauss data and elastic stiffness matrices.

    Entries map the original element index to a layered geometry block or a
    cached elastic matrix. Each layered block holds four consecutive Gauss
    records of 8×24 row-major B, 24 drilling coefficients, and dvol. The basis
    is already incorporated in these rows, which act on global DOFs.
    Elastic blocks hold one row-major 24×24 matrix per element.

    ConcreteS layer states also live here, element-major: four Gauss points
    per layered element, each with one state per ConcreteS layer. Every element
    owns a disjoint slice, so threaded sweeps can update them in place.
    """

    var entry_by_element: List[Int]
    var entries: List[ShellAssemblyEntry]
    var masses: List[ShellMass]
    var nonlinear: List[ShellNLDKGQState]
    var corotational: List[ASDCorotational]
    var layered_geometry: List[Float64]
    var elastic_matrices: List[Float64]
    var concrete_states: List[ConcreteSState]
    var work: ParallelElementWork
    var has_alternative_shells: Bool

    fn __init__(out self):
        self.entry_by_element = []
        self.entries = []
        self.masses = []
        self.nonlinear = []
        self.corotational = []
        self.layered_geometry = []
        self.elastic_matrices = []
        self.concrete_states = []
        self.work = ParallelElementWork()
        self.has_alternative_shells = False

    fn commit_layered_states(mut self):
        """Commit every ConcreteS trial state. Rebar states commit with the uniaxial list."""
        for i in range(len(self.concrete_states)):
            self.concrete_states[i].commit()
        for i in range(len(self.nonlinear)):
            self.nonlinear[i].commit()
        for i in range(len(self.corotational)):
            self.corotational[i].commit()

    fn revert_trial(mut self):
        for i in range(len(self.nonlinear)):
            self.nonlinear[i].revert()
        for i in range(len(self.corotational)):
            self.corotational[i].revert()

    fn prepare(
        mut self,
        shell_indices: List[Int],
        elements: List[ElementInput],
        nodes: List[NodeInput],
        sections: List[SectionInput],
        layered_sections: List[LayeredShellSectionDef],
        section_index: List[Int],
        section_state_offsets: List[Int],
        element_instance_offsets: List[Int],
    ):
        if len(shell_indices) == 0:
            return

        self.entry_by_element.resize(len(elements), -1)
        var layered_count = 0
        var elastic_count = 0
        var concrete_state_count = 0
        for e in shell_indices:
            if elements[e].type != "shell":
                self.has_alternative_shells = True
            if (
                elements[e].type != "ASDShellQ4"
                and sections[elements[e].section].type == "LayeredShellSection"
            ):
                concrete_state_count += 4 * len(
                    layered_sections[section_index[elements[e].section]].concretes
                )
            if elements[e].type != "ShellNLDKGQ":
                if (
                    elements[e].type != "ASDShellQ4"
                    and sections[elements[e].section].type == "LayeredShellSection"
                ):
                    layered_count += 1
                else:
                    elastic_count += 1
        # One allocation: growing per element reallocates and copies every state each time.
        self.concrete_states.reserve(len(self.concrete_states) + concrete_state_count)
        self.layered_geometry.resize(layered_count * SHELL4_GEOMETRY_SIZE, 0.0)
        self.elastic_matrices.resize(elastic_count * SHELL4_MATRIX_SIZE, 0.0)
        self.work = ParallelElementWork(
            min(_BATCH_SIZE, len(shell_indices)),
            SHELL4_MATRIX_SIZE,
            SHELL4_DOFS,
            SHELL4_DOFS + SHELL4_SCRATCH_SIZE,
            _MINIMUM_WORK,
        )

        var layer_offset = 0
        var elastic_offset = 0
        var geometry = InlineArray[Float64, SHELL4_GEOMETRY_SIZE](fill=0.0)
        for e in shell_indices:
            ref elem = elements[e]
            ref sec = sections[elem.section]
            if elem.type != "ASDShellQ4" and sec.type == "LayeredShellSection":
                ref definition = layered_sections[section_index[elem.section]]
                self.work.plan.append(32 + 8 * (len(definition.concretes) + len(definition.rebars)))
            elif elem.type == "ShellNLDKGQ" or elem.type == "ASDShellQ4":
                self.work.plan.append(128)
            else:
                self.work.plan.append(8)
            ref n1 = nodes[elem.node_index_1]
            ref n2 = nodes[elem.node_index_2]
            ref n3 = nodes[elem.node_index_3]
            var x: List[Float64]
            var y: List[Float64]
            var z: List[Float64]
            if elem.node_count == 4:
                ref n4 = nodes[elem.node_index_4]
                x = [n1.x, n2.x, n3.x, n4.x]
                y = [n1.y, n2.y, n3.y, n4.y]
                z = [n1.z, n2.z, n3.z, n4.z]
            else:
                x = [n1.x, n2.x, n3.x]
                y = [n1.y, n2.y, n3.y]
                z = [n1.z, n2.z, n3.z]
            var is_discrete_kirchhoff = elem.type == "ShellDKGT" or elem.type == "ShellDKGQ"
            var nonlinear_index = -1
            if elem.type == "ShellNLDKGQ":
                nonlinear_index = len(self.nonlinear)
                var state = ShellNLDKGQState(x, y, z)
                if sec.type != "LayeredShellSection":
                    if sec.type == "PlateFiber":
                        state.elastic_section.elastic_tangent = elastic_plate_fiber_section_tangent(
                            sec.E, sec.nu, sec.h
                        )
                    else:
                        state.elastic_section.elastic_tangent = elastic_membrane_plate_tangent(
                            sec.E, sec.nu, sec.h, sec.plate_mod
                        )
                self.nonlinear.append(state^)
            self.entry_by_element[e] = len(self.entries)
            if elem.type == "ASDShellQ4":
                var tangent: InlineArray[Float64, 64]
                if sec.type == "LayeredShellSection":
                    ref definition = layered_sections[section_index[elem.section]]
                    if len(definition.rebars) != 0 or len(definition.concretes) != 0:
                        abort("ASDShellQ4 currently requires an elastic shell section")
                    tangent = definition.elastic_tangent.copy()
                elif sec.type == "PlateFiber":
                    tangent = elastic_plate_fiber_section_tangent(sec.E, sec.nu, sec.h)
                else:
                    tangent = elastic_membrane_plate_tangent(sec.E, sec.nu, sec.h, sec.plate_mod)
                var points = ASDPoints(fill=ASDPoint(0.0))
                for i in range(4):
                    points[i] = ASDPoint(x[i], y[i], z[i], 0.0)
                var frame = ASDFrame(points)
                var gp_geometry = asd_shell_elastic_geometry(
                    frame,
                    UnsafePointer(to=tangent[0]),
                    elem.shell_eas,
                    max(0.0, min(1.0, elem.shell_drilling_stab)),
                )
                var matrix = self.elastic_matrices.unsafe_ptr() + elastic_offset
                var local_stiffness = InlineArray[Float64, SHELL4_MATRIX_SIZE](fill=0.0)
                var local = UnsafePointer(to=local_stiffness[0])
                shell4_integrate_tangent(
                    UnsafePointer(to=tangent[0]),
                    UnsafePointer(to=gp_geometry[0]),
                    local,
                    0.233 * tangent[18],
                )
                var corotational_index = -1
                if elem.geom_tag == GeomTransfTag.Corotational:
                    corotational_index = len(self.corotational)
                    self.corotational.append(ASDCorotational(points))
                    for i in range(SHELL4_MATRIX_SIZE):
                        matrix[i] = local[i]
                else:
                    asd_linear_stiffness(local, frame, matrix)
                self.entries.append(
                    ShellAssemblyEntry(
                        -1, elastic_offset, 0, 0, SHELL4_DOFS, -1, corotational_index
                    )
                )
                elastic_offset += SHELL4_MATRIX_SIZE
                continue
            if sec.type == "LayeredShellSection":
                var section = section_index[elem.section]
                var instance = element_instance_offsets[e]
                ref definition = layered_sections[section]
                var state_offset = section_state_offsets[section] + instance * len(
                    definition.rebars
                )
                var concrete_offset = len(self.concrete_states)
                self.concrete_states.resize(
                    concrete_offset + 4 * len(definition.concretes), ConcreteSState()
                )
                self.entries.append(
                    ShellAssemblyEntry(
                        section,
                        layer_offset,
                        state_offset,
                        concrete_offset,
                        elem.dof_count,
                        nonlinear_index,
                        -1,
                    )
                )
                if nonlinear_index < 0:
                    var target = self.layered_geometry.unsafe_ptr() + layer_offset
                    if is_discrete_kirchhoff:
                        discrete_kirchhoff_geometry(x, y, z, target)
                    else:
                        shell4_precompute_geometry(x, y, z, target)
                    layer_offset += SHELL4_GEOMETRY_SIZE
            else:
                self.entries.append(
                    ShellAssemblyEntry(
                        -1, elastic_offset, 0, 0, elem.dof_count, nonlinear_index, -1
                    )
                )
                if nonlinear_index >= 0:
                    continue
                var gp_geometry = UnsafePointer(to=geometry[0])
                var matrix = self.elastic_matrices.unsafe_ptr() + elastic_offset
                if is_discrete_kirchhoff:
                    for i in range(SHELL4_GEOMETRY_SIZE):
                        gp_geometry[i] = 0.0
                    discrete_kirchhoff_geometry(x, y, z, gp_geometry)
                else:
                    shell4_precompute_geometry(x, y, z, gp_geometry)
                if sec.type == "PlateFiber":
                    var tangent = elastic_plate_fiber_section_tangent(sec.E, sec.nu, sec.h)
                    var drill = 0.0
                    if not is_discrete_kirchhoff:
                        drill = tangent[18]
                    shell4_integrate_tangent(
                        UnsafePointer(to=tangent[0]), gp_geometry, matrix, drill
                    )
                else:
                    # The DK Gauss blocks have zero independent drilling and shear rows.
                    shell4_mindlin_stiffness(
                        sec.E, sec.nu, sec.h, gp_geometry, matrix, sec.plate_mod
                    )
                if elem.dof_count != SHELL4_DOFS:
                    compact_shell_matrix(matrix, elem.dof_count)
                elastic_offset += SHELL4_MATRIX_SIZE
        self.work.prepare_scratch()
