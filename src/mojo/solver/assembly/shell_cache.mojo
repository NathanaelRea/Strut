"""Geometry and workspace owned by one loaded model."""

from collections import InlineArray, List
from elements.shell4 import (
    SHELL4_DOFS,
    SHELL4_GEOMETRY_SIZE,
    SHELL4_MATRIX_SIZE,
    SHELL4_SCRATCH_SIZE,
    shell4_mindlin_stiffness,
    shell4_precompute_geometry,
)
from os import abort
from sections import LayeredShellSectionDef
from solver.assembly.parallel_elements import ParallelElementWork
from solver.run_case.input_types import ElementInput, NodeInput, SectionInput
from solver.threads import solver_thread_count


# Bound result storage to about 2.5 MB, even for very large models.
alias _BATCH_SIZE = 512
# At 64 layered shells, two 32-element ranges nearly halve compute time.
alias _MIN_PER_WORKER = 32


@fieldwise_init
struct ShellAssemblyEntry(ImplicitlyCopyable, Movable):
    var section: Int
    var offset: Int
    var state_offset: Int


struct ShellAssemblyCache(Movable):
    """Element-major layered Gauss data and elastic stiffness matrices.

    Entries map the original element index to a layered geometry block or a
    cached elastic matrix. Each layered block holds four consecutive Gauss
    records of 8×24 row-major B, 24 drilling coefficients, and dvol. The basis
    is already incorporated in these rows, which act on global DOFs.
    Elastic blocks hold one row-major 24×24 matrix per element.
    """

    var entry_by_element: List[Int]
    var entries: List[ShellAssemblyEntry]
    var layered_geometry: List[Float64]
    var elastic_matrices: List[Float64]
    var work: ParallelElementWork

    fn __init__(out self):
        self.entry_by_element = []
        self.entries = []
        self.layered_geometry = []
        self.elastic_matrices = []
        self.work = ParallelElementWork()

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
        for e in shell_indices:
            if sections[elements[e].section].type == "LayeredShellSection":
                layered_count += 1
        self.layered_geometry.resize(layered_count * SHELL4_GEOMETRY_SIZE, 0.0)
        self.elastic_matrices.resize(
            (len(shell_indices) - layered_count) * SHELL4_MATRIX_SIZE, 0.0
        )
        self.work = ParallelElementWork(
            min(_BATCH_SIZE, len(shell_indices)),
            SHELL4_MATRIX_SIZE,
            SHELL4_DOFS,
            SHELL4_DOFS + SHELL4_SCRATCH_SIZE,
            solver_thread_count(),
            _MIN_PER_WORKER,
        )

        var layer_offset = 0
        var elastic_offset = 0
        var geometry = InlineArray[Float64, SHELL4_GEOMETRY_SIZE](fill=0.0)
        for e in shell_indices:
            ref elem = elements[e]
            ref sec = sections[elem.section]
            ref n1 = nodes[elem.node_index_1]
            ref n2 = nodes[elem.node_index_2]
            ref n3 = nodes[elem.node_index_3]
            ref n4 = nodes[elem.node_index_4]
            var x = [n1.x, n2.x, n3.x, n4.x]
            var y = [n1.y, n2.y, n3.y, n4.y]
            var z = [n1.z, n2.z, n3.z, n4.z]
            self.entry_by_element[e] = len(self.entries)
            if sec.type == "LayeredShellSection":
                var section = section_index[elem.section]
                var instance = element_instance_offsets[e]
                ref definition = layered_sections[section]
                var state_offset = section_state_offsets[section] + instance * len(definition.rebars)
                self.entries.append(ShellAssemblyEntry(section, layer_offset, state_offset))
                shell4_precompute_geometry(x, y, z, self.layered_geometry.unsafe_ptr() + layer_offset)
                layer_offset += SHELL4_GEOMETRY_SIZE
            elif sec.type == "ElasticMembranePlateSection":
                self.entries.append(ShellAssemblyEntry(-1, elastic_offset, 0))
                shell4_precompute_geometry(x, y, z, UnsafePointer(to=geometry[0]))
                shell4_mindlin_stiffness(
                    sec.E, sec.nu, sec.h, UnsafePointer(to=geometry[0]),
                    self.elastic_matrices.unsafe_ptr() + elastic_offset,
                )
                elastic_offset += SHELL4_MATRIX_SIZE
            else:
                abort("shell requires ElasticMembranePlateSection or LayeredShellSection")
