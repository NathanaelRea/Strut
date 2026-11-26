from collections import InlineArray, List
from elements import quad4_plane_stress_stiffness
from elements.shell4 import (
    SHELL4_DOFS,
    SHELL4_MATRIX_SIZE,
    SHELL4_SCRATCH_SIZE,
    shell4_dot_row,
    shell4_layered_stiffness_and_residual,
)
from materials import UniMaterialDef, UniMaterialState
from materials.concrete_s import ConcreteSState
from sections import LayeredShellSectionDef
from solver.assembly.shell_cache import ShellAssemblyCache, ShellAssemblyEntry
from solver.assembly.stiffness_internal_shared import (
    _dot_row_simd,
    _elem_dof,
    _scatter_add_and_dot_row_simd,
)
from solver.profile import RuntimeProfileMetrics, _profile_metrics_note_element_timing
from solver.run_case.input_types import ElementInput, MaterialInput, NodeInput
from solver.sparse_pattern import ElementMatrixSink, NullMatrixSink
from tag_types import ElementTypeTag
from time import perf_counter_ns


@fieldwise_init
struct _ShellComputeData(ImplicitlyCopyable):
    var section_defs: UnsafePointer[LayeredShellSectionDef]
    var definitions: UnsafePointer[UniMaterialDef]
    var states: UnsafePointer[UniMaterialState]
    var concrete_states: UnsafePointer[ConcreteSState]
    var geometry: UnsafePointer[Float64]
    var elastic: UnsafePointer[Float64]

    @always_inline
    fn evaluate[
        tangent: Bool
    ](
        self,
        entry: ShellAssemblyEntry,
        u: UnsafePointer[Float64],
        matrix: UnsafePointer[Float64],
        force: UnsafePointer[Float64],
        scratch: UnsafePointer[Float64],
    ):
        if entry.section < 0:
            var stiffness = self.elastic + entry.offset
            for row in range(SHELL4_DOFS):
                force[row] = shell4_dot_row(stiffness + row * SHELL4_DOFS, u)
        else:
            shell4_layered_stiffness_and_residual[tangent](
                self.section_defs[entry.section],
                self.definitions,
                self.states + entry.state_offset,
                self.concrete_states + entry.concrete_offset,
                self.geometry + entry.offset,
                u,
                matrix,
                force,
                scratch,
            )


fn _scatter_shell_dense(
    mut K: List[List[Float64]],
    dofs: UnsafePointer[Int],
    matrix: UnsafePointer[Float64],
):
    for row in range(SHELL4_DOFS):
        var destination = K[dofs[row]].unsafe_ptr()
        var source = matrix + row * SHELL4_DOFS
        for col in range(SHELL4_DOFS):
            destination[dofs[col]] += source[col]


@fieldwise_init
struct _ShellElementMaps(ImplicitlyCopyable):
    var indices: UnsafePointer[Int]
    var entry_by_element: UnsafePointer[Int]
    var entries: UnsafePointer[ShellAssemblyEntry]
    var dof_offsets: UnsafePointer[Int]
    var dof_pool: UnsafePointer[Int]

    @always_inline
    fn entry(self, e: Int) -> ShellAssemblyEntry:
        return self.entries[self.entry_by_element[e]]

    @always_inline
    fn dofs(self, e: Int) -> UnsafePointer[Int]:
        return self.dof_pool + self.dof_offsets[e]


@fieldwise_init
struct _ShellComputeContext(ImplicitlyCopyable):
    var maps: _ShellElementMaps
    var data: _ShellComputeData
    var displacements: UnsafePointer[Float64]


@always_inline
fn _compute_shell[
    tangent: Bool
](
    ctx: _ShellComputeContext,
    idx: Int,
    matrix: UnsafePointer[Float64],
    force: UnsafePointer[Float64],
    scratch: UnsafePointer[Float64],
):
    var e = ctx.maps.indices[idx]
    var dofs = ctx.maps.dofs(e)
    for a in range(SHELL4_DOFS):
        scratch[a] = ctx.displacements[dofs[a]]
    ctx.data.evaluate[tangent](
        ctx.maps.entry(e),
        scratch,
        matrix,
        force,
        scratch + SHELL4_DOFS,
    )


@fieldwise_init
struct _ShellScatterContext[S: ElementMatrixSink](ImplicitlyCopyable):
    var maps: _ShellElementMaps
    var tangent: Bool
    var matrix_results: UnsafePointer[List[Float64]]
    var elastic_results: UnsafePointer[List[Float64]]
    var forces: UnsafePointer[Float64]
    var force_out: UnsafePointer[Float64]
    var dense_sink: UnsafePointer[List[List[Float64]]]
    var matrix_sink: UnsafePointer[S]
    var free_offsets: UnsafePointer[Int]
    var free_pool: UnsafePointer[List[Int]]


@always_inline
fn _scatter_shell[
    S: ElementMatrixSink, dense: Bool
](ctx: _ShellScatterContext[S], idx: Int, matrix_offset: Int, force_offset: Int,):
    var e = ctx.maps.indices[idx]
    var dofs = ctx.maps.dofs(e)
    var result = ctx.forces + force_offset
    if ctx.tangent:
        var entry = ctx.maps.entry(e)
        var matrix = ctx.matrix_results
        var offset = matrix_offset
        if entry.section < 0:
            matrix = ctx.elastic_results
            offset = entry.offset

        @parameter
        if dense:
            _scatter_shell_dense(ctx.dense_sink[], dofs, matrix[].unsafe_ptr() + offset)
        else:
            ctx.matrix_sink[].add_element_matrix_flat(
                e,
                ctx.free_pool[],
                ctx.free_offsets[e],
                SHELL4_DOFS,
                matrix[],
                offset,
            )
    for a in range(SHELL4_DOFS):
        ctx.force_out[dofs[a]] += result[a]


fn _assemble_shell_soa[
    S: ElementMatrixSink, dense: Bool
](
    shell_indices: List[Int],
    elem_dof_offsets: List[Int],
    elem_dof_pool: List[Int],
    elem_free_offsets: List[Int],
    elem_free_pool: List[Int],
    mut cache: ShellAssemblyCache,
    section_defs: List[LayeredShellSectionDef],
    definitions: List[UniMaterialDef],
    mut states: List[UniMaterialState],
    u: List[Float64],
    mut backend: S,
    mut K: List[List[Float64]],
    mut F_int: List[Float64],
    mut runtime_metrics: RuntimeProfileMetrics,
):
    var length = len(shell_indices)
    if length == 0:
        return
    var data = _ShellComputeData(
        section_defs.unsafe_ptr(),
        definitions.unsafe_ptr(),
        states.unsafe_ptr(),
        cache.concrete_states.unsafe_ptr(),
        cache.layered_geometry.unsafe_ptr(),
        cache.elastic_matrices.unsafe_ptr(),
    )
    var maps = _ShellElementMaps(
        shell_indices.unsafe_ptr(),
        cache.entry_by_element.unsafe_ptr(),
        cache.entries.unsafe_ptr(),
        elem_dof_offsets.unsafe_ptr(),
        elem_dof_pool.unsafe_ptr(),
    )
    var tangent = dense or backend.accepts_element_matrices()
    var compute = _ShellComputeContext(maps, data, u.unsafe_ptr())

    for start in range(0, length, cache.work.capacity):
        var count = min(length - start, cache.work.capacity)
        var compute_start = 0
        if runtime_metrics.enabled:
            compute_start = Int(perf_counter_ns())
        if tangent:
            cache.work.compute[_ShellComputeContext, _compute_shell[True]](compute, start, count)
        else:
            cache.work.compute[_ShellComputeContext, _compute_shell[False]](compute, start, count)

        var scatter_start = 0
        if runtime_metrics.enabled:
            scatter_start = Int(perf_counter_ns())
        var scatter = _ShellScatterContext[S](
            maps,
            tangent,
            UnsafePointer(to=cache.work.matrices),
            UnsafePointer(to=cache.elastic_matrices),
            cache.work.forces.unsafe_ptr(),
            F_int.unsafe_ptr(),
            UnsafePointer(to=K),
            UnsafePointer(to=backend),
            elem_free_offsets.unsafe_ptr(),
            UnsafePointer(to=elem_free_pool),
        )
        cache.work.scatter[_ShellScatterContext[S], _scatter_shell[S, dense]](scatter, start, count)
        if runtime_metrics.enabled:
            var end = Int(perf_counter_ns())
            runtime_metrics.shell_compute_ns += scatter_start - compute_start
            runtime_metrics.shell_scatter_ns += end - scatter_start
            runtime_metrics.element_type_call_counts[ElementTypeTag.Shell] += count
            runtime_metrics.element_type_total_ns[ElementTypeTag.Shell] += end - compute_start


fn _assemble_surface_native_soa_indices[
    S: ElementMatrixSink
](
    quad_elem_indices: List[Int],
    shell_elem_indices: List[Int],
    node_x: List[Float64],
    node_y: List[Float64],
    elem_dof_offsets: List[Int],
    elem_dof_pool: List[Int],
    elem_free_offsets: List[Int],
    elem_free_pool: List[Int],
    elem_node_offsets: List[Int],
    elem_node_pool: List[Int],
    elem_primary_material_ids: List[Int],
    mut shell_cache: ShellAssemblyCache,
    elem_thickness: List[Float64],
    materials_by_id: List[MaterialInput],
    layered_shell_section_defs: List[LayeredShellSectionDef],
    uniaxial_defs: List[UniMaterialDef],
    mut uniaxial_states: List[UniMaterialState],
    u: List[Float64],
    mut backend: S,
    mut F_int: List[Float64],
    mut runtime_metrics: RuntimeProfileMetrics,
) raises:
    for idx in range(len(quad_elem_indices)):
        var e = quad_elem_indices[idx]
        var t_elem_start = 0
        if runtime_metrics.enabled:
            t_elem_start = Int(perf_counter_ns())
        var node_offset = elem_node_offsets[e]
        var mat = materials_by_id[elem_primary_material_ids[e]]
        var x = [
            node_x[elem_node_pool[node_offset]],
            node_x[elem_node_pool[node_offset + 1]],
            node_x[elem_node_pool[node_offset + 2]],
            node_x[elem_node_pool[node_offset + 3]],
        ]
        var y = [
            node_y[elem_node_pool[node_offset]],
            node_y[elem_node_pool[node_offset + 1]],
            node_y[elem_node_pool[node_offset + 2]],
            node_y[elem_node_pool[node_offset + 3]],
        ]
        var k_global = quad4_plane_stress_stiffness(mat.E, mat.nu, elem_thickness[e], x, y)
        var dof_offset = elem_dof_offsets[e]
        var free_offset = elem_free_offsets[e]
        var dof_map = [
            elem_dof_pool[dof_offset],
            elem_dof_pool[dof_offset + 1],
            elem_dof_pool[dof_offset + 2],
            elem_dof_pool[dof_offset + 3],
            elem_dof_pool[dof_offset + 4],
            elem_dof_pool[dof_offset + 5],
            elem_dof_pool[dof_offset + 6],
            elem_dof_pool[dof_offset + 7],
        ]
        backend.add_element_matrix(e, elem_free_pool, free_offset, 8, k_global)
        for a in range(8):
            F_int[dof_map[a]] += _dot_row_simd(k_global[a], dof_map, u, 8)
        if runtime_metrics.enabled:
            _profile_metrics_note_element_timing(
                runtime_metrics,
                ElementTypeTag.FourNodeQuad,
                Int(perf_counter_ns()) - t_elem_start,
            )

    var unused_dense: List[List[Float64]] = []
    _assemble_shell_soa[S, False](
        shell_elem_indices,
        elem_dof_offsets,
        elem_dof_pool,
        elem_free_offsets,
        elem_free_pool,
        shell_cache,
        layered_shell_section_defs,
        uniaxial_defs,
        uniaxial_states,
        u,
        backend,
        unused_dense,
        F_int,
        runtime_metrics,
    )


fn _assemble_surface_soa_indices(
    quad_elem_indices: List[Int],
    shell_elem_indices: List[Int],
    node_x: List[Float64],
    node_y: List[Float64],
    elem_dof_offsets: List[Int],
    elem_dof_pool: List[Int],
    elem_node_offsets: List[Int],
    elem_node_pool: List[Int],
    elem_primary_material_ids: List[Int],
    mut shell_cache: ShellAssemblyCache,
    elem_thickness: List[Float64],
    materials_by_id: List[MaterialInput],
    layered_shell_section_defs: List[LayeredShellSectionDef],
    uniaxial_defs: List[UniMaterialDef],
    mut uniaxial_states: List[UniMaterialState],
    u: List[Float64],
    mut K: List[List[Float64]],
    mut F_int: List[Float64],
    mut runtime_metrics: RuntimeProfileMetrics,
) raises:
    for idx in range(len(quad_elem_indices)):
        var e = quad_elem_indices[idx]
        var t_elem_start = 0
        if runtime_metrics.enabled:
            t_elem_start = Int(perf_counter_ns())
        var node_offset = elem_node_offsets[e]
        var mat = materials_by_id[elem_primary_material_ids[e]]
        var x = [
            node_x[elem_node_pool[node_offset]],
            node_x[elem_node_pool[node_offset + 1]],
            node_x[elem_node_pool[node_offset + 2]],
            node_x[elem_node_pool[node_offset + 3]],
        ]
        var y = [
            node_y[elem_node_pool[node_offset]],
            node_y[elem_node_pool[node_offset + 1]],
            node_y[elem_node_pool[node_offset + 2]],
            node_y[elem_node_pool[node_offset + 3]],
        ]
        var k_global = quad4_plane_stress_stiffness(mat.E, mat.nu, elem_thickness[e], x, y)
        var dof_offset = elem_dof_offsets[e]
        var dof_map = [
            elem_dof_pool[dof_offset],
            elem_dof_pool[dof_offset + 1],
            elem_dof_pool[dof_offset + 2],
            elem_dof_pool[dof_offset + 3],
            elem_dof_pool[dof_offset + 4],
            elem_dof_pool[dof_offset + 5],
            elem_dof_pool[dof_offset + 6],
            elem_dof_pool[dof_offset + 7],
        ]
        for a in range(8):
            var Aidx = dof_map[a]
            F_int[Aidx] += _scatter_add_and_dot_row_simd(K, Aidx, k_global[a], dof_map, u, 8)
        if runtime_metrics.enabled:
            _profile_metrics_note_element_timing(
                runtime_metrics,
                ElementTypeTag.FourNodeQuad,
                Int(perf_counter_ns()) - t_elem_start,
            )

    var unused_sink = NullMatrixSink()
    var unused_free: List[Int] = []
    _assemble_shell_soa[NullMatrixSink, True](
        shell_elem_indices,
        elem_dof_offsets,
        elem_dof_pool,
        unused_free,
        unused_free,
        shell_cache,
        layered_shell_section_defs,
        uniaxial_defs,
        uniaxial_states,
        u,
        unused_sink,
        K,
        F_int,
        runtime_metrics,
    )


fn _assemble_surface_element(
    e: Int,
    elem: ElementInput,
    nodes: List[NodeInput],
    materials_by_id: List[MaterialInput],
    mut shell_cache: ShellAssemblyCache,
    layered_shell_section_defs: List[LayeredShellSectionDef],
    uniaxial_defs: List[UniMaterialDef],
    mut uniaxial_states: List[UniMaterialState],
    u: List[Float64],
    mut K: List[List[Float64]],
    mut F_int: List[Float64],
) raises:
    if elem.type_tag == ElementTypeTag.FourNodeQuad:
        var node1 = nodes[elem.node_index_1]
        var node2 = nodes[elem.node_index_2]
        var node3 = nodes[elem.node_index_3]
        var node4 = nodes[elem.node_index_4]
        var mat = materials_by_id[elem.material]
        var x: List[Float64] = []
        var y: List[Float64] = []
        x.resize(4, 0.0)
        y.resize(4, 0.0)
        x[0] = node1.x
        y[0] = node1.y
        x[1] = node2.x
        y[1] = node2.y
        x[2] = node3.x
        y[2] = node3.y
        x[3] = node4.x
        y[3] = node4.y
        var k_global = quad4_plane_stress_stiffness(mat.E, mat.nu, elem.thickness, x, y)
        var dof_map = [
            _elem_dof(elem, 0),
            _elem_dof(elem, 1),
            _elem_dof(elem, 2),
            _elem_dof(elem, 3),
            _elem_dof(elem, 4),
            _elem_dof(elem, 5),
            _elem_dof(elem, 6),
            _elem_dof(elem, 7),
        ]
        for a in range(8):
            var Aidx = dof_map[a]
            F_int[Aidx] += _scatter_add_and_dot_row_simd(K, Aidx, k_global[a], dof_map, u, 8)
        return

    if elem.type_tag == ElementTypeTag.Shell:
        var data = _ShellComputeData(
            layered_shell_section_defs.unsafe_ptr(),
            uniaxial_defs.unsafe_ptr(),
            uniaxial_states.unsafe_ptr(),
            shell_cache.concrete_states.unsafe_ptr(),
            shell_cache.layered_geometry.unsafe_ptr(),
            shell_cache.elastic_matrices.unsafe_ptr(),
        )
        var entry = shell_cache.entries[shell_cache.entry_by_element[e]]
        var dof_map = InlineArray[Int, SHELL4_DOFS](fill=0)
        var scratch = InlineArray[Float64, SHELL4_DOFS + SHELL4_SCRATCH_SIZE](fill=0.0)
        for a in range(SHELL4_DOFS):
            dof_map[a] = _elem_dof(elem, a)
            scratch[a] = u[dof_map[a]]
        var force = shell_cache.work.forces.unsafe_ptr()
        var matrix = shell_cache.work.matrices.unsafe_ptr()
        var displacement = UnsafePointer(to=scratch[0])
        data.evaluate[True](entry, displacement, matrix, force, displacement + SHELL4_DOFS)
        if entry.section < 0:
            matrix = data.elastic + entry.offset
        _scatter_shell_dense(K, UnsafePointer(to=dof_map[0]), matrix)
        for a in range(SHELL4_DOFS):
            F_int[dof_map[a]] += force[a]
