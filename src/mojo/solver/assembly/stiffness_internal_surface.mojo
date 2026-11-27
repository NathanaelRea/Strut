from collections import InlineArray, List
from elements.quad4 import QuadMatrix, QuadVector
from elements.quad_up import QuadUPVector
from elements.ssp_brick_up import BrickUPVector
from elements.shell4 import (
    SHELL4_DOFS,
    SHELL4_MATRIX_SIZE,
    SHELL4_SCRATCH_SIZE,
    compact_shell_matrix,
    shell4_dot_row,
    shell4_layered_stiffness_and_residual,
)
from elements.shell_asd_transform import ASDCorotational, asd_corotational_response
from elements.shell_nldkgq import ShellNLDKGQState
from elements.tri31 import tri31_elastic_stiffness
from materials import UniMaterialDef, UniMaterialState
from materials.concrete_s import ConcreteSState
from sections import LayeredShellSectionDef
from solver.assembly.shell_cache import ShellAssemblyCache, ShellAssemblyEntry
from solver.assembly.surface_cache import SurfaceAssemblyCache
from solver.assembly.stiffness_internal_shared import (
    _dot_row_simd,
    _scatter_add_and_dot_row_simd,
    _elem_dof,
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
    var nonlinear: UnsafePointer[ShellNLDKGQState]
    var corotational: UnsafePointer[ASDCorotational]

    @always_inline
    fn evaluate_special[
        tangent: Bool
    ](
        self,
        entry: ShellAssemblyEntry,
        u: UnsafePointer[Float64],
        matrix: UnsafePointer[Float64],
        force: UnsafePointer[Float64],
        scratch: UnsafePointer[Float64],
    ) -> Bool:
        if entry.nonlinear_index >= 0:
            ref state = self.nonlinear[entry.nonlinear_index]
            if entry.section < 0:
                state.evaluate[tangent](
                    state.elastic_section.copy(),
                    self.definitions,
                    self.states,
                    self.concrete_states,
                    u,
                    matrix,
                    force,
                    scratch,
                )
            else:
                state.evaluate[tangent](
                    self.section_defs[entry.section],
                    self.definitions,
                    self.states + entry.state_offset,
                    self.concrete_states + entry.concrete_offset,
                    u,
                    matrix,
                    force,
                    scratch,
                )
            return True
        if entry.corotational_index >= 0:
            var local = InlineArray[Float64, SHELL4_DOFS](fill=0.0)
            var local_u = UnsafePointer(to=local[0])
            var frame = self.corotational[entry.corotational_index].local_displacement(u, local_u)
            asd_corotational_response[tangent](
                frame, local_u, self.elastic + entry.offset, matrix, force
            )
            return True
        return False

    @always_inline
    fn evaluate[
        tangent: Bool,
        standard: Bool = False,
    ](
        self,
        entry: ShellAssemblyEntry,
        u: UnsafePointer[Float64],
        matrix: UnsafePointer[Float64],
        force: UnsafePointer[Float64],
        scratch: UnsafePointer[Float64],
    ):
        @parameter
        if not standard:
            if self.evaluate_special[tangent](entry, u, matrix, force, scratch):
                return
        if entry.section < 0:
            var stiffness = self.elastic + entry.offset

            @parameter
            if not standard:
                if entry.dof_count != SHELL4_DOFS:
                    for row in range(entry.dof_count):
                        var value = 0.0
                        for col in range(entry.dof_count):
                            value += stiffness[row * entry.dof_count + col] * u[col]
                        force[row] = value
                    return
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

            @parameter
            if tangent and not standard:
                if entry.dof_count != SHELL4_DOFS:
                    compact_shell_matrix(matrix, entry.dof_count)


@always_inline
fn _scatter_shell_dense(
    mut K: List[List[Float64]],
    dofs: UnsafePointer[Int],
    matrix: UnsafePointer[Float64],
    dof_count: Int,
):
    for row in range(dof_count):
        var destination = K[dofs[row]].unsafe_ptr()
        var source = matrix + row * dof_count
        for col in range(dof_count):
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
    tangent: Bool,
    standard: Bool,
](
    ctx: _ShellComputeContext,
    idx: Int,
    matrix: UnsafePointer[Float64],
    force: UnsafePointer[Float64],
    scratch: UnsafePointer[Float64],
):
    var e = ctx.maps.indices[idx]
    var dofs = ctx.maps.dofs(e)
    var entry = ctx.maps.entry(e)
    var dof_count = SHELL4_DOFS

    @parameter
    if not standard:
        dof_count = entry.dof_count
        for a in range(dof_count, SHELL4_DOFS):
            scratch[a] = 0.0
    for a in range(dof_count):
        scratch[a] = ctx.displacements[dofs[a]]
    ctx.data.evaluate[tangent, standard](
        entry,
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
    S: ElementMatrixSink,
    dense: Bool,
    standard: Bool,
](ctx: _ShellScatterContext[S], idx: Int, matrix_offset: Int, force_offset: Int,):
    var e = ctx.maps.indices[idx]
    var dofs = ctx.maps.dofs(e)
    var result = ctx.forces + force_offset
    var entry = ctx.maps.entry(e)
    var dof_count = SHELL4_DOFS

    @parameter
    if not standard:
        dof_count = entry.dof_count
    if ctx.tangent:
        var matrix = ctx.matrix_results
        var offset = matrix_offset
        var use_cached = entry.section < 0

        @parameter
        if not standard:
            use_cached = entry.uses_cached_matrix()
        if use_cached:
            matrix = ctx.elastic_results
            offset = entry.offset

        @parameter
        if dense:
            _scatter_shell_dense(ctx.dense_sink[], dofs, matrix[].unsafe_ptr() + offset, dof_count)
        else:
            ctx.matrix_sink[].add_element_matrix_flat(
                e,
                ctx.free_pool[],
                ctx.free_offsets[e],
                dof_count,
                matrix[],
                offset,
            )
    for a in range(dof_count):
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
        cache.nonlinear.unsafe_ptr(),
        cache.corotational.unsafe_ptr(),
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
        if cache.has_alternative_shells:
            if tangent:
                cache.work.compute[_ShellComputeContext, _compute_shell[True, False]](
                    compute, start, count
                )
            else:
                cache.work.compute[_ShellComputeContext, _compute_shell[False, False]](
                    compute, start, count
                )
        elif tangent:
            cache.work.compute[_ShellComputeContext, _compute_shell[True, True]](
                compute, start, count
            )
        else:
            cache.work.compute[_ShellComputeContext, _compute_shell[False, True]](
                compute, start, count
            )

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
        if cache.has_alternative_shells:
            cache.work.scatter[_ShellScatterContext[S], _scatter_shell[S, dense, False]](
                scatter, start, count
            )
        else:
            cache.work.scatter[_ShellScatterContext[S], _scatter_shell[S, dense, True]](
                scatter, start, count
            )
        if runtime_metrics.enabled:
            var end = Int(perf_counter_ns())
            runtime_metrics.shell_compute_ns += scatter_start - compute_start
            runtime_metrics.shell_scatter_ns += end - scatter_start
            runtime_metrics.element_type_call_counts[ElementTypeTag.Shell] += count
            runtime_metrics.element_type_total_ns[ElementTypeTag.Shell] += end - compute_start


fn _triangle_stiffness(
    elem: ElementInput,
    mat: MaterialInput,
    thickness: Float64,
    x: List[Float64],
    y: List[Float64],
) -> List[List[Float64]]:
    return tri31_elastic_stiffness(
        mat.E, mat.nu, thickness, elem.formulation == "PlaneStrain", x, y
    )


fn _assemble_surface_native_soa_indices[
    S: ElementMatrixSink
](
    elements: List[ElementInput],
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
    mut shell_cache: SurfaceAssemblyCache,
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
        var elem = elements[e]
        if elem.type_tag == ElementTypeTag.Tri31:
            var node_offset = elem_node_offsets[e]
            var mat = materials_by_id[elem_primary_material_ids[e]]
            var x: List[Float64] = []
            var y: List[Float64] = []
            for local_node in range(elem.node_count):
                var node_index = elem_node_pool[node_offset + local_node]
                x.append(node_x[node_index])
                y.append(node_y[node_index])
            var k_global = _triangle_stiffness(elem, mat, elem_thickness[e], x, y)
            var dof_map: List[Int] = []
            for a in range(elem.dof_count):
                dof_map.append(elem_dof_pool[elem_dof_offsets[e] + a])
            backend.add_element_matrix(
                e, elem_free_pool, elem_free_offsets[e], elem.dof_count, k_global
            )
            for a in range(elem.dof_count):
                F_int[dof_map[a]] += _dot_row_simd(k_global[a], dof_map, u, elem.dof_count)
        elif elem.type_tag == ElementTypeTag.BeamContact2D:
            var dof_offset = elem_dof_offsets[e]
            var flat: List[Float64] = []
            var tangent = backend.accepts_element_matrices()
            var force = shell_cache.evaluate_contact(e, u, tangent, flat)
            if tangent:
                backend.add_element_matrix_flat(
                    e, elem_free_pool, elem_free_offsets[e], elem.dof_count, flat
                )
            for i in range(elem.dof_count):
                F_int[elem_dof_pool[dof_offset + i]] += force[i]
        elif elem.type_tag == ElementTypeTag.SSPBrickUP:
            var dof_offset = elem_dof_offsets[e]
            var force = BrickUPVector(fill=0.0)
            var flat: List[Float64] = []
            var tangent = backend.accepts_element_matrices()
            shell_cache.evaluate_brick_up(e, dof_offset, elem_dof_pool, u, tangent, force, flat)
            if tangent:
                backend.add_element_matrix_flat(
                    e, elem_free_pool, elem_free_offsets[e], elem.dof_count, flat
                )
            for i in range(elem.dof_count):
                F_int[elem_dof_pool[dof_offset + i]] += force[i]
        elif elem.type_tag == ElementTypeTag.QuadUP:
            var dof_offset = elem_dof_offsets[e]
            var force = QuadUPVector(fill=0.0)
            var flat: List[Float64] = []
            var tangent = backend.accepts_element_matrices()
            shell_cache.evaluate_up(e, dof_offset, elem_dof_pool, u, tangent, force, flat)
            if tangent:
                backend.add_element_matrix_flat(
                    e, elem_free_pool, elem_free_offsets[e], elem.dof_count, flat
                )
            for i in range(elem.dof_count):
                F_int[elem_dof_pool[dof_offset + i]] += force[i]
        else:
            var dof_offset = elem_dof_offsets[e]
            var local_u = QuadVector(fill=0.0)
            for i in range(8):
                local_u[i] = u[elem_dof_pool[dof_offset + i]]
            var force = QuadVector(fill=0.0)
            var matrix = QuadMatrix(fill=0.0)
            var tangent = backend.accepts_element_matrices()
            shell_cache.quads[shell_cache.quad_index[e]].evaluate(
                shell_cache.pimy_definitions,
                shell_cache.pdmy_definitions,
                local_u,
                tangent,
                force,
                matrix,
            )
            if tangent:
                var flat: List[Float64] = []
                flat.resize(64, 0.0)
                for i in range(64):
                    flat[i] = matrix[i]
                backend.add_element_matrix_flat(e, elem_free_pool, elem_free_offsets[e], 8, flat)
            for i in range(8):
                F_int[elem_dof_pool[dof_offset + i]] += force[i]
        if runtime_metrics.enabled:
            _profile_metrics_note_element_timing(
                runtime_metrics,
                elem.type_tag,
                Int(perf_counter_ns()) - t_elem_start,
            )

    shell_cache.continuum.assemble_native[S](elem_free_offsets, elem_free_pool, u, backend, F_int)
    var unused_dense: List[List[Float64]] = []
    _assemble_shell_soa[S, False](
        shell_elem_indices,
        elem_dof_offsets,
        elem_dof_pool,
        elem_free_offsets,
        elem_free_pool,
        shell_cache.shell,
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
    elements: List[ElementInput],
    quad_elem_indices: List[Int],
    shell_elem_indices: List[Int],
    node_x: List[Float64],
    node_y: List[Float64],
    elem_dof_offsets: List[Int],
    elem_dof_pool: List[Int],
    elem_node_offsets: List[Int],
    elem_node_pool: List[Int],
    elem_primary_material_ids: List[Int],
    mut shell_cache: SurfaceAssemblyCache,
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
        var elem = elements[e]
        if elem.type_tag == ElementTypeTag.Tri31:
            var node_offset = elem_node_offsets[e]
            var mat = materials_by_id[elem_primary_material_ids[e]]
            var x: List[Float64] = []
            var y: List[Float64] = []
            for local_node in range(elem.node_count):
                var node_index = elem_node_pool[node_offset + local_node]
                x.append(node_x[node_index])
                y.append(node_y[node_index])
            var k_global = _triangle_stiffness(elem, mat, elem_thickness[e], x, y)
            var dof_map: List[Int] = []
            for a in range(elem.dof_count):
                dof_map.append(elem_dof_pool[elem_dof_offsets[e] + a])
            for a in range(elem.dof_count):
                var global_dof = dof_map[a]
                F_int[global_dof] += _scatter_add_and_dot_row_simd(
                    K, global_dof, k_global[a], dof_map, u, elem.dof_count
                )
        elif elem.type_tag == ElementTypeTag.BeamContact2D:
            var dof_offset = elem_dof_offsets[e]
            var flat: List[Float64] = []
            var force = shell_cache.evaluate_contact(e, u, True, flat)
            for i in range(elem.dof_count):
                var row = elem_dof_pool[dof_offset + i]
                F_int[row] += force[i]
                for j in range(elem.dof_count):
                    K[row][elem_dof_pool[dof_offset + j]] += flat[i * elem.dof_count + j]
        elif elem.type_tag == ElementTypeTag.SSPBrickUP:
            var dof_offset = elem_dof_offsets[e]
            var force = BrickUPVector(fill=0.0)
            var flat: List[Float64] = []
            shell_cache.evaluate_brick_up(e, dof_offset, elem_dof_pool, u, True, force, flat)
            for i in range(elem.dof_count):
                var row = elem_dof_pool[dof_offset + i]
                F_int[row] += force[i]
                for j in range(elem.dof_count):
                    K[row][elem_dof_pool[dof_offset + j]] += flat[i * elem.dof_count + j]
        elif elem.type_tag == ElementTypeTag.QuadUP:
            var dof_offset = elem_dof_offsets[e]
            var force = QuadUPVector(fill=0.0)
            var flat: List[Float64] = []
            shell_cache.evaluate_up(e, dof_offset, elem_dof_pool, u, True, force, flat)
            for i in range(elem.dof_count):
                var row = elem_dof_pool[dof_offset + i]
                F_int[row] += force[i]
                for j in range(elem.dof_count):
                    K[row][elem_dof_pool[dof_offset + j]] += flat[i * elem.dof_count + j]
        else:
            var dof_offset = elem_dof_offsets[e]
            var local_u = QuadVector(fill=0.0)
            for i in range(8):
                local_u[i] = u[elem_dof_pool[dof_offset + i]]
            var force = QuadVector(fill=0.0)
            var matrix = QuadMatrix(fill=0.0)
            shell_cache.quads[shell_cache.quad_index[e]].evaluate(
                shell_cache.pimy_definitions,
                shell_cache.pdmy_definitions,
                local_u,
                True,
                force,
                matrix,
            )
            for i in range(8):
                var row = elem_dof_pool[dof_offset + i]
                F_int[row] += force[i]
                for j in range(8):
                    K[row][elem_dof_pool[dof_offset + j]] += matrix[8 * i + j]
        if runtime_metrics.enabled:
            _profile_metrics_note_element_timing(
                runtime_metrics,
                elem.type_tag,
                Int(perf_counter_ns()) - t_elem_start,
            )

    shell_cache.continuum.assemble_dense(u, K, F_int)
    var unused_sink = NullMatrixSink()
    var unused_free: List[Int] = []
    _assemble_shell_soa[NullMatrixSink, True](
        shell_elem_indices,
        elem_dof_offsets,
        elem_dof_pool,
        unused_free,
        unused_free,
        shell_cache.shell,
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
    mut shell_cache: SurfaceAssemblyCache,
    layered_shell_section_defs: List[LayeredShellSectionDef],
    uniaxial_defs: List[UniMaterialDef],
    mut uniaxial_states: List[UniMaterialState],
    u: List[Float64],
    mut K: List[List[Float64]],
    mut F_int: List[Float64],
) raises:
    if elem.type_tag == ElementTypeTag.Tri31:
        var node_indices = [
            elem.node_index_1,
            elem.node_index_2,
            elem.node_index_3,
            elem.node_index_4,
        ]
        var x: List[Float64] = []
        var y: List[Float64] = []
        for a in range(elem.node_count):
            x.append(nodes[node_indices[a]].x)
            y.append(nodes[node_indices[a]].y)
        var mat = materials_by_id[elem.material]
        var k_global = _triangle_stiffness(elem, mat, elem.thickness, x, y)
        var dof_map: List[Int] = []
        for a in range(elem.dof_count):
            dof_map.append(_elem_dof(elem, a))
        for a in range(elem.dof_count):
            var global_dof = dof_map[a]
            F_int[global_dof] += _scatter_add_and_dot_row_simd(
                K, global_dof, k_global[a], dof_map, u, elem.dof_count
            )
        return

    if elem.type_tag == ElementTypeTag.BeamContact2D:
        var flat: List[Float64] = []
        var force = shell_cache.evaluate_contact(e, u, True, flat)
        for i in range(elem.dof_count):
            var row = _elem_dof(elem, i)
            F_int[row] += force[i]
            for j in range(elem.dof_count):
                K[row][_elem_dof(elem, j)] += flat[i * elem.dof_count + j]
        return

    if elem.type_tag == ElementTypeTag.SSPBrickUP:
        var dof_map: List[Int] = []
        for i in range(elem.dof_count):
            dof_map.append(_elem_dof(elem, i))
        var force = BrickUPVector(fill=0.0)
        var flat: List[Float64] = []
        shell_cache.evaluate_brick_up(e, 0, dof_map, u, True, force, flat)
        for i in range(elem.dof_count):
            F_int[dof_map[i]] += force[i]
            for j in range(elem.dof_count):
                K[dof_map[i]][dof_map[j]] += flat[i * elem.dof_count + j]
        return
    if elem.type_tag == ElementTypeTag.QuadUP:
        var dof_map: List[Int] = []
        for i in range(elem.dof_count):
            dof_map.append(_elem_dof(elem, i))
        var force = QuadUPVector(fill=0.0)
        var flat: List[Float64] = []
        shell_cache.evaluate_up(e, 0, dof_map, u, True, force, flat)
        for i in range(elem.dof_count):
            F_int[dof_map[i]] += force[i]
            for j in range(elem.dof_count):
                K[dof_map[i]][dof_map[j]] += flat[i * elem.dof_count + j]
        return

    if elem.type_tag == ElementTypeTag.FourNodeQuad:
        var local_u = QuadVector(fill=0.0)
        for i in range(8):
            local_u[i] = u[_elem_dof(elem, i)]
        var force = QuadVector(fill=0.0)
        var matrix = QuadMatrix(fill=0.0)
        shell_cache.quads[shell_cache.quad_index[e]].evaluate(
            shell_cache.pimy_definitions, shell_cache.pdmy_definitions, local_u, True, force, matrix
        )
        for i in range(8):
            var row = _elem_dof(elem, i)
            F_int[row] += force[i]
            for j in range(8):
                K[row][_elem_dof(elem, j)] += matrix[8 * i + j]
        return

    if elem.type_tag == ElementTypeTag.Shell:
        var data = _ShellComputeData(
            layered_shell_section_defs.unsafe_ptr(),
            uniaxial_defs.unsafe_ptr(),
            uniaxial_states.unsafe_ptr(),
            shell_cache.shell.concrete_states.unsafe_ptr(),
            shell_cache.shell.layered_geometry.unsafe_ptr(),
            shell_cache.shell.elastic_matrices.unsafe_ptr(),
            shell_cache.shell.nonlinear.unsafe_ptr(),
            shell_cache.shell.corotational.unsafe_ptr(),
        )
        var entry = shell_cache.shell.entries[shell_cache.shell.entry_by_element[e]]
        var dof_map = InlineArray[Int, SHELL4_DOFS](fill=0)
        var scratch = InlineArray[Float64, SHELL4_DOFS + SHELL4_SCRATCH_SIZE](fill=0.0)
        for a in range(entry.dof_count):
            dof_map[a] = _elem_dof(elem, a)
            scratch[a] = u[dof_map[a]]
        var force = shell_cache.shell.work.forces.unsafe_ptr()
        var matrix = shell_cache.shell.work.matrices.unsafe_ptr()
        var displacement = UnsafePointer(to=scratch[0])
        data.evaluate[True](entry, displacement, matrix, force, displacement + SHELL4_DOFS)
        if entry.uses_cached_matrix():
            matrix = data.elastic + entry.offset
        _scatter_shell_dense(K, UnsafePointer(to=dof_map[0]), matrix, entry.dof_count)
        for a in range(entry.dof_count):
            F_int[dof_map[a]] += force[a]
