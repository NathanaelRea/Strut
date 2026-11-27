from algorithm import parallelize
from collections import List
from os import abort
from time import perf_counter_ns

from elements import (
    ForceBeamColumn3dScratch,
    beam_column3d_fiber_global_tangent_and_internal,
    disp_beam_column3d_global_tangent_and_internal,
    force_beam_column3d_fiber_global_tangent_and_internal,
    force_beam_column3d_global_tangent_and_internal,
)
from materials import UniMaterialDef, UniMaterialState
from solver.assembly.frame3d_scratch import (
    FRAME3D_BATCH,
    FRAME3D_DOFS,
    FRAME3D_MATRIX_SIZE,
    FRAME3D_MIN_PER_WORKER,
    Frame3dAssemblyScratch,
)
from solver.assembly.stiffness_internal_shared import (
    _beam_integration_name_from_tag,
    _geom_transf_name_from_tag,
    _profile_scope_close,
    _profile_scope_open,
    _scatter_add_row,
)
from solver.profile import (
    PROFILE_FRAME_ASSEMBLE_FIBER_GEOMETRY,
    PROFILE_FRAME_ASSEMBLE_FIBER_INTERNAL_FORCE,
    PROFILE_FRAME_ASSEMBLE_FIBER_MATRIX_SCATTER,
    PROFILE_FRAME_ASSEMBLE_FIBER_SECTION_RESPONSE,
    RuntimeProfileMetrics,
    _profile_metrics_note_element_timing,
)
from solver.sparse_pattern import ElementMatrixSink
from solver.run_case.input_types import (
    ElementInput,
    ElementLoadInput,
    NodeInput,
    SectionInput,
)
from sections import FiberCell, FiberSection3dDef
from tag_types import ElementTypeTag, GeomTransfTag


fn _frame3d_element_tangent_and_force(
    e: Int,
    elements: List[ElementInput],
    node_x: List[Float64],
    node_y: List[Float64],
    node_z: List[Float64],
    elem_dof_offsets: List[Int],
    elem_dof_pool: List[Int],
    elem_node_offsets: List[Int],
    elem_node_pool: List[Int],
    elem_type_tags: List[Int],
    elem_geom_tags: List[Int],
    elem_section_ids: List[Int],
    elem_integration_tags: List[Int],
    elem_num_int_pts: List[Int],
    element_loads: List[ElementLoadInput],
    elem_load_offsets: List[Int],
    elem_load_pool: List[Int],
    load_scale: Float64,
    sections_by_id: List[SectionInput],
    u: List[Float64],
    uniaxial_defs: List[UniMaterialDef],
    mut uniaxial_states: List[UniMaterialState],
    elem_uniaxial_offsets: List[Int],
    elem_uniaxial_counts: List[Int],
    elem_uniaxial_state_ids: List[Int],
    force_basic_offsets: List[Int],
    force_basic_counts: List[Int],
    mut force_basic_q: List[Float64],
    mut fiber_section3d_defs: List[FiberSection3dDef],
    fiber_section3d_cells: List[FiberCell],
    fiber_section3d_index_by_id: List[Int],
    mut scratch: ForceBeamColumn3dScratch,
    mut u_elem12: List[Float64],
    mut k_elem12: List[List[Float64]],
    mut f_elem12: List[Float64],
):
    """Global tangent and internal force of one 3D beam-column element.

    Touches only this element's state (its fiber instances and basic forces)
    plus the scratch and output buffers passed in, so distinct elements may be
    evaluated concurrently when each caller brings its own scratch and buffers.
    """
    ref elem = elements[e]
    var node_offset = elem_node_offsets[e]
    var i1 = elem_node_pool[node_offset]
    var i2 = elem_node_pool[node_offset + 1]
    var x1 = node_x[i1]
    var y1 = node_y[i1]
    var z1 = node_z[i1]
    var x2 = node_x[i2]
    var y2 = node_y[i2]
    var z2 = node_z[i2]
    ref sec = sections_by_id[elem_section_ids[e]]
    var dof_offset = elem_dof_offsets[e]
    for i in range(FRAME3D_DOFS):
        u_elem12[i] = u[elem_dof_pool[dof_offset + i]]
    var elem_type = elem_type_tags[e]
    var geom_name = _geom_transf_name_from_tag(elem_geom_tags[e])
    if elem_type == ElementTypeTag.ElasticBeamColumn3d:
        if sec.type != "ElasticSection3d":
            abort("elasticBeamColumn3d requires ElasticSection3d")
        force_beam_column3d_global_tangent_and_internal(
            e,
            x1,
            y1,
            z1,
            x2,
            y2,
            z2,
            u_elem12,
            geom_name,
            elem.has_geom_vecxz,
            elem.geom_vecxz_1,
            elem.geom_vecxz_2,
            elem.geom_vecxz_3,
            element_loads,
            elem_load_offsets,
            elem_load_pool,
            load_scale,
            sec.E,
            sec.A,
            sec.Iy,
            sec.Iz,
            sec.beam_shear_modulus(elem_type),
            sec.J,
            scratch,
            k_elem12,
            f_elem12,
        )
    elif sec.type == "ElasticSection3d":
        if elem_type == ElementTypeTag.ForceBeamColumn3d:
            force_beam_column3d_global_tangent_and_internal(
                e,
                x1,
                y1,
                z1,
                x2,
                y2,
                z2,
                u_elem12,
                geom_name,
                elem.has_geom_vecxz,
                elem.geom_vecxz_1,
                elem.geom_vecxz_2,
                elem.geom_vecxz_3,
                element_loads,
                elem_load_offsets,
                elem_load_pool,
                load_scale,
                sec.E,
                sec.A,
                sec.Iy,
                sec.Iz,
                sec.beam_shear_modulus(elem_type),
                sec.J,
                scratch,
                k_elem12,
                f_elem12,
                sec.shear_rigidity_y,
                sec.shear_rigidity_z,
                elem.integration,
                elem.num_int_pts,
                elem.cbdi,
            )
        else:
            disp_beam_column3d_global_tangent_and_internal(
                e,
                x1,
                y1,
                z1,
                x2,
                y2,
                z2,
                u_elem12,
                geom_name,
                elem.has_geom_vecxz,
                elem.geom_vecxz_1,
                elem.geom_vecxz_2,
                elem.geom_vecxz_3,
                element_loads,
                elem_load_offsets,
                elem_load_pool,
                load_scale,
                sec.E,
                sec.A,
                sec.Iy,
                sec.Iz,
                sec.beam_shear_modulus(elem_type),
                sec.J,
                scratch,
                k_elem12,
                f_elem12,
            )
    elif sec.type == "FiberSection3d":
        var sec_index = fiber_section3d_index_by_id[elem_section_ids[e]]
        if sec_index < 0 or sec_index >= len(fiber_section3d_defs):
            abort("fiber section not found")
        var integration_name = _beam_integration_name_from_tag(elem_integration_tags[e])
        if elem_type == ElementTypeTag.ForceBeamColumn3d:
            force_beam_column3d_fiber_global_tangent_and_internal(
                e,
                x1,
                y1,
                z1,
                x2,
                y2,
                z2,
                u_elem12,
                geom_name,
                elem.has_geom_vecxz,
                elem.geom_vecxz_1,
                elem.geom_vecxz_2,
                elem.geom_vecxz_3,
                element_loads,
                elem_load_offsets,
                elem_load_pool,
                load_scale,
                fiber_section3d_defs[sec_index],
                fiber_section3d_cells,
                uniaxial_defs,
                uniaxial_states,
                elem_uniaxial_state_ids,
                elem_uniaxial_offsets[e],
                elem_uniaxial_counts[e],
                integration_name,
                elem_num_int_pts[e],
                sec.beam_shear_modulus(elem_type),
                sec.J,
                force_basic_q,
                force_basic_offsets[e],
                force_basic_counts[e],
                scratch,
                k_elem12,
                f_elem12,
            )
        else:
            beam_column3d_fiber_global_tangent_and_internal(
                e,
                x1,
                y1,
                z1,
                x2,
                y2,
                z2,
                u_elem12,
                geom_name,
                elem.has_geom_vecxz,
                elem.geom_vecxz_1,
                elem.geom_vecxz_2,
                elem.geom_vecxz_3,
                element_loads,
                elem_load_offsets,
                elem_load_pool,
                load_scale,
                fiber_section3d_defs[sec_index],
                fiber_section3d_cells,
                uniaxial_defs,
                uniaxial_states,
                elem_uniaxial_state_ids,
                elem_uniaxial_offsets[e],
                elem_uniaxial_counts[e],
                integration_name,
                elem_num_int_pts[e],
                sec.beam_shear_modulus(elem_type),
                sec.J,
                scratch,
                k_elem12,
                f_elem12,
            )
    else:
        abort("beam-column 3d requires ElasticSection3d or FiberSection3d")


@fieldwise_init
struct _Frame3dComputeContext(ImplicitlyCopyable):
    """Borrowed model and state arrays for one threaded sweep.

    The pointers stay valid because the sweep runs synchronously inside the
    function that owns the borrows.
    """

    var indices: UnsafePointer[Int]
    var elements: UnsafePointer[List[ElementInput]]
    var node_x: UnsafePointer[List[Float64]]
    var node_y: UnsafePointer[List[Float64]]
    var node_z: UnsafePointer[List[Float64]]
    var elem_dof_offsets: UnsafePointer[List[Int]]
    var elem_dof_pool: UnsafePointer[List[Int]]
    var elem_node_offsets: UnsafePointer[List[Int]]
    var elem_node_pool: UnsafePointer[List[Int]]
    var elem_type_tags: UnsafePointer[List[Int]]
    var elem_geom_tags: UnsafePointer[List[Int]]
    var elem_section_ids: UnsafePointer[List[Int]]
    var elem_integration_tags: UnsafePointer[List[Int]]
    var elem_num_int_pts: UnsafePointer[List[Int]]
    var element_loads: UnsafePointer[List[ElementLoadInput]]
    var elem_load_offsets: UnsafePointer[List[Int]]
    var elem_load_pool: UnsafePointer[List[Int]]
    var load_scale: Float64
    var sections_by_id: UnsafePointer[List[SectionInput]]
    var u: UnsafePointer[List[Float64]]
    var uniaxial_defs: UnsafePointer[List[UniMaterialDef]]
    var uniaxial_states: UnsafePointer[List[UniMaterialState]]
    var elem_uniaxial_offsets: UnsafePointer[List[Int]]
    var elem_uniaxial_counts: UnsafePointer[List[Int]]
    var elem_uniaxial_state_ids: UnsafePointer[List[Int]]
    var force_basic_offsets: UnsafePointer[List[Int]]
    var force_basic_counts: UnsafePointer[List[Int]]
    var force_basic_q: UnsafePointer[List[Float64]]
    var fiber_section3d_defs: UnsafePointer[List[FiberSection3dDef]]
    var fiber_section3d_cells: UnsafePointer[List[FiberCell]]
    var fiber_section3d_index_by_id: UnsafePointer[List[Int]]


fn _frame3d_compute_batch(
    ctx: _Frame3dComputeContext,
    mut scratch: Frame3dAssemblyScratch,
    start: Int,
    count: Int,
    workers: Int,
):
    """Evaluate elements [start, start + count) into scratch.matrices/forces.

    Each worker takes one contiguous slice and uses only its own kernel scratch
    and element buffers. Elements own disjoint fiber instances and basic-force
    slots, so the shared state lists are written at disjoint positions.
    """
    var worker_scratch = scratch.workers.unsafe_ptr()
    var worker_u = scratch.worker_u.unsafe_ptr()
    var worker_k = scratch.worker_k.unsafe_ptr()
    var worker_f = scratch.worker_f.unsafe_ptr()
    var matrices = scratch.matrices.unsafe_ptr()
    var forces = scratch.forces.unsafe_ptr()

    @parameter
    fn run(worker: Int):
        var end = (worker + 1) * count // workers
        for local_idx in range(worker * count // workers, end):
            _frame3d_element_tangent_and_force(
                ctx.indices[start + local_idx],
                ctx.elements[],
                ctx.node_x[],
                ctx.node_y[],
                ctx.node_z[],
                ctx.elem_dof_offsets[],
                ctx.elem_dof_pool[],
                ctx.elem_node_offsets[],
                ctx.elem_node_pool[],
                ctx.elem_type_tags[],
                ctx.elem_geom_tags[],
                ctx.elem_section_ids[],
                ctx.elem_integration_tags[],
                ctx.elem_num_int_pts[],
                ctx.element_loads[],
                ctx.elem_load_offsets[],
                ctx.elem_load_pool[],
                ctx.load_scale,
                ctx.sections_by_id[],
                ctx.u[],
                ctx.uniaxial_defs[],
                ctx.uniaxial_states[],
                ctx.elem_uniaxial_offsets[],
                ctx.elem_uniaxial_counts[],
                ctx.elem_uniaxial_state_ids[],
                ctx.force_basic_offsets[],
                ctx.force_basic_counts[],
                ctx.force_basic_q[],
                ctx.fiber_section3d_defs[],
                ctx.fiber_section3d_cells[],
                ctx.fiber_section3d_index_by_id[],
                worker_scratch[worker],
                worker_u[worker],
                worker_k[worker],
                worker_f[worker],
            )
            var matrix = matrices + local_idx * FRAME3D_MATRIX_SIZE
            for row in range(FRAME3D_DOFS):
                var source = worker_k[worker][row].unsafe_ptr()
                var destination = matrix + row * FRAME3D_DOFS
                for col in range(FRAME3D_DOFS):
                    destination[col] = source[col]
            var force = forces + local_idx * FRAME3D_DOFS
            var force_source = worker_f[worker].unsafe_ptr()
            for a in range(FRAME3D_DOFS):
                force[a] = force_source[a]

    parallelize[run](workers, workers)


fn _assemble_frame3d_native_soa_indices[
    S: ElementMatrixSink
](
    frame3d_elem_indices: List[Int],
    elements: List[ElementInput],
    node_x: List[Float64],
    node_y: List[Float64],
    node_z: List[Float64],
    elem_dof_offsets: List[Int],
    elem_dof_pool: List[Int],
    elem_free_offsets: List[Int],
    elem_free_pool: List[Int],
    elem_node_offsets: List[Int],
    elem_node_pool: List[Int],
    elem_type_tags: List[Int],
    elem_geom_tags: List[Int],
    elem_section_ids: List[Int],
    elem_integration_tags: List[Int],
    elem_num_int_pts: List[Int],
    element_loads: List[ElementLoadInput],
    elem_load_offsets: List[Int],
    elem_load_pool: List[Int],
    load_scale: Float64,
    sections_by_id: List[SectionInput],
    u: List[Float64],
    uniaxial_defs: List[UniMaterialDef],
    mut uniaxial_states: List[UniMaterialState],
    elem_uniaxial_offsets: List[Int],
    elem_uniaxial_counts: List[Int],
    elem_uniaxial_state_ids: List[Int],
    force_basic_offsets: List[Int],
    force_basic_counts: List[Int],
    mut force_basic_q: List[Float64],
    mut fiber_section3d_defs: List[FiberSection3dDef],
    fiber_section3d_cells: List[FiberCell],
    fiber_section3d_index_by_id: List[Int],
    mut force_beam_column3d_scratch: Frame3dAssemblyScratch,
    mut backend: S,
    mut F_int: List[Float64],
    do_profile: Bool,
    t0: Int,
    mut events: String,
    mut events_need_comma: Bool,
    frame_assemble_fiber: Int,
    mut runtime_metrics: RuntimeProfileMetrics,
) raises:
    var element_count = len(frame3d_elem_indices)
    # Event profiling records one scope per element, which only the serial loop can do.
    var workers = 1
    if not do_profile:
        workers = force_beam_column3d_scratch.worker_count(element_count)

    if workers > 1:
        for e in frame3d_elem_indices:
            if elem_geom_tags[e] == GeomTransfTag.Corotational:
                workers = 1
                break

    if workers > 1:
        force_beam_column3d_scratch.ensure_workers(workers)
        var ctx = _Frame3dComputeContext(
            frame3d_elem_indices.unsafe_ptr(),
            UnsafePointer(to=elements),
            UnsafePointer(to=node_x),
            UnsafePointer(to=node_y),
            UnsafePointer(to=node_z),
            UnsafePointer(to=elem_dof_offsets),
            UnsafePointer(to=elem_dof_pool),
            UnsafePointer(to=elem_node_offsets),
            UnsafePointer(to=elem_node_pool),
            UnsafePointer(to=elem_type_tags),
            UnsafePointer(to=elem_geom_tags),
            UnsafePointer(to=elem_section_ids),
            UnsafePointer(to=elem_integration_tags),
            UnsafePointer(to=elem_num_int_pts),
            UnsafePointer(to=element_loads),
            UnsafePointer(to=elem_load_offsets),
            UnsafePointer(to=elem_load_pool),
            load_scale,
            UnsafePointer(to=sections_by_id),
            UnsafePointer(to=u),
            UnsafePointer(to=uniaxial_defs),
            UnsafePointer(to=uniaxial_states),
            UnsafePointer(to=elem_uniaxial_offsets),
            UnsafePointer(to=elem_uniaxial_counts),
            UnsafePointer(to=elem_uniaxial_state_ids),
            UnsafePointer(to=force_basic_offsets),
            UnsafePointer(to=force_basic_counts),
            UnsafePointer(to=force_basic_q),
            UnsafePointer(to=fiber_section3d_defs),
            UnsafePointer(to=fiber_section3d_cells),
            UnsafePointer(to=fiber_section3d_index_by_id),
        )
        for start in range(0, element_count, FRAME3D_BATCH):
            var count = min(element_count - start, FRAME3D_BATCH)
            var batch_workers = max(1, min(workers, count // FRAME3D_MIN_PER_WORKER))
            var t_batch_start = 0
            if runtime_metrics.enabled:
                t_batch_start = Int(perf_counter_ns())
            _frame3d_compute_batch(ctx, force_beam_column3d_scratch, start, count, batch_workers)
            # Scatter on this thread in element order, so sums match the serial loop.
            for local_idx in range(count):
                var e = frame3d_elem_indices[start + local_idx]
                backend.add_element_matrix_flat(
                    e,
                    elem_free_pool,
                    elem_free_offsets[e],
                    FRAME3D_DOFS,
                    force_beam_column3d_scratch.matrices,
                    local_idx * FRAME3D_MATRIX_SIZE,
                )
                var dof_offset = elem_dof_offsets[e]
                var force_offset = local_idx * FRAME3D_DOFS
                for a in range(FRAME3D_DOFS):
                    F_int[elem_dof_pool[dof_offset + a]] += force_beam_column3d_scratch.forces[
                        force_offset + a
                    ]
            if runtime_metrics.enabled:
                # Wall time of the batch split evenly; per-element time is not observable here.
                var per_element_ns = (Int(perf_counter_ns()) - t_batch_start) // count
                for local_idx in range(count):
                    _profile_metrics_note_element_timing(
                        runtime_metrics,
                        elem_type_tags[frame3d_elem_indices[start + local_idx]],
                        per_element_ns,
                    )
        return

    var u_elem12: List[Float64] = []
    u_elem12.resize(FRAME3D_DOFS, 0.0)
    var k_elem12: List[List[Float64]] = []
    for _ in range(FRAME3D_DOFS):
        var row: List[Float64] = []
        row.resize(FRAME3D_DOFS, 0.0)
        k_elem12.append(row^)
    var f_elem12: List[Float64] = []
    f_elem12.resize(FRAME3D_DOFS, 0.0)
    for idx in range(element_count):
        var e = frame3d_elem_indices[idx]
        var elem_type = elem_type_tags[e]
        var t_elem_start = 0
        if runtime_metrics.enabled:
            t_elem_start = Int(perf_counter_ns())
        var profile_fiber = (
            do_profile
            and elem_type != ElementTypeTag.ElasticBeamColumn3d
            and sections_by_id[elem_section_ids[e]].type == "FiberSection3d"
        )
        if profile_fiber:
            _profile_scope_open(do_profile, events, events_need_comma, frame_assemble_fiber, t0)
            _profile_scope_open(
                do_profile,
                events,
                events_need_comma,
                PROFILE_FRAME_ASSEMBLE_FIBER_SECTION_RESPONSE,
                t0,
            )
        _frame3d_element_tangent_and_force(
            e,
            elements,
            node_x,
            node_y,
            node_z,
            elem_dof_offsets,
            elem_dof_pool,
            elem_node_offsets,
            elem_node_pool,
            elem_type_tags,
            elem_geom_tags,
            elem_section_ids,
            elem_integration_tags,
            elem_num_int_pts,
            element_loads,
            elem_load_offsets,
            elem_load_pool,
            load_scale,
            sections_by_id,
            u,
            uniaxial_defs,
            uniaxial_states,
            elem_uniaxial_offsets,
            elem_uniaxial_counts,
            elem_uniaxial_state_ids,
            force_basic_offsets,
            force_basic_counts,
            force_basic_q,
            fiber_section3d_defs,
            fiber_section3d_cells,
            fiber_section3d_index_by_id,
            force_beam_column3d_scratch.main,
            u_elem12,
            k_elem12,
            f_elem12,
        )
        if profile_fiber:
            _profile_scope_close(
                do_profile,
                events,
                events_need_comma,
                PROFILE_FRAME_ASSEMBLE_FIBER_SECTION_RESPONSE,
                t0,
            )
            _profile_scope_close(do_profile, events, events_need_comma, frame_assemble_fiber, t0)
        backend.add_element_matrix(e, elem_free_pool, elem_free_offsets[e], FRAME3D_DOFS, k_elem12)
        var dof_offset = elem_dof_offsets[e]
        for a in range(FRAME3D_DOFS):
            F_int[elem_dof_pool[dof_offset + a]] += f_elem12[a]
        if runtime_metrics.enabled:
            _profile_metrics_note_element_timing(
                runtime_metrics,
                elem_type,
                Int(perf_counter_ns()) - t_elem_start,
            )


fn _assemble_frame3d_soa_indices(
    frame3d_elem_indices: List[Int],
    elements: List[ElementInput],
    node_x: List[Float64],
    node_y: List[Float64],
    node_z: List[Float64],
    elem_dof_offsets: List[Int],
    elem_dof_pool: List[Int],
    elem_node_offsets: List[Int],
    elem_node_pool: List[Int],
    elem_type_tags: List[Int],
    elem_geom_tags: List[Int],
    elem_section_ids: List[Int],
    elem_integration_tags: List[Int],
    elem_num_int_pts: List[Int],
    element_loads: List[ElementLoadInput],
    elem_load_offsets: List[Int],
    elem_load_pool: List[Int],
    load_scale: Float64,
    sections_by_id: List[SectionInput],
    u: List[Float64],
    uniaxial_defs: List[UniMaterialDef],
    mut uniaxial_states: List[UniMaterialState],
    elem_uniaxial_offsets: List[Int],
    elem_uniaxial_counts: List[Int],
    elem_uniaxial_state_ids: List[Int],
    force_basic_offsets: List[Int],
    force_basic_counts: List[Int],
    mut force_basic_q: List[Float64],
    mut fiber_section3d_defs: List[FiberSection3dDef],
    fiber_section3d_cells: List[FiberCell],
    fiber_section3d_index_by_id: List[Int],
    mut force_beam_column3d_scratch: Frame3dAssemblyScratch,
    mut dof_map12: List[Int],
    mut K: List[List[Float64]],
    mut F_int: List[Float64],
    do_profile: Bool,
    t0: Int,
    mut events: String,
    mut events_need_comma: Bool,
    frame_assemble_fiber: Int,
    mut runtime_metrics: RuntimeProfileMetrics,
) raises:
    var u_elem12: List[Float64] = []
    u_elem12.resize(12, 0.0)
    var k_elem12: List[List[Float64]] = []
    for _ in range(12):
        var row: List[Float64] = []
        row.resize(12, 0.0)
        k_elem12.append(row^)
    var f_elem12: List[Float64] = []
    f_elem12.resize(12, 0.0)
    for idx in range(len(frame3d_elem_indices)):
        var e = frame3d_elem_indices[idx]
        ref elem = elements[e]
        var t_elem_start = 0
        if runtime_metrics.enabled:
            t_elem_start = Int(perf_counter_ns())
        _profile_scope_open(
            do_profile,
            events,
            events_need_comma,
            PROFILE_FRAME_ASSEMBLE_FIBER_GEOMETRY,
            t0,
        )
        var node_offset = elem_node_offsets[e]
        var i1 = elem_node_pool[node_offset]
        var i2 = elem_node_pool[node_offset + 1]
        var x1 = node_x[i1]
        var y1 = node_y[i1]
        var z1 = node_z[i1]
        var x2 = node_x[i2]
        var y2 = node_y[i2]
        var z2 = node_z[i2]
        ref sec = sections_by_id[elem_section_ids[e]]
        var dof_offset = elem_dof_offsets[e]
        for i in range(12):
            dof_map12[i] = elem_dof_pool[dof_offset + i]
            u_elem12[i] = u[dof_map12[i]]
        var elem_type = elem_type_tags[e]
        var geom_name = _geom_transf_name_from_tag(elem_geom_tags[e])
        _profile_scope_close(
            do_profile,
            events,
            events_need_comma,
            PROFILE_FRAME_ASSEMBLE_FIBER_GEOMETRY,
            t0,
        )
        if elem_type == ElementTypeTag.ElasticBeamColumn3d:
            if sec.type != "ElasticSection3d":
                abort("elasticBeamColumn3d requires ElasticSection3d")
            force_beam_column3d_global_tangent_and_internal(
                e,
                x1,
                y1,
                z1,
                x2,
                y2,
                z2,
                u_elem12,
                geom_name,
                elem.has_geom_vecxz,
                elem.geom_vecxz_1,
                elem.geom_vecxz_2,
                elem.geom_vecxz_3,
                element_loads,
                elem_load_offsets,
                elem_load_pool,
                load_scale,
                sec.E,
                sec.A,
                sec.Iy,
                sec.Iz,
                sec.beam_shear_modulus(elem_type),
                sec.J,
                force_beam_column3d_scratch.main,
                k_elem12,
                f_elem12,
            )
        else:
            if sec.type == "ElasticSection3d":
                if elem_type == ElementTypeTag.ForceBeamColumn3d:
                    force_beam_column3d_global_tangent_and_internal(
                        e,
                        x1,
                        y1,
                        z1,
                        x2,
                        y2,
                        z2,
                        u_elem12,
                        geom_name,
                        elem.has_geom_vecxz,
                        elem.geom_vecxz_1,
                        elem.geom_vecxz_2,
                        elem.geom_vecxz_3,
                        element_loads,
                        elem_load_offsets,
                        elem_load_pool,
                        load_scale,
                        sec.E,
                        sec.A,
                        sec.Iy,
                        sec.Iz,
                        sec.beam_shear_modulus(elem_type),
                        sec.J,
                        force_beam_column3d_scratch.main,
                        k_elem12,
                        f_elem12,
                        sec.shear_rigidity_y,
                        sec.shear_rigidity_z,
                        elem.integration,
                        elem.num_int_pts,
                        elem.cbdi,
                    )
                else:
                    disp_beam_column3d_global_tangent_and_internal(
                        e,
                        x1,
                        y1,
                        z1,
                        x2,
                        y2,
                        z2,
                        u_elem12,
                        geom_name,
                        elem.has_geom_vecxz,
                        elem.geom_vecxz_1,
                        elem.geom_vecxz_2,
                        elem.geom_vecxz_3,
                        element_loads,
                        elem_load_offsets,
                        elem_load_pool,
                        load_scale,
                        sec.E,
                        sec.A,
                        sec.Iy,
                        sec.Iz,
                        sec.beam_shear_modulus(elem_type),
                        sec.J,
                        force_beam_column3d_scratch.main,
                        k_elem12,
                        f_elem12,
                    )
            elif sec.type == "FiberSection3d":
                var sec_index = fiber_section3d_index_by_id[elem_section_ids[e]]
                if sec_index < 0 or sec_index >= len(fiber_section3d_defs):
                    abort("fiber section not found")
                var integration_name = _beam_integration_name_from_tag(elem_integration_tags[e])
                _profile_scope_open(
                    do_profile,
                    events,
                    events_need_comma,
                    frame_assemble_fiber,
                    t0,
                )
                _profile_scope_open(
                    do_profile,
                    events,
                    events_need_comma,
                    PROFILE_FRAME_ASSEMBLE_FIBER_SECTION_RESPONSE,
                    t0,
                )
                if elem_type == ElementTypeTag.ForceBeamColumn3d:
                    force_beam_column3d_fiber_global_tangent_and_internal(
                        e,
                        x1,
                        y1,
                        z1,
                        x2,
                        y2,
                        z2,
                        u_elem12,
                        geom_name,
                        elem.has_geom_vecxz,
                        elem.geom_vecxz_1,
                        elem.geom_vecxz_2,
                        elem.geom_vecxz_3,
                        element_loads,
                        elem_load_offsets,
                        elem_load_pool,
                        load_scale,
                        fiber_section3d_defs[sec_index],
                        fiber_section3d_cells,
                        uniaxial_defs,
                        uniaxial_states,
                        elem_uniaxial_state_ids,
                        elem_uniaxial_offsets[e],
                        elem_uniaxial_counts[e],
                        integration_name,
                        elem_num_int_pts[e],
                        sec.beam_shear_modulus(elem_type),
                        sec.J,
                        force_basic_q,
                        force_basic_offsets[e],
                        force_basic_counts[e],
                        force_beam_column3d_scratch.main,
                        k_elem12,
                        f_elem12,
                    )
                else:
                    beam_column3d_fiber_global_tangent_and_internal(
                        e,
                        x1,
                        y1,
                        z1,
                        x2,
                        y2,
                        z2,
                        u_elem12,
                        geom_name,
                        elem.has_geom_vecxz,
                        elem.geom_vecxz_1,
                        elem.geom_vecxz_2,
                        elem.geom_vecxz_3,
                        element_loads,
                        elem_load_offsets,
                        elem_load_pool,
                        load_scale,
                        fiber_section3d_defs[sec_index],
                        fiber_section3d_cells,
                        uniaxial_defs,
                        uniaxial_states,
                        elem_uniaxial_state_ids,
                        elem_uniaxial_offsets[e],
                        elem_uniaxial_counts[e],
                        integration_name,
                        elem_num_int_pts[e],
                        sec.beam_shear_modulus(elem_type),
                        sec.J,
                        force_beam_column3d_scratch.main,
                        k_elem12,
                        f_elem12,
                    )
                _profile_scope_close(
                    do_profile,
                    events,
                    events_need_comma,
                    PROFILE_FRAME_ASSEMBLE_FIBER_SECTION_RESPONSE,
                    t0,
                )
                _profile_scope_open(
                    do_profile,
                    events,
                    events_need_comma,
                    PROFILE_FRAME_ASSEMBLE_FIBER_MATRIX_SCATTER,
                    t0,
                )
            else:
                abort("3d beam requires ElasticSection3d or FiberSection3d")
        for a in range(12):
            var Aidx = dof_map12[a]
            _scatter_add_row[12](K, Aidx, k_elem12[a], dof_map12)
        if sec.type == "FiberSection3d":
            _profile_scope_close(
                do_profile,
                events,
                events_need_comma,
                PROFILE_FRAME_ASSEMBLE_FIBER_MATRIX_SCATTER,
                t0,
            )
            _profile_scope_open(
                do_profile,
                events,
                events_need_comma,
                PROFILE_FRAME_ASSEMBLE_FIBER_INTERNAL_FORCE,
                t0,
            )
        for a in range(12):
            var Aidx = dof_map12[a]
            F_int[Aidx] += f_elem12[a]
        if sec.type == "FiberSection3d":
            _profile_scope_close(
                do_profile,
                events,
                events_need_comma,
                PROFILE_FRAME_ASSEMBLE_FIBER_INTERNAL_FORCE,
                t0,
            )
            _profile_scope_close(
                do_profile,
                events,
                events_need_comma,
                frame_assemble_fiber,
                t0,
            )
        if runtime_metrics.enabled:
            _profile_metrics_note_element_timing(
                runtime_metrics,
                elem_type,
                Int(perf_counter_ns()) - t_elem_start,
            )


fn _assemble_frame3d_element(
    e: Int,
    elem: ElementInput,
    nodes: List[NodeInput],
    elem_dof_offsets: List[Int],
    elem_dof_pool: List[Int],
    element_loads: List[ElementLoadInput],
    elem_load_offsets: List[Int],
    elem_load_pool: List[Int],
    load_scale: Float64,
    sections_by_id: List[SectionInput],
    u: List[Float64],
    uniaxial_defs: List[UniMaterialDef],
    mut uniaxial_states: List[UniMaterialState],
    elem_uniaxial_offsets: List[Int],
    elem_uniaxial_counts: List[Int],
    elem_uniaxial_state_ids: List[Int],
    force_basic_offsets: List[Int],
    force_basic_counts: List[Int],
    mut force_basic_q: List[Float64],
    mut fiber_section3d_defs: List[FiberSection3dDef],
    fiber_section3d_cells: List[FiberCell],
    fiber_section3d_index_by_id: List[Int],
    mut force_beam_column3d_scratch: Frame3dAssemblyScratch,
    mut dof_map12: List[Int],
    mut K: List[List[Float64]],
    mut F_int: List[Float64],
) raises:
    var elem_type = elem.type_tag
    if (
        elem_type != ElementTypeTag.ElasticBeamColumn3d
        and elem_type != ElementTypeTag.ForceBeamColumn3d
        and elem_type != ElementTypeTag.DispBeamColumn3d
    ):
        return

    ref node1 = nodes[elem.node_index_1]
    ref node2 = nodes[elem.node_index_2]
    ref sec = sections_by_id[elem.section]
    var dof_offset = elem_dof_offsets[e]
    var u_elem12: List[Float64] = []
    u_elem12.resize(12, 0.0)
    for i in range(12):
        dof_map12[i] = elem_dof_pool[dof_offset + i]
        u_elem12[i] = u[dof_map12[i]]
    var k_elem12: List[List[Float64]] = []
    var f_elem12: List[Float64] = []
    if elem_type == ElementTypeTag.ElasticBeamColumn3d:
        if sec.type != "ElasticSection3d":
            abort("elasticBeamColumn3d requires ElasticSection3d")
        force_beam_column3d_global_tangent_and_internal(
            e,
            node1.x,
            node1.y,
            node1.z,
            node2.x,
            node2.y,
            node2.z,
            u_elem12,
            elem.geom_transf,
            elem.has_geom_vecxz,
            elem.geom_vecxz_1,
            elem.geom_vecxz_2,
            elem.geom_vecxz_3,
            element_loads,
            elem_load_offsets,
            elem_load_pool,
            load_scale,
            sec.E,
            sec.A,
            sec.Iy,
            sec.Iz,
            sec.beam_shear_modulus(elem_type),
            sec.J,
            force_beam_column3d_scratch.main,
            k_elem12,
            f_elem12,
        )
    else:
        if sec.type == "ElasticSection3d":
            if elem_type == ElementTypeTag.ForceBeamColumn3d:
                force_beam_column3d_global_tangent_and_internal(
                    e,
                    node1.x,
                    node1.y,
                    node1.z,
                    node2.x,
                    node2.y,
                    node2.z,
                    u_elem12,
                    elem.geom_transf,
                    elem.has_geom_vecxz,
                    elem.geom_vecxz_1,
                    elem.geom_vecxz_2,
                    elem.geom_vecxz_3,
                    element_loads,
                    elem_load_offsets,
                    elem_load_pool,
                    load_scale,
                    sec.E,
                    sec.A,
                    sec.Iy,
                    sec.Iz,
                    sec.beam_shear_modulus(elem_type),
                    sec.J,
                    force_beam_column3d_scratch.main,
                    k_elem12,
                    f_elem12,
                    sec.shear_rigidity_y,
                    sec.shear_rigidity_z,
                    elem.integration,
                    elem.num_int_pts,
                    elem.cbdi,
                )
            else:
                disp_beam_column3d_global_tangent_and_internal(
                    e,
                    node1.x,
                    node1.y,
                    node1.z,
                    node2.x,
                    node2.y,
                    node2.z,
                    u_elem12,
                    elem.geom_transf,
                    elem.has_geom_vecxz,
                    elem.geom_vecxz_1,
                    elem.geom_vecxz_2,
                    elem.geom_vecxz_3,
                    element_loads,
                    elem_load_offsets,
                    elem_load_pool,
                    load_scale,
                    sec.E,
                    sec.A,
                    sec.Iy,
                    sec.Iz,
                    sec.beam_shear_modulus(elem_type),
                    sec.J,
                    force_beam_column3d_scratch.main,
                    k_elem12,
                    f_elem12,
                )
        elif sec.type == "FiberSection3d":
            var sec_index = fiber_section3d_index_by_id[elem.section]
            if sec_index < 0 or sec_index >= len(fiber_section3d_defs):
                abort(elem.type + " fiber section not found")
            if elem_type == ElementTypeTag.ForceBeamColumn3d:
                force_beam_column3d_fiber_global_tangent_and_internal(
                    e,
                    node1.x,
                    node1.y,
                    node1.z,
                    node2.x,
                    node2.y,
                    node2.z,
                    u_elem12,
                    elem.geom_transf,
                    elem.has_geom_vecxz,
                    elem.geom_vecxz_1,
                    elem.geom_vecxz_2,
                    elem.geom_vecxz_3,
                    element_loads,
                    elem_load_offsets,
                    elem_load_pool,
                    load_scale,
                    fiber_section3d_defs[sec_index],
                    fiber_section3d_cells,
                    uniaxial_defs,
                    uniaxial_states,
                    elem_uniaxial_state_ids,
                    elem_uniaxial_offsets[e],
                    elem_uniaxial_counts[e],
                    elem.integration,
                    elem.num_int_pts,
                    sec.beam_shear_modulus(elem_type),
                    sec.J,
                    force_basic_q,
                    force_basic_offsets[e],
                    force_basic_counts[e],
                    force_beam_column3d_scratch.main,
                    k_elem12,
                    f_elem12,
                )
            else:
                beam_column3d_fiber_global_tangent_and_internal(
                    e,
                    node1.x,
                    node1.y,
                    node1.z,
                    node2.x,
                    node2.y,
                    node2.z,
                    u_elem12,
                    elem.geom_transf,
                    elem.has_geom_vecxz,
                    elem.geom_vecxz_1,
                    elem.geom_vecxz_2,
                    elem.geom_vecxz_3,
                    element_loads,
                    elem_load_offsets,
                    elem_load_pool,
                    load_scale,
                    fiber_section3d_defs[sec_index],
                    fiber_section3d_cells,
                    uniaxial_defs,
                    uniaxial_states,
                    elem_uniaxial_state_ids,
                    elem_uniaxial_offsets[e],
                    elem_uniaxial_counts[e],
                    elem.integration,
                    elem.num_int_pts,
                    sec.beam_shear_modulus(elem_type),
                    sec.J,
                    force_beam_column3d_scratch.main,
                    k_elem12,
                    f_elem12,
                )
        else:
            abort(elem.type + " requires ElasticSection3d or FiberSection3d")
    for a in range(12):
        var Aidx = dof_map12[a]
        _scatter_add_row[12](K, Aidx, k_elem12[a], dof_map12)
        F_int[Aidx] += f_elem12[a]
