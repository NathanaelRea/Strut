from elements.force_beam_column2d import force_beam_column2d_initial_global_stiffness
from elements.force_beam_column2d import elastic_beam_column2d_global_tangent_and_internal
from collections import List
from os import abort
from time import perf_counter_ns

from elements import (
    ForceBeamColumn2dScratch,
    beam2d_element_load_global,
    beam2d_element_load_global_array,
    beam2d_global_stiffness_flat,
    beam2d_pdelta_global_stiffness_flat,
    beam2d_corotational_global_tangent_and_internal,
    beam2d_pdelta_global_stiffness,
    beam2d_pdelta_global_internal_force,
    beam_global_stiffness,
    disp_beam_column2d_global_tangent_and_internal,
    force_beam_column2d_global_tangent_and_internal,
)
from materials import UniMaterialDef, UniMaterialState
from solver.assembly.frame2d_scratch import (
    FRAME2D_BATCH,
    FRAME2D_DOFS,
    FRAME2D_MINIMUM_WORK,
    Frame2dAssemblyScratch,
    Frame2dWorker,
)
from solver.assembly.parallel_elements import ParallelElementWork
from solver.assembly.stiffness_internal_shared import (
    _beam_integration_name_from_tag,
    _dot_row_simd,
    _elem_dof,
    _geom_transf_name_from_tag,
    _profile_scope_close,
    _profile_scope_open,
    _scatter_add_and_dot_row_simd,
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
from sections import FiberCell, FiberSection2dDef
from tag_types import ElementTypeTag, GeomTransfTag


@fieldwise_init
struct _Frame2dComputeContext(ImplicitlyCopyable):
    var frame2d_elem_indices: UnsafePointer[List[Int]]
    var node_x: UnsafePointer[List[Float64]]
    var node_y: UnsafePointer[List[Float64]]
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
    var fiber_section_defs: UnsafePointer[List[FiberSection2dDef]]
    var fiber_section_cells: UnsafePointer[List[FiberCell]]
    var fiber_section_index_by_id: UnsafePointer[List[Int]]
    var workers: UnsafePointer[Frame2dWorker]
    var tangent: Bool


fn _evaluate_frame2d_fiber(
    ctx: _Frame2dComputeContext,
    e: Int,
    elem_type: Int,
    x1: Float64,
    y1: Float64,
    x2: Float64,
    y2: Float64,
    u_elem6: List[Float64],
    mut scratch: ForceBeamColumn2dScratch,
    mut k_global: List[List[Float64]],
    mut f_global: List[Float64],
):
    var sec_index = ctx.fiber_section_index_by_id[][ctx.elem_section_ids[][e]]
    ref sec = ctx.sections_by_id[][ctx.elem_section_ids[][e]]
    ref sec_def = ctx.fiber_section_defs[][sec_index]
    var elem_offset = ctx.elem_uniaxial_offsets[][e]
    var elem_state_count = ctx.elem_uniaxial_counts[][e]
    var geom_name = _geom_transf_name_from_tag(ctx.elem_geom_tags[][e])
    var integration_name = _beam_integration_name_from_tag(ctx.elem_integration_tags[][e])
    if elem_type == ElementTypeTag.ForceBeamColumn2d:
        force_beam_column2d_global_tangent_and_internal(
            e,
            x1,
            y1,
            x2,
            y2,
            u_elem6,
            ctx.element_loads[],
            ctx.elem_load_offsets[],
            ctx.elem_load_pool[],
            ctx.load_scale,
            sec_def,
            ctx.fiber_section_cells[],
            ctx.uniaxial_defs[],
            ctx.uniaxial_states[],
            ctx.elem_uniaxial_state_ids[],
            elem_offset,
            elem_state_count,
            geom_name,
            integration_name,
            ctx.elem_num_int_pts[][e],
            ctx.force_basic_q[],
            ctx.force_basic_offsets[][e],
            ctx.force_basic_counts[][e],
            scratch,
            k_global,
            f_global,
            sec.hinge_radau,
        )
    else:
        disp_beam_column2d_global_tangent_and_internal(
            e,
            x1,
            y1,
            x2,
            y2,
            u_elem6,
            ctx.element_loads[],
            ctx.elem_load_offsets[],
            ctx.elem_load_pool[],
            ctx.load_scale,
            sec_def,
            ctx.fiber_section_cells[],
            ctx.uniaxial_defs[],
            ctx.uniaxial_states[],
            ctx.elem_uniaxial_state_ids[],
            elem_offset,
            elem_state_count,
            geom_name,
            integration_name,
            ctx.elem_num_int_pts[][e],
            k_global,
            f_global,
        )


fn _compute_frame2d_fiber(
    ctx: _Frame2dComputeContext,
    idx: Int,
    worker: Int,
    matrix: UnsafePointer[Float64],
    force: UnsafePointer[Float64],
    _scratch: UnsafePointer[Float64],
):
    var e = ctx.frame2d_elem_indices[][idx]
    if ctx.elem_uniaxial_counts[][e] <= 0:
        return
    var node_offset = ctx.elem_node_offsets[][e]
    var i1 = ctx.elem_node_pool[][node_offset]
    var i2 = ctx.elem_node_pool[][node_offset + 1]
    ref local = ctx.workers[worker]
    var dof_offset = ctx.elem_dof_offsets[][e]
    for a in range(FRAME2D_DOFS):
        local.u[a] = ctx.u[][ctx.elem_dof_pool[][dof_offset + a]]
    _evaluate_frame2d_fiber(
        ctx,
        e,
        ctx.elem_type_tags[][e],
        ctx.node_x[][i1],
        ctx.node_y[][i1],
        ctx.node_x[][i2],
        ctx.node_y[][i2],
        local.u,
        local.kernel,
        local.matrix,
        local.force,
    )
    for a in range(FRAME2D_DOFS):
        force[a] = local.force[a]
    if ctx.tangent:
        for a in range(FRAME2D_DOFS):
            var row = local.matrix[a].unsafe_ptr()
            for b in range(FRAME2D_DOFS):
                matrix[a * FRAME2D_DOFS + b] = row[b]


fn _assemble_frame2d_native_soa_indices[
    S: ElementMatrixSink
](
    frame2d_elem_indices: List[Int],
    node_x: List[Float64],
    node_y: List[Float64],
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
    mut fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    mut force_beam_column2d_scratch: ForceBeamColumn2dScratch,
    mut sweep: Frame2dAssemblyScratch,
    mut dof_map6: List[Int],
    mut u_elem6: List[Float64],
    mut k_flat: List[Float64],
    mut backend: S,
    mut F_int: List[Float64],
    do_profile: Bool,
    t0: Int,
    mut events: String,
    mut events_need_comma: Bool,
    frame_assemble_fiber: Int,
    mut runtime_metrics: RuntimeProfileMetrics,
    initial_tangent: Bool = False,
) raises:
    var count = len(frame2d_elem_indices)
    if count == 0:
        return
    var workers = 1
    if count > 1 and len(fiber_section_defs) > 0 and not initial_tangent and not do_profile:
        if len(sweep.work.plan.cumulative) != count + 1:
            var capacity = min(count, FRAME2D_BATCH)
            sweep.work = ParallelElementWork(
                capacity, FRAME2D_DOFS * FRAME2D_DOFS, FRAME2D_DOFS, 0, FRAME2D_MINIMUM_WORK
            )
            sweep.work.plan.cumulative.reserve(count + 1)
            for e in frame2d_elem_indices:
                sweep.work.plan.append(elem_uniaxial_counts[e])
        workers = sweep.work.plan.worker_count(0, count)
        if workers > 1:
            sweep.ensure_workers(workers)
    var tangent = backend.accepts_element_matrices()
    var ctx = _Frame2dComputeContext(
        UnsafePointer(to=frame2d_elem_indices),
        UnsafePointer(to=node_x),
        UnsafePointer(to=node_y),
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
        UnsafePointer(to=fiber_section_defs),
        UnsafePointer(to=fiber_section_cells),
        UnsafePointer(to=fiber_section_index_by_id),
        sweep.workers.unsafe_ptr(),
        tangent,
    )
    var batch_start = 0
    var batch_end = 0
    var per_element_ns = 0
    for idx in range(count):
        var e = frame2d_elem_indices[idx]
        if workers > 1:
            if idx == batch_end:
                batch_start = idx
                batch_end = min(count, idx + sweep.work.capacity)
                var start_ns = 0
                if runtime_metrics.enabled:
                    start_ns = Int(perf_counter_ns())
                sweep.work.compute[_Frame2dComputeContext, _compute_frame2d_fiber](
                    ctx, batch_start, batch_end - batch_start
                )
                if runtime_metrics.enabled:
                    per_element_ns = (Int(perf_counter_ns()) - start_ns) // (
                        batch_end - batch_start
                    )
            if elem_uniaxial_counts[e] > 0:
                var slot = idx - batch_start
                if tangent:
                    backend.add_element_matrix_flat(
                        e,
                        elem_free_pool,
                        elem_free_offsets[e],
                        FRAME2D_DOFS,
                        sweep.work.matrices,
                        slot * FRAME2D_DOFS * FRAME2D_DOFS,
                    )
                for a in range(FRAME2D_DOFS):
                    F_int[elem_dof_pool[elem_dof_offsets[e] + a]] += sweep.work.forces[
                        slot * FRAME2D_DOFS + a
                    ]
                if runtime_metrics.enabled:
                    _profile_metrics_note_element_timing(
                        runtime_metrics, elem_type_tags[e], per_element_ns
                    )
                continue
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
        var x2 = node_x[i2]
        var y2 = node_y[i2]
        var sec = sections_by_id[elem_section_ids[e]]
        var dof_offset = elem_dof_offsets[e]
        var free_offset = elem_free_offsets[e]
        for i in range(6):
            dof_map6[i] = elem_dof_pool[dof_offset + i]
        var elem_type = elem_type_tags[e]
        _profile_scope_close(
            do_profile,
            events,
            events_need_comma,
            PROFILE_FRAME_ASSEMBLE_FIBER_GEOMETRY,
            t0,
        )
        if initial_tangent:
            var initial: List[List[Float64]] = []
            if elem_type == ElementTypeTag.ElasticBeamColumn2d or sec.type == "ElasticSection2d":
                initial = beam_global_stiffness(sec.E, sec.A, sec.I, x1, y1, x2, y2)
                backend.add_element_matrix(e, elem_free_pool, free_offset, 6, initial)
                continue
            if elem_type == ElementTypeTag.ForceBeamColumn2d and not sec.hinge_radau.is_defined():
                var section_index = fiber_section_index_by_id[elem_section_ids[e]]
                force_beam_column2d_initial_global_stiffness(
                    x1,
                    y1,
                    x2,
                    y2,
                    _beam_integration_name_from_tag(elem_integration_tags[e]),
                    elem_num_int_pts[e],
                    fiber_section_defs[section_index],
                    fiber_section_cells,
                    uniaxial_defs,
                    force_beam_column2d_scratch,
                    initial,
                )
                backend.add_element_matrix(e, elem_free_pool, free_offset, 6, initial)
                continue
        if elem_type == ElementTypeTag.ElasticBeamColumn2d:
            var f_load_global = beam2d_element_load_global_array(
                element_loads,
                elem_load_offsets,
                elem_load_pool,
                e,
                load_scale,
                x1,
                y1,
                x2,
                y2,
            )
            if elem_geom_tags[e] == GeomTransfTag.Corotational:
                var k_global: List[List[Float64]] = []
                var f_elem: List[Float64] = []
                for i in range(6):
                    u_elem6[i] = u[dof_map6[i]]
                beam2d_corotational_global_tangent_and_internal(
                    sec.E, sec.A, sec.I, x1, y1, x2, y2, u_elem6, k_global, f_elem
                )
                backend.add_element_matrix(e, elem_free_pool, free_offset, 6, k_global)
                for a in range(6):
                    F_int[dof_map6[a]] += f_elem[a] - f_load_global[a]
            else:
                if elem_geom_tags[e] == GeomTransfTag.Linear:
                    beam2d_global_stiffness_flat(sec.E, sec.A, sec.I, x1, y1, x2, y2, k_flat)
                elif elem_geom_tags[e] == GeomTransfTag.PDelta:
                    for i in range(6):
                        u_elem6[i] = u[dof_map6[i]]
                    beam2d_pdelta_global_stiffness_flat(
                        sec.E, sec.A, sec.I, x1, y1, x2, y2, u_elem6, k_flat
                    )
                else:
                    abort("unsupported geomTransf tag")
                backend.add_element_matrix_flat(e, elem_free_pool, free_offset, 6, k_flat)
                if elem_geom_tags[e] == GeomTransfTag.PDelta:
                    # Basic forces first, as ElasticBeam2d does: K*u cancels large
                    # terms on nearly released, axially rigid members.
                    var f_elem = beam2d_pdelta_global_internal_force(
                        sec.E, sec.A, sec.I, x1, y1, x2, y2, u_elem6
                    )
                    for a in range(6):
                        F_int[dof_map6[a]] += f_elem[a] - f_load_global[a]
                else:
                    for a in range(6):
                        F_int[dof_map6[a]] += (
                            _dot_row_simd(k_flat, dof_map6, u, 6, a * 6) - f_load_global[a]
                        )
            if runtime_metrics.enabled:
                _profile_metrics_note_element_timing(
                    runtime_metrics,
                    elem_type,
                    Int(perf_counter_ns()) - t_elem_start,
                )
            continue

        for i in range(6):
            u_elem6[i] = u[dof_map6[i]]
        var k_global: List[List[Float64]] = []
        var f_global: List[Float64] = []
        if sec.type == "ElasticSection2d":
            if (
                sec.cbdi.enabled()
                or sec.shear_rigidity_y != 0.0
                or sec.hinge_radau.is_defined()
                or elem_geom_tags[e] == GeomTransfTag.Corotational
            ):
                var q = elastic_beam_column2d_global_tangent_and_internal(
                    e,
                    x1,
                    y1,
                    x2,
                    y2,
                    u_elem6,
                    _geom_transf_name_from_tag(elem_geom_tags[e]),
                    _beam_integration_name_from_tag(elem_integration_tags[e]),
                    elem_num_int_pts[e],
                    sec,
                    elem_type == ElementTypeTag.ForceBeamColumn2d,
                    element_loads,
                    elem_load_offsets,
                    elem_load_pool,
                    load_scale,
                    force_beam_column2d_scratch,
                    k_global,
                    f_global,
                )
                var q_offset = force_basic_offsets[e]
                if elem_type == ElementTypeTag.ForceBeamColumn2d and q_offset >= 0:
                    force_basic_q[q_offset] = q[0]
                    force_basic_q[q_offset + 1] = q[1]
                    force_basic_q[q_offset + 2] = q[2]
                backend.add_element_matrix(e, elem_free_pool, free_offset, 6, k_global)
                for a in range(6):
                    F_int[dof_map6[a]] += f_global[a]
                continue
            var f_load_global = beam2d_element_load_global(
                element_loads,
                elem_load_offsets,
                elem_load_pool,
                e,
                load_scale,
                x1,
                y1,
                x2,
                y2,
            )
            if elem_geom_tags[e] == GeomTransfTag.Linear:
                k_global = beam_global_stiffness(sec.E, sec.A, sec.I, x1, y1, x2, y2)
            elif elem_geom_tags[e] == GeomTransfTag.PDelta:
                k_global = beam2d_pdelta_global_stiffness(
                    sec.E, sec.A, sec.I, x1, y1, x2, y2, u_elem6
                )
            else:
                abort("forceBeamColumn2d/dispBeamColumn2d supports geomTransf Linear or PDelta")
            f_global.resize(6, 0.0)
            for a in range(6):
                f_global[a] = _dot_row_simd(k_global[a], dof_map6, u, 6)
            backend.add_element_matrix(e, elem_free_pool, free_offset, 6, k_global)
            for a in range(6):
                F_int[dof_map6[a]] += f_global[a] - f_load_global[a]
            if runtime_metrics.enabled:
                _profile_metrics_note_element_timing(
                    runtime_metrics,
                    elem_type,
                    Int(perf_counter_ns()) - t_elem_start,
                )
            continue

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
        _evaluate_frame2d_fiber(
            ctx,
            e,
            elem_type,
            x1,
            y1,
            x2,
            y2,
            u_elem6,
            force_beam_column2d_scratch,
            k_global,
            f_global,
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
        backend.add_element_matrix(e, elem_free_pool, free_offset, 6, k_global)
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
        for a in range(6):
            F_int[dof_map6[a]] += f_global[a]
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


fn _assemble_frame2d_soa_indices(
    frame2d_elem_indices: List[Int],
    node_x: List[Float64],
    node_y: List[Float64],
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
    mut fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    mut force_beam_column2d_scratch: ForceBeamColumn2dScratch,
    mut dof_map6: List[Int],
    mut u_elem6: List[Float64],
    mut K: List[List[Float64]],
    mut F_int: List[Float64],
    do_profile: Bool,
    t0: Int,
    mut events: String,
    mut events_need_comma: Bool,
    frame_assemble_fiber: Int,
    mut runtime_metrics: RuntimeProfileMetrics,
) raises:
    for idx in range(len(frame2d_elem_indices)):
        var e = frame2d_elem_indices[idx]
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
        var x2 = node_x[i2]
        var y2 = node_y[i2]
        var sec = sections_by_id[elem_section_ids[e]]
        var dof_offset = elem_dof_offsets[e]
        for i in range(6):
            dof_map6[i] = elem_dof_pool[dof_offset + i]
        var elem_type = elem_type_tags[e]
        _profile_scope_close(
            do_profile,
            events,
            events_need_comma,
            PROFILE_FRAME_ASSEMBLE_FIBER_GEOMETRY,
            t0,
        )
        if elem_type == ElementTypeTag.ElasticBeamColumn2d:
            var k_global: List[List[Float64]] = []
            var f_elem: List[Float64] = []
            var f_load_global = beam2d_element_load_global(
                element_loads,
                elem_load_offsets,
                elem_load_pool,
                e,
                load_scale,
                x1,
                y1,
                x2,
                y2,
            )
            if elem_geom_tags[e] == GeomTransfTag.Linear:
                k_global = beam_global_stiffness(sec.E, sec.A, sec.I, x1, y1, x2, y2)
            elif elem_geom_tags[e] == GeomTransfTag.PDelta:
                for i in range(6):
                    u_elem6[i] = u[dof_map6[i]]
                k_global = beam2d_pdelta_global_stiffness(
                    sec.E, sec.A, sec.I, x1, y1, x2, y2, u_elem6
                )
                f_elem = beam2d_pdelta_global_internal_force(
                    sec.E, sec.A, sec.I, x1, y1, x2, y2, u_elem6
                )
            elif elem_geom_tags[e] == GeomTransfTag.Corotational:
                for i in range(6):
                    u_elem6[i] = u[dof_map6[i]]
                beam2d_corotational_global_tangent_and_internal(
                    sec.E, sec.A, sec.I, x1, y1, x2, y2, u_elem6, k_global, f_elem
                )
            else:
                abort("unsupported geomTransf tag")

            if len(f_elem) != 0:
                for a in range(6):
                    var Aidx = dof_map6[a]
                    _scatter_add_row[6](K, Aidx, k_global[a], dof_map6)
                    F_int[Aidx] += f_elem[a] - f_load_global[a]
            else:
                for a in range(6):
                    var Aidx = dof_map6[a]
                    F_int[Aidx] += (
                        _scatter_add_and_dot_row_simd(K, Aidx, k_global[a], dof_map6, u, 6)
                        - f_load_global[a]
                    )
            if runtime_metrics.enabled:
                _profile_metrics_note_element_timing(
                    runtime_metrics,
                    elem_type,
                    Int(perf_counter_ns()) - t_elem_start,
                )
            continue

        for i in range(6):
            u_elem6[i] = u[dof_map6[i]]
        var k_global: List[List[Float64]] = []
        var f_global: List[Float64] = []
        if sec.type == "ElasticSection2d":
            if (
                sec.cbdi.enabled()
                or sec.shear_rigidity_y != 0.0
                or sec.hinge_radau.is_defined()
                or elem_geom_tags[e] == GeomTransfTag.Corotational
            ):
                var q = elastic_beam_column2d_global_tangent_and_internal(
                    e,
                    x1,
                    y1,
                    x2,
                    y2,
                    u_elem6,
                    _geom_transf_name_from_tag(elem_geom_tags[e]),
                    _beam_integration_name_from_tag(elem_integration_tags[e]),
                    elem_num_int_pts[e],
                    sec,
                    elem_type == ElementTypeTag.ForceBeamColumn2d,
                    element_loads,
                    elem_load_offsets,
                    elem_load_pool,
                    load_scale,
                    force_beam_column2d_scratch,
                    k_global,
                    f_global,
                )
                var q_offset = force_basic_offsets[e]
                if elem_type == ElementTypeTag.ForceBeamColumn2d and q_offset >= 0:
                    force_basic_q[q_offset] = q[0]
                    force_basic_q[q_offset + 1] = q[1]
                    force_basic_q[q_offset + 2] = q[2]
                for a in range(6):
                    var Aidx = dof_map6[a]
                    _scatter_add_row[6](K, Aidx, k_global[a], dof_map6)
                    F_int[Aidx] += f_global[a]
                continue
            var f_load_global = beam2d_element_load_global(
                element_loads,
                elem_load_offsets,
                elem_load_pool,
                e,
                load_scale,
                x1,
                y1,
                x2,
                y2,
            )
            if elem_geom_tags[e] == GeomTransfTag.Linear:
                k_global = beam_global_stiffness(sec.E, sec.A, sec.I, x1, y1, x2, y2)
            elif elem_geom_tags[e] == GeomTransfTag.PDelta:
                k_global = beam2d_pdelta_global_stiffness(
                    sec.E, sec.A, sec.I, x1, y1, x2, y2, u_elem6
                )
            else:
                abort("forceBeamColumn2d/dispBeamColumn2d supports geomTransf Linear or PDelta")
            f_global.resize(6, 0.0)
            for a in range(6):
                var sum = 0.0
                for b in range(6):
                    sum += k_global[a][b] * u_elem6[b]
                f_global[a] = sum
            for a in range(6):
                var Aidx = dof_map6[a]
                _scatter_add_row[6](K, Aidx, k_global[a], dof_map6)
                F_int[Aidx] += f_global[a] - f_load_global[a]
            if runtime_metrics.enabled:
                _profile_metrics_note_element_timing(
                    runtime_metrics,
                    elem_type,
                    Int(perf_counter_ns()) - t_elem_start,
                )
            continue

        var sec_index = fiber_section_index_by_id[elem_section_ids[e]]
        ref sec_def = fiber_section_defs[sec_index]
        var elem_offset = elem_uniaxial_offsets[e]
        var elem_state_count = elem_uniaxial_counts[e]
        var geom_name = _geom_transf_name_from_tag(elem_geom_tags[e])
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
        var k_elem6: List[List[Float64]] = []
        var f_elem6: List[Float64] = []
        if elem_type == ElementTypeTag.ForceBeamColumn2d:
            force_beam_column2d_global_tangent_and_internal(
                e,
                x1,
                y1,
                x2,
                y2,
                u_elem6,
                element_loads,
                elem_load_offsets,
                elem_load_pool,
                load_scale,
                sec_def,
                fiber_section_cells,
                uniaxial_defs,
                uniaxial_states,
                elem_uniaxial_state_ids,
                elem_offset,
                elem_state_count,
                geom_name,
                integration_name,
                elem_num_int_pts[e],
                force_basic_q,
                force_basic_offsets[e],
                force_basic_counts[e],
                runtime_metrics,
                force_beam_column2d_scratch,
                k_elem6,
                f_elem6,
                sec.hinge_radau,
            )
        else:
            disp_beam_column2d_global_tangent_and_internal(
                e,
                x1,
                y1,
                x2,
                y2,
                u_elem6,
                element_loads,
                elem_load_offsets,
                elem_load_pool,
                load_scale,
                sec_def,
                fiber_section_cells,
                uniaxial_defs,
                uniaxial_states,
                elem_uniaxial_state_ids,
                elem_offset,
                elem_state_count,
                geom_name,
                integration_name,
                elem_num_int_pts[e],
                k_elem6,
                f_elem6,
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
        for a in range(6):
            var Aidx = dof_map6[a]
            _scatter_add_row[6](K, Aidx, k_elem6[a], dof_map6)
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
        for a in range(6):
            var Aidx = dof_map6[a]
            F_int[Aidx] += f_elem6[a]
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


fn _assemble_frame2d_element(
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
    mut fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    mut force_beam_column2d_scratch: ForceBeamColumn2dScratch,
    mut dof_map6: List[Int],
    mut u_elem6: List[Float64],
    mut K: List[List[Float64]],
    mut F_int: List[Float64],
) raises:
    var elem_type = elem.type_tag
    if (
        elem_type != ElementTypeTag.ElasticBeamColumn2d
        and elem_type != ElementTypeTag.ForceBeamColumn2d
        and elem_type != ElementTypeTag.DispBeamColumn2d
    ):
        return

    var node1 = nodes[elem.node_index_1]
    var node2 = nodes[elem.node_index_2]
    var dof_offset = elem_dof_offsets[e]
    for i in range(6):
        dof_map6[i] = elem_dof_pool[dof_offset + i]

    if elem_type == ElementTypeTag.ElasticBeamColumn2d:
        var sec = sections_by_id[elem.section]
        var k_global: List[List[Float64]] = []
        var f_elem: List[Float64] = []
        var f_load_global = beam2d_element_load_global(
            element_loads,
            elem_load_offsets,
            elem_load_pool,
            e,
            load_scale,
            node1.x,
            node1.y,
            node2.x,
            node2.y,
        )
        var geom = elem.geom_transf
        if geom == "Linear":
            k_global = beam_global_stiffness(
                sec.E,
                sec.A,
                sec.I,
                node1.x,
                node1.y,
                node2.x,
                node2.y,
            )
        elif geom == "PDelta":
            for i in range(6):
                u_elem6[i] = u[dof_map6[i]]
            k_global = beam2d_pdelta_global_stiffness(
                sec.E,
                sec.A,
                sec.I,
                node1.x,
                node1.y,
                node2.x,
                node2.y,
                u_elem6,
            )
            f_elem = beam2d_pdelta_global_internal_force(
                sec.E, sec.A, sec.I, node1.x, node1.y, node2.x, node2.y, u_elem6
            )
        elif geom == "Corotational":
            for i in range(6):
                u_elem6[i] = u[dof_map6[i]]
            beam2d_corotational_global_tangent_and_internal(
                sec.E,
                sec.A,
                sec.I,
                node1.x,
                node1.y,
                node2.x,
                node2.y,
                u_elem6,
                k_global,
                f_elem,
            )
        else:
            abort("unsupported geomTransf: " + geom)

        if len(f_elem) != 0:
            for a in range(6):
                var Aidx = dof_map6[a]
                _scatter_add_row[6](K, Aidx, k_global[a], dof_map6)
                F_int[Aidx] += f_elem[a] - f_load_global[a]
        else:
            for a in range(6):
                var Aidx = dof_map6[a]
                F_int[Aidx] += (
                    _scatter_add_and_dot_row_simd(K, Aidx, k_global[a], dof_map6, u, 6)
                    - f_load_global[a]
                )
        return

    var sec = sections_by_id[elem.section]
    for i in range(6):
        u_elem6[i] = u[dof_map6[i]]
    var k_global: List[List[Float64]] = []
    var f_global: List[Float64] = []
    if sec.type == "ElasticSection2d":
        if (
            sec.cbdi.enabled()
            or sec.shear_rigidity_y != 0.0
            or sec.hinge_radau.is_defined()
            or elem.geom_tag == GeomTransfTag.Corotational
        ):
            var q = elastic_beam_column2d_global_tangent_and_internal(
                e,
                node1.x,
                node1.y,
                node2.x,
                node2.y,
                u_elem6,
                elem.geom_transf,
                elem.integration,
                elem.num_int_pts,
                sec,
                elem_type == ElementTypeTag.ForceBeamColumn2d,
                element_loads,
                elem_load_offsets,
                elem_load_pool,
                load_scale,
                force_beam_column2d_scratch,
                k_global,
                f_global,
            )
            var q_offset = force_basic_offsets[e]
            if elem_type == ElementTypeTag.ForceBeamColumn2d and q_offset >= 0:
                force_basic_q[q_offset] = q[0]
                force_basic_q[q_offset + 1] = q[1]
                force_basic_q[q_offset + 2] = q[2]
            for a in range(6):
                var Aidx = dof_map6[a]
                _scatter_add_row[6](K, Aidx, k_global[a], dof_map6)
                F_int[Aidx] += f_global[a]
            return
        var f_load_global = beam2d_element_load_global(
            element_loads,
            elem_load_offsets,
            elem_load_pool,
            e,
            load_scale,
            node1.x,
            node1.y,
            node2.x,
            node2.y,
        )
        var geom = elem.geom_transf
        if geom == "Linear":
            k_global = beam_global_stiffness(
                sec.E,
                sec.A,
                sec.I,
                node1.x,
                node1.y,
                node2.x,
                node2.y,
            )
        elif geom == "PDelta":
            k_global = beam2d_pdelta_global_stiffness(
                sec.E,
                sec.A,
                sec.I,
                node1.x,
                node1.y,
                node2.x,
                node2.y,
                u_elem6,
            )
        else:
            abort("forceBeamColumn2d/dispBeamColumn2d supports geomTransf Linear or PDelta")
        f_global.resize(6, 0.0)
        for a in range(6):
            var sum = 0.0
            for b in range(6):
                sum += k_global[a][b] * u_elem6[b]
            f_global[a] = sum
        for a in range(6):
            var Aidx = dof_map6[a]
            _scatter_add_row[6](K, Aidx, k_global[a], dof_map6)
            F_int[Aidx] += f_global[a] - f_load_global[a]
        return

    var sec_index = fiber_section_index_by_id[elem.section]
    ref sec_def = fiber_section_defs[sec_index]
    var elem_offset = elem_uniaxial_offsets[e]
    var elem_state_count = elem_uniaxial_counts[e]
    var k_elem6: List[List[Float64]] = []
    var f_elem6: List[Float64] = []
    if elem_type == ElementTypeTag.ForceBeamColumn2d:
        force_beam_column2d_global_tangent_and_internal(
            e,
            node1.x,
            node1.y,
            node2.x,
            node2.y,
            u_elem6,
            element_loads,
            elem_load_offsets,
            elem_load_pool,
            load_scale,
            sec_def,
            fiber_section_cells,
            uniaxial_defs,
            uniaxial_states,
            elem_uniaxial_state_ids,
            elem_offset,
            elem_state_count,
            elem.geom_transf,
            elem.integration,
            elem.num_int_pts,
            force_basic_q,
            force_basic_offsets[e],
            force_basic_counts[e],
            force_beam_column2d_scratch,
            k_elem6,
            f_elem6,
            sec.hinge_radau,
        )
    else:
        disp_beam_column2d_global_tangent_and_internal(
            e,
            node1.x,
            node1.y,
            node2.x,
            node2.y,
            u_elem6,
            element_loads,
            elem_load_offsets,
            elem_load_pool,
            load_scale,
            sec_def,
            fiber_section_cells,
            uniaxial_defs,
            uniaxial_states,
            elem_uniaxial_state_ids,
            elem_offset,
            elem_state_count,
            elem.geom_transf,
            elem.integration,
            elem.num_int_pts,
            k_elem6,
            f_elem6,
        )
    for a in range(6):
        var Aidx = dof_map6[a]
        _scatter_add_row[6](K, Aidx, k_elem6[a], dof_map6)
        F_int[Aidx] += f_elem6[a]
