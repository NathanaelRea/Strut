from collections import List
from solver.run_case.input_types import ElementLoadInput
from materials import UniMaterialDef, UniMaterialState
from python import Python

from linalg import gaussian_elimination
from solver.assembly import (
    assemble_global_stiffness_and_internal,
    assemble_global_stiffness_banded_frame2d_typed,
)
from solver.banded import banded_gaussian_elimination, estimate_bandwidth_typed
from solver.profile import _append_event
from solver.run_case.input_types import ElementInput, MaterialInput, NodeInput, SectionInput
from solver.run_case.helpers import (
    _collapse_matrix_by_rep,
    _collapse_vector_by_rep,
    _enforce_equal_dof_values,
)
from solver.run_case.load_state import (
    build_active_element_load_state,
    build_active_nodal_load,
)
from solver.time_series import TimeSeriesInput, eval_time_series_input
from sections import FiberCell, FiberSection2dDef, FiberSection3dDef
from tag_types import ElementTypeTag

fn run_static_linear(
    typed_nodes: List[NodeInput],
    typed_elements: List[ElementInput],
    elem_dof_offsets: List[Int],
    elem_dof_pool: List[Int],
    const_element_loads: List[ElementLoadInput],
    pattern_element_loads: List[ElementLoadInput],
    typed_sections_by_id: List[SectionInput],
    typed_materials_by_id: List[MaterialInput],
    id_to_index: List[Int],
    elem_id_to_index: List[Int],
    node_count: Int,
    ndf: Int,
    ndm: Int,
    mut u: List[Float64],
    F_const: List[Float64],
    F_pattern: List[Float64],
    uniaxial_defs: List[UniMaterialDef],
    uniaxial_state_defs: List[Int],
    mut uniaxial_states: List[UniMaterialState],
    elem_uniaxial_offsets: List[Int],
    elem_uniaxial_counts: List[Int],
    elem_uniaxial_state_ids: List[Int],
    force_basic_offsets: List[Int],
    force_basic_counts: List[Int],
    mut force_basic_q: List[Float64],
    fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    fiber_section3d_defs: List[FiberSection3dDef],
    fiber_section3d_cells: List[FiberCell],
    fiber_section3d_index_by_id: List[Int],
    use_banded_linear: Bool,
    free_index: List[Int],
    free: List[Int],
    ts_index: Int,
    time_series: List[TimeSeriesInput],
    time_series_values: List[Float64],
    time_series_times: List[Float64],
    do_profile: Bool,
    t0: Int,
    mut events: String,
    mut events_need_comma: Bool,
    frame_assemble_stiffness: Int,
    frame_kff_extract: Int,
    frame_solve_linear: Int,
    total_dofs: Int,
    has_transformation_mpc: Bool,
    rep_dof: List[Int],
    constrained: List[Bool],
) raises:
    var time = Python.import_module("time")
    var free_count = len(free)
    var asm_dof_map6: List[Int] = []
    var asm_dof_map12: List[Int] = []
    var asm_u_elem6: List[Float64] = []
    var asm_free_map6: List[Int] = []
    var asm_f_dummy: List[Float64] = []
    var load_scale = 1.0
    if ts_index >= 0:
        load_scale = eval_time_series_input(
            time_series[ts_index], 1.0, time_series_values, time_series_times
        )
    var F_active = build_active_nodal_load(F_const, F_pattern, load_scale)
    var active_element_load_state = build_active_element_load_state(
        const_element_loads,
        pattern_element_loads,
        load_scale,
        typed_elements,
        elem_id_to_index,
        ndm,
        ndf,
    )
    if do_profile:
        var t_asm_start = Int(time.perf_counter_ns())
        var asm_start_us = (t_asm_start - t0) // 1000
        _append_event(
            events, events_need_comma, "O", frame_assemble_stiffness, asm_start_us
        )
    var bw = 0
    var use_typed_banded = False
    var K_ff_banded: List[List[Float64]] = []
    var K: List[List[Float64]] = []
    var F_int_dummy: List[Float64] = []
    F_int_dummy.resize(total_dofs, 0.0)
    if use_banded_linear:
        bw = estimate_bandwidth_typed(typed_elements, free_index)
        if bw > free_count - 1:
            bw = free_count - 1
        use_typed_banded = True
        for e in range(len(typed_elements)):
            var elem_type = typed_elements[e].type_tag
            if (
                elem_type != ElementTypeTag.ElasticBeamColumn2d
                and elem_type != ElementTypeTag.ForceBeamColumn2d
                and elem_type != ElementTypeTag.DispBeamColumn2d
            ):
                use_typed_banded = False
                break
        if use_typed_banded:
            if len(active_element_load_state.element_loads) > 0:
                use_typed_banded = False
        if use_typed_banded:
            K_ff_banded = assemble_global_stiffness_banded_frame2d_typed(
                typed_nodes,
                typed_elements,
                active_element_load_state.element_loads,
                active_element_load_state.elem_load_offsets,
                active_element_load_state.elem_load_pool,
                1.0,
                typed_sections_by_id,
                u,
                uniaxial_defs,
                uniaxial_states,
                elem_uniaxial_offsets,
                elem_uniaxial_counts,
                elem_uniaxial_state_ids,
                force_basic_offsets,
                force_basic_counts,
                force_basic_q,
                fiber_section_defs,
                fiber_section_cells,
                fiber_section_index_by_id,
                free_index,
                free_count,
                bw,
                elem_dof_offsets,
                elem_dof_pool,
                asm_free_map6,
                asm_u_elem6,
                asm_f_dummy,
            )
    if not (use_banded_linear and use_typed_banded):
        for _ in range(total_dofs):
            var row: List[Float64] = []
            row.resize(total_dofs, 0.0)
            K.append(row^)
        assemble_global_stiffness_and_internal(
            typed_nodes,
            typed_elements,
            active_element_load_state.element_loads,
            active_element_load_state.elem_load_offsets,
            active_element_load_state.elem_load_pool,
            1.0,
            typed_sections_by_id,
            typed_materials_by_id,
            id_to_index,
            node_count,
            ndf,
            ndm,
            u,
            uniaxial_defs,
            uniaxial_state_defs,
            uniaxial_states,
            elem_uniaxial_offsets,
            elem_uniaxial_counts,
            elem_uniaxial_state_ids,
            force_basic_offsets,
            force_basic_counts,
            force_basic_q,
            fiber_section_defs,
            fiber_section_cells,
            fiber_section_index_by_id,
            fiber_section3d_defs,
            fiber_section3d_cells,
            fiber_section3d_index_by_id,
            elem_dof_offsets,
            elem_dof_pool,
            asm_dof_map6,
            asm_dof_map12,
            asm_u_elem6,
            K,
            F_int_dummy,
        )
        if has_transformation_mpc:
            K = _collapse_matrix_by_rep(K, rep_dof)
            F_int_dummy = _collapse_vector_by_rep(F_int_dummy, rep_dof)
    if do_profile:
        var t_asm_end = Int(time.perf_counter_ns())
        var asm_end_us = (t_asm_end - t0) // 1000
        _append_event(
            events, events_need_comma, "C", frame_assemble_stiffness, asm_end_us
        )
    var F_f: List[Float64] = []
    F_f.resize(free_count, 0.0)
    if do_profile:
        var t_kff_start = Int(time.perf_counter_ns())
        var kff_start_us = (t_kff_start - t0) // 1000
        _append_event(
            events, events_need_comma, "O", frame_kff_extract, kff_start_us
        )
    var K_ff: List[List[Float64]] = []
    if not (use_banded_linear and use_typed_banded):
        for _ in range(free_count):
            var row: List[Float64] = []
            row.resize(free_count, 0.0)
            K_ff.append(row^)
        for i in range(free_count):
            var row_i = free[i]
            F_f[i] = F_active[row_i] - F_int_dummy[row_i]
            for j in range(free_count):
                K_ff[i][j] = K[row_i][free[j]]
    else:
        for i in range(free_count):
            F_f[i] = F_active[free[i]]
    if do_profile:
        var t_kff_end = Int(time.perf_counter_ns())
        var kff_end_us = (t_kff_end - t0) // 1000
        _append_event(events, events_need_comma, "C", frame_kff_extract, kff_end_us)
    if do_profile:
        var t_solve_lin_start = Int(time.perf_counter_ns())
        var solve_lin_start_us = (t_solve_lin_start - t0) // 1000
        _append_event(
            events, events_need_comma, "O", frame_solve_linear, solve_lin_start_us
        )
    var u_f: List[Float64]
    if use_banded_linear and use_typed_banded:
        u_f = banded_gaussian_elimination(K_ff_banded, bw, F_f)
    else:
        u_f = gaussian_elimination(K_ff, F_f)
    if do_profile:
        var t_solve_lin_end = Int(time.perf_counter_ns())
        var solve_lin_end_us = (t_solve_lin_end - t0) // 1000
        _append_event(
            events, events_need_comma, "C", frame_solve_linear, solve_lin_end_us
        )
    for i in range(free_count):
        u[free[i]] = u_f[i]
    if has_transformation_mpc:
        _enforce_equal_dof_values(u, rep_dof, constrained)
