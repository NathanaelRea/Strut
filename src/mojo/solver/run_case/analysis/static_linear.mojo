from collections import List
from materials import UniMaterialDef, UniMaterialState
from python import Python

from linalg import gaussian_elimination
from solver.assembly import (
    assemble_global_stiffness_typed,
    assemble_global_stiffness_banded_frame2d_typed,
)
from solver.banded import banded_gaussian_elimination, estimate_bandwidth_typed
from solver.profile import _append_event
from solver.run_case.input_types import ElementInput, MaterialInput, NodeInput, SectionInput
from solver.run_case.helpers import _collapse_matrix_by_rep, _enforce_equal_dof_values
from solver.time_series import TimeSeriesInput, eval_time_series_input
from sections import FiberCell, FiberSection2dDef

fn run_static_linear(
    typed_nodes: List[NodeInput],
    typed_elements: List[ElementInput],
    typed_sections_by_id: List[SectionInput],
    typed_materials_by_id: List[MaterialInput],
    id_to_index: List[Int],
    node_count: Int,
    ndf: Int,
    ndm: Int,
    mut u: List[Float64],
    mut F_total: List[Float64],
    uniaxial_defs: List[UniMaterialDef],
    uniaxial_state_defs: List[Int],
    mut uniaxial_states: List[UniMaterialState],
    elem_uniaxial_offsets: List[Int],
    elem_uniaxial_counts: List[Int],
    elem_uniaxial_state_ids: List[Int],
    fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
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
    if ts_index >= 0:
        var factor = eval_time_series_input(
            time_series[ts_index], 1.0, time_series_values, time_series_times
        )
        for i in range(total_dofs):
            F_total[i] *= factor
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
    if use_banded_linear:
        bw = estimate_bandwidth_typed(typed_elements, free_index)
        if bw > len(free) - 1:
            bw = len(free) - 1
        use_typed_banded = True
        for e in range(len(typed_elements)):
            var elem_type = typed_elements[e].type_tag
            if elem_type != 1 and elem_type != 2:
                use_typed_banded = False
                break
        if use_typed_banded:
            K_ff_banded = assemble_global_stiffness_banded_frame2d_typed(
                typed_nodes,
                typed_elements,
                typed_sections_by_id,
                u,
                uniaxial_defs,
                uniaxial_states,
                elem_uniaxial_offsets,
                elem_uniaxial_counts,
                elem_uniaxial_state_ids,
                fiber_section_defs,
                fiber_section_cells,
                fiber_section_index_by_id,
                free_index,
                len(free),
                bw,
            )
    if not (use_banded_linear and use_typed_banded):
        K = assemble_global_stiffness_typed(
            typed_nodes,
            typed_elements,
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
            fiber_section_defs,
            fiber_section_cells,
            fiber_section_index_by_id,
        )
        if has_transformation_mpc:
            K = _collapse_matrix_by_rep(K, rep_dof)
    if do_profile:
        var t_asm_end = Int(time.perf_counter_ns())
        var asm_end_us = (t_asm_end - t0) // 1000
        _append_event(
            events, events_need_comma, "C", frame_assemble_stiffness, asm_end_us
        )
    var F_f: List[Float64] = []
    F_f.resize(len(free), 0.0)
    if do_profile:
        var t_kff_start = Int(time.perf_counter_ns())
        var kff_start_us = (t_kff_start - t0) // 1000
        _append_event(
            events, events_need_comma, "O", frame_kff_extract, kff_start_us
        )
    for i in range(len(free)):
        F_f[i] = F_total[free[i]]
    var K_ff: List[List[Float64]] = []
    if not (use_banded_linear and use_typed_banded):
        for _ in range(len(free)):
            var row: List[Float64] = []
            row.resize(len(free), 0.0)
            K_ff.append(row^)
        for i in range(len(free)):
            for j in range(len(free)):
                K_ff[i][j] = K[free[i]][free[j]]
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
    for i in range(len(free)):
        u[free[i]] = u_f[i]
    if has_transformation_mpc:
        _enforce_equal_dof_values(u, rep_dof, constrained)
