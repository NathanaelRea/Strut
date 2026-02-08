from collections import List
from materials import UniMaterialDef
from python import Python, PythonObject

from linalg import gaussian_elimination
from solver.assembly import assemble_global_stiffness, assemble_global_stiffness_banded
from solver.banded import banded_gaussian_elimination, estimate_bandwidth
from solver.profile import _append_event
from solver.time_series import eval_time_series

fn run_static_linear(
    nodes: PythonObject,
    elements: PythonObject,
    sections_by_id: List[PythonObject],
    materials_by_id: List[PythonObject],
    id_to_index: List[Int],
    node_count: Int,
    ndf: Int,
    ndm: Int,
    mut u: List[Float64],
    mut F_total: List[Float64],
    uniaxial_defs: List[UniMaterialDef],
    uniaxial_state_defs: List[Int],
    elem_uniaxial_offsets: List[Int],
    elem_uniaxial_counts: List[Int],
    elem_uniaxial_state_ids: List[Int],
    use_banded_linear: Bool,
    free_index: List[Int],
    free: List[Int],
    ts_index: Int,
    time_series: PythonObject,
    do_profile: Bool,
    t0: Int,
    mut events: String,
    mut events_need_comma: Bool,
    frame_assemble_stiffness: Int,
    frame_kff_extract: Int,
    frame_solve_linear: Int,
    total_dofs: Int,
) raises:
    var time = Python.import_module("time")
    if ts_index >= 0:
        var factor = eval_time_series(time_series[ts_index], 1.0)
        for i in range(total_dofs):
            F_total[i] *= factor
    if do_profile:
        var t_asm_start = Int(time.perf_counter_ns())
        var asm_start_us = (t_asm_start - t0) // 1000
        _append_event(
            events, events_need_comma, "O", frame_assemble_stiffness, asm_start_us
        )
    var bw = 0
    var K_ff_banded: List[List[Float64]] = []
    var K: List[List[Float64]] = []
    if use_banded_linear:
        bw = estimate_bandwidth(elements, id_to_index, ndf, free_index)
        if bw > len(free) - 1:
            bw = len(free) - 1
        K_ff_banded = assemble_global_stiffness_banded(
            nodes,
            elements,
            sections_by_id,
            materials_by_id,
            id_to_index,
            node_count,
            ndf,
            ndm,
            u,
            uniaxial_defs,
            uniaxial_state_defs,
            elem_uniaxial_offsets,
            elem_uniaxial_counts,
            elem_uniaxial_state_ids,
            free_index,
            len(free),
            bw,
        )
    else:
        K = assemble_global_stiffness(
            nodes,
            elements,
            sections_by_id,
            materials_by_id,
            id_to_index,
            node_count,
            ndf,
            ndm,
            u,
            uniaxial_defs,
            uniaxial_state_defs,
            elem_uniaxial_offsets,
            elem_uniaxial_counts,
            elem_uniaxial_state_ids,
        )
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
    if not use_banded_linear:
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
    if use_banded_linear:
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
