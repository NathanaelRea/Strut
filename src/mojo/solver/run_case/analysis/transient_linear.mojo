from collections import List
from materials import UniMaterialDef, UniMaterialState
from os import abort
from python import Python
from sections import FiberCell, FiberSection2dDef, FiberSection3dDef

from linalg import lu_factorize_into, lu_solve_into
from solver.assembly import assemble_global_stiffness_typed, assemble_internal_forces_typed
from solver.dof import node_dof_index, require_dof_in_range
from solver.profile import _append_event
from solver.run_case.input_types import (
    AnalysisInput,
    ElementLoadInput,
    ElementInput,
    MaterialInput,
    NodeInput,
    RecorderInput,
    SectionInput,
)
from solver.run_case.load_state import (
    build_active_element_load_state,
    build_active_nodal_load,
)
from solver.time_series import TimeSeriesInput, eval_time_series_input

from solver.run_case.helpers import (
    _append_output,
    _collapse_matrix_by_rep,
    _element_force_global_for_recorder,
    _enforce_equal_dof_values,
    _flush_envelope_outputs,
    _format_values_line,
    _has_recorder_type,
    _drift_value,
    _section_response_for_recorder,
    _update_envelope,
)
from tag_types import RecorderTypeTag


@always_inline
fn _copy_vector_simd4(mut dst: List[Float64], src: List[Float64]):
    if len(dst) != len(src):
        abort("vector size mismatch in _copy_vector_simd4")
    var n = len(dst)
    var i = 0
    while i + 3 < n:
        var chunk = SIMD[DType.float64, 4](src[i], src[i + 1], src[i + 2], src[i + 3])
        dst[i] = chunk[0]
        dst[i + 1] = chunk[1]
        dst[i + 2] = chunk[2]
        dst[i + 3] = chunk[3]
        i += 4
    while i < n:
        dst[i] = src[i]
        i += 1


@always_inline
fn _dot_row_simd4(row: List[Float64], vec: List[Float64], count: Int) -> Float64:
    var sum = 0.0
    var i = 0
    while i + 3 < count:
        var a = SIMD[DType.float64, 4](row[i], row[i + 1], row[i + 2], row[i + 3])
        var b = SIMD[DType.float64, 4](vec[i], vec[i + 1], vec[i + 2], vec[i + 3])
        sum += (a * b).reduce_add()
        i += 4
    while i < count:
        sum += row[i] * vec[i]
        i += 1
    return sum


@always_inline
fn _build_effective_rhs_newmark_simd4(
    free: List[Int],
    u: List[Float64],
    v: List[Float64],
    a: List[Float64],
    F_ext_step: List[Float64],
    M_f: List[Float64],
    a0: Float64,
    a1: Float64,
    a2: Float64,
    a3: Float64,
    a4: Float64,
    a5: Float64,
    mut P_ext_f: List[Float64],
    mut C_term: List[Float64],
    mut P_eff: List[Float64],
):
    var n = len(free)
    var i = 0
    var a0_vec = SIMD[DType.float64, 4](a0)
    var a1_vec = SIMD[DType.float64, 4](a1)
    var a2_vec = SIMD[DType.float64, 4](a2)
    var a3_vec = SIMD[DType.float64, 4](a3)
    var a4_vec = SIMD[DType.float64, 4](a4)
    var a5_vec = SIMD[DType.float64, 4](a5)
    while i + 3 < n:
        var idx0 = free[i]
        var idx1 = free[i + 1]
        var idx2 = free[i + 2]
        var idx3 = free[i + 3]
        var u_vec = SIMD[DType.float64, 4](u[idx0], u[idx1], u[idx2], u[idx3])
        var v_vec = SIMD[DType.float64, 4](v[idx0], v[idx1], v[idx2], v[idx3])
        var a_vec = SIMD[DType.float64, 4](a[idx0], a[idx1], a[idx2], a[idx3])
        var c_vec = a1_vec * u_vec + a4_vec * v_vec + a5_vec * a_vec
        var p_ext_vec = SIMD[DType.float64, 4](
            F_ext_step[idx0], F_ext_step[idx1], F_ext_step[idx2], F_ext_step[idx3]
        )
        var m_vec = SIMD[DType.float64, 4](M_f[i], M_f[i + 1], M_f[i + 2], M_f[i + 3])
        var p_eff_vec = p_ext_vec + m_vec * (a0_vec * u_vec + a2_vec * v_vec + a3_vec * a_vec)
        P_ext_f[i] = p_ext_vec[0]
        P_ext_f[i + 1] = p_ext_vec[1]
        P_ext_f[i + 2] = p_ext_vec[2]
        P_ext_f[i + 3] = p_ext_vec[3]
        C_term[i] = c_vec[0]
        C_term[i + 1] = c_vec[1]
        C_term[i + 2] = c_vec[2]
        C_term[i + 3] = c_vec[3]
        P_eff[i] = p_eff_vec[0]
        P_eff[i + 1] = p_eff_vec[1]
        P_eff[i + 2] = p_eff_vec[2]
        P_eff[i + 3] = p_eff_vec[3]
        i += 4
    while i < n:
        var idx = free[i]
        P_ext_f[i] = F_ext_step[idx]
        C_term[i] = a1 * u[idx] + a4 * v[idx] + a5 * a[idx]
        P_eff[i] = P_ext_f[i] + M_f[i] * (a0 * u[idx] + a2 * v[idx] + a3 * a[idx])
        i += 1


@always_inline
fn _update_newmark_state_from_solution_simd4(
    free: List[Int],
    mut u: List[Float64],
    mut v: List[Float64],
    mut a: List[Float64],
    u_f: List[Float64],
    a0: Float64,
    a2: Float64,
    a3: Float64,
    gamma: Float64,
    dt: Float64,
):
    var n = len(free)
    var i = 0
    var a0_vec = SIMD[DType.float64, 4](a0)
    var a2_vec = SIMD[DType.float64, 4](a2)
    var a3_vec = SIMD[DType.float64, 4](a3)
    var gamma_vec = SIMD[DType.float64, 4](gamma)
    var one_minus_gamma_vec = SIMD[DType.float64, 4](1.0 - gamma)
    var dt_vec = SIMD[DType.float64, 4](dt)
    while i + 3 < n:
        var idx0 = free[i]
        var idx1 = free[i + 1]
        var idx2 = free[i + 2]
        var idx3 = free[i + 3]
        var u_old = SIMD[DType.float64, 4](u[idx0], u[idx1], u[idx2], u[idx3])
        var v_old = SIMD[DType.float64, 4](v[idx0], v[idx1], v[idx2], v[idx3])
        var a_old = SIMD[DType.float64, 4](a[idx0], a[idx1], a[idx2], a[idx3])
        var u_next = SIMD[DType.float64, 4](u_f[i], u_f[i + 1], u_f[i + 2], u_f[i + 3])
        var a_next = a0_vec * (u_next - u_old) - a2_vec * v_old - a3_vec * a_old
        var v_next = v_old + dt_vec * (one_minus_gamma_vec * a_old + gamma_vec * a_next)
        u[idx0] = u_next[0]
        u[idx1] = u_next[1]
        u[idx2] = u_next[2]
        u[idx3] = u_next[3]
        v[idx0] = v_next[0]
        v[idx1] = v_next[1]
        v[idx2] = v_next[2]
        v[idx3] = v_next[3]
        a[idx0] = a_next[0]
        a[idx1] = a_next[1]
        a[idx2] = a_next[2]
        a[idx3] = a_next[3]
        i += 4
    while i < n:
        var idx = free[i]
        var u_next = u_f[i]
        var a_next = a0 * (u_next - u[idx]) - a2 * v[idx] - a3 * a[idx]
        var v_next = v[idx] + dt * ((1.0 - gamma) * a[idx] + gamma * a_next)
        u[idx] = u_next
        v[idx] = v_next
        a[idx] = a_next
        i += 1


fn run_transient_linear(
    analysis: AnalysisInput,
    steps: Int,
    ts_index: Int,
    time_series: List[TimeSeriesInput],
    time_series_values: List[Float64],
    time_series_times: List[Float64],
    pattern_type: String,
    uniform_excitation_direction: Int,
    uniform_accel_ts_index: Int,
    rayleigh_alpha_m: Float64,
    rayleigh_beta_k: Float64,
    rayleigh_beta_k_init: Float64,
    rayleigh_beta_k_comm: Float64,
    typed_nodes: List[NodeInput],
    typed_elements: List[ElementInput],
    const_element_loads: List[ElementLoadInput],
    pattern_element_loads: List[ElementLoadInput],
    typed_sections_by_id: List[SectionInput],
    typed_materials_by_id: List[MaterialInput],
    id_to_index: List[Int],
    node_count: Int,
    ndf: Int,
    ndm: Int,
    total_dofs: Int,
    mut u: List[Float64],
    uniaxial_defs: List[UniMaterialDef],
    uniaxial_state_defs: List[Int],
    mut uniaxial_states: List[UniMaterialState],
    elem_uniaxial_offsets: List[Int],
    elem_uniaxial_counts: List[Int],
    elem_uniaxial_state_ids: List[Int],
    force_basic_offsets: List[Int],
    force_basic_counts: List[Int],
    mut force_basic_q: List[Float64],
    F_const: List[Float64],
    F_pattern: List[Float64],
    M_total: List[Float64],
    free: List[Int],
    recorders: List[RecorderInput],
    recorder_nodes_pool: List[Int],
    recorder_elements_pool: List[Int],
    recorder_dofs_pool: List[Int],
    recorder_sections_pool: List[Int],
    elem_id_to_index: List[Int],
    fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    fiber_section3d_defs: List[FiberSection3dDef],
    fiber_section3d_cells: List[FiberCell],
    fiber_section3d_index_by_id: List[Int],
    mut transient_output_files: List[String],
    mut transient_output_buffers: List[List[String]],
    has_transformation_mpc: Bool,
    rep_dof: List[Int],
    constrained: List[Bool],
    do_profile: Bool,
    t0: Int,
    mut events: String,
    mut events_need_comma: Bool,
    frame_assemble_stiffness: Int,
    frame_solve_linear: Int,
    frame_time_series_eval: Int,
    frame_constraints: Int,
    frame_recorders: Int,
    frame_factorize: Int,
    frame_transient_step: Int,
) raises:
    var time = Python.import_module("time")

    var dt = analysis.dt
    if dt <= 0.0:
        abort("transient_linear requires dt > 0")
    if pattern_type != "Plain" and pattern_type != "UniformExcitation":
        abort("unsupported pattern type: " + pattern_type)
    if pattern_type == "UniformExcitation":
        if uniform_excitation_direction < 1 or uniform_excitation_direction > ndm:
            abort("UniformExcitation direction out of range")
        if uniform_accel_ts_index < 0:
            abort("UniformExcitation missing accel time_series")
    var uniform_excitation_dofs: List[Int] = []
    if pattern_type == "UniformExcitation":
        for i in range(total_dofs):
            if (i % ndf) + 1 == uniform_excitation_direction:
                uniform_excitation_dofs.append(i)
    var integrator_type = analysis.integrator_type
    if integrator_type == "":
        integrator_type = "Newmark"
    if integrator_type != "Newmark":
        abort("transient_linear only supports Newmark integrator")
    var gamma = analysis.integrator_gamma
    var beta = analysis.integrator_beta
    if beta <= 0.0:
        abort("Newmark beta must be > 0")

    var free_count = len(free)
    var M_f: List[Float64] = []
    M_f.resize(free_count, 0.0)
    var has_mass = False
    for i in range(free_count):
        var m = M_total[free[i]]
        M_f[i] = m
        if m != 0.0:
            has_mass = True
    if not has_mass:
        abort("transient_linear requires masses on free dofs")

    if do_profile:
        var t_asm_start = Int(time.perf_counter_ns())
        var asm_start_us = (t_asm_start - t0) // 1000
        _append_event(
            events, events_need_comma, "O", frame_assemble_stiffness, asm_start_us
        )
    var initial_pattern_scale = 0.0
    if pattern_type == "Plain":
        if ts_index >= 0:
            initial_pattern_scale = eval_time_series_input(
                time_series[ts_index], 0.0, time_series_values, time_series_times
            )
        else:
            initial_pattern_scale = 1.0
    var active_element_load_state = build_active_element_load_state(
        const_element_loads,
        pattern_element_loads,
        initial_pattern_scale,
        typed_elements,
        elem_id_to_index,
        ndm,
        ndf,
    )
    var K = assemble_global_stiffness_typed(
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
    )
    if do_profile:
        var t_asm_end = Int(time.perf_counter_ns())
        var asm_end_us = (t_asm_end - t0) // 1000
        _append_event(
            events, events_need_comma, "C", frame_assemble_stiffness, asm_end_us
        )
    if has_transformation_mpc:
        if do_profile:
            var t_constraints_start = Int(time.perf_counter_ns())
            var constraints_start_us = (t_constraints_start - t0) // 1000
            _append_event(
                events,
                events_need_comma,
                "O",
                frame_constraints,
                constraints_start_us,
            )
        K = _collapse_matrix_by_rep(K, rep_dof)
        if do_profile:
            var t_constraints_end = Int(time.perf_counter_ns())
            var constraints_end_us = (t_constraints_end - t0) // 1000
            _append_event(
                events,
                events_need_comma,
                "C",
                frame_constraints,
                constraints_end_us,
            )
    var K_ff: List[List[Float64]] = []
    for _ in range(free_count):
        var row: List[Float64] = []
        row.resize(free_count, 0.0)
        K_ff.append(row^)
    for i in range(free_count):
        for j in range(free_count):
            K_ff[i][j] = K[free[i]][free[j]]

    var a0 = 1.0 / (beta * dt * dt)
    var a1 = gamma / (beta * dt)
    var a2 = 1.0 / (beta * dt)
    var a3 = 1.0 / (2.0 * beta) - 1.0
    var a4 = gamma / beta - 1.0
    var a5 = dt * (gamma / (2.0 * beta) - 1.0)
    var beta_sum = rayleigh_beta_k + rayleigh_beta_k_init + rayleigh_beta_k_comm
    var C_ff: List[List[Float64]] = []
    for i in range(free_count):
        var row_c: List[Float64] = []
        row_c.resize(free_count, 0.0)
        C_ff.append(row_c^)
        C_ff[i][i] = rayleigh_alpha_m * M_f[i]
    if beta_sum != 0.0:
        for i in range(free_count):
            for j in range(free_count):
                C_ff[i][j] += beta_sum * K_ff[i][j]

    var K_eff: List[List[Float64]] = []
    for _ in range(free_count):
        var row_eff: List[Float64] = []
        row_eff.resize(free_count, 0.0)
        K_eff.append(row_eff^)
    for i in range(free_count):
        for j in range(free_count):
            K_eff[i][j] = K_ff[i][j] + a1 * C_ff[i][j]
        K_eff[i][i] += a0 * M_f[i]
    var K_lu: List[List[Float64]] = []
    for i in range(free_count):
        var row_lu: List[Float64] = []
        row_lu.resize(free_count, 0.0)
        K_lu.append(row_lu^)
        for j in range(free_count):
            K_lu[i][j] = K_eff[i][j]
    var lu_pivots: List[Int] = []
    lu_pivots.resize(free_count, 0)
    if do_profile:
        var t_fac_start = Int(time.perf_counter_ns())
        var fac_start_us = (t_fac_start - t0) // 1000
        _append_event(events, events_need_comma, "O", frame_factorize, fac_start_us)
    lu_factorize_into(K_lu, lu_pivots)
    if do_profile:
        var t_fac_end = Int(time.perf_counter_ns())
        var fac_end_us = (t_fac_end - t0) // 1000
        _append_event(events, events_need_comma, "C", frame_factorize, fac_end_us)

    var v: List[Float64] = []
    v.resize(total_dofs, 0.0)
    var a: List[Float64] = []
    a.resize(total_dofs, 0.0)

    var F_ext_step: List[Float64] = []
    F_ext_step.resize(total_dofs, 0.0)
    var P_ext_f: List[Float64] = []
    P_ext_f.resize(free_count, 0.0)
    var P_eff: List[Float64] = []
    P_eff.resize(free_count, 0.0)
    var C_term: List[Float64] = []
    C_term.resize(free_count, 0.0)
    var lu_rhs: List[Float64] = []
    lu_rhs.resize(free_count, 0.0)
    var lu_work: List[Float64] = []
    lu_work.resize(free_count, 0.0)
    var u_f: List[Float64] = []
    u_f.resize(free_count, 0.0)
    var record_reactions = _has_recorder_type(recorders, RecorderTypeTag.NodeReaction)
    var record_any_element_force = (
        _has_recorder_type(recorders, RecorderTypeTag.ElementForce) or _has_recorder_type(recorders, RecorderTypeTag.EnvelopeElementForce)
    )
    var elem_count = len(typed_elements)
    var elem_force_cached: List[Bool] = []
    var elem_force_values: List[List[Float64]] = []
    if record_any_element_force:
        elem_force_cached.resize(elem_count, False)
        for _ in range(elem_count):
            var empty_force: List[Float64] = []
            elem_force_values.append(empty_force^)
    var envelope_files: List[String] = []
    var envelope_min: List[List[Float64]] = []
    var envelope_max: List[List[Float64]] = []
    var envelope_abs: List[List[Float64]] = []

    for step in range(steps):
        if do_profile:
            var t_step_start = Int(time.perf_counter_ns())
            var step_start_us = (t_step_start - t0) // 1000
            _append_event(
                events, events_need_comma, "O", frame_transient_step, step_start_us
            )
        if has_transformation_mpc:
            if do_profile:
                var t_constraints_start = Int(time.perf_counter_ns())
                var constraints_start_us = (t_constraints_start - t0) // 1000
                _append_event(
                    events,
                    events_need_comma,
                    "O",
                    frame_constraints,
                    constraints_start_us,
                )
            _enforce_equal_dof_values(u, rep_dof, constrained)
            _enforce_equal_dof_values(v, rep_dof, constrained)
            _enforce_equal_dof_values(a, rep_dof, constrained)
            if do_profile:
                var t_constraints_end = Int(time.perf_counter_ns())
                var constraints_end_us = (t_constraints_end - t0) // 1000
                _append_event(
                    events,
                    events_need_comma,
                    "C",
                    frame_constraints,
                    constraints_end_us,
                )
        var t = Float64(step + 1) * dt
        if pattern_type == "UniformExcitation":
            _copy_vector_simd4(F_ext_step, F_const)
            active_element_load_state = build_active_element_load_state(
                const_element_loads,
                pattern_element_loads,
                0.0,
                typed_elements,
                elem_id_to_index,
                ndm,
                ndf,
            )
            if do_profile:
                var t_ts_start = Int(time.perf_counter_ns())
                var ts_start_us = (t_ts_start - t0) // 1000
                _append_event(
                    events,
                    events_need_comma,
                    "O",
                    frame_time_series_eval,
                    ts_start_us,
                )
            var ag = eval_time_series_input(
                time_series[uniform_accel_ts_index],
                t,
                time_series_values,
                time_series_times,
            )
            if do_profile:
                var t_ts_end = Int(time.perf_counter_ns())
                var ts_end_us = (t_ts_end - t0) // 1000
                _append_event(
                    events,
                    events_need_comma,
                    "C",
                    frame_time_series_eval,
                    ts_end_us,
                )
            for i in range(len(uniform_excitation_dofs)):
                var dof_idx = uniform_excitation_dofs[i]
                F_ext_step[dof_idx] += -M_total[dof_idx] * ag
        else:
            if ts_index >= 0:
                if do_profile:
                    var t_ts_start = Int(time.perf_counter_ns())
                    var ts_start_us = (t_ts_start - t0) // 1000
                    _append_event(
                        events,
                        events_need_comma,
                        "O",
                        frame_time_series_eval,
                        ts_start_us,
                    )
                var factor = eval_time_series_input(
                    time_series[ts_index],
                    t,
                    time_series_values,
                    time_series_times,
                )
                if do_profile:
                    var t_ts_end = Int(time.perf_counter_ns())
                    var ts_end_us = (t_ts_end - t0) // 1000
                    _append_event(
                        events,
                    events_need_comma,
                    "C",
                    frame_time_series_eval,
                    ts_end_us,
                )
                _copy_vector_simd4(
                    F_ext_step, build_active_nodal_load(F_const, F_pattern, factor)
                )
                active_element_load_state = build_active_element_load_state(
                    const_element_loads,
                    pattern_element_loads,
                    factor,
                    typed_elements,
                    elem_id_to_index,
                    ndm,
                    ndf,
                )
            else:
                _copy_vector_simd4(F_ext_step, build_active_nodal_load(F_const, F_pattern, 1.0))
                active_element_load_state = build_active_element_load_state(
                    const_element_loads,
                    pattern_element_loads,
                    1.0,
                    typed_elements,
                    elem_id_to_index,
                    ndm,
                    ndf,
                )
        _build_effective_rhs_newmark_simd4(
            free, u, v, a, F_ext_step, M_f, a0, a1, a2, a3, a4, a5, P_ext_f, C_term, P_eff
        )
        for i in range(free_count):
            lu_rhs[i] = P_eff[i] + _dot_row_simd4(C_ff[i], C_term, free_count)
        if do_profile:
            var t_solve_start = Int(time.perf_counter_ns())
            var solve_start_us = (t_solve_start - t0) // 1000
            _append_event(
                events, events_need_comma, "O", frame_solve_linear, solve_start_us
            )
        lu_solve_into(K_lu, lu_pivots, lu_rhs, lu_work, u_f)
        if do_profile:
            var t_solve_end = Int(time.perf_counter_ns())
            var solve_end_us = (t_solve_end - t0) // 1000
            _append_event(
                events, events_need_comma, "C", frame_solve_linear, solve_end_us
            )
        _update_newmark_state_from_solution_simd4(
            free, u, v, a, u_f, a0, a2, a3, gamma, dt
        )
        if has_transformation_mpc:
            if do_profile:
                var t_constraints_start = Int(time.perf_counter_ns())
                var constraints_start_us = (t_constraints_start - t0) // 1000
                _append_event(
                    events,
                    events_need_comma,
                    "O",
                    frame_constraints,
                    constraints_start_us,
                )
            _enforce_equal_dof_values(u, rep_dof, constrained)
            _enforce_equal_dof_values(v, rep_dof, constrained)
            _enforce_equal_dof_values(a, rep_dof, constrained)
            if do_profile:
                var t_constraints_end = Int(time.perf_counter_ns())
                var constraints_end_us = (t_constraints_end - t0) // 1000
                _append_event(
                    events,
                    events_need_comma,
                    "C",
                    frame_constraints,
                    constraints_end_us,
                )

        var F_int_reaction: List[Float64] = []
        if do_profile:
            var t_rec_start = Int(time.perf_counter_ns())
            var rec_start_us = (t_rec_start - t0) // 1000
            _append_event(events, events_need_comma, "O", frame_recorders, rec_start_us)
        if record_reactions:
            F_int_reaction = assemble_internal_forces_typed(
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
            )
        if record_any_element_force:
            for i in range(elem_count):
                elem_force_cached[i] = False

        for r in range(len(recorders)):
            var rec = recorders[r]
            if rec.type_tag == RecorderTypeTag.NodeDisplacement:
                for nidx in range(rec.node_count):
                    var node_id = recorder_nodes_pool[rec.node_offset + nidx]
                    var i = id_to_index[node_id]
                    var line = String()
                    for j in range(rec.dof_count):
                        var dof = recorder_dofs_pool[rec.dof_offset + j]
                        require_dof_in_range(dof, ndf, "recorder")
                        var value = u[node_dof_index(i, dof, ndf)]
                        if j > 0:
                            line += " "
                        line += String(value)
                    line += "\n"
                    var filename = rec.output + "_node" + String(node_id) + ".out"
                    _append_output(
                        transient_output_files, transient_output_buffers, filename, line
                    )
            elif rec.type_tag == RecorderTypeTag.ElementForce:
                for eidx in range(rec.element_count):
                    var elem_id = recorder_elements_pool[rec.element_offset + eidx]
                    if elem_id >= len(elem_id_to_index) or elem_id_to_index[elem_id] < 0:
                        abort("recorder element not found")
                    var elem_index = elem_id_to_index[elem_id]
                    if not elem_force_cached[elem_index]:
                        var elem = typed_elements[elem_index]
                        elem_force_values[elem_index] = (
                            _element_force_global_for_recorder(
                        elem_index,
                        elem,
                        ndf,
                        u,
                        active_element_load_state.element_loads,
                        active_element_load_state.elem_load_offsets,
                        active_element_load_state.elem_load_pool,
                        1.0,
                        typed_nodes,
                        typed_sections_by_id,
                                fiber_section_defs,
                                fiber_section_cells,
                                fiber_section_index_by_id,
                                fiber_section3d_defs,
                                fiber_section3d_cells,
                                fiber_section3d_index_by_id,
                                uniaxial_defs,
                                uniaxial_state_defs,
                                uniaxial_states,
                                elem_uniaxial_offsets,
                                elem_uniaxial_counts,
                                elem_uniaxial_state_ids,
                                force_basic_offsets,
                                force_basic_counts,
                                force_basic_q,
                            )
                        )
                        elem_force_cached[elem_index] = True
                    var line = _format_values_line(elem_force_values[elem_index])
                    var filename = rec.output + "_ele" + String(elem_id) + ".out"
                    _append_output(
                        transient_output_files, transient_output_buffers, filename, line
                    )
            elif rec.type_tag == RecorderTypeTag.NodeReaction:
                if not record_reactions:
                    abort("internal error: reaction recorder flag mismatch")
                for nidx in range(rec.node_count):
                    var node_id = recorder_nodes_pool[rec.node_offset + nidx]
                    var i = id_to_index[node_id]
                    var values: List[Float64] = []
                    values.resize(rec.dof_count, 0.0)
                    for j in range(rec.dof_count):
                        var dof = recorder_dofs_pool[rec.dof_offset + j]
                        require_dof_in_range(dof, ndf, "recorder")
                        var idx = node_dof_index(i, dof, ndf)
                        values[j] = F_int_reaction[idx] - F_ext_step[idx]
                    var line = _format_values_line(values)
                    var filename = rec.output + "_node" + String(node_id) + ".out"
                    _append_output(
                        transient_output_files, transient_output_buffers, filename, line
                    )
            elif rec.type_tag == RecorderTypeTag.Drift:
                var i_node = rec.i_node
                var j_node = rec.j_node
                var value = _drift_value(rec, typed_nodes, id_to_index, ndf, u)
                var filename = (
                    rec.output
                    + "_i"
                    + String(i_node)
                    + "_j"
                    + String(j_node)
                    + ".out"
                )
                _append_output(
                    transient_output_files,
                    transient_output_buffers,
                    filename,
                    _format_values_line([value]),
                )
            elif rec.type_tag == RecorderTypeTag.EnvelopeElementForce:
                for eidx in range(rec.element_count):
                    var elem_id = recorder_elements_pool[rec.element_offset + eidx]
                    if elem_id >= len(elem_id_to_index) or elem_id_to_index[elem_id] < 0:
                        abort("recorder element not found")
                    var elem_index = elem_id_to_index[elem_id]
                    if not elem_force_cached[elem_index]:
                        var elem = typed_elements[elem_index]
                        elem_force_values[elem_index] = (
                            _element_force_global_for_recorder(
                                elem_index,
                                elem,
                                ndf,
                                u,
                                active_element_load_state.element_loads,
                                active_element_load_state.elem_load_offsets,
                                active_element_load_state.elem_load_pool,
                                1.0,
                                typed_nodes,
                                typed_sections_by_id,
                                fiber_section_defs,
                                fiber_section_cells,
                                fiber_section_index_by_id,
                                fiber_section3d_defs,
                                fiber_section3d_cells,
                                fiber_section3d_index_by_id,
                                uniaxial_defs,
                                uniaxial_state_defs,
                                uniaxial_states,
                                elem_uniaxial_offsets,
                                elem_uniaxial_counts,
                                elem_uniaxial_state_ids,
                                force_basic_offsets,
                                force_basic_counts,
                                force_basic_q,
                            )
                        )
                        elem_force_cached[elem_index] = True
                    var filename = rec.output + "_ele" + String(elem_id) + ".out"
                    _update_envelope(
                        filename,
                        elem_force_values[elem_index],
                        envelope_files,
                        envelope_min,
                        envelope_max,
                        envelope_abs,
                    )
            elif (
                rec.type_tag == RecorderTypeTag.SectionForce
                or rec.type_tag == RecorderTypeTag.SectionDeformation
            ):
                var want_defo = rec.type_tag == RecorderTypeTag.SectionDeformation
                for eidx in range(rec.element_count):
                    var elem_id = recorder_elements_pool[rec.element_offset + eidx]
                    if elem_id >= len(elem_id_to_index) or elem_id_to_index[elem_id] < 0:
                        abort("recorder element not found")
                    var elem_index = elem_id_to_index[elem_id]
                    var elem = typed_elements[elem_index]
                    for sidx in range(rec.section_count):
                        var sec_no = recorder_sections_pool[rec.section_offset + sidx]
                        var values = _section_response_for_recorder(
                            elem_index,
                            elem,
                            sec_no,
                            ndf,
                            u,
                            active_element_load_state.element_loads,
                            active_element_load_state.elem_load_offsets,
                            active_element_load_state.elem_load_pool,
                            1.0,
                            typed_nodes,
                            typed_sections_by_id,
                            fiber_section_defs,
                            fiber_section_cells,
                            fiber_section_index_by_id,
                            fiber_section3d_defs,
                            fiber_section3d_cells,
                            fiber_section3d_index_by_id,
                            uniaxial_defs,
                            uniaxial_states,
                            elem_uniaxial_offsets,
                            elem_uniaxial_counts,
                            elem_uniaxial_state_ids,
                            force_basic_offsets,
                            force_basic_counts,
                            force_basic_q,
                            want_defo,
                        )
                        var filename = (
                            rec.output
                            + "_ele"
                            + String(elem_id)
                            + "_sec"
                            + String(sec_no)
                            + ".out"
                        )
                        _append_output(
                            transient_output_files,
                            transient_output_buffers,
                            filename,
                            _format_values_line(values),
                        )
            else:
                abort("unsupported recorder type")
        if do_profile:
            var t_rec_end = Int(time.perf_counter_ns())
            var rec_end_us = (t_rec_end - t0) // 1000
            _append_event(events, events_need_comma, "C", frame_recorders, rec_end_us)
            _append_event(
                events, events_need_comma, "C", frame_transient_step, rec_end_us
            )
    _flush_envelope_outputs(
        envelope_files,
        envelope_min,
        envelope_max,
        envelope_abs,
        transient_output_files,
        transient_output_buffers,
    )
