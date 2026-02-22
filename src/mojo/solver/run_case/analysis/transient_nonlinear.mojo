from collections import List
from math import sqrt
from materials import (
    UniMaterialDef,
    UniMaterialState,
    uniaxial_commit_all,
)
from os import abort
from python import Python
from sections import FiberCell, FiberSection2dDef

from linalg import gaussian_elimination_into, lu_factorize_into, lu_solve_into
from solver.assembly import (
    assemble_global_stiffness_and_internal,
    assemble_internal_forces_typed,
)
from solver.dof import node_dof_index, require_dof_in_range
from solver.profile import _append_event
from solver.run_case.input_types import (
    AnalysisInput,
    ElementInput,
    MaterialInput,
    NodeInput,
    RecorderInput,
    SectionInput,
)
from solver.time_series import TimeSeriesInput, eval_time_series_input

from solver.run_case.helpers import (
    _append_output,
    _collapse_matrix_by_rep,
    _collapse_vector_by_rep,
    _element_force_global_for_recorder,
    _enforce_equal_dof_values,
    _flush_envelope_outputs,
    _format_values_line,
    _has_recorder_type,
    _drift_value,
    _section_response_for_recorder,
    _update_envelope,
)
from tag_types import NonlinearAlgorithmMode, RecorderTypeTag


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
fn _copy_scaled_vector_simd4(mut dst: List[Float64], src: List[Float64], scale: Float64):
    if len(dst) != len(src):
        abort("vector size mismatch in _copy_scaled_vector_simd4")
    var n = len(dst)
    var i = 0
    var scale_vec = SIMD[DType.float64, 4](scale)
    while i + 3 < n:
        var chunk = SIMD[DType.float64, 4](src[i], src[i + 1], src[i + 2], src[i + 3])
        chunk = chunk * scale_vec
        dst[i] = chunk[0]
        dst[i + 1] = chunk[1]
        dst[i + 2] = chunk[2]
        dst[i + 3] = chunk[3]
        i += 4
    while i < n:
        dst[i] = src[i] * scale
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
fn _sum_sq_simd4(values: List[Float64], count: Int) -> Float64:
    var sum = 0.0
    var i = 0
    while i + 3 < count:
        var v = SIMD[DType.float64, 4](
            values[i], values[i + 1], values[i + 2], values[i + 3]
        )
        sum += (v * v).reduce_add()
        i += 4
    while i < count:
        sum += values[i] * values[i]
        i += 1
    return sum


@always_inline
fn _dot_simd4(lhs: List[Float64], rhs: List[Float64], count: Int) -> Float64:
    var sum = 0.0
    var i = 0
    while i + 3 < count:
        var a = SIMD[DType.float64, 4](lhs[i], lhs[i + 1], lhs[i + 2], lhs[i + 3])
        var b = SIMD[DType.float64, 4](rhs[i], rhs[i + 1], rhs[i + 2], rhs[i + 3])
        sum += (a * b).reduce_add()
        i += 4
    while i < count:
        sum += lhs[i] * rhs[i]
        i += 1
    return sum


@always_inline
fn _gather_free_state_simd4(
    free: List[Int],
    u: List[Float64],
    v: List[Float64],
    a: List[Float64],
    mut u_n: List[Float64],
    mut v_n: List[Float64],
    mut a_n: List[Float64],
):
    var n = len(free)
    var i = 0
    while i + 3 < n:
        var idx0 = free[i]
        var idx1 = free[i + 1]
        var idx2 = free[i + 2]
        var idx3 = free[i + 3]
        var u_vec = SIMD[DType.float64, 4](u[idx0], u[idx1], u[idx2], u[idx3])
        var v_vec = SIMD[DType.float64, 4](v[idx0], v[idx1], v[idx2], v[idx3])
        var a_vec = SIMD[DType.float64, 4](a[idx0], a[idx1], a[idx2], a[idx3])
        u_n[i] = u_vec[0]
        u_n[i + 1] = u_vec[1]
        u_n[i + 2] = u_vec[2]
        u_n[i + 3] = u_vec[3]
        v_n[i] = v_vec[0]
        v_n[i + 1] = v_vec[1]
        v_n[i + 2] = v_vec[2]
        v_n[i + 3] = v_vec[3]
        a_n[i] = a_vec[0]
        a_n[i + 1] = a_vec[1]
        a_n[i + 2] = a_vec[2]
        a_n[i + 3] = a_vec[3]
        i += 4
    while i < n:
        var idx = free[i]
        u_n[i] = u[idx]
        v_n[i] = v[idx]
        a_n[i] = a[idx]
        i += 1


@always_inline
fn _gather_from_free_simd4(free: List[Int], src: List[Float64], mut dst: List[Float64]):
    var n = len(free)
    var i = 0
    while i + 3 < n:
        var idx0 = free[i]
        var idx1 = free[i + 1]
        var idx2 = free[i + 2]
        var idx3 = free[i + 3]
        var v = SIMD[DType.float64, 4](src[idx0], src[idx1], src[idx2], src[idx3])
        dst[i] = v[0]
        dst[i + 1] = v[1]
        dst[i + 2] = v[2]
        dst[i + 3] = v[3]
        i += 4
    while i < n:
        dst[i] = src[free[i]]
        i += 1


@always_inline
fn _build_trial_state_newmark_simd4(
    free: List[Int],
    u: List[Float64],
    u_n: List[Float64],
    v_n: List[Float64],
    a_n: List[Float64],
    a0: Float64,
    a2: Float64,
    a3: Float64,
    gamma: Float64,
    dt: Float64,
    mut a_trial: List[Float64],
    mut v_trial: List[Float64],
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
        var u_curr = SIMD[DType.float64, 4](u[idx0], u[idx1], u[idx2], u[idx3])
        var u_prev = SIMD[DType.float64, 4](u_n[i], u_n[i + 1], u_n[i + 2], u_n[i + 3])
        var v_prev = SIMD[DType.float64, 4](v_n[i], v_n[i + 1], v_n[i + 2], v_n[i + 3])
        var a_prev = SIMD[DType.float64, 4](a_n[i], a_n[i + 1], a_n[i + 2], a_n[i + 3])
        var a_next = a0_vec * (u_curr - u_prev) - a2_vec * v_prev - a3_vec * a_prev
        var v_next = v_prev + dt_vec * (one_minus_gamma_vec * a_prev + gamma_vec * a_next)
        a_trial[i] = a_next[0]
        a_trial[i + 1] = a_next[1]
        a_trial[i + 2] = a_next[2]
        a_trial[i + 3] = a_next[3]
        v_trial[i] = v_next[0]
        v_trial[i + 1] = v_next[1]
        v_trial[i + 2] = v_next[2]
        v_trial[i + 3] = v_next[3]
        i += 4
    while i < n:
        var idx = free[i]
        a_trial[i] = a0 * (u[idx] - u_n[i]) - a2 * v_n[i] - a3 * a_n[i]
        v_trial[i] = v_n[i] + dt * ((1.0 - gamma) * a_n[i] + gamma * a_trial[i])
        i += 1


@always_inline
fn _scatter_add_from_free_simd4(free: List[Int], mut dst: List[Float64], add: List[Float64]):
    var n = len(free)
    var i = 0
    while i + 3 < n:
        var idx0 = free[i]
        var idx1 = free[i + 1]
        var idx2 = free[i + 2]
        var idx3 = free[i + 3]
        var dst_vec = SIMD[DType.float64, 4](dst[idx0], dst[idx1], dst[idx2], dst[idx3])
        var add_vec = SIMD[DType.float64, 4](add[i], add[i + 1], add[i + 2], add[i + 3])
        var out = dst_vec + add_vec
        dst[idx0] = out[0]
        dst[idx1] = out[1]
        dst[idx2] = out[2]
        dst[idx3] = out[3]
        i += 4
    while i < n:
        dst[free[i]] += add[i]
        i += 1


@always_inline
fn _update_post_step_newmark_simd4(
    free: List[Int],
    u: List[Float64],
    u_n: List[Float64],
    v_n: List[Float64],
    a_n: List[Float64],
    a0: Float64,
    a2: Float64,
    a3: Float64,
    gamma: Float64,
    dt: Float64,
    mut v: List[Float64],
    mut a: List[Float64],
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
        var u_curr = SIMD[DType.float64, 4](u[idx0], u[idx1], u[idx2], u[idx3])
        var u_prev = SIMD[DType.float64, 4](u_n[i], u_n[i + 1], u_n[i + 2], u_n[i + 3])
        var v_prev = SIMD[DType.float64, 4](v_n[i], v_n[i + 1], v_n[i + 2], v_n[i + 3])
        var a_prev = SIMD[DType.float64, 4](a_n[i], a_n[i + 1], a_n[i + 2], a_n[i + 3])
        var a_next = a0_vec * (u_curr - u_prev) - a2_vec * v_prev - a3_vec * a_prev
        var v_next = v_prev + dt_vec * (one_minus_gamma_vec * a_prev + gamma_vec * a_next)
        a[idx0] = a_next[0]
        a[idx1] = a_next[1]
        a[idx2] = a_next[2]
        a[idx3] = a_next[3]
        v[idx0] = v_next[0]
        v[idx1] = v_next[1]
        v[idx2] = v_next[2]
        v[idx3] = v_next[3]
        i += 4
    while i < n:
        var idx = free[i]
        var a_next = a0 * (u[idx] - u_n[i]) - a2 * v_n[i] - a3 * a_n[i]
        var v_next = v_n[i] + dt * ((1.0 - gamma) * a_n[i] + gamma * a_next)
        a[idx] = a_next
        v[idx] = v_next
        i += 1


fn run_transient_nonlinear(
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
    elem_dof_offsets: List[Int],
    elem_dof_pool: List[Int],
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
    mut F_total: List[Float64],
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
    frame_solve_nonlinear: Int,
    frame_nonlinear_step: Int,
    frame_nonlinear_iter: Int,
    frame_time_series_eval: Int,
    frame_constraints: Int,
    frame_recorders: Int,
    frame_factorize: Int,
    frame_transient_step: Int,
) raises:
    var time = Python.import_module("time")
    var asm_dof_map6: List[Int] = []
    var asm_dof_map12: List[Int] = []
    var asm_u_elem6: List[Float64] = []

    var dt = analysis.dt
    if dt <= 0.0:
        abort("transient_nonlinear requires dt > 0")
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

    var algorithm = analysis.algorithm
    var primary_algorithm_mode = NonlinearAlgorithmMode.Unknown
    if algorithm == "Newton":
        primary_algorithm_mode = NonlinearAlgorithmMode.Newton
    elif algorithm == "ModifiedNewton":
        primary_algorithm_mode = NonlinearAlgorithmMode.ModifiedNewton
    elif algorithm == "ModifiedNewtonInitial":
        primary_algorithm_mode = NonlinearAlgorithmMode.ModifiedNewtonInitial
    elif algorithm == "Broyden" or algorithm == "NewtonLineSearch":
        # Mapped alternative: currently follows Newton tangent refresh behavior.
        primary_algorithm_mode = NonlinearAlgorithmMode.Newton
    else:
        abort("unsupported transient_nonlinear algorithm: " + algorithm)

    var fallback_algorithm = analysis.fallback_algorithm
    var has_fallback = False
    var fallback_algorithm_mode = NonlinearAlgorithmMode.Unknown
    if len(fallback_algorithm) > 0:
        if fallback_algorithm == "Newton":
            fallback_algorithm_mode = NonlinearAlgorithmMode.Newton
        elif fallback_algorithm == "ModifiedNewton":
            fallback_algorithm_mode = NonlinearAlgorithmMode.ModifiedNewton
        elif fallback_algorithm == "ModifiedNewtonInitial":
            fallback_algorithm_mode = NonlinearAlgorithmMode.ModifiedNewtonInitial
        elif fallback_algorithm == "Broyden" or fallback_algorithm == "NewtonLineSearch":
            # Mapped alternative: currently follows Newton tangent refresh behavior.
            fallback_algorithm_mode = NonlinearAlgorithmMode.Newton
        else:
            abort(
                "unsupported transient_nonlinear fallback_algorithm: "
                + fallback_algorithm
            )
        has_fallback = True
    elif primary_algorithm_mode != NonlinearAlgorithmMode.Newton:
        # Preserve legacy fallback behavior for ModifiedNewton* primary modes.
        fallback_algorithm_mode = NonlinearAlgorithmMode.Newton
        has_fallback = True

    var integrator_type = analysis.integrator_type
    if integrator_type == "":
        integrator_type = "Newmark"
    if integrator_type != "Newmark":
        abort("transient_nonlinear only supports Newmark integrator")
    var gamma = analysis.integrator_gamma
    var beta = analysis.integrator_beta
    if beta <= 0.0:
        abort("Newmark beta must be > 0")

    var primary_test_mode = -1
    var test_type = analysis.test_type
    if test_type == "MaxDispIncr":
        primary_test_mode = 0
    elif test_type == "NormDispIncr":
        primary_test_mode = 1
    elif test_type == "NormUnbalance":
        primary_test_mode = 2
    elif test_type == "EnergyIncr":
        primary_test_mode = 3
    else:
        abort("unsupported transient_nonlinear test_type: " + test_type)
    var fallback_test_mode = -1
    var fallback_test_type = analysis.fallback_test_type
    if fallback_test_type == "MaxDispIncr":
        fallback_test_mode = 0
    elif fallback_test_type == "NormDispIncr":
        fallback_test_mode = 1
    elif fallback_test_type == "NormUnbalance":
        fallback_test_mode = 2
    elif fallback_test_type == "EnergyIncr":
        fallback_test_mode = 3
    else:
        abort(
            "unsupported transient_nonlinear fallback_test_type: "
            + fallback_test_type
        )

    var max_iters = analysis.max_iters
    var tol = analysis.tol
    var rel_tol = analysis.rel_tol
    var fallback_max_iters = analysis.fallback_max_iters
    var fallback_tol = analysis.fallback_tol
    var fallback_rel_tol = analysis.fallback_rel_tol
    if max_iters < 1:
        abort("transient_nonlinear max_iters must be >= 1")
    if fallback_max_iters < 1:
        abort("transient_nonlinear fallback_max_iters must be >= 1")

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
        abort("transient_nonlinear requires masses on free dofs")

    var a0 = 1.0 / (beta * dt * dt)
    var a1 = gamma / (beta * dt)
    var a2 = 1.0 / (beta * dt)
    var a3 = 1.0 / (2.0 * beta) - 1.0
    var need_comm_tangent = rayleigh_beta_k_comm != 0.0

    var K: List[List[Float64]] = []
    for _ in range(total_dofs):
        var row: List[Float64] = []
        row.resize(total_dofs, 0.0)
        K.append(row^)
    var F_int: List[Float64] = []
    F_int.resize(total_dofs, 0.0)

    var K_ff: List[List[Float64]] = []
    var K_init_ff: List[List[Float64]] = []
    var K_comm_ff: List[List[Float64]] = []
    var C_ff: List[List[Float64]] = []
    var K_eff: List[List[Float64]] = []
    var K_step: List[List[Float64]] = []
    var K_lu: List[List[Float64]] = []
    for _ in range(free_count):
        var row_kff: List[Float64] = []
        row_kff.resize(free_count, 0.0)
        K_ff.append(row_kff^)
        var row_kinit: List[Float64] = []
        row_kinit.resize(free_count, 0.0)
        K_init_ff.append(row_kinit^)
        var row_kcomm: List[Float64] = []
        row_kcomm.resize(free_count, 0.0)
        K_comm_ff.append(row_kcomm^)
        var row_c: List[Float64] = []
        row_c.resize(free_count, 0.0)
        C_ff.append(row_c^)
        var row_keff: List[Float64] = []
        row_keff.resize(free_count, 0.0)
        K_eff.append(row_keff^)
        var row_step: List[Float64] = []
        row_step.resize(free_count, 0.0)
        K_step.append(row_step^)
        var row_lu: List[Float64] = []
        row_lu.resize(free_count, 0.0)
        K_lu.append(row_lu^)

    if do_profile:
        var t_asm_start = Int(time.perf_counter_ns())
        var asm_start_us = (t_asm_start - t0) // 1000
        _append_event(
            events, events_need_comma, "O", frame_assemble_stiffness, asm_start_us
        )
    assemble_global_stiffness_and_internal(
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
        force_basic_offsets,
        force_basic_counts,
        force_basic_q,
        fiber_section_defs,
        fiber_section_cells,
        fiber_section_index_by_id,
        elem_dof_offsets,
        elem_dof_pool,
        asm_dof_map6,
        asm_dof_map12,
        asm_u_elem6,
        K,
        F_int,
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
    for i in range(free_count):
        for j in range(free_count):
            K_init_ff[i][j] = K[free[i]][free[j]]
            K_comm_ff[i][j] = K_init_ff[i][j]

    var v: List[Float64] = []
    v.resize(total_dofs, 0.0)
    var a: List[Float64] = []
    a.resize(total_dofs, 0.0)

    var u_n: List[Float64] = []
    u_n.resize(free_count, 0.0)
    var v_n: List[Float64] = []
    v_n.resize(free_count, 0.0)
    var a_n: List[Float64] = []
    a_n.resize(free_count, 0.0)
    var v_trial: List[Float64] = []
    v_trial.resize(free_count, 0.0)
    var a_trial: List[Float64] = []
    a_trial.resize(free_count, 0.0)

    var F_ext_step: List[Float64] = []
    F_ext_step.resize(total_dofs, 0.0)
    var P_ext_f: List[Float64] = []
    P_ext_f.resize(free_count, 0.0)
    var R_f: List[Float64] = []
    R_f.resize(free_count, 0.0)
    var R_step: List[Float64] = []
    R_step.resize(free_count, 0.0)
    var du_f: List[Float64] = []
    du_f.resize(free_count, 0.0)
    var lu_pivots: List[Int] = []
    lu_pivots.resize(free_count, 0)
    var lu_rhs: List[Float64] = []
    lu_rhs.resize(free_count, 0.0)
    var lu_work: List[Float64] = []
    lu_work.resize(free_count, 0.0)

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
            _append_event(
                events, events_need_comma, "O", frame_nonlinear_step, step_start_us
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

        _gather_free_state_simd4(free, u, v, a, u_n, v_n, a_n)

        var t = Float64(step + 1) * dt
        if pattern_type == "UniformExcitation":
            _copy_vector_simd4(F_ext_step, F_total)
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
                _copy_scaled_vector_simd4(F_ext_step, F_total, factor)
            else:
                _copy_vector_simd4(F_ext_step, F_total)
        _gather_from_free_simd4(free, F_ext_step, P_ext_f)

        var converged = False
        var attempt_algorithm_mode = primary_algorithm_mode
        var attempt_test_mode = primary_test_mode
        var attempt_max_iters = max_iters
        var attempt_tol = tol
        var attempt_rel_tol = rel_tol
        for attempt in range(2):
            if attempt == 1 and not has_fallback:
                break
            if attempt == 1:
                for i in range(free_count):
                    u[free[i]] = u_n[i]
                if has_transformation_mpc:
                    _enforce_equal_dof_values(u, rep_dof, constrained)
                attempt_algorithm_mode = fallback_algorithm_mode
                attempt_test_mode = fallback_test_mode
                attempt_max_iters = fallback_max_iters
                attempt_tol = fallback_tol
                attempt_rel_tol = fallback_rel_tol

            var tangent_initialized = False
            var damping_initialized = False
            var k_eff_initialized = False
            var k_eff_factored = False
            for _ in range(attempt_max_iters):
                var iter_closed = False
                if do_profile:
                    var t_iter_start = Int(time.perf_counter_ns())
                    var iter_start_us = (t_iter_start - t0) // 1000
                    _append_event(
                        events,
                        events_need_comma,
                        "O",
                        frame_nonlinear_iter,
                        iter_start_us,
                    )
                if do_profile:
                    var t_asm_iter_start = Int(time.perf_counter_ns())
                    var asm_start_us = (t_asm_iter_start - t0) // 1000
                    _append_event(
                        events,
                        events_need_comma,
                        "O",
                        frame_assemble_stiffness,
                        asm_start_us,
                    )
                assemble_global_stiffness_and_internal(
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
                    force_basic_offsets,
                    force_basic_counts,
                    force_basic_q,
                    fiber_section_defs,
                    fiber_section_cells,
                    fiber_section_index_by_id,
                    elem_dof_offsets,
                    elem_dof_pool,
                    asm_dof_map6,
                    asm_dof_map12,
                    asm_u_elem6,
                    K,
                    F_int,
                )
                if do_profile:
                    var t_asm_iter_end = Int(time.perf_counter_ns())
                    var asm_end_us = (t_asm_iter_end - t0) // 1000
                    _append_event(
                        events,
                        events_need_comma,
                        "C",
                        frame_assemble_stiffness,
                        asm_end_us,
                    )
                var need_tangent_matrix = (
                    attempt_algorithm_mode == NonlinearAlgorithmMode.Newton
                    or (
                        attempt_algorithm_mode
                        == NonlinearAlgorithmMode.ModifiedNewton
                        and not tangent_initialized
                    )
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
                    if need_tangent_matrix:
                        K = _collapse_matrix_by_rep(K, rep_dof)
                    F_int = _collapse_vector_by_rep(F_int, rep_dof)
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
                if attempt_algorithm_mode == NonlinearAlgorithmMode.Newton:
                    for i in range(free_count):
                        for j in range(free_count):
                            K_ff[i][j] = K[free[i]][free[j]]
                elif attempt_algorithm_mode == NonlinearAlgorithmMode.ModifiedNewton:
                    if not tangent_initialized:
                        for i in range(free_count):
                            for j in range(free_count):
                                K_ff[i][j] = K[free[i]][free[j]]
                        tangent_initialized = True
                else:
                    if not tangent_initialized:
                        for i in range(free_count):
                            for j in range(free_count):
                                K_ff[i][j] = K_init_ff[i][j]
                        tangent_initialized = True

                if (
                    attempt_algorithm_mode == NonlinearAlgorithmMode.Newton
                    or not damping_initialized
                ):
                    for i in range(free_count):
                        for j in range(free_count):
                            C_ff[i][j] = 0.0
                        C_ff[i][i] = rayleigh_alpha_m * M_f[i]
                    if rayleigh_beta_k != 0.0:
                        for i in range(free_count):
                            for j in range(free_count):
                                C_ff[i][j] += rayleigh_beta_k * K_ff[i][j]
                    if rayleigh_beta_k_init != 0.0:
                        for i in range(free_count):
                            for j in range(free_count):
                                C_ff[i][j] += rayleigh_beta_k_init * K_init_ff[i][j]
                    if rayleigh_beta_k_comm != 0.0:
                        for i in range(free_count):
                            for j in range(free_count):
                                C_ff[i][j] += rayleigh_beta_k_comm * K_comm_ff[i][j]
                    damping_initialized = True

                _build_trial_state_newmark_simd4(
                    free, u, u_n, v_n, a_n, a0, a2, a3, gamma, dt, a_trial, v_trial
                )

                for i in range(free_count):
                    var damping_force = _dot_row_simd4(C_ff[i], v_trial, free_count)
                    R_f[i] = (
                        P_ext_f[i]
                        - F_int[free[i]]
                        - damping_force
                        - M_f[i] * a_trial[i]
                    )
                if attempt_test_mode == 2:
                    var residual_norm = sqrt(_sum_sq_simd4(R_f, free_count))
                    if residual_norm <= attempt_tol:
                        if do_profile and not iter_closed:
                            var t_iter_end = Int(time.perf_counter_ns())
                            var iter_end_us = (t_iter_end - t0) // 1000
                            _append_event(
                                events,
                                events_need_comma,
                                "C",
                                frame_nonlinear_iter,
                                iter_end_us,
                            )
                        converged = True
                        break

                if (
                    attempt_algorithm_mode == NonlinearAlgorithmMode.Newton
                    or not k_eff_initialized
                ):
                    for i in range(free_count):
                        for j in range(free_count):
                            K_eff[i][j] = K_ff[i][j] + a1 * C_ff[i][j]
                        K_eff[i][i] += a0 * M_f[i]
                    k_eff_initialized = True

                if attempt_algorithm_mode == NonlinearAlgorithmMode.Newton:
                    for i in range(free_count):
                        for j in range(free_count):
                            K_step[i][j] = K_eff[i][j]
                    for i in range(free_count):
                        R_step[i] = R_f[i]
                    if do_profile:
                        var t_solve_start = Int(time.perf_counter_ns())
                        var solve_start_us = (t_solve_start - t0) // 1000
                        _append_event(
                            events,
                            events_need_comma,
                            "O",
                            frame_solve_nonlinear,
                            solve_start_us,
                        )
                        _append_event(
                            events,
                            events_need_comma,
                            "O",
                            frame_factorize,
                            solve_start_us,
                        )
                    gaussian_elimination_into(K_step, R_step, du_f)
                    if do_profile:
                        var t_solve_end = Int(time.perf_counter_ns())
                        var solve_end_us = (t_solve_end - t0) // 1000
                        _append_event(
                            events,
                            events_need_comma,
                            "C",
                            frame_factorize,
                            solve_end_us,
                        )
                        _append_event(
                            events,
                            events_need_comma,
                            "C",
                            frame_solve_nonlinear,
                            solve_end_us,
                        )
                else:
                    if not k_eff_factored:
                        for i in range(free_count):
                            for j in range(free_count):
                                K_lu[i][j] = K_eff[i][j]
                        if do_profile:
                            var t_fac_start = Int(time.perf_counter_ns())
                            var fac_start_us = (t_fac_start - t0) // 1000
                            _append_event(
                                events,
                                events_need_comma,
                                "O",
                                frame_factorize,
                                fac_start_us,
                            )
                        lu_factorize_into(K_lu, lu_pivots)
                        if do_profile:
                            var t_fac_end = Int(time.perf_counter_ns())
                            var fac_end_us = (t_fac_end - t0) // 1000
                            _append_event(
                                events,
                                events_need_comma,
                                "C",
                                frame_factorize,
                                fac_end_us,
                            )
                        k_eff_factored = True
                    for i in range(free_count):
                        lu_rhs[i] = R_f[i]
                    if do_profile:
                        var t_solve_start = Int(time.perf_counter_ns())
                        var solve_start_us = (t_solve_start - t0) // 1000
                        _append_event(
                            events,
                            events_need_comma,
                            "O",
                            frame_solve_nonlinear,
                            solve_start_us,
                        )
                    lu_solve_into(K_lu, lu_pivots, lu_rhs, lu_work, du_f)
                    if do_profile:
                        var t_solve_end = Int(time.perf_counter_ns())
                        var solve_end_us = (t_solve_end - t0) // 1000
                        _append_event(
                            events,
                            events_need_comma,
                            "C",
                            frame_solve_nonlinear,
                            solve_end_us,
                        )

                var max_diff = 0.0
                var max_u = 0.0
                for i in range(free_count):
                    var idx = free[i]
                    var du = du_f[i]
                    var value = u[idx] + du
                    var diff = abs(du)
                    if diff > max_diff:
                        max_diff = diff
                    var abs_val = abs(value)
                    if abs_val > max_u:
                        max_u = abs_val
                var disp_incr_norm = sqrt(_sum_sq_simd4(du_f, free_count))
                var energy_incr = abs(_dot_simd4(du_f, R_f, free_count))
                var scale_tol = attempt_rel_tol * max_u
                if scale_tol < attempt_rel_tol:
                    scale_tol = attempt_rel_tol

                _scatter_add_from_free_simd4(free, u, du_f)
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

                if attempt_test_mode == 1:
                    if disp_incr_norm <= attempt_tol:
                        if do_profile and not iter_closed:
                            var t_iter_end = Int(time.perf_counter_ns())
                            var iter_end_us = (t_iter_end - t0) // 1000
                            _append_event(
                                events,
                                events_need_comma,
                                "C",
                                frame_nonlinear_iter,
                                iter_end_us,
                            )
                        converged = True
                        break
                elif attempt_test_mode == 3:
                    if energy_incr <= attempt_tol:
                        if do_profile and not iter_closed:
                            var t_iter_end = Int(time.perf_counter_ns())
                            var iter_end_us = (t_iter_end - t0) // 1000
                            _append_event(
                                events,
                                events_need_comma,
                                "C",
                                frame_nonlinear_iter,
                                iter_end_us,
                            )
                        converged = True
                        break
                elif attempt_test_mode == 0:
                    if max_diff <= attempt_tol or max_diff <= scale_tol:
                        if do_profile and not iter_closed:
                            var t_iter_end = Int(time.perf_counter_ns())
                            var iter_end_us = (t_iter_end - t0) // 1000
                            _append_event(
                                events,
                                events_need_comma,
                                "C",
                                frame_nonlinear_iter,
                                iter_end_us,
                            )
                        converged = True
                        break
                if do_profile and not iter_closed:
                    var t_iter_end = Int(time.perf_counter_ns())
                    var iter_end_us = (t_iter_end - t0) // 1000
                    _append_event(
                        events, events_need_comma, "C", frame_nonlinear_iter, iter_end_us
                    )
            if converged:
                break

        if not converged:
            abort("transient_nonlinear did not converge at step " + String(step + 1))

        if need_comm_tangent:
            if do_profile:
                var t_asm_post_start = Int(time.perf_counter_ns())
                var asm_start_us = (t_asm_post_start - t0) // 1000
                _append_event(
                    events,
                    events_need_comma,
                    "O",
                    frame_assemble_stiffness,
                    asm_start_us,
                )
            assemble_global_stiffness_and_internal(
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
                force_basic_offsets,
                force_basic_counts,
                force_basic_q,
                fiber_section_defs,
                fiber_section_cells,
                fiber_section_index_by_id,
                elem_dof_offsets,
                elem_dof_pool,
                asm_dof_map6,
                asm_dof_map12,
                asm_u_elem6,
                K,
                F_int,
            )
            if do_profile:
                var t_asm_post_end = Int(time.perf_counter_ns())
                var asm_end_us = (t_asm_post_end - t0) // 1000
                _append_event(
                    events,
                    events_need_comma,
                    "C",
                    frame_assemble_stiffness,
                    asm_end_us,
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
            for i in range(free_count):
                for j in range(free_count):
                    K_comm_ff[i][j] = K[free[i]][free[j]]
        uniaxial_commit_all(uniaxial_states)

        _update_post_step_newmark_simd4(
            free, u, u_n, v_n, a_n, a0, a2, a3, gamma, dt, v, a
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

        if do_profile:
            var t_rec_start = Int(time.perf_counter_ns())
            var rec_start_us = (t_rec_start - t0) // 1000
            _append_event(events, events_need_comma, "O", frame_recorders, rec_start_us)
        var F_int_reaction: List[Float64] = []
        if record_reactions:
            F_int_reaction = assemble_internal_forces_typed(
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
                force_basic_offsets,
                force_basic_counts,
                force_basic_q,
                fiber_section_defs,
                fiber_section_cells,
                fiber_section_index_by_id,
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
                        elem_force_values[elem_index] = _element_force_global_for_recorder(
                            elem_index,
                            elem,
                            ndf,
                            u,
                            typed_nodes,
                            typed_sections_by_id,
                            fiber_section_defs,
                            fiber_section_cells,
                            fiber_section_index_by_id,
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
                        elem_force_values[elem_index] = _element_force_global_for_recorder(
                            elem_index,
                            elem,
                            ndf,
                            u,
                            typed_nodes,
                            typed_sections_by_id,
                            fiber_section_defs,
                            fiber_section_cells,
                            fiber_section_index_by_id,
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
                            typed_nodes,
                            typed_sections_by_id,
                            fiber_section_defs,
                            fiber_section_cells,
                            fiber_section_index_by_id,
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
                events, events_need_comma, "C", frame_nonlinear_step, rec_end_us
            )
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
