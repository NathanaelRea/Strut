from collections import List
from elements import (
    ForceBeamColumn2dScratch,
    ForceBeamColumn3dScratch,
    invalidate_force_beam_column2d_load_cache,
    invalidate_force_beam_column3d_load_cache,
    reset_force_beam_column2d_scratch,
    reset_force_beam_column3d_scratch,
)
from math import sqrt
from materials import (
    UniMaterialDef,
    UniMaterialState,
    uniaxial_commit_all,
    uniaxial_revert_trial_all,
)
from os import abort
from python import Python
from sections import FiberCell, FiberSection2dDef, FiberSection3dDef
from sys import simd_width_of

from linalg import gaussian_elimination_into, lu_factorize_into, lu_solve_into
from solver.assembly import (
    assemble_global_stiffness_and_internal_soa,
    assemble_link_stiffness_typed,
    assemble_internal_forces_typed_soa,
    assemble_zero_length_damping_trial_typed,
    assemble_zero_length_damping_typed,
)
from solver.dof import node_dof_index, require_dof_in_range
from solver.profile import _append_event
from solver.simd_contiguous import (
    copy_float64_contiguous,
    dot_float64_contiguous,
    load_float64_contiguous_simd,
    store_float64_contiguous_simd,
    sum_sq_float64_contiguous,
)
from solver.simd_indexed import (
    gather_float64_by_index_simd,
    scatter_float64_by_index_simd,
)
from solver.run_case.input_types import (
    AnalysisInput,
    DampingInput,
    ElementLoadInput,
    ElementInput,
    MaterialInput,
    NodeInput,
    RecorderInput,
    SectionInput,
    SolverAttemptInput,
)
from solver.run_case.load_state import (
    build_active_element_load_state,
    build_active_nodal_load,
)
from solver.time_series import TimeSeriesInput, eval_time_series_input, find_time_series_input

from solver.run_case.helpers import (
    _append_output,
    _element_basic_force_for_recorder,
    _element_deformation_for_recorder,
    _collapse_matrix_by_rep,
    _collapse_vector_by_rep,
    _element_force_global_for_recorder,
    _force_beam_column2d_force_global_from_basic_state,
    _force_beam_column2d_section_response_from_basic_state,
    _element_local_force_for_recorder,
    _enforce_equal_dof_values,
    _flush_envelope_outputs,
    _format_values_line,
    _has_recorder_type,
    _drift_value,
    _section_response_for_recorder,
    _sync_force_beam_column2d_committed_basic_states,
    _update_envelope,
)
from tag_types import (
    AnalysisAlgorithmTag,
    ElementTypeTag,
    IntegratorTypeTag,
    NonlinearAlgorithmMode,
    NonlinearTestTypeTag,
    PatternTypeTag,
    RecorderTypeTag,
)


@always_inline
fn _has_nonparticipating_link_for_rayleigh(elements: List[ElementInput]) -> Bool:
    for e in range(len(elements)):
        var elem = elements[e]
        if (
            elem.type_tag == ElementTypeTag.ZeroLength
            or elem.type_tag == ElementTypeTag.TwoNodeLink
        ) and not elem.do_rayleigh:
            return True
    return False


fn _has_zero_length_damp_mats(elements: List[ElementInput]) -> Bool:
    for i in range(len(elements)):
        var elem = elements[i]
        if (
            elem.type_tag == ElementTypeTag.ZeroLength
            and elem.damp_material_count > 0
        ):
            return True
    return False


fn _has_zero_length_dampers(elements: List[ElementInput]) -> Bool:
    for i in range(len(elements)):
        var elem = elements[i]
        if elem.type_tag == ElementTypeTag.ZeroLength and elem.damping_tag >= 0:
            return True
    return False


@always_inline
fn _scatter_free_vector_to_full_simd_impl[width: Int](
    free: List[Int], src: List[Float64], mut dst: List[Float64]
):
    var n = len(free)
    var i = 0
    while i + width <= n:
        scatter_float64_by_index_simd[width](
            free, i, dst, load_float64_contiguous_simd[width](src, i)
        )
        i += width
    while i < n:
        dst[free[i]] = src[i]
        i += 1


@always_inline
fn _scatter_free_vector_to_full(
    free: List[Int], src: List[Float64], mut dst: List[Float64]
):
    _scatter_free_vector_to_full_simd_impl[simd_width_of[DType.float64]()](
        free, src, dst
    )


@always_inline
fn _gather_from_free_simd_impl[width: Int](
    free: List[Int], src: List[Float64], mut dst: List[Float64]
):
    var n = len(free)
    var i = 0
    while i + width <= n:
        store_float64_contiguous_simd[width](
            dst, i, gather_float64_by_index_simd[width](free, i, src)
        )
        i += width
    while i < n:
        dst[i] = src[free[i]]
        i += 1


@always_inline
fn _gather_from_free_simd(free: List[Int], src: List[Float64], mut dst: List[Float64]):
    _gather_from_free_simd_impl[simd_width_of[DType.float64]()](
        free, src, dst
    )


@always_inline
fn _build_trial_state_newmark_simd_impl[width: Int](
    u_f: List[Float64],
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
    var n = len(u_f)
    var i = 0
    var a0_vec = SIMD[DType.float64, width](a0)
    var a2_vec = SIMD[DType.float64, width](a2)
    var a3_vec = SIMD[DType.float64, width](a3)
    var gamma_vec = SIMD[DType.float64, width](gamma)
    var one_minus_gamma_vec = SIMD[DType.float64, width](1.0 - gamma)
    var dt_vec = SIMD[DType.float64, width](dt)
    while i + width <= n:
        var u_curr = load_float64_contiguous_simd[width](u_f, i)
        var u_prev = load_float64_contiguous_simd[width](u_n, i)
        var v_prev = load_float64_contiguous_simd[width](v_n, i)
        var a_prev = load_float64_contiguous_simd[width](a_n, i)
        var a_next = a0_vec * (u_curr - u_prev) - a2_vec * v_prev - a3_vec * a_prev
        var v_next = v_prev + dt_vec * (one_minus_gamma_vec * a_prev + gamma_vec * a_next)
        store_float64_contiguous_simd[width](a_trial, i, a_next)
        store_float64_contiguous_simd[width](v_trial, i, v_next)
        i += width
    while i < n:
        a_trial[i] = a0 * (u_f[i] - u_n[i]) - a2 * v_n[i] - a3 * a_n[i]
        v_trial[i] = v_n[i] + dt * ((1.0 - gamma) * a_n[i] + gamma * a_trial[i])
        i += 1


@always_inline
fn _build_trial_state_newmark_simd(
    u_f: List[Float64],
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
    _build_trial_state_newmark_simd_impl[simd_width_of[DType.float64]()](
        u_f, u_n, v_n, a_n, a0, a2, a3, gamma, dt, a_trial, v_trial
    )


@always_inline
fn _predict_displacement_newmark_simd_impl[width: Int](
    mut u_f: List[Float64],
    u_n: List[Float64],
    v_n: List[Float64],
    a_n: List[Float64],
    dt: Float64,
    beta: Float64,
):
    var n = len(u_f)
    var i = 0
    var dt_vec = SIMD[DType.float64, width](dt)
    var coeff_vec = SIMD[DType.float64, width](dt * dt * (0.5 - beta))
    while i + width <= n:
        var u_prev = load_float64_contiguous_simd[width](u_n, i)
        var v_prev = load_float64_contiguous_simd[width](v_n, i)
        var a_prev = load_float64_contiguous_simd[width](a_n, i)
        var u_pred = u_prev + dt_vec * v_prev + coeff_vec * a_prev
        store_float64_contiguous_simd[width](u_f, i, u_pred)
        i += width
    var coeff = dt * dt * (0.5 - beta)
    while i < n:
        u_f[i] = u_n[i] + dt * v_n[i] + coeff * a_n[i]
        i += 1


@always_inline
fn _predict_displacement_newmark_simd(
    mut u_f: List[Float64],
    u_n: List[Float64],
    v_n: List[Float64],
    a_n: List[Float64],
    dt: Float64,
    beta: Float64,
):
    _predict_displacement_newmark_simd_impl[simd_width_of[DType.float64]()](
        u_f, u_n, v_n, a_n, dt, beta
    )


@always_inline
fn _update_post_step_newmark_simd_impl[width: Int](
    u_f: List[Float64],
    u_n: List[Float64],
    v_n: List[Float64],
    a_n: List[Float64],
    a0: Float64,
    a2: Float64,
    a3: Float64,
    gamma: Float64,
    dt: Float64,
    mut v_f: List[Float64],
    mut a_f: List[Float64],
):
    var n = len(u_f)
    var i = 0
    var a0_vec = SIMD[DType.float64, width](a0)
    var a2_vec = SIMD[DType.float64, width](a2)
    var a3_vec = SIMD[DType.float64, width](a3)
    var gamma_vec = SIMD[DType.float64, width](gamma)
    var one_minus_gamma_vec = SIMD[DType.float64, width](1.0 - gamma)
    var dt_vec = SIMD[DType.float64, width](dt)
    while i + width <= n:
        var u_curr = load_float64_contiguous_simd[width](u_f, i)
        var u_prev = load_float64_contiguous_simd[width](u_n, i)
        var v_prev = load_float64_contiguous_simd[width](v_n, i)
        var a_prev = load_float64_contiguous_simd[width](a_n, i)
        var a_next = a0_vec * (u_curr - u_prev) - a2_vec * v_prev - a3_vec * a_prev
        var v_next = v_prev + dt_vec * (one_minus_gamma_vec * a_prev + gamma_vec * a_next)
        store_float64_contiguous_simd[width](a_f, i, a_next)
        store_float64_contiguous_simd[width](v_f, i, v_next)
        i += width
    while i < n:
        var a_next = a0 * (u_f[i] - u_n[i]) - a2 * v_n[i] - a3 * a_n[i]
        var v_next = v_n[i] + dt * ((1.0 - gamma) * a_n[i] + gamma * a_next)
        a_f[i] = a_next
        v_f[i] = v_next
        i += 1


@always_inline
fn _update_post_step_newmark_simd(
    u_f: List[Float64],
    u_n: List[Float64],
    v_n: List[Float64],
    a_n: List[Float64],
    a0: Float64,
    a2: Float64,
    a3: Float64,
    gamma: Float64,
    dt: Float64,
    mut v_f: List[Float64],
    mut a_f: List[Float64],
):
    _update_post_step_newmark_simd_impl[simd_width_of[DType.float64]()](
        u_f, u_n, v_n, a_n, a0, a2, a3, gamma, dt, v_f, a_f
    )


fn _transient_nonlinear_algorithm_mode(
    algorithm_tag: Int, algorithm: String, label: String
) -> Int:
    if algorithm_tag == AnalysisAlgorithmTag.Newton:
        return NonlinearAlgorithmMode.Newton
    if algorithm_tag == AnalysisAlgorithmTag.ModifiedNewton:
        return NonlinearAlgorithmMode.ModifiedNewton
    if algorithm_tag == AnalysisAlgorithmTag.ModifiedNewtonInitial:
        return NonlinearAlgorithmMode.ModifiedNewtonInitial
    if (
        algorithm_tag == AnalysisAlgorithmTag.Broyden
        or algorithm_tag == AnalysisAlgorithmTag.NewtonLineSearch
    ):
        # Mapped alternative: currently follows Newton tangent refresh behavior.
        return NonlinearAlgorithmMode.Newton
    abort("unsupported " + label + " algorithm: " + algorithm)
    return NonlinearAlgorithmMode.Unknown


fn _transient_nonlinear_test_mode(
    test_type_tag: Int, test_type: String, label: String
) -> Int:
    if test_type_tag == NonlinearTestTypeTag.MaxDispIncr:
        return 0
    if test_type_tag == NonlinearTestTypeTag.NormDispIncr:
        return 1
    if test_type_tag == NonlinearTestTypeTag.NormUnbalance:
        return 2
    if test_type_tag == NonlinearTestTypeTag.EnergyIncr:
        return 3
    abort("unsupported " + label + " test_type: " + test_type)
    return -1


fn _append_transient_solver_attempt(
    algorithm: String,
    algorithm_tag: Int,
    line_search_eta: Float64,
    test_type: String,
    test_type_tag: Int,
    max_iters: Int,
    tol: Float64,
    rel_tol: Float64,
    label: String,
    mut retry_algorithm_tags: List[Int],
    mut retry_algorithm_modes: List[Int],
    mut retry_test_modes: List[Int],
    mut retry_max_iters: List[Int],
    mut retry_tols: List[Float64],
    mut retry_rel_tols: List[Float64],
    mut retry_line_search_etas: List[Float64],
):
    if len(algorithm) == 0:
        return
    retry_algorithm_tags.append(algorithm_tag)
    retry_algorithm_modes.append(
        _transient_nonlinear_algorithm_mode(algorithm_tag, algorithm, label)
    )
    retry_test_modes.append(
        _transient_nonlinear_test_mode(test_type_tag, test_type, label)
    )
    if max_iters < 1:
        abort(label + "_max_iters must be >= 1")
    retry_max_iters.append(max_iters)
    retry_tols.append(tol)
    retry_rel_tols.append(rel_tol)
    retry_line_search_etas.append(line_search_eta)


fn _append_transient_solver_attempt_from_input(
    attempt: SolverAttemptInput,
    label: String,
    mut retry_algorithm_tags: List[Int],
    mut retry_algorithm_modes: List[Int],
    mut retry_test_modes: List[Int],
    mut retry_max_iters: List[Int],
    mut retry_tols: List[Float64],
    mut retry_rel_tols: List[Float64],
    mut retry_line_search_etas: List[Float64],
):
    _append_transient_solver_attempt(
        attempt.algorithm,
        attempt.algorithm_tag,
        attempt.line_search_eta,
        attempt.test_type,
        attempt.test_type_tag,
        attempt.max_iters,
        attempt.tol,
        attempt.rel_tol,
        label,
        retry_algorithm_tags,
        retry_algorithm_modes,
        retry_test_modes,
        retry_max_iters,
        retry_tols,
        retry_rel_tols,
        retry_line_search_etas,
    )


fn _collect_transient_solver_chain(
    analysis: AnalysisInput,
    analysis_solver_chain_pool: List[SolverAttemptInput],
    mut retry_algorithm_tags: List[Int],
    mut retry_algorithm_modes: List[Int],
    mut retry_test_modes: List[Int],
    mut retry_max_iters: List[Int],
    mut retry_tols: List[Float64],
    mut retry_rel_tols: List[Float64],
    mut retry_line_search_etas: List[Float64],
):
    for i in range(analysis.solver_chain_count):
        _append_transient_solver_attempt_from_input(
            analysis_solver_chain_pool[analysis.solver_chain_offset + i],
            "transient_nonlinear solver_chain",
            retry_algorithm_tags,
            retry_algorithm_modes,
            retry_test_modes,
            retry_max_iters,
            retry_tols,
            retry_rel_tols,
            retry_line_search_etas,
        )
    if len(retry_algorithm_modes) == 0:
        _append_transient_solver_attempt(
            analysis.algorithm,
            analysis.algorithm_tag,
            1.0,
            analysis.test_type,
            analysis.test_type_tag,
            analysis.max_iters,
            analysis.tol,
            analysis.rel_tol,
            "transient_nonlinear primary",
            retry_algorithm_tags,
            retry_algorithm_modes,
            retry_test_modes,
            retry_max_iters,
            retry_tols,
            retry_rel_tols,
            retry_line_search_etas,
        )
    if analysis.has_solver_chain_override or len(retry_algorithm_modes) != 1:
        return
    if retry_algorithm_modes[0] == NonlinearAlgorithmMode.Newton:
        return
    _append_transient_solver_attempt(
        "Newton",
        AnalysisAlgorithmTag.Newton,
        1.0,
        analysis.test_type,
        analysis.test_type_tag,
        retry_max_iters[0],
        retry_tols[0],
        retry_rel_tols[0],
        "transient_nonlinear auto_fallback",
        retry_algorithm_tags,
        retry_algorithm_modes,
        retry_test_modes,
        retry_max_iters,
        retry_tols,
        retry_rel_tols,
        retry_line_search_etas,
    )


fn run_transient_nonlinear(
    analysis: AnalysisInput,
    steps: Int,
    ts_index: Int,
    time_series: List[TimeSeriesInput],
    time_series_values: List[Float64],
    time_series_times: List[Float64],
    analysis_solver_chain_pool: List[SolverAttemptInput],
    dampings: List[DampingInput],
    pattern_type: String,
    pattern_type_tag: Int,
    uniform_excitation_direction: Int,
    uniform_accel_ts_index: Int,
    rayleigh_alpha_m: Float64,
    rayleigh_beta_k: Float64,
    rayleigh_beta_k_init: Float64,
    rayleigh_beta_k_comm: Float64,
    typed_nodes: List[NodeInput],
    typed_elements: List[ElementInput],
    node_x: List[Float64],
    node_y: List[Float64],
    node_z: List[Float64],
    elem_dof_offsets: List[Int],
    elem_dof_pool: List[Int],
    elem_node_offsets: List[Int],
    elem_node_pool: List[Int],
    elem_primary_material_ids: List[Int],
    elem_type_tags: List[Int],
    elem_geom_tags: List[Int],
    elem_section_ids: List[Int],
    elem_integration_tags: List[Int],
    elem_num_int_pts: List[Int],
    elem_area: List[Float64],
    elem_thickness: List[Float64],
    frame2d_elem_indices: List[Int],
    frame3d_elem_indices: List[Int],
    truss_elem_indices: List[Int],
    zero_length_elem_indices: List[Int],
    two_node_link_elem_indices: List[Int],
    zero_length_section_elem_indices: List[Int],
    quad_elem_indices: List[Int],
    shell_elem_indices: List[Int],
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
    M_rayleigh_total: List[Float64],
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
    frame_assemble_uniaxial: Int,
    frame_assemble_fiber: Int,
    frame_solve_nonlinear: Int,
    frame_nonlinear_step: Int,
    frame_nonlinear_iter: Int,
    frame_time_series_eval: Int,
    frame_constraints: Int,
    frame_recorders: Int,
    frame_factorize: Int,
    frame_transient_step: Int,
    frame_uniaxial_revert_all: Int,
    frame_uniaxial_commit_all: Int,
) raises:
    var time = Python.import_module("time")
    var asm_dof_map6: List[Int] = []
    var asm_dof_map12: List[Int] = []
    var asm_u_elem6: List[Float64] = []
    var force_beam_column2d_scratch = ForceBeamColumn2dScratch()
    var force_beam_column3d_scratch = ForceBeamColumn3dScratch()

    var dt = analysis.dt
    if dt <= 0.0:
        abort("transient_nonlinear requires dt > 0")
    if (
        pattern_type_tag != PatternTypeTag.Plain
        and pattern_type_tag != PatternTypeTag.UniformExcitation
    ):
        abort("unsupported pattern type: " + pattern_type)
    if pattern_type_tag == PatternTypeTag.UniformExcitation:
        if uniform_excitation_direction < 1 or uniform_excitation_direction > ndm:
            abort("UniformExcitation direction out of range")
        if uniform_accel_ts_index < 0:
            abort("UniformExcitation missing accel time_series")
    var uniform_excitation_dofs: List[Int] = []
    if pattern_type_tag == PatternTypeTag.UniformExcitation:
        for i in range(total_dofs):
            if (i % ndf) + 1 == uniform_excitation_direction:
                uniform_excitation_dofs.append(i)

    var retry_algorithm_tags: List[Int] = []
    var retry_algorithm_modes: List[Int] = []
    var retry_test_modes: List[Int] = []
    var retry_max_iters: List[Int] = []
    var retry_tols: List[Float64] = []
    var retry_rel_tols: List[Float64] = []
    var retry_line_search_etas: List[Float64] = []
    _collect_transient_solver_chain(
        analysis,
        analysis_solver_chain_pool,
        retry_algorithm_tags,
        retry_algorithm_modes,
        retry_test_modes,
        retry_max_iters,
        retry_tols,
        retry_rel_tols,
        retry_line_search_etas,
    )
    if len(retry_algorithm_modes) == 0:
        abort("transient_nonlinear solver_chain must contain at least one attempt")
    var retry_attempt_count = len(retry_algorithm_modes)
    var primary_algorithm_tag = retry_algorithm_tags[0]
    var primary_algorithm_mode = retry_algorithm_modes[0]
    var primary_test_mode = retry_test_modes[0]
    var max_iters = retry_max_iters[0]
    var tol = retry_tols[0]
    var rel_tol = retry_rel_tols[0]
    var primary_line_search_eta = retry_line_search_etas[0]

    var integrator_tag = analysis.integrator_tag
    if integrator_tag == IntegratorTypeTag.Unknown:
        integrator_tag = IntegratorTypeTag.Newmark
    if integrator_tag != IntegratorTypeTag.Newmark:
        abort("transient_nonlinear only supports Newmark integrator")
    var gamma = analysis.integrator_gamma
    var beta = analysis.integrator_beta
    if beta <= 0.0:
        abort("Newmark beta must be > 0")

    var free_count = len(free)
    var M_f: List[Float64] = []
    M_f.resize(free_count, 0.0)
    var M_rayleigh_f: List[Float64] = []
    M_rayleigh_f.resize(free_count, 0.0)
    var has_mass = False
    for i in range(free_count):
        var m = M_total[free[i]]
        M_f[i] = m
        M_rayleigh_f[i] = M_rayleigh_total[free[i]]
        if m != 0.0:
            has_mass = True
    if not has_mass:
        abort("transient_nonlinear requires masses on free dofs")

    var a0 = 1.0 / (beta * dt * dt)
    var a1 = gamma / (beta * dt)
    var a2 = 1.0 / (beta * dt)
    var a3 = 1.0 / (2.0 * beta) - 1.0
    var beta_sum = rayleigh_beta_k + rayleigh_beta_k_init + rayleigh_beta_k_comm
    var need_comm_tangent = rayleigh_beta_k_comm != 0.0
    var need_link_rayleigh_filter = (
        beta_sum != 0.0 and _has_nonparticipating_link_for_rayleigh(typed_elements)
    )
    var has_zero_length_damp_mats = _has_zero_length_damp_mats(typed_elements)
    var has_zero_length_dampers = _has_zero_length_dampers(typed_elements)

    var K: List[List[Float64]] = []
    for _ in range(total_dofs):
        var row: List[Float64] = []
        row.resize(total_dofs, 0.0)
        K.append(row^)
    var F_int: List[Float64] = []
    F_int.resize(total_dofs, 0.0)
    var K_link_all: List[List[Float64]] = []
    var K_link_rayleigh: List[List[Float64]] = []
    if need_link_rayleigh_filter:
        for _ in range(total_dofs):
            var row_all: List[Float64] = []
            row_all.resize(total_dofs, 0.0)
            K_link_all.append(row_all^)
            var row_rayleigh: List[Float64] = []
            row_rayleigh.resize(total_dofs, 0.0)
            K_link_rayleigh.append(row_rayleigh^)

    var K_ff: List[List[Float64]] = []
    var K_damp_ff: List[List[Float64]] = []
    var K_init_ff: List[List[Float64]] = []
    var K_init_damp_ff: List[List[Float64]] = []
    var K_comm_ff: List[List[Float64]] = []
    var K_comm_damp_ff: List[List[Float64]] = []
    var K_current_damp_ff: List[List[Float64]] = []
    var C_ff: List[List[Float64]] = []
    var C_zero_length_damp_ff: List[List[Float64]] = []
    var K_zero_length_damp_ff: List[List[Float64]] = []
    var K_eff: List[List[Float64]] = []
    var K_step: List[List[Float64]] = []
    var K_lu: List[List[Float64]] = []
    for _ in range(free_count):
        var row_kff: List[Float64] = []
        row_kff.resize(free_count, 0.0)
        K_ff.append(row_kff^)
        var row_kdamp: List[Float64] = []
        row_kdamp.resize(free_count, 0.0)
        K_damp_ff.append(row_kdamp^)
        var row_kinit: List[Float64] = []
        row_kinit.resize(free_count, 0.0)
        K_init_ff.append(row_kinit^)
        var row_kinit_damp: List[Float64] = []
        row_kinit_damp.resize(free_count, 0.0)
        K_init_damp_ff.append(row_kinit_damp^)
        var row_kcomm: List[Float64] = []
        row_kcomm.resize(free_count, 0.0)
        K_comm_ff.append(row_kcomm^)
        var row_kcomm_damp: List[Float64] = []
        row_kcomm_damp.resize(free_count, 0.0)
        K_comm_damp_ff.append(row_kcomm_damp^)
        var row_kcurrent_damp: List[Float64] = []
        row_kcurrent_damp.resize(free_count, 0.0)
        K_current_damp_ff.append(row_kcurrent_damp^)
        var row_c: List[Float64] = []
        row_c.resize(free_count, 0.0)
        C_ff.append(row_c^)
        var row_zero_length_damp: List[Float64] = []
        row_zero_length_damp.resize(free_count, 0.0)
        C_zero_length_damp_ff.append(row_zero_length_damp^)
        var row_zero_length_damp_k: List[Float64] = []
        row_zero_length_damp_k.resize(free_count, 0.0)
        K_zero_length_damp_ff.append(row_zero_length_damp_k^)
        var row_keff: List[Float64] = []
        row_keff.resize(free_count, 0.0)
        K_eff.append(row_keff^)
        var row_step: List[Float64] = []
        row_step.resize(free_count, 0.0)
        K_step.append(row_step^)
        var row_lu: List[Float64] = []
        row_lu.resize(free_count, 0.0)
        K_lu.append(row_lu^)
    var F_zero_length_damp_f: List[Float64] = []
    F_zero_length_damp_f.resize(free_count, 0.0)

    if do_profile:
        var t_asm_start = Int(time.perf_counter_ns())
        var asm_start_us = (t_asm_start - t0) // 1000
        _append_event(
            events, events_need_comma, "O", frame_assemble_stiffness, asm_start_us
        )
    var initial_pattern_scale = 0.0
    if pattern_type_tag == PatternTypeTag.Plain:
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
    var refresh_step_element_load_cache = (
        pattern_type_tag == PatternTypeTag.Plain
        and ts_index >= 0
        and len(pattern_element_loads) > 0
    )
    reset_force_beam_column2d_scratch(force_beam_column2d_scratch)
    reset_force_beam_column3d_scratch(force_beam_column3d_scratch)
    assemble_global_stiffness_and_internal_soa(
        typed_nodes,
        typed_elements,
        node_x,
        node_y,
        node_z,
        elem_dof_offsets,
        elem_dof_pool,
        elem_node_offsets,
        elem_node_pool,
        elem_primary_material_ids,
        elem_type_tags,
        elem_geom_tags,
        elem_section_ids,
        elem_integration_tags,
        elem_num_int_pts,
        elem_area,
        elem_thickness,
        frame2d_elem_indices,
        frame3d_elem_indices,
        truss_elem_indices,
        zero_length_elem_indices,
        two_node_link_elem_indices,
        zero_length_section_elem_indices,
        quad_elem_indices,
        shell_elem_indices,
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
        force_beam_column2d_scratch,
        force_beam_column3d_scratch,
        asm_dof_map6,
        asm_dof_map12,
        asm_u_elem6,
        K,
        F_int,
        do_profile,
        t0,
        events,
        events_need_comma,
        frame_assemble_uniaxial,
        frame_assemble_fiber,
    )
    if do_profile:
        var t_asm_end = Int(time.perf_counter_ns())
        var asm_end_us = (t_asm_end - t0) // 1000
        _append_event(
            events, events_need_comma, "C", frame_assemble_stiffness, asm_end_us
        )
    if has_zero_length_damp_mats:
        var C_zero_length_global: List[List[Float64]] = []
        for _ in range(total_dofs):
            var row_global: List[Float64] = []
            row_global.resize(total_dofs, 0.0)
            C_zero_length_global.append(row_global^)
        assemble_zero_length_damping_typed(
            typed_nodes,
            typed_elements,
            typed_materials_by_id,
            node_count,
            ndf,
            ndm,
            C_zero_length_global,
        )
        if has_transformation_mpc:
            C_zero_length_global = _collapse_matrix_by_rep(C_zero_length_global, rep_dof)
        for i in range(free_count):
            for j in range(free_count):
                C_zero_length_damp_ff[i][j] = C_zero_length_global[free[i]][free[j]]
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
    if need_link_rayleigh_filter:
        assemble_link_stiffness_typed(
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
            fiber_section3d_defs,
            fiber_section3d_cells,
            fiber_section3d_index_by_id,
            elem_dof_offsets,
            elem_dof_pool,
            asm_dof_map6,
            asm_dof_map12,
            asm_u_elem6,
            False,
            K_link_all,
        )
        assemble_link_stiffness_typed(
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
            fiber_section3d_defs,
            fiber_section3d_cells,
            fiber_section3d_index_by_id,
            elem_dof_offsets,
            elem_dof_pool,
            asm_dof_map6,
            asm_dof_map12,
            asm_u_elem6,
            True,
            K_link_rayleigh,
        )
        if has_transformation_mpc:
            K_link_all = _collapse_matrix_by_rep(K_link_all, rep_dof)
            K_link_rayleigh = _collapse_matrix_by_rep(K_link_rayleigh, rep_dof)
        for i in range(free_count):
            for j in range(free_count):
                K_init_damp_ff[i][j] = (
                    K_init_ff[i][j]
                    - K_link_all[free[i]][free[j]]
                    + K_link_rayleigh[free[i]][free[j]]
                )
                K_comm_damp_ff[i][j] = K_init_damp_ff[i][j]
    else:
        for i in range(free_count):
            for j in range(free_count):
                K_init_damp_ff[i][j] = K_init_ff[i][j]
                K_comm_damp_ff[i][j] = K_init_ff[i][j]
    if do_profile:
        var t_revert_start = Int(time.perf_counter_ns())
        var revert_start_us = (t_revert_start - t0) // 1000
        _append_event(
            events,
            events_need_comma,
            "O",
            frame_uniaxial_revert_all,
            revert_start_us,
        )
    uniaxial_revert_trial_all(uniaxial_states)
    if do_profile:
        var t_revert_end = Int(time.perf_counter_ns())
        var revert_end_us = (t_revert_end - t0) // 1000
        _append_event(
            events,
            events_need_comma,
            "C",
            frame_uniaxial_revert_all,
            revert_end_us,
        )

    var v: List[Float64] = []
    v.resize(total_dofs, 0.0)
    var a: List[Float64] = []
    a.resize(total_dofs, 0.0)
    var u_f: List[Float64] = []
    u_f.resize(free_count, 0.0)
    var v_f: List[Float64] = []
    v_f.resize(free_count, 0.0)
    var a_f: List[Float64] = []
    a_f.resize(free_count, 0.0)
    _gather_from_free_simd(free, u, u_f)

    var u_n: List[Float64] = []
    u_n.resize(free_count, 0.0)
    var v_n: List[Float64] = []
    v_n.resize(free_count, 0.0)
    var a_n: List[Float64] = []
    a_n.resize(free_count, 0.0)
    var u_step_base_f: List[Float64] = []
    u_step_base_f.resize(free_count, 0.0)
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

        copy_float64_contiguous(u_n, u_f, free_count)
        copy_float64_contiguous(v_n, v_f, free_count)
        copy_float64_contiguous(a_n, a_f, free_count)
        _predict_displacement_newmark_simd(u_f, u_n, v_n, a_n, dt, beta)
        _scatter_free_vector_to_full(free, u_f, u)
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
        copy_float64_contiguous(u_step_base_f, u_f, free_count)
        var force_basic_q_base = force_basic_q.copy()

        var t = Float64(step + 1) * dt
        if pattern_type_tag == PatternTypeTag.UniformExcitation:
            copy_float64_contiguous(F_ext_step, F_const, len(F_ext_step))
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
                copy_float64_contiguous(
                    F_ext_step,
                    build_active_nodal_load(F_const, F_pattern, factor),
                    len(F_ext_step),
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
                copy_float64_contiguous(
                    F_ext_step,
                    build_active_nodal_load(F_const, F_pattern, 1.0),
                    len(F_ext_step),
                )
                active_element_load_state = build_active_element_load_state(
                    const_element_loads,
                    pattern_element_loads,
                    1.0,
                    typed_elements,
                    elem_id_to_index,
                    ndm,
                    ndf,
                )
        if refresh_step_element_load_cache:
            invalidate_force_beam_column2d_load_cache(force_beam_column2d_scratch)
            invalidate_force_beam_column3d_load_cache(force_beam_column3d_scratch)
        _gather_from_free_simd(free, F_ext_step, P_ext_f)

        var converged = False
        var attempt_algorithm_tag = primary_algorithm_tag
        var attempt_algorithm_mode = primary_algorithm_mode
        var attempt_line_search_eta = primary_line_search_eta
        var attempt_test_mode = primary_test_mode
        var attempt_max_iters = max_iters
        var attempt_tol = tol
        var attempt_rel_tol = rel_tol
        for attempt in range(retry_attempt_count):
            if attempt > 0:
                copy_float64_contiguous(u_f, u_step_base_f, free_count)
                _scatter_free_vector_to_full(free, u_f, u)
                force_basic_q = force_basic_q_base.copy()
                attempt_algorithm_tag = retry_algorithm_tags[attempt]
                attempt_algorithm_mode = retry_algorithm_modes[attempt]
                attempt_line_search_eta = retry_line_search_etas[attempt]
                attempt_test_mode = retry_test_modes[attempt]
                attempt_max_iters = retry_max_iters[attempt]
                attempt_tol = retry_tols[attempt]
                attempt_rel_tol = retry_rel_tols[attempt]

            var tangent_initialized = False
            var damping_initialized = False
            var k_eff_initialized = False
            var k_eff_factored = False
            for _ in range(attempt_max_iters):
                if do_profile:
                    var t_revert_start = Int(time.perf_counter_ns())
                    var revert_start_us = (t_revert_start - t0) // 1000
                    _append_event(
                        events,
                        events_need_comma,
                        "O",
                        frame_uniaxial_revert_all,
                        revert_start_us,
                    )
                uniaxial_revert_trial_all(uniaxial_states)
                if do_profile:
                    var t_revert_end = Int(time.perf_counter_ns())
                    var revert_end_us = (t_revert_end - t0) // 1000
                    _append_event(
                        events,
                        events_need_comma,
                        "C",
                        frame_uniaxial_revert_all,
                        revert_end_us,
                    )
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
                assemble_global_stiffness_and_internal_soa(
                    typed_nodes,
                    typed_elements,
                    node_x,
                    node_y,
                    node_z,
                    elem_dof_offsets,
                    elem_dof_pool,
                    elem_node_offsets,
                    elem_node_pool,
                    elem_primary_material_ids,
                    elem_type_tags,
                    elem_geom_tags,
                    elem_section_ids,
                    elem_integration_tags,
                    elem_num_int_pts,
                    elem_area,
                    elem_thickness,
                    frame2d_elem_indices,
                    frame3d_elem_indices,
                    truss_elem_indices,
                    zero_length_elem_indices,
                    two_node_link_elem_indices,
                    zero_length_section_elem_indices,
                    quad_elem_indices,
                    shell_elem_indices,
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
                    force_beam_column2d_scratch,
                    force_beam_column3d_scratch,
                    asm_dof_map6,
                    asm_dof_map12,
                    asm_u_elem6,
                    K,
                    F_int,
                    do_profile,
                    t0,
                    events,
                    events_need_comma,
                    frame_assemble_uniaxial,
                    frame_assemble_fiber,
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
                if need_link_rayleigh_filter:
                    assemble_link_stiffness_typed(
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
                        fiber_section3d_defs,
                        fiber_section3d_cells,
                        fiber_section3d_index_by_id,
                        elem_dof_offsets,
                        elem_dof_pool,
                        asm_dof_map6,
                        asm_dof_map12,
                        asm_u_elem6,
                        False,
                        K_link_all,
                    )
                    assemble_link_stiffness_typed(
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
                        fiber_section3d_defs,
                        fiber_section3d_cells,
                        fiber_section3d_index_by_id,
                        elem_dof_offsets,
                        elem_dof_pool,
                        asm_dof_map6,
                        asm_dof_map12,
                        asm_u_elem6,
                        True,
                        K_link_rayleigh,
                    )
                    if has_transformation_mpc:
                        K_link_all = _collapse_matrix_by_rep(K_link_all, rep_dof)
                        K_link_rayleigh = _collapse_matrix_by_rep(K_link_rayleigh, rep_dof)
                    for i in range(free_count):
                        for j in range(free_count):
                            K_current_damp_ff[i][j] = (
                                K[free[i]][free[j]]
                                - K_link_all[free[i]][free[j]]
                                + K_link_rayleigh[free[i]][free[j]]
                            )
                if attempt_algorithm_mode == NonlinearAlgorithmMode.Newton:
                    for i in range(free_count):
                        for j in range(free_count):
                            K_ff[i][j] = K[free[i]][free[j]]
                            if need_link_rayleigh_filter:
                                K_damp_ff[i][j] = K_current_damp_ff[i][j]
                            else:
                                K_damp_ff[i][j] = K_ff[i][j]
                elif attempt_algorithm_mode == NonlinearAlgorithmMode.ModifiedNewton:
                    if not tangent_initialized:
                        for i in range(free_count):
                            for j in range(free_count):
                                K_ff[i][j] = K[free[i]][free[j]]
                                if need_link_rayleigh_filter:
                                    K_damp_ff[i][j] = K_current_damp_ff[i][j]
                                else:
                                    K_damp_ff[i][j] = K_ff[i][j]
                        tangent_initialized = True
                else:
                    if not tangent_initialized:
                        for i in range(free_count):
                            for j in range(free_count):
                                K_ff[i][j] = K_init_ff[i][j]
                                K_damp_ff[i][j] = K_init_damp_ff[i][j]
                        tangent_initialized = True

                if has_zero_length_dampers:
                    var K_zero_length_damp_global: List[List[Float64]] = []
                    var F_zero_length_damp: List[Float64] = []
                    for _ in range(total_dofs):
                        var row_global: List[Float64] = []
                        row_global.resize(total_dofs, 0.0)
                        K_zero_length_damp_global.append(row_global^)
                    F_zero_length_damp.resize(total_dofs, 0.0)
                    assemble_zero_length_damping_trial_typed(
                        typed_nodes,
                        typed_elements,
                        dampings,
                        time_series,
                        time_series_values,
                        time_series_times,
                        ndf,
                        ndm,
                        t,
                        dt,
                        uniaxial_states,
                        elem_uniaxial_offsets,
                        elem_uniaxial_counts,
                        elem_uniaxial_state_ids,
                        K_zero_length_damp_global,
                        F_zero_length_damp,
                    )
                    if has_transformation_mpc:
                        K_zero_length_damp_global = _collapse_matrix_by_rep(
                            K_zero_length_damp_global, rep_dof
                        )
                        F_zero_length_damp = _collapse_vector_by_rep(
                            F_zero_length_damp, rep_dof
                        )
                    for i in range(free_count):
                        F_zero_length_damp_f[i] = F_zero_length_damp[free[i]]
                        for j in range(free_count):
                            K_zero_length_damp_ff[i][j] = K_zero_length_damp_global[
                                free[i]
                            ][free[j]]
                else:
                    for i in range(free_count):
                        F_zero_length_damp_f[i] = 0.0
                        for j in range(free_count):
                            K_zero_length_damp_ff[i][j] = 0.0

                if (
                    attempt_algorithm_mode == NonlinearAlgorithmMode.Newton
                    or not damping_initialized
                ):
                    for i in range(free_count):
                        for j in range(free_count):
                            C_ff[i][j] = 0.0
                        C_ff[i][i] = rayleigh_alpha_m * M_rayleigh_f[i]
                    if rayleigh_beta_k != 0.0:
                        for i in range(free_count):
                            for j in range(free_count):
                                C_ff[i][j] += rayleigh_beta_k * K_damp_ff[i][j]
                    if rayleigh_beta_k_init != 0.0:
                        for i in range(free_count):
                            for j in range(free_count):
                                C_ff[i][j] += rayleigh_beta_k_init * K_init_damp_ff[i][j]
                    if rayleigh_beta_k_comm != 0.0:
                        for i in range(free_count):
                            for j in range(free_count):
                                C_ff[i][j] += rayleigh_beta_k_comm * K_comm_damp_ff[i][j]
                    if has_zero_length_damp_mats:
                        for i in range(free_count):
                            for j in range(free_count):
                                C_ff[i][j] += C_zero_length_damp_ff[i][j]
                    damping_initialized = True

                _build_trial_state_newmark_simd(
                    u_f, u_n, v_n, a_n, a0, a2, a3, gamma, dt, a_trial, v_trial
                )

                for i in range(free_count):
                    var damping_force = dot_float64_contiguous(C_ff[i], v_trial, free_count)
                    R_f[i] = (
                        P_ext_f[i]
                        - F_int[free[i]]
                        - F_zero_length_damp_f[i]
                        - damping_force
                        - M_f[i] * a_trial[i]
                    )
                if attempt_test_mode == 2:
                    var residual_norm = sqrt(sum_sq_float64_contiguous(R_f, free_count))
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
                            K_eff[i][j] = (
                                K_ff[i][j]
                                + K_zero_length_damp_ff[i][j]
                                + a1 * C_ff[i][j]
                            )
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
                if attempt_algorithm_tag == AnalysisAlgorithmTag.NewtonLineSearch:
                    var line_search_eta = attempt_line_search_eta
                    if line_search_eta <= 0.0:
                        line_search_eta = 0.8
                    for i in range(free_count):
                        du_f[i] *= line_search_eta
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
                var disp_incr_norm = sqrt(sum_sq_float64_contiguous(du_f, free_count))
                var energy_incr = abs(dot_float64_contiguous(du_f, R_f, free_count))
                var scale_tol = attempt_rel_tol * max_u
                if scale_tol < attempt_rel_tol:
                    scale_tol = attempt_rel_tol

                for i in range(free_count):
                    u_f[i] += du_f[i]
                _scatter_free_vector_to_full(free, u_f, u)
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
            assemble_global_stiffness_and_internal_soa(
                typed_nodes,
                typed_elements,
                node_x,
                node_y,
                node_z,
                elem_dof_offsets,
                elem_dof_pool,
                elem_node_offsets,
                elem_node_pool,
                elem_primary_material_ids,
                elem_type_tags,
                elem_geom_tags,
                elem_section_ids,
                elem_integration_tags,
                elem_num_int_pts,
                elem_area,
                elem_thickness,
                frame2d_elem_indices,
                frame3d_elem_indices,
                truss_elem_indices,
                zero_length_elem_indices,
                two_node_link_elem_indices,
                zero_length_section_elem_indices,
                quad_elem_indices,
                shell_elem_indices,
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
                force_beam_column2d_scratch,
                force_beam_column3d_scratch,
                asm_dof_map6,
                asm_dof_map12,
                asm_u_elem6,
                K,
                F_int,
                do_profile,
                t0,
                events,
                events_need_comma,
                frame_assemble_uniaxial,
                frame_assemble_fiber,
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
            if need_link_rayleigh_filter:
                assemble_link_stiffness_typed(
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
                    fiber_section3d_defs,
                    fiber_section3d_cells,
                    fiber_section3d_index_by_id,
                    elem_dof_offsets,
                    elem_dof_pool,
                    asm_dof_map6,
                    asm_dof_map12,
                    asm_u_elem6,
                    False,
                    K_link_all,
                )
                assemble_link_stiffness_typed(
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
                    fiber_section3d_defs,
                    fiber_section3d_cells,
                    fiber_section3d_index_by_id,
                    elem_dof_offsets,
                    elem_dof_pool,
                    asm_dof_map6,
                    asm_dof_map12,
                    asm_u_elem6,
                    True,
                    K_link_rayleigh,
                )
                if has_transformation_mpc:
                    K_link_all = _collapse_matrix_by_rep(K_link_all, rep_dof)
                    K_link_rayleigh = _collapse_matrix_by_rep(K_link_rayleigh, rep_dof)
                for i in range(free_count):
                    for j in range(free_count):
                        K_comm_damp_ff[i][j] = (
                            K_comm_ff[i][j]
                            - K_link_all[free[i]][free[j]]
                            + K_link_rayleigh[free[i]][free[j]]
                        )
            else:
                for i in range(free_count):
                    for j in range(free_count):
                        K_comm_damp_ff[i][j] = K_comm_ff[i][j]
        if do_profile:
            var t_commit_start = Int(time.perf_counter_ns())
            var commit_start_us = (t_commit_start - t0) // 1000
            _append_event(
                events,
                events_need_comma,
                "O",
                frame_uniaxial_commit_all,
                commit_start_us,
            )
        uniaxial_commit_all(uniaxial_states)
        if do_profile:
            var t_commit_end = Int(time.perf_counter_ns())
            var commit_end_us = (t_commit_end - t0) // 1000
            _append_event(
                events,
                events_need_comma,
                "C",
                frame_uniaxial_commit_all,
                commit_end_us,
            )
        _sync_force_beam_column2d_committed_basic_states(
            typed_nodes,
            typed_elements,
            ndf,
            u,
            force_basic_offsets,
            force_basic_counts,
            force_basic_q,
        )

        _update_post_step_newmark_simd(
            u_f, u_n, v_n, a_n, a0, a2, a3, gamma, dt, v_f, a_f
        )
        _scatter_free_vector_to_full(free, u_f, u)
        _scatter_free_vector_to_full(free, v_f, v)
        _scatter_free_vector_to_full(free, a_f, a)
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
            F_int_reaction = assemble_internal_forces_typed_soa(
                typed_nodes,
                typed_elements,
                node_x,
                node_y,
                node_z,
                elem_dof_offsets,
                elem_dof_pool,
                elem_node_offsets,
                elem_node_pool,
                elem_primary_material_ids,
                elem_type_tags,
                elem_geom_tags,
                elem_section_ids,
                elem_integration_tags,
                elem_num_int_pts,
                elem_area,
                elem_thickness,
                frame2d_elem_indices,
                frame3d_elem_indices,
                truss_elem_indices,
                zero_length_elem_indices,
                two_node_link_elem_indices,
                zero_length_section_elem_indices,
                quad_elem_indices,
                shell_elem_indices,
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
                force_beam_column2d_scratch,
                force_beam_column3d_scratch,
            )
        if record_any_element_force:
            for i in range(elem_count):
                elem_force_cached[i] = False
        for r in range(len(recorders)):
            var rec = recorders[r]
            if (
                rec.type_tag == RecorderTypeTag.NodeDisplacement
                or rec.type_tag == RecorderTypeTag.EnvelopeNodeDisplacement
                or rec.type_tag == RecorderTypeTag.EnvelopeNodeAcceleration
            ):
                for nidx in range(rec.node_count):
                    var node_id = recorder_nodes_pool[rec.node_offset + nidx]
                    var i = id_to_index[node_id]
                    var values: List[Float64] = []
                    values.resize(rec.dof_count, 0.0)
                    for j in range(rec.dof_count):
                        var dof = recorder_dofs_pool[rec.dof_offset + j]
                        require_dof_in_range(dof, ndf, "recorder")
                        var value = u[node_dof_index(i, dof, ndf)]
                        if rec.type_tag == RecorderTypeTag.EnvelopeNodeAcceleration:
                            value = a[node_dof_index(i, dof, ndf)]
                            if rec.time_series_tag >= 0:
                                var ts_index = find_time_series_input(
                                    time_series, rec.time_series_tag
                                )
                                if ts_index < 0:
                                    abort("recorder time series not found")
                                value += eval_time_series_input(
                                    time_series[ts_index],
                                    t,
                                    time_series_values,
                                    time_series_times,
                                )
                        values[j] = value
                    var filename = rec.output + "_node" + String(node_id) + ".out"
                    if rec.type_tag == RecorderTypeTag.NodeDisplacement:
                        _append_output(
                            transient_output_files,
                            transient_output_buffers,
                            filename,
                            _format_values_line(values),
                        )
                    else:
                        _update_envelope(
                            filename,
                            values,
                            envelope_files,
                            envelope_min,
                            envelope_max,
                            envelope_abs,
                        )
            elif rec.type_tag == RecorderTypeTag.ElementForce:
                for eidx in range(rec.element_count):
                    var elem_id = recorder_elements_pool[rec.element_offset + eidx]
                    if elem_id >= len(elem_id_to_index) or elem_id_to_index[elem_id] < 0:
                        abort("recorder element not found")
                    var elem_index = elem_id_to_index[elem_id]
                    if not elem_force_cached[elem_index]:
                        var elem = typed_elements[elem_index]
                        if (
                            elem.type_tag == ElementTypeTag.ForceBeamColumn2d
                            and typed_sections_by_id[elem.section].type == "FiberSection2d"
                        ):
                            elem_force_values[elem_index] = _force_beam_column2d_force_global_from_basic_state(
                                elem_index,
                                elem,
                                typed_nodes,
                                ndf,
                                u,
                                active_element_load_state.element_loads,
                                active_element_load_state.elem_load_offsets,
                                active_element_load_state.elem_load_pool,
                                1.0,
                                force_basic_offsets,
                                force_basic_counts,
                                force_basic_q,
                            )
                        else:
                            elem_force_values[elem_index] = _element_force_global_for_recorder(
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
                        elem_force_cached[elem_index] = True
                    var line = _format_values_line(elem_force_values[elem_index])
                    var filename = rec.output + "_ele" + String(elem_id) + ".out"
                    _append_output(
                        transient_output_files, transient_output_buffers, filename, line
                    )
            elif rec.type_tag == RecorderTypeTag.ElementLocalForce:
                for eidx in range(rec.element_count):
                    var elem_id = recorder_elements_pool[rec.element_offset + eidx]
                    if elem_id >= len(elem_id_to_index) or elem_id_to_index[elem_id] < 0:
                        abort("recorder element not found")
                    var elem_index = elem_id_to_index[elem_id]
                    var elem = typed_elements[elem_index]
                    var values = _element_local_force_for_recorder(
                        elem_index,
                        elem,
                        ndf,
                        u,
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
                    var filename = rec.output + "_ele" + String(elem_id) + ".out"
                    _append_output(
                        transient_output_files,
                        transient_output_buffers,
                        filename,
                        _format_values_line(values),
                    )
            elif rec.type_tag == RecorderTypeTag.ElementBasicForce:
                for eidx in range(rec.element_count):
                    var elem_id = recorder_elements_pool[rec.element_offset + eidx]
                    if elem_id >= len(elem_id_to_index) or elem_id_to_index[elem_id] < 0:
                        abort("recorder element not found")
                    var elem_index = elem_id_to_index[elem_id]
                    var elem = typed_elements[elem_index]
                    var values = _element_basic_force_for_recorder(
                        elem_index,
                        elem,
                        ndf,
                        u,
                        typed_nodes,
                        uniaxial_defs,
                        uniaxial_state_defs,
                        uniaxial_states,
                        elem_uniaxial_offsets,
                        elem_uniaxial_counts,
                        elem_uniaxial_state_ids,
                    )
                    var filename = rec.output + "_ele" + String(elem_id) + ".out"
                    _append_output(
                        transient_output_files,
                        transient_output_buffers,
                        filename,
                        _format_values_line(values),
                    )
            elif rec.type_tag == RecorderTypeTag.ElementDeformation:
                for eidx in range(rec.element_count):
                    var elem_id = recorder_elements_pool[rec.element_offset + eidx]
                    if elem_id >= len(elem_id_to_index) or elem_id_to_index[elem_id] < 0:
                        abort("recorder element not found")
                    var elem_index = elem_id_to_index[elem_id]
                    var elem = typed_elements[elem_index]
                    var values = _element_deformation_for_recorder(
                        elem_index, elem, ndf, u, typed_nodes
                    )
                    var filename = rec.output + "_ele" + String(elem_id) + ".out"
                    _append_output(
                        transient_output_files,
                        transient_output_buffers,
                        filename,
                        _format_values_line(values),
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
                        if (
                            elem.type_tag == ElementTypeTag.ForceBeamColumn2d
                            and typed_sections_by_id[elem.section].type == "FiberSection2d"
                        ):
                            elem_force_values[elem_index] = _force_beam_column2d_force_global_from_basic_state(
                                elem_index,
                                elem,
                                typed_nodes,
                                ndf,
                                u,
                                active_element_load_state.element_loads,
                                active_element_load_state.elem_load_offsets,
                                active_element_load_state.elem_load_pool,
                                1.0,
                                force_basic_offsets,
                                force_basic_counts,
                                force_basic_q,
                            )
                        else:
                            elem_force_values[elem_index] = _element_force_global_for_recorder(
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
            elif rec.type_tag == RecorderTypeTag.EnvelopeElementLocalForce:
                for eidx in range(rec.element_count):
                    var elem_id = recorder_elements_pool[rec.element_offset + eidx]
                    if elem_id >= len(elem_id_to_index) or elem_id_to_index[elem_id] < 0:
                        abort("recorder element not found")
                    var elem_index = elem_id_to_index[elem_id]
                    var elem = typed_elements[elem_index]
                    var values = _element_local_force_for_recorder(
                        elem_index,
                        elem,
                        ndf,
                        u,
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
                    var filename = rec.output + "_ele" + String(elem_id) + ".out"
                    _update_envelope(
                        filename,
                        values,
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
                        var values: List[Float64]
                        if (
                            elem.type_tag == ElementTypeTag.ForceBeamColumn2d
                            and typed_sections_by_id[elem.section].type == "FiberSection2d"
                        ):
                            values = _force_beam_column2d_section_response_from_basic_state(
                                elem_index,
                                elem,
                                sec_no,
                                active_element_load_state.element_loads,
                                active_element_load_state.elem_load_offsets,
                                active_element_load_state.elem_load_pool,
                                1.0,
                                typed_nodes,
                                typed_sections_by_id,
                                force_basic_offsets,
                                force_basic_counts,
                                force_basic_q,
                                want_defo,
                            )
                        else:
                            values = _section_response_for_recorder(
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
