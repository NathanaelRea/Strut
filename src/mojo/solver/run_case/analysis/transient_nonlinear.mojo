from algorithm import vectorize
from collections import Dict, List
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
    uniaxial_commit,
    uniaxial_revert_trial,
)
from os import abort
from python import Python
from sections import (
    FiberCell,
    FiberSection2dDef,
    FiberSection3dDef,
    LayeredShellSectionDef,
    fiber_section2d_commit_runtime_all,
    fiber_section2d_revert_trial_runtime_all,
)
from sys import simd_width_of

from solver.run_case.linear_solver_backend import (
    LinearSolverBackend,
    add_diagonal,
    add_reduced_matrix,
    clear,
    factorize_loaded,
    initialize_symbolic_from_element_dof_map,
    initialize_structure,
    load_reduced_matrix,
    solve,
)
from solver.assembly import (
    assemble_global_stiffness_and_internal_soa,
    assemble_link_stiffness_typed,
    assemble_internal_forces_typed_soa,
    assemble_zero_length_damping_trial_typed,
    assemble_zero_length_damping_typed,
)
from solver.dof import node_dof_index, require_dof_in_range
from solver.profile import (
    PROFILE_FRAME_UNIAXIAL_COPY_RESET,
    RuntimeProfileMetrics,
    _append_event,
)
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
    _append_output_at_index,
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
    _output_buffer_index,
    _section_response_for_recorder,
    _sync_force_beam_column2d_committed_basic_states,
    _update_envelope,
)
from tag_types import (
    AnalysisAlgorithmTag,
    AnalysisSystemTag,
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
fn _default_broyden_count(count: Int) -> Int:
    if count < 1:
        return 10
    return count


@always_inline
fn _default_line_search_tol(tol: Float64) -> Float64:
    if tol <= 0.0:
        return 0.8
    return tol


@always_inline
fn _update_initial_interpolated_eta(
    s0: Float64, s: Float64, r: Float64, r0: Float64, eta_prev: Float64
) -> Float64:
    var denom = s0 - s
    if abs(denom) <= 1.0e-16:
        return eta_prev
    var eta = eta_prev * s0 / denom
    if eta > 10.0:
        eta = 10.0
    if r > r0:
        eta = 1.0
    if eta < 0.1:
        eta = 0.1
    return eta


fn _copy_vector(src: List[Float64], count: Int) -> List[Float64]:
    var dst: List[Float64] = []
    dst.resize(count, 0.0)
    for i in range(count):
        dst[i] = src[i]
    return dst^


fn _broyden_update_direction(
    mut du: List[Float64],
    history_s: List[List[Float64]],
    history_z: List[List[Float64]],
    mut current_z: List[Float64],
    count: Int,
):
    var eps = 1.0e-16
    for i in range(count):
        var p = -dot_float64_contiguous(history_s[i], history_z[i], len(du))
        if abs(p) < eps:
            break
        var sdotz = dot_float64_contiguous(history_s[i], current_z, len(du))
        for j in range(len(du)):
            current_z[j] += ((history_s[i][j] + history_z[i][j]) * sdotz) / p
    for i in range(count):
        var p = -dot_float64_contiguous(history_s[i], history_z[i], len(du))
        if abs(p) < eps:
            break
        var sdotdu = dot_float64_contiguous(history_s[i], du, len(du))
        for j in range(len(du)):
            du[j] += ((history_s[i][j] + history_z[i][j]) * sdotdu) / p
    var current_p = -dot_float64_contiguous(history_s[count], current_z, len(du))
    if abs(current_p) >= eps:
        var current_sdotdu = dot_float64_contiguous(history_s[count], du, len(du))
        for j in range(len(du)):
            du[j] += (
                ((history_s[count][j] + current_z[j]) * current_sdotdu) / current_p
            )


fn _transient_residual(
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
    active_element_load_state_element_loads: List[ElementLoadInput],
    active_element_load_state_elem_load_offsets: List[Int],
    active_element_load_state_elem_load_pool: List[Int],
    typed_sections_by_id: List[SectionInput],
    typed_materials_by_id: List[MaterialInput],
    id_to_index: List[Int],
    node_count: Int,
    ndf: Int,
    ndm: Int,
    u: List[Float64],
    u_f: List[Float64],
    u_n: List[Float64],
    v_n: List[Float64],
    a_n: List[Float64],
    a0: Float64,
    a2: Float64,
    a3: Float64,
    gamma: Float64,
    dt: Float64,
    P_ext_f: List[Float64],
    M_f: List[Float64],
    C_ff: List[List[Float64]],
    mut uniaxial_states: List[UniMaterialState],
    uniaxial_defs: List[UniMaterialDef],
    uniaxial_state_defs: List[Int],
    elem_uniaxial_offsets: List[Int],
    elem_uniaxial_counts: List[Int],
    elem_uniaxial_state_ids: List[Int],
    force_basic_offsets: List[Int],
    force_basic_counts: List[Int],
    mut force_basic_q: List[Float64],
    mut fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    fiber_section3d_defs: List[FiberSection3dDef],
    fiber_section3d_cells: List[FiberCell],
    fiber_section3d_index_by_id: List[Int],
    mut layered_shell_section_defs: List[LayeredShellSectionDef],
    layered_shell_section_index_by_id: List[Int],
    layered_shell_section_uniaxial_offsets: List[Int],
    layered_shell_section_uniaxial_counts: List[Int],
    shell_elem_instance_offsets: List[Int],
    mut force_beam_column2d_scratch: ForceBeamColumn2dScratch,
    mut force_beam_column3d_scratch: ForceBeamColumn3dScratch,
    has_transformation_mpc: Bool,
    rep_dof: List[Int],
    free: List[Int],
    free_count: Int,
    dampings: List[DampingInput],
    time_series: List[TimeSeriesInput],
    time_series_values: List[Float64],
    time_series_times: List[Float64],
    t: Float64,
    has_zero_length_dampers: Bool,
    mut R_f: List[Float64],
    mut a_trial: List[Float64],
    mut v_trial: List[Float64],
    mut F_zero_length_damp_f: List[Float64],
) raises:
    var F_int = assemble_internal_forces_typed_soa(
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
        active_element_load_state_element_loads,
        active_element_load_state_elem_load_offsets,
        active_element_load_state_elem_load_pool,
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
        layered_shell_section_defs,
        layered_shell_section_index_by_id,
        layered_shell_section_uniaxial_offsets,
        layered_shell_section_uniaxial_counts,
        shell_elem_instance_offsets,
        force_beam_column2d_scratch,
        force_beam_column3d_scratch,
    )
    if has_transformation_mpc:
        F_int = _collapse_vector_by_rep(F_int, rep_dof)
    if has_zero_length_dampers:
        var K_zero_length_damp_global: List[List[Float64]] = []
        var F_zero_length_damp: List[Float64] = []
        for _ in range(node_count * ndf):
            var row_global: List[Float64] = []
            row_global.resize(node_count * ndf, 0.0)
            K_zero_length_damp_global.append(row_global^)
        F_zero_length_damp.resize(node_count * ndf, 0.0)
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
            F_zero_length_damp = _collapse_vector_by_rep(F_zero_length_damp, rep_dof)
        for i in range(free_count):
            F_zero_length_damp_f[i] = F_zero_length_damp[free[i]]
    else:
        for i in range(free_count):
            F_zero_length_damp_f[i] = 0.0
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


fn _uniaxial_revert_trial_active_states(
    mut uniaxial_states: List[UniMaterialState], elem_uniaxial_state_ids: List[Int]
):
    for i in range(len(elem_uniaxial_state_ids)):
        var state_id = elem_uniaxial_state_ids[i]
        if state_id < 0:
            continue
        if state_id >= len(uniaxial_states):
            abort("active uniaxial state id out of range")
        ref state = uniaxial_states[state_id]
        uniaxial_revert_trial(state)


fn _uniaxial_commit_active_states(
    mut uniaxial_states: List[UniMaterialState], elem_uniaxial_state_ids: List[Int]
):
    for i in range(len(elem_uniaxial_state_ids)):
        var state_id = elem_uniaxial_state_ids[i]
        if state_id < 0:
            continue
        if state_id >= len(uniaxial_states):
            abort("active uniaxial state id out of range")
        ref state = uniaxial_states[state_id]
        uniaxial_commit(state)


@always_inline
fn _scatter_free_vector_to_full_simd_impl[width: Int](
    free: List[Int], src: List[Float64], mut dst: List[Float64]
):
    var n = len(free)

    @parameter
    fn scatter_chunk[chunk: Int](i: Int):
        var src_vec = load_float64_contiguous_simd[chunk](src, i)
        scatter_float64_by_index_simd[chunk](free, i, dst, src_vec)

    vectorize[scatter_chunk, width](n)


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

    @parameter
    fn gather_chunk[chunk: Int](i: Int):
        var gathered_vec = gather_float64_by_index_simd[chunk](free, i, src)
        store_float64_contiguous_simd[chunk](dst, i, gathered_vec)

    vectorize[gather_chunk, width](n)


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
    var one_minus_gamma = 1.0 - gamma

    @parameter
    fn build_chunk[chunk: Int](i: Int):
        var u_curr = load_float64_contiguous_simd[chunk](u_f, i)
        var u_prev = load_float64_contiguous_simd[chunk](u_n, i)
        var v_prev = load_float64_contiguous_simd[chunk](v_n, i)
        var a_prev = load_float64_contiguous_simd[chunk](a_n, i)
        var a_next = (
            SIMD[DType.float64, chunk](a0) * (u_curr - u_prev)
            - SIMD[DType.float64, chunk](a2) * v_prev
            - SIMD[DType.float64, chunk](a3) * a_prev
        )
        var v_next = v_prev + SIMD[DType.float64, chunk](dt) * (
            SIMD[DType.float64, chunk](one_minus_gamma) * a_prev
            + SIMD[DType.float64, chunk](gamma) * a_next
        )
        store_float64_contiguous_simd[chunk](a_trial, i, a_next)
        store_float64_contiguous_simd[chunk](v_trial, i, v_next)

    vectorize[build_chunk, width](n)


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
    var coeff = dt * dt * (0.5 - beta)

    @parameter
    fn predict_chunk[chunk: Int](i: Int):
        var u_prev = load_float64_contiguous_simd[chunk](u_n, i)
        var v_prev = load_float64_contiguous_simd[chunk](v_n, i)
        var a_prev = load_float64_contiguous_simd[chunk](a_n, i)
        var u_pred = (
            u_prev
            + SIMD[DType.float64, chunk](dt) * v_prev
            + SIMD[DType.float64, chunk](coeff) * a_prev
        )
        store_float64_contiguous_simd[chunk](u_f, i, u_pred)

    vectorize[predict_chunk, width](n)


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
    var one_minus_gamma = 1.0 - gamma

    @parameter
    fn update_chunk[chunk: Int](i: Int):
        var u_curr = load_float64_contiguous_simd[chunk](u_f, i)
        var u_prev = load_float64_contiguous_simd[chunk](u_n, i)
        var v_prev = load_float64_contiguous_simd[chunk](v_n, i)
        var a_prev = load_float64_contiguous_simd[chunk](a_n, i)
        var a_next = (
            SIMD[DType.float64, chunk](a0) * (u_curr - u_prev)
            - SIMD[DType.float64, chunk](a2) * v_prev
            - SIMD[DType.float64, chunk](a3) * a_prev
        )
        var v_next = v_prev + SIMD[DType.float64, chunk](dt) * (
            SIMD[DType.float64, chunk](one_minus_gamma) * a_prev
            + SIMD[DType.float64, chunk](gamma) * a_next
        )
        store_float64_contiguous_simd[chunk](a_f, i, a_next)
        store_float64_contiguous_simd[chunk](v_f, i, v_next)

    vectorize[update_chunk, width](n)


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
        # These algorithms still use Newton tangents, but add their own
        # per-iteration correction logic below.
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
    broyden_count: Int,
    line_search_eta: Float64,
    test_type: String,
    test_type_tag: Int,
    max_iters: Int,
    tol: Float64,
    label: String,
    mut retry_algorithm_tags: List[Int],
    mut retry_algorithm_modes: List[Int],
    mut retry_test_modes: List[Int],
    mut retry_max_iters: List[Int],
    mut retry_tols: List[Float64],
    mut retry_broyden_counts: List[Int],
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
    retry_broyden_counts.append(broyden_count)
    retry_line_search_etas.append(line_search_eta)


fn _append_transient_solver_attempt_from_input(
    attempt: SolverAttemptInput,
    label: String,
    mut retry_algorithm_tags: List[Int],
    mut retry_algorithm_modes: List[Int],
    mut retry_test_modes: List[Int],
    mut retry_max_iters: List[Int],
    mut retry_tols: List[Float64],
    mut retry_broyden_counts: List[Int],
    mut retry_line_search_etas: List[Float64],
):
    _append_transient_solver_attempt(
        attempt.algorithm,
        attempt.algorithm_tag,
        attempt.broyden_count,
        attempt.line_search_eta,
        attempt.test_type,
        attempt.test_type_tag,
        attempt.max_iters,
        attempt.tol,
        label,
        retry_algorithm_tags,
        retry_algorithm_modes,
        retry_test_modes,
        retry_max_iters,
        retry_tols,
        retry_broyden_counts,
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
    mut retry_broyden_counts: List[Int],
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
            retry_broyden_counts,
            retry_line_search_etas,
        )
    if len(retry_algorithm_modes) == 0:
        _append_transient_solver_attempt(
            analysis.algorithm,
            analysis.algorithm_tag,
            0,
            1.0,
            analysis.test_type,
            analysis.test_type_tag,
            analysis.max_iters,
            analysis.tol,
            "transient_nonlinear primary",
            retry_algorithm_tags,
            retry_algorithm_modes,
            retry_test_modes,
            retry_max_iters,
            retry_tols,
            retry_broyden_counts,
            retry_line_search_etas,
        )
    if analysis.has_solver_chain_override or len(retry_algorithm_modes) != 1:
        return
    if retry_algorithm_modes[0] == NonlinearAlgorithmMode.Newton:
        return
    _append_transient_solver_attempt(
        "Newton",
        AnalysisAlgorithmTag.Newton,
        0,
        1.0,
        analysis.test_type,
        analysis.test_type_tag,
        retry_max_iters[0],
        retry_tols[0],
        "transient_nonlinear auto_fallback",
        retry_algorithm_tags,
        retry_algorithm_modes,
        retry_test_modes,
        retry_max_iters,
        retry_tols,
        retry_broyden_counts,
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
    mut fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    fiber_section3d_defs: List[FiberSection3dDef],
    fiber_section3d_cells: List[FiberCell],
    fiber_section3d_index_by_id: List[Int],
    mut layered_shell_section_defs: List[LayeredShellSectionDef],
    layered_shell_section_index_by_id: List[Int],
    layered_shell_section_uniaxial_offsets: List[Int],
    layered_shell_section_uniaxial_counts: List[Int],
    shell_elem_instance_offsets: List[Int],
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
    mut runtime_metrics: RuntimeProfileMetrics,
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
    var retry_broyden_counts: List[Int] = []
    var retry_line_search_etas: List[Float64] = []
    _collect_transient_solver_chain(
        analysis,
        analysis_solver_chain_pool,
        retry_algorithm_tags,
        retry_algorithm_modes,
        retry_test_modes,
        retry_max_iters,
        retry_tols,
        retry_broyden_counts,
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
    var primary_broyden_count = retry_broyden_counts[0]
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
    var free_index: List[Int] = []
    free_index.resize(total_dofs, -1)
    var has_mass = False
    for i in range(free_count):
        free_index[free[i]] = i
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
    var backend = LinearSolverBackend()
    initialize_structure(backend, analysis, free_count)
    initialize_symbolic_from_element_dof_map(backend, elem_dof_offsets, elem_dof_pool, free_index)

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
    if do_profile:
        var t_reset_start = Int(time.perf_counter_ns())
        _append_event(
            events,
            events_need_comma,
            "O",
            PROFILE_FRAME_UNIAXIAL_COPY_RESET,
            (t_reset_start - t0) // 1000,
        )
    reset_force_beam_column2d_scratch(force_beam_column2d_scratch)
    reset_force_beam_column3d_scratch(force_beam_column3d_scratch)
    if do_profile:
        var t_reset_end = Int(time.perf_counter_ns())
        _append_event(
            events,
            events_need_comma,
            "C",
            PROFILE_FRAME_UNIAXIAL_COPY_RESET,
            (t_reset_end - t0) // 1000,
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
        layered_shell_section_defs,
        layered_shell_section_index_by_id,
        layered_shell_section_uniaxial_offsets,
        layered_shell_section_uniaxial_counts,
        shell_elem_instance_offsets,
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
        runtime_metrics,
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
        layered_shell_section_defs,
        layered_shell_section_index_by_id,
        layered_shell_section_uniaxial_offsets,
        layered_shell_section_uniaxial_counts,
        shell_elem_instance_offsets,
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
        layered_shell_section_defs,
        layered_shell_section_index_by_id,
        layered_shell_section_uniaxial_offsets,
        layered_shell_section_uniaxial_counts,
        shell_elem_instance_offsets,
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
    _uniaxial_revert_trial_active_states(uniaxial_states, elem_uniaxial_state_ids)
    fiber_section2d_revert_trial_runtime_all(fiber_section_defs)
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
    var lu_rhs: List[Float64] = []
    lu_rhs.resize(free_count, 0.0)

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

    if len(transient_output_files) != len(transient_output_buffers):
        abort("output files/buffers length mismatch")
    var output_file_index: Dict[String, Int] = {}
    for i in range(len(transient_output_files)):
        output_file_index[transient_output_files[i]] = i

    var recorder_count = len(recorders)
    var recorder_node_dof_index_offsets: List[Int] = []
    recorder_node_dof_index_offsets.resize(recorder_count, -1)
    var recorder_node_file_index_offsets: List[Int] = []
    recorder_node_file_index_offsets.resize(recorder_count, -1)
    var recorder_node_time_series_index: List[Int] = []
    recorder_node_time_series_index.resize(recorder_count, -1)
    var recorder_element_index_offsets: List[Int] = []
    recorder_element_index_offsets.resize(recorder_count, -1)
    var recorder_element_file_index_offsets: List[Int] = []
    recorder_element_file_index_offsets.resize(recorder_count, -1)
    var recorder_section_file_index_offsets: List[Int] = []
    recorder_section_file_index_offsets.resize(recorder_count, -1)
    var recorder_drift_i_dof_index: List[Int] = []
    recorder_drift_i_dof_index.resize(recorder_count, -1)
    var recorder_drift_j_dof_index: List[Int] = []
    recorder_drift_j_dof_index.resize(recorder_count, -1)
    var recorder_drift_denominator: List[Float64] = []
    recorder_drift_denominator.resize(recorder_count, 0.0)
    var recorder_drift_file_index: List[Int] = []
    recorder_drift_file_index.resize(recorder_count, -1)

    var recorder_node_dof_indices: List[Int] = []
    var recorder_node_file_indices: List[Int] = []
    var recorder_element_indices: List[Int] = []
    var recorder_element_file_indices: List[Int] = []
    var recorder_section_file_indices: List[Int] = []

    for r in range(recorder_count):
        var rec = recorders[r]

        if (
            rec.type_tag == RecorderTypeTag.NodeDisplacement
            or rec.type_tag == RecorderTypeTag.EnvelopeNodeDisplacement
            or rec.type_tag == RecorderTypeTag.EnvelopeNodeAcceleration
            or rec.type_tag == RecorderTypeTag.NodeReaction
        ):
            recorder_node_dof_index_offsets[r] = len(recorder_node_dof_indices)
            recorder_node_file_index_offsets[r] = len(recorder_node_file_indices)
            if (
                rec.type_tag == RecorderTypeTag.EnvelopeNodeAcceleration
                and rec.time_series_tag >= 0
            ):
                var ts_index = find_time_series_input(time_series, rec.time_series_tag)
                if ts_index < 0:
                    abort("recorder time series not found")
                recorder_node_time_series_index[r] = ts_index
            for nidx in range(rec.node_count):
                var node_id = recorder_nodes_pool[rec.node_offset + nidx]
                if node_id >= len(id_to_index) or id_to_index[node_id] < 0:
                    abort("recorder node not found")
                var node_index = id_to_index[node_id]
                var filename = rec.output + "_node" + String(node_id) + ".out"
                var file_index = _output_buffer_index(
                    transient_output_files,
                    transient_output_buffers,
                    output_file_index,
                    filename,
                )
                recorder_node_file_indices.append(file_index)
                for j in range(rec.dof_count):
                    var dof = recorder_dofs_pool[rec.dof_offset + j]
                    require_dof_in_range(dof, ndf, "recorder")
                    recorder_node_dof_indices.append(
                        node_dof_index(node_index, dof, ndf)
                    )

        if (
            rec.type_tag == RecorderTypeTag.ElementForce
            or rec.type_tag == RecorderTypeTag.ElementLocalForce
            or rec.type_tag == RecorderTypeTag.ElementBasicForce
            or rec.type_tag == RecorderTypeTag.ElementDeformation
            or rec.type_tag == RecorderTypeTag.EnvelopeElementForce
            or rec.type_tag == RecorderTypeTag.EnvelopeElementLocalForce
            or rec.type_tag == RecorderTypeTag.SectionForce
            or rec.type_tag == RecorderTypeTag.SectionDeformation
        ):
            recorder_element_index_offsets[r] = len(recorder_element_indices)
            if (
                rec.type_tag != RecorderTypeTag.SectionForce
                and rec.type_tag != RecorderTypeTag.SectionDeformation
            ):
                recorder_element_file_index_offsets[r] = len(
                    recorder_element_file_indices
                )
            for eidx in range(rec.element_count):
                var elem_id = recorder_elements_pool[rec.element_offset + eidx]
                if elem_id >= len(elem_id_to_index) or elem_id_to_index[elem_id] < 0:
                    abort("recorder element not found")
                var elem_index = elem_id_to_index[elem_id]
                recorder_element_indices.append(elem_index)
                if (
                    rec.type_tag != RecorderTypeTag.SectionForce
                    and rec.type_tag != RecorderTypeTag.SectionDeformation
                ):
                    var filename = rec.output + "_ele" + String(elem_id) + ".out"
                    var file_index = _output_buffer_index(
                        transient_output_files,
                        transient_output_buffers,
                        output_file_index,
                        filename,
                    )
                    recorder_element_file_indices.append(file_index)

        if (
            rec.type_tag == RecorderTypeTag.SectionForce
            or rec.type_tag == RecorderTypeTag.SectionDeformation
        ):
            recorder_section_file_index_offsets[r] = len(recorder_section_file_indices)
            for eidx in range(rec.element_count):
                var elem_id = recorder_elements_pool[rec.element_offset + eidx]
                for sidx in range(rec.section_count):
                    var sec_no = recorder_sections_pool[rec.section_offset + sidx]
                    var filename = (
                        rec.output
                        + "_ele"
                        + String(elem_id)
                        + "_sec"
                        + String(sec_no)
                        + ".out"
                    )
                    var file_index = _output_buffer_index(
                        transient_output_files,
                        transient_output_buffers,
                        output_file_index,
                        filename,
                    )
                    recorder_section_file_indices.append(file_index)

        if rec.type_tag == RecorderTypeTag.Drift:
            var i_node = rec.i_node
            var j_node = rec.j_node
            if i_node >= len(id_to_index) or id_to_index[i_node] < 0:
                abort("drift i_node not found")
            if j_node >= len(id_to_index) or id_to_index[j_node] < 0:
                abort("drift j_node not found")
            var dof = rec.drift_dof
            var perp_dirn = rec.perp_dirn
            require_dof_in_range(dof, ndf, "drift recorder dof")
            if perp_dirn < 1 or perp_dirn > 3:
                abort("drift perp_dirn must be in 1..3")
            var i_idx = id_to_index[i_node]
            var j_idx = id_to_index[j_node]
            var node_i = typed_nodes[i_idx]
            var node_j = typed_nodes[j_idx]
            if perp_dirn == 3 and (not node_i.has_z or not node_j.has_z):
                abort("drift perp_dirn=3 requires z coordinates")
            var dx = node_j.z - node_i.z
            if perp_dirn == 1:
                dx = node_j.x - node_i.x
            elif perp_dirn == 2:
                dx = node_j.y - node_i.y
            if dx == 0.0:
                abort("drift denominator is zero")
            recorder_drift_i_dof_index[r] = node_dof_index(i_idx, dof, ndf)
            recorder_drift_j_dof_index[r] = node_dof_index(j_idx, dof, ndf)
            recorder_drift_denominator[r] = dx
            var filename = (
                rec.output
                + "_i"
                + String(i_node)
                + "_j"
                + String(j_node)
                + ".out"
            )
            recorder_drift_file_index[r] = _output_buffer_index(
                transient_output_files,
                transient_output_buffers,
                output_file_index,
                filename,
            )

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
        var uniaxial_states_base = uniaxial_states.copy()

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
        var reaction_internal_current = False
        var attempt_algorithm_tag = primary_algorithm_tag
        var attempt_algorithm_mode = primary_algorithm_mode
        var attempt_line_search_eta = primary_line_search_eta
        var attempt_test_mode = primary_test_mode
        var attempt_max_iters = max_iters
        var attempt_tol = tol
        var attempt_broyden_count = primary_broyden_count
        for attempt in range(retry_attempt_count):
            if attempt > 0:
                copy_float64_contiguous(u_f, u_step_base_f, free_count)
                _scatter_free_vector_to_full(free, u_f, u)
                force_basic_q = force_basic_q_base.copy()
                uniaxial_states = uniaxial_states_base.copy()
                fiber_section2d_revert_trial_runtime_all(fiber_section_defs)
                attempt_algorithm_tag = retry_algorithm_tags[attempt]
                attempt_algorithm_mode = retry_algorithm_modes[attempt]
                attempt_line_search_eta = retry_line_search_etas[attempt]
                attempt_test_mode = retry_test_modes[attempt]
                attempt_max_iters = retry_max_iters[attempt]
                attempt_tol = retry_tols[attempt]
                attempt_broyden_count = retry_broyden_counts[attempt]
                reaction_internal_current = False

            var tangent_initialized = False
            var damping_initialized = False
            var k_eff_initialized = False
            var k_eff_factored = False
            var broyden_history_s: List[List[Float64]] = []
            var broyden_history_z: List[List[Float64]] = []
            var broyden_prev_residual: List[Float64] = []
            var has_broyden_prev_residual = False
            for _ in range(attempt_max_iters):
                var iter_closed = False
                if runtime_metrics.enabled:
                    runtime_metrics.global_nonlinear_iterations += 1
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
        layered_shell_section_defs,
        layered_shell_section_index_by_id,
        layered_shell_section_uniaxial_offsets,
        layered_shell_section_uniaxial_counts,
        shell_elem_instance_offsets,
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
                    runtime_metrics,
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
                var broyden_refresh_interval = _default_broyden_count(
                    attempt_broyden_count
                )
                var need_tangent_matrix: Bool
                if attempt_algorithm_mode == NonlinearAlgorithmMode.Newton:
                    if attempt_algorithm_tag == AnalysisAlgorithmTag.Broyden:
                        need_tangent_matrix = (
                            not tangent_initialized
                            or broyden_refresh_interval <= 1
                            or len(broyden_history_s) >= broyden_refresh_interval
                        )
                    else:
                        need_tangent_matrix = True
                elif attempt_algorithm_mode == NonlinearAlgorithmMode.ModifiedNewton:
                    need_tangent_matrix = not tangent_initialized
                else:
                    need_tangent_matrix = not tangent_initialized
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
                reaction_internal_current = True
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
        layered_shell_section_defs,
        layered_shell_section_index_by_id,
        layered_shell_section_uniaxial_offsets,
        layered_shell_section_uniaxial_counts,
        shell_elem_instance_offsets,
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
        layered_shell_section_defs,
        layered_shell_section_index_by_id,
        layered_shell_section_uniaxial_offsets,
        layered_shell_section_uniaxial_counts,
        shell_elem_instance_offsets,
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
                    if need_tangent_matrix:
                        for i in range(free_count):
                            for j in range(free_count):
                                K_ff[i][j] = K[free[i]][free[j]]
                                if need_link_rayleigh_filter:
                                    K_damp_ff[i][j] = K_current_damp_ff[i][j]
                                else:
                                    K_damp_ff[i][j] = K_ff[i][j]
                        tangent_initialized = True
                        k_eff_factored = False
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
                        k_eff_factored = False
                else:
                    if not tangent_initialized:
                        for i in range(free_count):
                            for j in range(free_count):
                                K_ff[i][j] = K_init_ff[i][j]
                                K_damp_ff[i][j] = K_init_damp_ff[i][j]
                        tangent_initialized = True
                        k_eff_factored = False

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

                if need_tangent_matrix or not damping_initialized:
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

                var effective_matrix_changed = (
                    need_tangent_matrix or not k_eff_initialized or has_zero_length_dampers
                )

                var direct_newton_solve = (
                    attempt_algorithm_mode == NonlinearAlgorithmMode.Newton
                    and attempt_algorithm_tag != AnalysisAlgorithmTag.Broyden
                )
                if direct_newton_solve:
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
                    load_reduced_matrix(backend, K_ff)
                    if has_zero_length_dampers:
                        add_reduced_matrix(backend, K_zero_length_damp_ff)
                    add_reduced_matrix(backend, C_ff, a1)
                    add_diagonal(backend, M_f, a0)
                    factorize_loaded(backend, runtime_metrics)
                    k_eff_initialized = True
                    k_eff_factored = True
                    for i in range(free_count):
                        R_step[i] = R_f[i]
                    solve(backend, R_step, du_f)
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
                    var fac_start_us = 0
                    if do_profile:
                        var t_fac_start = Int(time.perf_counter_ns())
                        fac_start_us = (t_fac_start - t0) // 1000
                    var did_factor = effective_matrix_changed or not k_eff_factored
                    if did_factor:
                        load_reduced_matrix(backend, K_ff)
                        if has_zero_length_dampers:
                            add_reduced_matrix(backend, K_zero_length_damp_ff)
                        add_reduced_matrix(backend, C_ff, a1)
                        add_diagonal(backend, M_f, a0)
                        factorize_loaded(backend, runtime_metrics)
                        k_eff_initialized = True
                    if did_factor:
                        if do_profile:
                            var t_fac_end = Int(time.perf_counter_ns())
                            var fac_end_us = (t_fac_end - t0) // 1000
                            _append_event(
                                events,
                                events_need_comma,
                                "O",
                                frame_factorize,
                                fac_start_us,
                            )
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
                    solve(backend, lu_rhs, du_f)
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
                if attempt_algorithm_tag == AnalysisAlgorithmTag.Broyden:
                    if need_tangent_matrix:
                        broyden_history_s = []
                        broyden_history_z = []
                        broyden_prev_residual = []
                        has_broyden_prev_residual = False
                    elif has_broyden_prev_residual and len(broyden_history_s) > len(
                        broyden_history_z
                    ):
                        var delta_residual: List[Float64] = []
                        delta_residual.resize(free_count, 0.0)
                        for i in range(free_count):
                            delta_residual[i] = R_f[i] - broyden_prev_residual[i]
                        var current_z: List[Float64] = []
                        current_z.resize(free_count, 0.0)
                        for i in range(free_count):
                            lu_rhs[i] = delta_residual[i]
                        solve(backend, lu_rhs, current_z)
                        _broyden_update_direction(
                            du_f,
                            broyden_history_s,
                            broyden_history_z,
                            current_z,
                            len(broyden_history_z),
                        )
                        broyden_history_z.append(current_z^)
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
                if attempt_algorithm_tag == AnalysisAlgorithmTag.NewtonLineSearch:
                    var line_search_tol = _default_line_search_tol(attempt_line_search_eta)
                    var s0 = -dot_float64_contiguous(du_f, R_f, free_count)
                    if abs(s0) > 0.0:
                        _transient_residual(
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
                            typed_sections_by_id,
                            typed_materials_by_id,
                            id_to_index,
                            node_count,
                            ndf,
                            ndm,
                            u,
                            u_f,
                            u_n,
                            v_n,
                            a_n,
                            a0,
                            a2,
                            a3,
                            gamma,
                            dt,
                            P_ext_f,
                            M_f,
                            C_ff,
                            uniaxial_states,
                            uniaxial_defs,
                            uniaxial_state_defs,
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
        layered_shell_section_defs,
        layered_shell_section_index_by_id,
        layered_shell_section_uniaxial_offsets,
        layered_shell_section_uniaxial_counts,
        shell_elem_instance_offsets,
                            force_beam_column2d_scratch,
                            force_beam_column3d_scratch,
                            has_transformation_mpc,
                            rep_dof,
                            free,
                            free_count,
                            dampings,
                            time_series,
                            time_series_values,
                            time_series_times,
                            t,
                            has_zero_length_dampers,
                            R_f,
                            a_trial,
                            v_trial,
                            F_zero_length_damp_f,
                        )
                        var s = -dot_float64_contiguous(du_f, R_f, free_count)
                        var r0 = abs(s / s0)
                        var r = r0
                        var eta_prev = 1.0
                        var line_search_iter = 0
                        while r > line_search_tol and line_search_iter < 10:
                            line_search_iter += 1
                            var eta = _update_initial_interpolated_eta(
                                s0, s, r, r0, eta_prev
                            )
                            if eta == eta_prev:
                                break
                            var delta_eta = eta - eta_prev
                            for i in range(free_count):
                                u_f[i] += delta_eta * du_f[i]
                            _scatter_free_vector_to_full(free, u_f, u)
                            if has_transformation_mpc:
                                _enforce_equal_dof_values(u, rep_dof, constrained)
                            _transient_residual(
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
                                typed_sections_by_id,
                                typed_materials_by_id,
                                id_to_index,
                                node_count,
                                ndf,
                                ndm,
                                u,
                                u_f,
                                u_n,
                                v_n,
                                a_n,
                                a0,
                                a2,
                                a3,
                                gamma,
                                dt,
                                P_ext_f,
                                M_f,
                                C_ff,
                                uniaxial_states,
                                uniaxial_defs,
                                uniaxial_state_defs,
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
        layered_shell_section_defs,
        layered_shell_section_index_by_id,
        layered_shell_section_uniaxial_offsets,
        layered_shell_section_uniaxial_counts,
        shell_elem_instance_offsets,
                                force_beam_column2d_scratch,
                                force_beam_column3d_scratch,
                                has_transformation_mpc,
                                rep_dof,
                                free,
                                free_count,
                                dampings,
                                time_series,
                                time_series_values,
                                time_series_times,
                                t,
                                has_zero_length_dampers,
                                R_f,
                                a_trial,
                                v_trial,
                                F_zero_length_damp_f,
                            )
                            s = dot_float64_contiguous(du_f, R_f, free_count)
                            r = abs(s / s0)
                            eta_prev = eta
                        for i in range(free_count):
                            du_f[i] *= eta_prev
                if attempt_algorithm_tag == AnalysisAlgorithmTag.Broyden:
                    broyden_history_s.append(_copy_vector(du_f, free_count))
                    broyden_prev_residual = _copy_vector(R_f, free_count)
                    has_broyden_prev_residual = True
                reaction_internal_current = False

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
                    if max_diff <= attempt_tol:
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
        layered_shell_section_defs,
        layered_shell_section_index_by_id,
        layered_shell_section_uniaxial_offsets,
        layered_shell_section_uniaxial_counts,
        shell_elem_instance_offsets,
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
                runtime_metrics,
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
            reaction_internal_current = True
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
        layered_shell_section_defs,
        layered_shell_section_index_by_id,
        layered_shell_section_uniaxial_offsets,
        layered_shell_section_uniaxial_counts,
        shell_elem_instance_offsets,
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
        layered_shell_section_defs,
        layered_shell_section_index_by_id,
        layered_shell_section_uniaxial_offsets,
        layered_shell_section_uniaxial_counts,
        shell_elem_instance_offsets,
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
        _uniaxial_commit_active_states(uniaxial_states, elem_uniaxial_state_ids)
        fiber_section2d_commit_runtime_all(fiber_section_defs)
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
        var use_solver_internal_for_reaction = False
        if record_reactions:
            if reaction_internal_current:
                use_solver_internal_for_reaction = True
            else:
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
        layered_shell_section_defs,
        layered_shell_section_index_by_id,
        layered_shell_section_uniaxial_offsets,
        layered_shell_section_uniaxial_counts,
        shell_elem_instance_offsets,
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
                var node_dof_index_offset = recorder_node_dof_index_offsets[r]
                var node_file_index_offset = recorder_node_file_index_offsets[r]
                for nidx in range(rec.node_count):
                    var values: List[Float64] = []
                    values.resize(rec.dof_count, 0.0)
                    var dof_index_base = node_dof_index_offset + nidx * rec.dof_count
                    for j in range(rec.dof_count):
                        var dof_index = recorder_node_dof_indices[dof_index_base + j]
                        var value = u[dof_index]
                        if rec.type_tag == RecorderTypeTag.EnvelopeNodeAcceleration:
                            value = a[dof_index]
                            var ts_index = recorder_node_time_series_index[r]
                            if ts_index >= 0:
                                value += eval_time_series_input(
                                    time_series[ts_index],
                                    t,
                                    time_series_values,
                                    time_series_times,
                                )
                        values[j] = value
                    var file_index = recorder_node_file_indices[
                        node_file_index_offset + nidx
                    ]
                    if rec.type_tag == RecorderTypeTag.NodeDisplacement:
                        _append_output_at_index(
                            transient_output_buffers,
                            file_index,
                            _format_values_line(values),
                        )
                    else:
                        _update_envelope(
                            transient_output_files[file_index],
                            values,
                            envelope_files,
                            envelope_min,
                            envelope_max,
                            envelope_abs,
                        )
            elif rec.type_tag == RecorderTypeTag.ElementForce:
                var elem_index_offset = recorder_element_index_offsets[r]
                var elem_file_index_offset = recorder_element_file_index_offsets[r]
                for eidx in range(rec.element_count):
                    var elem_index = recorder_element_indices[elem_index_offset + eidx]
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
                    _append_output_at_index(
                        transient_output_buffers,
                        recorder_element_file_indices[elem_file_index_offset + eidx],
                        line,
                    )
            elif rec.type_tag == RecorderTypeTag.ElementLocalForce:
                var elem_index_offset = recorder_element_index_offsets[r]
                var elem_file_index_offset = recorder_element_file_index_offsets[r]
                for eidx in range(rec.element_count):
                    var elem_index = recorder_element_indices[elem_index_offset + eidx]
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
                    _append_output_at_index(
                        transient_output_buffers,
                        recorder_element_file_indices[elem_file_index_offset + eidx],
                        _format_values_line(values),
                    )
            elif rec.type_tag == RecorderTypeTag.ElementBasicForce:
                var elem_index_offset = recorder_element_index_offsets[r]
                var elem_file_index_offset = recorder_element_file_index_offsets[r]
                for eidx in range(rec.element_count):
                    var elem_index = recorder_element_indices[elem_index_offset + eidx]
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
                    _append_output_at_index(
                        transient_output_buffers,
                        recorder_element_file_indices[elem_file_index_offset + eidx],
                        _format_values_line(values),
                    )
            elif rec.type_tag == RecorderTypeTag.ElementDeformation:
                var elem_index_offset = recorder_element_index_offsets[r]
                var elem_file_index_offset = recorder_element_file_index_offsets[r]
                for eidx in range(rec.element_count):
                    var elem_index = recorder_element_indices[elem_index_offset + eidx]
                    var elem = typed_elements[elem_index]
                    var values = _element_deformation_for_recorder(
                        elem_index, elem, ndf, u, typed_nodes
                    )
                    _append_output_at_index(
                        transient_output_buffers,
                        recorder_element_file_indices[elem_file_index_offset + eidx],
                        _format_values_line(values),
                    )
            elif rec.type_tag == RecorderTypeTag.NodeReaction:
                if not record_reactions:
                    abort("internal error: reaction recorder flag mismatch")
                var node_dof_index_offset = recorder_node_dof_index_offsets[r]
                var node_file_index_offset = recorder_node_file_index_offsets[r]
                for nidx in range(rec.node_count):
                    var values: List[Float64] = []
                    values.resize(rec.dof_count, 0.0)
                    var dof_index_base = node_dof_index_offset + nidx * rec.dof_count
                    for j in range(rec.dof_count):
                        var dof_index = recorder_node_dof_indices[dof_index_base + j]
                        var internal_force = F_int_reaction[dof_index]
                        if use_solver_internal_for_reaction:
                            internal_force = F_int[dof_index]
                        values[j] = internal_force - F_ext_step[dof_index]
                    _append_output_at_index(
                        transient_output_buffers,
                        recorder_node_file_indices[node_file_index_offset + nidx],
                        _format_values_line(values),
                    )
            elif rec.type_tag == RecorderTypeTag.Drift:
                var value = (
                    u[recorder_drift_j_dof_index[r]] - u[recorder_drift_i_dof_index[r]]
                ) / recorder_drift_denominator[r]
                _append_output_at_index(
                    transient_output_buffers,
                    recorder_drift_file_index[r],
                    _format_values_line([value]),
                )
            elif rec.type_tag == RecorderTypeTag.EnvelopeElementForce:
                var elem_index_offset = recorder_element_index_offsets[r]
                var elem_file_index_offset = recorder_element_file_index_offsets[r]
                for eidx in range(rec.element_count):
                    var elem_index = recorder_element_indices[elem_index_offset + eidx]
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
                    var file_index = recorder_element_file_indices[
                        elem_file_index_offset + eidx
                    ]
                    _update_envelope(
                        transient_output_files[file_index],
                        elem_force_values[elem_index],
                        envelope_files,
                        envelope_min,
                        envelope_max,
                        envelope_abs,
                    )
            elif rec.type_tag == RecorderTypeTag.EnvelopeElementLocalForce:
                var elem_index_offset = recorder_element_index_offsets[r]
                var elem_file_index_offset = recorder_element_file_index_offsets[r]
                for eidx in range(rec.element_count):
                    var elem_index = recorder_element_indices[elem_index_offset + eidx]
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
                    var file_index = recorder_element_file_indices[
                        elem_file_index_offset + eidx
                    ]
                    _update_envelope(
                        transient_output_files[file_index],
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
                var elem_index_offset = recorder_element_index_offsets[r]
                var section_file_index_offset = recorder_section_file_index_offsets[r]
                for eidx in range(rec.element_count):
                    var elem_index = recorder_element_indices[elem_index_offset + eidx]
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
                        var file_index = recorder_section_file_indices[
                            section_file_index_offset + eidx * rec.section_count + sidx
                        ]
                        _append_output_at_index(
                            transient_output_buffers,
                            file_index,
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
    clear(backend)
    _flush_envelope_outputs(
        envelope_files,
        envelope_min,
        envelope_max,
        envelope_abs,
        transient_output_files,
        transient_output_buffers,
        output_file_index,
    )
