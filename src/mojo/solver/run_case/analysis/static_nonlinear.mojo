from collections import List
from elements import (
    ForceBeamColumn2dScratch,
    ForceBeamColumn3dScratch,
    reset_force_beam_column2d_scratch,
    reset_force_beam_column3d_scratch,
)
from math import sqrt
from materials import UniMaterialDef, UniMaterialState, uniaxial_commit, uniaxial_revert_trial
from os import abort
from python import Python

from solver.run_case.linear_solver_backend import (
    LinearSolverBackend,
    clear,
    initialize_symbolic_from_element_dof_map,
    initialize_structure,
    refactor_loaded_if_needed,
    refactor_if_needed,
    solve,
)
from solver.assembly import (
    assemble_global_stiffness_and_internal_native_soa,
    assemble_global_stiffness_and_internal_soa,
    assemble_internal_forces_typed_soa,
)
from solver.banded import banded_gaussian_elimination, banded_matrix, estimate_bandwidth_typed
from solver.dof import node_dof_index, require_dof_in_range
from solver.profile import (
    PROFILE_FRAME_UNIAXIAL_COPY_RESET,
    RuntimeProfileMetrics,
    _append_event,
)
from solver.simd_contiguous import dot_float64_contiguous, sum_sq_float64_contiguous
from solver.run_case.input_types import (
    AnalysisInput,
    ElementLoadInput,
    ElementInput,
    MaterialInput,
    NodeInput,
    RecorderInput,
    SectionInput,
    SolverAttemptInput,
)
from solver.time_series import TimeSeriesInput, eval_time_series_input, find_time_series_input
from sections import (
    FiberCell,
    FiberSection2dDef,
    FiberSection3dDef,
    LayeredShellSectionDef,
    fiber_section2d_commit_runtime_all,
    fiber_section2d_revert_trial_runtime_all,
)
from tag_types import (
    AnalysisAlgorithmTag,
    AnalysisSystemTag,
    ElementTypeTag,
    IntegratorTypeTag,
    NonlinearAlgorithmMode,
    NonlinearTestTypeTag,
    RecorderTypeTag,
    TimeSeriesTypeTag,
)

from solver.run_case.helpers import (
    _append_output,
    _collapse_matrix_by_mpc,
    _collapse_vector_by_mpc,
    _drift_value,
    _element_basic_force_for_recorder,
    _element_deformation_for_recorder,
    _element_force_global_for_recorder,
    _force_beam_column2d_force_global_from_basic_state,
    _force_beam_column2d_section_response_from_basic_state,
    _element_local_force_for_recorder,
    _enforce_mpc_values,
    _flush_envelope_outputs,
    _format_values_line,
    _has_recorder_type,
    _section_response_for_recorder,
    _sync_force_beam_column2d_committed_basic_states,
    _write_run_progress,
    _update_envelope,
)
from solver.run_case.load_state import (
    build_active_element_load_state,
    build_active_nodal_load,
)
from solver.run_case.nonlinear_krylov import (
    copy_krylov_vector,
    default_krylov_max_dim,
    krylov_apply_acceleration,
    krylov_push_iteration_state,
)


fn _static_nonlinear_algorithm_mode(algorithm_tag: Int, algorithm: String, label: String) -> Int:
    if algorithm_tag == AnalysisAlgorithmTag.Newton:
        return NonlinearAlgorithmMode.Newton
    if algorithm_tag == AnalysisAlgorithmTag.ModifiedNewton:
        return NonlinearAlgorithmMode.ModifiedNewton
    if algorithm_tag == AnalysisAlgorithmTag.ModifiedNewtonInitial:
        return NonlinearAlgorithmMode.ModifiedNewtonInitial
    if (
        algorithm_tag == AnalysisAlgorithmTag.Broyden
        or algorithm_tag == AnalysisAlgorithmTag.NewtonLineSearch
        or algorithm_tag == AnalysisAlgorithmTag.KrylovNewton
    ):
        # These algorithms still use Newton tangents, but add their own
        # per-iteration correction logic below.
        return NonlinearAlgorithmMode.Newton
    abort("unsupported " + label + " algorithm: " + algorithm)
    return NonlinearAlgorithmMode.Unknown


fn _static_nonlinear_test_mode(test_type_tag: Int, test_type: String, label: String) -> Int:
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


@always_inline
fn _dense_flat_index(size: Int, row: Int, col: Int) -> Int:
    return row * size + col


fn _ensure_dense_flat_storage(mut values: List[Float64], size: Int):
    var count = size * size
    if len(values) != count:
        values.resize(count, 0.0)


fn _load_displacement_control_augmented_dense_flat(
    mut matrix_flat: List[Float64],
    K_ff: List[List[Float64]],
    load_column_free: List[Float64],
    control_free: Int,
):
    var free_count = len(load_column_free)
    var aug_size = free_count + 1
    _ensure_dense_flat_storage(matrix_flat, aug_size)
    for i in range(aug_size * aug_size):
        matrix_flat[i] = 0.0
    for i in range(free_count):
        for j in range(free_count):
            matrix_flat[_dense_flat_index(aug_size, i, j)] = K_ff[i][j]
        matrix_flat[_dense_flat_index(aug_size, i, free_count)] = load_column_free[i]
    matrix_flat[_dense_flat_index(aug_size, free_count, control_free)] = 1.0


fn _solve_linear_system_dense_flat(
    matrix_flat: List[Float64],
    rhs: List[Float64],
    mut solution_out: List[Float64],
    mut factor_work: List[Float64],
    mut rhs_work: List[Float64],
) -> Bool:
    var n = len(rhs)
    if len(matrix_flat) != n * n:
        return False
    solution_out.resize(n, 0.0)
    _ensure_dense_flat_storage(factor_work, n)
    rhs_work.resize(n, 0.0)
    for i in range(n * n):
        factor_work[i] = matrix_flat[i]
    for i in range(n):
        rhs_work[i] = rhs[i]

    var eps = 1.0e-18
    for i in range(n):
        var pivot = i
        var max_val = abs(factor_work[_dense_flat_index(n, i, i)])
        for row in range(i + 1, n):
            var candidate = abs(factor_work[_dense_flat_index(n, row, i)])
            if candidate > max_val:
                max_val = candidate
                pivot = row
        if max_val <= eps:
            return False
        if pivot != i:
            var pivot_base = pivot * n
            var row_base = i * n
            for col in range(n):
                var tmp = factor_work[row_base + col]
                factor_work[row_base + col] = factor_work[pivot_base + col]
                factor_work[pivot_base + col] = tmp
            var rhs_tmp = rhs_work[i]
            rhs_work[i] = rhs_work[pivot]
            rhs_work[pivot] = rhs_tmp

        var piv = factor_work[_dense_flat_index(n, i, i)]
        for j in range(i, n):
            factor_work[_dense_flat_index(n, i, j)] /= piv
        rhs_work[i] /= piv

        for row in range(i + 1, n):
            var factor = factor_work[_dense_flat_index(n, row, i)]
            if factor == 0.0:
                continue
            for col in range(i, n):
                factor_work[_dense_flat_index(n, row, col)] -= (
                    factor * factor_work[_dense_flat_index(n, i, col)]
                )
            rhs_work[row] -= factor * rhs_work[i]

    for i in range(n - 1, -1, -1):
        var sum = rhs_work[i]
        for j in range(i + 1, n):
            sum -= factor_work[_dense_flat_index(n, i, j)] * solution_out[j]
        solution_out[i] = sum
    return True


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
    var current_pair = count
    for i in range(current_pair):
        var p = -dot_float64_contiguous(history_s[i], history_z[i], len(du))
        if abs(p) < eps:
            break
        var sdotdu = dot_float64_contiguous(history_s[i], du, len(du))
        for j in range(len(du)):
            du[j] += ((history_s[i][j] + history_z[i][j]) * sdotdu) / p
    var current_p = -dot_float64_contiguous(history_s[current_pair], current_z, len(du))
    if abs(current_p) >= eps:
        var current_sdotdu = dot_float64_contiguous(history_s[current_pair], du, len(du))
        for j in range(len(du)):
            du[j] += (
                ((history_s[current_pair][j] + current_z[j]) * current_sdotdu)
                / current_p
            )


fn _solve_displacement_control_augmented_from_factorized_backend(
    mut backend: LinearSolverBackend,
    rhs_free: List[Float64],
    load_column_free: List[Float64],
    control_free: Int,
    disp_constraint_rhs: Float64,
    mut solution_aug: List[Float64],
    mut rhs_solve: List[Float64],
    mut load_solve: List[Float64],
) -> Bool:
    var free_count = len(rhs_free)
    if len(load_column_free) != free_count:
        abort("DisplacementControl load column size mismatch")
    if control_free < 0 or control_free >= free_count:
        abort("DisplacementControl control free index out of range")
    solve(backend, rhs_free, rhs_solve)
    solve(backend, load_column_free, load_solve)
    var denom = load_solve[control_free]
    if abs(denom) <= 1.0e-14:
        return False
    var delta_lambda = (rhs_solve[control_free] - disp_constraint_rhs) / denom
    if len(solution_aug) != free_count + 1:
        solution_aug.resize(free_count + 1, 0.0)
    for i in range(free_count):
        solution_aug[i] = rhs_solve[i] - load_solve[i] * delta_lambda
    solution_aug[free_count] = delta_lambda
    return True


fn _static_load_control_residual(
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
    uniaxial_defs: List[UniMaterialDef],
    uniaxial_state_defs: List[Int],
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
    F_step: List[Float64],
    free: List[Int],
    free_count: Int,
    has_transformation_mpc: Bool,
    mpc_row_offsets: List[Int],
    mpc_dof_pool: List[Int],
    mpc_coeff_pool: List[Float64],
    mut residual: List[Float64],
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
        F_int = _collapse_vector_by_mpc(
            F_int, mpc_row_offsets, mpc_dof_pool, mpc_coeff_pool
        )
    for i in range(free_count):
        var row_i = free[i]
        residual[i] = F_step[row_i] - F_int[row_i]


fn _static_displacement_control_residual(
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
    typed_sections_by_id: List[SectionInput],
    typed_materials_by_id: List[MaterialInput],
    id_to_index: List[Int],
    node_count: Int,
    ndf: Int,
    ndm: Int,
    u: List[Float64],
    uniaxial_defs: List[UniMaterialDef],
    uniaxial_state_defs: List[Int],
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
    free: List[Int],
    free_count: Int,
    has_transformation_mpc: Bool,
    mpc_row_offsets: List[Int],
    mpc_dof_pool: List[Int],
    mpc_coeff_pool: List[Float64],
    F_const_free: List[Float64],
    F_pattern_free: List[Float64],
    load_factor: Float64,
    attempt_du: Float64,
    control_idx: Int,
    u_base: List[Float64],
    mut residual_aug: List[Float64],
) raises:
    var empty_element_loads: List[ElementLoadInput] = []
    var empty_offsets: List[Int] = []
    var empty_pool: List[Int] = []
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
        empty_element_loads,
        empty_offsets,
        empty_pool,
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
        F_int = _collapse_vector_by_mpc(
            F_int, mpc_row_offsets, mpc_dof_pool, mpc_coeff_pool
        )
    for i in range(free_count):
        residual_aug[i] = (
            F_const_free[i] + load_factor * F_pattern_free[i] - F_int[free[i]]
        )
    residual_aug[free_count] = attempt_du - (u[control_idx] - u_base[control_idx])


fn _append_static_retry_attempt(
    algorithm: String,
    algorithm_tag: Int,
    broyden_count: Int,
    line_search_eta: Float64,
    krylov_max_dim: Int,
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
    mut retry_krylov_max_dims: List[Int],
):
    if len(algorithm) == 0:
        return
    retry_algorithm_tags.append(algorithm_tag)
    retry_algorithm_modes.append(
        _static_nonlinear_algorithm_mode(algorithm_tag, algorithm, label)
    )
    retry_test_modes.append(_static_nonlinear_test_mode(test_type_tag, test_type, label))
    if max_iters < 1:
        abort(label + "_max_iters must be >= 1")
    retry_max_iters.append(max_iters)
    retry_tols.append(tol)
    retry_broyden_counts.append(broyden_count)
    retry_line_search_etas.append(line_search_eta)
    retry_krylov_max_dims.append(krylov_max_dim)


fn _append_static_retry_attempt_from_input(
    attempt: SolverAttemptInput,
    label: String,
    mut retry_algorithm_tags: List[Int],
    mut retry_algorithm_modes: List[Int],
    mut retry_test_modes: List[Int],
    mut retry_max_iters: List[Int],
    mut retry_tols: List[Float64],
    mut retry_broyden_counts: List[Int],
    mut retry_line_search_etas: List[Float64],
    mut retry_krylov_max_dims: List[Int],
):
    _append_static_retry_attempt(
        attempt.algorithm,
        attempt.algorithm_tag,
        attempt.broyden_count,
        attempt.line_search_eta,
        attempt.krylov_max_dim,
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
        retry_krylov_max_dims,
    )


fn _collect_static_solver_chain(
    analysis: AnalysisInput,
    analysis_solver_chain_pool: List[SolverAttemptInput],
    has_force_beam_column2d: Bool,
    prefer_modified_newton_initial: Bool,
    mut retry_algorithm_tags: List[Int],
    mut retry_algorithm_modes: List[Int],
    mut retry_test_modes: List[Int],
    mut retry_max_iters: List[Int],
    mut retry_tols: List[Float64],
    mut retry_broyden_counts: List[Int],
    mut retry_line_search_etas: List[Float64],
    mut retry_krylov_max_dims: List[Int],
):
    for i in range(analysis.solver_chain_count):
        _append_static_retry_attempt_from_input(
            analysis_solver_chain_pool[analysis.solver_chain_offset + i],
            "static_nonlinear solver_chain",
            retry_algorithm_tags,
            retry_algorithm_modes,
            retry_test_modes,
            retry_max_iters,
            retry_tols,
            retry_broyden_counts,
            retry_line_search_etas,
            retry_krylov_max_dims,
        )
    if len(retry_algorithm_modes) == 0:
        _append_static_retry_attempt(
            analysis.algorithm,
            analysis.algorithm_tag,
            0,
            1.0,
            0,
            analysis.test_type,
            analysis.test_type_tag,
            analysis.max_iters,
            analysis.tol,
            "static_nonlinear primary",
            retry_algorithm_tags,
            retry_algorithm_modes,
            retry_test_modes,
            retry_max_iters,
            retry_tols,
            retry_broyden_counts,
            retry_line_search_etas,
            retry_krylov_max_dims,
        )
    if analysis.has_solver_chain_override or len(retry_algorithm_modes) != 1:
        return

    var primary_algorithm_mode = retry_algorithm_modes[0]
    var auto_algorithm = ""
    var auto_algorithm_tag = AnalysisAlgorithmTag.Unknown
    var auto_test_type = analysis.test_type
    var auto_test_type_tag = analysis.test_type_tag
    var auto_max_iters = retry_max_iters[0]
    var auto_tol = retry_tols[0]
    if has_force_beam_column2d and (
        primary_algorithm_mode == NonlinearAlgorithmMode.Newton
        or primary_algorithm_mode == NonlinearAlgorithmMode.ModifiedNewton
    ):
        if primary_algorithm_mode == NonlinearAlgorithmMode.Newton:
            if prefer_modified_newton_initial:
                auto_algorithm = "ModifiedNewtonInitial"
                auto_algorithm_tag = AnalysisAlgorithmTag.ModifiedNewtonInitial
            else:
                auto_algorithm = "ModifiedNewton"
                auto_algorithm_tag = AnalysisAlgorithmTag.ModifiedNewton
        else:
            auto_algorithm = "ModifiedNewtonInitial"
            auto_algorithm_tag = AnalysisAlgorithmTag.ModifiedNewtonInitial
        auto_test_type = "NormDispIncr"
        auto_test_type_tag = NonlinearTestTypeTag.NormDispIncr
        if auto_max_iters < 100:
            auto_max_iters = 100
        if auto_max_iters < retry_max_iters[0] * 5:
            auto_max_iters = retry_max_iters[0] * 5
        auto_tol = retry_tols[0]
        if not prefer_modified_newton_initial and auto_tol < 1.0e-10:
            auto_tol = 1.0e-10
    elif primary_algorithm_mode != NonlinearAlgorithmMode.Newton:
        auto_algorithm = "Newton"
        auto_algorithm_tag = AnalysisAlgorithmTag.Newton
    if len(auto_algorithm) == 0:
        return
    _append_static_retry_attempt(
        auto_algorithm,
        auto_algorithm_tag,
        0,
        1.0,
        0,
        auto_test_type,
        auto_test_type_tag,
        auto_max_iters,
        auto_tol,
        "static_nonlinear auto_fallback",
        retry_algorithm_tags,
        retry_algorithm_modes,
        retry_test_modes,
        retry_max_iters,
        retry_tols,
        retry_broyden_counts,
        retry_line_search_etas,
        retry_krylov_max_dims,
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


fn run_static_nonlinear_load_control(
    analysis: AnalysisInput,
    steps: Int,
    ts_index: Int,
    time_series: List[TimeSeriesInput],
    time_series_values: List[Float64],
    time_series_times: List[Float64],
    analysis_solver_chain_pool: List[SolverAttemptInput],
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
    total_dofs: Int,
    F_const: List[Float64],
    F_pattern: List[Float64],
    free: List[Int],
    free_index: List[Int],
    recorders: List[RecorderInput],
    recorder_nodes_pool: List[Int],
    recorder_elements_pool: List[Int],
    recorder_dofs_pool: List[Int],
    recorder_sections_pool: List[Int],
    elem_id_to_index: List[Int],
    mut static_output_files: List[String],
    mut static_output_buffers: List[String],
    progress_path: String,
    progress_stage_number: Int,
    progress_stage_count: Int,
    do_profile: Bool,
    t0: Int,
    mut events: String,
    mut events_need_comma: Bool,
    frame_assemble_stiffness: Int,
    frame_assemble_uniaxial: Int,
    frame_assemble_fiber: Int,
    frame_kff_extract: Int,
    frame_solve_nonlinear: Int,
    frame_nonlinear_step: Int,
    frame_nonlinear_iter: Int,
    frame_uniaxial_revert_all: Int,
    frame_uniaxial_commit_all: Int,
    has_transformation_mpc: Bool,
    mpc_slave_dof: List[Bool],
    mpc_row_offsets: List[Int],
    mpc_dof_pool: List[Int],
    mpc_coeff_pool: List[Float64],
    constrained: List[Bool],
    mut runtime_metrics: RuntimeProfileMetrics,
) raises:
    var time = Python.import_module("time")
    _write_run_progress(
        progress_path,
        "running_stage",
        "static_nonlinear",
        progress_stage_number,
        progress_stage_count,
        0,
        steps,
    )
    var asm_dof_map6: List[Int] = []
    var asm_dof_map12: List[Int] = []
    var asm_u_elem6: List[Float64] = []
    var force_beam_column2d_scratch = ForceBeamColumn2dScratch()
    var force_beam_column3d_scratch = ForceBeamColumn3dScratch()
    var has_force_beam_column2d = False
    for i in range(len(typed_elements)):
        if typed_elements[i].type_tag == ElementTypeTag.ForceBeamColumn2d:
            has_force_beam_column2d = True
            break
    var retry_algorithm_tags: List[Int] = []
    var retry_algorithm_modes: List[Int] = []
    var retry_test_modes: List[Int] = []
    var retry_max_iters: List[Int] = []
    var retry_tols: List[Float64] = []
    var retry_broyden_counts: List[Int] = []
    var retry_line_search_etas: List[Float64] = []
    var retry_krylov_max_dims: List[Int] = []
    _collect_static_solver_chain(
        analysis,
        analysis_solver_chain_pool,
        has_force_beam_column2d,
        False,
        retry_algorithm_tags,
        retry_algorithm_modes,
        retry_test_modes,
        retry_max_iters,
        retry_tols,
        retry_broyden_counts,
        retry_line_search_etas,
        retry_krylov_max_dims,
    )
    if len(retry_algorithm_modes) == 0:
        abort("static_nonlinear solver_chain must contain at least one attempt")
    var retry_attempt_count = len(retry_algorithm_modes)
    var primary_algorithm_tag = retry_algorithm_tags[0]
    var primary_algorithm_mode = retry_algorithm_modes[0]
    var primary_test_mode = retry_test_modes[0]
    var max_iters = retry_max_iters[0]
    var tol = retry_tols[0]
    var primary_broyden_count = retry_broyden_counts[0]
    var primary_line_search_eta = retry_line_search_etas[0]
    var primary_krylov_max_dim = default_krylov_max_dim(retry_krylov_max_dims[0])
    var chain_has_modified_newton_initial = False
    for i in range(retry_attempt_count):
        if retry_algorithm_modes[i] == NonlinearAlgorithmMode.ModifiedNewtonInitial:
            chain_has_modified_newton_initial = True
            break
    var free_count = len(free)
    var F_const_free: List[Float64] = []
    F_const_free.resize(free_count, 0.0)
    var F_pattern_free: List[Float64] = []
    F_pattern_free.resize(free_count, 0.0)
    for i in range(free_count):
        F_const_free[i] = F_const[free[i]]
        F_pattern_free[i] = F_pattern[free[i]]

    var F_int: List[Float64] = []
    F_int.resize(total_dofs, 0.0)

    var integrator_tag = analysis.integrator_tag
    if integrator_tag == IntegratorTypeTag.Unknown:
        integrator_tag = IntegratorTypeTag.LoadControl
    var use_banded_loadcontrol = (
        analysis.system_tag == AnalysisSystemTag.BandGeneral
        and not has_transformation_mpc
        and integrator_tag == IntegratorTypeTag.LoadControl
        and primary_algorithm_mode != NonlinearAlgorithmMode.ModifiedNewtonInitial
        and not chain_has_modified_newton_initial
    )
    var bw_nl = 0
    if use_banded_loadcontrol:
        bw_nl = estimate_bandwidth_typed(typed_elements, free_index)
        if bw_nl > free_count - 1:
            bw_nl = free_count - 1
        if runtime_metrics.enabled:
            runtime_metrics.active_bandwidth = bw_nl * 2 + 1
            runtime_metrics.active_nnz = 0
            runtime_metrics.active_profile_size = 0
    var K_ff: List[List[Float64]] = []
    var K_init_ff: List[List[Float64]] = []
    var backend = LinearSolverBackend()
    var native_direct_backend_available = (
        not has_transformation_mpc
        and not use_banded_loadcontrol
        and (
            analysis.system_tag == AnalysisSystemTag.FullGeneral
            or analysis.system_tag == AnalysisSystemTag.BandGeneral
            or analysis.system_tag == AnalysisSystemTag.BandSPD
            or analysis.system_tag == AnalysisSystemTag.ProfileSPD
            or analysis.system_tag == AnalysisSystemTag.SuperLU
            or analysis.system_tag == AnalysisSystemTag.UmfPack
            or analysis.system_tag == AnalysisSystemTag.SparseSYM
        )
    )
    var needs_dense_global_matrix = (
        use_banded_loadcontrol
        or not native_direct_backend_available
        or chain_has_modified_newton_initial
    )
    var K: List[List[Float64]] = []
    if needs_dense_global_matrix:
        for _ in range(total_dofs):
            var row: List[Float64] = []
            row.resize(total_dofs, 0.0)
            K.append(row^)
    var elem_free_pool_native: List[Int] = []
    if native_direct_backend_available:
        elem_free_pool_native.resize(len(elem_dof_pool), -1)
        for i in range(len(elem_dof_pool)):
            var dof = elem_dof_pool[i]
            if dof >= 0 and dof < len(free_index):
                elem_free_pool_native[i] = free_index[dof]
    var lu_rhs: List[Float64] = []
    var u_f_work: List[Float64] = []
    if not use_banded_loadcontrol:
        initialize_structure(backend, analysis, free_count)
        initialize_symbolic_from_element_dof_map(backend, elem_dof_offsets, elem_dof_pool, free_index)
        for _ in range(free_count):
            var row_ff: List[Float64] = []
            row_ff.resize(free_count, 0.0)
            K_ff.append(row_ff^)
            var row_kinit: List[Float64] = []
            row_kinit.resize(free_count, 0.0)
            K_init_ff.append(row_kinit^)
        lu_rhs.resize(free_count, 0.0)
        u_f_work.resize(free_count, 0.0)
    var K_ff_banded: List[List[Float64]] = []
    var K_ff_banded_step: List[List[Float64]] = []
    var banded_rhs: List[Float64] = []
    if use_banded_loadcontrol:
        K_ff_banded = banded_matrix(free_count, bw_nl)
        K_ff_banded_step = banded_matrix(free_count, bw_nl)
        banded_rhs.resize(free_count, 0.0)
    if chain_has_modified_newton_initial:
        var initial_element_load_state = build_active_element_load_state(
            const_element_loads,
            pattern_element_loads,
            0.0,
            typed_elements,
            elem_id_to_index,
            ndm,
            ndf,
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
            initial_element_load_state.element_loads,
            initial_element_load_state.elem_load_offsets,
            initial_element_load_state.elem_load_pool,
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
        if has_transformation_mpc:
            K = _collapse_matrix_by_mpc(K, mpc_row_offsets, mpc_dof_pool, mpc_coeff_pool)
        if not use_banded_loadcontrol:
            for i in range(free_count):
                for j in range(free_count):
                    K_init_ff[i][j] = K[free[i]][free[j]]
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
    var F_f: List[Float64] = []
    F_f.resize(free_count, 0.0)
    var has_reaction_recorder = _has_recorder_type(recorders, RecorderTypeTag.NodeReaction)
    var envelope_files: List[String] = []
    var envelope_min: List[List[Float64]] = []
    var envelope_max: List[List[Float64]] = []
    var envelope_abs: List[List[Float64]] = []
    var base_step_size = 1.0 / Float64(steps)
    if analysis.has_integrator_step:
        base_step_size = analysis.integrator_step
    var adaptive_step = (
        analysis.has_integrator_num_iter
        or analysis.has_integrator_min_step
        or analysis.has_integrator_max_step
    )
    var step_target_iters = analysis.integrator_num_iter
    if step_target_iters < 1:
        step_target_iters = 1
    var last_step_iters = step_target_iters
    var min_step_size = base_step_size
    if analysis.has_integrator_min_step:
        min_step_size = analysis.integrator_min_step
    var max_step_size = base_step_size
    if analysis.has_integrator_max_step:
        max_step_size = analysis.integrator_max_step
    var current_step_size = base_step_size
    var load_control_time = 0.0
    for step in range(steps):
        _write_run_progress(
            progress_path,
            "running_step",
            "static_nonlinear",
            progress_stage_number,
            progress_stage_count,
            step + 1,
            steps,
        )
        if do_profile:
            var t_step_start = Int(time.perf_counter_ns())
            var step_start_us = (t_step_start - t0) // 1000
            _append_event(
                events, events_need_comma, "O", frame_nonlinear_step, step_start_us
            )
        if has_transformation_mpc:
            _enforce_mpc_values(
                u,
                constrained,
                mpc_slave_dof,
                mpc_row_offsets,
                mpc_dof_pool,
                mpc_coeff_pool,
            )
        var step_size = current_step_size
        if adaptive_step:
            if last_step_iters > 0:
                step_size = (
                    current_step_size
                    * Float64(step_target_iters)
                    / Float64(last_step_iters)
                )
            var step_sign = 1.0
            if step_size < 0.0:
                step_sign = -1.0
            var step_mag = abs(step_size)
            var min_mag = abs(min_step_size)
            var max_mag = abs(max_step_size)
            if max_mag < min_mag:
                max_mag = min_mag
            if step_mag < min_mag:
                step_mag = min_mag
            if step_mag > max_mag:
                step_mag = max_mag
            step_size = step_sign * step_mag
        current_step_size = step_size
        load_control_time += step_size
        var scale = load_control_time
        if ts_index >= 0:
            scale = eval_time_series_input(
                time_series[ts_index], scale, time_series_values, time_series_times
            )
        var F_step = build_active_nodal_load(F_const, F_pattern, scale)
        var active_element_load_state = build_active_element_load_state(
            const_element_loads,
            pattern_element_loads,
            scale,
            typed_elements,
            elem_id_to_index,
            ndm,
            ndf,
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
        var u_base = u.copy()
        var force_basic_q_base = force_basic_q.copy()
        var uniaxial_states_base = uniaxial_states.copy()
        var converged = False
        var attempt_algorithm_tag = primary_algorithm_tag
        var attempt_algorithm_mode = primary_algorithm_mode
        var attempt_line_search_eta = primary_line_search_eta
        var attempt_test_mode = primary_test_mode
        var attempt_max_iters = max_iters
        var attempt_tol = tol
        var attempt_broyden_count = primary_broyden_count
        var attempt_krylov_max_dim = primary_krylov_max_dim
        var step_converged_iters = 0
        for attempt in range(retry_attempt_count):
            if attempt > 0:
                for i in range(total_dofs):
                    u[i] = u_base[i]
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
                attempt_krylov_max_dim = default_krylov_max_dim(
                    retry_krylov_max_dims[attempt]
                )

            var tangent_initialized = False
            var tangent_factored = False
            var broyden_history_s: List[List[Float64]] = []
            var broyden_history_z: List[List[Float64]] = []
            var broyden_prev_residual: List[Float64] = []
            var has_broyden_prev_residual = False
            var krylov_history_v: List[List[Float64]] = []
            var krylov_history_av: List[List[Float64]] = []
            var krylov_normal_matrix_flat: List[Float64] = []
            var krylov_normal_rhs: List[Float64] = []
            var krylov_coeffs: List[Float64] = []
            var krylov_factor_work: List[Float64] = []
            var krylov_rhs_work: List[Float64] = []
            for iter_idx in range(attempt_max_iters):
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
                var broyden_refresh_interval = _default_broyden_count(
                    attempt_broyden_count
                )
                var krylov_max_dim = attempt_krylov_max_dim
                if krylov_max_dim > free_count:
                    krylov_max_dim = free_count
                var refresh_tangent: Bool
                if attempt_algorithm_mode == NonlinearAlgorithmMode.Newton:
                    if attempt_algorithm_tag == AnalysisAlgorithmTag.Broyden:
                        refresh_tangent = (
                            not tangent_initialized
                            or broyden_refresh_interval <= 1
                            or len(broyden_history_s) >= broyden_refresh_interval
                        )
                    elif attempt_algorithm_tag == AnalysisAlgorithmTag.KrylovNewton:
                        refresh_tangent = (
                            not tangent_initialized
                            or len(krylov_history_v) > krylov_max_dim
                        )
                    else:
                        refresh_tangent = True
                elif attempt_algorithm_mode == NonlinearAlgorithmMode.ModifiedNewton:
                    refresh_tangent = not tangent_initialized
                else:
                    refresh_tangent = not tangent_initialized
                if refresh_tangent and attempt_algorithm_tag == AnalysisAlgorithmTag.KrylovNewton:
                    krylov_history_v = []
                    krylov_history_av = []
                var use_native_direct_solve = (
                    native_direct_backend_available
                    and attempt_algorithm_mode != NonlinearAlgorithmMode.ModifiedNewtonInitial
                )
                var use_native_tangent_assembly = (
                    use_native_direct_solve and refresh_tangent
                )
                var use_native_internal_force_only = (
                    use_native_direct_solve and not use_native_tangent_assembly
                )
                if do_profile:
                    var t_asm_start = Int(time.perf_counter_ns())
                    var asm_start_us = (t_asm_start - t0) // 1000
                    _append_event(
                        events,
                        events_need_comma,
                        "O",
                        frame_assemble_stiffness,
                        asm_start_us,
                    )
                if use_native_tangent_assembly:
                    assemble_global_stiffness_and_internal_native_soa(
                        typed_nodes,
                        typed_elements,
                        node_x,
                        node_y,
                        node_z,
                        elem_dof_offsets,
                        elem_dof_pool,
                        elem_dof_offsets,
                        elem_free_pool_native,
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
                        backend,
                        F_int,
                        do_profile,
                        t0,
                        events,
                        events_need_comma,
                        frame_assemble_uniaxial,
                        frame_assemble_fiber,
                        runtime_metrics,
                    )
                elif use_native_internal_force_only:
                    F_int = assemble_internal_forces_typed_soa(
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
                else:
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
                    if has_transformation_mpc:
                        K = _collapse_matrix_by_mpc(
                            K, mpc_row_offsets, mpc_dof_pool, mpc_coeff_pool
                        )
                        F_int = _collapse_vector_by_mpc(
                            F_int, mpc_row_offsets, mpc_dof_pool, mpc_coeff_pool
                        )
                if do_profile:
                    var t_asm_end = Int(time.perf_counter_ns())
                    var asm_end_us = (t_asm_end - t0) // 1000
                    _append_event(
                        events,
                        events_need_comma,
                        "C",
                        frame_assemble_stiffness,
                        asm_end_us,
                    )
                if do_profile:
                    var t_kff_start = Int(time.perf_counter_ns())
                    var kff_start_us = (t_kff_start - t0) // 1000
                    _append_event(
                        events, events_need_comma, "O", frame_kff_extract, kff_start_us
                    )
                if use_banded_loadcontrol:
                    if refresh_tangent:
                        var width = bw_nl * 2 + 1
                        for i in range(free_count):
                            for j in range(width):
                                K_ff_banded[i][j] = 0.0
                            var row_i = free[i]
                            F_f[i] = F_step[row_i] - F_int[row_i]
                            var j0 = i - bw_nl
                            if j0 < 0:
                                j0 = 0
                            var j1 = i + bw_nl
                            if j1 > free_count - 1:
                                j1 = free_count - 1
                            for j in range(j0, j1 + 1):
                                K_ff_banded[i][j - i + bw_nl] = K[row_i][free[j]]
                        tangent_initialized = True
                    else:
                        for i in range(free_count):
                            F_f[i] = F_step[free[i]] - F_int[free[i]]
                else:
                    if attempt_algorithm_mode == NonlinearAlgorithmMode.Newton:
                        if refresh_tangent:
                            for i in range(free_count):
                                var row_i = free[i]
                                F_f[i] = F_step[row_i] - F_int[row_i]
                                if not use_native_direct_solve:
                                    for j in range(free_count):
                                        K_ff[i][j] = K[row_i][free[j]]
                            tangent_initialized = True
                            tangent_factored = False
                        else:
                            for i in range(free_count):
                                F_f[i] = F_step[free[i]] - F_int[free[i]]
                    elif attempt_algorithm_mode == NonlinearAlgorithmMode.ModifiedNewton:
                        if not tangent_initialized:
                            for i in range(free_count):
                                var row_i = free[i]
                                F_f[i] = F_step[row_i] - F_int[row_i]
                                if not use_native_direct_solve:
                                    for j in range(free_count):
                                        K_ff[i][j] = K[row_i][free[j]]
                            tangent_initialized = True
                            tangent_factored = False
                        else:
                            for i in range(free_count):
                                F_f[i] = F_step[free[i]] - F_int[free[i]]
                    else:
                        if not tangent_initialized:
                            for i in range(free_count):
                                for j in range(free_count):
                                    K_ff[i][j] = K_init_ff[i][j]
                            tangent_initialized = True
                            tangent_factored = False
                        for i in range(free_count):
                            F_f[i] = F_step[free[i]] - F_int[free[i]]
                if do_profile:
                    var t_kff_end = Int(time.perf_counter_ns())
                    var kff_end_us = (t_kff_end - t0) // 1000
                    _append_event(
                        events, events_need_comma, "C", frame_kff_extract, kff_end_us
                    )

                var residual_norm_dbg = sqrt(sum_sq_float64_contiguous(F_f, free_count))

                if attempt_test_mode == 2:
                    if residual_norm_dbg <= attempt_tol:
                        if do_profile:
                            var t_iter_end = Int(time.perf_counter_ns())
                            var iter_end_us = (t_iter_end - t0) // 1000
                            _append_event(
                                events,
                                events_need_comma,
                                "C",
                                frame_nonlinear_iter,
                                iter_end_us,
                            )
                        step_converged_iters = iter_idx + 1
                        converged = True
                        break

                if do_profile:
                    var t_solve_nl_start = Int(time.perf_counter_ns())
                    var solve_nl_start_us = (t_solve_nl_start - t0) // 1000
                    _append_event(
                        events,
                        events_need_comma,
                        "O",
                        frame_solve_nonlinear,
                        solve_nl_start_us,
                    )
                var u_f: List[Float64]
                if use_banded_loadcontrol:
                    var width = bw_nl * 2 + 1
                    for i in range(free_count):
                        banded_rhs[i] = F_f[i]
                        for j in range(width):
                            K_ff_banded_step[i][j] = K_ff_banded[i][j]
                    if runtime_metrics.enabled:
                        runtime_metrics.tangent_factorizations += 1
                    u_f = banded_gaussian_elimination(K_ff_banded_step, bw_nl, banded_rhs)
                else:
                    var direct_newton_solve = (
                        attempt_algorithm_mode == NonlinearAlgorithmMode.Newton
                        and attempt_algorithm_tag != AnalysisAlgorithmTag.Broyden
                        and attempt_algorithm_tag != AnalysisAlgorithmTag.KrylovNewton
                    )
                    var force_refactor = direct_newton_solve or not tangent_factored
                    if use_native_direct_solve:
                        _ = refactor_loaded_if_needed(
                            backend, refresh_tangent, runtime_metrics, force_refactor
                        )
                    else:
                        _ = refactor_if_needed(
                            backend, K_ff, refresh_tangent, runtime_metrics, force_refactor
                        )
                    tangent_factored = True
                    for i in range(free_count):
                        lu_rhs[i] = F_f[i]
                    solve(backend, lu_rhs, u_f_work)
                    u_f = u_f_work.copy()
                var krylov_base_direction: List[Float64] = []
                if attempt_algorithm_tag == AnalysisAlgorithmTag.KrylovNewton:
                    krylov_base_direction = copy_krylov_vector(u_f, free_count)
                    _ = krylov_apply_acceleration(
                        u_f,
                        krylov_history_v,
                        krylov_history_av,
                        free_count,
                        krylov_normal_matrix_flat,
                        krylov_normal_rhs,
                        krylov_coeffs,
                        krylov_factor_work,
                        krylov_rhs_work,
                    )
                if attempt_algorithm_tag == AnalysisAlgorithmTag.Broyden:
                    if refresh_tangent:
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
                            delta_residual[i] = F_f[i] - broyden_prev_residual[i]
                        var current_z: List[Float64]
                        if use_banded_loadcontrol:
                            var width = bw_nl * 2 + 1
                            for i in range(free_count):
                                banded_rhs[i] = delta_residual[i]
                                for j in range(width):
                                    K_ff_banded_step[i][j] = K_ff_banded[i][j]
                            if runtime_metrics.enabled:
                                runtime_metrics.tangent_factorizations += 1
                            current_z = banded_gaussian_elimination(
                                K_ff_banded_step, bw_nl, banded_rhs
                            )
                        else:
                            for i in range(free_count):
                                lu_rhs[i] = delta_residual[i]
                            solve(backend, lu_rhs, u_f_work)
                            current_z = u_f_work.copy()
                        _broyden_update_direction(
                            u_f,
                            broyden_history_s,
                            broyden_history_z,
                            current_z,
                            len(broyden_history_z),
                        )
                        broyden_history_z.append(current_z^)
                if do_profile:
                    var t_solve_nl_end = Int(time.perf_counter_ns())
                    var solve_nl_end_us = (t_solve_nl_end - t0) // 1000
                    _append_event(
                        events,
                        events_need_comma,
                        "C",
                        frame_solve_nonlinear,
                        solve_nl_end_us,
                    )
                var max_diff = 0.0
                var max_u = 0.0
                for i in range(free_count):
                    var idx = free[i]
                    var du = u_f[i]
                    var value = u[idx] + du
                    u[idx] = value
                    var diff = abs(du)
                    if diff > max_diff:
                        max_diff = diff
                    var abs_val = abs(value)
                    if abs_val > max_u:
                        max_u = abs_val
                if has_transformation_mpc:
                    _enforce_mpc_values(
                        u,
                        constrained,
                        mpc_slave_dof,
                        mpc_row_offsets,
                        mpc_dof_pool,
                        mpc_coeff_pool,
                    )
                if attempt_algorithm_tag == AnalysisAlgorithmTag.NewtonLineSearch:
                    var line_search_tol = _default_line_search_tol(attempt_line_search_eta)
                    var s0 = -dot_float64_contiguous(u_f, F_f, free_count)
                    if abs(s0) > 0.0:
                        _static_load_control_residual(
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
                            F_step,
                            free,
                            free_count,
                            has_transformation_mpc,
                            mpc_row_offsets,
                            mpc_dof_pool,
                            mpc_coeff_pool,
                            F_f,
                        )
                        var s = -dot_float64_contiguous(u_f, F_f, free_count)
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
                                u[free[i]] += delta_eta * u_f[i]
                            if has_transformation_mpc:
                                _enforce_mpc_values(
                                    u,
                                    constrained,
                                    mpc_slave_dof,
                                    mpc_row_offsets,
                                    mpc_dof_pool,
                                    mpc_coeff_pool,
                                )
                            _static_load_control_residual(
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
                                F_step,
                                free,
                                free_count,
                                has_transformation_mpc,
                                mpc_row_offsets,
                                mpc_dof_pool,
                                mpc_coeff_pool,
                                F_f,
                            )
                            s = dot_float64_contiguous(u_f, F_f, free_count)
                            r = abs(s / s0)
                            eta_prev = eta
                        for i in range(free_count):
                            u_f[i] *= eta_prev
                if attempt_algorithm_tag == AnalysisAlgorithmTag.Broyden:
                    broyden_history_s.append(_copy_vector(u_f, free_count))
                    broyden_prev_residual = _copy_vector(F_f, free_count)
                    has_broyden_prev_residual = True
                elif attempt_algorithm_tag == AnalysisAlgorithmTag.KrylovNewton:
                    krylov_push_iteration_state(
                        krylov_base_direction,
                        u_f,
                        free_count,
                        krylov_history_v,
                        krylov_history_av,
                    )
                var disp_incr_norm = sqrt(sum_sq_float64_contiguous(u_f, free_count))
                var energy_incr = abs(dot_float64_contiguous(u_f, F_f, free_count))
                var converged_iter = max_diff <= attempt_tol
                if attempt_test_mode == 1:
                    converged_iter = disp_incr_norm <= attempt_tol
                elif attempt_test_mode == 3:
                    converged_iter = energy_incr <= attempt_tol
                if do_profile:
                    var t_iter_end = Int(time.perf_counter_ns())
                    var iter_end_us = (t_iter_end - t0) // 1000
                    _append_event(
                        events,
                        events_need_comma,
                        "C",
                        frame_nonlinear_iter,
                        iter_end_us,
                    )
                if converged_iter:
                    step_converged_iters = iter_idx + 1
                    converged = True
                    break
            if converged:
                break
        if do_profile:
            var t_step_end = Int(time.perf_counter_ns())
            var step_end_us = (t_step_end - t0) // 1000
            _append_event(
                events, events_need_comma, "C", frame_nonlinear_step, step_end_us
            )
        if not converged:
            abort("static_nonlinear did not converge")
        if adaptive_step:
            if step_converged_iters < 1:
                step_converged_iters = step_target_iters
            last_step_iters = step_converged_iters
        var F_int_reaction = assemble_internal_forces_typed_soa(
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
        var F_ext_reaction: List[Float64] = []
        if has_reaction_recorder:
            F_ext_reaction = F_step.copy()
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
                            value = 0.0
                        values[j] = value
                    var filename = rec.output + "_node" + String(node_id) + ".out"
                    if rec.type_tag == RecorderTypeTag.NodeDisplacement:
                        _append_output(
                            static_output_files,
                            static_output_buffers,
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
                    if (
                        elem_id >= len(elem_id_to_index)
                        or elem_id_to_index[elem_id] < 0
                    ):
                        abort("recorder element not found")
                    var elem_index = elem_id_to_index[elem_id]
                    var elem = typed_elements[elem_index]
                    var f_elem: List[Float64]
                    if (
                        elem.type_tag == ElementTypeTag.ForceBeamColumn2d
                        and typed_sections_by_id[elem.section].type == "FiberSection2d"
                    ):
                        f_elem = _force_beam_column2d_force_global_from_basic_state(
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
                        f_elem = _element_force_global_for_recorder(
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
                    var line = _format_values_line(f_elem)
                    var filename = rec.output + "_ele" + String(elem_id) + ".out"
                    _append_output(
                        static_output_files, static_output_buffers, filename, line
                    )
            elif rec.type_tag == RecorderTypeTag.ElementLocalForce:
                for eidx in range(rec.element_count):
                    var elem_id = recorder_elements_pool[rec.element_offset + eidx]
                    if (
                        elem_id >= len(elem_id_to_index)
                        or elem_id_to_index[elem_id] < 0
                    ):
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
                        static_output_files,
                        static_output_buffers,
                        filename,
                        _format_values_line(values),
                    )
            elif rec.type_tag == RecorderTypeTag.ElementBasicForce:
                for eidx in range(rec.element_count):
                    var elem_id = recorder_elements_pool[rec.element_offset + eidx]
                    if (
                        elem_id >= len(elem_id_to_index)
                        or elem_id_to_index[elem_id] < 0
                    ):
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
                        static_output_files,
                        static_output_buffers,
                        filename,
                        _format_values_line(values),
                    )
            elif rec.type_tag == RecorderTypeTag.ElementDeformation:
                for eidx in range(rec.element_count):
                    var elem_id = recorder_elements_pool[rec.element_offset + eidx]
                    if (
                        elem_id >= len(elem_id_to_index)
                        or elem_id_to_index[elem_id] < 0
                    ):
                        abort("recorder element not found")
                    var elem_index = elem_id_to_index[elem_id]
                    var elem = typed_elements[elem_index]
                    var values = _element_deformation_for_recorder(
                        elem_index,
                        elem,
                        ndf,
                        u,
                        typed_nodes,
                    )
                    var filename = rec.output + "_ele" + String(elem_id) + ".out"
                    _append_output(
                        static_output_files,
                        static_output_buffers,
                        filename,
                        _format_values_line(values),
                    )
            elif rec.type_tag == RecorderTypeTag.NodeReaction:
                for nidx in range(rec.node_count):
                    var node_id = recorder_nodes_pool[rec.node_offset + nidx]
                    var i = id_to_index[node_id]
                    var values: List[Float64] = []
                    values.resize(rec.dof_count, 0.0)
                    for j in range(rec.dof_count):
                        var dof = recorder_dofs_pool[rec.dof_offset + j]
                        require_dof_in_range(dof, ndf, "recorder")
                        var idx = node_dof_index(i, dof, ndf)
                        values[j] = F_int_reaction[idx] - F_ext_reaction[idx]
                    var filename = rec.output + "_node" + String(node_id) + ".out"
                    _append_output(
                        static_output_files,
                        static_output_buffers,
                        filename,
                        _format_values_line(values),
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
                    static_output_files,
                    static_output_buffers,
                    filename,
                    _format_values_line([value]),
                )
            elif rec.type_tag == RecorderTypeTag.EnvelopeElementForce:
                for eidx in range(rec.element_count):
                    var elem_id = recorder_elements_pool[rec.element_offset + eidx]
                    if (
                        elem_id >= len(elem_id_to_index)
                        or elem_id_to_index[elem_id] < 0
                    ):
                        abort("recorder element not found")
                    var elem_index = elem_id_to_index[elem_id]
                    var elem = typed_elements[elem_index]
                    var f_elem: List[Float64]
                    if (
                        elem.type_tag == ElementTypeTag.ForceBeamColumn2d
                        and typed_sections_by_id[elem.section].type == "FiberSection2d"
                    ):
                        f_elem = _force_beam_column2d_force_global_from_basic_state(
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
                        f_elem = _element_force_global_for_recorder(
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
                    var filename = rec.output + "_ele" + String(elem_id) + ".out"
                    _update_envelope(
                        filename,
                        f_elem,
                        envelope_files,
                        envelope_min,
                        envelope_max,
                        envelope_abs,
                    )
            elif rec.type_tag == RecorderTypeTag.EnvelopeElementLocalForce:
                for eidx in range(rec.element_count):
                    var elem_id = recorder_elements_pool[rec.element_offset + eidx]
                    if (
                        elem_id >= len(elem_id_to_index)
                        or elem_id_to_index[elem_id] < 0
                    ):
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
                    if (
                        elem_id >= len(elem_id_to_index)
                        or elem_id_to_index[elem_id] < 0
                    ):
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
                            static_output_files,
                            static_output_buffers,
                            filename,
                            _format_values_line(values),
                        )
            else:
                abort("unsupported recorder type")
    clear(backend)
    _flush_envelope_outputs(
        envelope_files,
        envelope_min,
        envelope_max,
        envelope_abs,
        static_output_files,
        static_output_buffers,
    )
    _write_run_progress(
        progress_path,
        "stage_complete",
        "static_nonlinear",
        progress_stage_number,
        progress_stage_count,
        steps,
        steps,
    )

fn run_static_nonlinear_displacement_control(
    analysis: AnalysisInput,
    mut steps: Int,
    ts_index: Int,
    time_series: List[TimeSeriesInput],
    analysis_integrator_targets_pool: List[Float64],
    analysis_solver_chain_pool: List[SolverAttemptInput],
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
    total_dofs: Int,
    F_const: List[Float64],
    F_pattern: List[Float64],
    constrained: List[Bool],
    free: List[Int],
    recorders: List[RecorderInput],
    recorder_nodes_pool: List[Int],
    recorder_elements_pool: List[Int],
    recorder_dofs_pool: List[Int],
    recorder_sections_pool: List[Int],
    elem_id_to_index: List[Int],
    mut static_output_files: List[String],
    mut static_output_buffers: List[String],
    progress_path: String,
    progress_stage_number: Int,
    progress_stage_count: Int,
    do_profile: Bool,
    t0: Int,
    mut events: String,
    mut events_need_comma: Bool,
    frame_assemble_stiffness: Int,
    frame_assemble_uniaxial: Int,
    frame_assemble_fiber: Int,
    frame_kff_extract: Int,
    frame_solve_nonlinear: Int,
    frame_nonlinear_step: Int,
    frame_nonlinear_iter: Int,
    frame_uniaxial_revert_all: Int,
    frame_uniaxial_commit_all: Int,
    has_transformation_mpc: Bool,
    mpc_slave_dof: List[Bool],
    mpc_row_offsets: List[Int],
    mpc_dof_pool: List[Int],
    mpc_coeff_pool: List[Float64],
    mut runtime_metrics: RuntimeProfileMetrics,
) raises -> Float64:
    var time = Python.import_module("time")
    _write_run_progress(
        progress_path,
        "running_stage",
        "static_nonlinear",
        progress_stage_number,
        progress_stage_count,
        0,
        steps,
    )
    var asm_dof_map6: List[Int] = []
    var asm_dof_map12: List[Int] = []
    var asm_u_elem6: List[Float64] = []
    var force_beam_column2d_scratch = ForceBeamColumn2dScratch()
    var force_beam_column3d_scratch = ForceBeamColumn3dScratch()
    var has_force_beam_column2d = False
    var has_fiber_beam_column2d = False
    for i in range(len(typed_elements)):
        if typed_elements[i].type_tag == ElementTypeTag.ForceBeamColumn2d:
            has_force_beam_column2d = True
            var sec_id = typed_elements[i].section
            if (
                sec_id >= 0
                and sec_id < len(typed_sections_by_id)
                and typed_sections_by_id[sec_id].type == "FiberSection2d"
            ):
                has_fiber_beam_column2d = True
        elif typed_elements[i].type_tag == ElementTypeTag.DispBeamColumn2d:
            var sec_id = typed_elements[i].section
            if (
                sec_id >= 0
                and sec_id < len(typed_sections_by_id)
                and typed_sections_by_id[sec_id].type == "FiberSection2d"
            ):
                has_fiber_beam_column2d = True
    var retry_algorithm_tags: List[Int] = []
    var retry_algorithm_modes: List[Int] = []
    var retry_test_modes: List[Int] = []
    var retry_max_iters: List[Int] = []
    var retry_tols: List[Float64] = []
    var retry_broyden_counts: List[Int] = []
    var retry_line_search_etas: List[Float64] = []
    var retry_krylov_max_dims: List[Int] = []
    _collect_static_solver_chain(
        analysis,
        analysis_solver_chain_pool,
        has_force_beam_column2d,
        True,
        retry_algorithm_tags,
        retry_algorithm_modes,
        retry_test_modes,
        retry_max_iters,
        retry_tols,
        retry_broyden_counts,
        retry_line_search_etas,
        retry_krylov_max_dims,
    )
    if len(retry_algorithm_modes) == 0:
        abort("static_nonlinear solver_chain must contain at least one attempt")
    var retry_attempt_count = len(retry_algorithm_modes)
    var primary_algorithm_tag = retry_algorithm_tags[0]
    var primary_algorithm_mode = retry_algorithm_modes[0]
    var primary_test_mode = retry_test_modes[0]
    var max_iters = retry_max_iters[0]
    var tol = retry_tols[0]
    var primary_broyden_count = retry_broyden_counts[0]
    var primary_line_search_eta = retry_line_search_etas[0]
    var primary_krylov_max_dim = default_krylov_max_dim(retry_krylov_max_dims[0])
    var chain_has_modified_newton_initial = False
    for i in range(retry_attempt_count):
        if retry_algorithm_modes[i] == NonlinearAlgorithmMode.ModifiedNewtonInitial:
            chain_has_modified_newton_initial = True
            break
    var free_count = len(free)
    var F_const_free: List[Float64] = []
    F_const_free.resize(free_count, 0.0)
    var F_pattern_free: List[Float64] = []
    F_pattern_free.resize(free_count, 0.0)
    for i in range(free_count):
        F_const_free[i] = F_const[free[i]]
        F_pattern_free[i] = F_pattern[free[i]]

    var F_int: List[Float64] = []
    F_int.resize(total_dofs, 0.0)

    var K_ff: List[List[Float64]] = []
    var K_init_ff: List[List[Float64]] = []
    for _ in range(free_count):
        var row_ff: List[Float64] = []
        row_ff.resize(free_count, 0.0)
        K_ff.append(row_ff^)
        var row_kinit: List[Float64] = []
        row_kinit.resize(free_count, 0.0)
        K_init_ff.append(row_kinit^)
    var backend = LinearSolverBackend()
    var native_direct_backend_available = (
        not has_transformation_mpc
        and (
            analysis.system_tag == AnalysisSystemTag.FullGeneral
            or analysis.system_tag == AnalysisSystemTag.BandGeneral
            or analysis.system_tag == AnalysisSystemTag.BandSPD
            or analysis.system_tag == AnalysisSystemTag.ProfileSPD
            or analysis.system_tag == AnalysisSystemTag.SuperLU
            or analysis.system_tag == AnalysisSystemTag.UmfPack
            or analysis.system_tag == AnalysisSystemTag.SparseSYM
        )
    )
    var needs_dense_global_matrix = (
        not native_direct_backend_available or chain_has_modified_newton_initial
    )
    var K: List[List[Float64]] = []
    if needs_dense_global_matrix:
        for _ in range(total_dofs):
            var row: List[Float64] = []
            row.resize(total_dofs, 0.0)
            K.append(row^)
    var free_index_native: List[Int] = []
    free_index_native.resize(total_dofs, -1)
    var elem_free_pool_native: List[Int] = []
    if native_direct_backend_available:
        for i in range(free_count):
            free_index_native[free[i]] = i
        elem_free_pool_native.resize(len(elem_dof_pool), -1)
        for i in range(len(elem_dof_pool)):
            var dof = elem_dof_pool[i]
            if dof >= 0 and dof < total_dofs:
                elem_free_pool_native[i] = free_index_native[dof]
        initialize_structure(backend, analysis, free_count)
        initialize_symbolic_from_element_dof_map(
            backend, elem_dof_offsets, elem_dof_pool, free_index_native
        )

    var load_scale_derivative = 1.0
    if ts_index >= 0:
        var ts = time_series[ts_index]
        if ts.type_tag != TimeSeriesTypeTag.Linear:
            abort("DisplacementControl only supports Linear time_series")
        load_scale_derivative = ts.factor
    if analysis.integrator_node < 0 or analysis.integrator_dof < 0:
        abort("DisplacementControl requires node and dof")
    var control_node = analysis.integrator_node
    var control_dof = analysis.integrator_dof
    require_dof_in_range(control_dof, ndf, "DisplacementControl")
    if control_node >= len(id_to_index) or id_to_index[control_node] < 0:
        abort("DisplacementControl node not found")
    var control_idx = node_dof_index(id_to_index[control_node], control_dof, ndf)
    if constrained[control_idx]:
        abort("DisplacementControl dof is constrained")
    if has_transformation_mpc and mpc_slave_dof[control_idx]:
        abort("DisplacementControl dof must be retained for transformation mp_constraints")
    var control_free = -1
    for i in range(free_count):
        if free[i] == control_idx:
            control_free = i
            break
    if control_free < 0:
        abort("DisplacementControl dof is not free")

    var cutback = analysis.integrator_cutback
    var max_cutbacks = analysis.integrator_max_cutbacks
    var min_du = analysis.integrator_min_du
    var adaptive_du = (
        analysis.has_integrator_num_iter
        or analysis.has_integrator_min_du
        or analysis.has_integrator_max_du
    )
    var du_target_iters = analysis.integrator_num_iter
    if du_target_iters < 1:
        du_target_iters = 1
    var du_last_iters = du_target_iters
    if analysis.step_retry_enabled:
        max_cutbacks = 0
    var bulk_retry_attempt_count = retry_attempt_count
    if analysis.step_retry_continue_after_failure and retry_attempt_count > 1:
        bulk_retry_attempt_count = 1
    var continuation_active = False
    var target_tol = 1.0e-14
    if cutback <= 0.0 or cutback >= 1.0:
        abort("DisplacementControl cutback must be in (0, 1)")
    if max_cutbacks < 0:
        abort("DisplacementControl max_cutbacks must be >= 0")
    if min_du <= 0.0:
        abort("DisplacementControl min_du must be > 0")
    if analysis.has_integrator_max_du and analysis.integrator_max_du <= 0.0:
        abort("DisplacementControl max_du must be > 0")

    var has_explicit_targets = analysis.integrator_targets_count > 0
    var target_disps: List[Float64] = []
    var current_du_step = 0.0
    var du_min_mag = 0.0
    var du_max_mag = 0.0
    if has_explicit_targets:
        for i in range(analysis.integrator_targets_count):
            target_disps.append(
                analysis_integrator_targets_pool[analysis.integrator_targets_offset + i]
            )
        if len(target_disps) == 0:
            abort("DisplacementControl targets must not be empty")
        steps = len(target_disps)
    else:
        if not analysis.has_integrator_du:
            abort("DisplacementControl requires du or targets")
        var du_step = analysis.integrator_du
        if du_step == 0.0:
            abort("DisplacementControl du must be nonzero")
        current_du_step = du_step
        du_min_mag = abs(du_step)
        if analysis.has_integrator_min_du:
            du_min_mag = abs(analysis.integrator_min_du)
        du_max_mag = abs(du_step)
        if analysis.has_integrator_max_du:
            du_max_mag = abs(analysis.integrator_max_du)
        if du_max_mag < du_min_mag:
            du_max_mag = du_min_mag

    var load_factor = 0.0
    var has_pattern_reference_load = False
    for i in range(free_count):
        if abs(F_pattern_free[i]) > 0.0:
            has_pattern_reference_load = True
            break
    if (
        not has_pattern_reference_load
        and len(pattern_element_loads) == 0
        and not has_fiber_beam_column2d
    ):
        return load_factor
    var R_f: List[Float64] = []
    R_f.resize(free_count, 0.0)
    var aug_size = free_count + 1
    var K_aug_flat: List[Float64] = []
    K_aug_flat.resize(aug_size * aug_size, 0.0)
    var K_aug_factor_flat: List[Float64] = []
    K_aug_factor_flat.resize(aug_size * aug_size, 0.0)
    var rhs_aug: List[Float64] = []
    rhs_aug.resize(aug_size, 0.0)
    var rhs_aug_work: List[Float64] = []
    rhs_aug_work.resize(aug_size, 0.0)
    var sol_aug: List[Float64] = []
    sol_aug.resize(aug_size, 0.0)
    var aug_load_column_free: List[Float64] = []
    aug_load_column_free.resize(free_count, 0.0)
    for i in range(free_count):
        aug_load_column_free[i] = -load_scale_derivative * F_pattern_free[i]
    var aug_rhs_solve: List[Float64] = []
    aug_rhs_solve.resize(free_count, 0.0)
    var aug_load_solve: List[Float64] = []
    aug_load_solve.resize(free_count, 0.0)
    var has_reaction_recorder = _has_recorder_type(recorders, RecorderTypeTag.NodeReaction)
    var envelope_files: List[String] = []
    var envelope_min: List[List[Float64]] = []
    var envelope_max: List[List[Float64]] = []
    var envelope_abs: List[List[Float64]] = []
    if chain_has_modified_newton_initial:
        var initial_element_load_state = build_active_element_load_state(
            const_element_loads,
            pattern_element_loads,
            0.0,
            typed_elements,
            elem_id_to_index,
            ndm,
            ndf,
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
            initial_element_load_state.element_loads,
            initial_element_load_state.elem_load_offsets,
            initial_element_load_state.elem_load_pool,
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
        if has_transformation_mpc:
            K = _collapse_matrix_by_mpc(K, mpc_row_offsets, mpc_dof_pool, mpc_coeff_pool)
        for i in range(free_count):
            for j in range(free_count):
                K_init_ff[i][j] = K[free[i]][free[j]]
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

    for step in range(steps):
        _write_run_progress(
            progress_path,
            "running_step",
            "static_nonlinear",
            progress_stage_number,
            progress_stage_count,
            step + 1,
            steps,
        )
        if do_profile:
            var t_step_start = Int(time.perf_counter_ns())
            var step_start_us = (t_step_start - t0) // 1000
            _append_event(
                events, events_need_comma, "O", frame_nonlinear_step, step_start_us
            )
        if has_transformation_mpc:
            _enforce_mpc_values(
                u,
                constrained,
                mpc_slave_dof,
                mpc_row_offsets,
                mpc_dof_pool,
                mpc_coeff_pool,
            )

        var target: Float64
        if has_explicit_targets:
            target = target_disps[step]
        else:
            if adaptive_du:
                var du_next = current_du_step
                if du_last_iters > 0:
                    du_next = (
                        current_du_step
                        * Float64(du_target_iters)
                        / Float64(du_last_iters)
                    )
                var du_sign = 1.0
                if du_next < 0.0:
                    du_sign = -1.0
                var du_mag = abs(du_next)
                if du_mag < du_min_mag:
                    du_mag = du_min_mag
                if du_mag > du_max_mag:
                    du_mag = du_max_mag
                current_du_step = du_sign * du_mag
            target = u[control_idx] + current_du_step
        var committed_F_int: List[Float64] = []
        var step_converged_iters = 0
        while True:
            var remaining = target - u[control_idx]
            # `min_du` is the cutback floor, not the target-reached tolerance.
            # Using it here skips exact-size displacement steps whenever
            # `remaining == min_du`, which is common for fixed-increment pushover.
            if abs(remaining) <= target_tol:
                break

            var u_base = u.copy()
            var force_basic_q_base = force_basic_q.copy()
            var uniaxial_states_base = uniaxial_states.copy()
            var lambda_base = load_factor
            var attempt_du = remaining
            var attempt_ok = False

            for _ in range(max_cutbacks + 1):
                var converged = False
                var converged_iters = 0
                var attempt_algorithm_tag = primary_algorithm_tag
                var attempt_algorithm_mode = primary_algorithm_mode
                var attempt_line_search_eta = primary_line_search_eta
                var attempt_test_mode = primary_test_mode
                var attempt_max_iters = max_iters
                var attempt_tol = tol
                var attempt_broyden_count = primary_broyden_count
                var attempt_krylov_max_dim = primary_krylov_max_dim
                var current_retry_attempt_count = retry_attempt_count
                if not continuation_active:
                    current_retry_attempt_count = bulk_retry_attempt_count
                for attempt in range(current_retry_attempt_count):
                    for i in range(total_dofs):
                        u[i] = u_base[i]
                    force_basic_q = force_basic_q_base.copy()
                    uniaxial_states = uniaxial_states_base.copy()
                    fiber_section2d_revert_trial_runtime_all(fiber_section_defs)
                    load_factor = lambda_base
                    if attempt > 0:
                        attempt_algorithm_tag = retry_algorithm_tags[attempt]
                        attempt_algorithm_mode = retry_algorithm_modes[attempt]
                        attempt_line_search_eta = retry_line_search_etas[attempt]
                        attempt_test_mode = retry_test_modes[attempt]
                        attempt_max_iters = retry_max_iters[attempt]
                        attempt_tol = retry_tols[attempt]
                        attempt_broyden_count = retry_broyden_counts[attempt]
                        attempt_krylov_max_dim = default_krylov_max_dim(
                            retry_krylov_max_dims[attempt]
                        )

                    var tangent_initialized = False
                    var tangent_factored = False
                    var broyden_history_s: List[List[Float64]] = []
                    var broyden_history_z: List[List[Float64]] = []
                    var broyden_prev_residual: List[Float64] = []
                    var has_broyden_prev_residual = False
                    var krylov_history_v: List[List[Float64]] = []
                    var krylov_history_av: List[List[Float64]] = []
                    var krylov_normal_matrix_flat: List[Float64] = []
                    var krylov_normal_rhs: List[Float64] = []
                    var krylov_coeffs: List[Float64] = []
                    var krylov_factor_work: List[Float64] = []
                    var krylov_rhs_work: List[Float64] = []
                    for iter_idx in range(attempt_max_iters):
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
                        var broyden_refresh_interval = _default_broyden_count(
                            attempt_broyden_count
                        )
                        var krylov_max_dim = attempt_krylov_max_dim
                        if krylov_max_dim > free_count + 1:
                            krylov_max_dim = free_count + 1
                        var refresh_tangent: Bool
                        if attempt_algorithm_mode == NonlinearAlgorithmMode.Newton:
                            if attempt_algorithm_tag == AnalysisAlgorithmTag.Broyden:
                                refresh_tangent = (
                                    not tangent_initialized
                                    or broyden_refresh_interval <= 1
                                    or len(broyden_history_s) >= broyden_refresh_interval
                                )
                            elif attempt_algorithm_tag == AnalysisAlgorithmTag.KrylovNewton:
                                refresh_tangent = (
                                    not tangent_initialized
                                    or len(krylov_history_v) > krylov_max_dim
                                )
                            else:
                                refresh_tangent = True
                        elif attempt_algorithm_mode == NonlinearAlgorithmMode.ModifiedNewton:
                            refresh_tangent = not tangent_initialized
                        else:
                            refresh_tangent = not tangent_initialized
                        if refresh_tangent and attempt_algorithm_tag == AnalysisAlgorithmTag.KrylovNewton:
                            krylov_history_v = []
                            krylov_history_av = []
                        var use_native_direct_solve = (
                            native_direct_backend_available
                        )
                        var use_native_tangent_assembly = (
                            use_native_direct_solve
                            and refresh_tangent
                            and attempt_algorithm_mode
                                != NonlinearAlgorithmMode.ModifiedNewtonInitial
                        )
                        var use_native_internal_force_only = (
                            use_native_direct_solve and not use_native_tangent_assembly
                        )
                        if do_profile:
                            var t_asm_start = Int(time.perf_counter_ns())
                            var asm_start_us = (t_asm_start - t0) // 1000
                            _append_event(
                                events,
                                events_need_comma,
                                "O",
                                frame_assemble_stiffness,
                                asm_start_us,
                            )
                        var load_scale = load_scale_derivative * load_factor
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
                        if use_native_tangent_assembly:
                            assemble_global_stiffness_and_internal_native_soa(
                                typed_nodes,
                                typed_elements,
                                node_x,
                                node_y,
                                node_z,
                                elem_dof_offsets,
                                elem_dof_pool,
                                elem_dof_offsets,
                                elem_free_pool_native,
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
                                backend,
                                F_int,
                                do_profile,
                                t0,
                                events,
                                events_need_comma,
                                frame_assemble_uniaxial,
                                frame_assemble_fiber,
                                runtime_metrics,
                            )
                        elif use_native_internal_force_only:
                            F_int = assemble_internal_forces_typed_soa(
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
                        else:
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
                            if has_transformation_mpc:
                                K = _collapse_matrix_by_mpc(
                                    K, mpc_row_offsets, mpc_dof_pool, mpc_coeff_pool
                                )
                                F_int = _collapse_vector_by_mpc(
                                    F_int,
                                    mpc_row_offsets,
                                    mpc_dof_pool,
                                    mpc_coeff_pool,
                                )
                        if do_profile:
                            var t_asm_end = Int(time.perf_counter_ns())
                            var asm_end_us = (t_asm_end - t0) // 1000
                            _append_event(
                                events,
                                events_need_comma,
                                "C",
                                frame_assemble_stiffness,
                                asm_end_us,
                            )

                        if do_profile:
                            var t_kff_start = Int(time.perf_counter_ns())
                            var kff_start_us = (t_kff_start - t0) // 1000
                            _append_event(
                                events,
                                events_need_comma,
                                "O",
                                frame_kff_extract,
                                kff_start_us,
                            )
                        var load_ext_scale = load_scale
                        if attempt_algorithm_mode == NonlinearAlgorithmMode.Newton:
                            if refresh_tangent:
                                for i in range(free_count):
                                    var row_i = free[i]
                                    R_f[i] = (
                                        F_const_free[i]
                                        + load_ext_scale * F_pattern_free[i]
                                        - F_int[row_i]
                                    )
                                    if not use_native_direct_solve:
                                        for j in range(free_count):
                                            K_ff[i][j] = K[row_i][free[j]]
                                tangent_initialized = True
                                tangent_factored = False
                            else:
                                for i in range(free_count):
                                    R_f[i] = (
                                        F_const_free[i]
                                        + load_ext_scale * F_pattern_free[i]
                                        - F_int[free[i]]
                                    )
                        elif attempt_algorithm_mode == NonlinearAlgorithmMode.ModifiedNewton:
                            if not tangent_initialized:
                                for i in range(free_count):
                                    var row_i = free[i]
                                    R_f[i] = (
                                        F_const_free[i]
                                        + load_ext_scale * F_pattern_free[i]
                                        - F_int[row_i]
                                    )
                                    if not use_native_direct_solve:
                                        for j in range(free_count):
                                            K_ff[i][j] = K[row_i][free[j]]
                                tangent_initialized = True
                                tangent_factored = False
                            else:
                                for i in range(free_count):
                                    R_f[i] = (
                                        F_const_free[i]
                                        + load_ext_scale * F_pattern_free[i]
                                        - F_int[free[i]]
                                    )
                        else:
                            if not tangent_initialized:
                                if not use_native_direct_solve:
                                    for i in range(free_count):
                                        for j in range(free_count):
                                            K_ff[i][j] = K_init_ff[i][j]
                                tangent_initialized = True
                                tangent_factored = False
                            for i in range(free_count):
                                R_f[i] = (
                                    F_const_free[i]
                                    + load_ext_scale * F_pattern_free[i]
                                    - F_int[free[i]]
                                )
                        if do_profile:
                            var t_kff_end = Int(time.perf_counter_ns())
                            var kff_end_us = (t_kff_end - t0) // 1000
                            _append_event(
                                events,
                                events_need_comma,
                                "C",
                                frame_kff_extract,
                                kff_end_us,
                            )

                        # For DisplacementControl, the pre-solve residual is evaluated at the
                        # current equilibrium state. Treating that as convergence before any
                        # augmented solve would leave the control displacement unchanged and
                        # spin forever on the same target.
                        if attempt_test_mode == 2 and abs(u[control_idx] - u_base[control_idx]) > 0.0:
                            var residual_norm = sqrt(sum_sq_float64_contiguous(R_f, free_count))
                            if residual_norm <= attempt_tol:
                                if do_profile:
                                    var t_iter_end = Int(time.perf_counter_ns())
                                    var iter_end_us = (t_iter_end - t0) // 1000
                                    _append_event(
                                        events,
                                        events_need_comma,
                                        "C",
                                        frame_nonlinear_iter,
                                        iter_end_us,
                                    )
                                converged_iters = iter_idx + 1
                                converged = True
                                break

                        for i in range(free_count):
                            rhs_aug[i] = R_f[i]
                        rhs_aug[free_count] = (
                            attempt_du - (u[control_idx] - u_base[control_idx])
                        )

                        if do_profile:
                            var t_solve_nl_start = Int(time.perf_counter_ns())
                            var solve_nl_start_us = (t_solve_nl_start - t0) // 1000
                            _append_event(
                                events,
                                events_need_comma,
                                "O",
                                frame_solve_nonlinear,
                                solve_nl_start_us,
                            )
                        var solved: Bool
                        if use_native_direct_solve:
                            if attempt_algorithm_mode == NonlinearAlgorithmMode.ModifiedNewtonInitial:
                                _ = refactor_if_needed(
                                    backend,
                                    K_init_ff,
                                    not tangent_factored,
                                    runtime_metrics,
                                    False,
                                )
                            else:
                                var direct_newton_solve = (
                                    attempt_algorithm_mode == NonlinearAlgorithmMode.Newton
                                    and attempt_algorithm_tag
                                        != AnalysisAlgorithmTag.Broyden
                                    and attempt_algorithm_tag
                                        != AnalysisAlgorithmTag.KrylovNewton
                                )
                                var force_refactor = direct_newton_solve or not tangent_factored
                                _ = refactor_loaded_if_needed(
                                    backend,
                                    refresh_tangent,
                                    runtime_metrics,
                                    force_refactor,
                                )
                            tangent_factored = True
                            solved = _solve_displacement_control_augmented_from_factorized_backend(
                                backend,
                                R_f,
                                aug_load_column_free,
                                control_free,
                                rhs_aug[free_count],
                                sol_aug,
                                aug_rhs_solve,
                                aug_load_solve,
                            )
                        else:
                            if refresh_tangent:
                                _load_displacement_control_augmented_dense_flat(
                                    K_aug_flat,
                                    K_ff,
                                    aug_load_column_free,
                                    control_free,
                                )
                            solved = _solve_linear_system_dense_flat(
                                K_aug_flat,
                                rhs_aug,
                                sol_aug,
                                K_aug_factor_flat,
                                rhs_aug_work,
                            )
                            if runtime_metrics.enabled:
                                runtime_metrics.tangent_factorizations += 1
                        if do_profile:
                            var t_solve_nl_end = Int(time.perf_counter_ns())
                            var solve_nl_end_us = (t_solve_nl_end - t0) // 1000
                            _append_event(
                                events,
                                events_need_comma,
                                "C",
                                frame_solve_nonlinear,
                                solve_nl_end_us,
                            )
                        if not solved:
                            break
                        var krylov_base_sol_aug: List[Float64] = []
                        if attempt_algorithm_tag == AnalysisAlgorithmTag.KrylovNewton:
                            krylov_base_sol_aug = copy_krylov_vector(
                                sol_aug, free_count + 1
                            )
                            _ = krylov_apply_acceleration(
                                sol_aug,
                                krylov_history_v,
                                krylov_history_av,
                                free_count + 1,
                                krylov_normal_matrix_flat,
                                krylov_normal_rhs,
                                krylov_coeffs,
                                krylov_factor_work,
                                krylov_rhs_work,
                            )
                        if attempt_algorithm_tag == AnalysisAlgorithmTag.Broyden:
                            if refresh_tangent:
                                broyden_history_s = []
                                broyden_history_z = []
                                broyden_prev_residual = []
                                has_broyden_prev_residual = False
                            elif has_broyden_prev_residual and len(
                                broyden_history_s
                            ) > len(broyden_history_z):
                                var delta_residual_aug: List[Float64] = []
                                delta_residual_aug.resize(free_count + 1, 0.0)
                                for i in range(free_count + 1):
                                    delta_residual_aug[i] = (
                                        rhs_aug[i] - broyden_prev_residual[i]
                                    )
                                var z_aug: List[Float64] = []
                                z_aug.resize(free_count + 1, 0.0)
                                var z_ok: Bool
                                if use_native_direct_solve:
                                    z_ok = _solve_displacement_control_augmented_from_factorized_backend(
                                        backend,
                                        delta_residual_aug,
                                        aug_load_column_free,
                                        control_free,
                                        delta_residual_aug[free_count],
                                        z_aug,
                                        aug_rhs_solve,
                                        aug_load_solve,
                                    )
                                else:
                                    if refresh_tangent:
                                        _load_displacement_control_augmented_dense_flat(
                                            K_aug_flat,
                                            K_ff,
                                            aug_load_column_free,
                                            control_free,
                                        )
                                    z_ok = _solve_linear_system_dense_flat(
                                        K_aug_flat,
                                        delta_residual_aug,
                                        z_aug,
                                        K_aug_factor_flat,
                                        rhs_aug_work,
                                    )
                                if z_ok:
                                    _broyden_update_direction(
                                        sol_aug,
                                        broyden_history_s,
                                        broyden_history_z,
                                        z_aug,
                                        len(broyden_history_z),
                                    )
                                    broyden_history_z.append(z_aug^)

                        var max_diff = 0.0
                        var max_u = 0.0
                        var update_eta = 1.0
                        for i in range(free_count):
                            var idx = free[i]
                            var du = sol_aug[i] * update_eta
                            var value = u[idx] + du
                            u[idx] = value
                            var diff = abs(du)
                            if diff > max_diff:
                                max_diff = diff
                            var abs_val = abs(value)
                            if abs_val > max_u:
                                max_u = abs_val
                        load_factor += sol_aug[free_count] * update_eta
                        if has_transformation_mpc:
                            _enforce_mpc_values(
                                u,
                                constrained,
                                mpc_slave_dof,
                                mpc_row_offsets,
                                mpc_dof_pool,
                                mpc_coeff_pool,
                            )
                        if attempt_algorithm_tag == AnalysisAlgorithmTag.NewtonLineSearch:
                            var line_search_tol = _default_line_search_tol(
                                attempt_line_search_eta
                            )
                            var s0 = -dot_float64_contiguous(
                                sol_aug, rhs_aug, free_count + 1
                            )
                            if abs(s0) > 0.0:
                                _static_displacement_control_residual(
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
                                    free,
                                    free_count,
                                    has_transformation_mpc,
                                    mpc_row_offsets,
                                    mpc_dof_pool,
                                    mpc_coeff_pool,
                                    F_const_free,
                                    F_pattern_free,
                                    load_factor,
                                    attempt_du,
                                    control_idx,
                                    u_base,
                                    rhs_aug,
                                )
                                var s = -dot_float64_contiguous(
                                    sol_aug, rhs_aug, free_count + 1
                                )
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
                                        u[free[i]] += delta_eta * sol_aug[i]
                                    load_factor += delta_eta * sol_aug[free_count]
                                    if has_transformation_mpc:
                                        _enforce_mpc_values(
                                            u,
                                            constrained,
                                            mpc_slave_dof,
                                            mpc_row_offsets,
                                            mpc_dof_pool,
                                            mpc_coeff_pool,
                                        )
                                    _static_displacement_control_residual(
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
                                        free,
                                        free_count,
                                        has_transformation_mpc,
                                        mpc_row_offsets,
                                        mpc_dof_pool,
                                        mpc_coeff_pool,
                                        F_const_free,
                                        F_pattern_free,
                                        load_factor,
                                        attempt_du,
                                        control_idx,
                                        u_base,
                                        rhs_aug,
                                    )
                                    s = dot_float64_contiguous(
                                        sol_aug, rhs_aug, free_count + 1
                                    )
                                    r = abs(s / s0)
                                    eta_prev = eta
                                for i in range(free_count + 1):
                                    sol_aug[i] *= eta_prev
                        if attempt_algorithm_tag == AnalysisAlgorithmTag.Broyden:
                            broyden_history_s.append(
                                _copy_vector(sol_aug, free_count + 1)
                            )
                            broyden_prev_residual = _copy_vector(
                                rhs_aug, free_count + 1
                            )
                            has_broyden_prev_residual = True
                        elif attempt_algorithm_tag == AnalysisAlgorithmTag.KrylovNewton:
                            krylov_push_iteration_state(
                                krylov_base_sol_aug,
                                sol_aug,
                                free_count + 1,
                                krylov_history_v,
                                krylov_history_av,
                            )
                        var disp_incr_norm = sqrt(sum_sq_float64_contiguous(sol_aug, free_count))
                        var energy_incr = abs(dot_float64_contiguous(sol_aug, rhs_aug, free_count))
                        var converged_iter = max_diff <= attempt_tol
                        if attempt_test_mode == 1:
                            converged_iter = disp_incr_norm <= attempt_tol
                        elif attempt_test_mode == 3:
                            converged_iter = energy_incr <= attempt_tol
                        if do_profile:
                            var t_iter_end = Int(time.perf_counter_ns())
                            var iter_end_us = (t_iter_end - t0) // 1000
                            _append_event(
                                events,
                                events_need_comma,
                                "C",
                                frame_nonlinear_iter,
                                iter_end_us,
                            )
                        if converged_iter:
                            converged_iters = iter_idx + 1
                            converged = True
                            break
                    if converged:
                        break
                if converged:
                    attempt_ok = True
                    step_converged_iters = converged_iters
                    break
                attempt_du *= cutback
                if abs(attempt_du) <= min_du:
                    break

            if not attempt_ok:
                if (
                    analysis.step_retry_continue_after_failure
                    and not continuation_active
                    and retry_attempt_count > bulk_retry_attempt_count
                ):
                    continuation_active = True
                    for i in range(total_dofs):
                        u[i] = u_base[i]
                    force_basic_q = force_basic_q_base.copy()
                    uniaxial_states = uniaxial_states_base.copy()
                    fiber_section2d_revert_trial_runtime_all(fiber_section_defs)
                    load_factor = lambda_base
                    continue
                abort("static_nonlinear did not converge (DisplacementControl)")

            var committed_load_scale = load_scale_derivative * load_factor
            var committed_element_load_state = build_active_element_load_state(
                const_element_loads,
                pattern_element_loads,
                committed_load_scale,
                typed_elements,
                elem_id_to_index,
                ndm,
                ndf,
            )
            committed_F_int = assemble_internal_forces_typed_soa(
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
                committed_element_load_state.element_loads,
                committed_element_load_state.elem_load_offsets,
                committed_element_load_state.elem_load_pool,
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

        if not has_explicit_targets and adaptive_du:
            if step_converged_iters < 1:
                step_converged_iters = du_target_iters
            du_last_iters = step_converged_iters

        if do_profile:
            var t_step_end = Int(time.perf_counter_ns())
            var step_end_us = (t_step_end - t0) // 1000
            _append_event(
                events, events_need_comma, "C", frame_nonlinear_step, step_end_us
            )

        var final_load_scale = load_scale_derivative * load_factor
        var F_step = build_active_nodal_load(F_const, F_pattern, final_load_scale)
        var active_element_load_state = build_active_element_load_state(
            const_element_loads,
            pattern_element_loads,
            final_load_scale,
            typed_elements,
            elem_id_to_index,
            ndm,
            ndf,
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
        var F_int_reaction: List[Float64] = []
        var F_ext_reaction: List[Float64] = []
        if has_reaction_recorder:
            if len(committed_F_int) == total_dofs:
                F_int_reaction = committed_F_int^
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
            F_ext_reaction = F_step.copy()

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
                            value = 0.0
                        values[j] = value
                    var filename = rec.output + "_node" + String(node_id) + ".out"
                    if rec.type_tag == RecorderTypeTag.NodeDisplacement:
                        _append_output(
                            static_output_files,
                            static_output_buffers,
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
                    if (
                        elem_id >= len(elem_id_to_index)
                        or elem_id_to_index[elem_id] < 0
                    ):
                        abort("recorder element not found")
                    var elem_index = elem_id_to_index[elem_id]
                    var elem = typed_elements[elem_index]
                    var f_elem: List[Float64]
                    if (
                        elem.type_tag == ElementTypeTag.ForceBeamColumn2d
                        and typed_sections_by_id[elem.section].type == "FiberSection2d"
                    ):
                        f_elem = _force_beam_column2d_force_global_from_basic_state(
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
                        f_elem = _element_force_global_for_recorder(
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
                    var line = _format_values_line(f_elem)
                    var filename = rec.output + "_ele" + String(elem_id) + ".out"
                    _append_output(
                        static_output_files, static_output_buffers, filename, line
                    )
            elif rec.type_tag == RecorderTypeTag.ElementLocalForce:
                for eidx in range(rec.element_count):
                    var elem_id = recorder_elements_pool[rec.element_offset + eidx]
                    if (
                        elem_id >= len(elem_id_to_index)
                        or elem_id_to_index[elem_id] < 0
                    ):
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
                        static_output_files,
                        static_output_buffers,
                        filename,
                        _format_values_line(values),
                    )
            elif rec.type_tag == RecorderTypeTag.ElementBasicForce:
                for eidx in range(rec.element_count):
                    var elem_id = recorder_elements_pool[rec.element_offset + eidx]
                    if (
                        elem_id >= len(elem_id_to_index)
                        or elem_id_to_index[elem_id] < 0
                    ):
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
                        static_output_files,
                        static_output_buffers,
                        filename,
                        _format_values_line(values),
                    )
            elif rec.type_tag == RecorderTypeTag.ElementDeformation:
                for eidx in range(rec.element_count):
                    var elem_id = recorder_elements_pool[rec.element_offset + eidx]
                    if (
                        elem_id >= len(elem_id_to_index)
                        or elem_id_to_index[elem_id] < 0
                    ):
                        abort("recorder element not found")
                    var elem_index = elem_id_to_index[elem_id]
                    var elem = typed_elements[elem_index]
                    var values = _element_deformation_for_recorder(
                        elem_index,
                        elem,
                        ndf,
                        u,
                        typed_nodes,
                    )
                    var filename = rec.output + "_ele" + String(elem_id) + ".out"
                    _append_output(
                        static_output_files,
                        static_output_buffers,
                        filename,
                        _format_values_line(values),
                    )
            elif rec.type_tag == RecorderTypeTag.NodeReaction:
                for nidx in range(rec.node_count):
                    var node_id = recorder_nodes_pool[rec.node_offset + nidx]
                    var i = id_to_index[node_id]
                    var values: List[Float64] = []
                    values.resize(rec.dof_count, 0.0)
                    for j in range(rec.dof_count):
                        var dof = recorder_dofs_pool[rec.dof_offset + j]
                        require_dof_in_range(dof, ndf, "recorder")
                        var idx = node_dof_index(i, dof, ndf)
                        values[j] = F_int_reaction[idx] - F_ext_reaction[idx]
                    var filename = rec.output + "_node" + String(node_id) + ".out"
                    _append_output(
                        static_output_files,
                        static_output_buffers,
                        filename,
                        _format_values_line(values),
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
                    static_output_files,
                    static_output_buffers,
                    filename,
                    _format_values_line([value]),
                )
            elif rec.type_tag == RecorderTypeTag.EnvelopeElementForce:
                for eidx in range(rec.element_count):
                    var elem_id = recorder_elements_pool[rec.element_offset + eidx]
                    if (
                        elem_id >= len(elem_id_to_index)
                        or elem_id_to_index[elem_id] < 0
                    ):
                        abort("recorder element not found")
                    var elem_index = elem_id_to_index[elem_id]
                    var elem = typed_elements[elem_index]
                    var f_elem: List[Float64]
                    if (
                        elem.type_tag == ElementTypeTag.ForceBeamColumn2d
                        and typed_sections_by_id[elem.section].type == "FiberSection2d"
                    ):
                        f_elem = _force_beam_column2d_force_global_from_basic_state(
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
                        f_elem = _element_force_global_for_recorder(
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
                    var filename = rec.output + "_ele" + String(elem_id) + ".out"
                    _update_envelope(
                        filename,
                        f_elem,
                        envelope_files,
                        envelope_min,
                        envelope_max,
                        envelope_abs,
                    )
            elif rec.type_tag == RecorderTypeTag.EnvelopeElementLocalForce:
                for eidx in range(rec.element_count):
                    var elem_id = recorder_elements_pool[rec.element_offset + eidx]
                    if (
                        elem_id >= len(elem_id_to_index)
                        or elem_id_to_index[elem_id] < 0
                    ):
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
                    if (
                        elem_id >= len(elem_id_to_index)
                        or elem_id_to_index[elem_id] < 0
                    ):
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
                            static_output_files,
                            static_output_buffers,
                            filename,
                            _format_values_line(values),
                        )
            else:
                abort("unsupported recorder type")
    _flush_envelope_outputs(
        envelope_files,
        envelope_min,
        envelope_max,
        envelope_abs,
        static_output_files,
        static_output_buffers,
    )
    _write_run_progress(
        progress_path,
        "stage_complete",
        "static_nonlinear",
        progress_stage_number,
        progress_stage_count,
        steps,
        steps,
    )
    return load_scale_derivative * load_factor
