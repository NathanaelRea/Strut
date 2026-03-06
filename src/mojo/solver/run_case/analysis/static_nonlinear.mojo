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
    initialize_structure,
    refactor_if_needed,
    solve,
)
from solver.assembly import (
    assemble_global_stiffness_and_internal_soa,
    assemble_internal_forces_typed_soa,
)
from solver.banded import banded_gaussian_elimination, banded_matrix, estimate_bandwidth_typed
from solver.dof import node_dof_index, require_dof_in_range
from solver.profile import _append_event
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
from sections import FiberCell, FiberSection2dDef, FiberSection3dDef
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
    _collapse_matrix_by_rep,
    _collapse_vector_by_rep,
    _drift_value,
    _element_basic_force_for_recorder,
    _element_deformation_for_recorder,
    _element_force_global_for_recorder,
    _force_beam_column2d_force_global_from_basic_state,
    _force_beam_column2d_section_response_from_basic_state,
    _element_local_force_for_recorder,
    _enforce_equal_dof_values,
    _flush_envelope_outputs,
    _format_values_line,
    _has_recorder_type,
    _section_response_for_recorder,
    _solve_linear_system,
    _sync_force_beam_column2d_committed_basic_states,
    _update_envelope,
)
from solver.run_case.load_state import (
    build_active_element_load_state,
    build_active_nodal_load,
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
    ):
        # Mapped alternative: currently follows Newton tangent refresh behavior.
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


fn _append_static_retry_attempt(
    algorithm: String,
    algorithm_tag: Int,
    broyden_count: Int,
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
    mut retry_broyden_counts: List[Int],
    mut retry_line_search_etas: List[Float64],
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
    retry_rel_tols.append(rel_tol)
    retry_broyden_counts.append(broyden_count)
    retry_line_search_etas.append(line_search_eta)


fn _append_static_retry_attempt_from_input(
    attempt: SolverAttemptInput,
    label: String,
    mut retry_algorithm_tags: List[Int],
    mut retry_algorithm_modes: List[Int],
    mut retry_test_modes: List[Int],
    mut retry_max_iters: List[Int],
    mut retry_tols: List[Float64],
    mut retry_rel_tols: List[Float64],
    mut retry_broyden_counts: List[Int],
    mut retry_line_search_etas: List[Float64],
):
    _append_static_retry_attempt(
        attempt.algorithm,
        attempt.algorithm_tag,
        attempt.broyden_count,
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
        retry_broyden_counts,
        retry_line_search_etas,
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
    mut retry_rel_tols: List[Float64],
    mut retry_broyden_counts: List[Int],
    mut retry_line_search_etas: List[Float64],
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
            retry_rel_tols,
            retry_broyden_counts,
            retry_line_search_etas,
        )
    if len(retry_algorithm_modes) == 0:
        _append_static_retry_attempt(
            analysis.algorithm,
            analysis.algorithm_tag,
            0,
            1.0,
            analysis.test_type,
            analysis.test_type_tag,
            analysis.max_iters,
            analysis.tol,
            analysis.rel_tol,
            "static_nonlinear primary",
            retry_algorithm_tags,
            retry_algorithm_modes,
            retry_test_modes,
            retry_max_iters,
            retry_tols,
            retry_rel_tols,
            retry_broyden_counts,
            retry_line_search_etas,
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
    var auto_rel_tol = retry_rel_tols[0]
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
        auto_rel_tol = retry_rel_tols[0]
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
        auto_test_type,
        auto_test_type_tag,
        auto_max_iters,
        auto_tol,
        auto_rel_tol,
        "static_nonlinear auto_fallback",
        retry_algorithm_tags,
        retry_algorithm_modes,
        retry_test_modes,
        retry_max_iters,
        retry_tols,
        retry_rel_tols,
        retry_broyden_counts,
        retry_line_search_etas,
    )


fn _uniaxial_revert_trial_active_states(
    mut uniaxial_states: List[UniMaterialState], elem_uniaxial_state_ids: List[Int]
):
    for i in range(len(elem_uniaxial_state_ids)):
        var state_id = elem_uniaxial_state_ids[i]
        if state_id < 0 or state_id >= len(uniaxial_states):
            abort("active uniaxial state id out of range")
        ref state = uniaxial_states[state_id]
        uniaxial_revert_trial(state)


fn _uniaxial_commit_active_states(
    mut uniaxial_states: List[UniMaterialState], elem_uniaxial_state_ids: List[Int]
):
    for i in range(len(elem_uniaxial_state_ids)):
        var state_id = elem_uniaxial_state_ids[i]
        if state_id < 0 or state_id >= len(uniaxial_states):
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
    fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    fiber_section3d_defs: List[FiberSection3dDef],
    fiber_section3d_cells: List[FiberCell],
    fiber_section3d_index_by_id: List[Int],
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
    mut static_output_buffers: List[List[String]],
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
    rep_dof: List[Int],
    constrained: List[Bool],
) raises:
    var time = Python.import_module("time")
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
    var retry_rel_tols: List[Float64] = []
    var retry_broyden_counts: List[Int] = []
    var retry_line_search_etas: List[Float64] = []
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
        retry_rel_tols,
        retry_broyden_counts,
        retry_line_search_etas,
    )
    if len(retry_algorithm_modes) == 0:
        abort("static_nonlinear solver_chain must contain at least one attempt")
    var retry_attempt_count = len(retry_algorithm_modes)
    var primary_algorithm_tag = retry_algorithm_tags[0]
    var primary_algorithm_mode = retry_algorithm_modes[0]
    var primary_test_mode = retry_test_modes[0]
    var max_iters = retry_max_iters[0]
    var tol = retry_tols[0]
    var rel_tol = retry_rel_tols[0]
    var primary_broyden_count = retry_broyden_counts[0]
    var primary_line_search_eta = retry_line_search_etas[0]
    var free_count = len(free)
    var F_const_free: List[Float64] = []
    F_const_free.resize(free_count, 0.0)
    var F_pattern_free: List[Float64] = []
    F_pattern_free.resize(free_count, 0.0)
    for i in range(free_count):
        F_const_free[i] = F_const[free[i]]
        F_pattern_free[i] = F_pattern[free[i]]

    var K: List[List[Float64]] = []
    for _ in range(total_dofs):
        var row: List[Float64] = []
        row.resize(total_dofs, 0.0)
        K.append(row^)
    var F_int: List[Float64] = []
    F_int.resize(total_dofs, 0.0)

    var integrator_tag = analysis.integrator_tag
    if integrator_tag == IntegratorTypeTag.Unknown:
        integrator_tag = IntegratorTypeTag.LoadControl
    var chain_has_modified_newton_initial = False
    for i in range(retry_attempt_count):
        if retry_algorithm_modes[i] == NonlinearAlgorithmMode.ModifiedNewtonInitial:
            chain_has_modified_newton_initial = True
            break
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
    var K_ff: List[List[Float64]] = []
    var K_init_ff: List[List[Float64]] = []
    var backend = LinearSolverBackend()
    var lu_rhs: List[Float64] = []
    var u_f_work: List[Float64] = []
    if not use_banded_loadcontrol:
        initialize_structure(backend, analysis, free_count)
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
        if has_transformation_mpc:
            K = _collapse_matrix_by_rep(K, rep_dof)
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
    for _ in range(steps):
        if do_profile:
            var t_step_start = Int(time.perf_counter_ns())
            var step_start_us = (t_step_start - t0) // 1000
            _append_event(
                events, events_need_comma, "O", frame_nonlinear_step, step_start_us
            )
        if has_transformation_mpc:
            _enforce_equal_dof_values(u, rep_dof, constrained)
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
        reset_force_beam_column2d_scratch(force_beam_column2d_scratch)
        reset_force_beam_column3d_scratch(force_beam_column3d_scratch)
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
        var attempt_rel_tol = rel_tol
        var attempt_broyden_count = primary_broyden_count
        var step_converged_iters = 0
        for attempt in range(retry_attempt_count):
            if attempt > 0:
                for i in range(total_dofs):
                    u[i] = u_base[i]
                force_basic_q = force_basic_q_base.copy()
                uniaxial_states = uniaxial_states_base.copy()
                attempt_algorithm_tag = retry_algorithm_tags[attempt]
                attempt_algorithm_mode = retry_algorithm_modes[attempt]
                attempt_line_search_eta = retry_line_search_etas[attempt]
                attempt_test_mode = retry_test_modes[attempt]
                attempt_max_iters = retry_max_iters[attempt]
                attempt_tol = retry_tols[attempt]
                attempt_rel_tol = retry_rel_tols[attempt]
                attempt_broyden_count = retry_broyden_counts[attempt]

            var tangent_initialized = False
            var tangent_factored = False
            for iter_idx in range(attempt_max_iters):
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
                    var t_asm_start = Int(time.perf_counter_ns())
                    var asm_start_us = (t_asm_start - t0) // 1000
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
                if has_transformation_mpc:
                    K = _collapse_matrix_by_rep(K, rep_dof)
                    F_int = _collapse_vector_by_rep(F_int, rep_dof)
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
                var broyden_refresh_interval = attempt_broyden_count
                if (
                    attempt_algorithm_tag == AnalysisAlgorithmTag.Broyden
                    and broyden_refresh_interval < 1
                ):
                    broyden_refresh_interval = 8
                var refresh_tangent: Bool
                if attempt_algorithm_mode == NonlinearAlgorithmMode.Newton:
                    if attempt_algorithm_tag == AnalysisAlgorithmTag.Broyden:
                        refresh_tangent = (
                            not tangent_initialized
                            or broyden_refresh_interval <= 1
                            or (iter_idx % broyden_refresh_interval == 0)
                        )
                    else:
                        refresh_tangent = True
                elif attempt_algorithm_mode == NonlinearAlgorithmMode.ModifiedNewton:
                    refresh_tangent = not tangent_initialized
                else:
                    refresh_tangent = not tangent_initialized
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
                    u_f = banded_gaussian_elimination(K_ff_banded_step, bw_nl, banded_rhs)
                else:
                    var direct_newton_solve = (
                        attempt_algorithm_mode == NonlinearAlgorithmMode.Newton
                        and attempt_algorithm_tag != AnalysisAlgorithmTag.Broyden
                    )
                    var force_refactor = direct_newton_solve or not tangent_factored
                    _ = refactor_if_needed(
                        backend, K_ff, refresh_tangent, force_refactor
                    )
                    tangent_factored = True
                    for i in range(free_count):
                        lu_rhs[i] = F_f[i]
                    solve(backend, lu_rhs, u_f_work)
                    u_f = u_f_work.copy()
                    if attempt_algorithm_tag == AnalysisAlgorithmTag.NewtonLineSearch:
                        var update_eta = attempt_line_search_eta
                        if update_eta <= 0.0:
                            update_eta = 0.8
                        for i in range(free_count):
                            u_f[i] *= update_eta
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
                    _enforce_equal_dof_values(u, rep_dof, constrained)
                var disp_incr_norm = sqrt(sum_sq_float64_contiguous(u_f, free_count))
                var energy_incr = abs(dot_float64_contiguous(u_f, F_f, free_count))
                var scale_tol = attempt_rel_tol * max_u
                if scale_tol < attempt_rel_tol:
                    scale_tol = attempt_rel_tol
                var converged_iter = max_diff <= attempt_tol or max_diff <= scale_tol
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
    fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    fiber_section3d_defs: List[FiberSection3dDef],
    fiber_section3d_cells: List[FiberCell],
    fiber_section3d_index_by_id: List[Int],
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
    mut static_output_buffers: List[List[String]],
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
    rep_dof: List[Int],
) raises -> Float64:
    var time = Python.import_module("time")
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
    var retry_rel_tols: List[Float64] = []
    var retry_broyden_counts: List[Int] = []
    var retry_line_search_etas: List[Float64] = []
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
        retry_rel_tols,
        retry_broyden_counts,
        retry_line_search_etas,
    )
    if len(retry_algorithm_modes) == 0:
        abort("static_nonlinear solver_chain must contain at least one attempt")
    var retry_attempt_count = len(retry_algorithm_modes)
    var primary_algorithm_tag = retry_algorithm_tags[0]
    var primary_algorithm_mode = retry_algorithm_modes[0]
    var primary_test_mode = retry_test_modes[0]
    var max_iters = retry_max_iters[0]
    var tol = retry_tols[0]
    var rel_tol = retry_rel_tols[0]
    var primary_broyden_count = retry_broyden_counts[0]
    var primary_line_search_eta = retry_line_search_etas[0]
    var free_count = len(free)
    var F_const_free: List[Float64] = []
    F_const_free.resize(free_count, 0.0)
    var F_pattern_free: List[Float64] = []
    F_pattern_free.resize(free_count, 0.0)
    for i in range(free_count):
        F_const_free[i] = F_const[free[i]]
        F_pattern_free[i] = F_pattern[free[i]]

    var K: List[List[Float64]] = []
    for _ in range(total_dofs):
        var row: List[Float64] = []
        row.resize(total_dofs, 0.0)
        K.append(row^)
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
    if has_transformation_mpc and rep_dof[control_idx] != control_idx:
        abort("DisplacementControl dof must be retained for equalDOF")
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
    var chain_has_modified_newton_initial = False
    for i in range(retry_attempt_count):
        if retry_algorithm_modes[i] == NonlinearAlgorithmMode.ModifiedNewtonInitial:
            chain_has_modified_newton_initial = True
            break
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
    var K_aug: List[List[Float64]] = []
    for _ in range(aug_size):
        var row_aug: List[Float64] = []
        row_aug.resize(aug_size, 0.0)
        K_aug.append(row_aug^)
    var rhs_aug: List[Float64] = []
    rhs_aug.resize(aug_size, 0.0)
    var sol_aug: List[Float64] = []
    sol_aug.resize(aug_size, 0.0)
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
        if has_transformation_mpc:
            K = _collapse_matrix_by_rep(K, rep_dof)
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
        if do_profile:
            var t_step_start = Int(time.perf_counter_ns())
            var step_start_us = (t_step_start - t0) // 1000
            _append_event(
                events, events_need_comma, "O", frame_nonlinear_step, step_start_us
            )
        if has_transformation_mpc:
            _enforce_equal_dof_values(u, rep_dof, constrained)

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
                var attempt_rel_tol = rel_tol
                var attempt_broyden_count = primary_broyden_count
                var current_retry_attempt_count = retry_attempt_count
                if not continuation_active:
                    current_retry_attempt_count = bulk_retry_attempt_count
                for attempt in range(current_retry_attempt_count):
                    for i in range(total_dofs):
                        u[i] = u_base[i]
                    force_basic_q = force_basic_q_base.copy()
                    uniaxial_states = uniaxial_states_base.copy()
                    load_factor = lambda_base
                    if attempt > 0:
                        attempt_algorithm_tag = retry_algorithm_tags[attempt]
                        attempt_algorithm_mode = retry_algorithm_modes[attempt]
                        attempt_line_search_eta = retry_line_search_etas[attempt]
                        attempt_test_mode = retry_test_modes[attempt]
                        attempt_max_iters = retry_max_iters[attempt]
                        attempt_tol = retry_tols[attempt]
                        attempt_rel_tol = retry_rel_tols[attempt]
                        attempt_broyden_count = retry_broyden_counts[attempt]

                    var tangent_initialized = False
                    for iter_idx in range(attempt_max_iters):
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
                        if has_transformation_mpc:
                            K = _collapse_matrix_by_rep(K, rep_dof)
                            F_int = _collapse_vector_by_rep(F_int, rep_dof)
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
                        var broyden_refresh_interval = attempt_broyden_count
                        if (
                            attempt_algorithm_tag == AnalysisAlgorithmTag.Broyden
                            and broyden_refresh_interval < 1
                        ):
                            broyden_refresh_interval = 8
                        var refresh_tangent: Bool
                        if attempt_algorithm_mode == NonlinearAlgorithmMode.Newton:
                            if attempt_algorithm_tag == AnalysisAlgorithmTag.Broyden:
                                refresh_tangent = (
                                    not tangent_initialized
                                    or broyden_refresh_interval <= 1
                                    or (iter_idx % broyden_refresh_interval == 0)
                                )
                            else:
                                refresh_tangent = True
                        elif attempt_algorithm_mode == NonlinearAlgorithmMode.ModifiedNewton:
                            refresh_tangent = not tangent_initialized
                        else:
                            refresh_tangent = not tangent_initialized
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
                                    for j in range(free_count):
                                        K_ff[i][j] = K[row_i][free[j]]
                                tangent_initialized = True
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
                                    for j in range(free_count):
                                        K_ff[i][j] = K[row_i][free[j]]
                                tangent_initialized = True
                            else:
                                for i in range(free_count):
                                    R_f[i] = (
                                        F_const_free[i]
                                        + load_ext_scale * F_pattern_free[i]
                                        - F_int[free[i]]
                                    )
                        else:
                            if not tangent_initialized:
                                for i in range(free_count):
                                    for j in range(free_count):
                                        K_ff[i][j] = K_init_ff[i][j]
                                tangent_initialized = True
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
                            for j in range(free_count):
                                K_aug[i][j] = K_ff[i][j]
                            K_aug[i][free_count] = -load_scale_derivative * F_pattern_free[i]
                        for j in range(free_count):
                            K_aug[free_count][j] = 0.0
                        K_aug[free_count][control_free] = 1.0
                        K_aug[free_count][free_count] = 0.0
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
                        var solved = _solve_linear_system(K_aug, rhs_aug, sol_aug)
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

                        var max_diff = 0.0
                        var max_u = 0.0
                        var update_eta = 1.0
                        if attempt_algorithm_tag == AnalysisAlgorithmTag.NewtonLineSearch:
                            update_eta = attempt_line_search_eta
                            if update_eta <= 0.0:
                                update_eta = 0.8
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
                            _enforce_equal_dof_values(u, rep_dof, constrained)
                        if update_eta != 1.0:
                            for i in range(free_count):
                                sol_aug[i] *= update_eta
                        var disp_incr_norm = sqrt(sum_sq_float64_contiguous(sol_aug, free_count))
                        var energy_incr = abs(dot_float64_contiguous(sol_aug, rhs_aug, free_count))
                        var scale_tol = attempt_rel_tol * max_u
                        if scale_tol < attempt_rel_tol:
                            scale_tol = attempt_rel_tol
                        var converged_iter = max_diff <= attempt_tol or max_diff <= scale_tol
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
        reset_force_beam_column2d_scratch(force_beam_column2d_scratch)
        reset_force_beam_column3d_scratch(force_beam_column3d_scratch)
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
    return load_scale_derivative * load_factor
