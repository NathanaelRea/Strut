from collections import List
from elements import (
    ForceBeamColumn2dScratch,
    ForceBeamColumn3dScratch,
    reset_force_beam_column2d_scratch,
    reset_force_beam_column3d_scratch,
)
from math import sqrt
from materials import UniMaterialDef, UniMaterialState
from os import abort
from python import Python

from linalg import gaussian_elimination_into, lu_factorize_into, lu_solve_into
from materials import uniaxial_commit_all, uniaxial_revert_trial_all
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
)
from solver.time_series import TimeSeriesInput, eval_time_series_input
from sections import FiberCell, FiberSection2dDef, FiberSection3dDef
from tag_types import ElementTypeTag, NonlinearAlgorithmMode, RecorderTypeTag, TimeSeriesTypeTag

from solver.run_case.helpers import (
    _append_output,
    _collapse_matrix_by_rep,
    _collapse_vector_by_rep,
    _drift_value,
    _element_basic_force_for_recorder,
    _element_deformation_for_recorder,
    _element_force_global_for_recorder,
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


fn _static_nonlinear_algorithm_mode(algorithm: String, label: String) -> Int:
    if algorithm == "Newton":
        return NonlinearAlgorithmMode.Newton
    if algorithm == "ModifiedNewton":
        return NonlinearAlgorithmMode.ModifiedNewton
    if algorithm == "ModifiedNewtonInitial":
        return NonlinearAlgorithmMode.ModifiedNewtonInitial
    abort("unsupported " + label + " algorithm: " + algorithm)
    return NonlinearAlgorithmMode.Unknown


fn _static_nonlinear_test_mode(test_type: String, label: String) -> Int:
    if test_type == "MaxDispIncr":
        return 0
    if test_type == "NormDispIncr":
        return 1
    if test_type == "NormUnbalance":
        return 2
    if test_type == "EnergyIncr":
        return 3
    abort("unsupported " + label + " test_type: " + test_type)
    return -1

fn run_static_nonlinear_load_control(
    analysis: AnalysisInput,
    steps: Int,
    ts_index: Int,
    time_series: List[TimeSeriesInput],
    time_series_values: List[Float64],
    time_series_times: List[Float64],
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
    use_banded_nonlinear: Bool,
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
    var max_iters = analysis.max_iters
    var tol = analysis.tol
    var rel_tol = analysis.rel_tol
    var has_force_beam_column2d = False
    for i in range(len(typed_elements)):
        if typed_elements[i].type_tag == ElementTypeTag.ForceBeamColumn2d:
            has_force_beam_column2d = True
            break
    var primary_algorithm_mode = _static_nonlinear_algorithm_mode(analysis.algorithm, "static_nonlinear")
    var fallback_algorithm = analysis.fallback_algorithm
    var has_fallback = False
    var fallback_algorithm_mode = NonlinearAlgorithmMode.Unknown
    if len(fallback_algorithm) > 0:
        fallback_algorithm_mode = _static_nonlinear_algorithm_mode(
            fallback_algorithm, "static_nonlinear fallback"
        )
        has_fallback = True
    elif has_force_beam_column2d and (
        primary_algorithm_mode == NonlinearAlgorithmMode.Newton
        or primary_algorithm_mode == NonlinearAlgorithmMode.ModifiedNewton
    ):
        if primary_algorithm_mode == NonlinearAlgorithmMode.Newton:
            fallback_algorithm_mode = NonlinearAlgorithmMode.ModifiedNewton
        else:
            fallback_algorithm_mode = NonlinearAlgorithmMode.ModifiedNewtonInitial
        has_fallback = True
    elif primary_algorithm_mode != NonlinearAlgorithmMode.Newton:
        fallback_algorithm_mode = NonlinearAlgorithmMode.Newton
        has_fallback = True
    var primary_test_mode = _static_nonlinear_test_mode(
        analysis.test_type, "static_nonlinear"
    )
    var fallback_test_mode = _static_nonlinear_test_mode(
        analysis.fallback_test_type, "static_nonlinear fallback"
    )
    if max_iters < 1:
        abort("max_iters must be >= 1")
    var fallback_max_iters = analysis.fallback_max_iters
    var fallback_tol = analysis.fallback_tol
    var fallback_rel_tol = analysis.fallback_rel_tol
    if len(fallback_algorithm) == 0 and has_force_beam_column2d and has_fallback:
        fallback_test_mode = 1
        if fallback_max_iters < 100:
            fallback_max_iters = 100
        if fallback_max_iters < max_iters * 5:
            fallback_max_iters = max_iters * 5
        fallback_tol = tol
        if fallback_tol < 1.0e-10:
            fallback_tol = 1.0e-10
        fallback_rel_tol = rel_tol
    if fallback_max_iters < 1:
        abort("static_nonlinear fallback_max_iters must be >= 1")
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

    var integrator_type = analysis.integrator_type
    if integrator_type == "":
        integrator_type = "LoadControl"
    var use_banded_loadcontrol = (
        use_banded_nonlinear
        and integrator_type == "LoadControl"
        and primary_algorithm_mode != NonlinearAlgorithmMode.ModifiedNewtonInitial
        and not (
            has_fallback
            and fallback_algorithm_mode == NonlinearAlgorithmMode.ModifiedNewtonInitial
        )
    )
    var bw_nl = 0
    if use_banded_loadcontrol:
        bw_nl = estimate_bandwidth_typed(typed_elements, free_index)
        if bw_nl > free_count - 1:
            bw_nl = free_count - 1
    var K_ff: List[List[Float64]] = []
    var K_init_ff: List[List[Float64]] = []
    var K_step: List[List[Float64]] = []
    var K_lu: List[List[Float64]] = []
    var lu_pivots: List[Int] = []
    var lu_rhs: List[Float64] = []
    var lu_work: List[Float64] = []
    var u_f_work: List[Float64] = []
    if not use_banded_loadcontrol:
        for _ in range(free_count):
            var row_ff: List[Float64] = []
            row_ff.resize(free_count, 0.0)
            K_ff.append(row_ff^)
            var row_kinit: List[Float64] = []
            row_kinit.resize(free_count, 0.0)
            K_init_ff.append(row_kinit^)
            var row_step: List[Float64] = []
            row_step.resize(free_count, 0.0)
            K_step.append(row_step^)
            var row_lu: List[Float64] = []
            row_lu.resize(free_count, 0.0)
            K_lu.append(row_lu^)
        lu_pivots.resize(free_count, 0)
        lu_rhs.resize(free_count, 0.0)
        lu_work.resize(free_count, 0.0)
        u_f_work.resize(free_count, 0.0)
    var K_ff_banded: List[List[Float64]] = []
    var K_ff_banded_step: List[List[Float64]] = []
    var banded_rhs: List[Float64] = []
    if use_banded_loadcontrol:
        K_ff_banded = banded_matrix(free_count, bw_nl)
        K_ff_banded_step = banded_matrix(free_count, bw_nl)
        banded_rhs.resize(free_count, 0.0)
    if primary_algorithm_mode == NonlinearAlgorithmMode.ModifiedNewtonInitial or (
        has_fallback and fallback_algorithm_mode == NonlinearAlgorithmMode.ModifiedNewtonInitial
    ):
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
    var F_f: List[Float64] = []
    F_f.resize(free_count, 0.0)
    var has_reaction_recorder = _has_recorder_type(recorders, RecorderTypeTag.NodeReaction)
    var envelope_files: List[String] = []
    var envelope_min: List[List[Float64]] = []
    var envelope_max: List[List[Float64]] = []
    var envelope_abs: List[List[Float64]] = []
    for step in range(steps):
        if do_profile:
            var t_step_start = Int(time.perf_counter_ns())
            var step_start_us = (t_step_start - t0) // 1000
            _append_event(
                events, events_need_comma, "O", frame_nonlinear_step, step_start_us
            )
        if has_transformation_mpc:
            _enforce_equal_dof_values(u, rep_dof, constrained)
        var scale = Float64(step + 1) / Float64(steps)
        if analysis.has_integrator_step:
            scale = Float64(step + 1) * analysis.integrator_step
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
                for i in range(total_dofs):
                    u[i] = u_base[i]
                force_basic_q = force_basic_q_base.copy()
                attempt_algorithm_mode = fallback_algorithm_mode
                attempt_test_mode = fallback_test_mode
                attempt_max_iters = fallback_max_iters
                attempt_tol = fallback_tol
                attempt_rel_tol = fallback_rel_tol

            var tangent_initialized = False
            var tangent_factored = False
            for _ in range(attempt_max_iters):
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
                if use_banded_loadcontrol:
                    var refresh_banded_tangent = (
                        attempt_algorithm_mode == NonlinearAlgorithmMode.Newton
                        or (
                            attempt_algorithm_mode == NonlinearAlgorithmMode.ModifiedNewton
                            and not tangent_initialized
                        )
                    )
                    if refresh_banded_tangent:
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
                        for i in range(free_count):
                            var row_i = free[i]
                            F_f[i] = F_step[row_i] - F_int[row_i]
                            for j in range(free_count):
                                K_ff[i][j] = K[row_i][free[j]]
                    elif attempt_algorithm_mode == NonlinearAlgorithmMode.ModifiedNewton:
                        if not tangent_initialized:
                            for i in range(free_count):
                                var row_i = free[i]
                                F_f[i] = F_step[row_i] - F_int[row_i]
                                for j in range(free_count):
                                    K_ff[i][j] = K[row_i][free[j]]
                            tangent_initialized = True
                        else:
                            for i in range(free_count):
                                F_f[i] = F_step[free[i]] - F_int[free[i]]
                    else:
                        if not tangent_initialized:
                            for i in range(free_count):
                                for j in range(free_count):
                                    K_ff[i][j] = K_init_ff[i][j]
                            tangent_initialized = True
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
                    if attempt_algorithm_mode == NonlinearAlgorithmMode.Newton:
                        for i in range(free_count):
                            lu_rhs[i] = F_f[i]
                            for j in range(free_count):
                                K_step[i][j] = K_ff[i][j]
                        gaussian_elimination_into(K_step, lu_rhs, u_f_work)
                    else:
                        if not tangent_factored:
                            for i in range(free_count):
                                for j in range(free_count):
                                    K_lu[i][j] = K_ff[i][j]
                            lu_factorize_into(K_lu, lu_pivots)
                            tangent_factored = True
                        for i in range(free_count):
                            lu_rhs[i] = F_f[i]
                        lu_solve_into(K_lu, lu_pivots, lu_rhs, lu_work, u_f_work)
                    u_f = u_f_work.copy()
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
        var F_int_reaction: List[Float64] = []
        var F_ext_reaction: List[Float64] = []
        if has_reaction_recorder:
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
                        static_output_files, static_output_buffers, filename, line
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
                    var f_elem = _element_force_global_for_recorder(
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
                    var f_elem = _element_force_global_for_recorder(
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

fn run_static_nonlinear_displacement_control(
    analysis: AnalysisInput,
    mut steps: Int,
    ts_index: Int,
    time_series: List[TimeSeriesInput],
    analysis_integrator_targets_pool: List[Float64],
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
    var max_iters = analysis.max_iters
    var tol = analysis.tol
    var rel_tol = analysis.rel_tol
    var has_force_beam_column2d = False
    for i in range(len(typed_elements)):
        if typed_elements[i].type_tag == ElementTypeTag.ForceBeamColumn2d:
            has_force_beam_column2d = True
            break
    var primary_algorithm_mode = _static_nonlinear_algorithm_mode(analysis.algorithm, "static_nonlinear")
    var fallback_algorithm = analysis.fallback_algorithm
    var has_fallback = False
    var fallback_algorithm_mode = NonlinearAlgorithmMode.Unknown
    if len(fallback_algorithm) > 0:
        fallback_algorithm_mode = _static_nonlinear_algorithm_mode(
            fallback_algorithm, "static_nonlinear fallback"
        )
        has_fallback = True
    elif has_force_beam_column2d and (
        primary_algorithm_mode == NonlinearAlgorithmMode.Newton
        or primary_algorithm_mode == NonlinearAlgorithmMode.ModifiedNewton
    ):
        fallback_algorithm_mode = NonlinearAlgorithmMode.ModifiedNewtonInitial
        has_fallback = True
    elif primary_algorithm_mode != NonlinearAlgorithmMode.Newton:
        fallback_algorithm_mode = NonlinearAlgorithmMode.Newton
        has_fallback = True
    var primary_test_mode = _static_nonlinear_test_mode(
        analysis.test_type, "static_nonlinear"
    )
    var fallback_test_mode = _static_nonlinear_test_mode(
        analysis.fallback_test_type, "static_nonlinear fallback"
    )
    if max_iters < 1:
        abort("max_iters must be >= 1")
    var fallback_max_iters = analysis.fallback_max_iters
    var fallback_tol = analysis.fallback_tol
    var fallback_rel_tol = analysis.fallback_rel_tol
    if len(fallback_algorithm) == 0 and has_force_beam_column2d and has_fallback:
        fallback_test_mode = 1
        if fallback_max_iters < 100:
            fallback_max_iters = 100
        if fallback_max_iters < max_iters * 5:
            fallback_max_iters = max_iters * 5
        fallback_tol = tol
        fallback_rel_tol = rel_tol
    if fallback_max_iters < 1:
        abort("static_nonlinear fallback_max_iters must be >= 1")
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
    var target_tol = 1.0e-14
    if cutback <= 0.0 or cutback >= 1.0:
        abort("DisplacementControl cutback must be in (0, 1)")
    if max_cutbacks < 0:
        abort("DisplacementControl max_cutbacks must be >= 0")
    if min_du <= 0.0:
        abort("DisplacementControl min_du must be > 0")

    var target_disps: List[Float64] = []
    if analysis.integrator_targets_count > 0:
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
        for i in range(steps):
            target_disps.append(du_step * Float64(i + 1))

    var load_factor = 0.0
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
    if primary_algorithm_mode == NonlinearAlgorithmMode.ModifiedNewtonInitial or (
        has_fallback and fallback_algorithm_mode == NonlinearAlgorithmMode.ModifiedNewtonInitial
    ):
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

    for step in range(steps):
        if do_profile:
            var t_step_start = Int(time.perf_counter_ns())
            var step_start_us = (t_step_start - t0) // 1000
            _append_event(
                events, events_need_comma, "O", frame_nonlinear_step, step_start_us
            )
        if has_transformation_mpc:
            _enforce_equal_dof_values(u, rep_dof, constrained)

        var target = target_disps[step]
        while True:
            var remaining = target - u[control_idx]
            # `min_du` is the cutback floor, not the target-reached tolerance.
            # Using it here skips exact-size displacement steps whenever
            # `remaining == min_du`, which is common for fixed-increment pushover.
            if abs(remaining) <= target_tol:
                break

            var u_base = u.copy()
            var force_basic_q_base = force_basic_q.copy()
            var lambda_base = load_factor
            var attempt_du = remaining
            var attempt_ok = False

            for _ in range(max_cutbacks + 1):
                var converged = False
                var attempt_algorithm_mode = primary_algorithm_mode
                var attempt_test_mode = primary_test_mode
                var attempt_max_iters = max_iters
                var attempt_tol = tol
                var attempt_rel_tol = rel_tol
                for attempt in range(2):
                    if attempt == 1 and not has_fallback:
                        break
                    for i in range(total_dofs):
                        u[i] = u_base[i]
                    force_basic_q = force_basic_q_base.copy()
                    load_factor = lambda_base
                    if attempt == 1:
                        attempt_algorithm_mode = fallback_algorithm_mode
                        attempt_test_mode = fallback_test_mode
                        attempt_max_iters = fallback_max_iters
                        attempt_tol = fallback_tol
                        attempt_rel_tol = fallback_rel_tol

                    var tangent_initialized = False
                    for _ in range(attempt_max_iters):
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
                        var load_ext_scale = load_scale
                        if attempt_algorithm_mode == NonlinearAlgorithmMode.Newton:
                            for i in range(free_count):
                                var row_i = free[i]
                                R_f[i] = (
                                    F_const_free[i]
                                    + load_ext_scale * F_pattern_free[i]
                                    - F_int[row_i]
                                )
                                for j in range(free_count):
                                    K_ff[i][j] = K[row_i][free[j]]
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
                        for i in range(free_count):
                            var idx = free[i]
                            var du = sol_aug[i]
                            var value = u[idx] + du
                            u[idx] = value
                            var diff = abs(du)
                            if diff > max_diff:
                                max_diff = diff
                            var abs_val = abs(value)
                            if abs_val > max_u:
                                max_u = abs_val
                        load_factor += sol_aug[free_count]
                        if has_transformation_mpc:
                            _enforce_equal_dof_values(u, rep_dof, constrained)
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
                            converged = True
                            break
                    if converged:
                        break
                if converged:
                    attempt_ok = True
                    break
                attempt_du *= cutback
                if abs(attempt_du) <= min_du:
                    break

            if not attempt_ok:
                abort("static_nonlinear did not converge (DisplacementControl)")

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
                        static_output_files, static_output_buffers, filename, line
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
                    var f_elem = _element_force_global_for_recorder(
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
                    var f_elem = _element_force_global_for_recorder(
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
