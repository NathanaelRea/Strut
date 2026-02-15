from collections import List
from math import sqrt
from materials import (
    UniMaterialDef,
    UniMaterialState,
    uniaxial_commit_all,
    uniaxial_revert_trial_all,
)
from os import abort
from sections import FiberCell, FiberSection2dDef

from linalg import gaussian_elimination_into
from solver.assembly import (
    assemble_global_stiffness_and_internal,
    assemble_internal_forces_typed,
)
from solver.dof import node_dof_index, require_dof_in_range
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
    _update_envelope,
)
from tag_types import NonlinearAlgorithmMode, RecorderTypeTag


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
    elem_id_to_index: List[Int],
    fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    mut transient_output_files: List[String],
    mut transient_output_buffers: List[List[String]],
    has_transformation_mpc: Bool,
    rep_dof: List[Int],
    constrained: List[Bool],
) raises:
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

    var algorithm = analysis.algorithm
    var primary_algorithm_mode = NonlinearAlgorithmMode.Unknown
    if algorithm == "Newton":
        primary_algorithm_mode = NonlinearAlgorithmMode.Newton
    elif algorithm == "ModifiedNewton":
        primary_algorithm_mode = NonlinearAlgorithmMode.ModifiedNewton
    elif algorithm == "ModifiedNewtonInitial":
        primary_algorithm_mode = NonlinearAlgorithmMode.ModifiedNewtonInitial
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

    var test_type = analysis.test_type
    if (
        test_type != "MaxDispIncr"
        and test_type != "NormDispIncr"
        and test_type != "NormUnbalance"
    ):
        abort("unsupported transient_nonlinear test_type: " + test_type)
    var fallback_test_type = analysis.fallback_test_type
    if (
        fallback_test_type != "MaxDispIncr"
        and fallback_test_type != "NormDispIncr"
        and fallback_test_type != "NormUnbalance"
    ):
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

    uniaxial_revert_trial_all(uniaxial_states)
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
        K,
        F_int,
    )
    if has_transformation_mpc:
        K = _collapse_matrix_by_rep(K, rep_dof)
    for i in range(free_count):
        for j in range(free_count):
            K_init_ff[i][j] = K[free[i]][free[j]]
            K_comm_ff[i][j] = K_init_ff[i][j]
    uniaxial_revert_trial_all(uniaxial_states)

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

    var record_reactions = _has_recorder_type(recorders, RecorderTypeTag.NodeReaction)
    var envelope_files: List[String] = []
    var envelope_min: List[List[Float64]] = []
    var envelope_max: List[List[Float64]] = []
    var envelope_abs: List[List[Float64]] = []

    for step in range(steps):
        if has_transformation_mpc:
            _enforce_equal_dof_values(u, rep_dof, constrained)
            _enforce_equal_dof_values(v, rep_dof, constrained)
            _enforce_equal_dof_values(a, rep_dof, constrained)

        for i in range(free_count):
            var idx = free[i]
            u_n[i] = u[idx]
            v_n[i] = v[idx]
            a_n[i] = a[idx]

        var t = Float64(step + 1) * dt
        for i in range(total_dofs):
            F_ext_step[i] = 0.0
        if pattern_type == "UniformExcitation":
            var ag = eval_time_series_input(
                time_series[uniform_accel_ts_index],
                t,
                time_series_values,
                time_series_times,
            )
            for i in range(total_dofs):
                if (i % ndf) + 1 == uniform_excitation_direction:
                    F_ext_step[i] = -M_total[i] * ag
        else:
            var factor = 1.0
            if ts_index >= 0:
                factor = eval_time_series_input(
                    time_series[ts_index],
                    t,
                    time_series_values,
                    time_series_times,
                )
            for i in range(total_dofs):
                F_ext_step[i] = F_total[i] * factor
        for i in range(free_count):
            P_ext_f[i] = F_ext_step[free[i]]

        var converged = False
        var attempt_algorithm_mode = primary_algorithm_mode
        var attempt_test_type = test_type
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
                attempt_test_type = fallback_test_type
                attempt_max_iters = fallback_max_iters
                attempt_tol = fallback_tol
                attempt_rel_tol = fallback_rel_tol

            var tangent_initialized = False
            for _ in range(attempt_max_iters):
                uniaxial_revert_trial_all(uniaxial_states)
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
                    K,
                    F_int,
                )
                if has_transformation_mpc:
                    K = _collapse_matrix_by_rep(K, rep_dof)
                    F_int = _collapse_vector_by_rep(F_int, rep_dof)
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

                for i in range(free_count):
                    var idx = free[i]
                    a_trial[i] = a0 * (u[idx] - u_n[i]) - a2 * v_n[i] - a3 * a_n[i]
                    v_trial[i] = v_n[i] + dt * (
                        (1.0 - gamma) * a_n[i] + gamma * a_trial[i]
                    )

                for i in range(free_count):
                    var damping_force = 0.0
                    for j in range(free_count):
                        damping_force += C_ff[i][j] * v_trial[j]
                    R_f[i] = (
                        P_ext_f[i]
                        - F_int[free[i]]
                        - damping_force
                        - M_f[i] * a_trial[i]
                    )
                if attempt_test_type == "NormUnbalance":
                    var residual_norm_sq = 0.0
                    for i in range(free_count):
                        residual_norm_sq += R_f[i] * R_f[i]
                    var residual_norm = sqrt(residual_norm_sq)
                    if residual_norm <= attempt_tol:
                        converged = True
                        break

                for i in range(free_count):
                    for j in range(free_count):
                        K_eff[i][j] = K_ff[i][j] + a1 * C_ff[i][j]
                    K_eff[i][i] += a0 * M_f[i]

                for i in range(free_count):
                    for j in range(free_count):
                        K_step[i][j] = K_eff[i][j]
                for i in range(free_count):
                    R_step[i] = R_f[i]
                gaussian_elimination_into(K_step, R_step, du_f)

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
                var disp_incr_norm_sq = 0.0
                for i in range(free_count):
                    disp_incr_norm_sq += du_f[i] * du_f[i]
                var disp_incr_norm = sqrt(disp_incr_norm_sq)
                var scale_tol = attempt_rel_tol * max_u
                if scale_tol < attempt_rel_tol:
                    scale_tol = attempt_rel_tol

                for i in range(free_count):
                    u[free[i]] += du_f[i]
                if has_transformation_mpc:
                    _enforce_equal_dof_values(u, rep_dof, constrained)

                if attempt_test_type == "NormDispIncr":
                    if disp_incr_norm <= attempt_tol:
                        converged = True
                        break
                elif attempt_test_type == "MaxDispIncr":
                    if max_diff <= attempt_tol or max_diff <= scale_tol:
                        converged = True
                        break
            if converged:
                break

        if not converged:
            abort("transient_nonlinear did not converge at step " + String(step + 1))

        uniaxial_revert_trial_all(uniaxial_states)
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
            K,
            F_int,
        )
        if has_transformation_mpc:
            K = _collapse_matrix_by_rep(K, rep_dof)
        for i in range(free_count):
            for j in range(free_count):
                K_comm_ff[i][j] = K[free[i]][free[j]]
        uniaxial_commit_all(uniaxial_states)

        for i in range(free_count):
            var idx = free[i]
            var a_next = a0 * (u[idx] - u_n[i]) - a2 * v_n[i] - a3 * a_n[i]
            var v_next = v_n[i] + dt * ((1.0 - gamma) * a_n[i] + gamma * a_next)
            a[idx] = a_next
            v[idx] = v_next
        if has_transformation_mpc:
            _enforce_equal_dof_values(u, rep_dof, constrained)
            _enforce_equal_dof_values(v, rep_dof, constrained)
            _enforce_equal_dof_values(a, rep_dof, constrained)

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
                    var elem = typed_elements[elem_index]
                    var f_elem = _element_force_global_for_recorder(
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
                    var line = _format_values_line(f_elem)
                    var filename = rec.output + "_ele" + String(elem_id) + ".out"
                    _append_output(
                        transient_output_files, transient_output_buffers, filename, line
                    )
            elif rec.type_tag == RecorderTypeTag.NodeReaction:
                if not record_reactions:
                    abort("internal error: reaction recorder flag mismatch")
                var F_int_reaction = assemble_internal_forces_typed(
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
                    var elem = typed_elements[elem_index]
                    var f_elem = _element_force_global_for_recorder(
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
                    var filename = rec.output + "_ele" + String(elem_id) + ".out"
                    _update_envelope(
                        filename,
                        f_elem,
                        envelope_files,
                        envelope_min,
                        envelope_max,
                        envelope_abs,
                    )
            else:
                abort("unsupported recorder type")
    _flush_envelope_outputs(
        envelope_files,
        envelope_min,
        envelope_max,
        envelope_abs,
        transient_output_files,
        transient_output_buffers,
    )
