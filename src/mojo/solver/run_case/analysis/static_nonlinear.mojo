from collections import List
from materials import UniMaterialDef, UniMaterialState
from os import abort
from python import Python

from linalg import gaussian_elimination
from materials import uniaxial_commit_all, uniaxial_revert_trial_all
from solver.assembly import (
    assemble_global_stiffness_and_internal,
    assemble_internal_forces_typed,
)
from solver.banded import banded_gaussian_elimination, banded_matrix, estimate_bandwidth_typed
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
from sections import FiberCell, FiberSection2dDef
from tag_types import RecorderTypeTag, TimeSeriesTypeTag

from solver.run_case.helpers import (
    _append_output,
    _collapse_matrix_by_rep,
    _collapse_vector_by_rep,
    _drift_value,
    _element_force_global_for_recorder,
    _enforce_equal_dof_values,
    _flush_envelope_outputs,
    _format_values_line,
    _has_recorder_type,
    _scaled_forces,
    _solve_linear_system,
    _update_envelope,
)

fn run_static_nonlinear_load_control(
    analysis: AnalysisInput,
    steps: Int,
    ts_index: Int,
    time_series: List[TimeSeriesInput],
    time_series_values: List[Float64],
    time_series_times: List[Float64],
    typed_nodes: List[NodeInput],
    typed_elements: List[ElementInput],
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
    fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    total_dofs: Int,
    mut F_total: List[Float64],
    use_banded_nonlinear: Bool,
    free: List[Int],
    free_index: List[Int],
    recorders: List[RecorderInput],
    recorder_nodes_pool: List[Int],
    recorder_elements_pool: List[Int],
    recorder_dofs_pool: List[Int],
    elem_id_to_index: List[Int],
    mut static_output_files: List[String],
    mut static_output_buffers: List[List[String]],
    do_profile: Bool,
    t0: Int,
    mut events: String,
    mut events_need_comma: Bool,
    frame_assemble_stiffness: Int,
    frame_kff_extract: Int,
    frame_solve_nonlinear: Int,
    frame_nonlinear_step: Int,
    frame_nonlinear_iter: Int,
    has_transformation_mpc: Bool,
    rep_dof: List[Int],
    constrained: List[Bool],
) raises:
    var time = Python.import_module("time")
    var max_iters = analysis.max_iters
    var tol = analysis.tol
    var rel_tol = analysis.rel_tol
    var algorithm = analysis.algorithm
    var use_modified_newton = False
    if algorithm == "Newton":
        pass
    elif algorithm == "ModifiedNewton":
        use_modified_newton = True
    else:
        abort("unsupported static_nonlinear algorithm: " + algorithm)
    if max_iters < 1:
        abort("max_iters must be >= 1")
    var free_count = len(free)
    var F_total_free: List[Float64] = []
    F_total_free.resize(free_count, 0.0)
    for i in range(free_count):
        F_total_free[i] = F_total[free[i]]

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
    var use_banded_loadcontrol = use_banded_nonlinear and integrator_type == "LoadControl"
    var bw_nl = 0
    if use_banded_loadcontrol:
        bw_nl = estimate_bandwidth_typed(typed_elements, free_index)
        if bw_nl > free_count - 1:
            bw_nl = free_count - 1
    var K_ff: List[List[Float64]] = []
    if not use_banded_loadcontrol:
        for _ in range(free_count):
            var row_ff: List[Float64] = []
            row_ff.resize(free_count, 0.0)
            K_ff.append(row_ff^)
    var K_ff_banded: List[List[Float64]] = []
    if use_banded_loadcontrol:
        K_ff_banded = banded_matrix(free_count, bw_nl)
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
        if ts_index >= 0:
            scale = eval_time_series_input(
                time_series[ts_index], scale, time_series_values, time_series_times
            )
        var converged = False
        var tangent_initialized = False
        for _ in range(max_iters):
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
            uniaxial_revert_trial_all(uniaxial_states)
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
                fiber_section_defs,
                fiber_section_cells,
                fiber_section_index_by_id,
                K,
                F_int,
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
            for i in range(free_count):
                F_f[i] = F_total_free[i] * scale - F_int[free[i]]
            if use_banded_loadcontrol:
                if not use_modified_newton or not tangent_initialized:
                    var width = bw_nl * 2 + 1
                    for i in range(free_count):
                        for j in range(width):
                            K_ff_banded[i][j] = 0.0
                    for i in range(free_count):
                        var row_i = free[i]
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
                if not use_modified_newton or not tangent_initialized:
                    for i in range(free_count):
                        for j in range(free_count):
                            K_ff[i][j] = K[free[i]][free[j]]
                    tangent_initialized = True
            if do_profile:
                var t_kff_end = Int(time.perf_counter_ns())
                var kff_end_us = (t_kff_end - t0) // 1000
                _append_event(
                    events, events_need_comma, "C", frame_kff_extract, kff_end_us
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
            var u_f: List[Float64]
            if use_banded_loadcontrol:
                u_f = banded_gaussian_elimination(K_ff_banded, bw_nl, F_f)
            else:
                u_f = gaussian_elimination(K_ff, F_f)
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
            for i in range(len(free)):
                var idx = free[i]
                var du = u_f[i]
                var value = u[idx] + du
                var diff = abs(du)
                if diff > max_diff:
                    max_diff = diff
                var abs_val = abs(value)
                if abs_val > max_u:
                    max_u = abs_val
            var scale_tol = rel_tol * max_u
            if scale_tol < rel_tol:
                scale_tol = rel_tol
            var converged_iter = False
            if max_diff <= tol or max_diff <= scale_tol:
                converged = True
                converged_iter = True
            for i in range(len(free)):
                u[free[i]] += u_f[i]
            if has_transformation_mpc:
                _enforce_equal_dof_values(u, rep_dof, constrained)
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
                break
        if do_profile:
            var t_step_end = Int(time.perf_counter_ns())
            var step_end_us = (t_step_end - t0) // 1000
            _append_event(
                events, events_need_comma, "C", frame_nonlinear_step, step_end_us
            )
        if not converged:
            abort("static_nonlinear did not converge")
        uniaxial_commit_all(uniaxial_states)
        var F_int_reaction: List[Float64] = []
        var F_ext_reaction: List[Float64] = []
        if has_reaction_recorder:
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
                fiber_section_defs,
                fiber_section_cells,
                fiber_section_index_by_id,
            )
            F_ext_reaction = _scaled_forces(F_total, scale)
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
                    )
                    var line = _format_values_line(f_elem)
                    var filename = rec.output + "_ele" + String(elem_id) + ".out"
                    _append_output(
                        static_output_files, static_output_buffers, filename, line
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
    fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    total_dofs: Int,
    mut F_total: List[Float64],
    constrained: List[Bool],
    free: List[Int],
    recorders: List[RecorderInput],
    recorder_nodes_pool: List[Int],
    recorder_elements_pool: List[Int],
    recorder_dofs_pool: List[Int],
    elem_id_to_index: List[Int],
    mut static_output_files: List[String],
    mut static_output_buffers: List[List[String]],
    do_profile: Bool,
    t0: Int,
    mut events: String,
    mut events_need_comma: Bool,
    frame_assemble_stiffness: Int,
    frame_kff_extract: Int,
    frame_solve_nonlinear: Int,
    frame_nonlinear_step: Int,
    frame_nonlinear_iter: Int,
    has_transformation_mpc: Bool,
    rep_dof: List[Int],
) raises:
    var time = Python.import_module("time")
    var max_iters = analysis.max_iters
    var tol = analysis.tol
    var rel_tol = analysis.rel_tol
    var algorithm = analysis.algorithm
    var use_modified_newton = False
    if algorithm == "Newton":
        pass
    elif algorithm == "ModifiedNewton":
        use_modified_newton = True
    else:
        abort("unsupported static_nonlinear algorithm: " + algorithm)
    if max_iters < 1:
        abort("max_iters must be >= 1")
    var free_count = len(free)
    var F_total_free: List[Float64] = []
    F_total_free.resize(free_count, 0.0)
    for i in range(free_count):
        F_total_free[i] = F_total[free[i]]

    var K: List[List[Float64]] = []
    for _ in range(total_dofs):
        var row: List[Float64] = []
        row.resize(total_dofs, 0.0)
        K.append(row^)
    var F_int: List[Float64] = []
    F_int.resize(total_dofs, 0.0)

    var K_ff: List[List[Float64]] = []
    for _ in range(free_count):
        var row_ff: List[Float64] = []
        row_ff.resize(free_count, 0.0)
        K_ff.append(row_ff^)

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
            if abs(remaining) <= min_du:
                break

            var u_base = u.copy()
            var lambda_base = load_factor
            var attempt_du = remaining
            var attempt_ok = False

            for _ in range(max_cutbacks + 1):
                for i in range(total_dofs):
                    u[i] = u_base[i]
                load_factor = lambda_base

                var converged = False
                var tangent_initialized = False
                for _ in range(max_iters):
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
                    uniaxial_revert_trial_all(uniaxial_states)
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
                        fiber_section_defs,
                        fiber_section_cells,
                        fiber_section_index_by_id,
                        K,
                        F_int,
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
                    var load_scale = load_scale_derivative * load_factor
                    for i in range(free_count):
                        R_f[i] = load_scale * F_total_free[i] - F_int[free[i]]
                    if not use_modified_newton or not tangent_initialized:
                        for i in range(free_count):
                            for j in range(free_count):
                                K_ff[i][j] = K[free[i]][free[j]]
                        tangent_initialized = True
                    if do_profile:
                        var t_kff_end = Int(time.perf_counter_ns())
                        var kff_end_us = (t_kff_end - t0) // 1000
                        _append_event(
                            events, events_need_comma, "C", frame_kff_extract, kff_end_us
                        )

                    for i in range(free_count):
                        rhs_aug[i] = R_f[i]
                        for j in range(free_count):
                            K_aug[i][j] = K_ff[i][j]
                        K_aug[i][free_count] = -load_scale_derivative * F_total_free[i]
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
                        converged = False
                        break

                    var max_diff = 0.0
                    var max_u = 0.0
                    for i in range(free_count):
                        var idx = free[i]
                        var du = sol_aug[i]
                        var value = u[idx] + du
                        var diff = abs(du)
                        if diff > max_diff:
                            max_diff = diff
                        var abs_val = abs(value)
                        if abs_val > max_u:
                            max_u = abs_val
                    for i in range(free_count):
                        u[free[i]] += sol_aug[i]
                    load_factor += sol_aug[free_count]
                    if has_transformation_mpc:
                        _enforce_equal_dof_values(u, rep_dof, constrained)
                    var scale_tol = rel_tol * max_u
                    if scale_tol < rel_tol:
                        scale_tol = rel_tol
                    if max_diff <= tol or max_diff <= scale_tol:
                        converged = True
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

            uniaxial_commit_all(uniaxial_states)

        if do_profile:
            var t_step_end = Int(time.perf_counter_ns())
            var step_end_us = (t_step_end - t0) // 1000
            _append_event(
                events, events_need_comma, "C", frame_nonlinear_step, step_end_us
            )

        var F_int_reaction: List[Float64] = []
        var F_ext_reaction: List[Float64] = []
        if has_reaction_recorder:
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
                fiber_section_defs,
                fiber_section_cells,
                fiber_section_index_by_id,
            )
            F_ext_reaction = _scaled_forces(F_total, load_factor)

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
                    )
                    var line = _format_values_line(f_elem)
                    var filename = rec.output + "_ele" + String(elem_id) + ".out"
                    _append_output(
                        static_output_files, static_output_buffers, filename, line
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
        static_output_files,
        static_output_buffers,
    )
