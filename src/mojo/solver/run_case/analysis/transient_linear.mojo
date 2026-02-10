from collections import List
from materials import UniMaterialDef, UniMaterialState
from os import abort
from sections import FiberCell, FiberSection2dDef

from linalg import gaussian_elimination
from solver.assembly import assemble_global_stiffness_typed, assemble_internal_forces_typed
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
    _element_force_global_for_recorder,
    _enforce_equal_dof_values,
    _flush_envelope_outputs,
    _format_values_line,
    _has_recorder_type,
    _drift_value,
    _update_envelope,
)
from tag_types import RecorderTypeTag

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
        abort("transient_linear requires dt > 0")
    if pattern_type != "Plain" and pattern_type != "UniformExcitation":
        abort("unsupported pattern type: " + pattern_type)
    if pattern_type == "UniformExcitation":
        if uniform_excitation_direction < 1 or uniform_excitation_direction > ndm:
            abort("UniformExcitation direction out of range")
        if uniform_accel_ts_index < 0:
            abort("UniformExcitation missing accel time_series")
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

    var K = assemble_global_stiffness_typed(
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
    if has_transformation_mpc:
        K = _collapse_matrix_by_rep(K, rep_dof)
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
    var record_reactions = _has_recorder_type(recorders, RecorderTypeTag.NodeReaction)
    var record_any_element_force = (
        _has_recorder_type(recorders, RecorderTypeTag.ElementForce) or _has_recorder_type(recorders, RecorderTypeTag.EnvelopeElementForce)
    )
    var envelope_files: List[String] = []
    var envelope_min: List[List[Float64]] = []
    var envelope_max: List[List[Float64]] = []
    var envelope_abs: List[List[Float64]] = []

    for step in range(steps):
        if has_transformation_mpc:
            _enforce_equal_dof_values(u, rep_dof, constrained)
            _enforce_equal_dof_values(v, rep_dof, constrained)
            _enforce_equal_dof_values(a, rep_dof, constrained)
        var t = Float64(step + 1) * dt
        for i in range(total_dofs):
            F_ext_step[i] = 0.0
        var factor = 1.0
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
            var idx = free[i]
            P_ext_f[i] = F_ext_step[idx]
            C_term[i] = a1 * u[idx] + a4 * v[idx] + a5 * a[idx]
            P_eff[i] = (
                P_ext_f[i]
                + M_f[i] * (a0 * u[idx] + a2 * v[idx] + a3 * a[idx])
            )
        for i in range(free_count):
            var damp = 0.0
            for j in range(free_count):
                damp += C_ff[i][j] * C_term[j]
            P_eff[i] += damp
        var K_eff_step: List[List[Float64]] = []
        for i in range(free_count):
            var row: List[Float64] = []
            row.resize(free_count, 0.0)
            K_eff_step.append(row^)
            for j in range(free_count):
                K_eff_step[i][j] = K_eff[i][j]
        var P_step: List[Float64] = []
        P_step.resize(free_count, 0.0)
        for i in range(free_count):
            P_step[i] = P_eff[i]
        var u_f = gaussian_elimination(K_eff_step, P_step)
        for i in range(free_count):
            var idx = free[i]
            var u_next = u_f[i]
            var a_next = a0 * (u_next - u[idx]) - a2 * v[idx] - a3 * a[idx]
            var v_next = v[idx] + dt * ((1.0 - gamma) * a[idx] + gamma * a_next)
            u[idx] = u_next
            v[idx] = v_next
            a[idx] = a_next
        if has_transformation_mpc:
            _enforce_equal_dof_values(u, rep_dof, constrained)
            _enforce_equal_dof_values(v, rep_dof, constrained)
            _enforce_equal_dof_values(a, rep_dof, constrained)

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
                fiber_section_defs,
                fiber_section_cells,
                fiber_section_index_by_id,
            )

        var elem_force_cached: List[Bool] = []
        var elem_force_values: List[List[Float64]] = []
        if record_any_element_force:
            elem_force_cached.resize(len(typed_elements), False)
            for _ in range(len(typed_elements)):
                var empty_force: List[Float64] = []
                elem_force_values.append(empty_force^)

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
