from collections import List
from os import abort
from python import Python, PythonObject

from solver.assembly import assemble_internal_forces_typed
from solver.dof import node_dof_index, require_dof_in_range
from solver.profile import (
    _append_event,
    _append_frame,
    _profile_enabled,
    _write_speedscope,
)

from solver.run_case.analysis.static_linear import run_static_linear
from solver.run_case.analysis.static_nonlinear import (
    run_static_nonlinear_displacement_control,
    run_static_nonlinear_load_control,
)
from solver.run_case.analysis.transient_linear import run_transient_linear
from solver.run_case.analysis.transient_nonlinear import run_transient_nonlinear
from solver.run_case.analysis.modal_eigen import run_modal_eigen
from solver.run_case.helpers import (
    _drift_value,
    _element_force_global_for_recorder,
    _enforce_equal_dof_values,
    _format_values_line,
    _has_recorder_type,
    _scaled_forces,
    _update_envelope,
)
from solver.run_case.loader import load_case_state


fn _write_output_chunk_files(
    out_dir: PythonObject, filenames: List[String], buffers: List[List[String]]
) raises:
    var builtins = Python.import_module("builtins")
    for i in range(len(filenames)):
        var file_path = out_dir.joinpath(filenames[i])
        var file_obj = builtins.open(file_path, "w")
        for j in range(len(buffers[i])):
            file_obj.write(PythonObject(buffers[i][j]))
        file_obj.close()


def run_case(data: PythonObject, output_path: String, profile_path: String):
    var time = Python.import_module("time")
    var t0 = Int(time.perf_counter_ns())
    var do_profile = _profile_enabled(profile_path)

    var frame_total = 0
    var frame_assemble = 1
    var frame_solve = 2
    var frame_output = 3
    var frame_assemble_stiffness = 4
    var frame_kff_extract = 5
    var frame_solve_linear = 6
    var frame_solve_nonlinear = 7
    var frame_nonlinear_step = 8
    var frame_nonlinear_iter = 9

    var frames = String()
    var events = String()
    var frames_need_comma = False
    var events_need_comma = False
    if do_profile:
        _append_frame(frames, frames_need_comma, "total")
        _append_frame(frames, frames_need_comma, "assemble")
        _append_frame(frames, frames_need_comma, "solve")
        _append_frame(frames, frames_need_comma, "output")
        _append_frame(frames, frames_need_comma, "assemble_stiffness")
        _append_frame(frames, frames_need_comma, "kff_extract")
        _append_frame(frames, frames_need_comma, "solve_linear")
        _append_frame(frames, frames_need_comma, "solve_nonlinear")
        _append_frame(frames, frames_need_comma, "nonlinear_step")
        _append_frame(frames, frames_need_comma, "nonlinear_iter")
        _append_event(events, events_need_comma, "O", frame_total, 0)
        _append_event(events, events_need_comma, "O", frame_assemble, 0)

    var state = load_case_state(data)

    var ndm = state.ndm
    var ndf = state.ndf
    var typed_nodes = state.typed_nodes.copy()
    var node_count = state.node_count
    var typed_elements = state.typed_elements.copy()
    var total_dofs = state.total_dofs
    var analysis = state.analysis
    var analysis_type = state.analysis_type
    var steps = state.steps
    var modal_num_modes = state.modal_num_modes
    var use_banded_linear = state.use_banded_linear
    var use_banded_nonlinear = state.use_banded_nonlinear
    var has_transformation_mpc = state.has_transformation_mpc
    var time_series = state.time_series.copy()
    var time_series_values = state.time_series_values.copy()
    var time_series_times = state.time_series_times.copy()
    var ts_index = state.ts_index
    var pattern_type = state.pattern_type
    var uniform_excitation_direction = state.uniform_excitation_direction
    var uniform_accel_ts_index = state.uniform_accel_ts_index
    var rayleigh_alpha_m = state.rayleigh_alpha_m
    var rayleigh_beta_k = state.rayleigh_beta_k
    var rayleigh_beta_k_init = state.rayleigh_beta_k_init
    var rayleigh_beta_k_comm = state.rayleigh_beta_k_comm
    var recorder_nodes_pool = state.recorder_nodes_pool.copy()
    var recorder_elements_pool = state.recorder_elements_pool.copy()
    var recorder_dofs_pool = state.recorder_dofs_pool.copy()
    var recorder_modes_pool = state.recorder_modes_pool.copy()
    var recorders = state.recorders.copy()

    var id_to_index = state.id_to_index.copy()
    var typed_sections_by_id = state.typed_sections_by_id.copy()
    var typed_materials_by_id = state.typed_materials_by_id.copy()
    var uniaxial_defs = state.uniaxial_defs.copy()
    var uniaxial_state_defs = state.uniaxial_state_defs.copy()
    var uniaxial_states = state.uniaxial_states.copy()
    var fiber_section_defs = state.fiber_section_defs.copy()
    var fiber_section_cells = state.fiber_section_cells.copy()
    var fiber_section_index_by_id = state.fiber_section_index_by_id.copy()
    var elem_id_to_index = state.elem_id_to_index.copy()
    var elem_uniaxial_offsets = state.elem_uniaxial_offsets.copy()
    var elem_uniaxial_counts = state.elem_uniaxial_counts.copy()
    var elem_uniaxial_state_ids = state.elem_uniaxial_state_ids.copy()
    var F_total = state.F_total.copy()
    var constrained = state.constrained.copy()
    var free = state.free.copy()
    var free_index = state.free_index.copy()
    var rep_dof = state.rep_dof.copy()
    var M_total = state.M_total.copy()
    var analysis_integrator_targets_pool = (
        state.analysis_integrator_targets_pool.copy()
    )

    var u: List[Float64] = []
    u.resize(total_dofs, 0.0)
    var transient_output_files: List[String] = []
    var transient_output_buffers: List[List[String]] = []
    var static_output_files: List[String] = []
    var static_output_buffers: List[List[String]] = []

    var t_solve_start = Int(time.perf_counter_ns())
    if do_profile:
        var assemble_end = (t_solve_start - t0) // 1000
        _append_event(events, events_need_comma, "C", frame_assemble, assemble_end)
        _append_event(events, events_need_comma, "O", frame_solve, assemble_end)

    if analysis_type == "static_linear":
        run_static_linear(
            typed_nodes,
            typed_elements,
            typed_sections_by_id,
            typed_materials_by_id,
            id_to_index,
            node_count,
            ndf,
            ndm,
            u,
            F_total,
            uniaxial_defs,
            uniaxial_state_defs,
            uniaxial_states,
            elem_uniaxial_offsets,
            elem_uniaxial_counts,
            elem_uniaxial_state_ids,
            fiber_section_defs,
            fiber_section_cells,
            fiber_section_index_by_id,
            use_banded_linear,
            free_index,
            free,
            ts_index,
            time_series,
            time_series_values,
            time_series_times,
            do_profile,
            t0,
            events,
            events_need_comma,
            frame_assemble_stiffness,
            frame_kff_extract,
            frame_solve_linear,
            total_dofs,
            has_transformation_mpc,
            rep_dof,
            constrained,
        )
    elif analysis_type == "static_nonlinear":
        var integrator_type = analysis.integrator_type
        if integrator_type == "":
            integrator_type = "LoadControl"
        if integrator_type == "LoadControl":
            run_static_nonlinear_load_control(
                analysis,
                steps,
                ts_index,
                time_series,
                time_series_values,
                time_series_times,
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
                total_dofs,
                F_total,
                use_banded_nonlinear,
                free,
                free_index,
                recorders,
                recorder_nodes_pool,
                recorder_elements_pool,
                recorder_dofs_pool,
                elem_id_to_index,
                static_output_files,
                static_output_buffers,
                do_profile,
                t0,
                events,
                events_need_comma,
                frame_assemble_stiffness,
                frame_kff_extract,
                frame_solve_nonlinear,
                frame_nonlinear_step,
                frame_nonlinear_iter,
                has_transformation_mpc,
                rep_dof,
                constrained,
            )
        elif integrator_type == "DisplacementControl":
            run_static_nonlinear_displacement_control(
                analysis,
                steps,
                ts_index,
                analysis_integrator_targets_pool,
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
                total_dofs,
                F_total,
                constrained,
                free,
                recorders,
                recorder_nodes_pool,
                recorder_elements_pool,
                recorder_dofs_pool,
                elem_id_to_index,
                static_output_files,
                static_output_buffers,
                do_profile,
                t0,
                events,
                events_need_comma,
                frame_assemble_stiffness,
                frame_kff_extract,
                frame_solve_nonlinear,
                frame_nonlinear_step,
                frame_nonlinear_iter,
                has_transformation_mpc,
                rep_dof,
            )
        else:
            abort("unsupported static_nonlinear integrator: " + integrator_type)
    elif analysis_type == "transient_linear":
        run_transient_linear(
            analysis,
            steps,
            ts_index,
            time_series,
            time_series_values,
            time_series_times,
            pattern_type,
            uniform_excitation_direction,
            uniform_accel_ts_index,
            rayleigh_alpha_m,
            rayleigh_beta_k,
            rayleigh_beta_k_init,
            rayleigh_beta_k_comm,
            typed_nodes,
            typed_elements,
            typed_sections_by_id,
            typed_materials_by_id,
            id_to_index,
            node_count,
            ndf,
            ndm,
            total_dofs,
            u,
            uniaxial_defs,
            uniaxial_state_defs,
            uniaxial_states,
            elem_uniaxial_offsets,
            elem_uniaxial_counts,
            elem_uniaxial_state_ids,
            F_total,
            M_total,
            free,
            recorders,
            recorder_nodes_pool,
            recorder_elements_pool,
            recorder_dofs_pool,
            elem_id_to_index,
            fiber_section_defs,
            fiber_section_cells,
            fiber_section_index_by_id,
            transient_output_files,
            transient_output_buffers,
            has_transformation_mpc,
            rep_dof,
            constrained,
        )
    elif analysis_type == "transient_nonlinear":
        run_transient_nonlinear(
            analysis,
            steps,
            ts_index,
            time_series,
            time_series_values,
            time_series_times,
            pattern_type,
            uniform_excitation_direction,
            uniform_accel_ts_index,
            rayleigh_alpha_m,
            rayleigh_beta_k,
            rayleigh_beta_k_init,
            rayleigh_beta_k_comm,
            typed_nodes,
            typed_elements,
            typed_sections_by_id,
            typed_materials_by_id,
            id_to_index,
            node_count,
            ndf,
            ndm,
            total_dofs,
            u,
            uniaxial_defs,
            uniaxial_state_defs,
            uniaxial_states,
            elem_uniaxial_offsets,
            elem_uniaxial_counts,
            elem_uniaxial_state_ids,
            F_total,
            M_total,
            free,
            recorders,
            recorder_nodes_pool,
            recorder_elements_pool,
            recorder_dofs_pool,
            elem_id_to_index,
            fiber_section_defs,
            fiber_section_cells,
            fiber_section_index_by_id,
            transient_output_files,
            transient_output_buffers,
            has_transformation_mpc,
            rep_dof,
            constrained,
        )
    elif analysis_type == "modal_eigen":
        run_modal_eigen(
            modal_num_modes,
            typed_nodes,
            typed_elements,
            typed_sections_by_id,
            typed_materials_by_id,
            id_to_index,
            node_count,
            ndf,
            ndm,
            total_dofs,
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
            M_total,
            constrained,
            free,
            has_transformation_mpc,
            rep_dof,
            recorders,
            recorder_nodes_pool,
            recorder_dofs_pool,
            recorder_modes_pool,
            static_output_files,
            static_output_buffers,
        )
    else:
        abort("unsupported analysis type: " + analysis_type)
    var t_solve_end = Int(time.perf_counter_ns())
    if do_profile:
        var solve_end_us = (t_solve_end - t0) // 1000
        _append_event(events, events_need_comma, "C", frame_solve, solve_end_us)
        _append_event(events, events_need_comma, "O", frame_output, solve_end_us)

    # Match OpenSees timing injection (`analyze`/`eigen`) by timing solve only.
    var analysis_us = (t_solve_end - t_solve_start) // 1000

    var pathlib = Python.import_module("pathlib")
    var out_dir = pathlib.Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    var analysis_path = out_dir.joinpath("analysis_time_us.txt")
    analysis_path.write_text(PythonObject(String(analysis_us) + "\n"))
    if analysis_type == "transient_linear" or analysis_type == "transient_nonlinear":
        _write_output_chunk_files(
            out_dir, transient_output_files, transient_output_buffers
        )
    elif analysis_type == "modal_eigen":
        _write_output_chunk_files(out_dir, static_output_files, static_output_buffers)
    elif analysis_type == "static_nonlinear" and len(static_output_files) > 0:
        _write_output_chunk_files(out_dir, static_output_files, static_output_buffers)
    else:
        if has_transformation_mpc:
            _enforce_equal_dof_values(u, rep_dof, constrained)
        var has_reaction_recorder = _has_recorder_type(recorders, 3)
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
            F_ext_reaction = _scaled_forces(F_total, 1.0)
        var envelope_files: List[String] = []
        var envelope_min: List[List[Float64]] = []
        var envelope_max: List[List[Float64]] = []
        var envelope_abs: List[List[Float64]] = []
        for r in range(len(recorders)):
            var rec = recorders[r]
            if rec.type_tag == 1:
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
                    var file_path = out_dir.joinpath(filename)
                    file_path.write_text(PythonObject(line))
            elif rec.type_tag == 2:
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
                        uniaxial_states,
                        elem_uniaxial_offsets,
                        elem_uniaxial_counts,
                        elem_uniaxial_state_ids,
                    )
                    var line = _format_values_line(f_elem)
                    var filename = rec.output + "_ele" + String(elem_id) + ".out"
                    var file_path = out_dir.joinpath(filename)
                    file_path.write_text(PythonObject(line))
            elif rec.type_tag == 3:
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
                    var file_path = out_dir.joinpath(filename)
                    file_path.write_text(PythonObject(_format_values_line(values)))
            elif rec.type_tag == 4:
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
                var file_path = out_dir.joinpath(filename)
                file_path.write_text(PythonObject(_format_values_line([value])))
            elif rec.type_tag == 5:
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
        for i in range(len(envelope_files)):
            var line = String()
            line += _format_values_line(envelope_min[i])
            line += _format_values_line(envelope_max[i])
            line += _format_values_line(envelope_abs[i])
            var file_path = out_dir.joinpath(envelope_files[i])
            file_path.write_text(PythonObject(line))

    var t2 = Int(time.perf_counter_ns())
    if do_profile:
        var total_us = (t2 - t0) // 1000
        _append_event(events, events_need_comma, "C", frame_output, total_us)
        _append_event(events, events_need_comma, "C", frame_total, total_us)
        _write_speedscope(profile_path, frames, events, total_us)
