from collections import List
from os import abort
from python import Python, PythonObject

from solver.dof import node_dof_index, require_dof_in_range
from solver.profile import (
    _append_event,
    _append_frame,
    _profile_enabled,
    _write_speedscope,
)
from strut_io import py_len

from solver.run_case.analysis.static_linear import run_static_linear
from solver.run_case.analysis.static_nonlinear import (
    run_static_nonlinear_displacement_control,
    run_static_nonlinear_load_control,
)
from solver.run_case.analysis.transient_linear import run_transient_linear
from solver.run_case.helpers import (
    _beam2d_element_force_global,
    _force_beam_column2d_element_force_global,
    _truss_element_force_global,
)
from solver.run_case.loader import load_case_state


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
    var nodes = state.nodes
    var node_count = state.node_count
    var elements = state.elements
    var total_dofs = state.total_dofs
    var analysis = state.analysis
    var analysis_type = state.analysis_type
    var steps = state.steps
    var use_banded_linear = state.use_banded_linear
    var use_banded_nonlinear = state.use_banded_nonlinear
    var time_series = state.time_series
    var ts_index = state.ts_index
    var recorders = state.recorders

    var id_to_index = state.id_to_index.copy()
    var sections_by_id = state.sections_by_id.copy()
    var materials_by_id = state.materials_by_id.copy()
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
    var M_total = state.M_total.copy()

    var u: List[Float64] = []
    u.resize(total_dofs, 0.0)
    var transient_output_files: List[String] = []
    var transient_output_buffers: List[String] = []
    var static_output_files: List[String] = []
    var static_output_buffers: List[String] = []

    var t_solve_start = Int(time.perf_counter_ns())
    if do_profile:
        var assemble_end = (t_solve_start - t0) // 1000
        _append_event(events, events_need_comma, "C", frame_assemble, assemble_end)
        _append_event(events, events_need_comma, "O", frame_solve, assemble_end)

    if analysis_type == "static_linear":
        run_static_linear(
            nodes,
            elements,
            sections_by_id,
            materials_by_id,
            id_to_index,
            node_count,
            ndf,
            ndm,
            u,
            F_total,
            uniaxial_defs,
            uniaxial_state_defs,
            elem_uniaxial_offsets,
            elem_uniaxial_counts,
            elem_uniaxial_state_ids,
            use_banded_linear,
            free_index,
            free,
            ts_index,
            time_series,
            do_profile,
            t0,
            events,
            events_need_comma,
            frame_assemble_stiffness,
            frame_kff_extract,
            frame_solve_linear,
            total_dofs,
        )
    elif analysis_type == "static_nonlinear":
        var integrator = analysis.get("integrator", {"type": "LoadControl"})
        var integrator_type = String(integrator.get("type", "LoadControl"))
        if integrator_type == "LoadControl":
            run_static_nonlinear_load_control(
                analysis,
                steps,
                ts_index,
                time_series,
                nodes,
                elements,
                sections_by_id,
                materials_by_id,
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
            )
        elif integrator_type == "DisplacementControl":
            run_static_nonlinear_displacement_control(
                analysis,
                steps,
                ts_index,
                nodes,
                elements,
                sections_by_id,
                materials_by_id,
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
            )
        else:
            abort("unsupported static_nonlinear integrator: " + integrator_type)
    elif analysis_type == "transient_linear":
        run_transient_linear(
            analysis,
            steps,
            ts_index,
            time_series,
            nodes,
            elements,
            sections_by_id,
            materials_by_id,
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
            elem_id_to_index,
            fiber_section_defs,
            fiber_section_cells,
            fiber_section_index_by_id,
            transient_output_files,
            transient_output_buffers,
        )
    else:
        abort("unsupported analysis type: " + analysis_type)
    var t_solve_end = Int(time.perf_counter_ns())
    if do_profile:
        var solve_end_us = (t_solve_end - t0) // 1000
        _append_event(events, events_need_comma, "C", frame_solve, solve_end_us)
        _append_event(events, events_need_comma, "O", frame_output, solve_end_us)

    var t_output_start = t_solve_end
    var t1 = Int(time.perf_counter_ns())
    var analysis_us = (t_output_start - t0) // 1000

    var pathlib = Python.import_module("pathlib")
    var out_dir = pathlib.Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    var analysis_path = out_dir.joinpath("analysis_time_us.txt")
    analysis_path.write_text(PythonObject(String(analysis_us) + "\n"))
    if analysis_type == "transient_linear":
        for i in range(len(transient_output_files)):
            var filename = transient_output_files[i]
            var file_path = out_dir.joinpath(filename)
            file_path.write_text(PythonObject(transient_output_buffers[i]))
    elif analysis_type == "static_nonlinear" and len(static_output_files) > 0:
        for i in range(len(static_output_files)):
            var filename = static_output_files[i]
            var file_path = out_dir.joinpath(filename)
            file_path.write_text(PythonObject(static_output_buffers[i]))
    else:
        for r in range(py_len(recorders)):
            var rec = recorders[r]
            var rec_type = String(rec["type"])
            if rec_type == "node_displacement":
                var dofs = rec["dofs"]
                var output = String(rec.get("output", "node_disp"))
                var nodes_out = rec["nodes"]
                for nidx in range(py_len(nodes_out)):
                    var node_id = Int(nodes_out[nidx])
                    var i = id_to_index[node_id]
                    var line = String()
                    for j in range(py_len(dofs)):
                        var dof = Int(dofs[j])
                        require_dof_in_range(dof, ndf, "recorder")
                        var value = u[node_dof_index(i, dof, ndf)]
                        if j > 0:
                            line += " "
                        line += String(value)
                    line += "\n"
                    var filename = output + "_node" + String(node_id) + ".out"
                    var file_path = out_dir.joinpath(filename)
                    file_path.write_text(PythonObject(line))
            elif rec_type == "element_force":
                var output = String(rec.get("output", "element_force"))
                var elements_out = rec["elements"]
                for eidx in range(py_len(elements_out)):
                    var elem_id = Int(elements_out[eidx])
                    if elem_id >= len(elem_id_to_index) or elem_id_to_index[elem_id] < 0:
                        abort("recorder element not found")
                    var elem_index = elem_id_to_index[elem_id]
                    var elem = elements[elem_index]
                    var elem_type = String(elem["type"])
                    var f_elem: List[Float64] = []
                    if elem_type == "elasticBeamColumn2d":
                        f_elem = _beam2d_element_force_global(
                            elem,
                            nodes,
                            sections_by_id,
                            id_to_index,
                            ndf,
                            u,
                        )
                    elif elem_type == "forceBeamColumn2d":
                        f_elem = _force_beam_column2d_element_force_global(
                            elem_index,
                            elem,
                            nodes,
                            id_to_index,
                            ndf,
                            u,
                            fiber_section_defs,
                            fiber_section_cells,
                            fiber_section_index_by_id,
                            uniaxial_defs,
                            uniaxial_states,
                            elem_uniaxial_offsets,
                            elem_uniaxial_counts,
                            elem_uniaxial_state_ids,
                        )
                    elif elem_type == "truss":
                        f_elem = _truss_element_force_global(
                            elem_index,
                            elem,
                            nodes,
                            id_to_index,
                            ndf,
                            uniaxial_states,
                            elem_uniaxial_offsets,
                            elem_uniaxial_counts,
                            elem_uniaxial_state_ids,
                        )
                    else:
                        abort(
                            "element_force recorder supports truss, "
                            "elasticBeamColumn2d, or forceBeamColumn2d only"
                        )
                    var line = String()
                    for j in range(len(f_elem)):
                        if j > 0:
                            line += " "
                        line += String(f_elem[j])
                    line += "\n"
                    var filename = output + "_ele" + String(elem_id) + ".out"
                    var file_path = out_dir.joinpath(filename)
                    file_path.write_text(PythonObject(line))
            else:
                abort("unsupported recorder type")

    var t2 = Int(time.perf_counter_ns())
    if do_profile:
        var total_us = (t2 - t0) // 1000
        _append_event(events, events_need_comma, "C", frame_output, total_us)
        _append_event(events, events_need_comma, "C", frame_total, total_us)
        _write_speedscope(profile_path, frames, events, total_us)
