from collections import List
from os import abort
from python import Python, PythonObject

from elements import (
    beam2d_corotational_global_internal_force,
    beam2d_pdelta_global_stiffness,
    beam_global_stiffness,
    beam_uniform_load_global,
)
from linalg import gaussian_elimination
from solver.assembly import (
    assemble_global_stiffness,
    assemble_global_stiffness_and_internal,
)
from solver.dof import node_dof_index, require_dof_in_range
from solver.profile import (
    _append_event,
    _append_frame,
    _profile_enabled,
    _write_speedscope,
)
from strut_io import py_len


fn _beam2d_element_force_global(
    elem: PythonObject,
    nodes: PythonObject,
    sections_by_id: List[PythonObject],
    id_to_index: List[Int],
    ndf: Int,
    u: List[Float64],
) raises -> List[Float64]:
    if ndf != 3:
        abort("elasticBeamColumn2d requires ndf=3")
    var n1 = Int(elem["nodes"][0])
    var n2 = Int(elem["nodes"][1])
    var i1 = id_to_index[n1]
    var i2 = id_to_index[n2]
    var node1 = nodes[i1]
    var node2 = nodes[i2]

    var sec_id = Int(elem["section"])
    if sec_id >= len(sections_by_id):
        abort("section not found")
    var sec = sections_by_id[sec_id]
    if sec is None:
        abort("section not found")

    var params = sec["params"]
    var E = Float64(params["E"])
    var A = Float64(params["A"])
    var I = Float64(params["I"])

    var dof_map = [
        node_dof_index(i1, 1, ndf),
        node_dof_index(i1, 2, ndf),
        node_dof_index(i1, 3, ndf),
        node_dof_index(i2, 1, ndf),
        node_dof_index(i2, 2, ndf),
        node_dof_index(i2, 3, ndf),
    ]
    var u_elem: List[Float64] = []
    u_elem.resize(6, 0.0)
    for i in range(6):
        u_elem[i] = u[dof_map[i]]

    var geom = String(elem.get("geomTransf", "Linear"))
    if geom == "Corotational":
        return beam2d_corotational_global_internal_force(
            E,
            A,
            I,
            Float64(node1["x"]),
            Float64(node1["y"]),
            Float64(node2["x"]),
            Float64(node2["y"]),
            u_elem,
        )

    var k_global: List[List[Float64]] = []
    if geom == "Linear":
        k_global = beam_global_stiffness(
            E,
            A,
            I,
            Float64(node1["x"]),
            Float64(node1["y"]),
            Float64(node2["x"]),
            Float64(node2["y"]),
        )
    elif geom == "PDelta":
        k_global = beam2d_pdelta_global_stiffness(
            E,
            A,
            I,
            Float64(node1["x"]),
            Float64(node1["y"]),
            Float64(node2["x"]),
            Float64(node2["y"]),
            u_elem,
        )
    else:
        abort("unsupported geomTransf: " + geom)

    var f_elem: List[Float64] = []
    f_elem.resize(6, 0.0)
    for i in range(6):
        var sum = 0.0
        for j in range(6):
            sum += k_global[i][j] * u_elem[j]
        f_elem[i] = sum
    return f_elem^


def run_case(data: PythonObject, output_path: String, profile_path: String):
    var model = data["model"]
    var ndm = Int(model["ndm"])
    var ndf = Int(model["ndf"])
    var is_2d = ndm == 2 and (ndf == 2 or ndf == 3)
    var is_3d_truss = ndm == 3 and ndf == 3
    var is_3d_shell = ndm == 3 and ndf == 6
    if not is_2d and not is_3d_truss and not is_3d_shell:
        abort("only ndm=2 ndf=2/3 and ndm=3 ndf=3/6 supported")

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

    var nodes = data["nodes"]
    var node_count = py_len(nodes)
    var node_ids: List[Int] = []
    node_ids.resize(node_count, 0)

    for i in range(node_count):
        var node = nodes[i]
        if ndm == 3 and not node.__contains__("z"):
            abort("ndm=3 requires node z coordinate")
        node_ids[i] = Int(node["id"])

    var id_to_index: List[Int] = []
    id_to_index.resize(10000, -1)
    for i in range(node_count):
        var nid = node_ids[i]
        if nid >= len(id_to_index):
            id_to_index.resize(nid + 1, -1)
        id_to_index[nid] = i

    var sections = data.get("sections", [])
    var materials = data.get("materials", [])
    var sections_by_id: List[PythonObject] = []
    sections_by_id.resize(0, None)
    for i in range(py_len(sections)):
        var sec = sections[i]
        var sid = Int(sec["id"])
        if sid >= len(sections_by_id):
            sections_by_id.resize(sid + 1, None)
        sections_by_id[sid] = sec
    var materials_by_id: List[PythonObject] = []
    materials_by_id.resize(0, None)
    for i in range(py_len(materials)):
        var mat = materials[i]
        var mid = Int(mat["id"])
        if mid >= len(materials_by_id):
            materials_by_id.resize(mid + 1, None)
        materials_by_id[mid] = mat
    var elements = data["elements"]
    var elem_count = py_len(elements)
    var elem_ids: List[Int] = []
    elem_ids.resize(elem_count, 0)
    var elem_id_to_index: List[Int] = []
    elem_id_to_index.resize(10000, -1)
    for i in range(elem_count):
        var elem = elements[i]
        var eid = Int(elem["id"])
        elem_ids[i] = eid
        if eid >= len(elem_id_to_index):
            elem_id_to_index.resize(eid + 1, -1)
        elem_id_to_index[eid] = i

    var total_dofs = node_count * ndf
    var F_total: List[Float64] = []
    F_total.resize(total_dofs, 0.0)

    var element_loads = data.get("element_loads", [])
    for i in range(py_len(element_loads)):
        var load = element_loads[i]
        var elem_id = Int(load["element"])
        if elem_id >= len(elem_id_to_index) or elem_id_to_index[elem_id] < 0:
            abort("element load refers to unknown element")
        var elem = elements[elem_id_to_index[elem_id]]
        var elem_type = String(elem["type"])
        var load_type = String(load["type"])
        if load_type == "beamUniform":
            if elem_type != "elasticBeamColumn2d":
                abort("beamUniform requires elasticBeamColumn2d")
            if ndf != 3:
                abort("beamUniform requires ndf=3")
            var n1 = Int(elem["nodes"][0])
            var n2 = Int(elem["nodes"][1])
            var i1 = id_to_index[n1]
            var i2 = id_to_index[n2]
            var node1 = nodes[i1]
            var node2 = nodes[i2]
            var w = Float64(load["w"])
            var f_global = beam_uniform_load_global(
                Float64(node1["x"]),
                Float64(node1["y"]),
                Float64(node2["x"]),
                Float64(node2["y"]),
                w,
            )
            var dof_map = [
                node_dof_index(i1, 1, ndf),
                node_dof_index(i1, 2, ndf),
                node_dof_index(i1, 3, ndf),
                node_dof_index(i2, 1, ndf),
                node_dof_index(i2, 2, ndf),
                node_dof_index(i2, 3, ndf),
            ]
            for a in range(6):
                F_total[dof_map[a]] += f_global[a]
        else:
            abort("unsupported element load type")

    var loads = data.get("loads", [])
    for i in range(py_len(loads)):
        var load = loads[i]
        var node_id = Int(load["node"])
        var dof = Int(load["dof"])
        require_dof_in_range(dof, ndf, "load")
        var idx = node_dof_index(id_to_index[node_id], dof, ndf)
        F_total[idx] += Float64(load["value"])

    var constrained: List[Bool] = []
    constrained.resize(total_dofs, False)
    for i in range(node_count):
        var node = nodes[i]
        if not node.__contains__("constraints"):
            continue
        var constraints = node["constraints"]
        for j in range(py_len(constraints)):
            var dof = Int(constraints[j])
            require_dof_in_range(dof, ndf, "constraint")
            var idx = node_dof_index(i, dof, ndf)
            constrained[idx] = True

    var free: List[Int] = []
    for i in range(total_dofs):
        if not constrained[i]:
            free.append(i)

    if len(free) == 0:
        abort("no free dofs")

    var analysis = data.get("analysis", {"type": "static_linear", "steps": 1})
    var analysis_type = String(analysis.get("type", "static_linear"))
    var steps = Int(analysis.get("steps", 1))
    if steps < 1:
        abort("analysis steps must be >= 1")

    var u: List[Float64] = []
    u.resize(total_dofs, 0.0)

    var t_solve_start = Int(time.perf_counter_ns())
    if do_profile:
        var assemble_end = (t_solve_start - t0) // 1000
        _append_event(events, events_need_comma, "C", frame_assemble, assemble_end)
        _append_event(events, events_need_comma, "O", frame_solve, assemble_end)
    if analysis_type == "static_linear":
        if do_profile:
            var t_asm_start = Int(time.perf_counter_ns())
            var asm_start_us = (t_asm_start - t0) // 1000
            _append_event(
                events, events_need_comma, "O", frame_assemble_stiffness, asm_start_us
            )
        var K = assemble_global_stiffness(
            nodes,
            elements,
            sections_by_id,
            materials_by_id,
            id_to_index,
            node_count,
            ndf,
            ndm,
            u,
        )
        if do_profile:
            var t_asm_end = Int(time.perf_counter_ns())
            var asm_end_us = (t_asm_end - t0) // 1000
            _append_event(
                events, events_need_comma, "C", frame_assemble_stiffness, asm_end_us
            )
        var K_ff: List[List[Float64]] = []
        var F_f: List[Float64] = []
        F_f.resize(len(free), 0.0)
        if do_profile:
            var t_kff_start = Int(time.perf_counter_ns())
            var kff_start_us = (t_kff_start - t0) // 1000
            _append_event(
                events, events_need_comma, "O", frame_kff_extract, kff_start_us
            )
        for i in range(len(free)):
            var row: List[Float64] = []
            row.resize(len(free), 0.0)
            K_ff.append(row^)
            F_f[i] = F_total[free[i]]
        for i in range(len(free)):
            for j in range(len(free)):
                K_ff[i][j] = K[free[i]][free[j]]
        if do_profile:
            var t_kff_end = Int(time.perf_counter_ns())
            var kff_end_us = (t_kff_end - t0) // 1000
            _append_event(events, events_need_comma, "C", frame_kff_extract, kff_end_us)
        if do_profile:
            var t_solve_lin_start = Int(time.perf_counter_ns())
            var solve_lin_start_us = (t_solve_lin_start - t0) // 1000
            _append_event(
                events, events_need_comma, "O", frame_solve_linear, solve_lin_start_us
            )
        var u_f = gaussian_elimination(K_ff, F_f)
        if do_profile:
            var t_solve_lin_end = Int(time.perf_counter_ns())
            var solve_lin_end_us = (t_solve_lin_end - t0) // 1000
            _append_event(
                events, events_need_comma, "C", frame_solve_linear, solve_lin_end_us
            )
        for i in range(len(free)):
            u[free[i]] = u_f[i]
    elif analysis_type == "static_nonlinear":
        var max_iters = Int(analysis.get("max_iters", 20))
        var tol = Float64(analysis.get("tol", 1.0e-10))
        var rel_tol = Float64(analysis.get("rel_tol", 1.0e-8))
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
        var F_f: List[Float64] = []
        F_f.resize(free_count, 0.0)
        for step in range(steps):
            if do_profile:
                var t_step_start = Int(time.perf_counter_ns())
                var step_start_us = (t_step_start - t0) // 1000
                _append_event(
                    events, events_need_comma, "O", frame_nonlinear_step, step_start_us
                )
            var scale = Float64(step + 1) / Float64(steps)
            var converged = False
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
                    nodes,
                    elements,
                    sections_by_id,
                    materials_by_id,
                    id_to_index,
                    node_count,
                    ndf,
                    ndm,
                    u,
                    K,
                    F_int,
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
                for i in range(free_count):
                    F_f[i] = F_total_free[i] * scale - F_int[free[i]]
                for i in range(free_count):
                    for j in range(free_count):
                        K_ff[i][j] = K[free[i]][free[j]]
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
                var u_f = gaussian_elimination(K_ff, F_f)
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
                    var diff = du
                    if diff < 0.0:
                        diff = -diff
                    if diff > max_diff:
                        max_diff = diff
                    var abs_val = value
                    if abs_val < 0.0:
                        abs_val = -abs_val
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
    var recorders = data.get("recorders", [])
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
                var elem = elements[elem_id_to_index[elem_id]]
                if String(elem["type"]) != "elasticBeamColumn2d":
                    abort("element_force recorder supports elasticBeamColumn2d only")
                var f_elem = _beam2d_element_force_global(
                    elem,
                    nodes,
                    sections_by_id,
                    id_to_index,
                    ndf,
                    u,
                )
                var line = String()
                for j in range(6):
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
