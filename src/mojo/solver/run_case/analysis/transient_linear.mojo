from collections import List
from materials import UniMaterialDef, UniMaterialState
from os import abort
from python import PythonObject
from sections import FiberCell, FiberSection2dDef

from linalg import gaussian_elimination
from solver.assembly import assemble_global_stiffness
from solver.dof import node_dof_index, require_dof_in_range
from solver.time_series import eval_time_series
from strut_io import py_len

from solver.run_case.helpers import (
    _append_output,
    _beam2d_element_force_global,
    _force_beam_column2d_element_force_global,
    _truss_element_force_global,
)

fn run_transient_linear(
    analysis: PythonObject,
    steps: Int,
    ts_index: Int,
    time_series: PythonObject,
    nodes: PythonObject,
    elements: PythonObject,
    sections_by_id: List[PythonObject],
    materials_by_id: List[PythonObject],
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
    recorders: PythonObject,
    elem_id_to_index: List[Int],
    fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    mut transient_output_files: List[String],
    mut transient_output_buffers: List[String],
) raises:
    var dt = Float64(analysis.get("dt", 0.0))
    if dt <= 0.0:
        abort("transient_linear requires dt > 0")
    var integrator = analysis.get("integrator", {"type": "Newmark"})
    var integrator_type = String(integrator.get("type", "Newmark"))
    if integrator_type != "Newmark":
        abort("transient_linear only supports Newmark integrator")
    var gamma = Float64(integrator.get("gamma", 0.5))
    var beta = Float64(integrator.get("beta", 0.25))
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
    var K_ff: List[List[Float64]] = []
    for _ in range(free_count):
        var row: List[Float64] = []
        row.resize(free_count, 0.0)
        K_ff.append(row^)
    for i in range(free_count):
        for j in range(free_count):
            K_ff[i][j] = K[free[i]][free[j]]

    var a0 = 1.0 / (beta * dt * dt)
    var a2 = 1.0 / (beta * dt)
    var a3 = 1.0 / (2.0 * beta) - 1.0
    var K_eff: List[List[Float64]] = []
    for _ in range(free_count):
        var row_eff: List[Float64] = []
        row_eff.resize(free_count, 0.0)
        K_eff.append(row_eff^)
    for i in range(free_count):
        for j in range(free_count):
            K_eff[i][j] = K_ff[i][j]
        K_eff[i][i] += a0 * M_f[i]

    var v: List[Float64] = []
    v.resize(total_dofs, 0.0)
    var a: List[Float64] = []
    a.resize(total_dofs, 0.0)

    var P_eff: List[Float64] = []
    P_eff.resize(free_count, 0.0)

    for step in range(steps):
        var t = Float64(step + 1) * dt
        var factor = 1.0
        if ts_index >= 0:
            factor = eval_time_series(time_series[ts_index], t)
        for i in range(free_count):
            var idx = free[i]
            P_eff[i] = (
                F_total[idx] * factor
                + M_f[i] * (a0 * u[idx] + a2 * v[idx] + a3 * a[idx])
            )
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
                    _append_output(
                        transient_output_files, transient_output_buffers, filename, line
                    )
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
                    _append_output(
                        transient_output_files, transient_output_buffers, filename, line
                    )
            else:
                abort("unsupported recorder type")
