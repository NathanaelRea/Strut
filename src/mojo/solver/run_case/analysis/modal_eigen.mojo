from collections import List
from math import sqrt
from os import abort
from python import PythonObject

from materials import UniMaterialDef, UniMaterialState
from solver.assembly import assemble_global_stiffness
from solver.dof import node_dof_index, require_dof_in_range
from solver.eigen import jacobi_symmetric_eigen
from sections import FiberCell, FiberSection2dDef
from strut_io import py_len

from solver.run_case.helpers import (
    _append_output,
    _collapse_matrix_by_rep,
    _enforce_equal_dof_values,
    _format_values_line,
)


fn _sorted_indices_ascending(values: List[Float64]) -> List[Int]:
    var n = len(values)
    var idx: List[Int] = []
    idx.resize(n, 0)
    for i in range(n):
        idx[i] = i
    for i in range(n):
        var min_i = i
        var min_v = values[idx[i]]
        for j in range(i + 1, n):
            var v = values[idx[j]]
            if v < min_v:
                min_v = v
                min_i = j
        if min_i != i:
            var tmp = idx[i]
            idx[i] = idx[min_i]
            idx[min_i] = tmp
    return idx^


fn run_modal_eigen(
    analysis: PythonObject,
    modal_num_modes: Int,
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
    fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    M_total: List[Float64],
    constrained: List[Bool],
    free: List[Int],
    has_transformation_mpc: Bool,
    rep_dof: List[Int],
    recorders: PythonObject,
    mut static_output_files: List[String],
    mut static_output_buffers: List[String],
) raises:
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
    if has_transformation_mpc:
        K = _collapse_matrix_by_rep(K, rep_dof)

    var free_count = len(free)
    if modal_num_modes > free_count:
        abort("modal_eigen num_modes exceeds free dof count")

    var K_ff: List[List[Float64]] = []
    for _ in range(free_count):
        var row: List[Float64] = []
        row.resize(free_count, 0.0)
        K_ff.append(row^)
    var M_f: List[Float64] = []
    M_f.resize(free_count, 0.0)
    for i in range(free_count):
        M_f[i] = M_total[free[i]]
        if M_f[i] <= 0.0:
            abort("modal_eigen requires positive mass at free dofs")
        for j in range(free_count):
            K_ff[i][j] = K[free[i]][free[j]]

    var A: List[List[Float64]] = []
    for i in range(free_count):
        var row: List[Float64] = []
        row.resize(free_count, 0.0)
        var mi = M_f[i]
        for j in range(free_count):
            row[j] = K_ff[i][j] / sqrt(mi * M_f[j])
        A.append(row^)

    var eigvals_raw: List[Float64] = []
    var eigvecs: List[List[Float64]] = []
    jacobi_symmetric_eigen(A, eigvals_raw, eigvecs)
    var order = _sorted_indices_ascending(eigvals_raw)

    var wrote_eigenvalues = False
    for r in range(py_len(recorders)):
        var rec = recorders[r]
        var rec_type = String(rec["type"])
        if rec_type != "modal_eigen":
            continue
        var output = String(rec.get("output", "modal"))
        if not wrote_eigenvalues:
            var eig_lines = String()
            for i in range(modal_num_modes):
                var col = order[i]
                eig_lines += String(eigvals_raw[col]) + "\n"
            _append_output(
                static_output_files,
                static_output_buffers,
                output + "_eigenvalues.out",
                eig_lines,
            )
            wrote_eigenvalues = True

        var modes = rec.get("modes", [])
        if py_len(modes) == 0:
            abort("modal_eigen recorder requires non-empty modes")

        var nodes_out = rec["nodes"]
        var dofs = rec["dofs"]
        for m in range(py_len(modes)):
            var mode_no = Int(modes[m])
            if mode_no < 1 or mode_no > modal_num_modes:
                abort("modal_eigen recorder mode out of range")
            var col = order[mode_no - 1]
            var full_mode: List[Float64] = []
            full_mode.resize(total_dofs, 0.0)
            for i in range(free_count):
                full_mode[free[i]] = eigvecs[i][col] / sqrt(M_f[i])
            _enforce_equal_dof_values(full_mode, rep_dof, constrained)
            for nidx in range(py_len(nodes_out)):
                var node_id = Int(nodes_out[nidx])
                if node_id >= len(id_to_index) or id_to_index[node_id] < 0:
                    abort("modal_eigen recorder node not found")
                var node_idx = id_to_index[node_id]
                var values: List[Float64] = []
                values.resize(py_len(dofs), 0.0)
                for j in range(py_len(dofs)):
                    var dof = Int(dofs[j])
                    require_dof_in_range(dof, ndf, "modal_eigen recorder")
                    values[j] = full_mode[node_dof_index(node_idx, dof, ndf)]
                var filename = (
                    output
                    + "_mode"
                    + String(mode_no)
                    + "_node"
                    + String(node_id)
                    + ".out"
                )
                _append_output(
                    static_output_files,
                    static_output_buffers,
                    filename,
                    _format_values_line(values),
                )
