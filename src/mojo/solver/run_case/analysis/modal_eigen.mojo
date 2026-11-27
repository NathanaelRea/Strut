from solver.assembly.shell_mass import add_shell_mass_reduced
from collections import List
from math import sqrt
from os import abort

from materials import UniMaterialDef, UniMaterialState
from solver.assembly.surface_cache import SurfaceAssemblyCache
from solver.assembly import assemble_global_stiffness_typed
from solver.dof import node_dof_index, require_dof_in_range
from solver.eigen import jacobi_symmetric_eigen
from solver.run_case.input_types import (
    ElementInput,
    MaterialInput,
    NodeInput,
    RecorderInput,
    SectionInput,
)
from solver.run_case.load_state import ActiveElementLoadState
from sections import FiberCell, FiberSection2dDef, FiberSection3dDef, LayeredShellSectionDef

from solver.run_case.helpers import (
    _append_output,
    _build_reduced_diagonal_matrix_by_mpc,
    _collapse_matrix_by_mpc,
    _enforce_mpc_values,
    _format_values_line,
)
from tag_types import RecorderTypeTag


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


fn _cholesky_factor_dense_spd(matrix: List[List[Float64]]) -> List[List[Float64]]:
    var n = len(matrix)
    var factor: List[List[Float64]] = []
    for _ in range(n):
        var row: List[Float64] = []
        row.resize(n, 0.0)
        factor.append(row^)

    var eps = 1.0e-18
    for i in range(n):
        for j in range(i + 1):
            var value = matrix[i][j]
            for k in range(j):
                value -= factor[i][k] * factor[j][k]
            if i == j:
                if value <= eps:
                    abort("modal_eigen matrix is not positive definite")
                factor[i][j] = sqrt(value)
            else:
                factor[i][j] = value / factor[j][j]
    return factor^


fn _solve_lower_triangular(lower: List[List[Float64]], rhs: List[Float64]) -> List[Float64]:
    var n = len(lower)
    var solution: List[Float64] = []
    solution.resize(n, 0.0)
    for i in range(n):
        var value = rhs[i]
        for j in range(i):
            value -= lower[i][j] * solution[j]
        solution[i] = value / lower[i][i]
    return solution^


fn _solve_upper_from_lower_transpose(
    lower: List[List[Float64]], rhs: List[Float64]
) -> List[Float64]:
    var n = len(lower)
    var solution: List[Float64] = []
    solution.resize(n, 0.0)
    for i in range(n - 1, -1, -1):
        var value = rhs[i]
        for j in range(i + 1, n):
            value -= lower[j][i] * solution[j]
        solution[i] = value / lower[i][i]
    return solution^


fn _multiply_dense_matrices(
    left: List[List[Float64]], right: List[List[Float64]]
) -> List[List[Float64]]:
    var rows = len(left)
    var inner = len(right)
    var cols = 0
    if inner > 0:
        cols = len(right[0])
    var out: List[List[Float64]] = []
    for _ in range(rows):
        var row: List[Float64] = []
        row.resize(cols, 0.0)
        out.append(row^)

    for i in range(rows):
        for k in range(inner):
            var left_ik = left[i][k]
            if left_ik == 0.0:
                continue
            for j in range(cols):
                out[i][j] += left_ik * right[k][j]
    return out^


fn _multiply_dense_by_transpose(
    left: List[List[Float64]], right: List[List[Float64]]
) -> List[List[Float64]]:
    var rows = len(left)
    var cols = len(right)
    var inner = 0
    if len(right) > 0:
        inner = len(right[0])
    var out: List[List[Float64]] = []
    for _ in range(rows):
        var row: List[Float64] = []
        row.resize(cols, 0.0)
        out.append(row^)

    for i in range(rows):
        for j in range(cols):
            var value = 0.0
            for k in range(inner):
                value += left[i][k] * right[j][k]
            out[i][j] = value
    return out^


fn run_modal_eigen(
    modal_num_modes: Int,
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
    mut fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    mut fiber_section3d_defs: List[FiberSection3dDef],
    fiber_section3d_cells: List[FiberCell],
    fiber_section3d_index_by_id: List[Int],
    layered_shell_section_defs: List[LayeredShellSectionDef],
    mut shell_cache: SurfaceAssemblyCache,
    active_element_loads: ActiveElementLoadState,
    M_total: List[Float64],
    constrained: List[Bool],
    free: List[Int],
    has_transformation_mpc: Bool,
    mpc_slave_dof: List[Bool],
    mpc_row_offsets: List[Int],
    mpc_dof_pool: List[Int],
    mpc_coeff_pool: List[Float64],
    recorders: List[RecorderInput],
    recorder_nodes_pool: List[Int],
    recorder_dofs_pool: List[Int],
    recorder_modes_pool: List[Int],
    mut static_output_files: List[String],
    mut static_output_buffers: List[String],
    finite_modes: Bool = False,
    standard: Bool = False,
) raises:
    var K = assemble_global_stiffness_typed(
        typed_nodes,
        typed_elements,
        active_element_loads.element_loads,
        active_element_loads.elem_load_offsets,
        active_element_loads.elem_load_pool,
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
        layered_shell_section_defs,
        shell_cache,
    )
    if has_transformation_mpc:
        K = _collapse_matrix_by_mpc(K, mpc_row_offsets, mpc_dof_pool, mpc_coeff_pool)

    var free_count = len(free)
    if modal_num_modes > free_count:
        abort("modal_eigen num_modes exceeds free dof count")

    var K_ff: List[List[Float64]] = []
    for _ in range(free_count):
        var row: List[Float64] = []
        row.resize(free_count, 0.0)
        K_ff.append(row^)
    for i in range(free_count):
        for j in range(free_count):
            K_ff[i][j] = K[free[i]][free[j]]
    var eigvals_raw: List[Float64] = []
    var eigvecs: List[List[Float64]] = []
    var factor: List[List[Float64]] = []
    var order: List[Int]
    var reciprocal_eigenvalues = not standard and (ndm == 3 or finite_modes)
    if standard:
        var scale = 0.0
        for i in range(free_count):
            for j in range(free_count):
                scale = max(scale, abs(K_ff[i][j]))
        jacobi_symmetric_eigen(
            K_ff, eigvals_raw, eigvecs, 8 * free_count * free_count, scale * 1.0e-15
        )
        order = _sorted_indices_ascending(eigvals_raw)
    else:
        var M_f: List[Float64] = []
        M_f.resize(free_count, 0.0)
        var free_index: List[Int] = []
        free_index.resize(total_dofs, -1)
        var max_mass = 0.0
        for i in range(free_count):
            free_index[free[i]] = i
        var M_ff = _build_reduced_diagonal_matrix_by_mpc(
            M_total,
            free_index,
            mpc_row_offsets,
            mpc_dof_pool,
            mpc_coeff_pool,
        )
        shell_cache.continuum.add_modal_mass(
            free_index,
            mpc_row_offsets,
            mpc_dof_pool,
            mpc_coeff_pool,
            M_ff,
        )
        add_shell_mass_reduced(
            shell_cache.shell.masses,
            free_index,
            mpc_row_offsets,
            mpc_dof_pool,
            mpc_coeff_pool,
            M_ff,
        )
        var has_mass = False
        for i in range(free_count):
            M_f[i] = M_ff[i][i]
            if M_f[i] > max_mass:
                max_mass = M_f[i]
            for j in range(free_count):
                if M_ff[i][j] != 0.0:
                    has_mass = True
        if not has_mass:
            abort("modal_eigen requires masses on free dofs")
        var matrix: List[List[Float64]]
        if reciprocal_eigenvalues:
            # Factor K instead of M so massless DOFs need no artificial mass.
            # The largest eigenvalues of L_K^-1 M L_K^-T are the reciprocals
            # of the smallest finite eigenvalues of K v = lambda M v.
            factor = _cholesky_factor_dense_spd(K_ff)
            matrix = M_ff^
        else:
            var msmall = 1.0e-10
            var mass_floor = max_mass * 1.0e-12
            if mass_floor <= 0.0:
                mass_floor = 1.0e-12
            for i in range(free_count):
                if M_f[i] == 0.0:
                    M_f[i] = K_ff[i][i] * msmall
                if M_f[i] <= 0.0:
                    M_f[i] = mass_floor
                M_ff[i][i] = M_f[i]
            factor = _cholesky_factor_dense_spd(M_ff)
            matrix = K_ff^
        var inverse_factor: List[List[Float64]] = []
        for _ in range(free_count):
            var row: List[Float64] = []
            row.resize(free_count, 0.0)
            inverse_factor.append(row^)
        for col in range(free_count):
            var basis: List[Float64] = []
            basis.resize(free_count, 0.0)
            basis[col] = 1.0
            var inverse_col = _solve_lower_triangular(factor, basis)
            for row in range(free_count):
                inverse_factor[row][col] = inverse_col[row]

        var A = _multiply_dense_matrices(inverse_factor, matrix)
        A = _multiply_dense_by_transpose(A, inverse_factor)

        if reciprocal_eigenvalues:
            var scale = 0.0
            for i in range(free_count):
                scale = max(scale, abs(A[i][i]))
            jacobi_symmetric_eigen(
                A, eigvals_raw, eigvecs, 8 * free_count * free_count, scale * 1.0e-13
            )
        else:
            jacobi_symmetric_eigen(A, eigvals_raw, eigvecs)
        order = _sorted_indices_ascending(eigvals_raw)
        if reciprocal_eigenvalues:
            for i in range(free_count // 2):
                var j = free_count - i - 1
                var col = order[i]
                order[i] = order[j]
                order[j] = col
            var threshold = eigvals_raw[order[0]] * 1.0e-12
            for i in range(modal_num_modes):
                var col = order[i]
                if eigvals_raw[col] <= threshold:
                    abort("modal_eigen num_modes exceeds the number of finite modes")
                eigvals_raw[col] = 1.0 / eigvals_raw[col]

    var wrote_eigenvalues = False
    for r in range(len(recorders)):
        var rec = recorders[r]
        if rec.type_tag != RecorderTypeTag.ModalEigen:
            continue
        var output = rec.output
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

        if rec.mode_count == 0:
            abort("modal_eigen recorder requires non-empty modes")

        for m in range(rec.mode_count):
            var mode_no = recorder_modes_pool[rec.mode_offset + m]
            if mode_no < 1 or mode_no > modal_num_modes:
                abort("modal_eigen recorder mode out of range")
            var col = order[mode_no - 1]
            var reduced_mode: List[Float64] = []
            reduced_mode.resize(free_count, 0.0)
            for i in range(free_count):
                reduced_mode[i] = eigvecs[i][col]
            if not standard:
                reduced_mode = _solve_upper_from_lower_transpose(factor, reduced_mode)
            if reciprocal_eigenvalues:
                var scale = sqrt(eigvals_raw[col])
                for i in range(free_count):
                    reduced_mode[i] *= scale
            var full_mode: List[Float64] = []
            full_mode.resize(total_dofs, 0.0)
            for i in range(free_count):
                full_mode[free[i]] = reduced_mode[i]
            _enforce_mpc_values(
                full_mode,
                constrained,
                mpc_slave_dof,
                mpc_row_offsets,
                mpc_dof_pool,
                mpc_coeff_pool,
            )
            for nidx in range(rec.node_count):
                var node_id = recorder_nodes_pool[rec.node_offset + nidx]
                if node_id >= len(id_to_index) or id_to_index[node_id] < 0:
                    abort("modal_eigen recorder node not found")
                var node_idx = id_to_index[node_id]
                var values: List[Float64] = []
                values.resize(rec.dof_count, 0.0)
                for j in range(rec.dof_count):
                    var dof = recorder_dofs_pool[rec.dof_offset + j]
                    require_dof_in_range(dof, ndf, "modal_eigen recorder")
                    values[j] = full_mode[node_dof_index(node_idx, dof, ndf)]
                var filename = (
                    output + "_mode" + String(mode_no) + "_node" + String(node_id) + ".out"
                )
                _append_output(
                    static_output_files,
                    static_output_buffers,
                    filename,
                    _format_values_line(values),
                )
