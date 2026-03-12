from collections import List
from math import sqrt
from os import abort

from elements import link_orientation_matrix
from materials import UniMaterialDef, UniMaterialState, uniaxial_set_trial_strain
from solver.assembly.stiffness_internal_shared import (
    _elem_dir,
    _elem_dof_map,
    _gather_element_u,
)
from solver.run_case.linear_solver_backend import (
    LinearSolverBackend,
    add_element_matrix_from_pool,
)
from solver.run_case.input_types import ElementInput, NodeInput


fn _zero_length_row(
    dir: Int, ndm: Int, ndf: Int, trans: List[List[Float64]]
) -> List[Float64]:
    var row: List[Float64] = []
    row.resize(2 * ndf, 0.0)
    if ndm == 2 and ndf == 2:
        var axis = dir - 1
        row[0] = -trans[axis][0]
        row[1] = -trans[axis][1]
        row[2] = trans[axis][0]
        row[3] = trans[axis][1]
        return row^
    if ndm == 2 and ndf == 3:
        if dir == 1 or dir == 2:
            var axis = dir - 1
            row[0] = -trans[axis][0]
            row[1] = -trans[axis][1]
            row[3] = trans[axis][0]
            row[4] = trans[axis][1]
        else:
            row[2] = -trans[2][2]
            row[5] = trans[2][2]
        return row^
    if ndm == 3 and ndf == 3:
        var axis = dir - 1
        row[0] = -trans[axis][0]
        row[1] = -trans[axis][1]
        row[2] = -trans[axis][2]
        row[3] = trans[axis][0]
        row[4] = trans[axis][1]
        row[5] = trans[axis][2]
        return row^
    if ndm == 3 and ndf == 6:
        if dir <= 3:
            var axis = dir - 1
            row[0] = -trans[axis][0]
            row[1] = -trans[axis][1]
            row[2] = -trans[axis][2]
            row[6] = trans[axis][0]
            row[7] = trans[axis][1]
            row[8] = trans[axis][2]
        else:
            var axis = dir - 4
            row[3] = -trans[axis][0]
            row[4] = -trans[axis][1]
            row[5] = -trans[axis][2]
            row[9] = trans[axis][0]
            row[10] = trans[axis][1]
            row[11] = trans[axis][2]
        return row^
    abort("unsupported zeroLength ndm/ndf combination")
    return row^


fn _two_node_link_tgl(
    ndm: Int, ndf: Int, trans: List[List[Float64]]
) -> List[List[Float64]]:
    var num_dof = 2 * ndf
    var tgl: List[List[Float64]] = []
    for _ in range(num_dof):
        var row: List[Float64] = []
        row.resize(num_dof, 0.0)
        tgl.append(row^)
    if ndm == 2 and ndf == 2:
        tgl[0][0] = trans[0][0]
        tgl[0][1] = trans[0][1]
        tgl[1][0] = trans[1][0]
        tgl[1][1] = trans[1][1]
        tgl[2][2] = trans[0][0]
        tgl[2][3] = trans[0][1]
        tgl[3][2] = trans[1][0]
        tgl[3][3] = trans[1][1]
        return tgl^
    if ndm == 2 and ndf == 3:
        tgl[0][0] = trans[0][0]
        tgl[0][1] = trans[0][1]
        tgl[1][0] = trans[1][0]
        tgl[1][1] = trans[1][1]
        tgl[2][2] = trans[2][2]
        tgl[3][3] = trans[0][0]
        tgl[3][4] = trans[0][1]
        tgl[4][3] = trans[1][0]
        tgl[4][4] = trans[1][1]
        tgl[5][5] = trans[2][2]
        return tgl^
    if ndm == 3 and ndf == 3:
        for i in range(3):
            for j in range(3):
                tgl[i][j] = trans[i][j]
                tgl[i + 3][j + 3] = trans[i][j]
        return tgl^
    if ndm == 3 and ndf == 6:
        for i in range(3):
            for j in range(3):
                tgl[i][j] = trans[i][j]
                tgl[i + 3][j + 3] = trans[i][j]
                tgl[i + 6][j + 6] = trans[i][j]
                tgl[i + 9][j + 9] = trans[i][j]
        return tgl^
    abort("unsupported twoNodeLink ndm/ndf combination")
    return tgl^


fn _two_node_link_tlb(
    ndm: Int, ndf: Int, elem: ElementInput, length: Float64
) -> List[List[Float64]]:
    var tlb: List[List[Float64]] = []
    for _ in range(elem.dir_count):
        var row: List[Float64] = []
        row.resize(elem.dof_count, 0.0)
        tlb.append(row^)
    var half = elem.dof_count // 2
    for i in range(elem.dir_count):
        var dir_id = _elem_dir(elem, i) - 1
        tlb[i][dir_id] = -1.0
        tlb[i][dir_id + half] = 1.0
        if ndm == 2 and ndf == 3 and dir_id == 1:
            tlb[i][2] = -elem.shear_dist_1 * length
            tlb[i][5] = -(1.0 - elem.shear_dist_1) * length
        elif ndm == 3 and ndf == 6 and dir_id == 1:
            tlb[i][5] = -elem.shear_dist_1 * length
            tlb[i][11] = -(1.0 - elem.shear_dist_1) * length
        elif ndm == 3 and ndf == 6 and dir_id == 2:
            tlb[i][4] = elem.shear_dist_2 * length
            tlb[i][10] = (1.0 - elem.shear_dist_2) * length
    return tlb^


fn _two_node_link_add_pdelta_forces(
    ndm: Int,
    ndf: Int,
    elem: ElementInput,
    length: Float64,
    ul: List[Float64],
    q_basic: List[Float64],
    mut p_local: List[Float64],
):
    if not elem.has_pdelta or length == 0.0:
        return
    var axial = 0.0
    var delta_y = 0.0
    var delta_z = 0.0
    var half = elem.dof_count // 2
    for i in range(elem.dir_count):
        var dir_id = _elem_dir(elem, i)
        if dir_id == 1:
            axial = q_basic[i]
        elif dir_id == 2:
            delta_y = ul[1 + half] - ul[1]
        elif dir_id == 3 and ndm == 3:
            delta_z = ul[2 + half] - ul[2]
    if axial == 0.0:
        return
    for i in range(elem.dir_count):
        var dir_id = _elem_dir(elem, i)
        if elem.dof_count == 4 and dir_id == 2:
            var vp = axial * delta_y / length
            vp *= 1.0 - elem.pdelta_3 - elem.pdelta_4
            p_local[1] -= vp
            p_local[3] += vp
        elif ndm == 2 and ndf == 3 and dir_id == 2:
            var vp = axial * delta_y / length
            vp *= 1.0 - elem.pdelta_3 - elem.pdelta_4
            p_local[1] -= vp
            p_local[4] += vp
        elif ndm == 2 and ndf == 3 and dir_id == 3:
            var mp = axial * delta_y
            p_local[2] += elem.pdelta_3 * mp
            p_local[5] += elem.pdelta_4 * mp
        elif ndm == 3 and ndf == 3 and dir_id == 2:
            var vp = axial * delta_y / length
            vp *= 1.0 - elem.pdelta_3 - elem.pdelta_4
            p_local[1] -= vp
            p_local[4] += vp
        elif ndm == 3 and ndf == 3 and dir_id == 3:
            var vp = axial * delta_z / length
            vp *= 1.0 - elem.pdelta_1 - elem.pdelta_2
            p_local[2] -= vp
            p_local[5] += vp
        elif elem.dof_count == 12 and dir_id == 2:
            var vp = axial * delta_y / length
            vp *= 1.0 - elem.pdelta_3 - elem.pdelta_4
            p_local[1] -= vp
            p_local[7] += vp
        elif elem.dof_count == 12 and dir_id == 3:
            var vp = axial * delta_z / length
            vp *= 1.0 - elem.pdelta_1 - elem.pdelta_2
            p_local[2] -= vp
            p_local[8] += vp
        elif elem.dof_count == 12 and dir_id == 5:
            var mp = axial * delta_z
            p_local[4] -= elem.pdelta_1 * mp
            p_local[10] -= elem.pdelta_2 * mp
        elif elem.dof_count == 12 and dir_id == 6:
            var mp = axial * delta_y
            p_local[5] += elem.pdelta_3 * mp
            p_local[11] += elem.pdelta_4 * mp


fn _two_node_link_add_pdelta_stiff(
    ndm: Int,
    ndf: Int,
    elem: ElementInput,
    length: Float64,
    q_basic: List[Float64],
    mut k_local: List[List[Float64]],
):
    if not elem.has_pdelta or length == 0.0:
        return
    var axial = 0.0
    for i in range(elem.dir_count):
        if _elem_dir(elem, i) == 1:
            axial = q_basic[i]
    if axial == 0.0:
        return
    for i in range(elem.dir_count):
        var dir_id = _elem_dir(elem, i)
        if elem.dof_count == 4 and dir_id == 2:
            var noverl = axial / length
            noverl *= 1.0 - elem.pdelta_3 - elem.pdelta_4
            k_local[1][1] += noverl
            k_local[1][3] -= noverl
            k_local[3][1] -= noverl
            k_local[3][3] += noverl
        elif ndm == 2 and ndf == 3 and dir_id == 2:
            var noverl = axial / length
            noverl *= 1.0 - elem.pdelta_3 - elem.pdelta_4
            k_local[1][1] += noverl
            k_local[1][4] -= noverl
            k_local[4][1] -= noverl
            k_local[4][4] += noverl
        elif ndm == 2 and ndf == 3 and dir_id == 3:
            k_local[2][1] -= elem.pdelta_3 * axial
            k_local[2][4] += elem.pdelta_3 * axial
            k_local[5][1] -= elem.pdelta_4 * axial
            k_local[5][4] += elem.pdelta_4 * axial
        elif ndm == 3 and ndf == 3 and dir_id == 2:
            var noverl = axial / length
            noverl *= 1.0 - elem.pdelta_3 - elem.pdelta_4
            k_local[1][1] += noverl
            k_local[1][4] -= noverl
            k_local[4][1] -= noverl
            k_local[4][4] += noverl
        elif ndm == 3 and ndf == 3 and dir_id == 3:
            var noverl = axial / length
            noverl *= 1.0 - elem.pdelta_1 - elem.pdelta_2
            k_local[2][2] += noverl
            k_local[2][5] -= noverl
            k_local[5][2] -= noverl
            k_local[5][5] += noverl
        elif elem.dof_count == 12 and dir_id == 2:
            var noverl = axial / length
            noverl *= 1.0 - elem.pdelta_3 - elem.pdelta_4
            k_local[1][1] += noverl
            k_local[1][7] -= noverl
            k_local[7][1] -= noverl
            k_local[7][7] += noverl
        elif elem.dof_count == 12 and dir_id == 3:
            var noverl = axial / length
            noverl *= 1.0 - elem.pdelta_1 - elem.pdelta_2
            k_local[2][2] += noverl
            k_local[2][8] -= noverl
            k_local[8][2] -= noverl
            k_local[8][8] += noverl
        elif elem.dof_count == 12 and dir_id == 5:
            k_local[4][2] += elem.pdelta_1 * axial
            k_local[4][8] -= elem.pdelta_1 * axial
            k_local[10][2] += elem.pdelta_2 * axial
            k_local[10][8] -= elem.pdelta_2 * axial
        elif elem.dof_count == 12 and dir_id == 6:
            k_local[5][1] -= elem.pdelta_3 * axial
            k_local[5][7] += elem.pdelta_3 * axial
            k_local[11][1] -= elem.pdelta_4 * axial
            k_local[11][7] += elem.pdelta_4 * axial


fn _assemble_zero_length_element(
    e: Int,
    elem: ElementInput,
    nodes: List[NodeInput],
    ndf: Int,
    ndm: Int,
    u: List[Float64],
    uniaxial_defs: List[UniMaterialDef],
    uniaxial_state_defs: List[Int],
    mut uniaxial_states: List[UniMaterialState],
    elem_uniaxial_offsets: List[Int],
    elem_uniaxial_counts: List[Int],
    elem_uniaxial_state_ids: List[Int],
    mut K: List[List[Float64]],
    mut F_int: List[Float64],
):
    var node1 = nodes[elem.node_index_1]
    var node2 = nodes[elem.node_index_2]
    var trans = link_orientation_matrix(
        ndm,
        node1.x,
        node1.y,
        node1.z,
        node2.x,
        node2.y,
        node2.z,
        elem.has_orient_x,
        elem.orient_x_1,
        elem.orient_x_2,
        elem.orient_x_3,
        elem.has_orient_y,
        elem.orient_y_1,
        elem.orient_y_2,
        elem.orient_y_3,
        False,
    )
    var dof_map = _elem_dof_map(elem)
    var ug = _gather_element_u(dof_map, u)
    var offset = elem_uniaxial_offsets[e]
    var count = elem_uniaxial_counts[e]
    for m in range(count):
        var row = _zero_length_row(_elem_dir(elem, m), ndm, ndf, trans)
        var strain = 0.0
        for i in range(elem.dof_count):
            strain += row[i] * ug[i]
        var state_index = elem_uniaxial_state_ids[offset + m]
        var def_index = uniaxial_state_defs[state_index]
        var mat_def = uniaxial_defs[def_index]
        ref state = uniaxial_states[state_index]
        uniaxial_set_trial_strain(mat_def, state, strain)
        var force = state.sig_t
        var tangent = state.tangent_t
        for i in range(elem.dof_count):
            var ri = row[i]
            if ri == 0.0:
                continue
            F_int[dof_map[i]] += ri * force
            for j in range(elem.dof_count):
                var rj = row[j]
                if rj != 0.0:
                    K[dof_map[i]][dof_map[j]] += ri * tangent * rj


fn _assemble_zero_length_element_native(
    e: Int,
    elem: ElementInput,
    nodes: List[NodeInput],
    ndf: Int,
    ndm: Int,
    u: List[Float64],
    uniaxial_defs: List[UniMaterialDef],
    uniaxial_state_defs: List[Int],
    mut uniaxial_states: List[UniMaterialState],
    elem_uniaxial_offsets: List[Int],
    elem_uniaxial_counts: List[Int],
    elem_uniaxial_state_ids: List[Int],
    elem_free_offsets: List[Int],
    elem_free_pool: List[Int],
    mut backend: LinearSolverBackend,
    mut F_int: List[Float64],
):
    var node1 = nodes[elem.node_index_1]
    var node2 = nodes[elem.node_index_2]
    var trans = link_orientation_matrix(
        ndm,
        node1.x,
        node1.y,
        node1.z,
        node2.x,
        node2.y,
        node2.z,
        elem.has_orient_x,
        elem.orient_x_1,
        elem.orient_x_2,
        elem.orient_x_3,
        elem.has_orient_y,
        elem.orient_y_1,
        elem.orient_y_2,
        elem.orient_y_3,
        False,
    )
    var dof_map = _elem_dof_map(elem)
    var ug = _gather_element_u(dof_map, u)
    var offset = elem_uniaxial_offsets[e]
    var count = elem_uniaxial_counts[e]
    var k_global: List[List[Float64]] = []
    for _ in range(elem.dof_count):
        var row: List[Float64] = []
        row.resize(elem.dof_count, 0.0)
        k_global.append(row^)
    for m in range(count):
        var row = _zero_length_row(_elem_dir(elem, m), ndm, ndf, trans)
        var strain = 0.0
        for i in range(elem.dof_count):
            strain += row[i] * ug[i]
        var state_index = elem_uniaxial_state_ids[offset + m]
        var def_index = uniaxial_state_defs[state_index]
        var mat_def = uniaxial_defs[def_index]
        ref state = uniaxial_states[state_index]
        uniaxial_set_trial_strain(mat_def, state, strain)
        var force = state.sig_t
        var tangent = state.tangent_t
        for i in range(elem.dof_count):
            var ri = row[i]
            if ri == 0.0:
                continue
            F_int[dof_map[i]] += ri * force
            for j in range(elem.dof_count):
                var rj = row[j]
                if rj != 0.0:
                    k_global[i][j] += ri * tangent * rj
    add_element_matrix_from_pool(
        backend,
        elem_free_pool,
        elem_free_offsets[e],
        elem.dof_count,
        k_global,
    )


fn _assemble_two_node_link_element(
    e: Int,
    elem: ElementInput,
    nodes: List[NodeInput],
    ndf: Int,
    ndm: Int,
    u: List[Float64],
    uniaxial_defs: List[UniMaterialDef],
    uniaxial_state_defs: List[Int],
    mut uniaxial_states: List[UniMaterialState],
    elem_uniaxial_offsets: List[Int],
    elem_uniaxial_counts: List[Int],
    elem_uniaxial_state_ids: List[Int],
    mut K: List[List[Float64]],
    mut F_int: List[Float64],
):
    var node1 = nodes[elem.node_index_1]
    var node2 = nodes[elem.node_index_2]
    var dx = node2.x - node1.x
    var dy = node2.y - node1.y
    var dz = node2.z - node1.z
    var length: Float64
    if ndm == 2:
        length = sqrt(dx * dx + dy * dy)
    else:
        length = sqrt(dx * dx + dy * dy + dz * dz)
    var trans = link_orientation_matrix(
        ndm,
        node1.x,
        node1.y,
        node1.z,
        node2.x,
        node2.y,
        node2.z,
        elem.has_orient_x,
        elem.orient_x_1,
        elem.orient_x_2,
        elem.orient_x_3,
        elem.has_orient_y,
        elem.orient_y_1,
        elem.orient_y_2,
        elem.orient_y_3,
        True,
    )
    var tgl = _two_node_link_tgl(ndm, ndf, trans)
    var tlb = _two_node_link_tlb(ndm, ndf, elem, length)
    var dof_map = _elem_dof_map(elem)
    var ug = _gather_element_u(dof_map, u)
    var ul: List[Float64] = []
    ul.resize(elem.dof_count, 0.0)
    for i in range(elem.dof_count):
        for j in range(elem.dof_count):
            ul[i] += tgl[i][j] * ug[j]
    var offset = elem_uniaxial_offsets[e]
    var count = elem_uniaxial_counts[e]
    var q_basic: List[Float64] = []
    var k_basic: List[Float64] = []
    q_basic.resize(count, 0.0)
    k_basic.resize(count, 0.0)
    for m in range(count):
        var ub = 0.0
        for j in range(elem.dof_count):
            ub += tlb[m][j] * ul[j]
        var state_index = elem_uniaxial_state_ids[offset + m]
        var def_index = uniaxial_state_defs[state_index]
        var mat_def = uniaxial_defs[def_index]
        ref state = uniaxial_states[state_index]
        uniaxial_set_trial_strain(mat_def, state, ub)
        q_basic[m] = state.sig_t
        k_basic[m] = state.tangent_t
    var k_local: List[List[Float64]] = []
    for _ in range(elem.dof_count):
        var row: List[Float64] = []
        row.resize(elem.dof_count, 0.0)
        k_local.append(row^)
    for m in range(count):
        for i in range(elem.dof_count):
            var ti = tlb[m][i]
            if ti == 0.0:
                continue
            for j in range(elem.dof_count):
                var tj = tlb[m][j]
                if tj != 0.0:
                    k_local[i][j] += ti * k_basic[m] * tj
    var p_local: List[Float64] = []
    p_local.resize(elem.dof_count, 0.0)
    for i in range(elem.dof_count):
        for m in range(count):
            p_local[i] += tlb[m][i] * q_basic[m]
    _two_node_link_add_pdelta_forces(ndm, ndf, elem, length, ul, q_basic, p_local)
    _two_node_link_add_pdelta_stiff(ndm, ndf, elem, length, q_basic, k_local)
    var p_global: List[Float64] = []
    p_global.resize(elem.dof_count, 0.0)
    for i in range(elem.dof_count):
        for j in range(elem.dof_count):
            p_global[i] += tgl[j][i] * p_local[j]
    for i in range(elem.dof_count):
        F_int[dof_map[i]] += p_global[i]
    for i in range(elem.dof_count):
        for j in range(elem.dof_count):
            var kij = 0.0
            for p in range(elem.dof_count):
                if tgl[p][i] == 0.0:
                    continue
                for q in range(elem.dof_count):
                    if tgl[q][j] != 0.0:
                        kij += tgl[p][i] * k_local[p][q] * tgl[q][j]
            K[dof_map[i]][dof_map[j]] += kij


fn _assemble_two_node_link_element_native(
    e: Int,
    elem: ElementInput,
    nodes: List[NodeInput],
    ndf: Int,
    ndm: Int,
    u: List[Float64],
    uniaxial_defs: List[UniMaterialDef],
    uniaxial_state_defs: List[Int],
    mut uniaxial_states: List[UniMaterialState],
    elem_uniaxial_offsets: List[Int],
    elem_uniaxial_counts: List[Int],
    elem_uniaxial_state_ids: List[Int],
    elem_free_offsets: List[Int],
    elem_free_pool: List[Int],
    mut backend: LinearSolverBackend,
    mut F_int: List[Float64],
):
    var node1 = nodes[elem.node_index_1]
    var node2 = nodes[elem.node_index_2]
    var dx = node2.x - node1.x
    var dy = node2.y - node1.y
    var dz = node2.z - node1.z
    var length: Float64
    if ndm == 2:
        length = sqrt(dx * dx + dy * dy)
    else:
        length = sqrt(dx * dx + dy * dy + dz * dz)
    var trans = link_orientation_matrix(
        ndm,
        node1.x,
        node1.y,
        node1.z,
        node2.x,
        node2.y,
        node2.z,
        elem.has_orient_x,
        elem.orient_x_1,
        elem.orient_x_2,
        elem.orient_x_3,
        elem.has_orient_y,
        elem.orient_y_1,
        elem.orient_y_2,
        elem.orient_y_3,
        True,
    )
    var tgl = _two_node_link_tgl(ndm, ndf, trans)
    var tlb = _two_node_link_tlb(ndm, ndf, elem, length)
    var dof_map = _elem_dof_map(elem)
    var ug = _gather_element_u(dof_map, u)
    var ul: List[Float64] = []
    ul.resize(elem.dof_count, 0.0)
    for i in range(elem.dof_count):
        for j in range(elem.dof_count):
            ul[i] += tgl[i][j] * ug[j]
    var offset = elem_uniaxial_offsets[e]
    var count = elem_uniaxial_counts[e]
    var q_basic: List[Float64] = []
    var k_basic: List[Float64] = []
    q_basic.resize(count, 0.0)
    k_basic.resize(count, 0.0)
    for m in range(count):
        var ub = 0.0
        for j in range(elem.dof_count):
            ub += tlb[m][j] * ul[j]
        var state_index = elem_uniaxial_state_ids[offset + m]
        var def_index = uniaxial_state_defs[state_index]
        var mat_def = uniaxial_defs[def_index]
        ref state = uniaxial_states[state_index]
        uniaxial_set_trial_strain(mat_def, state, ub)
        q_basic[m] = state.sig_t
        k_basic[m] = state.tangent_t
    var k_local: List[List[Float64]] = []
    for _ in range(elem.dof_count):
        var row: List[Float64] = []
        row.resize(elem.dof_count, 0.0)
        k_local.append(row^)
    for m in range(count):
        for i in range(elem.dof_count):
            var ti = tlb[m][i]
            if ti == 0.0:
                continue
            for j in range(elem.dof_count):
                var tj = tlb[m][j]
                if tj != 0.0:
                    k_local[i][j] += ti * k_basic[m] * tj
    var p_local: List[Float64] = []
    p_local.resize(elem.dof_count, 0.0)
    for i in range(elem.dof_count):
        for m in range(count):
            p_local[i] += tlb[m][i] * q_basic[m]
    _two_node_link_add_pdelta_forces(ndm, ndf, elem, length, ul, q_basic, p_local)
    _two_node_link_add_pdelta_stiff(ndm, ndf, elem, length, q_basic, k_local)
    var p_global: List[Float64] = []
    p_global.resize(elem.dof_count, 0.0)
    var k_global: List[List[Float64]] = []
    for _ in range(elem.dof_count):
        var row: List[Float64] = []
        row.resize(elem.dof_count, 0.0)
        k_global.append(row^)
    for i in range(elem.dof_count):
        for j in range(elem.dof_count):
            p_global[i] += tgl[j][i] * p_local[j]
            var kij = 0.0
            for p in range(elem.dof_count):
                if tgl[p][i] == 0.0:
                    continue
                for q in range(elem.dof_count):
                    if tgl[q][j] != 0.0:
                        kij += tgl[p][i] * k_local[p][q] * tgl[q][j]
            k_global[i][j] = kij
    for i in range(elem.dof_count):
        F_int[dof_map[i]] += p_global[i]
    add_element_matrix_from_pool(
        backend,
        elem_free_pool,
        elem_free_offsets[e],
        elem.dof_count,
        k_global,
    )
