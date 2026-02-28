from collections import List
from math import hypot, sqrt
from os import abort

from elements import (
    beam2d_element_load_global,
    beam_global_stiffness,
    beam2d_corotational_global_stiffness,
    beam2d_corotational_global_tangent_and_internal,
    beam2d_pdelta_global_stiffness,
    beam_column3d_fiber_global_tangent_and_internal,
    beam3d_corotational_global_tangent_and_internal,
    beam3d_global_stiffness,
    beam3d_pdelta_global_stiffness,
    disp_beam_column2d_global_tangent_and_internal,
    disp_beam_column3d_global_tangent_and_internal,
    force_beam_column2d_global_tangent_and_internal,
    force_beam_column3d_fiber_global_tangent_and_internal,
    force_beam_column3d_global_tangent_and_internal,
    link_orientation_matrix,
    quad4_plane_stress_stiffness,
    shell4_mindlin_stiffness,
    truss_global_stiffness,
    truss3d_global_stiffness,
)
from materials import (
    UniMaterialDef,
    UniMaterialState,
    uni_mat_initial_tangent,
    uniaxial_set_trial_strain,
)
from solver.banded import banded_add, banded_matrix
from solver.dof import node_dof_index
from solver.run_case.input_types import (
    DampingInput,
    ElementInput,
    ElementLoadInput,
    MaterialInput,
    NodeInput,
    SectionInput,
)
from solver.time_series import TimeSeriesInput, eval_time_series_input
from sections import FiberCell, FiberSection2dDef, FiberSection3dDef
from tag_types import ElementTypeTag, LinkDirectionTag


fn _zero_vector(mut vec: List[Float64]):
    for i in range(len(vec)):
        vec[i] = 0.0


fn _zero_matrix(mut mat: List[List[Float64]]):
    for i in range(len(mat)):
        for j in range(len(mat[i])):
            mat[i][j] = 0.0


@always_inline
fn _assembly_filter_accepts_element(elem: ElementInput, filter_mode: Int) -> Bool:
    if filter_mode == 0:
        return True

    var is_link_like = (
        elem.type_tag == ElementTypeTag.ZeroLength
        or elem.type_tag == ElementTypeTag.TwoNodeLink
    )
    if filter_mode == 1:
        return is_link_like
    if filter_mode == 2:
        return is_link_like and elem.do_rayleigh

    abort("unsupported assembly filter mode")
    return False


@always_inline
fn _scatter_add_row_unrolled4(
    mut K: List[List[Float64]],
    row_index: Int,
    k_row: List[Float64],
    dof_map: List[Int],
    count: Int,
):
    var b = 0
    while b + 3 < count:
        var b0 = dof_map[b]
        var b1 = dof_map[b + 1]
        var b2 = dof_map[b + 2]
        var b3 = dof_map[b + 3]
        K[row_index][b0] += k_row[b]
        K[row_index][b1] += k_row[b + 1]
        K[row_index][b2] += k_row[b + 2]
        K[row_index][b3] += k_row[b + 3]
        b += 4
    while b < count:
        var bidx = dof_map[b]
        K[row_index][bidx] += k_row[b]
        b += 1


@always_inline
fn _scatter_add_and_dot_row_simd4(
    mut K: List[List[Float64]],
    row_index: Int,
    k_row: List[Float64],
    dof_map: List[Int],
    u: List[Float64],
    count: Int,
) -> Float64:
    var sum = 0.0
    var b = 0
    while b + 3 < count:
        var b0 = dof_map[b]
        var b1 = dof_map[b + 1]
        var b2 = dof_map[b + 2]
        var b3 = dof_map[b + 3]
        var k0 = k_row[b]
        var k1 = k_row[b + 1]
        var k2 = k_row[b + 2]
        var k3 = k_row[b + 3]
        K[row_index][b0] += k0
        K[row_index][b1] += k1
        K[row_index][b2] += k2
        K[row_index][b3] += k3
        var k_vec = SIMD[DType.float64, 4](k0, k1, k2, k3)
        var u_vec = SIMD[DType.float64, 4](u[b0], u[b1], u[b2], u[b3])
        sum += (k_vec * u_vec).reduce_add()
        b += 4
    while b < count:
        var bidx = dof_map[b]
        var kval = k_row[b]
        K[row_index][bidx] += kval
        sum += kval * u[bidx]
        b += 1
    return sum


fn _elem_node(elem: ElementInput, idx: Int) -> Int:
    if idx == 0:
        return elem.node_1
    if idx == 1:
        return elem.node_2
    if idx == 2:
        return elem.node_3
    return elem.node_4


fn _elem_material(elem: ElementInput, idx: Int) -> Int:
    if idx == 0:
        return elem.material_1
    if idx == 1:
        return elem.material_2
    if idx == 2:
        return elem.material_3
    if idx == 3:
        return elem.material_4
    if idx == 4:
        return elem.material_5
    return elem.material_6


fn _elem_damp_material(elem: ElementInput, idx: Int) -> Int:
    if idx == 0:
        return elem.damp_material_1
    if idx == 1:
        return elem.damp_material_2
    if idx == 2:
        return elem.damp_material_3
    if idx == 3:
        return elem.damp_material_4
    if idx == 4:
        return elem.damp_material_5
    return elem.damp_material_6


fn _elem_dir(elem: ElementInput, idx: Int) -> Int:
    if idx == 0:
        return elem.dir_1
    if idx == 1:
        return elem.dir_2
    if idx == 2:
        return elem.dir_3
    if idx == 3:
        return elem.dir_4
    if idx == 4:
        return elem.dir_5
    return elem.dir_6


fn _elem_dof(elem: ElementInput, idx: Int) -> Int:
    if idx == 0:
        return elem.dof_1
    if idx == 1:
        return elem.dof_2
    if idx == 2:
        return elem.dof_3
    if idx == 3:
        return elem.dof_4
    if idx == 4:
        return elem.dof_5
    if idx == 5:
        return elem.dof_6
    if idx == 6:
        return elem.dof_7
    if idx == 7:
        return elem.dof_8
    if idx == 8:
        return elem.dof_9
    if idx == 9:
        return elem.dof_10
    if idx == 10:
        return elem.dof_11
    if idx == 11:
        return elem.dof_12
    if idx == 12:
        return elem.dof_13
    if idx == 13:
        return elem.dof_14
    if idx == 14:
        return elem.dof_15
    if idx == 15:
        return elem.dof_16
    if idx == 16:
        return elem.dof_17
    if idx == 17:
        return elem.dof_18
    if idx == 18:
        return elem.dof_19
    if idx == 19:
        return elem.dof_20
    if idx == 20:
        return elem.dof_21
    if idx == 21:
        return elem.dof_22
    if idx == 22:
        return elem.dof_23
    return elem.dof_24


fn _elem_dof_map(elem: ElementInput) -> List[Int]:
    var dof_map: List[Int] = []
    dof_map.resize(elem.dof_count, -1)
    for i in range(elem.dof_count):
        dof_map[i] = _elem_dof(elem, i)
    return dof_map^


fn _gather_element_u(dof_map: List[Int], u: List[Float64]) -> List[Float64]:
    var out: List[Float64] = []
    out.resize(len(dof_map), 0.0)
    for i in range(len(dof_map)):
        out[i] = u[dof_map[i]]
    return out^


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
        var state = uniaxial_states[state_index]
        uniaxial_set_trial_strain(mat_def, state, strain)
        uniaxial_states[state_index] = state
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
        var state = uniaxial_states[state_index]
        uniaxial_set_trial_strain(mat_def, state, ub)
        uniaxial_states[state_index] = state
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


fn _assemble_zero_length_damping_element(
    elem: ElementInput,
    nodes: List[NodeInput],
    ndf: Int,
    ndm: Int,
    materials_by_id: List[MaterialInput],
    mut C: List[List[Float64]],
):
    if elem.damp_material_count <= 0:
        return
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
    for m in range(elem.damp_material_count):
        var row = _zero_length_row(_elem_dir(elem, m), ndm, ndf, trans)
        var damp_mat_id = _elem_damp_material(elem, m)
        if damp_mat_id < 0 or damp_mat_id >= len(materials_by_id):
            abort("zeroLength damp material not found")
        var damp_mat = materials_by_id[damp_mat_id]
        if damp_mat.id < 0:
            abort("zeroLength damp material not found")
        var eta = damp_mat.E
        for i in range(elem.dof_count):
            var ri = row[i]
            if ri == 0.0:
                continue
            for j in range(elem.dof_count):
                var rj = row[j]
                if rj != 0.0:
                    C[dof_map[i]][dof_map[j]] += ri * eta * rj


fn assemble_zero_length_damping_typed(
    nodes: List[NodeInput],
    elements: List[ElementInput],
    materials_by_id: List[MaterialInput],
    node_count: Int,
    ndf: Int,
    ndm: Int,
    mut C: List[List[Float64]],
):
    _zero_matrix(C)
    var total_dofs = node_count * ndf
    if len(C) != total_dofs:
        abort("invalid damping matrix row count")
    for i in range(total_dofs):
        if len(C[i]) != total_dofs:
            abort("invalid damping matrix column count")
    for e in range(len(elements)):
        var elem = elements[e]
        if elem.type_tag != ElementTypeTag.ZeroLength:
            continue
        if elem.damp_material_count <= 0:
            continue
        _assemble_zero_length_damping_element(
            elem, nodes, ndf, ndm, materials_by_id, C
        )


fn _find_damping_input(dampings: List[DampingInput], tag: Int) -> Int:
    for i in range(len(dampings)):
        if dampings[i].tag == tag:
            return i
    return -1


fn _zero_length_secstif_active(
    damping: DampingInput,
    time: Float64,
    dt: Float64,
    time_series: List[TimeSeriesInput],
    time_series_values: List[Float64],
    time_series_times: List[Float64],
) raises -> Float64:
    if dt <= 0.0:
        return 0.0
    if time <= damping.activate_time or time >= damping.deactivate_time:
        return 0.0
    var factor = 1.0
    if damping.factor_ts_index >= 0:
        factor = eval_time_series_input(
            time_series[damping.factor_ts_index],
            time,
            time_series_values,
            time_series_times,
        )
    return factor


fn assemble_zero_length_damping_committed_typed(
    nodes: List[NodeInput],
    elements: List[ElementInput],
    dampings: List[DampingInput],
    time_series: List[TimeSeriesInput],
    time_series_values: List[Float64],
    time_series_times: List[Float64],
    ndf: Int,
    ndm: Int,
    time: Float64,
    dt: Float64,
    uniaxial_states: List[UniMaterialState],
    elem_uniaxial_offsets: List[Int],
    elem_uniaxial_counts: List[Int],
    elem_uniaxial_state_ids: List[Int],
    mut K_damp: List[List[Float64]],
    mut F_committed: List[Float64],
) raises:
    _zero_matrix(K_damp)
    _zero_vector(F_committed)
    for e in range(len(elements)):
        var elem = elements[e]
        if elem.type_tag != ElementTypeTag.ZeroLength or elem.damping_tag < 0:
            continue
        var damping_index = _find_damping_input(dampings, elem.damping_tag)
        if damping_index < 0:
            abort("zeroLength damping not found")
        var damping = dampings[damping_index]
        if damping.type != "SecStif":
            abort("unsupported zeroLength damping type")
        var factor = _zero_length_secstif_active(
            damping,
            time,
            dt,
            time_series,
            time_series_values,
            time_series_times,
        )
        if factor == 0.0:
            continue
        var km = damping.beta / dt
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
        var offset = elem_uniaxial_offsets[e]
        var count = elem_uniaxial_counts[e]
        for m in range(count):
            var row = _zero_length_row(_elem_dir(elem, m), ndm, ndf, trans)
            var state_index = elem_uniaxial_state_ids[offset + m]
            var state = uniaxial_states[state_index]
            var force_committed = factor * km * state.sig_c
            var tangent = km * state.tangent_c
            for i in range(elem.dof_count):
                var ri = row[i]
                if ri == 0.0:
                    continue
                F_committed[dof_map[i]] += ri * force_committed
                for j in range(elem.dof_count):
                    var rj = row[j]
                    if rj != 0.0:
                        K_damp[dof_map[i]][dof_map[j]] += ri * tangent * rj


fn assemble_zero_length_damping_trial_typed(
    nodes: List[NodeInput],
    elements: List[ElementInput],
    dampings: List[DampingInput],
    time_series: List[TimeSeriesInput],
    time_series_values: List[Float64],
    time_series_times: List[Float64],
    ndf: Int,
    ndm: Int,
    time: Float64,
    dt: Float64,
    uniaxial_states: List[UniMaterialState],
    elem_uniaxial_offsets: List[Int],
    elem_uniaxial_counts: List[Int],
    elem_uniaxial_state_ids: List[Int],
    mut K_damp: List[List[Float64]],
    mut F_damp: List[Float64],
) raises:
    _zero_matrix(K_damp)
    _zero_vector(F_damp)
    for e in range(len(elements)):
        var elem = elements[e]
        if elem.type_tag != ElementTypeTag.ZeroLength or elem.damping_tag < 0:
            continue
        var damping_index = _find_damping_input(dampings, elem.damping_tag)
        if damping_index < 0:
            abort("zeroLength damping not found")
        var damping = dampings[damping_index]
        if damping.type != "SecStif":
            abort("unsupported zeroLength damping type")
        var factor = _zero_length_secstif_active(
            damping,
            time,
            dt,
            time_series,
            time_series_values,
            time_series_times,
        )
        if factor == 0.0:
            continue
        var km = damping.beta / dt
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
        var offset = elem_uniaxial_offsets[e]
        var count = elem_uniaxial_counts[e]
        for m in range(count):
            var row = _zero_length_row(_elem_dir(elem, m), ndm, ndf, trans)
            var state_index = elem_uniaxial_state_ids[offset + m]
            var state = uniaxial_states[state_index]
            var force = factor * km * (state.sig_t - state.sig_c)
            var tangent = km * state.tangent_t
            for i in range(elem.dof_count):
                var ri = row[i]
                if ri == 0.0:
                    continue
                F_damp[dof_map[i]] += ri * force
                for j in range(elem.dof_count):
                    var rj = row[j]
                    if rj != 0.0:
                        K_damp[dof_map[i]][dof_map[j]] += ri * tangent * rj


fn _build_elem_dof_soa(
    elements: List[ElementInput],
    mut elem_dof_offsets: List[Int],
    mut elem_dof_pool: List[Int],
):
    var elem_count = len(elements)
    elem_dof_offsets.resize(elem_count + 1, 0)

    var total_elem_dofs = 0
    for e in range(elem_count):
        elem_dof_offsets[e] = total_elem_dofs
        total_elem_dofs += elements[e].dof_count
    elem_dof_offsets[elem_count] = total_elem_dofs

    elem_dof_pool.resize(total_elem_dofs, -1)
    for e in range(elem_count):
        var elem = elements[e]
        var offset = elem_dof_offsets[e]
        for d in range(elem.dof_count):
            elem_dof_pool[offset + d] = _elem_dof(elem, d)


fn assemble_global_stiffness_banded_frame2d_typed(
    nodes: List[NodeInput],
    elements: List[ElementInput],
    sections_by_id: List[SectionInput],
    u: List[Float64],
    uniaxial_defs: List[UniMaterialDef],
    mut uniaxial_states: List[UniMaterialState],
    elem_uniaxial_offsets: List[Int],
    elem_uniaxial_counts: List[Int],
    elem_uniaxial_state_ids: List[Int],
    force_basic_offsets: List[Int],
    force_basic_counts: List[Int],
    mut force_basic_q: List[Float64],
    fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    free_index: List[Int],
    free_count: Int,
    bw: Int,
) raises -> List[List[Float64]]:
    var empty_element_loads: List[ElementLoadInput] = []
    var empty_elem_load_offsets: List[Int] = []
    var empty_elem_load_pool: List[Int] = []
    return assemble_global_stiffness_banded_frame2d_typed(
        nodes,
        elements,
        empty_element_loads,
        empty_elem_load_offsets,
        empty_elem_load_pool,
        0.0,
        sections_by_id,
        u,
        uniaxial_defs,
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
        free_index,
        free_count,
        bw,
    )


fn assemble_global_stiffness_banded_frame2d_typed(
    nodes: List[NodeInput],
    elements: List[ElementInput],
    element_loads: List[ElementLoadInput],
    elem_load_offsets: List[Int],
    elem_load_pool: List[Int],
    load_scale: Float64,
    sections_by_id: List[SectionInput],
    u: List[Float64],
    uniaxial_defs: List[UniMaterialDef],
    mut uniaxial_states: List[UniMaterialState],
    elem_uniaxial_offsets: List[Int],
    elem_uniaxial_counts: List[Int],
    elem_uniaxial_state_ids: List[Int],
    force_basic_offsets: List[Int],
    force_basic_counts: List[Int],
    mut force_basic_q: List[Float64],
    fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    free_index: List[Int],
    free_count: Int,
    bw: Int,
    elem_dof_offsets: List[Int],
    elem_dof_pool: List[Int],
    mut free_map: List[Int],
    mut u_elem: List[Float64],
    mut f_dummy: List[Float64],
) raises -> List[List[Float64]]:
    var K = banded_matrix(free_count, bw)
    if len(elem_dof_offsets) != len(elements) + 1:
        abort("invalid elem_dof_offsets size for banded frame2d assembly")
    if len(elem_dof_pool) != elem_dof_offsets[len(elements)]:
        abort("invalid elem_dof_pool size for banded frame2d assembly")
    if len(free_map) != 6:
        free_map.resize(6, -1)
    if len(u_elem) != 6:
        u_elem.resize(6, 0.0)

    for e in range(len(elements)):
        var elem = elements[e]
        var elem_type = elem.type_tag
        if (
            elem_type != ElementTypeTag.ElasticBeamColumn2d
            and elem_type != ElementTypeTag.ForceBeamColumn2d
            and elem_type != ElementTypeTag.DispBeamColumn2d
        ):
            abort(
                "typed frame2d banded assembly requires elasticBeamColumn2d, "
                "forceBeamColumn2d, or dispBeamColumn2d"
            )

        var dof_offset = elem_dof_offsets[e]
        var d0 = elem_dof_pool[dof_offset]
        var d1 = elem_dof_pool[dof_offset + 1]
        var d2 = elem_dof_pool[dof_offset + 2]
        var d3 = elem_dof_pool[dof_offset + 3]
        var d4 = elem_dof_pool[dof_offset + 4]
        var d5 = elem_dof_pool[dof_offset + 5]
        free_map[0] = free_index[d0]
        free_map[1] = free_index[d1]
        free_map[2] = free_index[d2]
        free_map[3] = free_index[d3]
        free_map[4] = free_index[d4]
        free_map[5] = free_index[d5]

        if elem_type == ElementTypeTag.ElasticBeamColumn2d:
            var i1 = elem.node_index_1
            var i2 = elem.node_index_2
            var node1 = nodes[i1]
            var node2 = nodes[i2]
            var sec = sections_by_id[elem.section]

            var geom = elem.geom_transf
            var k_global: List[List[Float64]] = []
            if geom == "Linear":
                k_global = beam_global_stiffness(
                    sec.E,
                    sec.A,
                    sec.I,
                    node1.x,
                    node1.y,
                    node2.x,
                    node2.y,
                )
            elif geom == "PDelta":
                u_elem[0] = u[d0]
                u_elem[1] = u[d1]
                u_elem[2] = u[d2]
                u_elem[3] = u[d3]
                u_elem[4] = u[d4]
                u_elem[5] = u[d5]
                k_global = beam2d_pdelta_global_stiffness(
                    sec.E,
                    sec.A,
                    sec.I,
                    node1.x,
                    node1.y,
                    node2.x,
                    node2.y,
                    u_elem,
                )
            elif geom == "Corotational":
                u_elem[0] = u[d0]
                u_elem[1] = u[d1]
                u_elem[2] = u[d2]
                u_elem[3] = u[d3]
                u_elem[4] = u[d4]
                u_elem[5] = u[d5]
                k_global = beam2d_corotational_global_stiffness(
                    sec.E,
                    sec.A,
                    sec.I,
                    node1.x,
                    node1.y,
                    node2.x,
                    node2.y,
                    u_elem,
                )
            else:
                abort("unsupported geomTransf: " + geom)
            for a in range(6):
                var Aidx = free_map[a]
                if Aidx < 0:
                    continue
                for b in range(6):
                    var Bidx = free_map[b]
                    if Bidx < 0:
                        continue
                    banded_add(K, bw, Aidx, Bidx, k_global[a][b])
        else:
            var i1 = elem.node_index_1
            var i2 = elem.node_index_2
            var node1 = nodes[i1]
            var node2 = nodes[i2]
            var sec = sections_by_id[elem.section]

            u_elem[0] = u[d0]
            u_elem[1] = u[d1]
            u_elem[2] = u[d2]
            u_elem[3] = u[d3]
            u_elem[4] = u[d4]
            u_elem[5] = u[d5]
            var k_global: List[List[Float64]] = []
            if sec.type == "ElasticSection2d":
                var geom = elem.geom_transf
                if geom == "Linear":
                    k_global = beam_global_stiffness(
                        sec.E,
                        sec.A,
                        sec.I,
                        node1.x,
                        node1.y,
                        node2.x,
                        node2.y,
                    )
                elif geom == "PDelta":
                    k_global = beam2d_pdelta_global_stiffness(
                        sec.E,
                        sec.A,
                        sec.I,
                        node1.x,
                        node1.y,
                        node2.x,
                        node2.y,
                        u_elem,
                    )
                else:
                    abort(
                        "forceBeamColumn2d/dispBeamColumn2d supports geomTransf Linear or PDelta"
                    )
            else:
                var sec_index = fiber_section_index_by_id[elem.section]
                var sec_def = fiber_section_defs[sec_index]
                var elem_offset = elem_uniaxial_offsets[e]
                var elem_state_count = elem_uniaxial_counts[e]
                if elem_type == ElementTypeTag.ForceBeamColumn2d:
                    force_beam_column2d_global_tangent_and_internal(
                        e,
                        node1.x,
                        node1.y,
                        node2.x,
                        node2.y,
                        u_elem,
                        element_loads,
                        elem_load_offsets,
                        elem_load_pool,
                        load_scale,
                        sec_def,
                        fiber_section_cells,
                        uniaxial_defs,
                        uniaxial_states,
                        elem_uniaxial_state_ids,
                        elem_offset,
                        elem_state_count,
                        elem.geom_transf,
                        elem.integration,
                        elem.num_int_pts,
                        force_basic_q,
                        force_basic_offsets[e],
                        force_basic_counts[e],
                        k_global,
                        f_dummy,
                    )
                else:
                    disp_beam_column2d_global_tangent_and_internal(
                        e,
                        node1.x,
                        node1.y,
                        node2.x,
                        node2.y,
                        u_elem,
                        element_loads,
                        elem_load_offsets,
                        elem_load_pool,
                        load_scale,
                        sec_def,
                        fiber_section_cells,
                        uniaxial_defs,
                        uniaxial_states,
                        elem_uniaxial_state_ids,
                        elem_offset,
                        elem_state_count,
                        elem.geom_transf,
                        elem.integration,
                        elem.num_int_pts,
                        k_global,
                        f_dummy,
                    )
            for a in range(6):
                var Aidx = free_map[a]
                if Aidx < 0:
                    continue
                for b in range(6):
                    var Bidx = free_map[b]
                    if Bidx < 0:
                        continue
                    banded_add(K, bw, Aidx, Bidx, k_global[a][b])
    return K^


fn assemble_global_stiffness_banded_frame2d_typed(
    nodes: List[NodeInput],
    elements: List[ElementInput],
    sections_by_id: List[SectionInput],
    u: List[Float64],
    uniaxial_defs: List[UniMaterialDef],
    mut uniaxial_states: List[UniMaterialState],
    elem_uniaxial_offsets: List[Int],
    elem_uniaxial_counts: List[Int],
    elem_uniaxial_state_ids: List[Int],
    force_basic_offsets: List[Int],
    force_basic_counts: List[Int],
    mut force_basic_q: List[Float64],
    fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    free_index: List[Int],
    free_count: Int,
    bw: Int,
    elem_dof_offsets: List[Int],
    elem_dof_pool: List[Int],
    mut free_map: List[Int],
    mut u_elem: List[Float64],
    mut f_dummy: List[Float64],
) raises -> List[List[Float64]]:
    var empty_element_loads: List[ElementLoadInput] = []
    var empty_elem_load_offsets: List[Int] = []
    var empty_elem_load_pool: List[Int] = []
    return assemble_global_stiffness_banded_frame2d_typed(
        nodes,
        elements,
        empty_element_loads,
        empty_elem_load_offsets,
        empty_elem_load_pool,
        0.0,
        sections_by_id,
        u,
        uniaxial_defs,
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
        free_index,
        free_count,
        bw,
        elem_dof_offsets,
        elem_dof_pool,
        free_map,
        u_elem,
        f_dummy,
    )


fn assemble_global_stiffness_banded_frame2d_typed(
    nodes: List[NodeInput],
    elements: List[ElementInput],
    element_loads: List[ElementLoadInput],
    elem_load_offsets: List[Int],
    elem_load_pool: List[Int],
    load_scale: Float64,
    sections_by_id: List[SectionInput],
    u: List[Float64],
    uniaxial_defs: List[UniMaterialDef],
    mut uniaxial_states: List[UniMaterialState],
    elem_uniaxial_offsets: List[Int],
    elem_uniaxial_counts: List[Int],
    elem_uniaxial_state_ids: List[Int],
    force_basic_offsets: List[Int],
    force_basic_counts: List[Int],
    mut force_basic_q: List[Float64],
    fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    free_index: List[Int],
    free_count: Int,
    bw: Int,
) raises -> List[List[Float64]]:
    var elem_dof_offsets: List[Int] = []
    var elem_dof_pool: List[Int] = []
    _build_elem_dof_soa(elements, elem_dof_offsets, elem_dof_pool)
    var free_map: List[Int] = []
    var u_elem: List[Float64] = []
    var f_dummy: List[Float64] = []
    return assemble_global_stiffness_banded_frame2d_typed(
        nodes,
        elements,
        element_loads,
        elem_load_offsets,
        elem_load_pool,
        load_scale,
        sections_by_id,
        u,
        uniaxial_defs,
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
        free_index,
        free_count,
        bw,
        elem_dof_offsets,
        elem_dof_pool,
        free_map,
        u_elem,
        f_dummy,
    )


fn assemble_global_stiffness_typed(
    nodes: List[NodeInput],
    elements: List[ElementInput],
    sections_by_id: List[SectionInput],
    materials_by_id: List[MaterialInput],
    id_to_index: List[Int],
    node_count: Int,
    ndf: Int,
    ndm: Int,
    u: List[Float64],
    uniaxial_defs: List[UniMaterialDef],
    uniaxial_state_defs: List[Int],
    mut uniaxial_states: List[UniMaterialState],
    elem_uniaxial_offsets: List[Int],
    elem_uniaxial_counts: List[Int],
    elem_uniaxial_state_ids: List[Int],
    force_basic_offsets: List[Int],
    force_basic_counts: List[Int],
    mut force_basic_q: List[Float64],
    fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    fiber_section3d_defs: List[FiberSection3dDef],
    fiber_section3d_cells: List[FiberCell],
    fiber_section3d_index_by_id: List[Int],
) raises -> List[List[Float64]]:
    var empty_element_loads: List[ElementLoadInput] = []
    var empty_elem_load_offsets: List[Int] = []
    var empty_elem_load_pool: List[Int] = []
    return assemble_global_stiffness_typed(
        nodes,
        elements,
        empty_element_loads,
        empty_elem_load_offsets,
        empty_elem_load_pool,
        0.0,
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
        force_basic_offsets,
        force_basic_counts,
        force_basic_q,
        fiber_section_defs,
        fiber_section_cells,
        fiber_section_index_by_id,
        fiber_section3d_defs,
        fiber_section3d_cells,
        fiber_section3d_index_by_id,
    )


fn assemble_global_stiffness_typed(
    nodes: List[NodeInput],
    elements: List[ElementInput],
    element_loads: List[ElementLoadInput],
    elem_load_offsets: List[Int],
    elem_load_pool: List[Int],
    load_scale: Float64,
    sections_by_id: List[SectionInput],
    materials_by_id: List[MaterialInput],
    id_to_index: List[Int],
    node_count: Int,
    ndf: Int,
    ndm: Int,
    u: List[Float64],
    uniaxial_defs: List[UniMaterialDef],
    uniaxial_state_defs: List[Int],
    mut uniaxial_states: List[UniMaterialState],
    elem_uniaxial_offsets: List[Int],
    elem_uniaxial_counts: List[Int],
    elem_uniaxial_state_ids: List[Int],
    force_basic_offsets: List[Int],
    force_basic_counts: List[Int],
    mut force_basic_q: List[Float64],
    fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    fiber_section3d_defs: List[FiberSection3dDef],
    fiber_section3d_cells: List[FiberCell],
    fiber_section3d_index_by_id: List[Int],
) raises -> List[List[Float64]]:
    var total_dofs = node_count * ndf
    var K: List[List[Float64]] = []
    for _ in range(total_dofs):
        var row: List[Float64] = []
        row.resize(total_dofs, 0.0)
        K.append(row^)
    var F_int: List[Float64] = []
    F_int.resize(total_dofs, 0.0)
    assemble_global_stiffness_and_internal(
        nodes,
        elements,
        element_loads,
        elem_load_offsets,
        elem_load_pool,
        load_scale,
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
        force_basic_offsets,
        force_basic_counts,
        force_basic_q,
        fiber_section_defs,
        fiber_section_cells,
        fiber_section_index_by_id,
        fiber_section3d_defs,
        fiber_section3d_cells,
        fiber_section3d_index_by_id,
        K,
        F_int,
    )
    return K^


fn assemble_link_stiffness_typed(
    nodes: List[NodeInput],
    elements: List[ElementInput],
    sections_by_id: List[SectionInput],
    materials_by_id: List[MaterialInput],
    id_to_index: List[Int],
    node_count: Int,
    ndf: Int,
    ndm: Int,
    u: List[Float64],
    uniaxial_defs: List[UniMaterialDef],
    uniaxial_state_defs: List[Int],
    mut uniaxial_states: List[UniMaterialState],
    elem_uniaxial_offsets: List[Int],
    elem_uniaxial_counts: List[Int],
    elem_uniaxial_state_ids: List[Int],
    force_basic_offsets: List[Int],
    force_basic_counts: List[Int],
    mut force_basic_q: List[Float64],
    fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    fiber_section3d_defs: List[FiberSection3dDef],
    fiber_section3d_cells: List[FiberCell],
    fiber_section3d_index_by_id: List[Int],
    elem_dof_offsets: List[Int],
    elem_dof_pool: List[Int],
    mut dof_map6: List[Int],
    mut dof_map12: List[Int],
    mut u_elem6: List[Float64],
    only_rayleigh_participating: Bool,
    mut K: List[List[Float64]],
) raises:
    var total_dofs = node_count * ndf
    var F_dummy: List[Float64] = []
    F_dummy.resize(total_dofs, 0.0)
    var empty_element_loads: List[ElementLoadInput] = []
    var empty_elem_load_offsets: List[Int] = []
    var empty_elem_load_pool: List[Int] = []
    var filter_mode = 1
    if only_rayleigh_participating:
        filter_mode = 2
    _assemble_global_stiffness_and_internal_filtered(
        nodes,
        elements,
        empty_element_loads,
        empty_elem_load_offsets,
        empty_elem_load_pool,
        0.0,
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
        force_basic_offsets,
        force_basic_counts,
        force_basic_q,
        fiber_section_defs,
        fiber_section_cells,
        fiber_section_index_by_id,
        fiber_section3d_defs,
        fiber_section3d_cells,
        fiber_section3d_index_by_id,
        elem_dof_offsets,
        elem_dof_pool,
        dof_map6,
        dof_map12,
        u_elem6,
        filter_mode,
        K,
        F_dummy,
    )


fn assemble_internal_forces_typed(
    nodes: List[NodeInput],
    elements: List[ElementInput],
    sections_by_id: List[SectionInput],
    materials_by_id: List[MaterialInput],
    id_to_index: List[Int],
    node_count: Int,
    ndf: Int,
    ndm: Int,
    u: List[Float64],
    uniaxial_defs: List[UniMaterialDef],
    uniaxial_state_defs: List[Int],
    mut uniaxial_states: List[UniMaterialState],
    elem_uniaxial_offsets: List[Int],
    elem_uniaxial_counts: List[Int],
    elem_uniaxial_state_ids: List[Int],
    force_basic_offsets: List[Int],
    force_basic_counts: List[Int],
    mut force_basic_q: List[Float64],
    fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    fiber_section3d_defs: List[FiberSection3dDef],
    fiber_section3d_cells: List[FiberCell],
    fiber_section3d_index_by_id: List[Int],
) raises -> List[Float64]:
    var empty_element_loads: List[ElementLoadInput] = []
    var empty_elem_load_offsets: List[Int] = []
    var empty_elem_load_pool: List[Int] = []
    return assemble_internal_forces_typed(
        nodes,
        elements,
        empty_element_loads,
        empty_elem_load_offsets,
        empty_elem_load_pool,
        0.0,
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
        force_basic_offsets,
        force_basic_counts,
        force_basic_q,
        fiber_section_defs,
        fiber_section_cells,
        fiber_section_index_by_id,
        fiber_section3d_defs,
        fiber_section3d_cells,
        fiber_section3d_index_by_id,
    )


fn assemble_internal_forces_typed(
    nodes: List[NodeInput],
    elements: List[ElementInput],
    element_loads: List[ElementLoadInput],
    elem_load_offsets: List[Int],
    elem_load_pool: List[Int],
    load_scale: Float64,
    sections_by_id: List[SectionInput],
    materials_by_id: List[MaterialInput],
    id_to_index: List[Int],
    node_count: Int,
    ndf: Int,
    ndm: Int,
    u: List[Float64],
    uniaxial_defs: List[UniMaterialDef],
    uniaxial_state_defs: List[Int],
    mut uniaxial_states: List[UniMaterialState],
    elem_uniaxial_offsets: List[Int],
    elem_uniaxial_counts: List[Int],
    elem_uniaxial_state_ids: List[Int],
    force_basic_offsets: List[Int],
    force_basic_counts: List[Int],
    mut force_basic_q: List[Float64],
    fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    fiber_section3d_defs: List[FiberSection3dDef],
    fiber_section3d_cells: List[FiberCell],
    fiber_section3d_index_by_id: List[Int],
) raises -> List[Float64]:
    var total_dofs = node_count * ndf
    var K_dummy: List[List[Float64]] = []
    for _ in range(total_dofs):
        var row: List[Float64] = []
        row.resize(total_dofs, 0.0)
        K_dummy.append(row^)
    var F_int: List[Float64] = []
    F_int.resize(total_dofs, 0.0)
    assemble_global_stiffness_and_internal(
        nodes,
        elements,
        element_loads,
        elem_load_offsets,
        elem_load_pool,
        load_scale,
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
        force_basic_offsets,
        force_basic_counts,
        force_basic_q,
        fiber_section_defs,
        fiber_section_cells,
        fiber_section_index_by_id,
        fiber_section3d_defs,
        fiber_section3d_cells,
        fiber_section3d_index_by_id,
        K_dummy,
        F_int,
    )
    return F_int^


fn assemble_global_stiffness_and_internal(
    nodes: List[NodeInput],
    elements: List[ElementInput],
    sections_by_id: List[SectionInput],
    materials_by_id: List[MaterialInput],
    id_to_index: List[Int],
    node_count: Int,
    ndf: Int,
    ndm: Int,
    u: List[Float64],
    uniaxial_defs: List[UniMaterialDef],
    uniaxial_state_defs: List[Int],
    mut uniaxial_states: List[UniMaterialState],
    elem_uniaxial_offsets: List[Int],
    elem_uniaxial_counts: List[Int],
    elem_uniaxial_state_ids: List[Int],
    force_basic_offsets: List[Int],
    force_basic_counts: List[Int],
    mut force_basic_q: List[Float64],
    fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    fiber_section3d_defs: List[FiberSection3dDef],
    fiber_section3d_cells: List[FiberCell],
    fiber_section3d_index_by_id: List[Int],
    mut K: List[List[Float64]],
    mut F_int: List[Float64],
) raises:
    var empty_element_loads: List[ElementLoadInput] = []
    var empty_elem_load_offsets: List[Int] = []
    var empty_elem_load_pool: List[Int] = []
    assemble_global_stiffness_and_internal(
        nodes,
        elements,
        empty_element_loads,
        empty_elem_load_offsets,
        empty_elem_load_pool,
        0.0,
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
        force_basic_offsets,
        force_basic_counts,
        force_basic_q,
        fiber_section_defs,
        fiber_section_cells,
        fiber_section_index_by_id,
        fiber_section3d_defs,
        fiber_section3d_cells,
        fiber_section3d_index_by_id,
        K,
        F_int,
    )


fn assemble_global_stiffness_and_internal(
    nodes: List[NodeInput],
    elements: List[ElementInput],
    element_loads: List[ElementLoadInput],
    elem_load_offsets: List[Int],
    elem_load_pool: List[Int],
    load_scale: Float64,
    sections_by_id: List[SectionInput],
    materials_by_id: List[MaterialInput],
    id_to_index: List[Int],
    node_count: Int,
    ndf: Int,
    ndm: Int,
    u: List[Float64],
    uniaxial_defs: List[UniMaterialDef],
    uniaxial_state_defs: List[Int],
    mut uniaxial_states: List[UniMaterialState],
    elem_uniaxial_offsets: List[Int],
    elem_uniaxial_counts: List[Int],
    elem_uniaxial_state_ids: List[Int],
    force_basic_offsets: List[Int],
    force_basic_counts: List[Int],
    mut force_basic_q: List[Float64],
    fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    fiber_section3d_defs: List[FiberSection3dDef],
    fiber_section3d_cells: List[FiberCell],
    fiber_section3d_index_by_id: List[Int],
    mut K: List[List[Float64]],
    mut F_int: List[Float64],
) raises:
    var elem_dof_offsets: List[Int] = []
    var elem_dof_pool: List[Int] = []
    _build_elem_dof_soa(elements, elem_dof_offsets, elem_dof_pool)
    var dof_map6: List[Int] = []
    var dof_map12: List[Int] = []
    var u_elem6: List[Float64] = []
    assemble_global_stiffness_and_internal(
        nodes,
        elements,
        element_loads,
        elem_load_offsets,
        elem_load_pool,
        load_scale,
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
        force_basic_offsets,
        force_basic_counts,
        force_basic_q,
        fiber_section_defs,
        fiber_section_cells,
        fiber_section_index_by_id,
        fiber_section3d_defs,
        fiber_section3d_cells,
        fiber_section3d_index_by_id,
        elem_dof_offsets,
        elem_dof_pool,
        dof_map6,
        dof_map12,
        u_elem6,
        K,
        F_int,
    )


fn assemble_global_stiffness_and_internal(
    nodes: List[NodeInput],
    elements: List[ElementInput],
    sections_by_id: List[SectionInput],
    materials_by_id: List[MaterialInput],
    id_to_index: List[Int],
    node_count: Int,
    ndf: Int,
    ndm: Int,
    u: List[Float64],
    uniaxial_defs: List[UniMaterialDef],
    uniaxial_state_defs: List[Int],
    mut uniaxial_states: List[UniMaterialState],
    elem_uniaxial_offsets: List[Int],
    elem_uniaxial_counts: List[Int],
    elem_uniaxial_state_ids: List[Int],
    force_basic_offsets: List[Int],
    force_basic_counts: List[Int],
    mut force_basic_q: List[Float64],
    fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    fiber_section3d_defs: List[FiberSection3dDef],
    fiber_section3d_cells: List[FiberCell],
    fiber_section3d_index_by_id: List[Int],
    elem_dof_offsets: List[Int],
    elem_dof_pool: List[Int],
    mut dof_map6: List[Int],
    mut dof_map12: List[Int],
    mut u_elem6: List[Float64],
    mut K: List[List[Float64]],
    mut F_int: List[Float64],
) raises:
    var empty_element_loads: List[ElementLoadInput] = []
    var empty_elem_load_offsets: List[Int] = []
    var empty_elem_load_pool: List[Int] = []
    assemble_global_stiffness_and_internal(
        nodes,
        elements,
        empty_element_loads,
        empty_elem_load_offsets,
        empty_elem_load_pool,
        0.0,
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
        force_basic_offsets,
        force_basic_counts,
        force_basic_q,
        fiber_section_defs,
        fiber_section_cells,
        fiber_section_index_by_id,
        fiber_section3d_defs,
        fiber_section3d_cells,
        fiber_section3d_index_by_id,
        elem_dof_offsets,
        elem_dof_pool,
        dof_map6,
        dof_map12,
        u_elem6,
        K,
        F_int,
    )


fn assemble_global_stiffness_and_internal(
    nodes: List[NodeInput],
    elements: List[ElementInput],
    element_loads: List[ElementLoadInput],
    elem_load_offsets: List[Int],
    elem_load_pool: List[Int],
    load_scale: Float64,
    sections_by_id: List[SectionInput],
    materials_by_id: List[MaterialInput],
    id_to_index: List[Int],
    node_count: Int,
    ndf: Int,
    ndm: Int,
    u: List[Float64],
    uniaxial_defs: List[UniMaterialDef],
    uniaxial_state_defs: List[Int],
    mut uniaxial_states: List[UniMaterialState],
    elem_uniaxial_offsets: List[Int],
    elem_uniaxial_counts: List[Int],
    elem_uniaxial_state_ids: List[Int],
    force_basic_offsets: List[Int],
    force_basic_counts: List[Int],
    mut force_basic_q: List[Float64],
    fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    fiber_section3d_defs: List[FiberSection3dDef],
    fiber_section3d_cells: List[FiberCell],
    fiber_section3d_index_by_id: List[Int],
    elem_dof_offsets: List[Int],
    elem_dof_pool: List[Int],
    mut dof_map6: List[Int],
    mut dof_map12: List[Int],
    mut u_elem6: List[Float64],
    mut K: List[List[Float64]],
    mut F_int: List[Float64],
) raises:
    _assemble_global_stiffness_and_internal_filtered(
        nodes,
        elements,
        element_loads,
        elem_load_offsets,
        elem_load_pool,
        load_scale,
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
        force_basic_offsets,
        force_basic_counts,
        force_basic_q,
        fiber_section_defs,
        fiber_section_cells,
        fiber_section_index_by_id,
        fiber_section3d_defs,
        fiber_section3d_cells,
        fiber_section3d_index_by_id,
        elem_dof_offsets,
        elem_dof_pool,
        dof_map6,
        dof_map12,
        u_elem6,
        0,
        K,
        F_int,
    )


fn _assemble_global_stiffness_and_internal_filtered(
    nodes: List[NodeInput],
    elements: List[ElementInput],
    element_loads: List[ElementLoadInput],
    elem_load_offsets: List[Int],
    elem_load_pool: List[Int],
    load_scale: Float64,
    sections_by_id: List[SectionInput],
    materials_by_id: List[MaterialInput],
    id_to_index: List[Int],
    node_count: Int,
    ndf: Int,
    ndm: Int,
    u: List[Float64],
    uniaxial_defs: List[UniMaterialDef],
    uniaxial_state_defs: List[Int],
    mut uniaxial_states: List[UniMaterialState],
    elem_uniaxial_offsets: List[Int],
    elem_uniaxial_counts: List[Int],
    elem_uniaxial_state_ids: List[Int],
    force_basic_offsets: List[Int],
    force_basic_counts: List[Int],
    mut force_basic_q: List[Float64],
    fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    fiber_section3d_defs: List[FiberSection3dDef],
    fiber_section3d_cells: List[FiberCell],
    fiber_section3d_index_by_id: List[Int],
    elem_dof_offsets: List[Int],
    elem_dof_pool: List[Int],
    mut dof_map6: List[Int],
    mut dof_map12: List[Int],
    mut u_elem6: List[Float64],
    filter_mode: Int,
    mut K: List[List[Float64]],
    mut F_int: List[Float64],
) raises:
    _zero_matrix(K)
    _zero_vector(F_int)

    var elem_count = len(elements)
    if len(elem_dof_offsets) != elem_count + 1:
        abort("invalid elem_dof_offsets size for global assembly")
    if len(elem_dof_pool) != elem_dof_offsets[elem_count]:
        abort("invalid elem_dof_pool size for global assembly")
    if len(dof_map6) != 6:
        dof_map6.resize(6, 0)
    if len(dof_map12) != 12:
        dof_map12.resize(12, 0)
    if len(u_elem6) != 6:
        u_elem6.resize(6, 0.0)
    var k_elem6: List[List[Float64]] = []
    var f_elem6: List[Float64] = []
    var u_elem12: List[Float64] = []
    u_elem12.resize(12, 0.0)
    var k_elem12: List[List[Float64]] = []
    var f_elem12: List[Float64] = []

    for e in range(elem_count):
        var elem = elements[e]
        if not _assembly_filter_accepts_element(elem, filter_mode):
            continue
        var elem_type = elem.type_tag
        if elem_type == ElementTypeTag.ElasticBeamColumn2d:
            var i1 = elem.node_index_1
            var i2 = elem.node_index_2
            var node1 = nodes[i1]
            var node2 = nodes[i2]

            var sec = sections_by_id[elem.section]
            var E = sec.E
            var A = sec.A
            var I = sec.I
            var dof_offset = elem_dof_offsets[e]
            for i in range(6):
                dof_map6[i] = elem_dof_pool[dof_offset + i]

            var geom = elem.geom_transf
            var k_global: List[List[Float64]] = []
            var f_elem: List[Float64] = []
            var f_load_global = beam2d_element_load_global(
                element_loads,
                elem_load_offsets,
                elem_load_pool,
                e,
                load_scale,
                node1.x,
                node1.y,
                node2.x,
                node2.y,
            )
            if geom == "Linear":
                k_global = beam_global_stiffness(
                    E,
                    A,
                    I,
                    node1.x,
                    node1.y,
                    node2.x,
                    node2.y,
                )
            elif geom == "PDelta":
                for i in range(6):
                    u_elem6[i] = u[dof_map6[i]]
                k_global = beam2d_pdelta_global_stiffness(
                    E,
                    A,
                    I,
                    node1.x,
                    node1.y,
                    node2.x,
                    node2.y,
                    u_elem6,
                )
            elif geom == "Corotational":
                for i in range(6):
                    u_elem6[i] = u[dof_map6[i]]
                beam2d_corotational_global_tangent_and_internal(
                    E,
                    A,
                    I,
                    node1.x,
                    node1.y,
                    node2.x,
                    node2.y,
                    u_elem6,
                    k_global,
                    f_elem,
                )
            else:
                abort("unsupported geomTransf: " + geom)

            if geom == "Corotational":
                for a in range(6):
                    var Aidx = dof_map6[a]
                    _scatter_add_row_unrolled4(K, Aidx, k_global[a], dof_map6, 6)
                    F_int[Aidx] += f_elem[a] - f_load_global[a]
            else:
                for a in range(6):
                    var Aidx = dof_map6[a]
                    F_int[Aidx] += _scatter_add_and_dot_row_simd4(
                        K, Aidx, k_global[a], dof_map6, u, 6
                    ) - f_load_global[a]
        elif (
            elem_type == ElementTypeTag.ForceBeamColumn2d
            or elem_type == ElementTypeTag.DispBeamColumn2d
        ):
            var i1 = elem.node_index_1
            var i2 = elem.node_index_2
            var node1 = nodes[i1]
            var node2 = nodes[i2]
            var sec = sections_by_id[elem.section]
            var dof_offset = elem_dof_offsets[e]
            for i in range(6):
                dof_map6[i] = elem_dof_pool[dof_offset + i]
                u_elem6[i] = u[dof_map6[i]]
            var k_global: List[List[Float64]] = []
            var f_global: List[Float64] = []
            if sec.type == "ElasticSection2d":
                var geom = elem.geom_transf
                if geom == "Linear":
                    k_global = beam_global_stiffness(
                        sec.E,
                        sec.A,
                        sec.I,
                        node1.x,
                        node1.y,
                        node2.x,
                        node2.y,
                    )
                elif geom == "PDelta":
                    k_global = beam2d_pdelta_global_stiffness(
                        sec.E,
                        sec.A,
                        sec.I,
                        node1.x,
                        node1.y,
                        node2.x,
                        node2.y,
                        u_elem6,
                    )
                else:
                    abort(
                        "forceBeamColumn2d/dispBeamColumn2d supports geomTransf Linear or PDelta"
                    )
                f_global.resize(6, 0.0)
                for a in range(6):
                    var sum = 0.0
                    for b in range(6):
                        sum += k_global[a][b] * u_elem6[b]
                    f_global[a] = sum
            else:
                var sec_index = fiber_section_index_by_id[elem.section]
                var sec_def = fiber_section_defs[sec_index]
                var elem_offset = elem_uniaxial_offsets[e]
                var elem_state_count = elem_uniaxial_counts[e]
                if elem_type == ElementTypeTag.ForceBeamColumn2d:
                    force_beam_column2d_global_tangent_and_internal(
                        e,
                        node1.x,
                        node1.y,
                        node2.x,
                        node2.y,
                        u_elem6,
                        element_loads,
                        elem_load_offsets,
                        elem_load_pool,
                        load_scale,
                        sec_def,
                        fiber_section_cells,
                        uniaxial_defs,
                        uniaxial_states,
                        elem_uniaxial_state_ids,
                        elem_offset,
                        elem_state_count,
                        elem.geom_transf,
                        elem.integration,
                        elem.num_int_pts,
                        force_basic_q,
                        force_basic_offsets[e],
                        force_basic_counts[e],
                        k_elem6,
                        f_elem6,
                    )
                else:
                    disp_beam_column2d_global_tangent_and_internal(
                        e,
                        node1.x,
                        node1.y,
                        node2.x,
                        node2.y,
                        u_elem6,
                        element_loads,
                        elem_load_offsets,
                        elem_load_pool,
                        load_scale,
                        sec_def,
                        fiber_section_cells,
                        uniaxial_defs,
                        uniaxial_states,
                        elem_uniaxial_state_ids,
                        elem_offset,
                        elem_state_count,
                        elem.geom_transf,
                        elem.integration,
                        elem.num_int_pts,
                        k_elem6,
                        f_elem6,
                    )
                for a in range(6):
                    var Aidx = dof_map6[a]
                    _scatter_add_row_unrolled4(K, Aidx, k_elem6[a], dof_map6, 6)
                    F_int[Aidx] += f_elem6[a]
                continue
            for a in range(6):
                var Aidx = dof_map6[a]
                _scatter_add_row_unrolled4(K, Aidx, k_global[a], dof_map6, 6)
                F_int[Aidx] += f_global[a]
        elif (
            elem_type == ElementTypeTag.ElasticBeamColumn3d
            or elem_type == ElementTypeTag.ForceBeamColumn3d
            or elem_type == ElementTypeTag.DispBeamColumn3d
        ):
            var i1 = elem.node_index_1
            var i2 = elem.node_index_2
            var node1 = nodes[i1]
            var node2 = nodes[i2]

            var sec = sections_by_id[elem.section]
            var dof_offset = elem_dof_offsets[e]
            for i in range(12):
                dof_map12[i] = elem_dof_pool[dof_offset + i]
            for i in range(12):
                u_elem12[i] = u[dof_map12[i]]
            if elem_type == ElementTypeTag.ElasticBeamColumn3d:
                if sec.type != "ElasticSection3d":
                    abort("elasticBeamColumn3d requires ElasticSection3d")
                var E = sec.E
                var A = sec.A
                var Iz = sec.Iz
                var Iy = sec.Iy
                var G = sec.G
                var J = sec.J
                force_beam_column3d_global_tangent_and_internal(
                    e,
                    node1.x,
                    node1.y,
                    node1.z,
                    node2.x,
                    node2.y,
                    node2.z,
                    u_elem12,
                    elem.geom_transf,
                    element_loads,
                    elem_load_offsets,
                    elem_load_pool,
                    load_scale,
                    E,
                    A,
                    Iy,
                    Iz,
                    G,
                    J,
                    k_elem12,
                    f_elem12,
                )
            else:
                if sec.type == "ElasticSection3d":
                    var E = sec.E
                    var A = sec.A
                    var Iz = sec.Iz
                    var Iy = sec.Iy
                    var G = sec.G
                    var J = sec.J
                    if elem_type == ElementTypeTag.ForceBeamColumn3d:
                        force_beam_column3d_global_tangent_and_internal(
                            e,
                            node1.x,
                            node1.y,
                            node1.z,
                            node2.x,
                            node2.y,
                            node2.z,
                            u_elem12,
                            elem.geom_transf,
                            element_loads,
                            elem_load_offsets,
                            elem_load_pool,
                            load_scale,
                            E,
                            A,
                            Iy,
                            Iz,
                            G,
                            J,
                            k_elem12,
                            f_elem12,
                        )
                    else:
                        disp_beam_column3d_global_tangent_and_internal(
                            e,
                            node1.x,
                            node1.y,
                            node1.z,
                            node2.x,
                            node2.y,
                            node2.z,
                            u_elem12,
                            elem.geom_transf,
                            element_loads,
                            elem_load_offsets,
                            elem_load_pool,
                            load_scale,
                            E,
                            A,
                            Iy,
                            Iz,
                            G,
                            J,
                            k_elem12,
                            f_elem12,
                        )
                elif sec.type == "FiberSection3d":
                    var sec_index = fiber_section3d_index_by_id[elem.section]
                    if sec_index < 0 or sec_index >= len(fiber_section3d_defs):
                        abort(elem.type + " fiber section not found")
                    if elem_type == ElementTypeTag.ForceBeamColumn3d:
                        force_beam_column3d_fiber_global_tangent_and_internal(
                            e,
                            node1.x,
                            node1.y,
                            node1.z,
                            node2.x,
                            node2.y,
                            node2.z,
                            u_elem12,
                            elem.geom_transf,
                            element_loads,
                            elem_load_offsets,
                            elem_load_pool,
                            load_scale,
                            fiber_section3d_defs[sec_index],
                            fiber_section3d_cells,
                            uniaxial_defs,
                            uniaxial_states,
                            elem_uniaxial_state_ids,
                            elem_uniaxial_offsets[e],
                            elem_uniaxial_counts[e],
                            elem.integration,
                            elem.num_int_pts,
                            sec.G,
                            sec.J,
                            force_basic_q,
                            force_basic_offsets[e],
                            force_basic_counts[e],
                            k_elem12,
                            f_elem12,
                        )
                    else:
                        beam_column3d_fiber_global_tangent_and_internal(
                            e,
                            node1.x,
                            node1.y,
                            node1.z,
                            node2.x,
                            node2.y,
                            node2.z,
                            u_elem12,
                            elem.geom_transf,
                            element_loads,
                            elem_load_offsets,
                            elem_load_pool,
                            load_scale,
                            fiber_section3d_defs[sec_index],
                            fiber_section3d_cells,
                            uniaxial_defs,
                            uniaxial_states,
                            elem_uniaxial_state_ids,
                            elem_uniaxial_offsets[e],
                            elem_uniaxial_counts[e],
                            elem.integration,
                            elem.num_int_pts,
                            sec.G,
                            sec.J,
                            k_elem12,
                            f_elem12,
                        )
                else:
                    abort(elem.type + " requires ElasticSection3d or FiberSection3d")
            for a in range(12):
                var Aidx = dof_map12[a]
                _scatter_add_row_unrolled4(K, Aidx, k_elem12[a], dof_map12, 12)
                F_int[Aidx] += f_elem12[a]
        elif elem_type == ElementTypeTag.Truss:
            var i1 = elem.node_index_1
            var i2 = elem.node_index_2
            var node1 = nodes[i1]
            var node2 = nodes[i2]

            var offset = elem_uniaxial_offsets[e]
            var state_index = elem_uniaxial_state_ids[offset]
            var def_index = uniaxial_state_defs[state_index]
            var mat_def = uniaxial_defs[def_index]
            var state = uniaxial_states[state_index]
            var A = elem.area

            if ndf == 2:
                var dx = node2.x - node1.x
                var dy = node2.y - node1.y
                var L = hypot(dx, dy)
                if L == 0.0:
                    abort("zero-length element")
                var c = dx / L
                var s = dy / L
                var dof_map = [
                    _elem_dof(elem, 0),
                    _elem_dof(elem, 1),
                    _elem_dof(elem, 2),
                    _elem_dof(elem, 3),
                ]
                var du = (u[dof_map[2]] - u[dof_map[0]]) * c + (
                    u[dof_map[3]] - u[dof_map[1]]
                ) * s
                var eps = du / L
                uniaxial_set_trial_strain(mat_def, state, eps)
                uniaxial_states[state_index] = state
                var N = state.sig_t * A
                var k = state.tangent_t * A / L
                var k_global = [
                    [k * c * c, k * c * s, -k * c * c, -k * c * s],
                    [k * c * s, k * s * s, -k * c * s, -k * s * s],
                    [-k * c * c, -k * c * s, k * c * c, k * c * s],
                    [-k * c * s, -k * s * s, k * c * s, k * s * s],
                ]
                for a in range(4):
                    var Aidx = dof_map[a]
                    for b in range(4):
                        var Bidx = dof_map[b]
                        K[Aidx][Bidx] += k_global[a][b]
                F_int[dof_map[0]] -= N * c
                F_int[dof_map[1]] -= N * s
                F_int[dof_map[2]] += N * c
                F_int[dof_map[3]] += N * s
            else:
                var dx = node2.x - node1.x
                var dy = node2.y - node1.y
                var dz = node2.z - node1.z
                var L = sqrt(dx * dx + dy * dy + dz * dz)
                if L == 0.0:
                    abort("zero-length element")
                var l = dx / L
                var m = dy / L
                var n = dz / L
                var dof_map = [
                    _elem_dof(elem, 0),
                    _elem_dof(elem, 1),
                    _elem_dof(elem, 2),
                    _elem_dof(elem, 3),
                    _elem_dof(elem, 4),
                    _elem_dof(elem, 5),
                ]
                var du = (u[dof_map[3]] - u[dof_map[0]]) * l + (
                    u[dof_map[4]] - u[dof_map[1]]
                ) * m + (u[dof_map[5]] - u[dof_map[2]]) * n
                var eps = du / L
                uniaxial_set_trial_strain(mat_def, state, eps)
                uniaxial_states[state_index] = state
                var N = state.sig_t * A
                var k = state.tangent_t * A / L
                var k_global = [
                    [k * l * l, k * l * m, k * l * n, -k * l * l, -k * l * m, -k * l * n],
                    [k * l * m, k * m * m, k * m * n, -k * l * m, -k * m * m, -k * m * n],
                    [k * l * n, k * m * n, k * n * n, -k * l * n, -k * m * n, -k * n * n],
                    [-k * l * l, -k * l * m, -k * l * n, k * l * l, k * l * m, k * l * n],
                    [-k * l * m, -k * m * m, -k * m * n, k * l * m, k * m * m, k * m * n],
                    [-k * l * n, -k * m * n, -k * n * n, k * l * n, k * m * n, k * n * n],
                ]
                for a in range(6):
                    var Aidx = dof_map[a]
                    for b in range(6):
                        var Bidx = dof_map[b]
                        K[Aidx][Bidx] += k_global[a][b]
                F_int[dof_map[0]] -= N * l
                F_int[dof_map[1]] -= N * m
                F_int[dof_map[2]] -= N * n
                F_int[dof_map[3]] += N * l
                F_int[dof_map[4]] += N * m
                F_int[dof_map[5]] += N * n
        elif elem_type == ElementTypeTag.ZeroLength:
            _assemble_zero_length_element(
                e,
                elem,
                nodes,
                ndf,
                ndm,
                u,
                uniaxial_defs,
                uniaxial_state_defs,
                uniaxial_states,
                elem_uniaxial_offsets,
                elem_uniaxial_counts,
                elem_uniaxial_state_ids,
                K,
                F_int,
            )
        elif elem_type == ElementTypeTag.TwoNodeLink:
            _assemble_two_node_link_element(
                e,
                elem,
                nodes,
                ndf,
                ndm,
                u,
                uniaxial_defs,
                uniaxial_state_defs,
                uniaxial_states,
                elem_uniaxial_offsets,
                elem_uniaxial_counts,
                elem_uniaxial_state_ids,
                K,
                F_int,
            )
        elif elem_type == ElementTypeTag.ZeroLengthSection:
            var dof_map = [
                _elem_dof(elem, 0),
                _elem_dof(elem, 1),
                _elem_dof(elem, 2),
                _elem_dof(elem, 3),
                _elem_dof(elem, 4),
                _elem_dof(elem, 5),
            ]
            var delta_axial = u[dof_map[3]] - u[dof_map[0]]
            var delta_curv = u[dof_map[5]] - u[dof_map[2]]

            var sec = sections_by_id[elem.section]
            var axial_force = 0.0
            var moment_z = 0.0
            var k11 = 0.0
            var k12 = 0.0
            var k22 = 0.0
            if sec.type == "ElasticSection2d":
                k11 = sec.E * sec.A
                k22 = sec.E * sec.I
                axial_force = k11 * delta_axial
                moment_z = k22 * delta_curv
            elif sec.type == "FiberSection2d":
                var sec_index = fiber_section_index_by_id[elem.section]
                if sec_index < 0 or sec_index >= len(fiber_section_defs):
                    abort("zeroLengthSection fiber section not found")
                var sec_def = fiber_section_defs[sec_index]
                var elem_offset = elem_uniaxial_offsets[e]
                var elem_state_count = elem_uniaxial_counts[e]
                if elem_state_count != sec_def.fiber_count:
                    abort("zeroLengthSection fiber state count mismatch")
                if elem_offset < 0 or elem_offset + elem_state_count > len(
                    elem_uniaxial_state_ids
                ):
                    abort("zeroLengthSection fiber state range out of bounds")
                for i in range(elem_state_count):
                    var cell = fiber_section_cells[sec_def.fiber_offset + i]
                    var y_rel = cell.y - sec_def.y_bar
                    var eps = delta_axial - y_rel * delta_curv

                    var state_index = elem_uniaxial_state_ids[elem_offset + i]
                    if state_index < 0 or state_index >= len(uniaxial_states):
                        abort("zeroLengthSection fiber state index out of range")
                    var def_index = cell.def_index
                    if def_index < 0 or def_index >= len(uniaxial_defs):
                        abort("zeroLengthSection fiber material definition out of range")
                    var mat_def = uniaxial_defs[def_index]
                    var state = uniaxial_states[state_index]
                    uniaxial_set_trial_strain(mat_def, state, eps)
                    uniaxial_states[state_index] = state

                    var area = cell.area
                    var fs = state.sig_t * area
                    var ks = state.tangent_t * area
                    axial_force += fs
                    moment_z += -fs * y_rel
                    k11 += ks
                    k12 += -ks * y_rel
                    k22 += ks * y_rel * y_rel
            else:
                abort("zeroLengthSection requires FiberSection2d or ElasticSection2d")

            var u1 = dof_map[0]
            var r1 = dof_map[2]
            var u2 = dof_map[3]
            var r2 = dof_map[5]

            K[u1][u1] += k11
            K[u1][r1] += k12
            K[u1][u2] -= k11
            K[u1][r2] -= k12

            K[r1][u1] += k12
            K[r1][r1] += k22
            K[r1][u2] -= k12
            K[r1][r2] -= k22

            K[u2][u1] -= k11
            K[u2][r1] -= k12
            K[u2][u2] += k11
            K[u2][r2] += k12

            K[r2][u1] -= k12
            K[r2][r1] -= k22
            K[r2][u2] += k12
            K[r2][r2] += k22

            F_int[u1] -= axial_force
            F_int[r1] -= moment_z
            F_int[u2] += axial_force
            F_int[r2] += moment_z
        elif elem_type == ElementTypeTag.FourNodeQuad:
            var i1 = elem.node_index_1
            var i2 = elem.node_index_2
            var i3 = elem.node_index_3
            var i4 = elem.node_index_4
            var node1 = nodes[i1]
            var node2 = nodes[i2]
            var node3 = nodes[i3]
            var node4 = nodes[i4]

            var mat = materials_by_id[elem.material]
            var E = mat.E
            var nu = mat.nu
            var t = elem.thickness

            var x: List[Float64] = []
            var y: List[Float64] = []
            x.resize(4, 0.0)
            y.resize(4, 0.0)
            x[0] = node1.x
            y[0] = node1.y
            x[1] = node2.x
            y[1] = node2.y
            x[2] = node3.x
            y[2] = node3.y
            x[3] = node4.x
            y[3] = node4.y

            var k_global = quad4_plane_stress_stiffness(E, nu, t, x, y)
            var dof_map = [
                _elem_dof(elem, 0),
                _elem_dof(elem, 1),
                _elem_dof(elem, 2),
                _elem_dof(elem, 3),
                _elem_dof(elem, 4),
                _elem_dof(elem, 5),
                _elem_dof(elem, 6),
                _elem_dof(elem, 7),
            ]
            for a in range(8):
                var Aidx = dof_map[a]
                F_int[Aidx] += _scatter_add_and_dot_row_simd4(
                    K, Aidx, k_global[a], dof_map, u, 8
                )
        elif elem_type == ElementTypeTag.Shell:
            var i1 = elem.node_index_1
            var i2 = elem.node_index_2
            var i3 = elem.node_index_3
            var i4 = elem.node_index_4
            var node1 = nodes[i1]
            var node2 = nodes[i2]
            var node3 = nodes[i3]
            var node4 = nodes[i4]

            var sec = sections_by_id[elem.section]
            var E = sec.E
            var nu = sec.nu
            var h = sec.h

            var x: List[Float64] = []
            var y: List[Float64] = []
            var z: List[Float64] = []
            x.resize(4, 0.0)
            y.resize(4, 0.0)
            z.resize(4, 0.0)
            x[0] = node1.x
            y[0] = node1.y
            z[0] = node1.z
            x[1] = node2.x
            y[1] = node2.y
            z[1] = node2.z
            x[2] = node3.x
            y[2] = node3.y
            z[2] = node3.z
            x[3] = node4.x
            y[3] = node4.y
            z[3] = node4.z

            var k_global = shell4_mindlin_stiffness(E, nu, h, x, y, z)
            var dof_map = [
                _elem_dof(elem, 0),
                _elem_dof(elem, 1),
                _elem_dof(elem, 2),
                _elem_dof(elem, 3),
                _elem_dof(elem, 4),
                _elem_dof(elem, 5),
                _elem_dof(elem, 6),
                _elem_dof(elem, 7),
                _elem_dof(elem, 8),
                _elem_dof(elem, 9),
                _elem_dof(elem, 10),
                _elem_dof(elem, 11),
                _elem_dof(elem, 12),
                _elem_dof(elem, 13),
                _elem_dof(elem, 14),
                _elem_dof(elem, 15),
                _elem_dof(elem, 16),
                _elem_dof(elem, 17),
                _elem_dof(elem, 18),
                _elem_dof(elem, 19),
                _elem_dof(elem, 20),
                _elem_dof(elem, 21),
                _elem_dof(elem, 22),
                _elem_dof(elem, 23),
            ]
            for a in range(24):
                var Aidx = dof_map[a]
                F_int[Aidx] += _scatter_add_and_dot_row_simd4(
                    K, Aidx, k_global[a], dof_map, u, 24
                )
        else:
            abort("unsupported element type tag")
