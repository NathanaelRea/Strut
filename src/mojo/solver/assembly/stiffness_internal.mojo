from collections import List
from math import hypot, sqrt
from os import abort

from elements import (
    beam_global_stiffness,
    beam2d_corotational_global_stiffness,
    beam2d_corotational_global_tangent_and_internal,
    beam2d_pdelta_global_stiffness,
    beam3d_global_stiffness,
    force_beam_column2d_global_tangent_and_internal,
    link_global_stiffness,
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
from solver.run_case.input_types import ElementInput, MaterialInput, NodeInput, SectionInput
from sections import FiberCell, FiberSection2dDef


fn _zero_vector(mut vec: List[Float64]):
    for i in range(len(vec)):
        vec[i] = 0.0


fn _zero_matrix(mut mat: List[List[Float64]]):
    for i in range(len(mat)):
        for j in range(len(mat[i])):
            mat[i][j] = 0.0


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
    fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    free_index: List[Int],
    free_count: Int,
    bw: Int,
) raises -> List[List[Float64]]:
    var K = banded_matrix(free_count, bw)

    for e in range(len(elements)):
        var elem = elements[e]
        if elem.type_tag == 1:
            var i1 = elem.node_index_1
            var i2 = elem.node_index_2
            var node1 = nodes[i1]
            var node2 = nodes[i2]
            var sec = sections_by_id[elem.section]

            var dof_map = [
                _elem_dof(elem, 0),
                _elem_dof(elem, 1),
                _elem_dof(elem, 2),
                _elem_dof(elem, 3),
                _elem_dof(elem, 4),
                _elem_dof(elem, 5),
            ]
            var free_map: List[Int] = []
            free_map.resize(6, -1)
            for i in range(6):
                free_map[i] = free_index[dof_map[i]]

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
                var u_elem: List[Float64] = []
                u_elem.resize(6, 0.0)
                for i in range(6):
                    u_elem[i] = u[dof_map[i]]
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
                var u_elem: List[Float64] = []
                u_elem.resize(6, 0.0)
                for i in range(6):
                    u_elem[i] = u[dof_map[i]]
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
        elif elem.type_tag == 2:
            var i1 = elem.node_index_1
            var i2 = elem.node_index_2
            var node1 = nodes[i1]
            var node2 = nodes[i2]
            var sec = sections_by_id[elem.section]

            var dof_map = [
                _elem_dof(elem, 0),
                _elem_dof(elem, 1),
                _elem_dof(elem, 2),
                _elem_dof(elem, 3),
                _elem_dof(elem, 4),
                _elem_dof(elem, 5),
            ]
            var free_map: List[Int] = []
            free_map.resize(6, -1)
            for i in range(6):
                free_map[i] = free_index[dof_map[i]]

            var u_elem: List[Float64] = []
            u_elem.resize(6, 0.0)
            for i in range(6):
                u_elem[i] = u[dof_map[i]]
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
                    abort("forceBeamColumn2d/dispBeamColumn2d supports geomTransf Linear or PDelta")
            else:
                var sec_index = fiber_section_index_by_id[elem.section]
                var sec_def = fiber_section_defs[sec_index]
                var elem_offset = elem_uniaxial_offsets[e]
                var elem_state_count = elem_uniaxial_counts[e]
                var f_dummy: List[Float64] = []
                force_beam_column2d_global_tangent_and_internal(
                    node1.x,
                    node1.y,
                    node2.x,
                    node2.y,
                    u_elem,
                    sec_def,
                    fiber_section_cells,
                    uniaxial_defs,
                    uniaxial_states,
                    elem_uniaxial_state_ids,
                    elem_offset,
                    elem_state_count,
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
        else:
            abort(
                "typed frame2d banded assembly requires elasticBeamColumn2d, "
                "forceBeamColumn2d, or dispBeamColumn2d"
            )

    return K^


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
    fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
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
        K,
        F_int,
    )
    return K^


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
    fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
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
    fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    mut K: List[List[Float64]],
    mut F_int: List[Float64],
) raises:
    _zero_matrix(K)
    _zero_vector(F_int)

    var elem_count = len(elements)
    for e in range(elem_count):
        var elem = elements[e]
        var elem_type = elem.type_tag
        if elem_type == 1:
            var i1 = elem.node_index_1
            var i2 = elem.node_index_2
            var node1 = nodes[i1]
            var node2 = nodes[i2]

            var sec = sections_by_id[elem.section]
            var E = sec.E
            var A = sec.A
            var I = sec.I

            var dof_map = [
                _elem_dof(elem, 0),
                _elem_dof(elem, 1),
                _elem_dof(elem, 2),
                _elem_dof(elem, 3),
                _elem_dof(elem, 4),
                _elem_dof(elem, 5),
            ]

            var geom = elem.geom_transf
            var k_global: List[List[Float64]] = []
            var f_elem: List[Float64] = []
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
                var u_elem: List[Float64] = []
                u_elem.resize(6, 0.0)
                for i in range(6):
                    u_elem[i] = u[dof_map[i]]
                k_global = beam2d_pdelta_global_stiffness(
                    E,
                    A,
                    I,
                    node1.x,
                    node1.y,
                    node2.x,
                    node2.y,
                    u_elem,
                )
            elif geom == "Corotational":
                var u_elem: List[Float64] = []
                u_elem.resize(6, 0.0)
                for i in range(6):
                    u_elem[i] = u[dof_map[i]]
                beam2d_corotational_global_tangent_and_internal(
                    E,
                    A,
                    I,
                    node1.x,
                    node1.y,
                    node2.x,
                    node2.y,
                    u_elem,
                    k_global,
                    f_elem,
                )
            else:
                abort("unsupported geomTransf: " + geom)

            if geom == "Corotational":
                for a in range(6):
                    var Aidx = dof_map[a]
                    for b in range(6):
                        var Bidx = dof_map[b]
                        K[Aidx][Bidx] += k_global[a][b]
                    F_int[Aidx] += f_elem[a]
            else:
                for a in range(6):
                    var Aidx = dof_map[a]
                    var sum = 0.0
                    for b in range(6):
                        var Bidx = dof_map[b]
                        var kval = k_global[a][b]
                        K[Aidx][Bidx] += kval
                        sum += kval * u[Bidx]
                    F_int[Aidx] += sum
        elif elem_type == 2:
            var i1 = elem.node_index_1
            var i2 = elem.node_index_2
            var node1 = nodes[i1]
            var node2 = nodes[i2]
            var sec = sections_by_id[elem.section]

            var dof_map = [
                _elem_dof(elem, 0),
                _elem_dof(elem, 1),
                _elem_dof(elem, 2),
                _elem_dof(elem, 3),
                _elem_dof(elem, 4),
                _elem_dof(elem, 5),
            ]
            var u_elem: List[Float64] = []
            u_elem.resize(6, 0.0)
            for i in range(6):
                u_elem[i] = u[dof_map[i]]
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
                        u_elem,
                    )
                else:
                    abort("forceBeamColumn2d/dispBeamColumn2d supports geomTransf Linear or PDelta")
                f_global.resize(6, 0.0)
                for a in range(6):
                    var sum = 0.0
                    for b in range(6):
                        sum += k_global[a][b] * u_elem[b]
                    f_global[a] = sum
            else:
                var sec_index = fiber_section_index_by_id[elem.section]
                var sec_def = fiber_section_defs[sec_index]
                var elem_offset = elem_uniaxial_offsets[e]
                var elem_state_count = elem_uniaxial_counts[e]
                force_beam_column2d_global_tangent_and_internal(
                    node1.x,
                    node1.y,
                    node2.x,
                    node2.y,
                    u_elem,
                    sec_def,
                    fiber_section_cells,
                    uniaxial_defs,
                    uniaxial_states,
                    elem_uniaxial_state_ids,
                    elem_offset,
                    elem_state_count,
                    elem.num_int_pts,
                    k_global,
                    f_global,
                )
            for a in range(6):
                var Aidx = dof_map[a]
                for b in range(6):
                    var Bidx = dof_map[b]
                    K[Aidx][Bidx] += k_global[a][b]
                F_int[Aidx] += f_global[a]
        elif elem_type == 3:
            var i1 = elem.node_index_1
            var i2 = elem.node_index_2
            var node1 = nodes[i1]
            var node2 = nodes[i2]

            var sec = sections_by_id[elem.section]
            var E = sec.E
            var A = sec.A
            var Iz = sec.Iz
            var Iy = sec.Iy
            var G = sec.G
            var J = sec.J

            var k_global = beam3d_global_stiffness(
                E,
                A,
                Iy,
                Iz,
                G,
                J,
                node1.x,
                node1.y,
                node1.z,
                node2.x,
                node2.y,
                node2.z,
            )
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
            ]
            for a in range(12):
                var Aidx = dof_map[a]
                var sum = 0.0
                for b in range(12):
                    var Bidx = dof_map[b]
                    var kval = k_global[a][b]
                    K[Aidx][Bidx] += kval
                    sum += kval * u[Bidx]
                F_int[Aidx] += sum
        elif elem_type == 4:
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
        elif elem_type == 5:
            var dof_map = [
                _elem_dof(elem, 0),
                _elem_dof(elem, 1),
                _elem_dof(elem, 2),
                _elem_dof(elem, 3),
            ]
            var offset = elem_uniaxial_offsets[e]
            var count = elem_uniaxial_counts[e]

            for m in range(count):
                var state_index = elem_uniaxial_state_ids[offset + m]
                var def_index = uniaxial_state_defs[state_index]
                var mat_def = uniaxial_defs[def_index]
                var state = uniaxial_states[state_index]
                var dir = _elem_dir(elem, m)
                if dir == 1:
                    uniaxial_set_trial_strain(mat_def, state, u[dof_map[2]] - u[dof_map[0]])
                else:
                    uniaxial_set_trial_strain(mat_def, state, u[dof_map[3]] - u[dof_map[1]])
                uniaxial_states[state_index] = state
                var force = state.sig_t
                var k = state.tangent_t
                if dir == 1:
                    K[dof_map[0]][dof_map[0]] += k
                    K[dof_map[2]][dof_map[2]] += k
                    K[dof_map[0]][dof_map[2]] -= k
                    K[dof_map[2]][dof_map[0]] -= k
                    F_int[dof_map[0]] -= force
                    F_int[dof_map[2]] += force
                else:
                    K[dof_map[1]][dof_map[1]] += k
                    K[dof_map[3]][dof_map[3]] += k
                    K[dof_map[1]][dof_map[3]] -= k
                    K[dof_map[3]][dof_map[1]] -= k
                    F_int[dof_map[1]] -= force
                    F_int[dof_map[3]] += force
        elif elem_type == 8:
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
        elif elem_type == 6:
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
                var sum = 0.0
                for b in range(8):
                    var Bidx = dof_map[b]
                    var kval = k_global[a][b]
                    K[Aidx][Bidx] += kval
                    sum += kval * u[Bidx]
                F_int[Aidx] += sum
        elif elem_type == 7:
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
                var sum = 0.0
                for b in range(24):
                    var Bidx = dof_map[b]
                    var kval = k_global[a][b]
                    K[Aidx][Bidx] += kval
                    sum += kval * u[Bidx]
                F_int[Aidx] += sum
        else:
            abort("unsupported element type tag")
