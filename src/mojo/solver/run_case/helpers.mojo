from collections import Dict, List
from math import atan2, sqrt
from os import abort

from elements import (
    beam2d_basic_fixed_end_and_reactions,
    beam2d_element_load_global,
    beam2d_corotational_global_internal_force,
    beam2d_section_load_response,
    beam_column3d_fiber_global_tangent_and_internal,
    beam_column3d_fiber_section_response,
    beam_integration_validate_or_abort,
    beam_integration_xi_weight,
    beam2d_pdelta_global_stiffness,
    beam3d_section_load_response,
    beam3d_global_tangent_and_internal,
    beam_global_stiffness,
    beam_uniform_load_global_2d,
    disp_beam_column2d_global_tangent_and_internal,
    disp_beam_column3d_global_tangent_and_internal,
    force_beam_column2d_global_tangent_and_internal,
    force_beam_column3d_fiber_global_tangent_and_internal,
    force_beam_column3d_fiber_section_response,
    force_beam_column3d_global_tangent_and_internal,
    link_orientation_matrix,
)
from elements.beam3d import (
    _beam3d_rotation,
    _beam3d_transform_matrix,
    _beam3d_transform_u_global_to_local,
)
from elements.utils import (
    _beam2d_transform_force_local_to_global_in_place,
    _beam2d_transform_u_global_to_local,
    _ensure_zero_vector,
)
from materials import UniMaterialDef, UniMaterialState, uniaxial_set_trial_strain
from solver.dof import node_dof_index, require_dof_in_range
from solver.run_case.input_types import (
    ElementLoadInput,
    ElementInput,
    NodeInput,
    RecorderInput,
    SectionInput,
)
from sections import (
    FiberCell,
    FiberSection2dDef,
    FiberSection3dDef,
    fiber_section2d_set_trial_from_offset,
)
from tag_types import ElementTypeTag, GeomTransfTag


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


fn _aggregator_section2d_expected_state_count(sec: SectionInput) -> Int:
    if sec.base_section >= 0:
        abort("AggregatorSection2d with -section is not yet supported")
    var count = 0
    if sec.axial_material >= 0:
        count += 1
    if sec.flexural_material >= 0:
        count += 1
    if sec.moment_y_material >= 0 or sec.torsion_material >= 0:
        abort("AggregatorSection2d only supports P and Mz")
    if sec.shear_y_material >= 0 or sec.shear_z_material >= 0:
        abort("AggregatorSection2d shear responses are not yet supported")
    if count == 0:
        abort("AggregatorSection2d requires at least one uniaxial response")
    return count


fn aggregator_section2d_set_trial_from_offset(
    sec: SectionInput,
    uniaxial_defs: List[UniMaterialDef],
    uniaxial_state_defs: List[Int],
    mut uniaxial_states: List[UniMaterialState],
    elem_uniaxial_state_ids: List[Int],
    elem_offset: Int,
    elem_state_count: Int,
    delta_axial: Float64,
    delta_curv: Float64,
) raises -> (Float64, Float64, Float64, Float64, Float64):
    var expected = _aggregator_section2d_expected_state_count(sec)
    if elem_state_count != expected:
        abort("AggregatorSection2d state count mismatch")
    var axial_force = 0.0
    var moment_z = 0.0
    var k11 = 0.0
    var k12 = 0.0
    var k22 = 0.0
    var offset = elem_offset
    if sec.axial_material >= 0:
        if offset < 0 or offset >= len(elem_uniaxial_state_ids):
            abort("AggregatorSection2d axial state out of range")
        var state_id = elem_uniaxial_state_ids[offset]
        if state_id < 0 or state_id >= len(uniaxial_states):
            abort("AggregatorSection2d axial state id out of range")
        if state_id < 0 or state_id >= len(uniaxial_state_defs):
            abort("AggregatorSection2d axial state def out of range")
        var state = uniaxial_states[state_id]
        var def_index = uniaxial_state_defs[state_id]
        if def_index < 0 or def_index >= len(uniaxial_defs):
            abort("AggregatorSection2d axial material definition out of range")
        var mat_def = uniaxial_defs[def_index]
        uniaxial_set_trial_strain(mat_def, state, delta_axial)
        axial_force = state.sig_t
        k11 = state.tangent_t
        uniaxial_states[state_id] = state
        offset += 1
    if sec.flexural_material >= 0:
        if offset < 0 or offset >= len(elem_uniaxial_state_ids):
            abort("AggregatorSection2d flexural state out of range")
        var state_id = elem_uniaxial_state_ids[offset]
        if state_id < 0 or state_id >= len(uniaxial_states):
            abort("AggregatorSection2d flexural state id out of range")
        if state_id < 0 or state_id >= len(uniaxial_state_defs):
            abort("AggregatorSection2d flexural state def out of range")
        var state = uniaxial_states[state_id]
        var def_index = uniaxial_state_defs[state_id]
        if def_index < 0 or def_index >= len(uniaxial_defs):
            abort("AggregatorSection2d flexural material definition out of range")
        var mat_def = uniaxial_defs[def_index]
        uniaxial_set_trial_strain(mat_def, state, delta_curv)
        moment_z = state.sig_t
        k22 = state.tangent_t
        uniaxial_states[state_id] = state
    return (axial_force, moment_z, k11, k12, k22)


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
    for i in range(3):
        for j in range(3):
            tgl[i][j] = trans[i][j]
            tgl[i + 3][j + 3] = trans[i][j]
            tgl[i + 6][j + 6] = trans[i][j]
            tgl[i + 9][j + 9] = trans[i][j]
    return tgl^


fn _two_node_link_tlb(ndm: Int, ndf: Int, elem: ElementInput, length: Float64) -> List[List[Float64]]:
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


fn _beam_uniform_load_for_element_global(
    node1: NodeInput, node2: NodeInput, wy: Float64, wx: Float64
) -> List[Float64]:
    return beam_uniform_load_global_2d(node1.x, node1.y, node2.x, node2.y, wy, wx)


fn _beam2d_element_force_global(
    elem_index: Int,
    elem: ElementInput,
    nodes: List[NodeInput],
    sections_by_id: List[SectionInput],
    ndf: Int,
    u: List[Float64],
    element_loads: List[ElementLoadInput],
    elem_load_offsets: List[Int],
    elem_load_pool: List[Int],
    load_scale: Float64,
) raises -> List[Float64]:
    if ndf != 3:
        abort("elasticBeamColumn2d requires ndf=3")
    var i1 = elem.node_index_1
    var i2 = elem.node_index_2
    var node1 = nodes[i1]
    var node2 = nodes[i2]

    var sec_id = elem.section
    if sec_id >= len(sections_by_id):
        abort("section not found")
    var sec = sections_by_id[sec_id]
    if sec.id < 0:
        abort("section not found")
    if sec.type == "FiberSection2d":
        abort(
            "element_force for elasticBeamColumn2d with FiberSection2d requires "
            "forceBeamColumn2d or dispBeamColumn2d"
        )

    var E = sec.E
    var A = sec.A
    var I = sec.I

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

    var geom = elem.geom_tag
    var f_elem: List[Float64] = []
    if geom == GeomTransfTag.Corotational:
        f_elem = beam2d_corotational_global_internal_force(
            E,
            A,
            I,
            node1.x,
            node1.y,
            node2.x,
            node2.y,
            u_elem,
        )
    else:
        var k_global: List[List[Float64]] = []
        if geom == GeomTransfTag.Linear:
            k_global = beam_global_stiffness(
                E,
                A,
                I,
                node1.x,
                node1.y,
                node2.x,
                node2.y,
            )
        elif geom == GeomTransfTag.PDelta:
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
        else:
            abort("unsupported geomTransf: " + elem.geom_transf)

        f_elem.resize(6, 0.0)
        for i in range(6):
            var sum = 0.0
            for j in range(6):
                sum += k_global[i][j] * u_elem[j]
            f_elem[i] = sum

    var f_load = beam2d_element_load_global(
        element_loads,
        elem_load_offsets,
        elem_load_pool,
        elem_index,
        load_scale,
        node1.x,
        node1.y,
        node2.x,
        node2.y,
    )
    for i in range(6):
        f_elem[i] -= f_load[i]
    return f_elem^


fn _beam_column3d_element_force_global(
    elem_index: Int,
    elem: ElementInput,
    nodes: List[NodeInput],
    sections_by_id: List[SectionInput],
    ndf: Int,
    u: List[Float64],
    fiber_section3d_defs: List[FiberSection3dDef],
    fiber_section3d_cells: List[FiberCell],
    fiber_section3d_index_by_id: List[Int],
    uniaxial_defs: List[UniMaterialDef],
    mut uniaxial_states: List[UniMaterialState],
    elem_uniaxial_offsets: List[Int],
    elem_uniaxial_counts: List[Int],
    elem_uniaxial_state_ids: List[Int],
    force_basic_offsets: List[Int],
    force_basic_counts: List[Int],
    mut force_basic_q: List[Float64],
    element_loads: List[ElementLoadInput],
    elem_load_offsets: List[Int],
    elem_load_pool: List[Int],
    load_scale: Float64,
) raises -> List[Float64]:
    var beam_col_type = elem.type
    if ndf != 6:
        abort(beam_col_type + " requires ndf=6")
    var geom = elem.geom_tag
    if (
        geom != GeomTransfTag.Linear
        and geom != GeomTransfTag.PDelta
        and geom != GeomTransfTag.Corotational
    ):
        abort(beam_col_type + " supports geomTransf Linear, PDelta, or Corotational")
    if (
        elem.type_tag == ElementTypeTag.ForceBeamColumn3d
        or elem.type_tag == ElementTypeTag.DispBeamColumn3d
    ):
        beam_integration_validate_or_abort(
            beam_col_type, elem.integration, elem.num_int_pts
        )

    var i1 = elem.node_index_1
    var i2 = elem.node_index_2
    var node1 = nodes[i1]
    var node2 = nodes[i2]

    var sec_id = elem.section
    if sec_id >= len(sections_by_id):
        abort("section not found")
    var sec = sections_by_id[sec_id]
    if sec.id < 0:
        abort("section not found")

    var dof_map = [
        node_dof_index(i1, 1, ndf),
        node_dof_index(i1, 2, ndf),
        node_dof_index(i1, 3, ndf),
        node_dof_index(i1, 4, ndf),
        node_dof_index(i1, 5, ndf),
        node_dof_index(i1, 6, ndf),
        node_dof_index(i2, 1, ndf),
        node_dof_index(i2, 2, ndf),
        node_dof_index(i2, 3, ndf),
        node_dof_index(i2, 4, ndf),
        node_dof_index(i2, 5, ndf),
        node_dof_index(i2, 6, ndf),
    ]
    var u_elem: List[Float64] = []
    u_elem.resize(12, 0.0)
    for i in range(12):
        u_elem[i] = u[dof_map[i]]

    var k_global: List[List[Float64]] = []
    var f_elem: List[Float64] = []
    if sec.type == "ElasticSection3d":
        if elem.type_tag == ElementTypeTag.ForceBeamColumn3d:
            force_beam_column3d_global_tangent_and_internal(
                elem_index,
                node1.x,
                node1.y,
                node1.z,
                node2.x,
                node2.y,
                node2.z,
                u_elem,
                elem.geom_transf,
                element_loads,
                elem_load_offsets,
                elem_load_pool,
                load_scale,
                sec.E,
                sec.A,
                sec.Iy,
                sec.Iz,
                sec.G,
                sec.J,
                k_global,
                f_elem,
            )
        elif elem.type_tag == ElementTypeTag.DispBeamColumn3d:
            disp_beam_column3d_global_tangent_and_internal(
                elem_index,
                node1.x,
                node1.y,
                node1.z,
                node2.x,
                node2.y,
                node2.z,
                u_elem,
                elem.geom_transf,
                element_loads,
                elem_load_offsets,
                elem_load_pool,
                load_scale,
                sec.E,
                sec.A,
                sec.Iy,
                sec.Iz,
                sec.G,
                sec.J,
                k_global,
                f_elem,
            )
        else:
            force_beam_column3d_global_tangent_and_internal(
                elem_index,
                node1.x,
                node1.y,
                node1.z,
                node2.x,
                node2.y,
                node2.z,
                u_elem,
                elem.geom_transf,
                element_loads,
                elem_load_offsets,
                elem_load_pool,
                load_scale,
                sec.E,
                sec.A,
                sec.Iy,
                sec.Iz,
                sec.G,
                sec.J,
                k_global,
                f_elem,
            )
    elif sec.type == "FiberSection3d":
        var sec_index = fiber_section3d_index_by_id[sec_id]
        if sec_index < 0 or sec_index >= len(fiber_section3d_defs):
            abort(beam_col_type + " fiber section not found")
        if elem.type_tag == ElementTypeTag.ForceBeamColumn3d:
            force_beam_column3d_fiber_global_tangent_and_internal(
                elem_index,
                node1.x,
                node1.y,
                node1.z,
                node2.x,
                node2.y,
                node2.z,
                u_elem,
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
                elem_uniaxial_offsets[elem_index],
                elem_uniaxial_counts[elem_index],
                elem.integration,
                elem.num_int_pts,
                sec.G,
                sec.J,
                force_basic_q,
                force_basic_offsets[elem_index],
                force_basic_counts[elem_index],
                k_global,
                f_elem,
            )
        else:
            beam_column3d_fiber_global_tangent_and_internal(
                elem_index,
                node1.x,
                node1.y,
                node1.z,
                node2.x,
                node2.y,
                node2.z,
                u_elem,
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
                elem_uniaxial_offsets[elem_index],
                elem_uniaxial_counts[elem_index],
                elem.integration,
                elem.num_int_pts,
                sec.G,
                sec.J,
                k_global,
                f_elem,
            )
    else:
        abort(beam_col_type + " requires ElasticSection3d or FiberSection3d")
    return f_elem^


fn _truss_element_force_global(
    elem_index: Int,
    elem: ElementInput,
    nodes: List[NodeInput],
    ndf: Int,
    u: List[Float64],
    uniaxial_defs: List[UniMaterialDef],
    uniaxial_state_defs: List[Int],
    mut uniaxial_states: List[UniMaterialState],
    elem_uniaxial_offsets: List[Int],
    elem_uniaxial_counts: List[Int],
    elem_uniaxial_state_ids: List[Int],
) raises -> List[Float64]:
    if ndf != 2 and ndf != 3:
        abort("truss element_force requires ndf=2 or ndf=3")
    var i1 = elem.node_index_1
    var i2 = elem.node_index_2
    var node1 = nodes[i1]
    var node2 = nodes[i2]

    var offset = elem_uniaxial_offsets[elem_index]
    var count = elem_uniaxial_counts[elem_index]
    if count != 1:
        abort("truss requires one uniaxial material")
    var state_index = elem_uniaxial_state_ids[offset]
    var def_index = uniaxial_state_defs[state_index]
    var mat_def = uniaxial_defs[def_index]
    ref state = uniaxial_states[state_index]
    var A = elem.area

    if ndf == 2:
        var dx = node2.x - node1.x
        var dy = node2.y - node1.y
        var L = sqrt(dx * dx + dy * dy)
        if L == 0.0:
            abort("zero-length element")
        var c = dx / L
        var s = dy / L
        var dof_map = [
            node_dof_index(i1, 1, ndf),
            node_dof_index(i1, 2, ndf),
            node_dof_index(i2, 1, ndf),
            node_dof_index(i2, 2, ndf),
        ]
        var du = (u[dof_map[2]] - u[dof_map[0]]) * c + (
            u[dof_map[3]] - u[dof_map[1]]
        ) * s
        var eps = du / L
        uniaxial_set_trial_strain(mat_def, state, eps)
        var N = state.sig_t * A
        return [-N * c, -N * s, N * c, N * s]

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
        node_dof_index(i1, 1, ndf),
        node_dof_index(i1, 2, ndf),
        node_dof_index(i1, 3, ndf),
        node_dof_index(i2, 1, ndf),
        node_dof_index(i2, 2, ndf),
        node_dof_index(i2, 3, ndf),
    ]
    var du = (u[dof_map[3]] - u[dof_map[0]]) * l + (
        u[dof_map[4]] - u[dof_map[1]]
    ) * m + (u[dof_map[5]] - u[dof_map[2]]) * n
    var eps = du / L
    uniaxial_set_trial_strain(mat_def, state, eps)
    var N = state.sig_t * A
    return [-N * l, -N * m, -N * n, N * l, N * m, N * n]


fn _zero_length_element_force_global(
    elem_index: Int,
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
) raises -> List[Float64]:
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
    var f_elem: List[Float64] = []
    f_elem.resize(elem.dof_count, 0.0)
    var offset = elem_uniaxial_offsets[elem_index]
    var count = elem_uniaxial_counts[elem_index]
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
        for i in range(elem.dof_count):
            f_elem[i] += row[i] * state.sig_t
    return f_elem^


fn _zero_length_basic_force_for_recorder(
    elem_index: Int,
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
) raises -> List[Float64]:
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
    var q_basic: List[Float64] = []
    var count = elem_uniaxial_counts[elem_index]
    q_basic.resize(count, 0.0)
    var offset = elem_uniaxial_offsets[elem_index]
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
        q_basic[m] = state.sig_t
    return q_basic^


fn _zero_length_deformation_for_recorder(
    elem_index: Int,
    elem: ElementInput,
    nodes: List[NodeInput],
    ndf: Int,
    ndm: Int,
    u: List[Float64],
) raises -> List[Float64]:
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
    var deformation: List[Float64] = []
    deformation.resize(elem.dir_count, 0.0)
    for m in range(elem.dir_count):
        var row = _zero_length_row(_elem_dir(elem, m), ndm, ndf, trans)
        for i in range(elem.dof_count):
            deformation[m] += row[i] * ug[i]
    return deformation^


fn _two_node_link_element_force_global(
    elem_index: Int,
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
) raises -> List[Float64]:
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
    var q_basic: List[Float64] = []
    q_basic.resize(elem.dir_count, 0.0)
    var offset = elem_uniaxial_offsets[elem_index]
    for m in range(elem.dir_count):
        var ub = 0.0
        for j in range(elem.dof_count):
            ub += tlb[m][j] * ul[j]
        var state_index = elem_uniaxial_state_ids[offset + m]
        var def_index = uniaxial_state_defs[state_index]
        var mat_def = uniaxial_defs[def_index]
        ref state = uniaxial_states[state_index]
        uniaxial_set_trial_strain(mat_def, state, ub)
        q_basic[m] = state.sig_t
    var p_local: List[Float64] = []
    p_local.resize(elem.dof_count, 0.0)
    for i in range(elem.dof_count):
        for m in range(elem.dir_count):
            p_local[i] += tlb[m][i] * q_basic[m]
    _two_node_link_add_pdelta_forces(ndm, ndf, elem, length, ul, q_basic, p_local)
    var p_global: List[Float64] = []
    p_global.resize(elem.dof_count, 0.0)
    for i in range(elem.dof_count):
        for j in range(elem.dof_count):
            p_global[i] += tgl[j][i] * p_local[j]
    return p_global^


fn _two_node_link_local_force_for_recorder(
    elem_index: Int,
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
) raises -> List[Float64]:
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
    var q_basic: List[Float64] = []
    q_basic.resize(elem.dir_count, 0.0)
    var offset = elem_uniaxial_offsets[elem_index]
    for m in range(elem.dir_count):
        var ub = 0.0
        for j in range(elem.dof_count):
            ub += tlb[m][j] * ul[j]
        var state_index = elem_uniaxial_state_ids[offset + m]
        var def_index = uniaxial_state_defs[state_index]
        var mat_def = uniaxial_defs[def_index]
        ref state = uniaxial_states[state_index]
        uniaxial_set_trial_strain(mat_def, state, ub)
        q_basic[m] = state.sig_t
    var p_local: List[Float64] = []
    p_local.resize(elem.dof_count, 0.0)
    for i in range(elem.dof_count):
        for m in range(elem.dir_count):
            p_local[i] += tlb[m][i] * q_basic[m]
    _two_node_link_add_pdelta_forces(ndm, ndf, elem, length, ul, q_basic, p_local)
    return p_local^


fn _two_node_link_basic_force_for_recorder(
    elem_index: Int,
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
) raises -> List[Float64]:
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
    var q_basic: List[Float64] = []
    q_basic.resize(elem.dir_count, 0.0)
    var offset = elem_uniaxial_offsets[elem_index]
    for m in range(elem.dir_count):
        var ub = 0.0
        for j in range(elem.dof_count):
            ub += tlb[m][j] * ul[j]
        var state_index = elem_uniaxial_state_ids[offset + m]
        var def_index = uniaxial_state_defs[state_index]
        var mat_def = uniaxial_defs[def_index]
        ref state = uniaxial_states[state_index]
        uniaxial_set_trial_strain(mat_def, state, ub)
        q_basic[m] = state.sig_t
    return q_basic^


fn _two_node_link_deformation_for_recorder(
    elem: ElementInput,
    nodes: List[NodeInput],
    ndf: Int,
    ndm: Int,
    u: List[Float64],
) raises -> List[Float64]:
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
    var ub: List[Float64] = []
    ub.resize(elem.dir_count, 0.0)
    for m in range(elem.dir_count):
        for j in range(elem.dof_count):
            ub[m] += tlb[m][j] * ul[j]
    return ub^


fn _beam2d_force_global_to_local(
    node1: NodeInput, node2: NodeInput, f_global: List[Float64]
) raises -> List[Float64]:
    var dx = node2.x - node1.x
    var dy = node2.y - node1.y
    var L = sqrt(dx * dx + dy * dy)
    if L == 0.0:
        abort("zero-length element")
    var c = dx / L
    var s = dy / L
    var f_local: List[Float64] = []
    f_local.resize(6, 0.0)
    _beam2d_transform_u_global_to_local(c, s, f_global, f_local)
    return f_local^


fn _beam3d_force_global_to_local(
    node1: NodeInput, node2: NodeInput, f_global: List[Float64]
) -> List[Float64]:
    var R = _beam3d_rotation(node1.x, node1.y, node1.z, node2.x, node2.y, node2.z)
    var T = _beam3d_transform_matrix(R)
    return _beam3d_transform_u_global_to_local(T, f_global)


fn _force_beam_column2d_element_force_global(
    elem_index: Int,
    elem: ElementInput,
    nodes: List[NodeInput],
    sections_by_id: List[SectionInput],
    ndf: Int,
    u: List[Float64],
    fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    uniaxial_defs: List[UniMaterialDef],
    mut uniaxial_states: List[UniMaterialState],
    elem_uniaxial_offsets: List[Int],
    elem_uniaxial_counts: List[Int],
    elem_uniaxial_state_ids: List[Int],
    force_basic_offsets: List[Int],
    force_basic_counts: List[Int],
    mut force_basic_q: List[Float64],
) raises -> List[Float64]:
    var empty_element_loads: List[ElementLoadInput] = []
    var empty_elem_load_offsets: List[Int] = []
    var empty_elem_load_pool: List[Int] = []
    return _force_beam_column2d_element_force_global(
        elem_index,
        elem,
        nodes,
        sections_by_id,
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
        force_basic_offsets,
        force_basic_counts,
        force_basic_q,
        empty_element_loads,
        empty_elem_load_offsets,
        empty_elem_load_pool,
        0.0,
    )


fn _force_beam_column2d_element_force_global(
    elem_index: Int,
    elem: ElementInput,
    nodes: List[NodeInput],
    sections_by_id: List[SectionInput],
    ndf: Int,
    u: List[Float64],
    fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    uniaxial_defs: List[UniMaterialDef],
    mut uniaxial_states: List[UniMaterialState],
    elem_uniaxial_offsets: List[Int],
    elem_uniaxial_counts: List[Int],
    elem_uniaxial_state_ids: List[Int],
    force_basic_offsets: List[Int],
    force_basic_counts: List[Int],
    mut force_basic_q: List[Float64],
    element_loads: List[ElementLoadInput],
    elem_load_offsets: List[Int],
    elem_load_pool: List[Int],
    load_scale: Float64,
) raises -> List[Float64]:
    var beam_col_type = elem.type
    if ndf != 3:
        abort(beam_col_type + " requires ndf=3")
    var geom = elem.geom_tag
    if geom != GeomTransfTag.Linear and geom != GeomTransfTag.PDelta:
        abort(beam_col_type + " supports geomTransf Linear or PDelta")
    var integration = elem.integration
    var num_int_pts = elem.num_int_pts
    beam_integration_validate_or_abort(beam_col_type, integration, num_int_pts)

    var i1 = elem.node_index_1
    var i2 = elem.node_index_2
    var node1 = nodes[i1]
    var node2 = nodes[i2]
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
    var sec = sections_by_id[elem.section]

    if sec.type == "ElasticSection2d":
        var k_global: List[List[Float64]] = []
        if geom == GeomTransfTag.Linear:
            k_global = beam_global_stiffness(
                sec.E,
                sec.A,
                sec.I,
                node1.x,
                node1.y,
                node2.x,
                node2.y,
            )
        elif geom == GeomTransfTag.PDelta:
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
            abort(beam_col_type + " supports geomTransf Linear or PDelta")
        var f_global_elastic: List[Float64] = []
        f_global_elastic.resize(6, 0.0)
        for a in range(6):
            var sum = 0.0
            for b in range(6):
                sum += k_global[a][b] * u_elem[b]
            f_global_elastic[a] = sum
        var f_load = beam2d_element_load_global(
            element_loads,
            elem_load_offsets,
            elem_load_pool,
            elem_index,
            load_scale,
            node1.x,
            node1.y,
            node2.x,
            node2.y,
        )
        for a in range(6):
            f_global_elastic[a] -= f_load[a]
        if elem.uniform_load_wy != 0.0 or elem.uniform_load_wx != 0.0:
            var f_embedded = _beam_uniform_load_for_element_global(
                node1, node2, elem.uniform_load_wy, elem.uniform_load_wx
            )
            for a in range(6):
                f_global_elastic[a] -= f_embedded[a]
        var q_offset = force_basic_offsets[elem_index]
        if q_offset >= 0 and q_offset + 2 < len(force_basic_q):
            var f_local = _beam2d_force_global_to_local(node1, node2, f_global_elastic)
            force_basic_q[q_offset] = f_local[0]
            force_basic_q[q_offset + 1] = f_local[2]
            force_basic_q[q_offset + 2] = f_local[5]
        return f_global_elastic^

    var sec_id = elem.section
    if sec_id >= len(fiber_section_index_by_id):
        abort(beam_col_type + " section not found")
    var sec_index = fiber_section_index_by_id[sec_id]
    if sec_index < 0 or sec_index >= len(fiber_section_defs):
        abort(beam_col_type + " fiber section not found")
    var sec_def = fiber_section_defs[sec_index]

    var elem_offset = elem_uniaxial_offsets[elem_index]
    var elem_state_count = elem_uniaxial_counts[elem_index]
    var k_dummy: List[List[Float64]] = []
    var f_global: List[Float64] = []
    force_beam_column2d_global_tangent_and_internal(
        elem_index,
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
        integration,
        num_int_pts,
        force_basic_q,
        force_basic_offsets[elem_index],
        force_basic_counts[elem_index],
        k_dummy,
        f_global,
    )
    return f_global^


fn _force_beam_column2d_force_global_from_basic_state(
    elem_index: Int,
    elem: ElementInput,
    nodes: List[NodeInput],
    ndf: Int,
    u: List[Float64],
    element_loads: List[ElementLoadInput],
    elem_load_offsets: List[Int],
    elem_load_pool: List[Int],
    load_scale: Float64,
    force_basic_offsets: List[Int],
    force_basic_counts: List[Int],
    force_basic_q: List[Float64],
) raises -> List[Float64]:
    if ndf != 3:
        abort(elem.type + " requires ndf=3")
    if elem_index < 0 or elem_index >= len(force_basic_offsets):
        abort("forceBeamColumn2d basic state mapping missing")
    if elem_index >= len(force_basic_counts):
        abort("forceBeamColumn2d basic state count missing")

    var q_offset = force_basic_offsets[elem_index]
    if force_basic_counts[elem_index] < 3:
        abort("forceBeamColumn2d basic force state count mismatch")
    if q_offset < 0 or q_offset + 2 >= len(force_basic_q):
        abort("forceBeamColumn2d basic force state out of range")

    var i1 = elem.node_index_1
    var i2 = elem.node_index_2
    var node1 = nodes[i1]
    var node2 = nodes[i2]
    var dx = node2.x - node1.x
    var dy = node2.y - node1.y
    var L = sqrt(dx * dx + dy * dy)
    if L == 0.0:
        abort("zero-length element")
    var c = dx / L
    var s = dy / L
    var inv_L = 1.0 / L

    var q0 = force_basic_q[q_offset]
    var q1 = force_basic_q[q_offset + 1]
    var q2 = force_basic_q[q_offset + 2]
    var fixed_end = beam2d_basic_fixed_end_and_reactions(
        element_loads,
        elem_load_offsets,
        elem_load_pool,
        elem_index,
        load_scale,
        L,
    )
    var shear = (q1 + q2) * inv_L

    var f_local: List[Float64] = []
    f_local.resize(6, 0.0)
    f_local[0] = -q0 + fixed_end[3]
    f_local[1] = shear + fixed_end[4]
    f_local[2] = q1
    f_local[3] = q0
    f_local[4] = -shear + fixed_end[5]
    f_local[5] = q2

    if elem.geom_tag == GeomTransfTag.PDelta:
        var dof_map = [
            node_dof_index(i1, 1, ndf),
            node_dof_index(i1, 2, ndf),
            node_dof_index(i1, 3, ndf),
            node_dof_index(i2, 1, ndf),
            node_dof_index(i2, 2, ndf),
            node_dof_index(i2, 3, ndf),
        ]
        var u_local: List[Float64] = []
        u_local.resize(6, 0.0)
        u_local[0] = c * u[dof_map[0]] + s * u[dof_map[1]]
        u_local[1] = -s * u[dof_map[0]] + c * u[dof_map[1]]
        u_local[2] = u[dof_map[2]]
        u_local[3] = c * u[dof_map[3]] + s * u[dof_map[4]]
        u_local[4] = -s * u[dof_map[3]] + c * u[dof_map[4]]
        u_local[5] = u[dof_map[5]]
        var pdelta_shear = (u_local[1] - u_local[4]) * q0 * inv_L
        f_local[1] += pdelta_shear
        f_local[4] -= pdelta_shear
    elif elem.geom_tag == GeomTransfTag.Corotational:
        var dof_map = [
            node_dof_index(i1, 1, ndf),
            node_dof_index(i1, 2, ndf),
            node_dof_index(i1, 3, ndf),
            node_dof_index(i2, 1, ndf),
            node_dof_index(i2, 2, ndf),
            node_dof_index(i2, 3, ndf),
        ]
        var u_local: List[Float64] = []
        u_local.resize(6, 0.0)
        u_local[0] = c * u[dof_map[0]] + s * u[dof_map[1]]
        u_local[1] = -s * u[dof_map[0]] + c * u[dof_map[1]]
        u_local[2] = u[dof_map[2]]
        u_local[3] = c * u[dof_map[3]] + s * u[dof_map[4]]
        u_local[4] = -s * u[dof_map[3]] + c * u[dof_map[4]]
        u_local[5] = u[dof_map[5]]
        var dulx = u_local[3] - u_local[0]
        var duly = u_local[4] - u_local[1]
        var Lx = L + dulx
        var Ly = duly
        var Ln = sqrt(Lx * Lx + Ly * Ly)
        if Ln == 0.0:
            abort("zero-length element")
        var cos_alpha = Lx / Ln
        var sin_alpha = Ly / Ln
        var tbl0 = [
            -cos_alpha,
            -sin_alpha,
            0.0,
            cos_alpha,
            sin_alpha,
            0.0,
        ]
        var tbl1 = [
            -sin_alpha / Ln,
            cos_alpha / Ln,
            1.0,
            sin_alpha / Ln,
            -cos_alpha / Ln,
            0.0,
        ]
        var tbl2 = [
            -sin_alpha / Ln,
            cos_alpha / Ln,
            0.0,
            sin_alpha / Ln,
            -cos_alpha / Ln,
            1.0,
        ]
        _ensure_zero_vector(f_local, 6)
        for idx in range(6):
            f_local[idx] = tbl0[idx] * q0 + tbl1[idx] * q1 + tbl2[idx] * q2
        f_local[0] += fixed_end[3]
        f_local[1] += fixed_end[4]
        f_local[4] += fixed_end[5]
    elif elem.geom_tag != GeomTransfTag.Linear:
        abort(
            elem.type + " supports geomTransf Linear, PDelta, or Corotational"
        )

    _beam2d_transform_force_local_to_global_in_place(f_local, c, s)
    return f_local^


fn _disp_beam_column2d_element_force_global(
    elem_index: Int,
    elem: ElementInput,
    nodes: List[NodeInput],
    sections_by_id: List[SectionInput],
    ndf: Int,
    u: List[Float64],
    fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    uniaxial_defs: List[UniMaterialDef],
    mut uniaxial_states: List[UniMaterialState],
    elem_uniaxial_offsets: List[Int],
    elem_uniaxial_counts: List[Int],
    elem_uniaxial_state_ids: List[Int],
) raises -> List[Float64]:
    var empty_element_loads: List[ElementLoadInput] = []
    var empty_elem_load_offsets: List[Int] = []
    var empty_elem_load_pool: List[Int] = []
    return _disp_beam_column2d_element_force_global(
        elem_index,
        elem,
        nodes,
        sections_by_id,
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
        empty_element_loads,
        empty_elem_load_offsets,
        empty_elem_load_pool,
        0.0,
    )


fn _disp_beam_column2d_element_force_global(
    elem_index: Int,
    elem: ElementInput,
    nodes: List[NodeInput],
    sections_by_id: List[SectionInput],
    ndf: Int,
    u: List[Float64],
    fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    uniaxial_defs: List[UniMaterialDef],
    mut uniaxial_states: List[UniMaterialState],
    elem_uniaxial_offsets: List[Int],
    elem_uniaxial_counts: List[Int],
    elem_uniaxial_state_ids: List[Int],
    element_loads: List[ElementLoadInput],
    elem_load_offsets: List[Int],
    elem_load_pool: List[Int],
    load_scale: Float64,
) raises -> List[Float64]:
    var beam_col_type = elem.type
    if ndf != 3:
        abort(beam_col_type + " requires ndf=3")
    var geom = elem.geom_tag
    if geom != GeomTransfTag.Linear and geom != GeomTransfTag.PDelta:
        abort(beam_col_type + " supports geomTransf Linear or PDelta")
    var integration = elem.integration
    var num_int_pts = elem.num_int_pts
    beam_integration_validate_or_abort(beam_col_type, integration, num_int_pts)

    var i1 = elem.node_index_1
    var i2 = elem.node_index_2
    var node1 = nodes[i1]
    var node2 = nodes[i2]
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
    var sec = sections_by_id[elem.section]

    if sec.type == "ElasticSection2d":
        var k_global: List[List[Float64]] = []
        if geom == GeomTransfTag.Linear:
            k_global = beam_global_stiffness(
                sec.E,
                sec.A,
                sec.I,
                node1.x,
                node1.y,
                node2.x,
                node2.y,
            )
        elif geom == GeomTransfTag.PDelta:
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
            abort(beam_col_type + " supports geomTransf Linear or PDelta")
        var f_global_elastic: List[Float64] = []
        f_global_elastic.resize(6, 0.0)
        for a in range(6):
            var sum = 0.0
            for b in range(6):
                sum += k_global[a][b] * u_elem[b]
            f_global_elastic[a] = sum
        var f_load = beam2d_element_load_global(
            element_loads,
            elem_load_offsets,
            elem_load_pool,
            elem_index,
            load_scale,
            node1.x,
            node1.y,
            node2.x,
            node2.y,
        )
        for a in range(6):
            f_global_elastic[a] -= f_load[a]
        if elem.uniform_load_wy != 0.0 or elem.uniform_load_wx != 0.0:
            var f_load = _beam_uniform_load_for_element_global(
                node1, node2, elem.uniform_load_wy, elem.uniform_load_wx
            )
            for a in range(6):
                f_global_elastic[a] -= f_load[a]
        return f_global_elastic^

    var sec_id = elem.section
    if sec_id >= len(fiber_section_index_by_id):
        abort(beam_col_type + " section not found")
    var sec_index = fiber_section_index_by_id[sec_id]
    if sec_index < 0 or sec_index >= len(fiber_section_defs):
        abort(beam_col_type + " fiber section not found")
    var sec_def = fiber_section_defs[sec_index]

    var elem_offset = elem_uniaxial_offsets[elem_index]
    var elem_state_count = elem_uniaxial_counts[elem_index]
    var k_dummy: List[List[Float64]] = []
    var f_global: List[Float64] = []
    disp_beam_column2d_global_tangent_and_internal(
        elem_index,
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
        integration,
        num_int_pts,
        k_dummy,
        f_global,
    )
    return f_global^


fn _beam_integration_xi_for_section(integration: String, num_int_pts: Int, ip: Int) -> Float64:
    return beam_integration_xi_weight(integration, num_int_pts, ip)[0]


fn _sync_force_beam_column2d_committed_basic_states(
    typed_nodes: List[NodeInput],
    typed_elements: List[ElementInput],
    ndf: Int,
    u: List[Float64],
    force_basic_offsets: List[Int],
    force_basic_counts: List[Int],
    mut force_basic_q: List[Float64],
):
    if ndf != 3:
        return
    for e in range(len(typed_elements)):
        var elem = typed_elements[e]
        if elem.type_tag != ElementTypeTag.ForceBeamColumn2d:
            continue
        var num_int_pts = elem.num_int_pts
        var required_count = 3 + 2 * num_int_pts + 3
        if force_basic_counts[e] < required_count:
            continue
        var i1 = elem.node_index_1
        var i2 = elem.node_index_2
        var node1 = typed_nodes[i1]
        var node2 = typed_nodes[i2]
        var dx = node2.x - node1.x
        var dy = node2.y - node1.y
        var L = sqrt(dx * dx + dy * dy)
        if L == 0.0:
            abort("zero-length element")
        var c = dx / L
        var s = dy / L
        var dof_map = [
            node_dof_index(i1, 1, ndf),
            node_dof_index(i1, 2, ndf),
            node_dof_index(i1, 3, ndf),
            node_dof_index(i2, 1, ndf),
            node_dof_index(i2, 2, ndf),
            node_dof_index(i2, 3, ndf),
        ]
        var u_local: List[Float64] = []
        u_local.resize(6, 0.0)
        u_local[0] = c * u[dof_map[0]] + s * u[dof_map[1]]
        u_local[1] = -s * u[dof_map[0]] + c * u[dof_map[1]]
        u_local[2] = u[dof_map[2]]
        u_local[3] = c * u[dof_map[3]] + s * u[dof_map[4]]
        u_local[4] = -s * u[dof_map[3]] + c * u[dof_map[4]]
        u_local[5] = u[dof_map[5]]

        var chord_rotation = (u_local[4] - u_local[1]) / L
        var basic_state_offset = force_basic_offsets[e] + 3 + 2 * num_int_pts
        force_basic_q[basic_state_offset] = u_local[3] - u_local[0]
        force_basic_q[basic_state_offset + 1] = u_local[2] - chord_rotation
        force_basic_q[basic_state_offset + 2] = u_local[5] - chord_rotation


fn _fiber_section_force_from_offset(
    sec_def: FiberSection2dDef,
    fibers: List[FiberCell],
    uniaxial_defs: List[UniMaterialDef],
    mut uniaxial_states: List[UniMaterialState],
    state_offset: Int,
    state_count: Int,
    eps0: Float64,
    kappa: Float64,
) raises -> List[Float64]:
    var resp = fiber_section2d_set_trial_from_offset(
        sec_def,
        uniaxial_states,
        state_offset,
        state_count,
        eps0,
        kappa,
    )
    return [resp.axial_force, resp.moment_z]


fn _force_beam_column2d_section_response(
    elem_index: Int,
    elem: ElementInput,
    section_no: Int,
    u: List[Float64],
    element_loads: List[ElementLoadInput],
    elem_load_offsets: List[Int],
    elem_load_pool: List[Int],
    load_scale: Float64,
    nodes: List[NodeInput],
    sections_by_id: List[SectionInput],
    fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    uniaxial_defs: List[UniMaterialDef],
    mut uniaxial_states: List[UniMaterialState],
    elem_uniaxial_offsets: List[Int],
    elem_uniaxial_counts: List[Int],
    elem_uniaxial_state_ids: List[Int],
    force_basic_offsets: List[Int],
    force_basic_counts: List[Int],
    mut force_basic_q: List[Float64],
    want_deformation: Bool,
) raises -> List[Float64]:
    # Generic recorder path must refresh from the current displacement state
    # because linear analyses do not keep the force-beam cache synchronized
    # with the post-solve displacement vector.
    _ = _force_beam_column2d_element_force_global(
        elem_index,
        elem,
        nodes,
        sections_by_id,
        3,
        u,
        fiber_section_defs,
        fiber_section_cells,
        fiber_section_index_by_id,
        uniaxial_defs,
        uniaxial_states,
        elem_uniaxial_offsets,
        elem_uniaxial_counts,
        elem_uniaxial_state_ids,
        force_basic_offsets,
        force_basic_counts,
        force_basic_q,
        element_loads,
        elem_load_offsets,
        elem_load_pool,
        load_scale,
    )
    return _force_beam_column2d_section_response_from_basic_state(
        elem_index,
        elem,
        section_no,
        element_loads,
        elem_load_offsets,
        elem_load_pool,
        load_scale,
        nodes,
        sections_by_id,
        force_basic_offsets,
        force_basic_counts,
        force_basic_q,
        want_deformation,
    )


fn _force_beam_column2d_section_response_from_basic_state(
    elem_index: Int,
    elem: ElementInput,
    section_no: Int,
    element_loads: List[ElementLoadInput],
    elem_load_offsets: List[Int],
    elem_load_pool: List[Int],
    load_scale: Float64,
    nodes: List[NodeInput],
    sections_by_id: List[SectionInput],
    force_basic_offsets: List[Int],
    force_basic_counts: List[Int],
    force_basic_q: List[Float64],
    want_deformation: Bool,
) raises -> List[Float64]:
    var num_int_pts = elem.num_int_pts
    if section_no < 1 or section_no > num_int_pts:
        abort("section recorder section index out of range")
    var ip = section_no - 1
    var xi = _beam_integration_xi_for_section(elem.integration, num_int_pts, ip)
    var i1 = elem.node_index_1
    var i2 = elem.node_index_2
    var node1 = nodes[i1]
    var node2 = nodes[i2]
    var dx = node2.x - node1.x
    var dy = node2.y - node1.y
    var L = sqrt(dx * dx + dy * dy)
    if L == 0.0:
        abort("zero-length element")
    if elem_index < 0 or elem_index >= len(force_basic_offsets):
        abort("section recorder forceBeamColumn2d basic state mapping missing")
    if elem_index >= len(force_basic_counts):
        abort("section recorder forceBeamColumn2d basic state count missing")
    var q_offset = force_basic_offsets[elem_index]
    var q_count = force_basic_counts[elem_index]
    if q_count < 3 or q_offset < 0 or q_offset + 2 >= len(force_basic_q):
        abort("section recorder forceBeamColumn2d basic state missing")
    var q0 = force_basic_q[q_offset]
    var q1 = force_basic_q[q_offset + 1]
    var q2 = force_basic_q[q_offset + 2]
    var load_response = beam2d_section_load_response(
        element_loads,
        elem_load_offsets,
        elem_load_pool,
        elem_index,
        load_scale,
        xi * L,
        L,
    )
    var axial_force = q0 + load_response[0]
    var moment_z = (xi - 1.0) * q1 + xi * q2 + load_response[1]

    var sec = sections_by_id[elem.section]
    var eps0 = 0.0
    var kappa = 0.0
    var has_predictor = q_count >= 3 + 2 * num_int_pts
    if has_predictor:
        eps0 = force_basic_q[q_offset + 3 + ip]
        kappa = force_basic_q[q_offset + 3 + num_int_pts + ip]
    elif sec.type == "ElasticSection2d":
        if sec.E <= 0.0 or sec.A <= 0.0 or sec.I <= 0.0:
            abort("section recorder ElasticSection2d requires positive E/A/I")
        eps0 = axial_force / (sec.E * sec.A)
        kappa = moment_z / (sec.E * sec.I)

    if want_deformation:
        return [eps0, kappa]
    if sec.type == "ElasticSection2d":
        axial_force = -axial_force
    return [axial_force, moment_z]


fn _disp_beam_column2d_section_response(
    elem_index: Int,
    elem: ElementInput,
    section_no: Int,
    ndf: Int,
    u: List[Float64],
    element_loads: List[ElementLoadInput],
    elem_load_offsets: List[Int],
    elem_load_pool: List[Int],
    load_scale: Float64,
    nodes: List[NodeInput],
    sections_by_id: List[SectionInput],
    fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    uniaxial_defs: List[UniMaterialDef],
    mut uniaxial_states: List[UniMaterialState],
    elem_uniaxial_offsets: List[Int],
    elem_uniaxial_counts: List[Int],
    elem_uniaxial_state_ids: List[Int],
    want_deformation: Bool,
) raises -> List[Float64]:
    if ndf != 3:
        abort("dispBeamColumn2d section recorder requires ndf=3")
    var num_int_pts = elem.num_int_pts
    if section_no < 1 or section_no > num_int_pts:
        abort("section recorder section index out of range")
    var ip = section_no - 1
    var xi = _beam_integration_xi_for_section(elem.integration, num_int_pts, ip)

    var i1 = elem.node_index_1
    var i2 = elem.node_index_2
    var node1 = nodes[i1]
    var node2 = nodes[i2]
    var dx = node2.x - node1.x
    var dy = node2.y - node1.y
    var L = sqrt(dx * dx + dy * dy)
    if L == 0.0:
        abort("zero-length element")
    var c = dx / L
    var s = dy / L
    var dof_map = [
        node_dof_index(i1, 1, ndf),
        node_dof_index(i1, 2, ndf),
        node_dof_index(i1, 3, ndf),
        node_dof_index(i2, 1, ndf),
        node_dof_index(i2, 2, ndf),
        node_dof_index(i2, 3, ndf),
    ]
    var u_local: List[Float64] = []
    u_local.resize(6, 0.0)
    u_local[0] = c * u[dof_map[0]] + s * u[dof_map[1]]
    u_local[1] = -s * u[dof_map[0]] + c * u[dof_map[1]]
    u_local[2] = u[dof_map[2]]
    u_local[3] = c * u[dof_map[3]] + s * u[dof_map[4]]
    u_local[4] = -s * u[dof_map[3]] + c * u[dof_map[4]]
    u_local[5] = u[dof_map[5]]

    var inv_L = 1.0 / L
    var eps0 = (-inv_L) * u_local[0] + inv_L * u_local[3]
    var kappa = (
        ((-6.0 + 12.0 * xi) / (L * L)) * u_local[1]
        + ((-4.0 + 6.0 * xi) / L) * u_local[2]
        + ((6.0 - 12.0 * xi) / (L * L)) * u_local[4]
        + ((-2.0 + 6.0 * xi) / L) * u_local[5]
    )
    if want_deformation:
        return [eps0, kappa]

    var load_response = beam2d_section_load_response(
        element_loads,
        elem_load_offsets,
        elem_load_pool,
        elem_index,
        load_scale,
        xi * L,
        L,
    )
    var sec = sections_by_id[elem.section]
    if sec.type == "ElasticSection2d":
        if sec.E <= 0.0 or sec.A <= 0.0 or sec.I <= 0.0:
            abort("section recorder ElasticSection2d requires positive E/A/I")
        return [
            -(sec.E * sec.A * eps0 + load_response[0]),
            sec.E * sec.I * kappa + load_response[1],
        ]

    var sec_id = elem.section
    if sec_id >= len(fiber_section_index_by_id):
        abort("section recorder section not found")
    var sec_index = fiber_section_index_by_id[sec_id]
    if sec_index < 0 or sec_index >= len(fiber_section_defs):
        abort("section recorder fiber section not found")
    var sec_def = fiber_section_defs[sec_index]
    var elem_offset = elem_uniaxial_offsets[elem_index]
    var fibers_per_section = sec_def.fiber_count
    var ip_offset = elem_offset + ip * fibers_per_section
    var section_force = _fiber_section_force_from_offset(
        sec_def,
        fiber_section_cells,
        uniaxial_defs,
        uniaxial_states,
        ip_offset,
        fibers_per_section,
        eps0,
        kappa,
    )
    section_force[0] += load_response[0]
    section_force[1] += load_response[1]
    return section_force^


fn _beam_column3d_section_response(
    elem_index: Int,
    elem: ElementInput,
    section_no: Int,
    ndf: Int,
    u: List[Float64],
    element_loads: List[ElementLoadInput],
    elem_load_offsets: List[Int],
    elem_load_pool: List[Int],
    load_scale: Float64,
    nodes: List[NodeInput],
    sections_by_id: List[SectionInput],
    fiber_section3d_defs: List[FiberSection3dDef],
    fiber_section3d_cells: List[FiberCell],
    fiber_section3d_index_by_id: List[Int],
    uniaxial_defs: List[UniMaterialDef],
    mut uniaxial_states: List[UniMaterialState],
    elem_uniaxial_offsets: List[Int],
    elem_uniaxial_counts: List[Int],
    elem_uniaxial_state_ids: List[Int],
    force_basic_offsets: List[Int],
    force_basic_counts: List[Int],
    mut force_basic_q: List[Float64],
    want_deformation: Bool,
) raises -> List[Float64]:
    var beam_col_type = elem.type
    if ndf != 6:
        abort(beam_col_type + " section recorder requires ndf=6")
    var num_int_pts = elem.num_int_pts
    beam_integration_validate_or_abort(beam_col_type, elem.integration, num_int_pts)

    var i1 = elem.node_index_1
    var i2 = elem.node_index_2
    var node1 = nodes[i1]
    var node2 = nodes[i2]
    var dof_map = [
        node_dof_index(i1, 1, ndf),
        node_dof_index(i1, 2, ndf),
        node_dof_index(i1, 3, ndf),
        node_dof_index(i1, 4, ndf),
        node_dof_index(i1, 5, ndf),
        node_dof_index(i1, 6, ndf),
        node_dof_index(i2, 1, ndf),
        node_dof_index(i2, 2, ndf),
        node_dof_index(i2, 3, ndf),
        node_dof_index(i2, 4, ndf),
        node_dof_index(i2, 5, ndf),
        node_dof_index(i2, 6, ndf),
    ]
    var u_elem: List[Float64] = []
    u_elem.resize(12, 0.0)
    for i in range(12):
        u_elem[i] = u[dof_map[i]]

    var sec = sections_by_id[elem.section]
    if sec.type == "FiberSection3d":
        var sec_index = fiber_section3d_index_by_id[elem.section]
        if sec_index < 0 or sec_index >= len(fiber_section3d_defs):
            abort(beam_col_type + " fiber section not found")
        if elem.type_tag == ElementTypeTag.ForceBeamColumn3d:
            _ = _beam_column3d_element_force_global(
                elem_index,
                elem,
                nodes,
                sections_by_id,
                ndf,
                u,
                fiber_section3d_defs,
                fiber_section3d_cells,
                fiber_section3d_index_by_id,
                uniaxial_defs,
                uniaxial_states,
                elem_uniaxial_offsets,
                elem_uniaxial_counts,
                elem_uniaxial_state_ids,
                force_basic_offsets,
                force_basic_counts,
                force_basic_q,
                element_loads,
                elem_load_offsets,
                elem_load_pool,
                load_scale,
            )
            var xi = _beam_integration_xi_for_section(
                elem.integration, num_int_pts, section_no - 1
            )
            var q_offset = force_basic_offsets[elem_index]
            var q0 = force_basic_q[q_offset]
            var q1 = force_basic_q[q_offset + 1]
            var q2 = force_basic_q[q_offset + 2]
            var q3 = force_basic_q[q_offset + 3]
            var q4 = force_basic_q[q_offset + 4]
            var eps0_offset = q_offset + 5
            var ky_offset = eps0_offset + num_int_pts
            var kz_offset = ky_offset + num_int_pts
            var dx = node2.x - node1.x
            var dy = node2.y - node1.y
            var dz = node2.z - node1.z
            var L = sqrt(dx * dx + dy * dy + dz * dz)
            if L == 0.0:
                abort("zero-length element")
            var rot_i = (dx * u_elem[3] + dy * u_elem[4] + dz * u_elem[5]) / L
            var rot_j = (dx * u_elem[9] + dy * u_elem[10] + dz * u_elem[11]) / L
            var twist = (rot_j - rot_i) / L
            if want_deformation:
                return [
                    force_basic_q[eps0_offset + section_no - 1],
                    force_basic_q[kz_offset + section_no - 1],
                    force_basic_q[ky_offset + section_no - 1],
                    twist,
                ]
            var load_response = beam3d_section_load_response(
                element_loads,
                elem_load_offsets,
                elem_load_pool,
                elem_index,
                load_scale,
                xi * L,
                L,
            )
            return [
                q0 + load_response[0],
                (xi - 1.0) * q1 + xi * q2 + load_response[2],
                (xi - 1.0) * q3 + xi * q4 + load_response[1],
                sec.G * sec.J * twist,
            ]
        var values = beam_column3d_fiber_section_response(
            node1.x,
            node1.y,
            node1.z,
            node2.x,
            node2.y,
            node2.z,
            u_elem,
            elem.geom_transf,
            fiber_section3d_defs[sec_index],
            fiber_section3d_cells,
            uniaxial_defs,
            uniaxial_states,
            elem_uniaxial_state_ids,
            elem_uniaxial_offsets[elem_index],
            elem_uniaxial_counts[elem_index],
            elem.integration,
            elem.num_int_pts,
            sec.G,
            sec.J,
            section_no,
            want_deformation,
        )
        if want_deformation:
            return values^
        var load_response = beam3d_section_load_response(
            element_loads,
            elem_load_offsets,
            elem_load_pool,
            elem_index,
            load_scale,
            _beam_integration_xi_for_section(
                elem.integration, num_int_pts, section_no - 1
            ) * sqrt(
                (node2.x - node1.x) * (node2.x - node1.x)
                + (node2.y - node1.y) * (node2.y - node1.y)
                + (node2.z - node1.z) * (node2.z - node1.z)
            ),
            sqrt(
                (node2.x - node1.x) * (node2.x - node1.x)
                + (node2.y - node1.y) * (node2.y - node1.y)
                + (node2.z - node1.z) * (node2.z - node1.z)
            ),
        )
        values[0] += load_response[0]
        values[1] += load_response[2]
        values[2] -= load_response[1]
        return values^

    if sec.type != "ElasticSection3d":
        abort(beam_col_type + " requires ElasticSection3d or FiberSection3d")
    var xi = _beam_integration_xi_for_section(elem.integration, num_int_pts, section_no - 1)
    var dx = node2.x - node1.x
    var dy = node2.y - node1.y
    var dz = node2.z - node1.z
    var L = sqrt(dx * dx + dy * dy + dz * dz)
    if L == 0.0:
        abort("zero-length element")

    var rx = dx / L
    var ry = dy / L
    var rz = dz / L
    var vx = 1.0
    var vy = 0.0
    var vz = 0.0
    if abs(rx * vx + ry * vy + rz * vz) >= 0.9:
        vx = 0.0
        vy = 1.0
        vz = 0.0
        if abs(rx * vx + ry * vy + rz * vz) >= 0.9:
            vx = 0.0
            vy = 0.0
            vz = 1.0
    var sx = vy * rz - vz * ry
    var sy = vz * rx - vx * rz
    var sz = vx * ry - vy * rx
    var s_norm = sqrt(sx * sx + sy * sy + sz * sz)
    sx /= s_norm
    sy /= s_norm
    sz /= s_norm
    var tx = ry * sz - rz * sy
    var ty = rz * sx - rx * sz
    var tz = rx * sy - ry * sx

    var u_local: List[Float64] = []
    u_local.resize(12, 0.0)
    u_local[0] = rx * u_elem[0] + ry * u_elem[1] + rz * u_elem[2]
    u_local[1] = sx * u_elem[0] + sy * u_elem[1] + sz * u_elem[2]
    u_local[2] = tx * u_elem[0] + ty * u_elem[1] + tz * u_elem[2]
    u_local[3] = rx * u_elem[3] + ry * u_elem[4] + rz * u_elem[5]
    u_local[4] = sx * u_elem[3] + sy * u_elem[4] + sz * u_elem[5]
    u_local[5] = tx * u_elem[3] + ty * u_elem[4] + tz * u_elem[5]
    u_local[6] = rx * u_elem[6] + ry * u_elem[7] + rz * u_elem[8]
    u_local[7] = sx * u_elem[6] + sy * u_elem[7] + sz * u_elem[8]
    u_local[8] = tx * u_elem[6] + ty * u_elem[7] + tz * u_elem[8]
    u_local[9] = rx * u_elem[9] + ry * u_elem[10] + rz * u_elem[11]
    u_local[10] = sx * u_elem[9] + sy * u_elem[10] + sz * u_elem[11]
    u_local[11] = tx * u_elem[9] + ty * u_elem[10] + tz * u_elem[11]

    var eps0 = (-1.0 / L) * u_local[0] + (1.0 / L) * u_local[6]
    var kappa_y = (
        ((-6.0 + 12.0 * xi) / (L * L)) * u_local[2]
        + ((4.0 - 6.0 * xi) / L) * u_local[4]
        + ((6.0 - 12.0 * xi) / (L * L)) * u_local[8]
        + ((2.0 - 6.0 * xi) / L) * u_local[10]
    )
    var kappa_z = (
        ((-6.0 + 12.0 * xi) / (L * L)) * u_local[1]
        + ((-4.0 + 6.0 * xi) / L) * u_local[5]
        + ((6.0 - 12.0 * xi) / (L * L)) * u_local[7]
        + ((-2.0 + 6.0 * xi) / L) * u_local[11]
    )
    var twist = (u_local[9] - u_local[3]) / L
    if want_deformation:
        return [eps0, kappa_z, -kappa_y, twist]
    var load_response = beam3d_section_load_response(
        element_loads,
        elem_load_offsets,
        elem_load_pool,
        elem_index,
        load_scale,
        xi * L,
        L,
    )
    return [
        sec.E * sec.A * eps0 + load_response[0],
        sec.E * sec.Iz * kappa_z + load_response[2],
        -(sec.E * sec.Iy * kappa_y + load_response[1]),
        sec.G * sec.J * twist,
    ]
fn _section_response_for_recorder(
    elem_index: Int,
    elem: ElementInput,
    section_no: Int,
    ndf: Int,
    u: List[Float64],
    element_loads: List[ElementLoadInput],
    elem_load_offsets: List[Int],
    elem_load_pool: List[Int],
    load_scale: Float64,
    nodes: List[NodeInput],
    sections_by_id: List[SectionInput],
    fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    fiber_section3d_defs: List[FiberSection3dDef],
    fiber_section3d_cells: List[FiberCell],
    fiber_section3d_index_by_id: List[Int],
    uniaxial_defs: List[UniMaterialDef],
    mut uniaxial_states: List[UniMaterialState],
    elem_uniaxial_offsets: List[Int],
    elem_uniaxial_counts: List[Int],
    elem_uniaxial_state_ids: List[Int],
    force_basic_offsets: List[Int],
    force_basic_counts: List[Int],
    mut force_basic_q: List[Float64],
    want_deformation: Bool,
) raises -> List[Float64]:
    if elem.type_tag == ElementTypeTag.ForceBeamColumn2d:
        return _force_beam_column2d_section_response(
            elem_index,
            elem,
            section_no,
            u,
            element_loads,
            elem_load_offsets,
            elem_load_pool,
            load_scale,
            nodes,
            sections_by_id,
            fiber_section_defs,
            fiber_section_cells,
            fiber_section_index_by_id,
            uniaxial_defs,
            uniaxial_states,
            elem_uniaxial_offsets,
            elem_uniaxial_counts,
            elem_uniaxial_state_ids,
            force_basic_offsets,
            force_basic_counts,
            force_basic_q,
            want_deformation,
        )
    if elem.type_tag == ElementTypeTag.DispBeamColumn2d:
        return _disp_beam_column2d_section_response(
            elem_index,
            elem,
            section_no,
            ndf,
            u,
            element_loads,
            elem_load_offsets,
            elem_load_pool,
            load_scale,
            nodes,
            sections_by_id,
            fiber_section_defs,
            fiber_section_cells,
            fiber_section_index_by_id,
            uniaxial_defs,
            uniaxial_states,
            elem_uniaxial_offsets,
            elem_uniaxial_counts,
            elem_uniaxial_state_ids,
            want_deformation,
        )
    if (
        elem.type_tag == ElementTypeTag.ForceBeamColumn3d
        or elem.type_tag == ElementTypeTag.DispBeamColumn3d
    ):
        return _beam_column3d_section_response(
            elem_index,
            elem,
            section_no,
            ndf,
            u,
            element_loads,
            elem_load_offsets,
            elem_load_pool,
            load_scale,
            nodes,
            sections_by_id,
            fiber_section3d_defs,
            fiber_section3d_cells,
            fiber_section3d_index_by_id,
            uniaxial_defs,
            uniaxial_states,
            elem_uniaxial_offsets,
            elem_uniaxial_counts,
            elem_uniaxial_state_ids,
            force_basic_offsets,
            force_basic_counts,
            force_basic_q,
            want_deformation,
        )
    abort(
        "section recorder supports forceBeamColumn2d/3d or dispBeamColumn2d/3d only"
    )
    return []


fn _append_output(
    mut filenames: List[String],
    mut buffers: List[List[String]],
    filename: String,
    line: String,
):
    for i in range(len(filenames)):
        if filenames[i] == filename:
            buffers[i].append(line)
            return
    filenames.append(filename)
    var lines: List[String] = []
    lines.append(line)
    buffers.append(lines^)


fn _output_buffer_index(
    mut filenames: List[String],
    mut buffers: List[List[String]],
    mut filename_to_index: Dict[String, Int],
    filename: String,
) -> Int:
    var existing_index = filename_to_index.get(filename)
    if existing_index:
        return existing_index.value()
    var index = len(filenames)
    filenames.append(filename)
    var lines: List[String] = []
    buffers.append(lines^)
    filename_to_index[filename] = index
    return index


fn _append_output(
    mut filenames: List[String],
    mut buffers: List[List[String]],
    mut filename_to_index: Dict[String, Int],
    filename: String,
    line: String,
):
    var index = _output_buffer_index(filenames, buffers, filename_to_index, filename)
    buffers[index].append(line)


fn _append_output_at_index(
    mut buffers: List[List[String]], file_index: Int, line: String
) raises:
    if file_index < 0 or file_index >= len(buffers):
        abort("output buffer index out of range")
    buffers[file_index].append(line)


fn _has_recorder_type(recorders: List[RecorderInput], wanted_tag: Int) -> Bool:
    for r in range(len(recorders)):
        if recorders[r].type_tag == wanted_tag:
            return True
    return False


fn _format_values_line(values: List[Float64]) -> String:
    var line = String()
    for i in range(len(values)):
        if i > 0:
            line += " "
        line += String(values[i])
    line += "\n"
    return line


fn _element_force_global_for_recorder(
    elem_index: Int,
    elem: ElementInput,
    ndf: Int,
    u: List[Float64],
    nodes: List[NodeInput],
    sections_by_id: List[SectionInput],
    fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    fiber_section3d_defs: List[FiberSection3dDef],
    fiber_section3d_cells: List[FiberCell],
    fiber_section3d_index_by_id: List[Int],
    uniaxial_defs: List[UniMaterialDef],
    uniaxial_state_defs: List[Int],
    mut uniaxial_states: List[UniMaterialState],
    elem_uniaxial_offsets: List[Int],
    elem_uniaxial_counts: List[Int],
    elem_uniaxial_state_ids: List[Int],
    force_basic_offsets: List[Int],
    force_basic_counts: List[Int],
    mut force_basic_q: List[Float64],
) raises -> List[Float64]:
    var empty_element_loads: List[ElementLoadInput] = []
    var empty_elem_load_offsets: List[Int] = []
    var empty_elem_load_pool: List[Int] = []
    return _element_force_global_for_recorder(
        elem_index,
        elem,
        ndf,
        u,
        empty_element_loads,
        empty_elem_load_offsets,
        empty_elem_load_pool,
        0.0,
        nodes,
        sections_by_id,
        fiber_section_defs,
        fiber_section_cells,
        fiber_section_index_by_id,
        fiber_section3d_defs,
        fiber_section3d_cells,
        fiber_section3d_index_by_id,
        uniaxial_defs,
        uniaxial_state_defs,
        uniaxial_states,
        elem_uniaxial_offsets,
        elem_uniaxial_counts,
        elem_uniaxial_state_ids,
        force_basic_offsets,
        force_basic_counts,
        force_basic_q,
    )


fn _element_force_global_for_recorder(
    elem_index: Int,
    elem: ElementInput,
    ndf: Int,
    u: List[Float64],
    element_loads: List[ElementLoadInput],
    elem_load_offsets: List[Int],
    elem_load_pool: List[Int],
    load_scale: Float64,
    nodes: List[NodeInput],
    sections_by_id: List[SectionInput],
    fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    fiber_section3d_defs: List[FiberSection3dDef],
    fiber_section3d_cells: List[FiberCell],
    fiber_section3d_index_by_id: List[Int],
    uniaxial_defs: List[UniMaterialDef],
    uniaxial_state_defs: List[Int],
    mut uniaxial_states: List[UniMaterialState],
    elem_uniaxial_offsets: List[Int],
    elem_uniaxial_counts: List[Int],
    elem_uniaxial_state_ids: List[Int],
    force_basic_offsets: List[Int],
    force_basic_counts: List[Int],
    mut force_basic_q: List[Float64],
) raises -> List[Float64]:
    if elem.type_tag == ElementTypeTag.ElasticBeamColumn2d:
        return _beam2d_element_force_global(
            elem_index,
            elem,
            nodes,
            sections_by_id,
            ndf,
            u,
            element_loads,
            elem_load_offsets,
            elem_load_pool,
            load_scale,
        )
    if (
        elem.type_tag == ElementTypeTag.ElasticBeamColumn3d
        or elem.type_tag == ElementTypeTag.ForceBeamColumn3d
        or elem.type_tag == ElementTypeTag.DispBeamColumn3d
    ):
        return _beam_column3d_element_force_global(
            elem_index,
            elem,
            nodes,
            sections_by_id,
            ndf,
            u,
            fiber_section3d_defs,
            fiber_section3d_cells,
            fiber_section3d_index_by_id,
            uniaxial_defs,
            uniaxial_states,
            elem_uniaxial_offsets,
            elem_uniaxial_counts,
            elem_uniaxial_state_ids,
            force_basic_offsets,
            force_basic_counts,
            force_basic_q,
            element_loads,
            elem_load_offsets,
            elem_load_pool,
            load_scale,
        )
    if elem.type_tag == ElementTypeTag.ForceBeamColumn2d:
        return _force_beam_column2d_element_force_global(
            elem_index,
            elem,
            nodes,
            sections_by_id,
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
            force_basic_offsets,
            force_basic_counts,
            force_basic_q,
            element_loads,
            elem_load_offsets,
            elem_load_pool,
            load_scale,
        )
    if elem.type_tag == ElementTypeTag.DispBeamColumn2d:
        return _disp_beam_column2d_element_force_global(
            elem_index,
            elem,
            nodes,
            sections_by_id,
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
            element_loads,
            elem_load_offsets,
            elem_load_pool,
            load_scale,
        )
    if elem.type_tag == ElementTypeTag.Truss:
        return _truss_element_force_global(
            elem_index,
            elem,
            nodes,
            ndf,
            u,
            uniaxial_defs,
            uniaxial_state_defs,
            uniaxial_states,
            elem_uniaxial_offsets,
            elem_uniaxial_counts,
            elem_uniaxial_state_ids,
        )
    var elem_ndm = 2
    if ndf >= 6 or nodes[elem.node_index_1].has_z or nodes[elem.node_index_2].has_z:
        elem_ndm = 3
    if elem.type_tag == ElementTypeTag.ZeroLength:
        return _zero_length_element_force_global(
            elem_index,
            elem,
            nodes,
            ndf,
            elem_ndm,
            u,
            uniaxial_defs,
            uniaxial_state_defs,
            uniaxial_states,
            elem_uniaxial_offsets,
            elem_uniaxial_counts,
            elem_uniaxial_state_ids,
        )
    if elem.type_tag == ElementTypeTag.TwoNodeLink:
        return _two_node_link_element_force_global(
            elem_index,
            elem,
            nodes,
            ndf,
            elem_ndm,
            u,
            uniaxial_defs,
            uniaxial_state_defs,
            uniaxial_states,
            elem_uniaxial_offsets,
            elem_uniaxial_counts,
            elem_uniaxial_state_ids,
        )
    abort(
        "element_force recorder supports truss, zeroLength, twoNodeLink, "
        "elasticBeamColumn2d/3d, forceBeamColumn2d/3d, or dispBeamColumn2d/3d only"
    )
    return []


fn _element_local_force_for_recorder(
    elem_index: Int,
    elem: ElementInput,
    ndf: Int,
    u: List[Float64],
    nodes: List[NodeInput],
    sections_by_id: List[SectionInput],
    fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    fiber_section3d_defs: List[FiberSection3dDef],
    fiber_section3d_cells: List[FiberCell],
    fiber_section3d_index_by_id: List[Int],
    uniaxial_defs: List[UniMaterialDef],
    uniaxial_state_defs: List[Int],
    mut uniaxial_states: List[UniMaterialState],
    elem_uniaxial_offsets: List[Int],
    elem_uniaxial_counts: List[Int],
    elem_uniaxial_state_ids: List[Int],
    force_basic_offsets: List[Int],
    force_basic_counts: List[Int],
    mut force_basic_q: List[Float64],
) raises -> List[Float64]:
    var elem_ndm = 2
    if ndf >= 6 or nodes[elem.node_index_1].has_z or nodes[elem.node_index_2].has_z:
        elem_ndm = 3
    if elem.type_tag == ElementTypeTag.ZeroLength:
        return _zero_length_basic_force_for_recorder(
            elem_index,
            elem,
            nodes,
            ndf,
            elem_ndm,
            u,
            uniaxial_defs,
            uniaxial_state_defs,
            uniaxial_states,
            elem_uniaxial_offsets,
            elem_uniaxial_counts,
            elem_uniaxial_state_ids,
        )
    if elem.type_tag == ElementTypeTag.TwoNodeLink:
        return _two_node_link_local_force_for_recorder(
            elem_index,
            elem,
            nodes,
            ndf,
            elem_ndm,
            u,
            uniaxial_defs,
            uniaxial_state_defs,
            uniaxial_states,
            elem_uniaxial_offsets,
            elem_uniaxial_counts,
            elem_uniaxial_state_ids,
        )
    if (
        elem.type_tag == ElementTypeTag.ElasticBeamColumn2d
        or elem.type_tag == ElementTypeTag.ForceBeamColumn2d
        or elem.type_tag == ElementTypeTag.DispBeamColumn2d
    ):
        var f_global: List[Float64]
        if elem.type_tag == ElementTypeTag.ElasticBeamColumn2d:
            var empty_element_loads: List[ElementLoadInput] = []
            var empty_elem_load_offsets: List[Int] = []
            var empty_elem_load_pool: List[Int] = []
            f_global = _beam2d_element_force_global(
                elem_index,
                elem,
                nodes,
                sections_by_id,
                ndf,
                u,
                empty_element_loads,
                empty_elem_load_offsets,
                empty_elem_load_pool,
                0.0,
            )
        elif elem.type_tag == ElementTypeTag.ForceBeamColumn2d:
            f_global = _force_beam_column2d_element_force_global(
                elem_index,
                elem,
                nodes,
                sections_by_id,
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
                force_basic_offsets,
                force_basic_counts,
                force_basic_q,
            )
        else:
            f_global = _disp_beam_column2d_element_force_global(
                elem_index,
                elem,
                nodes,
                sections_by_id,
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
        return _beam2d_force_global_to_local(
            nodes[elem.node_index_1], nodes[elem.node_index_2], f_global
        )
    if (
        elem.type_tag == ElementTypeTag.ElasticBeamColumn3d
        or elem.type_tag == ElementTypeTag.ForceBeamColumn3d
        or elem.type_tag == ElementTypeTag.DispBeamColumn3d
    ):
        var empty_element_loads: List[ElementLoadInput] = []
        var empty_elem_load_offsets: List[Int] = []
        var empty_elem_load_pool: List[Int] = []
        var f_global = _beam_column3d_element_force_global(
            elem_index,
            elem,
            nodes,
            sections_by_id,
            ndf,
            u,
            fiber_section3d_defs,
            fiber_section3d_cells,
            fiber_section3d_index_by_id,
            uniaxial_defs,
            uniaxial_states,
            elem_uniaxial_offsets,
            elem_uniaxial_counts,
            elem_uniaxial_state_ids,
            force_basic_offsets,
            force_basic_counts,
            force_basic_q,
            empty_element_loads,
            empty_elem_load_offsets,
            empty_elem_load_pool,
            0.0,
        )
        return _beam3d_force_global_to_local(
            nodes[elem.node_index_1], nodes[elem.node_index_2], f_global
        )
    abort(
        "element_local_force recorder supports zeroLength, twoNodeLink, "
        "elasticBeamColumn2d/3d, forceBeamColumn2d/3d, or dispBeamColumn2d/3d only"
    )
    return []


fn _element_basic_force_for_recorder(
    elem_index: Int,
    elem: ElementInput,
    ndf: Int,
    u: List[Float64],
    nodes: List[NodeInput],
    uniaxial_defs: List[UniMaterialDef],
    uniaxial_state_defs: List[Int],
    mut uniaxial_states: List[UniMaterialState],
    elem_uniaxial_offsets: List[Int],
    elem_uniaxial_counts: List[Int],
    elem_uniaxial_state_ids: List[Int],
) raises -> List[Float64]:
    var elem_ndm = 2
    if ndf >= 6 or nodes[elem.node_index_1].has_z or nodes[elem.node_index_2].has_z:
        elem_ndm = 3
    if elem.type_tag == ElementTypeTag.ZeroLength:
        return _zero_length_basic_force_for_recorder(
            elem_index,
            elem,
            nodes,
            ndf,
            elem_ndm,
            u,
            uniaxial_defs,
            uniaxial_state_defs,
            uniaxial_states,
            elem_uniaxial_offsets,
            elem_uniaxial_counts,
            elem_uniaxial_state_ids,
        )
    if elem.type_tag == ElementTypeTag.TwoNodeLink:
        return _two_node_link_basic_force_for_recorder(
            elem_index,
            elem,
            nodes,
            ndf,
            elem_ndm,
            u,
            uniaxial_defs,
            uniaxial_state_defs,
            uniaxial_states,
            elem_uniaxial_offsets,
            elem_uniaxial_counts,
            elem_uniaxial_state_ids,
        )
    abort("element_basic_force recorder supports zeroLength and twoNodeLink only")
    return []


fn _beam2d_deformation_for_recorder(
    elem: ElementInput, nodes: List[NodeInput], ndf: Int, u: List[Float64]
) raises -> List[Float64]:
    if ndf != 3:
        abort("element_deformation recorder for beam-column 2d requires ndf=3")
    var i1 = elem.node_index_1
    var i2 = elem.node_index_2
    var node1 = nodes[i1]
    var node2 = nodes[i2]
    var dx = node2.x - node1.x
    var dy = node2.y - node1.y
    var L = sqrt(dx * dx + dy * dy)
    if L == 0.0:
        abort("zero-length element")
    var c = dx / L
    var s = dy / L
    var dof_map = [
        node_dof_index(i1, 1, ndf),
        node_dof_index(i1, 2, ndf),
        node_dof_index(i1, 3, ndf),
        node_dof_index(i2, 1, ndf),
        node_dof_index(i2, 2, ndf),
        node_dof_index(i2, 3, ndf),
    ]
    var u_global: List[Float64] = []
    u_global.resize(6, 0.0)
    for i in range(6):
        u_global[i] = u[dof_map[i]]
    var u_local: List[Float64] = []
    u_local.resize(6, 0.0)
    _beam2d_transform_u_global_to_local(c, s, u_global, u_local)
    if elem.geom_tag == GeomTransfTag.Corotational:
        var dulx = u_local[3] - u_local[0]
        var duly = u_local[4] - u_local[1]
        var Lx = L + dulx
        var Ly = duly
        var Ln = sqrt(Lx * Lx + Ly * Ly)
        if Ln == 0.0:
            abort("zero-length element")
        var alpha = atan2(Ly, Lx)
        return [Ln - L, u_local[2] - alpha, u_local[5] - alpha]
    var chord_rotation = (u_local[4] - u_local[1]) / L
    return [
        u_local[3] - u_local[0],
        u_local[2] - chord_rotation,
        u_local[5] - chord_rotation,
    ]


fn _element_deformation_for_recorder(
    elem_index: Int,
    elem: ElementInput,
    ndf: Int,
    u: List[Float64],
    nodes: List[NodeInput],
) raises -> List[Float64]:
    var elem_ndm = 2
    if ndf >= 6 or nodes[elem.node_index_1].has_z or nodes[elem.node_index_2].has_z:
        elem_ndm = 3
    if elem.type_tag == ElementTypeTag.ZeroLength:
        return _zero_length_deformation_for_recorder(
            elem_index, elem, nodes, ndf, elem_ndm, u
        )
    if elem.type_tag == ElementTypeTag.TwoNodeLink:
        return _two_node_link_deformation_for_recorder(
            elem, nodes, ndf, elem_ndm, u
        )
    if (
        elem.type_tag == ElementTypeTag.ElasticBeamColumn2d
        or elem.type_tag == ElementTypeTag.ForceBeamColumn2d
        or elem.type_tag == ElementTypeTag.DispBeamColumn2d
    ):
        return _beam2d_deformation_for_recorder(elem, nodes, ndf, u)
    abort(
        "element_deformation recorder supports zeroLength, twoNodeLink, "
        "or beam-column 2d elements only"
    )
    return []


fn _node_coordinate(node: NodeInput, coord_index: Int) raises -> Float64:
    if coord_index == 1:
        return node.x
    if coord_index == 2:
        return node.y
    if coord_index == 3:
        return node.z
    abort("coordinate index out of range")
    return 0.0


fn _drift_value(
    rec: RecorderInput,
    nodes: List[NodeInput],
    id_to_index: List[Int],
    ndf: Int,
    u: List[Float64],
) raises -> Float64:
    var i_node = rec.i_node
    var j_node = rec.j_node
    if i_node >= len(id_to_index) or id_to_index[i_node] < 0:
        abort("drift i_node not found")
    if j_node >= len(id_to_index) or id_to_index[j_node] < 0:
        abort("drift j_node not found")
    var dof = rec.drift_dof
    var perp_dirn = rec.perp_dirn
    require_dof_in_range(dof, ndf, "drift recorder dof")
    if perp_dirn < 1 or perp_dirn > 3:
        abort("drift perp_dirn must be in 1..3")
    var i_idx = id_to_index[i_node]
    var j_idx = id_to_index[j_node]
    var node_i = nodes[i_idx]
    var node_j = nodes[j_idx]
    if perp_dirn == 3 and (not node_i.has_z or not node_j.has_z):
        abort("drift perp_dirn=3 requires z coordinates")
    var dof_i = node_dof_index(i_idx, dof, ndf)
    var dof_j = node_dof_index(j_idx, dof, ndf)
    var dx = _node_coordinate(node_j, perp_dirn) - _node_coordinate(node_i, perp_dirn)
    if dx == 0.0:
        abort("drift denominator is zero")
    return (u[dof_j] - u[dof_i]) / dx


fn _scaled_forces(F_total: List[Float64], scale: Float64) -> List[Float64]:
    var scaled: List[Float64] = []
    scaled.resize(len(F_total), 0.0)
    for i in range(len(F_total)):
        scaled[i] = F_total[i] * scale
    return scaled^


fn _update_envelope(
    filename: String,
    values: List[Float64],
    mut envelope_files: List[String],
    mut envelope_min: List[List[Float64]],
    mut envelope_max: List[List[Float64]],
    mut envelope_abs: List[List[Float64]],
):
    for i in range(len(envelope_files)):
        if envelope_files[i] == filename:
            for j in range(len(values)):
                var value = values[j]
                if value < envelope_min[i][j]:
                    envelope_min[i][j] = value
                if value > envelope_max[i][j]:
                    envelope_max[i][j] = value
                var abs_val = abs(value)
                if abs_val > envelope_abs[i][j]:
                    envelope_abs[i][j] = abs_val
            return
    var mins: List[Float64] = []
    var maxs: List[Float64] = []
    var abss: List[Float64] = []
    mins.resize(len(values), 0.0)
    maxs.resize(len(values), 0.0)
    abss.resize(len(values), 0.0)
    for j in range(len(values)):
        mins[j] = values[j]
        maxs[j] = values[j]
        abss[j] = abs(values[j])
    envelope_files.append(filename)
    envelope_min.append(mins^)
    envelope_max.append(maxs^)
    envelope_abs.append(abss^)


fn _flush_envelope_outputs(
    envelope_files: List[String],
    envelope_min: List[List[Float64]],
    envelope_max: List[List[Float64]],
    envelope_abs: List[List[Float64]],
    mut output_files: List[String],
    mut output_buffers: List[List[String]],
):
    for i in range(len(envelope_files)):
        var line = String()
        line += _format_values_line(envelope_min[i])
        line += _format_values_line(envelope_max[i])
        line += _format_values_line(envelope_abs[i])
        _append_output(output_files, output_buffers, envelope_files[i], line)


fn _flush_envelope_outputs(
    envelope_files: List[String],
    envelope_min: List[List[Float64]],
    envelope_max: List[List[Float64]],
    envelope_abs: List[List[Float64]],
    mut output_files: List[String],
    mut output_buffers: List[List[String]],
    mut output_file_index: Dict[String, Int],
):
    for i in range(len(envelope_files)):
        var line = String()
        line += _format_values_line(envelope_min[i])
        line += _format_values_line(envelope_max[i])
        line += _format_values_line(envelope_abs[i])
        _append_output(
            output_files,
            output_buffers,
            output_file_index,
            envelope_files[i],
            line,
        )


fn _solve_linear_system(
    mut A: List[List[Float64]], mut b: List[Float64], mut x: List[Float64]
) -> Bool:
    var n = len(b)
    if len(A) != n:
        return False
    x.resize(n, 0.0)
    for i in range(n):
        if len(A[i]) != n:
            return False

    var eps = 1.0e-18
    for i in range(n):
        var pivot = i
        var max_val = abs(A[i][i])
        for r in range(i + 1, n):
            var candidate = abs(A[r][i])
            if candidate > max_val:
                max_val = candidate
                pivot = r
        if max_val <= eps:
            return False
        if pivot != i:
            var tmp = A[i].copy()
            A[i] = A[pivot].copy()
            A[pivot] = tmp^
            var tb = b[i]
            b[i] = b[pivot]
            b[pivot] = tb

        var piv = A[i][i]
        for j in range(i, n):
            A[i][j] /= piv
        b[i] /= piv

        for r in range(i + 1, n):
            var factor = A[r][i]
            if factor == 0.0:
                continue
            for c in range(i, n):
                A[r][c] -= factor * A[i][c]
            b[r] -= factor * b[i]

    for i in range(n - 1, -1, -1):
        var s = b[i]
        for j in range(i + 1, n):
            s -= A[i][j] * x[j]
        x[i] = s
    return True


fn _collapse_vector_by_rep(values: List[Float64], rep_dof: List[Int]) -> List[Float64]:
    var out: List[Float64] = []
    out.resize(len(values), 0.0)
    for i in range(len(values)):
        out[rep_dof[i]] += values[i]
    return out^


fn _collapse_matrix_by_rep(
    matrix: List[List[Float64]], rep_dof: List[Int]
) -> List[List[Float64]]:
    var n = len(matrix)
    var out: List[List[Float64]] = []
    for _ in range(n):
        var row: List[Float64] = []
        row.resize(n, 0.0)
        out.append(row^)
    for i in range(n):
        var ri = rep_dof[i]
        for j in range(n):
            out[ri][rep_dof[j]] += matrix[i][j]
    return out^


fn _enforce_equal_dof_values(
    mut values: List[Float64], rep_dof: List[Int], constrained: List[Bool]
):
    for i in range(len(values)):
        var rep = rep_dof[i]
        if constrained[rep]:
            values[i] = 0.0
        else:
            values[i] = values[rep]
