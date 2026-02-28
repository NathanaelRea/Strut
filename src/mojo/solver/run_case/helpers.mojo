from collections import List
from math import sqrt
from os import abort

from elements import (
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
from sections import FiberCell, FiberSection2dDef, FiberSection3dDef
from tag_types import ElementTypeTag


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

    var geom = elem.geom_transf
    var f_elem: List[Float64] = []
    if geom == "Corotational":
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
            abort("unsupported geomTransf: " + geom)

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
    var geom = elem.geom_transf
    if geom != "Linear" and geom != "PDelta" and geom != "Corotational":
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
                geom,
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
                geom,
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
                geom,
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
                geom,
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
                geom,
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
    var state = uniaxial_states[state_index]
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
        uniaxial_states[state_index] = state
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
    uniaxial_states[state_index] = state
    var N = state.sig_t * A
    return [-N * l, -N * m, -N * n, N * l, N * m, N * n]


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
    var geom = elem.geom_transf
    if geom != "Linear" and geom != "PDelta":
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
            abort(beam_col_type + " supports geomTransf Linear or PDelta")
        var f_global_elastic: List[Float64] = []
        f_global_elastic.resize(6, 0.0)
        for a in range(6):
            var sum = 0.0
            for b in range(6):
                sum += k_global[a][b] * u_elem[b]
            f_global_elastic[a] = sum
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
        geom,
        integration,
        num_int_pts,
        force_basic_q,
        force_basic_offsets[elem_index],
        force_basic_counts[elem_index],
        k_dummy,
        f_global,
    )
    return f_global^


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
    var geom = elem.geom_transf
    if geom != "Linear" and geom != "PDelta":
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
            abort(beam_col_type + " supports geomTransf Linear or PDelta")
        var f_global_elastic: List[Float64] = []
        f_global_elastic.resize(6, 0.0)
        for a in range(6):
            var sum = 0.0
            for b in range(6):
                sum += k_global[a][b] * u_elem[b]
            f_global_elastic[a] = sum
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
        geom,
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
    elem_state_ids: List[Int],
    state_offset: Int,
    state_count: Int,
    eps0: Float64,
    kappa: Float64,
) raises -> List[Float64]:
    if state_count != sec_def.fiber_count:
        abort("section recorder fiber state count mismatch")
    if state_offset < 0 or state_offset + state_count > len(elem_state_ids):
        abort("section recorder fiber state offset out of range")

    var axial_force = 0.0
    var moment_z = 0.0
    for i in range(state_count):
        var cell = fibers[sec_def.fiber_offset + i]
        var y_rel = cell.y - sec_def.y_bar
        var eps = eps0 - y_rel * kappa
        var state_index = elem_state_ids[state_offset + i]
        if state_index < 0 or state_index >= len(uniaxial_states):
            abort("section recorder fiber state index out of range")
        if cell.def_index < 0 or cell.def_index >= len(uniaxial_defs):
            abort("section recorder fiber material definition out of range")
        var mat_def = uniaxial_defs[cell.def_index]
        var state = uniaxial_states[state_index]
        uniaxial_set_trial_strain(mat_def, state, eps)
        uniaxial_states[state_index] = state

        var fs = state.sig_t * cell.area
        axial_force += fs
        moment_z += -fs * y_rel
    return [axial_force, moment_z]


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

    # Refresh basic force/predictor state at the current displacement.
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

    var q_offset = force_basic_offsets[elem_index]
    var q_count = force_basic_counts[elem_index]
    if q_count < 3:
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
        return [sec.E * sec.A * eps0 + load_response[0], sec.E * sec.I * kappa + load_response[1]]

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
        elem_uniaxial_state_ids,
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
    abort(
        "element_force recorder supports truss, "
        "elasticBeamColumn2d/3d, forceBeamColumn2d/3d, or dispBeamColumn2d/3d only"
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
