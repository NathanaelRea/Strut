from collections import List
from os import abort

from elements import (
    ForceBeamColumn2dScratch,
    beam2d_corotational_global_stiffness,
    beam2d_pdelta_global_stiffness,
    beam_global_stiffness,
    disp_beam_column2d_global_tangent_and_internal,
    force_beam_column2d_global_tangent_and_internal,
)
from materials import UniMaterialDef, UniMaterialState
from solver.assembly.stiffness_internal_shared import (
    _beam_integration_name_from_tag,
    _build_elem_dof_soa,
    _geom_transf_name_from_tag,
)
from solver.banded import banded_add, banded_matrix
from solver.run_case.input_types import (
    ElementInput,
    ElementLoadInput,
    NodeInput,
    SectionInput,
)
from sections import FiberCell, FiberSection2dDef
from tag_types import ElementTypeTag, GeomTransfTag


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
    var k_global6: List[List[Float64]] = []
    var force_beam_column2d_scratch = ForceBeamColumn2dScratch()

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
            if sec.type == "ElasticSection2d":
                var geom = elem.geom_transf
                if geom == "Linear":
                    k_global6 = beam_global_stiffness(
                        sec.E,
                        sec.A,
                        sec.I,
                        node1.x,
                        node1.y,
                        node2.x,
                        node2.y,
                    )
                elif geom == "PDelta":
                    k_global6 = beam2d_pdelta_global_stiffness(
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
                        force_beam_column2d_scratch,
                        k_global6,
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
                        k_global6,
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
                    banded_add(K, bw, Aidx, Bidx, k_global6[a][b])
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


fn assemble_global_stiffness_banded_frame2d_soa(
    node_x: List[Float64],
    node_y: List[Float64],
    elem_type_tags: List[Int],
    elem_geom_tags: List[Int],
    elem_node_offsets: List[Int],
    elem_node_pool: List[Int],
    elem_section_ids: List[Int],
    elem_integration_tags: List[Int],
    elem_num_int_pts: List[Int],
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
    free_count: Int,
    bw: Int,
    elem_dof_pool: List[Int],
    elem_free_offsets: List[Int],
    elem_free_pool: List[Int],
    mut free_map: List[Int],
    mut u_elem: List[Float64],
    mut f_dummy: List[Float64],
) raises -> List[List[Float64]]:
    var elem_count = len(elem_type_tags)
    var K = banded_matrix(free_count, bw)
    if len(elem_geom_tags) != elem_count:
        abort("invalid elem_geom_tags size for banded frame2d assembly")
    if len(elem_section_ids) != elem_count:
        abort("invalid elem_section_ids size for banded frame2d assembly")
    if len(elem_integration_tags) != elem_count:
        abort("invalid elem_integration_tags size for banded frame2d assembly")
    if len(elem_num_int_pts) != elem_count:
        abort("invalid elem_num_int_pts size for banded frame2d assembly")
    if len(elem_node_offsets) != elem_count + 1:
        abort("invalid elem_node_offsets size for banded frame2d assembly")
    if len(elem_node_pool) != elem_node_offsets[elem_count]:
        abort("invalid elem_node_pool size for banded frame2d assembly")
    if len(elem_free_offsets) != elem_count + 1:
        abort("invalid elem_free_offsets size for banded frame2d assembly")
    if len(elem_free_pool) != elem_free_offsets[elem_count]:
        abort("invalid elem_free_pool size for banded frame2d assembly")
    if len(elem_dof_pool) != elem_free_offsets[elem_count]:
        abort("invalid elem_dof_pool size for banded frame2d assembly")
    if len(free_map) != 6:
        free_map.resize(6, -1)
    if len(u_elem) != 6:
        u_elem.resize(6, 0.0)
    var k_global6: List[List[Float64]] = []
    var force_beam_column2d_scratch = ForceBeamColumn2dScratch()

    for e in range(elem_count):
        var elem_type = elem_type_tags[e]
        if (
            elem_type != ElementTypeTag.ElasticBeamColumn2d
            and elem_type != ElementTypeTag.ForceBeamColumn2d
            and elem_type != ElementTypeTag.DispBeamColumn2d
        ):
            abort(
                "typed frame2d banded assembly requires elasticBeamColumn2d, "
                "forceBeamColumn2d, or dispBeamColumn2d"
            )

        var node_offset = elem_node_offsets[e]
        if elem_node_offsets[e + 1] - node_offset != 2:
            abort("banded frame2d assembly requires exactly two nodes per element")
        var i1 = elem_node_pool[node_offset]
        var i2 = elem_node_pool[node_offset + 1]
        var x1 = node_x[i1]
        var y1 = node_y[i1]
        var x2 = node_x[i2]
        var y2 = node_y[i2]

        var dof_offset = elem_free_offsets[e]
        var d0 = elem_dof_pool[dof_offset]
        var d1 = elem_dof_pool[dof_offset + 1]
        var d2 = elem_dof_pool[dof_offset + 2]
        var d3 = elem_dof_pool[dof_offset + 3]
        var d4 = elem_dof_pool[dof_offset + 4]
        var d5 = elem_dof_pool[dof_offset + 5]
        free_map[0] = elem_free_pool[dof_offset]
        free_map[1] = elem_free_pool[dof_offset + 1]
        free_map[2] = elem_free_pool[dof_offset + 2]
        free_map[3] = elem_free_pool[dof_offset + 3]
        free_map[4] = elem_free_pool[dof_offset + 4]
        free_map[5] = elem_free_pool[dof_offset + 5]

        var sec = sections_by_id[elem_section_ids[e]]
        var geom_tag = elem_geom_tags[e]
        if elem_type == ElementTypeTag.ElasticBeamColumn2d:
            var k_global: List[List[Float64]] = []
            if geom_tag == GeomTransfTag.Linear:
                k_global = beam_global_stiffness(sec.E, sec.A, sec.I, x1, y1, x2, y2)
            elif geom_tag == GeomTransfTag.PDelta:
                u_elem[0] = u[d0]
                u_elem[1] = u[d1]
                u_elem[2] = u[d2]
                u_elem[3] = u[d3]
                u_elem[4] = u[d4]
                u_elem[5] = u[d5]
                k_global = beam2d_pdelta_global_stiffness(
                    sec.E, sec.A, sec.I, x1, y1, x2, y2, u_elem
                )
            elif geom_tag == GeomTransfTag.Corotational:
                u_elem[0] = u[d0]
                u_elem[1] = u[d1]
                u_elem[2] = u[d2]
                u_elem[3] = u[d3]
                u_elem[4] = u[d4]
                u_elem[5] = u[d5]
                k_global = beam2d_corotational_global_stiffness(
                    sec.E, sec.A, sec.I, x1, y1, x2, y2, u_elem
                )
            else:
                abort("unsupported geomTransf tag in frame2d banded assembly")
            for a in range(6):
                var Aidx = free_map[a]
                if Aidx < 0:
                    continue
                for b in range(6):
                    var Bidx = free_map[b]
                    if Bidx < 0:
                        continue
                    banded_add(K, bw, Aidx, Bidx, k_global[a][b])
            continue

        u_elem[0] = u[d0]
        u_elem[1] = u[d1]
        u_elem[2] = u[d2]
        u_elem[3] = u[d3]
        u_elem[4] = u[d4]
        u_elem[5] = u[d5]
        if sec.type == "ElasticSection2d":
            if geom_tag == GeomTransfTag.Linear:
                k_global6 = beam_global_stiffness(sec.E, sec.A, sec.I, x1, y1, x2, y2)
            elif geom_tag == GeomTransfTag.PDelta:
                k_global6 = beam2d_pdelta_global_stiffness(
                    sec.E, sec.A, sec.I, x1, y1, x2, y2, u_elem
                )
            else:
                abort(
                    "forceBeamColumn2d/dispBeamColumn2d supports geomTransf Linear or PDelta"
                )
        else:
            var sec_index = fiber_section_index_by_id[elem_section_ids[e]]
            var sec_def = fiber_section_defs[sec_index]
            var elem_offset = elem_uniaxial_offsets[e]
            var elem_state_count = elem_uniaxial_counts[e]
            var geom_name = _geom_transf_name_from_tag(geom_tag)
            var integration_name = _beam_integration_name_from_tag(
                elem_integration_tags[e]
            )
            if elem_type == ElementTypeTag.ForceBeamColumn2d:
                force_beam_column2d_global_tangent_and_internal(
                    e,
                    x1,
                    y1,
                    x2,
                    y2,
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
                    geom_name,
                    integration_name,
                    elem_num_int_pts[e],
                    force_basic_q,
                    force_basic_offsets[e],
                    force_basic_counts[e],
                    force_beam_column2d_scratch,
                    k_global6,
                    f_dummy,
                )
            else:
                disp_beam_column2d_global_tangent_and_internal(
                    e,
                    x1,
                    y1,
                    x2,
                    y2,
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
                    geom_name,
                    integration_name,
                    elem_num_int_pts[e],
                    k_global6,
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
                banded_add(K, bw, Aidx, Bidx, k_global6[a][b])
    return K^
