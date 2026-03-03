from collections import List
from os import abort

from elements import (
    ForceBeamColumn3dScratch,
    beam_column3d_fiber_global_tangent_and_internal,
    disp_beam_column3d_global_tangent_and_internal,
    force_beam_column3d_fiber_global_tangent_and_internal,
    force_beam_column3d_global_tangent_and_internal,
)
from materials import UniMaterialDef, UniMaterialState
from solver.assembly.stiffness_internal_shared import (
    _beam_integration_name_from_tag,
    _geom_transf_name_from_tag,
    _profile_scope_close,
    _profile_scope_open,
    _scatter_add_row_unrolled4,
)
from solver.run_case.input_types import (
    ElementInput,
    ElementLoadInput,
    NodeInput,
    SectionInput,
)
from sections import FiberCell, FiberSection3dDef
from tag_types import ElementTypeTag


fn _assemble_frame3d_soa_indices(
    frame3d_elem_indices: List[Int],
    node_x: List[Float64],
    node_y: List[Float64],
    node_z: List[Float64],
    elem_dof_offsets: List[Int],
    elem_dof_pool: List[Int],
    elem_node_offsets: List[Int],
    elem_node_pool: List[Int],
    elem_type_tags: List[Int],
    elem_geom_tags: List[Int],
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
    fiber_section3d_defs: List[FiberSection3dDef],
    fiber_section3d_cells: List[FiberCell],
    fiber_section3d_index_by_id: List[Int],
    mut force_beam_column3d_scratch: ForceBeamColumn3dScratch,
    mut dof_map12: List[Int],
    mut K: List[List[Float64]],
    mut F_int: List[Float64],
    do_profile: Bool,
    t0: Int,
    mut events: String,
    mut events_need_comma: Bool,
    frame_assemble_fiber: Int,
) raises:
    var u_elem12: List[Float64] = []
    u_elem12.resize(12, 0.0)
    for idx in range(len(frame3d_elem_indices)):
        var e = frame3d_elem_indices[idx]
        var node_offset = elem_node_offsets[e]
        var i1 = elem_node_pool[node_offset]
        var i2 = elem_node_pool[node_offset + 1]
        var x1 = node_x[i1]
        var y1 = node_y[i1]
        var z1 = node_z[i1]
        var x2 = node_x[i2]
        var y2 = node_y[i2]
        var z2 = node_z[i2]
        var sec = sections_by_id[elem_section_ids[e]]
        var dof_offset = elem_dof_offsets[e]
        for i in range(12):
            dof_map12[i] = elem_dof_pool[dof_offset + i]
            u_elem12[i] = u[dof_map12[i]]
        var elem_type = elem_type_tags[e]
        var geom_name = _geom_transf_name_from_tag(elem_geom_tags[e])
        var k_elem12: List[List[Float64]] = []
        var f_elem12: List[Float64] = []
        if elem_type == ElementTypeTag.ElasticBeamColumn3d:
            if sec.type != "ElasticSection3d":
                abort("elasticBeamColumn3d requires ElasticSection3d")
            force_beam_column3d_global_tangent_and_internal(
                e,
                x1,
                y1,
                z1,
                x2,
                y2,
                z2,
                u_elem12,
                geom_name,
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
                force_beam_column3d_scratch,
                k_elem12,
                f_elem12,
            )
        else:
            if sec.type == "ElasticSection3d":
                if elem_type == ElementTypeTag.ForceBeamColumn3d:
                    force_beam_column3d_global_tangent_and_internal(
                        e,
                        x1,
                        y1,
                        z1,
                        x2,
                        y2,
                        z2,
                        u_elem12,
                        geom_name,
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
                        force_beam_column3d_scratch,
                        k_elem12,
                        f_elem12,
                    )
                else:
                    disp_beam_column3d_global_tangent_and_internal(
                        e,
                        x1,
                        y1,
                        z1,
                        x2,
                        y2,
                        z2,
                        u_elem12,
                        geom_name,
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
                        k_elem12,
                        f_elem12,
                    )
            elif sec.type == "FiberSection3d":
                var sec_index = fiber_section3d_index_by_id[elem_section_ids[e]]
                if sec_index < 0 or sec_index >= len(fiber_section3d_defs):
                    abort("fiber section not found")
                var integration_name = _beam_integration_name_from_tag(
                    elem_integration_tags[e]
                )
                _profile_scope_open(
                    do_profile,
                    events,
                    events_need_comma,
                    frame_assemble_fiber,
                    t0,
                )
                if elem_type == ElementTypeTag.ForceBeamColumn3d:
                    force_beam_column3d_fiber_global_tangent_and_internal(
                        e,
                        x1,
                        y1,
                        z1,
                        x2,
                        y2,
                        z2,
                        u_elem12,
                        geom_name,
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
                        integration_name,
                        elem_num_int_pts[e],
                        sec.G,
                        sec.J,
                        force_basic_q,
                        force_basic_offsets[e],
                        force_basic_counts[e],
                        force_beam_column3d_scratch,
                        k_elem12,
                        f_elem12,
                    )
                else:
                    beam_column3d_fiber_global_tangent_and_internal(
                        e,
                        x1,
                        y1,
                        z1,
                        x2,
                        y2,
                        z2,
                        u_elem12,
                        geom_name,
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
                        integration_name,
                        elem_num_int_pts[e],
                        sec.G,
                        sec.J,
                        k_elem12,
                        f_elem12,
                    )
                _profile_scope_close(
                    do_profile,
                    events,
                    events_need_comma,
                    frame_assemble_fiber,
                    t0,
                )
            else:
                abort("3d beam requires ElasticSection3d or FiberSection3d")
        for a in range(12):
            var Aidx = dof_map12[a]
            _scatter_add_row_unrolled4(K, Aidx, k_elem12[a], dof_map12, 12)
            F_int[Aidx] += f_elem12[a]


fn _assemble_frame3d_element(
    e: Int,
    elem: ElementInput,
    nodes: List[NodeInput],
    elem_dof_offsets: List[Int],
    elem_dof_pool: List[Int],
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
    fiber_section3d_defs: List[FiberSection3dDef],
    fiber_section3d_cells: List[FiberCell],
    fiber_section3d_index_by_id: List[Int],
    mut force_beam_column3d_scratch: ForceBeamColumn3dScratch,
    mut dof_map12: List[Int],
    mut K: List[List[Float64]],
    mut F_int: List[Float64],
) raises:
    var elem_type = elem.type_tag
    if (
        elem_type != ElementTypeTag.ElasticBeamColumn3d
        and elem_type != ElementTypeTag.ForceBeamColumn3d
        and elem_type != ElementTypeTag.DispBeamColumn3d
    ):
        return

    var node1 = nodes[elem.node_index_1]
    var node2 = nodes[elem.node_index_2]
    var sec = sections_by_id[elem.section]
    var dof_offset = elem_dof_offsets[e]
    var u_elem12: List[Float64] = []
    u_elem12.resize(12, 0.0)
    for i in range(12):
        dof_map12[i] = elem_dof_pool[dof_offset + i]
        u_elem12[i] = u[dof_map12[i]]
    var k_elem12: List[List[Float64]] = []
    var f_elem12: List[Float64] = []
    if elem_type == ElementTypeTag.ElasticBeamColumn3d:
        if sec.type != "ElasticSection3d":
            abort("elasticBeamColumn3d requires ElasticSection3d")
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
            sec.E,
            sec.A,
            sec.Iy,
            sec.Iz,
            sec.G,
            sec.J,
            force_beam_column3d_scratch,
            k_elem12,
            f_elem12,
        )
    else:
        if sec.type == "ElasticSection3d":
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
                    sec.E,
                    sec.A,
                    sec.Iy,
                    sec.Iz,
                    sec.G,
                    sec.J,
                    force_beam_column3d_scratch,
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
                    sec.E,
                    sec.A,
                    sec.Iy,
                    sec.Iz,
                    sec.G,
                    sec.J,
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
                    force_beam_column3d_scratch,
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
