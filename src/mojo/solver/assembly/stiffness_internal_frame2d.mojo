from collections import List
from os import abort

from elements import (
    ForceBeamColumn2dScratch,
    beam2d_element_load_global,
    beam2d_corotational_global_tangent_and_internal,
    beam2d_pdelta_global_stiffness,
    beam_global_stiffness,
    disp_beam_column2d_global_tangent_and_internal,
    force_beam_column2d_global_tangent_and_internal,
)
from materials import UniMaterialDef, UniMaterialState
from solver.assembly.stiffness_internal_shared import (
    _beam_integration_name_from_tag,
    _elem_dof,
    _geom_transf_name_from_tag,
    _profile_scope_close,
    _profile_scope_open,
    _scatter_add_and_dot_row_simd,
    _scatter_add_row,
)
from solver.profile import (
    PROFILE_FRAME_ASSEMBLE_FIBER_GEOMETRY,
    PROFILE_FRAME_ASSEMBLE_FIBER_INTERNAL_FORCE,
    PROFILE_FRAME_ASSEMBLE_FIBER_MATRIX_SCATTER,
    PROFILE_FRAME_ASSEMBLE_FIBER_SECTION_RESPONSE,
)
from solver.run_case.input_types import (
    ElementInput,
    ElementLoadInput,
    NodeInput,
    SectionInput,
)
from sections import FiberCell, FiberSection2dDef
from tag_types import ElementTypeTag, GeomTransfTag


fn _assemble_frame2d_soa_indices(
    frame2d_elem_indices: List[Int],
    node_x: List[Float64],
    node_y: List[Float64],
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
    fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    mut force_beam_column2d_scratch: ForceBeamColumn2dScratch,
    mut dof_map6: List[Int],
    mut u_elem6: List[Float64],
    mut K: List[List[Float64]],
    mut F_int: List[Float64],
    do_profile: Bool,
    t0: Int,
    mut events: String,
    mut events_need_comma: Bool,
    frame_assemble_fiber: Int,
) raises:
    for idx in range(len(frame2d_elem_indices)):
        var e = frame2d_elem_indices[idx]
        _profile_scope_open(
            do_profile,
            events,
            events_need_comma,
            PROFILE_FRAME_ASSEMBLE_FIBER_GEOMETRY,
            t0,
        )
        var node_offset = elem_node_offsets[e]
        var i1 = elem_node_pool[node_offset]
        var i2 = elem_node_pool[node_offset + 1]
        var x1 = node_x[i1]
        var y1 = node_y[i1]
        var x2 = node_x[i2]
        var y2 = node_y[i2]
        var sec = sections_by_id[elem_section_ids[e]]
        var dof_offset = elem_dof_offsets[e]
        for i in range(6):
            dof_map6[i] = elem_dof_pool[dof_offset + i]
        var elem_type = elem_type_tags[e]
        _profile_scope_close(
            do_profile,
            events,
            events_need_comma,
            PROFILE_FRAME_ASSEMBLE_FIBER_GEOMETRY,
            t0,
        )
        if elem_type == ElementTypeTag.ElasticBeamColumn2d:
            var k_global: List[List[Float64]] = []
            var f_elem: List[Float64] = []
            var f_load_global = beam2d_element_load_global(
                element_loads,
                elem_load_offsets,
                elem_load_pool,
                e,
                load_scale,
                x1,
                y1,
                x2,
                y2,
            )
            if elem_geom_tags[e] == GeomTransfTag.Linear:
                k_global = beam_global_stiffness(sec.E, sec.A, sec.I, x1, y1, x2, y2)
            elif elem_geom_tags[e] == GeomTransfTag.PDelta:
                for i in range(6):
                    u_elem6[i] = u[dof_map6[i]]
                k_global = beam2d_pdelta_global_stiffness(
                    sec.E, sec.A, sec.I, x1, y1, x2, y2, u_elem6
                )
            elif elem_geom_tags[e] == GeomTransfTag.Corotational:
                for i in range(6):
                    u_elem6[i] = u[dof_map6[i]]
                beam2d_corotational_global_tangent_and_internal(
                    sec.E, sec.A, sec.I, x1, y1, x2, y2, u_elem6, k_global, f_elem
                )
            else:
                abort("unsupported geomTransf tag")

            if elem_geom_tags[e] == GeomTransfTag.Corotational:
                for a in range(6):
                    var Aidx = dof_map6[a]
                    _scatter_add_row[6](K, Aidx, k_global[a], dof_map6)
                    F_int[Aidx] += f_elem[a] - f_load_global[a]
            else:
                for a in range(6):
                    var Aidx = dof_map6[a]
                    F_int[Aidx] += _scatter_add_and_dot_row_simd(
                        K, Aidx, k_global[a], dof_map6, u, 6
                    ) - f_load_global[a]
            continue

        for i in range(6):
            u_elem6[i] = u[dof_map6[i]]
        var k_global: List[List[Float64]] = []
        var f_global: List[Float64] = []
        if sec.type == "ElasticSection2d":
            var f_load_global = beam2d_element_load_global(
                element_loads,
                elem_load_offsets,
                elem_load_pool,
                e,
                load_scale,
                x1,
                y1,
                x2,
                y2,
            )
            if elem_geom_tags[e] == GeomTransfTag.Linear:
                k_global = beam_global_stiffness(sec.E, sec.A, sec.I, x1, y1, x2, y2)
            elif elem_geom_tags[e] == GeomTransfTag.PDelta:
                k_global = beam2d_pdelta_global_stiffness(
                    sec.E, sec.A, sec.I, x1, y1, x2, y2, u_elem6
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
            for a in range(6):
                var Aidx = dof_map6[a]
                _scatter_add_row[6](K, Aidx, k_global[a], dof_map6)
                F_int[Aidx] += f_global[a] - f_load_global[a]
            continue

        var sec_index = fiber_section_index_by_id[elem_section_ids[e]]
        var sec_def = fiber_section_defs[sec_index]
        var elem_offset = elem_uniaxial_offsets[e]
        var elem_state_count = elem_uniaxial_counts[e]
        var geom_name = _geom_transf_name_from_tag(elem_geom_tags[e])
        var integration_name = _beam_integration_name_from_tag(elem_integration_tags[e])
        _profile_scope_open(
            do_profile,
            events,
            events_need_comma,
            frame_assemble_fiber,
            t0,
        )
        _profile_scope_open(
            do_profile,
            events,
            events_need_comma,
            PROFILE_FRAME_ASSEMBLE_FIBER_SECTION_RESPONSE,
            t0,
        )
        var k_elem6: List[List[Float64]] = []
        var f_elem6: List[Float64] = []
        if elem_type == ElementTypeTag.ForceBeamColumn2d:
            force_beam_column2d_global_tangent_and_internal(
                e,
                x1,
                y1,
                x2,
                y2,
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
                geom_name,
                integration_name,
                elem_num_int_pts[e],
                force_basic_q,
                force_basic_offsets[e],
                force_basic_counts[e],
                force_beam_column2d_scratch,
                k_elem6,
                f_elem6,
            )
        else:
            disp_beam_column2d_global_tangent_and_internal(
                e,
                x1,
                y1,
                x2,
                y2,
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
                geom_name,
                integration_name,
                elem_num_int_pts[e],
                k_elem6,
                f_elem6,
            )
        _profile_scope_close(
            do_profile,
            events,
            events_need_comma,
            PROFILE_FRAME_ASSEMBLE_FIBER_SECTION_RESPONSE,
            t0,
        )
        _profile_scope_open(
            do_profile,
            events,
            events_need_comma,
            PROFILE_FRAME_ASSEMBLE_FIBER_MATRIX_SCATTER,
            t0,
        )
        for a in range(6):
            var Aidx = dof_map6[a]
            _scatter_add_row[6](K, Aidx, k_elem6[a], dof_map6)
        _profile_scope_close(
            do_profile,
            events,
            events_need_comma,
            PROFILE_FRAME_ASSEMBLE_FIBER_MATRIX_SCATTER,
            t0,
        )
        _profile_scope_open(
            do_profile,
            events,
            events_need_comma,
            PROFILE_FRAME_ASSEMBLE_FIBER_INTERNAL_FORCE,
            t0,
        )
        for a in range(6):
            var Aidx = dof_map6[a]
            F_int[Aidx] += f_elem6[a]
        _profile_scope_close(
            do_profile,
            events,
            events_need_comma,
            PROFILE_FRAME_ASSEMBLE_FIBER_INTERNAL_FORCE,
            t0,
        )
        _profile_scope_close(
            do_profile,
            events,
            events_need_comma,
            frame_assemble_fiber,
            t0,
        )


fn _assemble_frame2d_element(
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
    fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    mut force_beam_column2d_scratch: ForceBeamColumn2dScratch,
    mut dof_map6: List[Int],
    mut u_elem6: List[Float64],
    mut K: List[List[Float64]],
    mut F_int: List[Float64],
) raises:
    var elem_type = elem.type_tag
    if (
        elem_type != ElementTypeTag.ElasticBeamColumn2d
        and elem_type != ElementTypeTag.ForceBeamColumn2d
        and elem_type != ElementTypeTag.DispBeamColumn2d
    ):
        return

    var node1 = nodes[elem.node_index_1]
    var node2 = nodes[elem.node_index_2]
    var dof_offset = elem_dof_offsets[e]
    for i in range(6):
        dof_map6[i] = elem_dof_pool[dof_offset + i]

    if elem_type == ElementTypeTag.ElasticBeamColumn2d:
        var sec = sections_by_id[elem.section]
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
            for i in range(6):
                u_elem6[i] = u[dof_map6[i]]
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
        elif geom == "Corotational":
            for i in range(6):
                u_elem6[i] = u[dof_map6[i]]
            beam2d_corotational_global_tangent_and_internal(
                sec.E,
                sec.A,
                sec.I,
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
                _scatter_add_row[6](K, Aidx, k_global[a], dof_map6)
                F_int[Aidx] += f_elem[a] - f_load_global[a]
        else:
            for a in range(6):
                var Aidx = dof_map6[a]
                F_int[Aidx] += _scatter_add_and_dot_row_simd(
                    K, Aidx, k_global[a], dof_map6, u, 6
                ) - f_load_global[a]
        return

    var sec = sections_by_id[elem.section]
    for i in range(6):
        u_elem6[i] = u[dof_map6[i]]
    var k_global: List[List[Float64]] = []
    var f_global: List[Float64] = []
    if sec.type == "ElasticSection2d":
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
        for a in range(6):
            var Aidx = dof_map6[a]
            _scatter_add_row[6](K, Aidx, k_global[a], dof_map6)
            F_int[Aidx] += f_global[a] - f_load_global[a]
        return

    var sec_index = fiber_section_index_by_id[elem.section]
    var sec_def = fiber_section_defs[sec_index]
    var elem_offset = elem_uniaxial_offsets[e]
    var elem_state_count = elem_uniaxial_counts[e]
    var k_elem6: List[List[Float64]] = []
    var f_elem6: List[Float64] = []
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
            force_beam_column2d_scratch,
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
        _scatter_add_row[6](K, Aidx, k_elem6[a], dof_map6)
        F_int[Aidx] += f_elem6[a]
