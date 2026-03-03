from collections import List
from os import abort

from elements import link_orientation_matrix
from materials import UniMaterialState
from solver.assembly.stiffness_internal_links import _zero_length_row
from solver.assembly.stiffness_internal_shared import (
    _elem_damp_material,
    _elem_dir,
    _elem_dof_map,
    _zero_matrix,
    _zero_vector,
)
from solver.run_case.input_types import (
    DampingInput,
    ElementInput,
    MaterialInput,
    NodeInput,
)
from solver.time_series import TimeSeriesInput, eval_time_series_input
from tag_types import ElementTypeTag


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
            ref state = uniaxial_states[state_index]
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
            ref state = uniaxial_states[state_index]
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
