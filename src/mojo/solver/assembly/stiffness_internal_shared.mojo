from collections import List
from os import abort
from sys import is_defined, simd_width_of
from time import perf_counter_ns

from solver.profile import _append_event
from solver.run_case.input_types import ElementInput
from tag_types import BeamIntegrationTag, ElementTypeTag, GeomTransfTag


fn _zero_vector(mut vec: List[Float64]):
    for i in range(len(vec)):
        vec[i] = 0.0


fn _zero_matrix(mut mat: List[List[Float64]]):
    for i in range(len(mat)):
        for j in range(len(mat[i])):
            mat[i][j] = 0.0


@always_inline
fn _profile_scope_open(
    do_profile: Bool,
    mut events: String,
    mut events_need_comma: Bool,
    frame: Int,
    t0: Int,
):
    @parameter
    if is_defined["STRUT_PROFILE"]():
        if do_profile:
            var start_us = Int(perf_counter_ns())
            _append_event(
                events, events_need_comma, "O", frame, (start_us - t0) // 1000
            )


@always_inline
fn _profile_scope_close(
    do_profile: Bool,
    mut events: String,
    mut events_need_comma: Bool,
    frame: Int,
    t0: Int,
):
    @parameter
    if is_defined["STRUT_PROFILE"]():
        if do_profile:
            var end_us = Int(perf_counter_ns())
            _append_event(
                events, events_need_comma, "C", frame, (end_us - t0) // 1000
            )


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
fn _geom_transf_name_from_tag(geom_tag: Int) -> String:
    if geom_tag == GeomTransfTag.Linear:
        return "Linear"
    if geom_tag == GeomTransfTag.PDelta:
        return "PDelta"
    if geom_tag == GeomTransfTag.Corotational:
        return "Corotational"
    abort("unsupported geomTransf tag")
    return ""


@always_inline
fn _beam_integration_name_from_tag(integration_tag: Int) -> String:
    if integration_tag == BeamIntegrationTag.Lobatto:
        return "Lobatto"
    if integration_tag == BeamIntegrationTag.Legendre:
        return "Legendre"
    if integration_tag == BeamIntegrationTag.Radau:
        return "Radau"
    abort("unsupported beam integration tag")
    return ""


@always_inline
fn _scatter_add_row[count: Int](
    mut K: List[List[Float64]],
    row_index: Int,
    k_row: List[Float64],
    dof_map: List[Int],
):
    @parameter
    for b in range(count):
        var bidx = dof_map[b]
        K[row_index][bidx] += k_row[b]


@always_inline
fn _scatter_add_and_dot_row_simd_impl[width: Int](
    mut K: List[List[Float64]],
    row_index: Int,
    k_row: List[Float64],
    dof_map: List[Int],
    u: List[Float64],
    count: Int,
) -> Float64:
    var sum = 0.0
    var b = 0
    while b + width <= count:
        var k_vec = SIMD[DType.float64, width](0.0)
        var u_vec = SIMD[DType.float64, width](0.0)
        @parameter
        for lane in range(width):
            var bidx = dof_map[b + lane]
            var kval = k_row[b + lane]
            K[row_index][bidx] += kval
            k_vec[lane] = kval
            u_vec[lane] = u[bidx]
        sum += (k_vec * u_vec).reduce_add()
        b += width
    while b < count:
        var bidx = dof_map[b]
        var kval = k_row[b]
        K[row_index][bidx] += kval
        sum += kval * u[bidx]
        b += 1
    return sum


@always_inline
fn _scatter_add_and_dot_row_simd(
    mut K: List[List[Float64]],
    row_index: Int,
    k_row: List[Float64],
    dof_map: List[Int],
    u: List[Float64],
    count: Int,
) -> Float64:
    return _scatter_add_and_dot_row_simd_impl[simd_width_of[DType.float64]()](
        K, row_index, k_row, dof_map, u, count
    )


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
