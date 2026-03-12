from collections import List

from elements import quad4_plane_stress_stiffness, shell4_mindlin_stiffness
from solver.profile import RuntimeProfileMetrics, _profile_metrics_note_element_timing
from solver.assembly.stiffness_internal_shared import (
    _elem_dof,
    _scatter_add_and_dot_row_simd,
)
from solver.run_case.input_types import ElementInput, MaterialInput, NodeInput, SectionInput
from tag_types import ElementTypeTag
from time import perf_counter_ns


fn _assemble_surface_soa_indices(
    quad_elem_indices: List[Int],
    shell_elem_indices: List[Int],
    node_x: List[Float64],
    node_y: List[Float64],
    node_z: List[Float64],
    elem_dof_offsets: List[Int],
    elem_dof_pool: List[Int],
    elem_node_offsets: List[Int],
    elem_node_pool: List[Int],
    elem_primary_material_ids: List[Int],
    elem_section_ids: List[Int],
    elem_thickness: List[Float64],
    materials_by_id: List[MaterialInput],
    sections_by_id: List[SectionInput],
    u: List[Float64],
    mut K: List[List[Float64]],
    mut F_int: List[Float64],
    mut runtime_metrics: RuntimeProfileMetrics,
):
    for idx in range(len(quad_elem_indices)):
        var e = quad_elem_indices[idx]
        var t_elem_start = 0
        if runtime_metrics.enabled:
            t_elem_start = Int(perf_counter_ns())
        var node_offset = elem_node_offsets[e]
        var mat = materials_by_id[elem_primary_material_ids[e]]
        var x = [node_x[elem_node_pool[node_offset]], node_x[elem_node_pool[node_offset + 1]], node_x[elem_node_pool[node_offset + 2]], node_x[elem_node_pool[node_offset + 3]]]
        var y = [node_y[elem_node_pool[node_offset]], node_y[elem_node_pool[node_offset + 1]], node_y[elem_node_pool[node_offset + 2]], node_y[elem_node_pool[node_offset + 3]]]
        var k_global = quad4_plane_stress_stiffness(mat.E, mat.nu, elem_thickness[e], x, y)
        var dof_offset = elem_dof_offsets[e]
        var dof_map = [
            elem_dof_pool[dof_offset],
            elem_dof_pool[dof_offset + 1],
            elem_dof_pool[dof_offset + 2],
            elem_dof_pool[dof_offset + 3],
            elem_dof_pool[dof_offset + 4],
            elem_dof_pool[dof_offset + 5],
            elem_dof_pool[dof_offset + 6],
            elem_dof_pool[dof_offset + 7],
        ]
        for a in range(8):
            var Aidx = dof_map[a]
            F_int[Aidx] += _scatter_add_and_dot_row_simd(K, Aidx, k_global[a], dof_map, u, 8)
        if runtime_metrics.enabled:
            _profile_metrics_note_element_timing(
                runtime_metrics,
                ElementTypeTag.FourNodeQuad,
                Int(perf_counter_ns()) - t_elem_start,
            )

    for idx in range(len(shell_elem_indices)):
        var e = shell_elem_indices[idx]
        var t_elem_start = 0
        if runtime_metrics.enabled:
            t_elem_start = Int(perf_counter_ns())
        var node_offset = elem_node_offsets[e]
        var sec = sections_by_id[elem_section_ids[e]]
        var x = [node_x[elem_node_pool[node_offset]], node_x[elem_node_pool[node_offset + 1]], node_x[elem_node_pool[node_offset + 2]], node_x[elem_node_pool[node_offset + 3]]]
        var y = [node_y[elem_node_pool[node_offset]], node_y[elem_node_pool[node_offset + 1]], node_y[elem_node_pool[node_offset + 2]], node_y[elem_node_pool[node_offset + 3]]]
        var z = [node_z[elem_node_pool[node_offset]], node_z[elem_node_pool[node_offset + 1]], node_z[elem_node_pool[node_offset + 2]], node_z[elem_node_pool[node_offset + 3]]]
        var k_global = shell4_mindlin_stiffness(sec.E, sec.nu, sec.h, x, y, z)
        var dof_offset = elem_dof_offsets[e]
        var dof_map = [
            elem_dof_pool[dof_offset],
            elem_dof_pool[dof_offset + 1],
            elem_dof_pool[dof_offset + 2],
            elem_dof_pool[dof_offset + 3],
            elem_dof_pool[dof_offset + 4],
            elem_dof_pool[dof_offset + 5],
            elem_dof_pool[dof_offset + 6],
            elem_dof_pool[dof_offset + 7],
            elem_dof_pool[dof_offset + 8],
            elem_dof_pool[dof_offset + 9],
            elem_dof_pool[dof_offset + 10],
            elem_dof_pool[dof_offset + 11],
            elem_dof_pool[dof_offset + 12],
            elem_dof_pool[dof_offset + 13],
            elem_dof_pool[dof_offset + 14],
            elem_dof_pool[dof_offset + 15],
            elem_dof_pool[dof_offset + 16],
            elem_dof_pool[dof_offset + 17],
            elem_dof_pool[dof_offset + 18],
            elem_dof_pool[dof_offset + 19],
            elem_dof_pool[dof_offset + 20],
            elem_dof_pool[dof_offset + 21],
            elem_dof_pool[dof_offset + 22],
            elem_dof_pool[dof_offset + 23],
        ]
        for a in range(24):
            var Aidx = dof_map[a]
            F_int[Aidx] += _scatter_add_and_dot_row_simd(K, Aidx, k_global[a], dof_map, u, 24)
        if runtime_metrics.enabled:
            _profile_metrics_note_element_timing(
                runtime_metrics,
                ElementTypeTag.Shell,
                Int(perf_counter_ns()) - t_elem_start,
            )


fn _assemble_surface_element(
    elem: ElementInput,
    nodes: List[NodeInput],
    materials_by_id: List[MaterialInput],
    sections_by_id: List[SectionInput],
    u: List[Float64],
    mut K: List[List[Float64]],
    mut F_int: List[Float64],
):
    if elem.type_tag == ElementTypeTag.FourNodeQuad:
        var node1 = nodes[elem.node_index_1]
        var node2 = nodes[elem.node_index_2]
        var node3 = nodes[elem.node_index_3]
        var node4 = nodes[elem.node_index_4]
        var mat = materials_by_id[elem.material]
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
        var k_global = quad4_plane_stress_stiffness(mat.E, mat.nu, elem.thickness, x, y)
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
            F_int[Aidx] += _scatter_add_and_dot_row_simd(K, Aidx, k_global[a], dof_map, u, 8)
        return

    if elem.type_tag == ElementTypeTag.Shell:
        var node1 = nodes[elem.node_index_1]
        var node2 = nodes[elem.node_index_2]
        var node3 = nodes[elem.node_index_3]
        var node4 = nodes[elem.node_index_4]
        var sec = sections_by_id[elem.section]
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
        var k_global = shell4_mindlin_stiffness(sec.E, sec.nu, sec.h, x, y, z)
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
            F_int[Aidx] += _scatter_add_and_dot_row_simd(K, Aidx, k_global[a], dof_map, u, 24)
