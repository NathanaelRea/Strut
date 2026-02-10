from collections import List
from math import sqrt
from os import abort

from elements import (
    beam2d_corotational_global_internal_force,
    beam2d_pdelta_global_stiffness,
    beam_global_stiffness,
    force_beam_column2d_global_tangent_and_internal,
)
from materials import UniMaterialDef, UniMaterialState
from solver.dof import node_dof_index, require_dof_in_range
from solver.run_case.input_types import (
    ElementInput,
    NodeInput,
    RecorderInput,
    SectionInput,
)
from sections import FiberCell, FiberSection2dDef


fn _beam2d_element_force_global(
    elem: ElementInput,
    nodes: List[NodeInput],
    sections_by_id: List[SectionInput],
    ndf: Int,
    u: List[Float64],
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
            "forceBeamColumn2d"
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
    if geom == "Corotational":
        return beam2d_corotational_global_internal_force(
            E,
            A,
            I,
            node1.x,
            node1.y,
            node2.x,
            node2.y,
            u_elem,
        )

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

    var f_elem: List[Float64] = []
    f_elem.resize(6, 0.0)
    for i in range(6):
        var sum = 0.0
        for j in range(6):
            sum += k_global[i][j] * u_elem[j]
        f_elem[i] = sum
    return f_elem^


fn _truss_element_force_global(
    elem_index: Int,
    elem: ElementInput,
    nodes: List[NodeInput],
    ndf: Int,
    uniaxial_states: List[UniMaterialState],
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
    var state = uniaxial_states[state_index]
    var A = elem.area
    var N = state.sig_c * A

    if ndf == 2:
        var dx = node2.x - node1.x
        var dy = node2.y - node1.y
        var L = sqrt(dx * dx + dy * dy)
        if L == 0.0:
            abort("zero-length element")
        var c = dx / L
        var s = dy / L
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
) raises -> List[Float64]:
    if ndf != 3:
        abort("forceBeamColumn2d requires ndf=3")
    var geom = elem.geom_transf
    if geom != "Linear" and geom != "PDelta":
        abort("forceBeamColumn2d supports geomTransf Linear or PDelta")
    var integration = elem.integration
    if integration != "Lobatto":
        abort("forceBeamColumn2d supports Lobatto integration only")
    var num_int_pts = elem.num_int_pts
    if num_int_pts != 3 and num_int_pts != 5:
        abort("forceBeamColumn2d supports num_int_pts=3 or 5")

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
            abort("forceBeamColumn2d supports geomTransf Linear or PDelta")
        var f_global_elastic: List[Float64] = []
        f_global_elastic.resize(6, 0.0)
        for a in range(6):
            var sum = 0.0
            for b in range(6):
                sum += k_global[a][b] * u_elem[b]
            f_global_elastic[a] = sum
        return f_global_elastic^

    var sec_id = elem.section
    if sec_id >= len(fiber_section_index_by_id):
        abort("forceBeamColumn2d section not found")
    var sec_index = fiber_section_index_by_id[sec_id]
    if sec_index < 0 or sec_index >= len(fiber_section_defs):
        abort("forceBeamColumn2d fiber section not found")
    var sec_def = fiber_section_defs[sec_index]

    var elem_offset = elem_uniaxial_offsets[elem_index]
    var elem_state_count = elem_uniaxial_counts[elem_index]
    var k_dummy: List[List[Float64]] = []
    var f_global: List[Float64] = []
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
        num_int_pts,
        k_dummy,
        f_global,
    )
    return f_global^


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
    uniaxial_defs: List[UniMaterialDef],
    mut uniaxial_states: List[UniMaterialState],
    elem_uniaxial_offsets: List[Int],
    elem_uniaxial_counts: List[Int],
    elem_uniaxial_state_ids: List[Int],
) raises -> List[Float64]:
    if elem.type_tag == 1:
        return _beam2d_element_force_global(
            elem,
            nodes,
            sections_by_id,
            ndf,
            u,
        )
    if elem.type_tag == 2:
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
        )
    if elem.type_tag == 4:
        return _truss_element_force_global(
            elem_index,
            elem,
            nodes,
            ndf,
            uniaxial_states,
            elem_uniaxial_offsets,
            elem_uniaxial_counts,
            elem_uniaxial_state_ids,
        )
    abort(
        "element_force recorder supports truss, "
        "elasticBeamColumn2d, or forceBeamColumn2d only"
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
