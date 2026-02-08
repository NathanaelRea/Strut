from collections import List
from math import sqrt
from os import abort
from python import Python, PythonObject

from elements import (
    beam2d_corotational_global_internal_force,
    beam2d_pdelta_global_stiffness,
    beam_global_stiffness,
    beam_uniform_load_global,
    force_beam_column2d_global_tangent_and_internal,
)
from linalg import gaussian_elimination
from materials import (
    UniMaterialDef,
    UniMaterialState,
    uni_mat_is_elastic,
    uniaxial_commit_all,
    uniaxial_revert_trial_all,
)
from solver.assembly import (
    assemble_global_stiffness,
    assemble_global_stiffness_banded,
    assemble_global_stiffness_and_internal,
)
from solver.banded import banded_gaussian_elimination, banded_matrix, estimate_bandwidth
from solver.dof import node_dof_index, require_dof_in_range
from solver.profile import (
    _append_event,
    _append_frame,
    _profile_enabled,
    _write_speedscope,
)
from solver.reorder import build_node_adjacency, rcm_order
from solver.time_series import eval_time_series, find_time_series, parse_time_series
from sections import FiberCell, FiberSection2dDef, append_fiber_section2d_from_json
from strut_io import py_len


fn _beam2d_element_force_global(
    elem: PythonObject,
    nodes: PythonObject,
    sections_by_id: List[PythonObject],
    id_to_index: List[Int],
    ndf: Int,
    u: List[Float64],
) raises -> List[Float64]:
    if ndf != 3:
        abort("elasticBeamColumn2d requires ndf=3")
    var n1 = Int(elem["nodes"][0])
    var n2 = Int(elem["nodes"][1])
    var i1 = id_to_index[n1]
    var i2 = id_to_index[n2]
    var node1 = nodes[i1]
    var node2 = nodes[i2]

    var sec_id = Int(elem["section"])
    if sec_id >= len(sections_by_id):
        abort("section not found")
    var sec = sections_by_id[sec_id]
    if sec is None:
        abort("section not found")
    if String(sec["type"]) == "FiberSection2d":
        abort(
            "element_force for elasticBeamColumn2d with FiberSection2d requires "
            "forceBeamColumn (not implemented)"
        )

    var params = sec["params"]
    var E = Float64(params["E"])
    var A = Float64(params["A"])
    var I = Float64(params["I"])

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

    var geom = String(elem.get("geomTransf", "Linear"))
    if geom == "Corotational":
        return beam2d_corotational_global_internal_force(
            E,
            A,
            I,
            Float64(node1["x"]),
            Float64(node1["y"]),
            Float64(node2["x"]),
            Float64(node2["y"]),
            u_elem,
        )

    var k_global: List[List[Float64]] = []
    if geom == "Linear":
        k_global = beam_global_stiffness(
            E,
            A,
            I,
            Float64(node1["x"]),
            Float64(node1["y"]),
            Float64(node2["x"]),
            Float64(node2["y"]),
        )
    elif geom == "PDelta":
        k_global = beam2d_pdelta_global_stiffness(
            E,
            A,
            I,
            Float64(node1["x"]),
            Float64(node1["y"]),
            Float64(node2["x"]),
            Float64(node2["y"]),
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
    elem: PythonObject,
    nodes: PythonObject,
    id_to_index: List[Int],
    ndf: Int,
    uniaxial_states: List[UniMaterialState],
    elem_uniaxial_offsets: List[Int],
    elem_uniaxial_counts: List[Int],
    elem_uniaxial_state_ids: List[Int],
) raises -> List[Float64]:
    if ndf != 2 and ndf != 3:
        abort("truss element_force requires ndf=2 or ndf=3")
    var n1 = Int(elem["nodes"][0])
    var n2 = Int(elem["nodes"][1])
    var i1 = id_to_index[n1]
    var i2 = id_to_index[n2]
    var node1 = nodes[i1]
    var node2 = nodes[i2]

    var offset = elem_uniaxial_offsets[elem_index]
    var count = elem_uniaxial_counts[elem_index]
    if count != 1:
        abort("truss requires one uniaxial material")
    var state_index = elem_uniaxial_state_ids[offset]
    var state = uniaxial_states[state_index]
    var A = Float64(elem["area"])
    var N = state.sig_c * A

    if ndf == 2:
        var dx = Float64(node2["x"]) - Float64(node1["x"])
        var dy = Float64(node2["y"]) - Float64(node1["y"])
        var L = sqrt(dx * dx + dy * dy)
        if L == 0.0:
            abort("zero-length element")
        var c = dx / L
        var s = dy / L
        return [-N * c, -N * s, N * c, N * s]

    var dx = Float64(node2["x"]) - Float64(node1["x"])
    var dy = Float64(node2["y"]) - Float64(node1["y"])
    var dz = Float64(node2["z"]) - Float64(node1["z"])
    var L = sqrt(dx * dx + dy * dy + dz * dz)
    if L == 0.0:
        abort("zero-length element")
    var l = dx / L
    var m = dy / L
    var n = dz / L
    return [-N * l, -N * m, -N * n, N * l, N * m, N * n]


fn _force_beam_column2d_element_force_global(
    elem_index: Int,
    elem: PythonObject,
    nodes: PythonObject,
    id_to_index: List[Int],
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
    var geom = String(elem.get("geomTransf", "Linear"))
    if geom != "Linear":
        abort("forceBeamColumn2d v1 supports geomTransf Linear only")
    var integration = String(elem.get("integration", "Lobatto"))
    if integration != "Lobatto":
        abort("forceBeamColumn2d v1 supports Lobatto integration only")
    var num_int_pts = Int(elem.get("num_int_pts", 3))
    if num_int_pts != 3:
        abort("forceBeamColumn2d v1 supports num_int_pts=3")

    var n1 = Int(elem["nodes"][0])
    var n2 = Int(elem["nodes"][1])
    var i1 = id_to_index[n1]
    var i2 = id_to_index[n2]
    var node1 = nodes[i1]
    var node2 = nodes[i2]

    var sec_id = Int(elem["section"])
    if sec_id >= len(fiber_section_index_by_id):
        abort("forceBeamColumn2d section not found")
    var sec_index = fiber_section_index_by_id[sec_id]
    if sec_index < 0 or sec_index >= len(fiber_section_defs):
        abort("forceBeamColumn2d requires FiberSection2d")
    var sec_def = fiber_section_defs[sec_index]

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

    var elem_offset = elem_uniaxial_offsets[elem_index]
    var elem_state_count = elem_uniaxial_counts[elem_index]
    var k_dummy: List[List[Float64]] = []
    var f_global: List[Float64] = []
    force_beam_column2d_global_tangent_and_internal(
        Float64(node1["x"]),
        Float64(node1["y"]),
        Float64(node2["x"]),
        Float64(node2["y"]),
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
    mut buffers: List[String],
    filename: String,
    line: String,
):
    for i in range(len(filenames)):
        if filenames[i] == filename:
            buffers[i] = buffers[i] + line
            return
    filenames.append(filename)
    buffers.append(line)


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


