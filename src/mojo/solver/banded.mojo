from collections import List
from os import abort
from python import PythonObject

from solver.dof import node_dof_index
from solver.run_case.input_types import ElementInput
from strut_io import py_len


fn banded_matrix(n: Int, bw: Int) -> List[List[Float64]]:
    var width = bw * 2 + 1
    var mat: List[List[Float64]] = []
    for _ in range(n):
        var row: List[Float64] = []
        row.resize(width, 0.0)
        mat.append(row^)
    return mat^


fn banded_add(mut mat: List[List[Float64]], bw: Int, i: Int, j: Int, value: Float64):
    var offset = j - i + bw
    var width = bw * 2 + 1
    if offset < 0 or offset >= width:
        abort("banded_add out of band")
    mat[i][offset] += value


fn banded_gaussian_elimination(
    mut mat: List[List[Float64]], bw: Int, mut b: List[Float64]
) -> List[Float64]:
    var n = len(b)
    if n == 0:
        return []
    for k in range(n):
        var pivot = mat[k][bw]
        if pivot == 0.0:
            abort("singular matrix")
        var j_end = k + bw
        if j_end > n - 1:
            j_end = n - 1
        for j in range(k + 1, j_end + 1):
            var idx = j - k + bw
            mat[k][idx] /= pivot
        b[k] /= pivot
        mat[k][bw] = 1.0

        var i_end = k + bw
        if i_end > n - 1:
            i_end = n - 1
        for i in range(k + 1, i_end + 1):
            var idx_ik = k - i + bw
            var factor = mat[i][idx_ik]
            if factor == 0.0:
                continue
            var row_end = i + bw
            if row_end > j_end:
                row_end = j_end
            if row_end > n - 1:
                row_end = n - 1
            for j in range(k + 1, row_end + 1):
                var idx_ij = j - i + bw
                var idx_kj = j - k + bw
                mat[i][idx_ij] -= factor * mat[k][idx_kj]
            b[i] -= factor * b[k]
            mat[i][idx_ik] = 0.0

    var x: List[Float64] = []
    x.resize(n, 0.0)
    var i = n - 1
    while True:
        var s = b[i]
        var j_end = i + bw
        if j_end > n - 1:
            j_end = n - 1
        for j in range(i + 1, j_end + 1):
            var idx = j - i + bw
            s -= mat[i][idx] * x[j]
        x[i] = s
        if i == 0:
            break
        i -= 1

    return x^


fn estimate_bandwidth(
    elements: PythonObject,
    id_to_index: List[Int],
    ndf: Int,
    free_index: List[Int],
) raises -> Int:
    var max_bw = 0
    var elem_count = py_len(elements)
    for e in range(elem_count):
        var elem = elements[e]
        var elem_nodes = elem["nodes"]
        var node_len = py_len(elem_nodes)
        if node_len == 0:
            continue
        var elem_dofs: List[Int] = []
        for i in range(node_len):
            var node_id = Int(elem_nodes[i])
            if node_id >= len(id_to_index):
                abort("element node id out of range")
            var node_idx = id_to_index[node_id]
            if node_idx < 0:
                abort("element node not found")
            for dof in range(1, ndf + 1):
                var dof_idx = node_dof_index(node_idx, dof, ndf)
                var free_idx = free_index[dof_idx]
                if free_idx >= 0:
                    elem_dofs.append(free_idx)
        var dof_len = len(elem_dofs)
        for a in range(dof_len):
            var ia = elem_dofs[a]
            for b in range(a + 1, dof_len):
                var diff = ia - elem_dofs[b]
                if diff < 0:
                    diff = -diff
                if diff > max_bw:
                    max_bw = diff
    return max_bw


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


fn estimate_bandwidth_typed(elements: List[ElementInput], free_index: List[Int]) -> Int:
    var max_bw = 0
    for e in range(len(elements)):
        var elem = elements[e]
        var dof_count = elem.dof_count
        for a in range(dof_count):
            var dof_a = _elem_dof(elem, a)
            if dof_a < 0 or dof_a >= len(free_index):
                continue
            var ia = free_index[dof_a]
            if ia < 0:
                continue
            for b in range(a + 1, dof_count):
                var dof_b = _elem_dof(elem, b)
                if dof_b < 0 or dof_b >= len(free_index):
                    continue
                var ib = free_index[dof_b]
                if ib < 0:
                    continue
                var diff = ia - ib
                if diff < 0:
                    diff = -diff
                if diff > max_bw:
                    max_bw = diff
    return max_bw
