from collections import List
from os import abort
from solver.simd_contiguous import (
    axpy_float64_segments,
    dot_float64_segments,
    scale_float64_segment,
)


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
    var width = bw * 2 + 1
    var pivot_row: List[Float64] = []
    pivot_row.resize(width, 0.0)
    for k in range(n):
        var pivot = mat[k][bw]
        if pivot == 0.0:
            abort("singular matrix")
        var inv_pivot = 1.0 / pivot
        var j_end = k + bw
        if j_end > n - 1:
            j_end = n - 1
        scale_float64_segment(mat[k], bw + 1, j_end - k, inv_pivot)
        b[k] *= inv_pivot
        mat[k][bw] = 1.0
        for c in range(width):
            pivot_row[c] = mat[k][c]

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
            var j_start = k + 1
            var count = row_end - j_start + 1
            if count > 0:
                axpy_float64_segments(
                    mat[i],
                    j_start - i + bw,
                    pivot_row,
                    j_start - k + bw,
                    -factor,
                    count,
                )
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
        var count = j_end - i
        if count > 0:
            s -= dot_float64_segments(mat[i], bw + 1, x, i + 1, count)
        x[i] = s
        if i == 0:
            break
        i -= 1

    return x^


fn estimate_bandwidth_typed(
    dof_offsets: List[Int], dof_pool: List[Int], free_index: List[Int]
) -> Int:
    var max_bw = 0
    for e in range(len(dof_offsets) - 1):
        var stop = dof_offsets[e + 1]
        for a in range(dof_offsets[e], stop):
            var dof_a = dof_pool[a]
            if dof_a < 0 or dof_a >= len(free_index):
                continue
            var ia = free_index[dof_a]
            if ia < 0:
                continue
            for b in range(a + 1, stop):
                var dof_b = dof_pool[b]
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
