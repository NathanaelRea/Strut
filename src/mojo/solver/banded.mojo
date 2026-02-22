from collections import List
from os import abort
from solver.run_case.input_types import ElementInput


@always_inline
fn _normalize_row_segment_simd4(
    mut row: List[Float64], start_idx: Int, end_idx: Int, inv_pivot: Float64
):
    if start_idx > end_idx:
        return
    var i = start_idx
    var inv_vec = SIMD[DType.float64, 4](inv_pivot)
    while i + 3 <= end_idx:
        var chunk = SIMD[DType.float64, 4](row[i], row[i + 1], row[i + 2], row[i + 3])
        chunk = chunk * inv_vec
        row[i] = chunk[0]
        row[i + 1] = chunk[1]
        row[i + 2] = chunk[2]
        row[i + 3] = chunk[3]
        i += 4
    while i <= end_idx:
        row[i] *= inv_pivot
        i += 1


@always_inline
fn _eliminate_row_segment_simd4(
    mut row_i: List[Float64],
    idx_ij_start: Int,
    row_k: List[Float64],
    idx_kj_start: Int,
    factor: Float64,
    count: Int,
):
    if count <= 0:
        return
    var t = 0
    var factor_vec = SIMD[DType.float64, 4](factor)
    while t + 3 < count:
        var a = SIMD[DType.float64, 4](
            row_i[idx_ij_start + t],
            row_i[idx_ij_start + t + 1],
            row_i[idx_ij_start + t + 2],
            row_i[idx_ij_start + t + 3],
        )
        var b = SIMD[DType.float64, 4](
            row_k[idx_kj_start + t],
            row_k[idx_kj_start + t + 1],
            row_k[idx_kj_start + t + 2],
            row_k[idx_kj_start + t + 3],
        )
        var out = a - factor_vec * b
        row_i[idx_ij_start + t] = out[0]
        row_i[idx_ij_start + t + 1] = out[1]
        row_i[idx_ij_start + t + 2] = out[2]
        row_i[idx_ij_start + t + 3] = out[3]
        t += 4
    while t < count:
        row_i[idx_ij_start + t] -= factor * row_k[idx_kj_start + t]
        t += 1


@always_inline
fn _dot_banded_row_tail_simd4(
    row: List[Float64],
    row_idx_start: Int,
    x: List[Float64],
    x_idx_start: Int,
    count: Int,
) -> Float64:
    if count <= 0:
        return 0.0
    var sum = 0.0
    var t = 0
    while t + 3 < count:
        var a = SIMD[DType.float64, 4](
            row[row_idx_start + t],
            row[row_idx_start + t + 1],
            row[row_idx_start + t + 2],
            row[row_idx_start + t + 3],
        )
        var b = SIMD[DType.float64, 4](
            x[x_idx_start + t], x[x_idx_start + t + 1], x[x_idx_start + t + 2], x[x_idx_start + t + 3]
        )
        sum += (a * b).reduce_add()
        t += 4
    while t < count:
        sum += row[row_idx_start + t] * x[x_idx_start + t]
        t += 1
    return sum


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
        _normalize_row_segment_simd4(mat[k], bw + 1, j_end - k + bw, inv_pivot)
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
                _eliminate_row_segment_simd4(
                    mat[i],
                    j_start - i + bw,
                    pivot_row,
                    j_start - k + bw,
                    factor,
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
            s -= _dot_banded_row_tail_simd4(mat[i], bw + 1, x, i + 1, count)
        x[i] = s
        if i == 0:
            break
        i -= 1

    return x^


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
