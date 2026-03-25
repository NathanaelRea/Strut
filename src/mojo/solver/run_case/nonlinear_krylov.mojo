from collections import List

from solver.simd_contiguous import dot_float64_contiguous


@always_inline
fn default_krylov_max_dim(max_dim: Int) -> Int:
    if max_dim < 1:
        return 3
    return max_dim


fn copy_krylov_vector(src: List[Float64], count: Int) -> List[Float64]:
    var dst: List[Float64] = []
    dst.resize(count, 0.0)
    for i in range(count):
        dst[i] = src[i]
    return dst^


@always_inline
fn _dense_flat_index(size: Int, row: Int, col: Int) -> Int:
    return row * size + col


fn _ensure_dense_flat_storage(mut values: List[Float64], size: Int):
    var count = size * size
    if len(values) != count:
        values.resize(count, 0.0)


fn _solve_dense_flat_system(
    matrix_flat: List[Float64],
    rhs: List[Float64],
    mut solution_out: List[Float64],
    mut factor_work: List[Float64],
    mut rhs_work: List[Float64],
) -> Bool:
    var n = len(rhs)
    if len(matrix_flat) != n * n:
        return False
    solution_out.resize(n, 0.0)
    _ensure_dense_flat_storage(factor_work, n)
    rhs_work.resize(n, 0.0)
    for i in range(n * n):
        factor_work[i] = matrix_flat[i]
    for i in range(n):
        rhs_work[i] = rhs[i]

    var eps = 1.0e-18
    for i in range(n):
        var pivot = i
        var max_val = abs(factor_work[_dense_flat_index(n, i, i)])
        for row in range(i + 1, n):
            var candidate = abs(factor_work[_dense_flat_index(n, row, i)])
            if candidate > max_val:
                max_val = candidate
                pivot = row
        if max_val <= eps:
            return False
        if pivot != i:
            for col in range(n):
                var idx_a = _dense_flat_index(n, i, col)
                var idx_b = _dense_flat_index(n, pivot, col)
                var tmp = factor_work[idx_a]
                factor_work[idx_a] = factor_work[idx_b]
                factor_work[idx_b] = tmp
            var rhs_tmp = rhs_work[i]
            rhs_work[i] = rhs_work[pivot]
            rhs_work[pivot] = rhs_tmp

        var diag = factor_work[_dense_flat_index(n, i, i)]
        for row in range(i + 1, n):
            var factor = factor_work[_dense_flat_index(n, row, i)] / diag
            factor_work[_dense_flat_index(n, row, i)] = 0.0
            if factor == 0.0:
                continue
            for col in range(i + 1, n):
                factor_work[_dense_flat_index(n, row, col)] -= (
                    factor * factor_work[_dense_flat_index(n, i, col)]
                )
            rhs_work[row] -= factor * rhs_work[i]

    for i in range(n - 1, -1, -1):
        var sum = rhs_work[i]
        for col in range(i + 1, n):
            sum -= (
                factor_work[_dense_flat_index(n, i, col)] * solution_out[col]
            )
        var diag = factor_work[_dense_flat_index(n, i, i)]
        if abs(diag) <= eps:
            return False
        solution_out[i] = sum / diag
    return True


fn krylov_apply_acceleration(
    mut update: List[Float64],
    history_v: List[List[Float64]],
    mut history_av: List[List[Float64]],
    count: Int,
    mut normal_matrix_flat: List[Float64],
    mut normal_rhs: List[Float64],
    mut coeffs: List[Float64],
    mut factor_work: List[Float64],
    mut rhs_work: List[Float64],
) -> Bool:
    var k = len(history_v)
    if k == 0:
        return True
    if len(history_av) != k:
        return False

    var latest = k - 1
    for i in range(count):
        history_av[latest][i] -= update[i]

    _ensure_dense_flat_storage(normal_matrix_flat, k)
    normal_rhs.resize(k, 0.0)
    coeffs.resize(k, 0.0)
    for i in range(k * k):
        normal_matrix_flat[i] = 0.0

    var max_diag = 0.0
    for i in range(k):
        var rhs_value = dot_float64_contiguous(history_av[i], update, count)
        normal_rhs[i] = rhs_value
        for j in range(i, k):
            var value = dot_float64_contiguous(
                history_av[i], history_av[j], count
            )
            normal_matrix_flat[_dense_flat_index(k, i, j)] = value
            normal_matrix_flat[_dense_flat_index(k, j, i)] = value
            if i == j and value > max_diag:
                max_diag = value
        if max_diag < normal_matrix_flat[_dense_flat_index(k, i, i)]:
            max_diag = normal_matrix_flat[_dense_flat_index(k, i, i)]
    var regularization = 1.0e-18
    if max_diag > 0.0:
        regularization = max_diag * 1.0e-12
        if regularization < 1.0e-18:
            regularization = 1.0e-18
    for i in range(k):
        normal_matrix_flat[_dense_flat_index(k, i, i)] += regularization

    if not _solve_dense_flat_system(
        normal_matrix_flat,
        normal_rhs,
        coeffs,
        factor_work,
        rhs_work,
    ):
        return False

    for j in range(k):
        var coeff = coeffs[j]
        if coeff == 0.0:
            continue
        for i in range(count):
            update[i] += coeff * (history_v[j][i] - history_av[j][i])
    return True


fn krylov_push_iteration_state(
    base_direction: List[Float64],
    applied_update: List[Float64],
    count: Int,
    mut history_v: List[List[Float64]],
    mut history_av: List[List[Float64]],
):
    history_v.append(copy_krylov_vector(applied_update, count))
    history_av.append(copy_krylov_vector(base_direction, count))
