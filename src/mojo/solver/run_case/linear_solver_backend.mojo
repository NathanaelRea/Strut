from collections import List
from math import sqrt
from os import abort

from solver.profile import RuntimeProfileMetrics
from solver.reorder import min_degree_order, rcm_order
from solver.run_case.input_types import AnalysisInput
from tag_types import AnalysisSystemTag


struct LinearSolverBackend(Movable):
    var system_tag: Int
    var free_count: Int
    var initialized: Bool
    var factorized: Bool
    var superlu_prefer_symmetric: Bool
    var superlu_enable_pivot: Bool
    var superlu_np_row: Int
    var superlu_np_col: Int
    var superlu_perm_spec: Int
    var umfpack_factor_once: Bool
    var umfpack_print_time: Bool
    var umfpack_lvalue_fact: Int
    var sparse_sym_ordering: Int
    var lu_matrix: List[List[Float64]]
    var chol_matrix: List[List[Float64]]
    var dense_matrix_flat: List[Float64]
    var lu_pivots: List[Int]
    var solve_work: List[Float64]
    var sparse_symbolic_ready: Bool
    var sparse_col_start: List[Int]
    var sparse_row_indices: List[Int]
    var sparse_values: List[Float64]
    var sparse_col_permutation: List[Int]
    var sparse_rhs_work: List[Float64]
    var sparse_solution_work: List[Float64]
    var sparse_factor_row_start: List[Int]
    var sparse_factor_col_indices: List[Int]
    var sparse_factor_values: List[Float64]
    var sparse_factor_diag_indices: List[Int]
    var sparse_row_permutation: List[Int]
    var band_general_half_bandwidth: Int
    var band_general_values: List[Float64]
    var band_general_dense_fallback: Bool
    var band_spd_half_bandwidth: Int
    var band_spd_values: List[Float64]
    var profile_symbolic_ready: Bool
    var profile_col_start: List[Int]
    var profile_first_row: List[Int]
    var profile_values: List[Float64]

    fn __init__(out self):
        self.system_tag = AnalysisSystemTag.Unknown
        self.free_count = 0
        self.initialized = False
        self.factorized = False
        self.superlu_prefer_symmetric = False
        self.superlu_enable_pivot = False
        self.superlu_np_row = -1
        self.superlu_np_col = -1
        self.superlu_perm_spec = -1
        self.umfpack_factor_once = False
        self.umfpack_print_time = False
        self.umfpack_lvalue_fact = -1
        self.sparse_sym_ordering = 0
        self.lu_matrix = []
        self.chol_matrix = []
        self.dense_matrix_flat = []
        self.lu_pivots = []
        self.solve_work = []
        self.sparse_symbolic_ready = False
        self.sparse_col_start = []
        self.sparse_row_indices = []
        self.sparse_values = []
        self.sparse_col_permutation = []
        self.sparse_rhs_work = []
        self.sparse_solution_work = []
        self.sparse_factor_row_start = []
        self.sparse_factor_col_indices = []
        self.sparse_factor_values = []
        self.sparse_factor_diag_indices = []
        self.sparse_row_permutation = []
        self.band_general_half_bandwidth = 0
        self.band_general_values = []
        self.band_general_dense_fallback = False
        self.band_spd_half_bandwidth = 0
        self.band_spd_values = []
        self.profile_symbolic_ready = False
        self.profile_col_start = []
        self.profile_first_row = []
        self.profile_values = []


fn _ensure_square_storage(mut mat: List[List[Float64]], n: Int):
    if len(mat) != n:
        mat.clear()
        for _ in range(n):
            var row: List[Float64] = []
            row.resize(n, 0.0)
            mat.append(row^)
        return
    for i in range(n):
        if len(mat[i]) != n:
            mat[i].resize(n, 0.0)


fn _zero_square_storage(mut mat: List[List[Float64]], n: Int):
    _ensure_square_storage(mat, n)
    for i in range(n):
        for j in range(n):
            mat[i][j] = 0.0


fn _ensure_vector_storage(mut vec: List[Float64], n: Int):
    if len(vec) != n:
        vec.resize(n, 0.0)


fn _dense_flat_index(n: Int, row: Int, col: Int) -> Int:
    return row * n + col


fn _ensure_dense_flat_storage(mut values: List[Float64], n: Int):
    var size = n * n
    if len(values) != size:
        values.resize(size, 0.0)


fn _zero_dense_flat_storage(mut values: List[Float64], n: Int):
    _ensure_dense_flat_storage(values, n)
    for i in range(n * n):
        values[i] = 0.0


fn _load_dense_flat_from_square_matrix(
    mut values: List[Float64], A: List[List[Float64]], n: Int
):
    _ensure_dense_flat_storage(values, n)
    for row in range(n):
        for col in range(n):
            values[_dense_flat_index(n, row, col)] = A[row][col]


fn _load_dense_flat_from_full_matrix(
    mut values: List[Float64], K_full: List[List[Float64]], free: List[Int], n: Int
):
    _ensure_dense_flat_storage(values, n)
    for row in range(n):
        var src_row = free[row]
        for col in range(n):
            values[_dense_flat_index(n, row, col)] = K_full[src_row][free[col]]


fn _swap_dense_flat_rows(mut values: List[Float64], n: Int, first: Int, second: Int):
    if first == second:
        return
    var first_base = first * n
    var second_base = second * n
    for col in range(n):
        var slot_first = first_base + col
        var slot_second = second_base + col
        var tmp = values[slot_first]
        values[slot_first] = values[slot_second]
        values[slot_second] = tmp


fn _validate_square_matrix_shape(A: List[List[Float64]], n: Int):
    if len(A) != n:
        abort("linear solver backend matrix row count mismatch")
    for i in range(n):
        if len(A[i]) != n:
            abort("linear solver backend matrix column count mismatch")


fn _try_factorize_dense_unsymmetric_loaded(mut backend: LinearSolverBackend) -> Bool:
    var n = backend.free_count
    if len(backend.lu_pivots) != n:
        backend.lu_pivots.resize(n, 0)
    for i in range(n):
        backend.lu_pivots[i] = i

    var eps = 1.0e-14
    for col in range(n):
        var pivot_row = col
        var pivot_abs = abs(backend.dense_matrix_flat[_dense_flat_index(n, col, col)])
        for row in range(col + 1, n):
            var candidate = abs(backend.dense_matrix_flat[_dense_flat_index(n, row, col)])
            if candidate > pivot_abs:
                pivot_abs = candidate
                pivot_row = row
        if pivot_abs <= eps:
            return False
        if pivot_row != col:
            _swap_dense_flat_rows(backend.dense_matrix_flat, n, col, pivot_row)
            var pivot_tmp = backend.lu_pivots[col]
            backend.lu_pivots[col] = backend.lu_pivots[pivot_row]
            backend.lu_pivots[pivot_row] = pivot_tmp
        var diag = backend.dense_matrix_flat[_dense_flat_index(n, col, col)]
        if abs(diag) <= eps:
            return False
        for row in range(col + 1, n):
            var row_col_slot = _dense_flat_index(n, row, col)
            var factor = backend.dense_matrix_flat[row_col_slot] / diag
            backend.dense_matrix_flat[row_col_slot] = factor
            if factor == 0.0:
                continue
            for rhs_col in range(col + 1, n):
                var row_slot = _dense_flat_index(n, row, rhs_col)
                var pivot_slot = _dense_flat_index(n, col, rhs_col)
                backend.dense_matrix_flat[row_slot] -= (
                    factor * backend.dense_matrix_flat[pivot_slot]
                )
    return True


fn _factorize_unsymmetric(mut backend: LinearSolverBackend, A: List[List[Float64]]):
    _load_dense_flat_from_square_matrix(backend.dense_matrix_flat, A, backend.free_count)
    if not _try_factorize_dense_unsymmetric_loaded(backend):
        abort("FullGeneral matrix is singular or near-singular")


fn _solve_dense_unsymmetric(
    mut backend: LinearSolverBackend, rhs: List[Float64], mut solution_out: List[Float64]
):
    var n = backend.free_count
    if len(solution_out) != n:
        solution_out.resize(n, 0.0)
    for i in range(n):
        var s = rhs[backend.lu_pivots[i]]
        for j in range(i):
            s -= backend.dense_matrix_flat[_dense_flat_index(n, i, j)] * backend.solve_work[j]
        backend.solve_work[i] = s

    for i in range(n - 1, -1, -1):
        var s = backend.solve_work[i]
        for j in range(i + 1, n):
            s -= backend.dense_matrix_flat[_dense_flat_index(n, i, j)] * solution_out[j]
        solution_out[i] = s / backend.dense_matrix_flat[_dense_flat_index(n, i, i)]


fn _solve_spd(
    mut backend: LinearSolverBackend, rhs: List[Float64], mut solution_out: List[Float64]
):
    var n = backend.free_count
    if len(solution_out) != n:
        solution_out.resize(n, 0.0)
    for i in range(n):
        var sum = rhs[i]
        for j in range(i):
            sum -= backend.chol_matrix[i][j] * backend.solve_work[j]
        var diag = backend.chol_matrix[i][i]
        if diag == 0.0:
            abort("singular cholesky factor")
        backend.solve_work[i] = sum / diag

    for i in range(n - 1, -1, -1):
        var sum = backend.solve_work[i]
        for j in range(i + 1, n):
            sum -= backend.chol_matrix[j][i] * solution_out[j]
        var diag = backend.chol_matrix[i][i]
        if diag == 0.0:
            abort("singular cholesky factor")
        solution_out[i] = sum / diag


fn _symmetric_entry(row: Int, col: Int, A: List[List[Float64]]) -> Float64:
    var lower = A[row][col]
    var upper = A[col][row]
    if abs(lower - upper) > 1.0e-10:
        abort("matrix is not symmetric")
    return lower


fn _append_unique_int(mut values: List[Int], candidate: Int):
    for i in range(len(values)):
        if values[i] == candidate:
            return
    values.append(candidate)


fn _insert_sorted_unique_int(mut values: List[Int], candidate: Int):
    for i in range(len(values)):
        if values[i] == candidate:
            return
        if values[i] > candidate:
            var out: List[Int] = []
            out.resize(len(values) + 1, 0)
            for j in range(i):
                out[j] = values[j]
            out[i] = candidate
            for j in range(i, len(values)):
                out[j + 1] = values[j]
            values = out^
            return
    values.append(candidate)


fn _identity_permutation(n: Int) -> List[Int]:
    var out: List[Int] = []
    out.resize(n, 0)
    for i in range(n):
        out[i] = i
    return out^


fn _is_valid_permutation(perm: List[Int], n: Int) -> Bool:
    if len(perm) != n:
        return False
    var seen: List[Bool] = []
    seen.resize(n, False)
    for i in range(n):
        var value = perm[i]
        if value < 0 or value >= n:
            return False
        if seen[value]:
            return False
        seen[value] = True
    return True


fn _sparse_entry_is_structural(row: Int, col: Int, value: Float64) -> Bool:
    if row == col:
        return True
    return value != 0.0


fn _sparse_symmetric_entry_is_structural(
    row: Int, col: Int, lower_value: Float64, upper_value: Float64
) -> Bool:
    if row == col:
        return True
    return lower_value != 0.0 or upper_value != 0.0


fn _detect_general_band_half_width(A: List[List[Float64]]) -> Int:
    var n = len(A)
    var max_bw = 0
    for row in range(n):
        for col in range(n):
            if not _sparse_entry_is_structural(row, col, A[row][col]):
                continue
            var diff = row - col
            if diff < 0:
                diff = -diff
            if diff > max_bw:
                max_bw = diff
    return max_bw


fn _detect_symmetric_band_half_width(A: List[List[Float64]]) -> Int:
    var n = len(A)
    var max_bw = 0
    for row in range(n):
        for col in range(row + 1):
            if not _sparse_symmetric_entry_is_structural(row, col, A[row][col], A[col][row]):
                continue
            var diff = row - col
            if diff > max_bw:
                max_bw = diff
    return max_bw


fn _ensure_banded_general_storage(mut backend: LinearSolverBackend, n: Int, half_bw: Int):
    var width = half_bw * 2 + 1
    var size = n * width
    if len(backend.band_general_values) != size:
        backend.band_general_values.resize(size, 0.0)


fn _ensure_banded_spd_storage(mut backend: LinearSolverBackend, n: Int, half_bw: Int):
    var width = half_bw + 1
    var size = n * width
    if len(backend.band_spd_values) != size:
        backend.band_spd_values.resize(size, 0.0)


fn _band_general_slot(half_bw: Int, row: Int, col: Int) -> Int:
    var offset = col - row + half_bw
    if offset < 0 or offset > half_bw * 2:
        return -1
    return row * (half_bw * 2 + 1) + offset


fn _band_spd_slot(half_bw: Int, row: Int, col: Int) -> Int:
    if col > row:
        return -1
    var offset = row - col
    if offset < 0 or offset > half_bw:
        return -1
    return row * (half_bw + 1) + offset


fn _zero_banded_general_storage(mut backend: LinearSolverBackend):
    for i in range(len(backend.band_general_values)):
        backend.band_general_values[i] = 0.0


fn _zero_banded_spd_storage(mut backend: LinearSolverBackend):
    for i in range(len(backend.band_spd_values)):
        backend.band_spd_values[i] = 0.0


fn _zero_profile_storage(mut backend: LinearSolverBackend):
    for i in range(len(backend.profile_values)):
        backend.profile_values[i] = 0.0


fn _zero_sparse_storage(mut backend: LinearSolverBackend):
    for i in range(len(backend.sparse_values)):
        backend.sparse_values[i] = 0.0


fn _update_banded_general_values_from_dense(mut backend: LinearSolverBackend, A: List[List[Float64]]):
    var n = len(A)
    var half_bw = backend.band_general_half_bandwidth
    _zero_banded_general_storage(backend)
    for i in range(n):
        var j0 = i - half_bw
        if j0 < 0:
            j0 = 0
        var j1 = i + half_bw
        if j1 > n - 1:
            j1 = n - 1
        for j in range(j0, j1 + 1):
            var slot = _band_general_slot(half_bw, i, j)
            backend.band_general_values[slot] = A[i][j]


fn _update_banded_spd_values_from_dense(mut backend: LinearSolverBackend, A: List[List[Float64]]):
    var n = len(A)
    var half_bw = backend.band_spd_half_bandwidth
    _zero_banded_spd_storage(backend)
    for i in range(n):
        var j0 = i - half_bw
        if j0 < 0:
            j0 = 0
        for j in range(j0, i + 1):
            var slot = _band_spd_slot(half_bw, i, j)
            backend.band_spd_values[slot] = _symmetric_entry(i, j, A)


fn _profile_first_row_for_col(A: List[List[Float64]], col: Int) -> Int:
    for row in range(col):
        if _sparse_symmetric_entry_is_structural(col, row, A[col][row], A[row][col]):
            return row
    return col


fn _build_profile_symbolic(mut backend: LinearSolverBackend, A: List[List[Float64]]):
    var n = len(A)
    backend.profile_col_start.resize(n + 1, 0)
    backend.profile_first_row.resize(n, 0)
    backend.profile_col_start[0] = 0
    for col in range(n):
        var first_row = _profile_first_row_for_col(A, col)
        backend.profile_first_row[col] = first_row
        backend.profile_col_start[col + 1] = (
            backend.profile_col_start[col] + (col - first_row + 1)
        )
    backend.profile_values.resize(backend.profile_col_start[n], 0.0)
    backend.profile_symbolic_ready = True


fn _dense_matches_profile_pattern(
    A: List[List[Float64]], col_start: List[Int], first_row: List[Int]
) -> Bool:
    var n = len(A)
    if len(first_row) != n:
        return False
    if len(col_start) != n + 1:
        return False
    if col_start[0] != 0:
        return False
    for col in range(n):
        var expected_first = _profile_first_row_for_col(A, col)
        if first_row[col] != expected_first:
            return False
        var expected_next = col_start[col] + (col - expected_first + 1)
        if col_start[col + 1] != expected_next:
            return False
    return True


fn _ensure_profile_symbolic(mut backend: LinearSolverBackend, A: List[List[Float64]]):
    var needs_symbolic = not backend.profile_symbolic_ready
    if not needs_symbolic:
        needs_symbolic = not _dense_matches_profile_pattern(
            A, backend.profile_col_start, backend.profile_first_row
        )
    if needs_symbolic:
        _build_profile_symbolic(backend, A)


fn _profile_value_index(col_start: List[Int], first_row: List[Int], col: Int, row: Int) -> Int:
    if col < 0 or col >= len(first_row):
        abort("profile index out of range")
    var first = first_row[col]
    if row < first or row > col:
        abort("profile index out of range")
    return col_start[col] + (row - first)


fn _update_profile_values_from_dense(mut backend: LinearSolverBackend, A: List[List[Float64]]):
    var n = len(A)
    if len(backend.profile_col_start) != n + 1:
        abort("profile symbolic layout mismatch")
    if len(backend.profile_first_row) != n:
        abort("profile symbolic layout mismatch")
    backend.profile_values.resize(backend.profile_col_start[n], 0.0)
    for col in range(n):
        var first = backend.profile_first_row[col]
        for row in range(first, col + 1):
            var idx = backend.profile_col_start[col] + (row - first)
            backend.profile_values[idx] = _symmetric_entry(col, row, A)


fn _factorize_profile_spd_loaded(mut backend: LinearSolverBackend):
    var n = backend.free_count
    for col in range(n):
        var first_col = backend.profile_first_row[col]
        for row in range(first_col, col):
            var idx_col_row = _profile_value_index(
                backend.profile_col_start, backend.profile_first_row, col, row
            )
            var sum = backend.profile_values[idx_col_row]
            var first_row = backend.profile_first_row[row]
            var k_start = first_col
            if first_row > k_start:
                k_start = first_row
            for k in range(k_start, row):
                var idx_col_k = _profile_value_index(
                    backend.profile_col_start, backend.profile_first_row, col, k
                )
                var idx_row_k = _profile_value_index(
                    backend.profile_col_start, backend.profile_first_row, row, k
                )
                sum -= backend.profile_values[idx_col_k] * backend.profile_values[idx_row_k]

            var idx_row_diag = _profile_value_index(
                backend.profile_col_start, backend.profile_first_row, row, row
            )
            var diag = backend.profile_values[idx_row_diag]
            if diag == 0.0:
                abort("ProfileSPD matrix is not positive definite")
            backend.profile_values[idx_col_row] = sum / diag

        var idx_diag = _profile_value_index(
            backend.profile_col_start, backend.profile_first_row, col, col
        )
        var sum_diag = backend.profile_values[idx_diag]
        for k in range(first_col, col):
            var idx_col_k = _profile_value_index(
                backend.profile_col_start, backend.profile_first_row, col, k
            )
            var l = backend.profile_values[idx_col_k]
            sum_diag -= l * l
        if sum_diag <= 0.0:
            abort("ProfileSPD matrix is not positive definite")
        backend.profile_values[idx_diag] = sqrt(sum_diag)


fn _factorize_profile_spd(mut backend: LinearSolverBackend, A: List[List[Float64]]):
    _ensure_profile_symbolic(backend, A)
    _update_profile_values_from_dense(backend, A)
    _factorize_profile_spd_loaded(backend)


fn _solve_profile_spd(
    mut backend: LinearSolverBackend, rhs: List[Float64], mut solution_out: List[Float64]
):
    var n = backend.free_count
    if len(solution_out) != n:
        solution_out.resize(n, 0.0)
    _ensure_vector_storage(backend.solve_work, n)

    for row in range(n):
        var first = backend.profile_first_row[row]
        var sum = rhs[row]
        for col in range(first, row):
            var idx = _profile_value_index(
                backend.profile_col_start, backend.profile_first_row, row, col
            )
            sum -= backend.profile_values[idx] * backend.solve_work[col]
        var idx_diag = _profile_value_index(
            backend.profile_col_start, backend.profile_first_row, row, row
        )
        var diag = backend.profile_values[idx_diag]
        if diag == 0.0:
            abort("singular profile cholesky factor")
        backend.solve_work[row] = sum / diag

    for i in range(n):
        solution_out[i] = 0.0
    for row in range(n - 1, -1, -1):
        var sum = backend.solve_work[row]
        for col in range(row + 1, n):
            if backend.profile_first_row[col] > row:
                continue
            var idx = _profile_value_index(
                backend.profile_col_start, backend.profile_first_row, col, row
            )
            sum -= backend.profile_values[idx] * solution_out[col]
        var idx_diag = _profile_value_index(
            backend.profile_col_start, backend.profile_first_row, row, row
        )
        var diag = backend.profile_values[idx_diag]
        if diag == 0.0:
            abort("singular profile cholesky factor")
        solution_out[row] = sum / diag


fn _copy_band_general_values_to_dense_flat(
    mut backend: LinearSolverBackend, band_values: List[Float64]
):
    var n = backend.free_count
    var half_bw = backend.band_general_half_bandwidth
    _zero_dense_flat_storage(backend.dense_matrix_flat, n)
    for row in range(n):
        var col0 = row - half_bw
        if col0 < 0:
            col0 = 0
        var col1 = row + half_bw
        if col1 > n - 1:
            col1 = n - 1
        for col in range(col0, col1 + 1):
            var slot = _band_general_slot(half_bw, row, col)
            if slot >= 0:
                backend.dense_matrix_flat[_dense_flat_index(n, row, col)] = band_values[slot]


fn _factorize_band_general_loaded(mut backend: LinearSolverBackend):
    var n = backend.free_count
    var half_bw = backend.band_general_half_bandwidth
    var pivot_tol = 1.0e-14
    var original_values = backend.band_general_values.copy()
    for k in range(n):
        var i1 = k + half_bw
        if i1 > n - 1:
            i1 = n - 1
        var pivot_row = k
        var pivot_abs = 0.0
        for row in range(k, i1 + 1):
            var row_slot = _band_general_slot(half_bw, row, k)
            if row_slot < 0:
                continue
            var candidate_abs = abs(backend.band_general_values[row_slot])
            if candidate_abs > pivot_abs:
                pivot_abs = candidate_abs
                pivot_row = row
        if pivot_abs <= pivot_tol:
            abort("BandGeneral matrix is singular or near-singular")
        if pivot_row != k:
            _copy_band_general_values_to_dense_flat(backend, original_values)
            if not _try_factorize_dense_unsymmetric_loaded(backend):
                abort("BandGeneral dense fallback matrix is singular or near-singular")
            backend.band_general_dense_fallback = True
            return
        var diag_slot = _band_general_slot(half_bw, k, k)
        var diag = backend.band_general_values[diag_slot]
        if abs(diag) <= pivot_tol:
            abort("BandGeneral matrix is singular or near-singular")
        var j1 = k + half_bw
        if j1 > n - 1:
            j1 = n - 1
        for i in range(k + 1, i1 + 1):
            var ik_slot = _band_general_slot(half_bw, i, k)
            if ik_slot < 0:
                continue
            var factor = backend.band_general_values[ik_slot] / diag
            backend.band_general_values[ik_slot] = factor
            if factor == 0.0:
                continue
            for j in range(k + 1, j1 + 1):
                var kj_slot = _band_general_slot(half_bw, k, j)
                var ij_slot = _band_general_slot(half_bw, i, j)
                if ij_slot < 0 or kj_slot < 0:
                    continue
                backend.band_general_values[ij_slot] -= factor * backend.band_general_values[kj_slot]
    backend.band_general_dense_fallback = False


fn _factorize_band_general(mut backend: LinearSolverBackend, A: List[List[Float64]]):
    backend.band_general_half_bandwidth = _detect_general_band_half_width(A)
    _ensure_banded_general_storage(backend, backend.free_count, backend.band_general_half_bandwidth)
    _update_banded_general_values_from_dense(backend, A)
    _factorize_band_general_loaded(backend)


fn _factorize_band_spd_loaded(mut backend: LinearSolverBackend):
    var n = backend.free_count
    var half_bw = backend.band_spd_half_bandwidth
    for row in range(n):
        var first_col = row - half_bw
        if first_col < 0:
            first_col = 0
        for col in range(first_col, row):
            var sum = backend.band_spd_values[_band_spd_slot(half_bw, row, col)]
            var k0 = first_col
            var col_first = col - half_bw
            if col_first < 0:
                col_first = 0
            if col_first > k0:
                k0 = col_first
            for k in range(k0, col):
                var row_k_slot = _band_spd_slot(half_bw, row, k)
                var col_k_slot = _band_spd_slot(half_bw, col, k)
                if row_k_slot < 0 or col_k_slot < 0:
                    continue
                sum -= backend.band_spd_values[row_k_slot] * backend.band_spd_values[col_k_slot]
            var diag_slot = _band_spd_slot(half_bw, col, col)
            var diag = backend.band_spd_values[diag_slot]
            if diag == 0.0:
                abort("BandSPD matrix is not positive definite")
            backend.band_spd_values[_band_spd_slot(half_bw, row, col)] = sum / diag

        var diag_sum = backend.band_spd_values[_band_spd_slot(half_bw, row, row)]
        for k in range(first_col, row):
            var row_k_slot = _band_spd_slot(half_bw, row, k)
            if row_k_slot < 0:
                continue
            var value = backend.band_spd_values[row_k_slot]
            diag_sum -= value * value
        if diag_sum <= 0.0:
            abort("BandSPD matrix is not positive definite")
        backend.band_spd_values[_band_spd_slot(half_bw, row, row)] = sqrt(diag_sum)


fn _factorize_band_spd(mut backend: LinearSolverBackend, A: List[List[Float64]]):
    backend.band_spd_half_bandwidth = _detect_symmetric_band_half_width(A)
    _ensure_banded_spd_storage(backend, backend.free_count, backend.band_spd_half_bandwidth)
    _update_banded_spd_values_from_dense(backend, A)
    _factorize_band_spd_loaded(backend)


fn _solve_band_general(
    mut backend: LinearSolverBackend, rhs: List[Float64], mut solution_out: List[Float64]
):
    var n = backend.free_count
    var half_bw = backend.band_general_half_bandwidth
    if len(solution_out) != n:
        solution_out.resize(n, 0.0)
    _ensure_vector_storage(backend.solve_work, n)

    for i in range(n):
        var sum = rhs[i]
        var j0 = i - half_bw
        if j0 < 0:
            j0 = 0
        for j in range(j0, i):
            var slot = _band_general_slot(half_bw, i, j)
            if slot < 0:
                continue
            sum -= backend.band_general_values[slot] * backend.solve_work[j]
        backend.solve_work[i] = sum

    for i in range(n - 1, -1, -1):
        var sum = backend.solve_work[i]
        var j1 = i + half_bw
        if j1 > n - 1:
            j1 = n - 1
        for j in range(i + 1, j1 + 1):
            var slot = _band_general_slot(half_bw, i, j)
            if slot < 0:
                continue
            sum -= backend.band_general_values[slot] * solution_out[j]
        var diag = backend.band_general_values[_band_general_slot(half_bw, i, i)]
        if diag == 0.0:
            abort("singular band LU factor")
        solution_out[i] = sum / diag


fn _solve_band_spd(
    mut backend: LinearSolverBackend, rhs: List[Float64], mut solution_out: List[Float64]
):
    var n = backend.free_count
    var half_bw = backend.band_spd_half_bandwidth
    if len(solution_out) != n:
        solution_out.resize(n, 0.0)
    _ensure_vector_storage(backend.solve_work, n)

    for row in range(n):
        var first = row - half_bw
        if first < 0:
            first = 0
        var sum = rhs[row]
        for col in range(first, row):
            var slot = _band_spd_slot(half_bw, row, col)
            if slot < 0:
                continue
            sum -= backend.band_spd_values[slot] * backend.solve_work[col]
        var diag = backend.band_spd_values[_band_spd_slot(half_bw, row, row)]
        if diag == 0.0:
            abort("singular band cholesky factor")
        backend.solve_work[row] = sum / diag

    for i in range(n):
        solution_out[i] = backend.solve_work[i]
    for row in range(n - 1, -1, -1):
        var diag = backend.band_spd_values[_band_spd_slot(half_bw, row, row)]
        if diag == 0.0:
            abort("singular band cholesky factor")
        solution_out[row] /= diag
        var x = solution_out[row]
        var first = row - half_bw
        if first < 0:
            first = 0
        for col in range(first, row):
            var slot = _band_spd_slot(half_bw, row, col)
            if slot < 0:
                continue
            solution_out[col] -= backend.band_spd_values[slot] * x


fn _build_csc_pattern_from_dense(
    A: List[List[Float64]], mut col_start: List[Int], mut row_indices: List[Int]
):
    var n = len(A)
    col_start.resize(n + 1, 0)
    col_start[0] = 0
    for col in range(n):
        var count = 0
        for row in range(n):
            if _sparse_entry_is_structural(row, col, A[row][col]):
                count += 1
        col_start[col + 1] = col_start[col] + count

    var nnz = col_start[n]
    row_indices.resize(nnz, 0)
    var write_idx = 0
    for col in range(n):
        for row in range(n):
            if _sparse_entry_is_structural(row, col, A[row][col]):
                row_indices[write_idx] = row
                write_idx += 1


fn _build_symmetric_csc_pattern_from_dense(
    A: List[List[Float64]], mut col_start: List[Int], mut row_indices: List[Int]
):
    var n = len(A)
    col_start.resize(n + 1, 0)
    col_start[0] = 0
    for col in range(n):
        var count = 0
        for row in range(col, n):
            if _sparse_symmetric_entry_is_structural(row, col, A[row][col], A[col][row]):
                count += 1
        col_start[col + 1] = col_start[col] + count

    var nnz = col_start[n]
    row_indices.resize(nnz, 0)
    var write_idx = 0
    for col in range(n):
        for row in range(col, n):
            if _sparse_symmetric_entry_is_structural(row, col, A[row][col], A[col][row]):
                row_indices[write_idx] = row
                write_idx += 1


fn _dense_matches_csc_pattern(
    A: List[List[Float64]], col_start: List[Int], row_indices: List[Int]
) -> Bool:
    var n = len(A)
    if len(col_start) != n + 1:
        return False
    if len(row_indices) != col_start[n]:
        return False

    var read_idx = 0
    for col in range(n):
        if col_start[col] != read_idx:
            return False
        for row in range(n):
            if _sparse_entry_is_structural(row, col, A[row][col]):
                if read_idx >= len(row_indices):
                    return False
                if row_indices[read_idx] != row:
                    return False
                read_idx += 1
        if col_start[col + 1] != read_idx:
            return False
    return read_idx == len(row_indices)


fn _dense_matches_symmetric_csc_pattern(
    A: List[List[Float64]], col_start: List[Int], row_indices: List[Int]
) -> Bool:
    var n = len(A)
    if len(col_start) != n + 1:
        return False
    if len(row_indices) != col_start[n]:
        return False

    var read_idx = 0
    for col in range(n):
        if col_start[col] != read_idx:
            return False
        for row in range(col, n):
            if _sparse_symmetric_entry_is_structural(row, col, A[row][col], A[col][row]):
                if read_idx >= len(row_indices):
                    return False
                if row_indices[read_idx] != row:
                    return False
                read_idx += 1
        if col_start[col + 1] != read_idx:
            return False
    return read_idx == len(row_indices)


fn _build_symmetric_adjacency_from_csc(
    n: Int, col_start: List[Int], row_indices: List[Int]
) -> List[List[Int]]:
    var adjacency: List[List[Int]] = []
    for _ in range(n):
        var row: List[Int] = []
        adjacency.append(row^)

    for col in range(n):
        for idx in range(col_start[col], col_start[col + 1]):
            var row = row_indices[idx]
            if row == col:
                continue
            _append_unique_int(adjacency[col], row)
            _append_unique_int(adjacency[row], col)
    return adjacency^


fn _build_superlu_column_permutation(
    backend: LinearSolverBackend, n: Int, col_start: List[Int], row_indices: List[Int]
) -> List[Int]:
    var perm_spec = backend.superlu_perm_spec
    if perm_spec < 0:
        perm_spec = 1
    if perm_spec == 0:
        return _identity_permutation(n)

    var adjacency = _build_symmetric_adjacency_from_csc(n, col_start, row_indices)
    if perm_spec == 2:
        var permutation = rcm_order(adjacency)
        if not _is_valid_permutation(permutation, n):
            return _identity_permutation(n)
        return permutation^
    if perm_spec == 1 or perm_spec == 3:
        if backend.superlu_prefer_symmetric:
            var permutation = min_degree_order(adjacency)
            if not _is_valid_permutation(permutation, n):
                return _identity_permutation(n)
            return permutation^
        var permutation = rcm_order(adjacency)
        if not _is_valid_permutation(permutation, n):
            return _identity_permutation(n)
        return permutation^

    var permutation = min_degree_order(adjacency)
    if not _is_valid_permutation(permutation, n):
        return _identity_permutation(n)
    return permutation^


fn _build_umfpack_column_permutation(
    backend: LinearSolverBackend, n: Int, col_start: List[Int], row_indices: List[Int]
) -> List[Int]:
    var adjacency = _build_symmetric_adjacency_from_csc(n, col_start, row_indices)
    if backend.umfpack_lvalue_fact == 0:
        var permutation = rcm_order(adjacency)
        if not _is_valid_permutation(permutation, n):
            return _identity_permutation(n)
        return permutation^

    var permutation = min_degree_order(adjacency)
    if not _is_valid_permutation(permutation, n):
        return _identity_permutation(n)
    return permutation^


fn _build_sparse_sym_column_permutation(
    backend: LinearSolverBackend, n: Int, col_start: List[Int], row_indices: List[Int]
) -> List[Int]:
    var ordering = backend.sparse_sym_ordering
    if ordering == 0:
        return _identity_permutation(n)

    var adjacency = _build_symmetric_adjacency_from_csc(n, col_start, row_indices)
    if ordering == 1:
        var permutation = min_degree_order(adjacency)
        if not _is_valid_permutation(permutation, n):
            return _identity_permutation(n)
        return permutation^
    if ordering == 2 or ordering == 3:
        # Ordering `2` maps to provisional nested dissection via RCM.
        var permutation = rcm_order(adjacency)
        if not _is_valid_permutation(permutation, n):
            return _identity_permutation(n)
        return permutation^
    return _identity_permutation(n)


fn _build_sparse_symbolic(mut backend: LinearSolverBackend, A: List[List[Float64]]):
    if backend.system_tag == AnalysisSystemTag.SparseSYM:
        _build_symmetric_csc_pattern_from_dense(
            A, backend.sparse_col_start, backend.sparse_row_indices
        )
    else:
        _build_csc_pattern_from_dense(A, backend.sparse_col_start, backend.sparse_row_indices)
    var n = backend.free_count
    if backend.system_tag == AnalysisSystemTag.SuperLU:
        backend.sparse_col_permutation = _build_superlu_column_permutation(
            backend, n, backend.sparse_col_start, backend.sparse_row_indices
        )
    elif backend.system_tag == AnalysisSystemTag.UmfPack:
        backend.sparse_col_permutation = _build_umfpack_column_permutation(
            backend, n, backend.sparse_col_start, backend.sparse_row_indices
        )
    elif backend.system_tag == AnalysisSystemTag.SparseSYM:
        backend.sparse_col_permutation = _build_sparse_sym_column_permutation(
            backend, n, backend.sparse_col_start, backend.sparse_row_indices
        )
    else:
        backend.sparse_col_permutation = _identity_permutation(n)

    if not _is_valid_permutation(backend.sparse_col_permutation, n):
        backend.sparse_col_permutation = _identity_permutation(n)

    backend.sparse_values.resize(len(backend.sparse_row_indices), 0.0)
    _ensure_vector_storage(backend.sparse_rhs_work, n)
    _ensure_vector_storage(backend.sparse_solution_work, n)
    backend.sparse_symbolic_ready = True


fn _ensure_sparse_symbolic(mut backend: LinearSolverBackend, A: List[List[Float64]]):
    var needs_symbolic = not backend.sparse_symbolic_ready
    if not needs_symbolic:
        if backend.system_tag == AnalysisSystemTag.SparseSYM:
            needs_symbolic = not _dense_matches_symmetric_csc_pattern(
                A, backend.sparse_col_start, backend.sparse_row_indices
            )
        else:
            needs_symbolic = not _dense_matches_csc_pattern(
                A, backend.sparse_col_start, backend.sparse_row_indices
            )
    if needs_symbolic:
        _build_sparse_symbolic(backend, A)


fn _update_sparse_values_from_dense(mut backend: LinearSolverBackend, A: List[List[Float64]]):
    var n = len(A)
    if len(backend.sparse_col_start) != n + 1:
        abort("sparse symbolic layout mismatch")
    if len(backend.sparse_row_indices) != backend.sparse_col_start[n]:
        abort("sparse symbolic layout mismatch")

    backend.sparse_values.resize(len(backend.sparse_row_indices), 0.0)
    if backend.system_tag == AnalysisSystemTag.SparseSYM:
        var symmetric_tol = 1.0e-10
        for col in range(n):
            for idx in range(backend.sparse_col_start[col], backend.sparse_col_start[col + 1]):
                var row = backend.sparse_row_indices[idx]
                var lower = A[row][col]
                var upper = A[col][row]
                if abs(lower - upper) > symmetric_tol:
                    abort("SparseSYM matrix is not symmetric")
                backend.sparse_values[idx] = lower
        return

    for col in range(n):
        for idx in range(backend.sparse_col_start[col], backend.sparse_col_start[col + 1]):
            var row = backend.sparse_row_indices[idx]
            backend.sparse_values[idx] = A[row][col]


fn _sparse_value_index(backend: LinearSolverBackend, row: Int, col: Int) -> Int:
    if col < 0 or col + 1 >= len(backend.sparse_col_start):
        abort("sparse symbolic column out of range")
    for idx in range(backend.sparse_col_start[col], backend.sparse_col_start[col + 1]):
        var stored_row = backend.sparse_row_indices[idx]
        if stored_row == row:
            return idx
        if stored_row > row:
            break
    abort("sparse symbolic layout missing structural entry")
    return -1


fn _build_inverse_permutation(permutation: List[Int], n: Int) -> List[Int]:
    var inverse: List[Int] = []
    inverse.resize(n, -1)
    for i in range(n):
        var src = permutation[i]
        if src < 0 or src >= n:
            abort("invalid sparse column permutation index")
        if inverse[src] != -1:
            abort("invalid sparse column permutation index")
        inverse[src] = i
    return inverse^


fn _append_sparse_row_value(
    mut row_cols: List[Int], mut row_vals: List[Float64], col: Int, value: Float64
):
    row_cols.append(col)
    row_vals.append(value)


fn _insert_sparse_row_value_sorted(
    mut row_cols: List[Int], mut row_vals: List[Float64], col: Int, value: Float64
):
    var out_cols: List[Int] = []
    var out_vals: List[Float64] = []
    var inserted = False
    var drop_tol = 1.0e-14
    for i in range(len(row_cols)):
        var current_col = row_cols[i]
        if not inserted and col < current_col:
            if abs(value) > drop_tol:
                out_cols.append(col)
                out_vals.append(value)
            inserted = True
        if current_col == col:
            var merged = row_vals[i] + value
            if abs(merged) > drop_tol:
                out_cols.append(current_col)
                out_vals.append(merged)
            inserted = True
            continue
        out_cols.append(current_col)
        out_vals.append(row_vals[i])
    if not inserted and abs(value) > drop_tol:
        out_cols.append(col)
        out_vals.append(value)
    row_cols = out_cols^
    row_vals = out_vals^


fn _sparse_row_value(cols: List[Int], vals: List[Float64], col: Int) -> Float64:
    for i in range(len(cols)):
        var current = cols[i]
        if current == col:
            return vals[i]
        if current > col:
            break
    return 0.0


fn _swap_sparse_factor_rows(
    mut row_cols_rows: List[List[Int]],
    mut row_vals_rows: List[List[Float64]],
    first: Int,
    second: Int,
):
    if first == second:
        return
    var cols_tmp = row_cols_rows[first].copy()
    row_cols_rows[first] = row_cols_rows[second].copy()
    row_cols_rows[second] = cols_tmp^
    var vals_tmp = row_vals_rows[first].copy()
    row_vals_rows[first] = row_vals_rows[second].copy()
    row_vals_rows[second] = vals_tmp^


fn _sparse_row_eliminate_with_pivot(
    pivot_col: Int,
    factor: Float64,
    pivot_cols: List[Int],
    pivot_vals: List[Float64],
    mut row_cols: List[Int],
    mut row_vals: List[Float64],
):
    var out_cols: List[Int] = []
    var out_vals: List[Float64] = []
    var row_idx = 0
    while row_idx < len(row_cols) and row_cols[row_idx] < pivot_col:
        out_cols.append(row_cols[row_idx])
        out_vals.append(row_vals[row_idx])
        row_idx += 1
    out_cols.append(pivot_col)
    out_vals.append(factor)
    if row_idx < len(row_cols) and row_cols[row_idx] == pivot_col:
        row_idx += 1
    var pivot_idx = 0
    while pivot_idx < len(pivot_cols) and pivot_cols[pivot_idx] <= pivot_col:
        pivot_idx += 1

    var drop_tol = 1.0e-14
    while row_idx < len(row_cols) or pivot_idx < len(pivot_cols):
        if pivot_idx >= len(pivot_cols) or (
            row_idx < len(row_cols) and row_cols[row_idx] < pivot_cols[pivot_idx]
        ):
            var value = row_vals[row_idx]
            if abs(value) > drop_tol:
                out_cols.append(row_cols[row_idx])
                out_vals.append(value)
            row_idx += 1
            continue
        if row_idx >= len(row_cols) or pivot_cols[pivot_idx] < row_cols[row_idx]:
            var update = -factor * pivot_vals[pivot_idx]
            if abs(update) > drop_tol:
                out_cols.append(pivot_cols[pivot_idx])
                out_vals.append(update)
            pivot_idx += 1
            continue

        var merged = row_vals[row_idx] - factor * pivot_vals[pivot_idx]
        if abs(merged) > drop_tol:
            out_cols.append(row_cols[row_idx])
            out_vals.append(merged)
        row_idx += 1
        pivot_idx += 1
    row_cols = out_cols^
    row_vals = out_vals^


fn _build_sparse_rows_from_csc(
    mut backend: LinearSolverBackend,
    mut row_cols_rows: List[List[Int]],
    mut row_vals_rows: List[List[Float64]],
):
    var n = backend.free_count
    row_cols_rows.clear()
    row_vals_rows.clear()
    for _ in range(n):
        var cols: List[Int] = []
        var vals: List[Float64] = []
        row_cols_rows.append(cols^)
        row_vals_rows.append(vals^)

    if len(backend.sparse_col_permutation) != n:
        backend.sparse_col_permutation = _identity_permutation(n)
    if not _is_valid_permutation(backend.sparse_col_permutation, n):
        backend.sparse_col_permutation = _identity_permutation(n)

    if backend.system_tag == AnalysisSystemTag.SparseSYM:
        var inverse_permutation = _build_inverse_permutation(backend.sparse_col_permutation, n)
        for perm_col in range(n):
            var src_col = backend.sparse_col_permutation[perm_col]
            for idx in range(backend.sparse_col_start[src_col], backend.sparse_col_start[src_col + 1]):
                var src_row = backend.sparse_row_indices[idx]
                var perm_row = inverse_permutation[src_row]
                _insert_sparse_row_value_sorted(
                    row_cols_rows[perm_row], row_vals_rows[perm_row], perm_col, backend.sparse_values[idx]
                )
                if perm_row != perm_col:
                    _insert_sparse_row_value_sorted(
                        row_cols_rows[perm_col], row_vals_rows[perm_col], perm_row, backend.sparse_values[idx]
                    )
        return

    for perm_col in range(n):
        var src_col = backend.sparse_col_permutation[perm_col]
        if src_col < 0 or src_col >= n:
            abort("invalid sparse column permutation index")
        for idx in range(backend.sparse_col_start[src_col], backend.sparse_col_start[src_col + 1]):
            var row = backend.sparse_row_indices[idx]
            _append_sparse_row_value(
                row_cols_rows[row], row_vals_rows[row], perm_col, backend.sparse_values[idx]
            )


fn _flatten_sparse_factor_rows(
    mut backend: LinearSolverBackend,
    row_cols_rows: List[List[Int]],
    row_vals_rows: List[List[Float64]],
    row_permutation: List[Int],
):
    var n = backend.free_count
    backend.sparse_factor_row_start.resize(n + 1, 0)
    backend.sparse_factor_diag_indices.resize(n, -1)
    backend.sparse_row_permutation = row_permutation.copy()

    var total = 0
    for row in range(n):
        backend.sparse_factor_row_start[row] = total
        total += len(row_cols_rows[row])
    backend.sparse_factor_row_start[n] = total
    backend.sparse_factor_col_indices.resize(total, 0)
    backend.sparse_factor_values.resize(total, 0.0)

    var write_idx = 0
    for row in range(n):
        for entry in range(len(row_cols_rows[row])):
            var col = row_cols_rows[row][entry]
            backend.sparse_factor_col_indices[write_idx] = col
            backend.sparse_factor_values[write_idx] = row_vals_rows[row][entry]
            if col == row:
                backend.sparse_factor_diag_indices[row] = write_idx
            write_idx += 1
        if backend.sparse_factor_diag_indices[row] < 0:
            abort("sparse factorization lost a diagonal entry")
    backend.sparse_factor_row_start[n] = write_idx


fn _factorize_sparse_unsymmetric_loaded(mut backend: LinearSolverBackend):
    var row_cols_rows: List[List[Int]] = []
    var row_vals_rows: List[List[Float64]] = []
    _build_sparse_rows_from_csc(backend, row_cols_rows, row_vals_rows)
    var n = backend.free_count
    var row_permutation = _identity_permutation(n)
    var pivot_tol = 1.0e-14

    for k in range(n):
        var pivot_row = k
        var pivot_value = _sparse_row_value(row_cols_rows[k], row_vals_rows[k], k)
        if backend.system_tag == AnalysisSystemTag.SuperLU and backend.superlu_enable_pivot:
            var max_value = abs(pivot_value)
            for row in range(k + 1, n):
                var candidate = _sparse_row_value(row_cols_rows[row], row_vals_rows[row], k)
                var candidate_abs = abs(candidate)
                if candidate_abs > max_value:
                    max_value = candidate_abs
                    pivot_row = row
            if max_value <= pivot_tol:
                abort("sparse matrix is singular or near-singular")
            _swap_sparse_factor_rows(row_cols_rows, row_vals_rows, k, pivot_row)
            var perm_tmp = row_permutation[k]
            row_permutation[k] = row_permutation[pivot_row]
            row_permutation[pivot_row] = perm_tmp
            pivot_value = _sparse_row_value(row_cols_rows[k], row_vals_rows[k], k)
        if abs(pivot_value) <= pivot_tol:
            abort("sparse matrix is singular or near-singular")

        var pivot_cols = row_cols_rows[k].copy()
        var pivot_vals = row_vals_rows[k].copy()
        for row in range(k + 1, n):
            var factor_value = _sparse_row_value(row_cols_rows[row], row_vals_rows[row], k)
            if factor_value == 0.0:
                continue
            var factor = factor_value / pivot_value
            _sparse_row_eliminate_with_pivot(
                k, factor, pivot_cols, pivot_vals, row_cols_rows[row], row_vals_rows[row]
            )
    _flatten_sparse_factor_rows(backend, row_cols_rows, row_vals_rows, row_permutation)


fn _factorize_sparse_unsymmetric(mut backend: LinearSolverBackend, A: List[List[Float64]]):
    _ensure_sparse_symbolic(backend, A)
    _update_sparse_values_from_dense(backend, A)
    _factorize_sparse_unsymmetric_loaded(backend)


fn _factorize_sparse_symmetric_loaded(mut backend: LinearSolverBackend):
    var row_cols_rows: List[List[Int]] = []
    var row_vals_rows: List[List[Float64]] = []
    _build_sparse_rows_from_csc(backend, row_cols_rows, row_vals_rows)
    var n = backend.free_count
    var row_permutation = _identity_permutation(n)
    var pivot_tol = 1.0e-14

    for k in range(n):
        var pivot_value = _sparse_row_value(row_cols_rows[k], row_vals_rows[k], k)
        if abs(pivot_value) <= pivot_tol:
            abort("SparseSYM matrix is singular or near-singular")

        var pivot_cols = row_cols_rows[k].copy()
        var pivot_vals = row_vals_rows[k].copy()
        for row in range(k + 1, n):
            var factor_value = _sparse_row_value(row_cols_rows[row], row_vals_rows[row], k)
            if factor_value == 0.0:
                continue
            var factor = factor_value / pivot_value
            _sparse_row_eliminate_with_pivot(
                k, factor, pivot_cols, pivot_vals, row_cols_rows[row], row_vals_rows[row]
            )
    _flatten_sparse_factor_rows(backend, row_cols_rows, row_vals_rows, row_permutation)


fn _factorize_sparse_symmetric(mut backend: LinearSolverBackend, A: List[List[Float64]]):
    _ensure_sparse_symbolic(backend, A)
    _update_sparse_values_from_dense(backend, A)
    _factorize_sparse_symmetric_loaded(backend)


fn _solve_sparse_unsymmetric(
    mut backend: LinearSolverBackend, rhs: List[Float64], mut solution_out: List[Float64]
):
    var n = backend.free_count
    if len(solution_out) != n:
        solution_out.resize(n, 0.0)
    _ensure_vector_storage(backend.sparse_rhs_work, n)
    _ensure_vector_storage(backend.sparse_solution_work, n)
    if len(backend.sparse_row_permutation) != n:
        backend.sparse_row_permutation = _identity_permutation(n)
    if len(backend.sparse_col_permutation) != n:
        backend.sparse_col_permutation = _identity_permutation(n)

    for row in range(n):
        backend.sparse_rhs_work[row] = rhs[backend.sparse_row_permutation[row]]

    for row in range(n):
        var sum = backend.sparse_rhs_work[row]
        var diag_idx = backend.sparse_factor_diag_indices[row]
        for idx in range(backend.sparse_factor_row_start[row], diag_idx):
            var col = backend.sparse_factor_col_indices[idx]
            sum -= backend.sparse_factor_values[idx] * backend.sparse_solution_work[col]
        backend.sparse_solution_work[row] = sum

    for row in range(n - 1, -1, -1):
        var diag_idx = backend.sparse_factor_diag_indices[row]
        var sum = backend.sparse_solution_work[row]
        for idx in range(diag_idx + 1, backend.sparse_factor_row_start[row + 1]):
            var col = backend.sparse_factor_col_indices[idx]
            sum -= backend.sparse_factor_values[idx] * backend.sparse_solution_work[col]
        var diag = backend.sparse_factor_values[diag_idx]
        if diag == 0.0:
            abort("singular sparse LU factor")
        backend.sparse_solution_work[row] = sum / diag

    for i in range(n):
        solution_out[i] = 0.0
    for perm_col in range(n):
        var src_col = backend.sparse_col_permutation[perm_col]
        if src_col < 0 or src_col >= n:
            abort("invalid sparse column permutation index")
        solution_out[src_col] = backend.sparse_solution_work[perm_col]


fn _solve_sparse_symmetric(
    mut backend: LinearSolverBackend, rhs: List[Float64], mut solution_out: List[Float64]
):
    var n = backend.free_count
    if len(solution_out) != n:
        solution_out.resize(n, 0.0)
    _ensure_vector_storage(backend.sparse_rhs_work, n)
    _ensure_vector_storage(backend.sparse_solution_work, n)
    if len(backend.sparse_col_permutation) != n:
        backend.sparse_col_permutation = _identity_permutation(n)
    if not _is_valid_permutation(backend.sparse_col_permutation, n):
        backend.sparse_col_permutation = _identity_permutation(n)

    for perm_col in range(n):
        var src_col = backend.sparse_col_permutation[perm_col]
        backend.sparse_rhs_work[perm_col] = rhs[src_col]
    if len(backend.sparse_row_permutation) != n:
        backend.sparse_row_permutation = _identity_permutation(n)
    for row in range(n):
        backend.sparse_solution_work[row] = backend.sparse_rhs_work[backend.sparse_row_permutation[row]]

    for row in range(n):
        var diag_idx = backend.sparse_factor_diag_indices[row]
        var sum = backend.sparse_solution_work[row]
        for idx in range(backend.sparse_factor_row_start[row], diag_idx):
            var col = backend.sparse_factor_col_indices[idx]
            sum -= backend.sparse_factor_values[idx] * backend.sparse_solution_work[col]
        backend.sparse_solution_work[row] = sum
    for row in range(n - 1, -1, -1):
        var diag_idx = backend.sparse_factor_diag_indices[row]
        var sum = backend.sparse_solution_work[row]
        for idx in range(diag_idx + 1, backend.sparse_factor_row_start[row + 1]):
            var col = backend.sparse_factor_col_indices[idx]
            sum -= backend.sparse_factor_values[idx] * backend.sparse_solution_work[col]
        var diag = backend.sparse_factor_values[diag_idx]
        if diag == 0.0:
            abort("singular sparse symmetric factor")
        backend.sparse_solution_work[row] = sum / diag
    for i in range(n):
        solution_out[i] = 0.0
    for perm_col in range(n):
        var src_col = backend.sparse_col_permutation[perm_col]
        solution_out[src_col] = backend.sparse_solution_work[perm_col]


fn _validate_full_matrix_and_free(
    K_full: List[List[Float64]], free: List[Int], free_count: Int
):
    if len(free) != free_count:
        abort("linear solver backend free index size mismatch")
    var total_dofs = len(K_full)
    for row in range(total_dofs):
        if len(K_full[row]) != total_dofs:
            abort("linear solver backend full matrix must be square")
    for i in range(free_count):
        var dof = free[i]
        if dof < 0 or dof >= total_dofs:
            abort("linear solver backend free index out of range")


fn _free_matrix_entry(
    K_full: List[List[Float64]], free: List[Int], row: Int, col: Int
) -> Float64:
    return K_full[free[row]][free[col]]


fn _detect_band_half_width_from_elem_free_map(
    elem_free_offsets: List[Int], elem_free_pool: List[Int]
) -> Int:
    if len(elem_free_offsets) == 0:
        return 0
    if len(elem_free_pool) != elem_free_offsets[len(elem_free_offsets) - 1]:
        abort("linear solver backend invalid elem_free_pool size")
    var max_bw = 0
    var elem_count = len(elem_free_offsets) - 1
    for e in range(elem_count):
        var start = elem_free_offsets[e]
        var end = elem_free_offsets[e + 1]
        for a in range(start, end):
            var ia = elem_free_pool[a]
            if ia < 0:
                continue
            for b in range(a + 1, end):
                var ib = elem_free_pool[b]
                if ib < 0:
                    continue
                var diff = ia - ib
                if diff < 0:
                    diff = -diff
                if diff > max_bw:
                    max_bw = diff
    return max_bw


fn _build_profile_symbolic_from_elem_free_map(
    free_count: Int,
    elem_free_offsets: List[Int],
    elem_free_pool: List[Int],
    mut col_start: List[Int],
    mut first_row: List[Int],
):
    first_row.resize(free_count, 0)
    for col in range(free_count):
        first_row[col] = col
    var elem_count = len(elem_free_offsets) - 1
    for e in range(elem_count):
        var start = elem_free_offsets[e]
        var end = elem_free_offsets[e + 1]
        for a in range(start, end):
            var ia = elem_free_pool[a]
            if ia < 0:
                continue
            for b in range(start, end):
                var ib = elem_free_pool[b]
                if ib < 0:
                    continue
                var row = ia
                var col = ib
                if row > col:
                    row = ib
                    col = ia
                if row < first_row[col]:
                    first_row[col] = row
    col_start.resize(free_count + 1, 0)
    col_start[0] = 0
    for col in range(free_count):
        col_start[col + 1] = col_start[col] + (col - first_row[col] + 1)


fn _build_csc_pattern_from_elem_free_map(
    free_count: Int,
    elem_free_offsets: List[Int],
    elem_free_pool: List[Int],
    symmetric_only: Bool,
    mut col_start: List[Int],
    mut row_indices: List[Int],
):
    var rows_by_col: List[List[Int]] = []
    for col in range(free_count):
        var rows: List[Int] = []
        rows.append(col)
        rows_by_col.append(rows^)

    var elem_count = len(elem_free_offsets) - 1
    for e in range(elem_count):
        var start = elem_free_offsets[e]
        var end = elem_free_offsets[e + 1]
        for a in range(start, end):
            var ia = elem_free_pool[a]
            if ia < 0:
                continue
            for b in range(start, end):
                var ib = elem_free_pool[b]
                if ib < 0:
                    continue
                if symmetric_only:
                    var row = ia
                    var col = ib
                    if row < col:
                        row = ib
                        col = ia
                    _insert_sorted_unique_int(rows_by_col[col], row)
                else:
                    _insert_sorted_unique_int(rows_by_col[ib], ia)
    col_start.resize(free_count + 1, 0)
    col_start[0] = 0
    for col in range(free_count):
        col_start[col + 1] = col_start[col] + len(rows_by_col[col])

    row_indices.resize(col_start[free_count], 0)
    var write_idx = 0
    for col in range(free_count):
        for i in range(len(rows_by_col[col])):
            row_indices[write_idx] = rows_by_col[col][i]
            write_idx += 1


fn initialize_symbolic_from_element_free_map(
    mut backend: LinearSolverBackend,
    elem_free_offsets: List[Int],
    elem_free_pool: List[Int],
):
    if not backend.initialized:
        abort("linear solver backend not initialized")
    if len(elem_free_offsets) == 0:
        abort("linear solver backend invalid elem_free_offsets size")
    if len(elem_free_pool) != elem_free_offsets[len(elem_free_offsets) - 1]:
        abort("linear solver backend invalid elem_free_pool size")

    if backend.system_tag == AnalysisSystemTag.BandGeneral:
        backend.band_general_half_bandwidth = _detect_band_half_width_from_elem_free_map(
            elem_free_offsets, elem_free_pool
        )
        _ensure_banded_general_storage(
            backend, backend.free_count, backend.band_general_half_bandwidth
        )
        backend.band_general_dense_fallback = False
        return

    if backend.system_tag == AnalysisSystemTag.BandSPD:
        backend.band_spd_half_bandwidth = _detect_band_half_width_from_elem_free_map(
            elem_free_offsets, elem_free_pool
        )
        _ensure_banded_spd_storage(backend, backend.free_count, backend.band_spd_half_bandwidth)
        return

    if backend.system_tag == AnalysisSystemTag.ProfileSPD:
        _build_profile_symbolic_from_elem_free_map(
            backend.free_count,
            elem_free_offsets,
            elem_free_pool,
            backend.profile_col_start,
            backend.profile_first_row,
        )
        backend.profile_values.resize(backend.profile_col_start[backend.free_count], 0.0)
        backend.profile_symbolic_ready = True
        return

    if (
        backend.system_tag == AnalysisSystemTag.SuperLU
        or backend.system_tag == AnalysisSystemTag.UmfPack
        or backend.system_tag == AnalysisSystemTag.SparseSYM
    ):
        _build_csc_pattern_from_elem_free_map(
            backend.free_count,
            elem_free_offsets,
            elem_free_pool,
            backend.system_tag == AnalysisSystemTag.SparseSYM,
            backend.sparse_col_start,
            backend.sparse_row_indices,
        )
        if backend.system_tag == AnalysisSystemTag.SuperLU:
            backend.sparse_col_permutation = _build_superlu_column_permutation(
                backend, backend.free_count, backend.sparse_col_start, backend.sparse_row_indices
            )
        elif backend.system_tag == AnalysisSystemTag.UmfPack:
            backend.sparse_col_permutation = _build_umfpack_column_permutation(
                backend, backend.free_count, backend.sparse_col_start, backend.sparse_row_indices
            )
        else:
            backend.sparse_col_permutation = _build_sparse_sym_column_permutation(
                backend, backend.free_count, backend.sparse_col_start, backend.sparse_row_indices
            )
        if not _is_valid_permutation(backend.sparse_col_permutation, backend.free_count):
            backend.sparse_col_permutation = _identity_permutation(backend.free_count)
        backend.sparse_values.resize(len(backend.sparse_row_indices), 0.0)
        backend.sparse_symbolic_ready = True


fn initialize_symbolic_from_element_dof_map(
    mut backend: LinearSolverBackend,
    elem_dof_offsets: List[Int],
    elem_dof_pool: List[Int],
    free_index: List[Int],
):
    var elem_free_pool: List[Int] = []
    elem_free_pool.resize(len(elem_dof_pool), -1)
    for i in range(len(elem_dof_pool)):
        var dof = elem_dof_pool[i]
        if dof >= 0 and dof < len(free_index):
            elem_free_pool[i] = free_index[dof]
    initialize_symbolic_from_element_free_map(backend, elem_dof_offsets, elem_free_pool)


fn _force_refactor_every_solve(backend: LinearSolverBackend) -> Bool:
    if backend.system_tag == AnalysisSystemTag.UmfPack:
        return not backend.umfpack_factor_once
    return False


fn initialize_structure(
    mut backend: LinearSolverBackend, analysis: AnalysisInput, free_count: Int
):
    backend.system_tag = analysis.system_tag
    backend.free_count = free_count
    backend.initialized = True
    backend.factorized = False
    backend.superlu_prefer_symmetric = analysis.superlu_prefer_symmetric
    backend.superlu_enable_pivot = analysis.superlu_enable_pivot
    backend.superlu_np_row = analysis.superlu_np_row
    backend.superlu_np_col = analysis.superlu_np_col
    backend.superlu_perm_spec = analysis.superlu_perm_spec
    backend.umfpack_factor_once = analysis.umfpack_factor_once
    backend.umfpack_print_time = analysis.umfpack_print_time
    backend.umfpack_lvalue_fact = analysis.umfpack_lvalue_fact
    backend.sparse_sym_ordering = analysis.sparse_sym_ordering
    _ensure_square_storage(backend.chol_matrix, free_count)
    _ensure_dense_flat_storage(backend.dense_matrix_flat, free_count)
    if len(backend.lu_pivots) != free_count:
        backend.lu_pivots.resize(free_count, 0)
    _ensure_vector_storage(backend.solve_work, free_count)

    backend.sparse_symbolic_ready = False
    backend.sparse_col_start.clear()
    backend.sparse_row_indices.clear()
    backend.sparse_values.clear()
    backend.sparse_col_permutation = _identity_permutation(free_count)
    _ensure_vector_storage(backend.sparse_rhs_work, free_count)
    _ensure_vector_storage(backend.sparse_solution_work, free_count)
    backend.sparse_factor_row_start.clear()
    backend.sparse_factor_col_indices.clear()
    backend.sparse_factor_values.clear()
    backend.sparse_factor_diag_indices.clear()
    backend.sparse_row_permutation = _identity_permutation(free_count)

    backend.band_general_half_bandwidth = 0
    backend.band_general_values.clear()
    backend.band_general_dense_fallback = False
    backend.band_spd_half_bandwidth = 0
    backend.band_spd_values.clear()
    backend.profile_symbolic_ready = False
    backend.profile_col_start.clear()
    backend.profile_first_row.clear()
    backend.profile_values.clear()


fn _extract_free_submatrix(K_full: List[List[Float64]], free: List[Int]) -> List[List[Float64]]:
    var n = len(free)
    var A: List[List[Float64]] = []
    for i in range(n):
        var row: List[Float64] = []
        row.resize(n, 0.0)
        var src_row = free[i]
        for j in range(n):
            row[j] = K_full[src_row][free[j]]
        A.append(row^)
    return A^


fn _update_banded_general_values_from_full_matrix(
    mut backend: LinearSolverBackend,
    K_full: List[List[Float64]],
    free: List[Int],
):
    var n = backend.free_count
    var half_bw = backend.band_general_half_bandwidth
    _zero_banded_general_storage(backend)
    for row in range(n):
        var col0 = row - half_bw
        if col0 < 0:
            col0 = 0
        var col1 = row + half_bw
        if col1 > n - 1:
            col1 = n - 1
        var src_row = free[row]
        for col in range(col0, col1 + 1):
            var slot = _band_general_slot(half_bw, row, col)
            backend.band_general_values[slot] = K_full[src_row][free[col]]


fn _update_banded_spd_values_from_full_matrix(
    mut backend: LinearSolverBackend,
    K_full: List[List[Float64]],
    free: List[Int],
):
    var n = backend.free_count
    var half_bw = backend.band_spd_half_bandwidth
    _zero_banded_spd_storage(backend)
    for row in range(n):
        var col0 = row - half_bw
        if col0 < 0:
            col0 = 0
        var src_row = free[row]
        for col in range(col0, row + 1):
            var slot = _band_spd_slot(half_bw, row, col)
            var lower = K_full[src_row][free[col]]
            var upper = K_full[free[col]][src_row]
            if abs(lower - upper) > 1.0e-10:
                abort("BandSPD matrix is not symmetric")
            backend.band_spd_values[slot] = lower


fn _update_profile_values_from_full_matrix(
    mut backend: LinearSolverBackend,
    K_full: List[List[Float64]],
    free: List[Int],
):
    var n = backend.free_count
    if len(backend.profile_col_start) != n + 1 or len(backend.profile_first_row) != n:
        abort("profile symbolic layout mismatch")
    backend.profile_values.resize(backend.profile_col_start[n], 0.0)
    for col in range(n):
        var first = backend.profile_first_row[col]
        var src_col = free[col]
        for row in range(first, col + 1):
            var idx = backend.profile_col_start[col] + (row - first)
            var lower = K_full[src_col][free[row]]
            var upper = K_full[free[row]][src_col]
            if abs(lower - upper) > 1.0e-10:
                abort("ProfileSPD matrix is not symmetric")
            backend.profile_values[idx] = lower


fn _update_sparse_values_from_full_matrix(
    mut backend: LinearSolverBackend,
    K_full: List[List[Float64]],
    free: List[Int],
):
    var n = backend.free_count
    if len(backend.sparse_col_start) != n + 1:
        abort("sparse symbolic layout mismatch")
    if len(backend.sparse_row_indices) != backend.sparse_col_start[n]:
        abort("sparse symbolic layout mismatch")
    backend.sparse_values.resize(len(backend.sparse_row_indices), 0.0)
    if backend.system_tag == AnalysisSystemTag.SparseSYM:
        for col in range(n):
            var src_col = free[col]
            for idx in range(backend.sparse_col_start[col], backend.sparse_col_start[col + 1]):
                var row = backend.sparse_row_indices[idx]
                var src_row = free[row]
                var lower = K_full[src_row][src_col]
                var upper = K_full[src_col][src_row]
                if abs(lower - upper) > 1.0e-10:
                    abort("SparseSYM matrix is not symmetric")
                backend.sparse_values[idx] = lower
        return
    for col in range(n):
        var src_col = free[col]
        for idx in range(backend.sparse_col_start[col], backend.sparse_col_start[col + 1]):
            var row = backend.sparse_row_indices[idx]
            backend.sparse_values[idx] = K_full[free[row]][src_col]


fn _load_from_full_matrix(mut backend: LinearSolverBackend, K_full: List[List[Float64]], free: List[Int]):
    if backend.system_tag == AnalysisSystemTag.BandGeneral:
        if len(backend.band_general_values) == 0:
            abort("linear solver backend missing BandGeneral symbolic storage")
        _update_banded_general_values_from_full_matrix(backend, K_full, free)
        backend.band_general_dense_fallback = False
    elif backend.system_tag == AnalysisSystemTag.BandSPD:
        if len(backend.band_spd_values) == 0:
            abort("linear solver backend missing BandSPD symbolic storage")
        _update_banded_spd_values_from_full_matrix(backend, K_full, free)
    elif backend.system_tag == AnalysisSystemTag.ProfileSPD:
        if not backend.profile_symbolic_ready:
            abort("linear solver backend missing ProfileSPD symbolic storage")
        _update_profile_values_from_full_matrix(backend, K_full, free)
    elif backend.system_tag == AnalysisSystemTag.SuperLU:
        if not backend.sparse_symbolic_ready:
            abort("linear solver backend missing sparse symbolic storage")
        _update_sparse_values_from_full_matrix(backend, K_full, free)
    elif backend.system_tag == AnalysisSystemTag.UmfPack:
        if not backend.sparse_symbolic_ready:
            abort("linear solver backend missing sparse symbolic storage")
        _update_sparse_values_from_full_matrix(backend, K_full, free)
    elif backend.system_tag == AnalysisSystemTag.SparseSYM:
        if not backend.sparse_symbolic_ready:
            abort("linear solver backend missing SparseSYM symbolic storage")
        _update_sparse_values_from_full_matrix(backend, K_full, free)
    elif backend.system_tag == AnalysisSystemTag.FullGeneral:
        _load_dense_flat_from_full_matrix(
            backend.dense_matrix_flat, K_full, free, backend.free_count
        )
    else:
        abort("unsupported analysis system tag")
    backend.factorized = False


fn load_reduced_matrix(mut backend: LinearSolverBackend, A: List[List[Float64]]):
    if not backend.initialized:
        abort("linear solver backend not initialized")
    _validate_square_matrix_shape(A, backend.free_count)

    if backend.system_tag == AnalysisSystemTag.BandGeneral:
        if len(backend.band_general_values) == 0:
            backend.band_general_half_bandwidth = _detect_general_band_half_width(A)
            _ensure_banded_general_storage(
                backend, backend.free_count, backend.band_general_half_bandwidth
            )
        _update_banded_general_values_from_dense(backend, A)
        backend.band_general_dense_fallback = False
    elif backend.system_tag == AnalysisSystemTag.BandSPD:
        if len(backend.band_spd_values) == 0:
            backend.band_spd_half_bandwidth = _detect_symmetric_band_half_width(A)
            _ensure_banded_spd_storage(
                backend, backend.free_count, backend.band_spd_half_bandwidth
            )
        _update_banded_spd_values_from_dense(backend, A)
    elif backend.system_tag == AnalysisSystemTag.ProfileSPD:
        _ensure_profile_symbolic(backend, A)
        _update_profile_values_from_dense(backend, A)
    elif backend.system_tag == AnalysisSystemTag.SuperLU:
        _ensure_sparse_symbolic(backend, A)
        _update_sparse_values_from_dense(backend, A)
    elif backend.system_tag == AnalysisSystemTag.UmfPack:
        _ensure_sparse_symbolic(backend, A)
        _update_sparse_values_from_dense(backend, A)
    elif backend.system_tag == AnalysisSystemTag.SparseSYM:
        _ensure_sparse_symbolic(backend, A)
        _update_sparse_values_from_dense(backend, A)
    elif backend.system_tag == AnalysisSystemTag.FullGeneral:
        _load_dense_flat_from_square_matrix(backend.dense_matrix_flat, A, backend.free_count)
    else:
        abort("unsupported analysis system tag")
    backend.factorized = False


fn add_reduced_matrix(mut backend: LinearSolverBackend, A: List[List[Float64]], scale: Float64 = 1.0):
    if not backend.initialized:
        abort("linear solver backend not initialized")
    _validate_square_matrix_shape(A, backend.free_count)
    if scale == 0.0:
        return

    var n = backend.free_count
    if backend.system_tag == AnalysisSystemTag.BandGeneral:
        var half_bw = backend.band_general_half_bandwidth
        for row in range(n):
            var col0 = row - half_bw
            if col0 < 0:
                col0 = 0
            var col1 = row + half_bw
            if col1 > n - 1:
                col1 = n - 1
            for col in range(col0, col1 + 1):
                var slot = _band_general_slot(half_bw, row, col)
                backend.band_general_values[slot] += scale * A[row][col]
    elif backend.system_tag == AnalysisSystemTag.BandSPD:
        var half_bw = backend.band_spd_half_bandwidth
        for row in range(n):
            var col0 = row - half_bw
            if col0 < 0:
                col0 = 0
            for col in range(col0, row + 1):
                var lower = A[row][col]
                var upper = A[col][row]
                if abs(lower - upper) > 1.0e-10:
                    abort("BandSPD matrix is not symmetric")
                var slot = _band_spd_slot(half_bw, row, col)
                backend.band_spd_values[slot] += scale * lower
    elif backend.system_tag == AnalysisSystemTag.ProfileSPD:
        for col in range(n):
            var first = backend.profile_first_row[col]
            for row in range(first, col + 1):
                var lower = A[col][row]
                var upper = A[row][col]
                if abs(lower - upper) > 1.0e-10:
                    abort("ProfileSPD matrix is not symmetric")
                var idx = backend.profile_col_start[col] + (row - first)
                backend.profile_values[idx] += scale * lower
    elif backend.system_tag == AnalysisSystemTag.SuperLU or backend.system_tag == AnalysisSystemTag.UmfPack:
        for col in range(n):
            for idx in range(backend.sparse_col_start[col], backend.sparse_col_start[col + 1]):
                var row = backend.sparse_row_indices[idx]
                backend.sparse_values[idx] += scale * A[row][col]
    elif backend.system_tag == AnalysisSystemTag.SparseSYM:
        for col in range(n):
            for idx in range(backend.sparse_col_start[col], backend.sparse_col_start[col + 1]):
                var row = backend.sparse_row_indices[idx]
                var lower = A[row][col]
                var upper = A[col][row]
                if abs(lower - upper) > 1.0e-10:
                    abort("SparseSYM matrix is not symmetric")
                backend.sparse_values[idx] += scale * lower
    elif backend.system_tag == AnalysisSystemTag.FullGeneral:
        for row in range(n):
            for col in range(n):
                var slot = _dense_flat_index(n, row, col)
                backend.dense_matrix_flat[slot] += scale * A[row][col]
    else:
        abort("unsupported analysis system tag")
    backend.factorized = False


fn zero_loaded_matrix(mut backend: LinearSolverBackend):
    if not backend.initialized:
        abort("linear solver backend not initialized")
    if backend.system_tag == AnalysisSystemTag.BandGeneral:
        _zero_banded_general_storage(backend)
        backend.band_general_dense_fallback = False
    elif backend.system_tag == AnalysisSystemTag.BandSPD:
        _zero_banded_spd_storage(backend)
    elif backend.system_tag == AnalysisSystemTag.ProfileSPD:
        _zero_profile_storage(backend)
    elif (
        backend.system_tag == AnalysisSystemTag.SuperLU
        or backend.system_tag == AnalysisSystemTag.UmfPack
        or backend.system_tag == AnalysisSystemTag.SparseSYM
    ):
        _zero_sparse_storage(backend)
    elif backend.system_tag == AnalysisSystemTag.FullGeneral:
        _zero_dense_flat_storage(backend.dense_matrix_flat, backend.free_count)
    else:
        abort("unsupported analysis system tag")
    backend.factorized = False


fn add_element_matrix_from_pool(
    mut backend: LinearSolverBackend,
    elem_free_pool: List[Int],
    offset: Int,
    dof_count: Int,
    k_elem: List[List[Float64]],
    scale: Float64 = 1.0,
):
    if not backend.initialized:
        abort("linear solver backend not initialized")
    if scale == 0.0:
        return
    if len(k_elem) != dof_count:
        abort("linear solver backend element matrix row count mismatch")
    for row in range(dof_count):
        if len(k_elem[row]) != dof_count:
            abort("linear solver backend element matrix column count mismatch")
    if offset < 0 or offset + dof_count > len(elem_free_pool):
        abort("linear solver backend element free map range out of bounds")

    var symmetric_tol = 1.0e-10
    if backend.system_tag == AnalysisSystemTag.BandGeneral:
        var half_bw = backend.band_general_half_bandwidth
        for a in range(dof_count):
            var row = elem_free_pool[offset + a]
            if row < 0:
                continue
            for b in range(dof_count):
                var col = elem_free_pool[offset + b]
                if col < 0:
                    continue
                var slot = _band_general_slot(half_bw, row, col)
                if slot >= 0:
                    backend.band_general_values[slot] += scale * k_elem[a][b]
        backend.band_general_dense_fallback = False
    elif backend.system_tag == AnalysisSystemTag.BandSPD:
        var half_bw = backend.band_spd_half_bandwidth
        for a in range(dof_count):
            var row = elem_free_pool[offset + a]
            if row < 0:
                continue
            for b in range(dof_count):
                var col = elem_free_pool[offset + b]
                if col < 0:
                    continue
                var value = k_elem[a][b]
                if row < col:
                    if abs(value - k_elem[b][a]) > symmetric_tol:
                        abort("BandSPD matrix is not symmetric")
                    continue
                var slot = _band_spd_slot(half_bw, row, col)
                if slot >= 0:
                    backend.band_spd_values[slot] += scale * value
        backend.band_general_dense_fallback = False
    elif backend.system_tag == AnalysisSystemTag.ProfileSPD:
        for a in range(dof_count):
            var row = elem_free_pool[offset + a]
            if row < 0:
                continue
            for b in range(dof_count):
                var col = elem_free_pool[offset + b]
                if col < 0:
                    continue
                var value = k_elem[a][b]
                if row < col:
                    if abs(value - k_elem[b][a]) > symmetric_tol:
                        abort("ProfileSPD matrix is not symmetric")
                    continue
                var idx = _profile_value_index(
                    backend.profile_col_start, backend.profile_first_row, row, col
                )
                backend.profile_values[idx] += scale * value
        backend.band_general_dense_fallback = False
    elif backend.system_tag == AnalysisSystemTag.SuperLU or backend.system_tag == AnalysisSystemTag.UmfPack:
        for a in range(dof_count):
            var row = elem_free_pool[offset + a]
            if row < 0:
                continue
            for b in range(dof_count):
                var col = elem_free_pool[offset + b]
                if col < 0:
                    continue
                var idx = _sparse_value_index(backend, row, col)
                backend.sparse_values[idx] += scale * k_elem[a][b]
        backend.band_general_dense_fallback = False
    elif backend.system_tag == AnalysisSystemTag.SparseSYM:
        for a in range(dof_count):
            var row = elem_free_pool[offset + a]
            if row < 0:
                continue
            for b in range(dof_count):
                var col = elem_free_pool[offset + b]
                if col < 0:
                    continue
                var value = k_elem[a][b]
                if row < col:
                    if abs(value - k_elem[b][a]) > symmetric_tol:
                        abort("SparseSYM matrix is not symmetric")
                    continue
                var idx = _sparse_value_index(backend, row, col)
                backend.sparse_values[idx] += scale * value
        backend.band_general_dense_fallback = False
    elif backend.system_tag == AnalysisSystemTag.FullGeneral:
        var n = backend.free_count
        for a in range(dof_count):
            var row = elem_free_pool[offset + a]
            if row < 0:
                continue
            for b in range(dof_count):
                var col = elem_free_pool[offset + b]
                if col < 0:
                    continue
                var slot = _dense_flat_index(n, row, col)
                backend.dense_matrix_flat[slot] += scale * k_elem[a][b]
        backend.band_general_dense_fallback = False
    else:
        abort("unsupported analysis system tag")
    backend.factorized = False


fn add_diagonal(mut backend: LinearSolverBackend, diag: List[Float64], scale: Float64 = 1.0):
    if not backend.initialized:
        abort("linear solver backend not initialized")
    if len(diag) != backend.free_count:
        abort("linear solver backend diagonal size mismatch")
    if scale == 0.0:
        return

    var n = backend.free_count
    if backend.system_tag == AnalysisSystemTag.BandGeneral:
        var half_bw = backend.band_general_half_bandwidth
        for row in range(n):
            var slot = _band_general_slot(half_bw, row, row)
            backend.band_general_values[slot] += scale * diag[row]
    elif backend.system_tag == AnalysisSystemTag.BandSPD:
        var half_bw = backend.band_spd_half_bandwidth
        for row in range(n):
            var slot = _band_spd_slot(half_bw, row, row)
            backend.band_spd_values[slot] += scale * diag[row]
    elif backend.system_tag == AnalysisSystemTag.ProfileSPD:
        for row in range(n):
            var idx = _profile_value_index(
                backend.profile_col_start, backend.profile_first_row, row, row
            )
            backend.profile_values[idx] += scale * diag[row]
    elif (
        backend.system_tag == AnalysisSystemTag.SuperLU
        or backend.system_tag == AnalysisSystemTag.UmfPack
        or backend.system_tag == AnalysisSystemTag.SparseSYM
    ):
        for col in range(n):
            var found = False
            for idx in range(backend.sparse_col_start[col], backend.sparse_col_start[col + 1]):
                if backend.sparse_row_indices[idx] == col:
                    backend.sparse_values[idx] += scale * diag[col]
                    found = True
                    break
            if not found:
                abort("sparse symbolic layout missing diagonal entry")
    elif backend.system_tag == AnalysisSystemTag.FullGeneral:
        for row in range(n):
            var slot = _dense_flat_index(n, row, row)
            backend.dense_matrix_flat[slot] += scale * diag[row]
    else:
        abort("unsupported analysis system tag")
    backend.factorized = False


fn factorize_loaded(mut backend: LinearSolverBackend, mut metrics: RuntimeProfileMetrics):
    if not backend.initialized:
        abort("linear solver backend not initialized")

    if backend.system_tag == AnalysisSystemTag.BandGeneral:
        _factorize_band_general_loaded(backend)
    elif backend.system_tag == AnalysisSystemTag.BandSPD:
        _factorize_band_spd_loaded(backend)
    elif backend.system_tag == AnalysisSystemTag.ProfileSPD:
        _factorize_profile_spd_loaded(backend)
    elif backend.system_tag == AnalysisSystemTag.SuperLU:
        _factorize_sparse_unsymmetric_loaded(backend)
    elif backend.system_tag == AnalysisSystemTag.UmfPack:
        _factorize_sparse_unsymmetric_loaded(backend)
    elif backend.system_tag == AnalysisSystemTag.SparseSYM:
        _factorize_sparse_symmetric_loaded(backend)
    elif backend.system_tag == AnalysisSystemTag.FullGeneral:
        if not _try_factorize_dense_unsymmetric_loaded(backend):
            abort("FullGeneral matrix is singular or near-singular")
    else:
        abort("unsupported analysis system tag")

    backend.factorized = True
    if metrics.enabled:
        metrics.tangent_factorizations += 1
    _update_active_structure_metrics(backend, metrics)


fn factorize_from_full_matrix(
    mut backend: LinearSolverBackend,
    K_full: List[List[Float64]],
    free: List[Int],
    mut metrics: RuntimeProfileMetrics,
):
    if not backend.initialized:
        abort("linear solver backend not initialized")
    _validate_full_matrix_and_free(K_full, free, backend.free_count)

    if (
        backend.system_tag != AnalysisSystemTag.FullGeneral
        and backend.system_tag != AnalysisSystemTag.BandGeneral
        and backend.system_tag != AnalysisSystemTag.BandSPD
        and backend.system_tag != AnalysisSystemTag.ProfileSPD
        and backend.system_tag != AnalysisSystemTag.SuperLU
        and backend.system_tag != AnalysisSystemTag.UmfPack
        and backend.system_tag != AnalysisSystemTag.SparseSYM
    ):
        var A_fallback = _extract_free_submatrix(K_full, free)
        factorize(backend, A_fallback, metrics)
        return
    _load_from_full_matrix(backend, K_full, free)
    factorize_loaded(backend, metrics)


fn refactor_from_full_matrix_if_needed(
    mut backend: LinearSolverBackend,
    K_full: List[List[Float64]],
    free: List[Int],
    matrix_changed: Bool,
    mut metrics: RuntimeProfileMetrics,
    force_refactor: Bool = False,
) -> Bool:
    var needs_refactor = force_refactor or not backend.factorized or matrix_changed
    if _force_refactor_every_solve(backend):
        needs_refactor = True
    if needs_refactor:
        factorize_from_full_matrix(backend, K_full, free, metrics)
    return needs_refactor


fn _update_active_structure_metrics(
    backend: LinearSolverBackend, mut metrics: RuntimeProfileMetrics
):
    if not metrics.enabled:
        return
    metrics.active_bandwidth = 0
    metrics.active_nnz = 0
    metrics.active_profile_size = 0
    if backend.system_tag == AnalysisSystemTag.BandGeneral:
        metrics.active_bandwidth = backend.band_general_half_bandwidth * 2 + 1
    elif backend.system_tag == AnalysisSystemTag.BandSPD:
        metrics.active_bandwidth = backend.band_spd_half_bandwidth * 2 + 1
    elif backend.system_tag == AnalysisSystemTag.ProfileSPD:
        metrics.active_profile_size = len(backend.profile_values)
    elif (
        backend.system_tag == AnalysisSystemTag.SuperLU
        or backend.system_tag == AnalysisSystemTag.UmfPack
        or backend.system_tag == AnalysisSystemTag.SparseSYM
    ):
        metrics.active_nnz = len(backend.sparse_row_indices)


fn factorize(
    mut backend: LinearSolverBackend,
    A: List[List[Float64]],
    mut metrics: RuntimeProfileMetrics,
):
    if not backend.initialized:
        abort("linear solver backend not initialized")
    load_reduced_matrix(backend, A)
    factorize_loaded(backend, metrics)


fn solve(
    mut backend: LinearSolverBackend, rhs: List[Float64], mut solution_out: List[Float64]
):
    if not backend.initialized:
        abort("linear solver backend not initialized")
    if not backend.factorized:
        abort("linear solver backend has no active factorization")
    if len(rhs) != backend.free_count:
        abort("linear solver backend rhs size mismatch")

    if backend.system_tag == AnalysisSystemTag.BandGeneral:
        if backend.band_general_dense_fallback:
            _solve_dense_unsymmetric(backend, rhs, solution_out)
        else:
            _solve_band_general(backend, rhs, solution_out)
    elif backend.system_tag == AnalysisSystemTag.BandSPD:
        _solve_band_spd(backend, rhs, solution_out)
    elif backend.system_tag == AnalysisSystemTag.ProfileSPD:
        _solve_profile_spd(backend, rhs, solution_out)
    elif backend.system_tag == AnalysisSystemTag.SuperLU:
        _solve_sparse_unsymmetric(backend, rhs, solution_out)
    elif backend.system_tag == AnalysisSystemTag.UmfPack:
        _solve_sparse_unsymmetric(backend, rhs, solution_out)
    elif backend.system_tag == AnalysisSystemTag.FullGeneral:
        _solve_dense_unsymmetric(backend, rhs, solution_out)
    elif backend.system_tag == AnalysisSystemTag.SparseSYM:
        _solve_sparse_symmetric(backend, rhs, solution_out)
    else:
        abort("unsupported analysis system tag")


fn refactor_if_needed(
    mut backend: LinearSolverBackend,
    A: List[List[Float64]],
    matrix_changed: Bool,
    mut metrics: RuntimeProfileMetrics,
    force_refactor: Bool = False,
) -> Bool:
    var needs_refactor = force_refactor or not backend.factorized or matrix_changed
    if _force_refactor_every_solve(backend):
        needs_refactor = True
    if needs_refactor:
        factorize(backend, A, metrics)
    return needs_refactor


fn refactor_loaded_if_needed(
    mut backend: LinearSolverBackend,
    matrix_changed: Bool,
    mut metrics: RuntimeProfileMetrics,
    force_refactor: Bool = False,
) -> Bool:
    var needs_refactor = force_refactor or not backend.factorized or matrix_changed
    if _force_refactor_every_solve(backend):
        needs_refactor = True
    if needs_refactor:
        factorize_loaded(backend, metrics)
    return needs_refactor


fn clear(mut backend: LinearSolverBackend):
    backend.system_tag = AnalysisSystemTag.Unknown
    backend.free_count = 0
    backend.initialized = False
    backend.factorized = False
    backend.lu_matrix.clear()
    backend.chol_matrix.clear()
    backend.dense_matrix_flat.clear()
    backend.lu_pivots.clear()
    backend.solve_work.clear()
    backend.sparse_symbolic_ready = False
    backend.sparse_col_start.clear()
    backend.sparse_row_indices.clear()
    backend.sparse_values.clear()
    backend.sparse_col_permutation.clear()
    backend.sparse_rhs_work.clear()
    backend.sparse_solution_work.clear()
    backend.sparse_factor_row_start.clear()
    backend.sparse_factor_col_indices.clear()
    backend.sparse_factor_values.clear()
    backend.sparse_factor_diag_indices.clear()
    backend.sparse_row_permutation.clear()
    backend.band_general_half_bandwidth = 0
    backend.band_general_values.clear()
    backend.band_general_dense_fallback = False
    backend.band_spd_half_bandwidth = 0
    backend.band_spd_values.clear()
    backend.profile_symbolic_ready = False
    backend.profile_col_start.clear()
    backend.profile_first_row.clear()
    backend.profile_values.clear()
