from collections import List
from math import sqrt
from os import abort

from linalg import cholesky_factorize_into, lu_factorize_into
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
    var lu_pivots: List[Int]
    var solve_work: List[Float64]
    var sparse_symbolic_ready: Bool
    var sparse_col_start: List[Int]
    var sparse_row_indices: List[Int]
    var sparse_values: List[Float64]
    var sparse_col_permutation: List[Int]
    var sparse_rhs_work: List[Float64]
    var sparse_solution_work: List[Float64]
    var band_general_half_bandwidth: Int
    var band_general_values: List[List[Float64]]
    var band_spd_half_bandwidth: Int
    var band_spd_values: List[List[Float64]]
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
        self.lu_pivots = []
        self.solve_work = []
        self.sparse_symbolic_ready = False
        self.sparse_col_start = []
        self.sparse_row_indices = []
        self.sparse_values = []
        self.sparse_col_permutation = []
        self.sparse_rhs_work = []
        self.sparse_solution_work = []
        self.band_general_half_bandwidth = 0
        self.band_general_values = []
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


fn _copy_square_matrix(src: List[List[Float64]], mut dst: List[List[Float64]], n: Int):
    _ensure_square_storage(dst, n)
    for i in range(n):
        for j in range(n):
            dst[i][j] = src[i][j]


fn _validate_square_matrix_shape(A: List[List[Float64]], n: Int):
    if len(A) != n:
        abort("linear solver backend matrix row count mismatch")
    for i in range(n):
        if len(A[i]) != n:
            abort("linear solver backend matrix column count mismatch")


fn _factorize_unsymmetric(mut backend: LinearSolverBackend, A: List[List[Float64]]):
    _copy_square_matrix(A, backend.lu_matrix, backend.free_count)
    lu_factorize_into(backend.lu_matrix, backend.lu_pivots)


fn _solve_unsymmetric(
    mut backend: LinearSolverBackend, rhs: List[Float64], mut solution_out: List[Float64]
):
    var n = backend.free_count
    if len(solution_out) != n:
        solution_out.resize(n, 0.0)
    for i in range(n):
        var s = rhs[backend.lu_pivots[i]]
        for j in range(i):
            s -= backend.lu_matrix[i][j] * backend.solve_work[j]
        backend.solve_work[i] = s

    for i in range(n - 1, -1, -1):
        var s = backend.solve_work[i]
        for j in range(i + 1, n):
            s -= backend.lu_matrix[i][j] * solution_out[j]
        solution_out[i] = s / backend.lu_matrix[i][i]


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
    if len(backend.band_general_values) != n:
        backend.band_general_values.clear()
        for _ in range(n):
            var row: List[Float64] = []
            row.resize(width, 0.0)
            backend.band_general_values.append(row^)
        return
    for i in range(n):
        if len(backend.band_general_values[i]) != width:
            backend.band_general_values[i].resize(width, 0.0)


fn _ensure_banded_spd_storage(mut backend: LinearSolverBackend, n: Int, half_bw: Int):
    var width = half_bw + 1
    if len(backend.band_spd_values) != n:
        backend.band_spd_values.clear()
        for _ in range(n):
            var row: List[Float64] = []
            row.resize(width, 0.0)
            backend.band_spd_values.append(row^)
        return
    for i in range(n):
        if len(backend.band_spd_values[i]) != width:
            backend.band_spd_values[i].resize(width, 0.0)


fn _update_banded_general_values_from_dense(mut backend: LinearSolverBackend, A: List[List[Float64]]):
    var n = len(A)
    var half_bw = backend.band_general_half_bandwidth
    var width = half_bw * 2 + 1
    for i in range(n):
        for k in range(width):
            backend.band_general_values[i][k] = 0.0
        var j0 = i - half_bw
        if j0 < 0:
            j0 = 0
        var j1 = i + half_bw
        if j1 > n - 1:
            j1 = n - 1
        for j in range(j0, j1 + 1):
            backend.band_general_values[i][j - i + half_bw] = A[i][j]


fn _assemble_dense_from_banded_general(mut backend: LinearSolverBackend):
    var n = backend.free_count
    var half_bw = backend.band_general_half_bandwidth
    _zero_square_storage(backend.lu_matrix, n)
    for i in range(n):
        var j0 = i - half_bw
        if j0 < 0:
            j0 = 0
        var j1 = i + half_bw
        if j1 > n - 1:
            j1 = n - 1
        for j in range(j0, j1 + 1):
            backend.lu_matrix[i][j] = backend.band_general_values[i][j - i + half_bw]


fn _update_banded_spd_values_from_dense(mut backend: LinearSolverBackend, A: List[List[Float64]]):
    var n = len(A)
    var half_bw = backend.band_spd_half_bandwidth
    var width = half_bw + 1
    for i in range(n):
        for k in range(width):
            backend.band_spd_values[i][k] = 0.0
        var j0 = i - half_bw
        if j0 < 0:
            j0 = 0
        for j in range(j0, i + 1):
            backend.band_spd_values[i][i - j] = _symmetric_entry(i, j, A)


fn _assemble_dense_symmetric_from_banded_spd(mut backend: LinearSolverBackend):
    var n = backend.free_count
    var half_bw = backend.band_spd_half_bandwidth
    _zero_square_storage(backend.lu_matrix, n)
    for i in range(n):
        var j0 = i - half_bw
        if j0 < 0:
            j0 = 0
        for j in range(j0, i + 1):
            var value = backend.band_spd_values[i][i - j]
            backend.lu_matrix[i][j] = value
            if i != j:
                backend.lu_matrix[j][i] = value


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


fn _factorize_profile_spd(mut backend: LinearSolverBackend, A: List[List[Float64]]):
    _ensure_profile_symbolic(backend, A)
    _update_profile_values_from_dense(backend, A)
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


fn _validate_band_general_pivots(backend: LinearSolverBackend):
    var n = backend.free_count
    var pivot_tol = 1.0e-14
    for i in range(n):
        if abs(backend.lu_matrix[i][i]) <= pivot_tol:
            abort("BandGeneral matrix is singular or near-singular after pivoting")


fn _factorize_band_general(mut backend: LinearSolverBackend, A: List[List[Float64]]):
    backend.band_general_half_bandwidth = _detect_general_band_half_width(A)
    _ensure_banded_general_storage(backend, backend.free_count, backend.band_general_half_bandwidth)
    _update_banded_general_values_from_dense(backend, A)
    _assemble_dense_from_banded_general(backend)
    lu_factorize_into(backend.lu_matrix, backend.lu_pivots)
    _validate_band_general_pivots(backend)


fn _factorize_band_spd(mut backend: LinearSolverBackend, A: List[List[Float64]]):
    backend.band_spd_half_bandwidth = _detect_symmetric_band_half_width(A)
    _ensure_banded_spd_storage(backend, backend.free_count, backend.band_spd_half_bandwidth)
    _update_banded_spd_values_from_dense(backend, A)
    _assemble_dense_symmetric_from_banded_spd(backend)
    cholesky_factorize_into(backend.lu_matrix, backend.chol_matrix)


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


fn _assemble_permuted_dense_from_sparse(mut backend: LinearSolverBackend):
    var n = backend.free_count
    _zero_square_storage(backend.lu_matrix, n)
    if len(backend.sparse_col_permutation) != n:
        backend.sparse_col_permutation = _identity_permutation(n)

    for perm_col in range(n):
        var src_col = backend.sparse_col_permutation[perm_col]
        if src_col < 0 or src_col >= n:
            abort("invalid sparse column permutation index")
        for idx in range(backend.sparse_col_start[src_col], backend.sparse_col_start[src_col + 1]):
            var row = backend.sparse_row_indices[idx]
            backend.lu_matrix[row][perm_col] = backend.sparse_values[idx]


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


fn _assemble_permuted_dense_symmetric_from_sparse(mut backend: LinearSolverBackend):
    var n = backend.free_count
    _zero_square_storage(backend.lu_matrix, n)
    if len(backend.sparse_col_permutation) != n:
        backend.sparse_col_permutation = _identity_permutation(n)
    if not _is_valid_permutation(backend.sparse_col_permutation, n):
        backend.sparse_col_permutation = _identity_permutation(n)

    var inverse_permutation = _build_inverse_permutation(backend.sparse_col_permutation, n)
    for perm_col in range(n):
        var src_col = backend.sparse_col_permutation[perm_col]
        for idx in range(backend.sparse_col_start[src_col], backend.sparse_col_start[src_col + 1]):
            var src_row = backend.sparse_row_indices[idx]
            var perm_row = inverse_permutation[src_row]
            var value = backend.sparse_values[idx]
            backend.lu_matrix[perm_row][perm_col] = value
            if perm_row != perm_col:
                backend.lu_matrix[perm_col][perm_row] = value


fn _factorize_sparse_unsymmetric(mut backend: LinearSolverBackend, A: List[List[Float64]]):
    _ensure_sparse_symbolic(backend, A)
    _update_sparse_values_from_dense(backend, A)
    _assemble_permuted_dense_from_sparse(backend)
    lu_factorize_into(backend.lu_matrix, backend.lu_pivots)


fn _factorize_sparse_symmetric(mut backend: LinearSolverBackend, A: List[List[Float64]]):
    _ensure_sparse_symbolic(backend, A)
    _update_sparse_values_from_dense(backend, A)
    _assemble_permuted_dense_symmetric_from_sparse(backend)
    cholesky_factorize_into(backend.lu_matrix, backend.chol_matrix)


fn _solve_sparse_unsymmetric(
    mut backend: LinearSolverBackend, rhs: List[Float64], mut solution_out: List[Float64]
):
    var n = backend.free_count
    if len(solution_out) != n:
        solution_out.resize(n, 0.0)
    _ensure_vector_storage(backend.sparse_rhs_work, n)
    _ensure_vector_storage(backend.sparse_solution_work, n)

    for i in range(n):
        backend.sparse_rhs_work[i] = rhs[i]
    var rhs_work = backend.sparse_rhs_work.copy()
    var solution_work = backend.sparse_solution_work.copy()
    _solve_unsymmetric(backend, rhs_work, solution_work)
    backend.sparse_solution_work = solution_work.copy()

    for i in range(n):
        solution_out[i] = 0.0
    if len(backend.sparse_col_permutation) != n:
        backend.sparse_col_permutation = _identity_permutation(n)

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

    var rhs_work = backend.sparse_rhs_work.copy()
    var solution_work = backend.sparse_solution_work.copy()
    _solve_spd(backend, rhs_work, solution_work)
    backend.sparse_solution_work = solution_work.copy()

    for i in range(n):
        solution_out[i] = 0.0
    for perm_col in range(n):
        var src_col = backend.sparse_col_permutation[perm_col]
        solution_out[src_col] = backend.sparse_solution_work[perm_col]


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
    _ensure_square_storage(backend.lu_matrix, free_count)
    _ensure_square_storage(backend.chol_matrix, free_count)
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

    backend.band_general_half_bandwidth = 0
    backend.band_general_values.clear()
    backend.band_spd_half_bandwidth = 0
    backend.band_spd_values.clear()
    backend.profile_symbolic_ready = False
    backend.profile_col_start.clear()
    backend.profile_first_row.clear()
    backend.profile_values.clear()


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
    _validate_square_matrix_shape(A, backend.free_count)

    if backend.system_tag == AnalysisSystemTag.BandGeneral:
        _factorize_band_general(backend, A)
    elif backend.system_tag == AnalysisSystemTag.BandSPD:
        _factorize_band_spd(backend, A)
    elif backend.system_tag == AnalysisSystemTag.ProfileSPD:
        _factorize_profile_spd(backend, A)
    elif backend.system_tag == AnalysisSystemTag.SuperLU:
        _factorize_sparse_unsymmetric(backend, A)
    elif backend.system_tag == AnalysisSystemTag.UmfPack:
        _factorize_sparse_unsymmetric(backend, A)
    elif backend.system_tag == AnalysisSystemTag.FullGeneral:
        _factorize_unsymmetric(backend, A)
    elif backend.system_tag == AnalysisSystemTag.SparseSYM:
        _factorize_sparse_symmetric(backend, A)
    else:
        abort("unsupported analysis system tag")
    backend.factorized = True
    if metrics.enabled:
        metrics.tangent_factorizations += 1
    _update_active_structure_metrics(backend, metrics)


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
        _solve_unsymmetric(backend, rhs, solution_out)
    elif backend.system_tag == AnalysisSystemTag.BandSPD:
        _solve_spd(backend, rhs, solution_out)
    elif backend.system_tag == AnalysisSystemTag.ProfileSPD:
        _solve_profile_spd(backend, rhs, solution_out)
    elif backend.system_tag == AnalysisSystemTag.SuperLU:
        _solve_sparse_unsymmetric(backend, rhs, solution_out)
    elif backend.system_tag == AnalysisSystemTag.UmfPack:
        _solve_sparse_unsymmetric(backend, rhs, solution_out)
    elif backend.system_tag == AnalysisSystemTag.FullGeneral:
        _solve_unsymmetric(backend, rhs, solution_out)
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


fn clear(mut backend: LinearSolverBackend):
    backend.system_tag = AnalysisSystemTag.Unknown
    backend.free_count = 0
    backend.initialized = False
    backend.factorized = False
    backend.lu_matrix.clear()
    backend.chol_matrix.clear()
    backend.lu_pivots.clear()
    backend.solve_work.clear()
    backend.sparse_symbolic_ready = False
    backend.sparse_col_start.clear()
    backend.sparse_row_indices.clear()
    backend.sparse_values.clear()
    backend.sparse_col_permutation.clear()
    backend.sparse_rhs_work.clear()
    backend.sparse_solution_work.clear()
    backend.band_general_half_bandwidth = 0
    backend.band_general_values.clear()
    backend.band_spd_half_bandwidth = 0
    backend.band_spd_values.clear()
    backend.profile_symbolic_ready = False
    backend.profile_col_start.clear()
    backend.profile_first_row.clear()
    backend.profile_values.clear()
