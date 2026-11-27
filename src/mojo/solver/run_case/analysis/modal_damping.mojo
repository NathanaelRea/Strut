from collections import List
from math import sqrt

from solver.eigen import jacobi_symmetric_eigen
from solver.run_case.analysis.modal_eigen import (
    _cholesky_factor_dense_spd,
    _multiply_dense_by_transpose,
    _multiply_dense_matrices,
    _solve_lower_triangular,
    _solve_upper_from_lower_transpose,
    _sorted_indices_ascending,
)
from solver.sparse_pattern import SparsePattern


fn build_modal_damping(
    pattern: SparsePattern,
    stiffness: List[Float64],
    mass: List[Float64],
    ratios: List[Float64],
) -> List[Float64]:
    var n = pattern.n
    var k: List[List[Float64]] = []
    var m: List[List[Float64]] = []
    for i in range(n):
        var k_row: List[Float64] = []
        var m_row: List[Float64] = []
        k_row.resize(n, 0.0)
        m_row.resize(n, 0.0)
        for slot in range(pattern.row_start[i], pattern.row_start[i + 1]):
            var j = pattern.col_index[slot]
            k_row[j] = stiffness[slot]
            m_row[j] = mass[slot]
        k.append(k_row^)
        m.append(m_row^)

    # Factor K so zero-mass DOFs do not need an artificial mass.
    var factor = _cholesky_factor_dense_spd(k)
    var inverse: List[List[Float64]] = []
    for _ in range(n):
        var row: List[Float64] = []
        row.resize(n, 0.0)
        inverse.append(row^)
    for j in range(n):
        var basis: List[Float64] = []
        basis.resize(n, 0.0)
        basis[j] = 1.0
        var column = _solve_lower_triangular(factor, basis)
        for i in range(n):
            inverse[i][j] = column[i]
    var scaled = _multiply_dense_matrices(inverse, m)
    scaled = _multiply_dense_by_transpose(scaled, inverse)
    var scale = 0.0
    for i in range(n):
        scale = max(scale, abs(scaled[i][i]))
    var eigenvalues: List[Float64] = []
    var eigenvectors: List[List[Float64]] = []
    jacobi_symmetric_eigen(scaled, eigenvalues, eigenvectors, 8 * n * n, scale * 1.0e-13)
    var order = _sorted_indices_ascending(eigenvalues)

    var damping: List[Float64] = []
    damping.resize(pattern.nnz(), 0.0)
    for mode in range(min(n, len(ratios))):
        var column = order[n - mode - 1]
        var reciprocal = eigenvalues[column]
        if reciprocal <= scale * 1.0e-12:
            break
        if ratios[mode] == 0.0:
            continue
        var omega = 1.0 / sqrt(reciprocal)
        var vector: List[Float64] = []
        for i in range(n):
            vector.append(eigenvectors[i][column])
        vector = _solve_upper_from_lower_transpose(factor, vector)

        # With mass-normalized eigenvectors phi, each mode contributes
        # 2 * ratio * omega * (M phi) (M phi)^T.
        var mass_mode: List[Float64] = []
        mass_mode.resize(n, 0.0)
        for i in range(n):
            for j in range(n):
                mass_mode[i] += m[i][j] * vector[j]
            mass_mode[i] *= omega
        for i in range(n):
            var coefficient = 2.0 * ratios[mode] * omega * mass_mode[i]
            for slot in range(pattern.row_start[i], pattern.row_start[i + 1]):
                damping[slot] += coefficient * mass_mode[pattern.col_index[slot]]
    return damping^
