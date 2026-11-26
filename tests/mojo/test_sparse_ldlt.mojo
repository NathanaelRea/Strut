from math import inf, nan
from testing import assert_equal, assert_raises, assert_true, TestSuite

from solver.sparse_ldlt import (
    estimate_sparse_ldlt_ordering,
    SparseFactorMode,
    SparseLDLT,
    SparseOrdering,
)
from sparse_ldlt_bench import (
    floor_dissection,
    load_csc_pattern,
    make_banded,
    make_lattice,
    make_rhs,
    relative_residual,
    SparseBenchMatrix,
)


fn _matrix(
    n: Int, var col_ptr: List[Int], var rows: List[Int], var values: List[Float64]
) -> SparseBenchMatrix:
    var result = SparseBenchMatrix(n, 0, 0)
    result.col_ptr = col_ptr^
    result.rows = rows^
    result.values = values^
    return result^


fn _check_modes(matrix: SparseBenchMatrix, negative: Int = 0, expected_nnz: Int = -1) raises:
    var rhs = make_rhs(matrix.n)
    var a = SparseLDLT(1)
    var b = SparseLDLT(1)
    var xa = rhs.copy()
    var xb = rhs.copy()
    for ordering in [SparseOrdering.AMD, SparseOrdering.RCM]:
        a.analyze(matrix.n, matrix.col_ptr, matrix.rows, ordering, SparseFactorMode.Simplicial)
        b.analyze(matrix.n, matrix.col_ptr, matrix.rows, ordering, SparseFactorMode.Supernodal)
        assert_equal(a.factor_nnz(), b.factor_nnz())
        assert_true(b.supernode_count() <= a.supernode_count())
        if expected_nnz >= 0:
            assert_equal(b.factor_nnz(), expected_nnz)
        assert_true(a.factorize(matrix.values))
        assert_true(b.factorize(matrix.values))
        assert_equal(a.negative_pivots(), negative)
        assert_equal(b.negative_pivots(), negative)
        a.solve(rhs, xa)
        b.solve(rhs, xb)
        assert_true(relative_residual(matrix, xa, rhs) <= 1.0e-10)
        assert_true(relative_residual(matrix, xb, rhs) <= 1.0e-10)
        for i in range(matrix.n):
            assert_true(abs(xa[i] - xb[i]) <= 5.0e-13 * max(1.0, abs(xa[i])))


def test_diagonal_and_empty():
    _check_modes(_matrix(0, [0], [], []), expected_nnz=0)
    _check_modes(_matrix(3, [0, 1, 2, 3], [0, 1, 2], [2.0, 3.0, 4.0]), expected_nnz=3)


def test_tridiagonal():
    _check_modes(make_banded(65, 1), expected_nnz=129)
    _check_modes(make_banded(145, 20), expected_nnz=2835)


def test_arrow():
    var n = 259
    var matrix = SparseBenchMatrix(n, n - 1, 2 * n - 1)
    matrix.col_ptr.append(0)
    matrix.append(0, 1.0 + Float64(n - 1) * 0.5)
    for i in range(1, n):
        matrix.append(i, -0.5)
    for i in range(1, n):
        matrix.col_ptr.append(len(matrix.rows))
        matrix.append(i, 1.5)
    matrix.col_ptr.append(len(matrix.rows))
    _check_modes(matrix, expected_nnz=2 * n - 1)


def test_grid_2d_and_grid_3d():
    _check_modes(make_lattice(16, 14, 1))
    _check_modes(make_lattice(7, 6, 5))


def test_dense_element_blocks():
    _check_modes(make_lattice(4, 3, 6, 6))


def test_indefinite_negative_pivots():
    _check_modes(
        _matrix(
            4,
            [0, 2, 5, 7, 8],
            [0, 1, 1, 2, 3, 2, 3, 3],
            [4.0, 1.0, -6.0, 0.5, 0.25, -3.0, 0.5, 3.0],
        ),
        negative=2,
    )


def test_pivot_failures_and_retry():
    for mode in [SparseFactorMode.Simplicial, SparseFactorMode.Supernodal]:
        var solver = SparseLDLT()
        assert_true(not solver.factorize([]))
        solver.analyze(2, [0, 2, 3], [0, 1, 1], mode=mode)
        assert_true(not solver.factorize([1.0, 1.0, 1.0]))
        assert_true(not solver.factorize([0.0, 1.0, 0.0]))
        assert_true(not solver.factorize([1.0, 1.0]))
        assert_true(not solver.factorize([1.0, inf[DType.float64](), 1.0]))
        assert_true(not solver.factorize([1.0, 1.0, nan[DType.float64]()]))
        assert_true(solver.factorize([2.0, -1.0, 2.0]))
        var rhs: List[Float64] = [1.0, 1.0]
        var x: List[Float64] = [0.0, 0.0]
        solver.solve(rhs, x)
        assert_true(abs(x[0] - 1.0) <= 1.0e-14)
        assert_true(abs(x[1] - 1.0) <= 1.0e-14)
        solver.analyze(1, [0, 1], [0], mode=mode)
        assert_true(not solver.factorize([1.0e-310]))
        assert_true(solver.factorize([1.0e-100]))


def test_repeated_numeric_factorization():
    var matrix = make_lattice(5, 6, 3, 6)
    var rhs = make_rhs(matrix.n)
    for mode in [SparseFactorMode.Simplicial, SparseFactorMode.Supernodal]:
        var solver = SparseLDLT()
        solver.analyze(matrix.n, matrix.col_ptr, matrix.rows, mode=mode)
        var x = rhs.copy()
        for _ in range(3):
            assert_true(solver.factorize(matrix.values))
            solver.solve(rhs, x)
            assert_true(relative_residual(matrix, x, rhs) <= 1.0e-10)
            for j in range(matrix.n):
                matrix.values[matrix.col_ptr[j]] += 0.5
            for j in range(len(rhs)):
                rhs[j] += 0.25


def test_auto_uses_the_lower_predicted_fill():
    var matrix = make_lattice(20, 17, 1)
    var solver = SparseLDLT()
    solver.analyze(matrix.n, matrix.col_ptr, matrix.rows, SparseOrdering.AMD)
    var amd_nnz = solver.factor_nnz()
    solver.analyze(matrix.n, matrix.col_ptr, matrix.rows, SparseOrdering.RCM)
    var rcm_nnz = solver.factor_nnz()
    assert_true(amd_nnz < rcm_nnz)
    solver.analyze(matrix.n, matrix.col_ptr, matrix.rows)
    assert_equal(solver.factor_nnz(), min(amd_nnz, rcm_nnz))


fn _check_threaded(matrix: SparseBenchMatrix, permutation: List[Int] = []) raises:
    var rhs = make_rhs(matrix.n)
    var x1 = rhs.copy()
    var x8 = rhs.copy()
    var repeat = rhs.copy()
    var serial = SparseLDLT(1)
    var parallel = SparseLDLT(8)
    serial.analyze(
        matrix.n,
        matrix.col_ptr,
        matrix.rows,
        SparseOrdering.AMD,
        permutation=permutation,
    )
    parallel.analyze(
        matrix.n,
        matrix.col_ptr,
        matrix.rows,
        SparseOrdering.AMD,
        permutation=permutation,
    )
    assert_true(serial.largest_supernode() >= 24)
    assert_true(serial.largest_front() >= serial.largest_supernode())
    assert_true(serial.factor_flops() >= 320_000)
    assert_equal(serial.factor_nnz(), parallel.factor_nnz())
    assert_equal(serial.factor_flops(), parallel.factor_flops())
    assert_true(serial.factorize(matrix.values))
    assert_true(parallel.factorize(matrix.values))
    assert_equal(serial.negative_pivots(), parallel.negative_pivots())
    serial.solve(rhs, x1)
    parallel.solve(rhs, x8)
    assert_true(relative_residual(matrix, x1, rhs) <= 1.0e-10)
    assert_true(relative_residual(matrix, x8, rhs) <= 1.0e-10)
    assert_true(parallel.factorize(matrix.values))
    parallel.solve(rhs, repeat)
    for i in range(matrix.n):
        assert_true(abs(x1[i] - x8[i]) <= 5.0e-12 * max(1.0, abs(x1[i])))
        assert_equal(x8[i], repeat[i])


def test_blocked_dense_reference():
    var n = 181
    var matrix = make_banded(n, n - 1)
    for j in range(0, n, 4):
        # Keep the matrix strictly dominant while giving D both signs.
        matrix.values[matrix.col_ptr[j]] *= -1.0
    _check_modes(matrix, negative=(n + 3) // 4, expected_nnz=n * (n + 1) // 2)


def test_parallel_dense_front():
    _check_threaded(make_banded(504, 503))


def test_threaded_failure_then_retry():
    var matrix = make_banded(504, 503)
    var solver = SparseLDLT(8)
    solver.analyze(matrix.n, matrix.col_ptr, matrix.rows)
    var broken = matrix.values.copy()
    for j in range(matrix.n):
        broken[matrix.col_ptr[j]] = 0.0
    assert_true(not solver.factorize(broken))
    assert_true(solver.factorize(matrix.values))
    var rhs = make_rhs(matrix.n)
    var x = rhs.copy()
    solver.solve(rhs, x)
    assert_true(relative_residual(matrix, x, rhs) <= 1.0e-10)


def test_floor_masters_and_independent_subtrees():
    var matrix = make_lattice(12, 12, 8, 6, True)
    _check_threaded(matrix)
    _check_threaded(matrix, floor_dissection(12, 12, 8))


def test_invalid_pattern_reports_error():
    var solver = SparseLDLT()
    with assert_raises(contains="column pointers"):
        solver.analyze(1, [0], [0])
    with assert_raises(contains="start with the diagonal"):
        solver.analyze(2, [0, 1, 2], [1, 1])
    with assert_raises(contains="sorted and unique"):
        solver.analyze(2, [0, 3, 4], [0, 1, 1, 1])
    with assert_raises(contains="permutation"):
        solver.analyze(2, [0, 2, 3], [0, 1, 1], permutation=[0, 0])


def test_load_csc_pattern_and_score_ordering():
    for scaled in [False, True]:
        var matrix = load_csc_pattern("tests/mojo/sparse_ldlt_pattern.txt", scaled)
        assert_equal(matrix.n, 4)
        assert_equal(len(matrix.rows), 8)
        assert_equal(matrix.col_ptr[1], 3)
        assert_equal(matrix.rows[2], 3)
        assert_true(matrix.values[2] < 0.0)
        var rhs = make_rhs(matrix.n)
        var x = rhs.copy()
        for ordering in [SparseOrdering.AMD, SparseOrdering.RCM, SparseOrdering.Auto]:
            var score = estimate_sparse_ldlt_ordering(
                matrix.n, matrix.col_ptr, matrix.rows, ordering
            )
            var solver = SparseLDLT(1)
            solver.analyze(matrix.n, matrix.col_ptr, matrix.rows, ordering)
            assert_equal(score.nnz, solver.factor_nnz())
            assert_equal(score.flops, solver.factor_flops())
            assert_true(solver.factorize(matrix.values))
            solver.solve(rhs, x)
            assert_true(relative_residual(matrix, x, rhs) <= 1.0e-10)


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
