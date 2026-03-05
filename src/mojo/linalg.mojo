from collections import List
from math import sqrt
from os import abort


fn transpose(a: List[List[Float64]]) -> List[List[Float64]]:
    var rows = len(a)
    var cols = len(a[0])
    var out: List[List[Float64]] = []
    for _ in range(cols):
        var row: List[Float64] = []
        row.resize(rows, 0.0)
        out.append(row^)
    for i in range(rows):
        for j in range(cols):
            out[j][i] = a[i][j]
    return out^


fn matmul(a: List[List[Float64]], b: List[List[Float64]]) -> List[List[Float64]]:
    var rows = len(a)
    var cols = len(b[0])
    var inner = len(b)
    var out: List[List[Float64]] = []
    for _ in range(rows):
        var row: List[Float64] = []
        row.resize(cols, 0.0)
        out.append(row^)
    for i in range(rows):
        for k in range(inner):
            var aik = a[i][k]
            if aik == 0.0:
                continue
            for j in range(cols):
                out[i][j] += aik * b[k][j]
    return out^


fn gaussian_elimination(
    mut A: List[List[Float64]], mut b: List[Float64]
) -> List[Float64]:
    var n = len(b)
    for i in range(n):
        var pivot = i
        var max_val = abs(A[i][i])
        for r in range(i + 1, n):
            if abs(A[r][i]) > max_val:
                max_val = abs(A[r][i])
                pivot = r
        if max_val == 0.0:
            abort("singular matrix")
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

    var x: List[Float64] = []
    x.resize(n, 0.0)
    for i in range(n - 1, -1, -1):
        var s = b[i]
        for j in range(i + 1, n):
            s -= A[i][j] * x[j]
        x[i] = s
    return x^


fn gaussian_elimination_into(
    mut A: List[List[Float64]], mut b: List[Float64], mut x_out: List[Float64]
):
    var n = len(b)
    if len(x_out) != n:
        x_out.resize(n, 0.0)
    for i in range(n):
        var pivot = i
        var max_val = abs(A[i][i])
        for r in range(i + 1, n):
            if abs(A[r][i]) > max_val:
                max_val = abs(A[r][i])
                pivot = r
        if max_val == 0.0:
            abort("singular matrix")
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
            s -= A[i][j] * x_out[j]
        x_out[i] = s


fn lu_factorize_into(mut A: List[List[Float64]], mut pivots: List[Int]):
    var n = len(A)
    if len(pivots) != n:
        pivots.resize(n, 0)
    for i in range(n):
        pivots[i] = i
    for i in range(n):
        var pivot = i
        var max_val = abs(A[i][i])
        for r in range(i + 1, n):
            var v = abs(A[r][i])
            if v > max_val:
                max_val = v
                pivot = r
        if max_val == 0.0:
            abort("singular matrix")
        if pivot != i:
            var tmp = A[i].copy()
            A[i] = A[pivot].copy()
            A[pivot] = tmp^
            var tp = pivots[i]
            pivots[i] = pivots[pivot]
            pivots[pivot] = tp
        var piv = A[i][i]
        for r in range(i + 1, n):
            A[r][i] /= piv
            var factor = A[r][i]
            if factor == 0.0:
                continue
            for c in range(i + 1, n):
                A[r][c] -= factor * A[i][c]


fn lu_solve_into(
    LU: List[List[Float64]],
    pivots: List[Int],
    b: List[Float64],
    mut work: List[Float64],
    mut x_out: List[Float64],
):
    var n = len(b)
    if len(work) != n:
        work.resize(n, 0.0)
    if len(x_out) != n:
        x_out.resize(n, 0.0)
    for i in range(n):
        var s = b[pivots[i]]
        for j in range(i):
            s -= LU[i][j] * work[j]
        work[i] = s

    for i in range(n - 1, -1, -1):
        var s = work[i]
        for j in range(i + 1, n):
            s -= LU[i][j] * x_out[j]
        x_out[i] = s / LU[i][i]


fn cholesky_factorize_into(
    A: List[List[Float64]],
    mut L: List[List[Float64]],
    symmetric_tol: Float64 = 1.0e-10,
):
    var n = len(A)
    if len(L) != n:
        L.clear()
        for _ in range(n):
            var row: List[Float64] = []
            row.resize(n, 0.0)
            L.append(row^)
    for i in range(n):
        if len(A[i]) != n:
            abort("cholesky_factorize_into expects square matrix")
        if len(L[i]) != n:
            L[i].resize(n, 0.0)
    for i in range(n):
        for j in range(i + 1):
            if abs(A[i][j] - A[j][i]) > symmetric_tol:
                abort("matrix is not symmetric")
            var sum = A[i][j]
            for k in range(j):
                sum -= L[i][k] * L[j][k]
            if i == j:
                if sum <= 0.0:
                    abort("matrix is not positive definite")
                L[i][j] = sqrt(sum)
            else:
                var diag = L[j][j]
                if diag == 0.0:
                    abort("matrix is not positive definite")
                L[i][j] = sum / diag
        for j in range(i + 1, n):
            L[i][j] = 0.0


fn cholesky_solve_into(
    L: List[List[Float64]],
    b: List[Float64],
    mut work: List[Float64],
    mut x_out: List[Float64],
):
    var n = len(b)
    if len(L) != n:
        abort("cholesky_solve_into dimension mismatch")
    if len(work) != n:
        work.resize(n, 0.0)
    if len(x_out) != n:
        x_out.resize(n, 0.0)

    # Forward solve: L y = b
    for i in range(n):
        var sum = b[i]
        for j in range(i):
            sum -= L[i][j] * work[j]
        var diag = L[i][i]
        if diag == 0.0:
            abort("singular cholesky factor")
        work[i] = sum / diag

    # Backward solve: L^T x = y
    for i in range(n - 1, -1, -1):
        var sum = work[i]
        for j in range(i + 1, n):
            sum -= L[j][i] * x_out[j]
        var diag = L[i][i]
        if diag == 0.0:
            abort("singular cholesky factor")
        x_out[i] = sum / diag
