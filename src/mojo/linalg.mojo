from collections import List
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
