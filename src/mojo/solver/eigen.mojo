from collections import List
from math import atan2, cos, sin


fn _identity_matrix(n: Int) -> List[List[Float64]]:
    var out: List[List[Float64]] = []
    for i in range(n):
        var row: List[Float64] = []
        row.resize(n, 0.0)
        row[i] = 1.0
        out.append(row^)
    return out^


fn jacobi_symmetric_eigen(
    mut A: List[List[Float64]],
    mut eigenvalues: List[Float64],
    mut eigenvectors: List[List[Float64]],
    max_sweeps: Int = 100,
    tol: Float64 = 1.0e-12,
):
    var n = len(A)
    var V = _identity_matrix(n)
    if n == 0:
        eigenvalues.resize(0, 0.0)
        eigenvectors.resize(0, [])
        return
    if n == 1:
        eigenvalues.resize(1, 0.0)
        eigenvalues[0] = A[0][0]
        eigenvectors = V^
        return

    for _ in range(max_sweeps):
        var p = 0
        var q = 1
        var max_off = abs(A[p][q])
        for i in range(n):
            for j in range(i + 1, n):
                var v = abs(A[i][j])
                if v > max_off:
                    max_off = v
                    p = i
                    q = j
        if max_off <= tol:
            break

        var app = A[p][p]
        var aqq = A[q][q]
        var apq = A[p][q]
        var phi = 0.5 * atan2(2.0 * apq, aqq - app)
        var c = cos(phi)
        var s = sin(phi)

        for i in range(n):
            if i == p or i == q:
                continue
            var aip = A[i][p]
            var aiq = A[i][q]
            A[i][p] = c * aip - s * aiq
            A[p][i] = A[i][p]
            A[i][q] = s * aip + c * aiq
            A[q][i] = A[i][q]

        A[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq
        A[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq
        A[p][q] = 0.0
        A[q][p] = 0.0

        for i in range(n):
            var vip = V[i][p]
            var viq = V[i][q]
            V[i][p] = c * vip - s * viq
            V[i][q] = s * vip + c * viq

    eigenvalues.resize(n, 0.0)
    for i in range(n):
        eigenvalues[i] = A[i][i]
    eigenvectors = V^
