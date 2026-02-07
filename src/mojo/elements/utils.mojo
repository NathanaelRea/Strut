from collections import List
from math import sqrt
from os import abort


fn _zero_matrix(rows: Int, cols: Int) -> List[List[Float64]]:
    var out: List[List[Float64]] = []
    for _ in range(rows):
        var row: List[Float64] = []
        row.resize(cols, 0.0)
        out.append(row^)
    return out^


fn _matvec(mat: List[List[Float64]], vec: List[Float64]) -> List[Float64]:
    var out: List[Float64] = []
    out.resize(len(vec), 0.0)
    for i in range(len(mat)):
        var sum = 0.0
        for j in range(len(vec)):
            sum += mat[i][j] * vec[j]
        out[i] = sum
    return out^


fn _dot(
    ax: Float64,
    ay: Float64,
    az: Float64,
    bx: Float64,
    by: Float64,
    bz: Float64,
) -> Float64:
    return ax * bx + ay * by + az * bz


fn _cross(
    ax: Float64,
    ay: Float64,
    az: Float64,
    bx: Float64,
    by: Float64,
    bz: Float64,
) -> (Float64, Float64, Float64):
    return (ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx)


fn _normalize(x: Float64, y: Float64, z: Float64) -> (Float64, Float64, Float64):
    var n = sqrt(x * x + y * y + z * z)
    if n == 0.0:
        abort("zero-length vector")
    return (x / n, y / n, z / n)
