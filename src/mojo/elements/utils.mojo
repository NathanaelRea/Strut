from collections import List
from math import sqrt
from os import abort


fn _ensure_zero_vector(mut vec: List[Float64], size: Int):
    if len(vec) != size:
        vec.resize(size, 0.0)
    for i in range(size):
        vec[i] = 0.0


fn _ensure_zero_matrix(mut mat: List[List[Float64]], rows: Int, cols: Int):
    if len(mat) != rows:
        mat = []
        for _ in range(rows):
            var row: List[Float64] = []
            row.resize(cols, 0.0)
            mat.append(row^)
        return
    for i in range(rows):
        if len(mat[i]) != cols:
            mat[i].resize(cols, 0.0)
        for j in range(cols):
            mat[i][j] = 0.0


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


fn _beam2d_transform_u_global_to_local(
    c: Float64, s: Float64, u_global: List[Float64], mut u_local_out: List[Float64]
):
    if len(u_global) != 6:
        abort("beam2d displacement vector size mismatch")
    _ensure_zero_vector(u_local_out, 6)
    u_local_out[0] = c * u_global[0] + s * u_global[1]
    u_local_out[1] = -s * u_global[0] + c * u_global[1]
    u_local_out[2] = u_global[2]
    u_local_out[3] = c * u_global[3] + s * u_global[4]
    u_local_out[4] = -s * u_global[3] + c * u_global[4]
    u_local_out[5] = u_global[5]


fn _beam2d_transform_force_local_to_global_in_place(
    mut f_local: List[Float64], c: Float64, s: Float64
):
    if len(f_local) != 6:
        abort("beam2d force vector size mismatch")
    var f0 = f_local[0]
    var f1 = f_local[1]
    var f3 = f_local[3]
    var f4 = f_local[4]
    f_local[0] = c * f0 - s * f1
    f_local[1] = s * f0 + c * f1
    f_local[3] = c * f3 - s * f4
    f_local[4] = s * f3 + c * f4


fn _beam2d_transform_stiffness_local_to_global_in_place(
    mut k_local: List[List[Float64]], c: Float64, s: Float64
):
    if len(k_local) != 6:
        abort("beam2d stiffness matrix size mismatch")
    for i in range(6):
        if len(k_local[i]) != 6:
            abort("beam2d stiffness matrix row size mismatch")

    for i in range(6):
        var k0 = k_local[i][0]
        var k1 = k_local[i][1]
        var k3 = k_local[i][3]
        var k4 = k_local[i][4]
        k_local[i][0] = c * k0 - s * k1
        k_local[i][1] = s * k0 + c * k1
        k_local[i][3] = c * k3 - s * k4
        k_local[i][4] = s * k3 + c * k4

    var r00 = k_local[0][0]
    var r01 = k_local[0][1]
    var r02 = k_local[0][2]
    var r03 = k_local[0][3]
    var r04 = k_local[0][4]
    var r05 = k_local[0][5]
    var r10 = k_local[1][0]
    var r11 = k_local[1][1]
    var r12 = k_local[1][2]
    var r13 = k_local[1][3]
    var r14 = k_local[1][4]
    var r15 = k_local[1][5]
    k_local[0][0] = c * r00 - s * r10
    k_local[0][1] = c * r01 - s * r11
    k_local[0][2] = c * r02 - s * r12
    k_local[0][3] = c * r03 - s * r13
    k_local[0][4] = c * r04 - s * r14
    k_local[0][5] = c * r05 - s * r15
    k_local[1][0] = s * r00 + c * r10
    k_local[1][1] = s * r01 + c * r11
    k_local[1][2] = s * r02 + c * r12
    k_local[1][3] = s * r03 + c * r13
    k_local[1][4] = s * r04 + c * r14
    k_local[1][5] = s * r05 + c * r15

    var r30 = k_local[3][0]
    var r31 = k_local[3][1]
    var r32 = k_local[3][2]
    var r33 = k_local[3][3]
    var r34 = k_local[3][4]
    var r35 = k_local[3][5]
    var r40 = k_local[4][0]
    var r41 = k_local[4][1]
    var r42 = k_local[4][2]
    var r43 = k_local[4][3]
    var r44 = k_local[4][4]
    var r45 = k_local[4][5]
    k_local[3][0] = c * r30 - s * r40
    k_local[3][1] = c * r31 - s * r41
    k_local[3][2] = c * r32 - s * r42
    k_local[3][3] = c * r33 - s * r43
    k_local[3][4] = c * r34 - s * r44
    k_local[3][5] = c * r35 - s * r45
    k_local[4][0] = s * r30 + c * r40
    k_local[4][1] = s * r31 + c * r41
    k_local[4][2] = s * r32 + c * r42
    k_local[4][3] = s * r33 + c * r43
    k_local[4][4] = s * r34 + c * r44
    k_local[4][5] = s * r35 + c * r45
