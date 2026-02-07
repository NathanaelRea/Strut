from collections import List
from math import hypot, sqrt
from os import abort

from linalg import matmul, transpose


fn _zero_matrix(rows: Int, cols: Int) -> List[List[Float64]]:
    var out: List[List[Float64]] = []
    for _ in range(rows):
        var row: List[Float64] = []
        row.resize(cols, 0.0)
        out.append(row^)
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


fn _quad_shape_gradients(
    xi: Float64,
    eta: Float64,
    mut N: List[Float64],
    mut dN_dxi: List[Float64],
    mut dN_deta: List[Float64],
):
    N.resize(4, 0.0)
    dN_dxi.resize(4, 0.0)
    dN_deta.resize(4, 0.0)

    N[0] = 0.25 * (1.0 - xi) * (1.0 - eta)
    N[1] = 0.25 * (1.0 + xi) * (1.0 - eta)
    N[2] = 0.25 * (1.0 + xi) * (1.0 + eta)
    N[3] = 0.25 * (1.0 - xi) * (1.0 + eta)

    dN_dxi[0] = -0.25 * (1.0 - eta)
    dN_dxi[1] = 0.25 * (1.0 - eta)
    dN_dxi[2] = 0.25 * (1.0 + eta)
    dN_dxi[3] = -0.25 * (1.0 + eta)

    dN_deta[0] = -0.25 * (1.0 - xi)
    dN_deta[1] = -0.25 * (1.0 + xi)
    dN_deta[2] = 0.25 * (1.0 + xi)
    dN_deta[3] = 0.25 * (1.0 - xi)


fn beam_local_stiffness(
    E: Float64, A: Float64, I: Float64, L: Float64
) -> List[List[Float64]]:
    var k: List[List[Float64]] = []
    for _ in range(6):
        var row: List[Float64] = []
        row.resize(6, 0.0)
        k.append(row^)

    var EA_L = E * A / L
    var EI = E * I
    var L2 = L * L
    var L3 = L2 * L

    k[0][0] = EA_L
    k[0][3] = -EA_L
    k[3][0] = -EA_L
    k[3][3] = EA_L

    k[1][1] = 12.0 * EI / L3
    k[1][2] = 6.0 * EI / L2
    k[1][4] = -12.0 * EI / L3
    k[1][5] = 6.0 * EI / L2

    k[2][1] = 6.0 * EI / L2
    k[2][2] = 4.0 * EI / L
    k[2][4] = -6.0 * EI / L2
    k[2][5] = 2.0 * EI / L

    k[4][1] = -12.0 * EI / L3
    k[4][2] = -6.0 * EI / L2
    k[4][4] = 12.0 * EI / L3
    k[4][5] = -6.0 * EI / L2

    k[5][1] = 6.0 * EI / L2
    k[5][2] = 2.0 * EI / L
    k[5][4] = -6.0 * EI / L2
    k[5][5] = 4.0 * EI / L

    return k^


fn beam_global_stiffness(
    E: Float64,
    A: Float64,
    I: Float64,
    x1: Float64,
    y1: Float64,
    x2: Float64,
    y2: Float64,
) -> List[List[Float64]]:
    var dx = x2 - x1
    var dy = y2 - y1
    var L = hypot(dx, dy)
    if L == 0.0:
        abort("zero-length element")
    var c = dx / L
    var s = dy / L

    var k_local = beam_local_stiffness(E, A, I, L)

    var T: List[List[Float64]] = [
        [c, s, 0.0, 0.0, 0.0, 0.0],
        [-s, c, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, c, s, 0.0],
        [0.0, 0.0, 0.0, -s, c, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ]

    return matmul(transpose(T), matmul(k_local, T))


fn truss_global_stiffness(
    E: Float64,
    A: Float64,
    x1: Float64,
    y1: Float64,
    x2: Float64,
    y2: Float64,
) -> List[List[Float64]]:
    var dx = x2 - x1
    var dy = y2 - y1
    var L = hypot(dx, dy)
    if L == 0.0:
        abort("zero-length element")
    var c = dx / L
    var s = dy / L
    var k = E * A / L

    return [
        [k * c * c, k * c * s, -k * c * c, -k * c * s],
        [k * c * s, k * s * s, -k * c * s, -k * s * s],
        [-k * c * c, -k * c * s, k * c * c, k * c * s],
        [-k * c * s, -k * s * s, k * c * s, k * s * s],
    ]


fn link_global_stiffness(
    dirs: List[Int],
    ks: List[Float64],
) -> List[List[Float64]]:
    if len(dirs) != len(ks):
        abort("dirs/materials length mismatch")

    var k_global: List[List[Float64]] = []
    for _ in range(4):
        var row: List[Float64] = []
        row.resize(4, 0.0)
        k_global.append(row^)

    for i in range(len(dirs)):
        var dof = dirs[i]
        var k = ks[i]
        var a = -1
        var b = -1
        if dof == 1:
            a = 0
            b = 2
        elif dof == 2:
            a = 1
            b = 3
        else:
            abort("unsupported link dir")

        k_global[a][a] += k
        k_global[b][b] += k
        k_global[a][b] -= k
        k_global[b][a] -= k

    return k_global^


fn quad4_plane_stress_stiffness(
    E: Float64,
    nu: Float64,
    t: Float64,
    x: List[Float64],
    y: List[Float64],
) -> List[List[Float64]]:
    if len(x) != 4 or len(y) != 4:
        abort("quad4 requires 4 nodes")

    var K = _zero_matrix(8, 8)

    var factor = E / (1.0 - nu * nu)
    var D: List[List[Float64]] = [
        [factor, factor * nu, 0.0],
        [factor * nu, factor, 0.0],
        [0.0, 0.0, factor * (1.0 - nu) / 2.0],
    ]

    var g = 1.0 / sqrt(3.0)
    var gauss = [-g, g]

    for xi in gauss:
        for eta in gauss:
            var N: List[Float64] = []
            var dN_dxi: List[Float64] = []
            var dN_deta: List[Float64] = []
            _quad_shape_gradients(xi, eta, N, dN_dxi, dN_deta)

            var dx_dxi = 0.0
            var dy_dxi = 0.0
            var dx_deta = 0.0
            var dy_deta = 0.0
            for i in range(4):
                dx_dxi += dN_dxi[i] * x[i]
                dy_dxi += dN_dxi[i] * y[i]
                dx_deta += dN_deta[i] * x[i]
                dy_deta += dN_deta[i] * y[i]

            var detJ = dx_dxi * dy_deta - dx_deta * dy_dxi
            if detJ == 0.0:
                abort("quad4 singular jacobian")

            var invJ11 = dy_deta / detJ
            var invJ12 = -dy_dxi / detJ
            var invJ21 = -dx_deta / detJ
            var invJ22 = dx_dxi / detJ

            var dN_dx: List[Float64] = []
            var dN_dy: List[Float64] = []
            dN_dx.resize(4, 0.0)
            dN_dy.resize(4, 0.0)
            for i in range(4):
                dN_dx[i] = invJ11 * dN_dxi[i] + invJ12 * dN_deta[i]
                dN_dy[i] = invJ21 * dN_dxi[i] + invJ22 * dN_deta[i]

            var B = _zero_matrix(3, 8)
            for i in range(4):
                var col_u = 2 * i
                var col_v = col_u + 1
                B[0][col_u] = dN_dx[i]
                B[1][col_v] = dN_dy[i]
                B[2][col_u] = dN_dy[i]
                B[2][col_v] = dN_dx[i]

            var temp = _zero_matrix(3, 8)
            for i in range(3):
                for j in range(8):
                    temp[i][j] = (
                        D[i][0] * B[0][j]
                        + D[i][1] * B[1][j]
                        + D[i][2] * B[2][j]
                    )

            for i in range(8):
                for j in range(8):
                    var sum = (
                        B[0][i] * temp[0][j]
                        + B[1][i] * temp[1][j]
                        + B[2][i] * temp[2][j]
                    )
                    K[i][j] += sum * t * detJ

    return K^


fn shell4_mindlin_stiffness(
    E: Float64,
    nu: Float64,
    h: Float64,
    x: List[Float64],
    y: List[Float64],
    z: List[Float64],
) -> List[List[Float64]]:
    if len(x) != 4 or len(y) != 4 or len(z) != 4:
        abort("shell4 requires 4 nodes")

    var v1x = x[1] - x[0]
    var v1y = y[1] - y[0]
    var v1z = z[1] - z[0]
    var v2x = x[3] - x[0]
    var v2y = y[3] - y[0]
    var v2z = z[3] - z[0]

    var nx: Float64
    var ny: Float64
    var nz: Float64
    (nx, ny, nz) = _cross(v1x, v1y, v1z, v2x, v2y, v2z)
    (nx, ny, nz) = _normalize(nx, ny, nz)

    var xx: Float64
    var xy: Float64
    var xz: Float64
    (xx, xy, xz) = _normalize(v1x, v1y, v1z)

    var yx: Float64
    var yy: Float64
    var yz: Float64
    (yx, yy, yz) = _cross(nx, ny, nz, xx, xy, xz)

    var xl: List[Float64] = []
    var yl: List[Float64] = []
    xl.resize(4, 0.0)
    yl.resize(4, 0.0)
    for i in range(4):
        var dx = x[i] - x[0]
        var dy = y[i] - y[0]
        var dz = z[i] - z[0]
        xl[i] = _dot(dx, dy, dz, xx, xy, xz)
        yl[i] = _dot(dx, dy, dz, yx, yy, yz)

    var factor = E / (1.0 - nu * nu)
    var Dm: List[List[Float64]] = [
        [factor, factor * nu, 0.0],
        [factor * nu, factor, 0.0],
        [0.0, 0.0, factor * (1.0 - nu) / 2.0],
    ]
    var bend_factor = E * h * h * h / (12.0 * (1.0 - nu * nu))
    var Db: List[List[Float64]] = [
        [bend_factor, bend_factor * nu, 0.0],
        [bend_factor * nu, bend_factor, 0.0],
        [0.0, 0.0, bend_factor * (1.0 - nu) / 2.0],
    ]
    var G = E / (2.0 * (1.0 + nu))
    var kappa = 5.0 / 6.0
    var Ds = kappa * G * h

    var K_local = _zero_matrix(24, 24)

    var g = 1.0 / sqrt(3.0)
    var gauss = [-g, g]
    for xi in gauss:
        for eta in gauss:
            var N: List[Float64] = []
            var dN_dxi: List[Float64] = []
            var dN_deta: List[Float64] = []
            _quad_shape_gradients(xi, eta, N, dN_dxi, dN_deta)

            var dx_dxi = 0.0
            var dy_dxi = 0.0
            var dx_deta = 0.0
            var dy_deta = 0.0
            for i in range(4):
                dx_dxi += dN_dxi[i] * xl[i]
                dy_dxi += dN_dxi[i] * yl[i]
                dx_deta += dN_deta[i] * xl[i]
                dy_deta += dN_deta[i] * yl[i]

            var detJ = dx_dxi * dy_deta - dx_deta * dy_dxi
            if detJ == 0.0:
                abort("shell4 singular jacobian")

            var invJ11 = dy_deta / detJ
            var invJ12 = -dy_dxi / detJ
            var invJ21 = -dx_deta / detJ
            var invJ22 = dx_dxi / detJ

            var dN_dx: List[Float64] = []
            var dN_dy: List[Float64] = []
            dN_dx.resize(4, 0.0)
            dN_dy.resize(4, 0.0)
            for i in range(4):
                dN_dx[i] = invJ11 * dN_dxi[i] + invJ12 * dN_deta[i]
                dN_dy[i] = invJ21 * dN_dxi[i] + invJ22 * dN_deta[i]

            var Bm = _zero_matrix(3, 8)
            var Bb = _zero_matrix(3, 8)
            var Bs = _zero_matrix(2, 12)
            for i in range(4):
                var col_uv = 2 * i
                var col_rx = 2 * i
                Bm[0][col_uv] = dN_dx[i]
                Bm[1][col_uv + 1] = dN_dy[i]
                Bm[2][col_uv] = dN_dy[i]
                Bm[2][col_uv + 1] = dN_dx[i]

                Bb[0][col_rx] = dN_dx[i]
                Bb[1][col_rx + 1] = dN_dy[i]
                Bb[2][col_rx] = dN_dy[i]
                Bb[2][col_rx + 1] = dN_dx[i]

                var col_w = 3 * i
                Bs[0][col_w] = dN_dx[i]
                Bs[0][col_w + 1] = -N[i]
                Bs[1][col_w] = dN_dy[i]
                Bs[1][col_w + 2] = -N[i]

            var temp_m = _zero_matrix(3, 8)
            for i in range(3):
                for j in range(8):
                    temp_m[i][j] = (
                        Dm[i][0] * Bm[0][j]
                        + Dm[i][1] * Bm[1][j]
                        + Dm[i][2] * Bm[2][j]
                    )

            for i in range(8):
                for j in range(8):
                    var sum = (
                        Bm[0][i] * temp_m[0][j]
                        + Bm[1][i] * temp_m[1][j]
                        + Bm[2][i] * temp_m[2][j]
                    )
                    var a = (i // 2) * 6 + (i % 2)
                    var b = (j // 2) * 6 + (j % 2)
                    K_local[a][b] += sum * h * detJ

            var temp_b = _zero_matrix(3, 8)
            for i in range(3):
                for j in range(8):
                    temp_b[i][j] = (
                        Db[i][0] * Bb[0][j]
                        + Db[i][1] * Bb[1][j]
                        + Db[i][2] * Bb[2][j]
                    )

            for i in range(8):
                for j in range(8):
                    var sum = (
                        Bb[0][i] * temp_b[0][j]
                        + Bb[1][i] * temp_b[1][j]
                        + Bb[2][i] * temp_b[2][j]
                    )
                    var a = (i // 2) * 6 + 3 + (i % 2)
                    var b = (j // 2) * 6 + 3 + (j % 2)
                    K_local[a][b] += sum * detJ

            var temp_s = _zero_matrix(2, 12)
            for i in range(2):
                for j in range(12):
                    var d = Ds
                    if i == 0:
                        temp_s[i][j] = d * Bs[0][j]
                    else:
                        temp_s[i][j] = d * Bs[1][j]

            for i in range(12):
                for j in range(12):
                    var sum = Bs[0][i] * temp_s[0][j] + Bs[1][i] * temp_s[1][j]
                    var a = (i // 3) * 6 + 2 + (i % 3)
                    var b = (j // 3) * 6 + 2 + (j % 3)
                    K_local[a][b] += sum * detJ

    var k_drill = 1e-6 * E * h
    for i in range(4):
        var idx = i * 6 + 5
        K_local[idx][idx] += k_drill

    var Rt: List[List[Float64]] = [
        [xx, xy, xz],
        [yx, yy, yz],
        [nx, ny, nz],
    ]
    var T = _zero_matrix(6, 6)
    for i in range(3):
        for j in range(3):
            T[i][j] = Rt[i][j]
            T[i + 3][j + 3] = Rt[i][j]

    var K_global = _zero_matrix(24, 24)
    for bi in range(4):
        for bj in range(4):
            var i0 = bi * 6
            var j0 = bj * 6
            for a in range(6):
                for b in range(6):
                    var sum = 0.0
                    for p in range(6):
                        for q in range(6):
                            sum += T[p][a] * K_local[i0 + p][j0 + q] * T[q][b]
                    K_global[i0 + a][j0 + b] = sum

    return K_global^
