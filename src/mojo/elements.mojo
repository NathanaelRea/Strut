from collections import List
from math import atan2, cos, hypot, sin, sqrt
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


fn beam_uniform_load_global(
    x1: Float64,
    y1: Float64,
    x2: Float64,
    y2: Float64,
    w: Float64,
) -> List[Float64]:
    var dx = x2 - x1
    var dy = y2 - y1
    var L = hypot(dx, dy)
    if L == 0.0:
        abort("zero-length element")
    var c = dx / L
    var s = dy / L

    var f_local: List[Float64] = [
        0.0,
        w * L / 2.0,
        w * L * L / 12.0,
        0.0,
        w * L / 2.0,
        -w * L * L / 12.0,
    ]

    var T: List[List[Float64]] = [
        [c, s, 0.0, 0.0, 0.0, 0.0],
        [-s, c, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, c, s, 0.0],
        [0.0, 0.0, 0.0, -s, c, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ]

    var f_global: List[Float64] = []
    f_global.resize(6, 0.0)
    for i in range(6):
        var sum = 0.0
        for j in range(6):
            sum += T[j][i] * f_local[j]
        f_global[i] = sum

    return f_global^


fn beam3d_local_stiffness(
    E: Float64,
    A: Float64,
    Iy: Float64,
    Iz: Float64,
    G: Float64,
    J: Float64,
    L: Float64,
) -> List[List[Float64]]:
    var k = _zero_matrix(12, 12)

    var EA_L = E * A / L
    var GJ_L = G * J / L

    var EIz = E * Iz
    var EIy = E * Iy
    var L2 = L * L
    var L3 = L2 * L

    var k1 = 12.0 * EIz / L3
    var k2 = 6.0 * EIz / L2
    var k3 = 4.0 * EIz / L
    var k4 = 2.0 * EIz / L

    var k5 = 12.0 * EIy / L3
    var k6 = 6.0 * EIy / L2
    var k7 = 4.0 * EIy / L
    var k8 = 2.0 * EIy / L

    k[0][0] = EA_L
    k[0][6] = -EA_L
    k[6][0] = -EA_L
    k[6][6] = EA_L

    k[3][3] = GJ_L
    k[3][9] = -GJ_L
    k[9][3] = -GJ_L
    k[9][9] = GJ_L

    k[1][1] = k1
    k[1][5] = k2
    k[1][7] = -k1
    k[1][11] = k2

    k[5][1] = k2
    k[5][5] = k3
    k[5][7] = -k2
    k[5][11] = k4

    k[7][1] = -k1
    k[7][5] = -k2
    k[7][7] = k1
    k[7][11] = -k2

    k[11][1] = k2
    k[11][5] = k4
    k[11][7] = -k2
    k[11][11] = k3

    k[2][2] = k5
    k[2][4] = -k6
    k[2][8] = -k5
    k[2][10] = -k6

    k[4][2] = -k6
    k[4][4] = k7
    k[4][8] = k6
    k[4][10] = k8

    k[8][2] = -k5
    k[8][4] = k6
    k[8][8] = k5
    k[8][10] = k6

    k[10][2] = -k6
    k[10][4] = k8
    k[10][8] = k6
    k[10][10] = k7

    return k^


fn _beam3d_rotation(
    x1: Float64,
    y1: Float64,
    z1: Float64,
    x2: Float64,
    y2: Float64,
    z2: Float64,
) -> List[List[Float64]]:
    var dx = x2 - x1
    var dy = y2 - y1
    var dz = z2 - z1
    var L = sqrt(dx * dx + dy * dy + dz * dz)
    if L == 0.0:
        abort("zero-length element")

    var lx: Float64
    var ly: Float64
    var lz: Float64
    (lx, ly, lz) = _normalize(dx, dy, dz)

    var rx = 0.0
    var ry = 0.0
    var rz = 1.0
    if abs(_dot(lx, ly, lz, rx, ry, rz)) > 0.9:
        rx = 0.0
        ry = 1.0
        rz = 0.0

    var yx: Float64
    var yy: Float64
    var yz: Float64
    (yx, yy, yz) = _cross(rx, ry, rz, lx, ly, lz)
    (yx, yy, yz) = _normalize(yx, yy, yz)

    var zx: Float64
    var zy: Float64
    var zz: Float64
    (zx, zy, zz) = _cross(lx, ly, lz, yx, yy, yz)

    return [
        [lx, ly, lz],
        [yx, yy, yz],
        [zx, zy, zz],
    ]


fn beam3d_global_stiffness(
    E: Float64,
    A: Float64,
    Iy: Float64,
    Iz: Float64,
    G: Float64,
    J: Float64,
    x1: Float64,
    y1: Float64,
    z1: Float64,
    x2: Float64,
    y2: Float64,
    z2: Float64,
) -> List[List[Float64]]:
    var dx = x2 - x1
    var dy = y2 - y1
    var dz = z2 - z1
    var L = sqrt(dx * dx + dy * dy + dz * dz)
    if L == 0.0:
        abort("zero-length element")

    var k_local = beam3d_local_stiffness(E, A, Iy, Iz, G, J, L)
    var R = _beam3d_rotation(x1, y1, z1, x2, y2, z2)

    var T = _zero_matrix(12, 12)
    var offsets = [0, 3, 6, 9]
    for b in range(4):
        var offset = offsets[b]
        for i in range(3):
            for j in range(3):
                T[offset + i][offset + j] = R[i][j]

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

    # Compute MITC4 basis vectors (g1, g2, g3) and local coords.
    var v1x = (x[2] + x[1] - x[3] - x[0]) * 0.5
    var v1y = (y[2] + y[1] - y[3] - y[0]) * 0.5
    var v1z = (z[2] + z[1] - z[3] - z[0]) * 0.5
    var v2x = (x[3] + x[2] - x[1] - x[0]) * 0.5
    var v2y = (y[3] + y[2] - y[1] - y[0]) * 0.5
    var v2z = (z[3] + z[2] - z[1] - z[0]) * 0.5

    var g1x: Float64
    var g1y: Float64
    var g1z: Float64
    (g1x, g1y, g1z) = _normalize(v1x, v1y, v1z)

    var alpha = _dot(v2x, v2y, v2z, g1x, g1y, g1z)
    v2x -= alpha * g1x
    v2y -= alpha * g1y
    v2z -= alpha * g1z

    var g2x: Float64
    var g2y: Float64
    var g2z: Float64
    (g2x, g2y, g2z) = _normalize(v2x, v2y, v2z)

    var g3x: Float64
    var g3y: Float64
    var g3z: Float64
    (g3x, g3y, g3z) = _cross(g1x, g1y, g1z, g2x, g2y, g2z)

    # Match OpenSees ShellMITC4 basis orientation without extra axis flips.

    var xl: List[List[Float64]] = []
    for _ in range(2):
        var row: List[Float64] = []
        row.resize(4, 0.0)
        xl.append(row^)
    for i in range(4):
        xl[0][i] = _dot(x[i], y[i], z[i], g1x, g1y, g1z)
        xl[1][i] = _dot(x[i], y[i], z[i], g2x, g2y, g2z)

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
    var Ktt = G * h

    # dd = [membrane(3), bending(3), shear(2)]
    var dd = _zero_matrix(8, 8)
    for i in range(3):
        for j in range(3):
            dd[i][j] = Dm[i][j] * h
            dd[i + 3][j + 3] = -Db[i][j]
    dd[6][6] = Ds
    dd[7][7] = Ds

    # MITC4 shear interpolation helpers.
    var dx34 = xl[0][2] - xl[0][3]
    var dy34 = xl[1][2] - xl[1][3]
    var dx21 = xl[0][1] - xl[0][0]
    var dy21 = xl[1][1] - xl[1][0]
    var dx32 = xl[0][2] - xl[0][1]
    var dy32 = xl[1][2] - xl[1][1]
    var dx41 = xl[0][3] - xl[0][0]
    var dy41 = xl[1][3] - xl[1][0]

    var Gm = _zero_matrix(4, 12)
    var one_over_four = 0.25
    Gm[0][0] = -0.5
    Gm[0][1] = -dy41 * one_over_four
    Gm[0][2] = dx41 * one_over_four
    Gm[0][9] = 0.5
    Gm[0][10] = -dy41 * one_over_four
    Gm[0][11] = dx41 * one_over_four

    Gm[1][0] = -0.5
    Gm[1][1] = -dy21 * one_over_four
    Gm[1][2] = dx21 * one_over_four
    Gm[1][3] = 0.5
    Gm[1][4] = -dy21 * one_over_four
    Gm[1][5] = dx21 * one_over_four

    Gm[2][3] = -0.5
    Gm[2][4] = -dy32 * one_over_four
    Gm[2][5] = dx32 * one_over_four
    Gm[2][6] = 0.5
    Gm[2][7] = -dy32 * one_over_four
    Gm[2][8] = dx32 * one_over_four

    Gm[3][6] = 0.5
    Gm[3][7] = -dy34 * one_over_four
    Gm[3][8] = dx34 * one_over_four
    Gm[3][9] = -0.5
    Gm[3][10] = -dy34 * one_over_four
    Gm[3][11] = dx34 * one_over_four

    var Ax = -xl[0][0] + xl[0][1] + xl[0][2] - xl[0][3]
    var Bx = xl[0][0] - xl[0][1] + xl[0][2] - xl[0][3]
    var Cx = -xl[0][0] - xl[0][1] + xl[0][2] + xl[0][3]

    var Ay = -xl[1][0] + xl[1][1] + xl[1][2] - xl[1][3]
    var By = xl[1][0] - xl[1][1] + xl[1][2] - xl[1][3]
    var Cy = -xl[1][0] - xl[1][1] + xl[1][2] + xl[1][3]

    var alph = atan2(Ay, Ax)
    var beta = 0.5 * 3.141592653589793 - atan2(Cx, Cy)
    var Rot = _zero_matrix(2, 2)
    Rot[0][0] = sin(beta)
    Rot[0][1] = -sin(alph)
    Rot[1][0] = -cos(beta)
    Rot[1][1] = cos(alph)

    var sg = [-1.0 / sqrt(3.0), 1.0 / sqrt(3.0), 1.0 / sqrt(3.0), -1.0 / sqrt(3.0)]
    var tg = [-1.0 / sqrt(3.0), -1.0 / sqrt(3.0), 1.0 / sqrt(3.0), 1.0 / sqrt(3.0)]
    var wg = [1.0, 1.0, 1.0, 1.0]

    var K_local = _zero_matrix(24, 24)
    for gp in range(4):
        var ss = sg[gp]
        var tt = tg[gp]

        var shp: List[List[Float64]] = []
        for _ in range(3):
            var row: List[Float64] = []
            row.resize(4, 0.0)
            shp.append(row^)

        var s = [-0.5, 0.5, 0.5, -0.5]
        var t = [-0.5, -0.5, 0.5, 0.5]
        for i in range(4):
            shp[2][i] = (0.5 + s[i] * ss) * (0.5 + t[i] * tt)
            shp[0][i] = s[i] * (0.5 + t[i] * tt)
            shp[1][i] = t[i] * (0.5 + s[i] * ss)

        var xs = _zero_matrix(2, 2)
        for i in range(2):
            for j in range(2):
                for k in range(4):
                    xs[i][j] += xl[i][k] * shp[j][k]

        var xsj = xs[0][0] * xs[1][1] - xs[0][1] * xs[1][0]
        if xsj == 0.0:
            abort("shell4 singular jacobian")
        var jinv = 1.0 / xsj
        var sx00 = xs[1][1] * jinv
        var sx11 = xs[0][0] * jinv
        var sx01 = -xs[0][1] * jinv
        var sx10 = -xs[1][0] * jinv

        for i in range(4):
            var temp = shp[0][i] * sx00 + shp[1][i] * sx10
            shp[1][i] = shp[0][i] * sx01 + shp[1][i] * sx11
            shp[0][i] = temp

        var dvol = wg[gp] * xsj

        var Ms = _zero_matrix(2, 4)
        Ms[1][0] = 1.0 - ss
        Ms[0][1] = 1.0 - tt
        Ms[1][2] = 1.0 + ss
        Ms[0][3] = 1.0 + tt

        var Bsv = _zero_matrix(2, 12)
        for i in range(2):
            for j in range(12):
                var sum = 0.0
                for k in range(4):
                    sum += Ms[i][k] * Gm[k][j]
                Bsv[i][j] = sum

        var r1 = Cx + ss * Bx
        var r3 = Cy + ss * By
        r1 = sqrt(r1 * r1 + r3 * r3)
        var r2 = Ax + tt * Bx
        r3 = Ay + tt * By
        r2 = sqrt(r2 * r2 + r3 * r3)

        for j in range(12):
            Bsv[0][j] = Bsv[0][j] * r1 / (8.0 * xsj)
            Bsv[1][j] = Bsv[1][j] * r2 / (8.0 * xsj)

        var Bs = _zero_matrix(2, 12)
        for j in range(12):
            Bs[0][j] = Rot[0][0] * Bsv[0][j] + Rot[0][1] * Bsv[1][j]
            Bs[1][j] = Rot[1][0] * Bsv[0][j] + Rot[1][1] * Bsv[1][j]

        var saveB: List[List[List[Float64]]] = []
        for _ in range(4):
            var nodeB: List[List[Float64]] = []
            for _ in range(8):
                var row: List[Float64] = []
                row.resize(6, 0.0)
                nodeB.append(row^)
            saveB.append(nodeB^)

        var Bdrill_all: List[List[Float64]] = []
        for _ in range(4):
            var row: List[Float64] = []
            row.resize(6, 0.0)
            Bdrill_all.append(row^)

        for j in range(4):
            var Bmem = _zero_matrix(3, 2)
            Bmem[0][0] = shp[0][j]
            Bmem[1][1] = shp[1][j]
            Bmem[2][0] = shp[1][j]
            Bmem[2][1] = shp[0][j]

            var Bb = _zero_matrix(3, 2)
            Bb[0][1] = -shp[0][j]
            Bb[1][0] = shp[1][j]
            Bb[2][0] = shp[0][j]
            Bb[2][1] = -shp[1][j]

            var Bshear = _zero_matrix(2, 3)
            for p in range(3):
                Bshear[0][p] = Bs[0][j * 3 + p]
                Bshear[1][p] = Bs[1][j * 3 + p]

            var Gmem = _zero_matrix(2, 3)
            Gmem[0][0] = g1x
            Gmem[0][1] = g1y
            Gmem[0][2] = g1z
            Gmem[1][0] = g2x
            Gmem[1][1] = g2y
            Gmem[1][2] = g2z

            var BmemShell = _zero_matrix(3, 3)
            var BbendShell = _zero_matrix(3, 3)
            for p in range(3):
                for q in range(3):
                    var sum_m = 0.0
                    var sum_b = 0.0
                    for r in range(2):
                        sum_m += Bmem[p][r] * Gmem[r][q]
                        sum_b += Bb[p][r] * Gmem[r][q]
                    BmemShell[p][q] = sum_m
                    BbendShell[p][q] = sum_b

            var Gshear = _zero_matrix(3, 6)
            Gshear[0][0] = g3x
            Gshear[0][1] = g3y
            Gshear[0][2] = g3z
            Gshear[1][3] = g1x
            Gshear[1][4] = g1y
            Gshear[1][5] = g1z
            Gshear[2][3] = g2x
            Gshear[2][4] = g2y
            Gshear[2][5] = g2z

            var BshearShell = _zero_matrix(2, 6)
            for p in range(2):
                for q in range(6):
                    var sum_s = 0.0
                    for r in range(3):
                        sum_s += Bshear[p][r] * Gshear[r][q]
                    BshearShell[p][q] = sum_s

            var B = _zero_matrix(8, 6)
            for p in range(3):
                for q in range(3):
                    B[p][q] = BmemShell[p][q]
            for p in range(3):
                for q in range(3):
                    B[p + 3][q + 3] = BbendShell[p][q]
            for p in range(2):
                for q in range(6):
                    B[p + 6][q] = BshearShell[p][q]

            saveB[j] = B^

            var B1 = -0.5 * shp[1][j]
            var B2 = 0.5 * shp[0][j]
            var B6 = -shp[2][j]
            Bdrill_all[j][0] = B1 * g1x + B2 * g2x
            Bdrill_all[j][1] = B1 * g1y + B2 * g2y
            Bdrill_all[j][2] = B1 * g1z + B2 * g2z
            Bdrill_all[j][3] = B6 * g3x
            Bdrill_all[j][4] = B6 * g3y
            Bdrill_all[j][5] = B6 * g3z

        for j in range(4):
            var BJ = _zero_matrix(8, 6)
            for p in range(8):
                for q in range(6):
                    BJ[p][q] = saveB[j][p][q]
            for p in range(3, 6):
                for q in range(3, 6):
                    BJ[p][q] *= -1.0
            var BJtranD = _zero_matrix(6, 8)
            for p in range(6):
                for q in range(8):
                    var sum = 0.0
                    for r in range(8):
                        sum += BJ[r][p] * dd[r][q]
                    BJtranD[p][q] = sum * dvol

            var BdrillJ: List[Float64] = []
            BdrillJ.resize(6, 0.0)
            for p in range(6):
                BdrillJ[p] = Bdrill_all[j][p]
            for p in range(6):
                BdrillJ[p] *= (Ktt * dvol)

            for k in range(4):
                var stiffJK = _zero_matrix(6, 6)
                for p in range(6):
                    for q in range(6):
                        var sum = 0.0
                        for r in range(8):
                            sum += BJtranD[p][r] * saveB[k][r][q]
                        stiffJK[p][q] = sum + BdrillJ[p] * Bdrill_all[k][q]
                var i0 = j * 6
                var j0 = k * 6
                for p in range(6):
                    for q in range(6):
                        K_local[i0 + p][j0 + q] += stiffJK[p][q]

    return K_local^
