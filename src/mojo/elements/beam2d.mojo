from collections import List
from math import atan2, hypot, sqrt
from os import abort

from elements.utils import _matvec, _zero_matrix
from linalg import matmul, transpose


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


fn beam2d_pdelta_global_stiffness(
    E: Float64,
    A: Float64,
    I: Float64,
    x1: Float64,
    y1: Float64,
    x2: Float64,
    y2: Float64,
    u_elem_global: List[Float64],
) -> List[List[Float64]]:
    var dx = x2 - x1
    var dy = y2 - y1
    var L = hypot(dx, dy)
    if L == 0.0:
        abort("zero-length element")
    var c = dx / L
    var s = dy / L

    var T: List[List[Float64]] = [
        [c, s, 0.0, 0.0, 0.0, 0.0],
        [-s, c, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, c, s, 0.0],
        [0.0, 0.0, 0.0, -s, c, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ]

    var u_local = _matvec(T, u_elem_global)
    var du = u_local[3] - u_local[0]
    var N = (E * A / L) * du
    var N_over_L = N / L

    var k_local = beam_local_stiffness(E, A, I, L)
    k_local[1][1] += N_over_L
    k_local[4][4] += N_over_L
    k_local[1][4] -= N_over_L
    k_local[4][1] -= N_over_L

    return matmul(transpose(T), matmul(k_local, T))


fn beam2d_corotational_global_stiffness(
    E: Float64,
    A: Float64,
    I: Float64,
    x1: Float64,
    y1: Float64,
    x2: Float64,
    y2: Float64,
    u_elem_global: List[Float64],
) -> List[List[Float64]]:
    var dx = x2 - x1
    var dy = y2 - y1
    var L = hypot(dx, dy)
    if L == 0.0:
        abort("zero-length element")
    var c = dx / L
    var s = dy / L

    var Tlg: List[List[Float64]] = [
        [c, s, 0.0, 0.0, 0.0, 0.0],
        [-s, c, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, c, s, 0.0],
        [0.0, 0.0, 0.0, -s, c, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ]

    var u_local = _matvec(Tlg, u_elem_global)
    var dulx = u_local[3] - u_local[0]
    var duly = u_local[4] - u_local[1]
    var Lx = L + dulx
    var Ly = duly
    var Ln = sqrt(Lx * Lx + Ly * Ly)
    if Ln == 0.0:
        abort("zero-length element")

    var cosAlpha = Lx / Ln
    var sinAlpha = Ly / Ln
    var alpha = atan2(sinAlpha, cosAlpha)

    var ub: List[Float64] = []
    ub.resize(3, 0.0)
    ub[0] = Ln - L
    ub[1] = u_local[2] - alpha
    ub[2] = u_local[5] - alpha

    var EA_L = E * A / L
    var EI_L = E * I / L

    var kb: List[List[Float64]] = [
        [EA_L, 0.0, 0.0],
        [0.0, 4.0 * EI_L, 2.0 * EI_L],
        [0.0, 2.0 * EI_L, 4.0 * EI_L],
    ]

    var pb: List[Float64] = []
    pb.resize(3, 0.0)
    for i in range(3):
        var sum = 0.0
        for j in range(3):
            sum += kb[i][j] * ub[j]
        pb[i] = sum

    var Tbl: List[List[Float64]] = [
        [-cosAlpha, -sinAlpha, 0.0, cosAlpha, sinAlpha, 0.0],
        [-sinAlpha / Ln, cosAlpha / Ln, 1.0, sinAlpha / Ln, -cosAlpha / Ln, 0.0],
        [-sinAlpha / Ln, cosAlpha / Ln, 0.0, sinAlpha / Ln, -cosAlpha / Ln, 1.0],
    ]

    var tmp: List[List[Float64]] = []
    for _ in range(3):
        var row: List[Float64] = []
        row.resize(6, 0.0)
        tmp.append(row^)
    for i in range(3):
        for k in range(3):
            var ik = kb[i][k]
            if ik == 0.0:
                continue
            for j in range(6):
                tmp[i][j] += ik * Tbl[k][j]

    var kl: List[List[Float64]] = []
    for _ in range(6):
        var row: List[Float64] = []
        row.resize(6, 0.0)
        kl.append(row^)
    for i in range(6):
        for k in range(3):
            var ik = Tbl[k][i]
            if ik == 0.0:
                continue
            for j in range(6):
                kl[i][j] += ik * tmp[k][j]

    var s2 = sinAlpha * sinAlpha
    var c2 = cosAlpha * cosAlpha
    var cs = sinAlpha * cosAlpha

    var kg0 = _zero_matrix(6, 6)
    kg0[0][0] = s2
    kg0[3][3] = s2
    kg0[0][1] = -cs
    kg0[3][4] = -cs
    kg0[1][0] = -cs
    kg0[4][3] = -cs
    kg0[1][1] = c2
    kg0[4][4] = c2

    kg0[0][3] = -s2
    kg0[3][0] = -s2
    kg0[0][4] = cs
    kg0[3][1] = cs
    kg0[1][3] = cs
    kg0[4][0] = cs
    kg0[1][4] = -c2
    kg0[4][1] = -c2

    var kg12 = _zero_matrix(6, 6)
    kg12[0][0] = -2.0 * cs
    kg12[3][3] = -2.0 * cs
    kg12[0][1] = c2 - s2
    kg12[3][4] = c2 - s2
    kg12[1][0] = c2 - s2
    kg12[4][3] = c2 - s2
    kg12[1][1] = 2.0 * cs
    kg12[4][4] = 2.0 * cs

    kg12[0][3] = 2.0 * cs
    kg12[3][0] = 2.0 * cs
    kg12[0][4] = -c2 + s2
    kg12[3][1] = -c2 + s2
    kg12[1][3] = -c2 + s2
    kg12[4][0] = -c2 + s2
    kg12[1][4] = -2.0 * cs
    kg12[4][1] = -2.0 * cs

    var scale0 = pb[0] / Ln
    var scale12 = (pb[1] + pb[2]) / (Ln * Ln)
    for i in range(6):
        for j in range(6):
            kl[i][j] += kg0[i][j] * scale0 + kg12[i][j] * scale12

    return matmul(transpose(Tlg), matmul(kl, Tlg))


fn beam2d_corotational_global_internal_force(
    E: Float64,
    A: Float64,
    I: Float64,
    x1: Float64,
    y1: Float64,
    x2: Float64,
    y2: Float64,
    u_elem_global: List[Float64],
) -> List[Float64]:
    var dx = x2 - x1
    var dy = y2 - y1
    var L = hypot(dx, dy)
    if L == 0.0:
        abort("zero-length element")
    var c = dx / L
    var s = dy / L

    var Tlg: List[List[Float64]] = [
        [c, s, 0.0, 0.0, 0.0, 0.0],
        [-s, c, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, c, s, 0.0],
        [0.0, 0.0, 0.0, -s, c, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ]

    var u_local = _matvec(Tlg, u_elem_global)
    var dulx = u_local[3] - u_local[0]
    var duly = u_local[4] - u_local[1]
    var Lx = L + dulx
    var Ly = duly
    var Ln = sqrt(Lx * Lx + Ly * Ly)
    if Ln == 0.0:
        abort("zero-length element")

    var cosAlpha = Lx / Ln
    var sinAlpha = Ly / Ln
    var alpha = atan2(sinAlpha, cosAlpha)

    var ub: List[Float64] = []
    ub.resize(3, 0.0)
    ub[0] = Ln - L
    ub[1] = u_local[2] - alpha
    ub[2] = u_local[5] - alpha

    var EA_L = E * A / L
    var EI_L = E * I / L

    var kb: List[List[Float64]] = [
        [EA_L, 0.0, 0.0],
        [0.0, 4.0 * EI_L, 2.0 * EI_L],
        [0.0, 2.0 * EI_L, 4.0 * EI_L],
    ]

    var pb: List[Float64] = []
    pb.resize(3, 0.0)
    for i in range(3):
        var sum = 0.0
        for j in range(3):
            sum += kb[i][j] * ub[j]
        pb[i] = sum

    var Tbl: List[List[Float64]] = [
        [-cosAlpha, -sinAlpha, 0.0, cosAlpha, sinAlpha, 0.0],
        [-sinAlpha / Ln, cosAlpha / Ln, 1.0, sinAlpha / Ln, -cosAlpha / Ln, 0.0],
        [-sinAlpha / Ln, cosAlpha / Ln, 0.0, sinAlpha / Ln, -cosAlpha / Ln, 1.0],
    ]

    var pl: List[Float64] = []
    pl.resize(6, 0.0)
    for i in range(6):
        var sum = 0.0
        for j in range(3):
            sum += Tbl[j][i] * pb[j]
        pl[i] = sum

    var pg: List[Float64] = []
    pg.resize(6, 0.0)
    for i in range(6):
        var sum = 0.0
        for j in range(6):
            sum += Tlg[j][i] * pl[j]
        pg[i] = sum

    return pg^
