from collections import List
from math import atan2, cos, sin, sqrt
from materials import UniMaterialDef, UniMaterialState
from os import abort

from elements.utils import _cross, _dot, _ensure_zero_vector, _normalize, _zero_matrix
from sections import LayeredShellSectionDef, layered_shell_set_trial_from_offset


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
    var s = [-0.5, 0.5, 0.5, -0.5]
    var t = [-0.5, -0.5, 0.5, 0.5]

    var K_local = _zero_matrix(24, 24)

    var shp = _zero_matrix(3, 4)
    var xs = _zero_matrix(2, 2)
    var Ms = _zero_matrix(2, 4)
    var Bsv = _zero_matrix(2, 12)
    var Bs = _zero_matrix(2, 12)

    var saveB: List[List[List[Float64]]] = []
    for _ in range(4):
        var nodeB = _zero_matrix(8, 6)
        saveB.append(nodeB^)

    var Bdrill_all: List[List[Float64]] = []
    for _ in range(4):
        var row: List[Float64] = []
        row.resize(6, 0.0)
        Bdrill_all.append(row^)

    var Bmem = _zero_matrix(3, 2)
    var Bb = _zero_matrix(3, 2)
    var Bshear = _zero_matrix(2, 3)

    var Gmem = _zero_matrix(2, 3)
    Gmem[0][0] = g1x
    Gmem[0][1] = g1y
    Gmem[0][2] = g1z
    Gmem[1][0] = g2x
    Gmem[1][1] = g2y
    Gmem[1][2] = g2z

    var BmemShell = _zero_matrix(3, 3)
    var BbendShell = _zero_matrix(3, 3)

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
    var BJ = _zero_matrix(8, 6)
    var BJtranD = _zero_matrix(6, 8)
    var BdrillJ: List[Float64] = []
    BdrillJ.resize(6, 0.0)
    var stiffJK = _zero_matrix(6, 6)

    for gp in range(4):
        var ss = sg[gp]
        var tt = tg[gp]

        for i in range(4):
            shp[2][i] = (0.5 + s[i] * ss) * (0.5 + t[i] * tt)
            shp[0][i] = s[i] * (0.5 + t[i] * tt)
            shp[1][i] = t[i] * (0.5 + s[i] * ss)

        for i in range(2):
            for j in range(2):
                xs[i][j] = 0.0
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

        for i in range(2):
            for j in range(4):
                Ms[i][j] = 0.0
        Ms[1][0] = 1.0 - ss
        Ms[0][1] = 1.0 - tt
        Ms[1][2] = 1.0 + ss
        Ms[0][3] = 1.0 + tt

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

        for j in range(12):
            Bs[0][j] = Rot[0][0] * Bsv[0][j] + Rot[0][1] * Bsv[1][j]
            Bs[1][j] = Rot[1][0] * Bsv[0][j] + Rot[1][1] * Bsv[1][j]

        for j in range(4):
            for p in range(3):
                for q in range(2):
                    Bmem[p][q] = 0.0
            Bmem[0][0] = shp[0][j]
            Bmem[1][1] = shp[1][j]
            Bmem[2][0] = shp[1][j]
            Bmem[2][1] = shp[0][j]

            for p in range(3):
                for q in range(2):
                    Bb[p][q] = 0.0
            Bb[0][1] = -shp[0][j]
            Bb[1][0] = shp[1][j]
            Bb[2][0] = shp[0][j]
            Bb[2][1] = -shp[1][j]

            for p in range(3):
                Bshear[0][p] = Bs[0][j * 3 + p]
                Bshear[1][p] = Bs[1][j * 3 + p]
            for p in range(3):
                for q in range(3):
                    var sum_m = 0.0
                    var sum_b = 0.0
                    for r in range(2):
                        sum_m += Bmem[p][r] * Gmem[r][q]
                        sum_b += Bb[p][r] * Gmem[r][q]
                    BmemShell[p][q] = sum_m
                    BbendShell[p][q] = sum_b

            for p in range(2):
                for q in range(6):
                    var sum_s = 0.0
                    for r in range(3):
                        sum_s += Bshear[p][r] * Gshear[r][q]
                    BshearShell[p][q] = sum_s

            for p in range(8):
                for q in range(6):
                    saveB[j][p][q] = 0.0
            for p in range(3):
                for q in range(3):
                    saveB[j][p][q] = BmemShell[p][q]
            for p in range(3):
                for q in range(3):
                    saveB[j][p + 3][q + 3] = BbendShell[p][q]
            for p in range(2):
                for q in range(6):
                    saveB[j][p + 6][q] = BshearShell[p][q]

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
            for p in range(8):
                for q in range(6):
                    BJ[p][q] = saveB[j][p][q]
            for p in range(3, 6):
                for q in range(3, 6):
                    BJ[p][q] *= -1.0
            for p in range(6):
                for q in range(8):
                    BJtranD[p][q] = 0.0
            for p in range(6):
                for q in range(8):
                    var sum = 0.0
                    for r in range(8):
                        sum += BJ[r][p] * dd[r][q]
                    BJtranD[p][q] = sum * dvol

            for p in range(6):
                BdrillJ[p] = Bdrill_all[j][p]
            for p in range(6):
                BdrillJ[p] *= (Ktt * dvol)

            for k in range(4):
                for p in range(6):
                    for q in range(6):
                        stiffJK[p][q] = 0.0
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


fn shell4_layered_stiffness_and_residual(
    sec_def: LayeredShellSectionDef,
    uniaxial_defs: List[UniMaterialDef],
    mut uniaxial_states: List[UniMaterialState],
    section_state_offset: Int,
    section_state_count: Int,
    instance_offset: Int,
    x: List[Float64],
    y: List[Float64],
    z: List[Float64],
    u_elem: List[Float64],
    mut k_out: List[List[Float64]],
    mut f_out: List[Float64],
) raises:
    if len(x) != 4 or len(y) != 4 or len(z) != 4:
        abort("shell4 requires 4 nodes")
    if len(u_elem) != 24:
        abort("shell4 layered response requires 24 element dofs")

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

    var xl: List[List[Float64]] = []
    for _ in range(2):
        var row: List[Float64] = []
        row.resize(4, 0.0)
        xl.append(row^)
    for i in range(4):
        xl[0][i] = _dot(x[i], y[i], z[i], g1x, g1y, g1z)
        xl[1][i] = _dot(x[i], y[i], z[i], g2x, g2y, g2z)

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
    var s = [-0.5, 0.5, 0.5, -0.5]
    var t = [-0.5, -0.5, 0.5, 0.5]

    var K_local = _zero_matrix(24, 24)
    var F_int: List[Float64] = []
    F_int.resize(24, 0.0)

    var shp = _zero_matrix(3, 4)
    var xs = _zero_matrix(2, 2)
    var Ms = _zero_matrix(2, 4)
    var Bsv = _zero_matrix(2, 12)
    var Bs = _zero_matrix(2, 12)
    var saveB: List[List[List[Float64]]] = []
    for _ in range(4):
        var nodeB = _zero_matrix(8, 6)
        saveB.append(nodeB^)
    var Bdrill_all: List[List[Float64]] = []
    for _ in range(4):
        var row: List[Float64] = []
        row.resize(6, 0.0)
        Bdrill_all.append(row^)
    var Bmem = _zero_matrix(3, 2)
    var Bb = _zero_matrix(3, 2)
    var Bshear = _zero_matrix(2, 3)
    var Gmem = _zero_matrix(2, 3)
    Gmem[0][0] = g1x
    Gmem[0][1] = g1y
    Gmem[0][2] = g1z
    Gmem[1][0] = g2x
    Gmem[1][1] = g2y
    Gmem[1][2] = g2z
    var BmemShell = _zero_matrix(3, 3)
    var BbendShell = _zero_matrix(3, 3)
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
    var BJ = _zero_matrix(8, 6)
    var BJtran = _zero_matrix(6, 8)
    var BJtranD = _zero_matrix(6, 8)
    var BK = _zero_matrix(8, 6)
    var BdrillJ: List[Float64] = []
    BdrillJ.resize(6, 0.0)
    var BdrillK: List[Float64] = []
    BdrillK.resize(6, 0.0)
    var stiffJK = _zero_matrix(6, 6)
    var strain_gp: List[Float64] = []
    strain_gp.resize(8, 0.0)
    var stress_gp: List[Float64] = []
    stress_gp.resize(8, 0.0)
    var dd_gp = _zero_matrix(8, 8)

    var ktt = 0.0

    for gp in range(4):
        var ss = sg[gp]
        var tt = tg[gp]

        for i in range(4):
            shp[2][i] = (0.5 + s[i] * ss) * (0.5 + t[i] * tt)
            shp[0][i] = s[i] * (0.5 + t[i] * tt)
            shp[1][i] = t[i] * (0.5 + s[i] * ss)

        for i in range(2):
            for j in range(2):
                xs[i][j] = 0.0
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

        for i in range(2):
            for j in range(4):
                Ms[i][j] = 0.0
        Ms[1][0] = 1.0 - ss
        Ms[0][1] = 1.0 - tt
        Ms[1][2] = 1.0 + ss
        Ms[0][3] = 1.0 + tt

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
        for j in range(12):
            Bs[0][j] = Rot[0][0] * Bsv[0][j] + Rot[0][1] * Bsv[1][j]
            Bs[1][j] = Rot[1][0] * Bsv[0][j] + Rot[1][1] * Bsv[1][j]

        for p in range(8):
            strain_gp[p] = 0.0
        var eps_drill = 0.0

        for j in range(4):
            for p in range(3):
                for q in range(2):
                    Bmem[p][q] = 0.0
            Bmem[0][0] = shp[0][j]
            Bmem[1][1] = shp[1][j]
            Bmem[2][0] = shp[1][j]
            Bmem[2][1] = shp[0][j]

            for p in range(3):
                for q in range(2):
                    Bb[p][q] = 0.0
            Bb[0][1] = -shp[0][j]
            Bb[1][0] = shp[1][j]
            Bb[2][0] = shp[0][j]
            Bb[2][1] = -shp[1][j]

            for p in range(3):
                Bshear[0][p] = Bs[0][j * 3 + p]
                Bshear[1][p] = Bs[1][j * 3 + p]
            for p in range(3):
                for q in range(3):
                    var sum_m = 0.0
                    var sum_b = 0.0
                    for r in range(2):
                        sum_m += Bmem[p][r] * Gmem[r][q]
                        sum_b += Bb[p][r] * Gmem[r][q]
                    BmemShell[p][q] = sum_m
                    BbendShell[p][q] = sum_b

            for p in range(2):
                for q in range(6):
                    var sum_s = 0.0
                    for r in range(3):
                        sum_s += Bshear[p][r] * Gshear[r][q]
                    BshearShell[p][q] = sum_s

            for p in range(8):
                for q in range(6):
                    saveB[j][p][q] = 0.0
            for p in range(3):
                for q in range(3):
                    saveB[j][p][q] = BmemShell[p][q]
            for p in range(3):
                for q in range(3):
                    saveB[j][p + 3][q + 3] = BbendShell[p][q]
            for p in range(2):
                for q in range(6):
                    saveB[j][p + 6][q] = BshearShell[p][q]

            var B1 = -0.5 * shp[1][j]
            var B2 = 0.5 * shp[0][j]
            var B6 = -shp[2][j]
            Bdrill_all[j][0] = B1 * g1x + B2 * g2x
            Bdrill_all[j][1] = B1 * g1y + B2 * g2y
            Bdrill_all[j][2] = B1 * g1z + B2 * g2z
            Bdrill_all[j][3] = B6 * g3x
            Bdrill_all[j][4] = B6 * g3y
            Bdrill_all[j][5] = B6 * g3z

            for p in range(8):
                for q in range(6):
                    BJ[p][q] = saveB[j][p][q]
            for p in range(6):
                var sum_u = 0.0
                for q in range(8):
                    sum_u += BJ[q][p] * u_elem[j * 6 + p]
                _ = sum_u
            for p in range(8):
                var sum = 0.0
                for q in range(6):
                    sum += saveB[j][p][q] * u_elem[j * 6 + q]
                strain_gp[p] += sum
            for p in range(6):
                eps_drill += Bdrill_all[j][p] * u_elem[j * 6 + p]
        layered_shell_set_trial_from_offset(
            sec_def,
            uniaxial_defs,
            uniaxial_states,
            section_state_offset,
            section_state_count,
            instance_offset + gp,
            strain_gp,
            stress_gp,
            dd_gp,
        )
        if gp == 0:
            ktt = dd_gp[2][2]
            if ktt <= 0.0:
                ktt = 1.0
        var tau_drill = ktt * eps_drill * dvol
        for p in range(8):
            stress_gp[p] *= dvol
        for p in range(8):
            for q in range(8):
                dd_gp[p][q] *= dvol

        for j in range(4):
            for p in range(8):
                for q in range(6):
                    BJ[p][q] = saveB[j][p][q]
            for p in range(3, 6):
                for q in range(3, 6):
                    BJ[p][q] *= -1.0
            for p in range(6):
                for q in range(8):
                    BJtran[p][q] = BJ[q][p]
            for p in range(6):
                var sum_force = 0.0
                for q in range(8):
                    sum_force += BJtran[p][q] * stress_gp[q]
                F_int[j * 6 + p] += sum_force + Bdrill_all[j][p] * tau_drill
            for p in range(6):
                for q in range(8):
                    var sum = 0.0
                    for r in range(8):
                        sum += BJtran[p][r] * dd_gp[r][q]
                    BJtranD[p][q] = sum
                BdrillJ[p] = Bdrill_all[j][p] * ktt * dvol
            for k in range(4):
                for p in range(8):
                    for q in range(6):
                        BK[p][q] = saveB[k][p][q]
                for p in range(6):
                    BdrillK[p] = Bdrill_all[k][p]
                for p in range(6):
                    for q in range(6):
                        var sum = 0.0
                        for r in range(8):
                            sum += BJtranD[p][r] * BK[r][q]
                        stiffJK[p][q] = sum + BdrillJ[p] * BdrillK[q]
                var i0 = j * 6
                var j0 = k * 6
                for p in range(6):
                    for q in range(6):
                        K_local[i0 + p][j0 + q] += stiffJK[p][q]
    k_out = K_local^
    f_out = F_int^
