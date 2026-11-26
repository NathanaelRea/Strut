from collections import InlineArray, List
from math import atan2, cos, sin, sqrt
from materials import UniMaterialDef, UniMaterialState
from os import abort
from sys import simd_width_of

from elements.utils import _cross, _dot, _normalize, _zero_matrix
from sections import LayeredShellSectionDef, layered_shell_set_trial_from_offset


alias SHELL4_DOFS = 24
alias SHELL4_MATRIX_SIZE = SHELL4_DOFS * SHELL4_DOFS
alias SHELL4_POINT_SIZE = 8 * SHELL4_DOFS + SHELL4_DOFS + 1
alias SHELL4_GEOMETRY_SIZE = 4 * SHELL4_POINT_SIZE
# Section scratch only. The assembler owns the displacement and output buffers.
alias SHELL4_SCRATCH_SIZE = 8 + 8 + 64
alias _WIDTH = min(8, simd_width_of[DType.float64]())


fn shell4_precompute_geometry(
    x: List[Float64],
    y: List[Float64],
    z: List[Float64],
    out_geometry: UnsafePointer[Float64],
):
    """Write four Gauss blocks of row-major B, a drilling row, and dvol.

    The basis is folded into B and the drilling row, so their columns already
    correspond to the global translational and rotational DOFs of each node.
    """
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

    var shp = _zero_matrix(3, 4)
    var xs = _zero_matrix(2, 2)
    var Ms = _zero_matrix(2, 4)
    var Bsv = _zero_matrix(2, 12)
    var Bs = _zero_matrix(2, 12)

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

        var point = out_geometry + gp * SHELL4_POINT_SIZE
        point[216] = wg[gp] * xsj
        for i in range(192):
            point[i] = 0.0

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

            for p in range(3):
                for q in range(3):
                    point[p * 24 + j * 6 + q] = BmemShell[p][q]
                    point[(p + 3) * 24 + j * 6 + q + 3] = BbendShell[p][q]
            for p in range(2):
                for q in range(6):
                    point[(p + 6) * 24 + j * 6 + q] = BshearShell[p][q]

            var B1 = -0.5 * shp[1][j]
            var B2 = 0.5 * shp[0][j]
            var B6 = -shp[2][j]
            var drill = point + 192 + j * 6
            drill[0] = B1 * g1x + B2 * g2x
            drill[1] = B1 * g1y + B2 * g2y
            drill[2] = B1 * g1z + B2 * g2z
            drill[3] = B6 * g3x
            drill[4] = B6 * g3y
            drill[5] = B6 * g3z


@always_inline
fn shell4_dot_row(
    row: UnsafePointer[Float64], u: UnsafePointer[Float64]
) -> Float64:
    var sum = SIMD[DType.float64, _WIDTH](0.0)
    @parameter
    for col in range(0, 24, _WIDTH):
        sum += row.load[width=_WIDTH](col) * u.load[width=_WIDTH](col)
    return sum.reduce_add()


@always_inline
fn _clear_matrix(matrix: UnsafePointer[Float64]):
    for i in range(0, SHELL4_MATRIX_SIZE, _WIDTH):
        matrix.store(i, SIMD[DType.float64, _WIDTH](0.0))


@always_inline
fn _add_gauss_tangent(
    b: UnsafePointer[Float64],
    d: UnsafePointer[Float64],
    matrix: UnsafePointer[Float64],
    ktt: Float64,
    dvol: Float64,
):
    var drill = b + 192
    for row in range(24):
        # OpenSees negates the bending rows on the left for equilibrium.
        var left_d = SIMD[DType.float64, 8](0.0)
        @parameter
        for j in range(8):
            var left = b[j * 24 + row]
            @parameter
            if 3 <= j and j < 6:
                left = -left
            left_d += left * (d.load[width=8](j * 8) * dvol)

        var drill_left = drill[row] * ktt * dvol
        @parameter
        for col in range(0, 24, _WIDTH):
            var value = SIMD[DType.float64, _WIDTH](0.0)
            @parameter
            for j in range(8):
                value += left_d[j] * b.load[width=_WIDTH](j * 24 + col)
            value += drill_left * drill.load[width=_WIDTH](col)
            var index = row * 24 + col
            matrix.store(index, matrix.load[width=_WIDTH](index) + value)


fn shell4_mindlin_stiffness(
    E: Float64,
    nu: Float64,
    h: Float64,
    geometry: UnsafePointer[Float64],
    matrix: UnsafePointer[Float64],
):
    """Fill the constant elastic shell stiffness during model setup."""
    var d = InlineArray[Float64, 64](fill=0.0)
    var factor = E / (1.0 - nu * nu)
    var bend_factor = E * h * h * h / (12.0 * (1.0 - nu * nu))
    var G = E / (2.0 * (1.0 + nu))
    d[0] = factor * h
    d[1] = factor * nu * h
    d[8] = d[1]
    d[9] = d[0]
    d[18] = (factor * (1.0 - nu) / 2.0) * h
    d[27] = -bend_factor
    d[28] = -bend_factor * nu
    d[35] = d[28]
    d[36] = d[27]
    d[45] = -(bend_factor * (1.0 - nu) / 2.0)
    d[54] = (5.0 / 6.0) * G * h
    d[63] = d[54]
    _clear_matrix(matrix)
    for gp in range(4):
        var point = geometry + gp * SHELL4_POINT_SIZE
        _add_gauss_tangent(point, UnsafePointer(to=d[0]), matrix, G * h, point[216])


fn shell4_layered_stiffness_and_residual[tangent: Bool](
    sec_def: LayeredShellSectionDef,
    uniaxial_defs: UnsafePointer[UniMaterialDef],
    states: UnsafePointer[UniMaterialState],
    geometry: UnsafePointer[Float64],
    u: UnsafePointer[Float64],
    matrix: UnsafePointer[Float64],
    force: UnsafePointer[Float64],
    scratch: UnsafePointer[Float64],
):
    var strain = scratch
    var stress = scratch + 8
    var d = scratch + 16

    @parameter
    if tangent:
        _clear_matrix(matrix)
    for col in range(0, 24, _WIDTH):
        force.store(col, SIMD[DType.float64, _WIDTH](0.0))

    var ktt = 0.0
    for gp in range(4):
        var point = geometry + gp * SHELL4_POINT_SIZE
        var drill = point + 192
        var dvol = point[216]
        for row in range(8):
            strain[row] = shell4_dot_row(point + row * 24, u)
        var drilling_tangent = layered_shell_set_trial_from_offset[tangent](
            sec_def, uniaxial_defs, states + gp * len(sec_def.rebars),
            strain, stress, d, gp == 0,
        )
        if gp == 0:
            ktt = drilling_tangent
            if ktt <= 0.0:
                ktt = 1.0
        var tau_drill = ktt * shell4_dot_row(drill, u) * dvol
        for col in range(0, 24, _WIDTH):
            var value = SIMD[DType.float64, _WIDTH](0.0)
            @parameter
            for row in range(8):
                var stress_component = stress[row] * dvol
                @parameter
                if 3 <= row and row < 6:
                    stress_component = -stress_component
                value += stress_component * point.load[width=_WIDTH](row * 24 + col)
            value += tau_drill * drill.load[width=_WIDTH](col)
            force.store(col, force.load[width=_WIDTH](col) + value)

        @parameter
        if tangent:
            _add_gauss_tangent(point, d, matrix, ktt, dvol)
