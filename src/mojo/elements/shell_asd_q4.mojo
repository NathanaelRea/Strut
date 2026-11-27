"""OpenSees ASDShellQ4 reference AGQI membrane and MITC shear interpolation."""

from collections import InlineArray, List
from math import atan, cos, sin, sqrt
from os import abort

from elements.shell4 import SHELL4_GEOMETRY_SIZE, SHELL4_POINT_SIZE
from elements.shell_asd_transform import ASDFrame, ASDPoints
from elements.utils import _zero_matrix
from linalg import lu_factorize_into, lu_solve_into


alias _Four = InlineArray[Float64, 4]
alias _Enhanced = InlineArray[Float64, 12]
alias _Row = InlineArray[Float64, 24]


fn _quad_params(points: ASDPoints, axis: Int) -> (Float64, Float64, Float64):
    var p0 = points[0][axis]
    var p1 = points[1][axis]
    var p2 = points[2][axis]
    var p3 = points[3][axis]
    return -p0 + p1 + p2 - p3, p0 - p1 + p2 - p3, -p0 - p1 + p2 + p3


fn _triangle_area(points: ASDPoints, i: Int, j: Int, k: Int) -> Float64:
    var a = points[j] - points[i]
    var b = points[k] - points[i]
    return 0.5 * (a[0] * b[1] - a[1] * b[0])


struct _AGQI(Copyable, Movable):
    var area: Float64
    var bx: _Four
    var by: _Four
    var g: _Four

    fn __init__(out self, points: ASDPoints):
        var a = _triangle_area(points, 0, 1, 3)
        self.area = a + _triangle_area(points, 1, 2, 3)
        self.g = _Four(fill=0.0)
        self.g[0] = a / self.area
        self.g[1] = _triangle_area(points, 0, 1, 2) / self.area
        self.g[2] = 1.0 - self.g[0]
        self.g[3] = 1.0 - self.g[1]
        self.bx = _Four(fill=0.0)
        self.by = _Four(fill=0.0)
        for i in range(4):
            var edge = points[(i + 1) % 4] - points[(i + 2) % 4]
            self.bx[i] = edge[1]
            self.by[i] = -edge[0]

    fn __copyinit__(out self, existing: Self):
        self.area = existing.area
        self.bx = existing.bx.copy()
        self.by = existing.by.copy()
        self.g = existing.g.copy()

    fn evaluate(
        self,
        s: Float64,
        t: Float64,
        mut dx: _Four,
        mut dy: _Four,
        mut enhanced: _Enhanced,
    ):
        var g = self.g.copy()
        var area_coords = [
            0.25 * (1.0 - s) * (g[1] * (1.0 - t) + g[2] * (1.0 + t)),
            0.25 * (1.0 - t) * (g[3] * (1.0 - s) + g[2] * (1.0 + s)),
            0.25 * (1.0 + s) * (g[0] * (1.0 - t) + g[3] * (1.0 + t)),
            0.25 * (1.0 + t) * (g[0] * (1.0 - s) + g[1] * (1.0 + s)),
        ]
        var sx = 0.0
        var sy = 0.0
        for i in range(4):
            var j = (i + 1) % 4
            var k = (i + 2) % 4
            var m = (i + 3) % 4
            var factor = 3.0 * (area_coords[j] - area_coords[m]) + g[j] - g[k]
            if i % 2 != 0:
                factor = -factor
            sx += self.bx[i] * factor
            sy += self.by[i] * factor
        var inv_area = 0.5 / self.area
        var enhance_scale = inv_area / (1.0 + g[0] * g[2] + g[1] * g[3])
        for i in range(4):
            var j = (i + 1) % 4
            var k = (i + 2) % 4
            var scale = enhance_scale * g[k]
            if i % 2 != 0:
                scale = -scale
            dx[i] = inv_area * (self.bx[i] + self.bx[j]) + scale * sx
            dy[i] = inv_area * (self.by[i] + self.by[j]) + scale * sy
        for i in range(2):
            var k = i + 2
            var x = inv_area * (self.bx[i] * area_coords[k] + self.bx[k] * area_coords[i])
            var y = inv_area * (self.by[i] * area_coords[k] + self.by[k] * area_coords[i])
            enhanced[2 * i] = x
            enhanced[5 + 2 * i] = y
            enhanced[8 + 2 * i] = y
            enhanced[9 + 2 * i] = x


fn _point_geometry(
    points: ASDPoints,
    agqi: _AGQI,
    use_eas: Bool,
    s: Float64,
    t: Float64,
    q: UnsafePointer[Float64],
    mut enhanced: _Enhanced,
):
    var ax, bx, cx = _quad_params(points, 0)
    var ay, by, cy = _quad_params(points, 1)
    var xs = ax + bx * t
    var xt = cx + bx * s
    var ys = ay + by * t
    var yt = cy + by * s
    var det = xs * yt - xt * ys
    if det <= 0.0:
        abort("ASDShellQ4 singular jacobian")
    q[216] = det / 16.0
    var sign_s = [-1.0, 1.0, 1.0, -1.0]
    var sign_t = [-1.0, -1.0, 1.0, 1.0]
    var dx = _Four(fill=0.0)
    var dy = _Four(fill=0.0)
    var shape = _Four(fill=0.0)
    for n in range(4):
        var sn = sign_s[n]
        var tn = sign_t[n]
        var si = 1.0 + sn * s
        var ti = 1.0 + tn * t
        shape[n] = 0.25 * si * ti
        dx[n] = (sn * ti * yt - tn * si * ys) / det
        dy[n] = (tn * si * xs - sn * ti * xt) / det
    if use_eas:
        agqi.evaluate(s, t, dx, dy, enhanced)
    for n in range(4):
        var p = 6 * n
        q[p] = dx[n]
        q[24 + p + 1] = dy[n]
        q[48 + p] = dy[n]
        q[48 + p + 1] = dx[n]
        q[72 + p + 4] = -dx[n]
        q[96 + p + 3] = dy[n]
        q[120 + p + 3] = dx[n]
        q[120 + p + 4] = -dy[n]
        q[192 + p] = -0.5 * dy[n]
        q[192 + p + 1] = 0.5 * dx[n]
        q[192 + p + 5] = -shape[n]
    var shear_s = _Row(fill=0.0)
    var shear_t = _Row(fill=0.0)
    for edge in range(4):
        var i = edge
        var j = (edge + 1) % 4
        if edge >= 2:
            swap(i, j)
        var delta = points[j] - points[i]
        var scale: Float64
        if edge % 2 == 0:
            scale = (1.0 + sign_t[edge] * t) * 2.0 * sqrt(xt * xt + yt * yt) / det
            _accumulate_shear(shear_s, i, j, delta[0], delta[1], scale)
        else:
            scale = (1.0 - sign_t[edge] * s) * 2.0 * sqrt(xs * xs + ys * ys) / det
            _accumulate_shear(shear_t, i, j, delta[0], delta[1], scale)
    var alpha = atan(ay / ax)
    var beta = 1.5707963267948966 - atan(cx / cy)
    for i in range(24):
        q[144 + i] = sin(beta) * shear_s[i] - sin(alpha) * shear_t[i]
        q[168 + i] = -cos(beta) * shear_s[i] + cos(alpha) * shear_t[i]


fn _accumulate_shear(mut row: _Row, i: Int, j: Int, dx: Float64, dy: Float64, scale: Float64):
    row[6 * i + 2] -= 0.5 * scale
    row[6 * j + 2] += 0.5 * scale
    for n in [i, j]:
        row[6 * n + 3] -= 0.25 * dy * scale
        row[6 * n + 4] += 0.25 * dx * scale


fn asd_shell_elastic_geometry(
    frame: ASDFrame,
    tangent: UnsafePointer[Float64],
    use_eas: Bool,
    drilling_stab: Float64,
) -> InlineArray[Float64, SHELL4_GEOMETRY_SIZE]:
    var geometry = InlineArray[Float64, SHELL4_GEOMETRY_SIZE](fill=0.0)
    var center = InlineArray[Float64, SHELL4_POINT_SIZE](fill=0.0)
    var enhanced = InlineArray[_Enhanced, 4](fill=_Enhanced(fill=0.0))
    var dummy = _Enhanced(fill=0.0)
    var p = UnsafePointer(to=geometry[0])
    var agqi = _AGQI(frame.points)
    _point_geometry(frame.points, agqi, use_eas, 0.0, 0.0, UnsafePointer(to=center[0]), dummy)
    var g = 0.577350269189626
    var sg = [-g, g, g, -g]
    var tg = [-g, -g, g, g]
    var area = 0.0
    var mean = _Enhanced(fill=0.0)
    for gp in range(4):
        var q = p + gp * SHELL4_POINT_SIZE
        _point_geometry(frame.points, agqi, use_eas, sg[gp], tg[gp], q, enhanced[gp])
        for d in range(24):
            q[192 + d] = (
                sqrt(drilling_stab) * q[192 + d] + sqrt(1.0 - drilling_stab) * center[192 + d]
            )
        area += q[216]
        for i in range(12):
            mean[i] += enhanced[gp][i] * q[216]
    if use_eas:
        for gp in range(4):
            for i in range(12):
                enhanced[gp][i] -= mean[i] / area
        _condense_enhanced(tangent, enhanced, p)
    return geometry^


fn _condense_enhanced(
    tangent: UnsafePointer[Float64],
    enhanced: InlineArray[_Enhanced, 4],
    geometry: UnsafePointer[Float64],
):
    # The enhanced strain operator is nonzero only in the membrane rows.
    # For constant elastic sections Q = -KQQ^-1 KQU U at equilibrium.
    var kqq = _zero_matrix(4, 4)
    var kqu = _zero_matrix(4, 24)
    for gp in range(4):
        var p = geometry + gp * SHELL4_POINT_SIZE
        for i in range(4):
            for row in range(3):
                var left = enhanced[gp][4 * row + i] * p[216]
                for k in range(3):
                    var weighted = left * tangent[8 * row + k]
                    for j in range(4):
                        kqq[i][j] += weighted * enhanced[gp][4 * k + j]
                for k in range(8):
                    var weighted = left * tangent[8 * row + k]
                    for j in range(24):
                        kqu[i][j] += weighted * p[24 * k + j]
    var pivots: List[Int] = []
    lu_factorize_into(kqq, pivots)
    var work: List[Float64] = []
    var result: List[Float64] = []
    var rhs = [0.0, 0.0, 0.0, 0.0]
    for j in range(24):
        for i in range(4):
            rhs[i] = kqu[i][j]
        lu_solve_into(kqq, pivots, rhs, work, result)
        for gp in range(4):
            var p = geometry + gp * SHELL4_POINT_SIZE
            for row in range(3):
                for i in range(4):
                    p[24 * row + j] -= enhanced[gp][4 * row + i] * result[i]
