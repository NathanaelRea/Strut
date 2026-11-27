"""Reference geometry for OpenSees ShellDKGT and ShellDKGQ."""

from collections import InlineArray, List
from math import sqrt
from os import abort

from elements.shell4 import SHELL4_DOFS, SHELL4_POINT_SIZE
from elements.utils import _cross, _dot, _normalize


alias _Vec3 = SIMD[DType.float64, 4]
alias _Four = InlineArray[Float64, 4]
alias _Eight = InlineArray[Float64, 8]


fn _write_b(
    point: UnsafePointer[Float64],
    count: Int,
    basis: InlineArray[_Vec3, 3],
    dx: _Four,
    dy: _Four,
    drill_x: _Four,
    drill_y: _Four,
    drill_xy: _Four,
    x: _Four,
    y: _Four,
    shape_x: _Eight,
    shape_y: _Eight,
    shape: _Eight,
    geometric: UnsafePointer[Float64],
):
    var b = InlineArray[Float64, 192](fill=0.0)
    for n in range(count):
        var p = 6 * n
        b[p] = dx[n]
        b[SHELL4_DOFS + p + 1] = dy[n]
        b[2 * SHELL4_DOFS + p] = dy[n]
        b[2 * SHELL4_DOFS + p + 1] = dx[n]
        b[p + 5] = drill_x[n]
        b[SHELL4_DOFS + p + 5] = drill_y[n]
        b[2 * SHELL4_DOFS + p + 5] = drill_xy[n]

        # The end slopes are interpolated with the quadratic corner shapes.
        # Each adjacent edge contributes its discrete Kirchhoff midpoint slope.
        var previous = (n + count - 1) % count
        var slopes = InlineArray[Float64, 18](fill=0.0)
        slopes[2] = shape_x[n]
        slopes[5] = shape_y[n]
        slopes[7] = -shape_x[n]
        slopes[10] = -shape_y[n]
        slopes[14] = shape[n]
        slopes[16] = -shape[n]
        for side in range(2):
            var edge = n
            var orientation = 1.0
            if side == 1:
                edge = previous
                orientation = -1.0
            var end = (edge + 1) % count
            var ex = x[end] - x[edge]
            var ey = y[end] - y[edge]
            var length_sq = ex * ex + ey * ey
            var cross_term = 0.75 * ex * ey / length_sq
            var cx = 0.25 * (ex * ex - 2.0 * ey * ey) / length_sq
            var cy = 0.25 * (ey * ey - 2.0 * ex * ex) / length_sq
            var mid = count + edge
            for axis in range(2):
                var shape = shape_x[mid]
                if axis == 1:
                    shape = shape_y[mid]
                var h_x = 3 * axis
                var h_y = h_x + 6
                slopes[h_x] += shape * orientation * 1.5 * ex / length_sq
                slopes[h_x + 1] += shape * cross_term
                slopes[h_x + 2] -= shape * cx
                slopes[h_y] += shape * orientation * 1.5 * ey / length_sq
                slopes[h_y + 1] += shape * cy
                slopes[h_y + 2] -= shape * cross_term
            if geometric:
                var m = shape[mid]
                slopes[12] += m * orientation * 1.5 * ex / length_sq
                slopes[13] += m * cross_term
                slopes[14] -= m * cx
                slopes[15] += m * orientation * 1.5 * ey / length_sq
                slopes[16] += m * cy
                slopes[17] -= m * cross_term
        for d in range(3):
            b[3 * SHELL4_DOFS + p + 2 + d] = -slopes[d]
            b[4 * SHELL4_DOFS + p + 2 + d] = -slopes[9 + d]
            b[5 * SHELL4_DOFS + p + 2 + d] = -(slopes[3 + d] + slopes[6 + d])

        if geometric:
            for row in range(2):
                for d in range(3):
                    geometric[row * SHELL4_DOFS + p + d] = -slopes[12 + 3 * row] * basis[2][d]
                    geometric[row * SHELL4_DOFS + p + 3 + d] = -(
                        slopes[13 + 3 * row] * basis[0][d] + slopes[14 + 3 * row] * basis[1][d]
                    )

        for row in range(6):
            for d in range(3):
                var translate = 0.0
                var rotate = 0.0
                for axis in range(3):
                    translate += b[row * SHELL4_DOFS + p + axis] * basis[axis][d]
                    rotate += b[row * SHELL4_DOFS + p + 3 + axis] * basis[axis][d]
                point[row * SHELL4_DOFS + p + d] = translate
                point[row * SHELL4_DOFS + p + 3 + d] = rotate


fn discrete_kirchhoff_geometry(
    x: List[Float64],
    y: List[Float64],
    z: List[Float64],
    geometry: UnsafePointer[Float64],
):
    var count = len(x)
    var xyz = InlineArray[_Vec3, 4](fill=_Vec3(0.0))
    for i in range(count):
        xyz[i] = _Vec3(x[i], y[i], z[i], 0.0)
    var basis = _dk_basis(xyz, count)
    _dk_geometry_in_basis(xyz, count, basis, geometry, UnsafePointer[Float64]())


fn discrete_kirchhoff_updated_geometry(
    reference: InlineArray[_Vec3, 4],
    committed: UnsafePointer[Float64],
    trial: UnsafePointer[Float64],
    geometry: UnsafePointer[Float64],
    geometric: UnsafePointer[Float64],
):
    var trial_positions = reference.copy()
    var committed_positions = reference.copy()
    for i in range(4):
        var p = 6 * i
        trial_positions[i] += _Vec3(trial[p], trial[p + 1], trial[p + 2], 0.0)
        committed_positions[i] += _Vec3(committed[p], committed[p + 1], committed[p + 2], 0.0)
    var basis = _dk_basis(trial_positions, 4)
    _dk_geometry_in_basis(committed_positions, 4, basis, geometry, geometric)


fn _dk_basis(xyz: InlineArray[_Vec3, 4], count: Int) -> InlineArray[_Vec3, 3]:
    var a = xyz[1] - xyz[0]
    var b = xyz[2] - xyz[0]
    if count == 4:
        a = xyz[2] + xyz[1] - xyz[3] - xyz[0]
        b = xyz[3] + xyz[2] - xyz[1] - xyz[0]
    var ax, ay, az = _normalize(a[0], a[1], a[2])
    b -= _Vec3(ax, ay, az, 0.0) * _dot(b[0], b[1], b[2], ax, ay, az)
    var bx, by, bz = _normalize(b[0], b[1], b[2])
    var cx, cy, cz = _cross(ax, ay, az, bx, by, bz)
    var basis = InlineArray[_Vec3, 3](fill=_Vec3(0.0))
    basis[0] = _Vec3(ax, ay, az, 0.0)
    basis[1] = _Vec3(bx, by, bz, 0.0)
    basis[2] = _Vec3(cx, cy, cz, 0.0)
    return basis^


fn _dk_geometry_in_basis(
    xyz: InlineArray[_Vec3, 4],
    count: Int,
    basis: InlineArray[_Vec3, 3],
    geometry: UnsafePointer[Float64],
    geometric: UnsafePointer[Float64],
):
    var xl = _Four(fill=0.0)
    var yl = _Four(fill=0.0)
    for i in range(count):
        var relative = xyz[i] - xyz[0]
        xl[i] = (relative * basis[0]).reduce_add()
        yl[i] = (relative * basis[1]).reduce_add()
    if count == 3:
        _triangle_geometry(basis, xl, yl, geometry)
    else:
        _quad_geometry(basis, xl, yl, geometry, geometric)


fn _triangle_geometry(
    basis: InlineArray[_Vec3, 3], x: _Four, y: _Four, geometry: UnsafePointer[Float64]
):
    var twice_area = x[1] * y[2] - x[2] * y[1]
    if twice_area <= 0.0:
        abort("ShellDKGT singular jacobian")
    var dx = _Four(fill=0.0)
    var dy = _Four(fill=0.0)
    for n in range(3):
        var j = (n + 1) % 3
        var k = (n + 2) % 3
        dx[n] = (y[j] - y[k]) / twice_area
        dy[n] = (x[k] - x[j]) / twice_area

    var dominant_vertex = [0, 1, 0, 2]
    for gp in range(4):
        var shape = _Four(fill=0.2)
        var volume = (25.0 / 48.0) * 0.5 * twice_area
        if gp == 0:
            shape = _Four(fill=1.0 / 3.0)
            volume = (-27.0 / 48.0) * 0.5 * twice_area
        else:
            shape[dominant_vertex[gp]] = 0.6
        var drill_x = _Four(fill=0.0)
        var drill_y = _Four(fill=0.0)
        var drill_xy = _Four(fill=0.0)
        var bend_x = _Eight(fill=0.0)
        var bend_y = _Eight(fill=0.0)
        for n in range(3):
            var j = (n + 1) % 3
            var k = (n + 2) % 3
            var drill_u = 0.5 * twice_area * (dx[k] * shape[j] - dx[j] * shape[k])
            var drill_v = 0.5 * twice_area * (dy[k] * shape[j] - dy[j] * shape[k])
            drill_x[n] = dx[n] * drill_u
            drill_y[n] = dy[n] * drill_v
            drill_xy[n] = dy[n] * drill_u + dx[n] * drill_v
            bend_x[n] = (4.0 * shape[n] - 1.0) * dx[n]
            bend_y[n] = (4.0 * shape[n] - 1.0) * dy[n]
            bend_x[3 + n] = 4.0 * (dx[n] * shape[j] + shape[n] * dx[j])
            bend_y[3 + n] = 4.0 * (dy[n] * shape[j] + shape[n] * dy[j])
        var point = geometry + gp * SHELL4_POINT_SIZE
        _write_b(
            point,
            3,
            basis,
            dx,
            dy,
            drill_x,
            drill_y,
            drill_xy,
            x,
            y,
            bend_x,
            bend_y,
            _Eight(fill=0.0),
            UnsafePointer[Float64](),
        )
        point[216] = volume


fn _quad_geometry(
    basis: InlineArray[_Vec3, 3],
    x: _Four,
    y: _Four,
    geometry: UnsafePointer[Float64],
    geometric: UnsafePointer[Float64],
):
    var a1 = 0.25 * (-x[0] + x[1] + x[2] - x[3])
    var a2 = 0.25 * (-x[0] - x[1] + x[2] + x[3])
    var a3 = 0.25 * (x[0] - x[1] + x[2] - x[3])
    var b1 = 0.25 * (-y[0] + y[1] + y[2] - y[3])
    var b2 = 0.25 * (-y[0] - y[1] + y[2] + y[3])
    var b3 = 0.25 * (y[0] - y[1] + y[2] - y[3])
    var sign_s = [-1.0, 1.0, 1.0, -1.0]
    var sign_t = [-1.0, -1.0, 1.0, 1.0]
    var g = 1.0 / sqrt(3.0)
    for gp in range(4):
        var s = g * sign_s[gp]
        var t = g * sign_t[gp]
        var xs = a1 + a3 * t
        var xt = a2 + a3 * s
        var ys = b1 + b3 * t
        var yt = b2 + b3 * s
        var det = xs * yt - xt * ys
        if det <= 0.0:
            abort("ShellDKGQ singular jacobian")
        var dx = _Four(fill=0.0)
        var dy = _Four(fill=0.0)
        var drill_x = _Four(fill=0.0)
        var drill_y = _Four(fill=0.0)
        var drill_xy = _Four(fill=0.0)
        var shape = _Eight(fill=0.0)
        var bend_s = _Eight(fill=0.0)
        var bend_t = _Eight(fill=0.0)
        for n in range(4):
            var sn = sign_s[n]
            var tn = sign_t[n]
            var si = 1.0 + sn * s
            var ti = 1.0 + tn * t
            var ds = 0.25 * sn * ti
            var dt = 0.25 * tn * si
            dx[n] = (ds * yt - dt * ys) / det
            dy[n] = (dt * xs - ds * xt) / det

            var ds1 = -0.25 * sn * s * ti
            var ds2 = 0.125 * sn * tn * (1.0 - t * t)
            var dt1 = 0.125 * sn * tn * (1.0 - s * s)
            var dt2 = -0.25 * tn * t * si
            var us = ds1 * (b1 + b3 * tn) + ds2 * (b2 + b3 * sn)
            var ut = dt1 * (b1 + b3 * tn) + dt2 * (b2 + b3 * sn)
            var vs = -(ds1 * (a1 + a3 * tn) + ds2 * (a2 + a3 * sn))
            var vt = -(dt1 * (a1 + a3 * tn) + dt2 * (a2 + a3 * sn))
            drill_x[n] = (us * yt - ut * ys) / det
            drill_y[n] = (vt * xs - vs * xt) / det
            drill_xy[n] = (ut * xs - us * xt + vs * yt - vt * ys) / det

            shape[n] = 0.25 * si * ti * (sn * s + tn * t - 1.0)
            bend_s[n] = 0.25 * ti * (2.0 * s + sn * tn * t)
            bend_t[n] = 0.25 * si * (2.0 * t + sn * tn * s)
            if n % 2 == 0:
                shape[4 + n] = 0.5 * ti * (1.0 - s * s)
                bend_s[4 + n] = -s * ti
                bend_t[4 + n] = 0.5 * tn * (1.0 - s * s)
            else:
                shape[4 + n] = 0.5 * si * (1.0 - t * t)
                bend_s[4 + n] = 0.5 * sn * (1.0 - t * t)
                bend_t[4 + n] = -t * si
        var bend_x = _Eight(fill=0.0)
        var bend_y = _Eight(fill=0.0)
        for i in range(8):
            bend_x[i] = (bend_s[i] * yt - bend_t[i] * ys) / det
            bend_y[i] = (bend_t[i] * xs - bend_s[i] * xt) / det
        var point = geometry + gp * SHELL4_POINT_SIZE
        var gradient = geometric
        if geometric:
            gradient += gp * 2 * SHELL4_DOFS
        _write_b(
            point,
            4,
            basis,
            dx,
            dy,
            drill_x,
            drill_y,
            drill_xy,
            x,
            y,
            bend_x,
            bend_y,
            shape,
            gradient,
        )
        point[216] = det
