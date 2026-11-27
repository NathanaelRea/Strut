"""Linear and polar-aligned corotational transformations for ASDShellQ4."""

from collections import InlineArray
from math import atan2, cos, sin, sqrt, tan
from os import abort

from elements.shell4 import SHELL4_DOFS, SHELL4_MATRIX_SIZE, shell4_dot_row


alias ASDPoint = SIMD[DType.float64, 4]
alias ASDPoints = InlineArray[ASDPoint, 4]
alias _Axes = InlineArray[ASDPoint, 3]
alias _Vector = InlineArray[Float64, SHELL4_DOFS]
alias _Matrix = InlineArray[Float64, SHELL4_MATRIX_SIZE]


fn _cross(a: ASDPoint, b: ASDPoint) -> ASDPoint:
    return ASDPoint(
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
        0.0,
    )


fn _dot(a: ASDPoint, b: ASDPoint) -> Float64:
    return (a * b).reduce_add()


fn _unit(a: ASDPoint) -> ASDPoint:
    var length = sqrt(_dot(a, a))
    if length == 0.0:
        abort("ASDShellQ4 degenerate reference or trial geometry")
    return a / length


fn _rotate(axes: _Axes, a: ASDPoint) -> ASDPoint:
    return ASDPoint(_dot(axes[0], a), _dot(axes[1], a), _dot(axes[2], a), 0.0)


struct ASDFrame(Copyable, Movable):
    var axes: _Axes
    var points: ASDPoints

    fn __init__(out self, points: ASDPoints, alpha: Float64 = 0.0):
        var center = 0.25 * (points[0] + points[1] + points[2] + points[3])
        var normal = _unit(_cross(points[2] - points[0], points[3] - points[1]))
        var side = points[1] - points[0]
        var x = _unit(side - _dot(side, normal) * normal)
        var y = _cross(normal, x)
        self.axes = _Axes(fill=ASDPoint(0.0))
        self.axes[0] = cos(alpha) * x + sin(alpha) * y
        self.axes[1] = cos(alpha) * y - sin(alpha) * x
        self.axes[2] = normal
        self.points = ASDPoints(fill=ASDPoint(0.0))
        for i in range(4):
            self.points[i] = _rotate(self.axes, points[i] - center)

    fn __copyinit__(out self, existing: Self):
        self.axes = existing.axes.copy()
        self.points = existing.points.copy()


fn _frame_rotation(ref_frame: ASDFrame, trial_frame: ASDFrame) -> Float64:
    var a = ref_frame.points[2] + ref_frame.points[1] - ref_frame.points[0] - ref_frame.points[3]
    var b = ref_frame.points[2] + ref_frame.points[3] - ref_frame.points[0] - ref_frame.points[1]
    var c = (
        trial_frame.points[2]
        + trial_frame.points[1]
        - trial_frame.points[0]
        - trial_frame.points[3]
    )
    var d = (
        trial_frame.points[2]
        + trial_frame.points[3]
        - trial_frame.points[0]
        - trial_frame.points[1]
    )
    # F is the center displacement map J_trial J_reference^-1. Its planar
    # polar rotation makes the local deformation gradient symmetric.
    var det = a[0] * b[1] - a[1] * b[0]
    var f11 = (c[0] * b[1] - d[0] * a[1]) / det
    var f12 = (d[0] * a[0] - c[0] * b[0]) / det
    var f21 = (c[1] * b[1] - d[1] * a[1]) / det
    var f22 = (d[1] * a[0] - c[1] * b[0]) / det
    return atan2(f21 - f12, f11 + f22)


# Quaternions store the vector part in lanes 0–2 and the scalar in lane 3.
alias _Quaternion = SIMD[DType.float64, 4]


fn _rotation_quaternion(v: ASDPoint) -> _Quaternion:
    var magnitude = sqrt(_dot(v, v))
    if magnitude == 0.0:
        return _Quaternion(0.0, 0.0, 0.0, 1.0)
    var q = (sin(0.5 * magnitude) / magnitude) * v
    q[3] = cos(0.5 * magnitude)
    return q


fn _quaternion_product(a: _Quaternion, b: _Quaternion) -> _Quaternion:
    var q = a[3] * b + b[3] * a + _cross(a, b)
    q[3] = 2.0 * a[3] * b[3] - _dot(a, b)
    return q


fn _frame_quaternion(axes: _Axes) -> _Quaternion:
    var tr = axes[0][0] + axes[1][1] + axes[2][2]
    if tr > max(axes[0][0], max(axes[1][1], axes[2][2])):
        var scale = 2.0 * sqrt(1.0 + tr)
        return _Quaternion(
            (axes[2][1] - axes[1][2]) / scale,
            (axes[0][2] - axes[2][0]) / scale,
            (axes[1][0] - axes[0][1]) / scale,
            0.25 * scale,
        )
    var i = 0
    if axes[1][1] >= axes[i][i]:
        i = 1
    if axes[2][2] >= axes[i][i]:
        i = 2
    var j = (i + 1) % 3
    var k = (i + 2) % 3
    var scale = 2.0 * sqrt(1.0 + axes[i][i] - axes[j][j] - axes[k][k])
    var q = _Quaternion(0.0)
    q[i] = 0.25 * scale
    q[j] = (axes[i][j] + axes[j][i]) / scale
    q[k] = (axes[i][k] + axes[k][i]) / scale
    q[3] = (axes[k][j] - axes[j][k]) / scale
    return q


fn _rotation_vector(q: _Quaternion) -> ASDPoint:
    var v = ASDPoint(q[0], q[1], q[2], 0.0)
    var length = sqrt(_dot(v, v))
    if length == 0.0:
        return v
    var scale = 2.0 * atan2(length, abs(q[3])) / length
    if q[3] < 0.0:
        scale = -scale
    return scale * v


fn _apply_axial_jacobian(v: ASDPoint, direction: ASDPoint) -> ASDPoint:
    var angle_sq = _dot(v, v)
    var eta = 1.0 / 12.0 + angle_sq * (
        1.0 / 720.0 + angle_sq * (1.0 / 30240.0 + angle_sq / 1209600.0)
    )
    if angle_sq >= 0.0025:
        var angle = sqrt(angle_sq)
        eta = (1.0 - 0.5 * angle * tan(1.5707963267948966 - 0.5 * angle)) / angle_sq
    var spin = _cross(v, direction)
    return direction - 0.5 * spin + eta * _cross(v, spin)


fn _project_translations(mut values: _Vector):
    for d in range(3):
        var mean = 0.25 * (values[d] + values[6 + d] + values[12 + d] + values[18 + d])
        for n in range(4):
            values[6 * n + d] -= mean


fn _load_point(values: UnsafePointer[Float64], offset: Int) -> ASDPoint:
    return ASDPoint(values[offset], values[offset + 1], values[offset + 2], 0.0)


fn _write_point(values: UnsafePointer[Float64], offset: Int, v: ASDPoint):
    for d in range(3):
        values[offset + d] = v[d]


fn asd_linear_transformation(frame: ASDFrame, out_transform: UnsafePointer[Float64]):
    # The linear transformation includes the warpage correction W * T.
    for n in range(4):
        var p = 6 * n
        var warp = frame.points[0][2] if n % 2 == 0 else -frame.points[0][2]
        for i in range(3):
            for j in range(3):
                out_transform[(p + i) * 24 + p + j] = frame.axes[i][j]
                out_transform[(p + i + 3) * 24 + p + j + 3] = frame.axes[i][j]
        for j in range(3):
            out_transform[p * 24 + p + 3 + j] = -warp * frame.axes[1][j]
            out_transform[(p + 1) * 24 + p + 3 + j] = warp * frame.axes[0][j]


fn _transform_tangent(
    local: UnsafePointer[Float64],
    left: UnsafePointer[Float64],
    right: UnsafePointer[Float64],
    matrix: UnsafePointer[Float64],
):
    var product = _Matrix(fill=0.0)
    for i in range(24):
        for k in range(24):
            var value = local[i * 24 + k]
            if value != 0.0:
                for j in range(24):
                    product[i * 24 + j] += value * right[k * 24 + j]
    for i in range(24):
        for j in range(24):
            var value = 0.0
            for k in range(24):
                value += left[k * 24 + i] * product[k * 24 + j]
            matrix[i * 24 + j] = value


fn asd_linear_stiffness(
    local: UnsafePointer[Float64], frame: ASDFrame, matrix: UnsafePointer[Float64]
):
    var transform = _Matrix(fill=0.0)
    var t = UnsafePointer(to=transform[0])
    asd_linear_transformation(frame, t)
    _transform_tangent(local, t, t, matrix)


struct ASDCorotational(Copyable, Movable):
    var points: ASDPoints
    var reference: ASDFrame
    var inverse_initial_rotation: _Quaternion
    var trial_q: InlineArray[_Quaternion, 4]
    var committed_q: InlineArray[_Quaternion, 4]
    var trial_r: ASDPoints
    var committed_r: ASDPoints

    fn __init__(out self, points: ASDPoints):
        self.points = points.copy()
        self.reference = ASDFrame(points)
        self.inverse_initial_rotation = _frame_quaternion(self.reference.axes)
        self.inverse_initial_rotation *= _Quaternion(-1.0, -1.0, -1.0, 1.0)
        self.trial_q = InlineArray[_Quaternion, 4](fill=_Quaternion(0.0, 0.0, 0.0, 1.0))
        self.committed_q = self.trial_q.copy()
        self.trial_r = ASDPoints(fill=ASDPoint(0.0))
        self.committed_r = self.trial_r.copy()

    fn __copyinit__(out self, existing: Self):
        self.points = existing.points.copy()
        self.reference = existing.reference.copy()
        self.inverse_initial_rotation = existing.inverse_initial_rotation
        self.trial_q = existing.trial_q.copy()
        self.committed_q = existing.committed_q.copy()
        self.trial_r = existing.trial_r.copy()
        self.committed_r = existing.committed_r.copy()

    fn commit(mut self):
        self.committed_q = self.trial_q.copy()
        self.committed_r = self.trial_r.copy()

    fn revert(mut self):
        self.trial_q = self.committed_q.copy()
        self.trial_r = self.committed_r.copy()

    fn local_displacement(
        mut self, u: UnsafePointer[Float64], local: UnsafePointer[Float64]
    ) -> ASDFrame:
        var positions = self.points.copy()
        for i in range(4):
            var p = 6 * i
            positions[i] += _load_point(u, p)
            var rotation = _load_point(u, p + 3)
            var delta_q = _rotation_quaternion(rotation - self.trial_r[i])
            self.trial_q[i] = _quaternion_product(delta_q, self.trial_q[i])
            self.trial_r[i] = rotation
        var initial_trial = ASDFrame(positions)
        var frame = ASDFrame(positions, _frame_rotation(self.reference, initial_trial))
        var q = _frame_quaternion(frame.axes)
        for i in range(4):
            var deformation_q = _quaternion_product(
                _quaternion_product(q, self.trial_q[i]), self.inverse_initial_rotation
            )
            _write_point(local, 6 * i, frame.points[i] - self.reference.points[i])
            _write_point(local, 6 * i + 3, _rotation_vector(deformation_q))
        return frame^


fn asd_corotational_response[
    tangent: Bool
](
    frame: ASDFrame,
    local_u: UnsafePointer[Float64],
    local_stiffness: UnsafePointer[Float64],
    matrix: UnsafePointer[Float64],
    force: UnsafePointer[Float64],
):
    var local_force = _Vector(fill=0.0)
    for i in range(24):
        local_force[i] = shell4_dot_row(local_stiffness + 24 * i, local_u)
    # With the source's polar alignment the rotation-gradient matrix is
    # zero. The EICR projector therefore only removes mean translation.
    _project_translations(local_force)
    for i in range(24):
        var value = 0.0
        var first = (i // 3) * 3
        for j in range(3):
            value += frame.axes[j][i % 3] * local_force[first + j]
        force[i] = value

    @parameter
    if tangent:
        var left = _Matrix(fill=0.0)
        var right = _Matrix(fill=0.0)
        for n in range(4):
            var p = n * 6
            var rotation = _load_point(local_u, p + 3)
            for j in range(3):
                var direction = ASDPoint(frame.axes[0][j], frame.axes[1][j], frame.axes[2][j], 0.0)
                var axial = _apply_axial_jacobian(rotation, direction)
                for i in range(3):
                    left[(p + i + 3) * 24 + p + j + 3] = direction[i]
                    right[(p + i + 3) * 24 + p + j + 3] = axial[i]
                    for m in range(4):
                        var factor = 0.75 if m == n else -0.25
                        var index = (6 * m + i) * 24 + p + j
                        left[index] = factor * direction[i]
                        right[index] = left[index]
        _transform_tangent(
            local_stiffness, UnsafePointer(to=left[0]), UnsafePointer(to=right[0]), matrix
        )
