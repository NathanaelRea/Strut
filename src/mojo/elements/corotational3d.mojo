"""Spatial beam transformation using the nodal triads of CorotCrdTransf3d."""

from math import asin, cos, sin, sqrt, tan
from os import abort
from elements.utils import _ensure_zero_matrix, _ensure_zero_vector
from utils import StaticTuple


alias _Vector = SIMD[DType.float64, 4]
alias _Mat3 = StaticTuple[_Vector, 3]
alias Corotational3dJacobian = StaticTuple[_Vector, 24]


fn _cross(a: _Vector, b: _Vector) -> _Vector:
    return _Vector(
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
        0.0,
    )


fn _dot(a: _Vector, b: _Vector) -> Float64:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


fn _outer(a: _Vector, b: _Vector) -> _Mat3:
    var result = _Mat3(fill=_Vector(0.0))
    for j in range(3):
        result[j] = b[j] * a
    return result


fn _skew(a: _Vector) -> _Mat3:
    var result = _Mat3(fill=_Vector(0.0))
    result[0] = _Vector(0.0, a[2], -a[1], 0.0)
    result[1] = _Vector(-a[2], 0.0, a[0], 0.0)
    result[2] = _Vector(a[1], -a[0], 0.0, 0.0)
    return result


fn _apply(a: _Mat3, b: _Vector) -> _Vector:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


fn _matmul(a: _Mat3, b: _Mat3) -> _Mat3:
    var result = _Mat3(fill=_Vector(0.0))
    for j in range(3):
        result[j] = _apply(a, b[j])
    return result


fn _combine(a: _Mat3, b: _Mat3, sa: Float64, sb: Float64) -> _Mat3:
    var result = _Mat3(fill=_Vector(0.0))
    for j in range(3):
        result[j] = sa * a[j] + sb * b[j]
    return result


fn _add_block(
    mut tangent: List[List[Float64]],
    a: _Mat3,
    row: Int,
    col: Int,
    scale: Float64,
    transpose: Bool = False,
):
    for i in range(3):
        for j in range(3):
            var entry = a[i][j] if transpose else a[j][i]
            tangent[row + i][col + j] += scale * entry


fn _add_translation_block(mut tangent: List[List[Float64]], a: _Mat3, scale: Float64):
    for i in range(2):
        for j in range(2):
            _add_block(tangent, a, 6 * i, 6 * j, scale if i == j else -scale)


fn _add_translation_rotation_blocks(
    mut tangent: List[List[Float64]], a: _Mat3, rotation: Int, scale: Float64
):
    _add_block(tangent, a, 0, rotation, scale)
    _add_block(tangent, a, 6, rotation, -scale)
    _add_block(tangent, a, rotation, 0, scale, True)
    _add_block(tangent, a, rotation, 6, -scale, True)


fn _quaternion(spin: _Vector) -> _Vector:
    var theta = sqrt(_dot(spin, spin))
    var q = _Vector(0.0)
    if theta != 0.0:
        q = (sin(0.5 * theta) / theta) * spin
    q[3] = cos(0.5 * theta)
    return q


fn _quaternion_product(a: _Vector, b: _Vector) -> _Vector:
    var q = a[3] * b + b[3] * a + _cross(a, b)
    q[3] = a[3] * b[3] - _dot(a, b)
    return q


fn _rotate(q: _Vector, axis: _Vector) -> _Vector:
    var qs = q[3]
    var qv = q
    qv[3] = 0.0
    return (qs * qs - _dot(qv, qv)) * axis + 2.0 * (_dot(qv, axis) * qv + qs * _cross(qv, axis))


@fieldwise_init
struct _NodalTriads(ImplicitlyCopyable, Movable):
    var spin_i: _Vector
    var spin_j: _Vector
    var q_i: _Vector
    var q_j: _Vector

    fn __init__(out self):
        var identity = _Vector(0.0, 0.0, 0.0, 1.0)
        self = _NodalTriads(_Vector(0.0), _Vector(0.0), identity, identity)


struct Corotational3dState(ImplicitlyCopyable, Movable):
    var trial: _NodalTriads
    var committed: _NodalTriads

    fn __init__(out self):
        self.trial = _NodalTriads()
        self.committed = _NodalTriads()

    fn update(mut self, u: List[Float64]):
        var spin_i = _Vector(u[3], u[4], u[5], 0.0)
        var spin_j = _Vector(u[9], u[10], u[11], 0.0)
        self.trial.q_i = _quaternion_product(
            _quaternion(spin_i - self.trial.spin_i), self.trial.q_i
        )
        self.trial.q_j = _quaternion_product(
            _quaternion(spin_j - self.trial.spin_j), self.trial.q_j
        )
        self.trial.spin_i = spin_i
        self.trial.spin_j = spin_j

    fn commit(mut self):
        self.committed = self.trial

    fn revert(mut self):
        self.trial = self.committed


@fieldwise_init
struct _Frame(Copyable, Movable):
    var reference_chord: _Vector
    var delta: _Vector
    var axes_i: _Mat3
    var axes_j: _Mat3
    var mean_axes: _Mat3

    fn __init__(
        out self,
        reference_axes: StaticTuple[Float64, 9],
        length: Float64,
        u: List[Float64],
        triads: _NodalTriads,
    ):
        self.delta = _Vector(u[6] - u[0], u[7] - u[1], u[8] - u[2], 0.0)
        var inverse_i = -triads.q_i
        inverse_i[3] = triads.q_i[3]
        var relative = _quaternion_product(triads.q_j, inverse_i)
        # The half tangent-scaled pseudo-vector gives the mean nodal triad.
        relative[3] *= 2.0
        relative /= sqrt(_dot(relative, relative) + relative[3] * relative[3])
        var mean = _quaternion_product(relative, triads.q_i)
        self.axes_i = _Mat3(fill=_Vector(0.0))
        self.axes_j = _Mat3(fill=_Vector(0.0))
        self.mean_axes = _Mat3(fill=_Vector(0.0))
        for i in range(3):
            var axis = _Vector(
                reference_axes[3 * i], reference_axes[3 * i + 1], reference_axes[3 * i + 2], 0.0
            )
            self.axes_i[i] = _rotate(triads.q_i, axis)
            self.axes_j[i] = _rotate(triads.q_j, axis)
            self.mean_axes[i] = _rotate(mean, axis)
        self.reference_chord = length * _Vector(
            reference_axes[0], reference_axes[1], reference_axes[2], 0.0
        )

    fn evaluate(
        self,
        mut basic: StaticTuple[Float64, 6],
        mut angles: StaticTuple[Float64, 6],
        mut jacobian: Corotational3dJacobian,
    ):
        var chord = self.reference_chord + self.delta
        var length = sqrt(_dot(chord, chord))
        if length == 0.0:
            abort("zero-length deformed beam")
        var e1 = chord / length
        var r1 = self.mean_axes[0]
        var r2 = self.mean_axes[1]
        var r3 = self.mean_axes[2]
        var midpoint = 0.5 * (e1 + r1)
        var e2 = r2 - _dot(r2, e1) * midpoint
        var e3 = r3 - _dot(r3, e1) * midpoint
        basic[0] = _dot(2.0 * self.reference_chord + self.delta, self.delta) / (
            length + sqrt(_dot(self.reference_chord, self.reference_chord))
        )
        jacobian[0] = -e1
        jacobian[2] = e1
        for end in range(2):
            var axes = self.axes_i.copy() if end == 0 else self.axes_j.copy()
            var a1 = axes[0]
            var a2 = axes[1]
            var a3 = axes[2]
            var spin_x = 0.5 * (_dot(a2, e3) - _dot(a3, e2))
            var spin_z = 0.5 * (_dot(a1, e2) - _dot(a2, e1))
            var spin_y = 0.5 * (_dot(a3, e1) - _dot(a1, e3))
            var torsion_sign = -1.0 if end == 0 else 1.0
            angles[3 * end] = asin(spin_x)
            angles[3 * end + 1] = asin(spin_z)
            angles[3 * end + 2] = -asin(spin_y)
            basic[5] += torsion_sign * angles[3 * end]
            basic[1 + end] = angles[3 * end + 1]
            basic[3 + end] = -angles[3 * end + 2]
            for j in range(3):
                var basis = _Vector(0.0)
                basis[j] = 1.0
                for block in range(4):
                    var de1 = _Vector(0.0)
                    var dr1 = _Vector(0.0)
                    var dr2 = _Vector(0.0)
                    var dr3 = _Vector(0.0)
                    var da1 = _Vector(0.0)
                    var da2 = _Vector(0.0)
                    var da3 = _Vector(0.0)
                    if block % 2 == 0:
                        var sign = -1.0 if block == 0 else 1.0
                        de1 = (sign / length) * (basis - _dot(basis, e1) * e1)
                    else:
                        # CorotCrdTransf3d's L matrix splits the mean spin
                        # equally between the two end rotations.
                        dr1 = 0.5 * _cross(basis, r1)
                        dr2 = 0.5 * _cross(basis, r2)
                        dr3 = 0.5 * _cross(basis, r3)
                        if block == 2 * end + 1:
                            da1 = _cross(basis, a1)
                            da2 = _cross(basis, a2)
                            da3 = _cross(basis, a3)
                    var dm = 0.5 * (de1 + dr1)
                    var de2 = dr2 - (_dot(dr2, e1) + _dot(r2, de1)) * midpoint - _dot(r2, e1) * dm
                    var de3 = dr3 - (_dot(dr3, e1) + _dot(r3, de1)) * midpoint - _dot(r3, e1) * dm
                    jacobian[20 + block][j] += (
                        torsion_sign
                        * (_dot(da2, e3) + _dot(a2, de3) - _dot(da3, e2) - _dot(a3, de2))
                        / (2.0 * sqrt(1.0 - spin_x * spin_x))
                    )
                    jacobian[4 * (1 + end) + block][j] = (
                        _dot(da1, e2) + _dot(a1, de2) - _dot(da2, e1) - _dot(a2, de1)
                    ) / (2.0 * sqrt(1.0 - spin_z * spin_z))
                    jacobian[4 * (3 + end) + block][j] = (
                        _dot(da3, e1) + _dot(a3, de1) - _dot(da1, e3) - _dot(a1, de3)
                    ) / (2.0 * sqrt(1.0 - spin_y * spin_y))

    fn geometric_tangent(
        self,
        q: StaticTuple[Float64, 6],
        angles: StaticTuple[Float64, 6],
        jacobian: Corotational3dJacobian,
        mut tangent: List[List[Float64]],
    ):
        var chord = self.reference_chord + self.delta
        var length = sqrt(_dot(chord, chord))
        var e1 = chord / length
        var r1 = self.mean_axes[0]
        var r2 = self.mean_axes[1]
        var r3 = self.mean_axes[2]
        var midpoint = 0.5 * (e1 + r1)
        var e2 = r2 - _dot(r2, e1) * midpoint
        var e3 = r3 - _dot(r3, e1) * midpoint
        var projection = _outer(-e1, e1)
        for i in range(3):
            projection[i][i] += 1.0
            projection[i] /= length
        _add_translation_block(tangent, projection, q[0])

        var l2 = _mean_axis_blocks(r1, r2, e1, projection)
        var l3 = _mean_axis_blocks(r1, r3, e1, projection)
        var local_force = StaticTuple[Float64, 6](fill=0.0)
        local_force[0] = -q[5]
        local_force[1] = q[1]
        local_force[2] = -q[3]
        local_force[3] = q[5]
        local_force[4] = q[2]
        local_force[5] = -q[4]
        var m = StaticTuple[Float64, 6](fill=0.0)
        for i in range(6):
            m[i] = local_force[i] / (2.0 * cos(angles[i]))

        var i_axes = self.axes_i.copy()
        var j_axes = self.axes_j.copy()
        var v = (m[1] * i_axes[1] + m[2] * i_axes[2] + m[4] * j_axes[1] + m[5] * j_axes[2]) / length
        var translation = _combine(
            _outer(_apply(projection, v), e1), _outer(e1, _apply(projection, v)), 1.0, 1.0
        )
        translation = _combine(translation, projection, 1.0, _dot(e1, v))
        _add_translation_block(tangent, translation, 1.0)

        for end in range(2):
            var axes = i_axes.copy() if end == 0 else j_axes.copy()
            var rotation = 3 + 6 * end
            var torsion = m[3] if end == 0 else -m[3]
            var bend_z = m[3 * end + 1]
            var bend_y = m[3 * end + 2]
            var skew_e1 = _skew(e1)
            var skew_e2 = _skew(e2)
            var skew_e3 = _skew(e3)
            var skew_r1 = _skew(axes[0])
            var skew_r2 = _skew(axes[1])
            var skew_r3 = _skew(axes[2])

            # Ks3 couples variation of the mean frame to the end triad.
            var sm2 = _skew(-torsion * axes[2] - bend_z * axes[0])
            var sm3 = _skew(torsion * axes[1] - bend_y * axes[0])
            var coupling = _combine(_matmul(l2[0], sm2), _matmul(l3[0], sm3), 1.0, 1.0)
            _add_translation_rotation_blocks(tangent, coupling, rotation, 1.0)
            coupling = _combine(_matmul(l2[1], sm2), _matmul(l3[1], sm3), 1.0, 1.0)
            for end_row in range(2):
                var row = 3 + 6 * end_row
                _add_block(tangent, coupling, row, rotation, 1.0)
                _add_block(tangent, coupling, rotation, row, 1.0, True)

            # Ks4 and Ks5 differentiate the terms attached to one end.
            var end_rotation = _combine(
                _matmul(skew_e2, skew_r3), _matmul(skew_e3, skew_r2), torsion, -torsion
            )
            end_rotation = _combine(end_rotation, _matmul(skew_e2, skew_r1), 1.0, bend_z)
            end_rotation = _combine(end_rotation, _matmul(skew_e3, skew_r1), 1.0, bend_y)
            end_rotation = _combine(
                end_rotation,
                _matmul(skew_e1, _skew(bend_z * axes[1] + bend_y * axes[2])),
                1.0,
                -1.0,
            )
            _add_block(tangent, end_rotation, rotation, rotation, 1.0)
            _add_translation_rotation_blocks(
                tangent,
                _matmul(projection, _skew(bend_z * axes[1] + bend_y * axes[2])),
                rotation,
                -1.0,
            )
            _add_mean_axis_geometric_tangent(
                tangent, projection, length, e1, r1, r2, axes[0], bend_z
            )
            _add_mean_axis_geometric_tangent(
                tangent, projection, length, e1, r1, r3, axes[0], bend_y
            )
        _add_mean_axis_geometric_tangent(
            tangent, projection, length, e1, r1, r2, i_axes[2] - j_axes[2], m[3]
        )
        _add_mean_axis_geometric_tangent(
            tangent, projection, length, e1, r1, r3, j_axes[1] - i_axes[1], m[3]
        )

        # The derivative of asin contributes the diagonal terms in T^T D T.
        for end in range(2):
            var weights = StaticTuple[Float64, 3](fill=0.0)
            for a in range(3):
                weights[a] = local_force[3 * end + a] * tan(angles[3 * end + a])
            var torsion_row = self._end_torsion_jacobian(end)
            for i in range(12):
                for j in range(12):
                    tangent[i][j] += (
                        weights[0] * torsion_row[i // 3][i % 3] * torsion_row[j // 3][j % 3]
                        + weights[1]
                        * jacobian[4 * (1 + end) + i // 3][i % 3]
                        * jacobian[4 * (1 + end) + j // 3][j % 3]
                        + weights[2]
                        * jacobian[4 * (3 + end) + i // 3][i % 3]
                        * jacobian[4 * (3 + end) + j // 3][j % 3]
                    )

    fn _end_torsion_jacobian(self, end: Int) -> StaticTuple[_Vector, 4]:
        var chord = self.reference_chord + self.delta
        var length = sqrt(_dot(chord, chord))
        var e1 = chord / length
        var r1 = self.mean_axes[0]
        var r2 = self.mean_axes[1]
        var r3 = self.mean_axes[2]
        var midpoint = 0.5 * (e1 + r1)
        var e2 = r2 - _dot(r2, e1) * midpoint
        var e3 = r3 - _dot(r3, e1) * midpoint
        var axes = self.axes_i.copy() if end == 0 else self.axes_j.copy()
        var sine = 0.5 * (_dot(axes[1], e3) - _dot(axes[2], e2))
        var denominator = 2.0 * sqrt(1.0 - sine * sine)
        var projection = _outer(-e1, e1)
        for i in range(3):
            projection[i][i] += 1.0
            projection[i] /= length
        var l2 = _mean_axis_blocks(r1, r2, e1, projection)
        var l3 = _mean_axis_blocks(r1, r3, e1, projection)
        var row = StaticTuple[_Vector, 4](fill=_Vector(0.0))
        row[0] = (_apply(l3[0], axes[1]) - _apply(l2[0], axes[2])) / denominator
        row[2] = -row[0]
        row[1] = (_apply(l3[1], axes[1]) - _apply(l2[1], axes[2])) / denominator
        row[3] = row[1]
        row[2 * end + 1] += (_cross(axes[1], e3) - _cross(axes[2], e2)) / denominator
        return row


fn _mean_axis_blocks(r1: _Vector, axis: _Vector, e1: _Vector, projection: _Mat3) -> (_Mat3, _Mat3):
    var along_chord = _dot(axis, e1)
    var chord_plus_axis = e1 + r1
    var skew_axis = _skew(axis)
    var translation = _combine(
        projection,
        _outer(_apply(projection, axis), chord_plus_axis),
        0.5 * along_chord,
        0.5,
    )
    var rotation = _combine(skew_axis, _skew(r1), 0.5, -0.25 * along_chord)
    rotation = _combine(rotation, _outer(_cross(axis, e1), chord_plus_axis), 1.0, -0.25)
    return (translation, rotation)


fn _add_mean_axis_geometric_tangent(
    mut tangent: List[List[Float64]],
    projection: _Mat3,
    length: Float64,
    e1: _Vector,
    r1: _Vector,
    axis: _Vector,
    z: _Vector,
    scale: Float64,
):
    # The Ks2 blocks in CorotCrdTransf3d::getKs2Matrix.
    var axis_e1 = _dot(axis, e1)
    var z_e1 = _dot(z, e1)
    var z_r1 = _dot(z, r1)
    var d = z_e1 + z_r1
    var az = _apply(projection, z)
    var ar = _apply(projection, axis)
    var u = _combine(_outer(az, ar), _outer(az, e1), -0.5, axis_e1 / (2.0 * length))
    u = _combine(u, _outer(ar, e1), 1.0, d / (2.0 * length))
    _add_translation_block(tangent, u, scale)
    var transpose = _Mat3(fill=_Vector(0.0))
    for i in range(3):
        for j in range(3):
            transpose[i][j] = u[j][i]
    _add_translation_block(tangent, transpose, scale)
    _add_translation_block(
        tangent, projection, scale * axis_e1 * (2.0 * z_e1 + z_r1) / (2.0 * length)
    )

    var sr = _skew(axis)
    var sr1 = _skew(r1)
    var sz = _skew(z)
    var se1 = _skew(e1)
    var mixed = _combine(_matmul(_outer(az, e1), sr), _matmul(_outer(ar, z), sr1), -0.25, -0.25)
    mixed = _combine(mixed, _matmul(projection, sr), 1.0, -0.25 * d)
    _add_translation_rotation_blocks(tangent, mixed, 3, scale)
    _add_translation_rotation_blocks(tangent, mixed, 9, scale)
    var rotational = _combine(
        _matmul(sz, sr1), _matmul(_outer(_cross(r1, z), e1), sr), -0.125 * axis_e1, 0.125
    )
    rotational = _combine(rotational, _matmul(_outer(_cross(axis, e1), z), sr1), 1.0, 0.125)
    rotational = _combine(rotational, _matmul(se1, sr), 1.0, -0.125 * d)
    rotational = _combine(rotational, _matmul(sz, sr), 1.0, 0.25)
    for i in range(2):
        for j in range(2):
            _add_block(tangent, rotational, 3 + 6 * i, 3 + 6 * j, scale)


fn corotational3d_basic_displacements(
    reference_axes: StaticTuple[Float64, 9],
    length: Float64,
    u: List[Float64],
    state: Corotational3dState,
) -> StaticTuple[Float64, 6]:
    var basic = StaticTuple[Float64, 6](fill=0.0)
    var jacobian = Corotational3dJacobian(fill=_Vector(0.0))
    var angles = StaticTuple[Float64, 6](fill=0.0)
    _Frame(reference_axes, length, u, state.trial).evaluate(basic, angles, jacobian)
    return basic


fn corotational3d_resisting_force_and_tangent(
    reference_axes: StaticTuple[Float64, 9],
    length: Float64,
    u: List[Float64],
    state: Corotational3dState,
    q: StaticTuple[Float64, 6],
    k_axial_bending: StaticTuple[Float64, 25],
    k_torsion: Float64,
    mut force: List[Float64],
    mut tangent: List[List[Float64]],
):
    var frame = _Frame(reference_axes, length, u, state.trial)
    var basic = StaticTuple[Float64, 6](fill=0.0)
    var jacobian = Corotational3dJacobian(fill=_Vector(0.0))
    var angles = StaticTuple[Float64, 6](fill=0.0)
    frame.evaluate(basic, angles, jacobian)
    _ensure_zero_matrix(tangent, 12, 12)
    _ensure_zero_vector(force, 12)
    for i in range(12):
        var i_block = i // 3
        var i_lane = i % 3
        force[i] = 0.0
        for a in range(6):
            force[i] += q[a] * jacobian[4 * a + i_block][i_lane]
        for j in range(12):
            var j_block = j // 3
            var j_lane = j % 3
            var value = k_torsion * jacobian[20 + i_block][i_lane] * jacobian[20 + j_block][j_lane]
            for a in range(5):
                var weighted = 0.0
                for b in range(5):
                    weighted += k_axial_bending[5 * a + b] * jacobian[4 * b + j_block][j_lane]
                value += jacobian[4 * a + i_block][i_lane] * weighted
            tangent[i][j] = value

    frame.geometric_tangent(q, angles, jacobian, tangent)
