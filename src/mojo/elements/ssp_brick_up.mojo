from collections import List
from math import sqrt
from os import abort
from utils import StaticTuple

from materials.multi_yield import SoilTangent, SoilVector


# OpenSees SSPbrickUP interpolations and hourglass stabilization (see COPYRIGHT.md).
alias SpatialVector = StaticTuple[Float64, 3]
alias BrickCoordinates = StaticTuple[SpatialVector, 8]
alias BrickUPVector = StaticTuple[Float64, 32]
alias _Matrix3 = StaticTuple[Float64, 9]
alias _NodeScalars = StaticTuple[Float64, 8]


fn _node_signs(node: Int) -> SpatialVector:
    var corner = node % 4
    return SpatialVector(
        1.0 if corner == 1 or corner == 2 else -1.0,
        1.0 if corner >= 2 else -1.0,
        1.0 if node >= 4 else -1.0,
    )


fn _det(matrix: _Matrix3) -> Float64:
    return (
        matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7])
        + matrix[1] * (matrix[5] * matrix[6] - matrix[3] * matrix[8])
        + matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6])
    )


fn _triple(a: SpatialVector, b: SpatialVector, c: SpatialVector) -> Float64:
    return _det(_Matrix3(a[0], b[0], c[0], a[1], b[1], c[1], a[2], b[2], c[2]))


fn _jacobian_polynomial(nodes: BrickCoordinates) -> StaticTuple[Float64, 20]:
    var modes = StaticTuple[SpatialVector, 7](fill=SpatialVector(fill=0.0))
    for i in range(8):
        var sign = _node_signs(i)
        var r = sign[0]
        var s = sign[1]
        var t = sign[2]
        var monomials = StaticTuple[Float64, 7](r, s, t, s * t, r * t, r * s, r * s * t)
        for j in range(7):
            for axis in range(3):
                modes[j][axis] += 0.125 * monomials[j] * nodes[i][axis]
    var r = modes[0]
    var s = modes[1]
    var t = modes[2]
    var st = modes[3]
    var rt = modes[4]
    var rs = modes[5]
    var rst = modes[6]
    return StaticTuple[Float64, 20](
        _triple(r, s, t),
        _triple(r, s, rt) + _triple(r, rs, t),
        _triple(r, s, st) + _triple(rs, s, t),
        _triple(rt, s, t) + _triple(r, st, t),
        _triple(rst, s, t) + _triple(st, rt, s) + _triple(st, t, rs),
        _triple(r, rst, t) + _triple(st, rt, r) + _triple(t, rt, rs),
        _triple(r, s, rst) + _triple(st, r, rs) + _triple(s, rt, rs),
        -_triple(r, rt, rs),
        -_triple(st, s, rs),
        -_triple(st, rt, t),
        _triple(s, st, rst),
        -_triple(t, st, rst),
        _triple(t, rt, rst),
        -_triple(r, rt, rst),
        _triple(r, rs, rst),
        -_triple(s, rs, rst),
        2.0 * _triple(st, rt, rs),
        _triple(rst, rt, rs),
        _triple(st, rst, rs),
        _triple(st, rt, rst),
    )


fn _inverse(matrix: _Matrix3, determinant: Float64) -> _Matrix3:
    var inverse = _Matrix3(fill=0.0)
    for i in range(3):
        var r1 = (i + 1) % 3
        var r2 = (i + 2) % 3
        for j in range(3):
            var c1 = (j + 1) % 3
            var c2 = (j + 2) % 3
            inverse[3 * j + i] = (
                matrix[3 * r1 + c1] * matrix[3 * r2 + c2]
                - matrix[3 * r1 + c2] * matrix[3 * r2 + c1]
            ) / determinant
    return inverse


fn _transform_gradient(local_derivative: SpatialVector, inverse: _Matrix3) -> SpatialVector:
    var result = SpatialVector(fill=0.0)
    for dim in range(3):
        for i in range(3):
            result[dim] += local_derivative[i] * inverse[3 * i + dim]
    return result


struct _BrickPoint(ImplicitlyCopyable, Movable):
    var gradient: BrickCoordinates
    var inverse: _Matrix3
    var det: Float64

    fn __init__(out self, nodes: BrickCoordinates, r: Float64, s: Float64, t: Float64):
        var local_derivatives = BrickCoordinates(fill=SpatialVector(fill=0.0))
        var jacobian = _Matrix3(fill=0.0)
        for node in range(8):
            var sign = _node_signs(node)
            var x = 1.0 + sign[0] * r
            var y = 1.0 + sign[1] * s
            var z = 1.0 + sign[2] * t
            var local = SpatialVector(
                0.125 * sign[0] * y * z, 0.125 * sign[1] * x * z, 0.125 * sign[2] * x * y
            )
            local_derivatives[node] = local
            for row in range(3):
                for col in range(3):
                    jacobian[3 * row + col] += nodes[node][row] * local[col]
        self.det = _det(jacobian)
        if self.det <= 0.0:
            abort("SSPbrickUP requires a positive Jacobian")
        self.inverse = _inverse(jacobian, self.det)
        self.gradient = BrickCoordinates(fill=SpatialVector(fill=0.0))
        for node in range(8):
            self.gradient[node] = _transform_gradient(local_derivatives[node], self.inverse)


fn _hourglass_values(sign: SpatialVector) -> StaticTuple[Float64, 4]:
    return StaticTuple[Float64, 4](
        0.125 * sign[0] * sign[1],
        0.125 * sign[1] * sign[2],
        0.125 * sign[0] * sign[2],
        0.125 * sign[0] * sign[1] * sign[2],
    )


fn _elastic_mode_entry(
    left: SpatialVector,
    right: SpatialVector,
    row: Int,
    col: Int,
    diagonal: Float64,
    off_diagonal: Float64,
    shear: Float64,
) -> Float64:
    if row == col:
        var value = diagonal * left[row] * right[row]
        for axis in range(3):
            if axis != row:
                value += shear * left[axis] * right[axis]
        return value
    return off_diagonal * left[row] * right[col] + shear * left[col] * right[row]


fn _condense_enhanced_modes(
    mut stiffness: List[Float64], mut enhanced: List[Float64], coupling: List[Float64]
):
    # K_stab is the Schur complement of the nine enhanced strain modes.
    for row in range(9):
        for col in range(row + 1):
            var value = enhanced[9 * row + col]
            for k in range(col):
                value -= enhanced[9 * row + k] * enhanced[9 * col + k]
            if row == col:
                if value <= 0.0:
                    abort("SSPbrickUP enhanced strain matrix is singular")
                enhanced[9 * row + col] = sqrt(value)
            else:
                enhanced[9 * row + col] = value / enhanced[9 * col + col]
    var transformed = coupling.copy()
    for row in range(9):
        for col in range(12):
            var value = transformed[12 * row + col]
            for k in range(row):
                value -= enhanced[9 * row + k] * transformed[12 * k + col]
            transformed[12 * row + col] = value / enhanced[9 * row + row]
    for row in range(12):
        for col in range(12):
            for k in range(9):
                stiffness[12 * row + col] -= transformed[12 * k + row] * transformed[12 * k + col]


struct SSPBrickUPGeometry(Copyable, Movable):
    var volume: Float64
    var gradient: BrickCoordinates
    var mass_weights: _NodeScalars
    var body_weights: _NodeScalars
    var permeability_gradients: List[_BrickPoint]
    var stabilization: List[Float64]
    var strain_matrix: List[Float64]

    fn __init__(out self, nodes: BrickCoordinates, initial_tangent: SoilTangent):
        self.volume = 0.0
        self.gradient = BrickCoordinates(fill=SpatialVector(fill=0.0))
        self.mass_weights = _NodeScalars(fill=0.0)
        self.body_weights = _NodeScalars(fill=0.0)
        self.permeability_gradients = []
        self.stabilization = []
        self.strain_matrix = []
        self.stabilization.resize(1024, 0.0)
        self.strain_matrix.resize(6 * 32, 0.0)
        var one_point = _BrickPoint(nodes, 0.0, 0.0, 0.0)
        var diagonal = initial_tangent[0]
        var off_diagonal = initial_tangent[1]
        var shear = initial_tangent[21]
        var hourglass: List[Float64] = []
        var enhanced: List[Float64] = []
        var coupling: List[Float64] = []
        hourglass.resize(12 * 12, 0.0)
        enhanced.resize(9 * 9, 0.0)
        coupling.resize(9 * 12, 0.0)

        var g = sqrt(3.0 / 5.0)
        var gauss = SpatialVector(-g, 0.0, g)
        var weights = SpatialVector(5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0)
        for ix in range(3):
            for iy in range(3):
                for iz in range(3):
                    var r = gauss[ix]
                    var s = gauss[iy]
                    var t = gauss[iz]
                    var point = _BrickPoint(nodes, r, s, t)
                    var weight = point.det * weights[ix] * weights[iy] * weights[iz]
                    for node in range(8):
                        for dim in range(3):
                            self.gradient[node][dim] += weight * point.gradient[node][dim]
                    var bending = StaticTuple[SpatialVector, 4](
                        _transform_gradient(SpatialVector(s, r, 0.0), one_point.inverse),
                        _transform_gradient(SpatialVector(0.0, t, s), one_point.inverse),
                        _transform_gradient(SpatialVector(t, 0.0, r), one_point.inverse),
                        _transform_gradient(SpatialVector(s * t, r * t, r * s), one_point.inverse),
                    )
                    var modes = StaticTuple[SpatialVector, 3](
                        _transform_gradient(SpatialVector(r, 0.0, 0.0), one_point.inverse),
                        _transform_gradient(SpatialVector(0.0, s, 0.0), one_point.inverse),
                        _transform_gradient(SpatialVector(0.0, 0.0, t), one_point.inverse),
                    )
                    for row in range(12):
                        for col in range(12):
                            var normal = row < 9 and col < 9
                            hourglass[12 * row + col] += weight * _elastic_mode_entry(
                                bending[row // 3],
                                bending[col // 3],
                                row % 3,
                                col % 3,
                                diagonal if normal else 0.0,
                                off_diagonal if normal else 0.0,
                                shear,
                            )
                    for row in range(9):
                        for col in range(9):
                            enhanced[9 * row + col] += weight * _elastic_mode_entry(
                                modes[row // 3],
                                modes[col // 3],
                                row % 3,
                                col % 3,
                                diagonal,
                                off_diagonal,
                                shear,
                            )
                            coupling[12 * row + col] += weight * _elastic_mode_entry(
                                modes[row // 3],
                                bending[col // 3],
                                row % 3,
                                col % 3,
                                diagonal,
                                off_diagonal,
                                shear,
                            )
        var jac = _jacobian_polynomial(nodes)
        self.volume = 8.0 * (jac[0] + (jac[7] + jac[8] + jac[9]) / 3.0)
        # Preserve the published element's higher-order Jacobian moments in the
        # shear-only block for the cubic hourglass mode.
        var cross_pairs = StaticTuple[Int, 6](1, 2, 0, 2, 0, 1)
        for k in range(3):
            var a = cross_pairs[2 * k]
            var b = cross_pairs[2 * k + 1]
            var left = SpatialVector(
                one_point.inverse[3 * a], one_point.inverse[3 * a + 1], one_point.inverse[3 * a + 2]
            )
            var right = SpatialVector(
                one_point.inverse[3 * b], one_point.inverse[3 * b + 1], one_point.inverse[3 * b + 2]
            )
            var correction = 8.0 * 63.0 * jac[17 + k] / 45.0
            for i in range(3):
                for j in range(3):
                    hourglass[12 * (9 + i) + 9 + j] += correction * (
                        _elastic_mode_entry(left, right, i, j, 0.0, 0.0, shear)
                        + _elastic_mode_entry(right, left, i, j, 0.0, 0.0, shear)
                    )
        _condense_enhanced_modes(hourglass, enhanced, coupling)

        var hourglass_projection = StaticTuple[SpatialVector, 4](fill=SpatialVector(fill=0.0))
        for node in range(8):
            var sign = _node_signs(node)
            var coord = _hourglass_values(sign)
            var rx = 0.125 * sign[0]
            var sy = 0.125 * sign[1]
            var tz = 0.125 * sign[2]
            var correction = (
                (jac[1] * rx + jac[2] * sy + jac[3] * tz + jac[7] + jac[8] + jac[9]) / 3.0
                + (
                    jac[4] * coord[1]
                    + jac[5] * coord[2]
                    + jac[6] * coord[0]
                    + (jac[10] + jac[13]) * tz
                    + (jac[11] + jac[14]) * sy
                    + (jac[12] + jac[15]) * rx
                )
                / 9.0
                + (
                    jac[16] * coord[3]
                    + jac[17] * coord[1]
                    + jac[18] * coord[2]
                    + jac[19] * coord[0]
                )
                / 27.0
            )
            self.mass_weights[node] = jac[0] + correction
            self.body_weights[node] = jac[0] * (1.0 + correction)
            var p = 1.0 / sqrt(3.0)
            self.permeability_gradients.append(
                _BrickPoint(nodes, p * sign[0], p * sign[1], p * sign[2])
            )
            for mode in range(4):
                for dim in range(3):
                    hourglass_projection[mode][dim] += coord[mode] * nodes[node][dim]
            for dim in range(3):
                self.gradient[node][dim] /= self.volume

        var gamma: List[Float64] = []
        gamma.resize(8 * 4, 0.0)
        for node in range(8):
            var d = self.gradient[node]
            var modes = _hourglass_values(_node_signs(node))
            for k in range(4):
                gamma[4 * node + k] = modes[k]
                for dim in range(3):
                    gamma[4 * node + k] -= d[dim] * hourglass_projection[k][dim]
            var col = 4 * node
            self.strain_matrix[col] = d[0]
            self.strain_matrix[32 + col + 1] = d[1]
            self.strain_matrix[64 + col + 2] = d[2]
            self.strain_matrix[96 + col] = d[1]
            self.strain_matrix[96 + col + 1] = d[0]
            self.strain_matrix[128 + col + 1] = d[2]
            self.strain_matrix[128 + col + 2] = d[1]
            self.strain_matrix[160 + col] = d[2]
            self.strain_matrix[160 + col + 2] = d[0]

        for i in range(8):
            for j in range(8):
                for a in range(3):
                    for b in range(3):
                        var value = 0.0
                        for k in range(4):
                            for l in range(4):
                                value += (
                                    gamma[4 * i + k]
                                    * hourglass[12 * (3 * k + a) + 3 * l + b]
                                    * gamma[4 * j + l]
                                )
                        self.stabilization[32 * (4 * i + a) + 4 * j + b] = value

    fn strain(self, displacement: BrickUPVector) -> SoilVector:
        var result = SoilVector(fill=0.0)
        for i in range(6):
            for j in range(32):
                result[i] += self.strain_matrix[32 * i + j] * displacement[j]
        return result

    fn add_solid_response(
        self,
        stress: SoilVector,
        tangent: SoilTangent,
        displacement: BrickUPVector,
        form_tangent: Bool,
        mut force: BrickUPVector,
        mut matrix: List[Float64],
    ):
        if form_tangent:
            matrix = self.stabilization.copy()
        for row in range(32):
            for i in range(6):
                force[row] += self.volume * self.strain_matrix[32 * i + row] * stress[i]
            for col in range(32):
                force[row] += self.stabilization[row * 32 + col] * displacement[col]
                if form_tangent:
                    var value = 0.0
                    for i in range(6):
                        for j in range(6):
                            value += (
                                self.strain_matrix[32 * i + row]
                                * tangent[6 * i + j]
                                * self.strain_matrix[32 * j + col]
                            )
                    matrix[row * 32 + col] += self.volume * value
