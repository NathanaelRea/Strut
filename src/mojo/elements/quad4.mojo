from collections import List
from os import abort
from utils import StaticTuple

from elements.utils import _zero_matrix
from materials.multi_yield import SoilVector, SoilTangent


alias QuadVector = StaticTuple[Float64, 8]
alias QuadMatrix = InlineArray[Float64, 64]
alias PlaneMatrix = StaticTuple[Float64, 9]
alias PlaneVector = StaticTuple[Float64, 3]


struct QuadPoint(ImplicitlyCopyable, Movable):
    var shape: StaticTuple[Float64, 4]
    var dx: StaticTuple[Float64, 4]
    var dy: StaticTuple[Float64, 4]
    var weight: Float64

    fn __init__(
        out self, xi: Float64, eta: Float64, t: Float64, x: List[Float64], y: List[Float64]
    ):
        var xm = 1.0 - xi
        var xp = 1.0 + xi
        var ym = 1.0 - eta
        var yp = 1.0 + eta
        self.shape = StaticTuple[Float64, 4](
            0.25 * xm * ym, 0.25 * xp * ym, 0.25 * xp * yp, 0.25 * xm * yp
        )
        var j00 = 0.25 * (-x[0] * ym + x[1] * ym + x[2] * yp - x[3] * yp)
        var j01 = 0.25 * (-x[0] * xm - x[1] * xp + x[2] * xp + x[3] * xm)
        var j10 = 0.25 * (-y[0] * ym + y[1] * ym + y[2] * yp - y[3] * yp)
        var j11 = 0.25 * (-y[0] * xm - y[1] * xp + y[2] * xp + y[3] * xm)
        var det = j00 * j11 - j01 * j10
        if det <= 0.0:
            abort("quad4 requires a positive Jacobian")
        self.weight = det * t
        var inverse_det = 1.0 / det
        var x_xi = 0.25 * (j11 * inverse_det)
        var x_eta = 0.25 * (-j10 * inverse_det)
        var y_xi = 0.25 * (-j01 * inverse_det)
        var y_eta = 0.25 * (j00 * inverse_det)
        self.dx = StaticTuple[Float64, 4](
            -x_xi * ym - x_eta * xm,
            x_xi * ym - x_eta * xp,
            x_xi * yp + x_eta * xp,
            -x_xi * yp + x_eta * xm,
        )
        self.dy = StaticTuple[Float64, 4](
            -y_xi * ym - y_eta * xm,
            y_xi * ym - y_eta * xp,
            y_xi * yp + y_eta * xp,
            -y_xi * yp + y_eta * xm,
        )

    fn strain(self, u: QuadVector) -> SoilVector:
        var strain = SoilVector(fill=0.0)
        for i in range(4):
            strain[0] += self.dx[i] * u[2 * i]
            strain[1] += self.dy[i] * u[2 * i + 1]
            strain[3] += self.dy[i] * u[2 * i] + self.dx[i] * u[2 * i + 1]
        return strain

    fn add_force(self, stress: SoilVector, bx: Float64, by: Float64, mut force: QuadVector):
        for i in range(4):
            force[2 * i] += self.weight * (self.dx[i] * stress[0] + self.dy[i] * stress[3])
            force[2 * i + 1] += self.weight * (self.dy[i] * stress[1] + self.dx[i] * stress[3])
            force[2 * i] -= self.weight * (self.shape[i] * bx)
            force[2 * i + 1] -= self.weight * (self.shape[i] * by)

    fn add_stiffness(self, tangent: PlaneMatrix, mut matrix: QuadMatrix):
        for j in range(4):
            var x = self.dx[j]
            var y = self.dy[j]
            var d00 = self.weight * (tangent[0] * x + tangent[2] * y)
            var d10 = self.weight * (tangent[3] * x + tangent[5] * y)
            var d20 = self.weight * (tangent[6] * x + tangent[8] * y)
            var d01 = self.weight * (tangent[1] * y + tangent[2] * x)
            var d11 = self.weight * (tangent[4] * y + tangent[5] * x)
            var d21 = self.weight * (tangent[7] * y + tangent[8] * x)
            for i in range(4):
                var a = self.dx[i]
                var b = self.dy[i]
                var index = 16 * i + 2 * j
                matrix[index] += a * d00 + b * d20
                matrix[index + 1] += a * d01 + b * d21
                matrix[index + 8] += b * d10 + a * d20
                matrix[index + 9] += b * d11 + a * d21


fn quad4_points(x: List[Float64], y: List[Float64], thickness: Float64) -> List[QuadPoint]:
    if len(x) != 4 or len(y) != 4:
        abort("quad4 requires 4 nodes")
    var g = 0.5773502691896258
    return [
        QuadPoint(-g, -g, thickness, x, y),
        QuadPoint(g, -g, thickness, x, y),
        QuadPoint(g, g, thickness, x, y),
        QuadPoint(-g, g, thickness, x, y),
    ]


fn quad4_plane_tangent(tangent: SoilTangent) -> PlaneMatrix:
    var result = PlaneMatrix(fill=0.0)
    var axes = StaticTuple[Int, 3](0, 1, 3)
    for i in range(3):
        for j in range(3):
            result[3 * i + j] = tangent[6 * axes[i] + axes[j]]
    return result


fn quad4_elastic_tangent(E: Float64, nu: Float64, plane_strain: Bool) -> PlaneMatrix:
    var D = PlaneMatrix(fill=0.0)
    D[8] = E / (2.0 * (1.0 + nu))
    if plane_strain:
        D[1] = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        D[0] = 2.0 * D[8] + D[1]
    else:
        D[0] = E / (1.0 - nu * nu)
        D[1] = nu * D[0]
    D[3] = D[1]
    D[4] = D[0]
    return D


fn quad4_plane_stress_stiffness(
    E: Float64, nu: Float64, t: Float64, x: List[Float64], y: List[Float64]
) -> List[List[Float64]]:
    var tangent = quad4_elastic_tangent(E, nu, False)
    var matrix = QuadMatrix(fill=0.0)
    for point in quad4_points(x, y, t):
        point.add_stiffness(tangent, matrix)
    var result = _zero_matrix(8, 8)
    for i in range(8):
        for j in range(8):
            result[i][j] = matrix[8 * i + j]
    return result^
