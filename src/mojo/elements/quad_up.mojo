from math import sqrt
from os import abort
from utils import StaticTuple

from elements.quad4 import PlaneMatrix
from materials.multi_yield import SoilVector


alias QuadUPVector = StaticTuple[Float64, 22]
alias QuadUPShape = StaticTuple[Float64, 9]


fn quad_up_displacement_dof(node: Int) -> Int:
    if node < 4:
        return 3 * node
    return 12 + 2 * (node - 4)


struct QuadUPPoint(ImplicitlyCopyable, Movable):
    var shape: QuadUPShape
    var dx: QuadUPShape
    var dy: QuadUPShape
    var weight: Float64

    fn __init__(
        out self,
        nodes: Int,
        xi: Float64,
        eta: Float64,
        weight: Float64,
        x: QuadUPShape,
        y: QuadUPShape,
    ):
        self.shape = QuadUPShape(fill=0.0)
        self.dx = QuadUPShape(fill=0.0)
        self.dy = QuadUPShape(fill=0.0)
        var signs_x = StaticTuple[Float64, 4](-0.5, 0.5, 0.5, -0.5)
        var signs_y = StaticTuple[Float64, 4](-0.5, -0.5, 0.5, 0.5)
        var center = 0.0
        var center_x = 0.0
        var center_y = 0.0
        if nodes == 9:
            var xi_mid = 1.0 - xi * xi
            var eta_mid = 1.0 - eta * eta
            center = xi_mid * eta_mid
            center_x = -2.0 * xi * eta_mid
            center_y = -2.0 * eta * xi_mid
            self.shape[8] = center
            self.dx[8] = center_x
            self.dy[8] = center_y
            self.shape[4] = 0.5 * xi_mid * (1.0 - eta) - 0.5 * center
            self.shape[5] = 0.5 * eta_mid * (1.0 + xi) - 0.5 * center
            self.shape[6] = 0.5 * xi_mid * (1.0 + eta) - 0.5 * center
            self.shape[7] = 0.5 * eta_mid * (1.0 - xi) - 0.5 * center
            self.dx[4] = -xi * (1.0 - eta) - 0.5 * center_x
            self.dx[5] = 0.5 * eta_mid - 0.5 * center_x
            self.dx[6] = -xi * (1.0 + eta) - 0.5 * center_x
            self.dx[7] = -0.5 * eta_mid - 0.5 * center_x
            self.dy[4] = -0.5 * xi_mid - 0.5 * center_y
            self.dy[5] = -eta * (1.0 + xi) - 0.5 * center_y
            self.dy[6] = 0.5 * xi_mid - 0.5 * center_y
            self.dy[7] = -eta * (1.0 - xi) - 0.5 * center_y
        for i in range(4):
            var a = 0.5 + signs_x[i] * xi
            var b = 0.5 + signs_y[i] * eta
            self.shape[i] = a * b - 0.25 * center
            self.dx[i] = signs_x[i] * b - 0.25 * center_x
            self.dy[i] = a * signs_y[i] - 0.25 * center_y
        if nodes == 9:
            for i in range(4):
                var next = (i + 1) % 4
                self.shape[i] -= 0.5 * self.shape[i + 4]
                self.shape[next] -= 0.5 * self.shape[i + 4]
                self.dx[i] -= 0.5 * self.dx[i + 4]
                self.dx[next] -= 0.5 * self.dx[i + 4]
                self.dy[i] -= 0.5 * self.dy[i + 4]
                self.dy[next] -= 0.5 * self.dy[i + 4]

        var x_xi = 0.0
        var y_xi = 0.0
        var x_eta = 0.0
        var y_eta = 0.0
        for i in range(nodes):
            x_xi += x[i] * self.dx[i]
            y_xi += y[i] * self.dx[i]
            x_eta += x[i] * self.dy[i]
            y_eta += y[i] * self.dy[i]
        var jacobian = x_xi * y_eta - y_xi * x_eta
        if jacobian <= 0.0:
            abort("u-p quadrilateral requires a positive Jacobian")
        self.weight = weight * jacobian
        for i in range(nodes):
            var xi_derivative = self.dx[i]
            self.dx[i] = (xi_derivative * y_eta - self.dy[i] * y_xi) / jacobian
            self.dy[i] = (self.dy[i] * x_xi - xi_derivative * x_eta) / jacobian

    fn strain(self, nodes: Int, u: QuadUPVector) -> SoilVector:
        var strain = SoilVector(fill=0.0)
        for i in range(nodes):
            var row = quad_up_displacement_dof(i)
            strain[0] += self.dx[i] * u[row]
            strain[1] += self.dy[i] * u[row + 1]
            strain[3] += self.dx[i] * u[row + 1] + self.dy[i] * u[row]
        return strain

    fn add_stiffness(self, nodes: Int, width: Int, tangent: PlaneMatrix, mut matrix: List[Float64]):
        for j in range(nodes):
            var x = self.dx[j]
            var y = self.dy[j]
            var d00 = self.weight * (tangent[0] * x + tangent[2] * y)
            var d10 = self.weight * (tangent[3] * x + tangent[5] * y)
            var d20 = self.weight * (tangent[6] * x + tangent[8] * y)
            var d01 = self.weight * (tangent[1] * y + tangent[2] * x)
            var d11 = self.weight * (tangent[4] * y + tangent[5] * x)
            var d21 = self.weight * (tangent[7] * y + tangent[8] * x)
            var col = quad_up_displacement_dof(j)
            for i in range(nodes):
                var a = self.dx[i]
                var b = self.dy[i]
                var offset = width * quad_up_displacement_dof(i) + col
                matrix[offset] += a * d00 + b * d20
                matrix[offset + 1] += a * d01 + b * d21
                matrix[offset + width] += b * d10 + a * d20
                matrix[offset + width + 1] += b * d11 + a * d21


fn quad_up_points(
    nodes: Int, order: Int, thickness: Float64, x: QuadUPShape, y: QuadUPShape
) -> List[QuadUPPoint]:
    var g = 1.0 / sqrt(3.0) if order == 4 else sqrt(3.0 / 5.0)
    var r = StaticTuple[Float64, 9](-g, g, g, -g, 0.0, g, 0.0, -g, 0.0)
    var s = StaticTuple[Float64, 9](-g, -g, g, g, -g, 0.0, g, 0.0, 0.0)
    var result: List[QuadUPPoint] = []
    for i in range(order):
        var weight = thickness
        if order == 9:
            if i < 4:
                weight *= 25.0 / 81.0
            elif i < 8:
                weight *= 40.0 / 81.0
            else:
                weight *= 64.0 / 81.0
        result.append(QuadUPPoint(nodes, r[i], s[i], weight, x, y))
    return result^
