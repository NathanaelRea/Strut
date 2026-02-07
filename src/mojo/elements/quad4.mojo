from collections import List
from math import sqrt
from os import abort

from elements.utils import _zero_matrix


fn _quad_shape_gradients(
    xi: Float64,
    eta: Float64,
    mut N: List[Float64],
    mut dN_dxi: List[Float64],
    mut dN_deta: List[Float64],
):
    N.resize(4, 0.0)
    dN_dxi.resize(4, 0.0)
    dN_deta.resize(4, 0.0)

    N[0] = 0.25 * (1.0 - xi) * (1.0 - eta)
    N[1] = 0.25 * (1.0 + xi) * (1.0 - eta)
    N[2] = 0.25 * (1.0 + xi) * (1.0 + eta)
    N[3] = 0.25 * (1.0 - xi) * (1.0 + eta)

    dN_dxi[0] = -0.25 * (1.0 - eta)
    dN_dxi[1] = 0.25 * (1.0 - eta)
    dN_dxi[2] = 0.25 * (1.0 + eta)
    dN_dxi[3] = -0.25 * (1.0 + eta)

    dN_deta[0] = -0.25 * (1.0 - xi)
    dN_deta[1] = -0.25 * (1.0 + xi)
    dN_deta[2] = 0.25 * (1.0 + xi)
    dN_deta[3] = 0.25 * (1.0 - xi)


fn quad4_plane_stress_stiffness(
    E: Float64,
    nu: Float64,
    t: Float64,
    x: List[Float64],
    y: List[Float64],
) -> List[List[Float64]]:
    if len(x) != 4 or len(y) != 4:
        abort("quad4 requires 4 nodes")

    var K = _zero_matrix(8, 8)

    var factor = E / (1.0 - nu * nu)
    var D: List[List[Float64]] = [
        [factor, factor * nu, 0.0],
        [factor * nu, factor, 0.0],
        [0.0, 0.0, factor * (1.0 - nu) / 2.0],
    ]

    var g = 1.0 / sqrt(3.0)
    var gauss = [-g, g]

    for xi in gauss:
        for eta in gauss:
            var N: List[Float64] = []
            var dN_dxi: List[Float64] = []
            var dN_deta: List[Float64] = []
            _quad_shape_gradients(xi, eta, N, dN_dxi, dN_deta)

            var dx_dxi = 0.0
            var dy_dxi = 0.0
            var dx_deta = 0.0
            var dy_deta = 0.0
            for i in range(4):
                dx_dxi += dN_dxi[i] * x[i]
                dy_dxi += dN_dxi[i] * y[i]
                dx_deta += dN_deta[i] * x[i]
                dy_deta += dN_deta[i] * y[i]

            var detJ = dx_dxi * dy_deta - dx_deta * dy_dxi
            if detJ == 0.0:
                abort("quad4 singular jacobian")

            var invJ11 = dy_deta / detJ
            var invJ12 = -dy_dxi / detJ
            var invJ21 = -dx_deta / detJ
            var invJ22 = dx_dxi / detJ

            var dN_dx: List[Float64] = []
            var dN_dy: List[Float64] = []
            dN_dx.resize(4, 0.0)
            dN_dy.resize(4, 0.0)
            for i in range(4):
                dN_dx[i] = invJ11 * dN_dxi[i] + invJ12 * dN_deta[i]
                dN_dy[i] = invJ21 * dN_dxi[i] + invJ22 * dN_deta[i]

            var B = _zero_matrix(3, 8)
            for i in range(4):
                var col_u = 2 * i
                var col_v = col_u + 1
                B[0][col_u] = dN_dx[i]
                B[1][col_v] = dN_dy[i]
                B[2][col_u] = dN_dy[i]
                B[2][col_v] = dN_dx[i]

            var temp = _zero_matrix(3, 8)
            for i in range(3):
                for j in range(8):
                    temp[i][j] = (
                        D[i][0] * B[0][j]
                        + D[i][1] * B[1][j]
                        + D[i][2] * B[2][j]
                    )

            for i in range(8):
                for j in range(8):
                    var sum = (
                        B[0][i] * temp[0][j]
                        + B[1][i] * temp[1][j]
                        + B[2][i] * temp[2][j]
                    )
                    K[i][j] += sum * t * detJ

    return K^
