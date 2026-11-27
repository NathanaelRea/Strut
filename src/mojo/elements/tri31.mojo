from collections import List
from os import abort

from elements.utils import _zero_matrix


fn tri31_elastic_stiffness(
    E: Float64,
    nu: Float64,
    thickness: Float64,
    plane_strain: Bool,
    x: List[Float64],
    y: List[Float64],
) -> List[List[Float64]]:
    # Tri31 uses a single integration point and a constant strain matrix.
    var dx_13 = x[0] - x[2]
    var dy_13 = y[0] - y[2]
    var dx_23 = x[1] - x[2]
    var dy_23 = y[1] - y[2]
    var det_jacobian = dx_13 * dy_23 - dx_23 * dy_13
    if det_jacobian <= 0.0:
        abort("tri31 requires positive area with counterclockwise node order")

    var dx = [dy_23 / det_jacobian, -dy_13 / det_jacobian, 0.0]
    var dy = [-dx_23 / det_jacobian, dx_13 / det_jacobian, 0.0]
    dx[2] = -dx[0] - dx[1]
    dy[2] = -dy[0] - dy[1]

    var shear = E / (2.0 * (1.0 + nu))
    var normal = E / (1.0 - nu * nu)
    var coupling = nu * normal
    if plane_strain:
        coupling = 2.0 * shear * nu / (1.0 - 2.0 * nu)
        normal = coupling + 2.0 * shear

    var volume = 0.5 * det_jacobian * thickness
    var stiffness = _zero_matrix(6, 6)
    for i in range(3):
        for j in range(3):
            stiffness[2 * i][2 * j] = volume * (normal * dx[i] * dx[j] + shear * dy[i] * dy[j])
            stiffness[2 * i][2 * j + 1] = volume * (
                coupling * dx[i] * dy[j] + shear * dy[i] * dx[j]
            )
            stiffness[2 * i + 1][2 * j] = volume * (
                coupling * dy[i] * dx[j] + shear * dx[i] * dy[j]
            )
            stiffness[2 * i + 1][2 * j + 1] = volume * (
                normal * dy[i] * dy[j] + shear * dx[i] * dx[j]
            )
    return stiffness^
