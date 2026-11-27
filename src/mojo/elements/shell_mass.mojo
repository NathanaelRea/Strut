"""Consistent translational mass for shell elements."""

from collections import InlineArray, List
from math import sqrt


@fieldwise_init
struct ShellMass(ImplicitlyCopyable, Movable):
    var element: Int
    var dofs: InlineArray[Int, 4]
    var weights: InlineArray[Float64, 16]
    var node_count: Int


fn shell_quad_mass(
    x: List[Float64], y: List[Float64], z: List[Float64], rho_area: Float64
) -> InlineArray[Float64, 16]:
    # The determinant projected onto the ShellMITC4 reference plane equals
    # the dot product of its unit normal with the natural-coordinate tangents.
    var ax = x[2] + x[1] - x[3] - x[0]
    var ay = y[2] + y[1] - y[3] - y[0]
    var az = z[2] + z[1] - z[3] - z[0]
    var bx = x[3] + x[2] - x[1] - x[0]
    var by = y[3] + y[2] - y[1] - y[0]
    var bz = z[3] + z[2] - z[1] - z[0]
    var nx = ay * bz - az * by
    var ny = az * bx - ax * bz
    var nz = ax * by - ay * bx
    var scale = rho_area / (16.0 * sqrt(nx * nx + ny * ny + nz * nz))
    var cx = x[0] - x[1] + x[2] - x[3]
    var cy = y[0] - y[1] + y[2] - y[3]
    var cz = z[0] - z[1] + z[2] - z[3]
    var g = 1.0 / sqrt(3.0)
    var weights = InlineArray[Float64, 16](fill=0.0)
    for s in [-g, g]:
        for t in [-g, g]:
            var dx = ax + cx * t
            var dy = ay + cy * t
            var dz = az + cz * t
            var ex = bx + cx * s
            var ey = by + cy * s
            var ez = bz + cz * s
            var weight = scale * (
                nx * (dy * ez - dz * ey) + ny * (dz * ex - dx * ez) + nz * (dx * ey - dy * ex)
            )
            var shape = [
                0.25 * (1.0 - s) * (1.0 - t),
                0.25 * (1.0 + s) * (1.0 - t),
                0.25 * (1.0 + s) * (1.0 + t),
                0.25 * (1.0 - s) * (1.0 + t),
            ]
            for i in range(4):
                for j in range(4):
                    weights[4 * i + j] += weight * shape[i] * shape[j]
    return weights^


fn shell_triangle_mass(area: Float64, rho_area: Float64) -> InlineArray[Float64, 16]:
    var weights = InlineArray[Float64, 16](fill=0.0)
    var scale = rho_area * area / 12.0
    for i in range(3):
        for j in range(3):
            weights[4 * i + j] = scale
        weights[5 * i] = 2.0 * scale
    return weights^
