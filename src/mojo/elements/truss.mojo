from collections import List
from math import hypot, sqrt
from os import abort


fn truss_global_stiffness(
    E: Float64,
    A: Float64,
    x1: Float64,
    y1: Float64,
    x2: Float64,
    y2: Float64,
) -> List[List[Float64]]:
    var dx = x2 - x1
    var dy = y2 - y1
    var L = hypot(dx, dy)
    if L == 0.0:
        abort("zero-length element")
    var c = dx / L
    var s = dy / L
    var k = E * A / L

    return [
        [k * c * c, k * c * s, -k * c * c, -k * c * s],
        [k * c * s, k * s * s, -k * c * s, -k * s * s],
        [-k * c * c, -k * c * s, k * c * c, k * c * s],
        [-k * c * s, -k * s * s, k * c * s, k * s * s],
    ]


fn truss3d_global_stiffness(
    E: Float64,
    A: Float64,
    x1: Float64,
    y1: Float64,
    z1: Float64,
    x2: Float64,
    y2: Float64,
    z2: Float64,
) -> List[List[Float64]]:
    var dx = x2 - x1
    var dy = y2 - y1
    var dz = z2 - z1
    var L = sqrt(dx * dx + dy * dy + dz * dz)
    if L == 0.0:
        abort("zero-length element")
    var l = dx / L
    var m = dy / L
    var n = dz / L
    var k = E * A / L

    return [
        [k * l * l, k * l * m, k * l * n, -k * l * l, -k * l * m, -k * l * n],
        [k * l * m, k * m * m, k * m * n, -k * l * m, -k * m * m, -k * m * n],
        [k * l * n, k * m * n, k * n * n, -k * l * n, -k * m * n, -k * n * n],
        [-k * l * l, -k * l * m, -k * l * n, k * l * l, k * l * m, k * l * n],
        [-k * l * m, -k * m * m, -k * m * n, k * l * m, k * m * m, k * m * n],
        [-k * l * n, -k * m * n, -k * n * n, k * l * n, k * m * n, k * n * n],
    ]
