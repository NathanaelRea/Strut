from collections import List
from math import sqrt
from os import abort

from elements.utils import _cross, _dot, _normalize, _zero_matrix
from linalg import matmul, transpose


fn beam3d_local_stiffness(
    E: Float64,
    A: Float64,
    Iy: Float64,
    Iz: Float64,
    G: Float64,
    J: Float64,
    L: Float64,
) -> List[List[Float64]]:
    var k = _zero_matrix(12, 12)

    var EA_L = E * A / L
    var GJ_L = G * J / L

    var EIz = E * Iz
    var EIy = E * Iy
    var L2 = L * L
    var L3 = L2 * L

    var k1 = 12.0 * EIz / L3
    var k2 = 6.0 * EIz / L2
    var k3 = 4.0 * EIz / L
    var k4 = 2.0 * EIz / L

    var k5 = 12.0 * EIy / L3
    var k6 = 6.0 * EIy / L2
    var k7 = 4.0 * EIy / L
    var k8 = 2.0 * EIy / L

    k[0][0] = EA_L
    k[0][6] = -EA_L
    k[6][0] = -EA_L
    k[6][6] = EA_L

    k[3][3] = GJ_L
    k[3][9] = -GJ_L
    k[9][3] = -GJ_L
    k[9][9] = GJ_L

    k[1][1] = k1
    k[1][5] = k2
    k[1][7] = -k1
    k[1][11] = k2

    k[5][1] = k2
    k[5][5] = k3
    k[5][7] = -k2
    k[5][11] = k4

    k[7][1] = -k1
    k[7][5] = -k2
    k[7][7] = k1
    k[7][11] = -k2

    k[11][1] = k2
    k[11][5] = k4
    k[11][7] = -k2
    k[11][11] = k3

    k[2][2] = k5
    k[2][4] = -k6
    k[2][8] = -k5
    k[2][10] = -k6

    k[4][2] = -k6
    k[4][4] = k7
    k[4][8] = k6
    k[4][10] = k8

    k[8][2] = -k5
    k[8][4] = k6
    k[8][8] = k5
    k[8][10] = k6

    k[10][2] = -k6
    k[10][4] = k8
    k[10][8] = k6
    k[10][10] = k7

    return k^


fn _beam3d_rotation(
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

    var lx: Float64
    var ly: Float64
    var lz: Float64
    (lx, ly, lz) = _normalize(dx, dy, dz)

    var rx = 0.0
    var ry = 0.0
    var rz = 1.0
    if abs(_dot(lx, ly, lz, rx, ry, rz)) > 0.9:
        rx = 0.0
        ry = 1.0
        rz = 0.0

    var yx: Float64
    var yy: Float64
    var yz: Float64
    (yx, yy, yz) = _cross(rx, ry, rz, lx, ly, lz)
    (yx, yy, yz) = _normalize(yx, yy, yz)

    var zx: Float64
    var zy: Float64
    var zz: Float64
    (zx, zy, zz) = _cross(lx, ly, lz, yx, yy, yz)

    return [
        [lx, ly, lz],
        [yx, yy, yz],
        [zx, zy, zz],
    ]


fn beam3d_global_stiffness(
    E: Float64,
    A: Float64,
    Iy: Float64,
    Iz: Float64,
    G: Float64,
    J: Float64,
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

    var k_local = beam3d_local_stiffness(E, A, Iy, Iz, G, J, L)
    var R = _beam3d_rotation(x1, y1, z1, x2, y2, z2)

    var T = _zero_matrix(12, 12)
    var offsets = [0, 3, 6, 9]
    for b in range(4):
        var offset = offsets[b]
        for i in range(3):
            for j in range(3):
                T[offset + i][offset + j] = R[i][j]

    return matmul(transpose(T), matmul(k_local, T))
