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


fn _beam3d_transform_matrix(R: List[List[Float64]]) -> List[List[Float64]]:
    var T = _zero_matrix(12, 12)
    var offsets = [0, 3, 6, 9]
    for b in range(4):
        var offset = offsets[b]
        for i in range(3):
            for j in range(3):
                T[offset + i][offset + j] = R[i][j]
    return T^


fn _beam3d_transform_u_global_to_local(T: List[List[Float64]], u_global: List[Float64]) -> List[Float64]:
    var u_local: List[Float64] = []
    u_local.resize(12, 0.0)
    for i in range(12):
        var sum = 0.0
        for j in range(12):
            sum += T[i][j] * u_global[j]
        u_local[i] = sum
    return u_local^


fn _beam3d_add_axial_geometric_stiffness(
    mut k_local: List[List[Float64]], N: Float64, L: Float64
):
    var N_over_L = N / L

    k_local[1][1] += N_over_L
    k_local[1][7] -= N_over_L
    k_local[7][1] -= N_over_L
    k_local[7][7] += N_over_L

    k_local[2][2] += N_over_L
    k_local[2][8] -= N_over_L
    k_local[8][2] -= N_over_L
    k_local[8][8] += N_over_L


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

    var T = _beam3d_transform_matrix(R)
    return matmul(transpose(T), matmul(k_local, T))


fn beam3d_pdelta_global_stiffness(
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
    u_elem_global: List[Float64],
) -> List[List[Float64]]:
    var dx = x2 - x1
    var dy = y2 - y1
    var dz = z2 - z1
    var L = sqrt(dx * dx + dy * dy + dz * dz)
    if L == 0.0:
        abort("zero-length element")

    var R = _beam3d_rotation(x1, y1, z1, x2, y2, z2)
    var T = _beam3d_transform_matrix(R)
    var u_local = _beam3d_transform_u_global_to_local(T, u_elem_global)

    var du = u_local[6] - u_local[0]
    var N = (E * A / L) * du

    var k_local = beam3d_local_stiffness(E, A, Iy, Iz, G, J, L)
    _beam3d_add_axial_geometric_stiffness(k_local, N, L)
    return matmul(transpose(T), matmul(k_local, T))


fn beam3d_corotational_global_tangent_and_internal(
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
    u_elem_global: List[Float64],
    mut k_global_out: List[List[Float64]],
    mut f_global_out: List[Float64],
):
    var dx0 = x2 - x1
    var dy0 = y2 - y1
    var dz0 = z2 - z1
    var L0 = sqrt(dx0 * dx0 + dy0 * dy0 + dz0 * dz0)
    if L0 == 0.0:
        abort("zero-length element")

    var x1_def = x1 + u_elem_global[0]
    var y1_def = y1 + u_elem_global[1]
    var z1_def = z1 + u_elem_global[2]
    var x2_def = x2 + u_elem_global[6]
    var y2_def = y2 + u_elem_global[7]
    var z2_def = z2 + u_elem_global[8]
    var dx = x2_def - x1_def
    var dy = y2_def - y1_def
    var dz = z2_def - z1_def
    var L_def = sqrt(dx * dx + dy * dy + dz * dz)
    if L_def == 0.0:
        abort("zero-length element")

    var R_def = _beam3d_rotation(x1_def, y1_def, z1_def, x2_def, y2_def, z2_def)
    var T_def = _beam3d_transform_matrix(R_def)
    var u_local = _beam3d_transform_u_global_to_local(T_def, u_elem_global)
    var du = u_local[6] - u_local[0]
    var N = (E * A / L0) * du

    var k_local = beam3d_local_stiffness(E, A, Iy, Iz, G, J, L0)
    _beam3d_add_axial_geometric_stiffness(k_local, N, L0)
    var k_global = matmul(transpose(T_def), matmul(k_local, T_def))

    f_global_out.resize(12, 0.0)
    for a in range(12):
        var sum = 0.0
        for b in range(12):
            sum += k_global[a][b] * u_elem_global[b]
        f_global_out[a] = sum
    k_global_out = k_global^


fn beam3d_global_tangent_and_internal(
    geom_transf: String,
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
    u_elem_global: List[Float64],
    mut k_global_out: List[List[Float64]],
    mut f_global_out: List[Float64],
):
    if geom_transf == "Linear":
        var k_global = beam3d_global_stiffness(
            E,
            A,
            Iy,
            Iz,
            G,
            J,
            x1,
            y1,
            z1,
            x2,
            y2,
            z2,
        )
        f_global_out.resize(12, 0.0)
        for a in range(12):
            var sum = 0.0
            for b in range(12):
                sum += k_global[a][b] * u_elem_global[b]
            f_global_out[a] = sum
        k_global_out = k_global^
        return

    if geom_transf == "PDelta":
        var k_global = beam3d_pdelta_global_stiffness(
            E,
            A,
            Iy,
            Iz,
            G,
            J,
            x1,
            y1,
            z1,
            x2,
            y2,
            z2,
            u_elem_global,
        )
        f_global_out.resize(12, 0.0)
        for a in range(12):
            var sum = 0.0
            for b in range(12):
                sum += k_global[a][b] * u_elem_global[b]
            f_global_out[a] = sum
        k_global_out = k_global^
        return

    if geom_transf == "Corotational":
        beam3d_corotational_global_tangent_and_internal(
            E,
            A,
            Iy,
            Iz,
            G,
            J,
            x1,
            y1,
            z1,
            x2,
            y2,
            z2,
            u_elem_global,
            k_global_out,
            f_global_out,
        )
        return

    abort("unsupported geomTransf: " + geom_transf)
