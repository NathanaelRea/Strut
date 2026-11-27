from collections import InlineArray, List
from math import atan2, hypot, sqrt
from os import abort

from solver.run_case.input_types import ElementLoadInput
from tag_types import ElementLoadTypeTag

from elements.utils import _zero_matrix


fn _beam2d_transform_u_global_to_local(
    c: Float64, s: Float64, u_global: List[Float64]
) -> List[Float64]:
    var u_local: List[Float64] = []
    u_local.resize(6, 0.0)
    u_local[0] = c * u_global[0] + s * u_global[1]
    u_local[1] = -s * u_global[0] + c * u_global[1]
    u_local[2] = u_global[2]
    u_local[3] = c * u_global[3] + s * u_global[4]
    u_local[4] = -s * u_global[3] + c * u_global[4]
    u_local[5] = u_global[5]
    return u_local^


fn _beam2d_transform_force_local_to_global_array(
    c: Float64, s: Float64, f_local: InlineArray[Float64, 6]
) -> InlineArray[Float64, 6]:
    var f_global = InlineArray[Float64, 6](fill=0.0)
    f_global[0] = c * f_local[0] - s * f_local[1]
    f_global[1] = s * f_local[0] + c * f_local[1]
    f_global[2] = f_local[2]
    f_global[3] = c * f_local[3] - s * f_local[4]
    f_global[4] = s * f_local[3] + c * f_local[4]
    f_global[5] = f_local[5]
    return f_global


fn _beam2d_transform_force_local_to_global(
    c: Float64, s: Float64, f_local: List[Float64]
) -> List[Float64]:
    var f_global: List[Float64] = []
    f_global.resize(6, 0.0)
    f_global[0] = c * f_local[0] - s * f_local[1]
    f_global[1] = s * f_local[0] + c * f_local[1]
    f_global[2] = f_local[2]
    f_global[3] = c * f_local[3] - s * f_local[4]
    f_global[4] = s * f_local[3] + c * f_local[4]
    f_global[5] = f_local[5]
    return f_global^


# The element matrices below are built in fixed-size stack arrays. A nested
# List[List[Float64]] costs about ten heap allocations per 6x6 matrix, which is most
# of a load step for a model with a handful of elements.


fn _beam2d_local_stiffness_array(
    E: Float64, A: Float64, I: Float64, L: Float64
) -> InlineArray[Float64, 36]:
    var k = InlineArray[Float64, 36](fill=0.0)

    var EA_L = E * A / L
    var EI = E * I
    var L2 = L * L
    var L3 = L2 * L

    k[0] = EA_L
    k[3] = -EA_L
    k[18] = -EA_L
    k[21] = EA_L

    k[7] = 12.0 * EI / L3
    k[8] = 6.0 * EI / L2
    k[10] = -12.0 * EI / L3
    k[11] = 6.0 * EI / L2

    k[13] = 6.0 * EI / L2
    k[14] = 4.0 * EI / L
    k[16] = -6.0 * EI / L2
    k[17] = 2.0 * EI / L

    k[25] = -12.0 * EI / L3
    k[26] = -6.0 * EI / L2
    k[28] = 12.0 * EI / L3
    k[29] = -6.0 * EI / L2

    k[31] = 6.0 * EI / L2
    k[32] = 2.0 * EI / L
    k[34] = -6.0 * EI / L2
    k[35] = 4.0 * EI / L

    return k


fn _beam2d_rotate_stiffness_array(
    c: Float64, s: Float64, k_local: InlineArray[Float64, 36]
) -> InlineArray[Float64, 36]:
    var tmp = InlineArray[Float64, 36](fill=0.0)
    for i in range(6):
        var row = i * 6
        var r0 = k_local[row]
        var r1 = k_local[row + 1]
        var r3 = k_local[row + 3]
        var r4 = k_local[row + 4]
        tmp[row] = c * r0 - s * r1
        tmp[row + 1] = s * r0 + c * r1
        tmp[row + 2] = k_local[row + 2]
        tmp[row + 3] = c * r3 - s * r4
        tmp[row + 4] = s * r3 + c * r4
        tmp[row + 5] = k_local[row + 5]

    var k_global = InlineArray[Float64, 36](fill=0.0)
    for j in range(6):
        var t0 = tmp[j]
        var t1 = tmp[6 + j]
        var t3 = tmp[18 + j]
        var t4 = tmp[24 + j]
        k_global[j] = c * t0 - s * t1
        k_global[6 + j] = s * t0 + c * t1
        k_global[12 + j] = tmp[12 + j]
        k_global[18 + j] = c * t3 - s * t4
        k_global[24 + j] = s * t3 + c * t4
        k_global[30 + j] = tmp[30 + j]

    return k_global


fn _beam2d_matrix_from_array(k: InlineArray[Float64, 36]) -> List[List[Float64]]:
    var out = List[List[Float64]](capacity=6)
    for i in range(6):
        var row = List[Float64](capacity=6)
        for j in range(6):
            row.append(k[i * 6 + j])
        out.append(row^)
    return out^


fn _beam2d_transform_stiffness_local_to_global(
    c: Float64, s: Float64, k_local: List[List[Float64]]
) -> List[List[Float64]]:
    var flat = InlineArray[Float64, 36](fill=0.0)
    for i in range(6):
        for j in range(6):
            flat[i * 6 + j] = k_local[i][j]
    return _beam2d_matrix_from_array(_beam2d_rotate_stiffness_array(c, s, flat))


fn _beam2d_global_stiffness_array(
    E: Float64,
    A: Float64,
    I: Float64,
    x1: Float64,
    y1: Float64,
    x2: Float64,
    y2: Float64,
) -> InlineArray[Float64, 36]:
    var dx = x2 - x1
    var dy = y2 - y1
    var L = hypot(dx, dy)
    if L == 0.0:
        abort("zero-length element")
    var c = dx / L
    var s = dy / L

    return _beam2d_rotate_stiffness_array(c, s, _beam2d_local_stiffness_array(E, A, I, L))


fn _beam2d_pdelta_global_stiffness_array(
    E: Float64,
    A: Float64,
    I: Float64,
    x1: Float64,
    y1: Float64,
    x2: Float64,
    y2: Float64,
    u_elem_global: List[Float64],
) -> InlineArray[Float64, 36]:
    var dx = x2 - x1
    var dy = y2 - y1
    var L = hypot(dx, dy)
    if L == 0.0:
        abort("zero-length element")
    var c = dx / L
    var s = dy / L

    var u_local_0 = c * u_elem_global[0] + s * u_elem_global[1]
    var u_local_3 = c * u_elem_global[3] + s * u_elem_global[4]
    var du = u_local_3 - u_local_0
    var N = (E * A / L) * du
    var N_over_L = N / L

    var k_local = _beam2d_local_stiffness_array(E, A, I, L)
    k_local[7] += N_over_L
    k_local[28] += N_over_L
    k_local[10] -= N_over_L
    k_local[25] -= N_over_L

    return _beam2d_rotate_stiffness_array(c, s, k_local)


fn beam_local_stiffness(E: Float64, A: Float64, I: Float64, L: Float64) -> List[List[Float64]]:
    return _beam2d_matrix_from_array(_beam2d_local_stiffness_array(E, A, I, L))


fn beam_global_stiffness(
    E: Float64,
    A: Float64,
    I: Float64,
    x1: Float64,
    y1: Float64,
    x2: Float64,
    y2: Float64,
) -> List[List[Float64]]:
    return _beam2d_matrix_from_array(_beam2d_global_stiffness_array(E, A, I, x1, y1, x2, y2))


fn beam2d_global_stiffness_flat(
    E: Float64,
    A: Float64,
    I: Float64,
    x1: Float64,
    y1: Float64,
    x2: Float64,
    y2: Float64,
    mut k_flat: List[Float64],
):
    """Row-major 6x6 global stiffness into `k_flat[0:36]`, without allocating."""
    var k = _beam2d_global_stiffness_array(E, A, I, x1, y1, x2, y2)
    for i in range(36):
        k_flat[i] = k[i]


fn beam2d_pdelta_global_stiffness_flat(
    E: Float64,
    A: Float64,
    I: Float64,
    x1: Float64,
    y1: Float64,
    x2: Float64,
    y2: Float64,
    u_elem_global: List[Float64],
    mut k_flat: List[Float64],
):
    """Row-major 6x6 PDelta global stiffness into `k_flat[0:36]`, without allocating."""
    var k = _beam2d_pdelta_global_stiffness_array(E, A, I, x1, y1, x2, y2, u_elem_global)
    for i in range(36):
        k_flat[i] = k[i]


fn beam_uniform_load_global_2d(
    x1: Float64,
    y1: Float64,
    x2: Float64,
    y2: Float64,
    wy: Float64,
    wx: Float64,
) -> List[Float64]:
    var dx = x2 - x1
    var dy = y2 - y1
    var L = hypot(dx, dy)
    if L == 0.0:
        abort("zero-length element")
    var c = dx / L
    var s = dy / L

    var f_local: List[Float64] = [
        wx * L / 2.0,
        wy * L / 2.0,
        wy * L * L / 12.0,
        wx * L / 2.0,
        wy * L / 2.0,
        -wy * L * L / 12.0,
    ]

    return _beam2d_transform_force_local_to_global(c, s, f_local)


fn beam_uniform_load_global(
    x1: Float64,
    y1: Float64,
    x2: Float64,
    y2: Float64,
    w: Float64,
) -> List[Float64]:
    return beam_uniform_load_global_2d(x1, y1, x2, y2, w, 0.0)


fn beam2d_element_load_global_array(
    element_loads: List[ElementLoadInput],
    elem_load_offsets: List[Int],
    elem_load_pool: List[Int],
    elem_index: Int,
    load_scale: Float64,
    x1: Float64,
    y1: Float64,
    x2: Float64,
    y2: Float64,
) -> InlineArray[Float64, 6]:
    var f_local = InlineArray[Float64, 6](fill=0.0)
    if load_scale == 0.0:
        return _beam2d_transform_force_local_to_global_array(1.0, 0.0, f_local)

    var dx = x2 - x1
    var dy = y2 - y1
    var L = hypot(dx, dy)
    if L == 0.0:
        abort("zero-length element")
    var c = dx / L
    var s = dy / L

    for slot in range(elem_load_offsets[elem_index], elem_load_offsets[elem_index + 1]):
        var load = element_loads[elem_load_pool[slot]]
        if load.type_tag == ElementLoadTypeTag.BeamUniform:
            var wx = load.wx * load_scale
            var wy = load.wy * load_scale
            f_local[0] += wx * L / 2.0
            f_local[1] += wy * L / 2.0
            f_local[2] += wy * L * L / 12.0
            f_local[3] += wx * L / 2.0
            f_local[4] += wy * L / 2.0
            f_local[5] -= wy * L * L / 12.0
        elif load.type_tag == ElementLoadTypeTag.BeamPoint:
            var xi = load.x
            if xi < 0.0 or xi > 1.0:
                continue
            var px = load.px * load_scale
            var py = load.py * load_scale
            var xi2 = xi * xi
            var xi3 = xi2 * xi
            f_local[0] += px * (1.0 - xi)
            f_local[3] += px * xi
            f_local[1] += py * (1.0 - 3.0 * xi2 + 2.0 * xi3)
            f_local[2] += py * L * (xi - 2.0 * xi2 + xi3)
            f_local[4] += py * (3.0 * xi2 - 2.0 * xi3)
            f_local[5] += py * L * (-xi2 + xi3)

    return _beam2d_transform_force_local_to_global_array(c, s, f_local)


fn beam2d_element_load_global(
    element_loads: List[ElementLoadInput],
    elem_load_offsets: List[Int],
    elem_load_pool: List[Int],
    elem_index: Int,
    load_scale: Float64,
    x1: Float64,
    y1: Float64,
    x2: Float64,
    y2: Float64,
) -> List[Float64]:
    var f = beam2d_element_load_global_array(
        element_loads, elem_load_offsets, elem_load_pool, elem_index, load_scale, x1, y1, x2, y2
    )
    var out = List[Float64](capacity=6)
    for i in range(6):
        out.append(f[i])
    return out^


fn beam2d_pdelta_global_stiffness(
    E: Float64,
    A: Float64,
    I: Float64,
    x1: Float64,
    y1: Float64,
    x2: Float64,
    y2: Float64,
    u_elem_global: List[Float64],
) -> List[List[Float64]]:
    return _beam2d_matrix_from_array(
        _beam2d_pdelta_global_stiffness_array(E, A, I, x1, y1, x2, y2, u_elem_global)
    )


fn beam2d_corotational_global_tangent_and_internal(
    E: Float64,
    A: Float64,
    I: Float64,
    x1: Float64,
    y1: Float64,
    x2: Float64,
    y2: Float64,
    u_elem_global: List[Float64],
    mut k_global: List[List[Float64]],
    mut f_global: List[Float64],
):
    var dx = x2 - x1
    var dy = y2 - y1
    var L = hypot(dx, dy)
    if L == 0.0:
        abort("zero-length element")
    var c = dx / L
    var s = dy / L

    var u_local = _beam2d_transform_u_global_to_local(c, s, u_elem_global)
    var dulx = u_local[3] - u_local[0]
    var duly = u_local[4] - u_local[1]
    var Lx = L + dulx
    var Ly = duly
    var Ln = sqrt(Lx * Lx + Ly * Ly)
    if Ln == 0.0:
        abort("zero-length element")

    var cos_alpha = Lx / Ln
    var sin_alpha = Ly / Ln
    var alpha = atan2(Ly, Lx)

    var ub0 = Ln - L
    var ub1 = u_local[2] - alpha
    var ub2 = u_local[5] - alpha

    var EA_L = E * A / L
    var EI_L = E * I / L
    var kb00 = EA_L
    var kb11 = 4.0 * EI_L
    var kb12 = 2.0 * EI_L
    var kb22 = 4.0 * EI_L

    var pb0 = kb00 * ub0
    var pb1 = kb11 * ub1 + kb12 * ub2
    var pb2 = kb12 * ub1 + kb22 * ub2

    var Tbl: List[List[Float64]] = [
        [-cos_alpha, -sin_alpha, 0.0, cos_alpha, sin_alpha, 0.0],
        [
            -sin_alpha / Ln,
            cos_alpha / Ln,
            1.0,
            sin_alpha / Ln,
            -cos_alpha / Ln,
            0.0,
        ],
        [
            -sin_alpha / Ln,
            cos_alpha / Ln,
            0.0,
            sin_alpha / Ln,
            -cos_alpha / Ln,
            1.0,
        ],
    ]

    var kl = _zero_matrix(6, 6)
    for i in range(6):
        var t0i = Tbl[0][i]
        var t1i = Tbl[1][i]
        var t2i = Tbl[2][i]
        for j in range(6):
            var t0j = Tbl[0][j]
            var t1j = Tbl[1][j]
            var t2j = Tbl[2][j]
            kl[i][j] = (
                kb00 * t0i * t0j
                + kb11 * t1i * t1j
                + kb12 * t1i * t2j
                + kb12 * t2i * t1j
                + kb22 * t2i * t2j
            )

    var s2 = sin_alpha * sin_alpha
    var c2 = cos_alpha * cos_alpha
    var cs = sin_alpha * cos_alpha

    var kg0 = _zero_matrix(6, 6)
    kg0[0][0] = s2
    kg0[3][3] = s2
    kg0[0][1] = -cs
    kg0[3][4] = -cs
    kg0[1][0] = -cs
    kg0[4][3] = -cs
    kg0[1][1] = c2
    kg0[4][4] = c2

    kg0[0][3] = -s2
    kg0[3][0] = -s2
    kg0[0][4] = cs
    kg0[3][1] = cs
    kg0[1][3] = cs
    kg0[4][0] = cs
    kg0[1][4] = -c2
    kg0[4][1] = -c2

    var kg12 = _zero_matrix(6, 6)
    kg12[0][0] = -2.0 * cs
    kg12[3][3] = -2.0 * cs
    kg12[0][1] = c2 - s2
    kg12[3][4] = c2 - s2
    kg12[1][0] = c2 - s2
    kg12[4][3] = c2 - s2
    kg12[1][1] = 2.0 * cs
    kg12[4][4] = 2.0 * cs

    kg12[0][3] = 2.0 * cs
    kg12[3][0] = 2.0 * cs
    kg12[0][4] = -c2 + s2
    kg12[3][1] = -c2 + s2
    kg12[1][3] = -c2 + s2
    kg12[4][0] = -c2 + s2
    kg12[1][4] = -2.0 * cs
    kg12[4][1] = -2.0 * cs

    var scale0 = pb0 / Ln
    var scale12 = (pb1 + pb2) / (Ln * Ln)
    for i in range(6):
        for j in range(6):
            kl[i][j] += kg0[i][j] * scale0 + kg12[i][j] * scale12

    var pl: List[Float64] = []
    pl.resize(6, 0.0)
    for i in range(6):
        pl[i] = Tbl[0][i] * pb0 + Tbl[1][i] * pb1 + Tbl[2][i] * pb2

    k_global = _beam2d_transform_stiffness_local_to_global(c, s, kl)
    f_global = _beam2d_transform_force_local_to_global(c, s, pl)


fn beam2d_corotational_global_stiffness(
    E: Float64,
    A: Float64,
    I: Float64,
    x1: Float64,
    y1: Float64,
    x2: Float64,
    y2: Float64,
    u_elem_global: List[Float64],
) -> List[List[Float64]]:
    var k_global: List[List[Float64]] = []
    var f_dummy: List[Float64] = []
    beam2d_corotational_global_tangent_and_internal(
        E,
        A,
        I,
        x1,
        y1,
        x2,
        y2,
        u_elem_global,
        k_global,
        f_dummy,
    )
    return k_global^


fn beam2d_corotational_global_internal_force(
    E: Float64,
    A: Float64,
    I: Float64,
    x1: Float64,
    y1: Float64,
    x2: Float64,
    y2: Float64,
    u_elem_global: List[Float64],
) -> List[Float64]:
    var k_dummy: List[List[Float64]] = []
    var f_global: List[Float64] = []
    beam2d_corotational_global_tangent_and_internal(
        E,
        A,
        I,
        x1,
        y1,
        x2,
        y2,
        u_elem_global,
        k_dummy,
        f_global,
    )
    return f_global^
