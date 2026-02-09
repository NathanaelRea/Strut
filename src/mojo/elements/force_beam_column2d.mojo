from collections import List
from math import hypot
from os import abort

from elements.utils import _matvec, _zero_matrix
from linalg import matmul, transpose
from materials import UniMaterialDef, UniMaterialState, uniaxial_set_trial_strain
from sections import FiberCell, FiberSection2dDef, FiberSection2dResponse


fn _beam2d_transform(c: Float64, s: Float64) -> List[List[Float64]]:
    return [
        [c, s, 0.0, 0.0, 0.0, 0.0],
        [-s, c, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, c, s, 0.0],
        [0.0, 0.0, 0.0, -s, c, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ]


fn _lobatto_xi_weight(num_int_pts: Int, ip: Int) -> (Float64, Float64):
    if num_int_pts == 3:
        if ip == 0:
            return (0.0, 1.0 / 6.0)
        if ip == 1:
            return (0.5, 2.0 / 3.0)
        return (1.0, 1.0 / 6.0)
    if num_int_pts == 5:
        if ip == 0:
            return (0.0, 0.05)
        if ip == 1:
            return (0.1726731646460114, 0.2722222222222222)
        if ip == 2:
            return (0.5, 0.35555555555555557)
        if ip == 3:
            return (0.8273268353539886, 0.2722222222222222)
        return (1.0, 0.05)
    abort("forceBeamColumn2d supports Lobatto num_int_pts=3 or 5")
    return (0.0, 0.0)


fn _fiber_section2d_set_trial_from_offset(
    sec_def: FiberSection2dDef,
    fibers: List[FiberCell],
    uniaxial_defs: List[UniMaterialDef],
    mut uniaxial_states: List[UniMaterialState],
    section_state_ids: List[Int],
    section_state_offset: Int,
    section_state_count: Int,
    eps0: Float64,
    kappa: Float64,
) -> FiberSection2dResponse:
    if section_state_count != sec_def.fiber_count:
        abort("forceBeamColumn2d fiber state count mismatch")
    if section_state_offset < 0 or section_state_offset + section_state_count > len(
        section_state_ids
    ):
        abort("forceBeamColumn2d fiber section state out of range")

    var axial_force = 0.0
    var moment_z = 0.0
    var k11 = 0.0
    var k12 = 0.0
    var k22 = 0.0

    for i in range(section_state_count):
        var cell = fibers[sec_def.fiber_offset + i]
        var y_rel = cell.y - sec_def.y_bar
        var eps = eps0 - y_rel * kappa

        var state_index = section_state_ids[section_state_offset + i]
        if state_index < 0 or state_index >= len(uniaxial_states):
            abort("forceBeamColumn2d fiber state index out of range")
        if cell.def_index < 0 or cell.def_index >= len(uniaxial_defs):
            abort("forceBeamColumn2d fiber material definition out of range")
        var mat_def = uniaxial_defs[cell.def_index]
        var state = uniaxial_states[state_index]
        uniaxial_set_trial_strain(mat_def, state, eps)
        uniaxial_states[state_index] = state

        var area = cell.area
        var fs = state.sig_t * area
        var ks = state.tangent_t * area
        axial_force += fs
        moment_z += -fs * y_rel
        k11 += ks
        k12 += -ks * y_rel
        k22 += ks * y_rel * y_rel

    return FiberSection2dResponse(axial_force, moment_z, k11, k12, k22)


fn force_beam_column2d_global_tangent_and_internal(
    x1: Float64,
    y1: Float64,
    x2: Float64,
    y2: Float64,
    u_elem_global: List[Float64],
    sec_def: FiberSection2dDef,
    fibers: List[FiberCell],
    uniaxial_defs: List[UniMaterialDef],
    mut uniaxial_states: List[UniMaterialState],
    elem_state_ids: List[Int],
    elem_state_offset: Int,
    elem_state_count: Int,
    num_int_pts: Int,
    mut k_global_out: List[List[Float64]],
    mut f_global_out: List[Float64],
):
    var dx = x2 - x1
    var dy = y2 - y1
    var L = hypot(dx, dy)
    if L == 0.0:
        abort("zero-length element")
    var c = dx / L
    var s = dy / L
    var T = _beam2d_transform(c, s)
    var u_local = _matvec(T, u_elem_global)

    var fibers_per_section = sec_def.fiber_count
    var required_state_count = num_int_pts * fibers_per_section
    if elem_state_count != required_state_count:
        abort("forceBeamColumn2d element state count mismatch")

    var b_axial: List[Float64] = [-1.0 / L, 0.0, 0.0, 1.0 / L, 0.0, 0.0]
    var k_local = _zero_matrix(6, 6)
    var f_local: List[Float64] = []
    f_local.resize(6, 0.0)

    for ip in range(num_int_pts):
        var xi_weight = _lobatto_xi_weight(num_int_pts, ip)
        var xi = xi_weight[0]
        var weight = xi_weight[1]
        var b_kappa: List[Float64] = [
            0.0,
            (-6.0 + 12.0 * xi) / (L * L),
            (-4.0 + 6.0 * xi) / L,
            0.0,
            (6.0 - 12.0 * xi) / (L * L),
            (-2.0 + 6.0 * xi) / L,
        ]
        var eps0 = 0.0
        var kappa = 0.0
        for a in range(6):
            eps0 += b_axial[a] * u_local[a]
            kappa += b_kappa[a] * u_local[a]

        var ip_state_offset = elem_state_offset + ip * fibers_per_section
        var resp = _fiber_section2d_set_trial_from_offset(
            sec_def,
            fibers,
            uniaxial_defs,
            uniaxial_states,
            elem_state_ids,
            ip_state_offset,
            fibers_per_section,
            eps0,
            kappa,
        )
        var wL = weight * L

        for a in range(6):
            var Ba_n = b_axial[a]
            var Ba_m = b_kappa[a]
            f_local[a] += wL * (Ba_n * resp.axial_force + Ba_m * resp.moment_z)
            for b in range(6):
                var Bb_n = b_axial[b]
                var Bb_m = b_kappa[b]
                var dN = resp.k11 * Bb_n + resp.k12 * Bb_m
                var dM = resp.k12 * Bb_n + resp.k22 * Bb_m
                k_local[a][b] += wL * (Ba_n * dN + Ba_m * dM)

    var k_global = matmul(transpose(T), matmul(k_local, T))
    f_global_out.resize(6, 0.0)
    for i in range(6):
        var sum = 0.0
        for j in range(6):
            sum += T[j][i] * f_local[j]
        f_global_out[i] = sum
    k_global_out = k_global^
