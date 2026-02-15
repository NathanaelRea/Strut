from collections import List
from math import hypot
from os import abort

from materials import UniMaterialDef, UniMaterialState, uniaxial_set_trial_strain
from sections import FiberCell, FiberSection2dDef, FiberSection2dResponse


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


fn _invert_3x3_values(
    a: Float64,
    b: Float64,
    c: Float64,
    d: Float64,
    e: Float64,
    f: Float64,
    g: Float64,
    h: Float64,
    i: Float64,
) -> (
    Bool,
    Float64,
    Float64,
    Float64,
    Float64,
    Float64,
    Float64,
    Float64,
    Float64,
    Float64,
):

    var co00 = e * i - f * h
    var co01 = -(d * i - f * g)
    var co02 = d * h - e * g
    var co10 = -(b * i - c * h)
    var co11 = a * i - c * g
    var co12 = -(a * h - b * g)
    var co20 = b * f - c * e
    var co21 = -(a * f - c * d)
    var co22 = a * e - b * d

    var det = a * co00 + b * co01 + c * co02
    if abs(det) <= 1.0e-40:
        return (False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    var inv_det = 1.0 / det
    return (
        True,
        co00 * inv_det,
        co10 * inv_det,
        co20 * inv_det,
        co01 * inv_det,
        co11 * inv_det,
        co21 * inv_det,
        co02 * inv_det,
        co12 * inv_det,
        co22 * inv_det,
    )


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


fn _fiber_section2d_solve_for_force(
    sec_def: FiberSection2dDef,
    fibers: List[FiberCell],
    uniaxial_defs: List[UniMaterialDef],
    mut uniaxial_states: List[UniMaterialState],
    section_state_ids: List[Int],
    section_state_offset: Int,
    section_state_count: Int,
    axial_force_target: Float64,
    moment_target: Float64,
    mut eps0_guess: Float64,
    mut kappa_guess: Float64,
    max_iters: Int,
    tol: Float64,
) -> FiberSection2dResponse:
    var iter = 0
    var resp = FiberSection2dResponse(0.0, 0.0, 0.0, 0.0, 0.0)
    while iter < max_iters:
        resp = _fiber_section2d_set_trial_from_offset(
            sec_def,
            fibers,
            uniaxial_defs,
            uniaxial_states,
            section_state_ids,
            section_state_offset,
            section_state_count,
            eps0_guess,
            kappa_guess,
        )

        var r_axial = axial_force_target - resp.axial_force
        var r_moment = moment_target - resp.moment_z
        if abs(r_axial) <= tol and abs(r_moment) <= tol:
            return resp

        var det = resp.k11 * resp.k22 - resp.k12 * resp.k12
        if abs(det) <= 1.0e-40:
            return resp

        var inv_det = 1.0 / det
        var de0 = inv_det * (resp.k22 * r_axial - resp.k12 * r_moment)
        var dkappa = inv_det * (-resp.k12 * r_axial + resp.k11 * r_moment)
        eps0_guess += de0
        kappa_guess += dkappa
        iter += 1
    return resp


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
    mut force_basic_q_state: List[Float64],
    force_basic_q_offset: Int,
    force_basic_q_count: Int,
    mut k_global_out: List[List[Float64]],
    mut f_global_out: List[Float64],
):
    var dx = x2 - x1
    var dy = y2 - y1
    var L = hypot(dx, dy)
    if L == 0.0:
        abort("zero-length element")

    var fibers_per_section = sec_def.fiber_count
    var required_state_count = num_int_pts * fibers_per_section
    if elem_state_count != required_state_count:
        abort("forceBeamColumn2d element state count mismatch")

    var predictor_state_count = 3 + 2 * num_int_pts
    if force_basic_q_count < predictor_state_count:
        abort("forceBeamColumn2d basic force state count mismatch")
    if (
        force_basic_q_offset < 0
        or force_basic_q_offset + predictor_state_count > len(force_basic_q_state)
    ):
        abort("forceBeamColumn2d basic force state out of range")

    var c = dx / L
    var s = dy / L
    var T: List[List[Float64]] = []
    for _ in range(6):
        var row_t: List[Float64] = []
        row_t.resize(6, 0.0)
        T.append(row_t^)
    T[0][0] = c
    T[0][1] = s
    T[1][0] = -s
    T[1][1] = c
    T[2][2] = 1.0
    T[3][3] = c
    T[3][4] = s
    T[4][3] = -s
    T[4][4] = c
    T[5][5] = 1.0

    var u_local: List[Float64] = []
    u_local.resize(6, 0.0)
    for i in range(6):
        var sum = 0.0
        for j in range(6):
            sum += T[i][j] * u_elem_global[j]
        u_local[i] = sum

    var chord_rotation = (u_local[4] - u_local[1]) / L
    var v_basic_0 = u_local[3] - u_local[0]
    var v_basic_1 = u_local[2] - chord_rotation
    var v_basic_2 = u_local[5] - chord_rotation

    var q0 = force_basic_q_state[force_basic_q_offset]
    var q1 = force_basic_q_state[force_basic_q_offset + 1]
    var q2 = force_basic_q_state[force_basic_q_offset + 2]
    var eps0_offset = force_basic_q_offset + 3
    var kappa_offset = eps0_offset + num_int_pts

    var elem_tol = 1.0e-10
    var section_tol = 1.0e-8
    var max_elem_iters = 40
    var max_section_iters = 60

    var k00 = 0.0
    var k01 = 0.0
    var k02 = 0.0
    var k10 = 0.0
    var k11 = 0.0
    var k12 = 0.0
    var k20 = 0.0
    var k21 = 0.0
    var k22 = 0.0
    var converged = False

    for _ in range(max_elem_iters):
        var f00 = 0.0
        var f01 = 0.0
        var f02 = 0.0
        var f10 = 0.0
        var f11 = 0.0
        var f12 = 0.0
        var f20 = 0.0
        var f21 = 0.0
        var f22 = 0.0

        var v_from_0 = 0.0
        var v_from_1 = 0.0
        var v_from_2 = 0.0

        for ip in range(num_int_pts):
            var xi_weight = _lobatto_xi_weight(num_int_pts, ip)
            var xi = xi_weight[0]
            var weight = xi_weight[1]
            var wL = weight * L

            var b_mi = xi - 1.0
            var b_mj = xi
            var axial_target = q0
            var moment_target = b_mi * q1 + b_mj * q2

            var ip_state_offset = elem_state_offset + ip * fibers_per_section
            var eps0 = force_basic_q_state[eps0_offset + ip]
            var kappa = force_basic_q_state[kappa_offset + ip]
            var resp = _fiber_section2d_solve_for_force(
                sec_def,
                fibers,
                uniaxial_defs,
                uniaxial_states,
                elem_state_ids,
                ip_state_offset,
                fibers_per_section,
                axial_target,
                moment_target,
                eps0,
                kappa,
                max_section_iters,
                section_tol,
            )
            force_basic_q_state[eps0_offset + ip] = eps0
            force_basic_q_state[kappa_offset + ip] = kappa

            var det_sec = resp.k11 * resp.k22 - resp.k12 * resp.k12
            if abs(det_sec) <= 1.0e-40:
                abort("forceBeamColumn2d singular section flexibility")
            var inv_det_sec = 1.0 / det_sec
            var sec_f00 = resp.k22 * inv_det_sec
            var sec_f01 = -resp.k12 * inv_det_sec
            var sec_f10 = sec_f01
            var sec_f11 = resp.k11 * inv_det_sec

            v_from_0 += wL * eps0
            v_from_1 += wL * b_mi * kappa
            v_from_2 += wL * b_mj * kappa

            f00 += wL * sec_f00
            f01 += wL * sec_f01 * b_mi
            f02 += wL * sec_f01 * b_mj
            f10 += wL * b_mi * sec_f10
            f11 += wL * b_mi * sec_f11 * b_mi
            f12 += wL * b_mi * sec_f11 * b_mj
            f20 += wL * b_mj * sec_f10
            f21 += wL * b_mj * sec_f11 * b_mi
            f22 += wL * b_mj * sec_f11 * b_mj

        var k_inv = _invert_3x3_values(
            f00,
            f01,
            f02,
            f10,
            f11,
            f12,
            f20,
            f21,
            f22,
        )
        if not k_inv[0]:
            abort("forceBeamColumn2d singular element flexibility")
        k00 = k_inv[1]
        k01 = k_inv[2]
        k02 = k_inv[3]
        k10 = k_inv[4]
        k11 = k_inv[5]
        k12 = k_inv[6]
        k20 = k_inv[7]
        k21 = k_inv[8]
        k22 = k_inv[9]

        var residual_0 = v_basic_0 - v_from_0
        var residual_1 = v_basic_1 - v_from_1
        var residual_2 = v_basic_2 - v_from_2

        var dq0 = k00 * residual_0 + k01 * residual_1 + k02 * residual_2
        var dq1 = k10 * residual_0 + k11 * residual_1 + k12 * residual_2
        var dq2 = k20 * residual_0 + k21 * residual_1 + k22 * residual_2

        q0 += dq0
        q1 += dq1
        q2 += dq2

        var max_residual = abs(residual_0)
        var abs_residual_1 = abs(residual_1)
        if abs_residual_1 > max_residual:
            max_residual = abs_residual_1
        var abs_residual_2 = abs(residual_2)
        if abs_residual_2 > max_residual:
            max_residual = abs_residual_2

        var work_norm = abs(residual_0 * dq0 + residual_1 * dq1 + residual_2 * dq2)
        if max_residual <= elem_tol or work_norm <= elem_tol:
            converged = True
            break

    if not converged:
        abort("forceBeamColumn2d element compatibility did not converge")

    force_basic_q_state[force_basic_q_offset] = q0
    force_basic_q_state[force_basic_q_offset + 1] = q1
    force_basic_q_state[force_basic_q_offset + 2] = q2

    var inv_L = 1.0 / L
    var a_matrix: List[List[Float64]] = [
        [-1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, inv_L, 1.0, 0.0, -inv_L, 0.0],
        [0.0, inv_L, 0.0, 0.0, -inv_L, 1.0],
    ]

    var k_local: List[List[Float64]] = []
    for _ in range(6):
        var row_k_local: List[Float64] = []
        row_k_local.resize(6, 0.0)
        k_local.append(row_k_local^)
    for a in range(6):
        for b in range(6):
            var a0 = a_matrix[0][a]
            var a1 = a_matrix[1][a]
            var a2 = a_matrix[2][a]
            var b0 = a_matrix[0][b]
            var b1 = a_matrix[1][b]
            var b2 = a_matrix[2][b]
            k_local[a][b] = (
                a0 * (k00 * b0 + k01 * b1 + k02 * b2)
                + a1 * (k10 * b0 + k11 * b1 + k12 * b2)
                + a2 * (k20 * b0 + k21 * b1 + k22 * b2)
            )

    var f_local: List[Float64] = []
    f_local.resize(6, 0.0)
    f_local[0] = -q0
    f_local[1] = (q1 + q2) * inv_L
    f_local[2] = q1
    f_local[3] = q0
    f_local[4] = -(q1 + q2) * inv_L
    f_local[5] = q2

    var k_temp: List[List[Float64]] = []
    var k_global: List[List[Float64]] = []
    for _ in range(6):
        var row_k_temp: List[Float64] = []
        row_k_temp.resize(6, 0.0)
        k_temp.append(row_k_temp^)
        var row_k_global: List[Float64] = []
        row_k_global.resize(6, 0.0)
        k_global.append(row_k_global^)

    for i in range(6):
        for j in range(6):
            var sum_temp = 0.0
            for k in range(6):
                sum_temp += k_local[i][k] * T[k][j]
            k_temp[i][j] = sum_temp
    for i in range(6):
        for j in range(6):
            var sum_global = 0.0
            for k in range(6):
                sum_global += T[k][i] * k_temp[k][j]
            k_global[i][j] = sum_global

    f_global_out.resize(6, 0.0)
    for i in range(6):
        var sum_f = 0.0
        for j in range(6):
            sum_f += T[j][i] * f_local[j]
        f_global_out[i] = sum_f
    k_global_out = k_global^
