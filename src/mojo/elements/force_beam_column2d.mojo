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


fn _invert_2x2(
    a11: Float64,
    a12: Float64,
    a21: Float64,
    a22: Float64,
    mut inv_out: List[List[Float64]],
) -> Bool:
    var det = a11 * a22 - a12 * a21
    if abs(det) <= 1.0e-40:
        return False
    var inv_det = 1.0 / det
    inv_out[0][0] = a22 * inv_det
    inv_out[0][1] = -a12 * inv_det
    inv_out[1][0] = -a21 * inv_det
    inv_out[1][1] = a11 * inv_det
    return True


fn _invert_3x3(mat: List[List[Float64]], mut inv_out: List[List[Float64]]) -> Bool:
    var a = mat[0][0]
    var b = mat[0][1]
    var c = mat[0][2]
    var d = mat[1][0]
    var e = mat[1][1]
    var f = mat[1][2]
    var g = mat[2][0]
    var h = mat[2][1]
    var i = mat[2][2]

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
        return False

    var inv_det = 1.0 / det
    inv_out[0][0] = co00 * inv_det
    inv_out[0][1] = co10 * inv_det
    inv_out[0][2] = co20 * inv_det
    inv_out[1][0] = co01 * inv_det
    inv_out[1][1] = co11 * inv_det
    inv_out[1][2] = co21 * inv_det
    inv_out[2][0] = co02 * inv_det
    inv_out[2][1] = co12 * inv_det
    inv_out[2][2] = co22 * inv_det
    return True


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

        var sec_tangent_inv = _zero_matrix(2, 2)
        if not _invert_2x2(
            resp.k11,
            resp.k12,
            resp.k12,
            resp.k22,
            sec_tangent_inv,
        ):
            return resp

        var de0 = sec_tangent_inv[0][0] * r_axial + sec_tangent_inv[0][1] * r_moment
        var dkappa = sec_tangent_inv[1][0] * r_axial + sec_tangent_inv[1][1] * r_moment
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

    if force_basic_q_count != 3:
        abort("forceBeamColumn2d basic force state count mismatch")
    if force_basic_q_offset < 0 or force_basic_q_offset + 3 > len(force_basic_q_state):
        abort("forceBeamColumn2d basic force state out of range")

    var c = dx / L
    var s = dy / L
    var T = _beam2d_transform(c, s)
    var u_local = _matvec(T, u_elem_global)

    var chord_rotation = (u_local[4] - u_local[1]) / L
    var v_basic: List[Float64] = [
        u_local[3] - u_local[0],
        u_local[2] - chord_rotation,
        u_local[5] - chord_rotation,
    ]

    var q: List[Float64] = []
    q.resize(3, 0.0)
    q[0] = force_basic_q_state[force_basic_q_offset]
    q[1] = force_basic_q_state[force_basic_q_offset + 1]
    q[2] = force_basic_q_state[force_basic_q_offset + 2]

    var section_eps0: List[Float64] = []
    var section_kappa: List[Float64] = []
    section_eps0.resize(num_int_pts, 0.0)
    section_kappa.resize(num_int_pts, 0.0)

    var elem_tol = 1.0e-10
    var section_tol = 1.0e-8
    var max_elem_iters = 40
    var max_section_iters = 60

    var flexibility = _zero_matrix(3, 3)
    var basic_stiffness = _zero_matrix(3, 3)
    var converged = False

    for _ in range(max_elem_iters):
        for i in range(3):
            for j in range(3):
                flexibility[i][j] = 0.0

        var v_from_section: List[Float64] = []
        v_from_section.resize(3, 0.0)

        for ip in range(num_int_pts):
            var xi_weight = _lobatto_xi_weight(num_int_pts, ip)
            var xi = xi_weight[0]
            var weight = xi_weight[1]
            var wL = weight * L

            var b_mi = xi - 1.0
            var b_mj = xi
            var axial_target = q[0]
            var moment_target = b_mi * q[1] + b_mj * q[2]

            var ip_state_offset = elem_state_offset + ip * fibers_per_section
            var eps0 = section_eps0[ip]
            var kappa = section_kappa[ip]
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
            section_eps0[ip] = eps0
            section_kappa[ip] = kappa

            var sec_flex = _zero_matrix(2, 2)
            if not _invert_2x2(
                resp.k11,
                resp.k12,
                resp.k12,
                resp.k22,
                sec_flex,
            ):
                abort("forceBeamColumn2d singular section flexibility")

            v_from_section[0] += wL * eps0
            v_from_section[1] += wL * b_mi * kappa
            v_from_section[2] += wL * b_mj * kappa

            flexibility[0][0] += wL * sec_flex[0][0]
            flexibility[0][1] += wL * sec_flex[0][1] * b_mi
            flexibility[0][2] += wL * sec_flex[0][1] * b_mj
            flexibility[1][0] += wL * b_mi * sec_flex[1][0]
            flexibility[1][1] += wL * b_mi * sec_flex[1][1] * b_mi
            flexibility[1][2] += wL * b_mi * sec_flex[1][1] * b_mj
            flexibility[2][0] += wL * b_mj * sec_flex[1][0]
            flexibility[2][1] += wL * b_mj * sec_flex[1][1] * b_mi
            flexibility[2][2] += wL * b_mj * sec_flex[1][1] * b_mj

        if not _invert_3x3(flexibility, basic_stiffness):
            abort("forceBeamColumn2d singular element flexibility")

        var residual: List[Float64] = []
        residual.resize(3, 0.0)
        for i in range(3):
            residual[i] = v_basic[i] - v_from_section[i]

        var dq: List[Float64] = []
        dq.resize(3, 0.0)
        for i in range(3):
            var sum = 0.0
            for j in range(3):
                sum += basic_stiffness[i][j] * residual[j]
            dq[i] = sum

        for i in range(3):
            q[i] += dq[i]

        var max_residual = 0.0
        for i in range(3):
            var abs_residual = abs(residual[i])
            if abs_residual > max_residual:
                max_residual = abs_residual

        var work_norm = abs(
            residual[0] * dq[0] + residual[1] * dq[1] + residual[2] * dq[2]
        )
        if max_residual <= elem_tol or work_norm <= elem_tol:
            converged = True
            break

    if not converged:
        abort("forceBeamColumn2d element compatibility did not converge")

    force_basic_q_state[force_basic_q_offset] = q[0]
    force_basic_q_state[force_basic_q_offset + 1] = q[1]
    force_basic_q_state[force_basic_q_offset + 2] = q[2]

    var a_matrix: List[List[Float64]] = [
        [-1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0 / L, 1.0, 0.0, -1.0 / L, 0.0],
        [0.0, 1.0 / L, 0.0, 0.0, -1.0 / L, 1.0],
    ]

    var k_local = _zero_matrix(6, 6)
    for a in range(6):
        for b in range(6):
            var sum = 0.0
            for i in range(3):
                for j in range(3):
                    sum += a_matrix[i][a] * basic_stiffness[i][j] * a_matrix[j][b]
            k_local[a][b] = sum

    var f_local: List[Float64] = []
    f_local.resize(6, 0.0)
    f_local[0] = -q[0]
    f_local[1] = (q[1] + q[2]) / L
    f_local[2] = q[1]
    f_local[3] = q[0]
    f_local[4] = -(q[1] + q[2]) / L
    f_local[5] = q[2]

    var k_global = matmul(transpose(T), matmul(k_local, T))
    f_global_out.resize(6, 0.0)
    for i in range(6):
        var sum = 0.0
        for j in range(6):
            sum += T[j][i] * f_local[j]
        f_global_out[i] = sum
    k_global_out = k_global^
