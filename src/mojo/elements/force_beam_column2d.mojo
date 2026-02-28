from collections import List
from math import hypot
from os import abort

from elements.beam_loads import beam2d_basic_fixed_end_and_reactions, beam2d_section_load_response
from elements.beam_integration import beam_integration_rule
from elements.utils import (
    _beam2d_transform_force_local_to_global_in_place,
    _beam2d_transform_stiffness_local_to_global_in_place,
    _beam2d_transform_u_global_to_local,
    _ensure_zero_matrix,
    _ensure_zero_vector,
)
from materials import (
    UniMaterialDef,
    UniMaterialState,
    uni_mat_is_elastic,
    uni_mat_initial_tangent,
    uniaxial_set_trial_strain,
)
from solver.run_case.input_types import ElementLoadInput
from sections import FiberCell, FiberSection2dDef, FiberSection2dResponse


@always_inline
fn _beam2d_local_basic_map(col: Int, inv_L: Float64) -> (Float64, Float64, Float64):
    if col == 0:
        return (-1.0, 0.0, 0.0)
    if col == 1:
        return (0.0, inv_L, inv_L)
    if col == 2:
        return (0.0, 1.0, 0.0)
    if col == 3:
        return (1.0, 0.0, 0.0)
    if col == 4:
        return (0.0, -inv_L, -inv_L)
    return (0.0, 0.0, 1.0)


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


@always_inline
fn _max_abs3(a: Float64, b: Float64, c: Float64) -> Float64:
    var max_val = abs(a)
    var abs_b = abs(b)
    if abs_b > max_val:
        max_val = abs_b
    var abs_c = abs(c)
    if abs_c > max_val:
        max_val = abs_c
    return max_val


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
) -> (Bool, FiberSection2dResponse, Float64, Float64):
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
            return (True, resp, eps0_guess, kappa_guess)

        var det = resp.k11 * resp.k22 - resp.k12 * resp.k12
        if abs(det) <= 1.0e-40:
            return (False, resp, eps0_guess, kappa_guess)

        var inv_det = 1.0 / det
        var de0 = inv_det * (resp.k22 * r_axial - resp.k12 * r_moment)
        var dkappa = inv_det * (-resp.k12 * r_axial + resp.k11 * r_moment)
        eps0_guess += de0
        kappa_guess += dkappa
        iter += 1
    return (False, resp, eps0_guess, kappa_guess)


fn _fiber_section2d_initial_flexibility(
    sec_def: FiberSection2dDef, fibers: List[FiberCell], uniaxial_defs: List[UniMaterialDef]
) -> (Bool, Float64, Float64, Float64):
    var k11 = 0.0
    var k12 = 0.0
    var k22 = 0.0
    for i in range(sec_def.fiber_count):
        var cell = fibers[sec_def.fiber_offset + i]
        if cell.def_index < 0 or cell.def_index >= len(uniaxial_defs):
            abort("forceBeamColumn2d fiber material definition out of range")
        var tangent = uni_mat_initial_tangent(uniaxial_defs[cell.def_index]) * cell.area
        var y_rel = cell.y - sec_def.y_bar
        k11 += tangent
        k12 += -tangent * y_rel
        k22 += tangent * y_rel * y_rel

    var det = k11 * k22 - k12 * k12
    if abs(det) <= 1.0e-40:
        return (False, 0.0, 0.0, 0.0)
    var inv_det = 1.0 / det
    return (True, k22 * inv_det, -k12 * inv_det, k11 * inv_det)


fn _fiber_section2d_response_flexibility(
    resp: FiberSection2dResponse
) -> (Bool, Float64, Float64, Float64):
    var det = resp.k11 * resp.k22 - resp.k12 * resp.k12
    if abs(det) <= 1.0e-40:
        return (False, 0.0, 0.0, 0.0)
    var inv_det = 1.0 / det
    return (True, resp.k22 * inv_det, -resp.k12 * inv_det, resp.k11 * inv_det)


fn _fiber_section2d_all_materials_elastic(
    sec_def: FiberSection2dDef, fibers: List[FiberCell], uniaxial_defs: List[UniMaterialDef]
) -> Bool:
    for i in range(sec_def.fiber_count):
        var cell = fibers[sec_def.fiber_offset + i]
        if cell.def_index < 0 or cell.def_index >= len(uniaxial_defs):
            abort("forceBeamColumn2d fiber material definition out of range")
        if not uni_mat_is_elastic(uniaxial_defs[cell.def_index]):
            return False
    return True


fn _copy_force_beam_column2d_states(
    elem_state_ids: List[Int],
    src_elem_state_offset: Int,
    dst_elem_state_offset: Int,
    elem_state_count: Int,
    mut uniaxial_states: List[UniMaterialState],
):
    if src_elem_state_offset < 0 or src_elem_state_offset + elem_state_count > len(
        elem_state_ids
    ):
        abort("forceBeamColumn2d fiber state source out of range")
    if dst_elem_state_offset < 0 or dst_elem_state_offset + elem_state_count > len(
        elem_state_ids
    ):
        abort("forceBeamColumn2d fiber state destination out of range")
    for i in range(elem_state_count):
        var src_state_index = elem_state_ids[src_elem_state_offset + i]
        if src_state_index < 0 or src_state_index >= len(uniaxial_states):
            abort("forceBeamColumn2d fiber state source index out of range")
        var dst_state_index = elem_state_ids[dst_elem_state_offset + i]
        if dst_state_index < 0 or dst_state_index >= len(uniaxial_states):
            abort("forceBeamColumn2d fiber state destination index out of range")
        uniaxial_states[dst_state_index] = uniaxial_states[src_state_index]


fn _copy_force_beam_column2d_basic_state(
    mut force_basic_q_state: List[Float64],
    src_offset: Int,
    dst_offset: Int,
    count: Int,
):
    if src_offset < 0 or src_offset + count > len(force_basic_q_state):
        abort("forceBeamColumn2d basic state source out of range")
    if dst_offset < 0 or dst_offset + count > len(force_basic_q_state):
        abort("forceBeamColumn2d basic state destination out of range")
    for i in range(count):
        force_basic_q_state[dst_offset + i] = force_basic_q_state[src_offset + i]


@always_inline
fn _set_force_beam_column2d_trial_basic_deformation(
    mut force_basic_q_state: List[Float64], basic_state_offset: Int, v0: Float64, v1: Float64, v2: Float64
):
    force_basic_q_state[basic_state_offset] = v0
    force_basic_q_state[basic_state_offset + 1] = v1
    force_basic_q_state[basic_state_offset + 2] = v2


fn _force_beam_column2d_exact_elastic_state(
    L: Float64,
    xis: List[Float64],
    weights: List[Float64],
    section_load_axial: List[Float64],
    section_load_moment: List[Float64],
    sec_def: FiberSection2dDef,
    fibers: List[FiberCell],
    uniaxial_defs: List[UniMaterialDef],
    mut uniaxial_states: List[UniMaterialState],
    elem_state_ids: List[Int],
    elem_state_offset: Int,
    fibers_per_section: Int,
    num_int_pts: Int,
    mut force_basic_q_state: List[Float64],
    force_basic_q_offset: Int,
    target_v0: Float64,
    target_v1: Float64,
    target_v2: Float64,
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
    Float64,
):
    var init_flex = _fiber_section2d_initial_flexibility(sec_def, fibers, uniaxial_defs)
    if not init_flex[0]:
        return (False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    var sec_f00 = init_flex[1]
    var sec_f01 = init_flex[2]
    var sec_f10 = sec_f01
    var sec_f11 = init_flex[3]
    var f00 = 0.0
    var f01 = 0.0
    var f02 = 0.0
    var f10 = 0.0
    var f11 = 0.0
    var f12 = 0.0
    var f20 = 0.0
    var f21 = 0.0
    var f22 = 0.0
    var v_load0 = 0.0
    var v_load1 = 0.0
    var v_load2 = 0.0

    for ip in range(num_int_pts):
        var xi = xis[ip]
        var weight = weights[ip]
        var wL = weight * L
        var b_mi = xi - 1.0
        var b_mj = xi

        f00 += wL * sec_f00
        f01 += wL * sec_f01 * b_mi
        f02 += wL * sec_f01 * b_mj
        f10 += wL * b_mi * sec_f10
        f11 += wL * b_mi * sec_f11 * b_mi
        f12 += wL * b_mi * sec_f11 * b_mj
        f20 += wL * b_mj * sec_f10
        f21 += wL * b_mj * sec_f11 * b_mi
        f22 += wL * b_mj * sec_f11 * b_mj
        var load_axial = section_load_axial[ip]
        var load_moment = section_load_moment[ip]
        v_load0 += wL * (sec_f00 * load_axial + sec_f01 * load_moment)
        v_load1 += wL * b_mi * (sec_f10 * load_axial + sec_f11 * load_moment)
        v_load2 += wL * b_mj * (sec_f10 * load_axial + sec_f11 * load_moment)

    var k_inv = _invert_3x3_values(f00, f01, f02, f10, f11, f12, f20, f21, f22)
    if not k_inv[0]:
        return (False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    var rhs0 = target_v0 - v_load0
    var rhs1 = target_v1 - v_load1
    var rhs2 = target_v2 - v_load2
    var q0 = k_inv[1] * rhs0 + k_inv[2] * rhs1 + k_inv[3] * rhs2
    var q1 = k_inv[4] * rhs0 + k_inv[5] * rhs1 + k_inv[6] * rhs2
    var q2 = k_inv[7] * rhs0 + k_inv[8] * rhs1 + k_inv[9] * rhs2
    var eps0_offset = force_basic_q_offset + 3
    var kappa_offset = eps0_offset + num_int_pts

    for ip in range(num_int_pts):
        var xi = xis[ip]
        var b_mi = xi - 1.0
        var b_mj = xi
        var axial = q0 + section_load_axial[ip]
        var moment = b_mi * q1 + b_mj * q2 + section_load_moment[ip]
        var eps0 = sec_f00 * axial + sec_f01 * moment
        var kappa = sec_f10 * axial + sec_f11 * moment
        force_basic_q_state[eps0_offset + ip] = eps0
        force_basic_q_state[kappa_offset + ip] = kappa

        var ip_state_offset = elem_state_offset + ip * fibers_per_section
        _ = _fiber_section2d_set_trial_from_offset(
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

    force_basic_q_state[force_basic_q_offset] = q0
    force_basic_q_state[force_basic_q_offset + 1] = q1
    force_basic_q_state[force_basic_q_offset + 2] = q2
    return (
        True,
        q0,
        q1,
        q2,
        k_inv[1],
        k_inv[2],
        k_inv[3],
        k_inv[4],
        k_inv[5],
        k_inv[6],
        k_inv[9],
    )


fn _force_beam_column2d_try_increment(
    L: Float64,
    xis: List[Float64],
    weights: List[Float64],
    section_load_axial: List[Float64],
    section_load_moment: List[Float64],
    sec_def: FiberSection2dDef,
    fibers: List[FiberCell],
    uniaxial_defs: List[UniMaterialDef],
    mut uniaxial_states: List[UniMaterialState],
    elem_state_ids: List[Int],
    elem_state_offset: Int,
    fibers_per_section: Int,
    num_int_pts: Int,
    mut force_basic_q_state: List[Float64],
    force_basic_q_offset: Int,
    base_v0: Float64,
    base_v1: Float64,
    base_v2: Float64,
    dv_trial0: Float64,
    dv_trial1: Float64,
    dv_trial2: Float64,
    use_initial_section_flexibility: Int,
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
    Float64,
):
    var q0 = force_basic_q_state[force_basic_q_offset]
    var q1 = force_basic_q_state[force_basic_q_offset + 1]
    var q2 = force_basic_q_state[force_basic_q_offset + 2]
    var eps0_offset = force_basic_q_offset + 3
    var kappa_offset = eps0_offset + num_int_pts

    var init_flex = _fiber_section2d_initial_flexibility(sec_def, fibers, uniaxial_defs)
    if use_initial_section_flexibility != 0 and not init_flex[0]:
        return (False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    var elem_tol = 1.0e-8
    var max_elem_iters = 80
    if use_initial_section_flexibility == 1:
        max_elem_iters *= 10

    var target_v0 = base_v0 + dv_trial0
    var target_v1 = base_v1 + dv_trial1
    var target_v2 = base_v2 + dv_trial2

    var k00: Float64
    var k01: Float64
    var k02: Float64
    var k10: Float64
    var k11: Float64
    var k12: Float64
    var k20: Float64
    var k21: Float64
    var k22: Float64

    var predictor_f00 = 0.0
    var predictor_f01 = 0.0
    var predictor_f02 = 0.0
    var predictor_f10 = 0.0
    var predictor_f11 = 0.0
    var predictor_f12 = 0.0
    var predictor_f20 = 0.0
    var predictor_f21 = 0.0
    var predictor_f22 = 0.0

    for ip in range(num_int_pts):
        var xi = xis[ip]
        var weight = weights[ip]
        var wL = weight * L
        var b_mi = xi - 1.0
        var b_mj = xi
        var ip_state_offset = elem_state_offset + ip * fibers_per_section
        var resp = _fiber_section2d_set_trial_from_offset(
            sec_def,
            fibers,
            uniaxial_defs,
            uniaxial_states,
            elem_state_ids,
            ip_state_offset,
            fibers_per_section,
            force_basic_q_state[eps0_offset + ip],
            force_basic_q_state[kappa_offset + ip],
        )
        var sec_flex = _fiber_section2d_response_flexibility(resp)
        if not sec_flex[0]:
            return (False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        var sec_f00 = sec_flex[1]
        var sec_f01 = sec_flex[2]
        var sec_f11 = sec_flex[3]
        var sec_f10 = sec_f01

        predictor_f00 += wL * sec_f00
        predictor_f01 += wL * sec_f01 * b_mi
        predictor_f02 += wL * sec_f01 * b_mj
        predictor_f10 += wL * b_mi * sec_f10
        predictor_f11 += wL * b_mi * sec_f11 * b_mi
        predictor_f12 += wL * b_mi * sec_f11 * b_mj
        predictor_f20 += wL * b_mj * sec_f10
        predictor_f21 += wL * b_mj * sec_f11 * b_mi
        predictor_f22 += wL * b_mj * sec_f11 * b_mj
    var predictor_k = _invert_3x3_values(
        predictor_f00,
        predictor_f01,
        predictor_f02,
        predictor_f10,
        predictor_f11,
        predictor_f12,
        predictor_f20,
        predictor_f21,
        predictor_f22,
    )
    if not predictor_k[0]:
        return (False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    q0 += predictor_k[1] * dv_trial0 + predictor_k[2] * dv_trial1 + predictor_k[3] * dv_trial2
    q1 += predictor_k[4] * dv_trial0 + predictor_k[5] * dv_trial1 + predictor_k[6] * dv_trial2
    q2 += predictor_k[7] * dv_trial0 + predictor_k[8] * dv_trial1 + predictor_k[9] * dv_trial2

    for iter in range(max_elem_iters):
        var f00 = 0.0
        var f01 = 0.0
        var f02 = 0.0
        var f10 = 0.0
        var f11 = 0.0
        var f12 = 0.0
        var f20 = 0.0
        var f21 = 0.0
        var f22 = 0.0

        var vr0 = 0.0
        var vr1 = 0.0
        var vr2 = 0.0

        for ip in range(num_int_pts):
            var xi = xis[ip]
            var weight = weights[ip]
            var wL = weight * L
            var b_mi = xi - 1.0
            var b_mj = xi
            var axial_target = q0 + section_load_axial[ip]
            var moment_target = b_mi * q1 + b_mj * q2 + section_load_moment[ip]
            var ip_state_offset = elem_state_offset + ip * fibers_per_section
            var eps0 = force_basic_q_state[eps0_offset + ip]
            var kappa = force_basic_q_state[kappa_offset + ip]

            var resp_current = _fiber_section2d_set_trial_from_offset(
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
            var current_flex = _fiber_section2d_response_flexibility(resp_current)
            if not current_flex[0]:
                return (False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

            var predictor_sec_f00: Float64
            var predictor_sec_f01: Float64
            var predictor_sec_f11: Float64
            if use_initial_section_flexibility == 1:
                predictor_sec_f00 = init_flex[1]
                predictor_sec_f01 = init_flex[2]
                predictor_sec_f11 = init_flex[3]
            elif use_initial_section_flexibility == 2 and iter == 0:
                predictor_sec_f00 = init_flex[1]
                predictor_sec_f01 = init_flex[2]
                predictor_sec_f11 = init_flex[3]
            else:
                predictor_sec_f00 = current_flex[1]
                predictor_sec_f01 = current_flex[2]
                predictor_sec_f11 = current_flex[3]

            var d_axial = axial_target - resp_current.axial_force
            var d_moment = moment_target - resp_current.moment_z
            var deps0 = predictor_sec_f00 * d_axial + predictor_sec_f01 * d_moment
            var dkappa = predictor_sec_f01 * d_axial + predictor_sec_f11 * d_moment
            eps0 += deps0
            kappa += dkappa

            var resp_trial = _fiber_section2d_set_trial_from_offset(
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
            force_basic_q_state[eps0_offset + ip] = eps0
            force_basic_q_state[kappa_offset + ip] = kappa

            var sec_flex = _fiber_section2d_response_flexibility(resp_trial)
            if not sec_flex[0]:
                return (False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            var sec_f00 = sec_flex[1]
            var sec_f01 = sec_flex[2]
            var sec_f11 = sec_flex[3]
            var sec_f10 = sec_f01

            var residual_axial = axial_target - resp_trial.axial_force
            var residual_moment = moment_target - resp_trial.moment_z
            var residual_eps0 = sec_f00 * residual_axial + sec_f01 * residual_moment
            var residual_kappa = sec_f10 * residual_axial + sec_f11 * residual_moment

            f00 += wL * sec_f00
            f01 += wL * sec_f01 * b_mi
            f02 += wL * sec_f01 * b_mj
            f10 += wL * b_mi * sec_f10
            f11 += wL * b_mi * sec_f11 * b_mi
            f12 += wL * b_mi * sec_f11 * b_mj
            f20 += wL * b_mj * sec_f10
            f21 += wL * b_mj * sec_f11 * b_mi
            f22 += wL * b_mj * sec_f11 * b_mj

            vr0 += wL * (eps0 + residual_eps0)
            vr1 += wL * b_mi * (kappa + residual_kappa)
            vr2 += wL * b_mj * (kappa + residual_kappa)
        var k_inv = _invert_3x3_values(f00, f01, f02, f10, f11, f12, f20, f21, f22)
        if not k_inv[0]:
            return (False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        k00 = k_inv[1]
        k01 = k_inv[2]
        k02 = k_inv[3]
        k10 = k_inv[4]
        k11 = k_inv[5]
        k12 = k_inv[6]
        k20 = k_inv[7]
        k21 = k_inv[8]
        k22 = k_inv[9]

        var residual_0 = target_v0 - vr0
        var residual_1 = target_v1 - vr1
        var residual_2 = target_v2 - vr2
        var dq0 = k00 * residual_0 + k01 * residual_1 + k02 * residual_2
        var dq1 = k10 * residual_0 + k11 * residual_1 + k12 * residual_2
        var dq2 = k20 * residual_0 + k21 * residual_1 + k22 * residual_2
        var work_norm = abs(residual_0 * dq0 + residual_1 * dq1 + residual_2 * dq2)

        q0 += dq0
        q1 += dq1
        q2 += dq2

        if work_norm <= elem_tol or _max_abs3(residual_0, residual_1, residual_2) <= elem_tol:
            force_basic_q_state[force_basic_q_offset] = q0
            force_basic_q_state[force_basic_q_offset + 1] = q1
            force_basic_q_state[force_basic_q_offset + 2] = q2
            return (True, q0, q1, q2, k00, k01, k02, k10, k11, k12, k22)

    return (False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


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
    geom_transf: String,
    integration: String,
    num_int_pts: Int,
    mut force_basic_q_state: List[Float64],
    force_basic_q_offset: Int,
    force_basic_q_count: Int,
    mut k_global_out: List[List[Float64]],
    mut f_global_out: List[Float64],
):
    var empty_element_loads: List[ElementLoadInput] = []
    var empty_elem_load_offsets: List[Int] = []
    var empty_elem_load_pool: List[Int] = []
    force_beam_column2d_global_tangent_and_internal(
        0,
        x1,
        y1,
        x2,
        y2,
        u_elem_global,
        empty_element_loads,
        empty_elem_load_offsets,
        empty_elem_load_pool,
        0.0,
        sec_def,
        fibers,
        uniaxial_defs,
        uniaxial_states,
        elem_state_ids,
        elem_state_offset,
        elem_state_count,
        geom_transf,
        integration,
        num_int_pts,
        force_basic_q_state,
        force_basic_q_offset,
        force_basic_q_count,
        k_global_out,
        f_global_out,
    )


fn force_beam_column2d_global_tangent_and_internal(
    elem_index: Int,
    x1: Float64,
    y1: Float64,
    x2: Float64,
    y2: Float64,
    u_elem_global: List[Float64],
    element_loads: List[ElementLoadInput],
    elem_load_offsets: List[Int],
    elem_load_pool: List[Int],
    load_scale: Float64,
    sec_def: FiberSection2dDef,
    fibers: List[FiberCell],
    uniaxial_defs: List[UniMaterialDef],
    mut uniaxial_states: List[UniMaterialState],
    elem_state_ids: List[Int],
    elem_state_offset: Int,
    elem_state_count: Int,
    geom_transf: String,
    integration: String,
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
    var xis: List[Float64] = []
    var weights: List[Float64] = []
    beam_integration_rule(integration, num_int_pts, xis, weights)
    var section_load_axial: List[Float64] = []
    var section_load_moment: List[Float64] = []
    section_load_axial.resize(num_int_pts, 0.0)
    section_load_moment.resize(num_int_pts, 0.0)
    for ip in range(num_int_pts):
        var loads = beam2d_section_load_response(
            element_loads,
            elem_load_offsets,
            elem_load_pool,
            elem_index,
            load_scale,
            xis[ip] * L,
            L,
        )
        section_load_axial[ip] = loads[0]
        section_load_moment[ip] = loads[1]
    var fixed_end = beam2d_basic_fixed_end_and_reactions(
        element_loads, elem_load_offsets, elem_load_pool, elem_index, load_scale, L
    )

    var c = dx / L
    var s = dy / L
    var u_local: List[Float64] = []
    _beam2d_transform_u_global_to_local(c, s, u_elem_global, u_local)

    var predictor_state_count = 3 + 2 * num_int_pts
    var active_basic_count = predictor_state_count + 3
    if force_basic_q_count < 2 * active_basic_count:
        abort("forceBeamColumn2d basic force state count mismatch")
    if (
        force_basic_q_offset < 0
        or force_basic_q_offset + 2 * active_basic_count > len(force_basic_q_state)
    ):
        abort("forceBeamColumn2d basic force state out of range")
    var basic_state_offset = force_basic_q_offset + predictor_state_count
    var basic_backup_offset = force_basic_q_offset + active_basic_count
    var elem_state_backup_offset = elem_state_offset + elem_state_count
    if (
        elem_state_count > 0
        and elem_state_backup_offset + elem_state_count > len(elem_state_ids)
    ):
        abort("forceBeamColumn2d fiber backup state out of range")

    var chord_rotation = (u_local[4] - u_local[1]) / L
    var v_basic_0 = u_local[3] - u_local[0]
    var v_basic_1 = u_local[2] - chord_rotation
    var v_basic_2 = u_local[5] - chord_rotation
    var basic_prev_0 = force_basic_q_state[basic_state_offset]
    var basic_prev_1 = force_basic_q_state[basic_state_offset + 1]
    var basic_prev_2 = force_basic_q_state[basic_state_offset + 2]

    _copy_force_beam_column2d_basic_state(
        force_basic_q_state,
        force_basic_q_offset,
        basic_backup_offset,
        active_basic_count,
    )
    if elem_state_count > 0:
        _copy_force_beam_column2d_states(
            elem_state_ids,
            elem_state_offset,
            elem_state_backup_offset,
            elem_state_count,
            uniaxial_states,
        )

    var accepted_basic_0 = basic_prev_0
    var accepted_basic_1 = basic_prev_1
    var accepted_basic_2 = basic_prev_2

    var remaining_0 = v_basic_0 - accepted_basic_0
    var remaining_1 = v_basic_1 - accepted_basic_1
    var remaining_2 = v_basic_2 - accepted_basic_2
    var attempt_0 = remaining_0
    var attempt_1 = remaining_1
    var attempt_2 = remaining_2
    var converged = False
    var all_materials_elastic = _fiber_section2d_all_materials_elastic(
        sec_def, fibers, uniaxial_defs
    )
    var best_solved: (
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
        Float64,
    ) = (
        False,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )

    var tolerance = 1.0e-10
    var max_subdivisions = 12
    var subdivisions = 0
    var cutback_factor = 10.0

    if all_materials_elastic:
        var solved = _force_beam_column2d_exact_elastic_state(
            L,
            xis,
            weights,
            section_load_axial,
            section_load_moment,
            sec_def,
            fibers,
            uniaxial_defs,
            uniaxial_states,
            elem_state_ids,
            elem_state_offset,
            fibers_per_section,
            num_int_pts,
            force_basic_q_state,
            force_basic_q_offset,
            v_basic_0,
            v_basic_1,
            v_basic_2,
        )
        if solved[0]:
            accepted_basic_0 = v_basic_0
            accepted_basic_1 = v_basic_1
            accepted_basic_2 = v_basic_2
            _set_force_beam_column2d_trial_basic_deformation(
                force_basic_q_state,
                basic_state_offset,
                accepted_basic_0,
                accepted_basic_1,
                accepted_basic_2,
            )
            _copy_force_beam_column2d_basic_state(
                force_basic_q_state,
                force_basic_q_offset,
                basic_backup_offset,
                active_basic_count,
            )
            if elem_state_count > 0:
                _copy_force_beam_column2d_states(
                    elem_state_ids,
                    elem_state_offset,
                    elem_state_backup_offset,
                    elem_state_count,
                    uniaxial_states,
                )
            converged = True
            remaining_0 = 0.0
            remaining_1 = 0.0
            remaining_2 = 0.0
            best_solved = solved

    if not converged and _max_abs3(remaining_0, remaining_1, remaining_2) <= tolerance:
        for use_initial in range(3):
            _copy_force_beam_column2d_basic_state(
                force_basic_q_state,
                basic_backup_offset,
                force_basic_q_offset,
                active_basic_count,
            )
            if elem_state_count > 0:
                _copy_force_beam_column2d_states(
                    elem_state_ids,
                    elem_state_backup_offset,
                    elem_state_offset,
                    elem_state_count,
                    uniaxial_states,
                )
            var solved = _force_beam_column2d_try_increment(
                L,
                xis,
                weights,
                section_load_axial,
                section_load_moment,
                sec_def,
                fibers,
                uniaxial_defs,
                uniaxial_states,
                elem_state_ids,
                elem_state_offset,
                fibers_per_section,
                num_int_pts,
                force_basic_q_state,
                force_basic_q_offset,
                accepted_basic_0,
                accepted_basic_1,
                accepted_basic_2,
                0.0,
                0.0,
                0.0,
                use_initial,
            )
            if not solved[0]:
                continue
            best_solved = solved
            converged = True
            break
    elif not converged:
        while True:
            if subdivisions > max_subdivisions:
                break
            var target_v0 = accepted_basic_0 + attempt_0
            var target_v1 = accepted_basic_1 + attempt_1
            var target_v2 = accepted_basic_2 + attempt_2

            var scheme_success = False
            for use_initial in range(3):
                _copy_force_beam_column2d_basic_state(
                    force_basic_q_state,
                    basic_backup_offset,
                    force_basic_q_offset,
                    active_basic_count,
                )
                if elem_state_count > 0:
                    _copy_force_beam_column2d_states(
                        elem_state_ids,
                        elem_state_backup_offset,
                        elem_state_offset,
                        elem_state_count,
                        uniaxial_states,
                    )

                var solved = _force_beam_column2d_try_increment(
                    L,
                    xis,
                    weights,
                    section_load_axial,
                    section_load_moment,
                    sec_def,
                    fibers,
                    uniaxial_defs,
                    uniaxial_states,
                    elem_state_ids,
                    elem_state_offset,
                    fibers_per_section,
                    num_int_pts,
                    force_basic_q_state,
                    force_basic_q_offset,
                    accepted_basic_0,
                    accepted_basic_1,
                    accepted_basic_2,
                    attempt_0,
                    attempt_1,
                    attempt_2,
                    use_initial,
                )
                if not solved[0]:
                    continue

                accepted_basic_0 = target_v0
                accepted_basic_1 = target_v1
                accepted_basic_2 = target_v2
                _set_force_beam_column2d_trial_basic_deformation(
                    force_basic_q_state,
                    basic_state_offset,
                    accepted_basic_0,
                    accepted_basic_1,
                    accepted_basic_2,
                )
                _copy_force_beam_column2d_basic_state(
                    force_basic_q_state,
                    force_basic_q_offset,
                    basic_backup_offset,
                    active_basic_count,
                )
                if elem_state_count > 0:
                    _copy_force_beam_column2d_states(
                        elem_state_ids,
                        elem_state_offset,
                        elem_state_backup_offset,
                        elem_state_count,
                        uniaxial_states,
                    )
                best_solved = solved
                remaining_0 = v_basic_0 - accepted_basic_0
                remaining_1 = v_basic_1 - accepted_basic_1
                remaining_2 = v_basic_2 - accepted_basic_2
                var remaining_norm = _max_abs3(remaining_0, remaining_1, remaining_2)
                if remaining_norm <= tolerance:
                    converged = True
                else:
                    attempt_0 = remaining_0
                    attempt_1 = remaining_1
                    attempt_2 = remaining_2
                    subdivisions = 0
                scheme_success = True
                break

            if converged:
                break
            if scheme_success:
                continue

            attempt_0 /= cutback_factor
            attempt_1 /= cutback_factor
            attempt_2 /= cutback_factor
            subdivisions += 1

    if not converged:
        abort("forceBeamColumn2d element compatibility did not converge")
    _copy_force_beam_column2d_basic_state(
        force_basic_q_state,
        basic_backup_offset,
        force_basic_q_offset,
        active_basic_count,
    )
    if elem_state_count > 0:
        _copy_force_beam_column2d_states(
            elem_state_ids,
            elem_state_backup_offset,
            elem_state_offset,
            elem_state_count,
            uniaxial_states,
        )
    if not best_solved[0]:
        abort("forceBeamColumn2d final tangent recovery did not converge")
    var q0 = best_solved[1]
    var q1 = best_solved[2]
    var q2 = best_solved[3]
    var k00 = best_solved[4]
    var k01 = best_solved[5]
    var k02 = best_solved[6]
    var k10 = best_solved[7]
    var k11 = best_solved[8]
    var k12 = best_solved[9]
    var k20 = k02
    var k21 = k12
    var k22 = best_solved[10]

    var inv_L = 1.0 / L
    _ensure_zero_matrix(k_global_out, 6, 6)
    for a in range(6):
        var a0: Float64
        var a1: Float64
        var a2: Float64
        (a0, a1, a2) = _beam2d_local_basic_map(a, inv_L)
        for b in range(6):
            var b0: Float64
            var b1: Float64
            var b2: Float64
            (b0, b1, b2) = _beam2d_local_basic_map(b, inv_L)
            k_global_out[a][b] = (
                a0 * (k00 * b0 + k01 * b1 + k02 * b2)
                + a1 * (k10 * b0 + k11 * b1 + k12 * b2)
                + a2 * (k20 * b0 + k21 * b1 + k22 * b2)
            )
    if geom_transf == "PDelta":
        var q0_over_l = q0 * inv_L
        k_global_out[1][1] += q0_over_l
        k_global_out[4][4] += q0_over_l
        k_global_out[1][4] -= q0_over_l
        k_global_out[4][1] -= q0_over_l

    _ensure_zero_vector(f_global_out, 6)
    f_global_out[0] = -q0
    f_global_out[1] = (q1 + q2) * inv_L
    f_global_out[2] = q1
    f_global_out[3] = q0
    f_global_out[4] = -(q1 + q2) * inv_L
    f_global_out[5] = q2
    f_global_out[0] += fixed_end[3]
    f_global_out[1] += fixed_end[4]
    f_global_out[4] += fixed_end[5]
    if geom_transf == "PDelta":
        var pdelta_shear = (u_local[1] - u_local[4]) * q0 * inv_L
        f_global_out[1] += pdelta_shear
        f_global_out[4] -= pdelta_shear

    _beam2d_transform_stiffness_local_to_global_in_place(k_global_out, c, s)
    _beam2d_transform_force_local_to_global_in_place(f_global_out, c, s)
