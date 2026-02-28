from collections import List
from math import sqrt
from os import abort

from elements.beam_loads import beam3d_basic_fixed_end_and_reactions, beam3d_section_load_response
from elements.beam_integration import beam_integration_rule
from elements.utils import (
    _cross,
    _dot,
    _ensure_zero_matrix,
    _ensure_zero_vector,
    _normalize,
)
from materials import UniMaterialDef, UniMaterialState, uniaxial_set_trial_strain
from solver.run_case.input_types import ElementLoadInput
from sections import FiberCell, FiberSection3dDef, FiberSection3dResponse


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

    var vx = 1.0
    var vy = 0.0
    var vz = 0.0
    if abs(_dot(lx, ly, lz, vx, vy, vz)) >= 0.9:
        vx = 0.0
        vy = 1.0
        vz = 0.0
        if abs(_dot(lx, ly, lz, vx, vy, vz)) >= 0.9:
            vx = 0.0
            vy = 0.0
            vz = 1.0

    var yx: Float64
    var yy: Float64
    var yz: Float64
    # Match OpenSees Linear/PDelta/Corotational vecxz orientation:
    # local y = vecxz x local x, local z = local x x local y.
    (yx, yy, yz) = _cross(vx, vy, vz, lx, ly, lz)
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


fn _beam3d_transform_u_global_to_local(
    R: List[List[Float64]], u_global: List[Float64], mut u_local: List[Float64]
):
    _ensure_zero_vector(u_local, 12)
    for block in range(4):
        var offset = 3 * block
        var g0 = u_global[offset]
        var g1 = u_global[offset + 1]
        var g2 = u_global[offset + 2]
        for i in range(3):
            u_local[offset + i] = R[i][0] * g0 + R[i][1] * g1 + R[i][2] * g2


fn _beam3d_transform_f_local_to_global(
    R: List[List[Float64]], f_local: List[Float64], mut f_global: List[Float64]
):
    _ensure_zero_vector(f_global, 12)
    for block in range(4):
        var offset = 3 * block
        var l0 = f_local[offset]
        var l1 = f_local[offset + 1]
        var l2 = f_local[offset + 2]
        for i in range(3):
            f_global[offset + i] = R[0][i] * l0 + R[1][i] * l1 + R[2][i] * l2


fn _beam3d_transform_stiffness_local_to_global(
    R: List[List[Float64]],
    k_local: List[List[Float64]],
    mut work: List[List[Float64]],
    mut k_global: List[List[Float64]],
):
    _ensure_zero_matrix(work, 12, 12)
    _ensure_zero_matrix(k_global, 12, 12)
    for row_block in range(4):
        var row_offset = 3 * row_block
        for col_block in range(4):
            var col_offset = 3 * col_block
            for i in range(3):
                var src0 = k_local[row_offset + i][col_offset]
                var src1 = k_local[row_offset + i][col_offset + 1]
                var src2 = k_local[row_offset + i][col_offset + 2]
                for j in range(3):
                    work[row_offset + i][col_offset + j] = (
                        src0 * R[0][j] + src1 * R[1][j] + src2 * R[2][j]
                    )
    for row_block in range(4):
        var row_offset = 3 * row_block
        for col in range(12):
            for i in range(3):
                k_global[row_offset + i][col] = (
                    R[0][i] * work[row_offset][col]
                    + R[1][i] * work[row_offset + 1][col]
                    + R[2][i] * work[row_offset + 2][col]
                )


@always_inline
fn _beam3d_local_basic_coeff(row: Int, col: Int, inv_L: Float64) -> Float64:
    if row == 0:
        if col == 0:
            return -1.0
        if col == 6:
            return 1.0
        return 0.0
    if row == 1:
        if col == 1:
            return inv_L
        if col == 5:
            return 1.0
        if col == 7:
            return -inv_L
        return 0.0
    if row == 2:
        if col == 1:
            return inv_L
        if col == 7:
            return -inv_L
        if col == 11:
            return 1.0
        return 0.0
    if row == 3:
        if col == 2:
            return -inv_L
        if col == 4:
            return 1.0
        if col == 8:
            return inv_L
        return 0.0
    if row == 4:
        if col == 2:
            return -inv_L
        if col == 8:
            return inv_L
        if col == 10:
            return 1.0
        return 0.0
    if col == 3:
        return -1.0
    if col == 9:
        return 1.0
    return 0.0


fn _beam3d_add_axial_geometric_stiffness(
    mut k_local: List[List[Float64]], axial_force: Float64, L: Float64
):
    var N_over_L = axial_force / L

    k_local[1][1] += N_over_L
    k_local[1][7] -= N_over_L
    k_local[7][1] -= N_over_L
    k_local[7][7] += N_over_L

    k_local[2][2] += N_over_L
    k_local[2][8] -= N_over_L
    k_local[8][2] -= N_over_L
    k_local[8][8] += N_over_L


fn _element_rotation(
    x1: Float64,
    y1: Float64,
    z1: Float64,
    x2: Float64,
    y2: Float64,
    z2: Float64,
    u_elem_global: List[Float64],
    geom_transf: String,
) -> List[List[Float64]]:
    var dx0 = x2 - x1
    var dy0 = y2 - y1
    var dz0 = z2 - z1
    var L0 = sqrt(dx0 * dx0 + dy0 * dy0 + dz0 * dz0)
    if L0 == 0.0:
        abort("zero-length element")

    if geom_transf == "Corotational":
        return _beam3d_rotation(
            x1 + u_elem_global[0],
            y1 + u_elem_global[1],
            z1 + u_elem_global[2],
            x2 + u_elem_global[6],
            y2 + u_elem_global[7],
            z2 + u_elem_global[8],
        )
    return _beam3d_rotation(x1, y1, z1, x2, y2, z2)


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


fn _invert_5x5_into(mut a: List[List[Float64]], mut inv_out: List[List[Float64]]) -> Bool:
    var n = 5
    _ensure_zero_matrix(inv_out, n, n)
    for i in range(n):
        inv_out[i][i] = 1.0

    for i in range(n):
        var pivot = i
        var max_val = abs(a[i][i])
        for r in range(i + 1, n):
            var abs_val = abs(a[r][i])
            if abs_val > max_val:
                max_val = abs_val
                pivot = r
        if max_val <= 1.0e-40:
            return False

        if pivot != i:
            for c in range(n):
                var tmp = a[i][c]
                a[i][c] = a[pivot][c]
                a[pivot][c] = tmp
                var inv_tmp = inv_out[i][c]
                inv_out[i][c] = inv_out[pivot][c]
                inv_out[pivot][c] = inv_tmp

        var piv = a[i][i]
        for c in range(n):
            a[i][c] /= piv
            inv_out[i][c] /= piv

        for r in range(n):
            if r == i:
                continue
            var factor = a[r][i]
            if factor == 0.0:
                continue
            for c in range(n):
                a[r][c] -= factor * a[i][c]
                inv_out[r][c] -= factor * inv_out[i][c]
    return True


fn _fiber_section3d_set_trial_from_offset(
    sec_def: FiberSection3dDef,
    fibers: List[FiberCell],
    uniaxial_defs: List[UniMaterialDef],
    mut uniaxial_states: List[UniMaterialState],
    section_state_ids: List[Int],
    section_state_offset: Int,
    section_state_count: Int,
    eps0: Float64,
    kappa_y: Float64,
    kappa_z: Float64,
) -> FiberSection3dResponse:
    if section_state_count != sec_def.fiber_count:
        abort("forceBeamColumn3d fiber state count mismatch")
    if section_state_offset < 0 or section_state_offset + section_state_count > len(
        section_state_ids
    ):
        abort("forceBeamColumn3d fiber section state out of range")

    var axial_force = 0.0
    var moment_y = 0.0
    var moment_z = 0.0
    var k11 = 0.0
    var k12 = 0.0
    var k13 = 0.0
    var k22 = 0.0
    var k23 = 0.0
    var k33 = 0.0

    for i in range(section_state_count):
        var cell = fibers[sec_def.fiber_offset + i]
        var y_rel = cell.y - sec_def.y_bar
        var z_rel = cell.z - sec_def.z_bar
        var eps = eps0 + z_rel * kappa_y - y_rel * kappa_z

        var state_index = section_state_ids[section_state_offset + i]
        if state_index < 0 or state_index >= len(uniaxial_states):
            abort("forceBeamColumn3d fiber state index out of range")
        if cell.def_index < 0 or cell.def_index >= len(uniaxial_defs):
            abort("forceBeamColumn3d fiber material definition out of range")
        var mat_def = uniaxial_defs[cell.def_index]
        var state = uniaxial_states[state_index]
        uniaxial_set_trial_strain(mat_def, state, eps)
        uniaxial_states[state_index] = state

        var area = cell.area
        var fs = state.sig_t * area
        var ks = state.tangent_t * area
        axial_force += fs
        moment_y += fs * z_rel
        moment_z += -fs * y_rel
        k11 += ks
        k12 += ks * z_rel
        k13 += -ks * y_rel
        k22 += ks * z_rel * z_rel
        k23 += -ks * z_rel * y_rel
        k33 += ks * y_rel * y_rel

    return FiberSection3dResponse(
        axial_force,
        moment_y,
        moment_z,
        k11,
        k12,
        k13,
        k22,
        k23,
        k33,
    )


fn _fiber_section3d_solve_for_force(
    sec_def: FiberSection3dDef,
    fibers: List[FiberCell],
    uniaxial_defs: List[UniMaterialDef],
    mut uniaxial_states: List[UniMaterialState],
    section_state_ids: List[Int],
    section_state_offset: Int,
    section_state_count: Int,
    axial_force_target: Float64,
    moment_y_target: Float64,
    moment_z_target: Float64,
    mut eps0_guess: Float64,
    mut kappa_y_guess: Float64,
    mut kappa_z_guess: Float64,
    max_iters: Int,
    tol: Float64,
) -> (FiberSection3dResponse, Float64, Float64, Float64):
    var iter = 0
    var resp = FiberSection3dResponse()
    while iter < max_iters:
        resp = _fiber_section3d_set_trial_from_offset(
            sec_def,
            fibers,
            uniaxial_defs,
            uniaxial_states,
            section_state_ids,
            section_state_offset,
            section_state_count,
            eps0_guess,
            kappa_y_guess,
            kappa_z_guess,
        )

        var r_axial = axial_force_target - resp.axial_force
        var r_my = moment_y_target - resp.moment_y
        var r_mz = moment_z_target - resp.moment_z
        if abs(r_axial) <= tol and abs(r_my) <= tol and abs(r_mz) <= tol:
            return (resp, eps0_guess, kappa_y_guess, kappa_z_guess)

        var inv_sec = _invert_3x3_values(
            resp.k11,
            resp.k12,
            resp.k13,
            resp.k12,
            resp.k22,
            resp.k23,
            resp.k13,
            resp.k23,
            resp.k33,
        )
        if not inv_sec[0]:
            return (resp, eps0_guess, kappa_y_guess, kappa_z_guess)

        var de0 = inv_sec[1] * r_axial + inv_sec[4] * r_my + inv_sec[7] * r_mz
        var dky = inv_sec[2] * r_axial + inv_sec[5] * r_my + inv_sec[8] * r_mz
        var dkz = inv_sec[3] * r_axial + inv_sec[6] * r_my + inv_sec[9] * r_mz
        eps0_guess += de0
        kappa_y_guess += dky
        kappa_z_guess += dkz
        iter += 1
    return (resp, eps0_guess, kappa_y_guess, kappa_z_guess)


fn _force_beam_column3d_basic_state(
    u_local: List[Float64], L: Float64
) -> (Float64, Float64, Float64, Float64, Float64, Float64):
    var chord_z = (u_local[7] - u_local[1]) / L
    var chord_y = (u_local[8] - u_local[2]) / L
    return (
        u_local[6] - u_local[0],
        u_local[5] - chord_z,
        u_local[11] - chord_z,
        u_local[4] + chord_y,
        u_local[10] + chord_y,
        u_local[9] - u_local[3],
    )


fn force_beam_column3d_global_tangent_and_internal(
    x1: Float64,
    y1: Float64,
    z1: Float64,
    x2: Float64,
    y2: Float64,
    z2: Float64,
    u_elem_global: List[Float64],
    geom_transf: String,
    E: Float64,
    A: Float64,
    Iy: Float64,
    Iz: Float64,
    G: Float64,
    J: Float64,
    mut k_global_out: List[List[Float64]],
    mut f_global_out: List[Float64],
):
    var empty_element_loads: List[ElementLoadInput] = []
    var empty_elem_load_offsets: List[Int] = []
    var empty_elem_load_pool: List[Int] = []
    force_beam_column3d_global_tangent_and_internal(
        0,
        x1,
        y1,
        z1,
        x2,
        y2,
        z2,
        u_elem_global,
        geom_transf,
        empty_element_loads,
        empty_elem_load_offsets,
        empty_elem_load_pool,
        0.0,
        E,
        A,
        Iy,
        Iz,
        G,
        J,
        k_global_out,
        f_global_out,
    )


fn force_beam_column3d_global_tangent_and_internal(
    elem_index: Int,
    x1: Float64,
    y1: Float64,
    z1: Float64,
    x2: Float64,
    y2: Float64,
    z2: Float64,
    u_elem_global: List[Float64],
    geom_transf: String,
    element_loads: List[ElementLoadInput],
    elem_load_offsets: List[Int],
    elem_load_pool: List[Int],
    load_scale: Float64,
    E: Float64,
    A: Float64,
    Iy: Float64,
    Iz: Float64,
    G: Float64,
    J: Float64,
    mut k_global_out: List[List[Float64]],
    mut f_global_out: List[Float64],
):
    var dx = x2 - x1
    var dy = y2 - y1
    var dz = z2 - z1
    var L = sqrt(dx * dx + dy * dy + dz * dz)
    if L == 0.0:
        abort("zero-length element")

    var R = _element_rotation(
        x1,
        y1,
        z1,
        x2,
        y2,
        z2,
        u_elem_global,
        geom_transf,
    )
    var u_local: List[Float64] = []
    _beam3d_transform_u_global_to_local(R, u_elem_global, u_local)
    var fixed_end = beam3d_basic_fixed_end_and_reactions(
        element_loads, elem_load_offsets, elem_load_pool, elem_index, load_scale, L
    )

    var v_basic_0: Float64
    var v_basic_1: Float64
    var v_basic_2: Float64
    var v_basic_3: Float64
    var v_basic_4: Float64
    var torsion_basic: Float64
    (v_basic_0, v_basic_1, v_basic_2, v_basic_3, v_basic_4, torsion_basic) = (
        _force_beam_column3d_basic_state(u_local, L)
    )
    var q_basic_0 = (E * A / L) * v_basic_0 + fixed_end[0]
    var q_basic_1 = (E * Iz / L) * (4.0 * v_basic_1 + 2.0 * v_basic_2) + fixed_end[1]
    var q_basic_2 = (E * Iz / L) * (2.0 * v_basic_1 + 4.0 * v_basic_2) + fixed_end[2]
    var q_basic_3 = (E * Iy / L) * (4.0 * v_basic_3 + 2.0 * v_basic_4) + fixed_end[3]
    var q_basic_4 = (E * Iy / L) * (2.0 * v_basic_3 + 4.0 * v_basic_4) + fixed_end[4]
    var q_basic_5 = 0.0
    if G > 0.0 and J > 0.0:
        q_basic_5 = (G * J / L) * torsion_basic

    var inv_L = 1.0 / L
    var k_local: List[List[Float64]] = []
    _ensure_zero_matrix(k_local, 12, 12)
    var axial_k = E * A / L
    var flex_z = E * Iz / L
    var flex_y = E * Iy / L
    var torsion_k = 0.0
    if G > 0.0 and J > 0.0:
        torsion_k = G * J / L
    for a in range(12):
        for b in range(12):
            var a0 = _beam3d_local_basic_coeff(0, a, inv_L)
            var a1 = _beam3d_local_basic_coeff(1, a, inv_L)
            var a2 = _beam3d_local_basic_coeff(2, a, inv_L)
            var a3 = _beam3d_local_basic_coeff(3, a, inv_L)
            var a4 = _beam3d_local_basic_coeff(4, a, inv_L)
            var a5 = _beam3d_local_basic_coeff(5, a, inv_L)
            var b0 = _beam3d_local_basic_coeff(0, b, inv_L)
            var b1 = _beam3d_local_basic_coeff(1, b, inv_L)
            var b2 = _beam3d_local_basic_coeff(2, b, inv_L)
            var b3 = _beam3d_local_basic_coeff(3, b, inv_L)
            var b4 = _beam3d_local_basic_coeff(4, b, inv_L)
            var b5 = _beam3d_local_basic_coeff(5, b, inv_L)
            k_local[a][b] += a0 * axial_k * b0
            k_local[a][b] += flex_z * (
                a1 * (4.0 * b1 + 2.0 * b2) + a2 * (2.0 * b1 + 4.0 * b2)
            )
            k_local[a][b] += flex_y * (
                a3 * (4.0 * b3 + 2.0 * b4) + a4 * (2.0 * b3 + 4.0 * b4)
            )
            if torsion_k != 0.0:
                k_local[a][b] += a5 * torsion_k * b5

    if geom_transf == "PDelta" or geom_transf == "Corotational":
        _beam3d_add_axial_geometric_stiffness(k_local, q_basic_0, L)

    var f_local: List[Float64] = []
    _ensure_zero_vector(f_local, 12)
    for a in range(12):
        f_local[a] += _beam3d_local_basic_coeff(0, a, inv_L) * q_basic_0
        f_local[a] += _beam3d_local_basic_coeff(1, a, inv_L) * q_basic_1
        f_local[a] += _beam3d_local_basic_coeff(2, a, inv_L) * q_basic_2
        f_local[a] += _beam3d_local_basic_coeff(3, a, inv_L) * q_basic_3
        f_local[a] += _beam3d_local_basic_coeff(4, a, inv_L) * q_basic_4
        f_local[a] += _beam3d_local_basic_coeff(5, a, inv_L) * q_basic_5
    f_local[0] += fixed_end[5]
    f_local[1] += fixed_end[6]
    f_local[2] += fixed_end[8]
    f_local[7] += fixed_end[7]
    f_local[8] += fixed_end[9]

    var k_work: List[List[Float64]] = []
    _beam3d_transform_stiffness_local_to_global(R, k_local, k_work, k_global_out)
    _beam3d_transform_f_local_to_global(R, f_local, f_global_out)


fn force_beam_column3d_fiber_global_tangent_and_internal(
    x1: Float64,
    y1: Float64,
    z1: Float64,
    x2: Float64,
    y2: Float64,
    z2: Float64,
    u_elem_global: List[Float64],
    geom_transf: String,
    sec_def: FiberSection3dDef,
    fibers: List[FiberCell],
    uniaxial_defs: List[UniMaterialDef],
    mut uniaxial_states: List[UniMaterialState],
    elem_state_ids: List[Int],
    elem_state_offset: Int,
    elem_state_count: Int,
    integration: String,
    num_int_pts: Int,
    G: Float64,
    J: Float64,
    mut force_basic_q_state: List[Float64],
    force_basic_q_offset: Int,
    force_basic_q_count: Int,
    mut k_global_out: List[List[Float64]],
    mut f_global_out: List[Float64],
):
    var empty_element_loads: List[ElementLoadInput] = []
    var empty_elem_load_offsets: List[Int] = []
    var empty_elem_load_pool: List[Int] = []
    force_beam_column3d_fiber_global_tangent_and_internal(
        0,
        x1,
        y1,
        z1,
        x2,
        y2,
        z2,
        u_elem_global,
        geom_transf,
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
        integration,
        num_int_pts,
        G,
        J,
        force_basic_q_state,
        force_basic_q_offset,
        force_basic_q_count,
        k_global_out,
        f_global_out,
    )


fn force_beam_column3d_fiber_global_tangent_and_internal(
    elem_index: Int,
    x1: Float64,
    y1: Float64,
    z1: Float64,
    x2: Float64,
    y2: Float64,
    z2: Float64,
    u_elem_global: List[Float64],
    geom_transf: String,
    element_loads: List[ElementLoadInput],
    elem_load_offsets: List[Int],
    elem_load_pool: List[Int],
    load_scale: Float64,
    sec_def: FiberSection3dDef,
    fibers: List[FiberCell],
    uniaxial_defs: List[UniMaterialDef],
    mut uniaxial_states: List[UniMaterialState],
    elem_state_ids: List[Int],
    elem_state_offset: Int,
    elem_state_count: Int,
    integration: String,
    num_int_pts: Int,
    G: Float64,
    J: Float64,
    mut force_basic_q_state: List[Float64],
    force_basic_q_offset: Int,
    force_basic_q_count: Int,
    mut k_global_out: List[List[Float64]],
    mut f_global_out: List[Float64],
):
    var fibers_per_section = sec_def.fiber_count
    if elem_state_count != num_int_pts * fibers_per_section:
        abort("forceBeamColumn3d element state count mismatch")

    var predictor_state_count = 5 + 3 * num_int_pts
    if force_basic_q_count < predictor_state_count:
        abort("forceBeamColumn3d basic force state count mismatch")
    if (
        force_basic_q_offset < 0
        or force_basic_q_offset + predictor_state_count > len(force_basic_q_state)
    ):
        abort("forceBeamColumn3d basic force state out of range")

    var dx = x2 - x1
    var dy = y2 - y1
    var dz = z2 - z1
    var L = sqrt(dx * dx + dy * dy + dz * dz)
    if L == 0.0:
        abort("zero-length element")

    var R = _element_rotation(
        x1,
        y1,
        z1,
        x2,
        y2,
        z2,
        u_elem_global,
        geom_transf,
    )
    var u_local: List[Float64] = []
    _beam3d_transform_u_global_to_local(R, u_elem_global, u_local)

    var xis: List[Float64] = []
    var weights: List[Float64] = []
    beam_integration_rule(integration, num_int_pts, xis, weights)
    var section_load_axial: List[Float64] = []
    var section_load_my: List[Float64] = []
    var section_load_mz: List[Float64] = []
    section_load_axial.resize(num_int_pts, 0.0)
    section_load_my.resize(num_int_pts, 0.0)
    section_load_mz.resize(num_int_pts, 0.0)
    for ip in range(num_int_pts):
        var loads = beam3d_section_load_response(
            element_loads,
            elem_load_offsets,
            elem_load_pool,
            elem_index,
            load_scale,
            xis[ip] * L,
            L,
        )
        section_load_axial[ip] = loads[0]
        section_load_my[ip] = loads[1]
        section_load_mz[ip] = loads[2]
    var fixed_end = beam3d_basic_fixed_end_and_reactions(
        element_loads, elem_load_offsets, elem_load_pool, elem_index, load_scale, L
    )

    var v_basic_0: Float64
    var v_basic_1: Float64
    var v_basic_2: Float64
    var v_basic_3: Float64
    var v_basic_4: Float64
    var torsion_basic: Float64
    (v_basic_0, v_basic_1, v_basic_2, v_basic_3, v_basic_4, torsion_basic) = (
        _force_beam_column3d_basic_state(u_local, L)
    )

    var q0 = force_basic_q_state[force_basic_q_offset]
    var q1 = force_basic_q_state[force_basic_q_offset + 1]
    var q2 = force_basic_q_state[force_basic_q_offset + 2]
    var q3 = force_basic_q_state[force_basic_q_offset + 3]
    var q4 = force_basic_q_state[force_basic_q_offset + 4]
    var eps0_offset = force_basic_q_offset + 5
    var ky_offset = eps0_offset + num_int_pts
    var kz_offset = ky_offset + num_int_pts

    var elem_tol = 1.0e-8
    var section_tol = 1.0e-8
    var max_elem_iters = 80
    var max_section_iters = 80
    var converged = False

    var k_basic: List[List[Float64]] = []
    var f_basic: List[List[Float64]] = []
    var f_basic_copy: List[List[Float64]] = []
    var v_from: List[Float64] = []
    var residual: List[Float64] = []
    var dq: List[Float64] = []
    _ensure_zero_matrix(k_basic, 5, 5)

    for _ in range(max_elem_iters):
        _ensure_zero_matrix(f_basic, 5, 5)
        _ensure_zero_vector(v_from, 5)

        for ip in range(num_int_pts):
            var xi = xis[ip]
            var weight = weights[ip]
            var wL = weight * L

            var b_mi = xi - 1.0
            var b_mj = xi
            var axial_target = q0 + section_load_axial[ip]
            var moment_z_target = b_mi * q1 + b_mj * q2 + section_load_mz[ip]
            var moment_y_target = b_mi * q3 + b_mj * q4 + section_load_my[ip]

            var ip_state_offset = elem_state_offset + ip * fibers_per_section
            var eps0 = force_basic_q_state[eps0_offset + ip]
            var kappa_y = force_basic_q_state[ky_offset + ip]
            var kappa_z = force_basic_q_state[kz_offset + ip]
            var solved = _fiber_section3d_solve_for_force(
                sec_def,
                fibers,
                uniaxial_defs,
                uniaxial_states,
                elem_state_ids,
                ip_state_offset,
                fibers_per_section,
                axial_target,
                moment_y_target,
                moment_z_target,
                eps0,
                kappa_y,
                kappa_z,
                max_section_iters,
                section_tol,
            )
            var resp = solved[0]
            eps0 = solved[1]
            kappa_y = solved[2]
            kappa_z = solved[3]
            force_basic_q_state[eps0_offset + ip] = eps0
            force_basic_q_state[ky_offset + ip] = kappa_y
            force_basic_q_state[kz_offset + ip] = kappa_z

            var inv_sec = _invert_3x3_values(
                resp.k11,
                resp.k12,
                resp.k13,
                resp.k12,
                resp.k22,
                resp.k23,
                resp.k13,
                resp.k23,
                resp.k33,
            )
            if not inv_sec[0]:
                abort("forceBeamColumn3d singular section flexibility")

            v_from[0] += wL * eps0
            v_from[1] += wL * b_mi * kappa_z
            v_from[2] += wL * b_mj * kappa_z
            v_from[3] += wL * b_mi * kappa_y
            v_from[4] += wL * b_mj * kappa_y

            for a in range(5):
                var Ba_n = 0.0
                var Ba_my = 0.0
                var Ba_mz = 0.0
                if a == 0:
                    Ba_n = 1.0
                elif a == 1:
                    Ba_mz = b_mi
                elif a == 2:
                    Ba_mz = b_mj
                elif a == 3:
                    Ba_my = b_mi
                else:
                    Ba_my = b_mj
                for b in range(5):
                    var Bb_n = 0.0
                    var Bb_my = 0.0
                    var Bb_mz = 0.0
                    if b == 0:
                        Bb_n = 1.0
                    elif b == 1:
                        Bb_mz = b_mi
                    elif b == 2:
                        Bb_mz = b_mj
                    elif b == 3:
                        Bb_my = b_mi
                    else:
                        Bb_my = b_mj
                    var dEps = inv_sec[1] * Bb_n + inv_sec[4] * Bb_my + inv_sec[7] * Bb_mz
                    var dKy = inv_sec[2] * Bb_n + inv_sec[5] * Bb_my + inv_sec[8] * Bb_mz
                    var dKz = inv_sec[3] * Bb_n + inv_sec[6] * Bb_my + inv_sec[9] * Bb_mz
                    f_basic[a][b] += wL * (Ba_n * dEps + Ba_my * dKy + Ba_mz * dKz)

        _ensure_zero_matrix(f_basic_copy, 5, 5)
        for i in range(5):
            for j in range(5):
                f_basic_copy[i][j] = f_basic[i][j]
        if not _invert_5x5_into(f_basic_copy, k_basic):
            abort("forceBeamColumn3d singular element flexibility")

        _ensure_zero_vector(residual, 5)
        residual[0] = v_basic_0 - v_from[0]
        residual[1] = v_basic_1 - v_from[1]
        residual[2] = v_basic_2 - v_from[2]
        residual[3] = v_basic_3 - v_from[3]
        residual[4] = v_basic_4 - v_from[4]
        _ensure_zero_vector(dq, 5)
        for i in range(5):
            for j in range(5):
                dq[i] += k_basic[i][j] * residual[j]

        q0 += dq[0]
        q1 += dq[1]
        q2 += dq[2]
        q3 += dq[3]
        q4 += dq[4]

        var max_residual = 0.0
        var work_norm = 0.0
        for i in range(5):
            var abs_residual = abs(residual[i])
            if abs_residual > max_residual:
                max_residual = abs_residual
            work_norm += residual[i] * dq[i]
        if max_residual <= elem_tol or abs(work_norm) <= elem_tol:
            converged = True
            break
    if not converged:
        abort("forceBeamColumn3d element compatibility did not converge")

    force_basic_q_state[force_basic_q_offset] = q0
    force_basic_q_state[force_basic_q_offset + 1] = q1
    force_basic_q_state[force_basic_q_offset + 2] = q2
    force_basic_q_state[force_basic_q_offset + 3] = q3
    force_basic_q_state[force_basic_q_offset + 4] = q4

    var inv_L = 1.0 / L
    var k_local: List[List[Float64]] = []
    _ensure_zero_matrix(k_local, 12, 12)

    for a in range(12):
        for b in range(12):
            var a0 = _beam3d_local_basic_coeff(0, a, inv_L)
            var a1 = _beam3d_local_basic_coeff(1, a, inv_L)
            var a2 = _beam3d_local_basic_coeff(2, a, inv_L)
            var a3 = _beam3d_local_basic_coeff(3, a, inv_L)
            var a4 = _beam3d_local_basic_coeff(4, a, inv_L)
            var b0 = _beam3d_local_basic_coeff(0, b, inv_L)
            var b1 = _beam3d_local_basic_coeff(1, b, inv_L)
            var b2 = _beam3d_local_basic_coeff(2, b, inv_L)
            var b3 = _beam3d_local_basic_coeff(3, b, inv_L)
            var b4 = _beam3d_local_basic_coeff(4, b, inv_L)
            for i in range(5):
                for j in range(5):
                    var ai = a4
                    if i == 0:
                        ai = a0
                    elif i == 1:
                        ai = a1
                    elif i == 2:
                        ai = a2
                    elif i == 3:
                        ai = a3
                    var bj = b4
                    if j == 0:
                        bj = b0
                    elif j == 1:
                        bj = b1
                    elif j == 2:
                        bj = b2
                    elif j == 3:
                        bj = b3
                    k_local[a][b] += ai * k_basic[i][j] * bj
    if G > 0.0 and J > 0.0:
        var torsion_k = G * J / L
        for a in range(12):
            for b in range(12):
                k_local[a][b] += (
                    _beam3d_local_basic_coeff(5, a, inv_L)
                    * torsion_k
                    * _beam3d_local_basic_coeff(5, b, inv_L)
                )

    if geom_transf == "PDelta" or geom_transf == "Corotational":
        _beam3d_add_axial_geometric_stiffness(k_local, q0, L)

    var f_local: List[Float64] = []
    _ensure_zero_vector(f_local, 12)
    var q_basic_5 = 0.0
    if G > 0.0 and J > 0.0:
        q_basic_5 = (G * J / L) * torsion_basic
    for a in range(12):
        f_local[a] += _beam3d_local_basic_coeff(0, a, inv_L) * q0
        f_local[a] += _beam3d_local_basic_coeff(1, a, inv_L) * q1
        f_local[a] += _beam3d_local_basic_coeff(2, a, inv_L) * q2
        f_local[a] += _beam3d_local_basic_coeff(3, a, inv_L) * q3
        f_local[a] += _beam3d_local_basic_coeff(4, a, inv_L) * q4
        f_local[a] += _beam3d_local_basic_coeff(5, a, inv_L) * q_basic_5
    f_local[0] += fixed_end[5]
    f_local[1] += fixed_end[6]
    f_local[2] += fixed_end[8]
    f_local[7] += fixed_end[7]
    f_local[8] += fixed_end[9]

    var k_work: List[List[Float64]] = []
    _beam3d_transform_stiffness_local_to_global(R, k_local, k_work, k_global_out)
    _beam3d_transform_f_local_to_global(R, f_local, f_global_out)


fn force_beam_column3d_fiber_section_response(
    x1: Float64,
    y1: Float64,
    z1: Float64,
    x2: Float64,
    y2: Float64,
    z2: Float64,
    u_elem_global: List[Float64],
    geom_transf: String,
    sec_def: FiberSection3dDef,
    fibers: List[FiberCell],
    uniaxial_defs: List[UniMaterialDef],
    mut uniaxial_states: List[UniMaterialState],
    elem_state_ids: List[Int],
    elem_state_offset: Int,
    elem_state_count: Int,
    integration: String,
    num_int_pts: Int,
    G: Float64,
    J: Float64,
    mut force_basic_q_state: List[Float64],
    force_basic_q_offset: Int,
    force_basic_q_count: Int,
    section_no: Int,
    want_deformation: Bool,
) -> List[Float64]:
    if section_no < 1 or section_no > num_int_pts:
        abort("forceBeamColumn3d section index out of range")
    var k_dummy: List[List[Float64]] = []
    var f_dummy: List[Float64] = []
    force_beam_column3d_fiber_global_tangent_and_internal(
        x1,
        y1,
        z1,
        x2,
        y2,
        z2,
        u_elem_global,
        geom_transf,
        sec_def,
        fibers,
        uniaxial_defs,
        uniaxial_states,
        elem_state_ids,
        elem_state_offset,
        elem_state_count,
        integration,
        num_int_pts,
        G,
        J,
        force_basic_q_state,
        force_basic_q_offset,
        force_basic_q_count,
        k_dummy,
        f_dummy,
    )

    var ip = section_no - 1
    var q0 = force_basic_q_state[force_basic_q_offset]
    var q1 = force_basic_q_state[force_basic_q_offset + 1]
    var q2 = force_basic_q_state[force_basic_q_offset + 2]
    var q3 = force_basic_q_state[force_basic_q_offset + 3]
    var q4 = force_basic_q_state[force_basic_q_offset + 4]
    var eps0_offset = force_basic_q_offset + 5
    var ky_offset = eps0_offset + num_int_pts
    var kz_offset = ky_offset + num_int_pts
    var u_local: List[Float64] = []
    _beam3d_transform_u_global_to_local(
        _element_rotation(x1, y1, z1, x2, y2, z2, u_elem_global, geom_transf),
        u_elem_global,
        u_local,
    )
    var twist = (u_local[9] - u_local[3]) / sqrt(
        (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1)
    )

    if want_deformation:
        return [
            force_basic_q_state[eps0_offset + ip],
            force_basic_q_state[kz_offset + ip],
            force_basic_q_state[ky_offset + ip],
            twist,
        ]

    var xis: List[Float64] = []
    var weights: List[Float64] = []
    beam_integration_rule(integration, num_int_pts, xis, weights)
    var xi = xis[ip]
    var b_mi = xi - 1.0
    var b_mj = xi
    return [
        q0,
        b_mi * q1 + b_mj * q2,
        b_mi * q3 + b_mj * q4,
        G * J * twist,
    ]
