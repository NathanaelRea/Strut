from collections import List
from math import sqrt
from os import abort

from elements.beam_loads import beam3d_basic_fixed_end_and_reactions
from elements.beam_integration import beam_integration_cache_ensure, beam_integration_rule
from elements.force_beam_column3d import (
    Beam3dMat3,
    Beam3dMat12,
    Beam3dVec10,
    Beam3dVec12,
    ForceBeamColumn3dScratch,
    _beam3d_add_axial_geometric_force,
    _beam3d_add_axial_geometric_stiffness_flat,
    _beam3d_assemble_local_force_from_basic,
    _beam3d_transform_f_local_to_global_static,
    _beam3d_transform_stiffness_local_to_global_flat,
    _beam3d_transform_u_global_to_local_static,
    _element_rotation_into_static,
    _ensure_force_beam_column3d_geometry_cache,
)
from materials import UniMaterialDef, UniMaterialState
from solver.run_case.input_types import ElementLoadInput
from sections import (
    FiberCell,
    FiberSection3dDef,
    FiberSection3dResponse,
    fiber_section3d_set_trial_from_offset,
    fiber_section3d_set_trial_instance,
)


@always_inline
fn _section_kinematics_at_xi_static(
    u_local: Beam3dVec12, inv_L: Float64, inv_L2: Float64, xi: Float64
) -> (Float64, Float64, Float64, Float64, Float64, Float64):
    var t = (-6.0 + 12.0 * xi) * inv_L2
    var r_i = (4.0 - 6.0 * xi) * inv_L
    var r_j = (2.0 - 6.0 * xi) * inv_L
    var eps0 = inv_L * (u_local[6] - u_local[0])
    var kappa_y = (
        t * (u_local[2] - u_local[8]) + r_i * u_local[4] + r_j * u_local[10]
    )
    var kappa_z = (
        t * (u_local[1] - u_local[7]) - r_i * u_local[5] - r_j * u_local[11]
    )
    return (eps0, kappa_y, kappa_z, t, r_i, r_j)


@always_inline
fn _write_disp_k_local_row_pair(
    row: Int,
    neg_row: Int,
    c0: Float64,
    c1: Float64,
    c2: Float64,
    c4: Float64,
    c5: Float64,
    c10: Float64,
    c11: Float64,
    mut k_local: Beam3dMat12,
):
    var rb = row * 12
    k_local[rb] = c0
    k_local[rb + 1] = c1
    k_local[rb + 2] = c2
    k_local[rb + 3] = 0.0
    k_local[rb + 4] = c4
    k_local[rb + 5] = c5
    k_local[rb + 6] = -c0
    k_local[rb + 7] = -c1
    k_local[rb + 8] = -c2
    k_local[rb + 9] = 0.0
    k_local[rb + 10] = c10
    k_local[rb + 11] = c11

    var nrb = neg_row * 12
    k_local[nrb] = -c0
    k_local[nrb + 1] = -c1
    k_local[nrb + 2] = -c2
    k_local[nrb + 3] = 0.0
    k_local[nrb + 4] = -c4
    k_local[nrb + 5] = -c5
    k_local[nrb + 6] = c0
    k_local[nrb + 7] = c1
    k_local[nrb + 8] = c2
    k_local[nrb + 9] = 0.0
    k_local[nrb + 10] = -c10
    k_local[nrb + 11] = -c11


@always_inline
fn _write_disp_k_local_row_single(
    row: Int,
    c0: Float64,
    c1: Float64,
    c2: Float64,
    c4: Float64,
    c5: Float64,
    c10: Float64,
    c11: Float64,
    mut k_local: Beam3dMat12,
):
    var rb = row * 12
    k_local[rb] = c0
    k_local[rb + 1] = c1
    k_local[rb + 2] = c2
    k_local[rb + 3] = 0.0
    k_local[rb + 4] = c4
    k_local[rb + 5] = c5
    k_local[rb + 6] = -c0
    k_local[rb + 7] = -c1
    k_local[rb + 8] = -c2
    k_local[rb + 9] = 0.0
    k_local[rb + 10] = c10
    k_local[rb + 11] = c11


fn beam_column3d_fiber_global_tangent_and_internal(
    x1: Float64,
    y1: Float64,
    z1: Float64,
    x2: Float64,
    y2: Float64,
    z2: Float64,
    u_elem_global: List[Float64],
    geom_transf: String,
    mut sec_def: FiberSection3dDef,
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
    mut k_global_out: List[List[Float64]],
    mut f_global_out: List[Float64],
):
    beam_column3d_fiber_global_tangent_and_internal(
        x1,
        y1,
        z1,
        x2,
        y2,
        z2,
        u_elem_global,
        geom_transf,
        False,
        0.0,
        0.0,
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
        k_global_out,
        f_global_out,
    )


fn beam_column3d_fiber_global_tangent_and_internal(
    x1: Float64,
    y1: Float64,
    z1: Float64,
    x2: Float64,
    y2: Float64,
    z2: Float64,
    u_elem_global: List[Float64],
    geom_transf: String,
    has_vecxz: Bool,
    vecxz_x: Float64,
    vecxz_y: Float64,
    vecxz_z: Float64,
    mut sec_def: FiberSection3dDef,
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
    mut k_global_out: List[List[Float64]],
    mut f_global_out: List[Float64],
):
    var empty_element_loads: List[ElementLoadInput] = []
    var empty_elem_load_offsets: List[Int] = []
    var empty_elem_load_pool: List[Int] = []
    var scratch = ForceBeamColumn3dScratch()
    beam_column3d_fiber_global_tangent_and_internal(
        0,
        x1,
        y1,
        z1,
        x2,
        y2,
        z2,
        u_elem_global,
        geom_transf,
        has_vecxz,
        vecxz_x,
        vecxz_y,
        vecxz_z,
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
        scratch,
        k_global_out,
        f_global_out,
    )


fn beam_column3d_fiber_global_tangent_and_internal(
    elem_index: Int,
    x1: Float64,
    y1: Float64,
    z1: Float64,
    x2: Float64,
    y2: Float64,
    z2: Float64,
    u_elem_global: List[Float64],
    geom_transf: String,
    has_vecxz: Bool,
    vecxz_x: Float64,
    vecxz_y: Float64,
    vecxz_z: Float64,
    element_loads: List[ElementLoadInput],
    elem_load_offsets: List[Int],
    elem_load_pool: List[Int],
    load_scale: Float64,
    mut sec_def: FiberSection3dDef,
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
    mut k_global_out: List[List[Float64]],
    mut f_global_out: List[Float64],
):
    var scratch = ForceBeamColumn3dScratch()
    beam_column3d_fiber_global_tangent_and_internal(
        elem_index,
        x1,
        y1,
        z1,
        x2,
        y2,
        z2,
        u_elem_global,
        geom_transf,
        has_vecxz,
        vecxz_x,
        vecxz_y,
        vecxz_z,
        element_loads,
        elem_load_offsets,
        elem_load_pool,
        load_scale,
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
        scratch,
        k_global_out,
        f_global_out,
    )


fn beam_column3d_fiber_global_tangent_and_internal(
    elem_index: Int,
    x1: Float64,
    y1: Float64,
    z1: Float64,
    x2: Float64,
    y2: Float64,
    z2: Float64,
    u_elem_global: List[Float64],
    geom_transf: String,
    has_vecxz: Bool,
    vecxz_x: Float64,
    vecxz_y: Float64,
    vecxz_z: Float64,
    element_loads: List[ElementLoadInput],
    elem_load_offsets: List[Int],
    elem_load_pool: List[Int],
    load_scale: Float64,
    mut sec_def: FiberSection3dDef,
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
    mut scratch: ForceBeamColumn3dScratch,
    mut k_global_out: List[List[Float64]],
    mut f_global_out: List[Float64],
):
    var fibers_per_section = sec_def.fiber_count
    var required_state_count = num_int_pts * fibers_per_section
    if elem_state_count != required_state_count:
        abort("beamColumn3d element state count mismatch")

    var local_state_offset = elem_state_offset - sec_def.runtime_state_offset_base
    if (
        local_state_offset < 0
        or local_state_offset + elem_state_count > sec_def.runtime_state_count
    ):
        abort("beamColumn3d section states out of range")
    var elem_inst_base = local_state_offset // fibers_per_section

    _ensure_force_beam_column3d_geometry_cache(
        elem_index, x1, y1, z1, x2, y2, z2, has_vecxz, vecxz_x, vecxz_y, vecxz_z, scratch
    )
    var L: Float64
    var inv_L: Float64
    if elem_index >= 0 and elem_index < len(scratch.cached_length):
        L = scratch.cached_length[elem_index]
        inv_L = scratch.cached_inv_length[elem_index]
    else:
        var dx = x2 - x1
        var dy = y2 - y1
        var dz = z2 - z1
        L = sqrt(dx * dx + dy * dy + dz * dz)
        if L == 0.0:
            abort("zero-length element")
        inv_L = 1.0 / L
    var inv_L2 = inv_L * inv_L

    if geom_transf == "Corotational":
        _element_rotation_into_static(
            x1,
            y1,
            z1,
            x2,
            y2,
            z2,
            u_elem_global,
            geom_transf,
            has_vecxz,
            vecxz_x,
            vecxz_y,
            vecxz_z,
            scratch.R,
        )
    elif elem_index >= 0 and elem_index < len(scratch.rotation_cache):
        scratch.R = scratch.rotation_cache[elem_index]
    else:
        _element_rotation_into_static(
            x1,
            y1,
            z1,
            x2,
            y2,
            z2,
            u_elem_global,
            geom_transf,
            has_vecxz,
            vecxz_x,
            vecxz_y,
            vecxz_z,
            scratch.R,
        )
    _beam3d_transform_u_global_to_local_static(scratch.R, u_elem_global, scratch.u_local)
    beam_integration_cache_ensure(scratch.integration_cache, integration, num_int_pts)

    var has_element_loads = (
        elem_index >= 0
        and load_scale != 0.0
        and elem_index + 1 < len(elem_load_offsets)
        and elem_load_offsets[elem_index] < elem_load_offsets[elem_index + 1]
    )

    var s11 = 0.0
    var s12_t = 0.0
    var s12_i = 0.0
    var s12_j = 0.0
    var s13_t = 0.0
    var s13_i = 0.0
    var s13_j = 0.0
    var s22_tt = 0.0
    var s22_ti = 0.0
    var s22_tj = 0.0
    var s22_ii = 0.0
    var s22_ij = 0.0
    var s22_jj = 0.0
    var s33_tt = 0.0
    var s33_ti = 0.0
    var s33_tj = 0.0
    var s33_ii = 0.0
    var s33_ij = 0.0
    var s33_jj = 0.0
    var s23_tt = 0.0
    var s23_ti = 0.0
    var s23_tj = 0.0
    var s23_ii = 0.0
    var s23_ij = 0.0
    var s23_jj = 0.0

    var axial_force_avg = 0.0
    var f_n = 0.0
    var f1 = 0.0
    var f2 = 0.0
    var f4 = 0.0
    var f5 = 0.0
    var f10 = 0.0
    var f11 = 0.0

    var q_basic_0 = 0.0
    var q_basic_1 = 0.0
    var q_basic_2 = 0.0
    var q_basic_3 = 0.0
    var q_basic_4 = 0.0

    var xis_ptr = scratch.integration_cache.xis.unsafe_ptr()
    var weights_ptr = scratch.integration_cache.weights.unsafe_ptr()
    for ip in range(num_int_pts):
        var xi = xis_ptr[ip]
        var weight = weights_ptr[ip]
        var wL = weight * L
        var kin = _section_kinematics_at_xi_static(scratch.u_local, inv_L, inv_L2, xi)
        var eps0 = kin[0]
        var kappa_y = kin[1]
        var kappa_z = kin[2]
        var t = kin[3]
        var r_i = kin[4]
        var r_j = kin[5]

        var resp = fiber_section3d_set_trial_instance(
            sec_def, elem_inst_base + ip, eps0, kappa_y, kappa_z
        )
        if has_element_loads:
            var xi6 = 6.0 * xi
            q_basic_0 += wL * resp.axial_force
            q_basic_1 += (xi6 - 4.0) * wL * resp.moment_z
            q_basic_2 += (xi6 - 2.0) * wL * resp.moment_z
            q_basic_3 += (xi6 - 4.0) * wL * resp.moment_y
            q_basic_4 += (xi6 - 2.0) * wL * resp.moment_y
        else:
            axial_force_avg += weight * resp.axial_force
            var wN = wL * resp.axial_force
            var wMy = wL * resp.moment_y
            var wMz = wL * resp.moment_z
            f_n += wN
            f1 += wMz * t
            f2 += wMy * t
            f4 += wMy * r_i
            f5 -= wMz * r_i
            f10 += wMy * r_j
            f11 -= wMz * r_j

        var wk11 = wL * resp.k11
        var wk12 = wL * resp.k12
        var wk13 = wL * resp.k13
        var wk22 = wL * resp.k22
        var wk23 = wL * resp.k23
        var wk33 = wL * resp.k33

        var tt = t * t
        var ti = t * r_i
        var tj = t * r_j
        var ii = r_i * r_i
        var ij = r_i * r_j
        var jj = r_j * r_j

        s11 += wk11
        s12_t += wk12 * t
        s12_i += wk12 * r_i
        s12_j += wk12 * r_j
        s13_t += wk13 * t
        s13_i += wk13 * r_i
        s13_j += wk13 * r_j
        s22_tt += wk22 * tt
        s22_ti += wk22 * ti
        s22_tj += wk22 * tj
        s22_ii += wk22 * ii
        s22_ij += wk22 * ij
        s22_jj += wk22 * jj
        s33_tt += wk33 * tt
        s33_ti += wk33 * ti
        s33_tj += wk33 * tj
        s33_ii += wk33 * ii
        s33_ij += wk33 * ij
        s33_jj += wk33 * jj
        s23_tt += wk23 * tt
        s23_ti += wk23 * ti
        s23_tj += wk23 * tj
        s23_ii += wk23 * ii
        s23_ij += wk23 * ij
        s23_jj += wk23 * jj

    var k00 = s11 * inv_L2
    var k01 = -s13_t * inv_L
    var k02 = -s12_t * inv_L
    var k04 = -s12_i * inv_L
    var k05 = s13_i * inv_L
    var k0_10 = -s12_j * inv_L
    var k0_11 = s13_j * inv_L

    var k11 = s33_tt
    var k12 = s23_tt
    var k14 = s23_ti
    var k15 = -s33_ti
    var k1_10 = s23_tj
    var k1_11 = -s33_tj

    var k22 = s22_tt
    var k24 = s22_ti
    var k25 = -s23_ti
    var k2_10 = s22_tj
    var k2_11 = -s23_tj

    var k44 = s22_ii
    var k45 = -s23_ii
    var k4_10 = s22_ij
    var k4_11 = -s23_ij

    var k55 = s33_ii
    var k5_10 = -s23_ij
    var k5_11 = s33_ij

    var k10_10 = s22_jj
    var k10_11 = -s23_jj
    var k11_11 = s33_jj

    _write_disp_k_local_row_pair(
        0, 6, k00, k01, k02, k04, k05, k0_10, k0_11, scratch.k_local_flat
    )
    _write_disp_k_local_row_pair(
        1, 7, k01, k11, k12, k14, k15, k1_10, k1_11, scratch.k_local_flat
    )
    _write_disp_k_local_row_pair(
        2, 8, k02, k12, k22, k24, k25, k2_10, k2_11, scratch.k_local_flat
    )
    _write_disp_k_local_row_single(
        4, k04, k14, k24, k44, k45, k4_10, k4_11, scratch.k_local_flat
    )
    _write_disp_k_local_row_single(
        5, k05, k15, k25, k45, k55, k5_10, k5_11, scratch.k_local_flat
    )
    _write_disp_k_local_row_single(
        10, k0_10, k1_10, k2_10, k4_10, k5_10, k10_10, k10_11, scratch.k_local_flat
    )
    _write_disp_k_local_row_single(
        11, k0_11, k1_11, k2_11, k4_11, k5_11, k10_11, k11_11, scratch.k_local_flat
    )
    for c in range(12):
        scratch.k_local_flat[3 * 12 + c] = 0.0
        scratch.k_local_flat[9 * 12 + c] = 0.0

    var torsion_force = 0.0
    if G > 0.0 and J > 0.0:
        var torsion_k = G * J * inv_L
        scratch.k_local_flat[3 * 12 + 3] = torsion_k
        scratch.k_local_flat[3 * 12 + 9] = -torsion_k
        scratch.k_local_flat[9 * 12 + 3] = -torsion_k
        scratch.k_local_flat[9 * 12 + 9] = torsion_k
        torsion_force = torsion_k * (scratch.u_local[9] - scratch.u_local[3])

    var axial_force_for_geom = axial_force_avg
    if has_element_loads:
        var fe0: Float64
        var fe1: Float64
        var fe2: Float64
        var fe3: Float64
        var fe4: Float64
        var fe5: Float64
        var fe6: Float64
        var fe7: Float64
        var fe8: Float64
        var fe9: Float64
        if elem_index < len(scratch.load_valid) and scratch.load_valid[elem_index]:
            ref fe = scratch.fixed_end_cache[elem_index]
            fe0 = fe[0]
            fe1 = fe[1]
            fe2 = fe[2]
            fe3 = fe[3]
            fe4 = fe[4]
            fe5 = fe[5]
            fe6 = fe[6]
            fe7 = fe[7]
            fe8 = fe[8]
            fe9 = fe[9]
        else:
            var fixed = beam3d_basic_fixed_end_and_reactions(
                element_loads, elem_load_offsets, elem_load_pool, elem_index, load_scale, L
            )
            fe0 = fixed[0]
            fe1 = fixed[1]
            fe2 = fixed[2]
            fe3 = fixed[3]
            fe4 = fixed[4]
            fe5 = fixed[5]
            fe6 = fixed[6]
            fe7 = fixed[7]
            fe8 = fixed[8]
            fe9 = fixed[9]
            if elem_index < len(scratch.fixed_end_cache):
                var fe_cache = Beam3dVec10(fill=0.0)
                fe_cache[0] = fe0
                fe_cache[1] = fe1
                fe_cache[2] = fe2
                fe_cache[3] = fe3
                fe_cache[4] = fe4
                fe_cache[5] = fe5
                fe_cache[6] = fe6
                fe_cache[7] = fe7
                fe_cache[8] = fe8
                fe_cache[9] = fe9
                scratch.fixed_end_cache[elem_index] = fe_cache
                scratch.load_has_nonzero[elem_index] = True
                scratch.load_valid[elem_index] = True
        q_basic_0 += fe0
        q_basic_1 += fe1
        q_basic_2 += fe2
        q_basic_3 += fe3
        q_basic_4 += fe4
        axial_force_for_geom = q_basic_0
        _beam3d_assemble_local_force_from_basic(
            q_basic_0,
            q_basic_1,
            q_basic_2,
            q_basic_3,
            q_basic_4,
            torsion_force,
            inv_L,
            scratch.f_local,
        )
        scratch.f_local[0] += fe5
        scratch.f_local[1] += fe6
        scratch.f_local[2] += fe8
        scratch.f_local[7] += fe7
        scratch.f_local[8] += fe9
    else:
        var f0 = -f_n * inv_L
        scratch.f_local[0] = f0
        scratch.f_local[1] = f1
        scratch.f_local[2] = f2
        scratch.f_local[3] = -torsion_force
        scratch.f_local[4] = f4
        scratch.f_local[5] = f5
        scratch.f_local[6] = -f0
        scratch.f_local[7] = -f1
        scratch.f_local[8] = -f2
        scratch.f_local[9] = torsion_force
        scratch.f_local[10] = f10
        scratch.f_local[11] = f11

    if geom_transf == "PDelta" or geom_transf == "Corotational":
        _beam3d_add_axial_geometric_stiffness_flat(
            scratch.k_local_flat, axial_force_for_geom, inv_L
        )
        _beam3d_add_axial_geometric_force(
            scratch.f_local, scratch.u_local, axial_force_for_geom, inv_L
        )

    _beam3d_transform_stiffness_local_to_global_flat(
        scratch.R, scratch.k_local_flat, scratch.k_work_flat, k_global_out
    )
    _beam3d_transform_f_local_to_global_static(scratch.R, scratch.f_local, f_global_out)


fn beam_column3d_fiber_section_response(
    x1: Float64,
    y1: Float64,
    z1: Float64,
    x2: Float64,
    y2: Float64,
    z2: Float64,
    u_elem_global: List[Float64],
    geom_transf: String,
    mut sec_def: FiberSection3dDef,
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
    section_no: Int,
    want_deformation: Bool,
) -> List[Float64]:
    if section_no < 1 or section_no > num_int_pts:
        abort("beamColumn3d section index out of range")
    var fibers_per_section = sec_def.fiber_count
    if elem_state_count != num_int_pts * fibers_per_section:
        abort("beamColumn3d section response state count mismatch")

    var dx = x2 - x1
    var dy = y2 - y1
    var dz = z2 - z1
    var L = sqrt(dx * dx + dy * dy + dz * dz)
    if L == 0.0:
        abort("zero-length element")
    var inv_L = 1.0 / L
    var inv_L2 = inv_L * inv_L
    var R = Beam3dMat3(fill=0.0)
    var u_local = Beam3dVec12(fill=0.0)
    _element_rotation_into_static(
        x1, y1, z1, x2, y2, z2, u_elem_global, geom_transf, R
    )
    _beam3d_transform_u_global_to_local_static(R, u_elem_global, u_local)
    var xis: List[Float64] = []
    var weights: List[Float64] = []
    beam_integration_rule(integration, num_int_pts, xis, weights)

    var ip = section_no - 1
    var kin = _section_kinematics_at_xi_static(u_local, inv_L, inv_L2, xis[ip])
    var eps0 = kin[0]
    var kappa_y = kin[1]
    var kappa_z = kin[2]
    var twist = (u_local[9] - u_local[3]) * inv_L
    if want_deformation:
        return [eps0, kappa_z, -kappa_y, twist]

    var resp = fiber_section3d_set_trial_from_offset(
        sec_def,
        uniaxial_states,
        elem_state_offset + ip * fibers_per_section,
        fibers_per_section,
        eps0,
        kappa_y,
        kappa_z,
    )
    return [resp.axial_force, resp.moment_z, -resp.moment_y, G * J * twist]
