from collections import List
from math import sqrt
from os import abort

from elements.beam_loads import beam3d_basic_fixed_end_and_reactions
from elements.beam_integration import beam_integration_rule
from elements.utils import _cross, _dot, _normalize, _zero_matrix
from linalg import matmul, transpose
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


fn _beam3d_transform_matrix(R: List[List[Float64]]) -> List[List[Float64]]:
    var T = _zero_matrix(12, 12)
    var offsets = [0, 3, 6, 9]
    for b in range(4):
        var offset = offsets[b]
        for i in range(3):
            for j in range(3):
                T[offset + i][offset + j] = R[i][j]
    return T^


fn _beam3d_transform_u_global_to_local(
    T: List[List[Float64]], u_global: List[Float64]
) -> List[Float64]:
    var u_local: List[Float64] = []
    u_local.resize(12, 0.0)
    for i in range(12):
        var sum = 0.0
        for j in range(12):
            sum += T[i][j] * u_global[j]
        u_local[i] = sum
    return u_local^


fn _beam3d_transform_f_local_to_global(
    T: List[List[Float64]], f_local: List[Float64]
) -> List[Float64]:
    var f_global: List[Float64] = []
    f_global.resize(12, 0.0)
    for i in range(12):
        var sum = 0.0
        for j in range(12):
            sum += T[j][i] * f_local[j]
        f_global[i] = sum
    return f_global^


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
        abort("beamColumn3d fiber state count mismatch")
    if section_state_offset < 0 or section_state_offset + section_state_count > len(
        section_state_ids
    ):
        abort("beamColumn3d fiber section state out of range")

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
            abort("beamColumn3d fiber state index out of range")
        if cell.def_index < 0 or cell.def_index >= len(uniaxial_defs):
            abort("beamColumn3d fiber material definition out of range")
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


fn _section_kinematics_at_xi(
    u_local: List[Float64], L: Float64, xi: Float64
) -> (Float64, Float64, Float64):
    var inv_L = 1.0 / L
    var eps0 = (-inv_L) * u_local[0] + inv_L * u_local[6]
    var kappa_y = (
        ((-6.0 + 12.0 * xi) / (L * L)) * u_local[2]
        + ((4.0 - 6.0 * xi) / L) * u_local[4]
        + ((6.0 - 12.0 * xi) / (L * L)) * u_local[8]
        + ((2.0 - 6.0 * xi) / L) * u_local[10]
    )
    var kappa_z = (
        ((-6.0 + 12.0 * xi) / (L * L)) * u_local[1]
        + ((-4.0 + 6.0 * xi) / L) * u_local[5]
        + ((6.0 - 12.0 * xi) / (L * L)) * u_local[7]
        + ((-2.0 + 6.0 * xi) / L) * u_local[11]
    )
    return (eps0, kappa_y, kappa_z)


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
        return _beam3d_transform_matrix(
            _beam3d_rotation(
                x1 + u_elem_global[0],
                y1 + u_elem_global[1],
                z1 + u_elem_global[2],
                x2 + u_elem_global[6],
                y2 + u_elem_global[7],
                z2 + u_elem_global[8],
            )
        )
    return _beam3d_transform_matrix(_beam3d_rotation(x1, y1, z1, x2, y2, z2))


fn beam_column3d_fiber_global_tangent_and_internal(
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
    mut k_global_out: List[List[Float64]],
    mut f_global_out: List[Float64],
):
    var empty_element_loads: List[ElementLoadInput] = []
    var empty_elem_load_offsets: List[Int] = []
    var empty_elem_load_pool: List[Int] = []
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
    mut k_global_out: List[List[Float64]],
    mut f_global_out: List[Float64],
):
    var fibers_per_section = sec_def.fiber_count
    var required_state_count = num_int_pts * fibers_per_section
    if elem_state_count != required_state_count:
        abort("beamColumn3d element state count mismatch")

    var dx = x2 - x1
    var dy = y2 - y1
    var dz = z2 - z1
    var L = sqrt(dx * dx + dy * dy + dz * dz)
    if L == 0.0:
        abort("zero-length element")
    var T = _element_rotation(
        x1,
        y1,
        z1,
        x2,
        y2,
        z2,
        u_elem_global,
        geom_transf,
    )
    var u_local = _beam3d_transform_u_global_to_local(T, u_elem_global)

    var xis: List[Float64] = []
    var weights: List[Float64] = []
    beam_integration_rule(integration, num_int_pts, xis, weights)
    var fixed_end = beam3d_basic_fixed_end_and_reactions(
        element_loads, elem_load_offsets, elem_load_pool, elem_index, load_scale, L
    )
    var has_element_loads = (
        load_scale != 0.0
        and elem_index + 1 < len(elem_load_offsets)
        and elem_load_offsets[elem_index] < elem_load_offsets[elem_index + 1]
    )

    var k_local = _zero_matrix(12, 12)
    var q_basic: List[Float64] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    var f_local: List[Float64] = []
    f_local.resize(12, 0.0)
    var axial_force_avg = 0.0

    for ip in range(num_int_pts):
        var xi = xis[ip]
        var weight = weights[ip]
        var wL = weight * L
        var eps0: Float64
        var kappa_y: Float64
        var kappa_z: Float64
        (eps0, kappa_y, kappa_z) = _section_kinematics_at_xi(u_local, L, xi)

        var resp = _fiber_section3d_set_trial_from_offset(
            sec_def,
            fibers,
            uniaxial_defs,
            uniaxial_states,
            elem_state_ids,
            elem_state_offset + ip * fibers_per_section,
            fibers_per_section,
            eps0,
            kappa_y,
            kappa_z,
        )
        var xi6 = 6.0 * xi
        if has_element_loads:
            q_basic[0] += wL * resp.axial_force
            q_basic[1] += (xi6 - 4.0) * wL * resp.moment_z
            q_basic[2] += (xi6 - 2.0) * wL * resp.moment_z
            q_basic[3] += (xi6 - 4.0) * wL * resp.moment_y
            q_basic[4] += (xi6 - 2.0) * wL * resp.moment_y
        else:
            axial_force_avg += weight * resp.axial_force

        var b_eps: List[Float64] = [
            -1.0 / L,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0 / L,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
        var b_ky: List[Float64] = [
            0.0,
            0.0,
            (-6.0 + 12.0 * xi) / (L * L),
            0.0,
            (4.0 - 6.0 * xi) / L,
            0.0,
            0.0,
            0.0,
            (6.0 - 12.0 * xi) / (L * L),
            0.0,
            (2.0 - 6.0 * xi) / L,
            0.0,
        ]
        var b_kz: List[Float64] = [
            0.0,
            (-6.0 + 12.0 * xi) / (L * L),
            0.0,
            0.0,
            0.0,
            (-4.0 + 6.0 * xi) / L,
            0.0,
            (6.0 - 12.0 * xi) / (L * L),
            0.0,
            0.0,
            0.0,
            (-2.0 + 6.0 * xi) / L,
        ]

        for a in range(12):
            var Ba_n = b_eps[a]
            var Ba_my = b_ky[a]
            var Ba_mz = b_kz[a]
            if not has_element_loads:
                f_local[a] += wL * (
                    Ba_n * resp.axial_force
                    + Ba_my * resp.moment_y
                    + Ba_mz * resp.moment_z
                )
            for b in range(12):
                var Bb_n = b_eps[b]
                var Bb_my = b_ky[b]
                var Bb_mz = b_kz[b]
                var dN = resp.k11 * Bb_n + resp.k12 * Bb_my + resp.k13 * Bb_mz
                var dMy = resp.k12 * Bb_n + resp.k22 * Bb_my + resp.k23 * Bb_mz
                var dMz = resp.k13 * Bb_n + resp.k23 * Bb_my + resp.k33 * Bb_mz
                k_local[a][b] += wL * (Ba_n * dN + Ba_my * dMy + Ba_mz * dMz)
    if has_element_loads:
        q_basic[0] += fixed_end[0]
        q_basic[1] += fixed_end[1]
        q_basic[2] += fixed_end[2]
        q_basic[3] += fixed_end[3]
        q_basic[4] += fixed_end[4]
    if G > 0.0 and J > 0.0:
        var torsion_k = G * J / L
        k_local[3][3] += torsion_k
        k_local[3][9] -= torsion_k
        k_local[9][3] -= torsion_k
        k_local[9][9] += torsion_k
        var torsion_force = torsion_k * (u_local[9] - u_local[3])
        q_basic[5] = torsion_force
    var axial_force_for_geom = axial_force_avg
    if has_element_loads:
        axial_force_for_geom = q_basic[0]
    if geom_transf == "PDelta" or geom_transf == "Corotational":
        _beam3d_add_axial_geometric_stiffness(k_local, axial_force_for_geom, L)

    var a_matrix = _zero_matrix(6, 12)
    a_matrix[0][0] = -1.0
    a_matrix[0][6] = 1.0
    a_matrix[1][1] = 1.0 / L
    a_matrix[1][5] = 1.0
    a_matrix[1][7] = -1.0 / L
    a_matrix[2][1] = 1.0 / L
    a_matrix[2][7] = -1.0 / L
    a_matrix[2][11] = 1.0
    a_matrix[3][2] = -1.0 / L
    a_matrix[3][4] = 1.0
    a_matrix[3][8] = 1.0 / L
    a_matrix[4][2] = -1.0 / L
    a_matrix[4][8] = 1.0 / L
    a_matrix[4][10] = 1.0
    a_matrix[5][3] = -1.0
    a_matrix[5][9] = 1.0

    for a in range(12):
        var start_i = 5
        if has_element_loads:
            start_i = 0
        for i in range(start_i, 6):
            f_local[a] += a_matrix[i][a] * q_basic[i]
    if has_element_loads:
        f_local[0] += fixed_end[5]
        f_local[1] += fixed_end[6]
        f_local[2] += fixed_end[8]
        f_local[7] += fixed_end[7]
        f_local[8] += fixed_end[9]

    k_global_out = matmul(transpose(T), matmul(k_local, T))
    f_global_out = _beam3d_transform_f_local_to_global(T, f_local)


fn beam_column3d_fiber_section_response(
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
    var T = _element_rotation(
        x1,
        y1,
        z1,
        x2,
        y2,
        z2,
        u_elem_global,
        geom_transf,
    )
    var u_local = _beam3d_transform_u_global_to_local(T, u_elem_global)
    var xis: List[Float64] = []
    var weights: List[Float64] = []
    beam_integration_rule(integration, num_int_pts, xis, weights)

    var ip = section_no - 1
    var eps0: Float64
    var kappa_y: Float64
    var kappa_z: Float64
    (eps0, kappa_y, kappa_z) = _section_kinematics_at_xi(u_local, L, xis[ip])
    var twist = (u_local[9] - u_local[3]) / L
    if want_deformation:
        return [eps0, kappa_z, -kappa_y, twist]

    var resp = _fiber_section3d_set_trial_from_offset(
        sec_def,
        fibers,
        uniaxial_defs,
        uniaxial_states,
        elem_state_ids,
        elem_state_offset + ip * fibers_per_section,
        fibers_per_section,
        eps0,
        kappa_y,
        kappa_z,
    )
    return [resp.axial_force, resp.moment_z, -resp.moment_y, G * J * twist]
