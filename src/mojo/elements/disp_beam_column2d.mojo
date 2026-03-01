from collections import List
from math import hypot
from os import abort

from elements.beam_loads import beam2d_basic_fixed_end_and_reactions
from elements.beam_integration import beam_integration_rule
from elements.utils import (
    _beam2d_transform_force_local_to_global_in_place,
    _beam2d_transform_stiffness_local_to_global_in_place,
    _beam2d_transform_u_global_to_local,
    _ensure_zero_matrix,
    _ensure_zero_vector,
)
from materials import UniMaterialDef, UniMaterialState
from solver.run_case.input_types import ElementLoadInput
from sections import (
    FiberCell,
    FiberSection2dDef,
    FiberSection2dResponse,
    fiber_section2d_set_trial_from_offset,
)


@always_inline
fn _beam2d_axial_shape(idx: Int, inv_L: Float64) -> Float64:
    if idx == 0:
        return -inv_L
    if idx == 3:
        return inv_L
    return 0.0


@always_inline
fn _beam2d_curvature_shape(
    idx: Int, inv_L: Float64, inv_L_sq: Float64, xi: Float64
) -> Float64:
    if idx == 1:
        return (-6.0 + 12.0 * xi) * inv_L_sq
    if idx == 2:
        return (-4.0 + 6.0 * xi) * inv_L
    if idx == 4:
        return (6.0 - 12.0 * xi) * inv_L_sq
    if idx == 5:
        return (-2.0 + 6.0 * xi) * inv_L
    return 0.0


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
    return fiber_section2d_set_trial_from_offset(
        sec_def,
        fibers,
        uniaxial_defs,
        section_state_ids,
        uniaxial_states,
        section_state_offset,
        section_state_count,
        eps0,
        kappa,
    )


fn disp_beam_column2d_global_tangent_and_internal(
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
    mut k_global_out: List[List[Float64]],
    mut f_global_out: List[Float64],
):
    var empty_element_loads: List[ElementLoadInput] = []
    var empty_elem_load_offsets: List[Int] = []
    var empty_elem_load_pool: List[Int] = []
    disp_beam_column2d_global_tangent_and_internal(
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
        k_global_out,
        f_global_out,
    )


fn disp_beam_column2d_global_tangent_and_internal(
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
    var inv_L = 1.0 / L
    var inv_L_sq = inv_L * inv_L
    var u_local: List[Float64] = []
    _beam2d_transform_u_global_to_local(c, s, u_elem_global, u_local)

    var fibers_per_section = sec_def.fiber_count
    var required_state_count = num_int_pts * fibers_per_section
    if elem_state_count != required_state_count:
        abort("dispBeamColumn2d element state count mismatch")
    var xis: List[Float64] = []
    var weights: List[Float64] = []
    beam_integration_rule(integration, num_int_pts, xis, weights)
    var has_element_loads = (
        load_scale != 0.0
        and elem_index + 1 < len(elem_load_offsets)
        and elem_load_offsets[elem_index] < elem_load_offsets[elem_index + 1]
    )

    _ensure_zero_matrix(k_global_out, 6, 6)
    _ensure_zero_vector(f_global_out, 6)
    var axial_force_avg = 0.0
    var q_basic_0 = 0.0
    var q_basic_1 = 0.0
    var q_basic_2 = 0.0

    for ip in range(num_int_pts):
        var xi = xis[ip]
        var weight = weights[ip]
        var b1 = _beam2d_curvature_shape(1, inv_L, inv_L_sq, xi)
        var b2 = _beam2d_curvature_shape(2, inv_L, inv_L_sq, xi)
        var b4 = _beam2d_curvature_shape(4, inv_L, inv_L_sq, xi)
        var b5 = _beam2d_curvature_shape(5, inv_L, inv_L_sq, xi)
        var eps0 = (-inv_L) * u_local[0] + inv_L * u_local[3]
        var kappa = b1 * u_local[1] + b2 * u_local[2] + b4 * u_local[4] + b5 * u_local[5]

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
        if has_element_loads:
            q_basic_0 += wL * resp.axial_force
            var xi6 = 6.0 * xi
            q_basic_1 += (xi6 - 4.0) * wL * resp.moment_z
            q_basic_2 += (xi6 - 2.0) * wL * resp.moment_z
        else:
            axial_force_avg += weight * resp.axial_force

        for a in range(6):
            var Ba_n = _beam2d_axial_shape(a, inv_L)
            var Ba_m = _beam2d_curvature_shape(a, inv_L, inv_L_sq, xi)
            if not has_element_loads:
                f_global_out[a] += wL * (Ba_n * resp.axial_force + Ba_m * resp.moment_z)
            for b in range(6):
                var Bb_n = _beam2d_axial_shape(b, inv_L)
                var Bb_m = _beam2d_curvature_shape(b, inv_L, inv_L_sq, xi)
                var dN = resp.k11 * Bb_n + resp.k12 * Bb_m
                var dM = resp.k12 * Bb_n + resp.k22 * Bb_m
                k_global_out[a][b] += wL * (Ba_n * dN + Ba_m * dM)

    if not has_element_loads:
        if geom_transf == "PDelta":
            var axial_over_l = axial_force_avg / L
            k_global_out[1][1] += axial_over_l
            k_global_out[4][4] += axial_over_l
            k_global_out[1][4] -= axial_over_l
            k_global_out[4][1] -= axial_over_l

            var pdelta_shear = (u_local[1] - u_local[4]) * axial_over_l
            f_global_out[1] += pdelta_shear
            f_global_out[4] -= pdelta_shear

        _beam2d_transform_stiffness_local_to_global_in_place(k_global_out, c, s)
        _beam2d_transform_force_local_to_global_in_place(f_global_out, c, s)
        return

    var fixed_end = beam2d_basic_fixed_end_and_reactions(
        element_loads, elem_load_offsets, elem_load_pool, elem_index, load_scale, L
    )
    q_basic_0 += fixed_end[0]
    q_basic_1 += fixed_end[1]
    q_basic_2 += fixed_end[2]

    if geom_transf == "PDelta":
        var axial_over_l = q_basic_0 / L
        k_global_out[1][1] += axial_over_l
        k_global_out[4][4] += axial_over_l
        k_global_out[1][4] -= axial_over_l
        k_global_out[4][1] -= axial_over_l

    var shear = (q_basic_1 + q_basic_2) / L
    _ensure_zero_vector(f_global_out, 6)
    f_global_out[0] = -q_basic_0 + fixed_end[3]
    f_global_out[1] = shear + fixed_end[4]
    f_global_out[2] = q_basic_1
    f_global_out[3] = q_basic_0
    f_global_out[4] = -shear + fixed_end[5]
    f_global_out[5] = q_basic_2
    if geom_transf == "PDelta":
        var pdelta_shear = (u_local[1] - u_local[4]) * q_basic_0 / L
        f_global_out[1] += pdelta_shear
        f_global_out[4] -= pdelta_shear

    _beam2d_transform_stiffness_local_to_global_in_place(k_global_out, c, s)
    _beam2d_transform_force_local_to_global_in_place(f_global_out, c, s)
