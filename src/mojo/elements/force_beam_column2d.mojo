from collections import List
from math import atan2, hypot, sqrt
from os import abort
from time import perf_counter_ns

from elements.beam_loads import beam2d_basic_fixed_end_and_reactions, beam2d_section_load_response
from elements.beam_integration import BeamIntegrationCache, beam_integration_cache_ensure
from elements.hinge_radau2d import HingeRadau2d
from elements.force_beam_column_cbdi import cbdi_elastic_basic_state
from elements.utils import (
    _beam2d_transform_force_local_to_global_in_place,
    _beam2d_transform_stiffness_local_to_global_in_place,
    _beam2d_transform_u_global_to_local,
    _ensure_zero_matrix,
    _ensure_zero_vector,
    _invert_3x3_values,
)
from materials import (
    UniMaterialDef,
    UniMaterialState,
    uni_mat_initial_tangent,
    uniaxial_set_trial_strain,
)
from solver.profile import RuntimeProfileMetrics
from solver.run_case.input_types import ElementLoadInput, SectionInput
from sections import (
    FIBER_SECTION2D_BATCH_FLAG_CORRECTOR,
    FIBER_SECTION2D_BATCH_FLAG_PREDICTOR,
    FiberSection2dBatchPoint,
    FiberSection2dBatchProfile,
    FiberSection2dBatchResult,
    FiberCell,
    FiberSection2dDef,
    FiberSection2dResponse,
    fiber_section2d_batch_eval_same_def,
    fiber_section2d_batch_profile_note_definition,
    fiber_section2d_batch_profile_reset,
    fiber_section2d_set_trial_from_offset,
)


struct ForceBeamColumn2dScratch(Movable):
    var integration_cache: BeamIntegrationCache
    var section_load_axial: List[Float64]
    var section_load_moment: List[Float64]
    var u_local: List[Float64]
    var geometry_valid: List[Bool]
    var load_valid: List[Bool]
    var cached_length: List[Float64]
    var cached_inv_length: List[Float64]
    var cached_cos: List[Float64]
    var cached_sin: List[Float64]
    var basic_map_cache: List[List[Float64]]
    var section_load_axial_cache: List[List[Float64]]
    var section_load_moment_cache: List[List[Float64]]
    var fixed_end_cache: List[List[Float64]]
    var section_vs_eps0: List[Float64]
    var section_vs_kappa: List[Float64]
    var section_vs_subdivide_eps0: List[Float64]
    var section_vs_subdivide_kappa: List[Float64]
    var section_ssr_axial: List[Float64]
    var section_ssr_moment: List[Float64]
    var section_ssr_subdivide_axial: List[Float64]
    var section_ssr_subdivide_moment: List[Float64]
    var section_fs00: List[Float64]
    var section_fs01: List[Float64]
    var section_fs11: List[Float64]
    var section_fs_subdivide00: List[Float64]
    var section_fs_subdivide01: List[Float64]
    var section_fs_subdivide11: List[Float64]
    var section_batch_points: List[FiberSection2dBatchPoint]
    var section_batch_results: List[FiberSection2dBatchResult]
    var section_batch_profile: FiberSection2dBatchProfile

    fn __init__(out self):
        self.integration_cache = BeamIntegrationCache()
        self.section_load_axial = []
        self.section_load_moment = []
        self.u_local = []
        self.geometry_valid = []
        self.load_valid = []
        self.cached_length = []
        self.cached_inv_length = []
        self.cached_cos = []
        self.cached_sin = []
        self.basic_map_cache = []
        self.section_load_axial_cache = []
        self.section_load_moment_cache = []
        self.fixed_end_cache = []
        self.section_vs_eps0 = []
        self.section_vs_kappa = []
        self.section_vs_subdivide_eps0 = []
        self.section_vs_subdivide_kappa = []
        self.section_ssr_axial = []
        self.section_ssr_moment = []
        self.section_ssr_subdivide_axial = []
        self.section_ssr_subdivide_moment = []
        self.section_fs00 = []
        self.section_fs01 = []
        self.section_fs11 = []
        self.section_fs_subdivide00 = []
        self.section_fs_subdivide01 = []
        self.section_fs_subdivide11 = []
        self.section_batch_points = []
        self.section_batch_results = []
        self.section_batch_profile = FiberSection2dBatchProfile()


fn reset_force_beam_column2d_scratch(mut scratch: ForceBeamColumn2dScratch):
    scratch.integration_cache.is_valid = False
    scratch.geometry_valid = []
    scratch.load_valid = []
    scratch.cached_length = []
    scratch.cached_inv_length = []
    scratch.cached_cos = []
    scratch.cached_sin = []
    scratch.basic_map_cache = []
    scratch.section_load_axial_cache = []
    scratch.section_load_moment_cache = []
    scratch.fixed_end_cache = []
    scratch.section_vs_eps0 = []
    scratch.section_vs_kappa = []
    scratch.section_vs_subdivide_eps0 = []
    scratch.section_vs_subdivide_kappa = []
    scratch.section_ssr_axial = []
    scratch.section_ssr_moment = []
    scratch.section_ssr_subdivide_axial = []
    scratch.section_ssr_subdivide_moment = []
    scratch.section_fs00 = []
    scratch.section_fs01 = []
    scratch.section_fs11 = []
    scratch.section_fs_subdivide00 = []
    scratch.section_fs_subdivide01 = []
    scratch.section_fs_subdivide11 = []
    scratch.section_batch_points = []
    scratch.section_batch_results = []
    fiber_section2d_batch_profile_reset(scratch.section_batch_profile)


fn invalidate_force_beam_column2d_load_cache(mut scratch: ForceBeamColumn2dScratch):
    for i in range(len(scratch.load_valid)):
        scratch.load_valid[i] = False


fn _ensure_force_beam_column2d_section_history_capacity(
    mut scratch: ForceBeamColumn2dScratch, num_int_pts: Int
):
    scratch.section_vs_eps0.resize(num_int_pts, 0.0)
    scratch.section_vs_kappa.resize(num_int_pts, 0.0)
    scratch.section_vs_subdivide_eps0.resize(num_int_pts, 0.0)
    scratch.section_vs_subdivide_kappa.resize(num_int_pts, 0.0)
    scratch.section_ssr_axial.resize(num_int_pts, 0.0)
    scratch.section_ssr_moment.resize(num_int_pts, 0.0)
    scratch.section_ssr_subdivide_axial.resize(num_int_pts, 0.0)
    scratch.section_ssr_subdivide_moment.resize(num_int_pts, 0.0)
    scratch.section_fs00.resize(num_int_pts, 0.0)
    scratch.section_fs01.resize(num_int_pts, 0.0)
    scratch.section_fs11.resize(num_int_pts, 0.0)
    scratch.section_fs_subdivide00.resize(num_int_pts, 0.0)
    scratch.section_fs_subdivide01.resize(num_int_pts, 0.0)
    scratch.section_fs_subdivide11.resize(num_int_pts, 0.0)


fn _clear_force_beam_column2d_section_batch(mut scratch: ForceBeamColumn2dScratch):
    scratch.section_batch_points.clear()
    scratch.section_batch_results.clear()


fn _append_force_beam_column2d_section_batch_point(
    mut scratch: ForceBeamColumn2dScratch,
    section_state_offset: Int,
    section_state_count: Int,
    eps0: Float64,
    kappa: Float64,
    lagged_axial_force: Float64,
    lagged_moment_z: Float64,
    lagged_f00: Float64,
    lagged_f01: Float64,
    lagged_f11: Float64,
    flags: Int,
):
    scratch.section_batch_points.append(
        FiberSection2dBatchPoint(
            0,
            section_state_offset,
            section_state_count,
            eps0,
            kappa,
            lagged_axial_force,
            lagged_moment_z,
            lagged_f00,
            lagged_f01,
            lagged_f11,
            flags,
        )
    )


fn _evaluate_force_beam_column2d_section_batch(
    mut sec_def: FiberSection2dDef,
    hinges: HingeRadau2d,
    uniaxial_defs: List[UniMaterialDef],
    mut uniaxial_states: List[UniMaterialState],
    elem_state_ids: List[Int],
    elem_state_offset: Int,
    mut scratch: ForceBeamColumn2dScratch,
) -> Bool:
    if hinges.is_defined():
        for ip in range(len(scratch.section_batch_points)):
            ref point = scratch.section_batch_points[ip]
            var ea = hinges.elastic_ea
            var ei = hinges.elastic_ei
            var axial_force = ea * point.eps0
            var bending_moment = ei * point.kappa
            if ip == 0 or ip == 5:
                var material = hinges.material_indices(ip)
                var offset = elem_state_offset + (0 if ip == 0 else 2)
                ref axial = uniaxial_states[elem_state_ids[offset]]
                ref flexural = uniaxial_states[elem_state_ids[offset + 1]]
                uniaxial_set_trial_strain(uniaxial_defs[material[0]], axial, point.eps0)
                uniaxial_set_trial_strain(uniaxial_defs[material[1]], flexural, point.kappa)
                axial_force = axial.sig_t
                bending_moment = flexural.sig_t
                ea = axial.tangent_t
                ei = flexural.tangent_t
            if ea == 0.0 or ei == 0.0:
                return False
            scratch.section_batch_results.append(
                FiberSection2dBatchResult(
                    FiberSection2dResponse(axial_force, bending_moment, ea, 0.0, ei),
                    True,
                    1.0 / ea,
                    0.0,
                    1.0 / ei,
                )
            )
        return True
    return fiber_section2d_batch_eval_same_def(
        sec_def,
        uniaxial_states,
        scratch.section_batch_points,
        scratch.section_batch_results,
        scratch.section_batch_profile,
    )


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
    mut sec_def: FiberSection2dDef,
    mut uniaxial_states: List[UniMaterialState],
    section_state_offset: Int,
    section_state_count: Int,
    eps0: Float64,
    kappa: Float64,
) -> FiberSection2dResponse:
    return fiber_section2d_set_trial_from_offset(
        sec_def,
        uniaxial_states,
        section_state_offset,
        section_state_count,
        eps0,
        kappa,
    )


fn _fiber_section2d_initial_flexibility(
    sec_def: FiberSection2dDef,
) -> (Bool, Float64, Float64, Float64):
    if not sec_def.initial_flex_valid:
        return (False, 0.0, 0.0, 0.0)
    return (
        True,
        sec_def.initial_f00,
        sec_def.initial_f01,
        sec_def.initial_f11,
    )


fn _fiber_section2d_response_flexibility(
    resp: FiberSection2dResponse,
) -> (Bool, Float64, Float64, Float64):
    var det = resp.k11 * resp.k22 - resp.k12 * resp.k12
    if abs(det) <= 1.0e-40:
        return (False, 0.0, 0.0, 0.0)
    var inv_det = 1.0 / det
    return (True, resp.k22 * inv_det, -resp.k12 * inv_det, resp.k11 * inv_det)


fn _fiber_section2d_all_materials_elastic(sec_def: FiberSection2dDef) -> Bool:
    return sec_def.nonlinear_count == 0


fn _restore_force_beam_column2d_predictor_state(
    mut force_basic_q_state: List[Float64],
    force_basic_q_offset: Int,
    num_int_pts: Int,
    q0: Float64,
    q1: Float64,
    q2: Float64,
    section_eps0: List[Float64],
    section_kappa: List[Float64],
):
    force_basic_q_state[force_basic_q_offset] = q0
    force_basic_q_state[force_basic_q_offset + 1] = q1
    force_basic_q_state[force_basic_q_offset + 2] = q2
    var eps0_offset = force_basic_q_offset + 3
    var kappa_offset = eps0_offset + num_int_pts
    for ip in range(num_int_pts):
        force_basic_q_state[eps0_offset + ip] = section_eps0[ip]
        force_basic_q_state[kappa_offset + ip] = section_kappa[ip]


@always_inline
fn _set_force_beam_column2d_trial_basic_deformation(
    mut force_basic_q_state: List[Float64],
    basic_state_offset: Int,
    v0: Float64,
    v1: Float64,
    v2: Float64,
):
    force_basic_q_state[basic_state_offset] = v0
    force_basic_q_state[basic_state_offset + 1] = v1
    force_basic_q_state[basic_state_offset + 2] = v2


fn _ensure_force_beam_column2d_cache_slot(mut scratch: ForceBeamColumn2dScratch, elem_index: Int):
    if elem_index < 0:
        return
    var needed = elem_index + 1
    if len(scratch.geometry_valid) < needed:
        scratch.geometry_valid.resize(needed, False)
        scratch.load_valid.resize(needed, False)
        scratch.cached_length.resize(needed, 0.0)
        scratch.cached_inv_length.resize(needed, 0.0)
        scratch.cached_cos.resize(needed, 0.0)
        scratch.cached_sin.resize(needed, 0.0)
        while len(scratch.basic_map_cache) < needed:
            scratch.basic_map_cache.append(List[Float64]())
        while len(scratch.section_load_axial_cache) < needed:
            scratch.section_load_axial_cache.append(List[Float64]())
        while len(scratch.section_load_moment_cache) < needed:
            scratch.section_load_moment_cache.append(List[Float64]())
        while len(scratch.fixed_end_cache) < needed:
            scratch.fixed_end_cache.append(List[Float64]())


fn _ensure_force_beam_column2d_geometry_cache(
    elem_index: Int,
    x1: Float64,
    y1: Float64,
    x2: Float64,
    y2: Float64,
    mut scratch: ForceBeamColumn2dScratch,
):
    if elem_index < 0:
        return
    _ensure_force_beam_column2d_cache_slot(scratch, elem_index)
    if scratch.geometry_valid[elem_index]:
        return
    var dx = x2 - x1
    var dy = y2 - y1
    var L = hypot(dx, dy)
    if L == 0.0:
        abort("zero-length element")
    var inv_L = 1.0 / L
    scratch.cached_length[elem_index] = L
    scratch.cached_inv_length[elem_index] = inv_L
    scratch.cached_cos[elem_index] = dx * inv_L
    scratch.cached_sin[elem_index] = dy * inv_L

    var basic_map = scratch.basic_map_cache[elem_index].copy()
    basic_map.resize(18, 0.0)
    for a in range(6):
        var a0: Float64
        var a1: Float64
        var a2: Float64
        (a0, a1, a2) = _beam2d_local_basic_map(a, inv_L)
        var offset = 3 * a
        basic_map[offset] = a0
        basic_map[offset + 1] = a1
        basic_map[offset + 2] = a2
    scratch.basic_map_cache[elem_index] = basic_map^
    scratch.geometry_valid[elem_index] = True


fn _ensure_force_beam_column2d_load_cache(
    elem_index: Int,
    num_int_pts: Int,
    element_loads: List[ElementLoadInput],
    elem_load_offsets: List[Int],
    elem_load_pool: List[Int],
    load_scale: Float64,
    mut scratch: ForceBeamColumn2dScratch,
):
    if elem_index < 0:
        return
    _ensure_force_beam_column2d_cache_slot(scratch, elem_index)
    if scratch.load_valid[elem_index]:
        return
    var L = scratch.cached_length[elem_index]
    var axial = scratch.section_load_axial_cache[elem_index].copy()
    var moment = scratch.section_load_moment_cache[elem_index].copy()
    axial.resize(num_int_pts, 0.0)
    moment.resize(num_int_pts, 0.0)
    for ip in range(num_int_pts):
        var loads = beam2d_section_load_response(
            element_loads,
            elem_load_offsets,
            elem_load_pool,
            elem_index,
            load_scale,
            scratch.integration_cache.xis[ip] * L,
            L,
        )
        axial[ip] = loads[0]
        moment[ip] = loads[1]
    scratch.section_load_axial_cache[elem_index] = axial^
    scratch.section_load_moment_cache[elem_index] = moment^

    var fixed_end = scratch.fixed_end_cache[elem_index].copy()
    fixed_end.resize(6, 0.0)
    var fixed = beam2d_basic_fixed_end_and_reactions(
        element_loads, elem_load_offsets, elem_load_pool, elem_index, load_scale, L
    )
    fixed_end[0] = fixed[0]
    fixed_end[1] = fixed[1]
    fixed_end[2] = fixed[2]
    fixed_end[3] = fixed[3]
    fixed_end[4] = fixed[4]
    fixed_end[5] = fixed[5]
    scratch.fixed_end_cache[elem_index] = fixed_end^
    scratch.load_valid[elem_index] = True


fn _force_beam_column2d_corotational_basic_to_global(
    L: Float64,
    c: Float64,
    s: Float64,
    u_local: List[Float64],
    q0: Float64,
    q1: Float64,
    q2: Float64,
    k00: Float64,
    k01: Float64,
    k02: Float64,
    k10: Float64,
    k11: Float64,
    k12: Float64,
    k22: Float64,
    fixed_end: (Float64, Float64, Float64, Float64, Float64, Float64),
    mut k_global_out: List[List[Float64]],
    mut f_global_out: List[Float64],
):
    var dulx = u_local[3] - u_local[0]
    var duly = u_local[4] - u_local[1]
    var Lx = L + dulx
    var Ly = duly
    var Ln = sqrt(Lx * Lx + Ly * Ly)
    if Ln == 0.0:
        abort("zero-length element")

    var cos_alpha = Lx / Ln
    var sin_alpha = Ly / Ln
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

    _ensure_zero_matrix(k_global_out, 6, 6)
    for i in range(6):
        var t0i = Tbl[0][i]
        var t1i = Tbl[1][i]
        var t2i = Tbl[2][i]
        for j in range(6):
            var t0j = Tbl[0][j]
            var t1j = Tbl[1][j]
            var t2j = Tbl[2][j]
            k_global_out[i][j] = (
                k00 * t0i * t0j
                + k01 * t0i * t1j
                + k02 * t0i * t2j
                + k10 * t1i * t0j
                + k11 * t1i * t1j
                + k12 * t1i * t2j
                + k02 * t2i * t0j
                + k12 * t2i * t1j
                + k22 * t2i * t2j
            )

    var s2 = sin_alpha * sin_alpha
    var c2 = cos_alpha * cos_alpha
    var cs = sin_alpha * cos_alpha

    var kg0: List[List[Float64]] = []
    _ensure_zero_matrix(kg0, 6, 6)
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

    var kg12: List[List[Float64]] = []
    _ensure_zero_matrix(kg12, 6, 6)
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

    var scale0 = q0 / Ln
    var scale12 = (q1 + q2) / (Ln * Ln)
    for i in range(6):
        for j in range(6):
            k_global_out[i][j] += kg0[i][j] * scale0 + kg12[i][j] * scale12

    _ensure_zero_vector(f_global_out, 6)
    for i in range(6):
        f_global_out[i] = Tbl[0][i] * q0 + Tbl[1][i] * q1 + Tbl[2][i] * q2

    # Distributed-load basic reactions are already folded into q*, but the
    # extra local shear/end-force terms still need to be carried through.
    f_global_out[0] += fixed_end[3]
    f_global_out[1] += fixed_end[4]
    f_global_out[4] += fixed_end[5]

    _beam2d_transform_stiffness_local_to_global_in_place(k_global_out, c, s)
    _beam2d_transform_force_local_to_global_in_place(f_global_out, c, s)


fn _force_beam_column2d_exact_elastic_state(
    L: Float64,
    xis: List[Float64],
    weights: List[Float64],
    section_load_axial: List[Float64],
    section_load_moment: List[Float64],
    mut sec_def: FiberSection2dDef,
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
    var init_flex = _fiber_section2d_initial_flexibility(sec_def)
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
            uniaxial_states,
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


fn _force_beam_column2d_initial_basic_tangent(
    L: Float64,
    xis: List[Float64],
    weights: List[Float64],
    mut sec_def: FiberSection2dDef,
    fibers: List[FiberCell],
    uniaxial_defs: List[UniMaterialDef],
) -> (Bool, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64,):
    var init_flex = _fiber_section2d_initial_flexibility(sec_def)
    if not init_flex[0]:
        return (False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

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

    for ip in range(len(xis)):
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

    var k_inv = _invert_3x3_values(f00, f01, f02, f10, f11, f12, f20, f21, f22)
    if not k_inv[0]:
        return (False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return (
        True,
        k_inv[1],
        k_inv[2],
        k_inv[3],
        k_inv[4],
        k_inv[5],
        k_inv[6],
        k_inv[8],
        k_inv[9],
    )


fn force_beam_column2d_fiber_state_count(num_points: Int) -> Int:
    # Forces, section deformations, previous basic displacement, tangent, initialized.
    return 3 + 2 * num_points + 3 + 7 + 1


fn force_beam_column2d_basic_displacement(
    length: Float64, local_u: List[Float64], corotational: Bool
) -> (Float64, Float64, Float64):
    var axial = local_u[3] - local_u[0]
    var transverse = local_u[4] - local_u[1]
    var rotation = transverse / length
    if corotational:
        var chord_x = length + axial
        axial = sqrt(chord_x * chord_x + transverse * transverse) - length
        rotation = atan2(transverse, chord_x)
    return (axial, local_u[2] - rotation, local_u[5] - rotation)


fn force_beam_column2d_initial_global_stiffness(
    x1: Float64,
    y1: Float64,
    x2: Float64,
    y2: Float64,
    integration: String,
    num_points: Int,
    mut section: FiberSection2dDef,
    fibers: List[FiberCell],
    uniaxial_defs: List[UniMaterialDef],
    mut scratch: ForceBeamColumn2dScratch,
    mut stiffness: List[List[Float64]],
):
    var length = hypot(x2 - x1, y2 - y1)
    beam_integration_cache_ensure(scratch.integration_cache, integration, num_points)
    var basic = _force_beam_column2d_initial_basic_tangent(
        length,
        scratch.integration_cache.xis,
        scratch.integration_cache.weights,
        section,
        fibers,
        uniaxial_defs,
    )
    if not basic[0]:
        abort("forceBeamColumn2d initial flexibility is singular")
    var zero = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    var force: List[Float64] = []
    _force_beam_column2d_corotational_basic_to_global(
        length,
        (x2 - x1) / length,
        (y2 - y1) / length,
        zero,
        0.0,
        0.0,
        0.0,
        basic[1],
        basic[2],
        basic[3],
        basic[4],
        basic[5],
        basic[6],
        basic[8],
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        stiffness,
        force,
    )


fn _force_beam_column2d_try_increment(
    L: Float64,
    mut sec_def: FiberSection2dDef,
    hinges: HingeRadau2d,
    fibers: List[FiberCell],
    uniaxial_defs: List[UniMaterialDef],
    mut uniaxial_states: List[UniMaterialState],
    elem_state_ids: List[Int],
    elem_state_offset: Int,
    fibers_per_section: Int,
    num_int_pts: Int,
    mut force_basic_q_state: List[Float64],
    force_basic_q_offset: Int,
    mut scratch: ForceBeamColumn2dScratch,
    base_v0: Float64,
    base_v1: Float64,
    base_v2: Float64,
    dv_trial0: Float64,
    dv_trial1: Float64,
    dv_trial2: Float64,
    use_initial_section_flexibility: Int,
    mut runtime_metrics: RuntimeProfileMetrics,
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
    ref xis = scratch.integration_cache.xis
    ref weights = scratch.integration_cache.weights
    ref section_load_axial = scratch.section_load_axial
    ref section_load_moment = scratch.section_load_moment

    var elem_tol = hinges.tolerance if hinges.is_defined() else 1.0e-12
    var max_elem_iters = hinges.max_iters if hinges.is_defined() else 10

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

    var initial_sec_flex = _fiber_section2d_initial_flexibility(sec_def)
    if not initial_sec_flex[0] and not hinges.is_defined():
        return (False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    _ensure_force_beam_column2d_section_history_capacity(scratch, num_int_pts)
    fiber_section2d_batch_profile_note_definition(
        sec_def,
        num_int_pts,
        scratch.section_batch_profile,
    )

    _clear_force_beam_column2d_section_batch(scratch)
    for ip in range(num_int_pts):
        var eps0 = force_basic_q_state[eps0_offset + ip]
        var kappa = force_basic_q_state[kappa_offset + ip]
        scratch.section_vs_eps0[ip] = eps0
        scratch.section_vs_kappa[ip] = kappa

        var ip_state_offset = elem_state_offset + ip * fibers_per_section
        _append_force_beam_column2d_section_batch_point(
            scratch,
            ip_state_offset,
            fibers_per_section,
            eps0,
            kappa,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            FIBER_SECTION2D_BATCH_FLAG_PREDICTOR,
        )
    var t_predictor_eval_start = 0
    if runtime_metrics.enabled:
        t_predictor_eval_start = Int(perf_counter_ns())
    if not _evaluate_force_beam_column2d_section_batch(
        sec_def,
        hinges,
        uniaxial_defs,
        uniaxial_states,
        elem_state_ids,
        elem_state_offset,
        scratch,
    ):
        return (False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if runtime_metrics.enabled:
        runtime_metrics.predictor_section_eval_ns += Int(perf_counter_ns()) - t_predictor_eval_start
        runtime_metrics.section_evaluations += len(scratch.section_batch_points)
    for ip in range(num_int_pts):
        var resp_trial = scratch.section_batch_results[ip]
        scratch.section_ssr_axial[ip] = resp_trial.response.axial_force
        scratch.section_ssr_moment[ip] = resp_trial.response.moment_z
        if not resp_trial.flexibility_valid:
            return (False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        scratch.section_fs00[ip] = resp_trial.f00
        scratch.section_fs01[ip] = resp_trial.f01
        scratch.section_fs11[ip] = resp_trial.f11
        scratch.section_vs_subdivide_eps0[ip] = scratch.section_vs_eps0[ip]
        scratch.section_vs_subdivide_kappa[ip] = scratch.section_vs_kappa[ip]
        scratch.section_ssr_subdivide_axial[ip] = resp_trial.response.axial_force
        scratch.section_ssr_subdivide_moment[ip] = resp_trial.response.moment_z
        scratch.section_fs_subdivide00[ip] = resp_trial.f00
        scratch.section_fs_subdivide01[ip] = resp_trial.f01
        scratch.section_fs_subdivide11[ip] = resp_trial.f11

    # Start each local solve from the current tangent predictor, matching
    # OpenSees' SeTrial = Se + kv * dvTrial initialization.
    var predictor_f00 = 0.0
    var predictor_f01 = 0.0
    var predictor_f02 = 0.0
    var predictor_f10 = 0.0
    var predictor_f11 = 0.0
    var predictor_f12 = 0.0
    var predictor_f20 = 0.0
    var predictor_f21 = 0.0
    var predictor_f22 = 0.0
    var t_flex_start = 0
    if runtime_metrics.enabled:
        t_flex_start = Int(perf_counter_ns())
    for ip in range(num_int_pts):
        var xi = xis[ip]
        var weight = weights[ip]
        var wL = weight * L
        var b_mi = xi - 1.0
        var b_mj = xi
        var sec_f00 = scratch.section_fs00[ip]
        var sec_f01 = scratch.section_fs01[ip]
        var sec_f10 = sec_f01
        var sec_f11 = scratch.section_fs11[ip]
        predictor_f00 += wL * sec_f00
        predictor_f01 += wL * sec_f01 * b_mi
        predictor_f02 += wL * sec_f01 * b_mj
        predictor_f10 += wL * b_mi * sec_f10
        predictor_f11 += wL * b_mi * sec_f11 * b_mi
        predictor_f12 += wL * b_mi * sec_f11 * b_mj
        predictor_f20 += wL * b_mj * sec_f10
        predictor_f21 += wL * b_mj * sec_f11 * b_mi
        predictor_f22 += wL * b_mj * sec_f11 * b_mj
    if runtime_metrics.enabled:
        runtime_metrics.local_flexibility_accumulation_ns += Int(perf_counter_ns()) - t_flex_start
    var t_solve3_start = 0
    if runtime_metrics.enabled:
        t_solve3_start = Int(perf_counter_ns())
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
    if runtime_metrics.enabled:
        runtime_metrics.local_3x3_solve_ns += Int(perf_counter_ns()) - t_solve3_start
    if predictor_k[0]:
        q0 += predictor_k[1] * dv_trial0 + predictor_k[2] * dv_trial1 + predictor_k[3] * dv_trial2
        q1 += predictor_k[4] * dv_trial0 + predictor_k[5] * dv_trial1 + predictor_k[6] * dv_trial2
        q2 += predictor_k[7] * dv_trial0 + predictor_k[8] * dv_trial1 + predictor_k[9] * dv_trial2

    var num_elem_iters = max_elem_iters
    if use_initial_section_flexibility == 1:
        # OpenSees gives the all-initial-tangent scheme a 10x iteration budget.
        num_elem_iters = 10 * max_elem_iters

    for elem_iter in range(num_elem_iters):
        if runtime_metrics.enabled:
            runtime_metrics.local_force_beam_column_iterations += 1
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

        var use_initial_flex = use_initial_section_flexibility == 1 or (
            use_initial_section_flexibility == 2 and elem_iter == 0
        )
        _clear_force_beam_column2d_section_batch(scratch)
        for ip in range(num_int_pts):
            var xi = xis[ip]
            var b_mi = xi - 1.0
            var b_mj = xi
            var ss0 = q0 + section_load_axial[ip]
            var ss1 = b_mi * q1 + b_mj * q2 + section_load_moment[ip]
            var dss0 = ss0 - scratch.section_ssr_subdivide_axial[ip]
            var dss1 = ss1 - scratch.section_ssr_subdivide_moment[ip]

            var solve_f00: Float64
            var solve_f01: Float64
            var solve_f11: Float64
            if use_initial_flex and hinges.is_defined():
                var ea = hinges.elastic_ea
                var ei = hinges.elastic_ei
                if ip == 0 or ip == 5:
                    var material = hinges.material_indices(ip)
                    ea = uni_mat_initial_tangent(uniaxial_defs[material[0]])
                    ei = uni_mat_initial_tangent(uniaxial_defs[material[1]])
                solve_f00 = 1.0 / ea
                solve_f01 = 0.0
                solve_f11 = 1.0 / ei
            elif use_initial_flex:
                solve_f00 = initial_sec_flex[1]
                solve_f01 = initial_sec_flex[2]
                solve_f11 = initial_sec_flex[3]
            else:
                solve_f00 = scratch.section_fs_subdivide00[ip]
                solve_f01 = scratch.section_fs_subdivide01[ip]
                solve_f11 = scratch.section_fs_subdivide11[ip]
            var dvs0 = solve_f00 * dss0 + solve_f01 * dss1
            var dvs1 = solve_f01 * dss0 + solve_f11 * dss1

            var sec_vs0 = scratch.section_vs_subdivide_eps0[ip] + dvs0
            var sec_vs1 = scratch.section_vs_subdivide_kappa[ip] + dvs1
            var ip_state_offset = elem_state_offset + ip * fibers_per_section
            _append_force_beam_column2d_section_batch_point(
                scratch,
                ip_state_offset,
                fibers_per_section,
                sec_vs0,
                sec_vs1,
                scratch.section_ssr_subdivide_axial[ip],
                scratch.section_ssr_subdivide_moment[ip],
                scratch.section_fs_subdivide00[ip],
                scratch.section_fs_subdivide01[ip],
                scratch.section_fs_subdivide11[ip],
                FIBER_SECTION2D_BATCH_FLAG_CORRECTOR,
            )
        var t_corrector_eval_start = 0
        if runtime_metrics.enabled:
            t_corrector_eval_start = Int(perf_counter_ns())
        if not _evaluate_force_beam_column2d_section_batch(
            sec_def,
            hinges,
            uniaxial_defs,
            uniaxial_states,
            elem_state_ids,
            elem_state_offset,
            scratch,
        ):
            return (False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        if runtime_metrics.enabled:
            runtime_metrics.corrector_section_eval_ns += (
                Int(perf_counter_ns()) - t_corrector_eval_start
            )
            runtime_metrics.section_evaluations += len(scratch.section_batch_points)
        t_flex_start = 0
        if runtime_metrics.enabled:
            t_flex_start = Int(perf_counter_ns())
        for ip in range(num_int_pts):
            var xi = xis[ip]
            var weight = weights[ip]
            var wL = weight * L
            var b_mi = xi - 1.0
            var b_mj = xi
            var ss0 = q0 + section_load_axial[ip]
            var ss1 = b_mi * q1 + b_mj * q2 + section_load_moment[ip]
            var point = scratch.section_batch_points[ip]
            var resp_trial = scratch.section_batch_results[ip]
            scratch.section_vs_subdivide_eps0[ip] = point.eps0
            scratch.section_vs_subdivide_kappa[ip] = point.kappa
            scratch.section_ssr_subdivide_axial[ip] = resp_trial.response.axial_force
            scratch.section_ssr_subdivide_moment[ip] = resp_trial.response.moment_z

            if not resp_trial.flexibility_valid:
                return (False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            var sec_f00 = resp_trial.f00
            var sec_f01 = resp_trial.f01
            var sec_f11 = resp_trial.f11
            scratch.section_fs_subdivide00[ip] = sec_f00
            scratch.section_fs_subdivide01[ip] = sec_f01
            scratch.section_fs_subdivide11[ip] = sec_f11
            var sec_f10 = sec_f01

            var dss_res0 = ss0 - scratch.section_ssr_subdivide_axial[ip]
            var dss_res1 = ss1 - scratch.section_ssr_subdivide_moment[ip]
            var dvs_res0 = sec_f00 * dss_res0 + sec_f01 * dss_res1
            var dvs_res1 = sec_f01 * dss_res0 + sec_f11 * dss_res1

            f00 += wL * sec_f00
            f01 += wL * sec_f01 * b_mi
            f02 += wL * sec_f01 * b_mj
            f10 += wL * b_mi * sec_f10
            f11 += wL * b_mi * sec_f11 * b_mi
            f12 += wL * b_mi * sec_f11 * b_mj
            f20 += wL * b_mj * sec_f10
            f21 += wL * b_mj * sec_f11 * b_mi
            f22 += wL * b_mj * sec_f11 * b_mj

            vr0 += wL * (point.eps0 + dvs_res0)
            vr1 += wL * b_mi * (point.kappa + dvs_res1)
            vr2 += wL * b_mj * (point.kappa + dvs_res1)
        if runtime_metrics.enabled:
            runtime_metrics.local_flexibility_accumulation_ns += (
                Int(perf_counter_ns()) - t_flex_start
            )
        t_solve3_start = 0
        if runtime_metrics.enabled:
            t_solve3_start = Int(perf_counter_ns())
        var k_inv = _invert_3x3_values(f00, f01, f02, f10, f11, f12, f20, f21, f22)
        if not k_inv[0]:
            return (False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        if runtime_metrics.enabled:
            runtime_metrics.local_3x3_solve_ns += Int(perf_counter_ns()) - t_solve3_start
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

        if work_norm < elem_tol:
            scratch.section_batch_profile.converged_points += num_int_pts
            var t_commit_start = 0
            if runtime_metrics.enabled:
                t_commit_start = Int(perf_counter_ns())
            force_basic_q_state[force_basic_q_offset] = q0
            force_basic_q_state[force_basic_q_offset + 1] = q1
            force_basic_q_state[force_basic_q_offset + 2] = q2
            for ip in range(num_int_pts):
                force_basic_q_state[eps0_offset + ip] = scratch.section_vs_subdivide_eps0[ip]
                force_basic_q_state[kappa_offset + ip] = scratch.section_vs_subdivide_kappa[ip]
                scratch.section_vs_eps0[ip] = scratch.section_vs_subdivide_eps0[ip]
                scratch.section_vs_kappa[ip] = scratch.section_vs_subdivide_kappa[ip]
                scratch.section_ssr_axial[ip] = scratch.section_ssr_subdivide_axial[ip]
                scratch.section_ssr_moment[ip] = scratch.section_ssr_subdivide_moment[ip]
                scratch.section_fs00[ip] = scratch.section_fs_subdivide00[ip]
                scratch.section_fs01[ip] = scratch.section_fs_subdivide01[ip]
                scratch.section_fs11[ip] = scratch.section_fs_subdivide11[ip]
            if runtime_metrics.enabled:
                runtime_metrics.local_commit_revert_ns += Int(perf_counter_ns()) - t_commit_start
            return (True, q0, q1, q2, k00, k01, k02, k10, k11, k12, k22)

    return (False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


fn force_beam_column2d_global_tangent_and_internal(
    x1: Float64,
    y1: Float64,
    x2: Float64,
    y2: Float64,
    u_elem_global: List[Float64],
    mut sec_def: FiberSection2dDef,
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
    mut scratch: ForceBeamColumn2dScratch,
    mut k_global_out: List[List[Float64]],
    mut f_global_out: List[Float64],
):
    var empty_element_loads: List[ElementLoadInput] = []
    var empty_elem_load_offsets: List[Int] = []
    var empty_elem_load_pool: List[Int] = []
    var runtime_metrics = RuntimeProfileMetrics()
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
        runtime_metrics,
        scratch,
        k_global_out,
        f_global_out,
    )


fn force_beam_column2d_global_tangent_and_internal(
    x1: Float64,
    y1: Float64,
    x2: Float64,
    y2: Float64,
    u_elem_global: List[Float64],
    mut sec_def: FiberSection2dDef,
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
    var scratch = ForceBeamColumn2dScratch()
    force_beam_column2d_global_tangent_and_internal(
        x1,
        y1,
        x2,
        y2,
        u_elem_global,
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
        scratch,
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
    mut sec_def: FiberSection2dDef,
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
    hinges: HingeRadau2d = HingeRadau2d(),
):
    var scratch = ForceBeamColumn2dScratch()
    var runtime_metrics = RuntimeProfileMetrics()
    force_beam_column2d_global_tangent_and_internal(
        elem_index,
        x1,
        y1,
        x2,
        y2,
        u_elem_global,
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
        geom_transf,
        integration,
        num_int_pts,
        force_basic_q_state,
        force_basic_q_offset,
        force_basic_q_count,
        runtime_metrics,
        scratch,
        k_global_out,
        f_global_out,
        hinges,
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
    mut sec_def: FiberSection2dDef,
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
    mut scratch: ForceBeamColumn2dScratch,
    mut k_global_out: List[List[Float64]],
    mut f_global_out: List[Float64],
    hinges: HingeRadau2d = HingeRadau2d(),
):
    var runtime_metrics = RuntimeProfileMetrics()
    force_beam_column2d_global_tangent_and_internal(
        elem_index,
        x1,
        y1,
        x2,
        y2,
        u_elem_global,
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
        geom_transf,
        integration,
        num_int_pts,
        force_basic_q_state,
        force_basic_q_offset,
        force_basic_q_count,
        runtime_metrics,
        scratch,
        k_global_out,
        f_global_out,
        hinges,
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
    mut sec_def: FiberSection2dDef,
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
    mut runtime_metrics: RuntimeProfileMetrics,
    mut scratch: ForceBeamColumn2dScratch,
    mut k_global_out: List[List[Float64]],
    mut f_global_out: List[Float64],
    hinges: HingeRadau2d = HingeRadau2d(),
):
    _ensure_force_beam_column2d_geometry_cache(elem_index, x1, y1, x2, y2, scratch)
    var dx = x2 - x1
    var dy = y2 - y1
    var L = hypot(dx, dy)
    if elem_index >= 0 and elem_index < len(scratch.cached_length):
        L = scratch.cached_length[elem_index]
    if L == 0.0:
        abort("zero-length element")

    var fibers_per_section = sec_def.fiber_count
    var required_state_count = 4 if hinges.is_defined() else num_int_pts * fibers_per_section
    if elem_state_count != required_state_count:
        abort("forceBeamColumn2d element state count mismatch")
    if hinges.is_defined():
        scratch.integration_cache.is_valid = False
        scratch.integration_cache.xis.resize(6, 0.0)
        scratch.integration_cache.weights.resize(6, 0.0)
        for ip in range(6):
            var point = hinges.xi_weight(ip, L)
            scratch.integration_cache.xis[ip] = point[0]
            scratch.integration_cache.weights[ip] = point[1]
    else:
        beam_integration_cache_ensure(scratch.integration_cache, integration, num_int_pts)
    _ensure_force_beam_column2d_load_cache(
        elem_index,
        num_int_pts,
        element_loads,
        elem_load_offsets,
        elem_load_pool,
        load_scale,
        scratch,
    )
    scratch.section_load_axial.resize(num_int_pts, 0.0)
    scratch.section_load_moment.resize(num_int_pts, 0.0)
    if elem_index >= 0 and elem_index < len(scratch.section_load_axial_cache):
        for ip in range(num_int_pts):
            scratch.section_load_axial[ip] = scratch.section_load_axial_cache[elem_index][ip]
            scratch.section_load_moment[ip] = scratch.section_load_moment_cache[elem_index][ip]
    else:
        for ip in range(num_int_pts):
            scratch.section_load_axial[ip] = 0.0
            scratch.section_load_moment[ip] = 0.0
    var fixed_end = beam2d_basic_fixed_end_and_reactions(
        element_loads, elem_load_offsets, elem_load_pool, elem_index, load_scale, L
    )
    if elem_index >= 0 and elem_index < len(scratch.fixed_end_cache):
        fixed_end = (
            scratch.fixed_end_cache[elem_index][0],
            scratch.fixed_end_cache[elem_index][1],
            scratch.fixed_end_cache[elem_index][2],
            scratch.fixed_end_cache[elem_index][3],
            scratch.fixed_end_cache[elem_index][4],
            scratch.fixed_end_cache[elem_index][5],
        )

    var c = dx / L
    var s = dy / L
    if elem_index >= 0 and elem_index < len(scratch.cached_cos):
        c = scratch.cached_cos[elem_index]
        s = scratch.cached_sin[elem_index]
    _beam2d_transform_u_global_to_local(c, s, u_elem_global, scratch.u_local)

    var predictor_state_count = 3 + 2 * num_int_pts
    var active_basic_count = force_beam_column2d_fiber_state_count(num_int_pts)
    if force_basic_q_count < active_basic_count:
        abort("forceBeamColumn2d basic force state count mismatch")
    if force_basic_q_offset < 0 or force_basic_q_offset + active_basic_count > len(
        force_basic_q_state
    ):
        abort("forceBeamColumn2d basic force state out of range")
    var eps0_offset = force_basic_q_offset + 3
    var kappa_offset = eps0_offset + num_int_pts
    var basic_state_offset = force_basic_q_offset + predictor_state_count
    var tangent_offset = basic_state_offset + 3
    var initialized_offset = tangent_offset + 7

    var basic = force_beam_column2d_basic_displacement(
        L, scratch.u_local, geom_transf == "Corotational"
    )
    var v_basic_0 = basic[0]
    var v_basic_1 = basic[1]
    var v_basic_2 = basic[2]
    var basic_prev_0 = force_basic_q_state[basic_state_offset]
    var basic_prev_1 = force_basic_q_state[basic_state_offset + 1]
    var basic_prev_2 = force_basic_q_state[basic_state_offset + 2]

    _ensure_force_beam_column2d_section_history_capacity(scratch, num_int_pts)

    var accepted_basic_0 = basic_prev_0
    var accepted_basic_1 = basic_prev_1
    var accepted_basic_2 = basic_prev_2
    var accepted_q0 = force_basic_q_state[force_basic_q_offset]
    var accepted_q1 = force_basic_q_state[force_basic_q_offset + 1]
    var accepted_q2 = force_basic_q_state[force_basic_q_offset + 2]
    for ip in range(num_int_pts):
        scratch.section_vs_eps0[ip] = force_basic_q_state[eps0_offset + ip]
        scratch.section_vs_kappa[ip] = force_basic_q_state[kappa_offset + ip]

    var remaining_0 = v_basic_0 - accepted_basic_0
    var remaining_1 = v_basic_1 - accepted_basic_1
    var remaining_2 = v_basic_2 - accepted_basic_2
    var attempt_0 = remaining_0
    var attempt_1 = remaining_1
    var attempt_2 = remaining_2
    var all_materials_elastic = not hinges.is_defined() and _fiber_section2d_all_materials_elastic(
        sec_def
    )
    var best_solved = (
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

    var has_element_loads = (
        elem_index >= 0
        and elem_index + 1 < len(elem_load_offsets)
        and elem_load_offsets[elem_index] < elem_load_offsets[elem_index + 1]
    )
    var converged = (
        not has_element_loads
        and not hinges.is_defined()
        and force_basic_q_state[initialized_offset] == 1.0
        and sqrt(remaining_0 * remaining_0 + remaining_1 * remaining_1 + remaining_2 * remaining_2)
        <= 2.220446049250313e-16
    )
    if converged:
        best_solved = (
            True,
            accepted_q0,
            accepted_q1,
            accepted_q2,
            force_basic_q_state[tangent_offset],
            force_basic_q_state[tangent_offset + 1],
            force_basic_q_state[tangent_offset + 2],
            force_basic_q_state[tangent_offset + 3],
            force_basic_q_state[tangent_offset + 4],
            force_basic_q_state[tangent_offset + 5],
            force_basic_q_state[tangent_offset + 6],
        )

    var tolerance = 1.0e-12
    var max_subdivisions = 4
    var num_subdivide = 1
    var cutback_factor = 10.0

    if all_materials_elastic and not converged:
        var solved = _force_beam_column2d_exact_elastic_state(
            L,
            scratch.integration_cache.xis,
            scratch.integration_cache.weights,
            scratch.section_load_axial,
            scratch.section_load_moment,
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
            accepted_q0 = solved[1]
            accepted_q1 = solved[2]
            accepted_q2 = solved[3]
            for ip in range(num_int_pts):
                scratch.section_vs_eps0[ip] = force_basic_q_state[eps0_offset + ip]
                scratch.section_vs_kappa[ip] = force_basic_q_state[kappa_offset + ip]
            converged = True
            remaining_0 = 0.0
            remaining_1 = 0.0
            remaining_2 = 0.0
            best_solved = solved

    if not converged and _max_abs3(remaining_0, remaining_1, remaining_2) <= 1.0e-16:
        for use_initial in range(3):
            _restore_force_beam_column2d_predictor_state(
                force_basic_q_state,
                force_basic_q_offset,
                num_int_pts,
                accepted_q0,
                accepted_q1,
                accepted_q2,
                scratch.section_vs_eps0,
                scratch.section_vs_kappa,
            )
            _set_force_beam_column2d_trial_basic_deformation(
                force_basic_q_state,
                basic_state_offset,
                accepted_basic_0,
                accepted_basic_1,
                accepted_basic_2,
            )
            var solved = _force_beam_column2d_try_increment(
                L,
                sec_def,
                hinges,
                fibers,
                uniaxial_defs,
                uniaxial_states,
                elem_state_ids,
                elem_state_offset,
                fibers_per_section,
                num_int_pts,
                force_basic_q_state,
                force_basic_q_offset,
                scratch,
                accepted_basic_0,
                accepted_basic_1,
                accepted_basic_2,
                0.0,
                0.0,
                0.0,
                use_initial,
                runtime_metrics,
            )
            if not solved[0]:
                continue
            best_solved = solved
            converged = True
            break
    elif not converged:
        while True:
            if num_subdivide > max_subdivisions:
                break
            var target_v0 = accepted_basic_0 + attempt_0
            var target_v1 = accepted_basic_1 + attempt_1
            var target_v2 = accepted_basic_2 + attempt_2

            var scheme_success = False
            for use_initial in range(3):
                _restore_force_beam_column2d_predictor_state(
                    force_basic_q_state,
                    force_basic_q_offset,
                    num_int_pts,
                    accepted_q0,
                    accepted_q1,
                    accepted_q2,
                    scratch.section_vs_eps0,
                    scratch.section_vs_kappa,
                )
                _set_force_beam_column2d_trial_basic_deformation(
                    force_basic_q_state,
                    basic_state_offset,
                    accepted_basic_0,
                    accepted_basic_1,
                    accepted_basic_2,
                )

                var solved = _force_beam_column2d_try_increment(
                    L,
                    sec_def,
                    hinges,
                    fibers,
                    uniaxial_defs,
                    uniaxial_states,
                    elem_state_ids,
                    elem_state_offset,
                    fibers_per_section,
                    num_int_pts,
                    force_basic_q_state,
                    force_basic_q_offset,
                    scratch,
                    accepted_basic_0,
                    accepted_basic_1,
                    accepted_basic_2,
                    attempt_0,
                    attempt_1,
                    attempt_2,
                    use_initial,
                    runtime_metrics,
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
                accepted_q0 = solved[1]
                accepted_q1 = solved[2]
                accepted_q2 = solved[3]
                for ip in range(num_int_pts):
                    scratch.section_vs_eps0[ip] = force_basic_q_state[eps0_offset + ip]
                    scratch.section_vs_kappa[ip] = force_basic_q_state[kappa_offset + ip]
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
                    num_subdivide = 1
                scheme_success = True
                break

            if converged:
                break
            if scheme_success:
                continue

            attempt_0 /= cutback_factor
            attempt_1 /= cutback_factor
            attempt_2 /= cutback_factor
            num_subdivide += 1
            if runtime_metrics.enabled:
                runtime_metrics.subdivision_fallback_iterations += 1

    if not converged:
        var t_revert_start = 0
        if runtime_metrics.enabled:
            t_revert_start = Int(perf_counter_ns())
        _restore_force_beam_column2d_predictor_state(
            force_basic_q_state,
            force_basic_q_offset,
            num_int_pts,
            accepted_q0,
            accepted_q1,
            accepted_q2,
            scratch.section_vs_eps0,
            scratch.section_vs_kappa,
        )
        _set_force_beam_column2d_trial_basic_deformation(
            force_basic_q_state,
            basic_state_offset,
            accepted_basic_0,
            accepted_basic_1,
            accepted_basic_2,
        )
        for ip in range(num_int_pts):
            var ip_state_offset = elem_state_offset + ip * fibers_per_section
            _ = _fiber_section2d_set_trial_from_offset(
                sec_def,
                uniaxial_states,
                ip_state_offset,
                fibers_per_section,
                scratch.section_vs_eps0[ip],
                scratch.section_vs_kappa[ip],
            )
        if runtime_metrics.enabled:
            runtime_metrics.local_commit_revert_ns += Int(perf_counter_ns()) - t_revert_start
    if not best_solved[0]:
        if runtime_metrics.enabled:
            runtime_metrics.subdivision_fallback_iterations += 1
        var fallback_tangent = _force_beam_column2d_initial_basic_tangent(
            L,
            scratch.integration_cache.xis,
            scratch.integration_cache.weights,
            sec_def,
            fibers,
            uniaxial_defs,
        )
        if not fallback_tangent[0]:
            abort("forceBeamColumn2d final tangent recovery did not converge")
        best_solved = (
            True,
            force_basic_q_state[force_basic_q_offset],
            force_basic_q_state[force_basic_q_offset + 1],
            force_basic_q_state[force_basic_q_offset + 2],
            fallback_tangent[1],
            fallback_tangent[2],
            fallback_tangent[3],
            fallback_tangent[4],
            fallback_tangent[5],
            fallback_tangent[6],
            fallback_tangent[8],
        )
    if converged:
        force_basic_q_state[tangent_offset] = best_solved[4]
        force_basic_q_state[tangent_offset + 1] = best_solved[5]
        force_basic_q_state[tangent_offset + 2] = best_solved[6]
        force_basic_q_state[tangent_offset + 3] = best_solved[7]
        force_basic_q_state[tangent_offset + 4] = best_solved[8]
        force_basic_q_state[tangent_offset + 5] = best_solved[9]
        force_basic_q_state[tangent_offset + 6] = best_solved[10]
        force_basic_q_state[initialized_offset] = -1.0 if has_element_loads else 1.0
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

    _force_beam_column2d_basic_to_global(
        elem_index,
        L,
        c,
        s,
        scratch.u_local,
        geom_transf,
        q0,
        q1,
        q2,
        k00,
        k01,
        k02,
        k10,
        k11,
        k12,
        k20,
        k21,
        k22,
        fixed_end,
        scratch.basic_map_cache,
        k_global_out,
        f_global_out,
    )


fn _force_beam_column2d_basic_to_global(
    elem_index: Int,
    L: Float64,
    c: Float64,
    s: Float64,
    u_local: List[Float64],
    geom_transf: String,
    q0: Float64,
    q1: Float64,
    q2: Float64,
    k00: Float64,
    k01: Float64,
    k02: Float64,
    k10: Float64,
    k11: Float64,
    k12: Float64,
    k20: Float64,
    k21: Float64,
    k22: Float64,
    fixed_end: (Float64, Float64, Float64, Float64, Float64, Float64),
    basic_map_cache: List[List[Float64]],
    mut k_global_out: List[List[Float64]],
    mut f_global_out: List[Float64],
):
    if geom_transf == "Corotational":
        _force_beam_column2d_corotational_basic_to_global(
            L,
            c,
            s,
            u_local,
            q0,
            q1,
            q2,
            k00,
            k01,
            k02,
            k10,
            k11,
            k12,
            k22,
            fixed_end,
            k_global_out,
            f_global_out,
        )
        return

    var inv_L = 1.0 / L
    _ensure_zero_matrix(k_global_out, 6, 6)
    var use_cached_basic_map = (
        elem_index >= 0
        and elem_index < len(basic_map_cache)
        and len(basic_map_cache[elem_index]) == 18
    )
    for a in range(6):
        var a_offset = 3 * a
        var a0: Float64
        var a1: Float64
        var a2: Float64
        if use_cached_basic_map:
            a0 = basic_map_cache[elem_index][a_offset]
            a1 = basic_map_cache[elem_index][a_offset + 1]
            a2 = basic_map_cache[elem_index][a_offset + 2]
        else:
            (a0, a1, a2) = _beam2d_local_basic_map(a, inv_L)
        for b in range(6):
            var b_offset = 3 * b
            var b0: Float64
            var b1: Float64
            var b2: Float64
            if use_cached_basic_map:
                b0 = basic_map_cache[elem_index][b_offset]
                b1 = basic_map_cache[elem_index][b_offset + 1]
                b2 = basic_map_cache[elem_index][b_offset + 2]
            else:
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


fn elastic_beam_column2d_global_tangent_and_internal(
    elem_index: Int,
    x1: Float64,
    y1: Float64,
    x2: Float64,
    y2: Float64,
    u_global: List[Float64],
    geom_transf: String,
    integration: String,
    num_int_pts: Int,
    sec: SectionInput,
    force_based: Bool,
    element_loads: List[ElementLoadInput],
    elem_load_offsets: List[Int],
    elem_load_pool: List[Int],
    load_scale: Float64,
    mut scratch: ForceBeamColumn2dScratch,
    mut k_global: List[List[Float64]],
    mut f_global: List[Float64],
) -> (Float64, Float64, Float64):
    var dx = x2 - x1
    var dy = y2 - y1
    var length = hypot(dx, dy)
    if length == 0.0:
        abort("zero-length element")
    var inv_length = 1.0 / length
    var c = dx * inv_length
    var s = dy * inv_length
    _beam2d_transform_u_global_to_local(c, s, u_global, scratch.u_local)
    var axial = scratch.u_local[3] - scratch.u_local[0]
    var transverse = scratch.u_local[4] - scratch.u_local[1]
    var chord_rotation = transverse * inv_length
    if geom_transf == "Corotational":
        axial = hypot(length + axial, transverse) - length
        chord_rotation = atan2(transverse, length + scratch.u_local[3] - scratch.u_local[0])
    var v1 = scratch.u_local[2] - chord_rotation
    var v2 = scratch.u_local[5] - chord_rotation
    var fixed = beam2d_basic_fixed_end_and_reactions(
        element_loads, elem_load_offsets, elem_load_pool, elem_index, load_scale, length
    )
    var ea = sec.E * sec.A
    var ei = sec.E * sec.I
    if sec.cbdi.enabled():
        beam_integration_cache_ensure(scratch.integration_cache, integration, num_int_pts)
        var loads: List[List[Float64]] = []
        _ensure_zero_matrix(loads, 2, num_int_pts)
        # OpenSees CBDI2d includes fixed-end reactions in the transform,
        # while its section equilibrium omits the applied element load.
        var q: List[Float64] = []
        var k: List[List[Float64]] = []
        cbdi_elastic_basic_state(
            length,
            ea,
            [ei],
            scratch.integration_cache,
            loads,
            [axial, v1, v2],
            sec.cbdi,
            q,
            k,
        )
        _force_beam_column2d_basic_to_global(
            -1,
            length,
            c,
            s,
            scratch.u_local,
            geom_transf,
            q[0],
            q[1],
            q[2],
            k[0][0],
            k[0][1],
            k[0][2],
            k[1][0],
            k[1][1],
            k[1][2],
            k[2][0],
            k[2][1],
            k[2][2],
            fixed,
            scratch.basic_map_cache,
            k_global,
            f_global,
        )
        return (q[0], q[1], q[2])
    var ka = ea * inv_length
    var k11 = 4.0 * ei * inv_length
    var k12 = 2.0 * ei * inv_length
    var k22 = k11
    var q0 = fixed[0]
    var q1 = fixed[1]
    var q2 = fixed[2]
    if force_based:
        var f00 = 0.0
        var f11 = 0.0
        var f12 = 0.0
        var f22 = 0.0
        var v0_load = 0.0
        var v1_load = 0.0
        var v2_load = 0.0
        var hinge = sec.hinge_radau
        if not hinge.is_defined():
            beam_integration_cache_ensure(scratch.integration_cache, integration, num_int_pts)
        for ip in range(num_int_pts):
            var xi: Float64
            var weight: Float64
            var point_ea = ea
            var point_ei = ei
            var point_ga = sec.shear_rigidity_y
            if hinge.is_defined():
                (xi, weight) = hinge.xi_weight(ip, length)
                if ip == 0 or ip == num_int_pts - 1:
                    var end = hinge.elastic_end_i if ip == 0 else hinge.elastic_end_j
                    point_ea = end[0]
                    point_ei = end[1]
                    point_ga = end[2]
            else:
                xi = scratch.integration_cache.xis[ip]
                weight = scratch.integration_cache.weights[ip]
            var wL = weight * length
            var b1 = xi - 1.0
            var b2 = xi
            var flexibility_a = wL / point_ea
            var flexibility_m = wL / point_ei
            var flexibility_v = 0.0 if point_ga == 0.0 else wL / point_ga
            var shear_term = flexibility_v * inv_length * inv_length
            var load = beam2d_section_load_response(
                element_loads,
                elem_load_offsets,
                elem_load_pool,
                elem_index,
                load_scale,
                xi * length,
                length,
            )
            f00 += flexibility_a
            f11 += flexibility_m * b1 * b1 + shear_term
            f12 += flexibility_m * b1 * b2 + shear_term
            f22 += flexibility_m * b2 * b2 + shear_term
            v0_load += flexibility_a * load[0]
            v1_load += flexibility_m * b1 * load[1] + flexibility_v * inv_length * load[2]
            v2_load += flexibility_m * b2 * load[1] + flexibility_v * inv_length * load[2]
        var det = f11 * f22 - f12 * f12
        if f00 == 0.0 or det == 0.0:
            abort("elastic forceBeamColumn2d flexibility is singular")
        ka = 1.0 / f00
        k11 = f22 / det
        k12 = -f12 / det
        k22 = f11 / det
        q0 = -ka * v0_load
        q1 = -k11 * v1_load - k12 * v2_load
        q2 = -k12 * v1_load - k22 * v2_load

    q0 += ka * axial
    q1 += k11 * v1 + k12 * v2
    q2 += k12 * v1 + k22 * v2
    _force_beam_column2d_basic_to_global(
        -1,
        length,
        c,
        s,
        scratch.u_local,
        geom_transf,
        q0,
        q1,
        q2,
        ka,
        0.0,
        0.0,
        0.0,
        k11,
        k12,
        0.0,
        k12,
        k22,
        fixed,
        scratch.basic_map_cache,
        k_global,
        f_global,
    )
    return (q0, q1, q2)
