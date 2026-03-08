from algorithm import vectorize
from collections import List
from os import abort
from python import PythonObject

from materials import (
    UniMaterialDef,
    UniMaterialState,
    uni_mat_is_elastic,
    uni_mat_initial_tangent,
    uniaxial_commit,
    uniaxial_revert_trial,
    uniaxial_set_trial_strain_concrete01,
    uniaxial_set_trial_strain_concrete02,
    uniaxial_set_trial_strain,
    uniaxial_set_trial_strain_steel01,
    uniaxial_set_trial_strain_steel02,
)
from solver.run_case.input_types import FiberLayerInput, FiberPatchInput, SectionInput
from solver.simd_contiguous import FLOAT64_SIMD_WIDTH, load_float64_contiguous_simd
from strut_io import py_len
from tag_types import UniMaterialTypeTag


struct FiberCell(Defaultable, Movable, ImplicitlyCopyable):
    var y: Float64
    var z: Float64
    var area: Float64
    var def_index: Int

    fn __init__(out self):
        self.y = 0.0
        self.z = 0.0
        self.area = 0.0
        self.def_index = -1

    fn __init__(
        out self, y: Float64, z: Float64, area: Float64, def_index: Int
    ):
        self.y = y
        self.z = z
        self.area = area
        self.def_index = def_index


struct FiberSection2dDef(Defaultable, Movable, ImplicitlyCopyable):
    var fiber_offset: Int
    var fiber_count: Int
    var y_bar: Float64
    var elastic_count: Int
    var nonlinear_count: Int
    var initial_flex_valid: Bool
    var initial_f00: Float64
    var initial_f01: Float64
    var initial_f11: Float64
    var elastic_y_rel: List[Float64]
    var elastic_area: List[Float64]
    var elastic_modulus: List[Float64]
    var elastic_def_index: List[Int]
    var nonlinear_y_rel: List[Float64]
    var nonlinear_area: List[Float64]
    var nonlinear_def_index: List[Int]
    var nonlinear_mat_defs: List[UniMaterialDef]
    var steel01_nonlinear_indices: List[Int]
    var concrete01_nonlinear_indices: List[Int]
    var steel02_nonlinear_indices: List[Int]
    var concrete02_nonlinear_indices: List[Int]
    var other_nonlinear_indices: List[Int]

    fn __init__(out self):
        self.fiber_offset = 0
        self.fiber_count = 0
        self.y_bar = 0.0
        self.elastic_count = 0
        self.nonlinear_count = 0
        self.initial_flex_valid = False
        self.initial_f00 = 0.0
        self.initial_f01 = 0.0
        self.initial_f11 = 0.0
        self.elastic_y_rel = []
        self.elastic_area = []
        self.elastic_modulus = []
        self.elastic_def_index = []
        self.nonlinear_y_rel = []
        self.nonlinear_area = []
        self.nonlinear_def_index = []
        self.nonlinear_mat_defs = []
        self.steel01_nonlinear_indices = []
        self.concrete01_nonlinear_indices = []
        self.steel02_nonlinear_indices = []
        self.concrete02_nonlinear_indices = []
        self.other_nonlinear_indices = []

    fn __init__(out self, fiber_offset: Int, fiber_count: Int, y_bar: Float64):
        self.fiber_offset = fiber_offset
        self.fiber_count = fiber_count
        self.y_bar = y_bar
        self.elastic_count = 0
        self.nonlinear_count = 0
        self.initial_flex_valid = False
        self.initial_f00 = 0.0
        self.initial_f01 = 0.0
        self.initial_f11 = 0.0
        self.elastic_y_rel = []
        self.elastic_area = []
        self.elastic_modulus = []
        self.elastic_def_index = []
        self.nonlinear_y_rel = []
        self.nonlinear_area = []
        self.nonlinear_def_index = []
        self.nonlinear_mat_defs = []
        self.steel01_nonlinear_indices = []
        self.concrete01_nonlinear_indices = []
        self.steel02_nonlinear_indices = []
        self.concrete02_nonlinear_indices = []
        self.other_nonlinear_indices = []

    fn __copyinit__(out self, existing: Self):
        self.fiber_offset = existing.fiber_offset
        self.fiber_count = existing.fiber_count
        self.y_bar = existing.y_bar
        self.elastic_count = existing.elastic_count
        self.nonlinear_count = existing.nonlinear_count
        self.initial_flex_valid = existing.initial_flex_valid
        self.initial_f00 = existing.initial_f00
        self.initial_f01 = existing.initial_f01
        self.initial_f11 = existing.initial_f11
        self.elastic_y_rel = existing.elastic_y_rel.copy()
        self.elastic_area = existing.elastic_area.copy()
        self.elastic_modulus = existing.elastic_modulus.copy()
        self.elastic_def_index = existing.elastic_def_index.copy()
        self.nonlinear_y_rel = existing.nonlinear_y_rel.copy()
        self.nonlinear_area = existing.nonlinear_area.copy()
        self.nonlinear_def_index = existing.nonlinear_def_index.copy()
        self.nonlinear_mat_defs = existing.nonlinear_mat_defs.copy()
        self.steel01_nonlinear_indices = existing.steel01_nonlinear_indices.copy()
        self.concrete01_nonlinear_indices = existing.concrete01_nonlinear_indices.copy()
        self.steel02_nonlinear_indices = existing.steel02_nonlinear_indices.copy()
        self.concrete02_nonlinear_indices = existing.concrete02_nonlinear_indices.copy()
        self.other_nonlinear_indices = existing.other_nonlinear_indices.copy()


struct FiberSection2dResponse(Defaultable, Movable, ImplicitlyCopyable):
    var axial_force: Float64
    var moment_z: Float64
    var k11: Float64
    var k12: Float64
    var k22: Float64

    fn __init__(out self):
        self.axial_force = 0.0
        self.moment_z = 0.0
        self.k11 = 0.0
        self.k12 = 0.0
        self.k22 = 0.0

    fn __init__(
        out self,
        axial_force: Float64,
        moment_z: Float64,
        k11: Float64,
        k12: Float64,
        k22: Float64,
    ):
        self.axial_force = axial_force
        self.moment_z = moment_z
        self.k11 = k11
        self.k12 = k12
        self.k22 = k22


alias FIBER_SECTION2D_BATCH_FLAG_PREDICTOR: Int = 1
alias FIBER_SECTION2D_BATCH_FLAG_CORRECTOR: Int = 2
alias FIBER_SECTION2D_BATCH_FLAG_RETRY: Int = 4
alias FIBER_SECTION2D_BATCH_FLAG_CONVERGED: Int = 8


struct FiberSection2dBatchPoint(Defaultable, Movable, ImplicitlyCopyable):
    var section_def_index: Int
    var section_state_offset: Int
    var section_state_count: Int
    var eps0: Float64
    var kappa: Float64
    var lagged_axial_force: Float64
    var lagged_moment_z: Float64
    var lagged_f00: Float64
    var lagged_f01: Float64
    var lagged_f11: Float64
    var flags: Int

    fn __init__(out self):
        self.section_def_index = -1
        self.section_state_offset = 0
        self.section_state_count = 0
        self.eps0 = 0.0
        self.kappa = 0.0
        self.lagged_axial_force = 0.0
        self.lagged_moment_z = 0.0
        self.lagged_f00 = 0.0
        self.lagged_f01 = 0.0
        self.lagged_f11 = 0.0
        self.flags = 0

    fn __init__(
        out self,
        section_def_index: Int,
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
        self.section_def_index = section_def_index
        self.section_state_offset = section_state_offset
        self.section_state_count = section_state_count
        self.eps0 = eps0
        self.kappa = kappa
        self.lagged_axial_force = lagged_axial_force
        self.lagged_moment_z = lagged_moment_z
        self.lagged_f00 = lagged_f00
        self.lagged_f01 = lagged_f01
        self.lagged_f11 = lagged_f11
        self.flags = flags


struct FiberSection2dBatchResult(Defaultable, Movable, ImplicitlyCopyable):
    var response: FiberSection2dResponse
    var flexibility_valid: Bool
    var f00: Float64
    var f01: Float64
    var f11: Float64

    fn __init__(out self):
        self.response = FiberSection2dResponse()
        self.flexibility_valid = False
        self.f00 = 0.0
        self.f01 = 0.0
        self.f11 = 0.0

    fn __init__(
        out self,
        response: FiberSection2dResponse,
        flexibility_valid: Bool,
        f00: Float64,
        f01: Float64,
        f11: Float64,
    ):
        self.response = response
        self.flexibility_valid = flexibility_valid
        self.f00 = f00
        self.f01 = f01
        self.f11 = f11


struct FiberSection2dBatchProfile(Defaultable, Movable, ImplicitlyCopyable):
    var batches_total: Int
    var homogeneous_batches: Int
    var irregular_batches: Int
    var scalar_fallback_batches: Int
    var section_point_evals: Int
    var section_point_reevals: Int
    var predictor_points: Int
    var corrector_points: Int
    var retry_points: Int
    var converged_points: Int
    var batch_hist_1: Int
    var batch_hist_2_4: Int
    var batch_hist_5_8: Int
    var batch_hist_9_plus: Int
    var traced_force_beam_elements: Int
    var traced_integration_points: Int
    var traced_fibers_total: Int
    var traced_max_fibers_per_section: Int
    var traced_elastic_fibers: Int
    var traced_nonlinear_fibers: Int
    var traced_elastic_material_points: Int
    var traced_steel01_material_points: Int
    var traced_concrete01_material_points: Int
    var traced_steel02_material_points: Int
    var traced_concrete02_material_points: Int
    var traced_other_material_points: Int

    fn __init__(out self):
        self.batches_total = 0
        self.homogeneous_batches = 0
        self.irregular_batches = 0
        self.scalar_fallback_batches = 0
        self.section_point_evals = 0
        self.section_point_reevals = 0
        self.predictor_points = 0
        self.corrector_points = 0
        self.retry_points = 0
        self.converged_points = 0
        self.batch_hist_1 = 0
        self.batch_hist_2_4 = 0
        self.batch_hist_5_8 = 0
        self.batch_hist_9_plus = 0
        self.traced_force_beam_elements = 0
        self.traced_integration_points = 0
        self.traced_fibers_total = 0
        self.traced_max_fibers_per_section = 0
        self.traced_elastic_fibers = 0
        self.traced_nonlinear_fibers = 0
        self.traced_elastic_material_points = 0
        self.traced_steel01_material_points = 0
        self.traced_concrete01_material_points = 0
        self.traced_steel02_material_points = 0
        self.traced_concrete02_material_points = 0
        self.traced_other_material_points = 0


fn fiber_section2d_batch_profile_reset(mut profile: FiberSection2dBatchProfile):
    profile = FiberSection2dBatchProfile()


fn _fiber_section2d_batch_profile_note_batch_size(
    batch_size: Int, mut profile: FiberSection2dBatchProfile
):
    profile.batches_total += 1
    if batch_size <= 1:
        profile.batch_hist_1 += 1
    elif batch_size <= 4:
        profile.batch_hist_2_4 += 1
    elif batch_size <= 8:
        profile.batch_hist_5_8 += 1
    else:
        profile.batch_hist_9_plus += 1


fn fiber_section2d_batch_profile_note_definition(
    sec_def: FiberSection2dDef,
    num_section_points: Int,
    mut profile: FiberSection2dBatchProfile,
):
    if num_section_points <= 0:
        return
    profile.traced_force_beam_elements += 1
    profile.traced_integration_points += num_section_points
    profile.traced_fibers_total += num_section_points * sec_def.fiber_count
    profile.traced_elastic_fibers += num_section_points * sec_def.elastic_count
    profile.traced_nonlinear_fibers += num_section_points * sec_def.nonlinear_count
    if sec_def.fiber_count > profile.traced_max_fibers_per_section:
        profile.traced_max_fibers_per_section = sec_def.fiber_count
    profile.traced_elastic_material_points += num_section_points * sec_def.elastic_count
    for i in range(sec_def.nonlinear_count):
        var mat_type = sec_def.nonlinear_mat_defs[i].mat_type
        if mat_type == UniMaterialTypeTag.Steel01:
            profile.traced_steel01_material_points += num_section_points
        elif mat_type == UniMaterialTypeTag.Concrete01:
            profile.traced_concrete01_material_points += num_section_points
        elif mat_type == UniMaterialTypeTag.Steel02:
            profile.traced_steel02_material_points += num_section_points
        elif mat_type == UniMaterialTypeTag.Concrete02:
            profile.traced_concrete02_material_points += num_section_points
        else:
            profile.traced_other_material_points += num_section_points


fn _fiber_section2d_batch_profile_note_flags(
    flags: Int, mut profile: FiberSection2dBatchProfile
):
    if flags & FIBER_SECTION2D_BATCH_FLAG_PREDICTOR != 0:
        profile.predictor_points += 1
    if flags & FIBER_SECTION2D_BATCH_FLAG_CORRECTOR != 0:
        profile.corrector_points += 1
        profile.section_point_reevals += 1
    if flags & FIBER_SECTION2D_BATCH_FLAG_RETRY != 0:
        profile.retry_points += 1
        profile.section_point_reevals += 1
    if flags & FIBER_SECTION2D_BATCH_FLAG_CONVERGED != 0:
        profile.converged_points += 1


@always_inline
fn _fiber_section2d_batch_accumulate_nonlinear_response(
    y_rel: Float64,
    area: Float64,
    sig_t: Float64,
    tangent_t: Float64,
    mut resp: FiberSection2dResponse,
):
    var fs = sig_t * area
    var ks = tangent_t * area
    resp.axial_force += fs
    resp.moment_z += -fs * y_rel
    resp.k11 += ks
    resp.k12 += -ks * y_rel
    resp.k22 += ks * y_rel * y_rel


fn _fiber_section2d_batch_apply_steel01_family(
    sec_def: FiberSection2dDef,
    subgroup_indices: List[Int],
    points: List[FiberSection2dBatchPoint],
    mut uniaxial_states: List[UniMaterialState],
    mut results: List[FiberSection2dBatchResult],
):
    for j in range(len(subgroup_indices)):
        var nonlinear_index = subgroup_indices[j]
        var y_rel = sec_def.nonlinear_y_rel[nonlinear_index]
        var area = sec_def.nonlinear_area[nonlinear_index]
        var mat_def = sec_def.nonlinear_mat_defs[nonlinear_index]
        for i in range(len(points)):
            var point = points[i]
            var state_index = point.section_state_offset + sec_def.elastic_count + nonlinear_index
            ref state = uniaxial_states[state_index]
            var eps = point.eps0 - y_rel * point.kappa
            uniaxial_set_trial_strain_steel01(mat_def, state, eps)
            _fiber_section2d_batch_accumulate_nonlinear_response(
                y_rel,
                area,
                state.sig_t,
                state.tangent_t,
                results[i].response,
            )


fn _fiber_section2d_batch_apply_concrete01_family(
    sec_def: FiberSection2dDef,
    subgroup_indices: List[Int],
    points: List[FiberSection2dBatchPoint],
    mut uniaxial_states: List[UniMaterialState],
    mut results: List[FiberSection2dBatchResult],
):
    for j in range(len(subgroup_indices)):
        var nonlinear_index = subgroup_indices[j]
        var y_rel = sec_def.nonlinear_y_rel[nonlinear_index]
        var area = sec_def.nonlinear_area[nonlinear_index]
        var mat_def = sec_def.nonlinear_mat_defs[nonlinear_index]
        for i in range(len(points)):
            var point = points[i]
            var state_index = point.section_state_offset + sec_def.elastic_count + nonlinear_index
            ref state = uniaxial_states[state_index]
            var eps = point.eps0 - y_rel * point.kappa
            uniaxial_set_trial_strain_concrete01(mat_def, state, eps)
            _fiber_section2d_batch_accumulate_nonlinear_response(
                y_rel,
                area,
                state.sig_t,
                state.tangent_t,
                results[i].response,
            )


fn _fiber_section2d_batch_apply_steel02_family(
    sec_def: FiberSection2dDef,
    subgroup_indices: List[Int],
    points: List[FiberSection2dBatchPoint],
    mut uniaxial_states: List[UniMaterialState],
    mut results: List[FiberSection2dBatchResult],
):
    for j in range(len(subgroup_indices)):
        var nonlinear_index = subgroup_indices[j]
        var y_rel = sec_def.nonlinear_y_rel[nonlinear_index]
        var area = sec_def.nonlinear_area[nonlinear_index]
        var mat_def = sec_def.nonlinear_mat_defs[nonlinear_index]
        for i in range(len(points)):
            var point = points[i]
            var state_index = point.section_state_offset + sec_def.elastic_count + nonlinear_index
            ref state = uniaxial_states[state_index]
            var eps = point.eps0 - y_rel * point.kappa
            uniaxial_set_trial_strain_steel02(mat_def, state, eps)
            _fiber_section2d_batch_accumulate_nonlinear_response(
                y_rel,
                area,
                state.sig_t,
                state.tangent_t,
                results[i].response,
            )


fn _fiber_section2d_batch_apply_concrete02_family(
    sec_def: FiberSection2dDef,
    subgroup_indices: List[Int],
    points: List[FiberSection2dBatchPoint],
    mut uniaxial_states: List[UniMaterialState],
    mut results: List[FiberSection2dBatchResult],
):
    for j in range(len(subgroup_indices)):
        var nonlinear_index = subgroup_indices[j]
        var y_rel = sec_def.nonlinear_y_rel[nonlinear_index]
        var area = sec_def.nonlinear_area[nonlinear_index]
        var mat_def = sec_def.nonlinear_mat_defs[nonlinear_index]
        for i in range(len(points)):
            var point = points[i]
            var state_index = point.section_state_offset + sec_def.elastic_count + nonlinear_index
            ref state = uniaxial_states[state_index]
            var eps = point.eps0 - y_rel * point.kappa
            uniaxial_set_trial_strain_concrete02(mat_def, state, eps)
            _fiber_section2d_batch_accumulate_nonlinear_response(
                y_rel,
                area,
                state.sig_t,
                state.tangent_t,
                results[i].response,
            )


fn _fiber_section2d_batch_apply_other_family(
    sec_def: FiberSection2dDef,
    subgroup_indices: List[Int],
    points: List[FiberSection2dBatchPoint],
    mut uniaxial_states: List[UniMaterialState],
    mut results: List[FiberSection2dBatchResult],
):
    for j in range(len(subgroup_indices)):
        var nonlinear_index = subgroup_indices[j]
        var y_rel = sec_def.nonlinear_y_rel[nonlinear_index]
        var area = sec_def.nonlinear_area[nonlinear_index]
        var mat_def = sec_def.nonlinear_mat_defs[nonlinear_index]
        for i in range(len(points)):
            var point = points[i]
            var state_index = point.section_state_offset + sec_def.elastic_count + nonlinear_index
            ref state = uniaxial_states[state_index]
            var eps = point.eps0 - y_rel * point.kappa
            uniaxial_set_trial_strain(mat_def, state, eps)
            _fiber_section2d_batch_accumulate_nonlinear_response(
                y_rel,
                area,
                state.sig_t,
                state.tangent_t,
                results[i].response,
            )


fn fiber_section2d_batch_eval_same_def(
    sec_def: FiberSection2dDef,
    mut uniaxial_states: List[UniMaterialState],
    points: List[FiberSection2dBatchPoint],
    mut results: List[FiberSection2dBatchResult],
    mut profile: FiberSection2dBatchProfile,
) -> Bool:
    var batch_size = len(points)
    if batch_size <= 0:
        results.clear()
        return True

    _fiber_section2d_batch_profile_note_batch_size(batch_size, profile)
    profile.homogeneous_batches += 1
    if batch_size <= 1:
        profile.scalar_fallback_batches += 1
        results.resize(batch_size, FiberSection2dBatchResult())
        for i in range(batch_size):
            var point = points[i]
            _fiber_section2d_batch_profile_note_flags(point.flags, profile)
            profile.section_point_evals += 1
            var resp = fiber_section2d_set_trial_from_offset(
                sec_def,
                uniaxial_states,
                point.section_state_offset,
                point.section_state_count,
                point.eps0,
                point.kappa,
            )
            var det = resp.k11 * resp.k22 - resp.k12 * resp.k12
            if abs(det) <= 1.0e-40:
                results[i] = FiberSection2dBatchResult(resp, False, 0.0, 0.0, 0.0)
                return False
            var inv_det = 1.0 / det
            results[i] = FiberSection2dBatchResult(
                resp,
                True,
                resp.k22 * inv_det,
                -resp.k12 * inv_det,
                resp.k11 * inv_det,
            )
        return True

    results.resize(batch_size, FiberSection2dBatchResult())
    for i in range(batch_size):
        var point = points[i]
        if point.section_state_count != sec_def.fiber_count:
            abort("FiberSection2d section state count mismatch")
        if point.section_state_offset + point.section_state_count > len(uniaxial_states):
            abort("FiberSection2d section states out of range")
        _fiber_section2d_batch_profile_note_flags(point.flags, profile)
        profile.section_point_evals += 1
        var elastic_resp = _fiber_section2d_elastic_response_simd[FLOAT64_SIMD_WIDTH](
            sec_def.elastic_y_rel,
            sec_def.elastic_area,
            sec_def.elastic_modulus,
            sec_def.elastic_count,
            point.eps0,
            point.kappa,
        )
        results[i] = FiberSection2dBatchResult(
            FiberSection2dResponse(
                elastic_resp[0],
                elastic_resp[1],
                elastic_resp[2],
                elastic_resp[3],
                elastic_resp[4],
            ),
            False,
            0.0,
            0.0,
            0.0,
        )

    _fiber_section2d_batch_apply_steel01_family(
        sec_def,
        sec_def.steel01_nonlinear_indices,
        points,
        uniaxial_states,
        results,
    )
    _fiber_section2d_batch_apply_concrete01_family(
        sec_def,
        sec_def.concrete01_nonlinear_indices,
        points,
        uniaxial_states,
        results,
    )
    _fiber_section2d_batch_apply_steel02_family(
        sec_def,
        sec_def.steel02_nonlinear_indices,
        points,
        uniaxial_states,
        results,
    )
    _fiber_section2d_batch_apply_concrete02_family(
        sec_def,
        sec_def.concrete02_nonlinear_indices,
        points,
        uniaxial_states,
        results,
    )
    _fiber_section2d_batch_apply_other_family(
        sec_def,
        sec_def.other_nonlinear_indices,
        points,
        uniaxial_states,
        results,
    )

    for i in range(batch_size):
        var resp = results[i].response
        var det = resp.k11 * resp.k22 - resp.k12 * resp.k12
        if abs(det) <= 1.0e-40:
            results[i] = FiberSection2dBatchResult(resp, False, 0.0, 0.0, 0.0)
            return False
        var inv_det = 1.0 / det
        results[i] = FiberSection2dBatchResult(
            resp,
            True,
            resp.k22 * inv_det,
            -resp.k12 * inv_det,
            resp.k11 * inv_det,
        )
    return True


fn fiber_section2d_batch_eval(
    defs: List[FiberSection2dDef],
    mut uniaxial_states: List[UniMaterialState],
    points: List[FiberSection2dBatchPoint],
    mut results: List[FiberSection2dBatchResult],
    mut profile: FiberSection2dBatchProfile,
) -> Bool:
    var batch_size = len(points)
    if batch_size <= 0:
        results.clear()
        return True

    var def_index0 = points[0].section_def_index
    var homogeneous = True
    for i in range(batch_size):
        if points[i].section_def_index != def_index0:
            homogeneous = False
            break

    if homogeneous:
        if def_index0 < 0 or def_index0 >= len(defs):
            abort("FiberSection2d batch section definition out of range")
        return fiber_section2d_batch_eval_same_def(
            defs[def_index0],
            uniaxial_states,
            points,
            results,
            profile,
        )

    _fiber_section2d_batch_profile_note_batch_size(batch_size, profile)
    profile.irregular_batches += 1
    if batch_size <= 1:
        profile.scalar_fallback_batches += 1

    results.resize(batch_size, FiberSection2dBatchResult())
    for i in range(batch_size):
        var point = points[i]
        var def_index = point.section_def_index
        if def_index < 0 or def_index >= len(defs):
            abort("FiberSection2d batch section definition out of range")
        _fiber_section2d_batch_profile_note_flags(point.flags, profile)
        profile.section_point_evals += 1
        var sec_def = defs[def_index]
        var resp = fiber_section2d_set_trial_from_offset(
            sec_def,
            uniaxial_states,
            point.section_state_offset,
            point.section_state_count,
            point.eps0,
            point.kappa,
        )
        var det = resp.k11 * resp.k22 - resp.k12 * resp.k12
        if abs(det) <= 1.0e-40:
            results[i] = FiberSection2dBatchResult(resp, False, 0.0, 0.0, 0.0)
            return False
        var inv_det = 1.0 / det
        results[i] = FiberSection2dBatchResult(
            resp,
            True,
            resp.k22 * inv_det,
            -resp.k12 * inv_det,
            resp.k11 * inv_det,
        )
    return True


fn _resolve_uniaxial_def_index(
    mat_id: Int, uniaxial_def_by_id: List[Int]
) -> Int:
    if mat_id < 0:
        abort("FiberSection2d material id must be >= 0")
    if mat_id >= len(uniaxial_def_by_id) or uniaxial_def_by_id[mat_id] < 0:
        abort("FiberSection2d requires uniaxial material for all fibers")
    return uniaxial_def_by_id[mat_id]


fn _build_fiber_section2d_def(
    fiber_offset: Int,
    fiber_count: Int,
    y_bar: Float64,
    fibers: List[FiberCell],
    uniaxial_defs: List[UniMaterialDef],
) -> FiberSection2dDef:
    var sec_def = FiberSection2dDef(fiber_offset, fiber_count, y_bar)
    var initial_k11 = 0.0
    var initial_k12 = 0.0
    var initial_k22 = 0.0
    for i in range(fiber_count):
        var cell = fibers[fiber_offset + i]
        var def_index = cell.def_index
        if def_index < 0 or def_index >= len(uniaxial_defs):
            abort("FiberSection2d fiber material definition out of range")
        var mat_def = uniaxial_defs[def_index]
        var y_rel = cell.y - y_bar
        var initial_ks = uni_mat_initial_tangent(mat_def) * cell.area
        initial_k11 += initial_ks
        initial_k12 += -initial_ks * y_rel
        initial_k22 += initial_ks * y_rel * y_rel
        if uni_mat_is_elastic(mat_def):
            sec_def.elastic_y_rel.append(y_rel)
            sec_def.elastic_area.append(cell.area)
            sec_def.elastic_modulus.append(uni_mat_initial_tangent(mat_def))
            sec_def.elastic_def_index.append(def_index)
        else:
            var nonlinear_index = len(sec_def.nonlinear_y_rel)
            sec_def.nonlinear_y_rel.append(y_rel)
            sec_def.nonlinear_area.append(cell.area)
            sec_def.nonlinear_def_index.append(def_index)
            sec_def.nonlinear_mat_defs.append(mat_def)
            if mat_def.mat_type == UniMaterialTypeTag.Steel01:
                sec_def.steel01_nonlinear_indices.append(nonlinear_index)
            elif mat_def.mat_type == UniMaterialTypeTag.Concrete01:
                sec_def.concrete01_nonlinear_indices.append(nonlinear_index)
            elif mat_def.mat_type == UniMaterialTypeTag.Steel02:
                sec_def.steel02_nonlinear_indices.append(nonlinear_index)
            elif mat_def.mat_type == UniMaterialTypeTag.Concrete02:
                sec_def.concrete02_nonlinear_indices.append(nonlinear_index)
            else:
                sec_def.other_nonlinear_indices.append(nonlinear_index)
    sec_def.elastic_count = len(sec_def.elastic_y_rel)
    sec_def.nonlinear_count = len(sec_def.nonlinear_y_rel)
    var det = initial_k11 * initial_k22 - initial_k12 * initial_k12
    if abs(det) > 1.0e-40:
        var inv_det = 1.0 / det
        sec_def.initial_flex_valid = True
        sec_def.initial_f00 = initial_k22 * inv_det
        sec_def.initial_f01 = -initial_k12 * inv_det
        sec_def.initial_f11 = initial_k11 * inv_det
    return sec_def


@always_inline
fn _fiber_section2d_elastic_response_simd[width: Int](
    y_rel: List[Float64],
    area: List[Float64],
    modulus: List[Float64],
    count: Int,
    eps0: Float64,
    kappa: Float64,
) -> (Float64, Float64, Float64, Float64, Float64):
    var axial_force = 0.0
    var moment_z = 0.0
    var k11 = 0.0
    var k12 = 0.0
    var k22 = 0.0

    @parameter
    fn accumulate_chunk[chunk: Int](i: Int):
        var y_vec = load_float64_contiguous_simd[chunk](y_rel, i)
        var area_vec = load_float64_contiguous_simd[chunk](area, i)
        var modulus_vec = load_float64_contiguous_simd[chunk](modulus, i)
        var ks_vec = modulus_vec * area_vec
        var fs_vec = ks_vec * (
            SIMD[DType.float64, chunk](eps0)
            - y_vec * SIMD[DType.float64, chunk](kappa)
        )
        axial_force += fs_vec.reduce_add()
        moment_z += (-fs_vec * y_vec).reduce_add()
        k11 += ks_vec.reduce_add()
        k12 += (-ks_vec * y_vec).reduce_add()
        k22 += (ks_vec * y_vec * y_vec).reduce_add()

    vectorize[accumulate_chunk, width](count)

    if count > 0:
        var modulus0 = modulus[0]
        var uniform_modulus = True
        for j in range(1, count):
            if modulus[j] != modulus0:
                uniform_modulus = False
                break
        if uniform_modulus:
            k12 = 0.0

    return (axial_force, moment_z, k11, k12, k22)


fn _append_rect_patch_cells(
    patch: PythonObject,
    uniaxial_def_by_id: List[Int],
    mut fibers: List[FiberCell],
    mut area_sum: Float64,
    mut qz_sum: Float64,
) raises:
    var mat_id = Int(patch["material"])
    var def_index = _resolve_uniaxial_def_index(mat_id, uniaxial_def_by_id)
    var ny = Int(patch["num_subdiv_y"])
    var nz = Int(patch["num_subdiv_z"])
    if ny <= 0 or nz <= 0:
        abort("FiberSection2d patch rect requires num_subdiv_y and num_subdiv_z > 0")

    var y_i = Float64(patch["y_i"])
    var z_i = Float64(patch["z_i"])
    var y_j = Float64(patch["y_j"])
    var z_j = Float64(patch["z_j"])

    var y_min = y_i
    var y_max = y_j
    if y_min > y_max:
        y_min = y_j
        y_max = y_i

    var z_min = z_i
    var z_max = z_j
    if z_min > z_max:
        z_min = z_j
        z_max = z_i

    var dy_total = y_max - y_min
    var dz_total = z_max - z_min
    if dy_total <= 0.0 or dz_total <= 0.0:
        abort("FiberSection2d patch rect must have non-zero side lengths")

    var dy = dy_total / Float64(ny)
    var dz = dz_total / Float64(nz)
    var area = dy * dz

    for iy in range(ny):
        var y = y_min + (Float64(iy) + 0.5) * dy
        for iz in range(nz):
            var z = z_min + (Float64(iz) + 0.5) * dz
            fibers.append(FiberCell(y, z, area, def_index))
            area_sum += area
            qz_sum += area * y


fn _append_quadr_patch_cells(
    patch: PythonObject,
    uniaxial_def_by_id: List[Int],
    mut fibers: List[FiberCell],
    mut area_sum: Float64,
    mut qz_sum: Float64,
) raises:
    var mat_id = Int(patch["material"])
    var def_index = _resolve_uniaxial_def_index(mat_id, uniaxial_def_by_id)
    var ny = Int(patch["num_subdiv_y"])
    var nz = Int(patch["num_subdiv_z"])
    if ny <= 0 or nz <= 0:
        abort("FiberSection2d patch quadr requires num_subdiv_y and num_subdiv_z > 0")

    var y_i = Float64(patch["y_i"])
    var z_i = Float64(patch["z_i"])
    var y_j = Float64(patch["y_j"])
    var z_j = Float64(patch["z_j"])
    var y_k = Float64(patch["y_k"])
    var z_k = Float64(patch["z_k"])
    var y_l = Float64(patch["y_l"])
    var z_l = Float64(patch["z_l"])

    for iy in range(ny):
        var u0 = Float64(iy) / Float64(ny)
        var u1 = Float64(iy + 1) / Float64(ny)
        var uc = 0.5 * (u0 + u1)
        for iz in range(nz):
            var v0 = Float64(iz) / Float64(nz)
            var v1 = Float64(iz + 1) / Float64(nz)
            var vc = 0.5 * (v0 + v1)

            var y00 = (1.0 - u0) * (1.0 - v0) * y_i + u0 * (1.0 - v0) * y_j + u0 * v0 * y_k + (1.0 - u0) * v0 * y_l
            var z00 = (1.0 - u0) * (1.0 - v0) * z_i + u0 * (1.0 - v0) * z_j + u0 * v0 * z_k + (1.0 - u0) * v0 * z_l
            var y10 = (1.0 - u1) * (1.0 - v0) * y_i + u1 * (1.0 - v0) * y_j + u1 * v0 * y_k + (1.0 - u1) * v0 * y_l
            var z10 = (1.0 - u1) * (1.0 - v0) * z_i + u1 * (1.0 - v0) * z_j + u1 * v0 * z_k + (1.0 - u1) * v0 * z_l
            var y11 = (1.0 - u1) * (1.0 - v1) * y_i + u1 * (1.0 - v1) * y_j + u1 * v1 * y_k + (1.0 - u1) * v1 * y_l
            var z11 = (1.0 - u1) * (1.0 - v1) * z_i + u1 * (1.0 - v1) * z_j + u1 * v1 * z_k + (1.0 - u1) * v1 * z_l
            var y01 = (1.0 - u0) * (1.0 - v1) * y_i + u0 * (1.0 - v1) * y_j + u0 * v1 * y_k + (1.0 - u0) * v1 * y_l
            var z01 = (1.0 - u0) * (1.0 - v1) * z_i + u0 * (1.0 - v1) * z_j + u0 * v1 * z_k + (1.0 - u0) * v1 * z_l

            var twice_area = (
                y00 * z10
                + y10 * z11
                + y11 * z01
                + y01 * z00
                - z00 * y10
                - z10 * y11
                - z11 * y01
                - z01 * y00
            )
            var area = abs(twice_area) * 0.5
            if area <= 0.0:
                abort("FiberSection2d patch quadr generated zero-area cell")

            var yc = (1.0 - uc) * (1.0 - vc) * y_i + uc * (1.0 - vc) * y_j + uc * vc * y_k + (1.0 - uc) * vc * y_l
            var zc = (1.0 - uc) * (1.0 - vc) * z_i + uc * (1.0 - vc) * z_j + uc * vc * z_k + (1.0 - uc) * vc * z_l
            fibers.append(FiberCell(yc, zc, area, def_index))
            area_sum += area
            qz_sum += area * yc


fn _append_straight_layer_cells(
    layer: PythonObject,
    uniaxial_def_by_id: List[Int],
    mut fibers: List[FiberCell],
    mut area_sum: Float64,
    mut qz_sum: Float64,
) raises:
    var mat_id = Int(layer["material"])
    var def_index = _resolve_uniaxial_def_index(mat_id, uniaxial_def_by_id)
    var num_bars = Int(layer["num_bars"])
    if num_bars <= 0:
        abort("FiberSection2d layer straight requires num_bars > 0")
    var bar_area = Float64(layer["bar_area"])
    if bar_area <= 0.0:
        abort("FiberSection2d layer straight requires bar_area > 0")

    var y_start = Float64(layer["y_start"])
    var z_start = Float64(layer["z_start"])
    var y_end = Float64(layer["y_end"])
    var z_end = Float64(layer["z_end"])

    if num_bars == 1:
        var y = 0.5 * (y_start + y_end)
        var z = 0.5 * (z_start + z_end)
        fibers.append(FiberCell(y, z, bar_area, def_index))
        area_sum += bar_area
        qz_sum += bar_area * y
        return

    for i in range(num_bars):
        var t = Float64(i) / Float64(num_bars - 1)
        var y = y_start + (y_end - y_start) * t
        var z = z_start + (z_end - z_start) * t
        fibers.append(FiberCell(y, z, bar_area, def_index))
        area_sum += bar_area
        qz_sum += bar_area * y


fn append_fiber_section2d_from_json(
    sec: PythonObject,
    uniaxial_def_by_id: List[Int],
    uniaxial_defs: List[UniMaterialDef],
    mut defs: List[FiberSection2dDef],
    mut fibers: List[FiberCell],
) raises:
    if String(sec["type"]) != "FiberSection2d":
        abort("append_fiber_section2d_from_json requires FiberSection2d")

    var params = sec["params"]
    var patches = params.get("patches", [])
    var layers = params.get("layers", [])
    if py_len(patches) == 0 and py_len(layers) == 0:
        abort("FiberSection2d requires at least one patch or layer")

    var fiber_offset = len(fibers)
    var area_sum = 0.0
    var qz_sum = 0.0

    for i in range(py_len(patches)):
        var patch = patches[i]
        var patch_type = String(patch["type"])
        if patch_type == "quad":
            patch_type = "quadr"
        if patch_type == "rect":
            _append_rect_patch_cells(
                patch, uniaxial_def_by_id, fibers, area_sum, qz_sum
            )
        elif patch_type == "quadr":
            _append_quadr_patch_cells(
                patch, uniaxial_def_by_id, fibers, area_sum, qz_sum
            )
        else:
            abort("unsupported FiberSection2d patch type: " + patch_type)

    for i in range(py_len(layers)):
        var layer = layers[i]
        var layer_type = String(layer["type"])
        if layer_type == "straight":
            _append_straight_layer_cells(
                layer, uniaxial_def_by_id, fibers, area_sum, qz_sum
            )
        else:
            abort("unsupported FiberSection2d layer type: " + layer_type)

    var fiber_count = len(fibers) - fiber_offset
    if fiber_count <= 0:
        abort("FiberSection2d produced no fibers")
    if area_sum <= 0.0:
        abort("FiberSection2d total area must be > 0")
    var y_bar = qz_sum / area_sum

    defs.append(
        _build_fiber_section2d_def(
            fiber_offset, fiber_count, y_bar, fibers, uniaxial_defs
        )
    )


fn _append_rect_patch_cells_input(
    patch: FiberPatchInput,
    uniaxial_def_by_id: List[Int],
    mut fibers: List[FiberCell],
    mut area_sum: Float64,
    mut qz_sum: Float64,
):
    var mat_id = patch.material
    var def_index = _resolve_uniaxial_def_index(mat_id, uniaxial_def_by_id)
    var ny = patch.num_subdiv_y
    var nz = patch.num_subdiv_z
    if ny <= 0 or nz <= 0:
        abort("FiberSection2d patch rect requires num_subdiv_y and num_subdiv_z > 0")

    var y_i = patch.y_i
    var z_i = patch.z_i
    var y_j = patch.y_j
    var z_j = patch.z_j

    var y_min = y_i
    var y_max = y_j
    if y_min > y_max:
        y_min = y_j
        y_max = y_i

    var z_min = z_i
    var z_max = z_j
    if z_min > z_max:
        z_min = z_j
        z_max = z_i

    var dy_total = y_max - y_min
    var dz_total = z_max - z_min
    if dy_total <= 0.0 or dz_total <= 0.0:
        abort("FiberSection2d patch rect must have non-zero side lengths")

    var dy = dy_total / Float64(ny)
    var dz = dz_total / Float64(nz)
    var area = dy * dz

    for iy in range(ny):
        var y = y_min + (Float64(iy) + 0.5) * dy
        for iz in range(nz):
            var z = z_min + (Float64(iz) + 0.5) * dz
            fibers.append(FiberCell(y, z, area, def_index))
            area_sum += area
            qz_sum += area * y


fn _append_quadr_patch_cells_input(
    patch: FiberPatchInput,
    uniaxial_def_by_id: List[Int],
    mut fibers: List[FiberCell],
    mut area_sum: Float64,
    mut qz_sum: Float64,
):
    var mat_id = patch.material
    var def_index = _resolve_uniaxial_def_index(mat_id, uniaxial_def_by_id)
    var ny = patch.num_subdiv_y
    var nz = patch.num_subdiv_z
    if ny <= 0 or nz <= 0:
        abort("FiberSection2d patch quadr requires num_subdiv_y and num_subdiv_z > 0")

    var y_i = patch.y_i
    var z_i = patch.z_i
    var y_j = patch.y_j
    var z_j = patch.z_j
    var y_k = patch.y_k
    var z_k = patch.z_k
    var y_l = patch.y_l
    var z_l = patch.z_l

    for iy in range(ny):
        var u0 = Float64(iy) / Float64(ny)
        var u1 = Float64(iy + 1) / Float64(ny)
        var uc = 0.5 * (u0 + u1)
        for iz in range(nz):
            var v0 = Float64(iz) / Float64(nz)
            var v1 = Float64(iz + 1) / Float64(nz)
            var vc = 0.5 * (v0 + v1)

            var y00 = (1.0 - u0) * (1.0 - v0) * y_i + u0 * (1.0 - v0) * y_j + u0 * v0 * y_k + (1.0 - u0) * v0 * y_l
            var z00 = (1.0 - u0) * (1.0 - v0) * z_i + u0 * (1.0 - v0) * z_j + u0 * v0 * z_k + (1.0 - u0) * v0 * z_l
            var y10 = (1.0 - u1) * (1.0 - v0) * y_i + u1 * (1.0 - v0) * y_j + u1 * v0 * y_k + (1.0 - u1) * v0 * y_l
            var z10 = (1.0 - u1) * (1.0 - v0) * z_i + u1 * (1.0 - v0) * z_j + u1 * v0 * z_k + (1.0 - u1) * v0 * z_l
            var y11 = (1.0 - u1) * (1.0 - v1) * y_i + u1 * (1.0 - v1) * y_j + u1 * v1 * y_k + (1.0 - u1) * v1 * y_l
            var z11 = (1.0 - u1) * (1.0 - v1) * z_i + u1 * (1.0 - v1) * z_j + u1 * v1 * z_k + (1.0 - u1) * v1 * z_l
            var y01 = (1.0 - u0) * (1.0 - v1) * y_i + u0 * (1.0 - v1) * y_j + u0 * v1 * y_k + (1.0 - u0) * v1 * y_l
            var z01 = (1.0 - u0) * (1.0 - v1) * z_i + u0 * (1.0 - v1) * z_j + u0 * v1 * z_k + (1.0 - u0) * v1 * z_l

            var twice_area = (
                y00 * z10
                + y10 * z11
                + y11 * z01
                + y01 * z00
                - z00 * y10
                - z10 * y11
                - z11 * y01
                - z01 * y00
            )
            var area = abs(twice_area) * 0.5
            if area <= 0.0:
                abort("FiberSection2d patch quadr generated zero-area cell")

            var yc = (1.0 - uc) * (1.0 - vc) * y_i + uc * (1.0 - vc) * y_j + uc * vc * y_k + (1.0 - uc) * vc * y_l
            var zc = (1.0 - uc) * (1.0 - vc) * z_i + uc * (1.0 - vc) * z_j + uc * vc * z_k + (1.0 - uc) * vc * z_l
            fibers.append(FiberCell(yc, zc, area, def_index))
            area_sum += area
            qz_sum += area * yc


fn _append_straight_layer_cells_input(
    layer: FiberLayerInput,
    uniaxial_def_by_id: List[Int],
    mut fibers: List[FiberCell],
    mut area_sum: Float64,
    mut qz_sum: Float64,
):
    var mat_id = layer.material
    var def_index = _resolve_uniaxial_def_index(mat_id, uniaxial_def_by_id)
    var num_bars = layer.num_bars
    if num_bars <= 0:
        abort("FiberSection2d layer straight requires num_bars > 0")
    var bar_area = layer.bar_area
    if bar_area <= 0.0:
        abort("FiberSection2d layer straight requires bar_area > 0")

    var y_start = layer.y_start
    var z_start = layer.z_start
    var y_end = layer.y_end
    var z_end = layer.z_end

    if num_bars == 1:
        var y = 0.5 * (y_start + y_end)
        var z = 0.5 * (z_start + z_end)
        fibers.append(FiberCell(y, z, bar_area, def_index))
        area_sum += bar_area
        qz_sum += bar_area * y
        return

    for i in range(num_bars):
        var t = Float64(i) / Float64(num_bars - 1)
        var y = y_start + (y_end - y_start) * t
        var z = z_start + (z_end - z_start) * t
        fibers.append(FiberCell(y, z, bar_area, def_index))
        area_sum += bar_area
        qz_sum += bar_area * y


fn append_fiber_section2d_from_input(
    sec: SectionInput,
    fiber_patches: List[FiberPatchInput],
    fiber_layers: List[FiberLayerInput],
    uniaxial_def_by_id: List[Int],
    uniaxial_defs: List[UniMaterialDef],
    mut defs: List[FiberSection2dDef],
    mut fibers: List[FiberCell],
):
    if sec.type != "FiberSection2d":
        abort("append_fiber_section2d_from_input requires FiberSection2d")
    if sec.fiber_patch_count == 0 and sec.fiber_layer_count == 0:
        abort("FiberSection2d requires at least one patch or layer")

    var fiber_offset = len(fibers)
    var area_sum = 0.0
    var qz_sum = 0.0

    for i in range(sec.fiber_patch_count):
        var patch = fiber_patches[sec.fiber_patch_offset + i]
        if patch.type == "rect":
            _append_rect_patch_cells_input(
                patch, uniaxial_def_by_id, fibers, area_sum, qz_sum
            )
        elif patch.type == "quadr":
            _append_quadr_patch_cells_input(
                patch, uniaxial_def_by_id, fibers, area_sum, qz_sum
            )
        else:
            abort("unsupported FiberSection2d patch type: " + patch.type)

    for i in range(sec.fiber_layer_count):
        var layer = fiber_layers[sec.fiber_layer_offset + i]
        if layer.type == "straight":
            _append_straight_layer_cells_input(
                layer, uniaxial_def_by_id, fibers, area_sum, qz_sum
            )
        else:
            abort("unsupported FiberSection2d layer type: " + layer.type)

    var fiber_count = len(fibers) - fiber_offset
    if fiber_count <= 0:
        abort("FiberSection2d produced no fibers")
    if area_sum <= 0.0:
        abort("FiberSection2d total area must be > 0")
    var y_bar = qz_sum / area_sum

    defs.append(
        _build_fiber_section2d_def(
            fiber_offset, fiber_count, y_bar, fibers, uniaxial_defs
        )
    )


fn fiber_section2d_init_states(
    defs: List[FiberSection2dDef],
    fibers: List[FiberCell],
    uniaxial_defs: List[UniMaterialDef],
    mut uniaxial_states: List[UniMaterialState],
    mut uniaxial_state_defs: List[Int],
    mut section_uniaxial_offsets: List[Int],
    mut section_uniaxial_counts: List[Int],
) -> Bool:
    section_uniaxial_offsets.resize(len(defs), 0)
    section_uniaxial_counts.resize(len(defs), 0)
    var used_nonelastic = False

    for s in range(len(defs)):
        var sec_def = defs[s]
        section_uniaxial_offsets[s] = len(uniaxial_states)
        section_uniaxial_counts[s] = sec_def.fiber_count
        if sec_def.fiber_offset < 0 or sec_def.fiber_offset + sec_def.fiber_count > len(
            fibers
        ):
            abort("FiberSection2d fiber data out of range")
        for i in range(sec_def.elastic_count):
            var def_index = sec_def.elastic_def_index[i]
            if def_index < 0 or def_index >= len(uniaxial_defs):
                abort("FiberSection2d fiber material definition out of range")
            var mat_def = uniaxial_defs[def_index]
            var state_index = len(uniaxial_states)
            uniaxial_states.append(UniMaterialState(mat_def))
            uniaxial_state_defs.append(def_index)
            if state_index != section_uniaxial_offsets[s] + i:
                abort("FiberSection2d elastic state layout must be contiguous")
        for i in range(sec_def.nonlinear_count):
            var mat_def = sec_def.nonlinear_mat_defs[i]
            var def_index = sec_def.nonlinear_def_index[i]
            var state_index = len(uniaxial_states)
            uniaxial_states.append(UniMaterialState(mat_def))
            uniaxial_state_defs.append(def_index)
            if state_index != (
                section_uniaxial_offsets[s] + sec_def.elastic_count + i
            ):
                abort("FiberSection2d nonlinear state layout must be contiguous")
            if not uni_mat_is_elastic(mat_def):
                used_nonelastic = True

    return used_nonelastic


fn fiber_section2d_set_trial_from_offset(
    sec_def: FiberSection2dDef,
    mut uniaxial_states: List[UniMaterialState],
    section_state_offset: Int,
    section_state_count: Int,
    eps0: Float64,
    kappa: Float64,
) -> FiberSection2dResponse:
    if section_state_count != sec_def.fiber_count:
        abort("FiberSection2d section state count mismatch")
    if section_state_offset + section_state_count > len(uniaxial_states):
        abort("FiberSection2d section states out of range")

    var elastic_resp = _fiber_section2d_elastic_response_simd[FLOAT64_SIMD_WIDTH](
        sec_def.elastic_y_rel,
        sec_def.elastic_area,
        sec_def.elastic_modulus,
        sec_def.elastic_count,
        eps0,
        kappa,
    )
    var axial_force = elastic_resp[0]
    var moment_z = elastic_resp[1]
    var k11 = elastic_resp[2]
    var k12 = elastic_resp[3]
    var k22 = elastic_resp[4]

    var nonlinear_state_offset = section_state_offset + sec_def.elastic_count
    for i in range(sec_def.nonlinear_count):
        var y_rel = sec_def.nonlinear_y_rel[i]
        var eps = eps0 - y_rel * kappa
        var state_index = nonlinear_state_offset + i
        var mat_def = sec_def.nonlinear_mat_defs[i]
        ref state = uniaxial_states[state_index]
        uniaxial_set_trial_strain(mat_def, state, eps)

        var area = sec_def.nonlinear_area[i]
        var fs = state.sig_t * area
        var ks = state.tangent_t * area
        axial_force += fs
        moment_z += -fs * y_rel
        k11 += ks
        k12 += -ks * y_rel
        k22 += ks * y_rel * y_rel

    return FiberSection2dResponse(axial_force, moment_z, k11, k12, k22)


fn fiber_section2d_set_trial(
    section_index: Int,
    defs: List[FiberSection2dDef],
    section_uniaxial_offsets: List[Int],
    section_uniaxial_counts: List[Int],
    mut uniaxial_states: List[UniMaterialState],
    eps0: Float64,
    kappa: Float64,
) -> FiberSection2dResponse:
    if section_index < 0 or section_index >= len(defs):
        abort("FiberSection2d section index out of range")
    if section_index >= len(section_uniaxial_offsets) or section_index >= len(
        section_uniaxial_counts
    ):
        abort("FiberSection2d section state mapping missing")

    var sec_def = defs[section_index]
    var offset = section_uniaxial_offsets[section_index]
    var count = section_uniaxial_counts[section_index]
    if count != sec_def.fiber_count:
        abort("FiberSection2d section state count mismatch")

    return fiber_section2d_set_trial_from_offset(
        sec_def,
        uniaxial_states,
        offset,
        count,
        eps0,
        kappa,
    )


fn fiber_section2d_commit_from_offset(
    mut uniaxial_states: List[UniMaterialState],
    section_state_offset: Int,
    section_state_count: Int,
):
    if section_state_offset + section_state_count > len(uniaxial_states):
        abort("FiberSection2d section states out of range")
    for i in range(section_state_count):
        ref state = uniaxial_states[section_state_offset + i]
        uniaxial_commit(state)


fn fiber_section2d_revert_trial_from_offset(
    mut uniaxial_states: List[UniMaterialState],
    section_state_offset: Int,
    section_state_count: Int,
):
    if section_state_offset + section_state_count > len(uniaxial_states):
        abort("FiberSection2d section states out of range")
    for i in range(section_state_count):
        ref state = uniaxial_states[section_state_offset + i]
        uniaxial_revert_trial(state)


fn fiber_section2d_commit(
    section_index: Int,
    section_uniaxial_offsets: List[Int],
    section_uniaxial_counts: List[Int],
    mut uniaxial_states: List[UniMaterialState],
):
    if section_index < 0 or section_index >= len(section_uniaxial_offsets):
        abort("FiberSection2d section index out of range")
    if section_index >= len(section_uniaxial_counts):
        abort("FiberSection2d section state mapping missing")

    var offset = section_uniaxial_offsets[section_index]
    var count = section_uniaxial_counts[section_index]
    fiber_section2d_commit_from_offset(uniaxial_states, offset, count)


fn fiber_section2d_revert_trial(
    section_index: Int,
    section_uniaxial_offsets: List[Int],
    section_uniaxial_counts: List[Int],
    mut uniaxial_states: List[UniMaterialState],
):
    if section_index < 0 or section_index >= len(section_uniaxial_offsets):
        abort("FiberSection2d section index out of range")
    if section_index >= len(section_uniaxial_counts):
        abort("FiberSection2d section state mapping missing")

    var offset = section_uniaxial_offsets[section_index]
    var count = section_uniaxial_counts[section_index]
    fiber_section2d_revert_trial_from_offset(uniaxial_states, offset, count)


fn fiber_section2d_commit_all(
    section_uniaxial_offsets: List[Int],
    section_uniaxial_counts: List[Int],
    mut uniaxial_states: List[UniMaterialState],
):
    for i in range(len(section_uniaxial_offsets)):
        fiber_section2d_commit(i, section_uniaxial_offsets, section_uniaxial_counts, uniaxial_states)


fn fiber_section2d_revert_trial_all(
    section_uniaxial_offsets: List[Int],
    section_uniaxial_counts: List[Int],
    mut uniaxial_states: List[UniMaterialState],
):
    for i in range(len(section_uniaxial_offsets)):
        fiber_section2d_revert_trial(
            i, section_uniaxial_offsets, section_uniaxial_counts, uniaxial_states
        )
