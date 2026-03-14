from algorithm import vectorize
from collections import List
from os import abort
from python import PythonObject

from materials import (
    UniMaterialDef,
    UniMaterialState,
    uni_mat_is_elastic,
    uni_mat_initial_tangent,
)
from materials.uniaxial.concrete01 import _concrete01_reload
from materials.uniaxial.concrete02 import _concrete02_compr_envlp, _concrete02_tens_envlp
from materials.uniaxial.core import _abs, _pow_abs, _sign
from solver.run_case.input_types import FiberLayerInput, FiberPatchInput, SectionInput
from solver.simd_contiguous import (
    FLOAT64_SIMD_WIDTH,
    copy_float64_contiguous_simd,
    load_float64_contiguous_simd,
    store_float64_contiguous_simd,
)
from strut_io import py_len
from tag_types import UniMaterialTypeTag


@always_inline
fn _fiber_section2d_padded_family_count(count: Int) -> Int:
    if count <= 0:
        return 0
    return ((count + FLOAT64_SIMD_WIDTH - 1) // FLOAT64_SIMD_WIDTH) * FLOAT64_SIMD_WIDTH


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
    var steel02_y_rel: List[Float64]
    var steel02_area: List[Float64]
    var steel02_mat_defs: List[UniMaterialDef]
    var steel02_single_definition: Bool
    var steel02_single_mat_def: UniMaterialDef
    var steel02_group_offsets: List[Int]
    var steel02_group_counts: List[Int]
    var steel02_group_padded_counts: List[Int]
    var steel02_group_mat_defs: List[UniMaterialDef]
    var steel02_fy: List[Float64]
    var steel02_e0: List[Float64]
    var steel02_b: List[Float64]
    var steel02_r0: List[Float64]
    var steel02_cr1: List[Float64]
    var steel02_cr2: List[Float64]
    var steel02_a1: List[Float64]
    var steel02_a2: List[Float64]
    var steel02_a3: List[Float64]
    var steel02_a4: List[Float64]
    var steel02_sigini: List[Float64]
    var steel02_esh: List[Float64]
    var steel02_epsy: List[Float64]
    var steel02_sigini_over_e0: List[Float64]
    var steel02_pos_inv_denom: List[Float64]
    var steel02_neg_inv_denom: List[Float64]
    var concrete02_y_rel: List[Float64]
    var concrete02_area: List[Float64]
    var concrete02_mat_defs: List[UniMaterialDef]
    var concrete02_single_definition: Bool
    var concrete02_single_mat_def: UniMaterialDef
    var concrete02_group_offsets: List[Int]
    var concrete02_group_counts: List[Int]
    var concrete02_group_padded_counts: List[Int]
    var concrete02_group_mat_defs: List[UniMaterialDef]
    var concrete02_fc: List[Float64]
    var concrete02_epsc0: List[Float64]
    var concrete02_fcu: List[Float64]
    var concrete02_epscu: List[Float64]
    var concrete02_rat: List[Float64]
    var concrete02_ft: List[Float64]
    var concrete02_ets: List[Float64]
    var concrete02_ec0: List[Float64]
    var concrete02_epsr: List[Float64]
    var concrete02_sigmr: List[Float64]
    var steel01_nonlinear_indices: List[Int]
    var concrete01_nonlinear_indices: List[Int]
    var steel02_nonlinear_indices: List[Int]
    var steel02_family_position_by_nonlinear_index: List[Int]
    var steel02_count: Int
    var steel02_padded_count: Int
    var steel02_instance_stride: Int
    var concrete02_nonlinear_indices: List[Int]
    var concrete02_family_position_by_nonlinear_index: List[Int]
    var concrete02_count: Int
    var concrete02_padded_count: Int
    var concrete02_instance_stride: Int
    var other_nonlinear_indices: List[Int]
    var runtime_state_offset_base: Int
    var runtime_state_count: Int
    var runtime_s2_state_count: Int
    var runtime_c2_state_count: Int
    var runtime_eps_c: List[Float64]
    var runtime_sig_c: List[Float64]
    var runtime_tangent_c: List[Float64]
    var runtime_eps_p_c: List[Float64]
    var runtime_alpha_c: List[Float64]
    var runtime_eps_t: List[Float64]
    var runtime_sig_t: List[Float64]
    var runtime_tangent_t: List[Float64]
    var runtime_eps_p_t: List[Float64]
    var runtime_alpha_t: List[Float64]
    var runtime_min_strain_c: List[Float64]
    var runtime_end_strain_c: List[Float64]
    var runtime_unload_slope_c: List[Float64]
    var runtime_min_strain_t: List[Float64]
    var runtime_end_strain_t: List[Float64]
    var runtime_unload_slope_t: List[Float64]
    var runtime_s2_eps_c: List[Float64]
    var runtime_s2_sig_c: List[Float64]
    var runtime_s2_tangent_c: List[Float64]
    var runtime_s2_epsmin_c: List[Float64]
    var runtime_s2_epsmax_c: List[Float64]
    var runtime_s2_epspl_c: List[Float64]
    var runtime_s2_epss0_c: List[Float64]
    var runtime_s2_sigs0_c: List[Float64]
    var runtime_s2_epsr_c: List[Float64]
    var runtime_s2_sigr_c: List[Float64]
    var runtime_s2_kon_c: List[Int]
    var runtime_s2_eps_t: List[Float64]
    var runtime_s2_sig_t: List[Float64]
    var runtime_s2_tangent_t: List[Float64]
    var runtime_s2_epsmin_t: List[Float64]
    var runtime_s2_epsmax_t: List[Float64]
    var runtime_s2_epspl_t: List[Float64]
    var runtime_s2_epss0_t: List[Float64]
    var runtime_s2_sigs0_t: List[Float64]
    var runtime_s2_epsr_t: List[Float64]
    var runtime_s2_sigr_t: List[Float64]
    var runtime_s2_kon_t: List[Int]
    var runtime_c2_eps_c: List[Float64]
    var runtime_c2_sig_c: List[Float64]
    var runtime_c2_tangent_c: List[Float64]
    var runtime_c2_eps_t: List[Float64]
    var runtime_c2_sig_t: List[Float64]
    var runtime_c2_tangent_t: List[Float64]
    var runtime_c2_ecmin_c: List[Float64]
    var runtime_c2_dept_c: List[Float64]
    var runtime_c2_ecmin_t: List[Float64]
    var runtime_c2_dept_t: List[Float64]

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
        self.steel02_y_rel = []
        self.steel02_area = []
        self.steel02_mat_defs = []
        self.steel02_single_definition = False
        self.steel02_single_mat_def = UniMaterialDef()
        self.steel02_group_offsets = []
        self.steel02_group_counts = []
        self.steel02_group_padded_counts = []
        self.steel02_group_mat_defs = []
        self.steel02_fy = []
        self.steel02_e0 = []
        self.steel02_b = []
        self.steel02_r0 = []
        self.steel02_cr1 = []
        self.steel02_cr2 = []
        self.steel02_a1 = []
        self.steel02_a2 = []
        self.steel02_a3 = []
        self.steel02_a4 = []
        self.steel02_sigini = []
        self.steel02_esh = []
        self.steel02_epsy = []
        self.steel02_sigini_over_e0 = []
        self.steel02_pos_inv_denom = []
        self.steel02_neg_inv_denom = []
        self.concrete02_y_rel = []
        self.concrete02_area = []
        self.concrete02_mat_defs = []
        self.concrete02_single_definition = False
        self.concrete02_single_mat_def = UniMaterialDef()
        self.concrete02_group_offsets = []
        self.concrete02_group_counts = []
        self.concrete02_group_padded_counts = []
        self.concrete02_group_mat_defs = []
        self.concrete02_fc = []
        self.concrete02_epsc0 = []
        self.concrete02_fcu = []
        self.concrete02_epscu = []
        self.concrete02_rat = []
        self.concrete02_ft = []
        self.concrete02_ets = []
        self.concrete02_ec0 = []
        self.concrete02_epsr = []
        self.concrete02_sigmr = []
        self.steel01_nonlinear_indices = []
        self.concrete01_nonlinear_indices = []
        self.steel02_nonlinear_indices = []
        self.steel02_family_position_by_nonlinear_index = []
        self.steel02_count = 0
        self.steel02_padded_count = 0
        self.steel02_instance_stride = 0
        self.concrete02_nonlinear_indices = []
        self.concrete02_family_position_by_nonlinear_index = []
        self.concrete02_count = 0
        self.concrete02_padded_count = 0
        self.concrete02_instance_stride = 0
        self.other_nonlinear_indices = []
        self.runtime_state_offset_base = 0
        self.runtime_state_count = 0
        self.runtime_s2_state_count = 0
        self.runtime_c2_state_count = 0
        self.runtime_eps_c = []
        self.runtime_sig_c = []
        self.runtime_tangent_c = []
        self.runtime_eps_p_c = []
        self.runtime_alpha_c = []
        self.runtime_eps_t = []
        self.runtime_sig_t = []
        self.runtime_tangent_t = []
        self.runtime_eps_p_t = []
        self.runtime_alpha_t = []
        self.runtime_min_strain_c = []
        self.runtime_end_strain_c = []
        self.runtime_unload_slope_c = []
        self.runtime_min_strain_t = []
        self.runtime_end_strain_t = []
        self.runtime_unload_slope_t = []
        self.runtime_s2_eps_c = []
        self.runtime_s2_sig_c = []
        self.runtime_s2_tangent_c = []
        self.runtime_s2_epsmin_c = []
        self.runtime_s2_epsmax_c = []
        self.runtime_s2_epspl_c = []
        self.runtime_s2_epss0_c = []
        self.runtime_s2_sigs0_c = []
        self.runtime_s2_epsr_c = []
        self.runtime_s2_sigr_c = []
        self.runtime_s2_kon_c = []
        self.runtime_s2_eps_t = []
        self.runtime_s2_sig_t = []
        self.runtime_s2_tangent_t = []
        self.runtime_s2_epsmin_t = []
        self.runtime_s2_epsmax_t = []
        self.runtime_s2_epspl_t = []
        self.runtime_s2_epss0_t = []
        self.runtime_s2_sigs0_t = []
        self.runtime_s2_epsr_t = []
        self.runtime_s2_sigr_t = []
        self.runtime_s2_kon_t = []
        self.runtime_c2_eps_c = []
        self.runtime_c2_sig_c = []
        self.runtime_c2_tangent_c = []
        self.runtime_c2_eps_t = []
        self.runtime_c2_sig_t = []
        self.runtime_c2_tangent_t = []
        self.runtime_c2_ecmin_c = []
        self.runtime_c2_dept_c = []
        self.runtime_c2_ecmin_t = []
        self.runtime_c2_dept_t = []

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
        self.steel02_y_rel = []
        self.steel02_area = []
        self.steel02_mat_defs = []
        self.steel02_single_definition = False
        self.steel02_single_mat_def = UniMaterialDef()
        self.steel02_group_offsets = []
        self.steel02_group_counts = []
        self.steel02_group_padded_counts = []
        self.steel02_group_mat_defs = []
        self.steel02_fy = []
        self.steel02_e0 = []
        self.steel02_b = []
        self.steel02_r0 = []
        self.steel02_cr1 = []
        self.steel02_cr2 = []
        self.steel02_a1 = []
        self.steel02_a2 = []
        self.steel02_a3 = []
        self.steel02_a4 = []
        self.steel02_sigini = []
        self.steel02_esh = []
        self.steel02_epsy = []
        self.steel02_sigini_over_e0 = []
        self.steel02_pos_inv_denom = []
        self.steel02_neg_inv_denom = []
        self.concrete02_y_rel = []
        self.concrete02_area = []
        self.concrete02_mat_defs = []
        self.concrete02_single_definition = False
        self.concrete02_single_mat_def = UniMaterialDef()
        self.concrete02_group_offsets = []
        self.concrete02_group_counts = []
        self.concrete02_group_padded_counts = []
        self.concrete02_group_mat_defs = []
        self.concrete02_fc = []
        self.concrete02_epsc0 = []
        self.concrete02_fcu = []
        self.concrete02_epscu = []
        self.concrete02_rat = []
        self.concrete02_ft = []
        self.concrete02_ets = []
        self.concrete02_ec0 = []
        self.concrete02_epsr = []
        self.concrete02_sigmr = []
        self.steel01_nonlinear_indices = []
        self.concrete01_nonlinear_indices = []
        self.steel02_nonlinear_indices = []
        self.steel02_family_position_by_nonlinear_index = []
        self.steel02_count = 0
        self.steel02_padded_count = 0
        self.steel02_instance_stride = 0
        self.concrete02_nonlinear_indices = []
        self.concrete02_family_position_by_nonlinear_index = []
        self.concrete02_count = 0
        self.concrete02_padded_count = 0
        self.concrete02_instance_stride = 0
        self.other_nonlinear_indices = []
        self.runtime_state_offset_base = 0
        self.runtime_state_count = 0
        self.runtime_s2_state_count = 0
        self.runtime_c2_state_count = 0
        self.runtime_eps_c = []
        self.runtime_sig_c = []
        self.runtime_tangent_c = []
        self.runtime_eps_p_c = []
        self.runtime_alpha_c = []
        self.runtime_eps_t = []
        self.runtime_sig_t = []
        self.runtime_tangent_t = []
        self.runtime_eps_p_t = []
        self.runtime_alpha_t = []
        self.runtime_min_strain_c = []
        self.runtime_end_strain_c = []
        self.runtime_unload_slope_c = []
        self.runtime_min_strain_t = []
        self.runtime_end_strain_t = []
        self.runtime_unload_slope_t = []
        self.runtime_s2_eps_c = []
        self.runtime_s2_sig_c = []
        self.runtime_s2_tangent_c = []
        self.runtime_s2_epsmin_c = []
        self.runtime_s2_epsmax_c = []
        self.runtime_s2_epspl_c = []
        self.runtime_s2_epss0_c = []
        self.runtime_s2_sigs0_c = []
        self.runtime_s2_epsr_c = []
        self.runtime_s2_sigr_c = []
        self.runtime_s2_kon_c = []
        self.runtime_s2_eps_t = []
        self.runtime_s2_sig_t = []
        self.runtime_s2_tangent_t = []
        self.runtime_s2_epsmin_t = []
        self.runtime_s2_epsmax_t = []
        self.runtime_s2_epspl_t = []
        self.runtime_s2_epss0_t = []
        self.runtime_s2_sigs0_t = []
        self.runtime_s2_epsr_t = []
        self.runtime_s2_sigr_t = []
        self.runtime_s2_kon_t = []
        self.runtime_c2_eps_c = []
        self.runtime_c2_sig_c = []
        self.runtime_c2_tangent_c = []
        self.runtime_c2_eps_t = []
        self.runtime_c2_sig_t = []
        self.runtime_c2_tangent_t = []
        self.runtime_c2_ecmin_c = []
        self.runtime_c2_dept_c = []
        self.runtime_c2_ecmin_t = []
        self.runtime_c2_dept_t = []

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
        self.steel02_y_rel = existing.steel02_y_rel.copy()
        self.steel02_area = existing.steel02_area.copy()
        self.steel02_mat_defs = existing.steel02_mat_defs.copy()
        self.steel02_single_definition = existing.steel02_single_definition
        self.steel02_single_mat_def = existing.steel02_single_mat_def
        self.steel02_group_offsets = existing.steel02_group_offsets.copy()
        self.steel02_group_counts = existing.steel02_group_counts.copy()
        self.steel02_group_padded_counts = existing.steel02_group_padded_counts.copy()
        self.steel02_group_mat_defs = existing.steel02_group_mat_defs.copy()
        self.steel02_fy = existing.steel02_fy.copy()
        self.steel02_e0 = existing.steel02_e0.copy()
        self.steel02_b = existing.steel02_b.copy()
        self.steel02_r0 = existing.steel02_r0.copy()
        self.steel02_cr1 = existing.steel02_cr1.copy()
        self.steel02_cr2 = existing.steel02_cr2.copy()
        self.steel02_a1 = existing.steel02_a1.copy()
        self.steel02_a2 = existing.steel02_a2.copy()
        self.steel02_a3 = existing.steel02_a3.copy()
        self.steel02_a4 = existing.steel02_a4.copy()
        self.steel02_sigini = existing.steel02_sigini.copy()
        self.steel02_esh = existing.steel02_esh.copy()
        self.steel02_epsy = existing.steel02_epsy.copy()
        self.steel02_sigini_over_e0 = existing.steel02_sigini_over_e0.copy()
        self.steel02_pos_inv_denom = existing.steel02_pos_inv_denom.copy()
        self.steel02_neg_inv_denom = existing.steel02_neg_inv_denom.copy()
        self.concrete02_y_rel = existing.concrete02_y_rel.copy()
        self.concrete02_area = existing.concrete02_area.copy()
        self.concrete02_mat_defs = existing.concrete02_mat_defs.copy()
        self.concrete02_single_definition = existing.concrete02_single_definition
        self.concrete02_single_mat_def = existing.concrete02_single_mat_def
        self.concrete02_group_offsets = existing.concrete02_group_offsets.copy()
        self.concrete02_group_counts = existing.concrete02_group_counts.copy()
        self.concrete02_group_padded_counts = (
            existing.concrete02_group_padded_counts.copy()
        )
        self.concrete02_group_mat_defs = existing.concrete02_group_mat_defs.copy()
        self.concrete02_fc = existing.concrete02_fc.copy()
        self.concrete02_epsc0 = existing.concrete02_epsc0.copy()
        self.concrete02_fcu = existing.concrete02_fcu.copy()
        self.concrete02_epscu = existing.concrete02_epscu.copy()
        self.concrete02_rat = existing.concrete02_rat.copy()
        self.concrete02_ft = existing.concrete02_ft.copy()
        self.concrete02_ets = existing.concrete02_ets.copy()
        self.concrete02_ec0 = existing.concrete02_ec0.copy()
        self.concrete02_epsr = existing.concrete02_epsr.copy()
        self.concrete02_sigmr = existing.concrete02_sigmr.copy()
        self.steel01_nonlinear_indices = existing.steel01_nonlinear_indices.copy()
        self.concrete01_nonlinear_indices = existing.concrete01_nonlinear_indices.copy()
        self.steel02_nonlinear_indices = existing.steel02_nonlinear_indices.copy()
        self.steel02_family_position_by_nonlinear_index = (
            existing.steel02_family_position_by_nonlinear_index.copy()
        )
        self.steel02_count = existing.steel02_count
        self.steel02_padded_count = existing.steel02_padded_count
        self.steel02_instance_stride = existing.steel02_instance_stride
        self.concrete02_nonlinear_indices = existing.concrete02_nonlinear_indices.copy()
        self.concrete02_family_position_by_nonlinear_index = (
            existing.concrete02_family_position_by_nonlinear_index.copy()
        )
        self.concrete02_count = existing.concrete02_count
        self.concrete02_padded_count = existing.concrete02_padded_count
        self.concrete02_instance_stride = existing.concrete02_instance_stride
        self.other_nonlinear_indices = existing.other_nonlinear_indices.copy()
        self.runtime_state_offset_base = existing.runtime_state_offset_base
        self.runtime_state_count = existing.runtime_state_count
        self.runtime_s2_state_count = existing.runtime_s2_state_count
        self.runtime_c2_state_count = existing.runtime_c2_state_count
        self.runtime_eps_c = existing.runtime_eps_c.copy()
        self.runtime_sig_c = existing.runtime_sig_c.copy()
        self.runtime_tangent_c = existing.runtime_tangent_c.copy()
        self.runtime_eps_p_c = existing.runtime_eps_p_c.copy()
        self.runtime_alpha_c = existing.runtime_alpha_c.copy()
        self.runtime_eps_t = existing.runtime_eps_t.copy()
        self.runtime_sig_t = existing.runtime_sig_t.copy()
        self.runtime_tangent_t = existing.runtime_tangent_t.copy()
        self.runtime_eps_p_t = existing.runtime_eps_p_t.copy()
        self.runtime_alpha_t = existing.runtime_alpha_t.copy()
        self.runtime_min_strain_c = existing.runtime_min_strain_c.copy()
        self.runtime_end_strain_c = existing.runtime_end_strain_c.copy()
        self.runtime_unload_slope_c = existing.runtime_unload_slope_c.copy()
        self.runtime_min_strain_t = existing.runtime_min_strain_t.copy()
        self.runtime_end_strain_t = existing.runtime_end_strain_t.copy()
        self.runtime_unload_slope_t = existing.runtime_unload_slope_t.copy()
        self.runtime_s2_eps_c = existing.runtime_s2_eps_c.copy()
        self.runtime_s2_sig_c = existing.runtime_s2_sig_c.copy()
        self.runtime_s2_tangent_c = existing.runtime_s2_tangent_c.copy()
        self.runtime_s2_epsmin_c = existing.runtime_s2_epsmin_c.copy()
        self.runtime_s2_epsmax_c = existing.runtime_s2_epsmax_c.copy()
        self.runtime_s2_epspl_c = existing.runtime_s2_epspl_c.copy()
        self.runtime_s2_epss0_c = existing.runtime_s2_epss0_c.copy()
        self.runtime_s2_sigs0_c = existing.runtime_s2_sigs0_c.copy()
        self.runtime_s2_epsr_c = existing.runtime_s2_epsr_c.copy()
        self.runtime_s2_sigr_c = existing.runtime_s2_sigr_c.copy()
        self.runtime_s2_kon_c = existing.runtime_s2_kon_c.copy()
        self.runtime_s2_eps_t = existing.runtime_s2_eps_t.copy()
        self.runtime_s2_sig_t = existing.runtime_s2_sig_t.copy()
        self.runtime_s2_tangent_t = existing.runtime_s2_tangent_t.copy()
        self.runtime_s2_epsmin_t = existing.runtime_s2_epsmin_t.copy()
        self.runtime_s2_epsmax_t = existing.runtime_s2_epsmax_t.copy()
        self.runtime_s2_epspl_t = existing.runtime_s2_epspl_t.copy()
        self.runtime_s2_epss0_t = existing.runtime_s2_epss0_t.copy()
        self.runtime_s2_sigs0_t = existing.runtime_s2_sigs0_t.copy()
        self.runtime_s2_epsr_t = existing.runtime_s2_epsr_t.copy()
        self.runtime_s2_sigr_t = existing.runtime_s2_sigr_t.copy()
        self.runtime_s2_kon_t = existing.runtime_s2_kon_t.copy()
        self.runtime_c2_eps_c = existing.runtime_c2_eps_c.copy()
        self.runtime_c2_sig_c = existing.runtime_c2_sig_c.copy()
        self.runtime_c2_tangent_c = existing.runtime_c2_tangent_c.copy()
        self.runtime_c2_eps_t = existing.runtime_c2_eps_t.copy()
        self.runtime_c2_sig_t = existing.runtime_c2_sig_t.copy()
        self.runtime_c2_tangent_t = existing.runtime_c2_tangent_t.copy()
        self.runtime_c2_ecmin_c = existing.runtime_c2_ecmin_c.copy()
        self.runtime_c2_dept_c = existing.runtime_c2_dept_c.copy()
        self.runtime_c2_ecmin_t = existing.runtime_c2_ecmin_t.copy()
        self.runtime_c2_dept_t = existing.runtime_c2_dept_t.copy()


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
fn _fiber_section2d_runtime_local_offset(
    sec_def: FiberSection2dDef, section_state_offset: Int, section_state_count: Int
) -> Int:
    if section_state_count != sec_def.fiber_count:
        abort("FiberSection2d section state count mismatch")
    var local_offset = section_state_offset - sec_def.runtime_state_offset_base
    if local_offset < 0 or local_offset + section_state_count > sec_def.runtime_state_count:
        abort("FiberSection2d section states out of range")
    return local_offset


fn _fiber_section2d_runtime_ensure_capacity(mut sec_def: FiberSection2dDef, new_count: Int):
    sec_def.runtime_eps_c.resize(new_count, 0.0)
    sec_def.runtime_sig_c.resize(new_count, 0.0)
    sec_def.runtime_tangent_c.resize(new_count, 0.0)
    sec_def.runtime_eps_p_c.resize(new_count, 0.0)
    sec_def.runtime_alpha_c.resize(new_count, 0.0)
    sec_def.runtime_eps_t.resize(new_count, 0.0)
    sec_def.runtime_sig_t.resize(new_count, 0.0)
    sec_def.runtime_tangent_t.resize(new_count, 0.0)
    sec_def.runtime_eps_p_t.resize(new_count, 0.0)
    sec_def.runtime_alpha_t.resize(new_count, 0.0)
    sec_def.runtime_min_strain_c.resize(new_count, 0.0)
    sec_def.runtime_end_strain_c.resize(new_count, 0.0)
    sec_def.runtime_unload_slope_c.resize(new_count, 0.0)
    sec_def.runtime_min_strain_t.resize(new_count, 0.0)
    sec_def.runtime_end_strain_t.resize(new_count, 0.0)
    sec_def.runtime_unload_slope_t.resize(new_count, 0.0)


fn _fiber_section2d_runtime_ensure_steel02_capacity(
    mut sec_def: FiberSection2dDef, new_count: Int
):
    sec_def.runtime_s2_eps_c.resize(new_count, 0.0)
    sec_def.runtime_s2_sig_c.resize(new_count, 0.0)
    sec_def.runtime_s2_tangent_c.resize(new_count, 0.0)
    sec_def.runtime_s2_epsmin_c.resize(new_count, 0.0)
    sec_def.runtime_s2_epsmax_c.resize(new_count, 0.0)
    sec_def.runtime_s2_epspl_c.resize(new_count, 0.0)
    sec_def.runtime_s2_epss0_c.resize(new_count, 0.0)
    sec_def.runtime_s2_sigs0_c.resize(new_count, 0.0)
    sec_def.runtime_s2_epsr_c.resize(new_count, 0.0)
    sec_def.runtime_s2_sigr_c.resize(new_count, 0.0)
    sec_def.runtime_s2_kon_c.resize(new_count, 0)
    sec_def.runtime_s2_eps_t.resize(new_count, 0.0)
    sec_def.runtime_s2_sig_t.resize(new_count, 0.0)
    sec_def.runtime_s2_tangent_t.resize(new_count, 0.0)
    sec_def.runtime_s2_epsmin_t.resize(new_count, 0.0)
    sec_def.runtime_s2_epsmax_t.resize(new_count, 0.0)
    sec_def.runtime_s2_epspl_t.resize(new_count, 0.0)
    sec_def.runtime_s2_epss0_t.resize(new_count, 0.0)
    sec_def.runtime_s2_sigs0_t.resize(new_count, 0.0)
    sec_def.runtime_s2_epsr_t.resize(new_count, 0.0)
    sec_def.runtime_s2_sigr_t.resize(new_count, 0.0)
    sec_def.runtime_s2_kon_t.resize(new_count, 0)


fn _fiber_section2d_runtime_ensure_concrete02_capacity(
    mut sec_def: FiberSection2dDef, new_count: Int
):
    sec_def.runtime_c2_eps_c.resize(new_count, 0.0)
    sec_def.runtime_c2_sig_c.resize(new_count, 0.0)
    sec_def.runtime_c2_tangent_c.resize(new_count, 0.0)
    sec_def.runtime_c2_eps_t.resize(new_count, 0.0)
    sec_def.runtime_c2_sig_t.resize(new_count, 0.0)
    sec_def.runtime_c2_tangent_t.resize(new_count, 0.0)
    sec_def.runtime_c2_ecmin_c.resize(new_count, 0.0)
    sec_def.runtime_c2_dept_c.resize(new_count, 0.0)
    sec_def.runtime_c2_ecmin_t.resize(new_count, 0.0)
    sec_def.runtime_c2_dept_t.resize(new_count, 0.0)


fn _fiber_section2d_runtime_init_slot(
    mut sec_def: FiberSection2dDef, slot: Int, mat_def: UniMaterialDef
):
    var state = UniMaterialState(mat_def)
    sec_def.runtime_eps_c[slot] = state.eps_c
    sec_def.runtime_sig_c[slot] = state.sig_c
    sec_def.runtime_tangent_c[slot] = state.tangent_c
    sec_def.runtime_eps_p_c[slot] = state.eps_p_c
    sec_def.runtime_alpha_c[slot] = state.alpha_c
    sec_def.runtime_eps_t[slot] = state.eps_t
    sec_def.runtime_sig_t[slot] = state.sig_t
    sec_def.runtime_tangent_t[slot] = state.tangent_t
    sec_def.runtime_eps_p_t[slot] = state.eps_p_t
    sec_def.runtime_alpha_t[slot] = state.alpha_t
    sec_def.runtime_min_strain_c[slot] = state.min_strain_c
    sec_def.runtime_end_strain_c[slot] = state.end_strain_c
    sec_def.runtime_unload_slope_c[slot] = state.unload_slope_c
    sec_def.runtime_min_strain_t[slot] = state.min_strain_t
    sec_def.runtime_end_strain_t[slot] = state.end_strain_t
    sec_def.runtime_unload_slope_t[slot] = state.unload_slope_t


fn _fiber_section2d_runtime_init_steel02_slot(
    mut sec_def: FiberSection2dDef, slot: Int, mat_def: UniMaterialDef
):
    var state = UniMaterialState(mat_def)
    sec_def.runtime_s2_eps_c[slot] = state.eps_c
    sec_def.runtime_s2_sig_c[slot] = state.sig_c
    sec_def.runtime_s2_tangent_c[slot] = state.tangent_c
    sec_def.runtime_s2_epsmin_c[slot] = state.s2_epsmin_c
    sec_def.runtime_s2_epsmax_c[slot] = state.s2_epsmax_c
    sec_def.runtime_s2_epspl_c[slot] = state.s2_epspl_c
    sec_def.runtime_s2_epss0_c[slot] = state.s2_epss0_c
    sec_def.runtime_s2_sigs0_c[slot] = state.s2_sigs0_c
    sec_def.runtime_s2_epsr_c[slot] = state.s2_epsr_c
    sec_def.runtime_s2_sigr_c[slot] = state.s2_sigr_c
    sec_def.runtime_s2_kon_c[slot] = state.s2_kon_c
    sec_def.runtime_s2_eps_t[slot] = state.eps_t
    sec_def.runtime_s2_sig_t[slot] = state.sig_t
    sec_def.runtime_s2_tangent_t[slot] = state.tangent_t
    sec_def.runtime_s2_epsmin_t[slot] = state.s2_epsmin_t
    sec_def.runtime_s2_epsmax_t[slot] = state.s2_epsmax_t
    sec_def.runtime_s2_epspl_t[slot] = state.s2_epspl_t
    sec_def.runtime_s2_epss0_t[slot] = state.s2_epss0_t
    sec_def.runtime_s2_sigs0_t[slot] = state.s2_sigs0_t
    sec_def.runtime_s2_epsr_t[slot] = state.s2_epsr_t
    sec_def.runtime_s2_sigr_t[slot] = state.s2_sigr_t
    sec_def.runtime_s2_kon_t[slot] = state.s2_kon_t


fn _fiber_section2d_runtime_init_concrete02_slot(
    mut sec_def: FiberSection2dDef, slot: Int, mat_def: UniMaterialDef
):
    var state = UniMaterialState(mat_def)
    sec_def.runtime_c2_eps_c[slot] = state.eps_c
    sec_def.runtime_c2_sig_c[slot] = state.sig_c
    sec_def.runtime_c2_tangent_c[slot] = state.tangent_c
    sec_def.runtime_c2_eps_t[slot] = state.eps_t
    sec_def.runtime_c2_sig_t[slot] = state.sig_t
    sec_def.runtime_c2_tangent_t[slot] = state.tangent_t
    sec_def.runtime_c2_ecmin_c[slot] = state.c2_ecmin_c
    sec_def.runtime_c2_dept_c[slot] = state.c2_dept_c
    sec_def.runtime_c2_ecmin_t[slot] = state.c2_ecmin_t
    sec_def.runtime_c2_dept_t[slot] = state.c2_dept_t


fn fiber_section2d_runtime_alloc_instances(
    mut sec_def: FiberSection2dDef, instance_count: Int
) -> Int:
    if instance_count <= 0:
        return sec_def.runtime_state_offset_base + sec_def.runtime_state_count
    if sec_def.runtime_state_count == 0:
        sec_def.runtime_state_offset_base = 0
    var old_count = sec_def.runtime_state_count
    var new_count = old_count + instance_count * sec_def.fiber_count
    var old_s2_count = sec_def.runtime_s2_state_count
    var new_s2_count = old_s2_count + instance_count * sec_def.steel02_instance_stride
    var old_c2_count = sec_def.runtime_c2_state_count
    var new_c2_count = old_c2_count + instance_count * sec_def.concrete02_instance_stride
    _fiber_section2d_runtime_ensure_capacity(sec_def, new_count)
    _fiber_section2d_runtime_ensure_steel02_capacity(sec_def, new_s2_count)
    _fiber_section2d_runtime_ensure_concrete02_capacity(sec_def, new_c2_count)
    for inst in range(instance_count):
        var instance_offset = old_count + inst * sec_def.fiber_count
        var steel02_instance_offset = (
            old_s2_count + inst * sec_def.steel02_instance_stride
        )
        var concrete02_instance_offset = (
            old_c2_count + inst * sec_def.concrete02_instance_stride
        )
        for i in range(sec_def.nonlinear_count):
            var mat_def = sec_def.nonlinear_mat_defs[i]
            if mat_def.mat_type == UniMaterialTypeTag.Steel02:
                var steel02_pos = sec_def.steel02_family_position_by_nonlinear_index[i]
                _fiber_section2d_runtime_init_steel02_slot(
                    sec_def, steel02_instance_offset + steel02_pos, mat_def
                )
            elif mat_def.mat_type == UniMaterialTypeTag.Concrete02:
                var concrete02_pos = sec_def.concrete02_family_position_by_nonlinear_index[
                    i
                ]
                _fiber_section2d_runtime_init_concrete02_slot(
                    sec_def, concrete02_instance_offset + concrete02_pos, mat_def
                )
            else:
                _fiber_section2d_runtime_init_slot(
                    sec_def, instance_offset + sec_def.elastic_count + i, mat_def
                )
    sec_def.runtime_state_count = new_count
    sec_def.runtime_s2_state_count = new_s2_count
    sec_def.runtime_c2_state_count = new_c2_count
    return sec_def.runtime_state_offset_base + old_count


@always_inline
fn _fiber_section2d_runtime_apply_elastic(
    mat_def: UniMaterialDef, mut sec_def: FiberSection2dDef, slot: Int, eps: Float64
):
    sec_def.runtime_eps_t[slot] = eps
    sec_def.runtime_sig_t[slot] = mat_def.p0 * eps
    sec_def.runtime_tangent_t[slot] = mat_def.p0
    sec_def.runtime_eps_p_t[slot] = sec_def.runtime_eps_p_c[slot]
    sec_def.runtime_alpha_t[slot] = sec_def.runtime_alpha_c[slot]


@always_inline
fn _fiber_section2d_runtime_apply_steel01(
    mat_def: UniMaterialDef, mut sec_def: FiberSection2dDef, slot: Int, eps: Float64
):
    var Fy = mat_def.p0
    var E0 = mat_def.p1
    var b = mat_def.p2
    var H = (b * E0) / (1.0 - b)
    var eps_p = sec_def.runtime_eps_p_c[slot]
    var alpha = sec_def.runtime_alpha_c[slot]
    var sigma_trial = E0 * (eps - eps_p)
    var xi = sigma_trial - alpha
    var f = _abs(xi) - Fy
    sec_def.runtime_eps_t[slot] = eps
    if f <= 0.0:
        sec_def.runtime_sig_t[slot] = sigma_trial
        sec_def.runtime_tangent_t[slot] = E0
        sec_def.runtime_eps_p_t[slot] = eps_p
        sec_def.runtime_alpha_t[slot] = alpha
        return
    var dg = f / (E0 + H)
    var sgn = _sign(xi)
    sec_def.runtime_eps_p_t[slot] = eps_p + dg * sgn
    sec_def.runtime_alpha_t[slot] = alpha + H * dg * sgn
    sec_def.runtime_sig_t[slot] = sigma_trial - E0 * dg * sgn
    sec_def.runtime_tangent_t[slot] = (E0 * H) / (E0 + H)


@always_inline
fn _fiber_section2d_runtime_apply_concrete01(
    mat_def: UniMaterialDef, mut sec_def: FiberSection2dDef, slot: Int, eps: Float64
):
    var fpc = mat_def.p0
    var epsc0 = mat_def.p1
    var fpcu = mat_def.p2
    var epscu = mat_def.p3
    var d_strain = eps - sec_def.runtime_eps_c[slot]
    sec_def.runtime_eps_t[slot] = eps
    if _abs(d_strain) < 1.0e-14:
        sec_def.runtime_sig_t[slot] = sec_def.runtime_sig_c[slot]
        sec_def.runtime_tangent_t[slot] = sec_def.runtime_tangent_c[slot]
        sec_def.runtime_min_strain_t[slot] = sec_def.runtime_min_strain_c[slot]
        sec_def.runtime_end_strain_t[slot] = sec_def.runtime_end_strain_c[slot]
        sec_def.runtime_unload_slope_t[slot] = sec_def.runtime_unload_slope_c[slot]
        return
    var min_strain = sec_def.runtime_min_strain_c[slot]
    var end_strain = sec_def.runtime_end_strain_c[slot]
    var unload_slope = sec_def.runtime_unload_slope_c[slot]
    sec_def.runtime_min_strain_t[slot] = min_strain
    sec_def.runtime_end_strain_t[slot] = end_strain
    sec_def.runtime_unload_slope_t[slot] = unload_slope
    if eps > 0.0:
        sec_def.runtime_sig_t[slot] = 0.0
        sec_def.runtime_tangent_t[slot] = 0.0
        return
    var temp_stress = (
        sec_def.runtime_sig_c[slot]
        + sec_def.runtime_unload_slope_t[slot] * eps
        - sec_def.runtime_unload_slope_t[slot] * sec_def.runtime_eps_c[slot]
    )
    if eps <= sec_def.runtime_eps_c[slot]:
        var reload = _concrete01_reload(
            fpc,
            epsc0,
            fpcu,
            epscu,
            eps,
            sec_def.runtime_min_strain_t[slot],
            sec_def.runtime_end_strain_t[slot],
            sec_def.runtime_unload_slope_t[slot],
        )
        sec_def.runtime_sig_t[slot] = reload.stress
        sec_def.runtime_tangent_t[slot] = reload.tangent
        if temp_stress > sec_def.runtime_sig_t[slot]:
            sec_def.runtime_sig_t[slot] = temp_stress
            sec_def.runtime_tangent_t[slot] = sec_def.runtime_unload_slope_t[slot]
    elif temp_stress <= 0.0:
        sec_def.runtime_sig_t[slot] = temp_stress
        sec_def.runtime_tangent_t[slot] = sec_def.runtime_unload_slope_t[slot]
    else:
        sec_def.runtime_sig_t[slot] = 0.0
        sec_def.runtime_tangent_t[slot] = 0.0


@always_inline
fn _fiber_section2d_runtime_apply_steel02(
    mat_def: UniMaterialDef, mut sec_def: FiberSection2dDef, slot: Int, trial_eps: Float64
):
    var Fy = mat_def.p0
    var E0 = mat_def.p1
    var b = mat_def.p2
    var R0 = mat_def.p3
    var cR1 = mat_def.p4
    var cR2 = mat_def.p5
    var a1 = mat_def.p6
    var a2 = mat_def.p7
    var a3 = mat_def.p8
    var a4 = mat_def.p9
    var sigini = mat_def.p10

    var Esh = b * E0
    var epsy = Fy / E0
    var eps = trial_eps
    if sigini != 0.0:
        eps += sigini / E0
    sec_def.runtime_s2_eps_t[slot] = eps
    var deps = eps - sec_def.runtime_s2_eps_c[slot]
    var epsmax = sec_def.runtime_s2_epsmax_c[slot]
    var epsmin = sec_def.runtime_s2_epsmin_c[slot]
    var epspl = sec_def.runtime_s2_epspl_c[slot]
    var epss0 = sec_def.runtime_s2_epss0_c[slot]
    var sigs0 = sec_def.runtime_s2_sigs0_c[slot]
    var epsr = sec_def.runtime_s2_epsr_c[slot]
    var sigr = sec_def.runtime_s2_sigr_c[slot]
    var kon = sec_def.runtime_s2_kon_c[slot]
    if kon == 0 or kon == 3:
        if _abs(deps) < 2.220446049250313e-15:
            sec_def.runtime_s2_tangent_t[slot] = E0
            sec_def.runtime_s2_sig_t[slot] = sigini
            sec_def.runtime_s2_kon_t[slot] = 3
            sec_def.runtime_s2_epsmin_t[slot] = epsmin
            sec_def.runtime_s2_epsmax_t[slot] = epsmax
            sec_def.runtime_s2_epspl_t[slot] = epspl
            sec_def.runtime_s2_epss0_t[slot] = epss0
            sec_def.runtime_s2_sigs0_t[slot] = sigs0
            sec_def.runtime_s2_epsr_t[slot] = epsr
            sec_def.runtime_s2_sigr_t[slot] = sigr
            return
        epsmax = epsy
        epsmin = -epsy
        if deps < 0.0:
            kon = 2
            epss0 = epsmin
            sigs0 = -Fy
            epspl = epsmin
        else:
            kon = 1
            epss0 = epsmax
            sigs0 = Fy
            epspl = epsmax
    if kon == 2 and deps > 0.0:
        kon = 1
        epsr = sec_def.runtime_s2_eps_c[slot]
        sigr = sec_def.runtime_s2_sig_c[slot]
        if sec_def.runtime_s2_eps_c[slot] < epsmin:
            epsmin = sec_def.runtime_s2_eps_c[slot]
        var d1 = (epsmax - epsmin) / (2.0 * (a4 * epsy))
        var shft = 1.0 + a3 * (d1 ** 0.8)
        epss0 = (Fy * shft - Esh * epsy * shft - sigr + E0 * epsr) / (E0 - Esh)
        sigs0 = Fy * shft + Esh * (epss0 - epsy * shft)
        epspl = epsmax
    elif kon == 1 and deps < 0.0:
        kon = 2
        epsr = sec_def.runtime_s2_eps_c[slot]
        sigr = sec_def.runtime_s2_sig_c[slot]
        if sec_def.runtime_s2_eps_c[slot] > epsmax:
            epsmax = sec_def.runtime_s2_eps_c[slot]
        var d1 = (epsmax - epsmin) / (2.0 * (a2 * epsy))
        var shft = 1.0 + a1 * (d1 ** 0.8)
        epss0 = (-Fy * shft + Esh * epsy * shft - sigr + E0 * epsr) / (E0 - Esh)
        sigs0 = -Fy * shft + Esh * (epss0 + epsy * shft)
        epspl = epsmin
    var xi = _abs((epspl - epss0) / epsy)
    var R = R0 * (1.0 - (cR1 * xi) / (cR2 + xi))
    var epsrat = (eps - epsr) / (epss0 - epsr)
    var dum1 = 1.0 + _pow_abs(epsrat, R)
    var dum2 = dum1 ** (1.0 / R)
    var sig = b * epsrat + (1.0 - b) * epsrat / dum2
    sig = sig * (sigs0 - sigr) + sigr
    var e = b + (1.0 - b) / (dum1 * dum2)
    e = e * (sigs0 - sigr) / (epss0 - epsr)
    sec_def.runtime_s2_sig_t[slot] = sig
    sec_def.runtime_s2_tangent_t[slot] = e
    sec_def.runtime_s2_epsmin_t[slot] = epsmin
    sec_def.runtime_s2_epsmax_t[slot] = epsmax
    sec_def.runtime_s2_epspl_t[slot] = epspl
    sec_def.runtime_s2_epss0_t[slot] = epss0
    sec_def.runtime_s2_sigs0_t[slot] = sigs0
    sec_def.runtime_s2_epsr_t[slot] = epsr
    sec_def.runtime_s2_sigr_t[slot] = sigr
    sec_def.runtime_s2_kon_t[slot] = kon


@always_inline
fn _fiber_section2d_runtime_apply_concrete02(
    mat_def: UniMaterialDef, mut sec_def: FiberSection2dDef, slot: Int, strain: Float64
):
    var fc = mat_def.p0
    var epsc0 = mat_def.p1
    var fcu = mat_def.p2
    var epscu = mat_def.p3
    var rat = mat_def.p4
    var ft = mat_def.p5
    var Ets = mat_def.p6
    var Ec0 = (2.0 * fc) / epsc0

    sec_def.runtime_c2_ecmin_t[slot] = sec_def.runtime_c2_ecmin_c[slot]
    sec_def.runtime_c2_dept_t[slot] = sec_def.runtime_c2_dept_c[slot]
    sec_def.runtime_c2_eps_t[slot] = strain
    var deps = strain - sec_def.runtime_c2_eps_c[slot]
    if _abs(deps) < 2.220446049250313e-16:
        sec_def.runtime_c2_sig_t[slot] = sec_def.runtime_c2_sig_c[slot]
        sec_def.runtime_c2_tangent_t[slot] = sec_def.runtime_c2_tangent_c[slot]
        return
    if strain < sec_def.runtime_c2_ecmin_t[slot]:
        var env = _concrete02_compr_envlp(fc, epsc0, fcu, epscu, strain)
        sec_def.runtime_c2_sig_t[slot] = env.stress
        sec_def.runtime_c2_tangent_t[slot] = env.tangent
        sec_def.runtime_c2_ecmin_t[slot] = strain
        return
    var epsr = (fcu - rat * Ec0 * epscu) / (Ec0 * (1.0 - rat))
    var sigmr = Ec0 * epsr
    var sigmm_env = _concrete02_compr_envlp(
        fc, epsc0, fcu, epscu, sec_def.runtime_c2_ecmin_t[slot]
    )
    var sigmm = sigmm_env.stress
    var er = (sigmm - sigmr) / (sec_def.runtime_c2_ecmin_t[slot] - epsr)
    var ept = sec_def.runtime_c2_ecmin_t[slot] - sigmm / er
    if strain <= ept:
        var sigmin = sigmm + er * (strain - sec_def.runtime_c2_ecmin_t[slot])
        var sigmax = 0.5 * er * (strain - ept)
        sec_def.runtime_c2_sig_t[slot] = sec_def.runtime_c2_sig_c[slot] + Ec0 * deps
        sec_def.runtime_c2_tangent_t[slot] = Ec0
        if sec_def.runtime_c2_sig_t[slot] <= sigmin:
            sec_def.runtime_c2_sig_t[slot] = sigmin
            sec_def.runtime_c2_tangent_t[slot] = er
        if sec_def.runtime_c2_sig_t[slot] >= sigmax:
            sec_def.runtime_c2_sig_t[slot] = sigmax
            sec_def.runtime_c2_tangent_t[slot] = 0.5 * er
        return
    var epn = ept + sec_def.runtime_c2_dept_t[slot]
    if strain <= epn:
        var sicn_env = _concrete02_tens_envlp(
            fc, epsc0, ft, Ets, sec_def.runtime_c2_dept_t[slot]
        )
        var e = Ec0
        if sec_def.runtime_c2_dept_t[slot] != 0.0:
            e = sicn_env.stress / sec_def.runtime_c2_dept_t[slot]
        sec_def.runtime_c2_tangent_t[slot] = e
        sec_def.runtime_c2_sig_t[slot] = e * (strain - ept)
    else:
        var env = _concrete02_tens_envlp(fc, epsc0, ft, Ets, strain - ept)
        sec_def.runtime_c2_sig_t[slot] = env.stress
        sec_def.runtime_c2_tangent_t[slot] = env.tangent
        sec_def.runtime_c2_dept_t[slot] = strain - ept


@always_inline
fn _fiber_section2d_runtime_steel02_slot(
    sec_def: FiberSection2dDef, local_offset: Int, nonlinear_index: Int
) -> Int:
    var steel02_pos = sec_def.steel02_family_position_by_nonlinear_index[
        nonlinear_index
    ]
    if steel02_pos < 0:
        abort("FiberSection2d Steel02 family slot missing")
    return (
        local_offset // sec_def.fiber_count
    ) * sec_def.steel02_instance_stride + steel02_pos


@always_inline
fn _fiber_section2d_runtime_concrete02_slot(
    sec_def: FiberSection2dDef, local_offset: Int, nonlinear_index: Int
) -> Int:
    var concrete02_pos = sec_def.concrete02_family_position_by_nonlinear_index[
        nonlinear_index
    ]
    if concrete02_pos < 0:
        abort("FiberSection2d Concrete02 family slot missing")
    return (
        local_offset // sec_def.fiber_count
    ) * sec_def.concrete02_instance_stride + concrete02_pos


@always_inline
fn _fiber_section2d_runtime_set_trial(
    mat_def: UniMaterialDef, mut sec_def: FiberSection2dDef, slot: Int, eps: Float64
):
    if mat_def.mat_type == UniMaterialTypeTag.Elastic:
        _fiber_section2d_runtime_apply_elastic(mat_def, sec_def, slot, eps)
        return
    if mat_def.mat_type == UniMaterialTypeTag.Steel01:
        _fiber_section2d_runtime_apply_steel01(mat_def, sec_def, slot, eps)
        return
    if mat_def.mat_type == UniMaterialTypeTag.Concrete01:
        _fiber_section2d_runtime_apply_concrete01(mat_def, sec_def, slot, eps)
        return
    if mat_def.mat_type == UniMaterialTypeTag.Steel02:
        abort("Steel02 requires dedicated family runtime slot")
    if mat_def.mat_type == UniMaterialTypeTag.Concrete02:
        abort("Concrete02 requires dedicated family runtime slot")
    abort("unsupported uniaxial material")


fn _fiber_section2d_runtime_commit_slot(mut sec_def: FiberSection2dDef, slot: Int):
    sec_def.runtime_eps_c[slot] = sec_def.runtime_eps_t[slot]
    sec_def.runtime_sig_c[slot] = sec_def.runtime_sig_t[slot]
    sec_def.runtime_tangent_c[slot] = sec_def.runtime_tangent_t[slot]
    sec_def.runtime_eps_p_c[slot] = sec_def.runtime_eps_p_t[slot]
    sec_def.runtime_alpha_c[slot] = sec_def.runtime_alpha_t[slot]
    sec_def.runtime_min_strain_c[slot] = sec_def.runtime_min_strain_t[slot]
    sec_def.runtime_end_strain_c[slot] = sec_def.runtime_end_strain_t[slot]
    sec_def.runtime_unload_slope_c[slot] = sec_def.runtime_unload_slope_t[slot]


fn _fiber_section2d_runtime_revert_slot(mut sec_def: FiberSection2dDef, slot: Int):
    sec_def.runtime_eps_t[slot] = sec_def.runtime_eps_c[slot]
    sec_def.runtime_sig_t[slot] = sec_def.runtime_sig_c[slot]
    sec_def.runtime_tangent_t[slot] = sec_def.runtime_tangent_c[slot]
    sec_def.runtime_eps_p_t[slot] = sec_def.runtime_eps_p_c[slot]
    sec_def.runtime_alpha_t[slot] = sec_def.runtime_alpha_c[slot]
    sec_def.runtime_min_strain_t[slot] = sec_def.runtime_min_strain_c[slot]
    sec_def.runtime_end_strain_t[slot] = sec_def.runtime_end_strain_c[slot]
    sec_def.runtime_unload_slope_t[slot] = sec_def.runtime_unload_slope_c[slot]


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


@always_inline
fn _fiber_section2d_apply_steel01_family_from_offset(
    mut sec_def: FiberSection2dDef,
    local_offset: Int,
    eps0: Float64,
    kappa: Float64,
    mut axial_force: Float64,
    mut moment_z: Float64,
    mut k11: Float64,
    mut k12: Float64,
    mut k22: Float64,
):
    if len(sec_def.steel01_nonlinear_indices) <= 0:
        return
    var nonlinear_state_offset = local_offset + sec_def.elastic_count
    for j in range(len(sec_def.steel01_nonlinear_indices)):
        var i = sec_def.steel01_nonlinear_indices[j]
        var y_rel = sec_def.nonlinear_y_rel[i]
        var area = sec_def.nonlinear_area[i]
        var state_index = nonlinear_state_offset + i
        var mat_def = sec_def.nonlinear_mat_defs[i]
        _fiber_section2d_runtime_apply_steel01(
            mat_def, sec_def, state_index, eps0 - y_rel * kappa
        )
        var fs = sec_def.runtime_sig_t[state_index] * area
        var ks = sec_def.runtime_tangent_t[state_index] * area
        axial_force += fs
        moment_z += -fs * y_rel
        k11 += ks
        k12 += -ks * y_rel
        k22 += ks * y_rel * y_rel


@always_inline
fn _fiber_section2d_apply_concrete01_family_from_offset(
    mut sec_def: FiberSection2dDef,
    local_offset: Int,
    eps0: Float64,
    kappa: Float64,
    mut axial_force: Float64,
    mut moment_z: Float64,
    mut k11: Float64,
    mut k12: Float64,
    mut k22: Float64,
):
    if len(sec_def.concrete01_nonlinear_indices) <= 0:
        return
    var nonlinear_state_offset = local_offset + sec_def.elastic_count
    for j in range(len(sec_def.concrete01_nonlinear_indices)):
        var i = sec_def.concrete01_nonlinear_indices[j]
        var y_rel = sec_def.nonlinear_y_rel[i]
        var area = sec_def.nonlinear_area[i]
        var state_index = nonlinear_state_offset + i
        var mat_def = sec_def.nonlinear_mat_defs[i]
        _fiber_section2d_runtime_apply_concrete01(
            mat_def, sec_def, state_index, eps0 - y_rel * kappa
        )
        var fs = sec_def.runtime_sig_t[state_index] * area
        var ks = sec_def.runtime_tangent_t[state_index] * area
        axial_force += fs
        moment_z += -fs * y_rel
        k11 += ks
        k12 += -ks * y_rel
        k22 += ks * y_rel * y_rel


@always_inline
fn _fiber_section2d_apply_other_nonlinear_fallback_from_offset(
    mut sec_def: FiberSection2dDef,
    local_offset: Int,
    eps0: Float64,
    kappa: Float64,
    mut axial_force: Float64,
    mut moment_z: Float64,
    mut k11: Float64,
    mut k12: Float64,
    mut k22: Float64,
):
    if len(sec_def.other_nonlinear_indices) <= 0:
        return
    var nonlinear_state_offset = local_offset + sec_def.elastic_count
    for j in range(len(sec_def.other_nonlinear_indices)):
        var i = sec_def.other_nonlinear_indices[j]
        var y_rel = sec_def.nonlinear_y_rel[i]
        var area = sec_def.nonlinear_area[i]
        var state_index = nonlinear_state_offset + i
        var mat_def = sec_def.nonlinear_mat_defs[i]
        _fiber_section2d_runtime_set_trial(
            mat_def, sec_def, state_index, eps0 - y_rel * kappa
        )
        var fs = sec_def.runtime_sig_t[state_index] * area
        var ks = sec_def.runtime_tangent_t[state_index] * area
        axial_force += fs
        moment_z += -fs * y_rel
        k11 += ks
        k12 += -ks * y_rel
        k22 += ks * y_rel * y_rel


@always_inline
fn _fiber_section2d_copy_float64_range_simd[width: Int](
    mut dst: List[Float64], dst_start: Int, src: List[Float64], src_start: Int, count: Int
):
    if count <= 0:
        return
    if dst_start == 0 and src_start == 0:
        copy_float64_contiguous_simd[width](dst, src, count)
        return

    @parameter
    fn copy_chunk[chunk: Int](i: Int):
        var value_vec = load_float64_contiguous_simd[chunk](src, src_start + i)
        store_float64_contiguous_simd[chunk](dst, dst_start + i, value_vec)

    vectorize[copy_chunk, width](count)


@always_inline
fn _fiber_section2d_copy_int_range(
    mut dst: List[Int], dst_start: Int, src: List[Int], src_start: Int, count: Int
):
    for i in range(count):
        dst[dst_start + i] = src[src_start + i]


@always_inline
fn _simd_abs_float64[width: Int](
    x: SIMD[DType.float64, width]
) -> SIMD[DType.float64, width]:
    return x.lt(0.0).select(-x, x)


@always_inline
fn _concrete02_compr_envlp_simd[width: Int](
    fc: SIMD[DType.float64, width],
    epsc0: SIMD[DType.float64, width],
    fcu: SIMD[DType.float64, width],
    epscu: SIMD[DType.float64, width],
    strain: SIMD[DType.float64, width],
) -> (SIMD[DType.float64, width], SIMD[DType.float64, width]):
    var ec0 = (SIMD[DType.float64, width](2.0) * fc) / epsc0
    var rat = strain / epsc0
    var sig = fc * rat * (SIMD[DType.float64, width](2.0) - rat)
    var tangent = ec0 * (SIMD[DType.float64, width](1.0) - rat)

    var softening_sig = (fcu - fc) * (strain - epsc0) / (epscu - epsc0) + fc
    var softening_tangent = (fcu - fc) / (epscu - epsc0)
    var peak_mask = strain.ge(epsc0)
    sig = peak_mask.select(sig, softening_sig)
    tangent = peak_mask.select(tangent, softening_tangent)

    var crush_mask = strain.le(epscu)
    sig = crush_mask.select(fcu, sig)
    tangent = crush_mask.select(SIMD[DType.float64, width](1.0e-10), tangent)
    return (sig, tangent)


@always_inline
fn _concrete02_tens_envlp_simd[width: Int](
    fc: SIMD[DType.float64, width],
    epsc0: SIMD[DType.float64, width],
    ft: SIMD[DType.float64, width],
    ets: SIMD[DType.float64, width],
    strain: SIMD[DType.float64, width],
) -> (SIMD[DType.float64, width], SIMD[DType.float64, width]):
    var ec0 = (SIMD[DType.float64, width](2.0) * fc) / epsc0
    var eps0 = ft / ec0
    var epsu = ft * (
        SIMD[DType.float64, width](1.0) / ets + SIMD[DType.float64, width](1.0) / ec0
    )
    var sig = strain * ec0
    var tangent = ec0

    var softening_mask = strain.gt(eps0)
    var softening_sig = ft - ets * (strain - eps0)
    sig = softening_mask.select(softening_sig, sig)
    tangent = softening_mask.select(-ets, tangent)

    var zero_mask = strain.gt(epsu)
    sig = zero_mask.select(SIMD[DType.float64, width](0.0), sig)
    tangent = zero_mask.select(SIMD[DType.float64, width](1.0e-10), tangent)
    return (sig, tangent)


@always_inline
fn _fiber_section2d_runtime_apply_concrete02_range_simd_mixed[width: Int](
    mut sec_def: FiberSection2dDef,
    slot_start: Int,
    count: Int,
    eps0: Float64,
    kappa: Float64,
    mut axial_force: Float64,
    mut moment_z: Float64,
    mut k11: Float64,
    mut k12: Float64,
    mut k22: Float64,
):
    if count <= 0:
        return

    var i = 0
    while i < count:
        var y_vec = load_float64_contiguous_simd[width](sec_def.concrete02_y_rel, i)
        var area_vec = load_float64_contiguous_simd[width](sec_def.concrete02_area, i)
        var strain = SIMD[DType.float64, width](eps0) - y_vec * SIMD[DType.float64, width](kappa)
        var slot = slot_start + i

        var eps_c = load_float64_contiguous_simd[width](sec_def.runtime_c2_eps_c, slot)
        var sig_c = load_float64_contiguous_simd[width](sec_def.runtime_c2_sig_c, slot)
        var tangent_c = load_float64_contiguous_simd[width](
            sec_def.runtime_c2_tangent_c, slot
        )
        var ecmin_c = load_float64_contiguous_simd[width](sec_def.runtime_c2_ecmin_c, slot)
        var dept_c = load_float64_contiguous_simd[width](sec_def.runtime_c2_dept_c, slot)

        var fc = load_float64_contiguous_simd[width](sec_def.concrete02_fc, i)
        var epsc0 = load_float64_contiguous_simd[width](sec_def.concrete02_epsc0, i)
        var fcu = load_float64_contiguous_simd[width](sec_def.concrete02_fcu, i)
        var epscu = load_float64_contiguous_simd[width](sec_def.concrete02_epscu, i)
        var ft = load_float64_contiguous_simd[width](sec_def.concrete02_ft, i)
        var ets = load_float64_contiguous_simd[width](sec_def.concrete02_ets, i)
        var ec0 = load_float64_contiguous_simd[width](sec_def.concrete02_ec0, i)
        var epsr = load_float64_contiguous_simd[width](sec_def.concrete02_epsr, i)
        var sigmr = load_float64_contiguous_simd[width](sec_def.concrete02_sigmr, i)
        var deps = strain - eps_c
        var tol = SIMD[DType.float64, width](2.220446049250313e-16)

        var eps_t = strain
        var sig_t = sig_c
        var tangent_t = tangent_c
        var ecmin_t = ecmin_c
        var dept_t = dept_c

        var no_change_mask = _simd_abs_float64[width](deps).lt(tol)
        var compression_mask = strain.lt(ecmin_t) & ~no_change_mask
        var compr_env = _concrete02_compr_envlp_simd[width](fc, epsc0, fcu, epscu, strain)
        sig_t = compression_mask.select(compr_env[0], sig_t)
        tangent_t = compression_mask.select(compr_env[1], tangent_t)
        ecmin_t = compression_mask.select(strain, ecmin_t)

        var remaining_mask = ~(no_change_mask | compression_mask)
        var half = SIMD[DType.float64, width](0.5)
        var sigmm_env = _concrete02_compr_envlp_simd[width](fc, epsc0, fcu, epscu, ecmin_t)
        var sigmm = sigmm_env[0]
        var er = (sigmm - sigmr) / (ecmin_t - epsr)
        var ept = ecmin_t - sigmm / er

        var reload_mask = remaining_mask & strain.le(ept)
        var sig_trial = sig_c + ec0 * deps
        var tangent_trial = ec0
        var sigmin = sigmm + er * (strain - ecmin_t)
        var sigmax = half * er * (strain - ept)
        var reload_low_mask = reload_mask & sig_trial.le(sigmin)
        var reload_high_mask = reload_mask & sig_trial.ge(sigmax)
        sig_trial = reload_low_mask.select(sigmin, sig_trial)
        tangent_trial = reload_low_mask.select(er, tangent_trial)
        sig_trial = reload_high_mask.select(sigmax, sig_trial)
        tangent_trial = reload_high_mask.select(half * er, tangent_trial)
        sig_t = reload_mask.select(sig_trial, sig_t)
        tangent_t = reload_mask.select(tangent_trial, tangent_t)

        var tension_mask = remaining_mask & ~reload_mask
        var epn = ept + dept_t
        var pre_tension_mask = tension_mask & strain.le(epn)
        var sicn_env = _concrete02_tens_envlp_simd[width](fc, epsc0, ft, ets, dept_t)
        var dept_nonzero_mask = dept_t.ne(0.0)
        var e = dept_nonzero_mask.select(sicn_env[0] / dept_t, ec0)
        var sig_pre_tension = e * (strain - ept)
        sig_t = pre_tension_mask.select(sig_pre_tension, sig_t)
        tangent_t = pre_tension_mask.select(e, tangent_t)

        var tension_env_mask = tension_mask & ~pre_tension_mask
        var tens_env = _concrete02_tens_envlp_simd[width](
            fc, epsc0, ft, ets, strain - ept
        )
        sig_t = tension_env_mask.select(tens_env[0], sig_t)
        tangent_t = tension_env_mask.select(tens_env[1], tangent_t)
        dept_t = tension_env_mask.select(strain - ept, dept_t)

        store_float64_contiguous_simd[width](sec_def.runtime_c2_eps_t, slot, eps_t)
        store_float64_contiguous_simd[width](sec_def.runtime_c2_sig_t, slot, sig_t)
        store_float64_contiguous_simd[width](
            sec_def.runtime_c2_tangent_t, slot, tangent_t
        )
        store_float64_contiguous_simd[width](sec_def.runtime_c2_ecmin_t, slot, ecmin_t)
        store_float64_contiguous_simd[width](sec_def.runtime_c2_dept_t, slot, dept_t)

        var fs_vec = sig_t * area_vec
        var ks_vec = tangent_t * area_vec
        axial_force += fs_vec.reduce_add()
        moment_z += (-fs_vec * y_vec).reduce_add()
        k11 += ks_vec.reduce_add()
        k12 += (-ks_vec * y_vec).reduce_add()
        k22 += (ks_vec * y_vec * y_vec).reduce_add()
        i += width


@always_inline
fn _fiber_section2d_runtime_apply_concrete02_range_simd_homogeneous[width: Int](
    mut sec_def: FiberSection2dDef,
    family_offset: Int,
    slot_start: Int,
    count: Int,
    mat_def: UniMaterialDef,
    eps0: Float64,
    kappa: Float64,
    mut axial_force: Float64,
    mut moment_z: Float64,
    mut k11: Float64,
    mut k12: Float64,
    mut k22: Float64,
):
    if count <= 0:
        return

    var fc = SIMD[DType.float64, width](mat_def.p0)
    var epsc0 = SIMD[DType.float64, width](mat_def.p1)
    var fcu = SIMD[DType.float64, width](mat_def.p2)
    var epscu = SIMD[DType.float64, width](mat_def.p3)
    var rat = SIMD[DType.float64, width](mat_def.p4)
    var ft = SIMD[DType.float64, width](mat_def.p5)
    var ets = SIMD[DType.float64, width](mat_def.p6)
    var ec0 = (SIMD[DType.float64, width](2.0) * fc) / epsc0
    var tol = SIMD[DType.float64, width](2.220446049250313e-16)
    var one = SIMD[DType.float64, width](1.0)
    var half = SIMD[DType.float64, width](0.5)

    var i = 0
    while i < count:
        var family_index = family_offset + i
        var y_vec = load_float64_contiguous_simd[width](
            sec_def.concrete02_y_rel, family_index
        )
        var area_vec = load_float64_contiguous_simd[width](
            sec_def.concrete02_area, family_index
        )
        var strain = SIMD[DType.float64, width](eps0) - y_vec * SIMD[DType.float64, width](kappa)
        var slot = slot_start + i

        var eps_c = load_float64_contiguous_simd[width](sec_def.runtime_c2_eps_c, slot)
        var sig_c = load_float64_contiguous_simd[width](sec_def.runtime_c2_sig_c, slot)
        var tangent_c = load_float64_contiguous_simd[width](
            sec_def.runtime_c2_tangent_c, slot
        )
        var ecmin_c = load_float64_contiguous_simd[width](sec_def.runtime_c2_ecmin_c, slot)
        var dept_c = load_float64_contiguous_simd[width](sec_def.runtime_c2_dept_c, slot)

        var deps = strain - eps_c
        var eps_t = strain
        var sig_t = sig_c
        var tangent_t = tangent_c
        var ecmin_t = ecmin_c
        var dept_t = dept_c

        var no_change_mask = _simd_abs_float64[width](deps).lt(tol)
        var compression_mask = strain.lt(ecmin_t) & ~no_change_mask
        var compr_env = _concrete02_compr_envlp_simd[width](fc, epsc0, fcu, epscu, strain)
        sig_t = compression_mask.select(compr_env[0], sig_t)
        tangent_t = compression_mask.select(compr_env[1], tangent_t)
        ecmin_t = compression_mask.select(strain, ecmin_t)

        var remaining_mask = ~(no_change_mask | compression_mask)
        var epsr = (fcu - rat * ec0 * epscu) / (ec0 * (one - rat))
        var sigmr = ec0 * epsr
        var sigmm_env = _concrete02_compr_envlp_simd[width](fc, epsc0, fcu, epscu, ecmin_t)
        var sigmm = sigmm_env[0]
        var er = (sigmm - sigmr) / (ecmin_t - epsr)
        var ept = ecmin_t - sigmm / er

        var reload_mask = remaining_mask & strain.le(ept)
        var sig_trial = sig_c + ec0 * deps
        var tangent_trial = ec0
        var sigmin = sigmm + er * (strain - ecmin_t)
        var sigmax = half * er * (strain - ept)
        var reload_low_mask = reload_mask & sig_trial.le(sigmin)
        var reload_high_mask = reload_mask & sig_trial.ge(sigmax)
        sig_trial = reload_low_mask.select(sigmin, sig_trial)
        tangent_trial = reload_low_mask.select(er, tangent_trial)
        sig_trial = reload_high_mask.select(sigmax, sig_trial)
        tangent_trial = reload_high_mask.select(half * er, tangent_trial)
        sig_t = reload_mask.select(sig_trial, sig_t)
        tangent_t = reload_mask.select(tangent_trial, tangent_t)

        var tension_mask = remaining_mask & ~reload_mask
        var epn = ept + dept_t
        var pre_tension_mask = tension_mask & strain.le(epn)
        var sicn_env = _concrete02_tens_envlp_simd[width](fc, epsc0, ft, ets, dept_t)
        var dept_nonzero_mask = dept_t.ne(0.0)
        var e = dept_nonzero_mask.select(sicn_env[0] / dept_t, ec0)
        var sig_pre_tension = e * (strain - ept)
        sig_t = pre_tension_mask.select(sig_pre_tension, sig_t)
        tangent_t = pre_tension_mask.select(e, tangent_t)

        var tension_env_mask = tension_mask & ~pre_tension_mask
        var tens_env = _concrete02_tens_envlp_simd[width](
            fc, epsc0, ft, ets, strain - ept
        )
        sig_t = tension_env_mask.select(tens_env[0], sig_t)
        tangent_t = tension_env_mask.select(tens_env[1], tangent_t)
        dept_t = tension_env_mask.select(strain - ept, dept_t)

        store_float64_contiguous_simd[width](sec_def.runtime_c2_eps_t, slot, eps_t)
        store_float64_contiguous_simd[width](sec_def.runtime_c2_sig_t, slot, sig_t)
        store_float64_contiguous_simd[width](
            sec_def.runtime_c2_tangent_t, slot, tangent_t
        )
        store_float64_contiguous_simd[width](sec_def.runtime_c2_ecmin_t, slot, ecmin_t)
        store_float64_contiguous_simd[width](sec_def.runtime_c2_dept_t, slot, dept_t)

        var fs_vec = sig_t * area_vec
        var ks_vec = tangent_t * area_vec
        axial_force += fs_vec.reduce_add()
        moment_z += (-fs_vec * y_vec).reduce_add()
        k11 += ks_vec.reduce_add()
        k12 += (-ks_vec * y_vec).reduce_add()
        k22 += (ks_vec * y_vec * y_vec).reduce_add()
        i += width


@always_inline
fn _fiber_section2d_runtime_apply_concrete02_range_simd[width: Int](
    mut sec_def: FiberSection2dDef,
    slot_start: Int,
    count: Int,
    eps0: Float64,
    kappa: Float64,
    mut axial_force: Float64,
    mut moment_z: Float64,
    mut k11: Float64,
    mut k12: Float64,
    mut k22: Float64,
):
    if sec_def.concrete02_single_definition:
        var mat_def = sec_def.concrete02_single_mat_def
        _fiber_section2d_runtime_apply_concrete02_range_simd_homogeneous[width](
            sec_def,
            0,
            slot_start,
            count,
            mat_def,
            eps0,
            kappa,
            axial_force,
            moment_z,
            k11,
            k12,
            k22,
        )
        return
    if len(sec_def.concrete02_group_offsets) > 0:
        for i in range(len(sec_def.concrete02_group_offsets)):
            var mat_def = sec_def.concrete02_group_mat_defs[i]
            _fiber_section2d_runtime_apply_concrete02_range_simd_homogeneous[width](
                sec_def,
                sec_def.concrete02_group_offsets[i],
                slot_start + sec_def.concrete02_group_offsets[i],
                sec_def.concrete02_group_padded_counts[i],
                mat_def,
                eps0,
                kappa,
                axial_force,
                moment_z,
                k11,
                k12,
                k22,
            )
        return
    _fiber_section2d_runtime_apply_concrete02_range_simd_mixed[width](
        sec_def, slot_start, count, eps0, kappa, axial_force, moment_z, k11, k12, k22
    )


@always_inline
fn _fiber_section2d_runtime_apply_steel02_range_simd_mixed[width: Int](
    mut sec_def: FiberSection2dDef,
    slot_start: Int,
    count: Int,
    eps0: Float64,
    kappa: Float64,
    mut axial_force: Float64,
    mut moment_z: Float64,
    mut k11: Float64,
    mut k12: Float64,
    mut k22: Float64,
):
    if count <= 0:
        return

    var i = 0
    while i < count:
        var y_vec = load_float64_contiguous_simd[width](sec_def.steel02_y_rel, i)
        var area_vec = load_float64_contiguous_simd[width](sec_def.steel02_area, i)
        var trial_eps = SIMD[DType.float64, width](eps0) - y_vec * SIMD[
            DType.float64, width
        ](kappa)
        var slot = slot_start + i

        var eps_c = load_float64_contiguous_simd[width](sec_def.runtime_s2_eps_c, slot)
        var sig_c = load_float64_contiguous_simd[width](sec_def.runtime_s2_sig_c, slot)
        var epsmin_c = load_float64_contiguous_simd[width](
            sec_def.runtime_s2_epsmin_c, slot
        )
        var epsmax_c = load_float64_contiguous_simd[width](
            sec_def.runtime_s2_epsmax_c, slot
        )
        var epspl_c = load_float64_contiguous_simd[width](
            sec_def.runtime_s2_epspl_c, slot
        )
        var epss0_c = load_float64_contiguous_simd[width](
            sec_def.runtime_s2_epss0_c, slot
        )
        var sigs0_c = load_float64_contiguous_simd[width](
            sec_def.runtime_s2_sigs0_c, slot
        )
        var epsr_c = load_float64_contiguous_simd[width](sec_def.runtime_s2_epsr_c, slot)
        var sigr_c = load_float64_contiguous_simd[width](sec_def.runtime_s2_sigr_c, slot)

        var fy = load_float64_contiguous_simd[width](sec_def.steel02_fy, i)
        var e0 = load_float64_contiguous_simd[width](sec_def.steel02_e0, i)
        var b = load_float64_contiguous_simd[width](sec_def.steel02_b, i)
        var r0 = load_float64_contiguous_simd[width](sec_def.steel02_r0, i)
        var cr1 = load_float64_contiguous_simd[width](sec_def.steel02_cr1, i)
        var cr2 = load_float64_contiguous_simd[width](sec_def.steel02_cr2, i)
        var a1 = load_float64_contiguous_simd[width](sec_def.steel02_a1, i)
        var a3 = load_float64_contiguous_simd[width](sec_def.steel02_a3, i)
        var sigini = load_float64_contiguous_simd[width](sec_def.steel02_sigini, i)
        var esh = load_float64_contiguous_simd[width](sec_def.steel02_esh, i)
        var epsy = load_float64_contiguous_simd[width](sec_def.steel02_epsy, i)
        var sigini_over_e0 = load_float64_contiguous_simd[width](
            sec_def.steel02_sigini_over_e0, i
        )
        var pos_inv_denom = load_float64_contiguous_simd[width](
            sec_def.steel02_pos_inv_denom, i
        )
        var neg_inv_denom = load_float64_contiguous_simd[width](
            sec_def.steel02_neg_inv_denom, i
        )
        var kon_c = SIMD[DType.int32, width](0)
        for lane in range(width):
            kon_c[lane] = Int32(sec_def.runtime_s2_kon_c[slot + lane])

        var eps = sigini.ne(0.0).select(trial_eps + sigini_over_e0, trial_eps)
        var deps = eps - eps_c

        var epsmin_t = epsmin_c
        var epsmax_t = epsmax_c
        var epspl_t = epspl_c
        var epss0_t = epss0_c
        var sigs0_t = sigs0_c
        var epsr_t = epsr_c
        var sigr_t = sigr_c
        var kon_t = kon_c
        var sig_t = sig_c
        var tangent_t = load_float64_contiguous_simd[width](
            sec_def.runtime_s2_tangent_c, slot
        )

        var tol = SIMD[DType.float64, width](2.220446049250313e-15)
        var no_change_mask = (kon_c.eq(0) | kon_c.eq(3)) & _simd_abs_float64[width](
            deps
        ).lt(tol)
        sig_t = no_change_mask.select(sigini, sig_t)
        tangent_t = no_change_mask.select(e0, tangent_t)
        kon_t = no_change_mask.select(SIMD[DType.int32, width](3), kon_t)

        var active_mask = ~no_change_mask
        var initial_mask = active_mask & (kon_c.eq(0) | kon_c.eq(3))
        epsmax_t = initial_mask.select(epsy, epsmax_t)
        epsmin_t = initial_mask.select(-epsy, epsmin_t)

        var initial_neg_mask = initial_mask & deps.lt(0.0)
        var initial_pos_mask = initial_mask & ~deps.lt(0.0)
        kon_t = initial_neg_mask.select(SIMD[DType.int32, width](2), kon_t)
        epss0_t = initial_neg_mask.select(epsmin_t, epss0_t)
        sigs0_t = initial_neg_mask.select(-fy, sigs0_t)
        epspl_t = initial_neg_mask.select(epsmin_t, epspl_t)

        kon_t = initial_pos_mask.select(SIMD[DType.int32, width](1), kon_t)
        epss0_t = initial_pos_mask.select(epsmax_t, epss0_t)
        sigs0_t = initial_pos_mask.select(fy, sigs0_t)
        epspl_t = initial_pos_mask.select(epsmax_t, epspl_t)

        var pos_reversal_mask = active_mask & kon_t.eq(2) & deps.gt(0.0)
        var epsmin_after_pos = eps_c.lt(epsmin_t).select(eps_c, epsmin_t)
        var d1_pos = (epsmax_t - epsmin_after_pos) * pos_inv_denom
        var shft_pos = SIMD[DType.float64, width](1.0) + a3 * (d1_pos ** SIMD[
            DType.float64, width
        ](0.8))
        var epss0_pos = (
            fy * shft_pos - esh * epsy * shft_pos - sig_c + e0 * eps_c
        ) / (e0 - esh)
        var sigs0_pos = fy * shft_pos + esh * (epss0_pos - epsy * shft_pos)
        kon_t = pos_reversal_mask.select(SIMD[DType.int32, width](1), kon_t)
        epsr_t = pos_reversal_mask.select(eps_c, epsr_t)
        sigr_t = pos_reversal_mask.select(sig_c, sigr_t)
        epsmin_t = pos_reversal_mask.select(epsmin_after_pos, epsmin_t)
        epss0_t = pos_reversal_mask.select(epss0_pos, epss0_t)
        sigs0_t = pos_reversal_mask.select(sigs0_pos, sigs0_t)
        epspl_t = pos_reversal_mask.select(epsmax_t, epspl_t)

        var neg_reversal_mask = active_mask & kon_t.eq(1) & deps.lt(0.0)
        var epsmax_after_neg = eps_c.gt(epsmax_t).select(eps_c, epsmax_t)
        var d1_neg = (epsmax_after_neg - epsmin_t) * neg_inv_denom
        var shft_neg = SIMD[DType.float64, width](1.0) + a1 * (d1_neg ** SIMD[
            DType.float64, width
        ](0.8))
        var epss0_neg = (
            -fy * shft_neg + esh * epsy * shft_neg - sig_c + e0 * eps_c
        ) / (e0 - esh)
        var sigs0_neg = -fy * shft_neg + esh * (epss0_neg + epsy * shft_neg)
        kon_t = neg_reversal_mask.select(SIMD[DType.int32, width](2), kon_t)
        epsr_t = neg_reversal_mask.select(eps_c, epsr_t)
        sigr_t = neg_reversal_mask.select(sig_c, sigr_t)
        epsmax_t = neg_reversal_mask.select(epsmax_after_neg, epsmax_t)
        epss0_t = neg_reversal_mask.select(epss0_neg, epss0_t)
        sigs0_t = neg_reversal_mask.select(sigs0_neg, sigs0_t)
        epspl_t = neg_reversal_mask.select(epsmin_t, epspl_t)

        var xi = _simd_abs_float64[width]((epspl_t - epss0_t) / epsy)
        var r = r0 * (
            SIMD[DType.float64, width](1.0) - (cr1 * xi) / (cr2 + xi)
        )
        var epsrat = (eps - epsr_t) / (epss0_t - epsr_t)
        var dum1 = SIMD[DType.float64, width](1.0) + (
            _simd_abs_float64[width](epsrat) ** r
        )
        var dum2 = dum1 ** (SIMD[DType.float64, width](1.0) / r)

        var sig_active = b * epsrat + (SIMD[DType.float64, width](1.0) - b) * epsrat / dum2
        sig_active = sig_active * (sigs0_t - sigr_t) + sigr_t
        var tangent_active = b + (SIMD[DType.float64, width](1.0) - b) / (dum1 * dum2)
        tangent_active = tangent_active * (sigs0_t - sigr_t) / (epss0_t - epsr_t)
        sig_t = active_mask.select(sig_active, sig_t)
        tangent_t = active_mask.select(tangent_active, tangent_t)

        store_float64_contiguous_simd[width](sec_def.runtime_s2_eps_t, slot, eps)
        store_float64_contiguous_simd[width](sec_def.runtime_s2_sig_t, slot, sig_t)
        store_float64_contiguous_simd[width](
            sec_def.runtime_s2_tangent_t, slot, tangent_t
        )
        store_float64_contiguous_simd[width](
            sec_def.runtime_s2_epsmin_t, slot, epsmin_t
        )
        store_float64_contiguous_simd[width](
            sec_def.runtime_s2_epsmax_t, slot, epsmax_t
        )
        store_float64_contiguous_simd[width](sec_def.runtime_s2_epspl_t, slot, epspl_t)
        store_float64_contiguous_simd[width](sec_def.runtime_s2_epss0_t, slot, epss0_t)
        store_float64_contiguous_simd[width](
            sec_def.runtime_s2_sigs0_t, slot, sigs0_t
        )
        store_float64_contiguous_simd[width](sec_def.runtime_s2_epsr_t, slot, epsr_t)
        store_float64_contiguous_simd[width](sec_def.runtime_s2_sigr_t, slot, sigr_t)
        for lane in range(width):
            sec_def.runtime_s2_kon_t[slot + lane] = Int(kon_t[lane])

        var fs_vec = sig_t * area_vec
        var ks_vec = tangent_t * area_vec
        axial_force += fs_vec.reduce_add()
        moment_z += (-fs_vec * y_vec).reduce_add()
        k11 += ks_vec.reduce_add()
        k12 += (-ks_vec * y_vec).reduce_add()
        k22 += (ks_vec * y_vec * y_vec).reduce_add()
        i += width


@always_inline
fn _fiber_section2d_runtime_apply_steel02_range_simd_homogeneous[width: Int](
    mut sec_def: FiberSection2dDef,
    family_offset: Int,
    slot_start: Int,
    count: Int,
    mat_def: UniMaterialDef,
    eps0: Float64,
    kappa: Float64,
    mut axial_force: Float64,
    mut moment_z: Float64,
    mut k11: Float64,
    mut k12: Float64,
    mut k22: Float64,
):
    if count <= 0:
        return

    var fy = SIMD[DType.float64, width](mat_def.p0)
    var e0 = SIMD[DType.float64, width](mat_def.p1)
    var b = SIMD[DType.float64, width](mat_def.p2)
    var r0 = SIMD[DType.float64, width](mat_def.p3)
    var cr1 = SIMD[DType.float64, width](mat_def.p4)
    var cr2 = SIMD[DType.float64, width](mat_def.p5)
    var a1 = SIMD[DType.float64, width](mat_def.p6)
    var a2 = SIMD[DType.float64, width](mat_def.p7)
    var a3 = SIMD[DType.float64, width](mat_def.p8)
    var a4 = SIMD[DType.float64, width](mat_def.p9)
    var sigini = SIMD[DType.float64, width](mat_def.p10)
    var esh = b * e0
    var epsy = fy / e0
    var tol = SIMD[DType.float64, width](2.220446049250313e-15)

    var i = 0
    while i < count:
        var family_index = family_offset + i
        var y_vec = load_float64_contiguous_simd[width](
            sec_def.steel02_y_rel, family_index
        )
        var area_vec = load_float64_contiguous_simd[width](
            sec_def.steel02_area, family_index
        )
        var trial_eps = SIMD[DType.float64, width](eps0) - y_vec * SIMD[
            DType.float64, width
        ](kappa)
        var slot = slot_start + i

        var eps_c = load_float64_contiguous_simd[width](sec_def.runtime_s2_eps_c, slot)
        var sig_c = load_float64_contiguous_simd[width](sec_def.runtime_s2_sig_c, slot)
        var epsmin_c = load_float64_contiguous_simd[width](
            sec_def.runtime_s2_epsmin_c, slot
        )
        var epsmax_c = load_float64_contiguous_simd[width](
            sec_def.runtime_s2_epsmax_c, slot
        )
        var epspl_c = load_float64_contiguous_simd[width](
            sec_def.runtime_s2_epspl_c, slot
        )
        var epss0_c = load_float64_contiguous_simd[width](
            sec_def.runtime_s2_epss0_c, slot
        )
        var sigs0_c = load_float64_contiguous_simd[width](
            sec_def.runtime_s2_sigs0_c, slot
        )
        var epsr_c = load_float64_contiguous_simd[width](sec_def.runtime_s2_epsr_c, slot)
        var sigr_c = load_float64_contiguous_simd[width](sec_def.runtime_s2_sigr_c, slot)
        var kon_c = SIMD[DType.int32, width](0)
        for lane in range(width):
            kon_c[lane] = Int32(sec_def.runtime_s2_kon_c[slot + lane])

        var eps = sigini.ne(0.0).select(trial_eps + sigini / e0, trial_eps)
        var deps = eps - eps_c

        var epsmin_t = epsmin_c
        var epsmax_t = epsmax_c
        var epspl_t = epspl_c
        var epss0_t = epss0_c
        var sigs0_t = sigs0_c
        var epsr_t = epsr_c
        var sigr_t = sigr_c
        var kon_t = kon_c
        var sig_t = sig_c
        var tangent_t = load_float64_contiguous_simd[width](
            sec_def.runtime_s2_tangent_c, slot
        )

        var no_change_mask = (kon_c.eq(0) | kon_c.eq(3)) & _simd_abs_float64[width](
            deps
        ).lt(tol)
        sig_t = no_change_mask.select(sigini, sig_t)
        tangent_t = no_change_mask.select(e0, tangent_t)
        kon_t = no_change_mask.select(SIMD[DType.int32, width](3), kon_t)

        var active_mask = ~no_change_mask
        var initial_mask = active_mask & (kon_c.eq(0) | kon_c.eq(3))
        epsmax_t = initial_mask.select(epsy, epsmax_t)
        epsmin_t = initial_mask.select(-epsy, epsmin_t)

        var initial_neg_mask = initial_mask & deps.lt(0.0)
        var initial_pos_mask = initial_mask & ~deps.lt(0.0)
        kon_t = initial_neg_mask.select(SIMD[DType.int32, width](2), kon_t)
        epss0_t = initial_neg_mask.select(epsmin_t, epss0_t)
        sigs0_t = initial_neg_mask.select(-fy, sigs0_t)
        epspl_t = initial_neg_mask.select(epsmin_t, epspl_t)

        kon_t = initial_pos_mask.select(SIMD[DType.int32, width](1), kon_t)
        epss0_t = initial_pos_mask.select(epsmax_t, epss0_t)
        sigs0_t = initial_pos_mask.select(fy, sigs0_t)
        epspl_t = initial_pos_mask.select(epsmax_t, epspl_t)

        var pos_reversal_mask = active_mask & kon_t.eq(2) & deps.gt(0.0)
        var epsmin_after_pos = eps_c.lt(epsmin_t).select(eps_c, epsmin_t)
        var d1_pos = (epsmax_t - epsmin_after_pos) / (
            SIMD[DType.float64, width](2.0) * (a4 * epsy)
        )
        var shft_pos = SIMD[DType.float64, width](1.0) + a3 * (d1_pos ** SIMD[
            DType.float64, width
        ](0.8))
        var epss0_pos = (
            fy * shft_pos - esh * epsy * shft_pos - sig_c + e0 * eps_c
        ) / (e0 - esh)
        var sigs0_pos = fy * shft_pos + esh * (epss0_pos - epsy * shft_pos)
        kon_t = pos_reversal_mask.select(SIMD[DType.int32, width](1), kon_t)
        epsr_t = pos_reversal_mask.select(eps_c, epsr_t)
        sigr_t = pos_reversal_mask.select(sig_c, sigr_t)
        epsmin_t = pos_reversal_mask.select(epsmin_after_pos, epsmin_t)
        epss0_t = pos_reversal_mask.select(epss0_pos, epss0_t)
        sigs0_t = pos_reversal_mask.select(sigs0_pos, sigs0_t)
        epspl_t = pos_reversal_mask.select(epsmax_t, epspl_t)

        var neg_reversal_mask = active_mask & kon_t.eq(1) & deps.lt(0.0)
        var epsmax_after_neg = eps_c.gt(epsmax_t).select(eps_c, epsmax_t)
        var d1_neg = (epsmax_after_neg - epsmin_t) / (
            SIMD[DType.float64, width](2.0) * (a2 * epsy)
        )
        var shft_neg = SIMD[DType.float64, width](1.0) + a1 * (d1_neg ** SIMD[
            DType.float64, width
        ](0.8))
        var epss0_neg = (
            -fy * shft_neg + esh * epsy * shft_neg - sig_c + e0 * eps_c
        ) / (e0 - esh)
        var sigs0_neg = -fy * shft_neg + esh * (epss0_neg + epsy * shft_neg)
        kon_t = neg_reversal_mask.select(SIMD[DType.int32, width](2), kon_t)
        epsr_t = neg_reversal_mask.select(eps_c, epsr_t)
        sigr_t = neg_reversal_mask.select(sig_c, sigr_t)
        epsmax_t = neg_reversal_mask.select(epsmax_after_neg, epsmax_t)
        epss0_t = neg_reversal_mask.select(epss0_neg, epss0_t)
        sigs0_t = neg_reversal_mask.select(sigs0_neg, sigs0_t)
        epspl_t = neg_reversal_mask.select(epsmin_t, epspl_t)

        var xi = _simd_abs_float64[width]((epspl_t - epss0_t) / epsy)
        var r = r0 * (
            SIMD[DType.float64, width](1.0) - (cr1 * xi) / (cr2 + xi)
        )
        var epsrat = (eps - epsr_t) / (epss0_t - epsr_t)
        var dum1 = SIMD[DType.float64, width](1.0) + (
            _simd_abs_float64[width](epsrat) ** r
        )
        var dum2 = dum1 ** (SIMD[DType.float64, width](1.0) / r)

        var sig_active = b * epsrat + (SIMD[DType.float64, width](1.0) - b) * epsrat / dum2
        sig_active = sig_active * (sigs0_t - sigr_t) + sigr_t
        var tangent_active = b + (SIMD[DType.float64, width](1.0) - b) / (dum1 * dum2)
        tangent_active = tangent_active * (sigs0_t - sigr_t) / (epss0_t - epsr_t)
        sig_t = active_mask.select(sig_active, sig_t)
        tangent_t = active_mask.select(tangent_active, tangent_t)

        store_float64_contiguous_simd[width](sec_def.runtime_s2_eps_t, slot, eps)
        store_float64_contiguous_simd[width](sec_def.runtime_s2_sig_t, slot, sig_t)
        store_float64_contiguous_simd[width](
            sec_def.runtime_s2_tangent_t, slot, tangent_t
        )
        store_float64_contiguous_simd[width](
            sec_def.runtime_s2_epsmin_t, slot, epsmin_t
        )
        store_float64_contiguous_simd[width](
            sec_def.runtime_s2_epsmax_t, slot, epsmax_t
        )
        store_float64_contiguous_simd[width](sec_def.runtime_s2_epspl_t, slot, epspl_t)
        store_float64_contiguous_simd[width](sec_def.runtime_s2_epss0_t, slot, epss0_t)
        store_float64_contiguous_simd[width](
            sec_def.runtime_s2_sigs0_t, slot, sigs0_t
        )
        store_float64_contiguous_simd[width](sec_def.runtime_s2_epsr_t, slot, epsr_t)
        store_float64_contiguous_simd[width](sec_def.runtime_s2_sigr_t, slot, sigr_t)
        for lane in range(width):
            sec_def.runtime_s2_kon_t[slot + lane] = Int(kon_t[lane])

        var fs_vec = sig_t * area_vec
        var ks_vec = tangent_t * area_vec
        axial_force += fs_vec.reduce_add()
        moment_z += (-fs_vec * y_vec).reduce_add()
        k11 += ks_vec.reduce_add()
        k12 += (-ks_vec * y_vec).reduce_add()
        k22 += (ks_vec * y_vec * y_vec).reduce_add()
        i += width


@always_inline
fn _fiber_section2d_runtime_apply_steel02_range_simd[width: Int](
    mut sec_def: FiberSection2dDef,
    slot_start: Int,
    count: Int,
    eps0: Float64,
    kappa: Float64,
    mut axial_force: Float64,
    mut moment_z: Float64,
    mut k11: Float64,
    mut k12: Float64,
    mut k22: Float64,
):
    if sec_def.steel02_single_definition:
        var mat_def = sec_def.steel02_single_mat_def
        _fiber_section2d_runtime_apply_steel02_range_simd_homogeneous[width](
            sec_def,
            0,
            slot_start,
            count,
            mat_def,
            eps0,
            kappa,
            axial_force,
            moment_z,
            k11,
            k12,
            k22,
        )
        return
    if len(sec_def.steel02_group_offsets) > 0:
        for i in range(len(sec_def.steel02_group_offsets)):
            var mat_def = sec_def.steel02_group_mat_defs[i]
            _fiber_section2d_runtime_apply_steel02_range_simd_homogeneous[width](
                sec_def,
                sec_def.steel02_group_offsets[i],
                slot_start + sec_def.steel02_group_offsets[i],
                sec_def.steel02_group_padded_counts[i],
                mat_def,
                eps0,
                kappa,
                axial_force,
                moment_z,
                k11,
                k12,
                k22,
            )
        return
    _fiber_section2d_runtime_apply_steel02_range_simd_mixed[width](
        sec_def, slot_start, count, eps0, kappa, axial_force, moment_z, k11, k12, k22
    )


@always_inline
fn _fiber_section2d_apply_steel02_range[width: Int](
    mut sec_def: FiberSection2dDef,
    local_offset: Int,
    eps0: Float64,
    kappa: Float64,
    mut axial_force: Float64,
    mut moment_z: Float64,
    mut k11: Float64,
    mut k12: Float64,
    mut k22: Float64,
):
    var slot_start = (
        local_offset // sec_def.fiber_count
    ) * sec_def.steel02_instance_stride
    _fiber_section2d_runtime_apply_steel02_range_simd[width](
        sec_def,
        slot_start,
        sec_def.steel02_padded_count,
        eps0,
        kappa,
        axial_force,
        moment_z,
        k11,
        k12,
        k22,
    )


@always_inline
fn _fiber_section2d_apply_concrete02_range[width: Int](
    mut sec_def: FiberSection2dDef,
    local_offset: Int,
    eps0: Float64,
    kappa: Float64,
    mut axial_force: Float64,
    mut moment_z: Float64,
    mut k11: Float64,
    mut k12: Float64,
    mut k22: Float64,
):
    var slot_start = (
        local_offset // sec_def.fiber_count
    ) * sec_def.concrete02_instance_stride
    _fiber_section2d_runtime_apply_concrete02_range_simd[width](
        sec_def,
        slot_start,
        sec_def.concrete02_padded_count,
        eps0,
        kappa,
        axial_force,
        moment_z,
        k11,
        k12,
        k22,
    )


fn _fiber_section2d_batch_apply_steel01_family(
    mut sec_def: FiberSection2dDef,
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
            var state_index = _fiber_section2d_runtime_local_offset(
                sec_def, point.section_state_offset, point.section_state_count
            ) + sec_def.elastic_count + nonlinear_index
            var eps = point.eps0 - y_rel * point.kappa
            _fiber_section2d_runtime_apply_steel01(mat_def, sec_def, state_index, eps)
            _fiber_section2d_batch_accumulate_nonlinear_response(
                y_rel,
                area,
                sec_def.runtime_sig_t[state_index],
                sec_def.runtime_tangent_t[state_index],
                results[i].response,
            )


fn _fiber_section2d_batch_apply_concrete01_family(
    mut sec_def: FiberSection2dDef,
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
            var state_index = _fiber_section2d_runtime_local_offset(
                sec_def, point.section_state_offset, point.section_state_count
            ) + sec_def.elastic_count + nonlinear_index
            var eps = point.eps0 - y_rel * point.kappa
            _fiber_section2d_runtime_apply_concrete01(mat_def, sec_def, state_index, eps)
            _fiber_section2d_batch_accumulate_nonlinear_response(
                y_rel,
                area,
                sec_def.runtime_sig_t[state_index],
                sec_def.runtime_tangent_t[state_index],
                results[i].response,
            )


fn _fiber_section2d_batch_apply_steel02_family(
    mut sec_def: FiberSection2dDef,
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
        var local_offset = _fiber_section2d_runtime_local_offset(
            sec_def, point.section_state_offset, point.section_state_count
        )
        var state_index = _fiber_section2d_runtime_steel02_slot(
            sec_def, local_offset, nonlinear_index
        )
        var eps = point.eps0 - y_rel * point.kappa
        var family_mat_def = mat_def
        _fiber_section2d_runtime_apply_steel02(
            family_mat_def, sec_def, state_index, eps
        )
            _fiber_section2d_batch_accumulate_nonlinear_response(
                y_rel,
                area,
                sec_def.runtime_s2_sig_t[state_index],
                sec_def.runtime_s2_tangent_t[state_index],
                results[i].response,
            )


fn _fiber_section2d_batch_apply_concrete02_family(
    mut sec_def: FiberSection2dDef,
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
            var local_offset = _fiber_section2d_runtime_local_offset(
                sec_def, point.section_state_offset, point.section_state_count
            )
            var state_index = _fiber_section2d_runtime_concrete02_slot(
                sec_def, local_offset, nonlinear_index
            )
            var eps = point.eps0 - y_rel * point.kappa
            var family_mat_def = mat_def
            _fiber_section2d_runtime_apply_concrete02(
                family_mat_def, sec_def, state_index, eps
            )
            _fiber_section2d_batch_accumulate_nonlinear_response(
                y_rel,
                area,
                sec_def.runtime_c2_sig_t[state_index],
                sec_def.runtime_c2_tangent_t[state_index],
                results[i].response,
            )


fn _fiber_section2d_batch_apply_other_family(
    mut sec_def: FiberSection2dDef,
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
            var state_index = _fiber_section2d_runtime_local_offset(
                sec_def, point.section_state_offset, point.section_state_count
            ) + sec_def.elastic_count + nonlinear_index
            var eps = point.eps0 - y_rel * point.kappa
            _fiber_section2d_runtime_set_trial(mat_def, sec_def, state_index, eps)
            _fiber_section2d_batch_accumulate_nonlinear_response(
                y_rel,
                area,
                sec_def.runtime_sig_t[state_index],
                sec_def.runtime_tangent_t[state_index],
                results[i].response,
            )


@always_inline
fn _fiber_section2d_commit_response_slot(mut sec_def: FiberSection2dDef, slot: Int):
    sec_def.runtime_eps_c[slot] = sec_def.runtime_eps_t[slot]
    sec_def.runtime_sig_c[slot] = sec_def.runtime_sig_t[slot]
    sec_def.runtime_tangent_c[slot] = sec_def.runtime_tangent_t[slot]


@always_inline
fn _fiber_section2d_revert_response_slot(mut sec_def: FiberSection2dDef, slot: Int):
    sec_def.runtime_eps_t[slot] = sec_def.runtime_eps_c[slot]
    sec_def.runtime_sig_t[slot] = sec_def.runtime_sig_c[slot]
    sec_def.runtime_tangent_t[slot] = sec_def.runtime_tangent_c[slot]


@always_inline
fn _fiber_section2d_commit_steel01_slot(mut sec_def: FiberSection2dDef, slot: Int):
    _fiber_section2d_commit_response_slot(sec_def, slot)
    sec_def.runtime_eps_p_c[slot] = sec_def.runtime_eps_p_t[slot]
    sec_def.runtime_alpha_c[slot] = sec_def.runtime_alpha_t[slot]


@always_inline
fn _fiber_section2d_revert_steel01_slot(mut sec_def: FiberSection2dDef, slot: Int):
    _fiber_section2d_revert_response_slot(sec_def, slot)
    sec_def.runtime_eps_p_t[slot] = sec_def.runtime_eps_p_c[slot]
    sec_def.runtime_alpha_t[slot] = sec_def.runtime_alpha_c[slot]


@always_inline
fn _fiber_section2d_commit_concrete01_slot(mut sec_def: FiberSection2dDef, slot: Int):
    _fiber_section2d_commit_response_slot(sec_def, slot)
    sec_def.runtime_min_strain_c[slot] = sec_def.runtime_min_strain_t[slot]
    sec_def.runtime_end_strain_c[slot] = sec_def.runtime_end_strain_t[slot]
    sec_def.runtime_unload_slope_c[slot] = sec_def.runtime_unload_slope_t[slot]


@always_inline
fn _fiber_section2d_revert_concrete01_slot(mut sec_def: FiberSection2dDef, slot: Int):
    _fiber_section2d_revert_response_slot(sec_def, slot)
    sec_def.runtime_min_strain_t[slot] = sec_def.runtime_min_strain_c[slot]
    sec_def.runtime_end_strain_t[slot] = sec_def.runtime_end_strain_c[slot]
    sec_def.runtime_unload_slope_t[slot] = sec_def.runtime_unload_slope_c[slot]


@always_inline
fn _fiber_section2d_commit_steel02_slot(mut sec_def: FiberSection2dDef, slot: Int):
    sec_def.runtime_s2_eps_c[slot] = sec_def.runtime_s2_eps_t[slot]
    sec_def.runtime_s2_sig_c[slot] = sec_def.runtime_s2_sig_t[slot]
    sec_def.runtime_s2_tangent_c[slot] = sec_def.runtime_s2_tangent_t[slot]
    sec_def.runtime_s2_epsmin_c[slot] = sec_def.runtime_s2_epsmin_t[slot]
    sec_def.runtime_s2_epsmax_c[slot] = sec_def.runtime_s2_epsmax_t[slot]
    sec_def.runtime_s2_epspl_c[slot] = sec_def.runtime_s2_epspl_t[slot]
    sec_def.runtime_s2_epss0_c[slot] = sec_def.runtime_s2_epss0_t[slot]
    sec_def.runtime_s2_sigs0_c[slot] = sec_def.runtime_s2_sigs0_t[slot]
    sec_def.runtime_s2_epsr_c[slot] = sec_def.runtime_s2_epsr_t[slot]
    sec_def.runtime_s2_sigr_c[slot] = sec_def.runtime_s2_sigr_t[slot]
    sec_def.runtime_s2_kon_c[slot] = sec_def.runtime_s2_kon_t[slot]


@always_inline
fn _fiber_section2d_revert_steel02_slot(mut sec_def: FiberSection2dDef, slot: Int):
    sec_def.runtime_s2_eps_t[slot] = sec_def.runtime_s2_eps_c[slot]
    sec_def.runtime_s2_sig_t[slot] = sec_def.runtime_s2_sig_c[slot]
    sec_def.runtime_s2_tangent_t[slot] = sec_def.runtime_s2_tangent_c[slot]
    sec_def.runtime_s2_epsmin_t[slot] = sec_def.runtime_s2_epsmin_c[slot]
    sec_def.runtime_s2_epsmax_t[slot] = sec_def.runtime_s2_epsmax_c[slot]
    sec_def.runtime_s2_epspl_t[slot] = sec_def.runtime_s2_epspl_c[slot]
    sec_def.runtime_s2_epss0_t[slot] = sec_def.runtime_s2_epss0_c[slot]
    sec_def.runtime_s2_sigs0_t[slot] = sec_def.runtime_s2_sigs0_c[slot]
    sec_def.runtime_s2_epsr_t[slot] = sec_def.runtime_s2_epsr_c[slot]
    sec_def.runtime_s2_sigr_t[slot] = sec_def.runtime_s2_sigr_c[slot]
    sec_def.runtime_s2_kon_t[slot] = sec_def.runtime_s2_kon_c[slot]


@always_inline
fn _fiber_section2d_commit_concrete02_slot(mut sec_def: FiberSection2dDef, slot: Int):
    sec_def.runtime_c2_eps_c[slot] = sec_def.runtime_c2_eps_t[slot]
    sec_def.runtime_c2_sig_c[slot] = sec_def.runtime_c2_sig_t[slot]
    sec_def.runtime_c2_tangent_c[slot] = sec_def.runtime_c2_tangent_t[slot]
    sec_def.runtime_c2_ecmin_c[slot] = sec_def.runtime_c2_ecmin_t[slot]
    sec_def.runtime_c2_dept_c[slot] = sec_def.runtime_c2_dept_t[slot]


@always_inline
fn _fiber_section2d_revert_concrete02_slot(mut sec_def: FiberSection2dDef, slot: Int):
    sec_def.runtime_c2_eps_t[slot] = sec_def.runtime_c2_eps_c[slot]
    sec_def.runtime_c2_sig_t[slot] = sec_def.runtime_c2_sig_c[slot]
    sec_def.runtime_c2_tangent_t[slot] = sec_def.runtime_c2_tangent_c[slot]
    sec_def.runtime_c2_ecmin_t[slot] = sec_def.runtime_c2_ecmin_c[slot]
    sec_def.runtime_c2_dept_t[slot] = sec_def.runtime_c2_dept_c[slot]


@always_inline
fn _fiber_section2d_commit_steel02_family_simd[width: Int](
    mut sec_def: FiberSection2dDef, slot_start: Int, count: Int
):
    if count <= 0:
        return
    _fiber_section2d_copy_float64_range_simd[width](
        sec_def.runtime_s2_eps_c, slot_start, sec_def.runtime_s2_eps_t, slot_start, count
    )
    _fiber_section2d_copy_float64_range_simd[width](
        sec_def.runtime_s2_sig_c, slot_start, sec_def.runtime_s2_sig_t, slot_start, count
    )
    _fiber_section2d_copy_float64_range_simd[width](
        sec_def.runtime_s2_tangent_c,
        slot_start,
        sec_def.runtime_s2_tangent_t,
        slot_start,
        count,
    )
    _fiber_section2d_copy_float64_range_simd[width](
        sec_def.runtime_s2_epsmin_c,
        slot_start,
        sec_def.runtime_s2_epsmin_t,
        slot_start,
        count,
    )
    _fiber_section2d_copy_float64_range_simd[width](
        sec_def.runtime_s2_epsmax_c,
        slot_start,
        sec_def.runtime_s2_epsmax_t,
        slot_start,
        count,
    )
    _fiber_section2d_copy_float64_range_simd[width](
        sec_def.runtime_s2_epspl_c, slot_start, sec_def.runtime_s2_epspl_t, slot_start, count
    )
    _fiber_section2d_copy_float64_range_simd[width](
        sec_def.runtime_s2_epss0_c, slot_start, sec_def.runtime_s2_epss0_t, slot_start, count
    )
    _fiber_section2d_copy_float64_range_simd[width](
        sec_def.runtime_s2_sigs0_c,
        slot_start,
        sec_def.runtime_s2_sigs0_t,
        slot_start,
        count,
    )
    _fiber_section2d_copy_float64_range_simd[width](
        sec_def.runtime_s2_epsr_c, slot_start, sec_def.runtime_s2_epsr_t, slot_start, count
    )
    _fiber_section2d_copy_float64_range_simd[width](
        sec_def.runtime_s2_sigr_c, slot_start, sec_def.runtime_s2_sigr_t, slot_start, count
    )
    _fiber_section2d_copy_int_range(
        sec_def.runtime_s2_kon_c, slot_start, sec_def.runtime_s2_kon_t, slot_start, count
    )


@always_inline
fn _fiber_section2d_revert_steel02_family_simd[width: Int](
    mut sec_def: FiberSection2dDef, slot_start: Int, count: Int
):
    if count <= 0:
        return
    _fiber_section2d_copy_float64_range_simd[width](
        sec_def.runtime_s2_eps_t, slot_start, sec_def.runtime_s2_eps_c, slot_start, count
    )
    _fiber_section2d_copy_float64_range_simd[width](
        sec_def.runtime_s2_sig_t, slot_start, sec_def.runtime_s2_sig_c, slot_start, count
    )
    _fiber_section2d_copy_float64_range_simd[width](
        sec_def.runtime_s2_tangent_t,
        slot_start,
        sec_def.runtime_s2_tangent_c,
        slot_start,
        count,
    )
    _fiber_section2d_copy_float64_range_simd[width](
        sec_def.runtime_s2_epsmin_t,
        slot_start,
        sec_def.runtime_s2_epsmin_c,
        slot_start,
        count,
    )
    _fiber_section2d_copy_float64_range_simd[width](
        sec_def.runtime_s2_epsmax_t,
        slot_start,
        sec_def.runtime_s2_epsmax_c,
        slot_start,
        count,
    )
    _fiber_section2d_copy_float64_range_simd[width](
        sec_def.runtime_s2_epspl_t, slot_start, sec_def.runtime_s2_epspl_c, slot_start, count
    )
    _fiber_section2d_copy_float64_range_simd[width](
        sec_def.runtime_s2_epss0_t, slot_start, sec_def.runtime_s2_epss0_c, slot_start, count
    )
    _fiber_section2d_copy_float64_range_simd[width](
        sec_def.runtime_s2_sigs0_t,
        slot_start,
        sec_def.runtime_s2_sigs0_c,
        slot_start,
        count,
    )
    _fiber_section2d_copy_float64_range_simd[width](
        sec_def.runtime_s2_epsr_t, slot_start, sec_def.runtime_s2_epsr_c, slot_start, count
    )
    _fiber_section2d_copy_float64_range_simd[width](
        sec_def.runtime_s2_sigr_t, slot_start, sec_def.runtime_s2_sigr_c, slot_start, count
    )
    _fiber_section2d_copy_int_range(
        sec_def.runtime_s2_kon_t, slot_start, sec_def.runtime_s2_kon_c, slot_start, count
    )


@always_inline
fn _fiber_section2d_commit_concrete02_family_simd[width: Int](
    mut sec_def: FiberSection2dDef, slot_start: Int, count: Int
):
    if count <= 0:
        return
    _fiber_section2d_copy_float64_range_simd[width](
        sec_def.runtime_c2_eps_c, slot_start, sec_def.runtime_c2_eps_t, slot_start, count
    )
    _fiber_section2d_copy_float64_range_simd[width](
        sec_def.runtime_c2_sig_c, slot_start, sec_def.runtime_c2_sig_t, slot_start, count
    )
    _fiber_section2d_copy_float64_range_simd[width](
        sec_def.runtime_c2_tangent_c,
        slot_start,
        sec_def.runtime_c2_tangent_t,
        slot_start,
        count,
    )
    _fiber_section2d_copy_float64_range_simd[width](
        sec_def.runtime_c2_ecmin_c, slot_start, sec_def.runtime_c2_ecmin_t, slot_start, count
    )
    _fiber_section2d_copy_float64_range_simd[width](
        sec_def.runtime_c2_dept_c, slot_start, sec_def.runtime_c2_dept_t, slot_start, count
    )


@always_inline
fn _fiber_section2d_revert_concrete02_family_simd[width: Int](
    mut sec_def: FiberSection2dDef, slot_start: Int, count: Int
):
    if count <= 0:
        return
    _fiber_section2d_copy_float64_range_simd[width](
        sec_def.runtime_c2_eps_t, slot_start, sec_def.runtime_c2_eps_c, slot_start, count
    )
    _fiber_section2d_copy_float64_range_simd[width](
        sec_def.runtime_c2_sig_t, slot_start, sec_def.runtime_c2_sig_c, slot_start, count
    )
    _fiber_section2d_copy_float64_range_simd[width](
        sec_def.runtime_c2_tangent_t,
        slot_start,
        sec_def.runtime_c2_tangent_c,
        slot_start,
        count,
    )
    _fiber_section2d_copy_float64_range_simd[width](
        sec_def.runtime_c2_ecmin_t, slot_start, sec_def.runtime_c2_ecmin_c, slot_start, count
    )
    _fiber_section2d_copy_float64_range_simd[width](
        sec_def.runtime_c2_dept_t, slot_start, sec_def.runtime_c2_dept_c, slot_start, count
    )


fn fiber_section2d_batch_eval_same_def(
    mut sec_def: FiberSection2dDef,
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
    # Batch sizes in forceBeamColumn2d are typically small, so point-major
    # traversal preserves contiguous per-section state access and avoids
    # repeated local-offset resolution inside family-major inner loops.
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


fn fiber_section2d_batch_eval(
    mut defs: List[FiberSection2dDef],
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
        ref sec_def = defs[def_index0]
        return fiber_section2d_batch_eval_same_def(
            sec_def,
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
        ref sec_def = defs[def_index]
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


fn _fiber_section2d_rebuild_grouped_family(
    nonlinear_indices: List[Int],
    nonlinear_def_index: List[Int],
    family_y_rel: List[Float64],
    family_area: List[Float64],
    family_mat_defs: List[UniMaterialDef],
    mut family_position_by_nonlinear_index: List[Int],
    mut grouped_nonlinear_indices: List[Int],
    mut grouped_y_rel: List[Float64],
    mut grouped_area: List[Float64],
    mut grouped_mat_defs: List[UniMaterialDef],
    mut group_offsets: List[Int],
    mut group_counts: List[Int],
    mut group_padded_counts: List[Int],
    mut group_mat_defs: List[UniMaterialDef],
):
    if len(nonlinear_indices) != len(family_y_rel) or len(nonlinear_indices) != len(
        family_area
    ) or len(nonlinear_indices) != len(family_mat_defs):
        abort("FiberSection2d grouped family arrays out of sync")

    var unique_def_indices: List[Int] = []
    for i in range(len(nonlinear_indices)):
        var nonlinear_index = nonlinear_indices[i]
        var def_index = nonlinear_def_index[nonlinear_index]
        var seen = False
        for j in range(len(unique_def_indices)):
            if unique_def_indices[j] == def_index:
                seen = True
                break
        if not seen:
            unique_def_indices.append(def_index)

    for group_index in range(len(unique_def_indices)):
        var def_index = unique_def_indices[group_index]
        var group_offset = len(grouped_y_rel)
        var group_count = 0
        var group_mat_def = UniMaterialDef()
        for i in range(len(nonlinear_indices)):
            var nonlinear_index = nonlinear_indices[i]
            if nonlinear_def_index[nonlinear_index] != def_index:
                continue
            grouped_nonlinear_indices.append(nonlinear_index)
            grouped_y_rel.append(family_y_rel[i])
            grouped_area.append(family_area[i])
            grouped_mat_defs.append(family_mat_defs[i])
            family_position_by_nonlinear_index[nonlinear_index] = len(grouped_y_rel) - 1
            if group_count == 0:
                group_mat_def = family_mat_defs[i]
            group_count += 1
        if group_count <= 0:
            continue
        var group_padded_count = _fiber_section2d_padded_family_count(group_count)
        group_offsets.append(group_offset)
        group_counts.append(group_count)
        group_padded_counts.append(group_padded_count)
        group_mat_defs.append(group_mat_def)
        for _ in range(group_count, group_padded_count):
            grouped_y_rel.append(0.0)
            grouped_area.append(0.0)
            grouped_mat_defs.append(group_mat_def)


fn _fiber_section2d_build_steel02_param_arrays(mut sec_def: FiberSection2dDef):
    sec_def.steel02_fy = []
    sec_def.steel02_e0 = []
    sec_def.steel02_b = []
    sec_def.steel02_r0 = []
    sec_def.steel02_cr1 = []
    sec_def.steel02_cr2 = []
    sec_def.steel02_a1 = []
    sec_def.steel02_a2 = []
    sec_def.steel02_a3 = []
    sec_def.steel02_a4 = []
    sec_def.steel02_sigini = []
    sec_def.steel02_esh = []
    sec_def.steel02_epsy = []
    sec_def.steel02_sigini_over_e0 = []
    sec_def.steel02_pos_inv_denom = []
    sec_def.steel02_neg_inv_denom = []
    for i in range(len(sec_def.steel02_mat_defs)):
        var mat_def = sec_def.steel02_mat_defs[i]
        var fy = mat_def.p0
        var e0 = mat_def.p1
        var b = mat_def.p2
        var epsy = fy / e0
        sec_def.steel02_fy.append(fy)
        sec_def.steel02_e0.append(e0)
        sec_def.steel02_b.append(b)
        sec_def.steel02_r0.append(mat_def.p3)
        sec_def.steel02_cr1.append(mat_def.p4)
        sec_def.steel02_cr2.append(mat_def.p5)
        sec_def.steel02_a1.append(mat_def.p6)
        sec_def.steel02_a2.append(mat_def.p7)
        sec_def.steel02_a3.append(mat_def.p8)
        sec_def.steel02_a4.append(mat_def.p9)
        sec_def.steel02_sigini.append(mat_def.p10)
        sec_def.steel02_esh.append(b * e0)
        sec_def.steel02_epsy.append(epsy)
        sec_def.steel02_sigini_over_e0.append(mat_def.p10 / e0)
        sec_def.steel02_pos_inv_denom.append(1.0 / (2.0 * mat_def.p9 * epsy))
        sec_def.steel02_neg_inv_denom.append(1.0 / (2.0 * mat_def.p7 * epsy))


fn _fiber_section2d_build_concrete02_param_arrays(mut sec_def: FiberSection2dDef):
    sec_def.concrete02_fc = []
    sec_def.concrete02_epsc0 = []
    sec_def.concrete02_fcu = []
    sec_def.concrete02_epscu = []
    sec_def.concrete02_rat = []
    sec_def.concrete02_ft = []
    sec_def.concrete02_ets = []
    sec_def.concrete02_ec0 = []
    sec_def.concrete02_epsr = []
    sec_def.concrete02_sigmr = []
    for i in range(len(sec_def.concrete02_mat_defs)):
        var mat_def = sec_def.concrete02_mat_defs[i]
        var fc = mat_def.p0
        var epsc0 = mat_def.p1
        var fcu = mat_def.p2
        var epscu = mat_def.p3
        var rat = mat_def.p4
        var ec0 = (2.0 * fc) / epsc0
        var epsr = (fcu - rat * ec0 * epscu) / (ec0 * (1.0 - rat))
        sec_def.concrete02_fc.append(fc)
        sec_def.concrete02_epsc0.append(epsc0)
        sec_def.concrete02_fcu.append(fcu)
        sec_def.concrete02_epscu.append(epscu)
        sec_def.concrete02_rat.append(rat)
        sec_def.concrete02_ft.append(mat_def.p5)
        sec_def.concrete02_ets.append(mat_def.p6)
        sec_def.concrete02_ec0.append(ec0)
        sec_def.concrete02_epsr.append(epsr)
        sec_def.concrete02_sigmr.append(ec0 * epsr)


fn _fiber_section2d_is_single_definition_family(
    nonlinear_indices: List[Int], nonlinear_def_index: List[Int]
) -> Bool:
    if len(nonlinear_indices) <= 0:
        return False
    var first_def_index = nonlinear_def_index[nonlinear_indices[0]]
    for i in range(1, len(nonlinear_indices)):
        if nonlinear_def_index[nonlinear_indices[i]] != first_def_index:
            return False
    return True


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
            sec_def.steel02_family_position_by_nonlinear_index.append(-1)
            sec_def.concrete02_family_position_by_nonlinear_index.append(-1)
            if mat_def.mat_type == UniMaterialTypeTag.Steel01:
                sec_def.steel01_nonlinear_indices.append(nonlinear_index)
            elif mat_def.mat_type == UniMaterialTypeTag.Concrete01:
                sec_def.concrete01_nonlinear_indices.append(nonlinear_index)
            elif mat_def.mat_type == UniMaterialTypeTag.Steel02:
                sec_def.steel02_y_rel.append(y_rel)
                sec_def.steel02_area.append(cell.area)
                sec_def.steel02_mat_defs.append(mat_def)
                sec_def.steel02_family_position_by_nonlinear_index[nonlinear_index] = (
                    len(sec_def.steel02_nonlinear_indices)
                )
                sec_def.steel02_nonlinear_indices.append(nonlinear_index)
            elif mat_def.mat_type == UniMaterialTypeTag.Concrete02:
                sec_def.concrete02_y_rel.append(y_rel)
                sec_def.concrete02_area.append(cell.area)
                sec_def.concrete02_mat_defs.append(mat_def)
                sec_def.concrete02_family_position_by_nonlinear_index[nonlinear_index] = (
                    len(sec_def.concrete02_nonlinear_indices)
                )
                sec_def.concrete02_nonlinear_indices.append(nonlinear_index)
            else:
                sec_def.other_nonlinear_indices.append(nonlinear_index)
    sec_def.elastic_count = len(sec_def.elastic_y_rel)
    sec_def.nonlinear_count = len(sec_def.nonlinear_y_rel)
    var steel02_nonlinear_indices = sec_def.steel02_nonlinear_indices.copy()
    var steel02_y_rel = sec_def.steel02_y_rel.copy()
    var steel02_area = sec_def.steel02_area.copy()
    var steel02_mat_defs = sec_def.steel02_mat_defs.copy()
    sec_def.steel02_nonlinear_indices = []
    sec_def.steel02_y_rel = []
    sec_def.steel02_area = []
    sec_def.steel02_mat_defs = []
    _fiber_section2d_rebuild_grouped_family(
        steel02_nonlinear_indices,
        sec_def.nonlinear_def_index,
        steel02_y_rel,
        steel02_area,
        steel02_mat_defs,
        sec_def.steel02_family_position_by_nonlinear_index,
        sec_def.steel02_nonlinear_indices,
        sec_def.steel02_y_rel,
        sec_def.steel02_area,
        sec_def.steel02_mat_defs,
        sec_def.steel02_group_offsets,
        sec_def.steel02_group_counts,
        sec_def.steel02_group_padded_counts,
        sec_def.steel02_group_mat_defs,
    )
    var concrete02_nonlinear_indices = sec_def.concrete02_nonlinear_indices.copy()
    var concrete02_y_rel = sec_def.concrete02_y_rel.copy()
    var concrete02_area = sec_def.concrete02_area.copy()
    var concrete02_mat_defs = sec_def.concrete02_mat_defs.copy()
    sec_def.concrete02_nonlinear_indices = []
    sec_def.concrete02_y_rel = []
    sec_def.concrete02_area = []
    sec_def.concrete02_mat_defs = []
    _fiber_section2d_rebuild_grouped_family(
        concrete02_nonlinear_indices,
        sec_def.nonlinear_def_index,
        concrete02_y_rel,
        concrete02_area,
        concrete02_mat_defs,
        sec_def.concrete02_family_position_by_nonlinear_index,
        sec_def.concrete02_nonlinear_indices,
        sec_def.concrete02_y_rel,
        sec_def.concrete02_area,
        sec_def.concrete02_mat_defs,
        sec_def.concrete02_group_offsets,
        sec_def.concrete02_group_counts,
        sec_def.concrete02_group_padded_counts,
        sec_def.concrete02_group_mat_defs,
    )
    _fiber_section2d_build_steel02_param_arrays(sec_def)
    _fiber_section2d_build_concrete02_param_arrays(sec_def)
    sec_def.steel02_count = len(sec_def.steel02_nonlinear_indices)
    sec_def.concrete02_count = len(sec_def.concrete02_nonlinear_indices)
    sec_def.steel02_single_definition = _fiber_section2d_is_single_definition_family(
        sec_def.steel02_nonlinear_indices, sec_def.nonlinear_def_index
    )
    if sec_def.steel02_single_definition:
        sec_def.steel02_single_mat_def = sec_def.steel02_group_mat_defs[0]
    sec_def.concrete02_single_definition = _fiber_section2d_is_single_definition_family(
        sec_def.concrete02_nonlinear_indices, sec_def.nonlinear_def_index
    )
    if sec_def.concrete02_single_definition:
        sec_def.concrete02_single_mat_def = sec_def.concrete02_group_mat_defs[0]
    sec_def.steel02_padded_count = len(sec_def.steel02_y_rel)
    sec_def.steel02_instance_stride = sec_def.steel02_padded_count
    sec_def.concrete02_padded_count = len(sec_def.concrete02_y_rel)
    sec_def.concrete02_instance_stride = sec_def.concrete02_padded_count
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
    mut defs: List[FiberSection2dDef],
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
        ref sec_def = defs[s]
        section_uniaxial_offsets[s] = fiber_section2d_runtime_alloc_instances(sec_def, 1)
        section_uniaxial_counts[s] = sec_def.fiber_count
        if sec_def.fiber_offset < 0 or sec_def.fiber_offset + sec_def.fiber_count > len(
            fibers
        ):
            abort("FiberSection2d fiber data out of range")
        for i in range(sec_def.elastic_count):
            var def_index = sec_def.elastic_def_index[i]
            if def_index < 0 or def_index >= len(uniaxial_defs):
                abort("FiberSection2d fiber material definition out of range")
            uniaxial_state_defs.append(def_index)
        for i in range(sec_def.nonlinear_count):
            var mat_def = sec_def.nonlinear_mat_defs[i]
            var def_index = sec_def.nonlinear_def_index[i]
            uniaxial_state_defs.append(def_index)
            if not uni_mat_is_elastic(mat_def):
                used_nonelastic = True

    return used_nonelastic


fn fiber_section2d_set_trial_from_offset(
    mut sec_def: FiberSection2dDef,
    mut uniaxial_states: List[UniMaterialState],
    section_state_offset: Int,
    section_state_count: Int,
    eps0: Float64,
    kappa: Float64,
) -> FiberSection2dResponse:
    var local_offset = _fiber_section2d_runtime_local_offset(
        sec_def, section_state_offset, section_state_count
    )

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

    _fiber_section2d_apply_steel02_range[FLOAT64_SIMD_WIDTH](
        sec_def, local_offset, eps0, kappa, axial_force, moment_z, k11, k12, k22
    )
    _fiber_section2d_apply_concrete02_range[FLOAT64_SIMD_WIDTH](
        sec_def, local_offset, eps0, kappa, axial_force, moment_z, k11, k12, k22
    )
    _fiber_section2d_apply_steel01_family_from_offset(
        sec_def, local_offset, eps0, kappa, axial_force, moment_z, k11, k12, k22
    )
    _fiber_section2d_apply_concrete01_family_from_offset(
        sec_def, local_offset, eps0, kappa, axial_force, moment_z, k11, k12, k22
    )
    _fiber_section2d_apply_other_nonlinear_fallback_from_offset(
        sec_def, local_offset, eps0, kappa, axial_force, moment_z, k11, k12, k22
    )

    return FiberSection2dResponse(axial_force, moment_z, k11, k12, k22)


fn fiber_section2d_set_trial(
    section_index: Int,
    mut defs: List[FiberSection2dDef],
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

    ref sec_def = defs[section_index]
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
    mut sec_def: FiberSection2dDef,
    mut uniaxial_states: List[UniMaterialState],
    section_state_offset: Int,
    section_state_count: Int,
):
    var local_offset = _fiber_section2d_runtime_local_offset(
        sec_def, section_state_offset, section_state_count
    )
    var nonlinear_state_offset = local_offset + sec_def.elastic_count
    for j in range(len(sec_def.steel01_nonlinear_indices)):
        _fiber_section2d_commit_steel01_slot(
            sec_def, nonlinear_state_offset + sec_def.steel01_nonlinear_indices[j]
        )
    for j in range(len(sec_def.concrete01_nonlinear_indices)):
        _fiber_section2d_commit_concrete01_slot(
            sec_def, nonlinear_state_offset + sec_def.concrete01_nonlinear_indices[j]
        )
    if sec_def.steel02_padded_count > 0:
        _fiber_section2d_commit_steel02_family_simd[FLOAT64_SIMD_WIDTH](
            sec_def,
            (local_offset // sec_def.fiber_count) * sec_def.steel02_instance_stride,
            sec_def.steel02_padded_count,
        )
    if sec_def.concrete02_padded_count > 0:
        _fiber_section2d_commit_concrete02_family_simd[FLOAT64_SIMD_WIDTH](
            sec_def,
            (local_offset // sec_def.fiber_count) * sec_def.concrete02_instance_stride,
            sec_def.concrete02_padded_count,
        )
    for j in range(len(sec_def.other_nonlinear_indices)):
        _fiber_section2d_runtime_commit_slot(
            sec_def, nonlinear_state_offset + sec_def.other_nonlinear_indices[j]
        )


fn fiber_section2d_revert_trial_from_offset(
    mut sec_def: FiberSection2dDef,
    mut uniaxial_states: List[UniMaterialState],
    section_state_offset: Int,
    section_state_count: Int,
):
    var local_offset = _fiber_section2d_runtime_local_offset(
        sec_def, section_state_offset, section_state_count
    )
    var nonlinear_state_offset = local_offset + sec_def.elastic_count
    for j in range(len(sec_def.steel01_nonlinear_indices)):
        _fiber_section2d_revert_steel01_slot(
            sec_def, nonlinear_state_offset + sec_def.steel01_nonlinear_indices[j]
        )
    for j in range(len(sec_def.concrete01_nonlinear_indices)):
        _fiber_section2d_revert_concrete01_slot(
            sec_def, nonlinear_state_offset + sec_def.concrete01_nonlinear_indices[j]
        )
    if sec_def.steel02_padded_count > 0:
        _fiber_section2d_revert_steel02_family_simd[FLOAT64_SIMD_WIDTH](
            sec_def,
            (local_offset // sec_def.fiber_count) * sec_def.steel02_instance_stride,
            sec_def.steel02_padded_count,
        )
    if sec_def.concrete02_padded_count > 0:
        _fiber_section2d_revert_concrete02_family_simd[FLOAT64_SIMD_WIDTH](
            sec_def,
            (local_offset // sec_def.fiber_count) * sec_def.concrete02_instance_stride,
            sec_def.concrete02_padded_count,
        )
    for j in range(len(sec_def.other_nonlinear_indices)):
        _fiber_section2d_runtime_revert_slot(
            sec_def, nonlinear_state_offset + sec_def.other_nonlinear_indices[j]
        )


fn fiber_section2d_commit(
    section_index: Int,
    mut defs: List[FiberSection2dDef],
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
    ref sec_def = defs[section_index]
    fiber_section2d_commit_from_offset(sec_def, uniaxial_states, offset, count)


fn fiber_section2d_revert_trial(
    section_index: Int,
    mut defs: List[FiberSection2dDef],
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
    ref sec_def = defs[section_index]
    fiber_section2d_revert_trial_from_offset(sec_def, uniaxial_states, offset, count)


fn fiber_section2d_commit_all(
    mut defs: List[FiberSection2dDef],
    section_uniaxial_offsets: List[Int],
    section_uniaxial_counts: List[Int],
    mut uniaxial_states: List[UniMaterialState],
):
    for i in range(len(section_uniaxial_offsets)):
        fiber_section2d_commit(
            i, defs, section_uniaxial_offsets, section_uniaxial_counts, uniaxial_states
        )


fn fiber_section2d_revert_trial_all(
    mut defs: List[FiberSection2dDef],
    section_uniaxial_offsets: List[Int],
    section_uniaxial_counts: List[Int],
    mut uniaxial_states: List[UniMaterialState],
):
    for i in range(len(section_uniaxial_offsets)):
        fiber_section2d_revert_trial(
            i, defs, section_uniaxial_offsets, section_uniaxial_counts, uniaxial_states
        )


fn fiber_section2d_commit_runtime_all(mut defs: List[FiberSection2dDef]):
    for s in range(len(defs)):
        ref sec_def = defs[s]
        if sec_def.fiber_count <= 0:
            continue
        var instance_count = sec_def.runtime_state_count // sec_def.fiber_count
        for inst in range(instance_count):
            var base = inst * sec_def.fiber_count + sec_def.elastic_count
            for j in range(len(sec_def.steel01_nonlinear_indices)):
                _fiber_section2d_commit_steel01_slot(
                    sec_def, base + sec_def.steel01_nonlinear_indices[j]
                )
            for j in range(len(sec_def.concrete01_nonlinear_indices)):
                _fiber_section2d_commit_concrete01_slot(
                    sec_def, base + sec_def.concrete01_nonlinear_indices[j]
                )
            _fiber_section2d_commit_steel02_family_simd[FLOAT64_SIMD_WIDTH](
                sec_def,
                inst * sec_def.steel02_instance_stride,
                sec_def.steel02_padded_count,
            )
            _fiber_section2d_commit_concrete02_family_simd[FLOAT64_SIMD_WIDTH](
                sec_def,
                inst * sec_def.concrete02_instance_stride,
                sec_def.concrete02_padded_count,
            )
            for j in range(len(sec_def.other_nonlinear_indices)):
                _fiber_section2d_runtime_commit_slot(
                    sec_def, base + sec_def.other_nonlinear_indices[j]
                )


fn fiber_section2d_revert_trial_runtime_all(mut defs: List[FiberSection2dDef]):
    for s in range(len(defs)):
        ref sec_def = defs[s]
        if sec_def.fiber_count <= 0:
            continue
        var instance_count = sec_def.runtime_state_count // sec_def.fiber_count
        for inst in range(instance_count):
            var base = inst * sec_def.fiber_count + sec_def.elastic_count
            for j in range(len(sec_def.steel01_nonlinear_indices)):
                _fiber_section2d_revert_steel01_slot(
                    sec_def, base + sec_def.steel01_nonlinear_indices[j]
                )
            for j in range(len(sec_def.concrete01_nonlinear_indices)):
                _fiber_section2d_revert_concrete01_slot(
                    sec_def, base + sec_def.concrete01_nonlinear_indices[j]
                )
            _fiber_section2d_revert_steel02_family_simd[FLOAT64_SIMD_WIDTH](
                sec_def,
                inst * sec_def.steel02_instance_stride,
                sec_def.steel02_padded_count,
            )
            _fiber_section2d_revert_concrete02_family_simd[FLOAT64_SIMD_WIDTH](
                sec_def,
                inst * sec_def.concrete02_instance_stride,
                sec_def.concrete02_padded_count,
            )
            for j in range(len(sec_def.other_nonlinear_indices)):
                _fiber_section2d_runtime_revert_slot(
                    sec_def, base + sec_def.other_nonlinear_indices[j]
                )
