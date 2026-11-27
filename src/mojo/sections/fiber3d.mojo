from sections.patch_cells import (
    circ_cell_area_and_centroid,
    circ_layer_effective_angles,
    quad_cell_centroid,
)
from collections import List
from math import acos, cos, sin
from os import abort
from python import PythonObject

from materials import (
    UniMaterialDef,
    UniMaterialState,
    uni_mat_initial_tangent,
    uni_mat_is_elastic,
    uniaxial_commit,
    uniaxial_revert_trial,
    uniaxial_set_trial_strain,
)
from sections.fiber2d import FiberCell
from sections.fiber_kernels import (
    _concrete01_eval_simd_step,
    _concrete02_eval_simd_step,
    _fiber_copy_float64_range_simd,
    _fiber_copy_int32_range_simd,
    _fiber_padded_family_count,
    _steel01_eval_simd_step,
    _steel02_eval_simd_step,
    load_int32_ptr_contiguous_simd,
    store_int32_ptr_contiguous_simd,
)
from solver.run_case.input_types import (
    FiberInput,
    FiberLayerInput,
    FiberPatchInput,
    SectionInput,
)
from solver.simd_contiguous import (
    FLOAT64_SIMD_WIDTH,
    load_float64_ptr_contiguous_simd,
    store_float64_ptr_contiguous_simd,
)
from strut_io import py_len
from tag_types import UniMaterialTypeTag


struct FiberSection3dDef(Defaultable, ImplicitlyCopyable, Movable):
    var fiber_offset: Int
    var fiber_count: Int
    var y_bar: Float64
    var z_bar: Float64
    var elastic_count: Int
    var nonlinear_count: Int
    var has_nonelastic: Bool
    var steel02_only_single_def: Bool
    var all_def_indices: List[Int]

    # Precomputed O(1) elastic fiber contributions
    var elastic_k11: Float64
    var elastic_k12: Float64
    var elastic_k13: Float64
    var elastic_k22: Float64
    var elastic_k23: Float64
    var elastic_k33: Float64
    var elastic_f12: Float64
    var elastic_f13: Float64

    # Grouped & SIMD-padded Steel02 family geometry
    var steel02_count: Int
    var steel02_padded_count: Int
    var steel02_instance_stride: Int
    var steel02_single_definition: Bool
    var steel02_single_mat_def: UniMaterialDef
    var steel02_y_rel: List[Float64]
    var steel02_z_rel: List[Float64]
    var steel02_area: List[Float64]
    var steel02_group_offsets: List[Int]
    var steel02_group_counts: List[Int]
    var steel02_group_padded_counts: List[Int]
    var steel02_group_mat_defs: List[UniMaterialDef]

    # Grouped & SIMD-padded Concrete02 family geometry
    var concrete02_count: Int
    var concrete02_padded_count: Int
    var concrete02_instance_stride: Int
    var concrete02_single_definition: Bool
    var concrete02_single_mat_def: UniMaterialDef
    var concrete02_y_rel: List[Float64]
    var concrete02_z_rel: List[Float64]
    var concrete02_area: List[Float64]
    var concrete02_group_offsets: List[Int]
    var concrete02_group_counts: List[Int]
    var concrete02_group_padded_counts: List[Int]
    var concrete02_group_mat_defs: List[UniMaterialDef]

    # Grouped & SIMD-padded Concrete01 family geometry
    var concrete01_count: Int
    var concrete01_padded_count: Int
    var concrete01_instance_stride: Int
    var concrete01_single_definition: Bool
    var concrete01_single_mat_def: UniMaterialDef
    var concrete01_y_rel: List[Float64]
    var concrete01_z_rel: List[Float64]
    var concrete01_area: List[Float64]
    var concrete01_group_offsets: List[Int]
    var concrete01_group_counts: List[Int]
    var concrete01_group_padded_counts: List[Int]
    var concrete01_group_mat_defs: List[UniMaterialDef]

    # Grouped & SIMD-padded Steel01 family geometry
    var steel01_count: Int
    var steel01_padded_count: Int
    var steel01_instance_stride: Int
    var steel01_single_definition: Bool
    var steel01_single_mat_def: UniMaterialDef
    var steel01_y_rel: List[Float64]
    var steel01_z_rel: List[Float64]
    var steel01_area: List[Float64]
    var steel01_group_offsets: List[Int]
    var steel01_group_counts: List[Int]
    var steel01_group_padded_counts: List[Int]
    var steel01_group_mat_defs: List[UniMaterialDef]

    # Scalar fallback family for any other uniaxial material
    var other_count: Int
    var other_y_rel: List[Float64]
    var other_z_rel: List[Float64]
    var other_area: List[Float64]
    var other_mat_defs: List[UniMaterialDef]

    # Per-definition runtime state bookkeeping
    var runtime_state_offset_base: Int
    var runtime_state_count: Int
    var runtime_instance_count: Int

    # Steel02 SoA runtime state (allocated only if steel02_padded_count > 0)
    var runtime_s2_state_count: Int
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
    var runtime_s2_kon_c: List[Int32]
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
    var runtime_s2_kon_t: List[Int32]

    # Concrete02 SoA runtime state (allocated only if concrete02_padded_count > 0)
    var runtime_c2_state_count: Int
    var runtime_c2_eps_c: List[Float64]
    var runtime_c2_sig_c: List[Float64]
    var runtime_c2_tangent_c: List[Float64]
    var runtime_c2_ecmin_c: List[Float64]
    var runtime_c2_dept_c: List[Float64]
    var runtime_c2_eps_t: List[Float64]
    var runtime_c2_sig_t: List[Float64]
    var runtime_c2_tangent_t: List[Float64]
    var runtime_c2_ecmin_t: List[Float64]
    var runtime_c2_dept_t: List[Float64]

    # Concrete01 SoA runtime state (allocated only if concrete01_padded_count > 0)
    var runtime_c1_state_count: Int
    var runtime_c1_eps_c: List[Float64]
    var runtime_c1_sig_c: List[Float64]
    var runtime_c1_tangent_c: List[Float64]
    var runtime_c1_min_strain_c: List[Float64]
    var runtime_c1_end_strain_c: List[Float64]
    var runtime_c1_unload_slope_c: List[Float64]
    var runtime_c1_eps_t: List[Float64]
    var runtime_c1_sig_t: List[Float64]
    var runtime_c1_tangent_t: List[Float64]
    var runtime_c1_min_strain_t: List[Float64]
    var runtime_c1_end_strain_t: List[Float64]
    var runtime_c1_unload_slope_t: List[Float64]

    # Steel01 SoA runtime state (allocated only if steel01_padded_count > 0)
    var runtime_s1_state_count: Int
    var runtime_s1_eps_c: List[Float64]
    var runtime_s1_sig_c: List[Float64]
    var runtime_s1_tangent_c: List[Float64]
    var runtime_s1_eps_p_c: List[Float64]
    var runtime_s1_alpha_c: List[Float64]
    var runtime_s1_eps_t: List[Float64]
    var runtime_s1_sig_t: List[Float64]
    var runtime_s1_tangent_t: List[Float64]
    var runtime_s1_eps_p_t: List[Float64]
    var runtime_s1_alpha_t: List[Float64]

    # Other nonlinear fallback runtime state (allocated only if other_count > 0)
    var runtime_other_states: List[UniMaterialState]

    var added_shear_flexibility_y: Float64
    var added_shear_flexibility_z: Float64

    fn __init__(out self):
        self.fiber_offset = 0
        self.fiber_count = 0
        self.y_bar = 0.0
        self.z_bar = 0.0
        self.added_shear_flexibility_y = 0.0
        self.added_shear_flexibility_z = 0.0
        self.elastic_count = 0
        self.nonlinear_count = 0
        self.has_nonelastic = False
        self.steel02_only_single_def = False
        self.all_def_indices = []
        self.elastic_k11 = 0.0
        self.elastic_k12 = 0.0
        self.elastic_k13 = 0.0
        self.elastic_k22 = 0.0
        self.elastic_k23 = 0.0
        self.elastic_k33 = 0.0
        self.elastic_f12 = 0.0
        self.elastic_f13 = 0.0
        self.steel02_count = 0
        self.steel02_padded_count = 0
        self.steel02_instance_stride = 0
        self.steel02_single_definition = False
        self.steel02_single_mat_def = UniMaterialDef()
        self.steel02_y_rel = []
        self.steel02_z_rel = []
        self.steel02_area = []
        self.steel02_group_offsets = []
        self.steel02_group_counts = []
        self.steel02_group_padded_counts = []
        self.steel02_group_mat_defs = []
        self.concrete02_count = 0
        self.concrete02_padded_count = 0
        self.concrete02_instance_stride = 0
        self.concrete02_single_definition = False
        self.concrete02_single_mat_def = UniMaterialDef()
        self.concrete02_y_rel = []
        self.concrete02_z_rel = []
        self.concrete02_area = []
        self.concrete02_group_offsets = []
        self.concrete02_group_counts = []
        self.concrete02_group_padded_counts = []
        self.concrete02_group_mat_defs = []
        self.concrete01_count = 0
        self.concrete01_padded_count = 0
        self.concrete01_instance_stride = 0
        self.concrete01_single_definition = False
        self.concrete01_single_mat_def = UniMaterialDef()
        self.concrete01_y_rel = []
        self.concrete01_z_rel = []
        self.concrete01_area = []
        self.concrete01_group_offsets = []
        self.concrete01_group_counts = []
        self.concrete01_group_padded_counts = []
        self.concrete01_group_mat_defs = []
        self.steel01_count = 0
        self.steel01_padded_count = 0
        self.steel01_instance_stride = 0
        self.steel01_single_definition = False
        self.steel01_single_mat_def = UniMaterialDef()
        self.steel01_y_rel = []
        self.steel01_z_rel = []
        self.steel01_area = []
        self.steel01_group_offsets = []
        self.steel01_group_counts = []
        self.steel01_group_padded_counts = []
        self.steel01_group_mat_defs = []
        self.other_count = 0
        self.other_y_rel = []
        self.other_z_rel = []
        self.other_area = []
        self.other_mat_defs = []
        self.runtime_state_offset_base = 0
        self.runtime_state_count = 0
        self.runtime_instance_count = 0
        self.runtime_s2_state_count = 0
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
        self.runtime_c2_state_count = 0
        self.runtime_c2_eps_c = []
        self.runtime_c2_sig_c = []
        self.runtime_c2_tangent_c = []
        self.runtime_c2_ecmin_c = []
        self.runtime_c2_dept_c = []
        self.runtime_c2_eps_t = []
        self.runtime_c2_sig_t = []
        self.runtime_c2_tangent_t = []
        self.runtime_c2_ecmin_t = []
        self.runtime_c2_dept_t = []
        self.runtime_c1_state_count = 0
        self.runtime_c1_eps_c = []
        self.runtime_c1_sig_c = []
        self.runtime_c1_tangent_c = []
        self.runtime_c1_min_strain_c = []
        self.runtime_c1_end_strain_c = []
        self.runtime_c1_unload_slope_c = []
        self.runtime_c1_eps_t = []
        self.runtime_c1_sig_t = []
        self.runtime_c1_tangent_t = []
        self.runtime_c1_min_strain_t = []
        self.runtime_c1_end_strain_t = []
        self.runtime_c1_unload_slope_t = []
        self.runtime_s1_state_count = 0
        self.runtime_s1_eps_c = []
        self.runtime_s1_sig_c = []
        self.runtime_s1_tangent_c = []
        self.runtime_s1_eps_p_c = []
        self.runtime_s1_alpha_c = []
        self.runtime_s1_eps_t = []
        self.runtime_s1_sig_t = []
        self.runtime_s1_tangent_t = []
        self.runtime_s1_eps_p_t = []
        self.runtime_s1_alpha_t = []
        self.runtime_other_states = []

    fn __init__(out self, fiber_offset: Int, fiber_count: Int, y_bar: Float64, z_bar: Float64):
        self = Self()
        self.fiber_offset = fiber_offset
        self.fiber_count = fiber_count
        self.y_bar = y_bar
        self.z_bar = z_bar

    fn __copyinit__(out self, existing: Self):
        self.fiber_offset = existing.fiber_offset
        self.fiber_count = existing.fiber_count
        self.y_bar = existing.y_bar
        self.z_bar = existing.z_bar
        self.added_shear_flexibility_y = existing.added_shear_flexibility_y
        self.added_shear_flexibility_z = existing.added_shear_flexibility_z
        self.elastic_count = existing.elastic_count
        self.nonlinear_count = existing.nonlinear_count
        self.has_nonelastic = existing.has_nonelastic
        self.all_def_indices = existing.all_def_indices.copy()
        self.elastic_k11 = existing.elastic_k11
        self.elastic_k12 = existing.elastic_k12
        self.elastic_k13 = existing.elastic_k13
        self.elastic_k22 = existing.elastic_k22
        self.elastic_k23 = existing.elastic_k23
        self.elastic_k33 = existing.elastic_k33
        self.elastic_f12 = existing.elastic_f12
        self.elastic_f13 = existing.elastic_f13
        self.steel02_count = existing.steel02_count
        self.steel02_padded_count = existing.steel02_padded_count
        self.steel02_instance_stride = existing.steel02_instance_stride
        self.steel02_single_definition = existing.steel02_single_definition
        self.steel02_only_single_def = existing.steel02_only_single_def
        self.steel02_single_mat_def = existing.steel02_single_mat_def
        self.steel02_y_rel = existing.steel02_y_rel.copy()
        self.steel02_z_rel = existing.steel02_z_rel.copy()
        self.steel02_area = existing.steel02_area.copy()
        self.steel02_group_offsets = existing.steel02_group_offsets.copy()
        self.steel02_group_counts = existing.steel02_group_counts.copy()
        self.steel02_group_padded_counts = existing.steel02_group_padded_counts.copy()
        self.steel02_group_mat_defs = existing.steel02_group_mat_defs.copy()
        self.concrete02_count = existing.concrete02_count
        self.concrete02_padded_count = existing.concrete02_padded_count
        self.concrete02_instance_stride = existing.concrete02_instance_stride
        self.concrete02_single_definition = existing.concrete02_single_definition
        self.concrete02_single_mat_def = existing.concrete02_single_mat_def
        self.concrete02_y_rel = existing.concrete02_y_rel.copy()
        self.concrete02_z_rel = existing.concrete02_z_rel.copy()
        self.concrete02_area = existing.concrete02_area.copy()
        self.concrete02_group_offsets = existing.concrete02_group_offsets.copy()
        self.concrete02_group_counts = existing.concrete02_group_counts.copy()
        self.concrete02_group_padded_counts = existing.concrete02_group_padded_counts.copy()
        self.concrete02_group_mat_defs = existing.concrete02_group_mat_defs.copy()
        self.concrete01_count = existing.concrete01_count
        self.concrete01_padded_count = existing.concrete01_padded_count
        self.concrete01_instance_stride = existing.concrete01_instance_stride
        self.concrete01_single_definition = existing.concrete01_single_definition
        self.concrete01_single_mat_def = existing.concrete01_single_mat_def
        self.concrete01_y_rel = existing.concrete01_y_rel.copy()
        self.concrete01_z_rel = existing.concrete01_z_rel.copy()
        self.concrete01_area = existing.concrete01_area.copy()
        self.concrete01_group_offsets = existing.concrete01_group_offsets.copy()
        self.concrete01_group_counts = existing.concrete01_group_counts.copy()
        self.concrete01_group_padded_counts = existing.concrete01_group_padded_counts.copy()
        self.concrete01_group_mat_defs = existing.concrete01_group_mat_defs.copy()
        self.steel01_count = existing.steel01_count
        self.steel01_padded_count = existing.steel01_padded_count
        self.steel01_instance_stride = existing.steel01_instance_stride
        self.steel01_single_definition = existing.steel01_single_definition
        self.steel01_single_mat_def = existing.steel01_single_mat_def
        self.steel01_y_rel = existing.steel01_y_rel.copy()
        self.steel01_z_rel = existing.steel01_z_rel.copy()
        self.steel01_area = existing.steel01_area.copy()
        self.steel01_group_offsets = existing.steel01_group_offsets.copy()
        self.steel01_group_counts = existing.steel01_group_counts.copy()
        self.steel01_group_padded_counts = existing.steel01_group_padded_counts.copy()
        self.steel01_group_mat_defs = existing.steel01_group_mat_defs.copy()
        self.other_count = existing.other_count
        self.other_y_rel = existing.other_y_rel.copy()
        self.other_z_rel = existing.other_z_rel.copy()
        self.other_area = existing.other_area.copy()
        self.other_mat_defs = existing.other_mat_defs.copy()
        self.runtime_state_offset_base = existing.runtime_state_offset_base
        self.runtime_state_count = existing.runtime_state_count
        self.runtime_instance_count = existing.runtime_instance_count
        self.runtime_s2_state_count = existing.runtime_s2_state_count
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
        self.runtime_c2_state_count = existing.runtime_c2_state_count
        self.runtime_c2_eps_c = existing.runtime_c2_eps_c.copy()
        self.runtime_c2_sig_c = existing.runtime_c2_sig_c.copy()
        self.runtime_c2_tangent_c = existing.runtime_c2_tangent_c.copy()
        self.runtime_c2_ecmin_c = existing.runtime_c2_ecmin_c.copy()
        self.runtime_c2_dept_c = existing.runtime_c2_dept_c.copy()
        self.runtime_c2_eps_t = existing.runtime_c2_eps_t.copy()
        self.runtime_c2_sig_t = existing.runtime_c2_sig_t.copy()
        self.runtime_c2_tangent_t = existing.runtime_c2_tangent_t.copy()
        self.runtime_c2_ecmin_t = existing.runtime_c2_ecmin_t.copy()
        self.runtime_c2_dept_t = existing.runtime_c2_dept_t.copy()
        self.runtime_c1_state_count = existing.runtime_c1_state_count
        self.runtime_c1_eps_c = existing.runtime_c1_eps_c.copy()
        self.runtime_c1_sig_c = existing.runtime_c1_sig_c.copy()
        self.runtime_c1_tangent_c = existing.runtime_c1_tangent_c.copy()
        self.runtime_c1_min_strain_c = existing.runtime_c1_min_strain_c.copy()
        self.runtime_c1_end_strain_c = existing.runtime_c1_end_strain_c.copy()
        self.runtime_c1_unload_slope_c = existing.runtime_c1_unload_slope_c.copy()
        self.runtime_c1_eps_t = existing.runtime_c1_eps_t.copy()
        self.runtime_c1_sig_t = existing.runtime_c1_sig_t.copy()
        self.runtime_c1_tangent_t = existing.runtime_c1_tangent_t.copy()
        self.runtime_c1_min_strain_t = existing.runtime_c1_min_strain_t.copy()
        self.runtime_c1_end_strain_t = existing.runtime_c1_end_strain_t.copy()
        self.runtime_c1_unload_slope_t = existing.runtime_c1_unload_slope_t.copy()
        self.runtime_s1_state_count = existing.runtime_s1_state_count
        self.runtime_s1_eps_c = existing.runtime_s1_eps_c.copy()
        self.runtime_s1_sig_c = existing.runtime_s1_sig_c.copy()
        self.runtime_s1_tangent_c = existing.runtime_s1_tangent_c.copy()
        self.runtime_s1_eps_p_c = existing.runtime_s1_eps_p_c.copy()
        self.runtime_s1_alpha_c = existing.runtime_s1_alpha_c.copy()
        self.runtime_s1_eps_t = existing.runtime_s1_eps_t.copy()
        self.runtime_s1_sig_t = existing.runtime_s1_sig_t.copy()
        self.runtime_s1_tangent_t = existing.runtime_s1_tangent_t.copy()
        self.runtime_s1_eps_p_t = existing.runtime_s1_eps_p_t.copy()
        self.runtime_s1_alpha_t = existing.runtime_s1_alpha_t.copy()
        self.runtime_other_states = existing.runtime_other_states.copy()


struct FiberSection3dResponse(Defaultable, ImplicitlyCopyable, Movable):
    var axial_force: Float64
    var moment_y: Float64
    var moment_z: Float64
    var k11: Float64
    var k12: Float64
    var k13: Float64
    var k22: Float64
    var k23: Float64
    var k33: Float64

    fn __init__(out self):
        self.axial_force = 0.0
        self.moment_y = 0.0
        self.moment_z = 0.0
        self.k11 = 0.0
        self.k12 = 0.0
        self.k13 = 0.0
        self.k22 = 0.0
        self.k23 = 0.0
        self.k33 = 0.0

    fn __init__(
        out self,
        axial_force: Float64,
        moment_y: Float64,
        moment_z: Float64,
        k11: Float64,
        k12: Float64,
        k13: Float64,
        k22: Float64,
        k23: Float64,
        k33: Float64,
    ):
        self.axial_force = axial_force
        self.moment_y = moment_y
        self.moment_z = moment_z
        self.k11 = k11
        self.k12 = k12
        self.k13 = k13
        self.k22 = k22
        self.k23 = k23
        self.k33 = k33


fn _resolve_uniaxial_def_index(mat_id: Int, uniaxial_def_by_id: List[Int]) -> Int:
    if mat_id < 0:
        abort("FiberSection3d material id must be >= 0")
    if mat_id >= len(uniaxial_def_by_id) or uniaxial_def_by_id[mat_id] < 0:
        abort("FiberSection3d requires uniaxial material for all fibers")
    return uniaxial_def_by_id[mat_id]


fn _fiber_section3d_pack_grouped_family(
    raw_def_indices: List[Int],
    raw_y_rel: List[Float64],
    raw_z_rel: List[Float64],
    raw_area: List[Float64],
    raw_mat_defs: List[UniMaterialDef],
    mut grouped_y_rel: List[Float64],
    mut grouped_z_rel: List[Float64],
    mut grouped_area: List[Float64],
    mut group_offsets: List[Int],
    mut group_counts: List[Int],
    mut group_padded_counts: List[Int],
    mut group_mat_defs: List[UniMaterialDef],
) -> (Int, Int, Bool, UniMaterialDef):
    var raw_count = len(raw_def_indices)
    if raw_count <= 0:
        return (0, 0, False, UniMaterialDef())

    var unique_def_indices: List[Int] = []
    for i in range(raw_count):
        var def_index = raw_def_indices[i]
        var seen = False
        for j in range(len(unique_def_indices)):
            if unique_def_indices[j] == def_index:
                seen = True
                break
        if not seen:
            unique_def_indices.append(def_index)

    for g in range(len(unique_def_indices)):
        var def_index = unique_def_indices[g]
        var group_offset = len(grouped_y_rel)
        var group_count = 0
        var group_mat_def = UniMaterialDef()
        for i in range(raw_count):
            if raw_def_indices[i] != def_index:
                continue
            grouped_y_rel.append(raw_y_rel[i])
            grouped_z_rel.append(raw_z_rel[i])
            grouped_area.append(raw_area[i])
            if group_count == 0:
                group_mat_def = raw_mat_defs[i]
            group_count += 1
        if group_count <= 0:
            continue
        var group_padded = _fiber_padded_family_count(group_count)
        group_offsets.append(group_offset)
        group_counts.append(group_count)
        group_padded_counts.append(group_padded)
        group_mat_defs.append(group_mat_def)
        for _ in range(group_count, group_padded):
            grouped_y_rel.append(0.0)
            grouped_z_rel.append(0.0)
            grouped_area.append(0.0)

    var padded_count = len(grouped_y_rel)
    var is_single = len(group_offsets) == 1
    var single_def = UniMaterialDef()
    if is_single:
        single_def = group_mat_defs[0]
    return (raw_count, padded_count, is_single, single_def)


fn _build_fiber_section3d_def(
    fiber_offset: Int,
    fiber_count: Int,
    y_bar: Float64,
    z_bar: Float64,
    fibers: List[FiberCell],
    uniaxial_defs: List[UniMaterialDef],
) -> FiberSection3dDef:
    var sec_def = FiberSection3dDef(fiber_offset, fiber_count, y_bar, z_bar)
    sec_def.all_def_indices.reserve(fiber_count)

    var elastic_count = 0
    var elastic_k11 = 0.0
    var elastic_k12 = 0.0
    var elastic_k13 = 0.0
    var elastic_k22 = 0.0
    var elastic_k23 = 0.0
    var elastic_k33 = 0.0
    var elastic_modulus0 = 0.0
    var elastic_uniform_modulus = True

    var s2_def_indices: List[Int] = []
    var s2_y: List[Float64] = []
    var s2_z: List[Float64] = []
    var s2_area: List[Float64] = []
    var s2_defs: List[UniMaterialDef] = []

    var c2_def_indices: List[Int] = []
    var c2_y: List[Float64] = []
    var c2_z: List[Float64] = []
    var c2_area: List[Float64] = []
    var c2_defs: List[UniMaterialDef] = []

    var c1_def_indices: List[Int] = []
    var c1_y: List[Float64] = []
    var c1_z: List[Float64] = []
    var c1_area: List[Float64] = []
    var c1_defs: List[UniMaterialDef] = []

    var s1_def_indices: List[Int] = []
    var s1_y: List[Float64] = []
    var s1_z: List[Float64] = []
    var s1_area: List[Float64] = []
    var s1_defs: List[UniMaterialDef] = []

    var nonlinear_count = 0

    for i in range(fiber_count):
        var cell = fibers[fiber_offset + i]
        var def_index = cell.def_index
        if def_index < 0 or def_index >= len(uniaxial_defs):
            abort("FiberSection3d fiber material definition out of range")
        sec_def.all_def_indices.append(def_index)
        var mat_def = uniaxial_defs[def_index]
        var y_rel = cell.y - y_bar
        var z_rel = cell.z - z_bar
        if uni_mat_is_elastic(mat_def):
            var modulus = uni_mat_initial_tangent(mat_def)
            if elastic_count == 0:
                elastic_modulus0 = modulus
            elif modulus != elastic_modulus0:
                elastic_uniform_modulus = False
            var ks = modulus * cell.area
            elastic_k11 += ks
            elastic_k12 += ks * z_rel
            elastic_k13 += -ks * y_rel
            elastic_k22 += ks * z_rel * z_rel
            elastic_k23 += -ks * z_rel * y_rel
            elastic_k33 += ks * y_rel * y_rel
            elastic_count += 1
        else:
            nonlinear_count += 1
            sec_def.has_nonelastic = True
            if mat_def.strain_limits:
                sec_def.other_y_rel.append(y_rel)
                sec_def.other_z_rel.append(z_rel)
                sec_def.other_area.append(cell.area)
                sec_def.other_mat_defs.append(mat_def)
            elif mat_def.mat_type == UniMaterialTypeTag.Steel02 and not mat_def.fatigue.enabled:
                s2_def_indices.append(def_index)
                s2_y.append(y_rel)
                s2_z.append(z_rel)
                s2_area.append(cell.area)
                s2_defs.append(mat_def)
            elif mat_def.mat_type == UniMaterialTypeTag.Concrete02:
                c2_def_indices.append(def_index)
                c2_y.append(y_rel)
                c2_z.append(z_rel)
                c2_area.append(cell.area)
                c2_defs.append(mat_def)
            elif mat_def.mat_type == UniMaterialTypeTag.Concrete01:
                c1_def_indices.append(def_index)
                c1_y.append(y_rel)
                c1_z.append(z_rel)
                c1_area.append(cell.area)
                c1_defs.append(mat_def)
            elif mat_def.mat_type == UniMaterialTypeTag.Steel01:
                s1_def_indices.append(def_index)
                s1_y.append(y_rel)
                s1_z.append(z_rel)
                s1_area.append(cell.area)
                s1_defs.append(mat_def)
            else:
                sec_def.other_y_rel.append(y_rel)
                sec_def.other_z_rel.append(z_rel)
                sec_def.other_area.append(cell.area)
                sec_def.other_mat_defs.append(mat_def)

    sec_def.elastic_count = elastic_count
    sec_def.nonlinear_count = nonlinear_count
    sec_def.elastic_k11 = elastic_k11
    sec_def.elastic_f12 = elastic_k12
    sec_def.elastic_f13 = elastic_k13
    sec_def.elastic_k12 = 0.0 if (elastic_count > 0 and elastic_uniform_modulus) else elastic_k12
    sec_def.elastic_k13 = 0.0 if (elastic_count > 0 and elastic_uniform_modulus) else elastic_k13
    sec_def.elastic_k22 = elastic_k22
    sec_def.elastic_k23 = elastic_k23
    sec_def.elastic_k33 = elastic_k33

    var s2_meta = _fiber_section3d_pack_grouped_family(
        s2_def_indices,
        s2_y,
        s2_z,
        s2_area,
        s2_defs,
        sec_def.steel02_y_rel,
        sec_def.steel02_z_rel,
        sec_def.steel02_area,
        sec_def.steel02_group_offsets,
        sec_def.steel02_group_counts,
        sec_def.steel02_group_padded_counts,
        sec_def.steel02_group_mat_defs,
    )
    sec_def.steel02_count = s2_meta[0]
    sec_def.steel02_padded_count = s2_meta[1]
    sec_def.steel02_instance_stride = s2_meta[1]
    sec_def.steel02_single_definition = s2_meta[2]
    sec_def.steel02_single_mat_def = s2_meta[3]

    var c2_meta = _fiber_section3d_pack_grouped_family(
        c2_def_indices,
        c2_y,
        c2_z,
        c2_area,
        c2_defs,
        sec_def.concrete02_y_rel,
        sec_def.concrete02_z_rel,
        sec_def.concrete02_area,
        sec_def.concrete02_group_offsets,
        sec_def.concrete02_group_counts,
        sec_def.concrete02_group_padded_counts,
        sec_def.concrete02_group_mat_defs,
    )
    sec_def.concrete02_count = c2_meta[0]
    sec_def.concrete02_padded_count = c2_meta[1]
    sec_def.concrete02_instance_stride = c2_meta[1]
    sec_def.concrete02_single_definition = c2_meta[2]
    sec_def.concrete02_single_mat_def = c2_meta[3]

    var c1_meta = _fiber_section3d_pack_grouped_family(
        c1_def_indices,
        c1_y,
        c1_z,
        c1_area,
        c1_defs,
        sec_def.concrete01_y_rel,
        sec_def.concrete01_z_rel,
        sec_def.concrete01_area,
        sec_def.concrete01_group_offsets,
        sec_def.concrete01_group_counts,
        sec_def.concrete01_group_padded_counts,
        sec_def.concrete01_group_mat_defs,
    )
    sec_def.concrete01_count = c1_meta[0]
    sec_def.concrete01_padded_count = c1_meta[1]
    sec_def.concrete01_instance_stride = c1_meta[1]
    sec_def.concrete01_single_definition = c1_meta[2]
    sec_def.concrete01_single_mat_def = c1_meta[3]

    var s1_meta = _fiber_section3d_pack_grouped_family(
        s1_def_indices,
        s1_y,
        s1_z,
        s1_area,
        s1_defs,
        sec_def.steel01_y_rel,
        sec_def.steel01_z_rel,
        sec_def.steel01_area,
        sec_def.steel01_group_offsets,
        sec_def.steel01_group_counts,
        sec_def.steel01_group_padded_counts,
        sec_def.steel01_group_mat_defs,
    )
    sec_def.steel01_count = s1_meta[0]
    sec_def.steel01_padded_count = s1_meta[1]
    sec_def.steel01_instance_stride = s1_meta[1]
    sec_def.steel01_single_definition = s1_meta[2]
    sec_def.steel01_single_mat_def = s1_meta[3]

    sec_def.other_count = len(sec_def.other_y_rel)
    sec_def.steel02_only_single_def = (
        sec_def.steel02_padded_count > 0
        and sec_def.steel02_single_definition
        and sec_def.elastic_count == 0
        and sec_def.concrete02_padded_count == 0
        and sec_def.concrete01_padded_count == 0
        and sec_def.steel01_padded_count == 0
        and sec_def.other_count == 0
    )
    return sec_def^


fn fiber_section3d_runtime_reserve_instances(mut sec_def: FiberSection3dDef, total_instances: Int):
    if total_instances <= 0:
        return
    if sec_def.steel02_instance_stride > 0:
        var cap = total_instances * sec_def.steel02_instance_stride
        sec_def.runtime_s2_eps_c.reserve(cap)
        sec_def.runtime_s2_sig_c.reserve(cap)
        sec_def.runtime_s2_tangent_c.reserve(cap)
        sec_def.runtime_s2_epsmin_c.reserve(cap)
        sec_def.runtime_s2_epsmax_c.reserve(cap)
        sec_def.runtime_s2_epspl_c.reserve(cap)
        sec_def.runtime_s2_epss0_c.reserve(cap)
        sec_def.runtime_s2_sigs0_c.reserve(cap)
        sec_def.runtime_s2_epsr_c.reserve(cap)
        sec_def.runtime_s2_sigr_c.reserve(cap)
        sec_def.runtime_s2_kon_c.reserve(cap)
        sec_def.runtime_s2_eps_t.reserve(cap)
        sec_def.runtime_s2_sig_t.reserve(cap)
        sec_def.runtime_s2_tangent_t.reserve(cap)
        sec_def.runtime_s2_epsmin_t.reserve(cap)
        sec_def.runtime_s2_epsmax_t.reserve(cap)
        sec_def.runtime_s2_epspl_t.reserve(cap)
        sec_def.runtime_s2_epss0_t.reserve(cap)
        sec_def.runtime_s2_sigs0_t.reserve(cap)
        sec_def.runtime_s2_epsr_t.reserve(cap)
        sec_def.runtime_s2_sigr_t.reserve(cap)
        sec_def.runtime_s2_kon_t.reserve(cap)
    if sec_def.concrete02_instance_stride > 0:
        var cap = total_instances * sec_def.concrete02_instance_stride
        sec_def.runtime_c2_eps_c.reserve(cap)
        sec_def.runtime_c2_sig_c.reserve(cap)
        sec_def.runtime_c2_tangent_c.reserve(cap)
        sec_def.runtime_c2_ecmin_c.reserve(cap)
        sec_def.runtime_c2_dept_c.reserve(cap)
        sec_def.runtime_c2_eps_t.reserve(cap)
        sec_def.runtime_c2_sig_t.reserve(cap)
        sec_def.runtime_c2_tangent_t.reserve(cap)
        sec_def.runtime_c2_ecmin_t.reserve(cap)
        sec_def.runtime_c2_dept_t.reserve(cap)
    if sec_def.concrete01_instance_stride > 0:
        var cap = total_instances * sec_def.concrete01_instance_stride
        sec_def.runtime_c1_eps_c.reserve(cap)
        sec_def.runtime_c1_sig_c.reserve(cap)
        sec_def.runtime_c1_tangent_c.reserve(cap)
        sec_def.runtime_c1_min_strain_c.reserve(cap)
        sec_def.runtime_c1_end_strain_c.reserve(cap)
        sec_def.runtime_c1_unload_slope_c.reserve(cap)
        sec_def.runtime_c1_eps_t.reserve(cap)
        sec_def.runtime_c1_sig_t.reserve(cap)
        sec_def.runtime_c1_tangent_t.reserve(cap)
        sec_def.runtime_c1_min_strain_t.reserve(cap)
        sec_def.runtime_c1_end_strain_t.reserve(cap)
        sec_def.runtime_c1_unload_slope_t.reserve(cap)
    if sec_def.steel01_instance_stride > 0:
        var cap = total_instances * sec_def.steel01_instance_stride
        sec_def.runtime_s1_eps_c.reserve(cap)
        sec_def.runtime_s1_sig_c.reserve(cap)
        sec_def.runtime_s1_tangent_c.reserve(cap)
        sec_def.runtime_s1_eps_p_c.reserve(cap)
        sec_def.runtime_s1_alpha_c.reserve(cap)
        sec_def.runtime_s1_eps_t.reserve(cap)
        sec_def.runtime_s1_sig_t.reserve(cap)
        sec_def.runtime_s1_tangent_t.reserve(cap)
        sec_def.runtime_s1_eps_p_t.reserve(cap)
        sec_def.runtime_s1_alpha_t.reserve(cap)
    if sec_def.other_count > 0:
        sec_def.runtime_other_states.reserve(total_instances * sec_def.other_count)


fn fiber_section3d_runtime_alloc_instances(
    mut sec_def: FiberSection3dDef, instance_count: Int
) -> Int:
    if instance_count <= 0:
        return sec_def.runtime_state_offset_base + sec_def.runtime_state_count
    if sec_def.runtime_state_count == 0:
        sec_def.runtime_state_offset_base = 0
    var old_count = sec_def.runtime_state_count
    var new_count = old_count + instance_count * sec_def.fiber_count

    if sec_def.steel02_instance_stride > 0:
        var old_s2 = sec_def.runtime_s2_state_count
        var new_s2 = old_s2 + instance_count * sec_def.steel02_instance_stride
        if sec_def.steel02_single_definition:
            var mat_def = sec_def.steel02_single_mat_def
            var fy = mat_def.p0
            var e0 = mat_def.p1
            var epsy = fy / e0
            var sigini = mat_def.p10
            var epsini = sigini / e0 if sigini != 0.0 else 0.0
            var sig0 = sigini if sigini != 0.0 else 0.0
            sec_def.runtime_s2_eps_c.resize(new_s2, epsini)
            sec_def.runtime_s2_sig_c.resize(new_s2, sig0)
            sec_def.runtime_s2_tangent_c.resize(new_s2, e0)
            sec_def.runtime_s2_epsmin_c.resize(new_s2, -epsy)
            sec_def.runtime_s2_epsmax_c.resize(new_s2, epsy)
            sec_def.runtime_s2_epspl_c.resize(new_s2, 0.0)
            sec_def.runtime_s2_epss0_c.resize(new_s2, 0.0)
            sec_def.runtime_s2_sigs0_c.resize(new_s2, 0.0)
            sec_def.runtime_s2_epsr_c.resize(new_s2, 0.0)
            sec_def.runtime_s2_sigr_c.resize(new_s2, 0.0)
            sec_def.runtime_s2_kon_c.resize(new_s2, Int32(0))
            sec_def.runtime_s2_eps_t.resize(new_s2, epsini)
            sec_def.runtime_s2_sig_t.resize(new_s2, sig0)
            sec_def.runtime_s2_tangent_t.resize(new_s2, e0)
            sec_def.runtime_s2_epsmin_t.resize(new_s2, -epsy)
            sec_def.runtime_s2_epsmax_t.resize(new_s2, epsy)
            sec_def.runtime_s2_epspl_t.resize(new_s2, 0.0)
            sec_def.runtime_s2_epss0_t.resize(new_s2, 0.0)
            sec_def.runtime_s2_sigs0_t.resize(new_s2, 0.0)
            sec_def.runtime_s2_epsr_t.resize(new_s2, 0.0)
            sec_def.runtime_s2_sigr_t.resize(new_s2, 0.0)
            sec_def.runtime_s2_kon_t.resize(new_s2, Int32(0))
        else:
            sec_def.runtime_s2_eps_c.resize(new_s2, 0.0)
            sec_def.runtime_s2_sig_c.resize(new_s2, 0.0)
            sec_def.runtime_s2_tangent_c.resize(new_s2, 0.0)
            sec_def.runtime_s2_epsmin_c.resize(new_s2, 0.0)
            sec_def.runtime_s2_epsmax_c.resize(new_s2, 0.0)
            sec_def.runtime_s2_epspl_c.resize(new_s2, 0.0)
            sec_def.runtime_s2_epss0_c.resize(new_s2, 0.0)
            sec_def.runtime_s2_sigs0_c.resize(new_s2, 0.0)
            sec_def.runtime_s2_epsr_c.resize(new_s2, 0.0)
            sec_def.runtime_s2_sigr_c.resize(new_s2, 0.0)
            sec_def.runtime_s2_kon_c.resize(new_s2, Int32(0))
            sec_def.runtime_s2_eps_t.resize(new_s2, 0.0)
            sec_def.runtime_s2_sig_t.resize(new_s2, 0.0)
            sec_def.runtime_s2_tangent_t.resize(new_s2, 0.0)
            sec_def.runtime_s2_epsmin_t.resize(new_s2, 0.0)
            sec_def.runtime_s2_epsmax_t.resize(new_s2, 0.0)
            sec_def.runtime_s2_epspl_t.resize(new_s2, 0.0)
            sec_def.runtime_s2_epss0_t.resize(new_s2, 0.0)
            sec_def.runtime_s2_sigs0_t.resize(new_s2, 0.0)
            sec_def.runtime_s2_epsr_t.resize(new_s2, 0.0)
            sec_def.runtime_s2_sigr_t.resize(new_s2, 0.0)
            sec_def.runtime_s2_kon_t.resize(new_s2, Int32(0))
            var eps_c_ptr = sec_def.runtime_s2_eps_c.unsafe_ptr()
            var sig_c_ptr = sec_def.runtime_s2_sig_c.unsafe_ptr()
            var tan_c_ptr = sec_def.runtime_s2_tangent_c.unsafe_ptr()
            var emin_c_ptr = sec_def.runtime_s2_epsmin_c.unsafe_ptr()
            var emax_c_ptr = sec_def.runtime_s2_epsmax_c.unsafe_ptr()
            var eps_t_ptr = sec_def.runtime_s2_eps_t.unsafe_ptr()
            var sig_t_ptr = sec_def.runtime_s2_sig_t.unsafe_ptr()
            var tan_t_ptr = sec_def.runtime_s2_tangent_t.unsafe_ptr()
            var emin_t_ptr = sec_def.runtime_s2_epsmin_t.unsafe_ptr()
            var emax_t_ptr = sec_def.runtime_s2_epsmax_t.unsafe_ptr()
            var stride = sec_def.steel02_instance_stride
            for g in range(len(sec_def.steel02_group_offsets)):
                var g_off = sec_def.steel02_group_offsets[g]
                var g_pad = sec_def.steel02_group_padded_counts[g]
                var mat_def = sec_def.steel02_group_mat_defs[g]
                var fy = mat_def.p0
                var e0 = mat_def.p1
                var epsy = fy / e0
                var sigini = mat_def.p10
                var has_sigini = sigini != 0.0
                var epsini = sigini / e0 if has_sigini else 0.0
                for inst in range(instance_count):
                    var base = old_s2 + inst * stride + g_off
                    for k in range(g_pad):
                        var slot = base + k
                        tan_c_ptr[slot] = e0
                        tan_t_ptr[slot] = e0
                        emin_c_ptr[slot] = -epsy
                        emin_t_ptr[slot] = -epsy
                        emax_c_ptr[slot] = epsy
                        emax_t_ptr[slot] = epsy
                        if has_sigini:
                            eps_c_ptr[slot] = epsini
                            eps_t_ptr[slot] = epsini
                            sig_c_ptr[slot] = sigini
                            sig_t_ptr[slot] = sigini
        sec_def.runtime_s2_state_count = new_s2

    if sec_def.concrete02_instance_stride > 0:
        var old_c2 = sec_def.runtime_c2_state_count
        var new_c2 = old_c2 + instance_count * sec_def.concrete02_instance_stride
        if sec_def.concrete02_single_definition:
            var mat_def = sec_def.concrete02_single_mat_def
            var ec0 = (2.0 * mat_def.p0) / mat_def.p1
            sec_def.runtime_c2_eps_c.resize(new_c2, 0.0)
            sec_def.runtime_c2_sig_c.resize(new_c2, 0.0)
            sec_def.runtime_c2_tangent_c.resize(new_c2, ec0)
            sec_def.runtime_c2_ecmin_c.resize(new_c2, 0.0)
            sec_def.runtime_c2_dept_c.resize(new_c2, 0.0)
            sec_def.runtime_c2_eps_t.resize(new_c2, 0.0)
            sec_def.runtime_c2_sig_t.resize(new_c2, 0.0)
            sec_def.runtime_c2_tangent_t.resize(new_c2, ec0)
            sec_def.runtime_c2_ecmin_t.resize(new_c2, 0.0)
            sec_def.runtime_c2_dept_t.resize(new_c2, 0.0)
        else:
            sec_def.runtime_c2_eps_c.resize(new_c2, 0.0)
            sec_def.runtime_c2_sig_c.resize(new_c2, 0.0)
            sec_def.runtime_c2_tangent_c.resize(new_c2, 0.0)
            sec_def.runtime_c2_ecmin_c.resize(new_c2, 0.0)
            sec_def.runtime_c2_dept_c.resize(new_c2, 0.0)
            sec_def.runtime_c2_eps_t.resize(new_c2, 0.0)
            sec_def.runtime_c2_sig_t.resize(new_c2, 0.0)
            sec_def.runtime_c2_tangent_t.resize(new_c2, 0.0)
            sec_def.runtime_c2_ecmin_t.resize(new_c2, 0.0)
            sec_def.runtime_c2_dept_t.resize(new_c2, 0.0)
            var tan_c_ptr = sec_def.runtime_c2_tangent_c.unsafe_ptr()
            var tan_t_ptr = sec_def.runtime_c2_tangent_t.unsafe_ptr()
            var stride = sec_def.concrete02_instance_stride
            for g in range(len(sec_def.concrete02_group_offsets)):
                var g_off = sec_def.concrete02_group_offsets[g]
                var g_pad = sec_def.concrete02_group_padded_counts[g]
                var mat_def = sec_def.concrete02_group_mat_defs[g]
                var ec0 = (2.0 * mat_def.p0) / mat_def.p1
                for inst in range(instance_count):
                    var base = old_c2 + inst * stride + g_off
                    for k in range(g_pad):
                        tan_c_ptr[base + k] = ec0
                        tan_t_ptr[base + k] = ec0
        sec_def.runtime_c2_state_count = new_c2

    if sec_def.concrete01_instance_stride > 0:
        var old_c1 = sec_def.runtime_c1_state_count
        var new_c1 = old_c1 + instance_count * sec_def.concrete01_instance_stride
        if sec_def.concrete01_single_definition:
            var mat_def = sec_def.concrete01_single_mat_def
            var ec0 = (2.0 * mat_def.p0) / mat_def.p1
            sec_def.runtime_c1_eps_c.resize(new_c1, 0.0)
            sec_def.runtime_c1_sig_c.resize(new_c1, 0.0)
            sec_def.runtime_c1_tangent_c.resize(new_c1, ec0)
            sec_def.runtime_c1_min_strain_c.resize(new_c1, 0.0)
            sec_def.runtime_c1_end_strain_c.resize(new_c1, 0.0)
            sec_def.runtime_c1_unload_slope_c.resize(new_c1, ec0)
            sec_def.runtime_c1_eps_t.resize(new_c1, 0.0)
            sec_def.runtime_c1_sig_t.resize(new_c1, 0.0)
            sec_def.runtime_c1_tangent_t.resize(new_c1, ec0)
            sec_def.runtime_c1_min_strain_t.resize(new_c1, 0.0)
            sec_def.runtime_c1_end_strain_t.resize(new_c1, 0.0)
            sec_def.runtime_c1_unload_slope_t.resize(new_c1, ec0)
        else:
            sec_def.runtime_c1_eps_c.resize(new_c1, 0.0)
            sec_def.runtime_c1_sig_c.resize(new_c1, 0.0)
            sec_def.runtime_c1_tangent_c.resize(new_c1, 0.0)
            sec_def.runtime_c1_min_strain_c.resize(new_c1, 0.0)
            sec_def.runtime_c1_end_strain_c.resize(new_c1, 0.0)
            sec_def.runtime_c1_unload_slope_c.resize(new_c1, 0.0)
            sec_def.runtime_c1_eps_t.resize(new_c1, 0.0)
            sec_def.runtime_c1_sig_t.resize(new_c1, 0.0)
            sec_def.runtime_c1_tangent_t.resize(new_c1, 0.0)
            sec_def.runtime_c1_min_strain_t.resize(new_c1, 0.0)
            sec_def.runtime_c1_end_strain_t.resize(new_c1, 0.0)
            sec_def.runtime_c1_unload_slope_t.resize(new_c1, 0.0)
            var tan_c_ptr = sec_def.runtime_c1_tangent_c.unsafe_ptr()
            var tan_t_ptr = sec_def.runtime_c1_tangent_t.unsafe_ptr()
            var us_c_ptr = sec_def.runtime_c1_unload_slope_c.unsafe_ptr()
            var us_t_ptr = sec_def.runtime_c1_unload_slope_t.unsafe_ptr()
            var stride = sec_def.concrete01_instance_stride
            for g in range(len(sec_def.concrete01_group_offsets)):
                var g_off = sec_def.concrete01_group_offsets[g]
                var g_pad = sec_def.concrete01_group_padded_counts[g]
                var mat_def = sec_def.concrete01_group_mat_defs[g]
                var ec0 = (2.0 * mat_def.p0) / mat_def.p1
                for inst in range(instance_count):
                    var base = old_c1 + inst * stride + g_off
                    for k in range(g_pad):
                        tan_c_ptr[base + k] = ec0
                        tan_t_ptr[base + k] = ec0
                        us_c_ptr[base + k] = ec0
                        us_t_ptr[base + k] = ec0
        sec_def.runtime_c1_state_count = new_c1

    if sec_def.steel01_instance_stride > 0:
        var old_s1 = sec_def.runtime_s1_state_count
        var new_s1 = old_s1 + instance_count * sec_def.steel01_instance_stride
        if sec_def.steel01_single_definition:
            var mat_def = sec_def.steel01_single_mat_def
            var e0 = mat_def.p1
            sec_def.runtime_s1_eps_c.resize(new_s1, 0.0)
            sec_def.runtime_s1_sig_c.resize(new_s1, 0.0)
            sec_def.runtime_s1_tangent_c.resize(new_s1, e0)
            sec_def.runtime_s1_eps_p_c.resize(new_s1, 0.0)
            sec_def.runtime_s1_alpha_c.resize(new_s1, 0.0)
            sec_def.runtime_s1_eps_t.resize(new_s1, 0.0)
            sec_def.runtime_s1_sig_t.resize(new_s1, 0.0)
            sec_def.runtime_s1_tangent_t.resize(new_s1, e0)
            sec_def.runtime_s1_eps_p_t.resize(new_s1, 0.0)
            sec_def.runtime_s1_alpha_t.resize(new_s1, 0.0)
        else:
            sec_def.runtime_s1_eps_c.resize(new_s1, 0.0)
            sec_def.runtime_s1_sig_c.resize(new_s1, 0.0)
            sec_def.runtime_s1_tangent_c.resize(new_s1, 0.0)
            sec_def.runtime_s1_eps_p_c.resize(new_s1, 0.0)
            sec_def.runtime_s1_alpha_c.resize(new_s1, 0.0)
            sec_def.runtime_s1_eps_t.resize(new_s1, 0.0)
            sec_def.runtime_s1_sig_t.resize(new_s1, 0.0)
            sec_def.runtime_s1_tangent_t.resize(new_s1, 0.0)
            sec_def.runtime_s1_eps_p_t.resize(new_s1, 0.0)
            sec_def.runtime_s1_alpha_t.resize(new_s1, 0.0)
            var tan_c_ptr = sec_def.runtime_s1_tangent_c.unsafe_ptr()
            var tan_t_ptr = sec_def.runtime_s1_tangent_t.unsafe_ptr()
            var stride = sec_def.steel01_instance_stride
            for g in range(len(sec_def.steel01_group_offsets)):
                var g_off = sec_def.steel01_group_offsets[g]
                var g_pad = sec_def.steel01_group_padded_counts[g]
                var mat_def = sec_def.steel01_group_mat_defs[g]
                var e0 = mat_def.p1
                for inst in range(instance_count):
                    var base = old_s1 + inst * stride + g_off
                    for k in range(g_pad):
                        tan_c_ptr[base + k] = e0
                        tan_t_ptr[base + k] = e0
        sec_def.runtime_s1_state_count = new_s1

    if sec_def.other_count > 0:
        for _ in range(instance_count):
            for i in range(sec_def.other_count):
                sec_def.runtime_other_states.append(UniMaterialState(sec_def.other_mat_defs[i]))

    sec_def.runtime_state_count = new_count
    sec_def.runtime_instance_count += instance_count
    return sec_def.runtime_state_offset_base + old_count


@always_inline
fn _fiber_section3d_runtime_apply_steel02_group_simd[
    width: Int
](
    mut sec_def: FiberSection3dDef,
    family_offset: Int,
    slot_start: Int,
    count: Int,
    mat_def: UniMaterialDef,
    eps0: Float64,
    kappa_y: Float64,
    kappa_z: Float64,
    mut axial_force: Float64,
    mut moment_y: Float64,
    mut moment_z: Float64,
    mut k11: Float64,
    mut k12: Float64,
    mut k13: Float64,
    mut k22: Float64,
    mut k23: Float64,
    mut k33: Float64,
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
    var sigini_over_e0 = sigini / e0
    var pos_inv_denom = SIMD[DType.float64, width](1.0) / (
        SIMD[DType.float64, width](2.0) * (a4 * epsy)
    )
    var neg_inv_denom = SIMD[DType.float64, width](1.0) / (
        SIMD[DType.float64, width](2.0) * (a2 * epsy)
    )

    var eps0_vec = SIMD[DType.float64, width](eps0)
    var ky_vec = SIMD[DType.float64, width](kappa_y)
    var kz_vec = SIMD[DType.float64, width](kappa_z)

    var y_ptr = sec_def.steel02_y_rel.unsafe_ptr()
    var z_ptr = sec_def.steel02_z_rel.unsafe_ptr()
    var area_ptr = sec_def.steel02_area.unsafe_ptr()
    var eps_c_ptr = sec_def.runtime_s2_eps_c.unsafe_ptr()
    var sig_c_ptr = sec_def.runtime_s2_sig_c.unsafe_ptr()
    var tan_c_ptr = sec_def.runtime_s2_tangent_c.unsafe_ptr()
    var emin_c_ptr = sec_def.runtime_s2_epsmin_c.unsafe_ptr()
    var emax_c_ptr = sec_def.runtime_s2_epsmax_c.unsafe_ptr()
    var epl_c_ptr = sec_def.runtime_s2_epspl_c.unsafe_ptr()
    var es0_c_ptr = sec_def.runtime_s2_epss0_c.unsafe_ptr()
    var ss0_c_ptr = sec_def.runtime_s2_sigs0_c.unsafe_ptr()
    var er_c_ptr = sec_def.runtime_s2_epsr_c.unsafe_ptr()
    var sr_c_ptr = sec_def.runtime_s2_sigr_c.unsafe_ptr()
    var kon_c_ptr = sec_def.runtime_s2_kon_c.unsafe_ptr()

    var eps_t_ptr = sec_def.runtime_s2_eps_t.unsafe_ptr()
    var sig_t_ptr = sec_def.runtime_s2_sig_t.unsafe_ptr()
    var tan_t_ptr = sec_def.runtime_s2_tangent_t.unsafe_ptr()
    var emin_t_ptr = sec_def.runtime_s2_epsmin_t.unsafe_ptr()
    var emax_t_ptr = sec_def.runtime_s2_epsmax_t.unsafe_ptr()
    var epl_t_ptr = sec_def.runtime_s2_epspl_t.unsafe_ptr()
    var es0_t_ptr = sec_def.runtime_s2_epss0_t.unsafe_ptr()
    var ss0_t_ptr = sec_def.runtime_s2_sigs0_t.unsafe_ptr()
    var er_t_ptr = sec_def.runtime_s2_epsr_t.unsafe_ptr()
    var sr_t_ptr = sec_def.runtime_s2_sigr_t.unsafe_ptr()
    var kon_t_ptr = sec_def.runtime_s2_kon_t.unsafe_ptr()

    var acc_n = SIMD[DType.float64, width](0.0)
    var acc_my = SIMD[DType.float64, width](0.0)
    var acc_mz_neg = SIMD[DType.float64, width](0.0)
    var acc_k11 = SIMD[DType.float64, width](0.0)
    var acc_k12 = SIMD[DType.float64, width](0.0)
    var acc_k13_neg = SIMD[DType.float64, width](0.0)
    var acc_k22 = SIMD[DType.float64, width](0.0)
    var acc_k23_neg = SIMD[DType.float64, width](0.0)
    var acc_k33 = SIMD[DType.float64, width](0.0)

    var i = 0
    while i < count:
        var fi = family_offset + i
        var slot = slot_start + i
        var y_vec = load_float64_ptr_contiguous_simd[width](y_ptr, fi)
        var z_vec = load_float64_ptr_contiguous_simd[width](z_ptr, fi)
        var area_vec = load_float64_ptr_contiguous_simd[width](area_ptr, fi)
        var trial_eps = eps0_vec + z_vec * ky_vec - y_vec * kz_vec

        var step = _steel02_eval_simd_step[width](
            trial_eps,
            load_float64_ptr_contiguous_simd[width](eps_c_ptr, slot),
            load_float64_ptr_contiguous_simd[width](sig_c_ptr, slot),
            load_float64_ptr_contiguous_simd[width](tan_c_ptr, slot),
            load_float64_ptr_contiguous_simd[width](emin_c_ptr, slot),
            load_float64_ptr_contiguous_simd[width](emax_c_ptr, slot),
            load_float64_ptr_contiguous_simd[width](epl_c_ptr, slot),
            load_float64_ptr_contiguous_simd[width](es0_c_ptr, slot),
            load_float64_ptr_contiguous_simd[width](ss0_c_ptr, slot),
            load_float64_ptr_contiguous_simd[width](er_c_ptr, slot),
            load_float64_ptr_contiguous_simd[width](sr_c_ptr, slot),
            load_int32_ptr_contiguous_simd[width](kon_c_ptr, slot),
            fy,
            e0,
            b,
            r0,
            cr1,
            cr2,
            a1,
            a3,
            sigini,
            esh,
            epsy,
            sigini_over_e0,
            pos_inv_denom,
            neg_inv_denom,
        )
        store_float64_ptr_contiguous_simd[width](eps_t_ptr, slot, step[0])
        store_float64_ptr_contiguous_simd[width](sig_t_ptr, slot, step[1])
        store_float64_ptr_contiguous_simd[width](tan_t_ptr, slot, step[2])
        store_float64_ptr_contiguous_simd[width](emin_t_ptr, slot, step[3])
        store_float64_ptr_contiguous_simd[width](emax_t_ptr, slot, step[4])
        store_float64_ptr_contiguous_simd[width](epl_t_ptr, slot, step[5])
        store_float64_ptr_contiguous_simd[width](es0_t_ptr, slot, step[6])
        store_float64_ptr_contiguous_simd[width](ss0_t_ptr, slot, step[7])
        store_float64_ptr_contiguous_simd[width](er_t_ptr, slot, step[8])
        store_float64_ptr_contiguous_simd[width](sr_t_ptr, slot, step[9])
        store_int32_ptr_contiguous_simd[width](kon_t_ptr, slot, step[10])

        var fs = step[1] * area_vec
        var ks = step[2] * area_vec
        var ks_z = ks * z_vec
        var ks_y = ks * y_vec
        acc_n += fs
        acc_my += fs * z_vec
        acc_mz_neg += fs * y_vec
        acc_k11 += ks
        acc_k12 += ks_z
        acc_k13_neg += ks_y
        acc_k22 += ks_z * z_vec
        acc_k23_neg += ks_z * y_vec
        acc_k33 += ks_y * y_vec
        i += width

    axial_force += acc_n.reduce_add()
    moment_y += acc_my.reduce_add()
    moment_z -= acc_mz_neg.reduce_add()
    k11 += acc_k11.reduce_add()
    k12 += acc_k12.reduce_add()
    k13 -= acc_k13_neg.reduce_add()
    k22 += acc_k22.reduce_add()
    k23 -= acc_k23_neg.reduce_add()
    k33 += acc_k33.reduce_add()


@always_inline
fn _fiber_section3d_runtime_apply_concrete02_group_simd[
    width: Int
](
    mut sec_def: FiberSection3dDef,
    family_offset: Int,
    slot_start: Int,
    count: Int,
    mat_def: UniMaterialDef,
    eps0: Float64,
    kappa_y: Float64,
    kappa_z: Float64,
    mut axial_force: Float64,
    mut moment_y: Float64,
    mut moment_z: Float64,
    mut k11: Float64,
    mut k12: Float64,
    mut k13: Float64,
    mut k22: Float64,
    mut k23: Float64,
    mut k33: Float64,
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
    var eps0_tension = ft / ec0
    var epsu_tension = ft * (
        SIMD[DType.float64, width](1.0) / ets + SIMD[DType.float64, width](1.0) / ec0
    )
    var one = SIMD[DType.float64, width](1.0)
    var epsr = (fcu - rat * ec0 * epscu) / (ec0 * (one - rat))
    var sigmr = ec0 * epsr

    var eps0_vec = SIMD[DType.float64, width](eps0)
    var ky_vec = SIMD[DType.float64, width](kappa_y)
    var kz_vec = SIMD[DType.float64, width](kappa_z)

    var y_ptr = sec_def.concrete02_y_rel.unsafe_ptr()
    var z_ptr = sec_def.concrete02_z_rel.unsafe_ptr()
    var area_ptr = sec_def.concrete02_area.unsafe_ptr()
    var eps_c_ptr = sec_def.runtime_c2_eps_c.unsafe_ptr()
    var sig_c_ptr = sec_def.runtime_c2_sig_c.unsafe_ptr()
    var tan_c_ptr = sec_def.runtime_c2_tangent_c.unsafe_ptr()
    var ecmin_c_ptr = sec_def.runtime_c2_ecmin_c.unsafe_ptr()
    var dept_c_ptr = sec_def.runtime_c2_dept_c.unsafe_ptr()

    var eps_t_ptr = sec_def.runtime_c2_eps_t.unsafe_ptr()
    var sig_t_ptr = sec_def.runtime_c2_sig_t.unsafe_ptr()
    var tan_t_ptr = sec_def.runtime_c2_tangent_t.unsafe_ptr()
    var ecmin_t_ptr = sec_def.runtime_c2_ecmin_t.unsafe_ptr()
    var dept_t_ptr = sec_def.runtime_c2_dept_t.unsafe_ptr()

    var acc_n = SIMD[DType.float64, width](0.0)
    var acc_my = SIMD[DType.float64, width](0.0)
    var acc_mz_neg = SIMD[DType.float64, width](0.0)
    var acc_k11 = SIMD[DType.float64, width](0.0)
    var acc_k12 = SIMD[DType.float64, width](0.0)
    var acc_k13_neg = SIMD[DType.float64, width](0.0)
    var acc_k22 = SIMD[DType.float64, width](0.0)
    var acc_k23_neg = SIMD[DType.float64, width](0.0)
    var acc_k33 = SIMD[DType.float64, width](0.0)

    var i = 0
    while i < count:
        var fi = family_offset + i
        var slot = slot_start + i
        var y_vec = load_float64_ptr_contiguous_simd[width](y_ptr, fi)
        var z_vec = load_float64_ptr_contiguous_simd[width](z_ptr, fi)
        var area_vec = load_float64_ptr_contiguous_simd[width](area_ptr, fi)
        var strain = eps0_vec + z_vec * ky_vec - y_vec * kz_vec

        var step = _concrete02_eval_simd_step[width](
            strain,
            load_float64_ptr_contiguous_simd[width](eps_c_ptr, slot),
            load_float64_ptr_contiguous_simd[width](sig_c_ptr, slot),
            load_float64_ptr_contiguous_simd[width](tan_c_ptr, slot),
            load_float64_ptr_contiguous_simd[width](ecmin_c_ptr, slot),
            load_float64_ptr_contiguous_simd[width](dept_c_ptr, slot),
            fc,
            epsc0,
            fcu,
            epscu,
            ft,
            ets,
            ec0,
            eps0_tension,
            epsu_tension,
            epsr,
            sigmr,
        )
        store_float64_ptr_contiguous_simd[width](eps_t_ptr, slot, strain)
        store_float64_ptr_contiguous_simd[width](sig_t_ptr, slot, step[0])
        store_float64_ptr_contiguous_simd[width](tan_t_ptr, slot, step[1])
        store_float64_ptr_contiguous_simd[width](ecmin_t_ptr, slot, step[2])
        store_float64_ptr_contiguous_simd[width](dept_t_ptr, slot, step[3])

        var fs = step[0] * area_vec
        var ks = step[1] * area_vec
        var ks_z = ks * z_vec
        var ks_y = ks * y_vec
        acc_n += fs
        acc_my += fs * z_vec
        acc_mz_neg += fs * y_vec
        acc_k11 += ks
        acc_k12 += ks_z
        acc_k13_neg += ks_y
        acc_k22 += ks_z * z_vec
        acc_k23_neg += ks_z * y_vec
        acc_k33 += ks_y * y_vec
        i += width

    axial_force += acc_n.reduce_add()
    moment_y += acc_my.reduce_add()
    moment_z -= acc_mz_neg.reduce_add()
    k11 += acc_k11.reduce_add()
    k12 += acc_k12.reduce_add()
    k13 -= acc_k13_neg.reduce_add()
    k22 += acc_k22.reduce_add()
    k23 -= acc_k23_neg.reduce_add()
    k33 += acc_k33.reduce_add()


@always_inline
fn _fiber_section3d_runtime_apply_concrete01_group_simd[
    width: Int
](
    mut sec_def: FiberSection3dDef,
    family_offset: Int,
    slot_start: Int,
    count: Int,
    mat_def: UniMaterialDef,
    eps0: Float64,
    kappa_y: Float64,
    kappa_z: Float64,
    mut axial_force: Float64,
    mut moment_y: Float64,
    mut moment_z: Float64,
    mut k11: Float64,
    mut k12: Float64,
    mut k13: Float64,
    mut k22: Float64,
    mut k23: Float64,
    mut k33: Float64,
):
    if count <= 0:
        return

    var fpc_s = mat_def.p0
    var epsc0_s = mat_def.p1
    var fpcu_s = mat_def.p2
    var epscu_s = mat_def.p3
    var ec0_s = (2.0 * fpc_s) / epsc0_s
    var soft_tan_s = (fpc_s - fpcu_s) / (epsc0_s - epscu_s) if (epsc0_s != epscu_s) else 0.0

    var fpc = SIMD[DType.float64, width](fpc_s)
    var epsc0 = SIMD[DType.float64, width](epsc0_s)
    var fpcu = SIMD[DType.float64, width](fpcu_s)
    var epscu = SIMD[DType.float64, width](epscu_s)
    var ec0 = SIMD[DType.float64, width](ec0_s)
    var softening_tangent = SIMD[DType.float64, width](soft_tan_s)

    var eps0_vec = SIMD[DType.float64, width](eps0)
    var ky_vec = SIMD[DType.float64, width](kappa_y)
    var kz_vec = SIMD[DType.float64, width](kappa_z)

    var y_ptr = sec_def.concrete01_y_rel.unsafe_ptr()
    var z_ptr = sec_def.concrete01_z_rel.unsafe_ptr()
    var area_ptr = sec_def.concrete01_area.unsafe_ptr()
    var eps_c_ptr = sec_def.runtime_c1_eps_c.unsafe_ptr()
    var sig_c_ptr = sec_def.runtime_c1_sig_c.unsafe_ptr()
    var tan_c_ptr = sec_def.runtime_c1_tangent_c.unsafe_ptr()
    var min_c_ptr = sec_def.runtime_c1_min_strain_c.unsafe_ptr()
    var end_c_ptr = sec_def.runtime_c1_end_strain_c.unsafe_ptr()
    var us_c_ptr = sec_def.runtime_c1_unload_slope_c.unsafe_ptr()

    var eps_t_ptr = sec_def.runtime_c1_eps_t.unsafe_ptr()
    var sig_t_ptr = sec_def.runtime_c1_sig_t.unsafe_ptr()
    var tan_t_ptr = sec_def.runtime_c1_tangent_t.unsafe_ptr()
    var min_t_ptr = sec_def.runtime_c1_min_strain_t.unsafe_ptr()
    var end_t_ptr = sec_def.runtime_c1_end_strain_t.unsafe_ptr()
    var us_t_ptr = sec_def.runtime_c1_unload_slope_t.unsafe_ptr()

    var acc_n = SIMD[DType.float64, width](0.0)
    var acc_my = SIMD[DType.float64, width](0.0)
    var acc_mz_neg = SIMD[DType.float64, width](0.0)
    var acc_k11 = SIMD[DType.float64, width](0.0)
    var acc_k12 = SIMD[DType.float64, width](0.0)
    var acc_k13_neg = SIMD[DType.float64, width](0.0)
    var acc_k22 = SIMD[DType.float64, width](0.0)
    var acc_k23_neg = SIMD[DType.float64, width](0.0)
    var acc_k33 = SIMD[DType.float64, width](0.0)

    var i = 0
    while i < count:
        var fi = family_offset + i
        var slot = slot_start + i
        var y_vec = load_float64_ptr_contiguous_simd[width](y_ptr, fi)
        var z_vec = load_float64_ptr_contiguous_simd[width](z_ptr, fi)
        var area_vec = load_float64_ptr_contiguous_simd[width](area_ptr, fi)
        var strain = eps0_vec + z_vec * ky_vec - y_vec * kz_vec

        var step = _concrete01_eval_simd_step[width](
            strain,
            load_float64_ptr_contiguous_simd[width](eps_c_ptr, slot),
            load_float64_ptr_contiguous_simd[width](sig_c_ptr, slot),
            load_float64_ptr_contiguous_simd[width](tan_c_ptr, slot),
            load_float64_ptr_contiguous_simd[width](min_c_ptr, slot),
            load_float64_ptr_contiguous_simd[width](end_c_ptr, slot),
            load_float64_ptr_contiguous_simd[width](us_c_ptr, slot),
            fpc,
            epsc0,
            fpcu,
            epscu,
            ec0,
            softening_tangent,
        )
        store_float64_ptr_contiguous_simd[width](eps_t_ptr, slot, strain)
        store_float64_ptr_contiguous_simd[width](sig_t_ptr, slot, step[0])
        store_float64_ptr_contiguous_simd[width](tan_t_ptr, slot, step[1])
        store_float64_ptr_contiguous_simd[width](min_t_ptr, slot, step[2])
        store_float64_ptr_contiguous_simd[width](end_t_ptr, slot, step[3])
        store_float64_ptr_contiguous_simd[width](us_t_ptr, slot, step[4])

        var fs = step[0] * area_vec
        var ks = step[1] * area_vec
        var ks_z = ks * z_vec
        var ks_y = ks * y_vec
        acc_n += fs
        acc_my += fs * z_vec
        acc_mz_neg += fs * y_vec
        acc_k11 += ks
        acc_k12 += ks_z
        acc_k13_neg += ks_y
        acc_k22 += ks_z * z_vec
        acc_k23_neg += ks_z * y_vec
        acc_k33 += ks_y * y_vec
        i += width

    axial_force += acc_n.reduce_add()
    moment_y += acc_my.reduce_add()
    moment_z -= acc_mz_neg.reduce_add()
    k11 += acc_k11.reduce_add()
    k12 += acc_k12.reduce_add()
    k13 -= acc_k13_neg.reduce_add()
    k22 += acc_k22.reduce_add()
    k23 -= acc_k23_neg.reduce_add()
    k33 += acc_k33.reduce_add()


@always_inline
fn _fiber_section3d_runtime_apply_steel01_group_simd[
    width: Int
](
    mut sec_def: FiberSection3dDef,
    family_offset: Int,
    slot_start: Int,
    count: Int,
    mat_def: UniMaterialDef,
    eps0: Float64,
    kappa_y: Float64,
    kappa_z: Float64,
    mut axial_force: Float64,
    mut moment_y: Float64,
    mut moment_z: Float64,
    mut k11: Float64,
    mut k12: Float64,
    mut k13: Float64,
    mut k22: Float64,
    mut k23: Float64,
    mut k33: Float64,
):
    if count <= 0:
        return

    var fy_s = mat_def.p0
    var e0_s = mat_def.p1
    var b_s = mat_def.p2
    if b_s >= 1.0:
        abort("Steel01 b must be < 1")
    var h_s = (b_s * e0_s) / (1.0 - b_s)
    var inv_e0_plus_h_s = 1.0 / (e0_s + h_s)
    var e_ep_s = (e0_s * h_s) * inv_e0_plus_h_s

    var fy = SIMD[DType.float64, width](fy_s)
    var e0 = SIMD[DType.float64, width](e0_s)
    var h = SIMD[DType.float64, width](h_s)
    var inv_e0_plus_h = SIMD[DType.float64, width](inv_e0_plus_h_s)
    var e_ep = SIMD[DType.float64, width](e_ep_s)

    var eps0_vec = SIMD[DType.float64, width](eps0)
    var ky_vec = SIMD[DType.float64, width](kappa_y)
    var kz_vec = SIMD[DType.float64, width](kappa_z)

    var y_ptr = sec_def.steel01_y_rel.unsafe_ptr()
    var z_ptr = sec_def.steel01_z_rel.unsafe_ptr()
    var area_ptr = sec_def.steel01_area.unsafe_ptr()
    var eps_p_c_ptr = sec_def.runtime_s1_eps_p_c.unsafe_ptr()
    var alpha_c_ptr = sec_def.runtime_s1_alpha_c.unsafe_ptr()

    var eps_t_ptr = sec_def.runtime_s1_eps_t.unsafe_ptr()
    var sig_t_ptr = sec_def.runtime_s1_sig_t.unsafe_ptr()
    var tan_t_ptr = sec_def.runtime_s1_tangent_t.unsafe_ptr()
    var eps_p_t_ptr = sec_def.runtime_s1_eps_p_t.unsafe_ptr()
    var alpha_t_ptr = sec_def.runtime_s1_alpha_t.unsafe_ptr()

    var acc_n = SIMD[DType.float64, width](0.0)
    var acc_my = SIMD[DType.float64, width](0.0)
    var acc_mz_neg = SIMD[DType.float64, width](0.0)
    var acc_k11 = SIMD[DType.float64, width](0.0)
    var acc_k12 = SIMD[DType.float64, width](0.0)
    var acc_k13_neg = SIMD[DType.float64, width](0.0)
    var acc_k22 = SIMD[DType.float64, width](0.0)
    var acc_k23_neg = SIMD[DType.float64, width](0.0)
    var acc_k33 = SIMD[DType.float64, width](0.0)

    var i = 0
    while i < count:
        var fi = family_offset + i
        var slot = slot_start + i
        var y_vec = load_float64_ptr_contiguous_simd[width](y_ptr, fi)
        var z_vec = load_float64_ptr_contiguous_simd[width](z_ptr, fi)
        var area_vec = load_float64_ptr_contiguous_simd[width](area_ptr, fi)
        var strain = eps0_vec + z_vec * ky_vec - y_vec * kz_vec

        var step = _steel01_eval_simd_step[width](
            strain,
            load_float64_ptr_contiguous_simd[width](eps_p_c_ptr, slot),
            load_float64_ptr_contiguous_simd[width](alpha_c_ptr, slot),
            fy,
            e0,
            h,
            inv_e0_plus_h,
            e_ep,
        )
        store_float64_ptr_contiguous_simd[width](eps_t_ptr, slot, strain)
        store_float64_ptr_contiguous_simd[width](sig_t_ptr, slot, step[0])
        store_float64_ptr_contiguous_simd[width](tan_t_ptr, slot, step[1])
        store_float64_ptr_contiguous_simd[width](eps_p_t_ptr, slot, step[2])
        store_float64_ptr_contiguous_simd[width](alpha_t_ptr, slot, step[3])

        var fs = step[0] * area_vec
        var ks = step[1] * area_vec
        var ks_z = ks * z_vec
        var ks_y = ks * y_vec
        acc_n += fs
        acc_my += fs * z_vec
        acc_mz_neg += fs * y_vec
        acc_k11 += ks
        acc_k12 += ks_z
        acc_k13_neg += ks_y
        acc_k22 += ks_z * z_vec
        acc_k23_neg += ks_z * y_vec
        acc_k33 += ks_y * y_vec
        i += width

    axial_force += acc_n.reduce_add()
    moment_y += acc_my.reduce_add()
    moment_z -= acc_mz_neg.reduce_add()
    k11 += acc_k11.reduce_add()
    k12 += acc_k12.reduce_add()
    k13 -= acc_k13_neg.reduce_add()
    k22 += acc_k22.reduce_add()
    k23 -= acc_k23_neg.reduce_add()
    k33 += acc_k33.reduce_add()


@always_inline
fn _fiber_section3d_commit_steel02_range_simd[
    width: Int
](mut sec_def: FiberSection3dDef, slot_start: Int, count: Int):
    if count <= 0:
        return
    _fiber_copy_float64_range_simd[width](
        sec_def.runtime_s2_eps_c, slot_start, sec_def.runtime_s2_eps_t, slot_start, count
    )
    _fiber_copy_float64_range_simd[width](
        sec_def.runtime_s2_sig_c, slot_start, sec_def.runtime_s2_sig_t, slot_start, count
    )
    _fiber_copy_float64_range_simd[width](
        sec_def.runtime_s2_tangent_c,
        slot_start,
        sec_def.runtime_s2_tangent_t,
        slot_start,
        count,
    )
    _fiber_copy_float64_range_simd[width](
        sec_def.runtime_s2_epsmin_c,
        slot_start,
        sec_def.runtime_s2_epsmin_t,
        slot_start,
        count,
    )
    _fiber_copy_float64_range_simd[width](
        sec_def.runtime_s2_epsmax_c,
        slot_start,
        sec_def.runtime_s2_epsmax_t,
        slot_start,
        count,
    )
    _fiber_copy_float64_range_simd[width](
        sec_def.runtime_s2_epspl_c,
        slot_start,
        sec_def.runtime_s2_epspl_t,
        slot_start,
        count,
    )
    _fiber_copy_float64_range_simd[width](
        sec_def.runtime_s2_epss0_c,
        slot_start,
        sec_def.runtime_s2_epss0_t,
        slot_start,
        count,
    )
    _fiber_copy_float64_range_simd[width](
        sec_def.runtime_s2_sigs0_c,
        slot_start,
        sec_def.runtime_s2_sigs0_t,
        slot_start,
        count,
    )
    _fiber_copy_float64_range_simd[width](
        sec_def.runtime_s2_epsr_c, slot_start, sec_def.runtime_s2_epsr_t, slot_start, count
    )
    _fiber_copy_float64_range_simd[width](
        sec_def.runtime_s2_sigr_c, slot_start, sec_def.runtime_s2_sigr_t, slot_start, count
    )
    _fiber_copy_int32_range_simd[width](
        sec_def.runtime_s2_kon_c, slot_start, sec_def.runtime_s2_kon_t, slot_start, count
    )


@always_inline
fn _fiber_section3d_revert_steel02_range_simd[
    width: Int
](mut sec_def: FiberSection3dDef, slot_start: Int, count: Int):
    if count <= 0:
        return
    _fiber_copy_float64_range_simd[width](
        sec_def.runtime_s2_eps_t, slot_start, sec_def.runtime_s2_eps_c, slot_start, count
    )
    _fiber_copy_float64_range_simd[width](
        sec_def.runtime_s2_sig_t, slot_start, sec_def.runtime_s2_sig_c, slot_start, count
    )
    _fiber_copy_float64_range_simd[width](
        sec_def.runtime_s2_tangent_t,
        slot_start,
        sec_def.runtime_s2_tangent_c,
        slot_start,
        count,
    )
    _fiber_copy_float64_range_simd[width](
        sec_def.runtime_s2_epsmin_t,
        slot_start,
        sec_def.runtime_s2_epsmin_c,
        slot_start,
        count,
    )
    _fiber_copy_float64_range_simd[width](
        sec_def.runtime_s2_epsmax_t,
        slot_start,
        sec_def.runtime_s2_epsmax_c,
        slot_start,
        count,
    )
    _fiber_copy_float64_range_simd[width](
        sec_def.runtime_s2_epspl_t,
        slot_start,
        sec_def.runtime_s2_epspl_c,
        slot_start,
        count,
    )
    _fiber_copy_float64_range_simd[width](
        sec_def.runtime_s2_epss0_t,
        slot_start,
        sec_def.runtime_s2_epss0_c,
        slot_start,
        count,
    )
    _fiber_copy_float64_range_simd[width](
        sec_def.runtime_s2_sigs0_t,
        slot_start,
        sec_def.runtime_s2_sigs0_c,
        slot_start,
        count,
    )
    _fiber_copy_float64_range_simd[width](
        sec_def.runtime_s2_epsr_t, slot_start, sec_def.runtime_s2_epsr_c, slot_start, count
    )
    _fiber_copy_float64_range_simd[width](
        sec_def.runtime_s2_sigr_t, slot_start, sec_def.runtime_s2_sigr_c, slot_start, count
    )
    _fiber_copy_int32_range_simd[width](
        sec_def.runtime_s2_kon_t, slot_start, sec_def.runtime_s2_kon_c, slot_start, count
    )


@always_inline
fn _fiber_section3d_commit_concrete02_range_simd[
    width: Int
](mut sec_def: FiberSection3dDef, slot_start: Int, count: Int):
    if count <= 0:
        return
    _fiber_copy_float64_range_simd[width](
        sec_def.runtime_c2_eps_c, slot_start, sec_def.runtime_c2_eps_t, slot_start, count
    )
    _fiber_copy_float64_range_simd[width](
        sec_def.runtime_c2_sig_c, slot_start, sec_def.runtime_c2_sig_t, slot_start, count
    )
    _fiber_copy_float64_range_simd[width](
        sec_def.runtime_c2_tangent_c,
        slot_start,
        sec_def.runtime_c2_tangent_t,
        slot_start,
        count,
    )
    _fiber_copy_float64_range_simd[width](
        sec_def.runtime_c2_ecmin_c,
        slot_start,
        sec_def.runtime_c2_ecmin_t,
        slot_start,
        count,
    )
    _fiber_copy_float64_range_simd[width](
        sec_def.runtime_c2_dept_c, slot_start, sec_def.runtime_c2_dept_t, slot_start, count
    )


@always_inline
fn _fiber_section3d_revert_concrete02_range_simd[
    width: Int
](mut sec_def: FiberSection3dDef, slot_start: Int, count: Int):
    if count <= 0:
        return
    _fiber_copy_float64_range_simd[width](
        sec_def.runtime_c2_eps_t, slot_start, sec_def.runtime_c2_eps_c, slot_start, count
    )
    _fiber_copy_float64_range_simd[width](
        sec_def.runtime_c2_sig_t, slot_start, sec_def.runtime_c2_sig_c, slot_start, count
    )
    _fiber_copy_float64_range_simd[width](
        sec_def.runtime_c2_tangent_t,
        slot_start,
        sec_def.runtime_c2_tangent_c,
        slot_start,
        count,
    )
    _fiber_copy_float64_range_simd[width](
        sec_def.runtime_c2_ecmin_t,
        slot_start,
        sec_def.runtime_c2_ecmin_c,
        slot_start,
        count,
    )
    _fiber_copy_float64_range_simd[width](
        sec_def.runtime_c2_dept_t, slot_start, sec_def.runtime_c2_dept_c, slot_start, count
    )


@always_inline
fn _fiber_section3d_commit_concrete01_range_simd[
    width: Int
](mut sec_def: FiberSection3dDef, slot_start: Int, count: Int):
    if count <= 0:
        return
    _fiber_copy_float64_range_simd[width](
        sec_def.runtime_c1_eps_c, slot_start, sec_def.runtime_c1_eps_t, slot_start, count
    )
    _fiber_copy_float64_range_simd[width](
        sec_def.runtime_c1_sig_c, slot_start, sec_def.runtime_c1_sig_t, slot_start, count
    )
    _fiber_copy_float64_range_simd[width](
        sec_def.runtime_c1_tangent_c,
        slot_start,
        sec_def.runtime_c1_tangent_t,
        slot_start,
        count,
    )
    _fiber_copy_float64_range_simd[width](
        sec_def.runtime_c1_min_strain_c,
        slot_start,
        sec_def.runtime_c1_min_strain_t,
        slot_start,
        count,
    )
    _fiber_copy_float64_range_simd[width](
        sec_def.runtime_c1_end_strain_c,
        slot_start,
        sec_def.runtime_c1_end_strain_t,
        slot_start,
        count,
    )
    _fiber_copy_float64_range_simd[width](
        sec_def.runtime_c1_unload_slope_c,
        slot_start,
        sec_def.runtime_c1_unload_slope_t,
        slot_start,
        count,
    )


@always_inline
fn _fiber_section3d_revert_concrete01_range_simd[
    width: Int
](mut sec_def: FiberSection3dDef, slot_start: Int, count: Int):
    if count <= 0:
        return
    _fiber_copy_float64_range_simd[width](
        sec_def.runtime_c1_eps_t, slot_start, sec_def.runtime_c1_eps_c, slot_start, count
    )
    _fiber_copy_float64_range_simd[width](
        sec_def.runtime_c1_sig_t, slot_start, sec_def.runtime_c1_sig_c, slot_start, count
    )
    _fiber_copy_float64_range_simd[width](
        sec_def.runtime_c1_tangent_t,
        slot_start,
        sec_def.runtime_c1_tangent_c,
        slot_start,
        count,
    )
    _fiber_copy_float64_range_simd[width](
        sec_def.runtime_c1_min_strain_t,
        slot_start,
        sec_def.runtime_c1_min_strain_c,
        slot_start,
        count,
    )
    _fiber_copy_float64_range_simd[width](
        sec_def.runtime_c1_end_strain_t,
        slot_start,
        sec_def.runtime_c1_end_strain_c,
        slot_start,
        count,
    )
    _fiber_copy_float64_range_simd[width](
        sec_def.runtime_c1_unload_slope_t,
        slot_start,
        sec_def.runtime_c1_unload_slope_c,
        slot_start,
        count,
    )


@always_inline
fn _fiber_section3d_commit_steel01_range_simd[
    width: Int
](mut sec_def: FiberSection3dDef, slot_start: Int, count: Int):
    if count <= 0:
        return
    _fiber_copy_float64_range_simd[width](
        sec_def.runtime_s1_eps_c, slot_start, sec_def.runtime_s1_eps_t, slot_start, count
    )
    _fiber_copy_float64_range_simd[width](
        sec_def.runtime_s1_sig_c, slot_start, sec_def.runtime_s1_sig_t, slot_start, count
    )
    _fiber_copy_float64_range_simd[width](
        sec_def.runtime_s1_tangent_c,
        slot_start,
        sec_def.runtime_s1_tangent_t,
        slot_start,
        count,
    )
    _fiber_copy_float64_range_simd[width](
        sec_def.runtime_s1_eps_p_c,
        slot_start,
        sec_def.runtime_s1_eps_p_t,
        slot_start,
        count,
    )
    _fiber_copy_float64_range_simd[width](
        sec_def.runtime_s1_alpha_c,
        slot_start,
        sec_def.runtime_s1_alpha_t,
        slot_start,
        count,
    )


@always_inline
fn _fiber_section3d_revert_steel01_range_simd[
    width: Int
](mut sec_def: FiberSection3dDef, slot_start: Int, count: Int):
    if count <= 0:
        return
    _fiber_copy_float64_range_simd[width](
        sec_def.runtime_s1_eps_t, slot_start, sec_def.runtime_s1_eps_c, slot_start, count
    )
    _fiber_copy_float64_range_simd[width](
        sec_def.runtime_s1_sig_t, slot_start, sec_def.runtime_s1_sig_c, slot_start, count
    )
    _fiber_copy_float64_range_simd[width](
        sec_def.runtime_s1_tangent_t,
        slot_start,
        sec_def.runtime_s1_tangent_c,
        slot_start,
        count,
    )
    _fiber_copy_float64_range_simd[width](
        sec_def.runtime_s1_eps_p_t,
        slot_start,
        sec_def.runtime_s1_eps_p_c,
        slot_start,
        count,
    )
    _fiber_copy_float64_range_simd[width](
        sec_def.runtime_s1_alpha_t,
        slot_start,
        sec_def.runtime_s1_alpha_c,
        slot_start,
        count,
    )


fn _append_rect_patch_cells(
    patch: PythonObject,
    uniaxial_def_by_id: List[Int],
    mut fibers: List[FiberCell],
    mut area_sum: Float64,
    mut qy_sum: Float64,
    mut qz_sum: Float64,
) raises:
    var mat_id = Int(patch["material"])
    var def_index = _resolve_uniaxial_def_index(mat_id, uniaxial_def_by_id)
    var ny = Int(patch["num_subdiv_y"])
    var nz = Int(patch["num_subdiv_z"])
    if ny <= 0 or nz <= 0:
        abort("FiberSection3d patch rect requires num_subdiv_y and num_subdiv_z > 0")

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
        abort("FiberSection3d patch rect must have non-zero side lengths")

    var dy = dy_total / Float64(ny)
    var dz = dz_total / Float64(nz)
    var area = dy * dz

    for iy in range(ny):
        var y = y_min + (Float64(iy) + 0.5) * dy
        for iz in range(nz):
            var z = z_min + (Float64(iz) + 0.5) * dz
            fibers.append(FiberCell(y, z, area, def_index))
            area_sum += area
            qy_sum += area * z
            qz_sum += area * y


fn _append_quadr_patch_cells(
    patch: PythonObject,
    uniaxial_def_by_id: List[Int],
    mut fibers: List[FiberCell],
    mut area_sum: Float64,
    mut qy_sum: Float64,
    mut qz_sum: Float64,
) raises:
    var mat_id = Int(patch["material"])
    var def_index = _resolve_uniaxial_def_index(mat_id, uniaxial_def_by_id)
    var ny = Int(patch["num_subdiv_y"])
    var nz = Int(patch["num_subdiv_z"])
    if ny <= 0 or nz <= 0:
        abort("FiberSection3d patch quadr requires num_subdiv_y and num_subdiv_z > 0")

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
        for iz in range(nz):
            var v0 = Float64(iz) / Float64(nz)
            var v1 = Float64(iz + 1) / Float64(nz)

            var y00 = (
                (1.0 - u0) * (1.0 - v0) * y_i
                + u0 * (1.0 - v0) * y_j
                + u0 * v0 * y_k
                + (1.0 - u0) * v0 * y_l
            )
            var z00 = (
                (1.0 - u0) * (1.0 - v0) * z_i
                + u0 * (1.0 - v0) * z_j
                + u0 * v0 * z_k
                + (1.0 - u0) * v0 * z_l
            )
            var y10 = (
                (1.0 - u1) * (1.0 - v0) * y_i
                + u1 * (1.0 - v0) * y_j
                + u1 * v0 * y_k
                + (1.0 - u1) * v0 * y_l
            )
            var z10 = (
                (1.0 - u1) * (1.0 - v0) * z_i
                + u1 * (1.0 - v0) * z_j
                + u1 * v0 * z_k
                + (1.0 - u1) * v0 * z_l
            )
            var y11 = (
                (1.0 - u1) * (1.0 - v1) * y_i
                + u1 * (1.0 - v1) * y_j
                + u1 * v1 * y_k
                + (1.0 - u1) * v1 * y_l
            )
            var z11 = (
                (1.0 - u1) * (1.0 - v1) * z_i
                + u1 * (1.0 - v1) * z_j
                + u1 * v1 * z_k
                + (1.0 - u1) * v1 * z_l
            )
            var y01 = (
                (1.0 - u0) * (1.0 - v1) * y_i
                + u0 * (1.0 - v1) * y_j
                + u0 * v1 * y_k
                + (1.0 - u0) * v1 * y_l
            )
            var z01 = (
                (1.0 - u0) * (1.0 - v1) * z_i
                + u0 * (1.0 - v1) * z_j
                + u0 * v1 * z_k
                + (1.0 - u0) * v1 * z_l
            )

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
                abort("FiberSection3d patch quadr generated zero-area cell")

            var centroid = quad_cell_centroid(y00, z00, y10, z10, y11, z11, y01, z01)
            var yc = centroid[0]
            var zc = centroid[1]
            fibers.append(FiberCell(yc, zc, area, def_index))
            area_sum += area
            qy_sum += area * zc
            qz_sum += area * yc


fn _append_straight_layer_cells(
    layer: PythonObject,
    uniaxial_def_by_id: List[Int],
    mut fibers: List[FiberCell],
    mut area_sum: Float64,
    mut qy_sum: Float64,
    mut qz_sum: Float64,
) raises:
    var mat_id = Int(layer["material"])
    var def_index = _resolve_uniaxial_def_index(mat_id, uniaxial_def_by_id)
    var num_bars = Int(layer["num_bars"])
    if num_bars <= 0:
        abort("FiberSection3d layer straight requires num_bars > 0")
    var bar_area = Float64(layer["bar_area"])
    if bar_area <= 0.0:
        abort("FiberSection3d layer straight requires bar_area > 0")

    var y_start = Float64(layer["y_start"])
    var z_start = Float64(layer["z_start"])
    var y_end = Float64(layer["y_end"])
    var z_end = Float64(layer["z_end"])

    if num_bars == 1:
        var y = 0.5 * (y_start + y_end)
        var z = 0.5 * (z_start + z_end)
        fibers.append(FiberCell(y, z, bar_area, def_index))
        area_sum += bar_area
        qy_sum += bar_area * z
        qz_sum += bar_area * y
        return

    for i in range(num_bars):
        var t = Float64(i) / Float64(num_bars - 1)
        var y = y_start + (y_end - y_start) * t
        var z = z_start + (z_end - z_start) * t
        fibers.append(FiberCell(y, z, bar_area, def_index))
        area_sum += bar_area
        qy_sum += bar_area * z
        qz_sum += bar_area * y


fn _append_circ_patch_cells(
    patch: PythonObject,
    uniaxial_def_by_id: List[Int],
    mut fibers: List[FiberCell],
    mut area_sum: Float64,
    mut qy_sum: Float64,
    mut qz_sum: Float64,
) raises:
    var mat_id = Int(py=patch["material"])
    var n_circ = Int(py=patch["num_subdiv_circ"])
    var n_rad = Int(py=patch["num_subdiv_rad"])
    if n_circ <= 0 or n_rad <= 0:
        abort("FiberSection3d patch circ requires num_subdiv_circ and num_subdiv_rad > 0")
    var y_center = Float64(py=patch["y_center"])
    var z_center = Float64(py=patch["z_center"])
    var int_rad = Float64(py=patch["int_rad"])
    var ext_rad = Float64(py=patch["ext_rad"])
    var start_ang = Float64(py=patch.get("start_ang", 0.0))
    var end_ang = Float64(py=patch.get("end_ang", 360.0))
    if int_rad < 0.0 or ext_rad <= int_rad:
        abort("FiberSection3d patch circ requires 0 <= int_rad < ext_rad")
    var def_index = _resolve_uniaxial_def_index(mat_id, uniaxial_def_by_id)
    var pi = acos(-1.0)
    var init_rad = pi * start_ang / 180.0
    var final_rad = pi * end_ang / 180.0
    var delta_rad = (ext_rad - int_rad) / Float64(n_rad)
    var delta_theta = (final_rad - init_rad) / Float64(n_circ)
    if delta_theta == 0.0:
        abort("FiberSection3d patch circ requires non-zero angular span")
    for j in range(n_rad):
        var rad_j = int_rad + delta_rad * Float64(j)
        var rad_j1 = rad_j + delta_rad
        for i in range(n_circ):
            var theta_mid = init_rad + delta_theta * Float64(i) + delta_theta / 2.0
            var cell = circ_cell_area_and_centroid(
                rad_j, rad_j1, delta_theta, theta_mid, y_center, z_center
            )
            var cell_area = cell[0]
            var y = cell[1]
            var z = cell[2]
            if cell_area <= 0.0:
                abort("FiberSection3d patch circ generated non-positive area cell")
            fibers.append(FiberCell(y, z, cell_area, def_index))
            area_sum += cell_area
            qy_sum += cell_area * z
            qz_sum += cell_area * y


fn _append_circ_layer_cells(
    layer: PythonObject,
    uniaxial_def_by_id: List[Int],
    mut fibers: List[FiberCell],
    mut area_sum: Float64,
    mut qy_sum: Float64,
    mut qz_sum: Float64,
) raises:
    var mat_id = Int(py=layer["material"])
    var num_bars = Int(py=layer["num_bars"])
    if num_bars <= 0:
        abort("FiberSection3d layer circ requires num_bars > 0")
    var bar_area = Float64(py=layer["bar_area"])
    if bar_area <= 0.0:
        abort("FiberSection3d layer circ requires bar_area > 0")
    var y_center = Float64(py=layer["y_center"])
    var z_center = Float64(py=layer["z_center"])
    var radius = Float64(py=layer["radius"])
    var has_angles = ("start_ang" in layer) and ("end_ang" in layer)
    var start_ang = Float64(py=layer.get("start_ang", 0.0))
    var end_ang = Float64(py=layer.get("end_ang", 360.0))
    var def_index = _resolve_uniaxial_def_index(mat_id, uniaxial_def_by_id)
    var ang = circ_layer_effective_angles(num_bars, start_ang, end_ang, has_angles)
    var init_rad = ang[0]
    var dtheta = ang[1]
    for i in range(num_bars):
        var theta = init_rad + dtheta * Float64(i)
        var y = y_center + radius * cos(theta)
        var z = z_center + radius * sin(theta)
        fibers.append(FiberCell(y, z, bar_area, def_index))
        area_sum += bar_area
        qy_sum += bar_area * z
        qz_sum += bar_area * y


fn append_fiber_section3d_from_json(
    sec: PythonObject,
    uniaxial_def_by_id: List[Int],
    uniaxial_defs: List[UniMaterialDef],
    mut defs: List[FiberSection3dDef],
    mut fibers: List[FiberCell],
) raises:
    if String(sec["type"]) != "FiberSection3d":
        abort("append_fiber_section3d_from_json requires FiberSection3d")

    var params = sec["params"]
    var patches = params.get("patches", [])
    var layers = params.get("layers", [])
    if py_len(patches) == 0 and py_len(layers) == 0:
        abort("FiberSection3d requires at least one patch or layer")

    var fiber_offset = len(fibers)
    var area_sum = 0.0
    var qy_sum = 0.0
    var qz_sum = 0.0

    for i in range(py_len(patches)):
        var patch = patches[i]
        var patch_type = String(patch["type"])
        if patch_type == "quad":
            patch_type = "quadr"
        if patch_type == "rect":
            _append_rect_patch_cells(patch, uniaxial_def_by_id, fibers, area_sum, qy_sum, qz_sum)
        elif patch_type == "quadr":
            _append_quadr_patch_cells(patch, uniaxial_def_by_id, fibers, area_sum, qy_sum, qz_sum)
        elif patch_type == "circ":
            _append_circ_patch_cells(patch, uniaxial_def_by_id, fibers, area_sum, qy_sum, qz_sum)
        else:
            abort("unsupported FiberSection3d patch type: " + patch_type)

    for i in range(py_len(layers)):
        var layer = layers[i]
        var layer_type = String(layer["type"])
        if layer_type == "straight":
            _append_straight_layer_cells(
                layer, uniaxial_def_by_id, fibers, area_sum, qy_sum, qz_sum
            )
        elif layer_type == "circ":
            _append_circ_layer_cells(layer, uniaxial_def_by_id, fibers, area_sum, qy_sum, qz_sum)
        else:
            abort("unsupported FiberSection3d layer type: " + layer_type)

    var fiber_count = len(fibers) - fiber_offset
    if fiber_count <= 0:
        abort("FiberSection3d produced no fibers")
    if area_sum <= 0.0:
        abort("FiberSection3d total area must be > 0")
    var y_bar = qz_sum / area_sum
    var z_bar = qy_sum / area_sum

    defs.append(
        _build_fiber_section3d_def(fiber_offset, fiber_count, y_bar, z_bar, fibers, uniaxial_defs)
    )


fn _append_rect_patch_cells_input(
    patch: FiberPatchInput,
    uniaxial_def_by_id: List[Int],
    mut fibers: List[FiberCell],
    mut area_sum: Float64,
    mut qy_sum: Float64,
    mut qz_sum: Float64,
):
    var mat_id = patch.material
    var def_index = _resolve_uniaxial_def_index(mat_id, uniaxial_def_by_id)
    var ny = patch.num_subdiv_y
    var nz = patch.num_subdiv_z
    if ny <= 0 or nz <= 0:
        abort("FiberSection3d patch rect requires num_subdiv_y and num_subdiv_z > 0")

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
        abort("FiberSection3d patch rect must have non-zero side lengths")

    var dy = dy_total / Float64(ny)
    var dz = dz_total / Float64(nz)
    var area = dy * dz

    for iy in range(ny):
        var y = y_min + (Float64(iy) + 0.5) * dy
        for iz in range(nz):
            var z = z_min + (Float64(iz) + 0.5) * dz
            fibers.append(FiberCell(y, z, area, def_index))
            area_sum += area
            qy_sum += area * z
            qz_sum += area * y


fn _append_quadr_patch_cells_input(
    patch: FiberPatchInput,
    uniaxial_def_by_id: List[Int],
    mut fibers: List[FiberCell],
    mut area_sum: Float64,
    mut qy_sum: Float64,
    mut qz_sum: Float64,
):
    var mat_id = patch.material
    var def_index = _resolve_uniaxial_def_index(mat_id, uniaxial_def_by_id)
    var ny = patch.num_subdiv_y
    var nz = patch.num_subdiv_z
    if ny <= 0 or nz <= 0:
        abort("FiberSection3d patch quadr requires num_subdiv_y and num_subdiv_z > 0")

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
        for iz in range(nz):
            var v0 = Float64(iz) / Float64(nz)
            var v1 = Float64(iz + 1) / Float64(nz)

            var y00 = (
                (1.0 - u0) * (1.0 - v0) * y_i
                + u0 * (1.0 - v0) * y_j
                + u0 * v0 * y_k
                + (1.0 - u0) * v0 * y_l
            )
            var z00 = (
                (1.0 - u0) * (1.0 - v0) * z_i
                + u0 * (1.0 - v0) * z_j
                + u0 * v0 * z_k
                + (1.0 - u0) * v0 * z_l
            )
            var y10 = (
                (1.0 - u1) * (1.0 - v0) * y_i
                + u1 * (1.0 - v0) * y_j
                + u1 * v0 * y_k
                + (1.0 - u1) * v0 * y_l
            )
            var z10 = (
                (1.0 - u1) * (1.0 - v0) * z_i
                + u1 * (1.0 - v0) * z_j
                + u1 * v0 * z_k
                + (1.0 - u1) * v0 * z_l
            )
            var y11 = (
                (1.0 - u1) * (1.0 - v1) * y_i
                + u1 * (1.0 - v1) * y_j
                + u1 * v1 * y_k
                + (1.0 - u1) * v1 * y_l
            )
            var z11 = (
                (1.0 - u1) * (1.0 - v1) * z_i
                + u1 * (1.0 - v1) * z_j
                + u1 * v1 * z_k
                + (1.0 - u1) * v1 * z_l
            )
            var y01 = (
                (1.0 - u0) * (1.0 - v1) * y_i
                + u0 * (1.0 - v1) * y_j
                + u0 * v1 * y_k
                + (1.0 - u0) * v1 * y_l
            )
            var z01 = (
                (1.0 - u0) * (1.0 - v1) * z_i
                + u0 * (1.0 - v1) * z_j
                + u0 * v1 * z_k
                + (1.0 - u0) * v1 * z_l
            )

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
                abort("FiberSection3d patch quadr generated zero-area cell")

            var centroid = quad_cell_centroid(y00, z00, y10, z10, y11, z11, y01, z01)
            var yc = centroid[0]
            var zc = centroid[1]
            fibers.append(FiberCell(yc, zc, area, def_index))
            area_sum += area
            qy_sum += area * zc
            qz_sum += area * yc


fn _append_straight_layer_cells_input(
    layer: FiberLayerInput,
    uniaxial_def_by_id: List[Int],
    mut fibers: List[FiberCell],
    mut area_sum: Float64,
    mut qy_sum: Float64,
    mut qz_sum: Float64,
):
    var mat_id = layer.material
    var def_index = _resolve_uniaxial_def_index(mat_id, uniaxial_def_by_id)
    var num_bars = layer.num_bars
    if num_bars <= 0:
        abort("FiberSection3d layer straight requires num_bars > 0")
    var bar_area = layer.bar_area
    if bar_area <= 0.0:
        abort("FiberSection3d layer straight requires bar_area > 0")

    var y_start = layer.y_start
    var z_start = layer.z_start
    var y_end = layer.y_end
    var z_end = layer.z_end

    if num_bars == 1:
        var y = 0.5 * (y_start + y_end)
        var z = 0.5 * (z_start + z_end)
        fibers.append(FiberCell(y, z, bar_area, def_index))
        area_sum += bar_area
        qy_sum += bar_area * z
        qz_sum += bar_area * y
        return

    for i in range(num_bars):
        var t = Float64(i) / Float64(num_bars - 1)
        var y = y_start + (y_end - y_start) * t
        var z = z_start + (z_end - z_start) * t
        fibers.append(FiberCell(y, z, bar_area, def_index))
        area_sum += bar_area
        qy_sum += bar_area * z
        qz_sum += bar_area * y


fn _append_direct_fibers_input(
    section_fibers: List[FiberInput],
    fiber_offset: Int,
    fiber_count: Int,
    uniaxial_def_by_id: List[Int],
    mut fibers: List[FiberCell],
    mut area_sum: Float64,
    mut qy_sum: Float64,
    mut qz_sum: Float64,
):
    for i in range(fiber_count):
        var fiber = section_fibers[fiber_offset + i]
        if fiber.area <= 0.0:
            abort("FiberSection3d fiber area must be > 0")
        var def_index = _resolve_uniaxial_def_index(fiber.material, uniaxial_def_by_id)
        fibers.append(FiberCell(fiber.y, fiber.z, fiber.area, def_index))
        area_sum += fiber.area
        qy_sum += fiber.area * fiber.z
        qz_sum += fiber.area * fiber.y


fn _append_circ_patch_cells_input(
    patch: FiberPatchInput,
    uniaxial_def_by_id: List[Int],
    mut fibers: List[FiberCell],
    mut area_sum: Float64,
    mut qy_sum: Float64,
    mut qz_sum: Float64,
):
    var n_circ = patch.num_subdiv_circ
    var n_rad = patch.num_subdiv_rad
    if n_circ <= 0 or n_rad <= 0:
        abort("FiberSection3d patch circ requires num_subdiv_circ and num_subdiv_rad > 0")
    if patch.int_rad < 0.0 or patch.ext_rad <= patch.int_rad:
        abort("FiberSection3d patch circ requires 0 <= int_rad < ext_rad")
    var def_index = _resolve_uniaxial_def_index(patch.material, uniaxial_def_by_id)
    var pi = acos(-1.0)
    var init_rad = pi * patch.start_ang / 180.0
    var final_rad = pi * patch.end_ang / 180.0
    var delta_rad = (patch.ext_rad - patch.int_rad) / Float64(n_rad)
    var delta_theta = (final_rad - init_rad) / Float64(n_circ)
    if delta_theta == 0.0:
        abort("FiberSection3d patch circ requires non-zero angular span")
    for j in range(n_rad):
        var rad_j = patch.int_rad + delta_rad * Float64(j)
        var rad_j1 = rad_j + delta_rad
        for i in range(n_circ):
            var theta_mid = init_rad + delta_theta * Float64(i) + delta_theta / 2.0
            var cell = circ_cell_area_and_centroid(
                rad_j, rad_j1, delta_theta, theta_mid, patch.y_center, patch.z_center
            )
            var cell_area = cell[0]
            var y = cell[1]
            var z = cell[2]
            if cell_area <= 0.0:
                abort("FiberSection3d patch circ generated non-positive area cell")
            fibers.append(FiberCell(y, z, cell_area, def_index))
            area_sum += cell_area
            qy_sum += cell_area * z
            qz_sum += cell_area * y


fn _append_circ_layer_cells_input(
    layer: FiberLayerInput,
    uniaxial_def_by_id: List[Int],
    mut fibers: List[FiberCell],
    mut area_sum: Float64,
    mut qy_sum: Float64,
    mut qz_sum: Float64,
):
    if layer.num_bars <= 0:
        abort("FiberSection3d layer circ requires num_bars > 0")
    if layer.bar_area <= 0.0:
        abort("FiberSection3d layer circ requires bar_area > 0")
    var def_index = _resolve_uniaxial_def_index(layer.material, uniaxial_def_by_id)
    var ang = circ_layer_effective_angles(
        layer.num_bars, layer.start_ang, layer.end_ang, layer.has_angles
    )
    var init_rad = ang[0]
    var dtheta = ang[1]
    for i in range(layer.num_bars):
        var theta = init_rad + dtheta * Float64(i)
        var y = layer.y_center + layer.radius * cos(theta)
        var z = layer.z_center + layer.radius * sin(theta)
        fibers.append(FiberCell(y, z, layer.bar_area, def_index))
        area_sum += layer.bar_area
        qy_sum += layer.bar_area * z
        qz_sum += layer.bar_area * y


fn append_fiber_section3d_from_input(
    sec: SectionInput,
    fiber_patches: List[FiberPatchInput],
    fiber_layers: List[FiberLayerInput],
    section_fibers: List[FiberInput],
    uniaxial_def_by_id: List[Int],
    uniaxial_defs: List[UniMaterialDef],
    mut defs: List[FiberSection3dDef],
    mut fibers: List[FiberCell],
):
    if sec.type != "FiberSection3d":
        abort("append_fiber_section3d_from_input requires FiberSection3d")
    if sec.fiber_patch_count == 0 and sec.fiber_layer_count == 0 and sec.fiber_count == 0:
        abort("FiberSection3d requires at least one patch, layer, or fiber")

    var fiber_offset = len(fibers)
    var area_sum = 0.0
    var qy_sum = 0.0
    var qz_sum = 0.0

    for i in range(sec.fiber_patch_count):
        var patch = fiber_patches[sec.fiber_patch_offset + i]
        if patch.type == "rect":
            _append_rect_patch_cells_input(
                patch, uniaxial_def_by_id, fibers, area_sum, qy_sum, qz_sum
            )
        elif patch.type == "quadr":
            _append_quadr_patch_cells_input(
                patch, uniaxial_def_by_id, fibers, area_sum, qy_sum, qz_sum
            )
        elif patch.type == "circ":
            _append_circ_patch_cells_input(
                patch, uniaxial_def_by_id, fibers, area_sum, qy_sum, qz_sum
            )
        else:
            abort("unsupported FiberSection3d patch type: " + patch.type)

    for i in range(sec.fiber_layer_count):
        var layer = fiber_layers[sec.fiber_layer_offset + i]
        if layer.type == "straight":
            _append_straight_layer_cells_input(
                layer, uniaxial_def_by_id, fibers, area_sum, qy_sum, qz_sum
            )
        elif layer.type == "circ":
            _append_circ_layer_cells_input(
                layer, uniaxial_def_by_id, fibers, area_sum, qy_sum, qz_sum
            )
        else:
            abort("unsupported FiberSection3d layer type: " + layer.type)

    _append_direct_fibers_input(
        section_fibers,
        sec.fiber_offset,
        sec.fiber_count,
        uniaxial_def_by_id,
        fibers,
        area_sum,
        qy_sum,
        qz_sum,
    )

    var fiber_count = len(fibers) - fiber_offset
    if fiber_count <= 0:
        abort("FiberSection3d produced no fibers")
    if area_sum <= 0.0:
        abort("FiberSection3d total area must be > 0")
    var y_bar = qz_sum / area_sum
    var z_bar = qy_sum / area_sum

    defs.append(
        _build_fiber_section3d_def(fiber_offset, fiber_count, y_bar, z_bar, fibers, uniaxial_defs)
    )


fn fiber_section3d_init_states(
    mut defs: List[FiberSection3dDef],
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
        if sec_def.fiber_offset < 0 or sec_def.fiber_offset + sec_def.fiber_count > len(fibers):
            abort("FiberSection3d fiber data out of range")
        section_uniaxial_offsets[s] = fiber_section3d_runtime_alloc_instances(sec_def, 1)
        section_uniaxial_counts[s] = sec_def.fiber_count
        for i in range(len(sec_def.all_def_indices)):
            var def_index = sec_def.all_def_indices[i]
            if def_index < 0 or def_index >= len(uniaxial_defs):
                abort("FiberSection3d fiber material definition out of range")
            uniaxial_state_defs.append(def_index)
        if sec_def.has_nonelastic:
            used_nonelastic = True

    return used_nonelastic


@always_inline
fn _fiber_section3d_runtime_instance_index(
    sec_def: FiberSection3dDef, section_state_offset: Int, section_state_count: Int
) -> Int:
    if section_state_count != sec_def.fiber_count:
        abort("FiberSection3d section state count mismatch")
    var local_offset = section_state_offset - sec_def.runtime_state_offset_base
    if local_offset < 0 or local_offset + section_state_count > sec_def.runtime_state_count:
        abort("FiberSection3d section states out of range")
    return local_offset // sec_def.fiber_count


fn fiber_section3d_set_trial_from_offset(
    mut sec_def: FiberSection3dDef,
    mut uniaxial_states: List[UniMaterialState],
    section_state_offset: Int,
    section_state_count: Int,
    eps0: Float64,
    kappa_y: Float64,
    kappa_z: Float64,
) -> FiberSection3dResponse:
    var inst = _fiber_section3d_runtime_instance_index(
        sec_def, section_state_offset, section_state_count
    )
    return fiber_section3d_set_trial_instance(sec_def, inst, eps0, kappa_y, kappa_z)


fn fiber_section3d_set_trial_instance(
    mut sec_def: FiberSection3dDef,
    inst: Int,
    eps0: Float64,
    kappa_y: Float64,
    kappa_z: Float64,
) -> FiberSection3dResponse:
    var axial_force = 0.0
    var moment_y = 0.0
    var moment_z = 0.0
    var k11 = 0.0
    var k12 = 0.0
    var k13 = 0.0
    var k22 = 0.0
    var k23 = 0.0
    var k33 = 0.0

    if sec_def.steel02_only_single_def:
        var slot_start = inst * sec_def.steel02_instance_stride
        var mat_def = sec_def.steel02_single_mat_def
        _fiber_section3d_runtime_apply_steel02_group_simd[FLOAT64_SIMD_WIDTH](
            sec_def,
            0,
            slot_start,
            sec_def.steel02_padded_count,
            mat_def,
            eps0,
            kappa_y,
            kappa_z,
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

    if sec_def.elastic_count > 0:
        axial_force = (
            sec_def.elastic_k11 * eps0
            + sec_def.elastic_f12 * kappa_y
            + sec_def.elastic_f13 * kappa_z
        )
        moment_y = (
            sec_def.elastic_f12 * eps0
            + sec_def.elastic_k22 * kappa_y
            + sec_def.elastic_k23 * kappa_z
        )
        moment_z = (
            sec_def.elastic_f13 * eps0
            + sec_def.elastic_k23 * kappa_y
            + sec_def.elastic_k33 * kappa_z
        )
        k11 = sec_def.elastic_k11
        k12 = sec_def.elastic_k12
        k13 = sec_def.elastic_k13
        k22 = sec_def.elastic_k22
        k23 = sec_def.elastic_k23
        k33 = sec_def.elastic_k33

    if sec_def.steel02_padded_count > 0:
        var slot_start = inst * sec_def.steel02_instance_stride
        if sec_def.steel02_single_definition:
            var mat_def = sec_def.steel02_single_mat_def
            _fiber_section3d_runtime_apply_steel02_group_simd[FLOAT64_SIMD_WIDTH](
                sec_def,
                0,
                slot_start,
                sec_def.steel02_padded_count,
                mat_def,
                eps0,
                kappa_y,
                kappa_z,
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
        else:
            for g in range(len(sec_def.steel02_group_offsets)):
                var g_off = sec_def.steel02_group_offsets[g]
                var mat_def = sec_def.steel02_group_mat_defs[g]
                _fiber_section3d_runtime_apply_steel02_group_simd[FLOAT64_SIMD_WIDTH](
                    sec_def,
                    g_off,
                    slot_start + g_off,
                    sec_def.steel02_group_padded_counts[g],
                    mat_def,
                    eps0,
                    kappa_y,
                    kappa_z,
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

    if sec_def.concrete02_padded_count > 0:
        var slot_start = inst * sec_def.concrete02_instance_stride
        if sec_def.concrete02_single_definition:
            var mat_def = sec_def.concrete02_single_mat_def
            _fiber_section3d_runtime_apply_concrete02_group_simd[FLOAT64_SIMD_WIDTH](
                sec_def,
                0,
                slot_start,
                sec_def.concrete02_padded_count,
                mat_def,
                eps0,
                kappa_y,
                kappa_z,
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
        else:
            for g in range(len(sec_def.concrete02_group_offsets)):
                var g_off = sec_def.concrete02_group_offsets[g]
                var mat_def = sec_def.concrete02_group_mat_defs[g]
                _fiber_section3d_runtime_apply_concrete02_group_simd[FLOAT64_SIMD_WIDTH](
                    sec_def,
                    g_off,
                    slot_start + g_off,
                    sec_def.concrete02_group_padded_counts[g],
                    mat_def,
                    eps0,
                    kappa_y,
                    kappa_z,
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

    if sec_def.concrete01_padded_count > 0:
        var slot_start = inst * sec_def.concrete01_instance_stride
        if sec_def.concrete01_single_definition:
            var mat_def = sec_def.concrete01_single_mat_def
            _fiber_section3d_runtime_apply_concrete01_group_simd[FLOAT64_SIMD_WIDTH](
                sec_def,
                0,
                slot_start,
                sec_def.concrete01_padded_count,
                mat_def,
                eps0,
                kappa_y,
                kappa_z,
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
        else:
            for g in range(len(sec_def.concrete01_group_offsets)):
                var g_off = sec_def.concrete01_group_offsets[g]
                var mat_def = sec_def.concrete01_group_mat_defs[g]
                _fiber_section3d_runtime_apply_concrete01_group_simd[FLOAT64_SIMD_WIDTH](
                    sec_def,
                    g_off,
                    slot_start + g_off,
                    sec_def.concrete01_group_padded_counts[g],
                    mat_def,
                    eps0,
                    kappa_y,
                    kappa_z,
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

    if sec_def.steel01_padded_count > 0:
        var slot_start = inst * sec_def.steel01_instance_stride
        if sec_def.steel01_single_definition:
            var mat_def = sec_def.steel01_single_mat_def
            _fiber_section3d_runtime_apply_steel01_group_simd[FLOAT64_SIMD_WIDTH](
                sec_def,
                0,
                slot_start,
                sec_def.steel01_padded_count,
                mat_def,
                eps0,
                kappa_y,
                kappa_z,
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
        else:
            for g in range(len(sec_def.steel01_group_offsets)):
                var g_off = sec_def.steel01_group_offsets[g]
                var mat_def = sec_def.steel01_group_mat_defs[g]
                _fiber_section3d_runtime_apply_steel01_group_simd[FLOAT64_SIMD_WIDTH](
                    sec_def,
                    g_off,
                    slot_start + g_off,
                    sec_def.steel01_group_padded_counts[g],
                    mat_def,
                    eps0,
                    kappa_y,
                    kappa_z,
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

    if sec_def.other_count > 0:
        var other_base = inst * sec_def.other_count
        for i in range(sec_def.other_count):
            var y_rel = sec_def.other_y_rel[i]
            var z_rel = sec_def.other_z_rel[i]
            var area = sec_def.other_area[i]
            var mat_def = sec_def.other_mat_defs[i]
            ref state = sec_def.runtime_other_states[other_base + i]
            uniaxial_set_trial_strain(mat_def, state, eps0 + z_rel * kappa_y - y_rel * kappa_z)
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


fn fiber_section3d_set_trial(
    section_index: Int,
    mut defs: List[FiberSection3dDef],
    section_uniaxial_offsets: List[Int],
    section_uniaxial_counts: List[Int],
    mut uniaxial_states: List[UniMaterialState],
    eps0: Float64,
    kappa_y: Float64,
    kappa_z: Float64,
) -> FiberSection3dResponse:
    if section_index < 0 or section_index >= len(defs):
        abort("FiberSection3d section index out of range")
    if section_index >= len(section_uniaxial_offsets) or section_index >= len(
        section_uniaxial_counts
    ):
        abort("FiberSection3d section state mapping missing")

    ref sec_def = defs[section_index]
    var offset = section_uniaxial_offsets[section_index]
    var count = section_uniaxial_counts[section_index]
    if count != sec_def.fiber_count:
        abort("FiberSection3d section state count mismatch")

    return fiber_section3d_set_trial_from_offset(
        sec_def,
        uniaxial_states,
        offset,
        count,
        eps0,
        kappa_y,
        kappa_z,
    )


fn fiber_section3d_commit_from_offset(
    mut sec_def: FiberSection3dDef,
    mut uniaxial_states: List[UniMaterialState],
    section_state_offset: Int,
    section_state_count: Int,
):
    var inst = _fiber_section3d_runtime_instance_index(
        sec_def, section_state_offset, section_state_count
    )
    if sec_def.steel02_padded_count > 0:
        _fiber_section3d_commit_steel02_range_simd[FLOAT64_SIMD_WIDTH](
            sec_def,
            inst * sec_def.steel02_instance_stride,
            sec_def.steel02_padded_count,
        )
    if sec_def.concrete02_padded_count > 0:
        _fiber_section3d_commit_concrete02_range_simd[FLOAT64_SIMD_WIDTH](
            sec_def,
            inst * sec_def.concrete02_instance_stride,
            sec_def.concrete02_padded_count,
        )
    if sec_def.concrete01_padded_count > 0:
        _fiber_section3d_commit_concrete01_range_simd[FLOAT64_SIMD_WIDTH](
            sec_def,
            inst * sec_def.concrete01_instance_stride,
            sec_def.concrete01_padded_count,
        )
    if sec_def.steel01_padded_count > 0:
        _fiber_section3d_commit_steel01_range_simd[FLOAT64_SIMD_WIDTH](
            sec_def,
            inst * sec_def.steel01_instance_stride,
            sec_def.steel01_padded_count,
        )
    if sec_def.other_count > 0:
        var other_base = inst * sec_def.other_count
        for i in range(sec_def.other_count):
            uniaxial_commit(sec_def.runtime_other_states[other_base + i])


fn fiber_section3d_revert_trial_from_offset(
    mut sec_def: FiberSection3dDef,
    mut uniaxial_states: List[UniMaterialState],
    section_state_offset: Int,
    section_state_count: Int,
):
    var inst = _fiber_section3d_runtime_instance_index(
        sec_def, section_state_offset, section_state_count
    )
    if sec_def.steel02_padded_count > 0:
        _fiber_section3d_revert_steel02_range_simd[FLOAT64_SIMD_WIDTH](
            sec_def,
            inst * sec_def.steel02_instance_stride,
            sec_def.steel02_padded_count,
        )
    if sec_def.concrete02_padded_count > 0:
        _fiber_section3d_revert_concrete02_range_simd[FLOAT64_SIMD_WIDTH](
            sec_def,
            inst * sec_def.concrete02_instance_stride,
            sec_def.concrete02_padded_count,
        )
    if sec_def.concrete01_padded_count > 0:
        _fiber_section3d_revert_concrete01_range_simd[FLOAT64_SIMD_WIDTH](
            sec_def,
            inst * sec_def.concrete01_instance_stride,
            sec_def.concrete01_padded_count,
        )
    if sec_def.steel01_padded_count > 0:
        _fiber_section3d_revert_steel01_range_simd[FLOAT64_SIMD_WIDTH](
            sec_def,
            inst * sec_def.steel01_instance_stride,
            sec_def.steel01_padded_count,
        )
    if sec_def.other_count > 0:
        var other_base = inst * sec_def.other_count
        for i in range(sec_def.other_count):
            uniaxial_revert_trial(sec_def.runtime_other_states[other_base + i])


fn fiber_section3d_commit(
    section_index: Int,
    mut defs: List[FiberSection3dDef],
    section_uniaxial_offsets: List[Int],
    section_uniaxial_counts: List[Int],
    mut uniaxial_states: List[UniMaterialState],
):
    if section_index < 0 or section_index >= len(section_uniaxial_offsets):
        abort("FiberSection3d section index out of range")
    if section_index >= len(section_uniaxial_counts):
        abort("FiberSection3d section state mapping missing")

    var offset = section_uniaxial_offsets[section_index]
    var count = section_uniaxial_counts[section_index]
    ref sec_def = defs[section_index]
    fiber_section3d_commit_from_offset(sec_def, uniaxial_states, offset, count)


fn fiber_section3d_revert_trial(
    section_index: Int,
    mut defs: List[FiberSection3dDef],
    section_uniaxial_offsets: List[Int],
    section_uniaxial_counts: List[Int],
    mut uniaxial_states: List[UniMaterialState],
):
    if section_index < 0 or section_index >= len(section_uniaxial_offsets):
        abort("FiberSection3d section index out of range")
    if section_index >= len(section_uniaxial_counts):
        abort("FiberSection3d section state mapping missing")

    var offset = section_uniaxial_offsets[section_index]
    var count = section_uniaxial_counts[section_index]
    ref sec_def = defs[section_index]
    fiber_section3d_revert_trial_from_offset(sec_def, uniaxial_states, offset, count)


fn fiber_section3d_commit_all(
    mut defs: List[FiberSection3dDef],
    section_uniaxial_offsets: List[Int],
    section_uniaxial_counts: List[Int],
    mut uniaxial_states: List[UniMaterialState],
):
    for i in range(len(section_uniaxial_offsets)):
        fiber_section3d_commit(
            i, defs, section_uniaxial_offsets, section_uniaxial_counts, uniaxial_states
        )


fn fiber_section3d_revert_trial_all(
    mut defs: List[FiberSection3dDef],
    section_uniaxial_offsets: List[Int],
    section_uniaxial_counts: List[Int],
    mut uniaxial_states: List[UniMaterialState],
):
    for i in range(len(section_uniaxial_offsets)):
        fiber_section3d_revert_trial(
            i, defs, section_uniaxial_offsets, section_uniaxial_counts, uniaxial_states
        )


fn fiber_section3d_commit_runtime_all(mut defs: List[FiberSection3dDef]):
    for s in range(len(defs)):
        ref sec_def = defs[s]
        if sec_def.runtime_instance_count <= 0:
            continue
        if sec_def.runtime_s2_state_count > 0:
            _fiber_section3d_commit_steel02_range_simd[FLOAT64_SIMD_WIDTH](
                sec_def, 0, sec_def.runtime_s2_state_count
            )
        if sec_def.runtime_c2_state_count > 0:
            _fiber_section3d_commit_concrete02_range_simd[FLOAT64_SIMD_WIDTH](
                sec_def, 0, sec_def.runtime_c2_state_count
            )
        if sec_def.runtime_c1_state_count > 0:
            _fiber_section3d_commit_concrete01_range_simd[FLOAT64_SIMD_WIDTH](
                sec_def, 0, sec_def.runtime_c1_state_count
            )
        if sec_def.runtime_s1_state_count > 0:
            _fiber_section3d_commit_steel01_range_simd[FLOAT64_SIMD_WIDTH](
                sec_def, 0, sec_def.runtime_s1_state_count
            )
        for i in range(len(sec_def.runtime_other_states)):
            uniaxial_commit(sec_def.runtime_other_states[i])


fn fiber_section3d_revert_trial_runtime_all(mut defs: List[FiberSection3dDef]):
    for s in range(len(defs)):
        ref sec_def = defs[s]
        if sec_def.runtime_instance_count <= 0:
            continue
        if sec_def.runtime_s2_state_count > 0:
            _fiber_section3d_revert_steel02_range_simd[FLOAT64_SIMD_WIDTH](
                sec_def, 0, sec_def.runtime_s2_state_count
            )
        if sec_def.runtime_c2_state_count > 0:
            _fiber_section3d_revert_concrete02_range_simd[FLOAT64_SIMD_WIDTH](
                sec_def, 0, sec_def.runtime_c2_state_count
            )
        if sec_def.runtime_c1_state_count > 0:
            _fiber_section3d_revert_concrete01_range_simd[FLOAT64_SIMD_WIDTH](
                sec_def, 0, sec_def.runtime_c1_state_count
            )
        if sec_def.runtime_s1_state_count > 0:
            _fiber_section3d_revert_steel01_range_simd[FLOAT64_SIMD_WIDTH](
                sec_def, 0, sec_def.runtime_s1_state_count
            )
        for i in range(len(sec_def.runtime_other_states)):
            uniaxial_revert_trial(sec_def.runtime_other_states[i])
