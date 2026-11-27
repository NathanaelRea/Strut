from sections.beam_aggregator import (
    axial_bending_components,
    elastic_aggregator_3d,
    fiber_aggregated_shear_flexibility,
)
from collections import InlineArray, List
from elements.continuum import continuum_dimension
from elements.force_beam_column2d import force_beam_column2d_fiber_state_count
from elements.force_beam_column3d import force_beam_column3d_fiber_state_count
from sections.plate_fiber import elastic_plate_fiber_base
from elements.shell_mass import ShellMass, shell_quad_mass, shell_triangle_mass
from materials.uniaxial.core import CoupledForceHistory
from elements import (
    beam_integration_validate_or_abort,
    link_element_dof_count,
    two_node_link_internal_dir,
    zero_length_internal_dir,
)
from math import sqrt
from os import abort

from materials import (
    UniMaterialDef,
    UniMaterialState,
    uni_mat_initial_tangent,
    uni_mat_is_elastic,
)
from materials.fsam import FSAMParameters, FSAMState
from materials.uniaxial.core import UniMaterialLeafState
from materials.uniaxial.fatigue import FatigueParameters
from materials.uniaxial.parallel import add_strain_limits, make_parallel_def
from materials.uniaxial.min_max import StrainLimits
from solver.assembly.surface_cache import SurfaceAssemblyCache
from solver.dof import node_dof_index, require_dof_in_range
from elements.quad_up import quad_up_displacement_dof
from solver.reorder import build_node_adjacency_typed, min_degree_order, rcm_order
from solver.run_case.helpers import (
    _aggregator_section2d_expected_state_count,
    _aggregator_section3d_expected_state_count,
    _collapse_vector_by_mpc,
    _write_run_progress,
)
from solver.run_case.input_types import (
    AnalysisInput,
    CaseInput,
    DampingInput,
    ElementLoadInput,
    ElementInput,
    MassInput,
    MaterialInput,
    MPConstraintInput,
    NodeInput,
    RayleighInput,
    RecorderInput,
    SectionInput,
    SolverAttemptInput,
    StageInput,
    UniformExcitationInput,
    beam_integration_tag,
)
from solver.time_series import TimeSeriesInput, find_time_series_input
from sections import (
    FiberCell,
    FiberSection2dDef,
    FiberSection3dDef,
    append_fiber_section2d_from_input,
    append_fiber_section3d_from_input,
    fiber_section2d_runtime_alloc_instances,
    fiber_section3d_runtime_alloc_instances,
    fiber_section3d_runtime_reserve_instances,
    LayeredShellSectionDef,
    append_layered_shell_section_from_input,
    layered_shell_runtime_alloc_instances,
)
from strut_io import py_len
from tag_types import (
    AnalysisTypeTag,
    AnalysisSystemTag,
    BeamIntegrationTag,
    ConstraintHandlerTag,
    ElementLoadTypeTag,
    ElementTypeTag,
    ForceBeamModeTag,
    GeomTransfTag,
    IntegratorTypeTag,
    LinkDirectionTag,
    NumbererTag,
    PatternTypeTag,
    UniMaterialTypeTag,
)


struct RunCaseState(Movable):
    var ndm: Int
    var ndf: Int

    var typed_nodes: List[NodeInput]
    var node_count: Int
    var id_to_index: List[Int]
    var node_x: List[Float64]
    var node_y: List[Float64]
    var node_z: List[Float64]
    var node_has_z: List[Bool]
    var node_constraint_offsets: List[Int]
    var node_constraint_pool: List[Int]

    var typed_sections_by_id: List[SectionInput]
    var typed_materials_by_id: List[MaterialInput]

    var uniaxial_defs: List[UniMaterialDef]
    var uniaxial_state_defs: List[Int]
    var uniaxial_states: List[UniMaterialState]

    var fiber_section_defs: List[FiberSection2dDef]
    var fiber_section_cells: List[FiberCell]
    var fiber_section_index_by_id: List[Int]
    var fiber_section3d_defs: List[FiberSection3dDef]
    var fiber_section3d_cells: List[FiberCell]
    var fiber_section3d_index_by_id: List[Int]
    var layered_shell_section_defs: List[LayeredShellSectionDef]

    var typed_elements: List[ElementInput]
    var elem_count: Int
    var elem_id_to_index: List[Int]
    var element_loads: List[ElementLoadInput]
    var elem_load_offsets: List[Int]
    var elem_load_pool: List[Int]
    var elem_dof_offsets: List[Int]
    var elem_dof_pool: List[Int]
    var elem_free_offsets: List[Int]
    var elem_free_pool: List[Int]
    var elem_node_offsets: List[Int]
    var elem_node_pool: List[Int]
    var elem_material_offsets: List[Int]
    var elem_material_pool: List[Int]
    var elem_primary_material_ids: List[Int]
    var elem_type_tags: List[Int]
    var elem_geom_tags: List[Int]
    var elem_section_ids: List[Int]
    var shell_cache: SurfaceAssemblyCache
    var elem_integration_tags: List[Int]
    var elem_num_int_pts: List[Int]
    var elem_dof_counts: List[Int]
    var elem_area: List[Float64]
    var elem_thickness: List[Float64]
    var frame2d_elem_indices: List[Int]
    var frame3d_elem_indices: List[Int]
    var truss_elem_indices: List[Int]
    var zero_length_elem_indices: List[Int]
    var two_node_link_elem_indices: List[Int]
    var zero_length_section_elem_indices: List[Int]
    var quad_elem_indices: List[Int]
    var shell_elem_indices: List[Int]
    var elem_uniaxial_offsets: List[Int]
    var elem_uniaxial_counts: List[Int]
    var elem_uniaxial_state_ids: List[Int]
    var force_basic_offsets: List[Int]
    var force_basic_counts: List[Int]
    var force_basic_q: List[Float64]

    var total_dofs: Int
    var F_total: List[Float64]
    var constrained: List[Bool]
    var prescribed_dofs: List[Int]
    var prescribed_values: List[Float64]

    var analysis: AnalysisInput
    var analysis_type: String
    var analysis_type_tag: Int
    var steps: Int
    var modal_num_modes: Int
    var constraints_handler: String
    var constraints_handler_tag: Int
    var use_banded_linear: Bool
    var use_banded_nonlinear: Bool
    var has_transformation_mpc: Bool
    var penalty_sp_dofs: List[Int]
    var penalty_mp_dofs: List[Int]
    var supports_linear_transient_fast_path: Bool

    var free: List[Int]
    var free_index: List[Int]
    var rep_dof: List[Int]
    var mpc_row_offsets: List[Int]
    var mpc_dof_pool: List[Int]
    var mpc_coeff_pool: List[Float64]
    var mpc_slave_dof: List[Bool]
    var active_index_by_dof: List[Int]

    var M_total: List[Float64]
    var M_rayleigh_total: List[Float64]
    var nodal_masses: List[MassInput]
    var rayleigh: RayleighInput
    var analysis_integrator_targets_pool: List[Float64]
    var analysis_solver_chain_pool: List[SolverAttemptInput]
    var time_series: List[TimeSeriesInput]
    var time_series_values: List[Float64]
    var time_series_times: List[Float64]
    var dampings: List[DampingInput]
    var stages: List[StageInput]
    var ts_index: Int
    var pattern_type: String
    var pattern_type_tag: Int
    var uniform_excitations: List[UniformExcitationInput]
    var rayleigh_alpha_m: Float64
    var rayleigh_beta_k: Float64
    var rayleigh_beta_k_init: Float64
    var rayleigh_beta_k_comm: Float64
    var recorder_nodes_pool: List[Int]
    var recorder_elements_pool: List[Int]
    var recorder_dofs_pool: List[Int]
    var recorder_modes_pool: List[Int]
    var recorder_sections_pool: List[Int]
    var recorders: List[RecorderInput]

    fn __init__(out self):
        self.ndm = 0
        self.ndf = 0
        self.typed_nodes = []
        self.node_count = 0
        self.id_to_index = []
        self.node_x = []
        self.node_y = []
        self.node_z = []
        self.node_has_z = []
        self.node_constraint_offsets = []
        self.node_constraint_pool = []
        self.typed_sections_by_id = []
        self.typed_materials_by_id = []
        self.uniaxial_defs = []
        self.uniaxial_state_defs = []
        self.uniaxial_states = []
        self.fiber_section_defs = []
        self.fiber_section_cells = []
        self.fiber_section_index_by_id = []
        self.fiber_section3d_defs = []
        self.fiber_section3d_cells = []
        self.fiber_section3d_index_by_id = []
        self.layered_shell_section_defs = []
        self.typed_elements = []
        self.elem_count = 0
        self.elem_id_to_index = []
        self.element_loads = []
        self.elem_load_offsets = []
        self.elem_load_pool = []
        self.elem_dof_offsets = []
        self.elem_dof_pool = []
        self.elem_free_offsets = []
        self.elem_free_pool = []
        self.elem_node_offsets = []
        self.elem_node_pool = []
        self.elem_material_offsets = []
        self.elem_material_pool = []
        self.elem_primary_material_ids = []
        self.elem_type_tags = []
        self.elem_geom_tags = []
        self.elem_section_ids = []
        self.shell_cache = SurfaceAssemblyCache()
        self.elem_integration_tags = []
        self.elem_num_int_pts = []
        self.elem_dof_counts = []
        self.elem_area = []
        self.elem_thickness = []
        self.frame2d_elem_indices = []
        self.frame3d_elem_indices = []
        self.truss_elem_indices = []
        self.zero_length_elem_indices = []
        self.two_node_link_elem_indices = []
        self.zero_length_section_elem_indices = []
        self.quad_elem_indices = []
        self.shell_elem_indices = []
        self.elem_uniaxial_offsets = []
        self.elem_uniaxial_counts = []
        self.elem_uniaxial_state_ids = []
        self.force_basic_offsets = []
        self.force_basic_counts = []
        self.force_basic_q = []
        self.total_dofs = 0
        self.F_total = []
        self.constrained = []
        self.prescribed_dofs = []
        self.prescribed_values = []
        self.analysis = AnalysisInput()
        self.analysis_type = ""
        self.analysis_type_tag = AnalysisTypeTag.Unknown
        self.steps = 0
        self.modal_num_modes = 0
        self.constraints_handler = "Plain"
        self.constraints_handler_tag = ConstraintHandlerTag.Plain
        self.use_banded_linear = False
        self.use_banded_nonlinear = False
        self.has_transformation_mpc = False
        self.penalty_sp_dofs = []
        self.penalty_mp_dofs = []
        self.supports_linear_transient_fast_path = False
        self.free = []
        self.free_index = []
        self.rep_dof = []
        self.mpc_row_offsets = []
        self.mpc_dof_pool = []
        self.mpc_coeff_pool = []
        self.mpc_slave_dof = []
        self.active_index_by_dof = []
        self.M_total = []
        self.M_rayleigh_total = []
        self.nodal_masses = []
        self.rayleigh = RayleighInput()
        self.analysis_integrator_targets_pool = []
        self.analysis_solver_chain_pool = []
        self.time_series = []
        self.time_series_values = []
        self.time_series_times = []
        self.dampings = []
        self.stages = []
        self.ts_index = -1
        self.pattern_type = "Plain"
        self.pattern_type_tag = PatternTypeTag.Plain
        self.uniform_excitations = []
        self.rayleigh_alpha_m = 0.0
        self.rayleigh_beta_k = 0.0
        self.rayleigh_beta_k_init = 0.0
        self.rayleigh_beta_k_comm = 0.0
        self.recorder_nodes_pool = []
        self.recorder_elements_pool = []
        self.recorder_dofs_pool = []
        self.recorder_modes_pool = []
        self.recorder_sections_pool = []
        self.recorders = []


fn _set_elem_dof(mut elem: ElementInput, idx: Int, value: Int):
    if idx == 0:
        elem.dof_1 = value
    elif idx == 1:
        elem.dof_2 = value
    elif idx == 2:
        elem.dof_3 = value
    elif idx == 3:
        elem.dof_4 = value
    elif idx == 4:
        elem.dof_5 = value
    elif idx == 5:
        elem.dof_6 = value
    elif idx == 6:
        elem.dof_7 = value
    elif idx == 7:
        elem.dof_8 = value
    elif idx == 8:
        elem.dof_9 = value
    elif idx == 9:
        elem.dof_10 = value
    elif idx == 10:
        elem.dof_11 = value
    elif idx == 11:
        elem.dof_12 = value
    elif idx == 12:
        elem.dof_13 = value
    elif idx == 13:
        elem.dof_14 = value
    elif idx == 14:
        elem.dof_15 = value
    elif idx == 15:
        elem.dof_16 = value
    elif idx == 16:
        elem.dof_17 = value
    elif idx == 17:
        elem.dof_18 = value
    elif idx == 18:
        elem.dof_19 = value
    elif idx == 19:
        elem.dof_20 = value
    elif idx == 20:
        elem.dof_21 = value
    elif idx == 21:
        elem.dof_22 = value
    elif idx == 22:
        elem.dof_23 = value
    elif idx == 23:
        elem.dof_24 = value
    else:
        elem.extra_dofs[idx - 24] = value


fn _set_elem_material(mut elem: ElementInput, idx: Int, value: Int):
    if idx == 0:
        elem.material_1 = value
    elif idx == 1:
        elem.material_2 = value
    elif idx == 2:
        elem.material_3 = value
    elif idx == 3:
        elem.material_4 = value
    elif idx == 4:
        elem.material_5 = value
    else:
        elem.material_6 = value


fn _set_elem_damp_material(mut elem: ElementInput, idx: Int, value: Int):
    if idx == 0:
        elem.damp_material_1 = value
    elif idx == 1:
        elem.damp_material_2 = value
    elif idx == 2:
        elem.damp_material_3 = value
    elif idx == 3:
        elem.damp_material_4 = value
    elif idx == 4:
        elem.damp_material_5 = value
    else:
        elem.damp_material_6 = value


fn _remap_compact_index_or_abort(raw_id: Int, raw_to_compact: List[Int], label: String) -> Int:
    if raw_id < 0:
        return -1
    if raw_id >= len(raw_to_compact) or raw_to_compact[raw_id] < 0:
        abort(label + " not found")
    return raw_to_compact[raw_id]


fn _set_elem_dir(mut elem: ElementInput, idx: Int, value: Int):
    if idx == 0:
        elem.dir_1 = value
    elif idx == 1:
        elem.dir_2 = value
    elif idx == 2:
        elem.dir_3 = value
    elif idx == 3:
        elem.dir_4 = value
    elif idx == 4:
        elem.dir_5 = value
    else:
        elem.dir_6 = value


fn _element_supports_linear_transient_fast_path(
    elem: ElementInput, sections_by_id: List[SectionInput]
) -> Bool:
    if (
        elem.type_tag == ElementTypeTag.ElasticBeamColumn2d
        or elem.type_tag == ElementTypeTag.ElasticBeamColumn3d
        or elem.type_tag == ElementTypeTag.Truss
        or elem.type_tag == ElementTypeTag.ZeroLength
        or elem.type_tag == ElementTypeTag.TwoNodeLink
        or elem.type_tag == ElementTypeTag.MVLEM
        or elem.type_tag == ElementTypeTag.FourNodeQuad
        or elem.type_tag == ElementTypeTag.Tri31
        or elem.type_tag == ElementTypeTag.Continuum
        or (
            elem.type_tag == ElementTypeTag.Shell
            and elem.type != "ShellNLDKGQ"
            and (elem.type != "ASDShellQ4" or elem.geom_tag != GeomTransfTag.Corotational)
        )
    ):
        return True
    if elem.type_tag == ElementTypeTag.ZeroLengthSection:
        if elem.section < 0 or elem.section >= len(sections_by_id):
            return False
        var sec = sections_by_id[elem.section]
        return sec.type == "ElasticSection2d" or sec.type == "ElasticSection3d"
    return False


fn _coerce_aggregator_section2d_for_beam_column(
    mut sec: SectionInput,
    sections_by_id: List[SectionInput],
    uniaxial_defs: List[UniMaterialDef],
    uniaxial_def_by_id: List[Int],
    beam_col_type: String,
) raises -> SectionInput:
    if sec.type != "AggregatorSection2d":
        return sec
    if sec.base_section >= 0 and sections_by_id[sec.base_section].type == "ElasticSection2d":
        var base = sections_by_id[sec.base_section]
        if (
            base.type != "ElasticSection2d"
            or sec.axial_material >= 0
            or sec.flexural_material >= 0
            or sec.shear_y_material < 0
            or sec.shear_z_material >= 0
            or sec.moment_y_material >= 0
            or sec.torsion_material >= 0
        ):
            abort(beam_col_type + " expects a single Vy addition to an elastic 2D base section")
        base.id = sec.id
        base.shear_rigidity_y = _elastic_aggregated_shear_rigidity(
            base.shear_rigidity_y, sec.shear_y_material, uniaxial_defs, uniaxial_def_by_id
        )
        return base
    var axial: Int
    var flexural: Int
    (axial, flexural) = axial_bending_components(sec, sections_by_id, uniaxial_def_by_id)
    var axial_def = uniaxial_defs[axial]
    var flexural_def = uniaxial_defs[flexural]
    sec.type = "ElasticSection2d"
    sec.E = 1.0
    sec.A = uni_mat_initial_tangent(axial_def)
    sec.I = uni_mat_initial_tangent(flexural_def)
    sec.shear_rigidity_y = _elastic_aggregated_shear_rigidity(
        0.0, sec.shear_y_material, uniaxial_defs, uniaxial_def_by_id
    )
    if sec.A <= 0.0 or sec.I <= 0.0:
        abort(beam_col_type + " AggregatorSection2d surrogate stiffness must be positive")
    return sec


fn _hinge_end_materials(
    section: Int,
    sections_by_id: List[SectionInput],
    fiber_defs: List[FiberSection2dDef],
    fiber_index: List[Int],
    mut uniaxial_defs: List[UniMaterialDef],
    uniaxial_def_by_id: List[Int],
) -> (Int, Int):
    ref end = sections_by_id[section]
    if end.type == "AggregatorSection2d":
        if end.shear_y_material >= 0:
            abort("beamWithHinges end sections require independent P and Mz responses")
        return axial_bending_components(end, sections_by_id, uniaxial_def_by_id)

    var ea = 0.0
    var ei = 0.0
    if end.type == "ElasticSection2d":
        ea = end.E * end.A
        ei = end.E * end.I
    elif end.type == "FiberSection2d":
        ref definition = fiber_defs[fiber_index[section]]
        if definition.elastic_count != definition.fiber_count or not definition.initial_flex_valid:
            abort("beamWithHinges requires an elastic fiber end section")
        if abs(definition.initial_f01) > 1.0e-12 * sqrt(
            definition.initial_f00 * definition.initial_f11
        ):
            abort("beamWithHinges elastic fiber ends require uncoupled axial and bending response")
        ea = 1.0 / definition.initial_f00
        ei = 1.0 / definition.initial_f11
    else:
        abort("unsupported beamWithHinges end section")

    var axial = len(uniaxial_defs)
    uniaxial_defs.append(UniMaterialDef(UniMaterialTypeTag.Elastic, ea, 0.0, 0.0, 0.0))
    var flexural = len(uniaxial_defs)
    uniaxial_defs.append(UniMaterialDef(UniMaterialTypeTag.Elastic, ei, 0.0, 0.0, 0.0))
    return (axial, flexural)


fn _elastic_aggregated_shear_rigidity(
    base: Float64,
    material: Int,
    uniaxial_defs: List[UniMaterialDef],
    uniaxial_def_by_id: List[Int],
) -> Float64:
    if material < 0:
        return base
    var index = uniaxial_def_by_id[material]
    if index < 0 or not uni_mat_is_elastic(uniaxial_defs[index]):
        abort("elastic beam section requires an elastic aggregated shear material")
    var rigidity = uni_mat_initial_tangent(uniaxial_defs[index])
    if rigidity <= 0.0:
        abort("aggregated shear stiffness must be positive")
    # Force interpolation applies the same shear to every response with
    # the same code. Their independent flexibilities add.
    if base != 0.0:
        return base * rigidity / (base + rigidity)
    return rigidity


fn _coerce_aggregator_section3d_for_beam_column(
    sec: SectionInput,
    sections: List[SectionInput],
    uniaxial_defs: List[UniMaterialDef],
    uniaxial_def_by_id: List[Int],
) -> SectionInput:
    if sec.type != "AggregatorSection3d":
        return sec
    if (
        sec.base_section < 0
        or sec.axial_material >= 0
        or sec.flexural_material >= 0
        or sec.moment_y_material >= 0
        or sec.torsion_material >= 0
    ):
        abort("3D beam section aggregator requires Vy or Vz and an elastic base")
    var base = sections[sec.base_section]
    if base.type != "ElasticSection3d":
        abort("3D beam section aggregator requires an elastic 3D base")
    base.id = sec.id
    base.base_section = sec.base_section
    base.shear_rigidity_y = _elastic_aggregated_shear_rigidity(
        base.shear_rigidity_y, sec.shear_y_material, uniaxial_defs, uniaxial_def_by_id
    )
    base.shear_rigidity_z = _elastic_aggregated_shear_rigidity(
        base.shear_rigidity_z, sec.shear_z_material, uniaxial_defs, uniaxial_def_by_id
    )
    return base


fn _node_constraint(node: NodeInput, idx: Int) -> Int:
    if idx == 0:
        return node.constraint_1
    if idx == 1:
        return node.constraint_2
    if idx == 2:
        return node.constraint_3
    if idx == 3:
        return node.constraint_4
    if idx == 4:
        return node.constraint_5
    return node.constraint_6


fn _elem_node(elem: ElementInput, idx: Int) -> Int:
    if idx == 0:
        return elem.node_index_1
    if idx == 1:
        return elem.node_index_2
    if idx == 2:
        return elem.node_index_3
    if idx == 3:
        return elem.node_index_4
    if elem.solid_extra_nodes:
        return elem.solid_extra_nodes.value()[idx - 4]
    return elem.extra_node_indices[idx - 4]


fn _elem_material(elem: ElementInput, idx: Int) -> Int:
    if idx == 0:
        return elem.material_1
    if idx == 1:
        return elem.material_2
    if idx == 2:
        return elem.material_3
    if idx == 3:
        return elem.material_4
    if idx == 4:
        return elem.material_5
    return elem.material_6


fn _elem_damp_material(elem: ElementInput, idx: Int) -> Int:
    if idx == 0:
        return elem.damp_material_1
    if idx == 1:
        return elem.damp_material_2
    if idx == 2:
        return elem.damp_material_3
    if idx == 3:
        return elem.damp_material_4
    if idx == 4:
        return elem.damp_material_5
    return elem.damp_material_6


fn _mp_constraint_dof(mpc: MPConstraintInput, idx: Int) -> Int:
    if idx == 0:
        return mpc.dof_1
    if idx == 1:
        return mpc.dof_2
    if idx == 2:
        return mpc.dof_3
    if idx == 3:
        return mpc.dof_4
    if idx == 4:
        return mpc.dof_5
    return mpc.dof_6


fn _mp_constraint_rigid_constrained_dof(mpc: MPConstraintInput, idx: Int) -> Int:
    if idx == 0:
        return mpc.rigid_constrained_dof_1
    if idx == 1:
        return mpc.rigid_constrained_dof_2
    return mpc.rigid_constrained_dof_3


fn _mp_constraint_rigid_retained_dof(mpc: MPConstraintInput, idx: Int) -> Int:
    if idx == 0:
        return mpc.rigid_retained_dof_1
    if idx == 1:
        return mpc.rigid_retained_dof_2
    return mpc.rigid_retained_dof_3


fn _mp_constraint_rigid_matrix_entry(mpc: MPConstraintInput, row: Int, col: Int) -> Float64:
    if row == 0:
        if col == 0:
            return mpc.rigid_matrix_11
        if col == 1:
            return mpc.rigid_matrix_12
        return mpc.rigid_matrix_13
    if row == 1:
        if col == 0:
            return mpc.rigid_matrix_21
        if col == 1:
            return mpc.rigid_matrix_22
        return mpc.rigid_matrix_23
    if col == 0:
        return mpc.rigid_matrix_31
    if col == 1:
        return mpc.rigid_matrix_32
    return mpc.rigid_matrix_33


fn _find_damping_input(dampings: List[DampingInput], tag: Int) -> Int:
    for i in range(len(dampings)):
        if dampings[i].tag == tag:
            return i
    return -1


fn _elem_dof(elem: ElementInput, idx: Int) -> Int:
    if elem.type_tag == ElementTypeTag.SFIMVLEM and idx >= 6:
        return node_dof_index(elem.wall_internal_node_start + idx - 6, 1, 3)
    if idx == 0:
        return elem.dof_1
    if idx == 1:
        return elem.dof_2
    if idx == 2:
        return elem.dof_3
    if idx == 3:
        return elem.dof_4
    if idx == 4:
        return elem.dof_5
    if idx == 5:
        return elem.dof_6
    if idx == 6:
        return elem.dof_7
    if idx == 7:
        return elem.dof_8
    if idx == 8:
        return elem.dof_9
    if idx == 9:
        return elem.dof_10
    if idx == 10:
        return elem.dof_11
    if idx == 11:
        return elem.dof_12
    if idx == 12:
        return elem.dof_13
    if idx == 13:
        return elem.dof_14
    if idx == 14:
        return elem.dof_15
    if idx == 15:
        return elem.dof_16
    if idx == 16:
        return elem.dof_17
    if idx == 17:
        return elem.dof_18
    if idx == 18:
        return elem.dof_19
    if idx == 19:
        return elem.dof_20
    if idx == 20:
        return elem.dof_21
    if idx == 21:
        return elem.dof_22
    if idx == 22:
        return elem.dof_23
    if idx == 23:
        return elem.dof_24
    return elem.extra_dofs[idx - 24]


fn _element_accepts_beam_load(elem: ElementInput, ndm: Int, ndf: Int) -> Bool:
    if ndm == 2 and ndf == 3:
        return (
            elem.type_tag == ElementTypeTag.ElasticBeamColumn2d
            or elem.type_tag == ElementTypeTag.ForceBeamColumn2d
            or elem.type_tag == ElementTypeTag.DispBeamColumn2d
        )
    if ndm == 3 and ndf == 6:
        return (
            elem.type_tag == ElementTypeTag.ElasticBeamColumn3d
            or elem.type_tag == ElementTypeTag.ForceBeamColumn3d
            or elem.type_tag == ElementTypeTag.DispBeamColumn3d
        )
    return False


fn _validate_element_load_for_element(
    load: ElementLoadInput, elem: ElementInput, ndm: Int, ndf: Int
):
    if load.type_tag == ElementLoadTypeTag.BeamThermal:
        if (
            ndm != 2
            or ndf != 3
            or elem.type_tag != ElementTypeTag.DispBeamColumn2d
            or not elem.thermal_beam
        ):
            abort("beamThermal requires a 2D dispBeamColumnThermal element")
        return
    if elem.type_tag == ElementTypeTag.CatenaryCable:
        if load.type_tag != ElementLoadTypeTag.BeamUniform:
            abort("CatenaryCable only accepts beamUniform element loads")
        return
    if not _element_accepts_beam_load(elem, ndm, ndf):
        abort(
            load.type
            + " requires elasticBeamColumn"
            + String(ndm)
            + "d, forceBeamColumn"
            + String(ndm)
            + "d, or dispBeamColumn"
            + String(ndm)
            + "d"
        )
    if load.type_tag == ElementLoadTypeTag.BeamUniform:
        if ndm == 2 and ndf != 3:
            abort("beamUniform requires ndf=3")
        if ndm == 3 and ndf != 6:
            abort("beamUniform requires ndf=6")
        return
    if load.type_tag == ElementLoadTypeTag.BeamPoint:
        if load.x < 0.0 or load.x > 1.0:
            abort("beamPoint requires x in [0, 1]")
        if ndm == 2 and ndf != 3:
            abort("beamPoint requires ndf=3")
        if ndm == 3 and ndf != 6:
            abort("beamPoint requires ndf=6")
        return
    abort("unsupported element load type")


fn resolve_uniform_excitations(
    inputs: List[UniformExcitationInput], time_series: List[TimeSeriesInput], ndm: Int
) -> List[UniformExcitationInput]:
    if len(inputs) == 0:
        abort("UniformExcitation requires at least one excitation")
    var resolved: List[UniformExcitationInput] = []
    for i in range(len(inputs)):
        var excitation = inputs[i]
        if excitation.direction < 1 or excitation.direction > ndm:
            abort("UniformExcitation direction out of range 1..ndm")
        var series_index = find_time_series_input(time_series, excitation.accel)
        if series_index < 0:
            abort("UniformExcitation accel time_series tag not found")
        excitation.accel = series_index
        resolved.append(excitation)
    return resolved^


fn _build_element_load_index(
    element_loads: List[ElementLoadInput],
    typed_elements: List[ElementInput],
    elem_id_to_index: List[Int],
    ndm: Int,
    ndf: Int,
    mut elem_load_offsets: List[Int],
    mut elem_load_pool: List[Int],
):
    var elem_count = len(typed_elements)
    elem_load_offsets.resize(elem_count + 1, 0)
    for i in range(len(element_loads)):
        var load = element_loads[i]
        var elem_id = load.element
        if elem_id < 0 or elem_id >= len(elem_id_to_index) or elem_id_to_index[elem_id] < 0:
            abort("element load refers to unknown element")
        var elem_index = elem_id_to_index[elem_id]
        _validate_element_load_for_element(load, typed_elements[elem_index], ndm, ndf)
        elem_load_offsets[elem_index + 1] += 1
    for e in range(elem_count):
        elem_load_offsets[e + 1] += elem_load_offsets[e]

    elem_load_pool.resize(len(element_loads), -1)
    var next_slot = elem_load_offsets.copy()
    for i in range(len(element_loads)):
        var load = element_loads[i]
        var elem_index = elem_id_to_index[load.element]
        var slot = next_slot[elem_index]
        elem_load_pool[slot] = i
        next_slot[elem_index] += 1


fn _accumulate_beam_element_lumped_mass(
    elem: ElementInput,
    rho: Float64,
    typed_nodes: List[NodeInput],
    ndf: Int,
    mut m_total: List[Float64],
):
    if rho == 0.0:
        return
    if (
        elem.type_tag == ElementTypeTag.DispBeamColumn2d
        or elem.type_tag == ElementTypeTag.DispBeamColumn3d
    ):
        if elem.use_cmass:
            abort(elem.type + " cMass is not yet supported")
    var node1 = typed_nodes[elem.node_index_1]
    var node2 = typed_nodes[elem.node_index_2]
    var dx = node2.x - node1.x
    var dy = node2.y - node1.y
    var dz = node2.z - node1.z
    var l = sqrt(dx * dx + dy * dy + dz * dz)
    if l == 0.0:
        abort("zero-length element")
    var lump = 0.5 * rho * l
    m_total[node_dof_index(elem.node_index_1, 1, ndf)] += lump
    m_total[node_dof_index(elem.node_index_2, 1, ndf)] += lump
    if ndf >= 2:
        m_total[node_dof_index(elem.node_index_1, 2, ndf)] += lump
        m_total[node_dof_index(elem.node_index_2, 2, ndf)] += lump
    if ndf >= 6:
        m_total[node_dof_index(elem.node_index_1, 3, ndf)] += lump
        m_total[node_dof_index(elem.node_index_2, 3, ndf)] += lump


fn _accumulate_two_node_link_lumped_mass(
    elem: ElementInput, ndm: Int, ndf: Int, mut m_total: List[Float64]
):
    if elem.element_mass <= 0.0:
        return
    var lump = 0.5 * elem.element_mass
    for d in range(ndm):
        m_total[node_dof_index(elem.node_index_1, d + 1, ndf)] += lump
        m_total[node_dof_index(elem.node_index_2, d + 1, ndf)] += lump


fn _elem_dir(elem: ElementInput, idx: Int) -> Int:
    if idx == 0:
        return elem.dir_1
    if idx == 1:
        return elem.dir_2
    if idx == 2:
        return elem.dir_3
    if idx == 3:
        return elem.dir_4
    if idx == 4:
        return elem.dir_5
    return elem.dir_6


fn _ensure_id_slot(mut table: List[Int], raw_id: Int):
    """Make `table[raw_id]` addressable, filling new slots with -1.

    Model ids are sparse and can run into the tens of millions. Growing the
    table to exactly `raw_id + 1` for every new id copies the whole table each
    time; growing geometrically keeps the total copy work linear in its size.
    """
    var size = len(table)
    if raw_id < size:
        return
    var grown = size + size // 2
    if grown < raw_id + 1:
        grown = raw_id + 1
    table.resize(grown, -1)


fn _resolve_composite_definition(
    material_id: Int,
    materials: List[MaterialInput],
    mut definitions: List[UniMaterialDef],
    mut index_by_id: List[Int],
) -> Int:
    var resolved = index_by_id[material_id]
    if resolved >= 0:
        return resolved
    if resolved == -2:
        abort("cyclic uniaxial material dependency")

    ref material = materials[material_id]
    index_by_id[material_id] = -2
    if material.type == "MinMax":
        if material.base_material < 0:
            abort("MinMax requires a base material")
        var child_index = _resolve_composite_definition(
            material.base_material, materials, definitions, index_by_id
        )
        var definition = definitions[child_index]
        add_strain_limits(
            definition.strain_limits, StrainLimits(material.strain_min, material.strain_max)
        )
        var index = len(definitions)
        definitions.append(definition)
        index_by_id[material_id] = index
        return index
    if not material.parallel_components:
        abort("composite requires uniaxial child materials")
    ref components = material.parallel_components.value()
    var children = List[UniMaterialDef](capacity=len(components))
    var factors = List[Float64](capacity=len(components))
    for i in range(len(components)):
        ref component = components[i]
        var child_index = _resolve_composite_definition(
            component.material, materials, definitions, index_by_id
        )
        children.append(definitions[child_index])
        factors.append(component.factor)
    var index = len(definitions)
    definitions.append(make_parallel_def(children, factors))
    index_by_id[material_id] = index
    return index


fn load_case_state_from_input(var input: CaseInput, progress_path: String) raises -> RunCaseState:
    var state = RunCaseState()
    var ndm = input.model.ndm
    var ndf = input.model.ndf
    ref typed_nodes = input.nodes
    ref sections = input.sections
    ref fiber_patches = input.fiber_patches
    ref fiber_layers = input.fiber_layers
    ref section_fibers = input.fibers
    ref shell_layers = input.shell_layers
    ref materials = input.materials
    ref typed_elements = input.elements
    ref element_loads = input.element_loads
    ref nodal_loads = input.loads
    ref masses = input.masses
    var analysis_input = input.analysis
    ref mp_constraints = input.mp_constraints
    var pattern_input = input.pattern
    var rayleigh_input = input.rayleigh
    ref time_series = input.time_series
    ref dampings = input.dampings
    var load_step_count = 6

    var is_2d = ndm == 2 and (ndf == 2 or ndf == 3)
    var is_3d_truss = ndm == 3 and ndf == 3
    var is_3d_shell = ndm == 3 and ndf == 6
    var is_3d_soil = ndm == 3 and ndf == 4
    if (
        not (ndm == 1 and ndf == 1)
        and not is_2d
        and not is_3d_truss
        and not is_3d_shell
        and not is_3d_soil
    ):
        abort("only ndm=1 ndf=1, ndm=2 ndf=2/3, and ndm=3 ndf=3/4/6 supported")

    var node_count = len(typed_nodes)
    var node_ids: List[Int] = []
    node_ids.resize(node_count, 0)

    for i in range(node_count):
        var node = typed_nodes[i]
        if ndm == 3 and not node.has_z:
            abort("ndm=3 requires node z coordinate")
        node_ids[i] = node.id

    var id_to_index: List[Int] = []
    id_to_index.resize(10000, -1)
    for i in range(node_count):
        var nid = node_ids[i]
        _ensure_id_slot(id_to_index, nid)
        id_to_index[nid] = i

    var material_index_by_raw_id: List[Int] = []
    material_index_by_raw_id.resize(10000, -1)
    for i in range(len(materials)):
        var raw_id = materials[i].id
        _ensure_id_slot(material_index_by_raw_id, raw_id)
        material_index_by_raw_id[raw_id] = i
    for i in range(len(input.stages)):
        ref updates = input.stages[i].material_stages
        for j in range(len(updates)):
            updates[j].material = _remap_compact_index_or_abort(
                updates[j].material, material_index_by_raw_id, "staged material"
            )
    for i in range(len(materials)):
        var mat = materials[i]
        if mat.base_material >= 0:
            mat.base_material = _remap_compact_index_or_abort(
                mat.base_material, material_index_by_raw_id, "base material"
            )
        if mat.parallel_components:
            ref components = mat.parallel_components.value()
            for j in range(len(components)):
                components[j].material = _remap_compact_index_or_abort(
                    components[j].material, material_index_by_raw_id, "Parallel child material"
                )
        mat.id = i
        materials[i] = mat
    for i in range(len(fiber_patches)):
        var patch = fiber_patches[i]
        patch.material = _remap_compact_index_or_abort(
            patch.material, material_index_by_raw_id, "fiber patch material"
        )
        fiber_patches[i] = patch
    for i in range(len(fiber_layers)):
        var layer = fiber_layers[i]
        layer.material = _remap_compact_index_or_abort(
            layer.material, material_index_by_raw_id, "fiber layer material"
        )
        fiber_layers[i] = layer
    for i in range(len(section_fibers)):
        var fiber = section_fibers[i]
        fiber.material = _remap_compact_index_or_abort(
            fiber.material, material_index_by_raw_id, "fiber material"
        )
        section_fibers[i] = fiber
    for i in range(len(shell_layers)):
        var layer = shell_layers[i]
        layer.material = _remap_compact_index_or_abort(
            layer.material, material_index_by_raw_id, "shell layer material"
        )
        shell_layers[i] = layer
    var typed_materials_by_id: List[MaterialInput] = []
    typed_materials_by_id.resize(len(materials), MaterialInput())
    for i in range(len(materials)):
        typed_materials_by_id[i] = materials[i]
    for i in range(len(materials)):
        ref wrapper = typed_materials_by_id[i]
        if wrapper.type == "InitialStateAnalysisWrapper":
            if wrapper.base_material < 0 or wrapper.soil_dimensions != 2:
                abort("InitialStateAnalysisWrapper requires a 2D multi-yield soil material")
            ref base = materials[wrapper.base_material]
            if base.soil_dimensions != 2 or (base.pimy_index < 0 and base.pdmy_index < 0):
                abort("InitialStateAnalysisWrapper requires a 2D multi-yield soil material")
            wrapper.pimy_index = base.pimy_index
            wrapper.pdmy_index = base.pdmy_index
            wrapper.rho = base.rho

    var section_index_by_raw_id: List[Int] = []
    section_index_by_raw_id.resize(10000, -1)
    for i in range(len(sections)):
        var raw_id = sections[i].id
        _ensure_id_slot(section_index_by_raw_id, raw_id)
        section_index_by_raw_id[raw_id] = i
    for i in range(len(sections)):
        var sec = sections[i]
        if sec.axial_material >= 0:
            sec.axial_material = _remap_compact_index_or_abort(
                sec.axial_material, material_index_by_raw_id, "AggregatorSection2d axial material"
            )
        if sec.flexural_material >= 0:
            sec.flexural_material = _remap_compact_index_or_abort(
                sec.flexural_material,
                material_index_by_raw_id,
                "AggregatorSection2d flexural material",
            )
        if sec.moment_y_material >= 0:
            sec.moment_y_material = _remap_compact_index_or_abort(
                sec.moment_y_material,
                material_index_by_raw_id,
                "AggregatorSection2d moment material",
            )
        if sec.torsion_material >= 0 and sec.added_torsional_rigidity == 0.0:
            sec.torsion_material = _remap_compact_index_or_abort(
                sec.torsion_material,
                material_index_by_raw_id,
                "AggregatorSection2d torsion material",
            )
        if sec.shear_y_material >= 0:
            sec.shear_y_material = _remap_compact_index_or_abort(
                sec.shear_y_material,
                material_index_by_raw_id,
                "AggregatorSection2d shear-y material",
            )
        if sec.shear_z_material >= 0:
            sec.shear_z_material = _remap_compact_index_or_abort(
                sec.shear_z_material,
                material_index_by_raw_id,
                "AggregatorSection2d shear-z material",
            )
        if sec.base_section >= 0 and sec.added_torsional_rigidity == 0.0:
            sec.base_section = _remap_compact_index_or_abort(
                sec.base_section, section_index_by_raw_id, "base section"
            )
        if sec.type == "PlateFiber":
            sec.plate_fiber_material = _remap_compact_index_or_abort(
                sec.plate_fiber_material, material_index_by_raw_id, "plate fiber material"
            )
            var material = elastic_plate_fiber_base(sec.plate_fiber_material, typed_materials_by_id)
            sec.E = material.E
            sec.nu = material.nu
            sec.rho = material.rho
        sec.id = i
        sections[i] = sec
    var typed_sections_by_id: List[SectionInput] = []
    typed_sections_by_id.resize(len(sections), SectionInput())
    for i in range(len(sections)):
        typed_sections_by_id[i] = sections[i]

    _write_run_progress(progress_path, "loading", "case", 0, 0, 1, load_step_count)

    var uniaxial_defs: List[UniMaterialDef] = []
    var uniaxial_def_by_id: List[Int] = []
    uniaxial_def_by_id.resize(len(typed_materials_by_id), -1)
    for i in range(len(materials)):
        var mat = materials[i]
        var mid = mat.id
        _ensure_id_slot(uniaxial_def_by_id, mid)
        var mat_type = mat.type
        if mat_type == "Elastic":
            var E = mat.E
            var mat_def = UniMaterialDef(UniMaterialTypeTag.Elastic, E, 0.0, 0.0, 0.0)
            uniaxial_def_by_id[mid] = len(uniaxial_defs)
            uniaxial_defs.append(mat_def)
        elif mat_type == "PySimple1" or mat_type == "TzSimple1" or mat_type == "QzSimple1":
            var tag = UniMaterialTypeTag.PySimple1
            var p3 = mat.soil_drag_or_suction
            var p4 = mat.soil_dashpot
            if mat_type == "TzSimple1":
                tag = UniMaterialTypeTag.TzSimple1
                p3 = mat.soil_dashpot
                p4 = 0.0
            elif mat_type == "QzSimple1":
                tag = UniMaterialTypeTag.QzSimple1
            uniaxial_def_by_id[mid] = len(uniaxial_defs)
            uniaxial_defs.append(
                UniMaterialDef(
                    tag,
                    Float64(mat.soil_type),
                    mat.soil_ultimate,
                    mat.soil_disp50,
                    p3,
                    p4,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                )
            )
        elif mat_type == "ENT":
            var mat_def = UniMaterialDef(UniMaterialTypeTag.ENT, mat.E, mat.ent_a, mat.ent_b, 0.0)
            uniaxial_def_by_id[mid] = len(uniaxial_defs)
            uniaxial_defs.append(mat_def)
        elif mat_type == "Parallel" or mat_type == "MinMax":
            continue
        elif mat_type == "Viscous":
            if mat.viscous_exponent != 1.0:
                abort("Viscous currently requires alpha = 1")
            uniaxial_def_by_id[mid] = len(uniaxial_defs)
            uniaxial_defs.append(
                UniMaterialDef(UniMaterialTypeTag.Viscous, mat.viscous_coefficient, 0.0, 0.0, 0.0)
            )
        elif mat_type == "Steel01" or mat_type == "Steel01Thermal":
            var Fy = mat.Fy
            var E0 = mat.E0
            var b = mat.b
            if Fy <= 0.0:
                abort("Steel01 Fy must be > 0")
            if E0 <= 0.0:
                abort("Steel01 E0 must be > 0")
            if b < 0.0 or b >= 1.0:
                abort("Steel01 b must be in [0, 1)")
            var tag = (
                UniMaterialTypeTag.Steel01Thermal if mat_type
                == "Steel01Thermal" else UniMaterialTypeTag.Steel01
            )
            var mat_def = UniMaterialDef(tag, Fy, E0, b, 0.0)
            uniaxial_def_by_id[mid] = len(uniaxial_defs)
            uniaxial_defs.append(mat_def)
        elif mat_type == "HystereticSmooth" or mat_type == "HystereticAsym":
            if mat.ka <= 0.0 or mat.kb >= mat.ka or mat.fbar <= 0.0:
                abort(mat_type + " requires ka > 0, kb < ka, and fbar > 0")
            var material_tag = UniMaterialTypeTag.HystereticSmooth
            if mat_type == "HystereticAsym":
                material_tag = UniMaterialTypeTag.HystereticAsym
            var mat_def = UniMaterialDef(material_tag, mat.ka, mat.kb, mat.fbar, mat.beta1)
            mat_def.p4 = mat.beta2
            mat_def.p5 = mat.gamma
            uniaxial_def_by_id[mid] = len(uniaxial_defs)
            uniaxial_defs.append(mat_def)
        elif mat_type == "Concrete01":
            var fpc = mat.fpc
            var epsc0 = mat.epsc0
            var fpcu = mat.fpcu
            var epscu = mat.epscu
            if fpc > 0.0:
                fpc = -fpc
            if epsc0 > 0.0:
                epsc0 = -epsc0
            if fpcu > 0.0:
                fpcu = -fpcu
            if epscu > 0.0:
                epscu = -epscu
            if fpc >= 0.0:
                abort("Concrete01 fpc must be < 0")
            if epsc0 >= 0.0:
                abort("Concrete01 epsc0 must be < 0")
            if epscu >= 0.0:
                abort("Concrete01 epscu must be < 0")
            if epscu >= epsc0:
                abort("Concrete01 epscu must be < epsc0")
            if fpcu > 0.0 or fpcu < fpc:
                abort("Concrete01 fpcu must be between fpc and 0")
            var mat_def = UniMaterialDef(UniMaterialTypeTag.Concrete01, fpc, epsc0, fpcu, epscu)
            uniaxial_def_by_id[mid] = len(uniaxial_defs)
            uniaxial_defs.append(mat_def)
        elif mat_type == "Steel02":
            var Fy = mat.Fy
            var E0 = mat.E0
            var b = mat.b
            if Fy < 0.0:
                abort("Steel02 Fy must be >= 0")
            if E0 <= 0.0:
                abort("Steel02 E0 must be > 0")
            if b < 0.0 or b >= 1.0:
                abort("Steel02 b must be in [0, 1)")

            var has_r0 = mat.has_r0
            var has_cr1 = mat.has_cr1
            var has_cr2 = mat.has_cr2
            var has_a1 = mat.has_a1
            var has_a2 = mat.has_a2
            var has_a3 = mat.has_a3
            var has_a4 = mat.has_a4
            var has_siginit = mat.has_siginit

            if has_r0 != has_cr1 or has_r0 != has_cr2:
                abort("Steel02 requires R0, cR1, cR2 together")
            if has_a1 != has_a2 or has_a1 != has_a3 or has_a1 != has_a4:
                abort("Steel02 requires a1, a2, a3, a4 together")
            if has_a1 and not has_r0:
                abort("Steel02 a1-a4 require R0, cR1, cR2")
            if has_siginit and not has_a1:
                abort("Steel02 sigInit requires a1-a4 and R0/cR1/cR2")

            var R0 = 15.0
            var cR1 = 0.925
            var cR2 = 0.15
            if has_r0:
                R0 = mat.R0
                cR1 = mat.cR1
                cR2 = mat.cR2
            if R0 <= 0.0:
                abort("Steel02 R0 must be > 0")
            if cR2 <= 0.0:
                abort("Steel02 cR2 must be > 0")

            var a1 = 0.0
            var a2 = 1.0
            var a3 = 0.0
            var a4 = 1.0
            if has_a1:
                a1 = mat.a1
                a2 = mat.a2
                a3 = mat.a3
                a4 = mat.a4
            if a2 <= 0.0:
                abort("Steel02 a2 must be > 0")
            if a4 <= 0.0:
                abort("Steel02 a4 must be > 0")

            var sig_init = 0.0
            if has_siginit:
                sig_init = mat.sigInit

            var mat_def = UniMaterialDef(
                UniMaterialTypeTag.Steel02,
                Fy,
                E0,
                b,
                R0,
                cR1,
                cR2,
                a1,
                a2,
                a3,
                a4,
                sig_init,
            )
            uniaxial_def_by_id[mid] = len(uniaxial_defs)
            uniaxial_defs.append(mat_def)
        elif mat_type == "Concrete02":
            var fpc = mat.fpc
            var epsc0 = mat.epsc0
            var fpcu = mat.fpcu
            var epscu = mat.epscu
            if fpc > 0.0:
                fpc = -fpc
            if epsc0 > 0.0:
                epsc0 = -epsc0
            if fpcu > 0.0:
                fpcu = -fpcu
            if epscu > 0.0:
                epscu = -epscu
            if fpc >= 0.0:
                abort("Concrete02 fpc must be < 0")
            if epsc0 >= 0.0:
                abort("Concrete02 epsc0 must be < 0")
            if epscu >= 0.0:
                abort("Concrete02 epscu must be < 0")
            if epscu >= epsc0:
                abort("Concrete02 epscu must be < epsc0")
            if fpcu > 0.0 or fpcu < fpc:
                abort("Concrete02 fpcu must be between fpc and 0")

            var has_rat = mat.has_rat
            var has_ft = mat.has_ft
            var has_ets = mat.has_ets
            if has_rat != has_ft or has_rat != has_ets:
                abort("Concrete02 requires rat, ft, Ets together")

            var rat = 0.1
            var ft = 0.1 * fpc
            if ft < 0.0:
                ft = -ft
            var Ets = 0.1 * fpc / epsc0
            if has_rat:
                rat = mat.rat
                ft = mat.ft
                Ets = mat.Ets

            if rat == 1.0:
                abort("Concrete02 rat must not be 1")
            if Ets <= 0.0:
                abort("Concrete02 Ets must be > 0")

            var mat_def = UniMaterialDef(
                UniMaterialTypeTag.Concrete02,
                fpc,
                epsc0,
                fpcu,
                epscu,
                rat,
                ft,
                Ets,
                0.0,
                0.0,
                0.0,
                0.0,
            )
            uniaxial_def_by_id[mid] = len(uniaxial_defs)
            uniaxial_defs.append(mat_def)
        elif mat_type == "ModIMKPeakOriented":
            var mat_def = UniMaterialDef(UniMaterialTypeTag.ModIMKPeakOriented, mat.uniaxial_values)
            uniaxial_def_by_id[mid] = len(uniaxial_defs)
            uniaxial_defs.append(mat_def)
        elif mat_type == "Bilin" or mat_type == "Hysteretic":
            var tag = (
                UniMaterialTypeTag.Bilin if mat_type == "Bilin" else UniMaterialTypeTag.Hysteretic
            )
            var mat_def = UniMaterialDef(tag, mat.uniaxial_values)
            uniaxial_def_by_id[mid] = len(uniaxial_defs)
            uniaxial_defs.append(mat_def)
        elif mat_type == "ViscousDamper":
            var alpha = mat.alpha
            if alpha < 0.0:
                alpha = 1.0
            var mat_def = UniMaterialDef(
                UniMaterialTypeTag.ViscousDamper,
                mat.K,
                mat.C,
                alpha,
                mat.LGap,
                mat.NM,
                mat.RelTol,
                mat.AbsTol,
                mat.MaxHalf,
                0.0,
                0.0,
                0.0,
            )
            uniaxial_def_by_id[mid] = len(uniaxial_defs)
            uniaxial_defs.append(mat_def)
        elif (
            mat_type == "ConcreteCM"
            or mat_type == "SteelMPF"
            or mat_type == "ElasticPP"
            or mat_type == "ElasticBilin"
            or mat_type == "Ratchet"
        ):
            uniaxial_def_by_id[mid] = len(uniaxial_defs)
            uniaxial_defs.append(mat.direct_uniaxial_def)
        elif (
            mat_type == "Fatigue"
            or mat_type == "ElasticIsotropic"
            or mat_type == "FSAM"
            or mat_type == "ElasticCrossAnisotropic"
            or mat_type == "DruckerPrager"
            or mat_type == "PressureIndependMultiYield"
            or mat_type == "PressureDependMultiYield"
            or mat_type == "PressureDependMultiYield02"
            or mat_type == "InitialStateAnalysisWrapper"
            or mat_type == "ContactMaterial2D"
            or mat_type == "PlaneStressUserMaterial"
            or mat_type == "ConcreteS"
            or mat_type == "PlateFromPlaneStress"
            or mat_type == "PlateFiber"
            or mat_type == "PlateRebar"
        ):
            continue
        else:
            abort("unsupported material type: " + mat_type)

    for i in range(len(materials)):
        var mat = materials[i]
        if mat.type != "Fatigue":
            continue
        if mat.base_material < 0 or typed_materials_by_id[mat.base_material].type != "Steel02":
            abort("Fatigue requires a Steel02 base material")
        var mat_def = uniaxial_defs[uniaxial_def_by_id[mat.base_material]]
        mat_def.fatigue = FatigueParameters(
            mat.fatigue_dmax, mat.E0, mat.fatigue_slope, mat.strain_min, mat.strain_max
        )
        uniaxial_def_by_id[mat.id] = len(uniaxial_defs)
        uniaxial_defs.append(mat_def)

    for i in range(len(materials)):
        if materials[i].type == "Parallel" or materials[i].type == "MinMax":
            _ = _resolve_composite_definition(i, materials, uniaxial_defs, uniaxial_def_by_id)

    _write_run_progress(progress_path, "loading", "case", 0, 0, 2, load_step_count)

    var fiber_section_defs: List[FiberSection2dDef] = []
    var fiber_section_cells: List[FiberCell] = []
    var fiber_section_index_by_id: List[Int] = []
    fiber_section_index_by_id.resize(len(typed_sections_by_id), -1)
    var fiber_section3d_defs: List[FiberSection3dDef] = []
    var fiber_section3d_cells: List[FiberCell] = []
    var fiber_section3d_index_by_id: List[Int] = []
    fiber_section3d_index_by_id.resize(len(typed_sections_by_id), -1)
    var layered_shell_section_defs: List[LayeredShellSectionDef] = []
    var layered_shell_section_index_by_id: List[Int] = []
    var layered_shell_section_uniaxial_offsets: List[Int] = []
    layered_shell_section_index_by_id.resize(len(typed_sections_by_id), -1)
    for i in range(len(sections)):
        var sec = sections[i]
        var sec_type = sec.type
        var sid = sec.id
        if sec_type == "FiberSection2d":
            _ensure_id_slot(fiber_section_index_by_id, sid)
            append_fiber_section2d_from_input(
                sec,
                fiber_patches,
                fiber_layers,
                section_fibers,
                uniaxial_def_by_id,
                uniaxial_defs,
                fiber_section_defs,
                fiber_section_cells,
            )
            fiber_section_index_by_id[sid] = len(fiber_section_defs) - 1
        elif sec_type == "FiberSection3d":
            _ensure_id_slot(fiber_section3d_index_by_id, sid)
            append_fiber_section3d_from_input(
                sec,
                fiber_patches,
                fiber_layers,
                section_fibers,
                uniaxial_def_by_id,
                uniaxial_defs,
                fiber_section3d_defs,
                fiber_section3d_cells,
            )
            fiber_section3d_index_by_id[sid] = len(fiber_section3d_defs) - 1
            ref fiber = fiber_section3d_defs[fiber_section3d_index_by_id[sid]]
            ref mapped = typed_sections_by_id[sid]
            fiber.added_shear_flexibility_y = fiber_aggregated_shear_flexibility(
                mapped.shear_y_material, uniaxial_defs, uniaxial_def_by_id
            )
            fiber.added_shear_flexibility_z = fiber_aggregated_shear_flexibility(
                mapped.shear_z_material, uniaxial_defs, uniaxial_def_by_id
            )
        elif sec_type == "LayeredShellSection":
            _ensure_id_slot(layered_shell_section_index_by_id, sid)
            append_layered_shell_section_from_input(
                layered_shell_section_defs,
                sec,
                shell_layers,
                typed_materials_by_id,
                uniaxial_def_by_id,
            )
            layered_shell_section_index_by_id[sid] = len(layered_shell_section_defs) - 1

    # Resolve hinge ends after all fiber layouts and aggregator bases exist.
    for i in range(len(sections)):
        var sec = sections[i]
        var sid = sec.id
        if sec.type == "HingeRadauSection2d":
            var hinge = sec.hinge_radau
            hinge.section_i = _remap_compact_index_or_abort(
                hinge.section_i, section_index_by_raw_id, "beamWithHinges section I"
            )
            hinge.section_j = _remap_compact_index_or_abort(
                hinge.section_j, section_index_by_raw_id, "beamWithHinges section J"
            )
            if hinge.section_i < 0 or hinge.section_j < 0:
                abort("beamWithHinges requires both end sections")
            var i_end = typed_sections_by_id[hinge.section_i]
            var j_end = typed_sections_by_id[hinge.section_j]
            if i_end.type == "AggregatorSection2d" and i_end.base_section >= 0:
                i_end = _coerce_aggregator_section2d_for_beam_column(
                    i_end, typed_sections_by_id, uniaxial_defs, uniaxial_def_by_id, "beamWithHinges"
                )
            if j_end.type == "AggregatorSection2d" and j_end.base_section >= 0:
                j_end = _coerce_aggregator_section2d_for_beam_column(
                    j_end, typed_sections_by_id, uniaxial_defs, uniaxial_def_by_id, "beamWithHinges"
                )
            if i_end.type == "ElasticSection2d" and j_end.type == "ElasticSection2d":
                hinge.elastic_end_i = (i_end.E * i_end.A, i_end.E * i_end.I, i_end.shear_rigidity_y)
                hinge.elastic_end_j = (j_end.E * j_end.A, j_end.E * j_end.I, j_end.shear_rigidity_y)
                hinge.elastic_ea = sec.E * sec.A
                hinge.elastic_ei = sec.E * sec.I
                sec.type = "ElasticSection2d"
                sec.hinge_radau = hinge
                typed_sections_by_id[i] = sec
                continue
            (hinge.axial_i, hinge.flexural_i) = _hinge_end_materials(
                hinge.section_i,
                typed_sections_by_id,
                fiber_section_defs,
                fiber_section_index_by_id,
                uniaxial_defs,
                uniaxial_def_by_id,
            )
            (hinge.axial_j, hinge.flexural_j) = _hinge_end_materials(
                hinge.section_j,
                typed_sections_by_id,
                fiber_section_defs,
                fiber_section_index_by_id,
                uniaxial_defs,
                uniaxial_def_by_id,
            )
            hinge.elastic_ea = sec.E * sec.A
            hinge.elastic_ei = sec.E * sec.I
            if (
                hinge.elastic_ea <= 0.0
                or hinge.elastic_ei <= 0.0
                or hinge.length_i < 0.0
                or hinge.length_j < 0.0
                or hinge.max_iters <= 0
                or hinge.tolerance <= 0.0
            ):
                abort("invalid beamWithHinges properties")
            sec.hinge_radau = hinge
            typed_sections_by_id[i] = sec
            # The force-based solver dispatches section evaluation using
            # hinge_radau and does not use a fiber layout for these sections.
            fiber_section_index_by_id[sid] = len(fiber_section_defs)
            fiber_section_defs.append(FiberSection2dDef())

    _write_run_progress(progress_path, "loading", "case", 0, 0, 3, load_step_count)

    var elem_count = len(typed_elements)
    var elem_ids: List[Int] = []
    elem_ids.resize(elem_count, 0)
    var elem_id_to_index: List[Int] = []
    elem_id_to_index.resize(10000, -1)
    for i in range(elem_count):
        var eid = typed_elements[i].id
        elem_ids[i] = eid
        _ensure_id_slot(elem_id_to_index, eid)
        elem_id_to_index[eid] = i

    for i in range(elem_count):
        var elem = typed_elements[i]
        if elem.section >= 0:
            elem.section = _remap_compact_index_or_abort(
                elem.section, section_index_by_raw_id, "section"
            )
        if elem.material >= 0:
            elem.material = _remap_compact_index_or_abort(
                elem.material, material_index_by_raw_id, "material"
            )
        if elem.type_tag == ElementTypeTag.MVLEM:
            for j in range(len(elem.wall.fibers)):
                ref fiber = elem.wall.fibers[j]
                fiber.concrete = _remap_compact_index_or_abort(
                    fiber.concrete,
                    material_index_by_raw_id,
                    "MVLEM concrete material",
                )
                fiber.steel = _remap_compact_index_or_abort(
                    fiber.steel,
                    material_index_by_raw_id,
                    "MVLEM steel material",
                )
        if elem.type_tag == ElementTypeTag.SFIMVLEM:
            for j in range(len(elem.wall.fibers)):
                ref panel = elem.wall.fibers[j]
                panel.concrete = _remap_compact_index_or_abort(
                    panel.concrete, material_index_by_raw_id, "SFI_MVLEM panel material"
                )
        for m in range(elem.material_count):
            _set_elem_material(
                elem,
                m,
                _remap_compact_index_or_abort(
                    _elem_material(elem, m),
                    material_index_by_raw_id,
                    "element material",
                ),
            )
        for m in range(elem.damp_material_count):
            _set_elem_damp_material(
                elem,
                m,
                _remap_compact_index_or_abort(
                    _elem_damp_material(elem, m),
                    material_index_by_raw_id,
                    "element damp material",
                ),
            )
        typed_elements[i] = elem

    var continuum_indices: List[Int] = []
    var has_force_beam_column2d = False
    for i in range(elem_count):
        var elem = typed_elements[i]
        if elem.type_tag == ElementTypeTag.Unknown:
            abort("unsupported element type: " + elem.type)

        if elem.node_1 >= len(id_to_index) or id_to_index[elem.node_1] < 0:
            abort("element node not found")
        if elem.node_2 >= len(id_to_index) or id_to_index[elem.node_2] < 0:
            abort("element node not found")
        elem.node_index_1 = id_to_index[elem.node_1]
        elem.node_index_2 = id_to_index[elem.node_2]
        if elem.node_count >= 3:
            if elem.node_3 >= len(id_to_index) or id_to_index[elem.node_3] < 0:
                abort("element node not found")
            elem.node_index_3 = id_to_index[elem.node_3]
        if elem.node_count >= 4:
            if elem.node_4 >= len(id_to_index) or id_to_index[elem.node_4] < 0:
                abort("element node not found")
            elem.node_index_4 = id_to_index[elem.node_4]
        for j in range(4, elem.node_count):
            if elem.solid_extra_nodes:
                break
            var id = elem.extra_nodes[j - 4]
            if id < 0 or id >= len(id_to_index) or id_to_index[id] < 0:
                abort("element node not found")
            elem.extra_node_indices[j - 4] = id_to_index[id]

        if elem.type_tag == ElementTypeTag.ElasticBeamColumn2d:
            if ndm != 2 or ndf != 3:
                abort("elasticBeamColumn2d requires ndm=2, ndf=3")
            if elem.node_count != 2:
                abort("elasticBeamColumn2d requires 2 nodes")
            if elem.section < 0 or elem.section >= len(typed_sections_by_id):
                abort("section not found")
            var sec = typed_sections_by_id[elem.section]
            if sec.id < 0:
                abort("section not found")
            if sec.type != "ElasticSection2d":
                abort("elasticBeamColumn2d requires ElasticSection2d")
            if (
                elem.geom_tag != GeomTransfTag.Linear
                and elem.geom_tag != GeomTransfTag.PDelta
                and elem.geom_tag != GeomTransfTag.Corotational
            ):
                abort("unsupported geomTransf: " + elem.geom_transf)
            if elem.section < len(fiber_section_index_by_id):
                if fiber_section_index_by_id[elem.section] >= 0:
                    abort(
                        "elasticBeamColumn2d with FiberSection2d requires forceBeamColumn2d "
                        "or dispBeamColumn2d"
                    )
            elem.dof_count = 6
            _set_elem_dof(elem, 0, node_dof_index(elem.node_index_1, 1, ndf))
            _set_elem_dof(elem, 1, node_dof_index(elem.node_index_1, 2, ndf))
            _set_elem_dof(elem, 2, node_dof_index(elem.node_index_1, 3, ndf))
            _set_elem_dof(elem, 3, node_dof_index(elem.node_index_2, 1, ndf))
            _set_elem_dof(elem, 4, node_dof_index(elem.node_index_2, 2, ndf))
            _set_elem_dof(elem, 5, node_dof_index(elem.node_index_2, 3, ndf))
        elif elem.type_tag == ElementTypeTag.ForceBeamColumn2d:
            has_force_beam_column2d = True
            var beam_col_type = elem.type
            if ndm != 2 or ndf != 3:
                abort(beam_col_type + " requires ndm=2, ndf=3")
            if elem.node_count != 2:
                abort(beam_col_type + " requires 2 nodes")
            if (
                elem.geom_tag != GeomTransfTag.Linear
                and elem.geom_tag != GeomTransfTag.PDelta
                and elem.geom_tag != GeomTransfTag.Corotational
            ):
                abort(beam_col_type + " supports geomTransf Linear, PDelta, or Corotational")
            if elem.section < 0 or elem.section >= len(typed_sections_by_id):
                abort(beam_col_type + " section not found")
            var sec = typed_sections_by_id[elem.section]
            if sec.id < 0:
                abort(beam_col_type + " section not found")
            if sec.hinge_radau.is_defined():
                if elem.integration != "HingeRadau" or elem.num_int_pts != 6:
                    abort("beamWithHinges requires six HingeRadau integration points")
            else:
                beam_integration_validate_or_abort(
                    beam_col_type, elem.integration, elem.num_int_pts
                )
                sec = _coerce_aggregator_section2d_for_beam_column(
                    sec, typed_sections_by_id, uniaxial_defs, uniaxial_def_by_id, beam_col_type
                )
                typed_sections_by_id[elem.section] = sec
                if sec.type != "FiberSection2d" and sec.type != "ElasticSection2d":
                    abort(beam_col_type + " requires FiberSection2d or ElasticSection2d")
            elem.dof_count = 6
            if elem.cbdi.enabled():
                if sec.type != "ElasticSection2d" or sec.shear_rigidity_y != 0.0:
                    abort("forceBeamColumnCBDI requires an elastic section without shear")
                sec.id = len(typed_sections_by_id)
                sec.cbdi = elem.cbdi
                elem.section = sec.id
                typed_sections_by_id.append(sec)
            _set_elem_dof(elem, 0, node_dof_index(elem.node_index_1, 1, ndf))
            _set_elem_dof(elem, 1, node_dof_index(elem.node_index_1, 2, ndf))
            _set_elem_dof(elem, 2, node_dof_index(elem.node_index_1, 3, ndf))
            _set_elem_dof(elem, 3, node_dof_index(elem.node_index_2, 1, ndf))
            _set_elem_dof(elem, 4, node_dof_index(elem.node_index_2, 2, ndf))
            _set_elem_dof(elem, 5, node_dof_index(elem.node_index_2, 3, ndf))
        elif elem.type_tag == ElementTypeTag.DispBeamColumn2d:
            has_force_beam_column2d = True
            if ndm != 2 or ndf != 3:
                abort("dispBeamColumn2d requires ndm=2, ndf=3")
            if elem.node_count != 2:
                abort("dispBeamColumn2d requires 2 nodes")
            if (
                elem.geom_tag != GeomTransfTag.Linear
                and elem.geom_tag != GeomTransfTag.PDelta
                and elem.geom_tag != GeomTransfTag.Corotational
            ):
                abort("dispBeamColumn2d supports geomTransf Linear, PDelta, or Corotational")
            beam_integration_validate_or_abort(
                "dispBeamColumn2d", elem.integration, elem.num_int_pts
            )
            if elem.section < 0 or elem.section >= len(typed_sections_by_id):
                abort("dispBeamColumn2d section not found")
            var sec = typed_sections_by_id[elem.section]
            if sec.id < 0:
                abort("dispBeamColumn2d section not found")
            sec = _coerce_aggregator_section2d_for_beam_column(
                sec, typed_sections_by_id, uniaxial_defs, uniaxial_def_by_id, "dispBeamColumn2d"
            )
            typed_sections_by_id[elem.section] = sec
            if sec.type != "FiberSection2d" and sec.type != "ElasticSection2d":
                abort("dispBeamColumn2d requires FiberSection2d or ElasticSection2d")
            if elem.geom_tag == GeomTransfTag.Corotational and (
                sec.type != "ElasticSection2d" or sec.shear_rigidity_y == 0.0
            ):
                abort("Corotational dispBeamColumn2d requires an elastic shear section")
            elem.dof_count = 6
            _set_elem_dof(elem, 0, node_dof_index(elem.node_index_1, 1, ndf))
            _set_elem_dof(elem, 1, node_dof_index(elem.node_index_1, 2, ndf))
            _set_elem_dof(elem, 2, node_dof_index(elem.node_index_1, 3, ndf))
            _set_elem_dof(elem, 3, node_dof_index(elem.node_index_2, 1, ndf))
            _set_elem_dof(elem, 4, node_dof_index(elem.node_index_2, 2, ndf))
            _set_elem_dof(elem, 5, node_dof_index(elem.node_index_2, 3, ndf))
        elif elem.type_tag == ElementTypeTag.ElasticBeamColumn3d:
            if ndm != 3 or ndf != 6:
                abort("elasticBeamColumn3d requires ndm=3, ndf=6")
            if elem.node_count != 2:
                abort("elasticBeamColumn3d requires 2 nodes")
            if (
                elem.geom_tag != GeomTransfTag.Linear
                and elem.geom_tag != GeomTransfTag.PDelta
                and elem.geom_tag != GeomTransfTag.Corotational
            ):
                abort("elasticBeamColumn3d supports geomTransf Linear, PDelta, or Corotational")
            if elem.section < 0 or elem.section >= len(typed_sections_by_id):
                abort("section not found")
            var sec = typed_sections_by_id[elem.section]
            if sec.id < 0:
                abort("section not found")
            if sec.type != "ElasticSection3d":
                abort("elasticBeamColumn3d requires ElasticSection3d")
            if elem.section < len(fiber_section_index_by_id):
                if fiber_section_index_by_id[elem.section] >= 0:
                    abort(
                        "elasticBeamColumn3d with FiberSection2d requires forceBeamColumn2d "
                        "or dispBeamColumn2d"
                    )
            if elem.section < len(fiber_section3d_index_by_id):
                if fiber_section3d_index_by_id[elem.section] >= 0:
                    abort("elasticBeamColumn3d requires ElasticSection3d")
            elem.dof_count = 12
            for d in range(6):
                _set_elem_dof(elem, d, node_dof_index(elem.node_index_1, d + 1, ndf))
                _set_elem_dof(elem, d + 6, node_dof_index(elem.node_index_2, d + 1, ndf))
        elif (
            elem.type_tag == ElementTypeTag.ForceBeamColumn3d
            or elem.type_tag == ElementTypeTag.DispBeamColumn3d
        ):
            var beam_col_type = elem.type
            if ndm != 3 or ndf != 6:
                abort(beam_col_type + " requires ndm=3, ndf=6")
            if elem.node_count != 2:
                abort(beam_col_type + " requires 2 nodes")
            if (
                elem.geom_tag != GeomTransfTag.Linear
                and elem.geom_tag != GeomTransfTag.PDelta
                and elem.geom_tag != GeomTransfTag.Corotational
            ):
                abort(beam_col_type + " supports geomTransf Linear, PDelta, or Corotational")
            beam_integration_validate_or_abort(beam_col_type, elem.integration, elem.num_int_pts)
            if elem.section < 0 or elem.section >= len(typed_sections_by_id):
                abort(beam_col_type + " section not found")
            var sec = typed_sections_by_id[elem.section]
            if sec.id < 0:
                abort(beam_col_type + " section not found")
            if sec.type == "AggregatorSection3d":
                if sec.base_section >= 0:
                    sec = _coerce_aggregator_section3d_for_beam_column(
                        sec, typed_sections_by_id, uniaxial_defs, uniaxial_def_by_id
                    )
                else:
                    sec = elastic_aggregator_3d(sec, uniaxial_defs, uniaxial_def_by_id)
                typed_sections_by_id[elem.section] = sec
            if sec.type != "ElasticSection3d" and sec.type != "FiberSection3d":
                abort(beam_col_type + " requires ElasticSection3d or FiberSection3d")
            if sec.type == "FiberSection3d" and (sec.G <= 0.0 or sec.J <= 0.0):
                abort(beam_col_type + " with FiberSection3d requires positive G and J")
            if elem.cbdi.enabled() and (
                elem.type_tag != ElementTypeTag.ForceBeamColumn3d
                or sec.type != "ElasticSection3d"
                or sec.shear_rigidity_y != 0.0
                or sec.shear_rigidity_z != 0.0
            ):
                abort("forceBeamColumnCBDI requires an elastic section without shear")
            elem.dof_count = 12
            for d in range(6):
                _set_elem_dof(elem, d, node_dof_index(elem.node_index_1, d + 1, ndf))
                _set_elem_dof(elem, d + 6, node_dof_index(elem.node_index_2, d + 1, ndf))
        elif elem.type_tag == ElementTypeTag.Truss:
            if elem.node_count != 2:
                abort("truss requires 2 nodes")
            if ndf != 2 and ndf != 3 and ndf != 6:
                abort("truss requires ndf=2, ndf=3, or ndf=6")
            if elem.area <= 0.0:
                abort("truss requires area > 0")
            if ndf == 2:
                elem.dof_count = 4
                _set_elem_dof(elem, 0, node_dof_index(elem.node_index_1, 1, ndf))
                _set_elem_dof(elem, 1, node_dof_index(elem.node_index_1, 2, ndf))
                _set_elem_dof(elem, 2, node_dof_index(elem.node_index_2, 1, ndf))
                _set_elem_dof(elem, 3, node_dof_index(elem.node_index_2, 2, ndf))
            else:
                elem.dof_count = 6
                _set_elem_dof(elem, 0, node_dof_index(elem.node_index_1, 1, ndf))
                _set_elem_dof(elem, 1, node_dof_index(elem.node_index_1, 2, ndf))
                _set_elem_dof(elem, 2, node_dof_index(elem.node_index_1, 3, ndf))
                _set_elem_dof(elem, 3, node_dof_index(elem.node_index_2, 1, ndf))
                _set_elem_dof(elem, 4, node_dof_index(elem.node_index_2, 2, ndf))
                _set_elem_dof(elem, 5, node_dof_index(elem.node_index_2, 3, ndf))
        elif elem.type_tag == ElementTypeTag.ZeroLength:
            if elem.node_count != 2:
                abort("zeroLength requires 2 nodes")
            if elem.type == "CoupledZeroLength":
                if elem.material_count != 1 or elem.dir_count != 2:
                    abort("CoupledZeroLength requires one material and two directions")
                var first = _elem_dir(elem, 0)
                var second = _elem_dir(elem, 1)
                if first < 1 or first > ndf or second < 1 or second > ndf or first == second:
                    abort("CoupledZeroLength requires two distinct nodal DOFs")
            elif elem.material_count != elem.dir_count:
                abort("zeroLength materials/dirs mismatch")
            for j in range(elem.material_count):
                var id = _elem_material(elem, j)
                if id >= 0 and id < len(typed_materials_by_id):
                    if typed_materials_by_id[id].type == "Viscous":
                        elem.has_viscous_material = True
            if elem.damp_material_count > 0 and elem.damp_material_count != elem.material_count:
                abort("zeroLength dampMats/materials mismatch")
            for m in range(elem.damp_material_count):
                var damp_mat_id = _elem_damp_material(elem, m)
                if damp_mat_id < 0 or damp_mat_id >= len(typed_materials_by_id):
                    abort("zeroLength damp material not found")
                var damp_mat = typed_materials_by_id[damp_mat_id]
                if damp_mat.id < 0:
                    abort("zeroLength damp material not found")
                if damp_mat.type != "Elastic":
                    abort("zeroLength dampMats currently require Elastic materials")
            if elem.damping_tag >= 0:
                if elem.damp_material_count > 0:
                    abort("zeroLength does not support both damp and dampMats")
            elem.dof_count = link_element_dof_count(ndm, ndf)
            if elem.type == "zeroLength":
                for d in range(elem.dir_count):
                    _set_elem_dir(
                        elem,
                        d,
                        zero_length_internal_dir(_elem_dir(elem, d), ndm, ndf),
                    )
            for d in range(ndf):
                _set_elem_dof(elem, d, node_dof_index(elem.node_index_1, d + 1, ndf))
                _set_elem_dof(elem, d + ndf, node_dof_index(elem.node_index_2, d + 1, ndf))
        elif elem.type_tag == ElementTypeTag.MVLEM or elem.type_tag == ElementTypeTag.SFIMVLEM:
            if ndm != 2 or ndf != 3 or elem.node_count != 2:
                abort("MVLEM requires two 2D, 3-DOF nodes")
            var bottom = typed_nodes[elem.node_index_1]
            var top = typed_nodes[elem.node_index_2]
            if top.x != bottom.x or top.y <= bottom.y:
                abort("MVLEM requires a vertical element with nodes ordered bottom to top")
            if len(elem.wall.fibers) == 0:
                abort("MVLEM requires at least one fiber")
            var height = top.y - bottom.y
            # OpenSees 3.7.1 computes the element mass before setDomain
            # assigns the height, leaving the translational mass at zero.
            _ = elem.wall.set_height(height)
            elem.dof_count = 6
            if elem.type_tag == ElementTypeTag.SFIMVLEM:
                elem.wall_internal_node_start = len(typed_nodes)
                elem.dof_count += len(elem.wall.fibers)
                for j in range(len(elem.wall.fibers)):
                    var tag = -(elem.id * 1000 + j + 1)
                    var x = bottom.x + elem.wall.fibers[j].center
                    var node = NodeInput(tag, x, 0.5 * (bottom.y + top.y), 0.0, False)
                    node.ndf = 1
                    typed_nodes.append(node)
                    node_ids.append(tag)

            for d in range(3):
                _set_elem_dof(elem, d, node_dof_index(elem.node_index_1, d + 1, ndf))
                _set_elem_dof(elem, d + 3, node_dof_index(elem.node_index_2, d + 1, ndf))
        elif elem.type_tag == ElementTypeTag.CatenaryCable:
            if ndm != 3 or ndf != 3 or elem.node_count != 2:
                abort("CatenaryCable requires two 3D, 3-DOF nodes")
            if (
                elem.cable.modulus <= 0.0
                or elem.cable.area <= 0.0
                or elem.cable.tolerance <= 0.0
                or elem.cable.substeps < 1
            ):
                abort("invalid CatenaryCable stiffness or iteration controls")
            if elem.cable.mass_type < 0 or elem.cable.mass_type > 1:
                abort("CatenaryCable supports lumped or integration-fallback mass")
            if elem.cable.length <= 0.0:
                ref first = typed_nodes[elem.node_index_1]
                ref last = typed_nodes[elem.node_index_2]
                var dx = last.x - first.x
                var dy = last.y - first.y
                var dz = last.z - first.z
                elem.cable.length = sqrt(dx * dx + dy * dy + dz * dz)
                if elem.cable.length == 0.0:
                    abort("CatenaryCable unstretched length must be positive")
            elem.dof_count = 6
            for d in range(3):
                _set_elem_dof(elem, d, node_dof_index(elem.node_index_1, d + 1, ndf))
                _set_elem_dof(elem, d + 3, node_dof_index(elem.node_index_2, d + 1, ndf))
        elif elem.type_tag == ElementTypeTag.TwoNodeLink:
            if elem.node_count != 2:
                abort("twoNodeLink requires 2 nodes")
            if elem.material_count != elem.dir_count:
                abort("twoNodeLink materials/dirs mismatch")
            if elem.damp_material_count > 0:
                abort("twoNodeLink does not support dampMats")
            if elem.damping_tag >= 0:
                abort("twoNodeLink does not support damp")
            elem.dof_count = link_element_dof_count(ndm, ndf)
            for d in range(elem.dir_count):
                _set_elem_dir(
                    elem,
                    d,
                    two_node_link_internal_dir(_elem_dir(elem, d), ndm, ndf),
                )
            for d in range(ndf):
                _set_elem_dof(elem, d, node_dof_index(elem.node_index_1, d + 1, ndf))
                _set_elem_dof(elem, d + ndf, node_dof_index(elem.node_index_2, d + 1, ndf))
        elif elem.type_tag == ElementTypeTag.ZeroLengthSection:
            if not ((ndm == 2 and ndf == 3) or (ndm == 3 and ndf == 6)):
                abort("zeroLengthSection requires (ndm=2, ndf=3) or (ndm=3, ndf=6)")
            if elem.node_count != 2:
                abort("zeroLengthSection requires 2 nodes")
            if elem.section < 0 or elem.section >= len(typed_sections_by_id):
                abort("zeroLengthSection section not found")
            var sec = typed_sections_by_id[elem.section]
            if sec.id < 0:
                abort("zeroLengthSection section not found")
            if ndm == 2 and ndf == 3:
                if (
                    sec.type != "FiberSection2d"
                    and sec.type != "ElasticSection2d"
                    and sec.type != "AggregatorSection2d"
                ):
                    abort(
                        "zeroLengthSection (2D) requires FiberSection2d, ElasticSection2d, or"
                        " AggregatorSection2d"
                    )
                elem.dof_count = 6
                for d in range(3):
                    _set_elem_dof(elem, d, node_dof_index(elem.node_index_1, d + 1, ndf))
                    _set_elem_dof(elem, d + 3, node_dof_index(elem.node_index_2, d + 1, ndf))
            else:
                if sec.type == "FiberSection3d" and (
                    sec.shear_y_material >= 0 or sec.shear_z_material >= 0
                ):
                    abort(
                        "zeroLengthSection does not support shear additions to a 3D fiber section"
                    )
                if (
                    sec.type != "FiberSection3d"
                    and sec.type != "ElasticSection3d"
                    and sec.type != "AggregatorSection3d"
                ):
                    abort(
                        "zeroLengthSection (3D) requires FiberSection3d, ElasticSection3d, or"
                        " AggregatorSection3d"
                    )
                elem.dof_count = 12
                for d in range(6):
                    _set_elem_dof(elem, d, node_dof_index(elem.node_index_1, d + 1, ndf))
                    _set_elem_dof(elem, d + 6, node_dof_index(elem.node_index_2, d + 1, ndf))
        elif elem.type_tag == ElementTypeTag.SSPBrickUP:
            if ndm != 3 or ndf != 4:
                abort("SSPbrickUP requires ndm=3, ndf=4")
            if elem.material < 0 or elem.material >= len(typed_materials_by_id):
                abort("SSPbrickUP material not found")
            ref mat = typed_materials_by_id[elem.material]
            if mat.type != "ElasticIsotropic" and not (
                mat.pdmy_index >= 0 and mat.soil_dimensions == 3
            ):
                abort("SSPbrickUP requires a supported three-dimensional soil material")
            if elem.up_bulk <= 0.0 or elem.up_void_ratio < 0.0 or elem.up_stabilization < 0.0:
                abort(
                    "SSPbrickUP requires positive fluid bulk modulus and nonnegative void ratio and"
                    " stabilization"
                )
            if elem.quad_density < 0.0 or min(elem.up_perm_x, elem.up_perm_y, elem.up_perm_z) < 0.0:
                abort("SSPbrickUP requires nonnegative fluid density and permeabilities")
            elem.dof_count = 32
            for j in range(8):
                var node = _elem_node(elem, j)
                if typed_nodes[node].ndf != 4:
                    abort("SSPbrickUP requires four DOFs at each node")
                for d in range(4):
                    _set_elem_dof(elem, 4 * j + d, node_dof_index(node, d + 1, ndf))
        elif elem.type_tag == ElementTypeTag.QuadUP:
            if ndm != 2 or ndf != 3:
                abort(elem.type + " requires ndm=2, ndf=3")
            if elem.material < 0 or elem.material >= len(typed_materials_by_id):
                abort("u-p quadrilateral material not found")
            ref mat = typed_materials_by_id[elem.material]
            if mat.type == "InitialStateAnalysisWrapper":
                abort("InitialStateAnalysisWrapper requires a fourNodeQuad")
            if mat.type != "ElasticIsotropic" and mat.pimy_index < 0 and mat.pdmy_index < 0:
                abort("u-p quadrilateral requires ElasticIsotropic or a supported soil material")
            if mat.type != "ElasticIsotropic" and mat.soil_dimensions != 2:
                abort("u-p quadrilateral requires plane strain soil material")
            if elem.thickness <= 0.0 or elem.up_bulk <= 0.0:
                abort("u-p quadrilateral requires positive thickness and fluid bulk modulus")
            if elem.quad_density < 0.0 or elem.up_perm_x < 0.0 or elem.up_perm_y < 0.0:
                abort("u-p quadrilateral requires nonnegative fluid density and permeabilities")
            elem.dof_count = 12 + 2 * (elem.node_count - 4)
            for j in range(elem.node_count):
                var node = _elem_node(elem, j)
                var local_ndf = 3 if j < 4 else 2
                if typed_nodes[node].ndf != local_ndf:
                    abort("u-p quadrilateral requires 3 DOF at corners and 2 DOF elsewhere")
                for d in range(local_ndf):
                    _set_elem_dof(
                        elem, quad_up_displacement_dof(j) + d, node_dof_index(node, d + 1, ndf)
                    )
        elif elem.type_tag == ElementTypeTag.Continuum:
            if ndm != continuum_dimension(elem.type) or ndf != ndm:
                abort(elem.type + " requires matching solid model dimensions")
            if elem.material < 0:
                abort("continuum element material not found")
            ref material = typed_materials_by_id[elem.material]
            if material.drucker_prager:
                ref definition = material.drucker_prager.value()
                if elem.type != "stdBrick" or definition.bulk <= 0.0 or definition.shear <= 0.0:
                    abort("DruckerPrager requires stdBrick and positive elastic moduli")
            elif material.type != "ElasticIsotropic" and (
                material.type != "ElasticCrossAnisotropic" or ndm != 3 or elem.type == "quad3d"
            ):
                abort("unsupported continuum element material: " + material.type)
            if ndm == 2 or elem.type == "quad3d":
                if elem.formulation != "PlaneStress" and elem.formulation != "PlaneStrain":
                    abort(elem.type + " requires PlaneStress or PlaneStrain formulation")
                if elem.thickness <= 0.0:
                    abort(elem.type + " requires positive thickness")
            if elem.solid_extra_nodes:
                var nodes = elem.solid_extra_nodes.value().copy()
                for n in range(len(nodes)):
                    nodes[n] = _remap_compact_index_or_abort(nodes[n], id_to_index, "element node")
                    if nodes[n] < 0:
                        abort("element node not found")
                elem.solid_extra_nodes = nodes^
            elem.dof_count = ndf * elem.node_count
            continuum_indices.append(i)
        elif elem.type_tag == ElementTypeTag.BeamContact2D:
            if ndm != 2 or ndf != 3 or elem.node_count != 4:
                abort(
                    "BeamContact2D requires two planar beam nodes, a secondary node and a"
                    " multiplier node"
                )
            if (
                elem.material < 0
                or typed_materials_by_id[elem.material].type != "ContactMaterial2D"
            ):
                abort("BeamContact2D requires ContactMaterial2D")
            if (
                elem.thickness <= 0.0
                or elem.contact_gap_tol <= 0.0
                or elem.contact_force_tol <= 0.0
            ):
                abort("BeamContact2D requires positive width and tolerances")
            ref a = typed_nodes[elem.node_index_1]
            ref b = typed_nodes[elem.node_index_2]
            if a.x == b.x and a.y == b.y:
                abort("BeamContact2D requires distinct beam-end coordinates")
            elem.dof_count = 10
            var local_dof = 0
            for j in range(4):
                var node = _elem_node(elem, j)
                var count = 3 if j < 2 else 2
                if typed_nodes[node].ndf != count:
                    abort("BeamContact2D nodes must have 3, 3, 2 and 2 DOFs")
                for d in range(count):
                    _set_elem_dof(elem, local_dof, node_dof_index(node, d + 1, ndf))
                    local_dof += 1
        elif elem.type_tag == ElementTypeTag.FourNodeQuad or elem.type_tag == ElementTypeTag.Tri31:
            if ndm != 2:
                abort(elem.type + " requires ndm=2")
            var is_triangle = elem.type_tag == ElementTypeTag.Tri31
            var expected_nodes = 3 if is_triangle else 4
            if elem.node_count != expected_nodes:
                abort(elem.type + " has an invalid node count")
            if elem.formulation != "PlaneStress" and elem.formulation != "PlaneStrain":
                abort("fourNodeQuad requires PlaneStress or PlaneStrain formulation")
            if elem.material < 0 or elem.material >= len(typed_materials_by_id):
                abort("material not found")
            var mat = typed_materials_by_id[elem.material]
            if mat.id < 0:
                abort("material not found")
            if mat.pimy_index >= 0 or mat.pdmy_index >= 0:
                if is_triangle or elem.formulation != "PlaneStrain" or mat.soil_dimensions != 2:
                    abort("fourNodeQuad soil material requires 2D PlaneStrain")
            elif mat.type != "ElasticIsotropic":
                abort("fourNodeQuad requires ElasticIsotropic or a supported soil material")
            if elem.thickness <= 0.0:
                abort(elem.type + " requires thickness > 0")
            elem.dof_count = 2 * expected_nodes
            var node_indices = [
                elem.node_index_1,
                elem.node_index_2,
                elem.node_index_3,
                elem.node_index_4,
            ]
            for local_node in range(expected_nodes):
                if typed_nodes[node_indices[local_node]].ndf != 2:
                    abort(elem.type + " requires two DOFs at each node")
                for d in range(2):
                    _set_elem_dof(
                        elem,
                        2 * local_node + d,
                        node_dof_index(node_indices[local_node], d + 1, ndf),
                    )
        elif elem.type_tag == ElementTypeTag.Shell:
            if ndm != 3 or ndf != 6:
                abort("shell requires ndm=3, ndf=6")
            var expected_nodes = 3 if elem.type == "ShellDKGT" else 4
            if elem.node_count != expected_nodes:
                abort("incorrect node count for shell element")
            if elem.section < 0 or elem.section >= len(typed_sections_by_id):
                abort("section not found")
            var sec = typed_sections_by_id[elem.section]
            if sec.id < 0:
                abort("section not found")
            if (
                sec.type != "ElasticMembranePlateSection"
                and sec.type != "LayeredShellSection"
                and sec.type != "PlateFiber"
            ):
                abort(
                    "shell requires ElasticMembranePlateSection, LayeredShellSection or PlateFiber"
                )
            elem.dof_count = 6 * expected_nodes
            for d in range(6):
                _set_elem_dof(elem, d, node_dof_index(elem.node_index_1, d + 1, ndf))
                _set_elem_dof(elem, d + 6, node_dof_index(elem.node_index_2, d + 1, ndf))
                _set_elem_dof(elem, d + 12, node_dof_index(elem.node_index_3, d + 1, ndf))
                if expected_nodes == 4:
                    _set_elem_dof(elem, d + 18, node_dof_index(elem.node_index_4, d + 1, ndf))
        else:
            abort("unsupported element type: " + elem.type)

        typed_elements[i] = elem

    node_count = len(typed_nodes)

    var elem_dof_offsets: List[Int] = []
    elem_dof_offsets.resize(elem_count + 1, 0)
    var total_elem_dofs = 0
    for e in range(elem_count):
        elem_dof_offsets[e] = total_elem_dofs
        total_elem_dofs += typed_elements[e].dof_count
    elem_dof_offsets[elem_count] = total_elem_dofs
    var elem_dof_pool: List[Int] = []
    elem_dof_pool.resize(total_elem_dofs, -1)
    for e in range(elem_count):
        var elem = typed_elements[e]
        var offset = elem_dof_offsets[e]
        if elem.type_tag == ElementTypeTag.Continuum:
            for n in range(elem.node_count):
                for d in range(ndf):
                    elem_dof_pool[offset + n * ndf + d] = node_dof_index(
                        _elem_node(elem, n), d + 1, ndf
                    )
        else:
            for d in range(elem.dof_count):
                elem_dof_pool[offset + d] = _elem_dof(elem, d)

    var node_x: List[Float64] = []
    var node_y: List[Float64] = []
    var node_z: List[Float64] = []
    var node_has_z: List[Bool] = []
    node_x.resize(node_count, 0.0)
    node_y.resize(node_count, 0.0)
    node_z.resize(node_count, 0.0)
    node_has_z.resize(node_count, False)
    var node_constraint_offsets: List[Int] = []
    node_constraint_offsets.resize(node_count + 1, 0)
    var total_node_constraints = 0
    for i in range(node_count):
        var node = typed_nodes[i]
        node_x[i] = node.x
        node_y[i] = node.y
        node_z[i] = node.z
        node_has_z[i] = node.has_z
        node_constraint_offsets[i] = total_node_constraints
        total_node_constraints += node.constraint_count
    node_constraint_offsets[node_count] = total_node_constraints
    var node_constraint_pool: List[Int] = []
    node_constraint_pool.resize(total_node_constraints, 0)
    for i in range(node_count):
        var node = typed_nodes[i]
        var offset = node_constraint_offsets[i]
        for c in range(node.constraint_count):
            node_constraint_pool[offset + c] = _node_constraint(node, c)

    var elem_node_offsets: List[Int] = []
    elem_node_offsets.resize(elem_count + 1, 0)
    var total_elem_nodes = 0
    for e in range(elem_count):
        elem_node_offsets[e] = total_elem_nodes
        total_elem_nodes += typed_elements[e].node_count
    elem_node_offsets[elem_count] = total_elem_nodes
    var elem_node_pool: List[Int] = []
    elem_node_pool.resize(total_elem_nodes, -1)
    for e in range(elem_count):
        var elem = typed_elements[e]
        var offset = elem_node_offsets[e]
        for n in range(elem.node_count):
            elem_node_pool[offset + n] = _elem_node(elem, n)

    var elem_material_offsets: List[Int] = []
    elem_material_offsets.resize(elem_count + 1, 0)
    var total_elem_materials = 0
    for e in range(elem_count):
        elem_material_offsets[e] = total_elem_materials
        var elem = typed_elements[e]
        if elem.material_count > 0:
            total_elem_materials += elem.material_count
        elif elem.material >= 0:
            total_elem_materials += 1
    elem_material_offsets[elem_count] = total_elem_materials
    var elem_material_pool: List[Int] = []
    elem_material_pool.resize(total_elem_materials, -1)
    for e in range(elem_count):
        var elem = typed_elements[e]
        var offset = elem_material_offsets[e]
        if elem.material_count > 0:
            for m in range(elem.material_count):
                elem_material_pool[offset + m] = _elem_material(elem, m)
        elif elem.material >= 0:
            elem_material_pool[offset] = elem.material

    var elem_type_tags: List[Int] = []
    var elem_geom_tags: List[Int] = []
    var elem_section_ids: List[Int] = []
    var shell_elem_instance_offsets: List[Int] = []
    var elem_integration_tags: List[Int] = []
    var elem_num_int_pts: List[Int] = []
    var elem_primary_material_ids: List[Int] = []
    var elem_dof_counts: List[Int] = []
    var elem_area: List[Float64] = []
    var elem_thickness: List[Float64] = []
    var frame2d_elem_indices: List[Int] = []
    var frame3d_elem_indices: List[Int] = []
    var truss_elem_indices: List[Int] = []
    var zero_length_elem_indices: List[Int] = []
    var two_node_link_elem_indices: List[Int] = []
    var zero_length_section_elem_indices: List[Int] = []
    var quad_elem_indices: List[Int] = []
    var shell_elem_indices: List[Int] = []
    elem_type_tags.resize(elem_count, ElementTypeTag.Unknown)
    elem_geom_tags.resize(elem_count, GeomTransfTag.Unknown)
    elem_section_ids.resize(elem_count, -1)
    shell_elem_instance_offsets.resize(elem_count, -1)
    elem_integration_tags.resize(elem_count, BeamIntegrationTag.Unknown)
    elem_num_int_pts.resize(elem_count, 0)
    elem_primary_material_ids.resize(elem_count, -1)
    elem_dof_counts.resize(elem_count, 0)
    elem_area.resize(elem_count, 0.0)
    elem_thickness.resize(elem_count, 0.0)
    for e in range(elem_count):
        var elem = typed_elements[e]
        elem_type_tags[e] = elem.type_tag
        elem_geom_tags[e] = elem.geom_tag
        elem_section_ids[e] = elem.section
        elem_integration_tags[e] = beam_integration_tag(elem.integration)
        elem_num_int_pts[e] = elem.num_int_pts
        elem_primary_material_ids[e] = elem.material
        elem_dof_counts[e] = elem.dof_count
        elem_area[e] = elem.area
        elem_thickness[e] = elem.thickness
        if (
            elem.type_tag == ElementTypeTag.ElasticBeamColumn2d
            or elem.type_tag == ElementTypeTag.ForceBeamColumn2d
            or elem.type_tag == ElementTypeTag.DispBeamColumn2d
        ):
            frame2d_elem_indices.append(e)
        elif (
            elem.type_tag == ElementTypeTag.ElasticBeamColumn3d
            or elem.type_tag == ElementTypeTag.ForceBeamColumn3d
            or elem.type_tag == ElementTypeTag.DispBeamColumn3d
        ):
            frame3d_elem_indices.append(e)
        elif elem.type_tag == ElementTypeTag.Truss:
            truss_elem_indices.append(e)
        elif elem.type_tag == ElementTypeTag.ZeroLength:
            zero_length_elem_indices.append(e)
        elif (
            elem.type_tag == ElementTypeTag.TwoNodeLink
            or elem.type_tag == ElementTypeTag.MVLEM
            or elem.type_tag == ElementTypeTag.SFIMVLEM
            or elem.type_tag == ElementTypeTag.CatenaryCable
        ):
            two_node_link_elem_indices.append(e)
        elif elem.type_tag == ElementTypeTag.ZeroLengthSection:
            zero_length_section_elem_indices.append(e)
        elif (
            elem.type_tag == ElementTypeTag.FourNodeQuad
            or elem.type_tag == ElementTypeTag.Tri31
            or elem.type_tag == ElementTypeTag.QuadUP
            or elem.type_tag == ElementTypeTag.SSPBrickUP
            or elem.type_tag == ElementTypeTag.BeamContact2D
        ):
            quad_elem_indices.append(e)
        elif elem.type_tag == ElementTypeTag.Shell:
            shell_elem_indices.append(e)

    var uniaxial_states: List[UniMaterialState] = []
    var uniaxial_state_defs: List[Int] = []
    var elem_uniaxial_offsets: List[Int] = []
    var elem_uniaxial_counts: List[Int] = []
    var elem_uniaxial_state_ids: List[Int] = []
    var force_basic_offsets: List[Int] = []
    var force_basic_counts: List[Int] = []
    var force_basic_q: List[Float64] = []
    elem_uniaxial_offsets.resize(elem_count, 0)
    elem_uniaxial_counts.resize(elem_count, 0)
    force_basic_offsets.resize(elem_count, 0)
    force_basic_counts.resize(elem_count, 0)
    var fiber3d_instance_counts: List[Int] = []
    if len(fiber_section3d_defs) > 0:
        fiber3d_instance_counts.resize(len(fiber_section3d_defs), 0)
        for e in range(elem_count):
            ref elem = typed_elements[e]
            if (
                elem.type_tag == ElementTypeTag.ForceBeamColumn3d
                or elem.type_tag == ElementTypeTag.DispBeamColumn3d
                or elem.type_tag == ElementTypeTag.ZeroLengthSection
            ):
                var sec_id = elem.section
                if sec_id >= 0 and sec_id < len(typed_sections_by_id):
                    ref sec = typed_sections_by_id[sec_id]
                    if sec.type == "FiberSection3d":
                        var sec_index = fiber_section3d_index_by_id[sec_id]
                        if sec_index >= 0 and sec_index < len(fiber_section3d_defs):
                            if elem.type_tag == ElementTypeTag.ZeroLengthSection:
                                fiber3d_instance_counts[sec_index] += 1
                            else:
                                fiber3d_instance_counts[sec_index] += elem.num_int_pts
        for s in range(len(fiber_section3d_defs)):
            if fiber3d_instance_counts[s] > 0:
                fiber_section3d_runtime_reserve_instances(
                    fiber_section3d_defs[s], fiber3d_instance_counts[s]
                )

    var used_nonelastic_uniaxial = False
    var force_beam_has_nonelastic = False
    for e in range(elem_count):
        var elem = typed_elements[e]
        force_basic_offsets[e] = len(force_basic_q)
        force_basic_counts[e] = 0
        if elem.type_tag == ElementTypeTag.Truss:
            var mat_id = elem.material
            if mat_id >= len(uniaxial_def_by_id) or uniaxial_def_by_id[mat_id] < 0:
                abort("truss requires uniaxial material")
            var def_index = uniaxial_def_by_id[mat_id]
            var mat_def = uniaxial_defs[def_index]
            if mat_def.mat_type == UniMaterialTypeTag.Viscous:
                abort("Viscous currently requires a zeroLength element")
            var state_index = len(uniaxial_states)
            uniaxial_states.append(UniMaterialState(mat_def))
            uniaxial_state_defs.append(def_index)
            elem_uniaxial_offsets[e] = len(elem_uniaxial_state_ids)
            elem_uniaxial_counts[e] = 1
            elem_uniaxial_state_ids.append(state_index)
            if not uni_mat_is_elastic(mat_def):
                used_nonelastic_uniaxial = True
        elif elem.type_tag == ElementTypeTag.ZeroLength:
            elem_uniaxial_offsets[e] = len(elem_uniaxial_state_ids)
            elem_uniaxial_counts[e] = elem.material_count
            for m in range(elem.material_count):
                var mat_id = _elem_material(elem, m)
                if mat_id >= len(uniaxial_def_by_id) or uniaxial_def_by_id[mat_id] < 0:
                    abort("zeroLength requires uniaxial material")
                var def_index = uniaxial_def_by_id[mat_id]
                var mat_def = uniaxial_defs[def_index]
                var state_index = len(uniaxial_states)
                var point = UniMaterialState(mat_def)
                if elem.type == "CoupledZeroLength":
                    if (
                        mat_def.mat_type == UniMaterialTypeTag.Viscous
                        or mat_def.mat_type == UniMaterialTypeTag.ViscousDamper
                    ):
                        abort("CoupledZeroLength requires a rate-independent material")
                    point.coupled_force = CoupledForceHistory()
                uniaxial_states.append(point^)
                uniaxial_state_defs.append(def_index)
                elem_uniaxial_state_ids.append(state_index)
                if mat_def.mat_type != UniMaterialTypeTag.Viscous and not uni_mat_is_elastic(
                    mat_def
                ):
                    used_nonelastic_uniaxial = True
        elif elem.type_tag == ElementTypeTag.MVLEM:
            elem_uniaxial_offsets[e] = len(elem_uniaxial_state_ids)
            var fiber_count = len(elem.wall.fibers)
            elem_uniaxial_counts[e] = 2 * fiber_count + 1
            for m in range(elem_uniaxial_counts[e]):
                var mat_id = elem.material
                if m < fiber_count:
                    mat_id = elem.wall.fibers[m].concrete
                elif m < 2 * fiber_count:
                    mat_id = elem.wall.fibers[m - fiber_count].steel
                if (
                    mat_id < 0
                    or mat_id >= len(uniaxial_def_by_id)
                    or uniaxial_def_by_id[mat_id] < 0
                ):
                    abort("MVLEM requires uniaxial concrete, steel and shear materials")
                var def_index = uniaxial_def_by_id[mat_id]
                var mat_def = uniaxial_defs[def_index]
                elem_uniaxial_state_ids.append(len(uniaxial_states))
                uniaxial_states.append(UniMaterialState(mat_def))
                uniaxial_state_defs.append(def_index)
                if not uni_mat_is_elastic(mat_def):
                    used_nonelastic_uniaxial = True
        elif elem.type_tag == ElementTypeTag.SFIMVLEM:
            elem_uniaxial_offsets[e] = len(elem_uniaxial_state_ids)
            elem_uniaxial_counts[e] = len(elem.wall.fibers)
            for fiber in elem.wall.fibers:
                ref material = typed_materials_by_id[fiber.concrete]
                if not material.fsam:
                    abort("SFI_MVLEM requires an FSAM material for each panel")
                var panel = material.fsam.value()
                var tags = [panel.steel_x, panel.steel_y, panel.concrete]
                var constituents: List[UniMaterialDef] = []
                for tag in tags:
                    var material_index = _remap_compact_index_or_abort(
                        tag, material_index_by_raw_id, "FSAM constituent material"
                    )
                    var definition_index = uniaxial_def_by_id[material_index]
                    if definition_index < 0:
                        abort("FSAM constituents must be uniaxial materials")
                    var definition = uniaxial_defs[definition_index]
                    if definition.mat_type == UniMaterialTypeTag.Parallel:
                        abort("FSAM requires individual uniaxial constituent materials")
                    constituents.append(definition)
                var concrete = constituents[2]
                if concrete.mat_type != UniMaterialTypeTag.ConcreteCM:
                    abort("FSAM requires a ConcreteCM concrete material")
                var concrete_params = concrete.concrete_cm_params()
                var panel_params = FSAMParameters(
                    panel.density,
                    panel.steel_x_ratio,
                    panel.steel_y_ratio,
                    panel.friction,
                    panel.dowel,
                    concrete_params.Ec,
                    concrete_params.epcc,
                    concrete_params.et,
                    uni_mat_initial_tangent(constituents[0]),
                    uni_mat_initial_tangent(constituents[1]),
                )
                # OpenSees creates two monotonic copies for the rotating
                # uncracked strut, then two cyclic copies per fixed strut.
                constituents[2].p10 = 1.0
                constituents.append(constituents[2])
                for _ in range(4):
                    constituents.append(concrete)
                var definitions = make_parallel_def(
                    constituents, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
                )
                var panel_state = UniMaterialState()
                var children: List[UniMaterialLeafState] = []
                ref leaf_definitions = definitions.components.value()[]
                for child in leaf_definitions:
                    children.append(UniMaterialLeafState(child.definition))
                panel_state.components = children^
                panel_state.fsam.append(FSAMState(panel_params))
                panel_state.fsam_components = definitions.components
                elem_uniaxial_state_ids.append(len(uniaxial_states))
                uniaxial_states.append(panel_state^)
                uniaxial_state_defs.append(-1)
            used_nonelastic_uniaxial = True
        elif elem.type_tag == ElementTypeTag.TwoNodeLink:
            elem_uniaxial_offsets[e] = len(elem_uniaxial_state_ids)
            elem_uniaxial_counts[e] = elem.material_count
            for m in range(elem.material_count):
                var mat_id = _elem_material(elem, m)
                if mat_id >= len(uniaxial_def_by_id) or uniaxial_def_by_id[mat_id] < 0:
                    abort("twoNodeLink requires uniaxial material")
                var def_index = uniaxial_def_by_id[mat_id]
                var mat_def = uniaxial_defs[def_index]
                if mat_def.mat_type == UniMaterialTypeTag.Viscous:
                    abort("Viscous currently requires a zeroLength element")
                var state_index = len(uniaxial_states)
                uniaxial_states.append(UniMaterialState(mat_def))
                uniaxial_state_defs.append(def_index)
                elem_uniaxial_state_ids.append(state_index)
                if not uni_mat_is_elastic(mat_def):
                    used_nonelastic_uniaxial = True
        elif elem.type_tag == ElementTypeTag.ForceBeamColumn2d:
            force_basic_offsets[e] = len(force_basic_q)
            var sec_id = elem.section
            elem_uniaxial_offsets[e] = len(elem_uniaxial_state_ids)
            var sec = typed_sections_by_id[sec_id]
            var beam_col_type = elem.type
            var has_material_hinges = (
                sec.hinge_radau.is_defined() and sec.type != "ElasticSection2d"
            )
            if sec.type == "FiberSection2d" or sec.hinge_radau.is_defined():
                force_basic_counts[e] = force_beam_column2d_fiber_state_count(elem.num_int_pts)
            elif sec.type == "ElasticSection2d":
                force_basic_counts[e] = 3
            else:
                abort(beam_col_type + " requires FiberSection2d or ElasticSection2d")
            for _ in range(force_basic_counts[e]):
                force_basic_q.append(0.0)
            if has_material_hinges:
                var hinge = sec.hinge_radau
                elem_uniaxial_counts[e] = 4
                for def_index in [hinge.axial_i, hinge.flexural_i, hinge.axial_j, hinge.flexural_j]:
                    var material = uniaxial_defs[def_index]
                    elem_uniaxial_state_ids.append(len(uniaxial_states))
                    uniaxial_states.append(UniMaterialState(material))
                    uniaxial_state_defs.append(def_index)
                    if not uni_mat_is_elastic(material):
                        used_nonelastic_uniaxial = True
                        force_beam_has_nonelastic = True
            elif sec.type == "FiberSection2d":
                var sec_index = fiber_section_index_by_id[sec_id]
                if sec_index < 0 or sec_index >= len(fiber_section_defs):
                    abort(beam_col_type + " fiber section not found")
                ref sec_def = fiber_section_defs[sec_index]
                var num_int_pts = elem.num_int_pts
                var state_count = num_int_pts * sec_def.fiber_count
                elem_uniaxial_offsets[e] = fiber_section2d_runtime_alloc_instances(
                    sec_def, num_int_pts
                )
                elem_uniaxial_counts[e] = state_count
                for _ in range(state_count):
                    elem_uniaxial_state_ids.append(-1)
                for i in range(sec_def.nonlinear_count):
                    var mat_def = sec_def.nonlinear_mat_defs[i]
                    if not uni_mat_is_elastic(mat_def):
                        used_nonelastic_uniaxial = True
                        force_beam_has_nonelastic = True
            elif sec.type == "ElasticSection2d":
                elem_uniaxial_counts[e] = 0
            else:
                abort(beam_col_type + " requires FiberSection2d or ElasticSection2d")
        elif elem.type_tag == ElementTypeTag.DispBeamColumn2d:
            var sec_id = elem.section
            elem_uniaxial_offsets[e] = len(elem_uniaxial_state_ids)
            var sec = typed_sections_by_id[sec_id]
            var beam_col_type = elem.type
            if sec.type == "FiberSection2d":
                var sec_index = fiber_section_index_by_id[sec_id]
                if sec_index < 0 or sec_index >= len(fiber_section_defs):
                    abort(beam_col_type + " fiber section not found")
                ref sec_def = fiber_section_defs[sec_index]
                var num_int_pts = elem.num_int_pts
                var state_count = num_int_pts * sec_def.fiber_count
                elem_uniaxial_offsets[e] = fiber_section2d_runtime_alloc_instances(
                    sec_def, num_int_pts
                )
                elem_uniaxial_counts[e] = state_count
                for _ in range(state_count):
                    elem_uniaxial_state_ids.append(-1)
                for i in range(sec_def.nonlinear_count):
                    var mat_def = sec_def.nonlinear_mat_defs[i]
                    if not uni_mat_is_elastic(mat_def):
                        used_nonelastic_uniaxial = True
                        force_beam_has_nonelastic = True
            elif sec.type == "ElasticSection2d":
                elem_uniaxial_counts[e] = 0
            else:
                abort(beam_col_type + " requires FiberSection2d or ElasticSection2d")
        elif (
            elem.type_tag == ElementTypeTag.ForceBeamColumn3d
            or elem.type_tag == ElementTypeTag.DispBeamColumn3d
        ):
            if elem.type_tag == ElementTypeTag.ForceBeamColumn3d:
                force_basic_offsets[e] = len(force_basic_q)
            var sec_id = elem.section
            elem_uniaxial_offsets[e] = len(elem_uniaxial_state_ids)
            var sec = typed_sections_by_id[sec_id]
            var beam_col_type = elem.type
            if sec.type == "FiberSection3d":
                var sec_index = fiber_section3d_index_by_id[sec_id]
                if sec_index < 0 or sec_index >= len(fiber_section3d_defs):
                    abort(beam_col_type + " fiber section not found")
                ref sec_def = fiber_section3d_defs[sec_index]
                var num_int_pts = elem.num_int_pts
                var state_count = num_int_pts * sec_def.fiber_count
                if elem.type_tag == ElementTypeTag.ForceBeamColumn3d:
                    force_basic_counts[e] = force_beam_column3d_fiber_state_count(num_int_pts)
                    for _ in range(force_basic_counts[e]):
                        force_basic_q.append(0.0)
                elem_uniaxial_offsets[e] = fiber_section3d_runtime_alloc_instances(
                    sec_def, num_int_pts
                )
                elem_uniaxial_counts[e] = state_count
                if sec_def.has_nonelastic:
                    used_nonelastic_uniaxial = True
                    if elem.type_tag == ElementTypeTag.ForceBeamColumn3d:
                        force_beam_has_nonelastic = True
            elif sec.type == "ElasticSection3d":
                elem_uniaxial_counts[e] = 0
            else:
                abort(beam_col_type + " requires ElasticSection3d or FiberSection3d")
        elif elem.type_tag == ElementTypeTag.ZeroLengthSection:
            var sec_id = elem.section
            elem_uniaxial_offsets[e] = len(elem_uniaxial_state_ids)
            var sec = typed_sections_by_id[sec_id]
            if sec.type == "FiberSection2d":
                var sec_index = fiber_section_index_by_id[sec_id]
                if sec_index < 0 or sec_index >= len(fiber_section_defs):
                    abort("zeroLengthSection fiber section not found")
                ref sec_def = fiber_section_defs[sec_index]
                var state_count = sec_def.fiber_count
                elem_uniaxial_offsets[e] = fiber_section2d_runtime_alloc_instances(sec_def, 1)
                elem_uniaxial_counts[e] = state_count
                for _ in range(state_count):
                    elem_uniaxial_state_ids.append(-1)
                for i in range(sec_def.nonlinear_count):
                    var mat_def = sec_def.nonlinear_mat_defs[i]
                    if not uni_mat_is_elastic(mat_def):
                        used_nonelastic_uniaxial = True
            elif sec.type == "FiberSection3d":
                var sec_index = fiber_section3d_index_by_id[sec_id]
                if sec_index < 0 or sec_index >= len(fiber_section3d_defs):
                    abort("zeroLengthSection 3D fiber section not found")
                ref sec_def = fiber_section3d_defs[sec_index]
                var state_count = sec_def.fiber_count
                elem_uniaxial_offsets[e] = fiber_section3d_runtime_alloc_instances(sec_def, 1)
                elem_uniaxial_counts[e] = state_count
                for _ in range(state_count):
                    elem_uniaxial_state_ids.append(-1)
                if sec_def.has_nonelastic:
                    used_nonelastic_uniaxial = True
            elif sec.type == "AggregatorSection2d":
                var expected_state_count = _aggregator_section2d_expected_state_count(sec)
                elem_uniaxial_counts[e] = expected_state_count
                var mat_slots = [sec.axial_material, sec.shear_y_material, sec.flexural_material]
                for s_idx in range(len(mat_slots)):
                    var mat_id = mat_slots[s_idx]
                    if mat_id < 0:
                        continue
                    if mat_id >= len(uniaxial_def_by_id) or uniaxial_def_by_id[mat_id] < 0:
                        abort("AggregatorSection2d material must be uniaxial")
                    var def_index = uniaxial_def_by_id[mat_id]
                    var mat_def = uniaxial_defs[def_index]
                    var state_index = len(uniaxial_states)
                    uniaxial_states.append(UniMaterialState(mat_def))
                    uniaxial_state_defs.append(def_index)
                    elem_uniaxial_state_ids.append(state_index)
                    if not uni_mat_is_elastic(mat_def):
                        used_nonelastic_uniaxial = True
            elif sec.type == "AggregatorSection3d":
                var expected_state_count = _aggregator_section3d_expected_state_count(sec)
                elem_uniaxial_counts[e] = expected_state_count
                var mat_slots = [
                    sec.axial_material,
                    sec.shear_y_material,
                    sec.shear_z_material,
                    sec.torsion_material,
                    sec.moment_y_material,
                    sec.flexural_material,
                ]
                for s_idx in range(len(mat_slots)):
                    var mat_id = mat_slots[s_idx]
                    if mat_id < 0:
                        continue
                    if mat_id >= len(uniaxial_def_by_id) or uniaxial_def_by_id[mat_id] < 0:
                        abort("AggregatorSection3d material must be uniaxial")
                    var def_index = uniaxial_def_by_id[mat_id]
                    var mat_def = uniaxial_defs[def_index]
                    var state_index = len(uniaxial_states)
                    uniaxial_states.append(UniMaterialState(mat_def))
                    uniaxial_state_defs.append(def_index)
                    elem_uniaxial_state_ids.append(state_index)
                    if not uni_mat_is_elastic(mat_def):
                        used_nonelastic_uniaxial = True
            elif sec.type == "ElasticSection2d" or sec.type == "ElasticSection3d":
                elem_uniaxial_counts[e] = 0
            else:
                abort(
                    "zeroLengthSection requires FiberSection2d/3d, ElasticSection2d/3d, or"
                    " AggregatorSection2d/3d"
                )
        else:
            elem_uniaxial_offsets[e] = len(elem_uniaxial_state_ids)
            elem_uniaxial_counts[e] = 0

    var layered_shell_instance_counts: List[Int] = []
    layered_shell_instance_counts.resize(len(layered_shell_section_defs), 0)
    for e in range(elem_count):
        var elem = typed_elements[e]
        if elem.type_tag != ElementTypeTag.Shell:
            continue
        var sec_id = elem.section
        if sec_id < 0 or sec_id >= len(typed_sections_by_id):
            abort("shell section not found")
        var sec = typed_sections_by_id[sec_id]
        if sec.type != "LayeredShellSection":
            continue
        if sec_id >= len(layered_shell_section_index_by_id):
            abort("LayeredShellSection index mapping missing")
        var sec_index = layered_shell_section_index_by_id[sec_id]
        if sec_index < 0 or sec_index >= len(layered_shell_section_defs):
            abort("LayeredShellSection definition not found")
        shell_elem_instance_offsets[e] = layered_shell_instance_counts[sec_index]
        layered_shell_instance_counts[sec_index] += 4
    var first_shell_rebar_state = len(uniaxial_states)
    if layered_shell_runtime_alloc_instances(
        layered_shell_section_defs,
        layered_shell_instance_counts,
        uniaxial_defs,
        uniaxial_states,
        uniaxial_state_defs,
        layered_shell_section_uniaxial_offsets,
    ):
        used_nonelastic_uniaxial = True
    for state_id in range(first_shell_rebar_state, len(uniaxial_states)):
        elem_uniaxial_state_ids.append(state_id)

    var shell_cache = SurfaceAssemblyCache()
    swap(shell_cache.pimy_definitions, input.pimy_definitions)
    swap(shell_cache.pdmy_definitions, input.pdmy_definitions)
    shell_cache.prepare_quads(
        quad_elem_indices,
        typed_elements,
        typed_nodes,
        typed_materials_by_id,
        elem_node_offsets,
        elem_node_pool,
    )
    shell_cache.continuum.prepare(
        continuum_indices,
        typed_elements,
        typed_nodes,
        typed_materials_by_id,
        elem_node_offsets,
        elem_node_pool,
        elem_dof_offsets,
        elem_dof_pool,
    )
    shell_cache.shell.prepare(
        shell_elem_indices,
        typed_elements,
        typed_nodes,
        typed_sections_by_id,
        layered_shell_section_defs,
        layered_shell_section_index_by_id,
        layered_shell_section_uniaxial_offsets,
        shell_elem_instance_offsets,
    )

    var total_dofs = node_count * ndf
    var F_total: List[Float64] = []
    F_total.resize(total_dofs, 0.0)
    var elem_load_offsets: List[Int] = []
    var elem_load_pool: List[Int] = []
    _build_element_load_index(
        element_loads,
        typed_elements,
        elem_id_to_index,
        ndm,
        ndf,
        elem_load_offsets,
        elem_load_pool,
    )

    for i in range(len(nodal_loads)):
        var load = nodal_loads[i]
        var node_id = load.node
        var dof = load.dof
        require_dof_in_range(dof, ndf, "load")
        var idx = node_dof_index(id_to_index[node_id], dof, ndf)
        F_total[idx] += load.value

    var M_total: List[Float64] = []
    M_total.resize(total_dofs, 0.0)
    var M_rayleigh_total: List[Float64] = []
    M_rayleigh_total.resize(total_dofs, 0.0)
    for i in range(len(masses)):
        var mass = masses[i]
        var node_id = mass.node
        var dof = mass.dof
        require_dof_in_range(dof, ndf, "mass")
        var idx = node_dof_index(id_to_index[node_id], dof, ndf)
        M_total[idx] += mass.value
        M_rayleigh_total[idx] += mass.value
    for e in range(elem_count):
        _accumulate_beam_element_lumped_mass(
            typed_elements[e], typed_elements[e].rho, typed_nodes, ndf, M_total
        )
        _accumulate_beam_element_lumped_mass(
            typed_elements[e],
            typed_elements[e].rho,
            typed_nodes,
            ndf,
            M_rayleigh_total,
        )
        if typed_elements[e].type_tag == ElementTypeTag.TwoNodeLink:
            _accumulate_two_node_link_lumped_mass(typed_elements[e], ndm, ndf, M_total)
            if typed_elements[e].do_rayleigh:
                _accumulate_two_node_link_lumped_mass(typed_elements[e], ndm, ndf, M_rayleigh_total)

    for elem in typed_elements:
        if elem.type_tag != ElementTypeTag.CatenaryCable:
            continue
        var mass = 0.5 * elem.cable.length * elem.cable.density
        for dof in range(6):
            M_total[_elem_dof(elem, dof)] += mass

    for elem in typed_elements:
        if elem.type_tag != ElementTypeTag.SFIMVLEM:
            continue
        var mass = 0.0
        for fiber in elem.wall.fibers:
            mass += typed_materials_by_id[fiber.concrete].fsam.value().density * fiber.concrete_area
        mass *= 0.5 * elem.wall.height
        for node in [elem.node_index_1, elem.node_index_2]:
            for dof in range(1, 3):
                var index = node_dof_index(node, dof, ndf)
                M_total[index] += mass
                M_rayleigh_total[index] += mass

    shell_cache.continuum.accumulate_lumped_mass(M_total, M_rayleigh_total)
    for e in quad_elem_indices:
        var elem = typed_elements[e]
        if (
            elem.type_tag == ElementTypeTag.QuadUP
            or elem.type_tag == ElementTypeTag.SSPBrickUP
            or elem.type_tag == ElementTypeTag.BeamContact2D
        ):
            continue
        if elem.type_tag == ElementTypeTag.Tri31:
            var rho = typed_materials_by_id[elem.material].rho
            if rho != 0.0:
                var n1 = typed_nodes[elem.node_index_1]
                var n2 = typed_nodes[elem.node_index_2]
                var n3 = typed_nodes[elem.node_index_3]
                var twice_area = (n1.x - n3.x) * (n2.y - n3.y) - (n2.x - n3.x) * (n1.y - n3.y)
                var lumped = rho * elem.thickness * twice_area / 6.0
                for node_idx in [elem.node_index_1, elem.node_index_2, elem.node_index_3]:
                    for d in range(1, 3):
                        var global_dof = node_dof_index(node_idx, d, ndf)
                        M_total[global_dof] += lumped
                        M_rayleigh_total[global_dof] += lumped
            continue
        ref quad = shell_cache.quads[shell_cache.quad_index[e]]
        for i in range(8):
            var dof = _elem_dof(typed_elements[e], i)
            M_total[dof] += quad.mass[i]
            M_rayleigh_total[dof] += quad.mass[i]

    for e in range(elem_count):
        var elem = typed_elements[e]
        if elem.type_tag != ElementTypeTag.Shell:
            continue
        if elem.section < 0 or elem.section >= len(typed_sections_by_id):
            abort("shell section not found")
        var sec = typed_sections_by_id[elem.section]
        var rho_area = 0.0
        if sec.type == "ElasticMembranePlateSection" or sec.type == "PlateFiber":
            rho_area = sec.rho * sec.h
        elif sec.type == "LayeredShellSection":
            if elem.section >= len(layered_shell_section_index_by_id):
                abort("LayeredShellSection index mapping missing")
            var sec_index = layered_shell_section_index_by_id[elem.section]
            if sec_index < 0 or sec_index >= len(layered_shell_section_defs):
                abort("LayeredShellSection definition not found")
            rho_area = layered_shell_section_defs[sec_index].rho_area
        if rho_area == 0.0:
            continue

        var n1 = typed_nodes[elem.node_index_1]
        var n2 = typed_nodes[elem.node_index_2]
        var n3 = typed_nodes[elem.node_index_3]
        var dofs = InlineArray[Int, 4](fill=0)
        dofs[0] = node_dof_index(elem.node_index_1, 1, ndf)
        dofs[1] = node_dof_index(elem.node_index_2, 1, ndf)
        dofs[2] = node_dof_index(elem.node_index_3, 1, ndf)
        if elem.node_count == 3:
            var ax = n2.x - n1.x
            var ay = n2.y - n1.y
            var az = n2.z - n1.z
            var bx = n3.x - n1.x
            var by = n3.y - n1.y
            var bz = n3.z - n1.z
            var cx = ay * bz - az * by
            var cy = az * bx - ax * bz
            var cz = ax * by - ay * bx
            var weights = shell_triangle_mass(0.5 * sqrt(cx * cx + cy * cy + cz * cz), rho_area)
            shell_cache.shell.masses.append(ShellMass(e, dofs^, weights^, 3))
            continue
        var n4 = typed_nodes[elem.node_index_4]
        dofs[3] = node_dof_index(elem.node_index_4, 1, ndf)
        var weights = shell_quad_mass(
            [n1.x, n2.x, n3.x, n4.x],
            [n1.y, n2.y, n3.y, n4.y],
            [n1.z, n2.z, n3.z, n4.z],
            rho_area,
        )
        if elem.type == "ASDShellQ4":
            for i in range(4):
                var lumped = 0.0
                for j in range(4):
                    lumped += weights[4 * i + j]
                for d in range(3):
                    M_total[dofs[i] + d] += lumped
                    M_rayleigh_total[dofs[i] + d] += lumped
        else:
            shell_cache.shell.masses.append(ShellMass(e, dofs^, weights^, 4))

    var root_uses_penalty = analysis_input.constraints_tag == ConstraintHandlerTag.Penalty
    var uses_penalty = root_uses_penalty
    for stage in input.stages:
        if stage.analysis.constraints_tag == ConstraintHandlerTag.Penalty:
            uses_penalty = True
    if root_uses_penalty and (
        analysis_input.type_tag != AnalysisTypeTag.Staged
        and analysis_input.type_tag != AnalysisTypeTag.TransientNonlinear
        and analysis_input.type_tag != AnalysisTypeTag.StaticNonlinear
    ):
        abort("Penalty requires static or transient nonlinear analysis")
    var constrained: List[Bool] = []
    constrained.resize(total_dofs, False)
    for i in range(node_count):
        var node = typed_nodes[i]
        if node.ndf < 1 or node.ndf > ndf:
            abort("node ndf must be in 1..model ndf")
        for dof in range(node.ndf + 1, ndf + 1):
            constrained[node_dof_index(i, dof, ndf)] = True
        if uses_penalty:
            for j in range(node.constraint_count):
                var dof = _node_constraint(node, j)
                require_dof_in_range(dof, node.ndf, "penalty constraint")
                state.penalty_sp_dofs.append(node_dof_index(i, dof, ndf))
        if root_uses_penalty:
            continue
        for j in range(node.constraint_count):
            if j == 0:
                var dof = node.constraint_1
                require_dof_in_range(dof, ndf, "constraint")
                var idx = node_dof_index(i, dof, ndf)
                constrained[idx] = True
            elif j == 1:
                var dof = node.constraint_2
                require_dof_in_range(dof, ndf, "constraint")
                var idx = node_dof_index(i, dof, ndf)
                constrained[idx] = True
            elif j == 2:
                var dof = node.constraint_3
                require_dof_in_range(dof, ndf, "constraint")
                var idx = node_dof_index(i, dof, ndf)
                constrained[idx] = True
            elif j == 3:
                var dof = node.constraint_4
                require_dof_in_range(dof, ndf, "constraint")
                var idx = node_dof_index(i, dof, ndf)
                constrained[idx] = True
            elif j == 4:
                var dof = node.constraint_5
                require_dof_in_range(dof, ndf, "constraint")
                var idx = node_dof_index(i, dof, ndf)
                constrained[idx] = True
            else:
                var dof = node.constraint_6
                require_dof_in_range(dof, ndf, "constraint")
                var idx = node_dof_index(i, dof, ndf)
                constrained[idx] = True

    _write_run_progress(progress_path, "loading", "case", 0, 0, 4, load_step_count)

    var analysis_type = analysis_input.type
    var analysis_type_tag = analysis_input.type_tag
    var constraints_handler = analysis_input.constraints
    var constraints_handler_tag = analysis_input.constraints_tag
    var sp_analysis = analysis_input
    var sp_pattern = input.pattern
    for stage in input.stages:
        if len(input.stages) == 1:
            sp_analysis = stage.analysis
            if stage.pattern.has_pattern:
                sp_pattern = stage.pattern
        elif (
            len(stage.pattern.sp_constraints) > 0
            and stage.analysis.constraints_tag != ConstraintHandlerTag.Penalty
        ):
            abort("sp constraints require a single static LoadControl analysis")
    if (
        len(sp_pattern.sp_constraints) > 0
        and sp_analysis.constraints_tag != ConstraintHandlerTag.Penalty
    ):
        if (
            sp_analysis.type_tag != AnalysisTypeTag.StaticNonlinear
            or (
                sp_analysis.integrator_tag != IntegratorTypeTag.Unknown
                and sp_analysis.integrator_tag != IntegratorTypeTag.LoadControl
            )
            or sp_analysis.constraints_tag != ConstraintHandlerTag.Transformation
        ):
            abort("sp constraints require static LoadControl and Transformation")
        if sp_pattern.type_tag != PatternTypeTag.Plain:
            abort("sp constraints require a Plain pattern")
        for sp in sp_pattern.sp_constraints:
            if sp.node < 0 or sp.node >= len(id_to_index) or id_to_index[sp.node] < 0:
                abort("sp constraint node not found")
            var node_index = id_to_index[sp.node]
            require_dof_in_range(sp.dof, typed_nodes[node_index].ndf, "sp constraint")
            var dof_index = node_dof_index(node_index, sp.dof, ndf)
            constrained[dof_index] = True
            state.prescribed_dofs.append(dof_index)
            state.prescribed_values.append(sp.value)
    if constraints_handler_tag == ConstraintHandlerTag.Unknown:
        abort("unsupported constraints handler: " + constraints_handler)
    var numberer_tag = analysis_input.numberer_tag
    if numberer_tag == NumbererTag.Unknown:
        numberer_tag = NumbererTag.RCM
    if numberer_tag != NumbererTag.RCM and numberer_tag != NumbererTag.Plain:
        abort("unsupported analysis numberer tag")
    var steps = analysis_input.steps
    var modal_num_modes = 0
    if analysis_type_tag == AnalysisTypeTag.ModalEigen:
        modal_num_modes = analysis_input.num_modes
        if modal_num_modes < 1:
            abort("modal_eigen requires num_modes >= 1")
    if steps < 1:
        abort("analysis steps must be >= 1")
    var force_beam_mode = analysis_input.force_beam_mode
    var force_beam_mode_tag = analysis_input.force_beam_mode_tag
    if force_beam_mode_tag == ForceBeamModeTag.Unknown:
        abort(
            "unsupported force_beam_mode: "
            + force_beam_mode
            + " (expected auto|linear_if_elastic|nonlinear)"
        )
    if has_force_beam_column2d:
        if (
            analysis_type_tag != AnalysisTypeTag.StaticLinear
            and analysis_type_tag != AnalysisTypeTag.StaticNonlinear
            and analysis_type_tag != AnalysisTypeTag.TransientNonlinear
            and analysis_type_tag != AnalysisTypeTag.Staged
        ):
            abort(
                "forceBeamColumn2d/dispBeamColumn2d requires static_linear, "
                "static_nonlinear, transient_nonlinear, or staged analysis"
            )
        if analysis_type_tag == AnalysisTypeTag.TransientNonlinear:
            if force_beam_mode_tag != ForceBeamModeTag.Auto:
                abort(
                    "force_beam_mode is only supported for static forceBeamColumn2d/"
                    "dispBeamColumn2d analyses"
                )
        elif analysis_type_tag == AnalysisTypeTag.Staged:
            if force_beam_mode_tag != ForceBeamModeTag.Auto:
                abort(
                    "force_beam_mode is only supported for non-staged static forceBeamColumn2d/"
                    "dispBeamColumn2d analyses"
                )
        else:
            if force_beam_mode_tag == ForceBeamModeTag.Nonlinear:
                if analysis_type_tag != AnalysisTypeTag.StaticNonlinear:
                    abort("force_beam_mode=nonlinear requires static_nonlinear analysis")
            elif force_beam_mode_tag == ForceBeamModeTag.LinearIfElastic:
                if force_beam_has_nonelastic:
                    if analysis_type_tag != AnalysisTypeTag.StaticNonlinear:
                        abort(
                            "forceBeamColumn2d/dispBeamColumn2d "
                            "with non-elastic fibers "
                            "requires static_nonlinear analysis"
                        )
                elif analysis_type_tag != AnalysisTypeTag.StaticLinear:
                    abort(
                        "force_beam_mode=linear_if_elastic requires static_linear "
                        "analysis for elastic forceBeamColumn2d/dispBeamColumn2d"
                    )
            elif analysis_type_tag == AnalysisTypeTag.StaticLinear and force_beam_has_nonelastic:
                abort(
                    "forceBeamColumn2d/dispBeamColumn2d with "
                    "non-elastic fibers requires "
                    "static_nonlinear analysis"
                )
    if (
        analysis_type_tag != AnalysisTypeTag.StaticNonlinear
        and analysis_type_tag != AnalysisTypeTag.TransientNonlinear
        and analysis_type_tag != AnalysisTypeTag.ModalEigen
        and analysis_type_tag != AnalysisTypeTag.Staged
        and used_nonelastic_uniaxial
    ):
        abort("nonlinear materials require static_nonlinear or transient_nonlinear analysis")

    var system_tag = analysis_input.system_tag
    if system_tag == AnalysisSystemTag.Unknown:
        abort("unsupported analysis system tag")
    var sparse_sym_ordering = analysis_input.sparse_sym_ordering

    var rep_dof: List[Int] = []
    rep_dof.resize(total_dofs, 0)
    var mpc_row_count: List[Int] = []
    mpc_row_count.resize(total_dofs, 1)
    var mpc_row_dof_1: List[Int] = []
    mpc_row_dof_1.resize(total_dofs, 0)
    var mpc_row_dof_2: List[Int] = []
    mpc_row_dof_2.resize(total_dofs, -1)
    var mpc_row_dof_3: List[Int] = []
    mpc_row_dof_3.resize(total_dofs, -1)
    var mpc_row_coeff_1: List[Float64] = []
    mpc_row_coeff_1.resize(total_dofs, 1.0)
    var mpc_row_coeff_2: List[Float64] = []
    mpc_row_coeff_2.resize(total_dofs, 0.0)
    var mpc_row_coeff_3: List[Float64] = []
    mpc_row_coeff_3.resize(total_dofs, 0.0)
    var mpc_slave_dof: List[Bool] = []
    mpc_slave_dof.resize(total_dofs, False)
    for i in range(total_dofs):
        rep_dof[i] = i
        mpc_row_dof_1[i] = i

    var has_transformation_mpc = False
    if len(mp_constraints) > 0:
        if (
            constraints_handler_tag != ConstraintHandlerTag.Transformation
            and constraints_handler_tag != ConstraintHandlerTag.Lagrange
            and constraints_handler_tag != ConstraintHandlerTag.Penalty
        ):
            abort("[load-fail] mp_constraints require Transformation, Lagrange, or Penalty")
        has_transformation_mpc = not root_uses_penalty
    var node_is_mpc_constrained: List[Bool] = []
    node_is_mpc_constrained.resize(len(id_to_index), False)
    for i in range(len(mp_constraints)):
        var mpc = mp_constraints[i]
        var constrained_node = mpc.constrained_node
        if constrained_node >= 0 and constrained_node < len(node_is_mpc_constrained):
            node_is_mpc_constrained[constrained_node] = True
    for i in range(len(mp_constraints)):
        var mpc = mp_constraints[i]
        var mpc_type = mpc.type
        if mpc_type != "equalDOF" and mpc_type != "rigidDiaphragm":
            abort("[load-fail] unsupported mp constraint type: " + mpc_type)
        var retained_node = mpc.retained_node
        var constrained_node = mpc.constrained_node
        if retained_node >= len(id_to_index) or id_to_index[retained_node] < 0:
            abort("[load-fail] " + mpc_type + " retained_node not found")
        if constrained_node >= len(id_to_index) or id_to_index[constrained_node] < 0:
            abort("[load-fail] " + mpc_type + " constrained_node not found")
        if mpc_type != "equalDOF" and node_is_mpc_constrained[retained_node]:
            abort("[load-fail] " + mpc_type + " retained_node cannot also be constrained")
        var retained_idx = id_to_index[retained_node]
        var constrained_idx = id_to_index[constrained_node]
        if uses_penalty and mpc_type != "equalDOF":
            abort("Penalty currently requires equalDOF for multi-point constraints")
        if mpc_type == "equalDOF":
            if mpc.dof_count == 0:
                abort("[load-fail] equalDOF requires non-empty dofs")
            for j in range(mpc.dof_count):
                var dof = _mp_constraint_dof(mpc, j)
                require_dof_in_range(dof, ndf, "equalDOF")
                var retained_dof = node_dof_index(retained_idx, dof, ndf)
                var constrained_dof = node_dof_index(constrained_idx, dof, ndf)
                if uses_penalty:
                    require_dof_in_range(dof, typed_nodes[retained_idx].ndf, "equalDOF")
                    require_dof_in_range(dof, typed_nodes[constrained_idx].ndf, "equalDOF")
                    state.penalty_mp_dofs.append(retained_dof)
                    state.penalty_mp_dofs.append(constrained_dof)
                if root_uses_penalty:
                    continue
                if mpc_slave_dof[constrained_dof]:
                    abort("[load-fail] equalDOF constrained dof already controlled")
                rep_dof[constrained_dof] = retained_dof
                mpc_row_count[constrained_dof] = 1
                mpc_row_dof_1[constrained_dof] = retained_dof
                mpc_row_dof_2[constrained_dof] = -1
                mpc_row_dof_3[constrained_dof] = -1
                mpc_row_coeff_1[constrained_dof] = 1.0
                mpc_row_coeff_2[constrained_dof] = 0.0
                mpc_row_coeff_3[constrained_dof] = 0.0
                mpc_slave_dof[constrained_dof] = True
        else:
            var rigid_requires_3d = ndm == 3 and ndf == 6
            var rigid_requires_2d = ndm == 2 and ndf == 3
            if not rigid_requires_3d and not rigid_requires_2d:
                abort("[load-fail] rigidDiaphragm requires a 3D/6DOF or 2D/3DOF model")
            if mpc.rigid_perp_dirn < 1 or mpc.rigid_perp_dirn > 3:
                abort("[load-fail] rigidDiaphragm perp_dirn must be in 1..3")
            if mpc.rigid_constrained_dof_count == 0 or mpc.rigid_retained_dof_count == 0:
                abort("[load-fail] rigidDiaphragm requires constrained_dofs and retained_dofs")
            if mpc.rigid_matrix_row_count != mpc.rigid_constrained_dof_count:
                abort("[load-fail] rigidDiaphragm matrix rows must match constrained_dofs")
            if mpc.rigid_matrix_col_count != mpc.rigid_retained_dof_count:
                abort("[load-fail] rigidDiaphragm matrix columns must match retained_dofs")
            for row in range(mpc.rigid_constrained_dof_count):
                var constrained_dof_no = _mp_constraint_rigid_constrained_dof(mpc, row)
                require_dof_in_range(constrained_dof_no, ndf, "rigidDiaphragm")
                var constrained_dof = node_dof_index(constrained_idx, constrained_dof_no, ndf)
                if mpc_slave_dof[constrained_dof]:
                    abort("[load-fail] rigidDiaphragm constrained dof already controlled")
                mpc_row_count[constrained_dof] = mpc.rigid_retained_dof_count
                mpc_row_dof_2[constrained_dof] = -1
                mpc_row_dof_3[constrained_dof] = -1
                mpc_row_coeff_2[constrained_dof] = 0.0
                mpc_row_coeff_3[constrained_dof] = 0.0
                for col in range(mpc.rigid_retained_dof_count):
                    var retained_dof_no = _mp_constraint_rigid_retained_dof(mpc, col)
                    require_dof_in_range(retained_dof_no, ndf, "rigidDiaphragm")
                    var retained_dof = node_dof_index(retained_idx, retained_dof_no, ndf)
                    var coeff = _mp_constraint_rigid_matrix_entry(mpc, row, col)
                    if col == 0:
                        mpc_row_dof_1[constrained_dof] = retained_dof
                        mpc_row_coeff_1[constrained_dof] = coeff
                        rep_dof[constrained_dof] = retained_dof
                    elif col == 1:
                        mpc_row_dof_2[constrained_dof] = retained_dof
                        mpc_row_coeff_2[constrained_dof] = coeff
                    else:
                        mpc_row_dof_3[constrained_dof] = retained_dof
                        mpc_row_coeff_3[constrained_dof] = coeff
                mpc_slave_dof[constrained_dof] = True
                constrained[constrained_dof] = True

    # TransformationDOF_Group::doneID indexes a retained group's ID by
    # nodal DOF. For a transformed group that ID lists local free slots
    # first, then the equation slots of the group's own retained node.
    var group_slot_dof: List[Int] = []
    group_slot_dof.resize(total_dofs, -1)
    if constraints_handler_tag == ConstraintHandlerTag.Transformation:
        for node in range(node_count):
            var tag = typed_nodes[node].id
            if tag < 0 or not node_is_mpc_constrained[tag]:
                continue
            var base = node * ndf
            var slot = base
            for dof in range(base, base + typed_nodes[node].ndf):
                if not mpc_slave_dof[dof]:
                    group_slot_dof[slot] = dof
                    slot += 1
            for dof in range(base, base + typed_nodes[node].ndf):
                if mpc_slave_dof[dof]:
                    group_slot_dof[slot] = dof
                    slot += 1
    for i in range(total_dofs):
        if not mpc_slave_dof[i] or mpc_row_count[i] != 1 or not mpc_slave_dof[rep_dof[i]]:
            continue
        var retained = rep_dof[i]
        var next_dof = group_slot_dof[retained]
        var hops = 0
        while next_dof >= 0 and mpc_slave_dof[next_dof]:
            if mpc_row_count[next_dof] != 1 or hops == total_dofs:
                abort("[load-fail] unsupported chained MPC equation")
            retained = rep_dof[next_dof]
            next_dof = group_slot_dof[retained]
            hops += 1
        if next_dof >= 0:
            retained = next_dof
        if mpc_slave_dof[retained]:
            abort("[load-fail] equalDOF retained dof cannot also be constrained")
        mpc_row_dof_1[i] = retained
    for dof in state.prescribed_dofs:
        if mpc_slave_dof[dof]:
            abort("sp constraint cannot prescribe an MPC slave DOF")

    var mpc_row_offsets: List[Int] = []
    mpc_row_offsets.resize(total_dofs + 1, 0)
    var mpc_pool_count = 0
    for i in range(total_dofs):
        mpc_row_offsets[i] = mpc_pool_count
        if constrained[i] and not mpc_slave_dof[i]:
            continue
        mpc_pool_count += mpc_row_count[i]
    mpc_row_offsets[total_dofs] = mpc_pool_count

    var mpc_dof_pool: List[Int] = []
    mpc_dof_pool.resize(mpc_pool_count, -1)
    var mpc_coeff_pool: List[Float64] = []
    mpc_coeff_pool.resize(mpc_pool_count, 0.0)
    for i in range(total_dofs):
        if constrained[i] and not mpc_slave_dof[i]:
            continue
        var base = mpc_row_offsets[i]
        var count = mpc_row_count[i]
        if count > 0:
            mpc_dof_pool[base] = mpc_row_dof_1[i]
            mpc_coeff_pool[base] = mpc_row_coeff_1[i]
        if count > 1:
            mpc_dof_pool[base + 1] = mpc_row_dof_2[i]
            mpc_coeff_pool[base + 1] = mpc_row_coeff_2[i]
        if count > 2:
            mpc_dof_pool[base + 2] = mpc_row_dof_3[i]
            mpc_coeff_pool[base + 2] = mpc_row_coeff_3[i]

    F_total = _collapse_vector_by_mpc(F_total, mpc_row_offsets, mpc_dof_pool, mpc_coeff_pool)

    var active_index_by_dof: List[Int] = []
    active_index_by_dof.resize(total_dofs, -1)
    var free_count = 0
    for i in range(total_dofs):
        if mpc_slave_dof[i]:
            constrained[i] = True
            continue
        if constrained[i]:
            continue
        active_index_by_dof[i] = free_count
        free_count += 1

    if free_count == 0:
        abort("no free dofs")

    var system_is_banded = (
        system_tag == AnalysisSystemTag.BandGeneral
        or system_tag == AnalysisSystemTag.BandSPD
        or system_tag == AnalysisSystemTag.ProfileSPD
    )
    var use_banded_linear = False
    var use_banded_nonlinear = False
    if not has_transformation_mpc:
        if analysis_type_tag == AnalysisTypeTag.StaticLinear and system_is_banded:
            use_banded_linear = True
        elif analysis_type_tag == AnalysisTypeTag.StaticNonlinear and system_is_banded:
            use_banded_nonlinear = True

    var free: List[Int] = []
    var free_index: List[Int] = []
    var use_sparse_sym_graph_ordering = (
        system_tag == AnalysisSystemTag.SparseSYM and sparse_sym_ordering != 0
    )
    if use_banded_linear or use_banded_nonlinear or use_sparse_sym_graph_ordering:
        var node_order: List[Int] = []
        var use_adjacency_ordering = (
            numberer_tag == NumbererTag.RCM or use_sparse_sym_graph_ordering
        )
        if use_adjacency_ordering:
            var adjacency = build_node_adjacency_typed(typed_elements, node_count)
            if use_sparse_sym_graph_ordering:
                if sparse_sym_ordering == 1:
                    node_order = min_degree_order(adjacency)
                elif sparse_sym_ordering == 2:
                    # Provisional ND approximation until a dedicated nested-dissection
                    # ordering is implemented.
                    node_order = rcm_order(adjacency)
                else:
                    node_order = rcm_order(adjacency)
            else:
                node_order = rcm_order(adjacency)
        if len(node_order) == 0:
            node_order.resize(node_count, 0)
            for i in range(node_count):
                node_order[i] = i
        free_index.resize(total_dofs, -1)
        for i in range(len(node_order)):
            var node_idx = node_order[i]
            for dof in range(1, ndf + 1):
                var idx = node_dof_index(node_idx, dof, ndf)
                if mpc_slave_dof[idx]:
                    continue
                if not constrained[idx]:
                    free_index[idx] = len(free)
                    free.append(idx)
    else:
        free_index.resize(total_dofs, -1)
        for i in range(total_dofs):
            if mpc_slave_dof[i]:
                continue
            if not constrained[i]:
                free_index[i] = len(free)
                free.append(i)

    var elem_free_offsets = elem_dof_offsets.copy()
    var elem_free_pool: List[Int] = []
    elem_free_pool.resize(len(elem_dof_pool), -1)
    for e in range(elem_count):
        var offset = elem_dof_offsets[e]
        var count = elem_dof_offsets[e + 1] - offset
        for d in range(count):
            elem_free_pool[offset + d] = free_index[elem_dof_pool[offset + d]]

    for i in range(len(typed_elements)):
        var elem = typed_elements[i]
        if elem.type_tag == ElementTypeTag.ZeroLength and elem.damping_tag >= 0:
            if _find_damping_input(dampings, elem.damping_tag) < 0:
                abort("zeroLength damping not found")
    var ts_index = -1
    var pattern_type = "Plain"
    var pattern_type_tag = PatternTypeTag.Plain
    var uniform_excitations: List[UniformExcitationInput] = []
    var supports_linear_transient_fast_path = (
        not used_nonelastic_uniaxial
        and not shell_cache.has_soil_materials()
        and len(shell_cache.continuum.nonlinear) == 0
    )
    if supports_linear_transient_fast_path:
        for i in range(len(typed_elements)):
            if not _element_supports_linear_transient_fast_path(
                typed_elements[i], typed_sections_by_id
            ):
                supports_linear_transient_fast_path = False
                break
    if pattern_input.has_pattern:
        pattern_type = pattern_input.type
        pattern_type_tag = pattern_input.type_tag
        if (
            pattern_type_tag != PatternTypeTag.Plain
            and pattern_type_tag != PatternTypeTag.UniformExcitation
        ):
            abort("unsupported pattern type: " + pattern_type)
    if len(time_series) > 0 or pattern_input.has_pattern:
        if not pattern_input.has_pattern:
            if analysis_type_tag == AnalysisTypeTag.Staged and shell_cache.has_coupled_elements():
                # The staged solver resolves these series when their pattern starts.
                pass
            elif len(time_series) == 1:
                var ts_tag = time_series[0].tag
                ts_index = find_time_series_input(time_series, ts_tag)
                if ts_index < 0:
                    abort("time_series tag not found")
            else:
                abort("pattern missing for multiple time_series")
        elif pattern_type_tag == PatternTypeTag.Plain:
            if not pattern_input.has_time_series:
                abort("pattern missing time_series")
            var ts_tag = pattern_input.time_series
            ts_index = find_time_series_input(time_series, ts_tag)
            if ts_index < 0:
                abort("time_series tag not found")
        else:
            if (
                analysis_type_tag != AnalysisTypeTag.TransientLinear
                and analysis_type_tag != AnalysisTypeTag.TransientNonlinear
                and analysis_type_tag != AnalysisTypeTag.Staged
            ):
                abort("UniformExcitation requires transient analysis")
            uniform_excitations = resolve_uniform_excitations(
                pattern_input.excitations, time_series, ndm
            )
            if len(nodal_loads) > 0 or len(element_loads) > 0:
                abort("UniformExcitation does not support nodal/element loads")

    var rayleigh_alpha_m = 0.0
    var rayleigh_beta_k = 0.0
    var rayleigh_beta_k_init = 0.0
    var rayleigh_beta_k_comm = 0.0
    if rayleigh_input.has_rayleigh:
        rayleigh_alpha_m = rayleigh_input.alpha_m
        rayleigh_beta_k = rayleigh_input.beta_k
        rayleigh_beta_k_init = rayleigh_input.beta_k_init
        rayleigh_beta_k_comm = rayleigh_input.beta_k_comm

    _write_run_progress(progress_path, "loading", "case", 0, 0, 5, load_step_count)

    state.ndm = ndm
    state.ndf = ndf
    swap(state.typed_nodes, input.nodes)
    state.node_count = node_count
    state.id_to_index = id_to_index^
    state.node_x = node_x^
    state.node_y = node_y^
    state.node_z = node_z^
    state.node_has_z = node_has_z^
    state.node_constraint_offsets = node_constraint_offsets^
    state.node_constraint_pool = node_constraint_pool^
    state.typed_sections_by_id = typed_sections_by_id^
    state.typed_materials_by_id = typed_materials_by_id^
    state.uniaxial_defs = uniaxial_defs^
    state.uniaxial_state_defs = uniaxial_state_defs^
    state.uniaxial_states = uniaxial_states^
    state.fiber_section_defs = fiber_section_defs^
    state.fiber_section_cells = fiber_section_cells^
    state.fiber_section_index_by_id = fiber_section_index_by_id^
    state.fiber_section3d_defs = fiber_section3d_defs^
    state.fiber_section3d_cells = fiber_section3d_cells^
    state.fiber_section3d_index_by_id = fiber_section3d_index_by_id^
    state.layered_shell_section_defs = layered_shell_section_defs^
    swap(state.typed_elements, input.elements)
    state.elem_count = elem_count
    swap(state.element_loads, input.element_loads)
    state.elem_id_to_index = elem_id_to_index^
    state.elem_load_offsets = elem_load_offsets^
    state.elem_load_pool = elem_load_pool^
    state.elem_dof_offsets = elem_dof_offsets^
    state.elem_dof_pool = elem_dof_pool^
    state.elem_free_offsets = elem_free_offsets^
    state.elem_free_pool = elem_free_pool^
    state.elem_node_offsets = elem_node_offsets^
    state.elem_node_pool = elem_node_pool^
    state.elem_material_offsets = elem_material_offsets^
    state.elem_material_pool = elem_material_pool^
    state.elem_primary_material_ids = elem_primary_material_ids^
    state.elem_type_tags = elem_type_tags^
    state.elem_geom_tags = elem_geom_tags^
    state.elem_section_ids = elem_section_ids^
    state.shell_cache = shell_cache^
    state.elem_integration_tags = elem_integration_tags^
    state.elem_num_int_pts = elem_num_int_pts^
    state.elem_dof_counts = elem_dof_counts^
    state.elem_area = elem_area^
    state.elem_thickness = elem_thickness^
    state.frame2d_elem_indices = frame2d_elem_indices^
    state.frame3d_elem_indices = frame3d_elem_indices^
    state.truss_elem_indices = truss_elem_indices^
    state.zero_length_elem_indices = zero_length_elem_indices^
    state.two_node_link_elem_indices = two_node_link_elem_indices^
    state.zero_length_section_elem_indices = zero_length_section_elem_indices^
    state.quad_elem_indices = quad_elem_indices^
    state.shell_elem_indices = shell_elem_indices^
    state.elem_uniaxial_offsets = elem_uniaxial_offsets^
    state.elem_uniaxial_counts = elem_uniaxial_counts^
    state.elem_uniaxial_state_ids = elem_uniaxial_state_ids^
    state.force_basic_offsets = force_basic_offsets^
    state.force_basic_counts = force_basic_counts^
    state.force_basic_q = force_basic_q^
    state.total_dofs = total_dofs
    state.F_total = F_total^
    state.constrained = constrained^
    state.analysis = analysis_input
    state.analysis_type = analysis_type
    state.analysis_type_tag = analysis_type_tag
    state.steps = steps
    state.modal_num_modes = modal_num_modes
    state.constraints_handler = constraints_handler
    state.constraints_handler_tag = constraints_handler_tag
    state.use_banded_linear = use_banded_linear
    state.use_banded_nonlinear = use_banded_nonlinear
    state.has_transformation_mpc = has_transformation_mpc
    state.supports_linear_transient_fast_path = supports_linear_transient_fast_path
    state.free = free^
    state.free_index = free_index^
    state.rep_dof = rep_dof^
    state.mpc_row_offsets = mpc_row_offsets^
    state.mpc_dof_pool = mpc_dof_pool^
    state.mpc_coeff_pool = mpc_coeff_pool^
    state.mpc_slave_dof = mpc_slave_dof^
    state.active_index_by_dof = active_index_by_dof^
    state.M_total = M_total^
    state.M_rayleigh_total = M_rayleigh_total^
    state.nodal_masses = masses.copy()
    state.rayleigh = rayleigh_input
    swap(
        state.analysis_integrator_targets_pool,
        input.analysis_integrator_targets_pool,
    )
    swap(state.analysis_solver_chain_pool, input.analysis_solver_chain_pool)
    swap(state.time_series, input.time_series)
    swap(state.time_series_values, input.time_series_values)
    swap(state.time_series_times, input.time_series_times)
    swap(state.dampings, input.dampings)
    swap(state.stages, input.stages)
    state.ts_index = ts_index
    state.pattern_type = pattern_type
    state.pattern_type_tag = pattern_type_tag
    state.uniform_excitations = uniform_excitations^
    state.rayleigh_alpha_m = rayleigh_alpha_m
    state.rayleigh_beta_k = rayleigh_beta_k
    state.rayleigh_beta_k_init = rayleigh_beta_k_init
    state.rayleigh_beta_k_comm = rayleigh_beta_k_comm
    swap(state.recorder_nodes_pool, input.recorder_nodes_pool)
    swap(state.recorder_elements_pool, input.recorder_elements_pool)
    swap(state.recorder_dofs_pool, input.recorder_dofs_pool)
    swap(state.recorder_modes_pool, input.recorder_modes_pool)
    swap(state.recorder_sections_pool, input.recorder_sections_pool)
    swap(state.recorders, input.recorders)
    _write_run_progress(progress_path, "loading", analysis_type, 0, 0, 6, load_step_count)

    return state^
