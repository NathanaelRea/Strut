from collections import List
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
from solver.dof import node_dof_index, require_dof_in_range
from solver.reorder import build_node_adjacency_typed, min_degree_order, rcm_order
from solver.run_case.helpers import (
    _aggregator_section2d_expected_state_count,
    _collapse_vector_by_mpc,
    _write_run_progress,
)
from solver.run_case.input_types import (
    AnalysisInput,
    CaseInput,
    DampingInput,
    ElementLoadInput,
    ElementInput,
    MaterialInput,
    MPConstraintInput,
    NodeInput,
    RecorderInput,
    SectionInput,
    SolverAttemptInput,
    StageInput,
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
    var layered_shell_section_index_by_id: List[Int]
    var layered_shell_section_uniaxial_offsets: List[Int]
    var layered_shell_section_uniaxial_counts: List[Int]

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
    var shell_elem_instance_offsets: List[Int]
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
    var uniform_excitation_direction: Int
    var uniform_accel_ts_index: Int
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
        self.layered_shell_section_index_by_id = []
        self.layered_shell_section_uniaxial_offsets = []
        self.layered_shell_section_uniaxial_counts = []
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
        self.shell_elem_instance_offsets = []
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
        self.uniform_excitation_direction = 0
        self.uniform_accel_ts_index = -1
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
    else:
        elem.dof_24 = value


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


fn _remap_compact_index_or_abort(
    raw_id: Int, raw_to_compact: List[Int], label: String
) -> Int:
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
        or elem.type_tag == ElementTypeTag.FourNodeQuad
        or elem.type_tag == ElementTypeTag.Shell
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
    uniaxial_defs: List[UniMaterialDef],
    uniaxial_def_by_id: List[Int],
    beam_col_type: String,
) raises -> SectionInput:
    if sec.type != "AggregatorSection2d":
        return sec
    if sec.base_section >= 0:
        abort(beam_col_type + " does not support AggregatorSection2d with -section")
    if sec.axial_material < 0 or sec.flexural_material < 0:
        abort(beam_col_type + " requires AggregatorSection2d with P and Mz responses")
    if (
        sec.axial_material >= len(uniaxial_def_by_id)
        or uniaxial_def_by_id[sec.axial_material] < 0
    ):
        abort(beam_col_type + " AggregatorSection2d axial material must be uniaxial")
    if (
        sec.flexural_material >= len(uniaxial_def_by_id)
        or uniaxial_def_by_id[sec.flexural_material] < 0
    ):
        abort(
            beam_col_type + " AggregatorSection2d flexural material must be uniaxial"
        )
    var axial_def = uniaxial_defs[uniaxial_def_by_id[sec.axial_material]]
    var flexural_def = uniaxial_defs[uniaxial_def_by_id[sec.flexural_material]]
    sec.type = "ElasticSection2d"
    sec.E = 1.0
    sec.A = uni_mat_initial_tangent(axial_def)
    sec.I = uni_mat_initial_tangent(flexural_def)
    if sec.A <= 0.0 or sec.I <= 0.0:
        abort(beam_col_type + " AggregatorSection2d surrogate stiffness must be positive")
    return sec


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
    return elem.node_index_4


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
    return elem.dof_24


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
    if elem.type_tag == ElementTypeTag.DispBeamColumn2d or elem.type_tag == ElementTypeTag.DispBeamColumn3d:
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


fn load_case_state_from_input(
    var input: CaseInput, progress_path: String
) raises -> RunCaseState:
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
    ref shell_material_props = input.shell_material_props
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
    if not is_2d and not is_3d_truss and not is_3d_shell:
        abort("only ndm=2 ndf=2/3 and ndm=3 ndf=3/6 supported")

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
        if nid >= len(id_to_index):
            id_to_index.resize(nid + 1, -1)
        id_to_index[nid] = i

    var material_index_by_raw_id: List[Int] = []
    material_index_by_raw_id.resize(10000, -1)
    for i in range(len(materials)):
        var raw_id = materials[i].id
        if raw_id >= len(material_index_by_raw_id):
            material_index_by_raw_id.resize(raw_id + 1, -1)
        material_index_by_raw_id[raw_id] = i
    for i in range(len(materials)):
        var mat = materials[i]
        if mat.base_material >= 0:
            mat.base_material = _remap_compact_index_or_abort(
                mat.base_material, material_index_by_raw_id, "base material"
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

    var section_index_by_raw_id: List[Int] = []
    section_index_by_raw_id.resize(10000, -1)
    for i in range(len(sections)):
        var raw_id = sections[i].id
        if raw_id >= len(section_index_by_raw_id):
            section_index_by_raw_id.resize(raw_id + 1, -1)
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
        if sec.torsion_material >= 0:
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
        if sec.base_section >= 0:
            sec.base_section = _remap_compact_index_or_abort(
                sec.base_section, section_index_by_raw_id, "base section"
            )
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
        if mid >= len(uniaxial_def_by_id):
            uniaxial_def_by_id.resize(mid + 1, -1)
        var mat_type = mat.type
        if mat_type == "Elastic":
            var E = mat.E
            if E <= 0.0:
                abort("Elastic material E must be > 0")
            var mat_def = UniMaterialDef(UniMaterialTypeTag.Elastic, E, 0.0, 0.0, 0.0)
            uniaxial_def_by_id[mid] = len(uniaxial_defs)
            uniaxial_defs.append(mat_def)
        elif mat_type == "Steel01":
            var Fy = mat.Fy
            var E0 = mat.E0
            var b = mat.b
            if Fy <= 0.0:
                abort("Steel01 Fy must be > 0")
            if E0 <= 0.0:
                abort("Steel01 E0 must be > 0")
            if b < 0.0 or b >= 1.0:
                abort("Steel01 b must be in [0, 1)")
            var mat_def = UniMaterialDef(UniMaterialTypeTag.Steel01, Fy, E0, b, 0.0)
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
            if ft < 0.0:
                abort("Concrete02 ft must be >= 0")
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
        elif (
            mat_type == "ElasticIsotropic"
            or mat_type == "PlaneStressUserMaterial"
            or mat_type == "PlateFromPlaneStress"
            or mat_type == "PlateRebar"
        ):
            continue
        else:
            abort("unsupported material type: " + mat_type)

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
    var layered_shell_section_uniaxial_counts: List[Int] = []
    layered_shell_section_index_by_id.resize(len(typed_sections_by_id), -1)
    for i in range(len(sections)):
        var sec = sections[i]
        var sec_type = sec.type
        var sid = sec.id
        if sec_type == "FiberSection2d":
            if sid >= len(fiber_section_index_by_id):
                fiber_section_index_by_id.resize(sid + 1, -1)
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
            if sid >= len(fiber_section3d_index_by_id):
                fiber_section3d_index_by_id.resize(sid + 1, -1)
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
        elif sec_type == "LayeredShellSection":
            if sid >= len(layered_shell_section_index_by_id):
                layered_shell_section_index_by_id.resize(sid + 1, -1)
            append_layered_shell_section_from_input(
                layered_shell_section_defs,
                sec,
                shell_layers,
                typed_materials_by_id,
                uniaxial_def_by_id,
                shell_material_props,
            )
            layered_shell_section_index_by_id[sid] = len(layered_shell_section_defs) - 1

    _write_run_progress(progress_path, "loading", "case", 0, 0, 3, load_step_count)

    var elem_count = len(typed_elements)
    var elem_ids: List[Int] = []
    elem_ids.resize(elem_count, 0)
    var elem_id_to_index: List[Int] = []
    elem_id_to_index.resize(10000, -1)
    for i in range(elem_count):
        var eid = typed_elements[i].id
        elem_ids[i] = eid
        if eid >= len(elem_id_to_index):
            elem_id_to_index.resize(eid + 1, -1)
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
                abort(
                    beam_col_type
                    + " supports geomTransf Linear, PDelta, or Corotational"
                )
            beam_integration_validate_or_abort(
                beam_col_type, elem.integration, elem.num_int_pts
            )
            if elem.section < 0 or elem.section >= len(typed_sections_by_id):
                abort(beam_col_type + " section not found")
            var sec = typed_sections_by_id[elem.section]
            if sec.id < 0:
                abort(beam_col_type + " section not found")
            sec = _coerce_aggregator_section2d_for_beam_column(
                sec, uniaxial_defs, uniaxial_def_by_id, beam_col_type
            )
            typed_sections_by_id[elem.section] = sec
            if sec.type != "FiberSection2d" and sec.type != "ElasticSection2d":
                abort(beam_col_type + " requires FiberSection2d or ElasticSection2d")
            elem.dof_count = 6
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
            ):
                abort("dispBeamColumn2d supports geomTransf Linear or PDelta")
            beam_integration_validate_or_abort(
                "dispBeamColumn2d", elem.integration, elem.num_int_pts
            )
            if elem.section < 0 or elem.section >= len(typed_sections_by_id):
                abort("dispBeamColumn2d section not found")
            var sec = typed_sections_by_id[elem.section]
            if sec.id < 0:
                abort("dispBeamColumn2d section not found")
            sec = _coerce_aggregator_section2d_for_beam_column(
                sec, uniaxial_defs, uniaxial_def_by_id, "dispBeamColumn2d"
            )
            typed_sections_by_id[elem.section] = sec
            if sec.type != "FiberSection2d" and sec.type != "ElasticSection2d":
                abort("dispBeamColumn2d requires FiberSection2d or ElasticSection2d")
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
                abort(
                    beam_col_type + " supports geomTransf Linear, PDelta, or Corotational"
                )
            beam_integration_validate_or_abort(
                beam_col_type, elem.integration, elem.num_int_pts
            )
            if elem.section < 0 or elem.section >= len(typed_sections_by_id):
                abort(beam_col_type + " section not found")
            var sec = typed_sections_by_id[elem.section]
            if sec.id < 0:
                abort(beam_col_type + " section not found")
            if sec.type != "ElasticSection3d" and sec.type != "FiberSection3d":
                abort(beam_col_type + " requires ElasticSection3d or FiberSection3d")
            if sec.type == "FiberSection3d" and (sec.G <= 0.0 or sec.J <= 0.0):
                abort(beam_col_type + " with FiberSection3d requires positive G and J")
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
            if elem.material_count != elem.dir_count:
                abort("zeroLength materials/dirs mismatch")
            if (
                elem.damp_material_count > 0
                and elem.damp_material_count != elem.material_count
            ):
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
            for d in range(elem.dir_count):
                _set_elem_dir(
                    elem,
                    d,
                    zero_length_internal_dir(_elem_dir(elem, d), ndm, ndf),
                )
            for d in range(ndf):
                _set_elem_dof(elem, d, node_dof_index(elem.node_index_1, d + 1, ndf))
                _set_elem_dof(
                    elem, d + ndf, node_dof_index(elem.node_index_2, d + 1, ndf)
                )
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
                _set_elem_dof(
                    elem, d + ndf, node_dof_index(elem.node_index_2, d + 1, ndf)
                )
        elif elem.type_tag == ElementTypeTag.ZeroLengthSection:
            if ndm != 2 or ndf != 3:
                abort("zeroLengthSection requires ndm=2, ndf=3")
            if elem.node_count != 2:
                abort("zeroLengthSection requires 2 nodes")
            if elem.section < 0 or elem.section >= len(typed_sections_by_id):
                abort("zeroLengthSection section not found")
            var sec = typed_sections_by_id[elem.section]
            if sec.id < 0:
                abort("zeroLengthSection section not found")
            if (
                sec.type != "FiberSection2d"
                and sec.type != "ElasticSection2d"
                and sec.type != "AggregatorSection2d"
            ):
                abort(
                    "zeroLengthSection requires FiberSection2d, ElasticSection2d, or AggregatorSection2d"
                )
            elem.dof_count = 6
            _set_elem_dof(elem, 0, node_dof_index(elem.node_index_1, 1, ndf))
            _set_elem_dof(elem, 1, node_dof_index(elem.node_index_1, 2, ndf))
            _set_elem_dof(elem, 2, node_dof_index(elem.node_index_1, 3, ndf))
            _set_elem_dof(elem, 3, node_dof_index(elem.node_index_2, 1, ndf))
            _set_elem_dof(elem, 4, node_dof_index(elem.node_index_2, 2, ndf))
            _set_elem_dof(elem, 5, node_dof_index(elem.node_index_2, 3, ndf))
        elif elem.type_tag == ElementTypeTag.FourNodeQuad:
            if ndm != 2 or ndf != 2:
                abort("fourNodeQuad requires ndm=2, ndf=2")
            if elem.node_count != 4:
                abort("fourNodeQuad requires 4 nodes")
            if elem.formulation != "PlaneStress":
                abort("fourNodeQuad only supports PlaneStress formulation")
            if elem.material < 0 or elem.material >= len(typed_materials_by_id):
                abort("material not found")
            var mat = typed_materials_by_id[elem.material]
            if mat.id < 0:
                abort("material not found")
            if mat.type != "ElasticIsotropic":
                abort("fourNodeQuad requires ElasticIsotropic material")
            if elem.thickness <= 0.0:
                abort("fourNodeQuad requires thickness > 0")
            elem.dof_count = 8
            _set_elem_dof(elem, 0, node_dof_index(elem.node_index_1, 1, ndf))
            _set_elem_dof(elem, 1, node_dof_index(elem.node_index_1, 2, ndf))
            _set_elem_dof(elem, 2, node_dof_index(elem.node_index_2, 1, ndf))
            _set_elem_dof(elem, 3, node_dof_index(elem.node_index_2, 2, ndf))
            _set_elem_dof(elem, 4, node_dof_index(elem.node_index_3, 1, ndf))
            _set_elem_dof(elem, 5, node_dof_index(elem.node_index_3, 2, ndf))
            _set_elem_dof(elem, 6, node_dof_index(elem.node_index_4, 1, ndf))
            _set_elem_dof(elem, 7, node_dof_index(elem.node_index_4, 2, ndf))
        elif elem.type_tag == ElementTypeTag.Shell:
            if ndm != 3 or ndf != 6:
                abort("shell requires ndm=3, ndf=6")
            if elem.node_count != 4:
                abort("shell requires 4 nodes")
            if elem.section < 0 or elem.section >= len(typed_sections_by_id):
                abort("section not found")
            var sec = typed_sections_by_id[elem.section]
            if sec.id < 0:
                abort("section not found")
            if sec.type != "ElasticMembranePlateSection" and sec.type != "LayeredShellSection":
                abort("shell requires ElasticMembranePlateSection or LayeredShellSection")
            elem.dof_count = 24
            for d in range(6):
                _set_elem_dof(elem, d, node_dof_index(elem.node_index_1, d + 1, ndf))
                _set_elem_dof(elem, d + 6, node_dof_index(elem.node_index_2, d + 1, ndf))
                _set_elem_dof(elem, d + 12, node_dof_index(elem.node_index_3, d + 1, ndf))
                _set_elem_dof(elem, d + 18, node_dof_index(elem.node_index_4, d + 1, ndf))
        else:
            abort("unsupported element type: " + elem.type)

        typed_elements[i] = elem

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
        elif elem.type_tag == ElementTypeTag.TwoNodeLink:
            two_node_link_elem_indices.append(e)
        elif elem.type_tag == ElementTypeTag.ZeroLengthSection:
            zero_length_section_elem_indices.append(e)
        elif elem.type_tag == ElementTypeTag.FourNodeQuad:
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
                uniaxial_states.append(UniMaterialState(mat_def))
                uniaxial_state_defs.append(def_index)
                elem_uniaxial_state_ids.append(state_index)
                if not uni_mat_is_elastic(mat_def):
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
            var predictor_slots = 0
            if sec.type == "FiberSection2d":
                predictor_slots = 2 * elem.num_int_pts + 3
            elif sec.type != "ElasticSection2d":
                abort(beam_col_type + " requires FiberSection2d or ElasticSection2d")
            var active_basic_count = 3 + predictor_slots
            force_basic_counts[e] = active_basic_count
            for _ in range(force_basic_counts[e]):
                force_basic_q.append(0.0)
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
                var sec_def = fiber_section3d_defs[sec_index]
                var num_int_pts = elem.num_int_pts
                var state_count = num_int_pts * sec_def.fiber_count
                if elem.type_tag == ElementTypeTag.ForceBeamColumn3d:
                    force_basic_counts[e] = 5 + 3 * num_int_pts
                    for _ in range(force_basic_counts[e]):
                        force_basic_q.append(0.0)
                elem_uniaxial_counts[e] = state_count
                for _ in range(num_int_pts):
                    for i in range(sec_def.elastic_count):
                        var def_index = sec_def.elastic_def_index[i]
                        if def_index < 0 or def_index >= len(uniaxial_defs):
                            abort(beam_col_type + " fiber material definition out of range")
                        var mat_def = uniaxial_defs[def_index]
                        var state_index = len(uniaxial_states)
                        uniaxial_states.append(UniMaterialState(mat_def))
                        uniaxial_state_defs.append(def_index)
                        elem_uniaxial_state_ids.append(state_index)
                    for i in range(sec_def.nonlinear_count):
                        var def_index = sec_def.nonlinear_def_index[i]
                        if def_index < 0 or def_index >= len(uniaxial_defs):
                            abort(beam_col_type + " fiber material definition out of range")
                        var mat_def = uniaxial_defs[def_index]
                        var state_index = len(uniaxial_states)
                        uniaxial_states.append(UniMaterialState(mat_def))
                        uniaxial_state_defs.append(def_index)
                        elem_uniaxial_state_ids.append(state_index)
                        if not uni_mat_is_elastic(mat_def):
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
                elem_uniaxial_offsets[e] = fiber_section2d_runtime_alloc_instances(
                    sec_def, 1
                )
                elem_uniaxial_counts[e] = state_count
                for _ in range(state_count):
                    elem_uniaxial_state_ids.append(-1)
                for i in range(sec_def.nonlinear_count):
                    var mat_def = sec_def.nonlinear_mat_defs[i]
                    if not uni_mat_is_elastic(mat_def):
                        used_nonelastic_uniaxial = True
            elif sec.type == "AggregatorSection2d":
                var expected_state_count = _aggregator_section2d_expected_state_count(sec)
                elem_uniaxial_counts[e] = expected_state_count
                if sec.axial_material >= 0:
                    if (
                        sec.axial_material >= len(uniaxial_def_by_id)
                        or uniaxial_def_by_id[sec.axial_material] < 0
                    ):
                        abort("AggregatorSection2d axial material must be uniaxial")
                    var def_index = uniaxial_def_by_id[sec.axial_material]
                    var mat_def = uniaxial_defs[def_index]
                    var state_index = len(uniaxial_states)
                    uniaxial_states.append(UniMaterialState(mat_def))
                    uniaxial_state_defs.append(def_index)
                    elem_uniaxial_state_ids.append(state_index)
                    if not uni_mat_is_elastic(mat_def):
                        used_nonelastic_uniaxial = True
                if sec.flexural_material >= 0:
                    if (
                        sec.flexural_material >= len(uniaxial_def_by_id)
                        or uniaxial_def_by_id[sec.flexural_material] < 0
                    ):
                        abort("AggregatorSection2d flexural material must be uniaxial")
                    var def_index = uniaxial_def_by_id[sec.flexural_material]
                    var mat_def = uniaxial_defs[def_index]
                    var state_index = len(uniaxial_states)
                    uniaxial_states.append(UniMaterialState(mat_def))
                    uniaxial_state_defs.append(def_index)
                    elem_uniaxial_state_ids.append(state_index)
                    if not uni_mat_is_elastic(mat_def):
                        used_nonelastic_uniaxial = True
            elif sec.type == "ElasticSection2d":
                elem_uniaxial_counts[e] = 0
            else:
                abort(
                    "zeroLengthSection requires FiberSection2d, ElasticSection2d, or AggregatorSection2d"
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
    if layered_shell_runtime_alloc_instances(
        layered_shell_section_defs,
        layered_shell_instance_counts,
        uniaxial_defs,
        uniaxial_states,
        uniaxial_state_defs,
        layered_shell_section_uniaxial_offsets,
        layered_shell_section_uniaxial_counts,
    ):
        used_nonelastic_uniaxial = True

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
            _accumulate_two_node_link_lumped_mass(
                typed_elements[e], ndm, ndf, M_total
            )
            if typed_elements[e].do_rayleigh:
                _accumulate_two_node_link_lumped_mass(
                    typed_elements[e], ndm, ndf, M_rayleigh_total
                )

    # Lump translational mass from fourNodeQuad/bbarQuad density when provided.
    for e in range(elem_count):
        var elem = typed_elements[e]
        if elem.type_tag != ElementTypeTag.FourNodeQuad:
            continue
        if elem.material < 0 or elem.material >= len(typed_materials_by_id):
            abort("material not found")
        var mat = typed_materials_by_id[elem.material]
        if mat.id < 0:
            abort("material not found")
        var rho = mat.rho
        if rho == 0.0:
            continue

        var n1 = typed_nodes[elem.node_index_1]
        var n2 = typed_nodes[elem.node_index_2]
        var n3 = typed_nodes[elem.node_index_3]
        var n4 = typed_nodes[elem.node_index_4]
        var twice_area = (
            n1.x * n2.y
            - n1.y * n2.x
            + n2.x * n3.y
            - n2.y * n3.x
            + n3.x * n4.y
            - n3.y * n4.x
            + n4.x * n1.y
            - n4.y * n1.x
        )
        var area = abs(twice_area) * 0.5
        if area <= 0.0:
            abort("fourNodeQuad area must be > 0")
        var lumped = rho * elem.thickness * area / 4.0
        if lumped == 0.0:
            continue
        M_total[node_dof_index(elem.node_index_1, 1, ndf)] += lumped
        M_total[node_dof_index(elem.node_index_1, 2, ndf)] += lumped
        M_total[node_dof_index(elem.node_index_2, 1, ndf)] += lumped
        M_total[node_dof_index(elem.node_index_2, 2, ndf)] += lumped
        M_total[node_dof_index(elem.node_index_3, 1, ndf)] += lumped
        M_total[node_dof_index(elem.node_index_3, 2, ndf)] += lumped
        M_total[node_dof_index(elem.node_index_4, 1, ndf)] += lumped
        M_total[node_dof_index(elem.node_index_4, 2, ndf)] += lumped

    for e in range(elem_count):
        var elem = typed_elements[e]
        if elem.type_tag != ElementTypeTag.Shell:
            continue
        if elem.section < 0 or elem.section >= len(typed_sections_by_id):
            abort("shell section not found")
        var sec = typed_sections_by_id[elem.section]
        var rho_area = 0.0
        if sec.type == "ElasticMembranePlateSection":
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
        var n4 = typed_nodes[elem.node_index_4]
        var ax = (n2.x - n1.x) + (n3.x - n4.x)
        var ay = (n2.y - n1.y) + (n3.y - n4.y)
        var az = (n2.z - n1.z) + (n3.z - n4.z)
        var bx = (n4.x - n1.x) + (n3.x - n2.x)
        var by = (n4.y - n1.y) + (n3.y - n2.y)
        var bz = (n4.z - n1.z) + (n3.z - n2.z)
        var cross_x = ay * bz - az * by
        var cross_y = az * bx - ax * bz
        var cross_z = ax * by - ay * bx
        var area = 0.25 * sqrt(cross_x * cross_x + cross_y * cross_y + cross_z * cross_z)
        if area <= 0.0:
            abort("shell area must be > 0")
        var lumped = rho_area * area / 4.0
        for dof in range(3):
            M_total[node_dof_index(elem.node_index_1, dof + 1, ndf)] += lumped
            M_total[node_dof_index(elem.node_index_2, dof + 1, ndf)] += lumped
            M_total[node_dof_index(elem.node_index_3, dof + 1, ndf)] += lumped
            M_total[node_dof_index(elem.node_index_4, dof + 1, ndf)] += lumped

    var constrained: List[Bool] = []
    constrained.resize(total_dofs, False)
    for i in range(node_count):
        var node = typed_nodes[i]
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
            elif (
                analysis_type_tag == AnalysisTypeTag.StaticLinear
                and force_beam_has_nonelastic
            ):
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
        abort("nonlinear uniaxial materials require static_nonlinear or transient_nonlinear analysis")

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
        ):
            abort(
                "[load-fail] mp_constraints require analysis.constraints=Transformation or Lagrange"
            )
        has_transformation_mpc = True
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
        if node_is_mpc_constrained[retained_node]:
            abort("[load-fail] " + mpc_type + " retained_node cannot also be constrained")
        var retained_idx = id_to_index[retained_node]
        var constrained_idx = id_to_index[constrained_node]
        if mpc_type == "equalDOF":
            if mpc.dof_count == 0:
                abort("[load-fail] equalDOF requires non-empty dofs")
            for j in range(mpc.dof_count):
                var dof = _mp_constraint_dof(mpc, j)
                require_dof_in_range(dof, ndf, "equalDOF")
                var retained_dof = node_dof_index(retained_idx, dof, ndf)
                var constrained_dof = node_dof_index(constrained_idx, dof, ndf)
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
                var constrained_dof = node_dof_index(
                    constrained_idx, constrained_dof_no, ndf
                )
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
    var uniform_excitation_direction = 0
    var uniform_accel_ts_index = -1
    var supports_linear_transient_fast_path = not used_nonelastic_uniaxial
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
            if len(time_series) == 1:
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
            if not pattern_input.has_direction:
                abort("UniformExcitation pattern missing direction")
            uniform_excitation_direction = pattern_input.direction
            if uniform_excitation_direction < 1 or uniform_excitation_direction > ndm:
                abort("UniformExcitation direction out of range 1..ndm")
            var accel_tag = -1
            if pattern_input.has_accel:
                accel_tag = pattern_input.accel
            elif pattern_input.has_time_series:
                accel_tag = pattern_input.time_series
            else:
                abort("UniformExcitation pattern missing accel time_series tag")
            uniform_accel_ts_index = find_time_series_input(time_series, accel_tag)
            if uniform_accel_ts_index < 0:
                abort("UniformExcitation accel time_series tag not found")
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
    state.layered_shell_section_index_by_id = layered_shell_section_index_by_id^
    state.layered_shell_section_uniaxial_offsets = (
        layered_shell_section_uniaxial_offsets^
    )
    state.layered_shell_section_uniaxial_counts = (
        layered_shell_section_uniaxial_counts^
    )
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
    state.shell_elem_instance_offsets = shell_elem_instance_offsets^
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
    state.uniform_excitation_direction = uniform_excitation_direction
    state.uniform_accel_ts_index = uniform_accel_ts_index
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
