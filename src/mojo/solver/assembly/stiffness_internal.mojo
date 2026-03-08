from collections import List
from math import hypot, sqrt
from os import abort

from elements import (
    ForceBeamColumn2dScratch,
    ForceBeamColumn3dScratch,
    truss_global_stiffness,
    truss3d_global_stiffness,
)
from materials import (
    UniMaterialDef,
    UniMaterialState,
    uni_mat_initial_tangent,
    uniaxial_set_trial_strain,
)
from solver.assembly.stiffness_internal_frame2d import (
    _assemble_frame2d_element,
    _assemble_frame2d_soa_indices,
)
from solver.assembly.stiffness_internal_frame3d import (
    _assemble_frame3d_element,
    _assemble_frame3d_soa_indices,
)
from solver.assembly.stiffness_internal_links import (
    _assemble_two_node_link_element,
    _assemble_zero_length_element,
)
from solver.assembly.stiffness_internal_shared import (
    _assembly_filter_accepts_element,
    _beam_integration_name_from_tag,
    _build_elem_dof_soa,
    _elem_dof,
    _elem_dof_map,
    _geom_transf_name_from_tag,
    _profile_scope_close,
    _profile_scope_open,
    _scatter_add_and_dot_row_simd,
    _scatter_add_row,
    _zero_matrix,
    _zero_vector,
)
from solver.assembly.stiffness_internal_surface import (
    _assemble_surface_element,
    _assemble_surface_soa_indices,
)
from solver.dof import node_dof_index
from solver.profile import (
    PROFILE_FRAME_ASSEMBLE_FIBER_GEOMETRY,
    PROFILE_FRAME_ASSEMBLE_FIBER_INTERNAL_FORCE,
    PROFILE_FRAME_ASSEMBLE_FIBER_MATRIX_SCATTER,
    PROFILE_FRAME_ASSEMBLE_FIBER_SECTION_RESPONSE,
    PROFILE_FRAME_UNIAXIAL_TRIAL_UPDATE,
)
from solver.run_case.helpers import aggregator_section2d_set_trial_from_offset
from solver.run_case.input_types import (
    ElementInput,
    ElementLoadInput,
    MaterialInput,
    NodeInput,
    SectionInput,
)
from sections import (
    FiberCell,
    FiberSection2dDef,
    FiberSection3dDef,
    fiber_section2d_set_trial_from_offset,
)
from tag_types import BeamIntegrationTag, ElementTypeTag, GeomTransfTag


fn assemble_global_stiffness_typed(
    nodes: List[NodeInput],
    elements: List[ElementInput],
    sections_by_id: List[SectionInput],
    materials_by_id: List[MaterialInput],
    id_to_index: List[Int],
    node_count: Int,
    ndf: Int,
    ndm: Int,
    u: List[Float64],
    uniaxial_defs: List[UniMaterialDef],
    uniaxial_state_defs: List[Int],
    mut uniaxial_states: List[UniMaterialState],
    elem_uniaxial_offsets: List[Int],
    elem_uniaxial_counts: List[Int],
    elem_uniaxial_state_ids: List[Int],
    force_basic_offsets: List[Int],
    force_basic_counts: List[Int],
    mut force_basic_q: List[Float64],
    fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    fiber_section3d_defs: List[FiberSection3dDef],
    fiber_section3d_cells: List[FiberCell],
    fiber_section3d_index_by_id: List[Int],
) raises -> List[List[Float64]]:
    var empty_element_loads: List[ElementLoadInput] = []
    var empty_elem_load_offsets: List[Int] = []
    var empty_elem_load_pool: List[Int] = []
    return assemble_global_stiffness_typed(
        nodes,
        elements,
        empty_element_loads,
        empty_elem_load_offsets,
        empty_elem_load_pool,
        0.0,
        sections_by_id,
        materials_by_id,
        id_to_index,
        node_count,
        ndf,
        ndm,
        u,
        uniaxial_defs,
        uniaxial_state_defs,
        uniaxial_states,
        elem_uniaxial_offsets,
        elem_uniaxial_counts,
        elem_uniaxial_state_ids,
        force_basic_offsets,
        force_basic_counts,
        force_basic_q,
        fiber_section_defs,
        fiber_section_cells,
        fiber_section_index_by_id,
        fiber_section3d_defs,
        fiber_section3d_cells,
        fiber_section3d_index_by_id,
    )


fn assemble_global_stiffness_typed(
    nodes: List[NodeInput],
    elements: List[ElementInput],
    element_loads: List[ElementLoadInput],
    elem_load_offsets: List[Int],
    elem_load_pool: List[Int],
    load_scale: Float64,
    sections_by_id: List[SectionInput],
    materials_by_id: List[MaterialInput],
    id_to_index: List[Int],
    node_count: Int,
    ndf: Int,
    ndm: Int,
    u: List[Float64],
    uniaxial_defs: List[UniMaterialDef],
    uniaxial_state_defs: List[Int],
    mut uniaxial_states: List[UniMaterialState],
    elem_uniaxial_offsets: List[Int],
    elem_uniaxial_counts: List[Int],
    elem_uniaxial_state_ids: List[Int],
    force_basic_offsets: List[Int],
    force_basic_counts: List[Int],
    mut force_basic_q: List[Float64],
    fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    fiber_section3d_defs: List[FiberSection3dDef],
    fiber_section3d_cells: List[FiberCell],
    fiber_section3d_index_by_id: List[Int],
) raises -> List[List[Float64]]:
    var total_dofs = node_count * ndf
    var K: List[List[Float64]] = []
    for _ in range(total_dofs):
        var row: List[Float64] = []
        row.resize(total_dofs, 0.0)
        K.append(row^)
    var F_int: List[Float64] = []
    F_int.resize(total_dofs, 0.0)
    assemble_global_stiffness_and_internal(
        nodes,
        elements,
        element_loads,
        elem_load_offsets,
        elem_load_pool,
        load_scale,
        sections_by_id,
        materials_by_id,
        id_to_index,
        node_count,
        ndf,
        ndm,
        u,
        uniaxial_defs,
        uniaxial_state_defs,
        uniaxial_states,
        elem_uniaxial_offsets,
        elem_uniaxial_counts,
        elem_uniaxial_state_ids,
        force_basic_offsets,
        force_basic_counts,
        force_basic_q,
        fiber_section_defs,
        fiber_section_cells,
        fiber_section_index_by_id,
        fiber_section3d_defs,
        fiber_section3d_cells,
        fiber_section3d_index_by_id,
        K,
        F_int,
    )
    return K^


fn assemble_link_stiffness_typed(
    nodes: List[NodeInput],
    elements: List[ElementInput],
    sections_by_id: List[SectionInput],
    materials_by_id: List[MaterialInput],
    id_to_index: List[Int],
    node_count: Int,
    ndf: Int,
    ndm: Int,
    u: List[Float64],
    uniaxial_defs: List[UniMaterialDef],
    uniaxial_state_defs: List[Int],
    mut uniaxial_states: List[UniMaterialState],
    elem_uniaxial_offsets: List[Int],
    elem_uniaxial_counts: List[Int],
    elem_uniaxial_state_ids: List[Int],
    force_basic_offsets: List[Int],
    force_basic_counts: List[Int],
    mut force_basic_q: List[Float64],
    fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    fiber_section3d_defs: List[FiberSection3dDef],
    fiber_section3d_cells: List[FiberCell],
    fiber_section3d_index_by_id: List[Int],
    elem_dof_offsets: List[Int],
    elem_dof_pool: List[Int],
    mut dof_map6: List[Int],
    mut dof_map12: List[Int],
    mut u_elem6: List[Float64],
    only_rayleigh_participating: Bool,
    mut K: List[List[Float64]],
) raises:
    var total_dofs = node_count * ndf
    var F_dummy: List[Float64] = []
    F_dummy.resize(total_dofs, 0.0)
    var empty_element_loads: List[ElementLoadInput] = []
    var empty_elem_load_offsets: List[Int] = []
    var empty_elem_load_pool: List[Int] = []
    var filter_mode = 1
    if only_rayleigh_participating:
        filter_mode = 2
    _assemble_global_stiffness_and_internal_filtered(
        nodes,
        elements,
        empty_element_loads,
        empty_elem_load_offsets,
        empty_elem_load_pool,
        0.0,
        sections_by_id,
        materials_by_id,
        id_to_index,
        node_count,
        ndf,
        ndm,
        u,
        uniaxial_defs,
        uniaxial_state_defs,
        uniaxial_states,
        elem_uniaxial_offsets,
        elem_uniaxial_counts,
        elem_uniaxial_state_ids,
        force_basic_offsets,
        force_basic_counts,
        force_basic_q,
        fiber_section_defs,
        fiber_section_cells,
        fiber_section_index_by_id,
        fiber_section3d_defs,
        fiber_section3d_cells,
        fiber_section3d_index_by_id,
        elem_dof_offsets,
        elem_dof_pool,
        dof_map6,
        dof_map12,
        u_elem6,
        filter_mode,
        K,
        F_dummy,
    )


fn assemble_internal_forces_typed_soa(
    nodes: List[NodeInput],
    elements: List[ElementInput],
    node_x: List[Float64],
    node_y: List[Float64],
    node_z: List[Float64],
    elem_dof_offsets: List[Int],
    elem_dof_pool: List[Int],
    elem_node_offsets: List[Int],
    elem_node_pool: List[Int],
    elem_primary_material_ids: List[Int],
    elem_type_tags: List[Int],
    elem_geom_tags: List[Int],
    elem_section_ids: List[Int],
    elem_integration_tags: List[Int],
    elem_num_int_pts: List[Int],
    elem_area: List[Float64],
    elem_thickness: List[Float64],
    frame2d_elem_indices: List[Int],
    frame3d_elem_indices: List[Int],
    truss_elem_indices: List[Int],
    zero_length_elem_indices: List[Int],
    two_node_link_elem_indices: List[Int],
    zero_length_section_elem_indices: List[Int],
    quad_elem_indices: List[Int],
    shell_elem_indices: List[Int],
    sections_by_id: List[SectionInput],
    materials_by_id: List[MaterialInput],
    id_to_index: List[Int],
    node_count: Int,
    ndf: Int,
    ndm: Int,
    u: List[Float64],
    uniaxial_defs: List[UniMaterialDef],
    uniaxial_state_defs: List[Int],
    mut uniaxial_states: List[UniMaterialState],
    elem_uniaxial_offsets: List[Int],
    elem_uniaxial_counts: List[Int],
    elem_uniaxial_state_ids: List[Int],
    force_basic_offsets: List[Int],
    force_basic_counts: List[Int],
    mut force_basic_q: List[Float64],
    fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    fiber_section3d_defs: List[FiberSection3dDef],
    fiber_section3d_cells: List[FiberCell],
    fiber_section3d_index_by_id: List[Int],
) raises -> List[Float64]:
    var empty_element_loads: List[ElementLoadInput] = []
    var empty_elem_load_offsets: List[Int] = []
    var empty_elem_load_pool: List[Int] = []
    return assemble_internal_forces_typed_soa(
        nodes,
        elements,
        node_x,
        node_y,
        node_z,
        elem_dof_offsets,
        elem_dof_pool,
        elem_node_offsets,
        elem_node_pool,
        elem_primary_material_ids,
        elem_type_tags,
        elem_geom_tags,
        elem_section_ids,
        elem_integration_tags,
        elem_num_int_pts,
        elem_area,
        elem_thickness,
        frame2d_elem_indices,
        frame3d_elem_indices,
        truss_elem_indices,
        zero_length_elem_indices,
        two_node_link_elem_indices,
        zero_length_section_elem_indices,
        quad_elem_indices,
        shell_elem_indices,
        empty_element_loads,
        empty_elem_load_offsets,
        empty_elem_load_pool,
        0.0,
        sections_by_id,
        materials_by_id,
        id_to_index,
        node_count,
        ndf,
        ndm,
        u,
        uniaxial_defs,
        uniaxial_state_defs,
        uniaxial_states,
        elem_uniaxial_offsets,
        elem_uniaxial_counts,
        elem_uniaxial_state_ids,
        force_basic_offsets,
        force_basic_counts,
        force_basic_q,
        fiber_section_defs,
        fiber_section_cells,
        fiber_section_index_by_id,
        fiber_section3d_defs,
        fiber_section3d_cells,
        fiber_section3d_index_by_id,
    )


fn assemble_internal_forces_typed_soa(
    nodes: List[NodeInput],
    elements: List[ElementInput],
    node_x: List[Float64],
    node_y: List[Float64],
    node_z: List[Float64],
    elem_dof_offsets: List[Int],
    elem_dof_pool: List[Int],
    elem_node_offsets: List[Int],
    elem_node_pool: List[Int],
    elem_primary_material_ids: List[Int],
    elem_type_tags: List[Int],
    elem_geom_tags: List[Int],
    elem_section_ids: List[Int],
    elem_integration_tags: List[Int],
    elem_num_int_pts: List[Int],
    elem_area: List[Float64],
    elem_thickness: List[Float64],
    frame2d_elem_indices: List[Int],
    frame3d_elem_indices: List[Int],
    truss_elem_indices: List[Int],
    zero_length_elem_indices: List[Int],
    two_node_link_elem_indices: List[Int],
    zero_length_section_elem_indices: List[Int],
    quad_elem_indices: List[Int],
    shell_elem_indices: List[Int],
    element_loads: List[ElementLoadInput],
    elem_load_offsets: List[Int],
    elem_load_pool: List[Int],
    load_scale: Float64,
    sections_by_id: List[SectionInput],
    materials_by_id: List[MaterialInput],
    id_to_index: List[Int],
    node_count: Int,
    ndf: Int,
    ndm: Int,
    u: List[Float64],
    uniaxial_defs: List[UniMaterialDef],
    uniaxial_state_defs: List[Int],
    mut uniaxial_states: List[UniMaterialState],
    elem_uniaxial_offsets: List[Int],
    elem_uniaxial_counts: List[Int],
    elem_uniaxial_state_ids: List[Int],
    force_basic_offsets: List[Int],
    force_basic_counts: List[Int],
    mut force_basic_q: List[Float64],
    fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    fiber_section3d_defs: List[FiberSection3dDef],
    fiber_section3d_cells: List[FiberCell],
    fiber_section3d_index_by_id: List[Int],
) raises -> List[Float64]:
    var force_beam_column2d_scratch = ForceBeamColumn2dScratch()
    var force_beam_column3d_scratch = ForceBeamColumn3dScratch()
    return assemble_internal_forces_typed_soa(
        nodes,
        elements,
        node_x,
        node_y,
        node_z,
        elem_dof_offsets,
        elem_dof_pool,
        elem_node_offsets,
        elem_node_pool,
        elem_primary_material_ids,
        elem_type_tags,
        elem_geom_tags,
        elem_section_ids,
        elem_integration_tags,
        elem_num_int_pts,
        elem_area,
        elem_thickness,
        frame2d_elem_indices,
        frame3d_elem_indices,
        truss_elem_indices,
        zero_length_elem_indices,
        two_node_link_elem_indices,
        zero_length_section_elem_indices,
        quad_elem_indices,
        shell_elem_indices,
        element_loads,
        elem_load_offsets,
        elem_load_pool,
        load_scale,
        sections_by_id,
        materials_by_id,
        id_to_index,
        node_count,
        ndf,
        ndm,
        u,
        uniaxial_defs,
        uniaxial_state_defs,
        uniaxial_states,
        elem_uniaxial_offsets,
        elem_uniaxial_counts,
        elem_uniaxial_state_ids,
        force_basic_offsets,
        force_basic_counts,
        force_basic_q,
        fiber_section_defs,
        fiber_section_cells,
        fiber_section_index_by_id,
        fiber_section3d_defs,
        fiber_section3d_cells,
        fiber_section3d_index_by_id,
        force_beam_column2d_scratch,
        force_beam_column3d_scratch,
    )


fn assemble_internal_forces_typed_soa(
    nodes: List[NodeInput],
    elements: List[ElementInput],
    node_x: List[Float64],
    node_y: List[Float64],
    node_z: List[Float64],
    elem_dof_offsets: List[Int],
    elem_dof_pool: List[Int],
    elem_node_offsets: List[Int],
    elem_node_pool: List[Int],
    elem_primary_material_ids: List[Int],
    elem_type_tags: List[Int],
    elem_geom_tags: List[Int],
    elem_section_ids: List[Int],
    elem_integration_tags: List[Int],
    elem_num_int_pts: List[Int],
    elem_area: List[Float64],
    elem_thickness: List[Float64],
    frame2d_elem_indices: List[Int],
    frame3d_elem_indices: List[Int],
    truss_elem_indices: List[Int],
    zero_length_elem_indices: List[Int],
    two_node_link_elem_indices: List[Int],
    zero_length_section_elem_indices: List[Int],
    quad_elem_indices: List[Int],
    shell_elem_indices: List[Int],
    element_loads: List[ElementLoadInput],
    elem_load_offsets: List[Int],
    elem_load_pool: List[Int],
    load_scale: Float64,
    sections_by_id: List[SectionInput],
    materials_by_id: List[MaterialInput],
    id_to_index: List[Int],
    node_count: Int,
    ndf: Int,
    ndm: Int,
    u: List[Float64],
    uniaxial_defs: List[UniMaterialDef],
    uniaxial_state_defs: List[Int],
    mut uniaxial_states: List[UniMaterialState],
    elem_uniaxial_offsets: List[Int],
    elem_uniaxial_counts: List[Int],
    elem_uniaxial_state_ids: List[Int],
    force_basic_offsets: List[Int],
    force_basic_counts: List[Int],
    mut force_basic_q: List[Float64],
    fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    fiber_section3d_defs: List[FiberSection3dDef],
    fiber_section3d_cells: List[FiberCell],
    fiber_section3d_index_by_id: List[Int],
    mut force_beam_column2d_scratch: ForceBeamColumn2dScratch,
    mut force_beam_column3d_scratch: ForceBeamColumn3dScratch,
) raises -> List[Float64]:
    var total_dofs = node_count * ndf
    var K_dummy: List[List[Float64]] = []
    for _ in range(total_dofs):
        var row: List[Float64] = []
        row.resize(total_dofs, 0.0)
        K_dummy.append(row^)
    var F_int: List[Float64] = []
    F_int.resize(total_dofs, 0.0)
    var dof_map6: List[Int] = []
    var dof_map12: List[Int] = []
    var u_elem6: List[Float64] = []
    var profile_events = String()
    var profile_events_need_comma = False
    assemble_global_stiffness_and_internal_soa(
        nodes,
        elements,
        node_x,
        node_y,
        node_z,
        elem_dof_offsets,
        elem_dof_pool,
        elem_node_offsets,
        elem_node_pool,
        elem_primary_material_ids,
        elem_type_tags,
        elem_geom_tags,
        elem_section_ids,
        elem_integration_tags,
        elem_num_int_pts,
        elem_area,
        elem_thickness,
        frame2d_elem_indices,
        frame3d_elem_indices,
        truss_elem_indices,
        zero_length_elem_indices,
        two_node_link_elem_indices,
        zero_length_section_elem_indices,
        quad_elem_indices,
        shell_elem_indices,
        element_loads,
        elem_load_offsets,
        elem_load_pool,
        load_scale,
        sections_by_id,
        materials_by_id,
        id_to_index,
        node_count,
        ndf,
        ndm,
        u,
        uniaxial_defs,
        uniaxial_state_defs,
        uniaxial_states,
        elem_uniaxial_offsets,
        elem_uniaxial_counts,
        elem_uniaxial_state_ids,
        force_basic_offsets,
        force_basic_counts,
        force_basic_q,
        fiber_section_defs,
        fiber_section_cells,
        fiber_section_index_by_id,
        fiber_section3d_defs,
        fiber_section3d_cells,
        fiber_section3d_index_by_id,
        force_beam_column2d_scratch,
        force_beam_column3d_scratch,
        dof_map6,
        dof_map12,
        u_elem6,
        K_dummy,
        F_int,
        False,
        0,
        profile_events,
        profile_events_need_comma,
        0,
        0,
    )
    return F_int^


fn assemble_internal_forces_typed(
    nodes: List[NodeInput],
    elements: List[ElementInput],
    sections_by_id: List[SectionInput],
    materials_by_id: List[MaterialInput],
    id_to_index: List[Int],
    node_count: Int,
    ndf: Int,
    ndm: Int,
    u: List[Float64],
    uniaxial_defs: List[UniMaterialDef],
    uniaxial_state_defs: List[Int],
    mut uniaxial_states: List[UniMaterialState],
    elem_uniaxial_offsets: List[Int],
    elem_uniaxial_counts: List[Int],
    elem_uniaxial_state_ids: List[Int],
    force_basic_offsets: List[Int],
    force_basic_counts: List[Int],
    mut force_basic_q: List[Float64],
    fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    fiber_section3d_defs: List[FiberSection3dDef],
    fiber_section3d_cells: List[FiberCell],
    fiber_section3d_index_by_id: List[Int],
) raises -> List[Float64]:
    var empty_element_loads: List[ElementLoadInput] = []
    var empty_elem_load_offsets: List[Int] = []
    var empty_elem_load_pool: List[Int] = []
    return assemble_internal_forces_typed(
        nodes,
        elements,
        empty_element_loads,
        empty_elem_load_offsets,
        empty_elem_load_pool,
        0.0,
        sections_by_id,
        materials_by_id,
        id_to_index,
        node_count,
        ndf,
        ndm,
        u,
        uniaxial_defs,
        uniaxial_state_defs,
        uniaxial_states,
        elem_uniaxial_offsets,
        elem_uniaxial_counts,
        elem_uniaxial_state_ids,
        force_basic_offsets,
        force_basic_counts,
        force_basic_q,
        fiber_section_defs,
        fiber_section_cells,
        fiber_section_index_by_id,
        fiber_section3d_defs,
        fiber_section3d_cells,
        fiber_section3d_index_by_id,
    )


fn assemble_internal_forces_typed(
    nodes: List[NodeInput],
    elements: List[ElementInput],
    element_loads: List[ElementLoadInput],
    elem_load_offsets: List[Int],
    elem_load_pool: List[Int],
    load_scale: Float64,
    sections_by_id: List[SectionInput],
    materials_by_id: List[MaterialInput],
    id_to_index: List[Int],
    node_count: Int,
    ndf: Int,
    ndm: Int,
    u: List[Float64],
    uniaxial_defs: List[UniMaterialDef],
    uniaxial_state_defs: List[Int],
    mut uniaxial_states: List[UniMaterialState],
    elem_uniaxial_offsets: List[Int],
    elem_uniaxial_counts: List[Int],
    elem_uniaxial_state_ids: List[Int],
    force_basic_offsets: List[Int],
    force_basic_counts: List[Int],
    mut force_basic_q: List[Float64],
    fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    fiber_section3d_defs: List[FiberSection3dDef],
    fiber_section3d_cells: List[FiberCell],
    fiber_section3d_index_by_id: List[Int],
) raises -> List[Float64]:
    var total_dofs = node_count * ndf
    var K_dummy: List[List[Float64]] = []
    for _ in range(total_dofs):
        var row: List[Float64] = []
        row.resize(total_dofs, 0.0)
        K_dummy.append(row^)
    var F_int: List[Float64] = []
    F_int.resize(total_dofs, 0.0)
    assemble_global_stiffness_and_internal(
        nodes,
        elements,
        element_loads,
        elem_load_offsets,
        elem_load_pool,
        load_scale,
        sections_by_id,
        materials_by_id,
        id_to_index,
        node_count,
        ndf,
        ndm,
        u,
        uniaxial_defs,
        uniaxial_state_defs,
        uniaxial_states,
        elem_uniaxial_offsets,
        elem_uniaxial_counts,
        elem_uniaxial_state_ids,
        force_basic_offsets,
        force_basic_counts,
        force_basic_q,
        fiber_section_defs,
        fiber_section_cells,
        fiber_section_index_by_id,
        fiber_section3d_defs,
        fiber_section3d_cells,
        fiber_section3d_index_by_id,
        K_dummy,
        F_int,
    )
    return F_int^


fn assemble_global_stiffness_typed_soa(
    nodes: List[NodeInput],
    elements: List[ElementInput],
    node_x: List[Float64],
    node_y: List[Float64],
    node_z: List[Float64],
    elem_dof_offsets: List[Int],
    elem_dof_pool: List[Int],
    elem_node_offsets: List[Int],
    elem_node_pool: List[Int],
    elem_primary_material_ids: List[Int],
    elem_type_tags: List[Int],
    elem_geom_tags: List[Int],
    elem_section_ids: List[Int],
    elem_integration_tags: List[Int],
    elem_num_int_pts: List[Int],
    elem_area: List[Float64],
    elem_thickness: List[Float64],
    frame2d_elem_indices: List[Int],
    frame3d_elem_indices: List[Int],
    truss_elem_indices: List[Int],
    zero_length_elem_indices: List[Int],
    two_node_link_elem_indices: List[Int],
    zero_length_section_elem_indices: List[Int],
    quad_elem_indices: List[Int],
    shell_elem_indices: List[Int],
    element_loads: List[ElementLoadInput],
    elem_load_offsets: List[Int],
    elem_load_pool: List[Int],
    load_scale: Float64,
    sections_by_id: List[SectionInput],
    materials_by_id: List[MaterialInput],
    id_to_index: List[Int],
    node_count: Int,
    ndf: Int,
    ndm: Int,
    u: List[Float64],
    uniaxial_defs: List[UniMaterialDef],
    uniaxial_state_defs: List[Int],
    mut uniaxial_states: List[UniMaterialState],
    elem_uniaxial_offsets: List[Int],
    elem_uniaxial_counts: List[Int],
    elem_uniaxial_state_ids: List[Int],
    force_basic_offsets: List[Int],
    force_basic_counts: List[Int],
    mut force_basic_q: List[Float64],
    fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    fiber_section3d_defs: List[FiberSection3dDef],
    fiber_section3d_cells: List[FiberCell],
    fiber_section3d_index_by_id: List[Int],
) raises -> List[List[Float64]]:
    var force_beam_column2d_scratch = ForceBeamColumn2dScratch()
    var force_beam_column3d_scratch = ForceBeamColumn3dScratch()
    return assemble_global_stiffness_typed_soa(
        nodes,
        elements,
        node_x,
        node_y,
        node_z,
        elem_dof_offsets,
        elem_dof_pool,
        elem_node_offsets,
        elem_node_pool,
        elem_primary_material_ids,
        elem_type_tags,
        elem_geom_tags,
        elem_section_ids,
        elem_integration_tags,
        elem_num_int_pts,
        elem_area,
        elem_thickness,
        frame2d_elem_indices,
        frame3d_elem_indices,
        truss_elem_indices,
        zero_length_elem_indices,
        two_node_link_elem_indices,
        zero_length_section_elem_indices,
        quad_elem_indices,
        shell_elem_indices,
        element_loads,
        elem_load_offsets,
        elem_load_pool,
        load_scale,
        sections_by_id,
        materials_by_id,
        id_to_index,
        node_count,
        ndf,
        ndm,
        u,
        uniaxial_defs,
        uniaxial_state_defs,
        uniaxial_states,
        elem_uniaxial_offsets,
        elem_uniaxial_counts,
        elem_uniaxial_state_ids,
        force_basic_offsets,
        force_basic_counts,
        force_basic_q,
        fiber_section_defs,
        fiber_section_cells,
        fiber_section_index_by_id,
        fiber_section3d_defs,
        fiber_section3d_cells,
        fiber_section3d_index_by_id,
        force_beam_column2d_scratch,
        force_beam_column3d_scratch,
    )


fn assemble_global_stiffness_typed_soa(
    nodes: List[NodeInput],
    elements: List[ElementInput],
    node_x: List[Float64],
    node_y: List[Float64],
    node_z: List[Float64],
    elem_dof_offsets: List[Int],
    elem_dof_pool: List[Int],
    elem_node_offsets: List[Int],
    elem_node_pool: List[Int],
    elem_primary_material_ids: List[Int],
    elem_type_tags: List[Int],
    elem_geom_tags: List[Int],
    elem_section_ids: List[Int],
    elem_integration_tags: List[Int],
    elem_num_int_pts: List[Int],
    elem_area: List[Float64],
    elem_thickness: List[Float64],
    frame2d_elem_indices: List[Int],
    frame3d_elem_indices: List[Int],
    truss_elem_indices: List[Int],
    zero_length_elem_indices: List[Int],
    two_node_link_elem_indices: List[Int],
    zero_length_section_elem_indices: List[Int],
    quad_elem_indices: List[Int],
    shell_elem_indices: List[Int],
    element_loads: List[ElementLoadInput],
    elem_load_offsets: List[Int],
    elem_load_pool: List[Int],
    load_scale: Float64,
    sections_by_id: List[SectionInput],
    materials_by_id: List[MaterialInput],
    id_to_index: List[Int],
    node_count: Int,
    ndf: Int,
    ndm: Int,
    u: List[Float64],
    uniaxial_defs: List[UniMaterialDef],
    uniaxial_state_defs: List[Int],
    mut uniaxial_states: List[UniMaterialState],
    elem_uniaxial_offsets: List[Int],
    elem_uniaxial_counts: List[Int],
    elem_uniaxial_state_ids: List[Int],
    force_basic_offsets: List[Int],
    force_basic_counts: List[Int],
    mut force_basic_q: List[Float64],
    fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    fiber_section3d_defs: List[FiberSection3dDef],
    fiber_section3d_cells: List[FiberCell],
    fiber_section3d_index_by_id: List[Int],
    mut force_beam_column2d_scratch: ForceBeamColumn2dScratch,
    mut force_beam_column3d_scratch: ForceBeamColumn3dScratch,
) raises -> List[List[Float64]]:
    var total_dofs = node_count * ndf
    var K: List[List[Float64]] = []
    for _ in range(total_dofs):
        var row: List[Float64] = []
        row.resize(total_dofs, 0.0)
        K.append(row^)
    var F_int_dummy: List[Float64] = []
    F_int_dummy.resize(total_dofs, 0.0)
    var dof_map6: List[Int] = []
    var dof_map12: List[Int] = []
    var u_elem6: List[Float64] = []
    var profile_events = String()
    var profile_events_need_comma = False
    assemble_global_stiffness_and_internal_soa(
        nodes,
        elements,
        node_x,
        node_y,
        node_z,
        elem_dof_offsets,
        elem_dof_pool,
        elem_node_offsets,
        elem_node_pool,
        elem_primary_material_ids,
        elem_type_tags,
        elem_geom_tags,
        elem_section_ids,
        elem_integration_tags,
        elem_num_int_pts,
        elem_area,
        elem_thickness,
        frame2d_elem_indices,
        frame3d_elem_indices,
        truss_elem_indices,
        zero_length_elem_indices,
        two_node_link_elem_indices,
        zero_length_section_elem_indices,
        quad_elem_indices,
        shell_elem_indices,
        element_loads,
        elem_load_offsets,
        elem_load_pool,
        load_scale,
        sections_by_id,
        materials_by_id,
        id_to_index,
        node_count,
        ndf,
        ndm,
        u,
        uniaxial_defs,
        uniaxial_state_defs,
        uniaxial_states,
        elem_uniaxial_offsets,
        elem_uniaxial_counts,
        elem_uniaxial_state_ids,
        force_basic_offsets,
        force_basic_counts,
        force_basic_q,
        fiber_section_defs,
        fiber_section_cells,
        fiber_section_index_by_id,
        fiber_section3d_defs,
        fiber_section3d_cells,
        fiber_section3d_index_by_id,
        force_beam_column2d_scratch,
        force_beam_column3d_scratch,
        dof_map6,
        dof_map12,
        u_elem6,
        K,
        F_int_dummy,
        False,
        0,
        profile_events,
        profile_events_need_comma,
        0,
        0,
    )
    return K^


fn assemble_global_stiffness_and_internal_soa(
    nodes: List[NodeInput],
    elements: List[ElementInput],
    node_x: List[Float64],
    node_y: List[Float64],
    node_z: List[Float64],
    elem_dof_offsets: List[Int],
    elem_dof_pool: List[Int],
    elem_node_offsets: List[Int],
    elem_node_pool: List[Int],
    elem_primary_material_ids: List[Int],
    elem_type_tags: List[Int],
    elem_geom_tags: List[Int],
    elem_section_ids: List[Int],
    elem_integration_tags: List[Int],
    elem_num_int_pts: List[Int],
    elem_area: List[Float64],
    elem_thickness: List[Float64],
    frame2d_elem_indices: List[Int],
    frame3d_elem_indices: List[Int],
    truss_elem_indices: List[Int],
    zero_length_elem_indices: List[Int],
    two_node_link_elem_indices: List[Int],
    zero_length_section_elem_indices: List[Int],
    quad_elem_indices: List[Int],
    shell_elem_indices: List[Int],
    element_loads: List[ElementLoadInput],
    elem_load_offsets: List[Int],
    elem_load_pool: List[Int],
    load_scale: Float64,
    sections_by_id: List[SectionInput],
    materials_by_id: List[MaterialInput],
    id_to_index: List[Int],
    node_count: Int,
    ndf: Int,
    ndm: Int,
    u: List[Float64],
    uniaxial_defs: List[UniMaterialDef],
    uniaxial_state_defs: List[Int],
    mut uniaxial_states: List[UniMaterialState],
    elem_uniaxial_offsets: List[Int],
    elem_uniaxial_counts: List[Int],
    elem_uniaxial_state_ids: List[Int],
    force_basic_offsets: List[Int],
    force_basic_counts: List[Int],
    mut force_basic_q: List[Float64],
    fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    fiber_section3d_defs: List[FiberSection3dDef],
    fiber_section3d_cells: List[FiberCell],
    fiber_section3d_index_by_id: List[Int],
    mut dof_map6: List[Int],
    mut dof_map12: List[Int],
    mut u_elem6: List[Float64],
    mut K: List[List[Float64]],
    mut F_int: List[Float64],
    do_profile: Bool,
    t0: Int,
    mut events: String,
    mut events_need_comma: Bool,
    frame_assemble_uniaxial: Int,
    frame_assemble_fiber: Int,
) raises:
    var force_beam_column2d_scratch = ForceBeamColumn2dScratch()
    var force_beam_column3d_scratch = ForceBeamColumn3dScratch()
    assemble_global_stiffness_and_internal_soa(
        nodes,
        elements,
        node_x,
        node_y,
        node_z,
        elem_dof_offsets,
        elem_dof_pool,
        elem_node_offsets,
        elem_node_pool,
        elem_primary_material_ids,
        elem_type_tags,
        elem_geom_tags,
        elem_section_ids,
        elem_integration_tags,
        elem_num_int_pts,
        elem_area,
        elem_thickness,
        frame2d_elem_indices,
        frame3d_elem_indices,
        truss_elem_indices,
        zero_length_elem_indices,
        two_node_link_elem_indices,
        zero_length_section_elem_indices,
        quad_elem_indices,
        shell_elem_indices,
        element_loads,
        elem_load_offsets,
        elem_load_pool,
        load_scale,
        sections_by_id,
        materials_by_id,
        id_to_index,
        node_count,
        ndf,
        ndm,
        u,
        uniaxial_defs,
        uniaxial_state_defs,
        uniaxial_states,
        elem_uniaxial_offsets,
        elem_uniaxial_counts,
        elem_uniaxial_state_ids,
        force_basic_offsets,
        force_basic_counts,
        force_basic_q,
        fiber_section_defs,
        fiber_section_cells,
        fiber_section_index_by_id,
        fiber_section3d_defs,
        fiber_section3d_cells,
        fiber_section3d_index_by_id,
        force_beam_column2d_scratch,
        force_beam_column3d_scratch,
        dof_map6,
        dof_map12,
        u_elem6,
        K,
        F_int,
        do_profile,
        t0,
        events,
        events_need_comma,
        frame_assemble_uniaxial,
        frame_assemble_fiber,
    )


fn assemble_global_stiffness_and_internal_soa(
    nodes: List[NodeInput],
    elements: List[ElementInput],
    node_x: List[Float64],
    node_y: List[Float64],
    node_z: List[Float64],
    elem_dof_offsets: List[Int],
    elem_dof_pool: List[Int],
    elem_node_offsets: List[Int],
    elem_node_pool: List[Int],
    elem_primary_material_ids: List[Int],
    elem_type_tags: List[Int],
    elem_geom_tags: List[Int],
    elem_section_ids: List[Int],
    elem_integration_tags: List[Int],
    elem_num_int_pts: List[Int],
    elem_area: List[Float64],
    elem_thickness: List[Float64],
    frame2d_elem_indices: List[Int],
    frame3d_elem_indices: List[Int],
    truss_elem_indices: List[Int],
    zero_length_elem_indices: List[Int],
    two_node_link_elem_indices: List[Int],
    zero_length_section_elem_indices: List[Int],
    quad_elem_indices: List[Int],
    shell_elem_indices: List[Int],
    element_loads: List[ElementLoadInput],
    elem_load_offsets: List[Int],
    elem_load_pool: List[Int],
    load_scale: Float64,
    sections_by_id: List[SectionInput],
    materials_by_id: List[MaterialInput],
    id_to_index: List[Int],
    node_count: Int,
    ndf: Int,
    ndm: Int,
    u: List[Float64],
    uniaxial_defs: List[UniMaterialDef],
    uniaxial_state_defs: List[Int],
    mut uniaxial_states: List[UniMaterialState],
    elem_uniaxial_offsets: List[Int],
    elem_uniaxial_counts: List[Int],
    elem_uniaxial_state_ids: List[Int],
    force_basic_offsets: List[Int],
    force_basic_counts: List[Int],
    mut force_basic_q: List[Float64],
    fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    fiber_section3d_defs: List[FiberSection3dDef],
    fiber_section3d_cells: List[FiberCell],
    fiber_section3d_index_by_id: List[Int],
    mut force_beam_column2d_scratch: ForceBeamColumn2dScratch,
    mut force_beam_column3d_scratch: ForceBeamColumn3dScratch,
    mut dof_map6: List[Int],
    mut dof_map12: List[Int],
    mut u_elem6: List[Float64],
    mut K: List[List[Float64]],
    mut F_int: List[Float64],
    do_profile: Bool,
    t0: Int,
    mut events: String,
    mut events_need_comma: Bool,
    frame_assemble_uniaxial: Int,
    frame_assemble_fiber: Int,
) raises:
    _zero_matrix(K)
    _zero_vector(F_int)

    if len(dof_map6) != 6:
        dof_map6.resize(6, 0)
    if len(dof_map12) != 12:
        dof_map12.resize(12, 0)
    if len(u_elem6) != 6:
        u_elem6.resize(6, 0.0)

    _assemble_frame2d_soa_indices(
        frame2d_elem_indices,
        node_x,
        node_y,
        elem_dof_offsets,
        elem_dof_pool,
        elem_node_offsets,
        elem_node_pool,
        elem_type_tags,
        elem_geom_tags,
        elem_section_ids,
        elem_integration_tags,
        elem_num_int_pts,
        element_loads,
        elem_load_offsets,
        elem_load_pool,
        load_scale,
        sections_by_id,
        u,
        uniaxial_defs,
        uniaxial_states,
        elem_uniaxial_offsets,
        elem_uniaxial_counts,
        elem_uniaxial_state_ids,
        force_basic_offsets,
        force_basic_counts,
        force_basic_q,
        fiber_section_defs,
        fiber_section_cells,
        fiber_section_index_by_id,
        force_beam_column2d_scratch,
        dof_map6,
        u_elem6,
        K,
        F_int,
        do_profile,
        t0,
        events,
        events_need_comma,
        frame_assemble_fiber,
    )

    _assemble_frame3d_soa_indices(
        frame3d_elem_indices,
        node_x,
        node_y,
        node_z,
        elem_dof_offsets,
        elem_dof_pool,
        elem_node_offsets,
        elem_node_pool,
        elem_type_tags,
        elem_geom_tags,
        elem_section_ids,
        elem_integration_tags,
        elem_num_int_pts,
        element_loads,
        elem_load_offsets,
        elem_load_pool,
        load_scale,
        sections_by_id,
        u,
        uniaxial_defs,
        uniaxial_states,
        elem_uniaxial_offsets,
        elem_uniaxial_counts,
        elem_uniaxial_state_ids,
        force_basic_offsets,
        force_basic_counts,
        force_basic_q,
        fiber_section3d_defs,
        fiber_section3d_cells,
        fiber_section3d_index_by_id,
        force_beam_column3d_scratch,
        dof_map12,
        K,
        F_int,
        do_profile,
        t0,
        events,
        events_need_comma,
        frame_assemble_fiber,
    )

    for idx in range(len(truss_elem_indices)):
        var e = truss_elem_indices[idx]
        var node_offset = elem_node_offsets[e]
        var i1 = elem_node_pool[node_offset]
        var i2 = elem_node_pool[node_offset + 1]
        var x1 = node_x[i1]
        var y1 = node_y[i1]
        var z1 = node_z[i1]
        var x2 = node_x[i2]
        var y2 = node_y[i2]
        var z2 = node_z[i2]
        var offset = elem_uniaxial_offsets[e]
        var state_index = elem_uniaxial_state_ids[offset]
        var def_index = uniaxial_state_defs[state_index]
        var mat_def = uniaxial_defs[def_index]
        ref state = uniaxial_states[state_index]
        var A = elem_area[e]
        var dof_offset = elem_dof_offsets[e]
        _profile_scope_open(
            do_profile,
            events,
            events_need_comma,
            frame_assemble_uniaxial,
            t0,
        )
        if ndf == 2:
            var d0 = elem_dof_pool[dof_offset]
            var d1 = elem_dof_pool[dof_offset + 1]
            var d2 = elem_dof_pool[dof_offset + 2]
            var d3 = elem_dof_pool[dof_offset + 3]
            var dx = x2 - x1
            var dy = y2 - y1
            var L = hypot(dx, dy)
            if L == 0.0:
                abort("zero-length element")
            var c = dx / L
            var s = dy / L
            var du = (u[d2] - u[d0]) * c + (u[d3] - u[d1]) * s
            var eps = du / L
            _profile_scope_open(
                do_profile,
                events,
                events_need_comma,
                PROFILE_FRAME_UNIAXIAL_TRIAL_UPDATE,
                t0,
            )
            uniaxial_set_trial_strain(mat_def, state, eps)
            _profile_scope_close(
                do_profile,
                events,
                events_need_comma,
                PROFILE_FRAME_UNIAXIAL_TRIAL_UPDATE,
                t0,
            )
            var N = state.sig_t * A
            var k = state.tangent_t * A / L
            var dof_map = [d0, d1, d2, d3]
            var k_global = [
                [k * c * c, k * c * s, -k * c * c, -k * c * s],
                [k * c * s, k * s * s, -k * c * s, -k * s * s],
                [-k * c * c, -k * c * s, k * c * c, k * c * s],
                [-k * c * s, -k * s * s, k * c * s, k * s * s],
            ]
            for a in range(4):
                var Aidx = dof_map[a]
                for b in range(4):
                    var Bidx = dof_map[b]
                    K[Aidx][Bidx] += k_global[a][b]
            F_int[d0] -= N * c
            F_int[d1] -= N * s
            F_int[d2] += N * c
            F_int[d3] += N * s
        else:
            var d0 = elem_dof_pool[dof_offset]
            var d1 = elem_dof_pool[dof_offset + 1]
            var d2 = elem_dof_pool[dof_offset + 2]
            var d3 = elem_dof_pool[dof_offset + 3]
            var d4 = elem_dof_pool[dof_offset + 4]
            var d5 = elem_dof_pool[dof_offset + 5]
            var dx = x2 - x1
            var dy = y2 - y1
            var dz = z2 - z1
            var L = sqrt(dx * dx + dy * dy + dz * dz)
            if L == 0.0:
                abort("zero-length element")
            var l = dx / L
            var m = dy / L
            var n = dz / L
            var du = (u[d3] - u[d0]) * l + (u[d4] - u[d1]) * m + (u[d5] - u[d2]) * n
            var eps = du / L
            _profile_scope_open(
                do_profile,
                events,
                events_need_comma,
                PROFILE_FRAME_UNIAXIAL_TRIAL_UPDATE,
                t0,
            )
            uniaxial_set_trial_strain(mat_def, state, eps)
            _profile_scope_close(
                do_profile,
                events,
                events_need_comma,
                PROFILE_FRAME_UNIAXIAL_TRIAL_UPDATE,
                t0,
            )
            var N = state.sig_t * A
            var k = state.tangent_t * A / L
            var dof_map = [d0, d1, d2, d3, d4, d5]
            var k_global = [
                [k * l * l, k * l * m, k * l * n, -k * l * l, -k * l * m, -k * l * n],
                [k * l * m, k * m * m, k * m * n, -k * l * m, -k * m * m, -k * m * n],
                [k * l * n, k * m * n, k * n * n, -k * l * n, -k * m * n, -k * n * n],
                [-k * l * l, -k * l * m, -k * l * n, k * l * l, k * l * m, k * l * n],
                [-k * l * m, -k * m * m, -k * m * n, k * l * m, k * m * m, k * m * n],
                [-k * l * n, -k * m * n, -k * n * n, k * l * n, k * m * n, k * n * n],
            ]
            for a in range(6):
                var Aidx = dof_map[a]
                for b in range(6):
                    var Bidx = dof_map[b]
                    K[Aidx][Bidx] += k_global[a][b]
            F_int[d0] -= N * l
            F_int[d1] -= N * m
            F_int[d2] -= N * n
            F_int[d3] += N * l
            F_int[d4] += N * m
            F_int[d5] += N * n
        _profile_scope_close(
            do_profile,
            events,
            events_need_comma,
            frame_assemble_uniaxial,
            t0,
        )

    for idx in range(len(zero_length_elem_indices)):
        var e = zero_length_elem_indices[idx]
        _profile_scope_open(
            do_profile,
            events,
            events_need_comma,
            frame_assemble_uniaxial,
            t0,
        )
        _profile_scope_open(
            do_profile,
            events,
            events_need_comma,
            PROFILE_FRAME_UNIAXIAL_TRIAL_UPDATE,
            t0,
        )
        _assemble_zero_length_element(
            e,
            elements[e],
            nodes,
            ndf,
            ndm,
            u,
            uniaxial_defs,
            uniaxial_state_defs,
            uniaxial_states,
            elem_uniaxial_offsets,
            elem_uniaxial_counts,
            elem_uniaxial_state_ids,
            K,
            F_int,
        )
        _profile_scope_close(
            do_profile,
            events,
            events_need_comma,
            PROFILE_FRAME_UNIAXIAL_TRIAL_UPDATE,
            t0,
        )
        _profile_scope_close(
            do_profile,
            events,
            events_need_comma,
            frame_assemble_uniaxial,
            t0,
        )
    for idx in range(len(two_node_link_elem_indices)):
        var e = two_node_link_elem_indices[idx]
        _profile_scope_open(
            do_profile,
            events,
            events_need_comma,
            frame_assemble_uniaxial,
            t0,
        )
        _profile_scope_open(
            do_profile,
            events,
            events_need_comma,
            PROFILE_FRAME_UNIAXIAL_TRIAL_UPDATE,
            t0,
        )
        _assemble_two_node_link_element(
            e,
            elements[e],
            nodes,
            ndf,
            ndm,
            u,
            uniaxial_defs,
            uniaxial_state_defs,
            uniaxial_states,
            elem_uniaxial_offsets,
            elem_uniaxial_counts,
            elem_uniaxial_state_ids,
            K,
            F_int,
        )
        _profile_scope_close(
            do_profile,
            events,
            events_need_comma,
            PROFILE_FRAME_UNIAXIAL_TRIAL_UPDATE,
            t0,
        )
        _profile_scope_close(
            do_profile,
            events,
            events_need_comma,
            frame_assemble_uniaxial,
            t0,
        )

    for idx in range(len(zero_length_section_elem_indices)):
        var e = zero_length_section_elem_indices[idx]
        _profile_scope_open(
            do_profile,
            events,
            events_need_comma,
            PROFILE_FRAME_ASSEMBLE_FIBER_GEOMETRY,
            t0,
        )
        var dof_offset = elem_dof_offsets[e]
        var u1 = elem_dof_pool[dof_offset]
        var r1 = elem_dof_pool[dof_offset + 2]
        var u2 = elem_dof_pool[dof_offset + 3]
        var r2 = elem_dof_pool[dof_offset + 5]
        var delta_axial = u[u2] - u[u1]
        var delta_curv = u[r2] - u[r1]
        var sec = sections_by_id[elem_section_ids[e]]
        var elem_offset = elem_uniaxial_offsets[e]
        var elem_state_count = elem_uniaxial_counts[e]
        var axial_force = 0.0
        var moment_z = 0.0
        var k11 = 0.0
        var k12 = 0.0
        var k22 = 0.0
        _profile_scope_close(
            do_profile,
            events,
            events_need_comma,
            PROFILE_FRAME_ASSEMBLE_FIBER_GEOMETRY,
            t0,
        )
        if sec.type == "ElasticSection2d":
            k11 = sec.E * sec.A
            k22 = sec.E * sec.I
            axial_force = k11 * delta_axial
            moment_z = k22 * delta_curv
        elif sec.type == "AggregatorSection2d":
            _profile_scope_open(
                do_profile,
                events,
                events_need_comma,
                PROFILE_FRAME_UNIAXIAL_TRIAL_UPDATE,
                t0,
            )
            (axial_force, moment_z, k11, k12, k22) = aggregator_section2d_set_trial_from_offset(
                sec,
                uniaxial_defs,
                uniaxial_state_defs,
                uniaxial_states,
                elem_uniaxial_state_ids,
                elem_offset,
                elem_state_count,
                delta_axial,
                delta_curv,
            )
            _profile_scope_close(
                do_profile,
                events,
                events_need_comma,
                PROFILE_FRAME_UNIAXIAL_TRIAL_UPDATE,
                t0,
            )
        elif sec.type == "FiberSection2d":
            var sec_index = fiber_section_index_by_id[elem_section_ids[e]]
            if sec_index < 0 or sec_index >= len(fiber_section_defs):
                abort("zeroLengthSection fiber section not found")
            var sec_def = fiber_section_defs[sec_index]
            var elem_offset = elem_uniaxial_offsets[e]
            var elem_state_count = elem_uniaxial_counts[e]
            if elem_state_count != sec_def.fiber_count:
                abort("zeroLengthSection fiber state count mismatch")
            _profile_scope_open(
                do_profile,
                events,
                events_need_comma,
                frame_assemble_fiber,
                t0,
            )
            _profile_scope_open(
                do_profile,
                events,
                events_need_comma,
                PROFILE_FRAME_ASSEMBLE_FIBER_SECTION_RESPONSE,
                t0,
            )
            var resp = fiber_section2d_set_trial_from_offset(
                sec_def,
                fiber_section_cells,
                uniaxial_defs,
                uniaxial_states,
                elem_offset,
                elem_state_count,
                delta_axial,
                delta_curv,
            )
            _profile_scope_close(
                do_profile,
                events,
                events_need_comma,
                PROFILE_FRAME_ASSEMBLE_FIBER_SECTION_RESPONSE,
                t0,
            )
            _profile_scope_close(
                do_profile,
                events,
                events_need_comma,
                frame_assemble_fiber,
                t0,
            )
            axial_force = resp.axial_force
            moment_z = resp.moment_z
            k11 = resp.k11
            k12 = resp.k12
            k22 = resp.k22
        else:
            abort(
                "zeroLengthSection requires FiberSection2d, ElasticSection2d, or AggregatorSection2d"
            )
        _profile_scope_open(
            do_profile,
            events,
            events_need_comma,
            PROFILE_FRAME_ASSEMBLE_FIBER_MATRIX_SCATTER,
            t0,
        )
        K[u1][u1] += k11
        K[u1][r1] += k12
        K[u1][u2] -= k11
        K[u1][r2] -= k12
        K[r1][u1] += k12
        K[r1][r1] += k22
        K[r1][u2] -= k12
        K[r1][r2] -= k22
        K[u2][u1] -= k11
        K[u2][r1] -= k12
        K[u2][u2] += k11
        K[u2][r2] += k12
        K[r2][u1] -= k12
        K[r2][r1] -= k22
        K[r2][u2] += k12
        K[r2][r2] += k22
        _profile_scope_close(
            do_profile,
            events,
            events_need_comma,
            PROFILE_FRAME_ASSEMBLE_FIBER_MATRIX_SCATTER,
            t0,
        )
        _profile_scope_open(
            do_profile,
            events,
            events_need_comma,
            PROFILE_FRAME_ASSEMBLE_FIBER_INTERNAL_FORCE,
            t0,
        )
        F_int[u1] -= axial_force
        F_int[r1] -= moment_z
        F_int[u2] += axial_force
        F_int[r2] += moment_z
        _profile_scope_close(
            do_profile,
            events,
            events_need_comma,
            PROFILE_FRAME_ASSEMBLE_FIBER_INTERNAL_FORCE,
            t0,
        )

    _assemble_surface_soa_indices(
        quad_elem_indices,
        shell_elem_indices,
        node_x,
        node_y,
        node_z,
        elem_dof_offsets,
        elem_dof_pool,
        elem_node_offsets,
        elem_node_pool,
        elem_primary_material_ids,
        elem_section_ids,
        elem_thickness,
        materials_by_id,
        sections_by_id,
        u,
        K,
        F_int,
    )


fn assemble_global_stiffness_and_internal(
    nodes: List[NodeInput],
    elements: List[ElementInput],
    sections_by_id: List[SectionInput],
    materials_by_id: List[MaterialInput],
    id_to_index: List[Int],
    node_count: Int,
    ndf: Int,
    ndm: Int,
    u: List[Float64],
    uniaxial_defs: List[UniMaterialDef],
    uniaxial_state_defs: List[Int],
    mut uniaxial_states: List[UniMaterialState],
    elem_uniaxial_offsets: List[Int],
    elem_uniaxial_counts: List[Int],
    elem_uniaxial_state_ids: List[Int],
    force_basic_offsets: List[Int],
    force_basic_counts: List[Int],
    mut force_basic_q: List[Float64],
    fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    fiber_section3d_defs: List[FiberSection3dDef],
    fiber_section3d_cells: List[FiberCell],
    fiber_section3d_index_by_id: List[Int],
    mut K: List[List[Float64]],
    mut F_int: List[Float64],
) raises:
    var empty_element_loads: List[ElementLoadInput] = []
    var empty_elem_load_offsets: List[Int] = []
    var empty_elem_load_pool: List[Int] = []
    assemble_global_stiffness_and_internal(
        nodes,
        elements,
        empty_element_loads,
        empty_elem_load_offsets,
        empty_elem_load_pool,
        0.0,
        sections_by_id,
        materials_by_id,
        id_to_index,
        node_count,
        ndf,
        ndm,
        u,
        uniaxial_defs,
        uniaxial_state_defs,
        uniaxial_states,
        elem_uniaxial_offsets,
        elem_uniaxial_counts,
        elem_uniaxial_state_ids,
        force_basic_offsets,
        force_basic_counts,
        force_basic_q,
        fiber_section_defs,
        fiber_section_cells,
        fiber_section_index_by_id,
        fiber_section3d_defs,
        fiber_section3d_cells,
        fiber_section3d_index_by_id,
        K,
        F_int,
    )


fn assemble_global_stiffness_and_internal(
    nodes: List[NodeInput],
    elements: List[ElementInput],
    element_loads: List[ElementLoadInput],
    elem_load_offsets: List[Int],
    elem_load_pool: List[Int],
    load_scale: Float64,
    sections_by_id: List[SectionInput],
    materials_by_id: List[MaterialInput],
    id_to_index: List[Int],
    node_count: Int,
    ndf: Int,
    ndm: Int,
    u: List[Float64],
    uniaxial_defs: List[UniMaterialDef],
    uniaxial_state_defs: List[Int],
    mut uniaxial_states: List[UniMaterialState],
    elem_uniaxial_offsets: List[Int],
    elem_uniaxial_counts: List[Int],
    elem_uniaxial_state_ids: List[Int],
    force_basic_offsets: List[Int],
    force_basic_counts: List[Int],
    mut force_basic_q: List[Float64],
    fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    fiber_section3d_defs: List[FiberSection3dDef],
    fiber_section3d_cells: List[FiberCell],
    fiber_section3d_index_by_id: List[Int],
    mut K: List[List[Float64]],
    mut F_int: List[Float64],
) raises:
    var elem_dof_offsets: List[Int] = []
    var elem_dof_pool: List[Int] = []
    _build_elem_dof_soa(elements, elem_dof_offsets, elem_dof_pool)
    var dof_map6: List[Int] = []
    var dof_map12: List[Int] = []
    var u_elem6: List[Float64] = []
    assemble_global_stiffness_and_internal(
        nodes,
        elements,
        element_loads,
        elem_load_offsets,
        elem_load_pool,
        load_scale,
        sections_by_id,
        materials_by_id,
        id_to_index,
        node_count,
        ndf,
        ndm,
        u,
        uniaxial_defs,
        uniaxial_state_defs,
        uniaxial_states,
        elem_uniaxial_offsets,
        elem_uniaxial_counts,
        elem_uniaxial_state_ids,
        force_basic_offsets,
        force_basic_counts,
        force_basic_q,
        fiber_section_defs,
        fiber_section_cells,
        fiber_section_index_by_id,
        fiber_section3d_defs,
        fiber_section3d_cells,
        fiber_section3d_index_by_id,
        elem_dof_offsets,
        elem_dof_pool,
        dof_map6,
        dof_map12,
        u_elem6,
        K,
        F_int,
    )


fn assemble_global_stiffness_and_internal(
    nodes: List[NodeInput],
    elements: List[ElementInput],
    sections_by_id: List[SectionInput],
    materials_by_id: List[MaterialInput],
    id_to_index: List[Int],
    node_count: Int,
    ndf: Int,
    ndm: Int,
    u: List[Float64],
    uniaxial_defs: List[UniMaterialDef],
    uniaxial_state_defs: List[Int],
    mut uniaxial_states: List[UniMaterialState],
    elem_uniaxial_offsets: List[Int],
    elem_uniaxial_counts: List[Int],
    elem_uniaxial_state_ids: List[Int],
    force_basic_offsets: List[Int],
    force_basic_counts: List[Int],
    mut force_basic_q: List[Float64],
    fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    fiber_section3d_defs: List[FiberSection3dDef],
    fiber_section3d_cells: List[FiberCell],
    fiber_section3d_index_by_id: List[Int],
    elem_dof_offsets: List[Int],
    elem_dof_pool: List[Int],
    mut dof_map6: List[Int],
    mut dof_map12: List[Int],
    mut u_elem6: List[Float64],
    mut K: List[List[Float64]],
    mut F_int: List[Float64],
) raises:
    var empty_element_loads: List[ElementLoadInput] = []
    var empty_elem_load_offsets: List[Int] = []
    var empty_elem_load_pool: List[Int] = []
    assemble_global_stiffness_and_internal(
        nodes,
        elements,
        empty_element_loads,
        empty_elem_load_offsets,
        empty_elem_load_pool,
        0.0,
        sections_by_id,
        materials_by_id,
        id_to_index,
        node_count,
        ndf,
        ndm,
        u,
        uniaxial_defs,
        uniaxial_state_defs,
        uniaxial_states,
        elem_uniaxial_offsets,
        elem_uniaxial_counts,
        elem_uniaxial_state_ids,
        force_basic_offsets,
        force_basic_counts,
        force_basic_q,
        fiber_section_defs,
        fiber_section_cells,
        fiber_section_index_by_id,
        fiber_section3d_defs,
        fiber_section3d_cells,
        fiber_section3d_index_by_id,
        elem_dof_offsets,
        elem_dof_pool,
        dof_map6,
        dof_map12,
        u_elem6,
        K,
        F_int,
    )


fn assemble_global_stiffness_and_internal(
    nodes: List[NodeInput],
    elements: List[ElementInput],
    element_loads: List[ElementLoadInput],
    elem_load_offsets: List[Int],
    elem_load_pool: List[Int],
    load_scale: Float64,
    sections_by_id: List[SectionInput],
    materials_by_id: List[MaterialInput],
    id_to_index: List[Int],
    node_count: Int,
    ndf: Int,
    ndm: Int,
    u: List[Float64],
    uniaxial_defs: List[UniMaterialDef],
    uniaxial_state_defs: List[Int],
    mut uniaxial_states: List[UniMaterialState],
    elem_uniaxial_offsets: List[Int],
    elem_uniaxial_counts: List[Int],
    elem_uniaxial_state_ids: List[Int],
    force_basic_offsets: List[Int],
    force_basic_counts: List[Int],
    mut force_basic_q: List[Float64],
    fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    fiber_section3d_defs: List[FiberSection3dDef],
    fiber_section3d_cells: List[FiberCell],
    fiber_section3d_index_by_id: List[Int],
    elem_dof_offsets: List[Int],
    elem_dof_pool: List[Int],
    mut dof_map6: List[Int],
    mut dof_map12: List[Int],
    mut u_elem6: List[Float64],
    mut K: List[List[Float64]],
    mut F_int: List[Float64],
) raises:
    _assemble_global_stiffness_and_internal_filtered(
        nodes,
        elements,
        element_loads,
        elem_load_offsets,
        elem_load_pool,
        load_scale,
        sections_by_id,
        materials_by_id,
        id_to_index,
        node_count,
        ndf,
        ndm,
        u,
        uniaxial_defs,
        uniaxial_state_defs,
        uniaxial_states,
        elem_uniaxial_offsets,
        elem_uniaxial_counts,
        elem_uniaxial_state_ids,
        force_basic_offsets,
        force_basic_counts,
        force_basic_q,
        fiber_section_defs,
        fiber_section_cells,
        fiber_section_index_by_id,
        fiber_section3d_defs,
        fiber_section3d_cells,
        fiber_section3d_index_by_id,
        elem_dof_offsets,
        elem_dof_pool,
        dof_map6,
        dof_map12,
        u_elem6,
        0,
        K,
        F_int,
    )


fn _assemble_global_stiffness_and_internal_filtered(
    nodes: List[NodeInput],
    elements: List[ElementInput],
    element_loads: List[ElementLoadInput],
    elem_load_offsets: List[Int],
    elem_load_pool: List[Int],
    load_scale: Float64,
    sections_by_id: List[SectionInput],
    materials_by_id: List[MaterialInput],
    id_to_index: List[Int],
    node_count: Int,
    ndf: Int,
    ndm: Int,
    u: List[Float64],
    uniaxial_defs: List[UniMaterialDef],
    uniaxial_state_defs: List[Int],
    mut uniaxial_states: List[UniMaterialState],
    elem_uniaxial_offsets: List[Int],
    elem_uniaxial_counts: List[Int],
    elem_uniaxial_state_ids: List[Int],
    force_basic_offsets: List[Int],
    force_basic_counts: List[Int],
    mut force_basic_q: List[Float64],
    fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    fiber_section3d_defs: List[FiberSection3dDef],
    fiber_section3d_cells: List[FiberCell],
    fiber_section3d_index_by_id: List[Int],
    elem_dof_offsets: List[Int],
    elem_dof_pool: List[Int],
    mut dof_map6: List[Int],
    mut dof_map12: List[Int],
    mut u_elem6: List[Float64],
    filter_mode: Int,
    mut K: List[List[Float64]],
    mut F_int: List[Float64],
) raises:
    _zero_matrix(K)
    _zero_vector(F_int)

    var elem_count = len(elements)
    if len(elem_dof_offsets) != elem_count + 1:
        abort("invalid elem_dof_offsets size for global assembly")
    if len(elem_dof_pool) != elem_dof_offsets[elem_count]:
        abort("invalid elem_dof_pool size for global assembly")
    if len(dof_map6) != 6:
        dof_map6.resize(6, 0)
    if len(dof_map12) != 12:
        dof_map12.resize(12, 0)
    if len(u_elem6) != 6:
        u_elem6.resize(6, 0.0)
    var force_beam_column2d_scratch = ForceBeamColumn2dScratch()
    var force_beam_column3d_scratch = ForceBeamColumn3dScratch()

    for e in range(elem_count):
        var elem = elements[e]
        if not _assembly_filter_accepts_element(elem, filter_mode):
            continue
        var elem_type = elem.type_tag
        if (
            elem_type == ElementTypeTag.ElasticBeamColumn2d
            or elem_type == ElementTypeTag.ForceBeamColumn2d
            or elem_type == ElementTypeTag.DispBeamColumn2d
        ):
            _assemble_frame2d_element(
                e,
                elem,
                nodes,
                elem_dof_offsets,
                elem_dof_pool,
                element_loads,
                elem_load_offsets,
                elem_load_pool,
                load_scale,
                sections_by_id,
                u,
                uniaxial_defs,
                uniaxial_states,
                elem_uniaxial_offsets,
                elem_uniaxial_counts,
                elem_uniaxial_state_ids,
                force_basic_offsets,
                force_basic_counts,
                force_basic_q,
                fiber_section_defs,
                fiber_section_cells,
                fiber_section_index_by_id,
                force_beam_column2d_scratch,
                dof_map6,
                u_elem6,
                K,
                F_int,
            )
        elif (
            elem_type == ElementTypeTag.ElasticBeamColumn3d
            or elem_type == ElementTypeTag.ForceBeamColumn3d
            or elem_type == ElementTypeTag.DispBeamColumn3d
        ):
            _assemble_frame3d_element(
                e,
                elem,
                nodes,
                elem_dof_offsets,
                elem_dof_pool,
                element_loads,
                elem_load_offsets,
                elem_load_pool,
                load_scale,
                sections_by_id,
                u,
                uniaxial_defs,
                uniaxial_states,
                elem_uniaxial_offsets,
                elem_uniaxial_counts,
                elem_uniaxial_state_ids,
                force_basic_offsets,
                force_basic_counts,
                force_basic_q,
                fiber_section3d_defs,
                fiber_section3d_cells,
                fiber_section3d_index_by_id,
                force_beam_column3d_scratch,
                dof_map12,
                K,
                F_int,
            )
        elif elem_type == ElementTypeTag.Truss:
            var i1 = elem.node_index_1
            var i2 = elem.node_index_2
            var node1 = nodes[i1]
            var node2 = nodes[i2]

            var offset = elem_uniaxial_offsets[e]
            var state_index = elem_uniaxial_state_ids[offset]
            var def_index = uniaxial_state_defs[state_index]
            var mat_def = uniaxial_defs[def_index]
            ref state = uniaxial_states[state_index]
            var A = elem.area

            if ndf == 2:
                var dx = node2.x - node1.x
                var dy = node2.y - node1.y
                var L = hypot(dx, dy)
                if L == 0.0:
                    abort("zero-length element")
                var c = dx / L
                var s = dy / L
                var dof_map = [
                    _elem_dof(elem, 0),
                    _elem_dof(elem, 1),
                    _elem_dof(elem, 2),
                    _elem_dof(elem, 3),
                ]
                var du = (u[dof_map[2]] - u[dof_map[0]]) * c + (
                    u[dof_map[3]] - u[dof_map[1]]
                ) * s
                var eps = du / L
                uniaxial_set_trial_strain(mat_def, state, eps)
                var N = state.sig_t * A
                var k = state.tangent_t * A / L
                var k_global = [
                    [k * c * c, k * c * s, -k * c * c, -k * c * s],
                    [k * c * s, k * s * s, -k * c * s, -k * s * s],
                    [-k * c * c, -k * c * s, k * c * c, k * c * s],
                    [-k * c * s, -k * s * s, k * c * s, k * s * s],
                ]
                for a in range(4):
                    var Aidx = dof_map[a]
                    for b in range(4):
                        var Bidx = dof_map[b]
                        K[Aidx][Bidx] += k_global[a][b]
                F_int[dof_map[0]] -= N * c
                F_int[dof_map[1]] -= N * s
                F_int[dof_map[2]] += N * c
                F_int[dof_map[3]] += N * s
            else:
                var dx = node2.x - node1.x
                var dy = node2.y - node1.y
                var dz = node2.z - node1.z
                var L = sqrt(dx * dx + dy * dy + dz * dz)
                if L == 0.0:
                    abort("zero-length element")
                var l = dx / L
                var m = dy / L
                var n = dz / L
                var dof_map = [
                    _elem_dof(elem, 0),
                    _elem_dof(elem, 1),
                    _elem_dof(elem, 2),
                    _elem_dof(elem, 3),
                    _elem_dof(elem, 4),
                    _elem_dof(elem, 5),
                ]
                var du = (u[dof_map[3]] - u[dof_map[0]]) * l + (
                    u[dof_map[4]] - u[dof_map[1]]
                ) * m + (u[dof_map[5]] - u[dof_map[2]]) * n
                var eps = du / L
                uniaxial_set_trial_strain(mat_def, state, eps)
                var N = state.sig_t * A
                var k = state.tangent_t * A / L
                var k_global = [
                    [k * l * l, k * l * m, k * l * n, -k * l * l, -k * l * m, -k * l * n],
                    [k * l * m, k * m * m, k * m * n, -k * l * m, -k * m * m, -k * m * n],
                    [k * l * n, k * m * n, k * n * n, -k * l * n, -k * m * n, -k * n * n],
                    [-k * l * l, -k * l * m, -k * l * n, k * l * l, k * l * m, k * l * n],
                    [-k * l * m, -k * m * m, -k * m * n, k * l * m, k * m * m, k * m * n],
                    [-k * l * n, -k * m * n, -k * n * n, k * l * n, k * m * n, k * n * n],
                ]
                for a in range(6):
                    var Aidx = dof_map[a]
                    for b in range(6):
                        var Bidx = dof_map[b]
                        K[Aidx][Bidx] += k_global[a][b]
                F_int[dof_map[0]] -= N * l
                F_int[dof_map[1]] -= N * m
                F_int[dof_map[2]] -= N * n
                F_int[dof_map[3]] += N * l
                F_int[dof_map[4]] += N * m
                F_int[dof_map[5]] += N * n
        elif elem_type == ElementTypeTag.ZeroLength:
            _assemble_zero_length_element(
                e,
                elem,
                nodes,
                ndf,
                ndm,
                u,
                uniaxial_defs,
                uniaxial_state_defs,
                uniaxial_states,
                elem_uniaxial_offsets,
                elem_uniaxial_counts,
                elem_uniaxial_state_ids,
                K,
                F_int,
            )
        elif elem_type == ElementTypeTag.TwoNodeLink:
            _assemble_two_node_link_element(
                e,
                elem,
                nodes,
                ndf,
                ndm,
                u,
                uniaxial_defs,
                uniaxial_state_defs,
                uniaxial_states,
                elem_uniaxial_offsets,
                elem_uniaxial_counts,
                elem_uniaxial_state_ids,
                K,
                F_int,
            )
        elif elem_type == ElementTypeTag.ZeroLengthSection:
            var dof_map = [
                _elem_dof(elem, 0),
                _elem_dof(elem, 1),
                _elem_dof(elem, 2),
                _elem_dof(elem, 3),
                _elem_dof(elem, 4),
                _elem_dof(elem, 5),
            ]
            var delta_axial = u[dof_map[3]] - u[dof_map[0]]
            var delta_curv = u[dof_map[5]] - u[dof_map[2]]

            var sec = sections_by_id[elem.section]
            var elem_offset = elem_uniaxial_offsets[e]
            var elem_state_count = elem_uniaxial_counts[e]
            var axial_force = 0.0
            var moment_z = 0.0
            var k11 = 0.0
            var k12 = 0.0
            var k22 = 0.0
            if sec.type == "ElasticSection2d":
                k11 = sec.E * sec.A
                k22 = sec.E * sec.I
                axial_force = k11 * delta_axial
                moment_z = k22 * delta_curv
            elif sec.type == "AggregatorSection2d":
                (axial_force, moment_z, k11, k12, k22) = aggregator_section2d_set_trial_from_offset(
                    sec,
                    uniaxial_defs,
                    uniaxial_state_defs,
                    uniaxial_states,
                    elem_uniaxial_state_ids,
                    elem_offset,
                    elem_state_count,
                    delta_axial,
                    delta_curv,
                )
            elif sec.type == "FiberSection2d":
                var sec_index = fiber_section_index_by_id[elem.section]
                if sec_index < 0 or sec_index >= len(fiber_section_defs):
                    abort("zeroLengthSection fiber section not found")
                var sec_def = fiber_section_defs[sec_index]
                var elem_offset = elem_uniaxial_offsets[e]
                var elem_state_count = elem_uniaxial_counts[e]
                if elem_state_count != sec_def.fiber_count:
                    abort("zeroLengthSection fiber state count mismatch")
                if elem_offset < 0 or elem_offset + elem_state_count > len(
                    elem_uniaxial_state_ids
                ):
                    abort("zeroLengthSection fiber state range out of bounds")
                var resp = fiber_section2d_set_trial_from_offset(
                    sec_def,
                    fiber_section_cells,
                    uniaxial_defs,
                    uniaxial_states,
                    elem_offset,
                    elem_state_count,
                    delta_axial,
                    delta_curv,
                )
                axial_force = resp.axial_force
                moment_z = resp.moment_z
                k11 = resp.k11
                k12 = resp.k12
                k22 = resp.k22
            else:
                abort(
                    "zeroLengthSection requires FiberSection2d, ElasticSection2d, or AggregatorSection2d"
                )

            var u1 = dof_map[0]
            var r1 = dof_map[2]
            var u2 = dof_map[3]
            var r2 = dof_map[5]

            K[u1][u1] += k11
            K[u1][r1] += k12
            K[u1][u2] -= k11
            K[u1][r2] -= k12

            K[r1][u1] += k12
            K[r1][r1] += k22
            K[r1][u2] -= k12
            K[r1][r2] -= k22

            K[u2][u1] -= k11
            K[u2][r1] -= k12
            K[u2][u2] += k11
            K[u2][r2] += k12

            K[r2][u1] -= k12
            K[r2][r1] -= k22
            K[r2][u2] += k12
            K[r2][r2] += k22

            F_int[u1] -= axial_force
            F_int[r1] -= moment_z
            F_int[u2] += axial_force
            F_int[r2] += moment_z
        elif (
            elem_type == ElementTypeTag.FourNodeQuad
            or elem_type == ElementTypeTag.Shell
        ):
            _assemble_surface_element(
                elem, nodes, materials_by_id, sections_by_id, u, K, F_int
            )
        else:
            abort("unsupported element type tag")
