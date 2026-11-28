from collections import List
from math import hypot, sqrt
from os import abort
from time import perf_counter_ns

from elements import (
    ForceBeamColumn2dScratch,
    truss_global_stiffness,
    truss3d_global_stiffness,
)
from elements.corot_truss import CorotTrussChord, TrussVector
from materials import (
    UniMaterialDef,
    UniMaterialState,
    uni_mat_initial_tangent,
    uniaxial_set_trial_strain,
    uniaxial_set_trial_strain_rate,
)
from solver.assembly.frame3d_scratch import (
    Frame3dAssemblyScratch,
)
from solver.assembly.surface_cache import SurfaceAssemblyCache
from solver.assembly.stiffness_internal_frame2d import (
    _assemble_frame2d_element,
    _assemble_frame2d_native_soa_indices,
    _assemble_frame2d_soa_indices,
)
from solver.assembly.stiffness_internal_frame3d import (
    _assemble_frame3d_element,
    _assemble_frame3d_native_soa_indices,
    _assemble_frame3d_soa_indices,
)
from solver.assembly.stiffness_internal_links import (
    _assemble_two_node_link_element_native,
    _assemble_two_node_link_element,
    _assemble_zero_length_element_native,
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
    _assemble_surface_native_soa_indices,
    _assemble_surface_soa_indices,
)
from solver.dof import node_dof_index
from solver.profile import (
    PROFILE_FRAME_ASSEMBLE_FIBER_GEOMETRY,
    PROFILE_FRAME_ASSEMBLE_FIBER_INTERNAL_FORCE,
    PROFILE_FRAME_ASSEMBLE_FIBER_MATRIX_SCATTER,
    PROFILE_FRAME_ASSEMBLE_FIBER_SECTION_RESPONSE,
    PROFILE_FRAME_UNIAXIAL_TRIAL_UPDATE,
    RuntimeProfileMetrics,
    _profile_metrics_note_element_timing,
)
from solver.run_case.helpers import (
    aggregator_section2d_set_trial_from_offset,
    aggregator_section3d_set_trial_from_offset,
)
from solver.sparse_pattern import ElementMatrixSink, NullMatrixSink
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
    LayeredShellSectionDef,
    fiber_section2d_set_trial_from_offset,
    fiber_section3d_set_trial_from_offset,
)
from tag_types import BeamIntegrationTag, ElementTypeTag, GeomTransfTag


fn _eval_zero_length_section3d(
    sec: SectionInput,
    sec_id: Int,
    uniaxial_defs: List[UniMaterialDef],
    uniaxial_state_defs: List[Int],
    mut uniaxial_states: List[UniMaterialState],
    elem_uniaxial_state_ids: List[Int],
    elem_offset: Int,
    elem_state_count: Int,
    fiber_section3d_index_by_id: List[Int],
    mut fiber_section3d_defs: List[FiberSection3dDef],
    delta_p: Float64,
    delta_vy: Float64,
    delta_vz: Float64,
    delta_t: Float64,
    delta_my: Float64,
    delta_mz: Float64,
    mut f_sec: List[Float64],
    mut k_sec: List[Float64],
) raises:
    for i in range(6):
        f_sec[i] = 0.0
    for i in range(36):
        k_sec[i] = 0.0
    if sec.type == "ElasticSection3d":
        var k_p = sec.E * sec.A
        var k_t = sec.G * sec.J
        var k_my = sec.E * sec.Iy
        var k_mz = sec.E * sec.Iz
        f_sec[0] = k_p * delta_p
        f_sec[3] = k_t * delta_t
        f_sec[4] = k_my * delta_my
        f_sec[5] = k_mz * delta_mz
        k_sec[0 * 6 + 0] = k_p
        k_sec[3 * 6 + 3] = k_t
        k_sec[4 * 6 + 4] = k_my
        k_sec[5 * 6 + 5] = k_mz
    elif sec.type == "AggregatorSection3d":
        var resp = aggregator_section3d_set_trial_from_offset(
            sec,
            uniaxial_defs,
            uniaxial_state_defs,
            uniaxial_states,
            elem_uniaxial_state_ids,
            elem_offset,
            elem_state_count,
            delta_p,
            delta_vy,
            delta_vz,
            delta_t,
            delta_my,
            delta_mz,
        )
        f_sec[0] = resp[0]
        f_sec[1] = resp[1]
        f_sec[2] = resp[2]
        f_sec[3] = resp[3]
        f_sec[4] = resp[4]
        f_sec[5] = resp[5]
        k_sec[0 * 6 + 0] = resp[6]
        k_sec[1 * 6 + 1] = resp[7]
        k_sec[2 * 6 + 2] = resp[8]
        k_sec[3 * 6 + 3] = resp[9]
        k_sec[4 * 6 + 4] = resp[10]
        k_sec[5 * 6 + 5] = resp[11]
    elif sec.type == "FiberSection3d":
        var sec_index = fiber_section3d_index_by_id[sec_id]
        if sec_index < 0 or sec_index >= len(fiber_section3d_defs):
            abort("zeroLengthSection 3D fiber section not found")
        ref sec_def = fiber_section3d_defs[sec_index]
        if elem_state_count != sec_def.fiber_count:
            abort("zeroLengthSection 3D fiber state count mismatch")
        var resp = fiber_section3d_set_trial_from_offset(
            sec_def,
            uniaxial_states,
            elem_offset,
            elem_state_count,
            delta_p,
            delta_my,
            delta_mz,
        )
        f_sec[0] = resp.axial_force
        f_sec[4] = resp.moment_y
        f_sec[5] = resp.moment_z
        k_sec[0 * 6 + 0] = resp.k11
        k_sec[0 * 6 + 4] = resp.k12
        k_sec[0 * 6 + 5] = resp.k13
        k_sec[4 * 6 + 0] = resp.k12
        k_sec[4 * 6 + 4] = resp.k22
        k_sec[4 * 6 + 5] = resp.k23
        k_sec[5 * 6 + 0] = resp.k13
        k_sec[5 * 6 + 4] = resp.k23
        k_sec[5 * 6 + 5] = resp.k33
        if sec.G > 0.0 and sec.J > 0.0:
            var k_t = sec.G * sec.J
            f_sec[3] = k_t * delta_t
            k_sec[3 * 6 + 3] = k_t
    else:
        abort(
            "zeroLengthSection (3D) requires FiberSection3d, ElasticSection3d, or"
            " AggregatorSection3d"
        )


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
    mut fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    mut fiber_section3d_defs: List[FiberSection3dDef],
    fiber_section3d_cells: List[FiberCell],
    fiber_section3d_index_by_id: List[Int],
    layered_shell_section_defs: List[LayeredShellSectionDef],
    mut shell_cache: SurfaceAssemblyCache,
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
        layered_shell_section_defs,
        shell_cache,
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
    mut fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    mut fiber_section3d_defs: List[FiberSection3dDef],
    fiber_section3d_cells: List[FiberCell],
    fiber_section3d_index_by_id: List[Int],
    layered_shell_section_defs: List[LayeredShellSectionDef],
    mut shell_cache: SurfaceAssemblyCache,
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
        layered_shell_section_defs,
        shell_cache,
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
    mut fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    mut fiber_section3d_defs: List[FiberSection3dDef],
    fiber_section3d_cells: List[FiberCell],
    fiber_section3d_index_by_id: List[Int],
    layered_shell_section_defs: List[LayeredShellSectionDef],
    mut shell_cache: SurfaceAssemblyCache,
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
        layered_shell_section_defs,
        shell_cache,
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
    mut fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    mut fiber_section3d_defs: List[FiberSection3dDef],
    fiber_section3d_cells: List[FiberCell],
    fiber_section3d_index_by_id: List[Int],
    layered_shell_section_defs: List[LayeredShellSectionDef],
    mut shell_cache: SurfaceAssemblyCache,
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
        layered_shell_section_defs,
        shell_cache,
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
    mut fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    mut fiber_section3d_defs: List[FiberSection3dDef],
    fiber_section3d_cells: List[FiberCell],
    fiber_section3d_index_by_id: List[Int],
    layered_shell_section_defs: List[LayeredShellSectionDef],
    mut shell_cache: SurfaceAssemblyCache,
) raises -> List[Float64]:
    var force_beam_column2d_scratch = ForceBeamColumn2dScratch()
    var force_beam_column3d_scratch = Frame3dAssemblyScratch()
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
        layered_shell_section_defs,
        shell_cache,
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
    mut fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    mut fiber_section3d_defs: List[FiberSection3dDef],
    fiber_section3d_cells: List[FiberCell],
    fiber_section3d_index_by_id: List[Int],
    layered_shell_section_defs: List[LayeredShellSectionDef],
    mut shell_cache: SurfaceAssemblyCache,
    mut force_beam_column2d_scratch: ForceBeamColumn2dScratch,
    mut force_beam_column3d_scratch: Frame3dAssemblyScratch,
    v: List[Float64] = List[Float64](),
    dt: Float64 = 0.0,
) raises -> List[Float64]:
    var total_dofs = node_count * ndf
    # No tangent is needed here, so element matrices go to a sink that drops
    # them. The sink ignores the free-equation map; the DOF map stands in for it.
    var matrix_sink = NullMatrixSink()
    var F_int: List[Float64] = []
    F_int.resize(total_dofs, 0.0)
    var dof_map6: List[Int] = []
    var dof_map12: List[Int] = []
    var u_elem6: List[Float64] = []
    var profile_events = String()
    var profile_events_need_comma = False
    var runtime_metrics = RuntimeProfileMetrics()
    assemble_global_stiffness_and_internal_native_soa(
        nodes,
        elements,
        node_x,
        node_y,
        node_z,
        elem_dof_offsets,
        elem_dof_pool,
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
        layered_shell_section_defs,
        shell_cache,
        force_beam_column2d_scratch,
        force_beam_column3d_scratch,
        dof_map6,
        dof_map12,
        u_elem6,
        matrix_sink,
        F_int,
        False,
        0,
        profile_events,
        profile_events_need_comma,
        0,
        0,
        runtime_metrics,
        v,
        dt,
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
    mut fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    mut fiber_section3d_defs: List[FiberSection3dDef],
    fiber_section3d_cells: List[FiberCell],
    fiber_section3d_index_by_id: List[Int],
    layered_shell_section_defs: List[LayeredShellSectionDef],
    mut shell_cache: SurfaceAssemblyCache,
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
        layered_shell_section_defs,
        shell_cache,
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
    mut fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    mut fiber_section3d_defs: List[FiberSection3dDef],
    fiber_section3d_cells: List[FiberCell],
    fiber_section3d_index_by_id: List[Int],
    layered_shell_section_defs: List[LayeredShellSectionDef],
    mut shell_cache: SurfaceAssemblyCache,
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
        layered_shell_section_defs,
        shell_cache,
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
    mut fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    mut fiber_section3d_defs: List[FiberSection3dDef],
    fiber_section3d_cells: List[FiberCell],
    fiber_section3d_index_by_id: List[Int],
    layered_shell_section_defs: List[LayeredShellSectionDef],
    mut shell_cache: SurfaceAssemblyCache,
) raises -> List[List[Float64]]:
    var force_beam_column2d_scratch = ForceBeamColumn2dScratch()
    var force_beam_column3d_scratch = Frame3dAssemblyScratch()
    var runtime_metrics = RuntimeProfileMetrics()
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
        layered_shell_section_defs,
        shell_cache,
        force_beam_column2d_scratch,
        force_beam_column3d_scratch,
        runtime_metrics,
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
    mut fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    mut fiber_section3d_defs: List[FiberSection3dDef],
    fiber_section3d_cells: List[FiberCell],
    fiber_section3d_index_by_id: List[Int],
    layered_shell_section_defs: List[LayeredShellSectionDef],
    mut shell_cache: SurfaceAssemblyCache,
    mut force_beam_column2d_scratch: ForceBeamColumn2dScratch,
    mut force_beam_column3d_scratch: Frame3dAssemblyScratch,
    mut runtime_metrics: RuntimeProfileMetrics,
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
        layered_shell_section_defs,
        shell_cache,
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
        runtime_metrics,
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
    mut fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    mut fiber_section3d_defs: List[FiberSection3dDef],
    fiber_section3d_cells: List[FiberCell],
    fiber_section3d_index_by_id: List[Int],
    layered_shell_section_defs: List[LayeredShellSectionDef],
    mut shell_cache: SurfaceAssemblyCache,
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
    var force_beam_column3d_scratch = Frame3dAssemblyScratch()
    var runtime_metrics = RuntimeProfileMetrics()
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
        layered_shell_section_defs,
        shell_cache,
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
        runtime_metrics,
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
    mut fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    mut fiber_section3d_defs: List[FiberSection3dDef],
    fiber_section3d_cells: List[FiberCell],
    fiber_section3d_index_by_id: List[Int],
    layered_shell_section_defs: List[LayeredShellSectionDef],
    mut shell_cache: SurfaceAssemblyCache,
    mut force_beam_column2d_scratch: ForceBeamColumn2dScratch,
    mut force_beam_column3d_scratch: Frame3dAssemblyScratch,
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
    mut runtime_metrics: RuntimeProfileMetrics,
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
        runtime_metrics,
    )

    _assemble_frame3d_soa_indices(
        frame3d_elem_indices,
        elements,
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
        runtime_metrics,
    )

    for idx in range(len(truss_elem_indices)):
        var e = truss_elem_indices[idx]
        var t_elem_start = 0
        if runtime_metrics.enabled:
            t_elem_start = Int(perf_counter_ns())
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
        if elements[e].type == "corotTruss":
            var dof_count = elements[e].dof_count
            var k_flat: List[Float64] = []
            k_flat.resize(dof_count * dof_count, 0.0)
            _assemble_corot_truss(
                ndm,
                dof_count,
                dof_offset,
                elem_dof_pool,
                TrussVector(x2 - x1, y2 - y1, z2 - z1, 0.0),
                u,
                List[Float64](),
                0.0,
                A,
                mat_def,
                state,
                k_flat,
                F_int,
            )
            for i in range(dof_count):
                var row = elem_dof_pool[dof_offset + i]
                for j in range(dof_count):
                    var col = elem_dof_pool[dof_offset + j]
                    K[row][col] += k_flat[i * dof_count + j]
        elif ndf == 2:
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
        if runtime_metrics.enabled:
            _profile_metrics_note_element_timing(
                runtime_metrics,
                ElementTypeTag.Truss,
                Int(perf_counter_ns()) - t_elem_start,
            )

    for idx in range(len(zero_length_elem_indices)):
        var e = zero_length_elem_indices[idx]
        var t_elem_start = 0
        if runtime_metrics.enabled:
            t_elem_start = Int(perf_counter_ns())
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
        if runtime_metrics.enabled:
            _profile_metrics_note_element_timing(
                runtime_metrics,
                ElementTypeTag.ZeroLength,
                Int(perf_counter_ns()) - t_elem_start,
            )
    for idx in range(len(two_node_link_elem_indices)):
        var e = two_node_link_elem_indices[idx]
        var t_elem_start = 0
        if runtime_metrics.enabled:
            t_elem_start = Int(perf_counter_ns())
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
            element_loads=element_loads,
            elem_load_offsets=elem_load_offsets,
            elem_load_pool=elem_load_pool,
            load_scale=load_scale,
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
        if runtime_metrics.enabled:
            _profile_metrics_note_element_timing(
                runtime_metrics,
                ElementTypeTag.TwoNodeLink,
                Int(perf_counter_ns()) - t_elem_start,
            )

    var zls_f_sec6: List[Float64] = []
    var zls_k_sec36: List[Float64] = []
    if len(zero_length_section_elem_indices) > 0:
        zls_f_sec6.resize(6, 0.0)
        zls_k_sec36.resize(36, 0.0)
    for idx in range(len(zero_length_section_elem_indices)):
        var e = zero_length_section_elem_indices[idx]
        var t_elem_start = 0
        if runtime_metrics.enabled:
            t_elem_start = Int(perf_counter_ns())
        _profile_scope_open(
            do_profile,
            events,
            events_need_comma,
            PROFILE_FRAME_ASSEMBLE_FIBER_GEOMETRY,
            t0,
        )
        var dof_offset = elem_dof_offsets[e]
        var elem_ndof = elem_dof_offsets[e + 1] - dof_offset
        var sec_id = elem_section_ids[e]
        var sec = sections_by_id[sec_id]
        var elem_offset = elem_uniaxial_offsets[e]
        var elem_state_count = elem_uniaxial_counts[e]
        if elem_ndof == 12:
            var d0 = elem_dof_pool[dof_offset]
            var d1 = elem_dof_pool[dof_offset + 1]
            var d2 = elem_dof_pool[dof_offset + 2]
            var d3 = elem_dof_pool[dof_offset + 3]
            var d4 = elem_dof_pool[dof_offset + 4]
            var d5 = elem_dof_pool[dof_offset + 5]
            var d6 = elem_dof_pool[dof_offset + 6]
            var d7 = elem_dof_pool[dof_offset + 7]
            var d8 = elem_dof_pool[dof_offset + 8]
            var d9 = elem_dof_pool[dof_offset + 9]
            var d10 = elem_dof_pool[dof_offset + 10]
            var d11 = elem_dof_pool[dof_offset + 11]
            _profile_scope_close(
                do_profile,
                events,
                events_need_comma,
                PROFILE_FRAME_ASSEMBLE_FIBER_GEOMETRY,
                t0,
            )
            _eval_zero_length_section3d(
                sec,
                sec_id,
                uniaxial_defs,
                uniaxial_state_defs,
                uniaxial_states,
                elem_uniaxial_state_ids,
                elem_offset,
                elem_state_count,
                fiber_section3d_index_by_id,
                fiber_section3d_defs,
                u[d6] - u[d0],
                u[d7] - u[d1],
                u[d8] - u[d2],
                u[d9] - u[d3],
                u[d10] - u[d4],
                u[d11] - u[d5],
                zls_f_sec6,
                zls_k_sec36,
            )
            for a in range(6):
                var da1 = elem_dof_pool[dof_offset + a]
                var da2 = elem_dof_pool[dof_offset + 6 + a]
                for b in range(6):
                    var kab = zls_k_sec36[a * 6 + b]
                    if kab != 0.0:
                        var db1 = elem_dof_pool[dof_offset + b]
                        var db2 = elem_dof_pool[dof_offset + 6 + b]
                        K[da1][db1] += kab
                        K[da1][db2] -= kab
                        K[da2][db1] -= kab
                        K[da2][db2] += kab
                var fa = zls_f_sec6[a]
                F_int[da1] -= fa
                F_int[da2] += fa
            if runtime_metrics.enabled:
                _profile_metrics_note_element_timing(
                    runtime_metrics,
                    ElementTypeTag.ZeroLengthSection,
                    Int(perf_counter_ns()) - t_elem_start,
                )
            continue

        var u1 = elem_dof_pool[dof_offset]
        var v1 = elem_dof_pool[dof_offset + 1]
        var r1 = elem_dof_pool[dof_offset + 2]
        var u2 = elem_dof_pool[dof_offset + 3]
        var v2 = elem_dof_pool[dof_offset + 4]
        var r2 = elem_dof_pool[dof_offset + 5]
        var delta_axial = u[u2] - u[u1]
        var delta_shear_y = u[v2] - u[v1]
        var delta_curv = u[r2] - u[r1]
        var axial_force = 0.0
        var shear_y = 0.0
        var moment_z = 0.0
        var k11 = 0.0
        var k12 = 0.0
        var k22 = 0.0
        var k_vy = 0.0
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
            (
                axial_force,
                shear_y,
                moment_z,
                k11,
                k_vy,
                k22,
            ) = aggregator_section2d_set_trial_from_offset(
                sec,
                uniaxial_defs,
                uniaxial_state_defs,
                uniaxial_states,
                elem_uniaxial_state_ids,
                elem_offset,
                elem_state_count,
                delta_axial,
                delta_shear_y,
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
            var sec_index = fiber_section_index_by_id[sec_id]
            if sec_index < 0 or sec_index >= len(fiber_section_defs):
                abort("zeroLengthSection fiber section not found")
            ref sec_def = fiber_section_defs[sec_index]
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
                "zeroLengthSection requires FiberSection2d, ElasticSection2d, or"
                " AggregatorSection2d"
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
        if k_vy != 0.0:
            K[v1][v1] += k_vy
            K[v1][v2] -= k_vy
            K[v2][v1] -= k_vy
            K[v2][v2] += k_vy
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
        F_int[v1] -= shear_y
        F_int[r1] -= moment_z
        F_int[u2] += axial_force
        F_int[v2] += shear_y
        F_int[r2] += moment_z
        _profile_scope_close(
            do_profile,
            events,
            events_need_comma,
            PROFILE_FRAME_ASSEMBLE_FIBER_INTERNAL_FORCE,
            t0,
        )
        if runtime_metrics.enabled:
            _profile_metrics_note_element_timing(
                runtime_metrics,
                ElementTypeTag.ZeroLengthSection,
                Int(perf_counter_ns()) - t_elem_start,
            )

    _assemble_surface_soa_indices(
        elements,
        quad_elem_indices,
        shell_elem_indices,
        node_x,
        node_y,
        elem_dof_offsets,
        elem_dof_pool,
        elem_node_offsets,
        elem_node_pool,
        elem_primary_material_ids,
        shell_cache,
        elem_thickness,
        materials_by_id,
        layered_shell_section_defs,
        uniaxial_defs,
        uniaxial_states,
        u,
        K,
        F_int,
        runtime_metrics,
    )


@always_inline
fn _truss_flat_stiffness[
    dim: Int
](k: Float64, cosines: SIMD[DType.float64, 4], mut k_flat: List[Float64]):
    """Truss global stiffness into `k_flat` (row-major, 2*dim square).

    Entry (a, b) is `+-k * u[min] * u[max]` with `u` the direction cosines of
    the local DOFs, evaluated in the same order as the list-literal form.
    """
    alias count = 2 * dim
    for a in range(count):
        var ia = a % dim
        for b in range(count):
            var ib = b % dim
            var lo = ia
            var hi = ib
            if lo > hi:
                lo = ib
                hi = ia
            var value = k * cosines[lo] * cosines[hi]
            if (a < dim) != (b < dim):
                value = -value
            k_flat[a * count + b] = value


fn _assemble_corot_truss(
    ndm: Int,
    dof_count: Int,
    dof_offset: Int,
    elem_dof_pool: List[Int],
    reference: TrussVector,
    u: List[Float64],
    v: List[Float64],
    dt: Float64,
    area: Float64,
    mat_def: UniMaterialDef,
    mut state: UniMaterialState,
    mut k_flat: List[Float64],
    mut F_int: List[Float64],
):
    var stride = dof_count // 2
    var relative_disp = TrussVector(0.0)
    var relative_vel = TrussVector(0.0)
    for i in range(ndm):
        var node_i_dof = elem_dof_pool[dof_offset + i]
        var node_j_dof = elem_dof_pool[dof_offset + stride + i]
        relative_disp[i] = u[node_j_dof] - u[node_i_dof]
        if len(v) == len(u):
            relative_vel[i] = v[node_j_dof] - v[node_i_dof]
    var chord = CorotTrussChord(reference, relative_disp)
    uniaxial_set_trial_strain_rate(
        mat_def, state, chord.strain(), chord.strain_rate(relative_vel), dt
    )
    var force = area * state.sig_t
    var axial = area * state.tangent_t / chord.initial_length
    var geometric = force / chord.length
    for row in range(dof_count):
        for col in range(dof_count):
            k_flat[row * dof_count + col] = 0.0
    for i in range(ndm):
        var fi = force * chord.direction[i]
        F_int[elem_dof_pool[dof_offset + i]] -= fi
        F_int[elem_dof_pool[dof_offset + stride + i]] += fi
        for j in range(ndm):
            var kij = (axial - geometric) * chord.direction[i] * chord.direction[j]
            if i == j:
                kij += geometric
            k_flat[i * dof_count + j] = kij
            k_flat[i * dof_count + stride + j] = -kij
            k_flat[(i + stride) * dof_count + j] = -kij
            k_flat[(i + stride) * dof_count + stride + j] = kij


fn assemble_global_stiffness_and_internal_native_soa[
    S: ElementMatrixSink
](
    nodes: List[NodeInput],
    elements: List[ElementInput],
    node_x: List[Float64],
    node_y: List[Float64],
    node_z: List[Float64],
    elem_dof_offsets: List[Int],
    elem_dof_pool: List[Int],
    elem_free_offsets: List[Int],
    elem_free_pool: List[Int],
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
    mut fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    mut fiber_section3d_defs: List[FiberSection3dDef],
    fiber_section3d_cells: List[FiberCell],
    fiber_section3d_index_by_id: List[Int],
    layered_shell_section_defs: List[LayeredShellSectionDef],
    mut shell_cache: SurfaceAssemblyCache,
    mut force_beam_column2d_scratch: ForceBeamColumn2dScratch,
    mut force_beam_column3d_scratch: Frame3dAssemblyScratch,
    mut dof_map6: List[Int],
    mut dof_map12: List[Int],
    mut u_elem6: List[Float64],
    mut backend: S,
    mut F_int: List[Float64],
    do_profile: Bool,
    t0: Int,
    mut events: String,
    mut events_need_comma: Bool,
    frame_assemble_uniaxial: Int,
    frame_assemble_fiber: Int,
    mut runtime_metrics: RuntimeProfileMetrics,
    v: List[Float64] = List[Float64](),
    dt: Float64 = 0.0,
    initial_wall_tangent: Bool = False,
    initial_frame_tangent: Bool = False,
) raises:
    _zero_vector(F_int)
    backend.zero_matrix()

    var elem_count = len(elements)
    if len(elem_dof_offsets) != elem_count + 1:
        abort("invalid elem_dof_offsets size for native global assembly")
    if len(elem_free_offsets) != elem_count + 1:
        abort("invalid elem_free_offsets size for native global assembly")
    if len(elem_dof_pool) != elem_dof_offsets[elem_count]:
        abort("invalid elem_dof_pool size for native global assembly")
    if len(elem_free_pool) != elem_free_offsets[elem_count]:
        abort("invalid elem_free_pool size for native global assembly")
    if len(dof_map6) != 6:
        dof_map6.resize(6, 0)
    if len(dof_map12) != 12:
        dof_map12.resize(12, 0)
    if len(u_elem6) != 6:
        u_elem6.resize(6, 0.0)
    # Flat element-matrix scratch reused by the truss and zeroLengthSection loops.
    var k_flat: List[Float64] = []
    k_flat.resize(144, 0.0)

    _assemble_frame2d_native_soa_indices(
        frame2d_elem_indices,
        node_x,
        node_y,
        elem_dof_offsets,
        elem_dof_pool,
        elem_free_offsets,
        elem_free_pool,
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
        force_beam_column3d_scratch.frame2d,
        dof_map6,
        u_elem6,
        k_flat,
        backend,
        F_int,
        do_profile,
        t0,
        events,
        events_need_comma,
        frame_assemble_fiber,
        runtime_metrics,
        initial_frame_tangent,
    )

    _assemble_frame3d_native_soa_indices(
        frame3d_elem_indices,
        elements,
        node_x,
        node_y,
        node_z,
        elem_dof_offsets,
        elem_dof_pool,
        elem_free_offsets,
        elem_free_pool,
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
        backend,
        F_int,
        do_profile,
        t0,
        events,
        events_need_comma,
        frame_assemble_fiber,
        runtime_metrics,
    )

    for idx in range(len(truss_elem_indices)):
        var e = truss_elem_indices[idx]
        var t_elem_start = 0
        if runtime_metrics.enabled:
            t_elem_start = Int(perf_counter_ns())
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
        var free_offset = elem_free_offsets[e]
        _profile_scope_open(
            do_profile,
            events,
            events_need_comma,
            frame_assemble_uniaxial,
            t0,
        )
        if elements[e].type == "corotTruss":
            var dof_count = elements[e].dof_count
            _assemble_corot_truss(
                ndm,
                dof_count,
                dof_offset,
                elem_dof_pool,
                TrussVector(x2 - x1, y2 - y1, z2 - z1, 0.0),
                u,
                v,
                dt,
                A,
                mat_def,
                state,
                k_flat,
                F_int,
            )
            backend.add_element_matrix_flat(e, elem_free_pool, free_offset, dof_count, k_flat)
        elif ndf == 2:
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
            _truss_flat_stiffness[2](k, SIMD[DType.float64, 4](c, s, 0.0, 0.0), k_flat)
            backend.add_element_matrix_flat(e, elem_free_pool, free_offset, 4, k_flat)
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
            _truss_flat_stiffness[3](k, SIMD[DType.float64, 4](l, m, n, 0.0), k_flat)
            backend.add_element_matrix_flat(e, elem_free_pool, free_offset, 6, k_flat)
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
        if runtime_metrics.enabled:
            _profile_metrics_note_element_timing(
                runtime_metrics,
                ElementTypeTag.Truss,
                Int(perf_counter_ns()) - t_elem_start,
            )

    for idx in range(len(zero_length_elem_indices)):
        var e = zero_length_elem_indices[idx]
        var t_elem_start = 0
        if runtime_metrics.enabled:
            t_elem_start = Int(perf_counter_ns())
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
        _assemble_zero_length_element_native(
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
            elem_free_offsets,
            elem_free_pool,
            backend,
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
        if runtime_metrics.enabled:
            _profile_metrics_note_element_timing(
                runtime_metrics,
                ElementTypeTag.ZeroLength,
                Int(perf_counter_ns()) - t_elem_start,
            )

    for idx in range(len(two_node_link_elem_indices)):
        var e = two_node_link_elem_indices[idx]
        var t_elem_start = 0
        if runtime_metrics.enabled:
            t_elem_start = Int(perf_counter_ns())
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
        _assemble_two_node_link_element_native(
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
            elem_free_offsets,
            elem_free_pool,
            backend,
            F_int,
            v,
            dt,
            initial_wall_tangent,
            element_loads,
            elem_load_offsets,
            elem_load_pool,
            load_scale,
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
        if runtime_metrics.enabled:
            _profile_metrics_note_element_timing(
                runtime_metrics,
                ElementTypeTag.TwoNodeLink,
                Int(perf_counter_ns()) - t_elem_start,
            )

    var zls_f_sec6: List[Float64] = []
    var zls_k_sec36: List[Float64] = []
    if len(zero_length_section_elem_indices) > 0:
        zls_f_sec6.resize(6, 0.0)
        zls_k_sec36.resize(36, 0.0)
    for idx in range(len(zero_length_section_elem_indices)):
        var e = zero_length_section_elem_indices[idx]
        var t_elem_start = 0
        if runtime_metrics.enabled:
            t_elem_start = Int(perf_counter_ns())
        _profile_scope_open(
            do_profile,
            events,
            events_need_comma,
            PROFILE_FRAME_ASSEMBLE_FIBER_GEOMETRY,
            t0,
        )
        var dof_offset = elem_dof_offsets[e]
        var elem_ndof = elem_dof_offsets[e + 1] - dof_offset
        var free_offset = elem_free_offsets[e]
        var sec_id = elem_section_ids[e]
        var sec = sections_by_id[sec_id]
        var elem_offset = elem_uniaxial_offsets[e]
        var elem_state_count = elem_uniaxial_counts[e]
        if elem_ndof == 12:
            var d0 = elem_dof_pool[dof_offset]
            var d1 = elem_dof_pool[dof_offset + 1]
            var d2 = elem_dof_pool[dof_offset + 2]
            var d3 = elem_dof_pool[dof_offset + 3]
            var d4 = elem_dof_pool[dof_offset + 4]
            var d5 = elem_dof_pool[dof_offset + 5]
            var d6 = elem_dof_pool[dof_offset + 6]
            var d7 = elem_dof_pool[dof_offset + 7]
            var d8 = elem_dof_pool[dof_offset + 8]
            var d9 = elem_dof_pool[dof_offset + 9]
            var d10 = elem_dof_pool[dof_offset + 10]
            var d11 = elem_dof_pool[dof_offset + 11]
            _profile_scope_close(
                do_profile,
                events,
                events_need_comma,
                PROFILE_FRAME_ASSEMBLE_FIBER_GEOMETRY,
                t0,
            )
            _eval_zero_length_section3d(
                sec,
                sec_id,
                uniaxial_defs,
                uniaxial_state_defs,
                uniaxial_states,
                elem_uniaxial_state_ids,
                elem_offset,
                elem_state_count,
                fiber_section3d_index_by_id,
                fiber_section3d_defs,
                u[d6] - u[d0],
                u[d7] - u[d1],
                u[d8] - u[d2],
                u[d9] - u[d3],
                u[d10] - u[d4],
                u[d11] - u[d5],
                zls_f_sec6,
                zls_k_sec36,
            )
            for a in range(6):
                for b in range(6):
                    var kab = zls_k_sec36[a * 6 + b]
                    k_flat[a * 12 + b] = kab
                    k_flat[a * 12 + (b + 6)] = -kab
                    k_flat[(a + 6) * 12 + b] = -kab
                    k_flat[(a + 6) * 12 + (b + 6)] = kab
            backend.add_element_matrix_flat(e, elem_free_pool, free_offset, 12, k_flat)
            for a in range(6):
                var da1 = elem_dof_pool[dof_offset + a]
                var da2 = elem_dof_pool[dof_offset + 6 + a]
                var fa = zls_f_sec6[a]
                F_int[da1] -= fa
                F_int[da2] += fa
            if runtime_metrics.enabled:
                _profile_metrics_note_element_timing(
                    runtime_metrics,
                    ElementTypeTag.ZeroLengthSection,
                    Int(perf_counter_ns()) - t_elem_start,
                )
            continue

        var u1 = elem_dof_pool[dof_offset]
        var v1 = elem_dof_pool[dof_offset + 1]
        var r1 = elem_dof_pool[dof_offset + 2]
        var u2 = elem_dof_pool[dof_offset + 3]
        var v2 = elem_dof_pool[dof_offset + 4]
        var r2 = elem_dof_pool[dof_offset + 5]
        var delta_axial = u[u2] - u[u1]
        var delta_shear_y = u[v2] - u[v1]
        var delta_curv = u[r2] - u[r1]
        var axial_force = 0.0
        var shear_y = 0.0
        var moment_z = 0.0
        var k11 = 0.0
        var k12 = 0.0
        var k22 = 0.0
        var k_vy = 0.0
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
            (
                axial_force,
                shear_y,
                moment_z,
                k11,
                k_vy,
                k22,
            ) = aggregator_section2d_set_trial_from_offset(
                sec,
                uniaxial_defs,
                uniaxial_state_defs,
                uniaxial_states,
                elem_uniaxial_state_ids,
                elem_offset,
                elem_state_count,
                delta_axial,
                delta_shear_y,
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
            var sec_index = fiber_section_index_by_id[sec_id]
            if sec_index < 0 or sec_index >= len(fiber_section_defs):
                abort("zeroLengthSection fiber section not found")
            ref sec_def = fiber_section_defs[sec_index]
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
                "zeroLengthSection requires FiberSection2d, ElasticSection2d, or"
                " AggregatorSection2d"
            )
        var zls_row0 = SIMD[DType.float64, 8](k11, 0.0, k12, -k11, 0.0, -k12, 0.0, 0.0)
        var zls_row1 = SIMD[DType.float64, 8](0.0, k_vy, 0.0, 0.0, -k_vy, 0.0, 0.0, 0.0)
        var zls_row2 = SIMD[DType.float64, 8](k12, 0.0, k22, -k12, 0.0, -k22, 0.0, 0.0)
        for b in range(6):
            k_flat[0 * 6 + b] = zls_row0[b]
            k_flat[1 * 6 + b] = zls_row1[b]
            k_flat[2 * 6 + b] = zls_row2[b]
            k_flat[3 * 6 + b] = -zls_row0[b]
            k_flat[4 * 6 + b] = -zls_row1[b]
            k_flat[5 * 6 + b] = -zls_row2[b]
        _profile_scope_open(
            do_profile,
            events,
            events_need_comma,
            PROFILE_FRAME_ASSEMBLE_FIBER_MATRIX_SCATTER,
            t0,
        )
        backend.add_element_matrix_flat(e, elem_free_pool, free_offset, 6, k_flat)
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
        F_int[v1] -= shear_y
        F_int[r1] -= moment_z
        F_int[u2] += axial_force
        F_int[v2] += shear_y
        F_int[r2] += moment_z
        _profile_scope_close(
            do_profile,
            events,
            events_need_comma,
            PROFILE_FRAME_ASSEMBLE_FIBER_INTERNAL_FORCE,
            t0,
        )
        if runtime_metrics.enabled:
            _profile_metrics_note_element_timing(
                runtime_metrics,
                ElementTypeTag.ZeroLengthSection,
                Int(perf_counter_ns()) - t_elem_start,
            )

    _assemble_surface_native_soa_indices(
        elements,
        quad_elem_indices,
        shell_elem_indices,
        node_x,
        node_y,
        elem_dof_offsets,
        elem_dof_pool,
        elem_free_offsets,
        elem_free_pool,
        elem_node_offsets,
        elem_node_pool,
        elem_primary_material_ids,
        shell_cache,
        elem_thickness,
        materials_by_id,
        layered_shell_section_defs,
        uniaxial_defs,
        uniaxial_states,
        u,
        backend,
        F_int,
        runtime_metrics,
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
    mut fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    mut fiber_section3d_defs: List[FiberSection3dDef],
    fiber_section3d_cells: List[FiberCell],
    fiber_section3d_index_by_id: List[Int],
    layered_shell_section_defs: List[LayeredShellSectionDef],
    mut shell_cache: SurfaceAssemblyCache,
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
        layered_shell_section_defs,
        shell_cache,
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
    mut fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    mut fiber_section3d_defs: List[FiberSection3dDef],
    fiber_section3d_cells: List[FiberCell],
    fiber_section3d_index_by_id: List[Int],
    layered_shell_section_defs: List[LayeredShellSectionDef],
    mut shell_cache: SurfaceAssemblyCache,
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
        layered_shell_section_defs,
        shell_cache,
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
    mut fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    mut fiber_section3d_defs: List[FiberSection3dDef],
    fiber_section3d_cells: List[FiberCell],
    fiber_section3d_index_by_id: List[Int],
    layered_shell_section_defs: List[LayeredShellSectionDef],
    mut shell_cache: SurfaceAssemblyCache,
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
        layered_shell_section_defs,
        shell_cache,
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
    mut fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    mut fiber_section3d_defs: List[FiberSection3dDef],
    fiber_section3d_cells: List[FiberCell],
    fiber_section3d_index_by_id: List[Int],
    layered_shell_section_defs: List[LayeredShellSectionDef],
    mut shell_cache: SurfaceAssemblyCache,
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
        layered_shell_section_defs,
        shell_cache,
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
    mut fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    mut fiber_section3d_defs: List[FiberSection3dDef],
    fiber_section3d_cells: List[FiberCell],
    fiber_section3d_index_by_id: List[Int],
    layered_shell_section_defs: List[LayeredShellSectionDef],
    mut shell_cache: SurfaceAssemblyCache,
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
    var force_beam_column3d_scratch = Frame3dAssemblyScratch()

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
                var du = (u[dof_map[2]] - u[dof_map[0]]) * c + (u[dof_map[3]] - u[dof_map[1]]) * s
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
                var du = (
                    (u[dof_map[3]] - u[dof_map[0]]) * l
                    + (u[dof_map[4]] - u[dof_map[1]]) * m
                    + (u[dof_map[5]] - u[dof_map[2]]) * n
                )
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
        elif (
            elem_type == ElementTypeTag.TwoNodeLink
            or elem_type == ElementTypeTag.MVLEM
            or elem_type == ElementTypeTag.SFIMVLEM
            or elem_type == ElementTypeTag.CatenaryCable
        ):
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
                element_loads=element_loads,
                elem_load_offsets=elem_load_offsets,
                elem_load_pool=elem_load_pool,
                load_scale=load_scale,
            )
        elif elem_type == ElementTypeTag.ZeroLengthSection:
            var sec = sections_by_id[elem.section]
            var elem_offset = elem_uniaxial_offsets[e]
            var elem_state_count = elem_uniaxial_counts[e]
            if elem.dof_count == 12:
                var d = [
                    _elem_dof(elem, 0),
                    _elem_dof(elem, 1),
                    _elem_dof(elem, 2),
                    _elem_dof(elem, 3),
                    _elem_dof(elem, 4),
                    _elem_dof(elem, 5),
                    _elem_dof(elem, 6),
                    _elem_dof(elem, 7),
                    _elem_dof(elem, 8),
                    _elem_dof(elem, 9),
                    _elem_dof(elem, 10),
                    _elem_dof(elem, 11),
                ]
                var f_sec6: List[Float64] = []
                var k_sec36: List[Float64] = []
                f_sec6.resize(6, 0.0)
                k_sec36.resize(36, 0.0)
                _eval_zero_length_section3d(
                    sec,
                    elem.section,
                    uniaxial_defs,
                    uniaxial_state_defs,
                    uniaxial_states,
                    elem_uniaxial_state_ids,
                    elem_offset,
                    elem_state_count,
                    fiber_section3d_index_by_id,
                    fiber_section3d_defs,
                    u[d[6]] - u[d[0]],
                    u[d[7]] - u[d[1]],
                    u[d[8]] - u[d[2]],
                    u[d[9]] - u[d[3]],
                    u[d[10]] - u[d[4]],
                    u[d[11]] - u[d[5]],
                    f_sec6,
                    k_sec36,
                )
                for a in range(6):
                    var da1 = d[a]
                    var da2 = d[6 + a]
                    for b in range(6):
                        var kab = k_sec36[a * 6 + b]
                        if kab != 0.0:
                            var db1 = d[b]
                            var db2 = d[6 + b]
                            K[da1][db1] += kab
                            K[da1][db2] -= kab
                            K[da2][db1] -= kab
                            K[da2][db2] += kab
                    var fa = f_sec6[a]
                    F_int[da1] -= fa
                    F_int[da2] += fa
                continue

            var dof_map = [
                _elem_dof(elem, 0),
                _elem_dof(elem, 1),
                _elem_dof(elem, 2),
                _elem_dof(elem, 3),
                _elem_dof(elem, 4),
                _elem_dof(elem, 5),
            ]
            var delta_axial = u[dof_map[3]] - u[dof_map[0]]
            var delta_shear_y = u[dof_map[4]] - u[dof_map[1]]
            var delta_curv = u[dof_map[5]] - u[dof_map[2]]

            var axial_force = 0.0
            var shear_y = 0.0
            var moment_z = 0.0
            var k11 = 0.0
            var k12 = 0.0
            var k22 = 0.0
            var k_vy = 0.0
            if sec.type == "ElasticSection2d":
                k11 = sec.E * sec.A
                k22 = sec.E * sec.I
                axial_force = k11 * delta_axial
                moment_z = k22 * delta_curv
            elif sec.type == "AggregatorSection2d":
                (
                    axial_force,
                    shear_y,
                    moment_z,
                    k11,
                    k_vy,
                    k22,
                ) = aggregator_section2d_set_trial_from_offset(
                    sec,
                    uniaxial_defs,
                    uniaxial_state_defs,
                    uniaxial_states,
                    elem_uniaxial_state_ids,
                    elem_offset,
                    elem_state_count,
                    delta_axial,
                    delta_shear_y,
                    delta_curv,
                )
            elif sec.type == "FiberSection2d":
                var sec_index = fiber_section_index_by_id[elem.section]
                if sec_index < 0 or sec_index >= len(fiber_section_defs):
                    abort("zeroLengthSection fiber section not found")
                ref sec_def = fiber_section_defs[sec_index]
                if elem_state_count != sec_def.fiber_count:
                    abort("zeroLengthSection fiber state count mismatch")
                if elem_offset < 0 or elem_offset + elem_state_count > len(elem_uniaxial_state_ids):
                    abort("zeroLengthSection fiber state range out of bounds")
                var resp = fiber_section2d_set_trial_from_offset(
                    sec_def,
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
                    "zeroLengthSection requires FiberSection2d, ElasticSection2d, or"
                    " AggregatorSection2d"
                )

            var u1 = dof_map[0]
            var v1 = dof_map[1]
            var r1 = dof_map[2]
            var u2 = dof_map[3]
            var v2 = dof_map[4]
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

            if k_vy != 0.0:
                K[v1][v1] += k_vy
                K[v1][v2] -= k_vy
                K[v2][v1] -= k_vy
                K[v2][v2] += k_vy

            F_int[u1] -= axial_force
            F_int[v1] -= shear_y
            F_int[r1] -= moment_z
            F_int[u2] += axial_force
            F_int[v2] += shear_y
            F_int[r2] += moment_z
        elif elem_type == ElementTypeTag.Continuum:
            shell_cache.continuum.assemble_dense_element(e, u, K, F_int)
        elif (
            elem_type == ElementTypeTag.FourNodeQuad
            or elem_type == ElementTypeTag.BeamContact2D
            or elem_type == ElementTypeTag.QuadUP
            or elem_type == ElementTypeTag.SSPBrickUP
            or elem_type == ElementTypeTag.Tri31
            or elem_type == ElementTypeTag.Shell
        ):
            _assemble_surface_element(
                e,
                elem,
                nodes,
                materials_by_id,
                shell_cache,
                layered_shell_section_defs,
                uniaxial_defs,
                uniaxial_states,
                u,
                K,
                F_int,
            )
        else:
            abort("unsupported element type tag")


fn assemble_link_stiffness_native[
    S: ElementMatrixSink
](
    nodes: List[NodeInput],
    elements: List[ElementInput],
    zero_length_elem_indices: List[Int],
    two_node_link_elem_indices: List[Int],
    ndf: Int,
    ndm: Int,
    u: List[Float64],
    uniaxial_defs: List[UniMaterialDef],
    uniaxial_state_defs: List[Int],
    mut uniaxial_states: List[UniMaterialState],
    elem_uniaxial_offsets: List[Int],
    elem_uniaxial_counts: List[Int],
    elem_uniaxial_state_ids: List[Int],
    elem_free_offsets: List[Int],
    elem_free_pool: List[Int],
    only_rayleigh_participating: Bool,
    mut sink: S,
    mut F_scratch: List[Float64],
):
    """Link-element (zeroLength, twoNodeLink, truss) tangent scattered into `sink`.

    Sparse counterpart of `assemble_link_stiffness_typed`: filter mode 1
    (all links) or 2 (links with `do_rayleigh`). Internal forces go to
    `F_scratch`, which callers discard.
    """
    var filter_mode = 1
    if only_rayleigh_participating:
        filter_mode = 2
    sink.zero_matrix()
    _zero_vector(F_scratch)
    # Trusses use the tangent of their current trial state, which the main
    # assembly has just set at this displacement.
    var truss_k_flat: List[Float64] = []
    truss_k_flat.resize(36, 0.0)
    for e in range(len(elements)):
        ref elem = elements[e]
        if elem.type_tag != ElementTypeTag.Truss:
            continue
        if not _assembly_filter_accepts_element(elem, filter_mode):
            continue
        ref n1 = nodes[elem.node_index_1]
        ref n2 = nodes[elem.node_index_2]
        var dx = n2.x - n1.x
        var dy = n2.y - n1.y
        var state_index = elem_uniaxial_state_ids[elem_uniaxial_offsets[e]]
        var tangent = uniaxial_states[state_index].tangent_t
        if ndf == 2:
            var L = hypot(dx, dy)
            if L == 0.0:
                abort("zero-length element")
            _truss_flat_stiffness[2](
                tangent * elem.area / L,
                SIMD[DType.float64, 4](dx / L, dy / L, 0.0, 0.0),
                truss_k_flat,
            )
            sink.add_element_matrix_flat(e, elem_free_pool, elem_free_offsets[e], 4, truss_k_flat)
        else:
            var dz = n2.z - n1.z
            var L = sqrt(dx * dx + dy * dy + dz * dz)
            if L == 0.0:
                abort("zero-length element")
            _truss_flat_stiffness[3](
                tangent * elem.area / L,
                SIMD[DType.float64, 4](dx / L, dy / L, dz / L, 0.0),
                truss_k_flat,
            )
            sink.add_element_matrix_flat(e, elem_free_pool, elem_free_offsets[e], 6, truss_k_flat)
    for idx in range(len(zero_length_elem_indices)):
        var e = zero_length_elem_indices[idx]
        if not _assembly_filter_accepts_element(elements[e], filter_mode):
            continue
        _assemble_zero_length_element_native(
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
            elem_free_offsets,
            elem_free_pool,
            sink,
            F_scratch,
        )
    for idx in range(len(two_node_link_elem_indices)):
        var e = two_node_link_elem_indices[idx]
        if not _assembly_filter_accepts_element(elements[e], filter_mode):
            continue
        _assemble_two_node_link_element_native(
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
            elem_free_offsets,
            elem_free_pool,
            sink,
            F_scratch,
        )
