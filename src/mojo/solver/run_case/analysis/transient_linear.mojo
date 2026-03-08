from algorithm import vectorize
from collections import List
from elements import (
    ForceBeamColumn2dScratch,
    ForceBeamColumn3dScratch,
    beam2d_element_load_global,
    reset_force_beam_column2d_scratch,
    reset_force_beam_column3d_scratch,
)
from materials import UniMaterialDef, UniMaterialState, uniaxial_commit_all
from os import abort
from python import Python
from sections import (
    FiberCell,
    FiberSection2dDef,
    FiberSection3dDef,
    fiber_section2d_commit_runtime_all,
)
from sys import simd_width_of

from solver.run_case.linear_solver_backend import (
    LinearSolverBackend,
    clear,
    initialize_structure,
    refactor_if_needed,
    solve,
)
from solver.assembly import (
    assemble_global_stiffness_typed_soa,
    assemble_internal_forces_typed_soa,
    assemble_link_stiffness_typed,
    assemble_zero_length_damping_committed_typed,
    assemble_zero_length_damping_typed,
)
from solver.dof import node_dof_index, require_dof_in_range
from solver.profile import PROFILE_FRAME_UNIAXIAL_COPY_RESET, _append_event
from solver.simd_contiguous import (
    copy_float64_contiguous,
    dot_float64_contiguous,
    load_float64_contiguous_simd,
    store_float64_contiguous_simd,
)
from solver.simd_indexed import (
    gather_float64_by_index_simd,
    scatter_float64_by_index_simd,
)
from solver.run_case.input_types import (
    AnalysisInput,
    DampingInput,
    ElementLoadInput,
    ElementInput,
    MaterialInput,
    NodeInput,
    RecorderInput,
    SectionInput,
)
from solver.run_case.load_state import (
    build_active_element_load_state,
    build_active_nodal_load,
)
from solver.time_series import TimeSeriesInput, eval_time_series_input, find_time_series_input

from solver.run_case.helpers import (
    _append_output,
    _collapse_matrix_by_rep,
    _collapse_vector_by_rep,
    _element_basic_force_for_recorder,
    _element_deformation_for_recorder,
    _element_force_global_for_recorder,
    _element_local_force_for_recorder,
    _enforce_equal_dof_values,
    _flush_envelope_outputs,
    _format_values_line,
    _has_recorder_type,
    _drift_value,
    _section_response_for_recorder,
    _update_envelope,
)
from tag_types import (
    ElementTypeTag,
    IntegratorTypeTag,
    PatternTypeTag,
    RecorderTypeTag,
)


@always_inline
fn _has_nonparticipating_link_for_rayleigh(elements: List[ElementInput]) -> Bool:
    for e in range(len(elements)):
        var elem = elements[e]
        if (
            elem.type_tag == ElementTypeTag.ZeroLength
            or elem.type_tag == ElementTypeTag.TwoNodeLink
        ) and not elem.do_rayleigh:
            return True
    return False


fn _has_zero_length_damp_mats(elements: List[ElementInput]) -> Bool:
    for i in range(len(elements)):
        var elem = elements[i]
        if (
            elem.type_tag == ElementTypeTag.ZeroLength
            and elem.damp_material_count > 0
        ):
            return True
    return False


fn _has_zero_length_dampers(elements: List[ElementInput]) -> Bool:
    for i in range(len(elements)):
        var elem = elements[i]
        if elem.type_tag == ElementTypeTag.ZeroLength and elem.damping_tag >= 0:
            return True
    return False


fn _build_solver_load_vector_with_element_loads(
    typed_nodes: List[NodeInput],
    typed_elements: List[ElementInput],
    elem_dof_offsets: List[Int],
    elem_dof_pool: List[Int],
    active_element_loads: List[ElementLoadInput],
    elem_load_offsets: List[Int],
    elem_load_pool: List[Int],
    source: List[Float64],
    mut dst: List[Float64],
):
    copy_float64_contiguous(dst, source, len(dst))
    for elem_index in range(len(typed_elements)):
        if elem_load_offsets[elem_index] == elem_load_offsets[elem_index + 1]:
            continue
        var elem = typed_elements[elem_index]
        if elem.type_tag != ElementTypeTag.ElasticBeamColumn2d:
            abort(
                "transient_linear solver load assembly supports active element loads "
                "on elasticBeamColumn2d only"
            )
        var node1 = typed_nodes[elem.node_index_1]
        var node2 = typed_nodes[elem.node_index_2]
        var f_load = beam2d_element_load_global(
            active_element_loads,
            elem_load_offsets,
            elem_load_pool,
            elem_index,
            1.0,
            node1.x,
            node1.y,
            node2.x,
            node2.y,
        )
        var dof_offset = elem_dof_offsets[elem_index]
        for a in range(6):
            dst[elem_dof_pool[dof_offset + a]] += f_load[a]



@always_inline
fn _build_effective_rhs_newmark_simd_impl[width: Int](
    free: List[Int],
    u_f: List[Float64],
    v_f: List[Float64],
    a_f: List[Float64],
    F_ext_step: List[Float64],
    M_f: List[Float64],
    a0: Float64,
    a1: Float64,
    a2: Float64,
    a3: Float64,
    a4: Float64,
    a5: Float64,
    mut P_ext_f: List[Float64],
    mut C_term: List[Float64],
    mut P_eff: List[Float64],
):
    var n = len(free)

    @parameter
    fn build_chunk[chunk: Int](i: Int):
        var u_vec = load_float64_contiguous_simd[chunk](u_f, i)
        var v_vec = load_float64_contiguous_simd[chunk](v_f, i)
        var a_vec = load_float64_contiguous_simd[chunk](a_f, i)
        var c_vec = (
            SIMD[DType.float64, chunk](a1) * u_vec
            + SIMD[DType.float64, chunk](a4) * v_vec
            + SIMD[DType.float64, chunk](a5) * a_vec
        )
        var p_ext_vec = gather_float64_by_index_simd[chunk](free, i, F_ext_step)
        var m_vec = load_float64_contiguous_simd[chunk](M_f, i)
        var p_eff_vec = p_ext_vec + m_vec * (
            SIMD[DType.float64, chunk](a0) * u_vec
            + SIMD[DType.float64, chunk](a2) * v_vec
            + SIMD[DType.float64, chunk](a3) * a_vec
        )
        store_float64_contiguous_simd[chunk](P_ext_f, i, p_ext_vec)
        store_float64_contiguous_simd[chunk](C_term, i, c_vec)
        store_float64_contiguous_simd[chunk](P_eff, i, p_eff_vec)

    vectorize[build_chunk, width](n)


@always_inline
fn _build_effective_rhs_newmark_simd(
    free: List[Int],
    u_f: List[Float64],
    v_f: List[Float64],
    a_f: List[Float64],
    F_ext_step: List[Float64],
    M_f: List[Float64],
    a0: Float64,
    a1: Float64,
    a2: Float64,
    a3: Float64,
    a4: Float64,
    a5: Float64,
    mut P_ext_f: List[Float64],
    mut C_term: List[Float64],
    mut P_eff: List[Float64],
):
    _build_effective_rhs_newmark_simd_impl[simd_width_of[DType.float64]()](
        free,
        u_f,
        v_f,
        a_f,
        F_ext_step,
        M_f,
        a0,
        a1,
        a2,
        a3,
        a4,
        a5,
        P_ext_f,
        C_term,
        P_eff,
    )


@always_inline
fn _update_newmark_state_from_solution_simd_impl[width: Int](
    mut u_f: List[Float64],
    mut v_f: List[Float64],
    mut a_f: List[Float64],
    u_next_f: List[Float64],
    a0: Float64,
    a2: Float64,
    a3: Float64,
    gamma: Float64,
    dt: Float64,
):
    var n = len(u_f)
    var one_minus_gamma = 1.0 - gamma

    @parameter
    fn update_chunk[chunk: Int](i: Int):
        var u_old = load_float64_contiguous_simd[chunk](u_f, i)
        var v_old = load_float64_contiguous_simd[chunk](v_f, i)
        var a_old = load_float64_contiguous_simd[chunk](a_f, i)
        var u_next = load_float64_contiguous_simd[chunk](u_next_f, i)
        var a_next = (
            SIMD[DType.float64, chunk](a0) * (u_next - u_old)
            - SIMD[DType.float64, chunk](a2) * v_old
            - SIMD[DType.float64, chunk](a3) * a_old
        )
        var v_next = v_old + SIMD[DType.float64, chunk](dt) * (
            SIMD[DType.float64, chunk](one_minus_gamma) * a_old
            + SIMD[DType.float64, chunk](gamma) * a_next
        )
        store_float64_contiguous_simd[chunk](u_f, i, u_next)
        store_float64_contiguous_simd[chunk](v_f, i, v_next)
        store_float64_contiguous_simd[chunk](a_f, i, a_next)

    vectorize[update_chunk, width](n)


@always_inline
fn _update_newmark_state_from_solution_simd(
    mut u_f: List[Float64],
    mut v_f: List[Float64],
    mut a_f: List[Float64],
    u_next_f: List[Float64],
    a0: Float64,
    a2: Float64,
    a3: Float64,
    gamma: Float64,
    dt: Float64,
):
    _update_newmark_state_from_solution_simd_impl[simd_width_of[DType.float64]()](
        u_f, v_f, a_f, u_next_f, a0, a2, a3, gamma, dt
    )


@always_inline
fn _scatter_free_vector_to_full_simd_impl[width: Int](
    free: List[Int], src: List[Float64], mut dst: List[Float64]
):
    var n = len(free)

    @parameter
    fn scatter_chunk[chunk: Int](i: Int):
        var src_vec = load_float64_contiguous_simd[chunk](src, i)
        scatter_float64_by_index_simd[chunk](free, i, dst, src_vec)

    vectorize[scatter_chunk, width](n)


@always_inline
fn _scatter_free_vector_to_full(
    free: List[Int], src: List[Float64], mut dst: List[Float64]
):
    _scatter_free_vector_to_full_simd_impl[simd_width_of[DType.float64]()](
        free, src, dst
    )


fn run_transient_linear(
    analysis: AnalysisInput,
    steps: Int,
    ts_index: Int,
    time_series: List[TimeSeriesInput],
    time_series_values: List[Float64],
    time_series_times: List[Float64],
    dampings: List[DampingInput],
    pattern_type: String,
    pattern_type_tag: Int,
    uniform_excitation_direction: Int,
    uniform_accel_ts_index: Int,
    rayleigh_alpha_m: Float64,
    rayleigh_beta_k: Float64,
    rayleigh_beta_k_init: Float64,
    rayleigh_beta_k_comm: Float64,
    typed_nodes: List[NodeInput],
    typed_elements: List[ElementInput],
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
    const_element_loads: List[ElementLoadInput],
    pattern_element_loads: List[ElementLoadInput],
    typed_sections_by_id: List[SectionInput],
    typed_materials_by_id: List[MaterialInput],
    id_to_index: List[Int],
    node_count: Int,
    ndf: Int,
    ndm: Int,
    total_dofs: Int,
    mut u: List[Float64],
    uniaxial_defs: List[UniMaterialDef],
    uniaxial_state_defs: List[Int],
    mut uniaxial_states: List[UniMaterialState],
    elem_uniaxial_offsets: List[Int],
    elem_uniaxial_counts: List[Int],
    elem_uniaxial_state_ids: List[Int],
    force_basic_offsets: List[Int],
    force_basic_counts: List[Int],
    mut force_basic_q: List[Float64],
    F_const: List[Float64],
    F_pattern: List[Float64],
    M_total: List[Float64],
    M_rayleigh_total: List[Float64],
    free: List[Int],
    recorders: List[RecorderInput],
    recorder_nodes_pool: List[Int],
    recorder_elements_pool: List[Int],
    recorder_dofs_pool: List[Int],
    recorder_sections_pool: List[Int],
    elem_id_to_index: List[Int],
    mut fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    fiber_section3d_defs: List[FiberSection3dDef],
    fiber_section3d_cells: List[FiberCell],
    fiber_section3d_index_by_id: List[Int],
    mut transient_output_files: List[String],
    mut transient_output_buffers: List[List[String]],
    has_transformation_mpc: Bool,
    rep_dof: List[Int],
    constrained: List[Bool],
    do_profile: Bool,
    t0: Int,
    mut events: String,
    mut events_need_comma: Bool,
    frame_assemble_stiffness: Int,
    frame_solve_linear: Int,
    frame_time_series_eval: Int,
    frame_constraints: Int,
    frame_recorders: Int,
    frame_factorize: Int,
    frame_transient_step: Int,
    frame_uniaxial_commit_all: Int,
) raises:
    var time = Python.import_module("time")

    var dt = analysis.dt
    if dt <= 0.0:
        abort("transient_linear requires dt > 0")
    if (
        pattern_type_tag != PatternTypeTag.Plain
        and pattern_type_tag != PatternTypeTag.UniformExcitation
    ):
        abort("unsupported pattern type: " + pattern_type)
    if pattern_type_tag == PatternTypeTag.UniformExcitation:
        if uniform_excitation_direction < 1 or uniform_excitation_direction > ndm:
            abort("UniformExcitation direction out of range")
        if uniform_accel_ts_index < 0:
            abort("UniformExcitation missing accel time_series")
    var uniform_excitation_dofs: List[Int] = []
    if pattern_type_tag == PatternTypeTag.UniformExcitation:
        for i in range(total_dofs):
            if (i % ndf) + 1 == uniform_excitation_direction:
                uniform_excitation_dofs.append(i)
    var integrator_tag = analysis.integrator_tag
    if integrator_tag == IntegratorTypeTag.Unknown:
        integrator_tag = IntegratorTypeTag.Newmark
    if integrator_tag != IntegratorTypeTag.Newmark:
        abort("transient_linear only supports Newmark integrator")
    var gamma = analysis.integrator_gamma
    var beta = analysis.integrator_beta
    if beta <= 0.0:
        abort("Newmark beta must be > 0")

    var free_count = len(free)
    var M_f: List[Float64] = []
    M_f.resize(free_count, 0.0)
    var M_rayleigh_f: List[Float64] = []
    M_rayleigh_f.resize(free_count, 0.0)
    var has_mass = False
    for i in range(free_count):
        var m = M_total[free[i]]
        M_f[i] = m
        M_rayleigh_f[i] = M_rayleigh_total[free[i]]
        if m != 0.0:
            has_mass = True
    if not has_mass:
        abort("transient_linear requires masses on free dofs")
    var force_beam_column2d_scratch = ForceBeamColumn2dScratch()
    var force_beam_column3d_scratch = ForceBeamColumn3dScratch()

    if do_profile:
        var t_asm_start = Int(time.perf_counter_ns())
        var asm_start_us = (t_asm_start - t0) // 1000
        _append_event(
            events, events_need_comma, "O", frame_assemble_stiffness, asm_start_us
        )
    var initial_pattern_scale = 0.0
    if pattern_type_tag == PatternTypeTag.Plain:
        if ts_index >= 0:
            initial_pattern_scale = eval_time_series_input(
                time_series[ts_index], 0.0, time_series_values, time_series_times
            )
        else:
            initial_pattern_scale = 1.0
    var active_element_load_state = build_active_element_load_state(
        const_element_loads,
        pattern_element_loads,
        initial_pattern_scale,
        typed_elements,
        elem_id_to_index,
        ndm,
        ndf,
    )
    if do_profile:
        var t_reset_start = Int(time.perf_counter_ns())
        _append_event(
            events,
            events_need_comma,
            "O",
            PROFILE_FRAME_UNIAXIAL_COPY_RESET,
            (t_reset_start - t0) // 1000,
        )
    reset_force_beam_column2d_scratch(force_beam_column2d_scratch)
    reset_force_beam_column3d_scratch(force_beam_column3d_scratch)
    if do_profile:
        var t_reset_end = Int(time.perf_counter_ns())
        _append_event(
            events,
            events_need_comma,
            "C",
            PROFILE_FRAME_UNIAXIAL_COPY_RESET,
            (t_reset_end - t0) // 1000,
        )
    var K = assemble_global_stiffness_typed_soa(
        typed_nodes,
        typed_elements,
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
        active_element_load_state.element_loads,
        active_element_load_state.elem_load_offsets,
        active_element_load_state.elem_load_pool,
        1.0,
        typed_sections_by_id,
        typed_materials_by_id,
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
    if do_profile:
        var t_asm_end = Int(time.perf_counter_ns())
        var asm_end_us = (t_asm_end - t0) // 1000
        _append_event(
            events, events_need_comma, "C", frame_assemble_stiffness, asm_end_us
        )
    if has_transformation_mpc:
        if do_profile:
            var t_constraints_start = Int(time.perf_counter_ns())
            var constraints_start_us = (t_constraints_start - t0) // 1000
            _append_event(
                events,
                events_need_comma,
                "O",
                frame_constraints,
                constraints_start_us,
            )
        K = _collapse_matrix_by_rep(K, rep_dof)
        if do_profile:
            var t_constraints_end = Int(time.perf_counter_ns())
            var constraints_end_us = (t_constraints_end - t0) // 1000
            _append_event(
                events,
                events_need_comma,
                "C",
                frame_constraints,
                constraints_end_us,
            )
    var K_ff: List[List[Float64]] = []
    for _ in range(free_count):
        var row: List[Float64] = []
        row.resize(free_count, 0.0)
        K_ff.append(row^)
    for i in range(free_count):
        for j in range(free_count):
            K_ff[i][j] = K[free[i]][free[j]]

    var a0 = 1.0 / (beta * dt * dt)
    var a1 = gamma / (beta * dt)
    var a2 = 1.0 / (beta * dt)
    var a3 = 1.0 / (2.0 * beta) - 1.0
    var a4 = gamma / beta - 1.0
    var a5 = dt * (gamma / (2.0 * beta) - 1.0)
    var beta_sum = rayleigh_beta_k + rayleigh_beta_k_init + rayleigh_beta_k_comm
    var K_damping_ff: List[List[Float64]] = []
    for _ in range(free_count):
        var row_damp: List[Float64] = []
        row_damp.resize(free_count, 0.0)
        K_damping_ff.append(row_damp^)
    var need_link_rayleigh_filter = (
        beta_sum != 0.0 and _has_nonparticipating_link_for_rayleigh(typed_elements)
    )
    var has_zero_length_damp_mats = _has_zero_length_damp_mats(typed_elements)
    var has_zero_length_dampers = _has_zero_length_dampers(typed_elements)
    if need_link_rayleigh_filter:
        var asm_dof_map6: List[Int] = []
        var asm_dof_map12: List[Int] = []
        var asm_u_elem6: List[Float64] = []
        var K_link_all: List[List[Float64]] = []
        var K_link_rayleigh: List[List[Float64]] = []
        for _ in range(total_dofs):
            var row_all: List[Float64] = []
            row_all.resize(total_dofs, 0.0)
            K_link_all.append(row_all^)
            var row_rayleigh: List[Float64] = []
            row_rayleigh.resize(total_dofs, 0.0)
            K_link_rayleigh.append(row_rayleigh^)
        assemble_link_stiffness_typed(
            typed_nodes,
            typed_elements,
            typed_sections_by_id,
            typed_materials_by_id,
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
            asm_dof_map6,
            asm_dof_map12,
            asm_u_elem6,
            False,
            K_link_all,
        )
        assemble_link_stiffness_typed(
            typed_nodes,
            typed_elements,
            typed_sections_by_id,
            typed_materials_by_id,
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
            asm_dof_map6,
            asm_dof_map12,
            asm_u_elem6,
            True,
            K_link_rayleigh,
        )
        if has_transformation_mpc:
            K_link_all = _collapse_matrix_by_rep(K_link_all, rep_dof)
            K_link_rayleigh = _collapse_matrix_by_rep(K_link_rayleigh, rep_dof)
        for i in range(free_count):
            for j in range(free_count):
                K_damping_ff[i][j] = (
                    K_ff[i][j]
                    - K_link_all[free[i]][free[j]]
                    + K_link_rayleigh[free[i]][free[j]]
                )
    else:
        for i in range(free_count):
            for j in range(free_count):
                K_damping_ff[i][j] = K_ff[i][j]

    var C_ff: List[List[Float64]] = []
    var C_zero_length_damp_ff: List[List[Float64]] = []
    for i in range(free_count):
        var row_c: List[Float64] = []
        row_c.resize(free_count, 0.0)
        C_ff.append(row_c^)
        var row_zero_length_damp: List[Float64] = []
        row_zero_length_damp.resize(free_count, 0.0)
        C_zero_length_damp_ff.append(row_zero_length_damp^)
        C_ff[i][i] = rayleigh_alpha_m * M_rayleigh_f[i]
    if has_zero_length_damp_mats:
        var C_zero_length_global: List[List[Float64]] = []
        for _ in range(total_dofs):
            var row_global: List[Float64] = []
            row_global.resize(total_dofs, 0.0)
            C_zero_length_global.append(row_global^)
        assemble_zero_length_damping_typed(
            typed_nodes,
            typed_elements,
            typed_materials_by_id,
            node_count,
            ndf,
            ndm,
            C_zero_length_global,
        )
        if has_transformation_mpc:
            C_zero_length_global = _collapse_matrix_by_rep(C_zero_length_global, rep_dof)
        for i in range(free_count):
            for j in range(free_count):
                C_zero_length_damp_ff[i][j] = C_zero_length_global[free[i]][free[j]]
    if beta_sum != 0.0:
        for i in range(free_count):
            for j in range(free_count):
                C_ff[i][j] += beta_sum * K_damping_ff[i][j]
    if has_zero_length_damp_mats:
        for i in range(free_count):
            for j in range(free_count):
                C_ff[i][j] += C_zero_length_damp_ff[i][j]

    var K_eff: List[List[Float64]] = []
    for _ in range(free_count):
        var row_eff: List[Float64] = []
        row_eff.resize(free_count, 0.0)
        K_eff.append(row_eff^)
    for i in range(free_count):
        for j in range(free_count):
            K_eff[i][j] = K_ff[i][j] + a1 * C_ff[i][j]
        K_eff[i][i] += a0 * M_f[i]
    var K_lu: List[List[Float64]] = []
    for _ in range(free_count):
        var row_lu: List[Float64] = []
        row_lu.resize(free_count, 0.0)
        K_lu.append(row_lu^)
    var backend = LinearSolverBackend()
    initialize_structure(backend, analysis, free_count)
    var K_zero_length_damp_ff: List[List[Float64]] = []
    for _ in range(free_count):
        var row_zero_length_damp: List[Float64] = []
        row_zero_length_damp.resize(free_count, 0.0)
        K_zero_length_damp_ff.append(row_zero_length_damp^)
    var F_zero_length_damp_committed_f: List[Float64] = []
    F_zero_length_damp_committed_f.resize(free_count, 0.0)
    if not has_zero_length_dampers:
        if do_profile:
            var t_fac_start = Int(time.perf_counter_ns())
            var fac_start_us = (t_fac_start - t0) // 1000
            _append_event(events, events_need_comma, "O", frame_factorize, fac_start_us)
        _ = refactor_if_needed(backend, K_eff, True, True)
        if do_profile:
            var t_fac_end = Int(time.perf_counter_ns())
            var fac_end_us = (t_fac_end - t0) // 1000
            _append_event(events, events_need_comma, "C", frame_factorize, fac_end_us)

    var v: List[Float64] = []
    v.resize(total_dofs, 0.0)
    var a: List[Float64] = []
    a.resize(total_dofs, 0.0)
    var u_f: List[Float64] = []
    u_f.resize(free_count, 0.0)
    var v_f: List[Float64] = []
    v_f.resize(free_count, 0.0)
    var a_f: List[Float64] = []
    a_f.resize(free_count, 0.0)
    for i in range(free_count):
        u_f[i] = u[free[i]]

    var F_ext_step: List[Float64] = []
    F_ext_step.resize(total_dofs, 0.0)
    var F_solve_step: List[Float64] = []
    F_solve_step.resize(total_dofs, 0.0)
    var P_ext_f: List[Float64] = []
    P_ext_f.resize(free_count, 0.0)
    var P_eff: List[Float64] = []
    P_eff.resize(free_count, 0.0)
    var C_term: List[Float64] = []
    C_term.resize(free_count, 0.0)
    var lu_rhs: List[Float64] = []
    lu_rhs.resize(free_count, 0.0)
    var u_next_f: List[Float64] = []
    u_next_f.resize(free_count, 0.0)
    var record_reactions = _has_recorder_type(recorders, RecorderTypeTag.NodeReaction)
    var record_any_element_force = (
        _has_recorder_type(recorders, RecorderTypeTag.ElementForce) or _has_recorder_type(recorders, RecorderTypeTag.EnvelopeElementForce)
    )
    var elem_count = len(typed_elements)
    var elem_force_cached: List[Bool] = []
    var elem_force_values: List[List[Float64]] = []
    if record_any_element_force:
        elem_force_cached.resize(elem_count, False)
        for _ in range(elem_count):
            var empty_force: List[Float64] = []
            elem_force_values.append(empty_force^)
    var envelope_files: List[String] = []
    var envelope_min: List[List[Float64]] = []
    var envelope_max: List[List[Float64]] = []
    var envelope_abs: List[List[Float64]] = []

    for step in range(steps):
        if do_profile:
            var t_step_start = Int(time.perf_counter_ns())
            var step_start_us = (t_step_start - t0) // 1000
            _append_event(
                events, events_need_comma, "O", frame_transient_step, step_start_us
            )
        if has_transformation_mpc:
            if do_profile:
                var t_constraints_start = Int(time.perf_counter_ns())
                var constraints_start_us = (t_constraints_start - t0) // 1000
                _append_event(
                    events,
                    events_need_comma,
                    "O",
                    frame_constraints,
                    constraints_start_us,
                )
            _enforce_equal_dof_values(u, rep_dof, constrained)
            _enforce_equal_dof_values(v, rep_dof, constrained)
            _enforce_equal_dof_values(a, rep_dof, constrained)
            if do_profile:
                var t_constraints_end = Int(time.perf_counter_ns())
                var constraints_end_us = (t_constraints_end - t0) // 1000
                _append_event(
                    events,
                    events_need_comma,
                    "C",
                    frame_constraints,
                    constraints_end_us,
                )
        var t = Float64(step + 1) * dt
        if pattern_type_tag == PatternTypeTag.UniformExcitation:
            copy_float64_contiguous(F_ext_step, F_const, len(F_ext_step))
            active_element_load_state = build_active_element_load_state(
                const_element_loads,
                pattern_element_loads,
                0.0,
                typed_elements,
                elem_id_to_index,
                ndm,
                ndf,
            )
            if do_profile:
                var t_reset_start = Int(time.perf_counter_ns())
                _append_event(
                    events,
                    events_need_comma,
                    "O",
                    PROFILE_FRAME_UNIAXIAL_COPY_RESET,
                    (t_reset_start - t0) // 1000,
                )
            reset_force_beam_column2d_scratch(force_beam_column2d_scratch)
            reset_force_beam_column3d_scratch(force_beam_column3d_scratch)
            if do_profile:
                var t_reset_end = Int(time.perf_counter_ns())
                _append_event(
                    events,
                    events_need_comma,
                    "C",
                    PROFILE_FRAME_UNIAXIAL_COPY_RESET,
                    (t_reset_end - t0) // 1000,
                )
            if do_profile:
                var t_ts_start = Int(time.perf_counter_ns())
                var ts_start_us = (t_ts_start - t0) // 1000
                _append_event(
                    events,
                    events_need_comma,
                    "O",
                    frame_time_series_eval,
                    ts_start_us,
                )
            var ag = eval_time_series_input(
                time_series[uniform_accel_ts_index],
                t,
                time_series_values,
                time_series_times,
            )
            if do_profile:
                var t_ts_end = Int(time.perf_counter_ns())
                var ts_end_us = (t_ts_end - t0) // 1000
                _append_event(
                    events,
                    events_need_comma,
                    "C",
                    frame_time_series_eval,
                    ts_end_us,
                )
            for i in range(len(uniform_excitation_dofs)):
                var dof_idx = uniform_excitation_dofs[i]
                F_ext_step[dof_idx] += -M_total[dof_idx] * ag
        else:
            if ts_index >= 0:
                if do_profile:
                    var t_ts_start = Int(time.perf_counter_ns())
                    var ts_start_us = (t_ts_start - t0) // 1000
                    _append_event(
                        events,
                        events_need_comma,
                        "O",
                        frame_time_series_eval,
                        ts_start_us,
                    )
                var factor = eval_time_series_input(
                    time_series[ts_index],
                    t,
                    time_series_values,
                    time_series_times,
                )
                if do_profile:
                    var t_ts_end = Int(time.perf_counter_ns())
                    var ts_end_us = (t_ts_end - t0) // 1000
                    _append_event(
                        events,
                    events_need_comma,
                    "C",
                    frame_time_series_eval,
                    ts_end_us,
                )
                copy_float64_contiguous(
                    F_ext_step,
                    build_active_nodal_load(F_const, F_pattern, factor),
                    len(F_ext_step),
                )
                active_element_load_state = build_active_element_load_state(
                    const_element_loads,
                    pattern_element_loads,
                    factor,
                    typed_elements,
                    elem_id_to_index,
                    ndm,
                    ndf,
                )
                if do_profile:
                    var t_reset_start = Int(time.perf_counter_ns())
                    _append_event(
                        events,
                        events_need_comma,
                        "O",
                        PROFILE_FRAME_UNIAXIAL_COPY_RESET,
                        (t_reset_start - t0) // 1000,
                    )
                reset_force_beam_column2d_scratch(force_beam_column2d_scratch)
                reset_force_beam_column3d_scratch(force_beam_column3d_scratch)
                if do_profile:
                    var t_reset_end = Int(time.perf_counter_ns())
                    _append_event(
                        events,
                        events_need_comma,
                        "C",
                        PROFILE_FRAME_UNIAXIAL_COPY_RESET,
                        (t_reset_end - t0) // 1000,
                    )
            else:
                copy_float64_contiguous(
                    F_ext_step,
                    build_active_nodal_load(F_const, F_pattern, 1.0),
                    len(F_ext_step),
                )
                active_element_load_state = build_active_element_load_state(
                    const_element_loads,
                    pattern_element_loads,
                    1.0,
                    typed_elements,
                    elem_id_to_index,
                    ndm,
                    ndf,
                )
                if do_profile:
                    var t_reset_start = Int(time.perf_counter_ns())
                    _append_event(
                        events,
                        events_need_comma,
                        "O",
                        PROFILE_FRAME_UNIAXIAL_COPY_RESET,
                        (t_reset_start - t0) // 1000,
                    )
                reset_force_beam_column2d_scratch(force_beam_column2d_scratch)
                reset_force_beam_column3d_scratch(force_beam_column3d_scratch)
                if do_profile:
                    var t_reset_end = Int(time.perf_counter_ns())
                    _append_event(
                        events,
                        events_need_comma,
                        "C",
                        PROFILE_FRAME_UNIAXIAL_COPY_RESET,
                        (t_reset_end - t0) // 1000,
                    )
        _build_solver_load_vector_with_element_loads(
            typed_nodes,
            typed_elements,
            elem_dof_offsets,
            elem_dof_pool,
            active_element_load_state.element_loads,
            active_element_load_state.elem_load_offsets,
            active_element_load_state.elem_load_pool,
            F_ext_step,
            F_solve_step,
        )
        _build_effective_rhs_newmark_simd(
            free,
            u_f,
            v_f,
            a_f,
            F_solve_step,
            M_f,
            a0,
            a1,
            a2,
            a3,
            a4,
            a5,
            P_ext_f,
            C_term,
            P_eff,
        )
        if has_zero_length_dampers:
            var K_zero_length_damp_global: List[List[Float64]] = []
            var F_zero_length_damp_committed: List[Float64] = []
            for _ in range(total_dofs):
                var row_global: List[Float64] = []
                row_global.resize(total_dofs, 0.0)
                K_zero_length_damp_global.append(row_global^)
            F_zero_length_damp_committed.resize(total_dofs, 0.0)
            assemble_zero_length_damping_committed_typed(
                typed_nodes,
                typed_elements,
                dampings,
                time_series,
                time_series_values,
                time_series_times,
                ndf,
                ndm,
                t,
                dt,
                uniaxial_states,
                elem_uniaxial_offsets,
                elem_uniaxial_counts,
                elem_uniaxial_state_ids,
                K_zero_length_damp_global,
                F_zero_length_damp_committed,
            )
            if has_transformation_mpc:
                K_zero_length_damp_global = _collapse_matrix_by_rep(
                    K_zero_length_damp_global, rep_dof
                )
                F_zero_length_damp_committed = _collapse_vector_by_rep(
                    F_zero_length_damp_committed, rep_dof
                )
            for i in range(free_count):
                F_zero_length_damp_committed_f[i] = F_zero_length_damp_committed[free[i]]
                for j in range(free_count):
                    K_zero_length_damp_ff[i][j] = K_zero_length_damp_global[free[i]][free[j]]
                    K_lu[i][j] = K_eff[i][j] + K_zero_length_damp_ff[i][j]
            if do_profile:
                var t_fac_start = Int(time.perf_counter_ns())
                var fac_start_us = (t_fac_start - t0) // 1000
                _append_event(events, events_need_comma, "O", frame_factorize, fac_start_us)
            _ = refactor_if_needed(backend, K_lu, True, True)
            if do_profile:
                var t_fac_end = Int(time.perf_counter_ns())
                var fac_end_us = (t_fac_end - t0) // 1000
                _append_event(events, events_need_comma, "C", frame_factorize, fac_end_us)
        for i in range(free_count):
            lu_rhs[i] = P_eff[i] + dot_float64_contiguous(C_ff[i], C_term, free_count)
            if has_zero_length_dampers:
                lu_rhs[i] += F_zero_length_damp_committed_f[i]
        if do_profile:
            var t_solve_start = Int(time.perf_counter_ns())
            var solve_start_us = (t_solve_start - t0) // 1000
            _append_event(
                events, events_need_comma, "O", frame_solve_linear, solve_start_us
            )
        solve(backend, lu_rhs, u_next_f)
        if do_profile:
            var t_solve_end = Int(time.perf_counter_ns())
            var solve_end_us = (t_solve_end - t0) // 1000
            _append_event(
                events, events_need_comma, "C", frame_solve_linear, solve_end_us
            )
        _update_newmark_state_from_solution_simd(
            u_f, v_f, a_f, u_next_f, a0, a2, a3, gamma, dt
        )
        var need_full_state = (
            has_transformation_mpc or record_reactions or len(recorders) > 0 or step == steps - 1
        )
        if need_full_state:
            _scatter_free_vector_to_full(free, u_f, u)
            _scatter_free_vector_to_full(free, v_f, v)
            _scatter_free_vector_to_full(free, a_f, a)
        if has_transformation_mpc:
            if do_profile:
                var t_constraints_start = Int(time.perf_counter_ns())
                var constraints_start_us = (t_constraints_start - t0) // 1000
                _append_event(
                    events,
                    events_need_comma,
                    "O",
                    frame_constraints,
                    constraints_start_us,
                )
            _enforce_equal_dof_values(u, rep_dof, constrained)
            _enforce_equal_dof_values(v, rep_dof, constrained)
            _enforce_equal_dof_values(a, rep_dof, constrained)
            if do_profile:
                var t_constraints_end = Int(time.perf_counter_ns())
                var constraints_end_us = (t_constraints_end - t0) // 1000
                _append_event(
                    events,
                    events_need_comma,
                    "C",
                    frame_constraints,
                    constraints_end_us,
                )

        var F_int_reaction: List[Float64] = []
        if do_profile:
            var t_rec_start = Int(time.perf_counter_ns())
            var rec_start_us = (t_rec_start - t0) // 1000
            _append_event(events, events_need_comma, "O", frame_recorders, rec_start_us)
        if record_reactions:
            F_int_reaction = assemble_internal_forces_typed_soa(
                typed_nodes,
                typed_elements,
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
                active_element_load_state.element_loads,
                active_element_load_state.elem_load_offsets,
                active_element_load_state.elem_load_pool,
                1.0,
                typed_sections_by_id,
                typed_materials_by_id,
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
        elif has_zero_length_dampers:
            _ = assemble_internal_forces_typed_soa(
                typed_nodes,
                typed_elements,
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
                active_element_load_state.element_loads,
                active_element_load_state.elem_load_offsets,
                active_element_load_state.elem_load_pool,
                1.0,
                typed_sections_by_id,
                typed_materials_by_id,
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
        if record_any_element_force:
            for i in range(elem_count):
                elem_force_cached[i] = False

        for r in range(len(recorders)):
            var rec = recorders[r]
            if (
                rec.type_tag == RecorderTypeTag.NodeDisplacement
                or rec.type_tag == RecorderTypeTag.EnvelopeNodeDisplacement
                or rec.type_tag == RecorderTypeTag.EnvelopeNodeAcceleration
            ):
                for nidx in range(rec.node_count):
                    var node_id = recorder_nodes_pool[rec.node_offset + nidx]
                    var i = id_to_index[node_id]
                    var values: List[Float64] = []
                    values.resize(rec.dof_count, 0.0)
                    for j in range(rec.dof_count):
                        var dof = recorder_dofs_pool[rec.dof_offset + j]
                        require_dof_in_range(dof, ndf, "recorder")
                        var value = u[node_dof_index(i, dof, ndf)]
                        if rec.type_tag == RecorderTypeTag.EnvelopeNodeAcceleration:
                            value = a[node_dof_index(i, dof, ndf)]
                            if rec.time_series_tag >= 0:
                                var ts_index = find_time_series_input(
                                    time_series, rec.time_series_tag
                                )
                                if ts_index < 0:
                                    abort("recorder time series not found")
                                value += eval_time_series_input(
                                    time_series[ts_index],
                                    t,
                                    time_series_values,
                                    time_series_times,
                                )
                        values[j] = value
                    var filename = rec.output + "_node" + String(node_id) + ".out"
                    if rec.type_tag == RecorderTypeTag.NodeDisplacement:
                        _append_output(
                            transient_output_files,
                            transient_output_buffers,
                            filename,
                            _format_values_line(values),
                        )
                    else:
                        _update_envelope(
                            filename,
                            values,
                            envelope_files,
                            envelope_min,
                            envelope_max,
                            envelope_abs,
                        )
            elif rec.type_tag == RecorderTypeTag.ElementForce:
                for eidx in range(rec.element_count):
                    var elem_id = recorder_elements_pool[rec.element_offset + eidx]
                    if elem_id >= len(elem_id_to_index) or elem_id_to_index[elem_id] < 0:
                        abort("recorder element not found")
                    var elem_index = elem_id_to_index[elem_id]
                    if not elem_force_cached[elem_index]:
                        var elem = typed_elements[elem_index]
                        elem_force_values[elem_index] = (
                            _element_force_global_for_recorder(
                        elem_index,
                        elem,
                        ndf,
                        u,
                        active_element_load_state.element_loads,
                        active_element_load_state.elem_load_offsets,
                        active_element_load_state.elem_load_pool,
                        1.0,
                        typed_nodes,
                        typed_sections_by_id,
                                fiber_section_defs,
                                fiber_section_cells,
                                fiber_section_index_by_id,
                                fiber_section3d_defs,
                                fiber_section3d_cells,
                                fiber_section3d_index_by_id,
                                uniaxial_defs,
                                uniaxial_state_defs,
                                uniaxial_states,
                                elem_uniaxial_offsets,
                                elem_uniaxial_counts,
                                elem_uniaxial_state_ids,
                                force_basic_offsets,
                                force_basic_counts,
                                force_basic_q,
                            )
                        )
                        elem_force_cached[elem_index] = True
                    var line = _format_values_line(elem_force_values[elem_index])
                    var filename = rec.output + "_ele" + String(elem_id) + ".out"
                    _append_output(
                        transient_output_files, transient_output_buffers, filename, line
                    )
            elif rec.type_tag == RecorderTypeTag.ElementLocalForce:
                for eidx in range(rec.element_count):
                    var elem_id = recorder_elements_pool[rec.element_offset + eidx]
                    if elem_id >= len(elem_id_to_index) or elem_id_to_index[elem_id] < 0:
                        abort("recorder element not found")
                    var elem_index = elem_id_to_index[elem_id]
                    var elem = typed_elements[elem_index]
                    var values = _element_local_force_for_recorder(
                        elem_index,
                        elem,
                        ndf,
                        u,
                        typed_nodes,
                        typed_sections_by_id,
                        fiber_section_defs,
                        fiber_section_cells,
                        fiber_section_index_by_id,
                        fiber_section3d_defs,
                        fiber_section3d_cells,
                        fiber_section3d_index_by_id,
                        uniaxial_defs,
                        uniaxial_state_defs,
                        uniaxial_states,
                        elem_uniaxial_offsets,
                        elem_uniaxial_counts,
                        elem_uniaxial_state_ids,
                        force_basic_offsets,
                        force_basic_counts,
                        force_basic_q,
                    )
                    var filename = rec.output + "_ele" + String(elem_id) + ".out"
                    _append_output(
                        transient_output_files,
                        transient_output_buffers,
                        filename,
                        _format_values_line(values),
                    )
            elif rec.type_tag == RecorderTypeTag.ElementBasicForce:
                for eidx in range(rec.element_count):
                    var elem_id = recorder_elements_pool[rec.element_offset + eidx]
                    if elem_id >= len(elem_id_to_index) or elem_id_to_index[elem_id] < 0:
                        abort("recorder element not found")
                    var elem_index = elem_id_to_index[elem_id]
                    var elem = typed_elements[elem_index]
                    var values = _element_basic_force_for_recorder(
                        elem_index,
                        elem,
                        ndf,
                        u,
                        typed_nodes,
                        uniaxial_defs,
                        uniaxial_state_defs,
                        uniaxial_states,
                        elem_uniaxial_offsets,
                        elem_uniaxial_counts,
                        elem_uniaxial_state_ids,
                    )
                    var filename = rec.output + "_ele" + String(elem_id) + ".out"
                    _append_output(
                        transient_output_files,
                        transient_output_buffers,
                        filename,
                        _format_values_line(values),
                    )
            elif rec.type_tag == RecorderTypeTag.ElementDeformation:
                for eidx in range(rec.element_count):
                    var elem_id = recorder_elements_pool[rec.element_offset + eidx]
                    if elem_id >= len(elem_id_to_index) or elem_id_to_index[elem_id] < 0:
                        abort("recorder element not found")
                    var elem_index = elem_id_to_index[elem_id]
                    var elem = typed_elements[elem_index]
                    var values = _element_deformation_for_recorder(
                        elem_index, elem, ndf, u, typed_nodes
                    )
                    var filename = rec.output + "_ele" + String(elem_id) + ".out"
                    _append_output(
                        transient_output_files,
                        transient_output_buffers,
                        filename,
                        _format_values_line(values),
                    )
            elif rec.type_tag == RecorderTypeTag.NodeReaction:
                if not record_reactions:
                    abort("internal error: reaction recorder flag mismatch")
                for nidx in range(rec.node_count):
                    var node_id = recorder_nodes_pool[rec.node_offset + nidx]
                    var i = id_to_index[node_id]
                    var values: List[Float64] = []
                    values.resize(rec.dof_count, 0.0)
                    for j in range(rec.dof_count):
                        var dof = recorder_dofs_pool[rec.dof_offset + j]
                        require_dof_in_range(dof, ndf, "recorder")
                        var idx = node_dof_index(i, dof, ndf)
                        values[j] = F_int_reaction[idx] - F_ext_step[idx]
                    var line = _format_values_line(values)
                    var filename = rec.output + "_node" + String(node_id) + ".out"
                    _append_output(
                        transient_output_files, transient_output_buffers, filename, line
                    )
            elif rec.type_tag == RecorderTypeTag.Drift:
                var i_node = rec.i_node
                var j_node = rec.j_node
                var value = _drift_value(rec, typed_nodes, id_to_index, ndf, u)
                var filename = (
                    rec.output
                    + "_i"
                    + String(i_node)
                    + "_j"
                    + String(j_node)
                    + ".out"
                )
                _append_output(
                    transient_output_files,
                    transient_output_buffers,
                    filename,
                    _format_values_line([value]),
                )
            elif rec.type_tag == RecorderTypeTag.EnvelopeElementForce:
                for eidx in range(rec.element_count):
                    var elem_id = recorder_elements_pool[rec.element_offset + eidx]
                    if elem_id >= len(elem_id_to_index) or elem_id_to_index[elem_id] < 0:
                        abort("recorder element not found")
                    var elem_index = elem_id_to_index[elem_id]
                    if not elem_force_cached[elem_index]:
                        var elem = typed_elements[elem_index]
                        elem_force_values[elem_index] = (
                            _element_force_global_for_recorder(
                                elem_index,
                                elem,
                                ndf,
                                u,
                                active_element_load_state.element_loads,
                                active_element_load_state.elem_load_offsets,
                                active_element_load_state.elem_load_pool,
                                1.0,
                                typed_nodes,
                                typed_sections_by_id,
                                fiber_section_defs,
                                fiber_section_cells,
                                fiber_section_index_by_id,
                                fiber_section3d_defs,
                                fiber_section3d_cells,
                                fiber_section3d_index_by_id,
                                uniaxial_defs,
                                uniaxial_state_defs,
                                uniaxial_states,
                                elem_uniaxial_offsets,
                                elem_uniaxial_counts,
                                elem_uniaxial_state_ids,
                                force_basic_offsets,
                                force_basic_counts,
                                force_basic_q,
                            )
                        )
                        elem_force_cached[elem_index] = True
                    var filename = rec.output + "_ele" + String(elem_id) + ".out"
                    _update_envelope(
                        filename,
                        elem_force_values[elem_index],
                        envelope_files,
                        envelope_min,
                        envelope_max,
                        envelope_abs,
                    )
            elif rec.type_tag == RecorderTypeTag.EnvelopeElementLocalForce:
                for eidx in range(rec.element_count):
                    var elem_id = recorder_elements_pool[rec.element_offset + eidx]
                    if elem_id >= len(elem_id_to_index) or elem_id_to_index[elem_id] < 0:
                        abort("recorder element not found")
                    var elem_index = elem_id_to_index[elem_id]
                    var elem = typed_elements[elem_index]
                    var values = _element_local_force_for_recorder(
                        elem_index,
                        elem,
                        ndf,
                        u,
                        typed_nodes,
                        typed_sections_by_id,
                        fiber_section_defs,
                        fiber_section_cells,
                        fiber_section_index_by_id,
                        fiber_section3d_defs,
                        fiber_section3d_cells,
                        fiber_section3d_index_by_id,
                        uniaxial_defs,
                        uniaxial_state_defs,
                        uniaxial_states,
                        elem_uniaxial_offsets,
                        elem_uniaxial_counts,
                        elem_uniaxial_state_ids,
                        force_basic_offsets,
                        force_basic_counts,
                        force_basic_q,
                    )
                    var filename = rec.output + "_ele" + String(elem_id) + ".out"
                    _update_envelope(
                        filename,
                        values,
                        envelope_files,
                        envelope_min,
                        envelope_max,
                        envelope_abs,
                    )
            elif (
                rec.type_tag == RecorderTypeTag.SectionForce
                or rec.type_tag == RecorderTypeTag.SectionDeformation
            ):
                var want_defo = rec.type_tag == RecorderTypeTag.SectionDeformation
                for eidx in range(rec.element_count):
                    var elem_id = recorder_elements_pool[rec.element_offset + eidx]
                    if elem_id >= len(elem_id_to_index) or elem_id_to_index[elem_id] < 0:
                        abort("recorder element not found")
                    var elem_index = elem_id_to_index[elem_id]
                    var elem = typed_elements[elem_index]
                    for sidx in range(rec.section_count):
                        var sec_no = recorder_sections_pool[rec.section_offset + sidx]
                        var values = _section_response_for_recorder(
                            elem_index,
                            elem,
                            sec_no,
                            ndf,
                            u,
                            active_element_load_state.element_loads,
                            active_element_load_state.elem_load_offsets,
                            active_element_load_state.elem_load_pool,
                            1.0,
                            typed_nodes,
                            typed_sections_by_id,
                            fiber_section_defs,
                            fiber_section_cells,
                            fiber_section_index_by_id,
                            fiber_section3d_defs,
                            fiber_section3d_cells,
                            fiber_section3d_index_by_id,
                            uniaxial_defs,
                            uniaxial_states,
                            elem_uniaxial_offsets,
                            elem_uniaxial_counts,
                            elem_uniaxial_state_ids,
                            force_basic_offsets,
                            force_basic_counts,
                            force_basic_q,
                            want_defo,
                        )
                        var filename = (
                            rec.output
                            + "_ele"
                            + String(elem_id)
                            + "_sec"
                            + String(sec_no)
                            + ".out"
                        )
                        _append_output(
                            transient_output_files,
                            transient_output_buffers,
                            filename,
                            _format_values_line(values),
                        )
            else:
                abort("unsupported recorder type")
        if do_profile:
            var t_rec_end = Int(time.perf_counter_ns())
            var rec_end_us = (t_rec_end - t0) // 1000
            _append_event(events, events_need_comma, "C", frame_recorders, rec_end_us)
            _append_event(
                events, events_need_comma, "C", frame_transient_step, rec_end_us
            )
        if has_zero_length_dampers:
            if do_profile:
                var t_commit_start = Int(time.perf_counter_ns())
                var commit_start_us = (t_commit_start - t0) // 1000
                _append_event(
                    events,
                    events_need_comma,
                    "O",
                    frame_uniaxial_commit_all,
                    commit_start_us,
                )
            uniaxial_commit_all(uniaxial_states)
            fiber_section2d_commit_runtime_all(fiber_section_defs)
            if do_profile:
                var t_commit_end = Int(time.perf_counter_ns())
                var commit_end_us = (t_commit_end - t0) // 1000
                _append_event(
                    events,
                    events_need_comma,
                    "C",
                    frame_uniaxial_commit_all,
                    commit_end_us,
                )
    clear(backend)
    _flush_envelope_outputs(
        envelope_files,
        envelope_min,
        envelope_max,
        envelope_abs,
        transient_output_files,
        transient_output_buffers,
    )
