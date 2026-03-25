from collections import List
from os import abort
from python import Python, PythonObject

from solver.assembly import assemble_internal_forces_typed_soa
from solver.dof import node_dof_index, require_dof_in_range
from solver.profile import (
    PROFILE_FRAME_ASSEMBLE,
    PROFILE_FRAME_ASSEMBLE_FIBER,
    PROFILE_FRAME_ASSEMBLE_FIBER_GEOMETRY,
    PROFILE_FRAME_ASSEMBLE_FIBER_INTERNAL_FORCE,
    PROFILE_FRAME_ASSEMBLE_FIBER_MATRIX_SCATTER,
    PROFILE_FRAME_ASSEMBLE_FIBER_SECTION_RESPONSE,
    PROFILE_FRAME_ASSEMBLE_STIFFNESS,
    PROFILE_FRAME_ASSEMBLE_UNIAXIAL,
    PROFILE_FRAME_CASE_LOAD_PARSE,
    PROFILE_FRAME_CONSTRAINTS,
    PROFILE_FRAME_FACTORIZE,
    PROFILE_FRAME_KFF_EXTRACT,
    PROFILE_FRAME_MODEL_BUILD_DOF_MAP,
    PROFILE_FRAME_NONLINEAR_ITER,
    PROFILE_FRAME_NONLINEAR_STEP,
    PROFILE_FRAME_OUTPUT,
    PROFILE_FRAME_RECORDERS,
    PROFILE_FRAME_SOLVE,
    PROFILE_FRAME_SOLVE_LINEAR,
    PROFILE_FRAME_SOLVE_NONLINEAR,
    PROFILE_FRAME_TIME_SERIES_EVAL,
    PROFILE_FRAME_TOTAL,
    PROFILE_FRAME_TRANSIENT_STEP,
    PROFILE_FRAME_UNIAXIAL_COMMIT_ALL,
    PROFILE_FRAME_UNIAXIAL_COPY_RESET,
    PROFILE_FRAME_UNIAXIAL_REVERT_ALL,
    PROFILE_FRAME_UNIAXIAL_TRIAL_UPDATE,
    RuntimeProfileMetrics,
    _append_event,
    _append_frame,
    _profile_metrics_element_type_name,
    _profile_enabled,
    _write_speedscope,
)

from solver.run_case.analysis.static_linear import run_static_linear
from solver.run_case.analysis.static_nonlinear import (
    run_static_nonlinear_displacement_control,
    run_static_nonlinear_load_control,
)
from solver.run_case.analysis.transient_linear import run_transient_linear
from solver.run_case.analysis.transient_nonlinear import run_transient_nonlinear
from solver.run_case.analysis.modal_eigen import run_modal_eigen
from solver.run_case.input_types import (
    CaseInput,
    ElementLoadInput,
    ElementInput,
    NodalLoadInput,
    NodeInput,
    StageInput,
    parse_case_input_native_from_source,
)
from solver.run_case.helpers import (
    _collapse_vector_by_mpc,
    _drift_value,
    _element_basic_force_for_recorder,
    _element_force_global_for_recorder,
    _element_local_force_for_recorder,
    _element_deformation_for_recorder,
    _enforce_mpc_values,
    _format_values_line,
    _has_recorder_type,
    _section_response_for_recorder,
    _write_run_progress,
    _update_envelope,
)
from strut_io import CaseSourceInfo
from json_native import JsonDocument
from solver.run_case.load_state import (
    append_scaled_element_loads,
    build_active_element_load_state,
    build_active_nodal_load,
)
from solver.run_case.loader import load_case_state_from_input
from solver.run_case.precheck import precheck_case_input_native
from solver.time_series import TimeSeriesInput, eval_time_series_input, find_time_series_input
from tag_types import (
    AnalysisTypeTag,
    ElementTypeTag,
    IntegratorTypeTag,
    PatternTypeTag,
    RecorderTypeTag,
)


fn _write_output_chunk_files(
    out_dir: PythonObject, filenames: List[String], buffers: List[String]
) raises:
    var builtins = Python.import_module("builtins")
    for i in range(len(filenames)):
        var file_path = out_dir.joinpath(filenames[i])
        var file_obj = builtins.open(file_path, "w")
        file_obj.write(PythonObject(buffers[i]))
        file_obj.close()


fn _build_stage_nodal_force_vector(
    loads: List[NodalLoadInput],
    id_to_index: List[Int],
    ndf: Int,
    total_dofs: Int,
    has_transformation_mpc: Bool,
    mpc_row_offsets: List[Int],
    mpc_dof_pool: List[Int],
    mpc_coeff_pool: List[Float64],
) raises -> List[Float64]:
    var F_stage: List[Float64] = []
    F_stage.resize(total_dofs, 0.0)
    for i in range(len(loads)):
        var load = loads[i]
        var node_id = load.node
        if node_id >= len(id_to_index) or id_to_index[node_id] < 0:
            abort("load node not found")
        var dof = load.dof
        require_dof_in_range(dof, ndf, "load")
        var idx = node_dof_index(id_to_index[node_id], dof, ndf)
        F_stage[idx] += load.value
    if has_transformation_mpc:
        return _collapse_vector_by_mpc(
            F_stage, mpc_row_offsets, mpc_dof_pool, mpc_coeff_pool
        )
    return F_stage^


fn _zero_nodal_force_vector(total_dofs: Int) -> List[Float64]:
    var F_stage: List[Float64] = []
    F_stage.resize(total_dofs, 0.0)
    return F_stage^


fn _append_stage_time_series(
    stage: StageInput,
    mut time_series: List[TimeSeriesInput],
    mut time_series_values: List[Float64],
    mut time_series_times: List[Float64],
) raises:
    var values_offset = len(time_series_values)
    var time_offset = len(time_series_times)
    for i in range(len(stage.time_series_values)):
        time_series_values.append(stage.time_series_values[i])
    for i in range(len(stage.time_series_times)):
        time_series_times.append(stage.time_series_times[i])

    for i in range(len(stage.time_series)):
        var ts = stage.time_series[i]
        if find_time_series_input(time_series, ts.tag) >= 0:
            abort("duplicate time_series tag in staged analysis")
        ts.values_offset += values_offset
        ts.time_offset += time_offset
        time_series.append(ts)


def run_case_input(
    var input: CaseInput, output_path: String, profile_path: String, case_load_us: Int
):
    var time = Python.import_module("time")
    var pathlib = Python.import_module("pathlib")
    var out_dir = pathlib.Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    var progress_path = output_path + "/run_progress.json"
    var t_start = Int(time.perf_counter_ns())
    var safe_case_load_us = case_load_us
    if safe_case_load_us < 0:
        safe_case_load_us = 0
    var t0 = t_start - safe_case_load_us * 1000
    var do_profile = _profile_enabled(profile_path)
    var runtime_metrics = RuntimeProfileMetrics()
    runtime_metrics.enabled = do_profile
    _write_run_progress(progress_path, "loading", "case", 0, 0, 0, 0)

    var frame_total = PROFILE_FRAME_TOTAL
    var frame_assemble = PROFILE_FRAME_ASSEMBLE
    var frame_solve = PROFILE_FRAME_SOLVE
    var frame_output = PROFILE_FRAME_OUTPUT
    var frame_assemble_stiffness = PROFILE_FRAME_ASSEMBLE_STIFFNESS
    var frame_kff_extract = PROFILE_FRAME_KFF_EXTRACT
    var frame_solve_linear = PROFILE_FRAME_SOLVE_LINEAR
    var frame_solve_nonlinear = PROFILE_FRAME_SOLVE_NONLINEAR
    var frame_nonlinear_step = PROFILE_FRAME_NONLINEAR_STEP
    var frame_nonlinear_iter = PROFILE_FRAME_NONLINEAR_ITER
    var frame_time_series_eval = PROFILE_FRAME_TIME_SERIES_EVAL
    var frame_constraints = PROFILE_FRAME_CONSTRAINTS
    var frame_recorders = PROFILE_FRAME_RECORDERS
    var frame_factorize = PROFILE_FRAME_FACTORIZE
    var frame_transient_step = PROFILE_FRAME_TRANSIENT_STEP
    var frame_case_load_parse = PROFILE_FRAME_CASE_LOAD_PARSE
    var frame_model_build_dof_map = PROFILE_FRAME_MODEL_BUILD_DOF_MAP
    var frame_assemble_uniaxial = PROFILE_FRAME_ASSEMBLE_UNIAXIAL
    var frame_assemble_fiber = PROFILE_FRAME_ASSEMBLE_FIBER
    var frame_uniaxial_revert_all = PROFILE_FRAME_UNIAXIAL_REVERT_ALL
    var frame_uniaxial_commit_all = PROFILE_FRAME_UNIAXIAL_COMMIT_ALL

    var frames = String()
    var events = String()
    var frames_need_comma = False
    var events_need_comma = False
    if do_profile:
        _append_frame(frames, frames_need_comma, "total")
        _append_frame(frames, frames_need_comma, "assemble")
        _append_frame(frames, frames_need_comma, "solve")
        _append_frame(frames, frames_need_comma, "output")
        _append_frame(frames, frames_need_comma, "assemble_stiffness")
        _append_frame(frames, frames_need_comma, "kff_extract")
        _append_frame(frames, frames_need_comma, "solve_linear")
        _append_frame(frames, frames_need_comma, "solve_nonlinear")
        _append_frame(frames, frames_need_comma, "nonlinear_step")
        _append_frame(frames, frames_need_comma, "nonlinear_iter")
        _append_frame(frames, frames_need_comma, "time_series_eval")
        _append_frame(frames, frames_need_comma, "constraints")
        _append_frame(frames, frames_need_comma, "recorders")
        _append_frame(frames, frames_need_comma, "factorize")
        _append_frame(frames, frames_need_comma, "transient_step")
        _append_frame(frames, frames_need_comma, "case_load_parse")
        _append_frame(frames, frames_need_comma, "model_build_dof_map")
        _append_frame(frames, frames_need_comma, "assemble_uniaxial")
        _append_frame(frames, frames_need_comma, "assemble_fiber")
        _append_frame(frames, frames_need_comma, "uniaxial_revert_all")
        _append_frame(frames, frames_need_comma, "uniaxial_commit_all")
        _append_frame(frames, frames_need_comma, "assemble_fiber_geometry")
        _append_frame(frames, frames_need_comma, "assemble_fiber_section_response")
        _append_frame(frames, frames_need_comma, "assemble_fiber_matrix_scatter")
        _append_frame(frames, frames_need_comma, "assemble_fiber_internal_force")
        _append_frame(frames, frames_need_comma, "uniaxial_trial_update")
        _append_frame(frames, frames_need_comma, "uniaxial_copy_reset")
        _append_event(events, events_need_comma, "O", frame_total, 0)
        _append_event(events, events_need_comma, "O", frame_case_load_parse, 0)
        _append_event(
            events, events_need_comma, "C", frame_case_load_parse, safe_case_load_us
        )
        _append_event(events, events_need_comma, "O", frame_assemble, safe_case_load_us)
        _append_event(
            events,
            events_need_comma,
            "O",
            frame_model_build_dof_map,
            safe_case_load_us,
        )

    var state = load_case_state_from_input(input^, progress_path)

    var ndm = state.ndm
    var ndf = state.ndf
    ref typed_nodes = state.typed_nodes
    var node_count = state.node_count
    ref typed_elements = state.typed_elements
    var total_dofs = state.total_dofs
    ref analysis = state.analysis
    ref analysis_type = state.analysis_type
    var analysis_type_tag = state.analysis_type_tag
    var steps = state.steps
    var modal_num_modes = state.modal_num_modes
    var has_transformation_mpc = state.has_transformation_mpc
    var supports_linear_transient_fast_path = (
        state.supports_linear_transient_fast_path
    )
    ref time_series = state.time_series
    ref time_series_values = state.time_series_values
    ref time_series_times = state.time_series_times
    ref dampings = state.dampings
    var ts_index = state.ts_index
    ref pattern_type = state.pattern_type
    var pattern_type_tag = state.pattern_type_tag
    var uniform_excitation_direction = state.uniform_excitation_direction
    var uniform_accel_ts_index = state.uniform_accel_ts_index
    var rayleigh_alpha_m = state.rayleigh_alpha_m
    var rayleigh_beta_k = state.rayleigh_beta_k
    var rayleigh_beta_k_init = state.rayleigh_beta_k_init
    var rayleigh_beta_k_comm = state.rayleigh_beta_k_comm
    ref recorder_nodes_pool = state.recorder_nodes_pool
    ref recorder_elements_pool = state.recorder_elements_pool
    ref recorder_dofs_pool = state.recorder_dofs_pool
    ref recorder_modes_pool = state.recorder_modes_pool
    ref recorder_sections_pool = state.recorder_sections_pool
    ref recorders = state.recorders
    ref stages = state.stages

    ref id_to_index = state.id_to_index
    ref typed_sections_by_id = state.typed_sections_by_id
    ref typed_materials_by_id = state.typed_materials_by_id
    ref uniaxial_defs = state.uniaxial_defs
    ref uniaxial_state_defs = state.uniaxial_state_defs
    ref uniaxial_states = state.uniaxial_states
    ref fiber_section_defs = state.fiber_section_defs
    ref fiber_section_cells = state.fiber_section_cells
    ref fiber_section_index_by_id = state.fiber_section_index_by_id
    ref fiber_section3d_defs = state.fiber_section3d_defs
    ref fiber_section3d_cells = state.fiber_section3d_cells
    ref fiber_section3d_index_by_id = state.fiber_section3d_index_by_id
    ref layered_shell_section_defs = state.layered_shell_section_defs
    ref layered_shell_section_index_by_id = state.layered_shell_section_index_by_id
    ref layered_shell_section_uniaxial_offsets = (
        state.layered_shell_section_uniaxial_offsets
    )
    ref layered_shell_section_uniaxial_counts = (
        state.layered_shell_section_uniaxial_counts
    )
    ref shell_elem_instance_offsets = state.shell_elem_instance_offsets
    ref elem_id_to_index = state.elem_id_to_index
    ref pattern_element_loads = state.element_loads
    ref node_x = state.node_x
    ref node_y = state.node_y
    ref node_z = state.node_z
    ref elem_dof_offsets = state.elem_dof_offsets
    ref elem_dof_pool = state.elem_dof_pool
    ref elem_free_offsets = state.elem_free_offsets
    ref elem_free_pool = state.elem_free_pool
    ref elem_node_offsets = state.elem_node_offsets
    ref elem_node_pool = state.elem_node_pool
    ref elem_primary_material_ids = state.elem_primary_material_ids
    ref elem_type_tags = state.elem_type_tags
    ref elem_geom_tags = state.elem_geom_tags
    ref elem_section_ids = state.elem_section_ids
    ref elem_integration_tags = state.elem_integration_tags
    ref elem_num_int_pts = state.elem_num_int_pts
    ref elem_area = state.elem_area
    ref elem_thickness = state.elem_thickness
    ref frame2d_elem_indices = state.frame2d_elem_indices
    ref frame3d_elem_indices = state.frame3d_elem_indices
    ref truss_elem_indices = state.truss_elem_indices
    ref zero_length_elem_indices = state.zero_length_elem_indices
    ref two_node_link_elem_indices = state.two_node_link_elem_indices
    ref zero_length_section_elem_indices = state.zero_length_section_elem_indices
    ref quad_elem_indices = state.quad_elem_indices
    ref shell_elem_indices = state.shell_elem_indices
    ref elem_uniaxial_offsets = state.elem_uniaxial_offsets
    ref elem_uniaxial_counts = state.elem_uniaxial_counts
    ref elem_uniaxial_state_ids = state.elem_uniaxial_state_ids
    ref force_basic_offsets = state.force_basic_offsets
    ref force_basic_counts = state.force_basic_counts
    ref force_basic_q = state.force_basic_q
    ref F_pattern = state.F_total
    ref constrained = state.constrained
    ref free = state.free
    ref free_index = state.free_index
    ref mpc_row_offsets = state.mpc_row_offsets
    ref mpc_dof_pool = state.mpc_dof_pool
    ref mpc_coeff_pool = state.mpc_coeff_pool
    ref mpc_slave_dof = state.mpc_slave_dof
    ref M_total = state.M_total
    ref M_rayleigh_total = state.M_rayleigh_total
    ref analysis_integrator_targets_pool = state.analysis_integrator_targets_pool
    ref analysis_solver_chain_pool = state.analysis_solver_chain_pool
    var F_const: List[Float64] = []
    F_const.resize(total_dofs, 0.0)
    var const_element_loads: List[ElementLoadInput] = []

    var u: List[Float64] = []
    u.resize(total_dofs, 0.0)
    var transient_output_files: List[String] = []
    var transient_output_buffers: List[String] = []
    var static_output_files: List[String] = []
    var static_output_buffers: List[String] = []

    var t_solve_start = Int(time.perf_counter_ns())
    var model_build_dof_map_us = (t_solve_start - t_start) // 1000
    if do_profile:
        var assemble_end = (t_solve_start - t0) // 1000
        _append_event(
            events,
            events_need_comma,
            "C",
            frame_model_build_dof_map,
            assemble_end,
        )
        _append_event(events, events_need_comma, "C", frame_assemble, assemble_end)
        _append_event(events, events_need_comma, "O", frame_solve, assemble_end)

    if analysis_type_tag == AnalysisTypeTag.StaticLinear:
        run_static_linear(
            typed_nodes,
            typed_elements,
            node_x,
            node_y,
            node_z,
            elem_dof_offsets,
            elem_dof_pool,
            elem_free_offsets,
            elem_free_pool,
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
            const_element_loads,
            pattern_element_loads,
            typed_sections_by_id,
            typed_materials_by_id,
            id_to_index,
            elem_id_to_index,
            node_count,
            ndf,
            ndm,
            u,
            F_const,
            F_pattern,
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
        layered_shell_section_index_by_id,
        layered_shell_section_uniaxial_offsets,
        layered_shell_section_uniaxial_counts,
        shell_elem_instance_offsets,
            analysis,
            free_index,
            free,
            ts_index,
            time_series,
            time_series_values,
            time_series_times,
            do_profile,
            t0,
            events,
            events_need_comma,
            frame_assemble_stiffness,
            frame_assemble_uniaxial,
            frame_assemble_fiber,
            frame_kff_extract,
            frame_solve_linear,
            total_dofs,
            has_transformation_mpc,
            mpc_slave_dof,
            mpc_row_offsets,
            mpc_dof_pool,
            mpc_coeff_pool,
            constrained,
            runtime_metrics,
        )
    elif analysis_type_tag == AnalysisTypeTag.StaticNonlinear:
        var integrator_type = analysis.integrator_type
        var integrator_tag = analysis.integrator_tag
        if integrator_tag == IntegratorTypeTag.Unknown:
            integrator_type = "LoadControl"
            integrator_tag = IntegratorTypeTag.LoadControl
        if integrator_tag == IntegratorTypeTag.LoadControl:
            run_static_nonlinear_load_control(
                analysis,
                steps,
                ts_index,
                time_series,
                time_series_values,
                time_series_times,
                analysis_solver_chain_pool,
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
                const_element_loads,
                pattern_element_loads,
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
        layered_shell_section_defs,
        layered_shell_section_index_by_id,
        layered_shell_section_uniaxial_offsets,
        layered_shell_section_uniaxial_counts,
        shell_elem_instance_offsets,
                total_dofs,
                F_const,
                F_pattern,
                free,
                free_index,
                recorders,
                recorder_nodes_pool,
                recorder_elements_pool,
                recorder_dofs_pool,
                recorder_sections_pool,
                elem_id_to_index,
                static_output_files,
                static_output_buffers,
                progress_path,
                1,
                1,
                do_profile,
                t0,
                events,
                events_need_comma,
                frame_assemble_stiffness,
                frame_assemble_uniaxial,
                frame_assemble_fiber,
                frame_kff_extract,
                frame_solve_nonlinear,
                frame_nonlinear_step,
                frame_nonlinear_iter,
                frame_uniaxial_revert_all,
                frame_uniaxial_commit_all,
                has_transformation_mpc,
                mpc_slave_dof,
                mpc_row_offsets,
                mpc_dof_pool,
                mpc_coeff_pool,
                constrained,
                runtime_metrics,
            )
        elif integrator_tag == IntegratorTypeTag.DisplacementControl:
            _ = run_static_nonlinear_displacement_control(
                analysis,
                steps,
                ts_index,
                time_series,
                analysis_integrator_targets_pool,
                analysis_solver_chain_pool,
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
                const_element_loads,
                pattern_element_loads,
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
        layered_shell_section_defs,
        layered_shell_section_index_by_id,
        layered_shell_section_uniaxial_offsets,
        layered_shell_section_uniaxial_counts,
        shell_elem_instance_offsets,
                total_dofs,
                F_const,
                F_pattern,
                constrained,
                free,
                recorders,
                recorder_nodes_pool,
                recorder_elements_pool,
                recorder_dofs_pool,
                recorder_sections_pool,
                elem_id_to_index,
                static_output_files,
                static_output_buffers,
                progress_path,
                1,
                1,
                do_profile,
                t0,
                events,
                events_need_comma,
                frame_assemble_stiffness,
                frame_assemble_uniaxial,
                frame_assemble_fiber,
                frame_kff_extract,
                frame_solve_nonlinear,
                frame_nonlinear_step,
                frame_nonlinear_iter,
                frame_uniaxial_revert_all,
                frame_uniaxial_commit_all,
                has_transformation_mpc,
                mpc_slave_dof,
                mpc_row_offsets,
                mpc_dof_pool,
                mpc_coeff_pool,
                runtime_metrics,
            )
        else:
            abort("unsupported static_nonlinear integrator: " + integrator_type)
    elif (
        analysis_type_tag == AnalysisTypeTag.TransientLinear
        or (
            analysis_type_tag == AnalysisTypeTag.TransientNonlinear
            and supports_linear_transient_fast_path
        )
    ):
        run_transient_linear(
            analysis,
            steps,
            ts_index,
            time_series,
            time_series_values,
            time_series_times,
            dampings,
            pattern_type,
            pattern_type_tag,
            uniform_excitation_direction,
            uniform_accel_ts_index,
            rayleigh_alpha_m,
            rayleigh_beta_k,
            rayleigh_beta_k_init,
            rayleigh_beta_k_comm,
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
            const_element_loads,
            pattern_element_loads,
            typed_sections_by_id,
            typed_materials_by_id,
            id_to_index,
            node_count,
            ndf,
            ndm,
            total_dofs,
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
            F_const,
            F_pattern,
            M_total,
            M_rayleigh_total,
            free,
            recorders,
            recorder_nodes_pool,
            recorder_elements_pool,
            recorder_dofs_pool,
            recorder_sections_pool,
            elem_id_to_index,
            fiber_section_defs,
            fiber_section_cells,
            fiber_section_index_by_id,
            fiber_section3d_defs,
            fiber_section3d_cells,
            fiber_section3d_index_by_id,
        layered_shell_section_defs,
        layered_shell_section_index_by_id,
        layered_shell_section_uniaxial_offsets,
        layered_shell_section_uniaxial_counts,
        shell_elem_instance_offsets,
            transient_output_files,
            transient_output_buffers,
            has_transformation_mpc,
            mpc_slave_dof,
            mpc_row_offsets,
            mpc_dof_pool,
            mpc_coeff_pool,
            constrained,
            do_profile,
            t0,
            events,
            events_need_comma,
            frame_assemble_stiffness,
            frame_solve_linear,
            frame_time_series_eval,
            frame_constraints,
            frame_recorders,
            frame_factorize,
            frame_transient_step,
            frame_uniaxial_commit_all,
            runtime_metrics,
        )
    elif analysis_type_tag == AnalysisTypeTag.TransientNonlinear:
        run_transient_nonlinear(
            analysis,
            steps,
            ts_index,
            time_series,
            time_series_values,
            time_series_times,
            analysis_solver_chain_pool,
            dampings,
            pattern_type,
            pattern_type_tag,
            uniform_excitation_direction,
            uniform_accel_ts_index,
            rayleigh_alpha_m,
            rayleigh_beta_k,
            rayleigh_beta_k_init,
            rayleigh_beta_k_comm,
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
            const_element_loads,
            pattern_element_loads,
            typed_sections_by_id,
            typed_materials_by_id,
            id_to_index,
            node_count,
            ndf,
            ndm,
            total_dofs,
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
            F_const,
            F_pattern,
            M_total,
            M_rayleigh_total,
            free,
            recorders,
            recorder_nodes_pool,
            recorder_elements_pool,
            recorder_dofs_pool,
            recorder_sections_pool,
            elem_id_to_index,
            fiber_section_defs,
            fiber_section_cells,
            fiber_section_index_by_id,
            fiber_section3d_defs,
            fiber_section3d_cells,
            fiber_section3d_index_by_id,
        layered_shell_section_defs,
        layered_shell_section_index_by_id,
        layered_shell_section_uniaxial_offsets,
        layered_shell_section_uniaxial_counts,
        shell_elem_instance_offsets,
            transient_output_files,
            transient_output_buffers,
            progress_path,
            1,
            1,
            has_transformation_mpc,
            mpc_slave_dof,
            mpc_row_offsets,
            mpc_dof_pool,
            mpc_coeff_pool,
            constrained,
            do_profile,
            t0,
            events,
            events_need_comma,
            frame_assemble_stiffness,
            frame_assemble_uniaxial,
            frame_assemble_fiber,
            frame_solve_nonlinear,
            frame_nonlinear_step,
            frame_nonlinear_iter,
            frame_time_series_eval,
            frame_constraints,
            frame_recorders,
            frame_factorize,
            frame_transient_step,
            frame_uniaxial_revert_all,
            frame_uniaxial_commit_all,
            runtime_metrics,
        )
    elif analysis_type_tag == AnalysisTypeTag.Staged:
        var stage_pattern_type = pattern_type
        var stage_pattern_type_tag = pattern_type_tag
        var stage_ts_index = ts_index
        var stage_uniform_excitation_direction = uniform_excitation_direction
        var stage_uniform_accel_ts_index = uniform_accel_ts_index
        var stage_rayleigh_alpha_m = rayleigh_alpha_m
        var stage_rayleigh_beta_k = rayleigh_beta_k
        var stage_rayleigh_beta_k_init = rayleigh_beta_k_init
        var stage_rayleigh_beta_k_comm = rayleigh_beta_k_comm
        var stage_F = F_pattern.copy()
        var stage_element_loads = pattern_element_loads.copy()

        for stage_idx in range(len(stages)):
            var stage = stages[stage_idx]
            var stage_analysis_targets_pool = (
                stage.analysis_integrator_targets_pool.copy()
            )
            var stage_analysis_solver_chain_pool = (
                stage.analysis_solver_chain_pool.copy()
            )
            var stage_analysis = stage.analysis
            var stage_type = stage_analysis.type
            var stage_type_tag = stage_analysis.type_tag
            if stage_analysis.steps < 1:
                abort("analysis steps must be >= 1")
            if (
                stage_analysis.constraints != "Plain"
                and stage_analysis.constraints != "Transformation"
                and stage_analysis.constraints != "Lagrange"
            ):
                abort(
                    "unsupported staged constraints handler: "
                    + stage_analysis.constraints
                )
            if stage_analysis.constraints != state.constraints_handler:
                abort(
                    "staged analysis requires a single constraints handler "
                    "for all stages"
                )

            if len(stage.time_series) > 0:
                _append_stage_time_series(
                    stage,
                    time_series,
                    time_series_values,
                    time_series_times,
                )

            var has_stage_loads = len(stage.loads) > 0
            var has_stage_element_loads = len(stage.element_loads) > 0

            if stage.pattern.has_pattern:
                var stage_pattern_type_next = stage.pattern.type
                var stage_pattern_type_next_tag = stage.pattern.type_tag
                if stage_pattern_type_next_tag == PatternTypeTag.Plain:
                    if not stage.pattern.has_time_series:
                        abort("pattern missing time_series")
                    var stage_ts_tag = stage.pattern.time_series
                    stage_ts_index = find_time_series_input(time_series, stage_ts_tag)
                    if stage_ts_index < 0:
                        abort("time_series tag not found")
                    if has_stage_loads:
                        stage_F = _build_stage_nodal_force_vector(
                            stage.loads,
                            id_to_index,
                            ndf,
                            total_dofs,
                            has_transformation_mpc,
                            mpc_row_offsets,
                            mpc_dof_pool,
                            mpc_coeff_pool,
                        )
                    elif stage_idx > 0:
                        stage_F = _zero_nodal_force_vector(total_dofs)
                    if has_stage_element_loads:
                        stage_element_loads = stage.element_loads.copy()
                    elif stage_idx > 0:
                        stage_element_loads = []
                    stage_pattern_type = "Plain"
                    stage_pattern_type_tag = PatternTypeTag.Plain
                    stage_uniform_excitation_direction = 0
                    stage_uniform_accel_ts_index = -1
                elif stage_pattern_type_next_tag == PatternTypeTag.UniformExcitation:
                    if has_stage_loads or has_stage_element_loads:
                        abort("UniformExcitation does not support nodal/element loads")
                    if not stage.pattern.has_direction:
                        abort("UniformExcitation pattern missing direction")
                    stage_uniform_excitation_direction = stage.pattern.direction
                    if (
                        stage_uniform_excitation_direction < 1
                        or stage_uniform_excitation_direction > ndm
                    ):
                        abort("UniformExcitation direction out of range 1..ndm")
                    var accel_tag = -1
                    if stage.pattern.has_accel:
                        accel_tag = stage.pattern.accel
                    elif stage.pattern.has_time_series:
                        accel_tag = stage.pattern.time_series
                    else:
                        abort("UniformExcitation pattern missing accel time_series tag")
                    stage_uniform_accel_ts_index = find_time_series_input(
                        time_series, accel_tag
                    )
                    if stage_uniform_accel_ts_index < 0:
                        abort("UniformExcitation accel time_series tag not found")
                    stage_F = _zero_nodal_force_vector(total_dofs)
                    stage_element_loads = []
                    stage_pattern_type = "UniformExcitation"
                    stage_pattern_type_tag = PatternTypeTag.UniformExcitation
                    stage_ts_index = -1
                elif stage_pattern_type_next_tag == PatternTypeTag.`None`:
                    if has_stage_loads or has_stage_element_loads:
                        abort("None pattern does not support nodal/element loads")
                    stage_F = _zero_nodal_force_vector(total_dofs)
                    stage_element_loads = []
                    stage_pattern_type = "Plain"
                    stage_pattern_type_tag = PatternTypeTag.Plain
                    stage_ts_index = -1
                    stage_uniform_excitation_direction = 0
                    stage_uniform_accel_ts_index = -1
                else:
                    abort("unsupported staged pattern type: " + stage_pattern_type_next)
            elif stage_idx > 0 and (has_stage_loads or has_stage_element_loads):
                abort("staged analysis stage loads require an explicit stage pattern")

            if stage.rayleigh.has_rayleigh:
                stage_rayleigh_alpha_m = stage.rayleigh.alpha_m
                stage_rayleigh_beta_k = stage.rayleigh.beta_k
                stage_rayleigh_beta_k_init = stage.rayleigh.beta_k_init
                stage_rayleigh_beta_k_comm = stage.rayleigh.beta_k_comm
            _write_run_progress(
                progress_path,
                "running_stage",
                stage_type,
                stage_idx + 1,
                len(stages),
                0,
                stage_analysis.steps,
            )
            var stage_final_pattern_scale = 0.0
            if stage_type_tag == AnalysisTypeTag.StaticLinear:
                if stage_pattern_type_tag != PatternTypeTag.Plain:
                    abort("staged static_linear only supports Plain pattern")
                run_static_linear(
                    typed_nodes,
                    typed_elements,
                    node_x,
                    node_y,
                    node_z,
                    elem_dof_offsets,
                    elem_dof_pool,
                    elem_free_offsets,
                    elem_free_pool,
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
                    const_element_loads,
                    stage_element_loads,
                    typed_sections_by_id,
                    typed_materials_by_id,
                    id_to_index,
                    elem_id_to_index,
                    node_count,
                    ndf,
                    ndm,
                    u,
                    F_const,
                    stage_F,
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
        layered_shell_section_index_by_id,
        layered_shell_section_uniaxial_offsets,
        layered_shell_section_uniaxial_counts,
        shell_elem_instance_offsets,
                    stage_analysis,
                    free_index,
                    free,
                    stage_ts_index,
                    time_series,
                    time_series_values,
                    time_series_times,
                    do_profile,
                    t0,
                    events,
                    events_need_comma,
                    frame_assemble_stiffness,
                    frame_assemble_uniaxial,
                    frame_assemble_fiber,
                    frame_kff_extract,
                    frame_solve_linear,
                    total_dofs,
                    has_transformation_mpc,
                    mpc_slave_dof,
                    mpc_row_offsets,
                    mpc_dof_pool,
                    mpc_coeff_pool,
                    constrained,
                    runtime_metrics,
                )
                stage_final_pattern_scale = 1.0
                if stage_ts_index >= 0:
                    stage_final_pattern_scale = eval_time_series_input(
                        time_series[stage_ts_index],
                        1.0,
                        time_series_values,
                        time_series_times,
                    )
            elif stage_type_tag == AnalysisTypeTag.StaticNonlinear:
                var stage_integrator_type = stage_analysis.integrator_type
                var stage_integrator_tag = stage_analysis.integrator_tag
                if stage_integrator_tag == IntegratorTypeTag.Unknown:
                    stage_integrator_type = "LoadControl"
                    stage_integrator_tag = IntegratorTypeTag.LoadControl
                if stage_integrator_tag == IntegratorTypeTag.LoadControl:
                    run_static_nonlinear_load_control(
                        stage_analysis,
                        stage_analysis.steps,
                        stage_ts_index,
                        time_series,
                        time_series_values,
                        time_series_times,
                        stage_analysis_solver_chain_pool,
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
                        const_element_loads,
                        stage_element_loads,
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
        layered_shell_section_defs,
        layered_shell_section_index_by_id,
        layered_shell_section_uniaxial_offsets,
        layered_shell_section_uniaxial_counts,
        shell_elem_instance_offsets,
                        total_dofs,
                        F_const,
                        stage_F,
                        free,
                        free_index,
                        recorders,
                        recorder_nodes_pool,
                        recorder_elements_pool,
                        recorder_dofs_pool,
                        recorder_sections_pool,
                        elem_id_to_index,
                        transient_output_files,
                        transient_output_buffers,
                        progress_path,
                        stage_idx + 1,
                        len(stages),
                        do_profile,
                        t0,
                        events,
                        events_need_comma,
                        frame_assemble_stiffness,
                        frame_assemble_uniaxial,
                        frame_assemble_fiber,
                        frame_kff_extract,
                        frame_solve_nonlinear,
                        frame_nonlinear_step,
                        frame_nonlinear_iter,
                        frame_uniaxial_revert_all,
                        frame_uniaxial_commit_all,
                        has_transformation_mpc,
                        mpc_slave_dof,
                        mpc_row_offsets,
                        mpc_dof_pool,
                        mpc_coeff_pool,
                        constrained,
                        runtime_metrics,
                    )
                    stage_final_pattern_scale = 1.0
                    if stage_ts_index >= 0:
                        stage_final_pattern_scale = eval_time_series_input(
                            time_series[stage_ts_index],
                            1.0,
                            time_series_values,
                            time_series_times,
                        )
                elif stage_integrator_tag == IntegratorTypeTag.DisplacementControl:
                    var stage_disp_steps = stage_analysis.steps
                    stage_final_pattern_scale = run_static_nonlinear_displacement_control(
                        stage_analysis,
                        stage_disp_steps,
                        stage_ts_index,
                        time_series,
                        stage_analysis_targets_pool,
                        stage_analysis_solver_chain_pool,
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
                        const_element_loads,
                        stage_element_loads,
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
        layered_shell_section_defs,
        layered_shell_section_index_by_id,
        layered_shell_section_uniaxial_offsets,
        layered_shell_section_uniaxial_counts,
        shell_elem_instance_offsets,
                        total_dofs,
                        F_const,
                        stage_F,
                        constrained,
                        free,
                        recorders,
                        recorder_nodes_pool,
                        recorder_elements_pool,
                        recorder_dofs_pool,
                        recorder_sections_pool,
                        elem_id_to_index,
                        transient_output_files,
                        transient_output_buffers,
                        progress_path,
                        stage_idx + 1,
                        len(stages),
                        do_profile,
                        t0,
                        events,
                        events_need_comma,
                        frame_assemble_stiffness,
                        frame_assemble_uniaxial,
                        frame_assemble_fiber,
                        frame_kff_extract,
                        frame_solve_nonlinear,
                        frame_nonlinear_step,
                        frame_nonlinear_iter,
                        frame_uniaxial_revert_all,
                        frame_uniaxial_commit_all,
                        has_transformation_mpc,
                        mpc_slave_dof,
                        mpc_row_offsets,
                        mpc_dof_pool,
                        mpc_coeff_pool,
                        runtime_metrics,
                    )
                else:
                    abort(
                        "unsupported static_nonlinear integrator: "
                        + stage_integrator_type
                    )
            elif (
                stage_type_tag == AnalysisTypeTag.TransientLinear
                or (
                    stage_type_tag == AnalysisTypeTag.TransientNonlinear
                    and supports_linear_transient_fast_path
                )
            ):
                run_transient_linear(
                    stage_analysis,
                    stage_analysis.steps,
                    stage_ts_index,
                    time_series,
                    time_series_values,
                    time_series_times,
                    dampings,
                    stage_pattern_type,
                    stage_pattern_type_tag,
                    stage_uniform_excitation_direction,
                    stage_uniform_accel_ts_index,
                    stage_rayleigh_alpha_m,
                    stage_rayleigh_beta_k,
                    stage_rayleigh_beta_k_init,
                    stage_rayleigh_beta_k_comm,
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
                    const_element_loads,
                    stage_element_loads,
                    typed_sections_by_id,
                    typed_materials_by_id,
                    id_to_index,
                    node_count,
                    ndf,
                    ndm,
                    total_dofs,
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
                    F_const,
                    stage_F,
                    M_total,
                    M_rayleigh_total,
                    free,
                    recorders,
                    recorder_nodes_pool,
                    recorder_elements_pool,
                    recorder_dofs_pool,
                    recorder_sections_pool,
                    elem_id_to_index,
                    fiber_section_defs,
                    fiber_section_cells,
                    fiber_section_index_by_id,
                    fiber_section3d_defs,
                    fiber_section3d_cells,
                    fiber_section3d_index_by_id,
        layered_shell_section_defs,
        layered_shell_section_index_by_id,
        layered_shell_section_uniaxial_offsets,
        layered_shell_section_uniaxial_counts,
        shell_elem_instance_offsets,
                    transient_output_files,
                    transient_output_buffers,
                    has_transformation_mpc,
                    mpc_slave_dof,
                    mpc_row_offsets,
                    mpc_dof_pool,
                    mpc_coeff_pool,
                    constrained,
                    do_profile,
                    t0,
                    events,
                    events_need_comma,
                    frame_assemble_stiffness,
                    frame_solve_linear,
                    frame_time_series_eval,
                    frame_constraints,
                    frame_recorders,
                    frame_factorize,
                    frame_transient_step,
                    frame_uniaxial_commit_all,
                    runtime_metrics,
                )
                if stage_pattern_type_tag == PatternTypeTag.Plain:
                    stage_final_pattern_scale = 1.0
                    if stage_ts_index >= 0:
                        stage_final_pattern_scale = eval_time_series_input(
                            time_series[stage_ts_index],
                            Float64(stage_analysis.steps) * stage_analysis.dt,
                            time_series_values,
                            time_series_times,
                        )
            elif stage_type_tag == AnalysisTypeTag.TransientNonlinear:
                run_transient_nonlinear(
                    stage_analysis,
                    stage_analysis.steps,
                    stage_ts_index,
                    time_series,
                    time_series_values,
                    time_series_times,
                    stage_analysis_solver_chain_pool,
                    dampings,
                    stage_pattern_type,
                    stage_pattern_type_tag,
                    stage_uniform_excitation_direction,
                    stage_uniform_accel_ts_index,
                    stage_rayleigh_alpha_m,
                    stage_rayleigh_beta_k,
                    stage_rayleigh_beta_k_init,
                    stage_rayleigh_beta_k_comm,
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
                    const_element_loads,
                    stage_element_loads,
                    typed_sections_by_id,
                    typed_materials_by_id,
                    id_to_index,
                    node_count,
                    ndf,
                    ndm,
                    total_dofs,
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
                    F_const,
                    stage_F,
                    M_total,
                    M_rayleigh_total,
                    free,
                    recorders,
                    recorder_nodes_pool,
                    recorder_elements_pool,
                    recorder_dofs_pool,
                    recorder_sections_pool,
                    elem_id_to_index,
                    fiber_section_defs,
                    fiber_section_cells,
                    fiber_section_index_by_id,
                    fiber_section3d_defs,
                    fiber_section3d_cells,
                    fiber_section3d_index_by_id,
        layered_shell_section_defs,
        layered_shell_section_index_by_id,
        layered_shell_section_uniaxial_offsets,
        layered_shell_section_uniaxial_counts,
        shell_elem_instance_offsets,
                    transient_output_files,
                    transient_output_buffers,
                    progress_path,
                    stage_idx + 1,
                    len(stages),
                    has_transformation_mpc,
                    mpc_slave_dof,
                    mpc_row_offsets,
                    mpc_dof_pool,
                    mpc_coeff_pool,
                    constrained,
                    do_profile,
                    t0,
                    events,
                    events_need_comma,
                    frame_assemble_stiffness,
                    frame_assemble_uniaxial,
                    frame_assemble_fiber,
                    frame_solve_nonlinear,
                    frame_nonlinear_step,
                    frame_nonlinear_iter,
                    frame_time_series_eval,
                    frame_constraints,
                    frame_recorders,
                    frame_factorize,
                    frame_transient_step,
                    frame_uniaxial_revert_all,
                    frame_uniaxial_commit_all,
                    runtime_metrics,
                )
                if stage_pattern_type_tag == PatternTypeTag.Plain:
                    stage_final_pattern_scale = 1.0
                    if stage_ts_index >= 0:
                        stage_final_pattern_scale = eval_time_series_input(
                            time_series[stage_ts_index],
                            Float64(stage_analysis.steps) * stage_analysis.dt,
                            time_series_values,
                            time_series_times,
                        )
            elif stage_type_tag == AnalysisTypeTag.ModalEigen:
                if stage_analysis.num_modes < 1:
                    abort("modal_eigen requires num_modes >= 1")
                run_modal_eigen(
                    stage_analysis.num_modes,
                    typed_nodes,
                    typed_elements,
                    typed_sections_by_id,
                    typed_materials_by_id,
                    id_to_index,
                    node_count,
                    ndf,
                    ndm,
                    total_dofs,
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
        layered_shell_section_index_by_id,
        layered_shell_section_uniaxial_offsets,
        layered_shell_section_uniaxial_counts,
        shell_elem_instance_offsets,
                    M_total,
                    constrained,
                    free,
                    has_transformation_mpc,
                    mpc_slave_dof,
                    mpc_row_offsets,
                    mpc_dof_pool,
                    mpc_coeff_pool,
                    recorders,
                    recorder_nodes_pool,
                    recorder_dofs_pool,
                    recorder_modes_pool,
                    transient_output_files,
                    transient_output_buffers,
                )
            else:
                abort("unsupported staged analysis type: " + stage_type)

            if stage.has_load_const:
                if stage_pattern_type_tag == PatternTypeTag.Plain:
                    F_const = build_active_nodal_load(
                        F_const, stage_F, stage_final_pattern_scale
                    )
                    append_scaled_element_loads(
                        const_element_loads,
                        stage_element_loads,
                        stage_final_pattern_scale,
                    )
                stage_F = _zero_nodal_force_vector(total_dofs)
                stage_element_loads = []
                stage_pattern_type = "Plain"
                stage_pattern_type_tag = PatternTypeTag.Plain
                stage_ts_index = -1
                stage_uniform_excitation_direction = 0
                stage_uniform_accel_ts_index = -1
    elif analysis_type_tag == AnalysisTypeTag.ModalEigen:
        run_modal_eigen(
            modal_num_modes,
            typed_nodes,
            typed_elements,
            typed_sections_by_id,
            typed_materials_by_id,
            id_to_index,
            node_count,
            ndf,
            ndm,
            total_dofs,
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
        layered_shell_section_index_by_id,
        layered_shell_section_uniaxial_offsets,
        layered_shell_section_uniaxial_counts,
        shell_elem_instance_offsets,
            M_total,
            constrained,
            free,
            has_transformation_mpc,
            mpc_slave_dof,
            mpc_row_offsets,
            mpc_dof_pool,
            mpc_coeff_pool,
            recorders,
            recorder_nodes_pool,
            recorder_dofs_pool,
            recorder_modes_pool,
            static_output_files,
            static_output_buffers,
        )
    else:
        abort("unsupported analysis type: " + analysis_type)
    var t_solve_end = Int(time.perf_counter_ns())
    if do_profile:
        var solve_end_us = (t_solve_end - t0) // 1000
        _append_event(events, events_need_comma, "C", frame_solve, solve_end_us)
        _append_event(events, events_need_comma, "O", frame_output, solve_end_us)

    var analysis_us = (t_solve_end - t_solve_start) // 1000

    _write_run_progress(progress_path, "writing_output", analysis_type, 0, 0, 0, 0)
    var analysis_path = out_dir.joinpath("analysis_time_us.txt")
    analysis_path.write_text(PythonObject(String(analysis_us) + "\n"))
    if (
        analysis_type_tag == AnalysisTypeTag.TransientLinear
        or analysis_type_tag == AnalysisTypeTag.TransientNonlinear
        or (
            analysis_type_tag == AnalysisTypeTag.Staged
            and len(transient_output_files) > 0
        )
    ):
        _write_output_chunk_files(
            out_dir, transient_output_files, transient_output_buffers
        )
    elif analysis_type_tag == AnalysisTypeTag.ModalEigen:
        _write_output_chunk_files(out_dir, static_output_files, static_output_buffers)
    elif (
        analysis_type_tag == AnalysisTypeTag.StaticNonlinear
        and len(static_output_files) > 0
    ):
        _write_output_chunk_files(out_dir, static_output_files, static_output_buffers)
    else:
        if has_transformation_mpc:
            _enforce_mpc_values(
                u,
                constrained,
                mpc_slave_dof,
                mpc_row_offsets,
                mpc_dof_pool,
                mpc_coeff_pool,
            )
        var has_reaction_recorder = _has_recorder_type(recorders, RecorderTypeTag.NodeReaction)
        var final_load_scale = 1.0
        if analysis_type_tag == AnalysisTypeTag.StaticLinear and ts_index >= 0:
            final_load_scale = eval_time_series_input(
                time_series[ts_index], 1.0, time_series_values, time_series_times
            )
        var final_F = build_active_nodal_load(F_const, F_pattern, final_load_scale)
        var final_element_load_state = build_active_element_load_state(
            const_element_loads,
            pattern_element_loads,
            final_load_scale,
            typed_elements,
            elem_id_to_index,
            ndm,
            ndf,
        )
        var F_int_reaction: List[Float64] = []
        var F_ext_reaction: List[Float64] = []
        if has_reaction_recorder:
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
                final_element_load_state.element_loads,
                final_element_load_state.elem_load_offsets,
                final_element_load_state.elem_load_pool,
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
        layered_shell_section_defs,
        layered_shell_section_index_by_id,
        layered_shell_section_uniaxial_offsets,
        layered_shell_section_uniaxial_counts,
        shell_elem_instance_offsets,
            )
            F_ext_reaction = final_F.copy()
        var envelope_files: List[String] = []
        var envelope_min: List[List[Float64]] = []
        var envelope_max: List[List[Float64]] = []
        var envelope_abs: List[List[Float64]] = []
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
                            value = 0.0
                            if rec.time_series_tag >= 0:
                                var ts_index = find_time_series_input(
                                    time_series, rec.time_series_tag
                                )
                                if ts_index < 0:
                                    abort("recorder time series not found")
                                value += eval_time_series_input(
                                    time_series[ts_index],
                                    0.0,
                                    time_series_values,
                                    time_series_times,
                                )
                        values[j] = value
                    var filename = rec.output + "_node" + String(node_id) + ".out"
                    if rec.type_tag == RecorderTypeTag.NodeDisplacement:
                        var file_path = out_dir.joinpath(filename)
                        file_path.write_text(PythonObject(_format_values_line(values)))
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
                    var elem = typed_elements[elem_index]
                    var f_elem = _element_force_global_for_recorder(
                        elem_index,
                        elem,
                        ndf,
                        u,
                        final_element_load_state.element_loads,
                        final_element_load_state.elem_load_offsets,
                        final_element_load_state.elem_load_pool,
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
                    var line = _format_values_line(f_elem)
                    var filename = rec.output + "_ele" + String(elem_id) + ".out"
                    var file_path = out_dir.joinpath(filename)
                    file_path.write_text(PythonObject(line))
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
                    var file_path = out_dir.joinpath(filename)
                    file_path.write_text(PythonObject(_format_values_line(values)))
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
                    var file_path = out_dir.joinpath(filename)
                    file_path.write_text(PythonObject(_format_values_line(values)))
            elif rec.type_tag == RecorderTypeTag.ElementDeformation:
                for eidx in range(rec.element_count):
                    var elem_id = recorder_elements_pool[rec.element_offset + eidx]
                    if elem_id >= len(elem_id_to_index) or elem_id_to_index[elem_id] < 0:
                        abort("recorder element not found")
                    var elem_index = elem_id_to_index[elem_id]
                    var elem = typed_elements[elem_index]
                    var values = _element_deformation_for_recorder(
                        elem_index,
                        elem,
                        ndf,
                        u,
                        typed_nodes,
                    )
                    var filename = rec.output + "_ele" + String(elem_id) + ".out"
                    var file_path = out_dir.joinpath(filename)
                    file_path.write_text(PythonObject(_format_values_line(values)))
            elif rec.type_tag == RecorderTypeTag.NodeReaction:
                for nidx in range(rec.node_count):
                    var node_id = recorder_nodes_pool[rec.node_offset + nidx]
                    var i = id_to_index[node_id]
                    var values: List[Float64] = []
                    values.resize(rec.dof_count, 0.0)
                    for j in range(rec.dof_count):
                        var dof = recorder_dofs_pool[rec.dof_offset + j]
                        require_dof_in_range(dof, ndf, "recorder")
                        var idx = node_dof_index(i, dof, ndf)
                        values[j] = F_int_reaction[idx] - F_ext_reaction[idx]
                    var filename = rec.output + "_node" + String(node_id) + ".out"
                    var file_path = out_dir.joinpath(filename)
                    file_path.write_text(PythonObject(_format_values_line(values)))
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
                var file_path = out_dir.joinpath(filename)
                file_path.write_text(PythonObject(_format_values_line([value])))
            elif rec.type_tag == RecorderTypeTag.EnvelopeElementForce:
                for eidx in range(rec.element_count):
                    var elem_id = recorder_elements_pool[rec.element_offset + eidx]
                    if elem_id >= len(elem_id_to_index) or elem_id_to_index[elem_id] < 0:
                        abort("recorder element not found")
                    var elem_index = elem_id_to_index[elem_id]
                    var elem = typed_elements[elem_index]
                    var f_elem = _element_force_global_for_recorder(
                        elem_index,
                        elem,
                        ndf,
                        u,
                        final_element_load_state.element_loads,
                        final_element_load_state.elem_load_offsets,
                        final_element_load_state.elem_load_pool,
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
                    var filename = rec.output + "_ele" + String(elem_id) + ".out"
                    _update_envelope(
                        filename,
                        f_elem,
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
                            final_element_load_state.element_loads,
                            final_element_load_state.elem_load_offsets,
                            final_element_load_state.elem_load_pool,
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
                        var file_path = out_dir.joinpath(filename)
                        file_path.write_text(PythonObject(_format_values_line(values)))
            else:
                abort("unsupported recorder type")
        for i in range(len(envelope_files)):
            var line = String()
            line += _format_values_line(envelope_min[i])
            line += _format_values_line(envelope_max[i])
            line += _format_values_line(envelope_abs[i])
            var file_path = out_dir.joinpath(envelope_files[i])
            file_path.write_text(PythonObject(line))

    var t2 = Int(time.perf_counter_ns())
    var total_us = (t2 - t0) // 1000
    var output_write_us = (t2 - t_solve_end) // 1000
    var predictor_section_eval_us = runtime_metrics.predictor_section_eval_ns // 1000
    var corrector_section_eval_us = runtime_metrics.corrector_section_eval_ns // 1000
    var local_flexibility_accumulation_us = (
        runtime_metrics.local_flexibility_accumulation_ns // 1000
    )
    var local_3x3_solve_us = runtime_metrics.local_3x3_solve_ns // 1000
    var local_commit_revert_us = runtime_metrics.local_commit_revert_ns // 1000
    var element_state_update_ns = (
        runtime_metrics.predictor_section_eval_ns
        + runtime_metrics.corrector_section_eval_ns
        + runtime_metrics.local_flexibility_accumulation_ns
        + runtime_metrics.local_3x3_solve_ns
        + runtime_metrics.local_commit_revert_ns
    )
    var element_state_update_us = element_state_update_ns // 1000
    var phase_json = String()
    phase_json += "{\n"
    phase_json += "  \"case_load_parse_us\": " + String(safe_case_load_us) + ",\n"
    phase_json += (
        "  \"model_build_dof_map_us\": " + String(model_build_dof_map_us) + ",\n"
    )
    phase_json += "  \"analysis_us\": " + String(analysis_us) + ",\n"
    phase_json += (
        "  \"element_state_update_us\": " + String(element_state_update_us) + ",\n"
    )
    phase_json += (
        "  \"predictor_section_eval_us\": "
        + String(predictor_section_eval_us)
        + ",\n"
    )
    phase_json += (
        "  \"corrector_section_eval_us\": "
        + String(corrector_section_eval_us)
        + ",\n"
    )
    phase_json += (
        "  \"local_flexibility_accumulation_us\": "
        + String(local_flexibility_accumulation_us)
        + ",\n"
    )
    phase_json += (
        "  \"local_3x3_solve_us\": "
        + String(local_3x3_solve_us)
        + ",\n"
    )
    phase_json += (
        "  \"local_commit_revert_us\": "
        + String(local_commit_revert_us)
        + ",\n"
    )
    phase_json += "  \"solve_total_us\": " + String(analysis_us) + ",\n"
    phase_json += (
        "  \"global_nonlinear_iterations\": "
        + String(runtime_metrics.global_nonlinear_iterations)
        + ",\n"
    )
    phase_json += (
        "  \"local_force_beam_column_iterations\": "
        + String(runtime_metrics.local_force_beam_column_iterations)
        + ",\n"
    )
    phase_json += (
        "  \"subdivision_fallback_iterations\": "
        + String(runtime_metrics.subdivision_fallback_iterations)
        + ",\n"
    )
    phase_json += (
        "  \"tangent_factorizations\": "
        + String(runtime_metrics.tangent_factorizations)
        + ",\n"
    )
    phase_json += (
        "  \"section_evaluations\": "
        + String(runtime_metrics.section_evaluations)
        + ",\n"
    )
    phase_json += (
        "  \"active_bandwidth\": " + String(runtime_metrics.active_bandwidth) + ",\n"
    )
    phase_json += "  \"active_nnz\": " + String(runtime_metrics.active_nnz) + ",\n"
    phase_json += (
        "  \"active_profile_size\": "
        + String(runtime_metrics.active_profile_size)
        + ",\n"
    )
    phase_json += "  \"output_write_us\": " + String(output_write_us) + ",\n"
    phase_json += "  \"total_case_us\": " + String(total_us) + ",\n"
    phase_json += "  \"element_type_timing_us\": {\n"
    var need_elem_comma = False
    for element_type in range(len(runtime_metrics.element_type_total_ns)):
        var total_type_us = runtime_metrics.element_type_total_ns[element_type] // 1000
        var type_name = _profile_metrics_element_type_name(element_type)
        if total_type_us <= 0 or type_name == "":
            continue
        if need_elem_comma:
            phase_json += ",\n"
        phase_json += (
            "    \""
            + type_name
            + "\": "
            + String(total_type_us)
        )
        need_elem_comma = True
    phase_json += "\n  },\n"
    phase_json += "  \"element_type_call_counts\": {\n"
    need_elem_comma = False
    for element_type in range(len(runtime_metrics.element_type_call_counts)):
        var total_calls = runtime_metrics.element_type_call_counts[element_type]
        var type_name = _profile_metrics_element_type_name(element_type)
        if total_calls <= 0 or type_name == "":
            continue
        if need_elem_comma:
            phase_json += ",\n"
        phase_json += (
            "    \""
            + type_name
            + "\": "
            + String(total_calls)
        )
        need_elem_comma = True
    phase_json += "\n  }\n"
    phase_json += "}\n"
    var phase_path = out_dir.joinpath("phase_times_us.json")
    phase_path.write_text(PythonObject(phase_json))
    _write_run_progress(progress_path, "completed", analysis_type, 0, 0, 0, 0)

    if do_profile:
        _append_event(events, events_need_comma, "C", frame_output, total_us)
        _append_event(events, events_need_comma, "C", frame_total, total_us)
        _write_speedscope(profile_path, frames, events, total_us)


def run_case_from_native_source(
    doc: JsonDocument,
    source_info: CaseSourceInfo,
    output_path: String,
    profile_path: String,
    case_load_us: Int,
    include_recorders: Bool,
):
    precheck_case_input_native(doc, include_recorders)
    run_case_input(
        parse_case_input_native_from_source(doc, source_info, include_recorders),
        output_path,
        profile_path,
        case_load_us,
    )
