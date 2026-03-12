from collections import List
from materials import UniMaterialState
from os import abort
from python import Python
from solver.run_case.loader import RunCaseState, load_case_state_from_input
from solver.run_case.input_types import parse_case_input
from solver.simd_contiguous import FLOAT64_SIMD_WIDTH
from sections import (
    FiberSection2dDef,
    fiber_section2d_commit_from_offset,
    fiber_section2d_runtime_alloc_instances,
    fiber_section2d_set_trial_from_offset,
)
from sections.fiber2d import (
    _fiber_section2d_runtime_apply_concrete02,
    _fiber_section2d_runtime_apply_steel02,
    _fiber_section2d_apply_concrete02_range,
    _fiber_section2d_apply_steel02_range,
)
from strut_io import load_json
from sys.arg import argv


alias SCENARIO_COUNT: Int = 3


struct BenchScenario(Movable):
    var name: String
    var target_eps0: Float64
    var target_kappa: Float64

    fn __init__(out self, name: String, target_eps0: Float64, target_kappa: Float64):
        self.name = name
        self.target_eps0 = target_eps0
        self.target_kappa = target_kappa


struct BenchArgs(Movable):
    var input_path: String
    var section_id: Int
    var target_fibers: Int
    var iterations: Int
    var samples: Int

    fn __init__(
        out self,
        input_path: String,
        section_id: Int,
        target_fibers: Int,
        iterations: Int,
        samples: Int,
    ):
        self.input_path = input_path
        self.section_id = section_id
        self.target_fibers = target_fibers
        self.iterations = iterations
        self.samples = samples


fn _arg_value(
    args: VariadicList[StringSlice[StaticConstantOrigin]], name: String, default_value: String
) -> String:
    for i in range(len(args) - 1):
        if String(args[i]) == name:
            return String(args[i + 1])
    return default_value


fn _parse_args() raises -> BenchArgs:
    var args = argv()
    var input_path = _arg_value(args, "--input", "")
    var section_id = -1
    var target_fibers = 250000
    var iterations = 32
    var samples = 5
    var section_id_raw = _arg_value(args, "--section-id", "")
    if section_id_raw != "":
        section_id = Int(section_id_raw)
    var target_fibers_raw = _arg_value(args, "--target-fibers", "")
    if target_fibers_raw != "":
        target_fibers = Int(target_fibers_raw)
    var iterations_raw = _arg_value(args, "--iterations", "")
    if iterations_raw != "":
        iterations = Int(iterations_raw)
    var samples_raw = _arg_value(args, "--samples", "")
    if samples_raw != "":
        samples = Int(samples_raw)
    if input_path == "":
        abort("missing --input")
    if target_fibers <= 0:
        abort("--target-fibers must be > 0")
    if iterations <= 0:
        abort("--iterations must be > 0")
    if samples <= 0:
        abort("--samples must be > 0")
    return BenchArgs(input_path, section_id, target_fibers, iterations, samples)


fn _scenario_definition(index: Int) -> BenchScenario:
    if index == 0:
        return BenchScenario("high_push", -0.012, 1.2e-3)
    if index == 1:
        return BenchScenario("post_crushing_rebound", 0.0035, -7.5e-4)
    if index == 2:
        return BenchScenario("post_12x_reversal", -0.008, 1.4e-3)
    abort("scenario index out of range")
    return BenchScenario("", 0.0, 0.0)


fn _apply_preload_history(
    mut sec_def: FiberSection2dDef,
    mut uniaxial_states: List[UniMaterialState],
    base_offset: Int,
    instance_count: Int,
    scenario_index: Int,
):
    @always_inline
    fn apply_step(
        mut sec_def: FiberSection2dDef,
        mut uniaxial_states: List[UniMaterialState],
        base_offset: Int,
        instance_count: Int,
        eps0: Float64,
        kappa: Float64,
    ):
        for inst in range(instance_count):
            var offset = base_offset + inst * sec_def.fiber_count
            _ = fiber_section2d_set_trial_from_offset(
                sec_def,
                uniaxial_states,
                offset,
                sec_def.fiber_count,
                eps0,
                kappa,
            )
            fiber_section2d_commit_from_offset(
                sec_def,
                uniaxial_states,
                offset,
                sec_def.fiber_count,
            )

    if scenario_index == 0:
        apply_step(sec_def, uniaxial_states, base_offset, instance_count, -2.0e-3, 2.0e-4)
        apply_step(sec_def, uniaxial_states, base_offset, instance_count, -5.0e-3, 5.0e-4)
        apply_step(sec_def, uniaxial_states, base_offset, instance_count, -8.0e-3, 8.0e-4)
        return
    if scenario_index == 1:
        apply_step(sec_def, uniaxial_states, base_offset, instance_count, -2.5e-3, 2.5e-4)
        apply_step(sec_def, uniaxial_states, base_offset, instance_count, -6.0e-3, 6.0e-4)
        apply_step(sec_def, uniaxial_states, base_offset, instance_count, -1.0e-2, 1.0e-3)
        apply_step(sec_def, uniaxial_states, base_offset, instance_count, -6.0e-3, 6.0e-4)
        apply_step(sec_def, uniaxial_states, base_offset, instance_count, -2.0e-3, 2.0e-4)
        apply_step(sec_def, uniaxial_states, base_offset, instance_count, 1.2e-3, -3.5e-4)
        return
    for cycle in range(12):
        var scale = Float64(cycle + 1) / 12.0
        apply_step(
            sec_def,
            uniaxial_states,
            base_offset,
            instance_count,
            -1.2e-2 * scale,
            1.0e-3 * scale,
        )
        apply_step(
            sec_def,
            uniaxial_states,
            base_offset,
            instance_count,
            5.0e-3 * scale,
            -8.0e-4 * scale,
        )


fn _time_ns() raises -> Int:
    var time = Python.import_module("time")
    return Int(time.perf_counter_ns())


fn _print_sample(
    section_id: Int,
    section_index: Int,
    section_name: String,
    fiber_count: Int,
    elastic_count: Int,
    steel02_count: Int,
    steel02_padded_count: Int,
    concrete02_count: Int,
    concrete02_padded_count: Int,
    instance_count: Int,
    scenario_name: String,
    benchmark: String,
    sample_index: Int,
    iterations: Int,
    total_updates: Int,
    elapsed_ns: Int,
):
    print(
        String(section_id)
        + ","
        + String(section_index)
        + ","
        + section_name
        + ","
        + String(fiber_count)
        + ","
        + String(elastic_count)
        + ","
        + String(steel02_count)
        + ","
        + String(steel02_padded_count)
        + ","
        + String(concrete02_count)
        + ","
        + String(concrete02_padded_count)
        + ","
        + String(instance_count)
        + ","
        + scenario_name
        + ","
        + benchmark
        + ","
        + String(sample_index)
        + ","
        + String(iterations)
        + ","
        + String(total_updates)
        + ","
        + String(elapsed_ns)
    )


fn _run_full_section_bench(
    section_id: Int,
    section_index: Int,
    section_name: String,
    sec_source: FiberSection2dDef,
    state: RunCaseState,
    instance_count: Int,
    iterations: Int,
    samples: Int,
    scenario_index: Int,
) raises:
    var scenario = _scenario_definition(scenario_index)
    var sec_def = sec_source
    var uniaxial_states = state.uniaxial_states.copy()
    var base_offset = fiber_section2d_runtime_alloc_instances(sec_def, instance_count)
    _apply_preload_history(sec_def, uniaxial_states, base_offset, instance_count, scenario_index)
    var total_updates = instance_count * iterations * sec_def.fiber_count
    for sample in range(samples):
        var started_ns = _time_ns()
        for _ in range(iterations):
            for inst in range(instance_count):
                var offset = base_offset + inst * sec_def.fiber_count
                _ = fiber_section2d_set_trial_from_offset(
                    sec_def,
                    uniaxial_states,
                    offset,
                    sec_def.fiber_count,
                    scenario.target_eps0,
                    scenario.target_kappa,
                )
        var elapsed_ns = _time_ns() - started_ns
        _print_sample(
            section_id,
            section_index,
            section_name,
            sec_def.fiber_count,
            sec_def.elastic_count,
            sec_def.steel02_count,
            sec_def.steel02_padded_count,
            sec_def.concrete02_count,
            sec_def.concrete02_padded_count,
            instance_count,
            scenario.name,
            "full_section",
            sample,
            iterations,
            total_updates,
            elapsed_ns,
        )


fn _run_family_bench(
    section_id: Int,
    section_index: Int,
    section_name: String,
    sec_source: FiberSection2dDef,
    state: RunCaseState,
    instance_count: Int,
    iterations: Int,
    samples: Int,
    scenario_index: Int,
    is_steel: Bool,
) raises:
    var scenario = _scenario_definition(scenario_index)
    var sec_def = sec_source
    var uniaxial_states = state.uniaxial_states.copy()
    var base_offset = fiber_section2d_runtime_alloc_instances(sec_def, instance_count)
    _apply_preload_history(sec_def, uniaxial_states, base_offset, instance_count, scenario_index)
    var family_count = sec_def.concrete02_count
    var benchmark_name = String("family_concrete02")
    if is_steel:
        family_count = sec_def.steel02_count
        benchmark_name = "family_steel02"
    if family_count <= 0:
        return
    var total_updates = instance_count * iterations * family_count
    for sample in range(samples):
        var started_ns = _time_ns()
        for _ in range(iterations):
            for inst in range(instance_count):
                var local_offset = inst * sec_def.fiber_count
                var axial_force = 0.0
                var moment_z = 0.0
                var k11 = 0.0
                var k12 = 0.0
                var k22 = 0.0
                if is_steel:
                    _fiber_section2d_apply_steel02_range[FLOAT64_SIMD_WIDTH](
                        sec_def,
                        local_offset,
                        scenario.target_eps0,
                        scenario.target_kappa,
                        axial_force,
                        moment_z,
                        k11,
                        k12,
                        k22,
                    )
                else:
                    _fiber_section2d_apply_concrete02_range[FLOAT64_SIMD_WIDTH](
                        sec_def,
                        local_offset,
                        scenario.target_eps0,
                        scenario.target_kappa,
                        axial_force,
                        moment_z,
                        k11,
                        k12,
                        k22,
                    )
        var elapsed_ns = _time_ns() - started_ns
        _print_sample(
            section_id,
            section_index,
            section_name,
            sec_def.fiber_count,
            sec_def.elastic_count,
            sec_def.steel02_count,
            sec_def.steel02_padded_count,
            sec_def.concrete02_count,
            sec_def.concrete02_padded_count,
            instance_count,
            scenario.name,
            benchmark_name,
            sample,
            iterations,
            total_updates,
            elapsed_ns,
        )


fn _run_material_scalar_bench(
    section_id: Int,
    section_index: Int,
    section_name: String,
    sec_source: FiberSection2dDef,
    state: RunCaseState,
    instance_count: Int,
    iterations: Int,
    samples: Int,
    scenario_index: Int,
    is_steel: Bool,
) raises:
    var scenario = _scenario_definition(scenario_index)
    var sec_def = sec_source
    var uniaxial_states = state.uniaxial_states.copy()
    var base_offset = fiber_section2d_runtime_alloc_instances(sec_def, instance_count)
    _apply_preload_history(sec_def, uniaxial_states, base_offset, instance_count, scenario_index)
    var family_count = sec_def.concrete02_count
    var benchmark_name = String("material_scalar_concrete02")
    if is_steel:
        family_count = sec_def.steel02_count
        benchmark_name = "material_scalar_steel02"
    if family_count <= 0:
        return
    var total_updates = instance_count * iterations * family_count
    for sample in range(samples):
        var started_ns = _time_ns()
        for _ in range(iterations):
            for inst in range(instance_count):
                if is_steel:
                    var slot_start = inst * sec_def.steel02_instance_stride
                    for i in range(sec_def.steel02_count):
                        var y_rel = sec_def.steel02_y_rel[i]
                        var strain = scenario.target_eps0 - y_rel * scenario.target_kappa
                        var mat_def = sec_def.steel02_mat_defs[i]
                        _fiber_section2d_runtime_apply_steel02(
                            mat_def, sec_def, slot_start + i, strain
                        )
                else:
                    var slot_start = inst * sec_def.concrete02_instance_stride
                    for i in range(sec_def.concrete02_count):
                        var y_rel = sec_def.concrete02_y_rel[i]
                        var strain = scenario.target_eps0 - y_rel * scenario.target_kappa
                        var mat_def = sec_def.concrete02_mat_defs[i]
                        _fiber_section2d_runtime_apply_concrete02(
                            mat_def, sec_def, slot_start + i, strain
                        )
        var elapsed_ns = _time_ns() - started_ns
        _print_sample(
            section_id,
            section_index,
            section_name,
            sec_def.fiber_count,
            sec_def.elastic_count,
            sec_def.steel02_count,
            sec_def.steel02_padded_count,
            sec_def.concrete02_count,
            sec_def.concrete02_padded_count,
            instance_count,
            scenario.name,
            benchmark_name,
            sample,
            iterations,
            total_updates,
            elapsed_ns,
        )


fn _run_section_benchmarks(
    section_id: Int,
    section_index: Int,
    state: RunCaseState,
    target_fibers: Int,
    iterations: Int,
    samples: Int,
) raises:
    ref sec_source = state.fiber_section_defs[section_index]
    var fiber_count = sec_source.fiber_count
    if fiber_count <= 0:
        return
    var instance_count = (target_fibers + fiber_count - 1) // fiber_count
    if instance_count <= 0:
        instance_count = 1
    var section_name = "section_" + String(section_id)
    for scenario_index in range(SCENARIO_COUNT):
        _run_material_scalar_bench(
            section_id,
            section_index,
            section_name,
            sec_source,
            state,
            instance_count,
            iterations,
            samples,
            scenario_index,
            False,
        )
        _run_material_scalar_bench(
            section_id,
            section_index,
            section_name,
            sec_source,
            state,
            instance_count,
            iterations,
            samples,
            scenario_index,
            True,
        )
        _run_family_bench(
            section_id,
            section_index,
            section_name,
            sec_source,
            state,
            instance_count,
            iterations,
            samples,
            scenario_index,
            False,
        )
        _run_family_bench(
            section_id,
            section_index,
            section_name,
            sec_source,
            state,
            instance_count,
            iterations,
            samples,
            scenario_index,
            True,
        )
        _run_full_section_bench(
            section_id,
            section_index,
            section_name,
            sec_source,
            state,
            instance_count,
            iterations,
            samples,
            scenario_index,
        )


fn main() raises:
    var args = _parse_args()
    var input = parse_case_input(load_json(args.input_path))
    var state = load_case_state_from_input(input)
    print(
        "section_id,section_index,section_name,fiber_count,elastic_count,"
        "steel02_count,steel02_padded_count,concrete02_count,concrete02_padded_count,"
        "instances,scenario,benchmark,sample,iterations,total_updates,elapsed_ns"
    )
    var matched_any = False
    for sec_id in range(len(state.fiber_section_index_by_id)):
        var sec_index = state.fiber_section_index_by_id[sec_id]
        if sec_index < 0 or sec_index >= len(state.fiber_section_defs):
            continue
        if args.section_id >= 0 and sec_id != args.section_id:
            continue
        matched_any = True
        _run_section_benchmarks(
            sec_id,
            sec_index,
            state,
            args.target_fibers,
            args.iterations,
            args.samples,
        )
    if not matched_any:
        abort("no matching FiberSection2d sections found")
