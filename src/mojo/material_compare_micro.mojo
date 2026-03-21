from collections import List
from materials import UniMaterialDef, UniMaterialState, uniaxial_commit
from materials.uniaxial import (
    uniaxial_set_trial_strain_concrete01,
    uniaxial_set_trial_strain_elastic,
    uniaxial_set_trial_strain_steel01,
)
from materials.uniaxial.concrete02 import _concrete02_set_trial
from materials.uniaxial.steel02 import _steel02_set_trial
from os import abort
from python import Python
from sections import FiberSection2dDef
from sections.fiber2d import (
    _fiber_section2d_runtime_apply_concrete01,
    _fiber_section2d_runtime_apply_concrete02,
    _fiber_section2d_runtime_apply_steel01,
    _fiber_section2d_runtime_apply_steel02,
)
from solver.run_case.input_types import MaterialInput, parse_case_input_native_from_source
from strut_io import case_source_from_path, load_json_native
from sys.arg import argv
from tag_types import UniMaterialTypeTag


alias IMPL_ELASTIC_UNIAXIAL: Int = 1
alias IMPL_ELASTIC_FIBER_SCALAR: Int = 2
alias IMPL_STEEL01_UNIAXIAL: Int = 3
alias IMPL_STEEL01_FIBER_SCALAR: Int = 4
alias IMPL_CONCRETE01_UNIAXIAL: Int = 5
alias IMPL_CONCRETE01_FIBER_SCALAR: Int = 6
alias IMPL_STEEL02_UNIAXIAL: Int = 7
alias IMPL_STEEL02_FIBER_SCALAR: Int = 8
alias IMPL_CONCRETE02_UNIAXIAL: Int = 9
alias IMPL_CONCRETE02_FIBER_SCALAR: Int = 10

alias DEFAULT_ATOL: Float64 = 1.0e-10
alias DEFAULT_RTOL: Float64 = 1.0e-9
alias SCENARIO_COUNT: Int = 5


struct CompareArgs(Movable):
    var input_path: String
    var left_impl: String
    var right_impl: String
    var states: Int
    var steps: Int
    var samples: Int

    fn __init__(
        out self,
        input_path: String,
        left_impl: String,
        right_impl: String,
        states: Int,
        steps: Int,
        samples: Int,
    ):
        self.input_path = input_path
        self.left_impl = left_impl
        self.right_impl = right_impl
        self.states = states
        self.steps = steps
        self.samples = samples


fn _arg_value(
    args: VariadicList[StringSlice[StaticConstantOrigin]], name: String, default_value: String
) -> String:
    for i in range(len(args) - 1):
        if String(args[i]) == name:
            return String(args[i + 1])
    return default_value


fn _parse_args() raises -> CompareArgs:
    var args = argv()
    var input_path = _arg_value(args, "--input", "")
    var left_impl = _arg_value(args, "--left-impl", "")
    var right_impl = _arg_value(args, "--right-impl", "")
    var states = 1024
    var steps = 128
    var samples = 5
    var states_raw = _arg_value(args, "--states", "")
    if states_raw != "":
        states = Int(states_raw)
    var steps_raw = _arg_value(args, "--steps", "")
    if steps_raw != "":
        steps = Int(steps_raw)
    var samples_raw = _arg_value(args, "--samples", "")
    if samples_raw != "":
        samples = Int(samples_raw)
    if input_path == "":
        abort("missing --input")
    if states <= 0:
        abort("--states must be > 0")
    if steps <= 0:
        abort("--steps must be > 0")
    if samples <= 0:
        abort("--samples must be > 0")
    return CompareArgs(input_path, left_impl, right_impl, states, steps, samples)


fn _scenario_name(index: Int) -> String:
    if index == 0:
        return "monotonic_compression"
    if index == 1:
        return "monotonic_tension"
    if index == 2:
        return "symmetric_reversal"
    if index == 3:
        return "compression_partial_unload"
    if index == 4:
        return "random_walk_extreme"
    abort("scenario index out of range")
    return ""


fn _impl_id(name: String) -> Int:
    if name == "elastic_uniaxial":
        return IMPL_ELASTIC_UNIAXIAL
    if name == "elastic_fiber_scalar":
        return IMPL_ELASTIC_FIBER_SCALAR
    if name == "steel01_uniaxial":
        return IMPL_STEEL01_UNIAXIAL
    if name == "steel01_fiber_scalar":
        return IMPL_STEEL01_FIBER_SCALAR
    if name == "concrete01_uniaxial":
        return IMPL_CONCRETE01_UNIAXIAL
    if name == "concrete01_fiber_scalar":
        return IMPL_CONCRETE01_FIBER_SCALAR
    if name == "steel02_uniaxial":
        return IMPL_STEEL02_UNIAXIAL
    if name == "steel02_fiber_scalar":
        return IMPL_STEEL02_FIBER_SCALAR
    if name == "concrete02_uniaxial":
        return IMPL_CONCRETE02_UNIAXIAL
    if name == "concrete02_fiber_scalar":
        return IMPL_CONCRETE02_FIBER_SCALAR
    return 0


fn _impl_name(impl_id: Int) -> String:
    if impl_id == IMPL_ELASTIC_UNIAXIAL:
        return "elastic_uniaxial"
    if impl_id == IMPL_ELASTIC_FIBER_SCALAR:
        return "elastic_fiber_scalar"
    if impl_id == IMPL_STEEL01_UNIAXIAL:
        return "steel01_uniaxial"
    if impl_id == IMPL_STEEL01_FIBER_SCALAR:
        return "steel01_fiber_scalar"
    if impl_id == IMPL_CONCRETE01_UNIAXIAL:
        return "concrete01_uniaxial"
    if impl_id == IMPL_CONCRETE01_FIBER_SCALAR:
        return "concrete01_fiber_scalar"
    if impl_id == IMPL_STEEL02_UNIAXIAL:
        return "steel02_uniaxial"
    if impl_id == IMPL_STEEL02_FIBER_SCALAR:
        return "steel02_fiber_scalar"
    if impl_id == IMPL_CONCRETE02_UNIAXIAL:
        return "concrete02_uniaxial"
    if impl_id == IMPL_CONCRETE02_FIBER_SCALAR:
        return "concrete02_fiber_scalar"
    return ""


fn _impl_material_type(impl_id: Int) -> Int:
    if impl_id == IMPL_ELASTIC_UNIAXIAL or impl_id == IMPL_ELASTIC_FIBER_SCALAR:
        return UniMaterialTypeTag.Elastic
    if impl_id == IMPL_STEEL01_UNIAXIAL or impl_id == IMPL_STEEL01_FIBER_SCALAR:
        return UniMaterialTypeTag.Steel01
    if impl_id == IMPL_CONCRETE01_UNIAXIAL or impl_id == IMPL_CONCRETE01_FIBER_SCALAR:
        return UniMaterialTypeTag.Concrete01
    if impl_id == IMPL_STEEL02_UNIAXIAL or impl_id == IMPL_STEEL02_FIBER_SCALAR:
        return UniMaterialTypeTag.Steel02
    if impl_id == IMPL_CONCRETE02_UNIAXIAL or impl_id == IMPL_CONCRETE02_FIBER_SCALAR:
        return UniMaterialTypeTag.Concrete02
    return -1


fn _material_type_name(material_type: Int) -> String:
    if material_type == UniMaterialTypeTag.Elastic:
        return "Elastic"
    if material_type == UniMaterialTypeTag.Steel01:
        return "Steel01"
    if material_type == UniMaterialTypeTag.Concrete01:
        return "Concrete01"
    if material_type == UniMaterialTypeTag.Concrete02:
        return "Concrete02"
    if material_type == UniMaterialTypeTag.Steel02:
        return "Steel02"
    return ""


fn _material_input_to_def(mat: MaterialInput) -> UniMaterialDef:
    if mat.type == "Elastic":
        return UniMaterialDef(UniMaterialTypeTag.Elastic, mat.E, 0.0, 0.0, 0.0)
    if mat.type == "Steel01":
        return UniMaterialDef(UniMaterialTypeTag.Steel01, mat.Fy, mat.E0, mat.b, 0.0)
    if mat.type == "Concrete01":
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
        return UniMaterialDef(UniMaterialTypeTag.Concrete01, fpc, epsc0, fpcu, epscu)
    if mat.type == "Steel02":
        var Fy = mat.Fy
        var E0 = mat.E0
        var b = mat.b
        var R0 = 15.0
        var cR1 = 0.925
        var cR2 = 0.15
        if mat.has_r0:
            R0 = mat.R0
            cR1 = mat.cR1
            cR2 = mat.cR2
        var a1 = 0.0
        var a2 = 1.0
        var a3 = 0.0
        var a4 = 1.0
        if mat.has_a1:
            a1 = mat.a1
            a2 = mat.a2
            a3 = mat.a3
            a4 = mat.a4
        var sig_init = 0.0
        if mat.has_siginit:
            sig_init = mat.sigInit
        return UniMaterialDef(
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
    if mat.type == "Concrete02":
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
        var rat = 0.1
        var ft = -0.1 * fpc
        var ets = 0.1 * fpc / epsc0
        if mat.has_rat:
            rat = mat.rat
            ft = mat.ft
            ets = mat.Ets
        return UniMaterialDef(
            UniMaterialTypeTag.Concrete02,
            fpc,
            epsc0,
            fpcu,
            epscu,
            rat,
            ft,
            ets,
            0.0,
            0.0,
            0.0,
            0.0,
        )
    abort("unsupported material type for compare harness")
    return UniMaterialDef()


fn _lcg_next(mut state: Int) -> Int:
    state = (state * 1103515245 + 12345) % 2147483647
    if state <= 0:
        state += 2147483646
    return state


fn _rand_unit(mut state: Int) -> Float64:
    state = _lcg_next(state)
    return Float64(state) / 2147483647.0


fn _generate_strains_for_scenario(
    material_type: Int,
    material_id: Int,
    state_count: Int,
    steps: Int,
    scenario_index: Int,
) -> List[Float64]:
    var strains: List[Float64] = []
    strains.resize(state_count * steps, 0.0)
    var rng = 7919 + material_id * 104729 + material_type * 31337
    var compression_limit = -0.03
    var tension_limit = 0.03
    if material_type == UniMaterialTypeTag.Concrete02:
        compression_limit = -0.02
        tension_limit = 0.004
    for state_index in range(state_count):
        var amplitude_bias = 0.85 + 0.15 * Float64(state_index % 13) / 12.0
        var prev = 0.0
        for step_index in range(steps):
            var x = 0.0
            if steps > 1:
                x = Float64(step_index) / Float64(steps - 1)
            var strain: Float64
            if scenario_index == 0:
                strain = compression_limit * (0.15 + 0.85 * x) * amplitude_bias
            elif scenario_index == 1:
                strain = tension_limit * (0.15 + 0.85 * x) * amplitude_bias
            elif scenario_index == 2:
                var amplitude = tension_limit * (0.25 + 0.75 * x) * amplitude_bias
                var cycle_pos = Float64(step_index % 24) / 24.0
                if cycle_pos < 0.25:
                    strain = amplitude * (cycle_pos / 0.25)
                elif cycle_pos < 0.5:
                    strain = amplitude * (1.0 - 2.0 * (cycle_pos - 0.25) / 0.25)
                elif cycle_pos < 0.75:
                    strain = -amplitude * ((cycle_pos - 0.5) / 0.25)
                else:
                    strain = -amplitude * (1.0 - (cycle_pos - 0.75) / 0.25)
                if material_type == UniMaterialTypeTag.Concrete02 and strain < 0.0:
                    strain = 1.6 * strain
            elif scenario_index == 3:
                if x < 0.55:
                    strain = compression_limit * (0.1 + 1.65 * x) * amplitude_bias
                else:
                    var unload = (x - 0.55) / 0.45
                    strain = compression_limit * (1.0 - 0.82 * unload) * amplitude_bias
                    if material_type == UniMaterialTypeTag.Steel02:
                        strain += tension_limit * 0.45 * unload
                    else:
                        strain += tension_limit * 0.12 * unload
            else:
                rng = _lcg_next(rng)
                var r0 = Float64(rng) / 2147483647.0
                rng = _lcg_next(rng)
                var r1 = Float64(rng) / 2147483647.0
                var target: Float64
                if material_type == UniMaterialTypeTag.Steel02:
                    target = (2.0 * r0 - 1.0) * tension_limit * amplitude_bias
                else:
                    if r0 < 0.82:
                        target = compression_limit * r1 * amplitude_bias
                    else:
                        target = tension_limit * r1 * amplitude_bias
                prev = 0.72 * prev + 0.28 * target
                strain = prev
            strains[state_index * steps + step_index] = strain
        rng += state_index + 17
    return strains^


fn _init_steel02_candidate(mut sec_def: FiberSection2dDef, mat_def: UniMaterialDef, state_count: Int):
    sec_def.runtime_s2_eps_c.resize(state_count, 0.0)
    sec_def.runtime_s2_sig_c.resize(state_count, 0.0)
    sec_def.runtime_s2_tangent_c.resize(state_count, 0.0)
    sec_def.runtime_s2_epsmin_c.resize(state_count, 0.0)
    sec_def.runtime_s2_epsmax_c.resize(state_count, 0.0)
    sec_def.runtime_s2_epspl_c.resize(state_count, 0.0)
    sec_def.runtime_s2_epss0_c.resize(state_count, 0.0)
    sec_def.runtime_s2_sigs0_c.resize(state_count, 0.0)
    sec_def.runtime_s2_epsr_c.resize(state_count, 0.0)
    sec_def.runtime_s2_sigr_c.resize(state_count, 0.0)
    sec_def.runtime_s2_kon_c.resize(state_count, 0)
    sec_def.runtime_s2_eps_t.resize(state_count, 0.0)
    sec_def.runtime_s2_sig_t.resize(state_count, 0.0)
    sec_def.runtime_s2_tangent_t.resize(state_count, 0.0)
    sec_def.runtime_s2_epsmin_t.resize(state_count, 0.0)
    sec_def.runtime_s2_epsmax_t.resize(state_count, 0.0)
    sec_def.runtime_s2_epspl_t.resize(state_count, 0.0)
    sec_def.runtime_s2_epss0_t.resize(state_count, 0.0)
    sec_def.runtime_s2_sigs0_t.resize(state_count, 0.0)
    sec_def.runtime_s2_epsr_t.resize(state_count, 0.0)
    sec_def.runtime_s2_sigr_t.resize(state_count, 0.0)
    sec_def.runtime_s2_kon_t.resize(state_count, 0)
    for i in range(state_count):
        var state = UniMaterialState(mat_def)
        sec_def.runtime_s2_eps_c[i] = state.eps_c
        sec_def.runtime_s2_sig_c[i] = state.sig_c
        sec_def.runtime_s2_tangent_c[i] = state.tangent_c
        sec_def.runtime_s2_epsmin_c[i] = state.s2_epsmin_c
        sec_def.runtime_s2_epsmax_c[i] = state.s2_epsmax_c
        sec_def.runtime_s2_epspl_c[i] = state.s2_epspl_c
        sec_def.runtime_s2_epss0_c[i] = state.s2_epss0_c
        sec_def.runtime_s2_sigs0_c[i] = state.s2_sigs0_c
        sec_def.runtime_s2_epsr_c[i] = state.s2_epsr_c
        sec_def.runtime_s2_sigr_c[i] = state.s2_sigr_c
        sec_def.runtime_s2_kon_c[i] = state.s2_kon_c
        sec_def.runtime_s2_eps_t[i] = state.eps_t
        sec_def.runtime_s2_sig_t[i] = state.sig_t
        sec_def.runtime_s2_tangent_t[i] = state.tangent_t
        sec_def.runtime_s2_epsmin_t[i] = state.s2_epsmin_t
        sec_def.runtime_s2_epsmax_t[i] = state.s2_epsmax_t
        sec_def.runtime_s2_epspl_t[i] = state.s2_epspl_t
        sec_def.runtime_s2_epss0_t[i] = state.s2_epss0_t
        sec_def.runtime_s2_sigs0_t[i] = state.s2_sigs0_t
        sec_def.runtime_s2_epsr_t[i] = state.s2_epsr_t
        sec_def.runtime_s2_sigr_t[i] = state.s2_sigr_t
        sec_def.runtime_s2_kon_t[i] = state.s2_kon_t


fn _init_concrete02_candidate(mut sec_def: FiberSection2dDef, mat_def: UniMaterialDef, state_count: Int):
    sec_def.runtime_c2_eps_c.resize(state_count, 0.0)
    sec_def.runtime_c2_sig_c.resize(state_count, 0.0)
    sec_def.runtime_c2_tangent_c.resize(state_count, 0.0)
    sec_def.runtime_c2_eps_t.resize(state_count, 0.0)
    sec_def.runtime_c2_sig_t.resize(state_count, 0.0)
    sec_def.runtime_c2_tangent_t.resize(state_count, 0.0)
    sec_def.runtime_c2_ecmin_c.resize(state_count, 0.0)
    sec_def.runtime_c2_dept_c.resize(state_count, 0.0)
    sec_def.runtime_c2_ecmin_t.resize(state_count, 0.0)
    sec_def.runtime_c2_dept_t.resize(state_count, 0.0)
    for i in range(state_count):
        var state = UniMaterialState(mat_def)
        sec_def.runtime_c2_eps_c[i] = state.eps_c
        sec_def.runtime_c2_sig_c[i] = state.sig_c
        sec_def.runtime_c2_tangent_c[i] = state.tangent_c
        sec_def.runtime_c2_eps_t[i] = state.eps_t
        sec_def.runtime_c2_sig_t[i] = state.sig_t
        sec_def.runtime_c2_tangent_t[i] = state.tangent_t
        sec_def.runtime_c2_ecmin_c[i] = state.c2_ecmin_c
        sec_def.runtime_c2_dept_c[i] = state.c2_dept_c
        sec_def.runtime_c2_ecmin_t[i] = state.c2_ecmin_t
        sec_def.runtime_c2_dept_t[i] = state.c2_dept_t


fn _init_runtime_scalar_candidate(
    mut sec_def: FiberSection2dDef, mat_def: UniMaterialDef, state_count: Int
):
    sec_def.runtime_eps_c.resize(state_count, 0.0)
    sec_def.runtime_sig_c.resize(state_count, 0.0)
    sec_def.runtime_tangent_c.resize(state_count, 0.0)
    sec_def.runtime_eps_p_c.resize(state_count, 0.0)
    sec_def.runtime_alpha_c.resize(state_count, 0.0)
    sec_def.runtime_eps_t.resize(state_count, 0.0)
    sec_def.runtime_sig_t.resize(state_count, 0.0)
    sec_def.runtime_tangent_t.resize(state_count, 0.0)
    sec_def.runtime_eps_p_t.resize(state_count, 0.0)
    sec_def.runtime_alpha_t.resize(state_count, 0.0)
    sec_def.runtime_min_strain_c.resize(state_count, 0.0)
    sec_def.runtime_end_strain_c.resize(state_count, 0.0)
    sec_def.runtime_unload_slope_c.resize(state_count, 0.0)
    sec_def.runtime_min_strain_t.resize(state_count, 0.0)
    sec_def.runtime_end_strain_t.resize(state_count, 0.0)
    sec_def.runtime_unload_slope_t.resize(state_count, 0.0)
    for i in range(state_count):
        var state = UniMaterialState(mat_def)
        sec_def.runtime_eps_c[i] = state.eps_c
        sec_def.runtime_sig_c[i] = state.sig_c
        sec_def.runtime_tangent_c[i] = state.tangent_c
        sec_def.runtime_eps_p_c[i] = state.eps_p_c
        sec_def.runtime_alpha_c[i] = state.alpha_c
        sec_def.runtime_eps_t[i] = state.eps_t
        sec_def.runtime_sig_t[i] = state.sig_t
        sec_def.runtime_tangent_t[i] = state.tangent_t
        sec_def.runtime_eps_p_t[i] = state.eps_p_t
        sec_def.runtime_alpha_t[i] = state.alpha_t
        sec_def.runtime_min_strain_c[i] = state.min_strain_c
        sec_def.runtime_end_strain_c[i] = state.end_strain_c
        sec_def.runtime_unload_slope_c[i] = state.unload_slope_c
        sec_def.runtime_min_strain_t[i] = state.min_strain_t
        sec_def.runtime_end_strain_t[i] = state.end_strain_t
        sec_def.runtime_unload_slope_t[i] = state.unload_slope_t


fn _impl_uses_runtime_scalar_slot(impl_id: Int) -> Bool:
    return (
        impl_id == IMPL_ELASTIC_FIBER_SCALAR
        or impl_id == IMPL_STEEL01_FIBER_SCALAR
        or impl_id == IMPL_CONCRETE01_FIBER_SCALAR
    )


fn _apply_impl(
    impl_id: Int,
    mat_def: UniMaterialDef,
    mut uni_states: List[UniMaterialState],
    mut sec_def: FiberSection2dDef,
    slot: Int,
    strain: Float64,
):
    if impl_id == IMPL_ELASTIC_UNIAXIAL:
        uniaxial_set_trial_strain_elastic(mat_def, uni_states[slot], strain)
        return
    if impl_id == IMPL_STEEL01_UNIAXIAL:
        uniaxial_set_trial_strain_steel01(mat_def, uni_states[slot], strain)
        return
    if impl_id == IMPL_CONCRETE01_UNIAXIAL:
        uniaxial_set_trial_strain_concrete01(mat_def, uni_states[slot], strain)
        return
    if impl_id == IMPL_CONCRETE02_UNIAXIAL:
        _concrete02_set_trial(mat_def, uni_states[slot], strain)
        return
    if impl_id == IMPL_STEEL02_UNIAXIAL:
        _steel02_set_trial(mat_def, uni_states[slot], strain)
        return
    if _impl_uses_runtime_scalar_slot(impl_id):
        if impl_id == IMPL_ELASTIC_FIBER_SCALAR:
            sec_def.runtime_eps_t[slot] = strain
            sec_def.runtime_sig_t[slot] = mat_def.p0 * strain
            sec_def.runtime_tangent_t[slot] = mat_def.p0
            sec_def.runtime_eps_p_t[slot] = sec_def.runtime_eps_p_c[slot]
            sec_def.runtime_alpha_t[slot] = sec_def.runtime_alpha_c[slot]
            sec_def.runtime_min_strain_t[slot] = sec_def.runtime_min_strain_c[slot]
            sec_def.runtime_end_strain_t[slot] = sec_def.runtime_end_strain_c[slot]
            sec_def.runtime_unload_slope_t[slot] = sec_def.runtime_unload_slope_c[slot]
            return
        if impl_id == IMPL_STEEL01_FIBER_SCALAR:
            _fiber_section2d_runtime_apply_steel01(mat_def, sec_def, slot, strain)
            return
        _fiber_section2d_runtime_apply_concrete01(mat_def, sec_def, slot, strain)
        return
    if impl_id == IMPL_CONCRETE02_FIBER_SCALAR:
        _fiber_section2d_runtime_apply_concrete02(mat_def, sec_def, slot, strain)
        return
    if impl_id == IMPL_STEEL02_FIBER_SCALAR:
        _fiber_section2d_runtime_apply_steel02(mat_def, sec_def, slot, strain)
        return
    abort("unknown implementation id")


fn _commit_impl(
    impl_id: Int,
    mut uni_states: List[UniMaterialState],
    mut sec_def: FiberSection2dDef,
    slot: Int,
):
    if (
        impl_id == IMPL_ELASTIC_UNIAXIAL
        or impl_id == IMPL_STEEL01_UNIAXIAL
        or impl_id == IMPL_CONCRETE01_UNIAXIAL
        or impl_id == IMPL_CONCRETE02_UNIAXIAL
        or impl_id == IMPL_STEEL02_UNIAXIAL
    ):
        uniaxial_commit(uni_states[slot])
        return
    if impl_id == IMPL_ELASTIC_FIBER_SCALAR:
        sec_def.runtime_eps_c[slot] = sec_def.runtime_eps_t[slot]
        sec_def.runtime_sig_c[slot] = sec_def.runtime_sig_t[slot]
        sec_def.runtime_tangent_c[slot] = sec_def.runtime_tangent_t[slot]
        sec_def.runtime_eps_p_c[slot] = sec_def.runtime_eps_p_t[slot]
        sec_def.runtime_alpha_c[slot] = sec_def.runtime_alpha_t[slot]
        sec_def.runtime_min_strain_c[slot] = sec_def.runtime_min_strain_t[slot]
        sec_def.runtime_end_strain_c[slot] = sec_def.runtime_end_strain_t[slot]
        sec_def.runtime_unload_slope_c[slot] = sec_def.runtime_unload_slope_t[slot]
        return
    if impl_id == IMPL_STEEL01_FIBER_SCALAR:
        sec_def.runtime_eps_c[slot] = sec_def.runtime_eps_t[slot]
        sec_def.runtime_sig_c[slot] = sec_def.runtime_sig_t[slot]
        sec_def.runtime_tangent_c[slot] = sec_def.runtime_tangent_t[slot]
        sec_def.runtime_eps_p_c[slot] = sec_def.runtime_eps_p_t[slot]
        sec_def.runtime_alpha_c[slot] = sec_def.runtime_alpha_t[slot]
        return
    if impl_id == IMPL_CONCRETE01_FIBER_SCALAR:
        sec_def.runtime_eps_c[slot] = sec_def.runtime_eps_t[slot]
        sec_def.runtime_sig_c[slot] = sec_def.runtime_sig_t[slot]
        sec_def.runtime_tangent_c[slot] = sec_def.runtime_tangent_t[slot]
        sec_def.runtime_min_strain_c[slot] = sec_def.runtime_min_strain_t[slot]
        sec_def.runtime_end_strain_c[slot] = sec_def.runtime_end_strain_t[slot]
        sec_def.runtime_unload_slope_c[slot] = sec_def.runtime_unload_slope_t[slot]
        sec_def.runtime_eps_p_c[slot] = sec_def.runtime_eps_p_t[slot]
        sec_def.runtime_alpha_c[slot] = sec_def.runtime_alpha_t[slot]
        return
    if impl_id == IMPL_CONCRETE02_FIBER_SCALAR:
        sec_def.runtime_c2_eps_c[slot] = sec_def.runtime_c2_eps_t[slot]
        sec_def.runtime_c2_sig_c[slot] = sec_def.runtime_c2_sig_t[slot]
        sec_def.runtime_c2_tangent_c[slot] = sec_def.runtime_c2_tangent_t[slot]
        sec_def.runtime_c2_ecmin_c[slot] = sec_def.runtime_c2_ecmin_t[slot]
        sec_def.runtime_c2_dept_c[slot] = sec_def.runtime_c2_dept_t[slot]
        return
    if impl_id == IMPL_STEEL02_FIBER_SCALAR:
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
        return


fn _trial_stress(impl_id: Int, uni_states: List[UniMaterialState], sec_def: FiberSection2dDef, slot: Int) -> Float64:
    if (
        impl_id == IMPL_ELASTIC_UNIAXIAL
        or impl_id == IMPL_STEEL01_UNIAXIAL
        or impl_id == IMPL_CONCRETE01_UNIAXIAL
        or impl_id == IMPL_CONCRETE02_UNIAXIAL
        or impl_id == IMPL_STEEL02_UNIAXIAL
    ):
        return uni_states[slot].sig_t
    if _impl_uses_runtime_scalar_slot(impl_id):
        return sec_def.runtime_sig_t[slot]
    if impl_id == IMPL_CONCRETE02_FIBER_SCALAR:
        return sec_def.runtime_c2_sig_t[slot]
    return sec_def.runtime_s2_sig_t[slot]


fn _trial_tangent(impl_id: Int, uni_states: List[UniMaterialState], sec_def: FiberSection2dDef, slot: Int) -> Float64:
    if (
        impl_id == IMPL_ELASTIC_UNIAXIAL
        or impl_id == IMPL_STEEL01_UNIAXIAL
        or impl_id == IMPL_CONCRETE01_UNIAXIAL
        or impl_id == IMPL_CONCRETE02_UNIAXIAL
        or impl_id == IMPL_STEEL02_UNIAXIAL
    ):
        return uni_states[slot].tangent_t
    if _impl_uses_runtime_scalar_slot(impl_id):
        return sec_def.runtime_tangent_t[slot]
    if impl_id == IMPL_CONCRETE02_FIBER_SCALAR:
        return sec_def.runtime_c2_tangent_t[slot]
    return sec_def.runtime_s2_tangent_t[slot]


fn _state_field_count(material_type: Int) -> Int:
    if material_type == UniMaterialTypeTag.Elastic:
        return 1
    if material_type == UniMaterialTypeTag.Steel01:
        return 3
    if material_type == UniMaterialTypeTag.Concrete01:
        return 4
    if material_type == UniMaterialTypeTag.Concrete02:
        return 3
    if material_type == UniMaterialTypeTag.Steel02:
        return 9
    return 0


fn _state_field_value(
    material_type: Int,
    field_index: Int,
    impl_id: Int,
    uni_states: List[UniMaterialState],
    sec_def: FiberSection2dDef,
    slot: Int,
) -> Float64:
    if material_type == UniMaterialTypeTag.Elastic:
        if impl_id == IMPL_ELASTIC_UNIAXIAL:
            return uni_states[slot].eps_t
        return sec_def.runtime_eps_t[slot]
    if material_type == UniMaterialTypeTag.Steel01:
        if field_index == 0:
            if impl_id == IMPL_STEEL01_UNIAXIAL:
                return uni_states[slot].eps_t
            return sec_def.runtime_eps_t[slot]
        if field_index == 1:
            if impl_id == IMPL_STEEL01_UNIAXIAL:
                return uni_states[slot].eps_p_t
            return sec_def.runtime_eps_p_t[slot]
        if impl_id == IMPL_STEEL01_UNIAXIAL:
            return uni_states[slot].alpha_t
        return sec_def.runtime_alpha_t[slot]
    if material_type == UniMaterialTypeTag.Concrete01:
        if field_index == 0:
            if impl_id == IMPL_CONCRETE01_UNIAXIAL:
                return uni_states[slot].eps_t
            return sec_def.runtime_eps_t[slot]
        if field_index == 1:
            if impl_id == IMPL_CONCRETE01_UNIAXIAL:
                return uni_states[slot].min_strain_t
            return sec_def.runtime_min_strain_t[slot]
        if field_index == 2:
            if impl_id == IMPL_CONCRETE01_UNIAXIAL:
                return uni_states[slot].end_strain_t
            return sec_def.runtime_end_strain_t[slot]
        if impl_id == IMPL_CONCRETE01_UNIAXIAL:
            return uni_states[slot].unload_slope_t
        return sec_def.runtime_unload_slope_t[slot]
    if material_type == UniMaterialTypeTag.Concrete02:
        if field_index == 0:
            if impl_id == IMPL_CONCRETE02_UNIAXIAL:
                return uni_states[slot].eps_t
            return sec_def.runtime_c2_eps_t[slot]
        if field_index == 1:
            if impl_id == IMPL_CONCRETE02_UNIAXIAL:
                return uni_states[slot].c2_ecmin_t
            return sec_def.runtime_c2_ecmin_t[slot]
        if field_index == 2:
            if impl_id == IMPL_CONCRETE02_UNIAXIAL:
                return uni_states[slot].c2_dept_t
            return sec_def.runtime_c2_dept_t[slot]
        return 0.0
    if field_index == 0:
        if impl_id == IMPL_STEEL02_UNIAXIAL:
            return uni_states[slot].eps_t
        return sec_def.runtime_s2_eps_t[slot]
    if field_index == 1:
        if impl_id == IMPL_STEEL02_UNIAXIAL:
            return uni_states[slot].s2_epsmin_t
        return sec_def.runtime_s2_epsmin_t[slot]
    if field_index == 2:
        if impl_id == IMPL_STEEL02_UNIAXIAL:
            return uni_states[slot].s2_epsmax_t
        return sec_def.runtime_s2_epsmax_t[slot]
    if field_index == 3:
        if impl_id == IMPL_STEEL02_UNIAXIAL:
            return uni_states[slot].s2_epspl_t
        return sec_def.runtime_s2_epspl_t[slot]
    if field_index == 4:
        if impl_id == IMPL_STEEL02_UNIAXIAL:
            return uni_states[slot].s2_epss0_t
        return sec_def.runtime_s2_epss0_t[slot]
    if field_index == 5:
        if impl_id == IMPL_STEEL02_UNIAXIAL:
            return uni_states[slot].s2_sigs0_t
        return sec_def.runtime_s2_sigs0_t[slot]
    if field_index == 6:
        if impl_id == IMPL_STEEL02_UNIAXIAL:
            return uni_states[slot].s2_epsr_t
        return sec_def.runtime_s2_epsr_t[slot]
    if field_index == 7:
        if impl_id == IMPL_STEEL02_UNIAXIAL:
            return uni_states[slot].s2_sigr_t
        return sec_def.runtime_s2_sigr_t[slot]
    if impl_id == IMPL_STEEL02_UNIAXIAL:
        return Float64(uni_states[slot].s2_kon_t)
    return Float64(sec_def.runtime_s2_kon_t[slot])


fn _in_tolerance(diff: Float64, lhs: Float64, rhs: Float64) -> Bool:
    var scale = 1.0
    var lhs_abs = lhs
    if lhs_abs < 0.0:
        lhs_abs = -lhs_abs
    var rhs_abs = rhs
    if rhs_abs < 0.0:
        rhs_abs = -rhs_abs
    if lhs_abs > scale:
        scale = lhs_abs
    if rhs_abs > scale:
        scale = rhs_abs
    return diff <= DEFAULT_ATOL + DEFAULT_RTOL * scale


fn _time_ns() raises -> Int:
    var time = Python.import_module("time")
    return Int(time.perf_counter_ns())


fn _print_validation_row(
    compare_name: String,
    scenario_name: String,
    left_impl: String,
    right_impl: String,
    material_id: Int,
    material_type: String,
    states: Int,
    steps: Int,
    max_abs_stress_diff: Float64,
    max_abs_tangent_diff: Float64,
    max_abs_state_diff: Float64,
    mismatch_count: Int,
):
    print(
        "validation,"
        + compare_name
        + ","
        + scenario_name
        + ","
        + left_impl
        + ","
        + right_impl
        + ",,"
        + String(material_id)
        + ","
        + material_type
        + ","
        + String(states)
        + ","
        + String(steps)
        + ",0,0,0,"
        + String(max_abs_stress_diff)
        + ","
        + String(max_abs_tangent_diff)
        + ","
        + String(max_abs_state_diff)
        + ","
        + String(mismatch_count)
    )


fn _print_benchmark_row(
    compare_name: String,
    scenario_name: String,
    left_impl: String,
    right_impl: String,
    bench_impl: String,
    material_id: Int,
    material_type: String,
    states: Int,
    steps: Int,
    sample: Int,
    updates: Int,
    elapsed_ns: Int,
):
    print(
        "benchmark,"
        + compare_name
        + ","
        + scenario_name
        + ","
        + left_impl
        + ","
        + right_impl
        + ","
        + bench_impl
        + ","
        + String(material_id)
        + ","
        + material_type
        + ","
        + String(states)
        + ","
        + String(steps)
        + ","
        + String(sample)
        + ","
        + String(updates)
        + ","
        + String(elapsed_ns)
        + ",0,0,0,0"
    )


fn _run_validation(
    material_id: Int,
    material_type: Int,
    mat_def: UniMaterialDef,
    left_impl_id: Int,
    right_impl_id: Int,
    scenario_name: String,
    states: Int,
    steps: Int,
    strains: List[Float64],
):
    var left_uni: List[UniMaterialState] = []
    var right_uni: List[UniMaterialState] = []
    left_uni.resize(states, UniMaterialState())
    right_uni.resize(states, UniMaterialState())
    for i in range(states):
        left_uni[i] = UniMaterialState(mat_def)
        right_uni[i] = UniMaterialState(mat_def)
    var left_sec = FiberSection2dDef()
    var right_sec = FiberSection2dDef()
    if _impl_uses_runtime_scalar_slot(left_impl_id):
        _init_runtime_scalar_candidate(left_sec, mat_def, states)
    elif left_impl_id == IMPL_CONCRETE02_FIBER_SCALAR:
        _init_concrete02_candidate(left_sec, mat_def, states)
    elif left_impl_id == IMPL_STEEL02_FIBER_SCALAR:
        _init_steel02_candidate(left_sec, mat_def, states)
    if _impl_uses_runtime_scalar_slot(right_impl_id):
        _init_runtime_scalar_candidate(right_sec, mat_def, states)
    elif right_impl_id == IMPL_CONCRETE02_FIBER_SCALAR:
        _init_concrete02_candidate(right_sec, mat_def, states)
    elif right_impl_id == IMPL_STEEL02_FIBER_SCALAR:
        _init_steel02_candidate(right_sec, mat_def, states)

    var max_abs_stress_diff = 0.0
    var max_abs_tangent_diff = 0.0
    var max_abs_state_diff = 0.0
    var mismatch_count = 0
    for state_index in range(states):
        for step in range(steps):
            var strain = strains[state_index * steps + step]
            _apply_impl(left_impl_id, mat_def, left_uni, left_sec, state_index, strain)
            _apply_impl(right_impl_id, mat_def, right_uni, right_sec, state_index, strain)
            var left_stress = _trial_stress(left_impl_id, left_uni, left_sec, state_index)
            var right_stress = _trial_stress(right_impl_id, right_uni, right_sec, state_index)
            var stress_diff = left_stress - right_stress
            if stress_diff < 0.0:
                stress_diff = -stress_diff
            if stress_diff > max_abs_stress_diff:
                max_abs_stress_diff = stress_diff
            var left_tangent = _trial_tangent(left_impl_id, left_uni, left_sec, state_index)
            var right_tangent = _trial_tangent(right_impl_id, right_uni, right_sec, state_index)
            var tangent_diff = left_tangent - right_tangent
            if tangent_diff < 0.0:
                tangent_diff = -tangent_diff
            if tangent_diff > max_abs_tangent_diff:
                max_abs_tangent_diff = tangent_diff
            var state_field_count = _state_field_count(material_type)
            var state_diff_max_step = 0.0
            for field_index in range(state_field_count):
                var left_value = _state_field_value(
                    material_type, field_index, left_impl_id, left_uni, left_sec, state_index
                )
                var right_value = _state_field_value(
                    material_type, field_index, right_impl_id, right_uni, right_sec, state_index
                )
                var diff = left_value - right_value
                if diff < 0.0:
                    diff = -diff
                if diff > state_diff_max_step:
                    state_diff_max_step = diff
            if state_diff_max_step > max_abs_state_diff:
                max_abs_state_diff = state_diff_max_step
            if not _in_tolerance(stress_diff, left_stress, right_stress):
                mismatch_count += 1
            elif not _in_tolerance(tangent_diff, left_tangent, right_tangent):
                mismatch_count += 1
            elif not _in_tolerance(state_diff_max_step, 0.0, state_diff_max_step):
                mismatch_count += 1
            _commit_impl(left_impl_id, left_uni, left_sec, state_index)
            _commit_impl(right_impl_id, right_uni, right_sec, state_index)
    _print_validation_row(
        _impl_name(left_impl_id) + "__vs__" + _impl_name(right_impl_id),
        scenario_name,
        _impl_name(left_impl_id),
        _impl_name(right_impl_id),
        material_id,
        _material_type_name(material_type),
        states,
        steps,
        max_abs_stress_diff,
        max_abs_tangent_diff,
        max_abs_state_diff,
        mismatch_count,
    )


fn _run_benchmark_impl(
    material_id: Int,
    material_type: Int,
    mat_def: UniMaterialDef,
    compare_name: String,
    scenario_name: String,
    left_impl: String,
    right_impl: String,
    bench_impl_id: Int,
    states: Int,
    steps: Int,
    samples: Int,
    strains: List[Float64],
) raises:
    var updates = states * steps
    for sample in range(samples):
        var uni_states: List[UniMaterialState] = []
        uni_states.resize(states, UniMaterialState())
        for i in range(states):
            uni_states[i] = UniMaterialState(mat_def)
        var sec_def = FiberSection2dDef()
        if _impl_uses_runtime_scalar_slot(bench_impl_id):
            _init_runtime_scalar_candidate(sec_def, mat_def, states)
        elif bench_impl_id == IMPL_CONCRETE02_FIBER_SCALAR:
            _init_concrete02_candidate(sec_def, mat_def, states)
        elif bench_impl_id == IMPL_STEEL02_FIBER_SCALAR:
            _init_steel02_candidate(sec_def, mat_def, states)

        var started_ns = _time_ns()
        for state_index in range(states):
            for step in range(steps):
                var strain = strains[state_index * steps + step]
                _apply_impl(bench_impl_id, mat_def, uni_states, sec_def, state_index, strain)
                _commit_impl(bench_impl_id, uni_states, sec_def, state_index)
        var elapsed_ns = _time_ns() - started_ns
        _print_benchmark_row(
            compare_name,
            scenario_name,
            left_impl,
            right_impl,
            _impl_name(bench_impl_id),
            material_id,
            _material_type_name(material_type),
            states,
            steps,
            sample,
            updates,
            elapsed_ns,
        )


fn _run_compare_for_material(
    material_id: Int,
    mat_def: UniMaterialDef,
    left_impl_id: Int,
    right_impl_id: Int,
    scenario_index: Int,
    states: Int,
    steps: Int,
    samples: Int,
) raises:
    var material_type = mat_def.mat_type
    var compare_name = _impl_name(left_impl_id) + "__vs__" + _impl_name(right_impl_id)
    var scenario_name = _scenario_name(scenario_index)
    var strains = _generate_strains_for_scenario(
        material_type, material_id, states, steps, scenario_index
    )
    _run_validation(
        material_id,
        material_type,
        mat_def,
        left_impl_id,
        right_impl_id,
        scenario_name,
        states,
        steps,
        strains,
    )
    _run_benchmark_impl(
        material_id,
        material_type,
        mat_def,
        compare_name,
        scenario_name,
        _impl_name(left_impl_id),
        _impl_name(right_impl_id),
        left_impl_id,
        states,
        steps,
        samples,
        strains,
    )
    _run_benchmark_impl(
        material_id,
        material_type,
        mat_def,
        compare_name,
        scenario_name,
        _impl_name(left_impl_id),
        _impl_name(right_impl_id),
        right_impl_id,
        states,
        steps,
        samples,
        strains,
    )


fn _run_default_pair_for_material(
    material_id: Int,
    mat_def: UniMaterialDef,
    states: Int,
    steps: Int,
    samples: Int,
) raises:
    for scenario_index in range(SCENARIO_COUNT):
        if mat_def.mat_type == UniMaterialTypeTag.Elastic:
            _run_compare_for_material(
                material_id,
                mat_def,
                IMPL_ELASTIC_UNIAXIAL,
                IMPL_ELASTIC_FIBER_SCALAR,
                scenario_index,
                states,
                steps,
                samples,
            )
            continue
        if mat_def.mat_type == UniMaterialTypeTag.Steel01:
            _run_compare_for_material(
                material_id,
                mat_def,
                IMPL_STEEL01_UNIAXIAL,
                IMPL_STEEL01_FIBER_SCALAR,
                scenario_index,
                states,
                steps,
                samples,
            )
            continue
        if mat_def.mat_type == UniMaterialTypeTag.Concrete01:
            _run_compare_for_material(
                material_id,
                mat_def,
                IMPL_CONCRETE01_UNIAXIAL,
                IMPL_CONCRETE01_FIBER_SCALAR,
                scenario_index,
                states,
                steps,
                samples,
            )
            continue
        if mat_def.mat_type == UniMaterialTypeTag.Concrete02:
            _run_compare_for_material(
                material_id,
                mat_def,
                IMPL_CONCRETE02_UNIAXIAL,
                IMPL_CONCRETE02_FIBER_SCALAR,
                scenario_index,
                states,
                steps,
                samples,
            )
            continue
        if mat_def.mat_type == UniMaterialTypeTag.Steel02:
            _run_compare_for_material(
                material_id,
                mat_def,
                IMPL_STEEL02_UNIAXIAL,
                IMPL_STEEL02_FIBER_SCALAR,
                scenario_index,
                states,
                steps,
                samples,
            )


fn main() raises:
    var args = _parse_args()
    var source_info = case_source_from_path(args.input_path)
    var doc = load_json_native(args.input_path)
    var input = parse_case_input_native_from_source(doc, source_info)
    var left_impl_id = _impl_id(args.left_impl)
    var right_impl_id = _impl_id(args.right_impl)
    if (args.left_impl == "") != (args.right_impl == ""):
        abort("use both --left-impl and --right-impl together")
    if args.left_impl != "":
        if left_impl_id == 0 or right_impl_id == 0:
            abort("unknown compare implementation")
        if _impl_material_type(left_impl_id) != _impl_material_type(right_impl_id):
            abort("compare implementations must target the same material type")
    print(
        "row_type,compare,scenario,left_impl,right_impl,bench_impl,material_id,material_type,"
        "states,steps,sample,updates,elapsed_ns,max_abs_stress_diff,"
        "max_abs_tangent_diff,max_abs_state_diff,mismatch_count"
    )
    var matched_any = False
    for i in range(len(input.materials)):
        var mat = input.materials[i]
        if (
            mat.type != "Elastic"
            and mat.type != "Steel01"
            and mat.type != "Concrete01"
            and mat.type != "Concrete02"
            and mat.type != "Steel02"
        ):
            continue
        var mat_def = _material_input_to_def(mat)
        if args.left_impl != "":
            if mat_def.mat_type != _impl_material_type(left_impl_id):
                continue
            matched_any = True
            for scenario_index in range(SCENARIO_COUNT):
                _run_compare_for_material(
                    mat.id,
                    mat_def,
                    left_impl_id,
                    right_impl_id,
                    scenario_index,
                    args.states,
                    args.steps,
                    args.samples,
                )
        else:
            matched_any = True
            _run_default_pair_for_material(mat.id, mat_def, args.states, args.steps, args.samples)
    if not matched_any:
        abort("no matching materials found for compare harness")
