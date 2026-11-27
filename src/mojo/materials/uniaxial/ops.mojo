from collections import InlineArray, List
from materials.fsam import PanelVector
from os import abort

from materials.uniaxial.concrete01 import _concrete01_reload
from materials.uniaxial.concrete02 import _concrete02_set_trial
from materials.uniaxial.core import (
    _abs,
    _sign,
    UniMaterialDef,
    UniMaterialDefinition,
    UniMaterialPointState,
    UniMaterialState,
)
from materials.uniaxial.ent import ent_set_trial
from materials.uniaxial.elastic_pp import elastic_pp_set_trial
from materials.uniaxial.elastic_bilin import elastic_bilin_set_trial
from materials.uniaxial.py_simple1 import py_simple1_set_trial
from materials.uniaxial.qz_simple1 import qz_simple1_set_trial
from materials.uniaxial.hysteretic import hysteretic_asym_set_trial, hysteretic_smooth_set_trial
from materials.uniaxial.concrete_cm import ConcreteCMTrial
from materials.uniaxial.steel_mpf import SteelMPFPoint
from materials.uniaxial.steel02 import _steel02_set_trial
from materials.uniaxial.steel01_thermal import steel01_thermal_set_trial
from materials.uniaxial.tz_simple1 import tz_simple1_set_trial
from materials.uniaxial.viscous_damper import _viscous_damper_set_trial
from tag_types import UniMaterialTypeTag


fn uniaxial_set_trial_strain_elastic(
    mat_def: UniMaterialDefinition, mut state: UniMaterialPointState, eps: Float64
):
    if eps == state.eps_t:
        return
    state.eps_t = eps
    var E = mat_def.p0
    state.sig_t = E * eps
    state.tangent_t = E
    state.eps_p_t = state.eps_p_c
    state.alpha_t = state.alpha_c
    return


fn uniaxial_set_trial_strain_steel01(
    mat_def: UniMaterialDefinition, mut state: UniMaterialPointState, eps: Float64
):
    if eps == state.eps_t:
        return
    state.eps_t = eps
    var Fy = mat_def.p0
    var E0 = mat_def.p1
    var b = mat_def.p2
    if b >= 1.0:
        abort("Steel01 b must be < 1")
    var H = (b * E0) / (1.0 - b)
    var eps_p = state.eps_p_c
    var alpha = state.alpha_c
    var sigma_trial = E0 * (eps - eps_p)
    var xi = sigma_trial - alpha
    var f = _abs(xi) - Fy
    if f <= 0.0:
        state.sig_t = sigma_trial
        state.tangent_t = E0
        state.eps_p_t = eps_p
        state.alpha_t = alpha
        return
    var dg = f / (E0 + H)
    var sgn = _sign(xi)
    state.eps_p_t = eps_p + dg * sgn
    state.alpha_t = alpha + H * dg * sgn
    state.sig_t = sigma_trial - E0 * dg * sgn
    state.tangent_t = (E0 * H) / (E0 + H)
    return


fn uniaxial_set_trial_strain_concrete01(
    mat_def: UniMaterialDefinition, mut state: UniMaterialPointState, eps: Float64
):
    if eps == state.eps_t:
        return
    state.eps_t = eps
    var fpc = mat_def.p0
    var epsc0 = mat_def.p1
    var fpcu = mat_def.p2
    var epscu = mat_def.p3
    var d_strain = eps - state.eps_c
    if _abs(d_strain) < 1.0e-14:
        state.sig_t = state.sig_c
        state.tangent_t = state.tangent_c
        state.eps_p_t = state.eps_p_c
        state.alpha_t = state.alpha_c
        state.min_strain_t = state.min_strain_c
        state.end_strain_t = state.end_strain_c
        state.unload_slope_t = state.unload_slope_c
        return
    state.min_strain_t = state.min_strain_c
    state.end_strain_t = state.end_strain_c
    state.unload_slope_t = state.unload_slope_c
    if eps > 0.0:
        state.sig_t = 0.0
        state.tangent_t = 0.0
        state.eps_p_t = state.eps_p_c
        state.alpha_t = state.alpha_c
        return
    var temp_stress = state.sig_c + state.unload_slope_t * eps - state.unload_slope_t * state.eps_c
    if eps <= state.eps_c:
        var reload = _concrete01_reload(
            fpc,
            epsc0,
            fpcu,
            epscu,
            eps,
            state.min_strain_t,
            state.end_strain_t,
            state.unload_slope_t,
        )
        state.sig_t = reload.stress
        state.tangent_t = reload.tangent
        if temp_stress > state.sig_t:
            state.sig_t = temp_stress
            state.tangent_t = state.unload_slope_t
    elif temp_stress <= 0.0:
        state.sig_t = temp_stress
        state.tangent_t = state.unload_slope_t
    else:
        state.sig_t = 0.0
        state.tangent_t = 0.0
    state.eps_p_t = state.eps_p_c
    state.alpha_t = state.alpha_c
    return


fn uniaxial_set_trial_strain_steel02(
    mat_def: UniMaterialDefinition, mut state: UniMaterialPointState, eps: Float64
):
    if eps == state.eps_t:
        return
    _steel02_set_trial(mat_def, state, eps)


fn uniaxial_set_trial_strain_concrete02(
    mat_def: UniMaterialDefinition, mut state: UniMaterialPointState, eps: Float64
):
    if eps == state.eps_t:
        return
    _concrete02_set_trial(mat_def, state, eps)


fn uniaxial_set_trial_strain_py_simple1(
    mat_def: UniMaterialDefinition, mut state: UniMaterialPointState, eps: Float64
):
    py_simple1_set_trial(
        mat_def.p0,
        mat_def.p1,
        mat_def.p2,
        mat_def.p3,
        mat_def.p4,
        state.eps_t,
        state.sig_t,
        state.tangent_t,
        state.py_tz_qz,
        eps,
    )


fn uniaxial_set_trial_strain_tz_simple1(
    mat_def: UniMaterialDefinition, mut state: UniMaterialPointState, eps: Float64
):
    tz_simple1_set_trial(
        mat_def.p0,
        mat_def.p1,
        mat_def.p2,
        mat_def.p3,
        state.eps_t,
        state.sig_t,
        state.tangent_t,
        state.py_tz_qz,
        eps,
    )


fn uniaxial_set_trial_strain_qz_simple1(
    mat_def: UniMaterialDefinition, mut state: UniMaterialPointState, eps: Float64
):
    qz_simple1_set_trial(
        mat_def.p0,
        mat_def.p1,
        mat_def.p2,
        mat_def.p3,
        mat_def.p4,
        state.eps_t,
        state.sig_t,
        state.tangent_t,
        state.py_tz_qz,
        eps,
    )


fn uniaxial_set_trial_strain_viscous_damper(
    mat_def: UniMaterialDefinition,
    mut state: UniMaterialPointState,
    eps: Float64,
    strain_rate: Float64,
    dt: Float64,
):
    _viscous_damper_set_trial(mat_def, state, eps, strain_rate, dt)


fn _set_trial_component(
    mat_def: UniMaterialDefinition, mut state: UniMaterialPointState, eps: Float64
):
    if mat_def.mat_type == UniMaterialTypeTag.Elastic:
        uniaxial_set_trial_strain_elastic(mat_def, state, eps)
        return
    if mat_def.mat_type == UniMaterialTypeTag.Steel01:
        uniaxial_set_trial_strain_steel01(mat_def, state, eps)
        return
    if mat_def.mat_type == UniMaterialTypeTag.Steel01Thermal:
        steel01_thermal_set_trial(mat_def, state, eps, 0.0)
        return
    if mat_def.mat_type == UniMaterialTypeTag.Concrete01:
        uniaxial_set_trial_strain_concrete01(mat_def, state, eps)
        return
    if mat_def.mat_type == UniMaterialTypeTag.Steel02:
        if not mat_def.fatigue.enabled:
            uniaxial_set_trial_strain_steel02(mat_def, state, eps)
        else:
            state.fatigue.trial_strain = eps
            if state.fatigue.failed:
                _steel02_set_trial(mat_def, state, eps)
                state.sig_t *= 1.0e-8
                state.tangent_t = 1.0e-8 * mat_def.p1
            else:
                uniaxial_set_trial_strain_steel02(mat_def, state, eps)
        return
    if mat_def.mat_type == UniMaterialTypeTag.Concrete02:
        uniaxial_set_trial_strain_concrete02(mat_def, state, eps)
        return
    if mat_def.mat_type == UniMaterialTypeTag.Bilin:
        state.bilin_history[1] = state.bilin_history[0]
        ref trial = state.bilin_history[1]
        trial.set_trial(mat_def.bilin, state.eps_c, state.bilin_has_commit, eps)
        state.eps_t = trial.U
        state.sig_t = trial.fP
        state.tangent_t = trial.Tangent
        return
    if mat_def.mat_type == UniMaterialTypeTag.Hysteretic:
        _hysteretic_set_trial(mat_def, state, eps)
        return
    if mat_def.mat_type == UniMaterialTypeTag.PySimple1:
        uniaxial_set_trial_strain_py_simple1(mat_def, state, eps)
        return
    if mat_def.mat_type == UniMaterialTypeTag.TzSimple1:
        uniaxial_set_trial_strain_tz_simple1(mat_def, state, eps)
        return
    if mat_def.mat_type == UniMaterialTypeTag.QzSimple1:
        uniaxial_set_trial_strain_qz_simple1(mat_def, state, eps)
        return
    if mat_def.mat_type == UniMaterialTypeTag.ViscousDamper:
        uniaxial_set_trial_strain_viscous_damper(mat_def, state, eps, 0.0, 0.0)
        return
    if mat_def.mat_type == UniMaterialTypeTag.HystereticSmooth:
        hysteretic_smooth_set_trial(mat_def, state, eps)
        return
    if mat_def.mat_type == UniMaterialTypeTag.HystereticAsym:
        hysteretic_asym_set_trial(mat_def, state, eps)
        return
    if mat_def.mat_type == UniMaterialTypeTag.ENT:
        ent_set_trial(mat_def, state, eps)
        return
    if mat_def.mat_type == UniMaterialTypeTag.ModIMKPeakOriented:
        state.imk_peak_history[1] = state.imk_peak_history[0]
        ref history = state.imk_peak_history[1]
        history.evaluate(mat_def.imk_peak, eps, state.imk_peak_has_commit)
        state.eps_t = history.strain
        state.sig_t = history.stress
        state.tangent_t = history.tangent
        return
    if mat_def.mat_type == UniMaterialTypeTag.ConcreteCM:
        var concrete = ConcreteCMTrial(mat_def.concrete_cm_params(), state.wall_history.concrete[0])
        concrete.evaluate(eps)
        state.wall_history.concrete[1] = concrete.trial
        state.eps_t = eps
        state.sig_t = concrete.trial.stress
        state.tangent_t = concrete.trial.tangent
        return
    if mat_def.mat_type == UniMaterialTypeTag.SteelMPF:
        state.wall_history.steel[1] = state.wall_history.steel[0]
        ref steel = state.wall_history.steel[1]
        var response = steel.evaluate(
            mat_def.steel_mpf_params(),
            SteelMPFPoint(state.eps_c, state.sig_c),
            eps,
        )
        state.eps_t = eps
        state.sig_t = response[0]
        state.tangent_t = response[1]
        return
    if mat_def.mat_type == UniMaterialTypeTag.ElasticPP:
        elastic_pp_set_trial(mat_def, state, eps)
        return
    if mat_def.mat_type == UniMaterialTypeTag.ElasticBilin:
        elastic_bilin_set_trial(mat_def, state, eps)
        return
    if mat_def.mat_type == UniMaterialTypeTag.Ratchet:
        var response = state.ratchet.value().evaluate(eps)
        state.eps_t = eps
        state.sig_t = response[0]
        state.tangent_t = response[1]
        return
    if mat_def.mat_type == UniMaterialTypeTag.Viscous:
        state.eps_t = eps
        return
    abort("unsupported uniaxial material")


fn _hysteretic_set_trial(
    mat_def: UniMaterialDefinition, mut state: UniMaterialPointState, eps: Float64
):
    if state.hysteretic_t.direction == 0 and eps == 0.0:
        return
    ref trial = state.hysteretic_t
    var previous_direction = trial.direction
    trial = state.hysteretic_c
    state.eps_t = eps
    var delta = eps - state.eps_c
    if abs(delta) < 2.220446049250313e-16:
        trial.direction = previous_direction
        return

    var direction = 1 if delta > 0.0 else -1
    if eps >= trial.strain_max:
        trial.strain_max = eps
        trial.direction = 1
        state.sig_t = mat_def.hysteretic.positive.stress(eps)
        state.tangent_t = mat_def.hysteretic.positive.tangent(eps)
    elif eps <= trial.strain_min:
        trial.strain_min = eps
        trial.direction = -1
        state.sig_t = -mat_def.hysteretic.negative.stress(-eps)
        state.tangent_t = mat_def.hysteretic.negative.tangent(-eps)
    else:
        var response = trial.increment(mat_def.hysteretic, direction, state.eps_c, state.sig_c, eps)
        state.sig_t = response[0]
        state.tangent_t = response[1]
    trial.energy += 0.5 * (state.sig_c + state.sig_t) * delta


fn _accept_trial(mut state: UniMaterialPointState, eps: Float64) -> Bool:
    if state.min_max:
        return state.min_max.value().set_trial(eps, state.sig_t, state.tangent_t)
    return True


fn _set_trial_material(mat_def: UniMaterialDef, mut state: UniMaterialState, eps: Float64):
    if mat_def.mat_type != UniMaterialTypeTag.Parallel:
        _set_trial_component(mat_def, state, eps)
        return
    ref definitions = mat_def.components.value()[]
    ref children = state.components.value()
    state.eps_t = eps
    state.sig_t = 0.0
    state.tangent_t = 0.0
    for i in range(len(definitions)):
        ref component = definitions[i]
        ref child = children[i]
        if _accept_trial(child, eps):
            _set_trial_component(component.definition, child, eps)
        state.sig_t += component.factor * child.sig_t
        state.tangent_t += component.factor * child.tangent_t


fn uniaxial_set_trial_strain(mat_def: UniMaterialDef, mut state: UniMaterialState, eps: Float64):
    if _accept_trial(state, eps):
        _set_trial_material(mat_def, state, eps)


fn uniaxial_set_trial_strain_rate(
    mat_def: UniMaterialDef,
    mut state: UniMaterialState,
    eps: Float64,
    strain_rate: Float64,
    dt: Float64,
):
    if mat_def.mat_type == UniMaterialTypeTag.ViscousDamper:
        if _accept_trial(state, eps):
            uniaxial_set_trial_strain_viscous_damper(mat_def, state, eps, strain_rate, dt)
        return
    uniaxial_set_trial_strain(mat_def, state, eps)


fn _commit_point(mut state: UniMaterialPointState):
    if state.fatigue.params.enabled:
        if state.fatigue.failed:
            return
        state.fatigue.commit(state.sig_t)
        if state.fatigue.failed:
            state.sig_t *= 1.0e-8
            state.tangent_t = 1.0e-8 * state.fatigue.initial_tangent
            return
    if state.ratchet:
        state.ratchet.value().commit(state.eps_t, state.eps_c)
    state.wall_history.commit()
    if len(state.imk_peak_history) != 0:
        state.imk_peak_history[0] = state.imk_peak_history[1]
        state.imk_peak_has_commit = True
    state.eps_c = state.eps_t
    state.sig_c = state.sig_t
    state.tangent_c = state.tangent_t
    state.eps_p_c = state.eps_p_t
    state.alpha_c = state.alpha_t
    state.min_strain_c = state.min_strain_t
    state.end_strain_c = state.end_strain_t
    state.unload_slope_c = state.unload_slope_t

    state.s2_epsmin_c = state.s2_epsmin_t
    state.s2_epsmax_c = state.s2_epsmax_t
    state.s2_epspl_c = state.s2_epspl_t
    state.s2_epss0_c = state.s2_epss0_t
    state.s2_sigs0_c = state.s2_sigs0_t
    state.s2_epsr_c = state.s2_epsr_t
    state.s2_sigr_c = state.s2_sigr_t
    state.s2_kon_c = state.s2_kon_t

    state.c2_ecmin_c = state.c2_ecmin_t
    state.c2_dept_c = state.c2_dept_t
    state.hysteretic_c = state.hysteretic_t
    if len(state.bilin_history) != 0:
        state.bilin_history[0] = state.bilin_history[1]
        state.bilin_has_commit = True
    if state.has_py_tz_qz:
        state.py_tz_qz.commit()

    state.vd_vel_c = state.vd_vel_t
    state.vd_pugr_c = state.vd_pugr_t
    state.vd_nugr_c = state.vd_nugr_t


fn _revert_point(mut state: UniMaterialPointState):
    if state.fatigue.failed or state.retain_trial_response:
        return
    if state.ratchet:
        state.ratchet.value().revert()
    state.wall_history.revert()
    if len(state.imk_peak_history) != 0:
        state.imk_peak_history[1] = state.imk_peak_history[0]
    state.eps_t = state.eps_c
    state.sig_t = state.sig_c
    # Hysteretic and ElasticPP leave the trial tangent intact on revert.
    if not state.retain_trial_tangent:
        state.tangent_t = state.tangent_c
    state.eps_p_t = state.eps_p_c
    state.alpha_t = state.alpha_c
    state.min_strain_t = state.min_strain_c
    state.end_strain_t = state.end_strain_c
    state.unload_slope_t = state.unload_slope_c

    state.s2_epsmin_t = state.s2_epsmin_c
    state.s2_epsmax_t = state.s2_epsmax_c
    state.s2_epspl_t = state.s2_epspl_c
    state.s2_epss0_t = state.s2_epss0_c
    state.s2_sigs0_t = state.s2_sigs0_c
    state.s2_epsr_t = state.s2_epsr_c
    state.s2_sigr_t = state.s2_sigr_c
    state.s2_kon_t = state.s2_kon_c

    state.c2_ecmin_t = state.c2_ecmin_c
    state.c2_dept_t = state.c2_dept_c
    state.hysteretic_t = state.hysteretic_c
    if len(state.bilin_history) != 0:
        state.bilin_history[1] = state.bilin_history[0]
    if state.has_py_tz_qz:
        state.py_tz_qz.revert_trial()

    state.vd_vel_t = state.vd_vel_c
    state.vd_pugr_t = state.vd_pugr_c
    state.vd_nugr_t = state.vd_nugr_c


fn fsam_set_trial_strain(mut state: UniMaterialState, strain: PanelVector):
    ref panel = state.fsam[0]
    panel.set_strain(strain)
    ref definitions = state.fsam_components.value()[]
    ref children = state.components.value()
    var tangents = InlineArray[Float64, 8](fill=0.0)
    var concrete_start = 2 if panel.crack_count == 0 else 4
    var concrete_end = 4 + 2 * panel.crack_count
    for i in range(concrete_start, concrete_end):
        _set_trial_component(definitions[i].definition, children[i], panel.trial.material_strain[i])
        panel.trial.material_stress[i] = children[i].sig_t
        tangents[i] = children[i].tangent_t
    for i in range(2):
        _set_trial_component(definitions[i].definition, children[i], panel.trial.material_strain[i])
        panel.trial.material_stress[i] = children[i].sig_t
        tangents[i] = children[i].tangent_t
    panel.set_response(tangents)


fn _commit_fsam(mut state: UniMaterialState):
    ref panel = state.fsam[0]
    panel.converged_stress = panel.trial.stress
    var strain = panel.trial.strain
    fsam_set_trial_strain(state, strain)
    var crack_count = panel.crack_count
    if crack_count == 0 and panel.try_first_crack():
        panel.cracks[0] = panel.trial_cracks[0]
        fsam_set_trial_strain(state, strain)
    ref children = state.components.value()
    for i in range(len(children)):
        _commit_point(children[i])
    panel.cracks = panel.trial_cracks
    if crack_count == 1 and panel.try_second_crack(children[5].wall_history.concrete[0].eunp):
        fsam_set_trial_strain(state, strain)
        for i in range(len(children)):
            _commit_point(children[i])
        panel.cracks = panel.trial_cracks
    panel.committed = panel.trial


fn _commit_material_point(mut state: UniMaterialPointState) -> Bool:
    if state.min_max:
        ref limits = state.min_max.value()
        limits.committed_failed = limits.trial_failed
        if limits.committed_failed:
            return False
    _commit_point(state)
    return True


fn _revert_material_point(mut state: UniMaterialPointState) -> Bool:
    if state.min_max:
        ref limits = state.min_max.value()
        if limits.committed_failed:
            return False
        if limits.trial_failed:
            limits.restore_response(state.sig_t, state.tangent_t)
            _revert_point(state)
            limits.hide_response(state.sig_t, state.tangent_t)
            return True
    _revert_point(state)
    return True


fn uniaxial_commit(mut state: UniMaterialState):
    if len(state.fsam) != 0:
        _commit_fsam(state)
        return
    if state.coupled_force and state.eps_t != 0.0 and state.sig_t != 0.0:
        ref force = state.coupled_force.value()
        force.committed = force.trial
    if _commit_material_point(state) and state.components:
        ref children = state.components.value()
        for i in range(len(children)):
            _ = _commit_material_point(children[i])


fn uniaxial_revert_trial(mut state: UniMaterialState):
    if len(state.fsam) != 0:
        state.fsam[0].revert()
    if _revert_material_point(state) and state.components:
        ref children = state.components.value()
        for i in range(len(children)):
            _ = _revert_material_point(children[i])


fn uniaxial_commit_all(mut states: List[UniMaterialState]):
    for i in range(len(states)):
        ref state = states[i]
        uniaxial_commit(state)


fn uniaxial_revert_trial_all(mut states: List[UniMaterialState]):
    for i in range(len(states)):
        ref state = states[i]
        uniaxial_revert_trial(state)
