from collections import List
from os import abort

from materials.uniaxial.concrete01 import _concrete01_reload
from materials.uniaxial.concrete02 import _concrete02_set_trial
from materials.uniaxial.core import _abs, _sign, UniMaterialDef, UniMaterialState
from materials.uniaxial.steel02 import _steel02_set_trial
from tag_types import UniMaterialTypeTag

fn uniaxial_set_trial_strain_elastic(
    mat_def: UniMaterialDef, mut state: UniMaterialState, eps: Float64
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
    mat_def: UniMaterialDef, mut state: UniMaterialState, eps: Float64
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
    mat_def: UniMaterialDef, mut state: UniMaterialState, eps: Float64
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
    var temp_stress = (
        state.sig_c
        + state.unload_slope_t * eps
        - state.unload_slope_t * state.eps_c
    )
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
    mat_def: UniMaterialDef, mut state: UniMaterialState, eps: Float64
):
    if eps == state.eps_t:
        return
    _steel02_set_trial(mat_def, state, eps)


fn uniaxial_set_trial_strain_concrete02(
    mat_def: UniMaterialDef, mut state: UniMaterialState, eps: Float64
):
    if eps == state.eps_t:
        return
    _concrete02_set_trial(mat_def, state, eps)


fn uniaxial_set_trial_strain(
    mat_def: UniMaterialDef, mut state: UniMaterialState, eps: Float64
):
    if mat_def.mat_type == UniMaterialTypeTag.Elastic:
        uniaxial_set_trial_strain_elastic(mat_def, state, eps)
        return
    if mat_def.mat_type == UniMaterialTypeTag.Steel01:
        uniaxial_set_trial_strain_steel01(mat_def, state, eps)
        return
    if mat_def.mat_type == UniMaterialTypeTag.Concrete01:
        uniaxial_set_trial_strain_concrete01(mat_def, state, eps)
        return
    if mat_def.mat_type == UniMaterialTypeTag.Steel02:
        uniaxial_set_trial_strain_steel02(mat_def, state, eps)
        return
    if mat_def.mat_type == UniMaterialTypeTag.Concrete02:
        uniaxial_set_trial_strain_concrete02(mat_def, state, eps)
        return
    abort("unsupported uniaxial material")


fn uniaxial_commit(mut state: UniMaterialState):
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


fn uniaxial_revert_trial(mut state: UniMaterialState):
    state.eps_t = state.eps_c
    state.sig_t = state.sig_c
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


fn uniaxial_commit_all(mut states: List[UniMaterialState]):
    for i in range(len(states)):
        ref state = states[i]
        uniaxial_commit(state)


fn uniaxial_revert_trial_all(mut states: List[UniMaterialState]):
    for i in range(len(states)):
        ref state = states[i]
        uniaxial_revert_trial(state)
