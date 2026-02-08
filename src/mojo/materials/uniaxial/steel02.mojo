from materials.uniaxial.core import _abs, _pow_abs, UniMaterialDef, UniMaterialState

fn _steel02_set_trial(
    mat_def: UniMaterialDef, mut state: UniMaterialState, trial_eps: Float64
):
    var Fy = mat_def.p0
    var E0 = mat_def.p1
    var b = mat_def.p2
    var R0 = mat_def.p3
    var cR1 = mat_def.p4
    var cR2 = mat_def.p5
    var a1 = mat_def.p6
    var a2 = mat_def.p7
    var a3 = mat_def.p8
    var a4 = mat_def.p9
    var sigini = mat_def.p10

    var Esh = b * E0
    var epsy = Fy / E0

    var eps = trial_eps
    if sigini != 0.0:
        var epsini = sigini / E0
        eps = trial_eps + epsini
    state.eps_t = eps

    var deps = eps - state.eps_c

    var epsmax = state.s2_epsmax_c
    var epsmin = state.s2_epsmin_c
    var epspl = state.s2_epspl_c
    var epss0 = state.s2_epss0_c
    var sigs0 = state.s2_sigs0_c
    var epsr = state.s2_epsr_c
    var sigr = state.s2_sigr_c
    var kon = state.s2_kon_c

    if kon == 0 or kon == 3:
        if _abs(deps) < 2.220446049250313e-15:
            state.tangent_t = E0
            state.sig_t = sigini
            state.s2_kon_t = 3
            state.s2_epsmin_t = epsmin
            state.s2_epsmax_t = epsmax
            state.s2_epspl_t = epspl
            state.s2_epss0_t = epss0
            state.s2_sigs0_t = sigs0
            state.s2_epsr_t = epsr
            state.s2_sigr_t = sigr
            state.eps_p_t = state.eps_p_c
            state.alpha_t = state.alpha_c
            return
        else:
            epsmax = epsy
            epsmin = -epsy
            if deps < 0.0:
                kon = 2
                epss0 = epsmin
                sigs0 = -Fy
                epspl = epsmin
            else:
                kon = 1
                epss0 = epsmax
                sigs0 = Fy
                epspl = epsmax

    if kon == 2 and deps > 0.0:
        kon = 1
        epsr = state.eps_c
        sigr = state.sig_c
        if state.eps_c < epsmin:
            epsmin = state.eps_c
        var d1 = (epsmax - epsmin) / (2.0 * (a4 * epsy))
        var shft = 1.0 + a3 * (d1 ** 0.8)
        epss0 = (Fy * shft - Esh * epsy * shft - sigr + E0 * epsr) / (E0 - Esh)
        sigs0 = Fy * shft + Esh * (epss0 - epsy * shft)
        epspl = epsmax
    elif kon == 1 and deps < 0.0:
        kon = 2
        epsr = state.eps_c
        sigr = state.sig_c
        if state.eps_c > epsmax:
            epsmax = state.eps_c
        var d1 = (epsmax - epsmin) / (2.0 * (a2 * epsy))
        var shft = 1.0 + a1 * (d1 ** 0.8)
        epss0 = (-Fy * shft + Esh * epsy * shft - sigr + E0 * epsr) / (E0 - Esh)
        sigs0 = -Fy * shft + Esh * (epss0 + epsy * shft)
        epspl = epsmin

    var xi = _abs((epspl - epss0) / epsy)
    var R = R0 * (1.0 - (cR1 * xi) / (cR2 + xi))
    var epsrat = (eps - epsr) / (epss0 - epsr)
    var dum1 = 1.0 + _pow_abs(epsrat, R)
    var dum2 = dum1 ** (1.0 / R)

    var sig = b * epsrat + (1.0 - b) * epsrat / dum2
    sig = sig * (sigs0 - sigr) + sigr

    var e = b + (1.0 - b) / (dum1 * dum2)
    e = e * (sigs0 - sigr) / (epss0 - epsr)

    state.sig_t = sig
    state.tangent_t = e
    state.eps_p_t = state.eps_p_c
    state.alpha_t = state.alpha_c

    state.s2_epsmin_t = epsmin
    state.s2_epsmax_t = epsmax
    state.s2_epspl_t = epspl
    state.s2_epss0_t = epss0
    state.s2_sigs0_t = sigs0
    state.s2_epsr_t = epsr
    state.s2_sigr_t = sigr
    state.s2_kon_t = kon


