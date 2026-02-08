from materials.uniaxial.core import _abs, StressTangent, UniMaterialDef, UniMaterialState

fn _concrete02_tens_envlp(
    fc: Float64, epsc0: Float64, ft: Float64, Ets: Float64, strain: Float64
) -> StressTangent:
    var Ec0 = (2.0 * fc) / epsc0
    var eps0 = ft / Ec0
    var epsu = ft * (1.0 / Ets + 1.0 / Ec0)
    if strain <= eps0:
        return StressTangent(strain * Ec0, Ec0)
    elif strain <= epsu:
        return StressTangent(ft - Ets * (strain - eps0), -Ets)
    else:
        return StressTangent(0.0, 1.0e-10)


fn _concrete02_compr_envlp(
    fc: Float64, epsc0: Float64, fcu: Float64, epscu: Float64, strain: Float64
) -> StressTangent:
    var Ec0 = (2.0 * fc) / epsc0
    var rat = strain / epsc0
    if strain >= epsc0:
        var sig = fc * rat * (2.0 - rat)
        var Ect = Ec0 * (1.0 - rat)
        return StressTangent(sig, Ect)
    elif strain > epscu:
        var sig = (fcu - fc) * (strain - epsc0) / (epscu - epsc0) + fc
        var Ect = (fcu - fc) / (epscu - epsc0)
        return StressTangent(sig, Ect)
    else:
        return StressTangent(fcu, 1.0e-10)


fn _concrete02_set_trial(
    mat_def: UniMaterialDef, mut state: UniMaterialState, strain: Float64
):
    var fc = mat_def.p0
    var epsc0 = mat_def.p1
    var fcu = mat_def.p2
    var epscu = mat_def.p3
    var rat = mat_def.p4
    var ft = mat_def.p5
    var Ets = mat_def.p6

    var Ec0 = (2.0 * fc) / epsc0

    state.c2_ecmin_t = state.c2_ecmin_c
    state.c2_dept_t = state.c2_dept_c

    state.eps_t = strain
    var deps = strain - state.eps_c
    if _abs(deps) < 2.220446049250313e-16:
        state.sig_t = state.sig_c
        state.tangent_t = state.tangent_c
        state.eps_p_t = state.eps_p_c
        state.alpha_t = state.alpha_c
        return

    if strain < state.c2_ecmin_t:
        var env = _concrete02_compr_envlp(fc, epsc0, fcu, epscu, strain)
        state.sig_t = env.stress
        state.tangent_t = env.tangent
        state.c2_ecmin_t = strain
        state.eps_p_t = state.eps_p_c
        state.alpha_t = state.alpha_c
        return

    var epsr = (fcu - rat * Ec0 * epscu) / (Ec0 * (1.0 - rat))
    var sigmr = Ec0 * epsr

    var sigmm_env = _concrete02_compr_envlp(fc, epsc0, fcu, epscu, state.c2_ecmin_t)
    var sigmm = sigmm_env.stress

    var er = (sigmm - sigmr) / (state.c2_ecmin_t - epsr)
    var ept = state.c2_ecmin_t - sigmm / er

    if strain <= ept:
        var sigmin = sigmm + er * (strain - state.c2_ecmin_t)
        var sigmax = 0.5 * er * (strain - ept)
        state.sig_t = state.sig_c + Ec0 * deps
        state.tangent_t = Ec0
        if state.sig_t <= sigmin:
            state.sig_t = sigmin
            state.tangent_t = er
        if state.sig_t >= sigmax:
            state.sig_t = sigmax
            state.tangent_t = 0.5 * er
        state.eps_p_t = state.eps_p_c
        state.alpha_t = state.alpha_c
        return

    var epn = ept + state.c2_dept_t
    if strain <= epn:
        var sicn_env = _concrete02_tens_envlp(
            fc, epsc0, ft, Ets, state.c2_dept_t
        )
        var e = Ec0
        if state.c2_dept_t != 0.0:
            e = sicn_env.stress / state.c2_dept_t
        state.tangent_t = e
        state.sig_t = e * (strain - ept)
    else:
        var env = _concrete02_tens_envlp(fc, epsc0, ft, Ets, strain - ept)
        state.sig_t = env.stress
        state.tangent_t = env.tangent
        state.c2_dept_t = strain - ept

    state.eps_p_t = state.eps_p_c
    state.alpha_t = state.alpha_c


