from collections import List
from os import abort


fn _abs(x: Float64) -> Float64:
    if x < 0.0:
        return -x
    return x


fn _sign(x: Float64) -> Float64:
    if x < 0.0:
        return -1.0
    return 1.0


fn _pow_abs(x: Float64, p: Float64) -> Float64:
    return _abs(x) ** p


struct StressTangent(Defaultable, Movable, ImplicitlyCopyable):
    var stress: Float64
    var tangent: Float64

    fn __init__(out self):
        self.stress = 0.0
        self.tangent = 0.0

    fn __init__(out self, stress: Float64, tangent: Float64):
        self.stress = stress
        self.tangent = tangent


struct EndStrainSlope(Defaultable, Movable, ImplicitlyCopyable):
    var end_strain: Float64
    var unload_slope: Float64

    fn __init__(out self):
        self.end_strain = 0.0
        self.unload_slope = 0.0

    fn __init__(out self, end_strain: Float64, unload_slope: Float64):
        self.end_strain = end_strain
        self.unload_slope = unload_slope


struct UniMaterialDef(Defaultable, Movable, ImplicitlyCopyable):
    var mat_type: Int
    var p0: Float64
    var p1: Float64
    var p2: Float64
    var p3: Float64
    var p4: Float64
    var p5: Float64
    var p6: Float64
    var p7: Float64
    var p8: Float64
    var p9: Float64
    var p10: Float64

    fn __init__(out self):
        self.mat_type = 0
        self.p0 = 0.0
        self.p1 = 0.0
        self.p2 = 0.0
        self.p3 = 0.0
        self.p4 = 0.0
        self.p5 = 0.0
        self.p6 = 0.0
        self.p7 = 0.0
        self.p8 = 0.0
        self.p9 = 0.0
        self.p10 = 0.0

    fn __init__(
        out self, mat_type: Int, p0: Float64, p1: Float64, p2: Float64, p3: Float64
    ):
        self.mat_type = mat_type
        self.p0 = p0
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = 0.0
        self.p5 = 0.0
        self.p6 = 0.0
        self.p7 = 0.0
        self.p8 = 0.0
        self.p9 = 0.0
        self.p10 = 0.0

    fn __init__(
        out self,
        mat_type: Int,
        p0: Float64,
        p1: Float64,
        p2: Float64,
        p3: Float64,
        p4: Float64,
        p5: Float64,
        p6: Float64,
        p7: Float64,
        p8: Float64,
        p9: Float64,
        p10: Float64,
    ):
        self.mat_type = mat_type
        self.p0 = p0
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
        self.p5 = p5
        self.p6 = p6
        self.p7 = p7
        self.p8 = p8
        self.p9 = p9
        self.p10 = p10


struct UniMaterialState(Defaultable, Movable, ImplicitlyCopyable):
    var eps_c: Float64
    var sig_c: Float64
    var tangent_c: Float64
    var eps_p_c: Float64
    var alpha_c: Float64

    var eps_t: Float64
    var sig_t: Float64
    var tangent_t: Float64
    var eps_p_t: Float64
    var alpha_t: Float64
    var min_strain_c: Float64
    var end_strain_c: Float64
    var unload_slope_c: Float64
    var min_strain_t: Float64
    var end_strain_t: Float64
    var unload_slope_t: Float64

    # Steel02 history variables
    var s2_epsmin_c: Float64
    var s2_epsmax_c: Float64
    var s2_epspl_c: Float64
    var s2_epss0_c: Float64
    var s2_sigs0_c: Float64
    var s2_epsr_c: Float64
    var s2_sigr_c: Float64
    var s2_kon_c: Int

    var s2_epsmin_t: Float64
    var s2_epsmax_t: Float64
    var s2_epspl_t: Float64
    var s2_epss0_t: Float64
    var s2_sigs0_t: Float64
    var s2_epsr_t: Float64
    var s2_sigr_t: Float64
    var s2_kon_t: Int

    # Concrete02 history variables
    var c2_ecmin_c: Float64
    var c2_dept_c: Float64
    var c2_ecmin_t: Float64
    var c2_dept_t: Float64

    fn __init__(out self):
        self.eps_c = 0.0
        self.sig_c = 0.0
        self.tangent_c = 0.0
        self.eps_p_c = 0.0
        self.alpha_c = 0.0
        self.eps_t = 0.0
        self.sig_t = 0.0
        self.tangent_t = 0.0
        self.eps_p_t = 0.0
        self.alpha_t = 0.0
        self.min_strain_c = 0.0
        self.end_strain_c = 0.0
        self.unload_slope_c = 0.0
        self.min_strain_t = 0.0
        self.end_strain_t = 0.0
        self.unload_slope_t = 0.0
        self.s2_epsmin_c = 0.0
        self.s2_epsmax_c = 0.0
        self.s2_epspl_c = 0.0
        self.s2_epss0_c = 0.0
        self.s2_sigs0_c = 0.0
        self.s2_epsr_c = 0.0
        self.s2_sigr_c = 0.0
        self.s2_kon_c = 0
        self.s2_epsmin_t = 0.0
        self.s2_epsmax_t = 0.0
        self.s2_epspl_t = 0.0
        self.s2_epss0_t = 0.0
        self.s2_sigs0_t = 0.0
        self.s2_epsr_t = 0.0
        self.s2_sigr_t = 0.0
        self.s2_kon_t = 0
        self.c2_ecmin_c = 0.0
        self.c2_dept_c = 0.0
        self.c2_ecmin_t = 0.0
        self.c2_dept_t = 0.0

    fn __init__(out self, mat_def: UniMaterialDef):
        self.eps_c = 0.0
        self.sig_c = 0.0
        self.eps_p_c = 0.0
        self.alpha_c = 0.0
        self.eps_t = 0.0
        self.sig_t = 0.0
        self.eps_p_t = 0.0
        self.alpha_t = 0.0
        self.min_strain_c = 0.0
        self.end_strain_c = 0.0
        self.unload_slope_c = 0.0
        self.min_strain_t = 0.0
        self.end_strain_t = 0.0
        self.unload_slope_t = 0.0

        self.s2_epsmin_c = 0.0
        self.s2_epsmax_c = 0.0
        self.s2_epspl_c = 0.0
        self.s2_epss0_c = 0.0
        self.s2_sigs0_c = 0.0
        self.s2_epsr_c = 0.0
        self.s2_sigr_c = 0.0
        self.s2_kon_c = 0
        self.s2_epsmin_t = 0.0
        self.s2_epsmax_t = 0.0
        self.s2_epspl_t = 0.0
        self.s2_epss0_t = 0.0
        self.s2_sigs0_t = 0.0
        self.s2_epsr_t = 0.0
        self.s2_sigr_t = 0.0
        self.s2_kon_t = 0

        self.c2_ecmin_c = 0.0
        self.c2_dept_c = 0.0
        self.c2_ecmin_t = 0.0
        self.c2_dept_t = 0.0

        var tangent = uni_mat_initial_tangent(mat_def)
        self.tangent_c = tangent
        self.tangent_t = tangent

        if mat_def.mat_type == 2:
            self.unload_slope_c = tangent
            self.unload_slope_t = tangent

        if mat_def.mat_type == 3:
            var Fy = mat_def.p0
            var E0 = mat_def.p1
            var epsy = Fy / E0
            self.s2_epsmax_c = epsy
            self.s2_epsmax_t = epsy
            self.s2_epsmin_c = -epsy
            self.s2_epsmin_t = -epsy
            var sigini = mat_def.p10
            if sigini != 0.0:
                var epsini = sigini / E0
                self.eps_c = epsini
                self.eps_t = epsini
                self.sig_c = sigini
                self.sig_t = sigini


fn uni_mat_is_elastic(mat_def: UniMaterialDef) -> Bool:
    return mat_def.mat_type == 0


fn uni_mat_initial_tangent(mat_def: UniMaterialDef) -> Float64:
    if mat_def.mat_type == 0:
        return mat_def.p0
    if mat_def.mat_type == 1:
        return mat_def.p1
    if mat_def.mat_type == 2:
        return (2.0 * mat_def.p0) / mat_def.p1
    if mat_def.mat_type == 3:
        return mat_def.p1
    if mat_def.mat_type == 4:
        return (2.0 * mat_def.p0) / mat_def.p1
    abort("unsupported uniaxial material")
    return 0.0


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


fn uniaxial_set_trial_strain(
    mat_def: UniMaterialDef, mut state: UniMaterialState, eps: Float64
):
    state.eps_t = eps
    if mat_def.mat_type == 0:
        var E = mat_def.p0
        state.sig_t = E * eps
        state.tangent_t = E
        state.eps_p_t = state.eps_p_c
        state.alpha_t = state.alpha_c
        return
    if mat_def.mat_type == 1:
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
    if mat_def.mat_type == 2:
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
    if mat_def.mat_type == 3:
        _steel02_set_trial(mat_def, state, eps)
        return
    if mat_def.mat_type == 4:
        _concrete02_set_trial(mat_def, state, eps)
        return
    abort("unsupported uniaxial material")


fn _concrete01_envelope(
    fpc: Float64, epsc0: Float64, fpcu: Float64, epscu: Float64, strain: Float64
) -> StressTangent:
    if strain > epsc0:
        var eta = strain / epsc0
        var stress = fpc * (2.0 * eta - eta * eta)
        var Ec0 = (2.0 * fpc) / epsc0
        var tangent = Ec0 * (1.0 - eta)
        return StressTangent(stress, tangent)
    elif strain > epscu:
        var tangent = (fpc - fpcu) / (epsc0 - epscu)
        var stress = fpc + tangent * (strain - epsc0)
        return StressTangent(stress, tangent)
    else:
        return StressTangent(fpcu, 0.0)


fn _concrete01_unload(
    fpc: Float64,
    epsc0: Float64,
    epscu: Float64,
    min_strain: Float64,
    stress: Float64,
) -> EndStrainSlope:
    var temp_strain = min_strain
    if temp_strain < epscu:
        temp_strain = epscu
    var eta = temp_strain / epsc0
    var ratio = 0.707 * (eta - 2.0) + 0.834
    if eta < 2.0:
        ratio = 0.145 * eta * eta + 0.13 * eta
    var end_strain = ratio * epsc0
    var temp1 = min_strain - end_strain
    var Ec0 = (2.0 * fpc) / epsc0
    var temp2 = stress / Ec0
    if temp1 > -1.0e-14:
        return EndStrainSlope(end_strain, Ec0)
    elif temp1 <= temp2:
        end_strain = min_strain - temp1
        return EndStrainSlope(end_strain, stress / temp1)
    else:
        end_strain = min_strain - temp2
        return EndStrainSlope(end_strain, Ec0)


fn _concrete01_reload(
    fpc: Float64,
    epsc0: Float64,
    fpcu: Float64,
    epscu: Float64,
    strain: Float64,
    mut min_strain: Float64,
    mut end_strain: Float64,
    mut unload_slope: Float64,
) -> StressTangent:
    if strain <= min_strain:
        min_strain = strain
        var env = _concrete01_envelope(fpc, epsc0, fpcu, epscu, strain)
        var unload = _concrete01_unload(fpc, epsc0, epscu, min_strain, env.stress)
        end_strain = unload.end_strain
        unload_slope = unload.unload_slope
        return env
    elif strain <= end_strain:
        var tangent = unload_slope
        var stress = tangent * (strain - end_strain)
        return StressTangent(stress, tangent)
    else:
        return StressTangent(0.0, 0.0)


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
        var state = states[i]
        uniaxial_commit(state)
        states[i] = state


fn uniaxial_revert_trial_all(mut states: List[UniMaterialState]):
    for i in range(len(states)):
        var state = states[i]
        uniaxial_revert_trial(state)
        states[i] = state
