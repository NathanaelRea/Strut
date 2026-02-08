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

    fn __init__(out self):
        self.mat_type = 0
        self.p0 = 0.0
        self.p1 = 0.0
        self.p2 = 0.0
        self.p3 = 0.0

    fn __init__(
        out self, mat_type: Int, p0: Float64, p1: Float64, p2: Float64, p3: Float64
    ):
        self.mat_type = mat_type
        self.p0 = p0
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3


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
        var tangent = uni_mat_initial_tangent(mat_def)
        self.tangent_c = tangent
        self.tangent_t = tangent
        if mat_def.mat_type == 2:
            self.unload_slope_c = tangent
            self.unload_slope_t = tangent


fn uni_mat_is_elastic(mat_def: UniMaterialDef) -> Bool:
    return mat_def.mat_type == 0


fn uni_mat_initial_tangent(mat_def: UniMaterialDef) -> Float64:
    if mat_def.mat_type == 0:
        return mat_def.p0
    if mat_def.mat_type == 1:
        return mat_def.p1
    if mat_def.mat_type == 2:
        return (2.0 * mat_def.p0) / mat_def.p1
    abort("unsupported uniaxial material")
    return 0.0


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


fn uniaxial_revert_trial(mut state: UniMaterialState):
    state.eps_t = state.eps_c
    state.sig_t = state.sig_c
    state.tangent_t = state.tangent_c
    state.eps_p_t = state.eps_p_c
    state.alpha_t = state.alpha_c
    state.min_strain_t = state.min_strain_c
    state.end_strain_t = state.end_strain_c
    state.unload_slope_t = state.unload_slope_c


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
