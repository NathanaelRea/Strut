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

    fn __init__(out self, mat_def: UniMaterialDef):
        self.eps_c = 0.0
        self.sig_c = 0.0
        self.eps_p_c = 0.0
        self.alpha_c = 0.0
        self.eps_t = 0.0
        self.sig_t = 0.0
        self.eps_p_t = 0.0
        self.alpha_t = 0.0
        var tangent = uni_mat_initial_tangent(mat_def)
        self.tangent_c = tangent
        self.tangent_t = tangent


fn uni_mat_is_elastic(mat_def: UniMaterialDef) -> Bool:
    return mat_def.mat_type == 0


fn uni_mat_initial_tangent(mat_def: UniMaterialDef) -> Float64:
    if mat_def.mat_type == 0:
        return mat_def.p0
    if mat_def.mat_type == 1:
        return mat_def.p1
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
    abort("unsupported uniaxial material")


fn uniaxial_commit(mut state: UniMaterialState):
    state.eps_c = state.eps_t
    state.sig_c = state.sig_t
    state.tangent_c = state.tangent_t
    state.eps_p_c = state.eps_p_t
    state.alpha_c = state.alpha_t


fn uniaxial_revert_trial(mut state: UniMaterialState):
    state.eps_t = state.eps_c
    state.sig_t = state.sig_c
    state.tangent_t = state.tangent_c
    state.eps_p_t = state.eps_p_c
    state.alpha_t = state.alpha_c


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
