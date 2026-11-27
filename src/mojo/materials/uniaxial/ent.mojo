from sys.ffi import external_call

from materials.uniaxial.core import UniMaterialDefinition, UniMaterialPointState


fn ent_set_trial(mat_def: UniMaterialDefinition, mut state: UniMaterialPointState, eps: Float64):
    # OpenSees ENTMaterial uses the compressive tangent at the origin.
    state.eps_t = eps
    var E = mat_def.p0
    if eps <= 0.0:
        state.sig_t = E * eps
        state.tangent_t = E
    elif mat_def.p1 == 0.0:
        state.sig_t = 0.0
        state.tangent_t = 0.0
    else:
        var tension = external_call["tanh", Float64](eps * mat_def.p2)
        var scale = E * mat_def.p1
        state.sig_t = scale * tension
        state.tangent_t = scale * mat_def.p2 * (1.0 - tension * tension)
