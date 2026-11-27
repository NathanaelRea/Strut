from materials.uniaxial.core import UniMaterialDefinition, UniMaterialPointState


fn elastic_pp_set_trial(
    material: UniMaterialDefinition, mut state: UniMaterialPointState, strain: Float64
):
    var E = material.p0
    var positive_yield = E * material.p1
    var negative_yield = E * material.p2
    var predictor = E * (strain - material.p3 - state.eps_p_c)
    var excess = predictor - positive_yield if predictor >= 0.0 else negative_yield - predictor

    state.eps_t = strain
    state.eps_p_t = state.eps_p_c
    if excess <= -E * 2.220446049250313e-16:
        state.sig_t = predictor
        state.tangent_t = E
    else:
        state.sig_t = positive_yield if predictor > 0.0 else negative_yield
        state.tangent_t = 0.0
        state.eps_p_t += excess / E if predictor > 0.0 else -excess / E
