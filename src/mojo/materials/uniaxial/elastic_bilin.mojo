from materials.uniaxial.core import UniMaterialDefinition, UniMaterialPointState


fn elastic_bilin_set_trial(
    material: UniMaterialDefinition, mut state: UniMaterialPointState, strain: Float64
):
    var initial = material.p0
    var post_break = material.p1
    var breakpoint = material.p2
    var past_break = strain >= breakpoint
    if strain < 0.0:
        initial = material.p3
        post_break = material.p4
        breakpoint = material.p5
        past_break = strain <= breakpoint
    state.eps_t = strain
    if past_break:
        state.sig_t = initial * breakpoint + post_break * (strain - breakpoint)
        state.tangent_t = post_break
    else:
        state.sig_t = initial * strain
        state.tangent_t = initial
