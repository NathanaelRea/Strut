from math import sqrt
from materials.uniaxial.core import UniMaterialDef, UniMaterialState
from materials.uniaxial.ops import uniaxial_set_trial_strain


fn evaluate_coupled_zero_length(
    material: UniMaterialDef, mut state: UniMaterialState, dx: Float64, dy: Float64
) -> Tuple[Float64, Float64]:
    var strain = sqrt(dx * dx + dy * dy)
    if dx + dy < 0.0:
        strain = -strain
    uniaxial_set_trial_strain(material, state, strain)
    if state.eps_t != 0.0:
        var scale = state.sig_t / state.eps_t
        var x = scale * dx
        var y = scale * dy
        if state.sig_t != 0.0:
            state.coupled_force.value().trial = (x, y)
        return (x, y)

    var committed = state.coupled_force.value().committed
    var norm = sqrt(committed[0] * committed[0] + committed[1] * committed[1])
    if norm != 0.0:
        var scale = state.sig_t / norm
        return (scale * committed[0], scale * committed[1])
    return (state.sig_t, state.sig_t)
