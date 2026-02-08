from materials.uniaxial.core import EndStrainSlope, StressTangent

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


