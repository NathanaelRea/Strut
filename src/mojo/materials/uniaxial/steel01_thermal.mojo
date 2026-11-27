from math import sqrt
from os import abort
from utils import StaticTuple

from materials.uniaxial.core import UniMaterialDefinition, UniMaterialPointState


alias _MACHINE_EPSILON = 2.220446049250313e-16


@fieldwise_init
struct Steel01ThermalProperties(Copyable, Movable):
    var modulus: Float64
    var yield_stress: Float64
    var proportional_stress: Float64
    var elongation: Float64


fn steel01_thermal_properties(
    mat: UniMaterialDefinition, temperature: Float64
) -> Steel01ThermalProperties:
    if temperature >= 1200.0:
        abort("Steel01Thermal requires a temperature below 1200 degrees")
    var modulus_factor = 1.0
    var yield_factor = 1.0
    var proportional_factor = 1.0
    if temperature > 100.0:
        var modulus = StaticTuple[Float64, 12](
            1.0, 0.9, 0.8, 0.7, 0.6, 0.31, 0.13, 0.09, 0.0675, 0.045, 0.0225, 0.0
        )
        var strength = StaticTuple[Float64, 12](
            1.0, 1.0, 1.0, 1.0, 0.78, 0.47, 0.23, 0.11, 0.06, 0.04, 0.02, 0.0
        )
        var proportional = StaticTuple[Float64, 12](
            1.0, 0.807, 0.613, 0.42, 0.36, 0.18, 0.075, 0.05, 0.0375, 0.025, 0.0125, 0.0
        )
        var interval = Int(temperature / 100.0) - 1
        var fraction = (temperature - Float64(interval + 1) * 100.0) / 100.0
        modulus_factor = modulus[interval] + fraction * (modulus[interval + 1] - modulus[interval])
        yield_factor = strength[interval] + fraction * (strength[interval + 1] - strength[interval])
        proportional_factor = proportional[interval] + fraction * (
            proportional[interval + 1] - proportional[interval]
        )

    var elongation = 1.52466e-20
    if temperature > 860.0:
        elongation = -0.00619160 + 2.0e-5 * temperature
    elif temperature > 750.0:
        elongation = 0.01100840
    elif temperature > 20.0:
        elongation = -2.416e-4 + 1.2e-5 * temperature + 0.4e-8 * temperature * temperature
    return Steel01ThermalProperties(
        mat.p1 * modulus_factor, mat.p0 * yield_factor, mat.p0 * proportional_factor, elongation
    )


struct _Steel01ThermalEnvelope(Copyable, Movable):
    var modulus: Float64
    var proportional_stress: Float64
    var proportional_strain: Float64
    var increment: Float64
    var a: Float64
    var b: Float64
    var c: Float64

    fn __init__(out self, properties: Steel01ThermalProperties):
        self.modulus = properties.modulus
        self.proportional_stress = properties.proportional_stress
        self.proportional_strain = self.proportional_stress / self.modulus
        var span = 0.02 - self.proportional_strain
        self.increment = span / 500.0
        var delta = properties.yield_stress - self.proportional_stress
        self.c = delta * delta / (span * self.modulus - 2.0 * delta)
        self.b = sqrt(self.c * span * self.modulus + self.c * self.c)
        self.a = span * (span + self.c / self.modulus)

    fn sample(self, index: Int) -> (Float64, Float64):
        var strain = self.proportional_strain + Float64(index) * self.increment
        var distance = 0.02 - strain
        var stress = (
            self.proportional_stress
            - self.c
            + self.b / sqrt(self.a) * sqrt(self.a - distance * distance)
        )
        return (strain, stress)

    fn upper_strain(self, plastic_strain: Float64) -> Float64:
        var low = 0
        var high = 500
        while high - low > 1:
            var mid = (high + low) // 2
            var point = self.sample(mid)
            if point[0] - point[1] / self.modulus < plastic_strain:
                low = mid
            else:
                high = mid
        var a = self.sample(low)
        var b = self.sample(high)
        var plastic_a = a[0] - a[1] / self.modulus
        var plastic_b = b[0] - b[1] / self.modulus
        return a[0] + (plastic_strain - plastic_a) * (b[0] - a[0]) / (plastic_b - plastic_a)

    fn stress_and_tangent(self, strain: Float64) -> (Float64, Float64):
        var segment = 0
        if strain >= self.proportional_strain and strain <= 0.02:
            segment = min(Int((strain - self.proportional_strain) / self.increment), 499)
        var a = self.sample(segment)
        var b = self.sample(segment + 1)
        var tangent = (b[1] - a[1]) / (b[0] - a[0])
        return (a[1] + tangent * (strain - a[0]), abs(tangent))


fn steel01_thermal_set_trial(
    mat: UniMaterialDefinition,
    mut state: UniMaterialPointState,
    total_strain: Float64,
    temperature: Float64,
):
    var properties = steel01_thermal_properties(mat, temperature)
    var strain = total_strain - properties.elongation
    var increment = strain - state.eps_c
    state.eps_t = state.eps_c
    state.sig_t = state.sig_c
    state.tangent_t = state.tangent_c
    state.unload_slope_t = state.unload_slope_c
    if increment != 0.0 and abs(increment) <= _MACHINE_EPSILON:
        return

    state.eps_t = strain
    var e = properties.modulus
    var fy = properties.yield_stress
    if e == mat.p1:
        var elastic_stress = state.sig_c + e * increment
        var hardening_stress = mat.p2 * e * strain
        var limit = fy * (1.0 - mat.p2)
        state.sig_t = max(hardening_stress - limit, min(hardening_stress + limit, elastic_stress))
        state.tangent_t = e if abs(state.sig_t - elastic_stress) < _MACHINE_EPSILON else mat.p2 * e
        return

    var fp = properties.proportional_stress
    var ep = fp / e
    var plastic = state.eps_c - state.sig_c / state.unload_slope_c
    var plastic_sign = -1.0 if plastic < 0.0 else 1.0
    var strain_sign = -1.0 if strain < 0.0 else 1.0
    var rounded = abs(plastic) + fy / e < 0.02
    var envelope = _Steel01ThermalEnvelope(properties)
    var bound = abs(plastic) + fy / e
    if rounded:
        bound = envelope.upper_strain(abs(plastic))
    var opposite = bound - 2.0 * ep
    var directed_strain = plastic_sign * strain
    var stress: Float64
    var tangent: Float64
    if opposite <= directed_strain and directed_strain <= bound:
        stress = e * (strain - plastic)
        tangent = e
    elif (rounded and abs(strain) >= 0.02) or (
        not rounded and abs(strain) > max(abs(bound), abs(opposite))
    ):
        stress = strain_sign * fy
        tangent = 1.0e-10 * e
    elif rounded and abs(strain) > abs(bound):
        var envelope_point = envelope.stress_and_tangent(abs(strain))
        stress = strain_sign * envelope_point[0]
        tangent = envelope_point[1]
    else:
        var bound_stress = fy
        if rounded:
            bound_stress = envelope.stress_and_tangent(abs(bound))[0]
        tangent = abs((2.0 * bound_stress - 2.0 * fp) / (opposite + bound))
        stress = plastic_sign * (
            bound_stress - 2.0 * fp - tangent * abs(directed_strain - opposite)
        )

    state.sig_t = min(fy, max(-fy, stress))
    state.tangent_t = tangent
    state.unload_slope_t = e
