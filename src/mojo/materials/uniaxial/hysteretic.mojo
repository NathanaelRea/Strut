# OpenSees HystereticMaterial.cpp by Michael H. Scott and Filip C. Filippou.
# Copyright 1999, The Regents of the University of California. See the
# OpenSees COPYRIGHT file for usage and redistribution terms.

from collections import InlineArray
from math import exp, log
from materials.uniaxial.core import _sign, UniMaterialDefinition, UniMaterialPointState
from os import abort


struct HystereticBackbone(ImplicitlyCopyable, Movable):
    # Coordinates are oriented along the loading direction.
    var force1: Float64
    var strain1: Float64
    var force2: Float64
    var strain2: Float64
    var force3: Float64
    var strain3: Float64
    var slope1: Float64
    var slope2: Float64
    var slope3: Float64
    var unload: Float64

    fn __init__(out self):
        self.force1 = 0.0
        self.strain1 = 0.0
        self.force2 = 0.0
        self.strain2 = 0.0
        self.force3 = 0.0
        self.strain3 = 0.0
        self.slope1 = 0.0
        self.slope2 = 0.0
        self.slope3 = 0.0
        self.unload = 0.0

    fn __init__(out self, values: InlineArray[Float64, 24], offset: Int, direction: Float64):
        self.force1 = direction * values[offset]
        self.strain1 = direction * values[offset + 1]
        self.force2 = direction * values[offset + 2]
        self.strain2 = direction * values[offset + 3]
        self.force3 = direction * values[offset + 4]
        self.strain3 = direction * values[offset + 5]
        if self.strain1 <= 0.0 or self.strain2 <= self.strain1 or self.strain3 <= self.strain2:
            abort("Hysteretic backbone strains must increase in magnitude")
        self.slope1 = self.force1 / self.strain1
        self.slope2 = (self.force2 - self.force1) / (self.strain2 - self.strain1)
        self.slope3 = (self.force3 - self.force2) / (self.strain3 - self.strain2)
        self.unload = max(self.slope1, self.slope2, self.slope3)

    fn energy(self) -> Float64:
        return (
            self.strain1 * self.force1
            + (self.strain2 - self.strain1) * (self.force2 + self.force1)
            + (self.strain3 - self.strain2) * (self.force3 + self.force2)
        )

    fn stress(self, strain: Float64) -> Float64:
        if strain <= 0.0:
            return 0.0
        if strain <= self.strain1:
            return self.slope1 * strain
        if strain <= self.strain2:
            return self.force1 + self.slope2 * (strain - self.strain1)
        if strain <= self.strain3 or self.slope3 > 0.0:
            return self.force2 + self.slope3 * (strain - self.strain2)
        return self.force3

    fn tangent(self, strain: Float64) -> Float64:
        if strain < 0.0:
            return self.slope1 * 1.0e-9
        if strain <= self.strain1:
            return self.slope1
        if strain <= self.strain2:
            return self.slope2
        if strain <= self.strain3 or self.slope3 > 0.0:
            return self.slope3
        return self.slope1 * 1.0e-9

    fn zero_stress_limit(self, strain: Float64) -> Float64:
        var limit = Float64(1.0e16)
        if strain > self.strain1 and strain <= self.strain2 and self.slope2 < 0.0:
            limit = self.strain1 - self.force1 / self.slope2
        if strain > self.strain2 and self.slope3 < 0.0:
            limit = self.strain2 - self.force2 / self.slope3
        if limit == 1.0e16 or self.stress(limit) > 0.0:
            return 1.0e16
        return limit


struct HystereticParameters(ImplicitlyCopyable, Movable):
    var positive: HystereticBackbone
    var negative: HystereticBackbone
    var pinch_x: Float64
    var pinch_y: Float64
    var damage_ductility: Float64
    var damage_energy: Float64
    var beta: Float64
    var envelope_energy: Float64

    fn __init__(out self):
        self.positive = HystereticBackbone()
        self.negative = HystereticBackbone()
        self.pinch_x = 0.0
        self.pinch_y = 0.0
        self.damage_ductility = 0.0
        self.damage_energy = 0.0
        self.beta = 0.0
        self.envelope_energy = 0.0

    fn __init__(out self, values: InlineArray[Float64, 24]):
        self.positive = HystereticBackbone(values, 0, 1.0)
        self.negative = HystereticBackbone(values, 6, -1.0)
        self.pinch_x = values[12]
        self.pinch_y = values[13]
        self.damage_ductility = values[14]
        self.damage_energy = values[15]
        self.beta = values[16]
        self.envelope_energy = 0.5 * (self.positive.energy() + self.negative.energy())


struct HystereticHistory(ImplicitlyCopyable, Movable):
    var strain_max: Float64
    var strain_min: Float64
    var unload_positive: Float64
    var unload_negative: Float64
    var energy: Float64
    var direction: Int

    fn __init__(out self):
        self.strain_max = 0.0
        self.strain_min = 0.0
        self.unload_positive = 0.0
        self.unload_negative = 0.0
        self.energy = 0.0
        self.direction = 0

    fn increment(
        mut self,
        params: HystereticParameters,
        direction: Int,
        committed_strain: Float64,
        committed_stress: Float64,
        strain: Float64,
    ) -> (Float64, Float64):
        var sign = Float64(direction)
        var loading = params.positive if direction == 1 else params.negative
        var unloading = params.negative if direction == 1 else params.positive
        var peak = self.strain_max if direction == 1 else -self.strain_min
        var opposite_peak = -self.strain_min if direction == 1 else self.strain_max
        var reversal = self.unload_negative if direction == 1 else -self.unload_positive
        var load_factor = (peak / loading.strain1) ** params.beta
        load_factor = 1.0 if load_factor < 1.0 else 1.0 / load_factor
        var unload_factor = (opposite_peak / unloading.strain1) ** params.beta
        unload_factor = 1.0 if unload_factor < 1.0 else 1.0 / unload_factor
        var load_slope = loading.unload * load_factor
        var unload_slope = unloading.unload * unload_factor
        var previous_stress = sign * committed_stress
        if self.direction == -direction and previous_stress <= 0.0:
            reversal = sign * committed_strain - previous_stress / unload_slope
            var energy = self.energy - 0.5 * previous_stress / unload_slope * previous_stress
            if opposite_peak > unloading.strain1:
                var damage = params.damage_energy * energy / params.envelope_energy
                damage += (
                    params.damage_ductility
                    * (opposite_peak - unloading.strain1)
                    / unloading.strain1
                )
                peak *= 1.0 + damage
        self.direction = direction
        peak = max(loading.strain1, min(peak, 1.0e16))
        if direction == 1:
            self.strain_max = peak
            self.unload_negative = reversal
        else:
            self.strain_min = -peak
            self.unload_positive = -reversal

        var peak_stress = loading.stress(peak)
        var release = max(-unloading.zero_stress_limit(opposite_peak), reversal)
        var pinch_target = peak - (1.0 - params.pinch_y) * peak_stress / load_slope
        var pinch_strain = release + (pinch_target - release) * params.pinch_x
        var x = sign * strain
        var dx = sign * (strain - committed_strain)
        if x < reversal:
            var stress = previous_stress + unload_slope * dx
            if stress >= 0.0:
                return (0.0, unloading.unload * 1.0e-9)
            return (sign * stress, unload_slope)

        var tangent: Float64
        var pinched_stress: Float64
        if x < pinch_strain:
            if x <= release:
                return (0.0, loading.unload * 1.0e-9)
            tangent = peak_stress * params.pinch_y / (pinch_strain - release)
            pinched_stress = (x - release) * tangent
        else:
            tangent = (1.0 - params.pinch_y) * peak_stress / (peak - pinch_strain)
            pinched_stress = params.pinch_y * peak_stress + (x - pinch_strain) * tangent

        var elastic_stress = previous_stress + load_slope * dx
        if elastic_stress < pinched_stress:
            return (sign * elastic_stress, load_slope)
        return (sign * pinched_stress, tangent)


# OpenSees 3.7.1 HystereticSmooth::setTrialStrain.
fn hysteretic_smooth_set_trial(
    material: UniMaterialDefinition, mut state: UniMaterialPointState, strain: Float64
):
    var ka = material.p0
    var kb = material.p1
    var fbar = material.p2
    var beta = material.p3
    var stiffness_difference = ka - kb
    var a = 0.5 * stiffness_difference / fbar
    var committed_strain = state.eps_c
    var direction = _sign(strain - committed_strain)
    var committed_exp = exp(beta * committed_strain) - exp(-beta * committed_strain)
    var shift = (
        committed_strain * direction
        + log(
            a
            * direction
            / stiffness_difference
            * (
                (kb - 2.0 * beta) * committed_strain
                + committed_exp
                + fbar * direction
                - state.sig_c
            )
        )
        / a
    )
    var decay = exp(-a * (strain * direction - shift))
    var positive_exp = exp(beta * strain)
    var negative_exp = exp(-beta * strain)

    state.eps_t = strain
    state.sig_t = (
        (kb - 2.0 * beta) * strain
        + positive_exp
        - negative_exp
        - direction * stiffness_difference / a * decay
        + fbar * direction
    )
    state.tangent_t = (
        kb - 2.0 * beta + beta * (positive_exp + negative_exp) + stiffness_difference * decay
    )


# OpenSees 3.7.1 HystereticAsym::setTrialStrain.
fn hysteretic_asym_set_trial(
    material: UniMaterialDefinition, mut state: UniMaterialPointState, strain: Float64
):
    var ka = material.p0
    var kb = material.p1
    var fbar = material.p2
    var beta1 = material.p3
    var beta2 = material.p4
    var gamma = material.p5
    var stiffness_difference = ka - kb
    var a = 0.5 * stiffness_difference / fbar
    var committed_strain = state.eps_c
    var direction = _sign(strain - committed_strain)
    var committed_elastic_force = (
        -(beta1 + beta2) * committed_strain
        + exp(beta1 * committed_strain)
        - exp(-beta2 * committed_strain)
    )
    var delta = (
        (committed_elastic_force - state.sig_c) * exp(-gamma * committed_strain)
        + kb * committed_strain
        + direction * fbar
    )
    var shift = committed_strain * direction + log(direction * a * delta / stiffness_difference) / a
    var decay = exp(-a * (strain * direction - shift))
    var component = kb * strain - direction * stiffness_difference / a * decay + fbar * direction
    var positive_exp = exp(beta1 * strain)
    var negative_exp = exp(-beta2 * strain)
    var asymmetry = exp(gamma * strain)

    state.eps_t = strain
    state.sig_t = -(beta1 + beta2) * strain + positive_exp - negative_exp + component * asymmetry
    state.tangent_t = (
        -(beta1 + beta2)
        + beta1 * positive_exp
        + beta2 * negative_exp
        + asymmetry * (gamma * component + kb + stiffness_difference * decay)
    )
