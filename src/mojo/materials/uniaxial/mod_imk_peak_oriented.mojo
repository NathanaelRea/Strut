# Modified Ibarra–Medina–Krawinkler peak-oriented material from OpenSees 3.7.1.
from collections import InlineArray
from sys.ffi import external_call


@fieldwise_init
struct IMKPeakSide(Defaultable, ImplicitlyCopyable, Movable):
    var hardening: Float64
    var yield_force: Float64
    var plastic_rotation: Float64
    var post_cap_rotation: Float64
    var residual: Float64
    var ultimate_rotation: Float64
    var deterioration: Float64

    fn __init__(out self):
        self = IMKPeakSide(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


struct IMKPeakParameters(Defaultable, ImplicitlyCopyable, Movable):
    var stiffness: Float64
    var n_factor: Float64
    var strength_energy: Float64
    var capping_energy: Float64
    var reloading_energy: Float64
    var unloading_energy: Float64
    var strength_exponent: Float64
    var capping_exponent: Float64
    var reloading_exponent: Float64
    var unloading_exponent: Float64
    var sides: InlineArray[IMKPeakSide, 2]

    fn __init__(out self):
        self = Self(InlineArray[Float64, 24](fill=0.0))

    fn __init__(out self, values: InlineArray[Float64, 24]):
        self.stiffness = values[0]
        self.n_factor = values[23]
        self.strength_energy = values[5] * values[3]
        self.capping_energy = values[6] * values[3]
        self.reloading_energy = values[7] * values[3]
        self.unloading_energy = 2.0 * values[8] * values[3]
        self.strength_exponent = values[9]
        self.capping_exponent = values[10]
        self.reloading_exponent = values[11]
        self.unloading_exponent = values[12]
        self.sides = InlineArray[IMKPeakSide, 2](fill=IMKPeakSide())
        for i in range(2):
            self.sides[i] = IMKPeakSide(
                values[1 + i],
                values[3 + i],
                values[13 + i],
                values[15 + i],
                values[17 + i],
                values[19 + i],
                values[21 + i],
            )

    fn initial_tangent(self) -> Float64:
        return self.stiffness * (1.0 + self.n_factor)

    fn spring_slope(self, slope: Float64) -> Float64:
        return slope / (1.0 + self.n_factor * (1.0 - slope))

    fn member_slope(self, slope: Float64) -> Float64:
        return slope * (1.0 + self.n_factor) / (1.0 + self.n_factor * slope)


@fieldwise_init
struct IMKPeakEnvelope(ImplicitlyCopyable, Movable):
    var yield_force: Float64
    var hardening: Float64
    var cap_slope: Float64
    var cap_rotation: Float64
    var cap_intercept: Float64
    var peak_rotation: Float64
    var peak_force: Float64
    var last_rotation: Float64
    var last_force: Float64
    var zero_rotation: Float64

    fn __init__(out self, params: IMKPeakParameters, side: IMKPeakSide, sign: Float64):
        var stiffness = params.initial_tangent()
        var dy = side.yield_force / stiffness
        var hardening = params.spring_slope(side.hardening)
        var plastic_rotation = sign * side.plastic_rotation
        var cap_rotation = dy + plastic_rotation
        var cap_force = side.yield_force + stiffness * hardening * plastic_rotation
        var cap_slope = params.spring_slope(
            -sign * cap_force / (side.post_cap_rotation * params.stiffness)
        )
        self = IMKPeakEnvelope(
            side.yield_force,
            hardening,
            cap_slope,
            cap_rotation,
            cap_force - cap_slope * stiffness * cap_rotation,
            dy,
            side.yield_force,
            dy,
            side.yield_force,
            0.0,
        )

    fn backbone(
        self,
        side: IMKPeakSide,
        sign: Float64,
        stiffness: Float64,
        mut d: Float64,
        mut stopped: Bool,
    ) -> (Float64, Float64):
        var dy = self.yield_force / stiffness
        var cap_force = stiffness * self.cap_rotation
        var residual = side.residual * cap_force
        var hardening = sign * dy <= sign * self.cap_rotation
        if hardening:
            cap_force = self.yield_force + self.hardening * stiffness * (self.cap_rotation - dy)
            residual = side.residual * side.yield_force
        var cap_tangent = self.cap_slope * stiffness
        var residual_rotation = self.cap_rotation + (residual - cap_force) / cap_tangent

        if sign * d >= side.ultimate_rotation:
            d = sign * side.ultimate_rotation
            stopped = True
            return (1e-10, 1e-7)
        if sign * d < 0.0:
            return (0.0, 1e-7)
        if sign * d <= sign * (dy if hardening else self.cap_rotation):
            return (stiffness * d, stiffness)
        if sign * d <= sign * self.cap_rotation:
            var tangent = stiffness * self.hardening
            return (self.yield_force + tangent * (d - dy), tangent)
        if sign * d <= sign * residual_rotation:
            return (cap_force + cap_tangent * (d - self.cap_rotation), cap_tangent)
        return (residual + 1e-7 * d, 1e-7)

    fn degrade(
        mut self,
        params: IMKPeakParameters,
        side: IMKPeakSide,
        sign: Float64,
        strength: Float64,
        cap: Float64,
        reload: Float64,
        mut stopped: Bool,
    ):
        var stiffness = params.initial_tangent()
        var factor = 1.0 - strength * side.deterioration
        var hardening = params.member_slope(self.hardening) * factor
        self.yield_force *= factor
        self.hardening = params.spring_slope(hardening)
        self.cap_intercept *= 1.0 - cap * side.deterioration
        if stiffness > params.stiffness:
            var member = self
            member.hardening = params.member_slope(self.hardening)
            member.cap_slope = params.member_slope(self.cap_slope)
            member.move_cap(params.stiffness, sign)
            self.peak_rotation += self.peak_force / params.stiffness
            self.peak_rotation *= 1.0 + reload
            var response = member.backbone(
                side, sign, params.stiffness, self.peak_rotation, stopped
            )
            self.peak_rotation -= response[0] / params.stiffness
        else:
            self.peak_rotation *= 1.0 + reload
        self.move_cap(stiffness, sign)

    fn move_cap(mut self, stiffness: Float64, sign: Float64):
        var cap_tangent = self.cap_slope * stiffness
        var hardening = self.hardening * stiffness
        var elastic_cap = self.cap_intercept / (stiffness - cap_tangent)
        var plastic_cap = (
            self.cap_intercept + hardening * (self.yield_force / stiffness) - self.yield_force
        ) / (hardening - cap_tangent)
        self.cap_rotation = elastic_cap if sign * elastic_cap > sign * plastic_cap else plastic_cap


struct IMKPeakHistory(ImplicitlyCopyable, Movable):
    var envelopes: InlineArray[IMKPeakEnvelope, 2]
    var strain: Float64
    var stress: Float64
    var tangent: Float64
    var direction: Int
    var unloading: Bool
    var stopped: Bool
    var excursion_energy: Float64
    var total_energy: Float64
    var unloading_stiffness: Float64
    var unloading_product: Float64

    fn __init__(out self, params: IMKPeakParameters):
        var positive = IMKPeakEnvelope(params, params.sides[0], 1.0)
        var negative = IMKPeakEnvelope(params, params.sides[1], -1.0)
        self.envelopes = InlineArray[IMKPeakEnvelope, 2](positive, negative)
        self.strain = 0.0
        self.stress = 0.0
        self.tangent = params.initial_tangent()
        self.direction = -1
        self.unloading = False
        self.stopped = False
        self.excursion_energy = 0.0
        self.total_energy = 0.0
        self.unloading_stiffness = self.tangent
        self.unloading_product = 1.0

    fn _reverse(mut self, params: IMKPeakParameters, direction: Int, sign: Float64):
        self.unloading = True
        var recoverable = 0.5 * self.stress * self.stress / self.unloading_stiffness
        var dissipated = self.total_energy - recoverable
        var remaining = params.unloading_energy - dissipated
        if params.unloading_energy != 0.0:
            if direction == 0 and remaining < 0.0:
                self.stopped = True
            var beta = 1.0 if dissipated >= params.unloading_energy else external_call[
                "pow", Float64
            ]((self.excursion_energy - recoverable) / remaining, params.unloading_exponent)
            self.unloading_product *= 1.0 - beta
            self.unloading_stiffness = params.initial_tangent() * params.spring_slope(
                self.unloading_product
            )
            var previous_hardening = (
                self.envelopes[1 - direction].hardening * params.initial_tangent()
            )
            if self.unloading_stiffness <= previous_hardening:
                self.stopped = True
        ref dest = self.envelopes[direction]
        ref origin = self.envelopes[1 - direction]
        if sign * self.stress < 0.0:
            var zero = self.strain - self.stress / self.unloading_stiffness
            var dy = params.sides[direction].yield_force / params.initial_tangent()
            if abs(dest.peak_rotation - dy) >= 1e-10 and abs(zero) <= 1e-10:
                zero = 1e-9
            dest.zero_rotation = zero
        if abs(origin.peak_rotation - self.strain) <= 1e-10:
            origin.zero_rotation = dest.zero_rotation + sign * 1e-10

    fn _deteriorate(mut self, params: IMKPeakParameters, direction: Int, sign: Float64):
        var strength = 0.0
        var reload = 0.0
        var cap = 0.0
        if self.total_energy >= params.strength_energy and params.strength_energy != 0.0:
            strength = 1.0
        elif self.total_energy >= params.reloading_energy and params.reloading_energy != 0.0:
            reload = 1.0
        elif self.total_energy >= params.capping_energy and params.capping_energy == 0.0:
            cap = 1.0
        else:
            strength = self._energy_ratio(params.strength_energy, params.strength_exponent)
            reload = self._energy_ratio(params.reloading_energy, params.reloading_exponent)
            cap = self._energy_ratio(params.capping_energy, params.capping_exponent)
        self.excursion_energy = 0.0
        self.envelopes[direction].degrade(
            params, params.sides[direction], sign, strength, cap, reload, self.stopped
        )

    fn _energy_ratio(self, capacity: Float64, exponent: Float64) -> Float64:
        if capacity == 0.0:
            return 0.0
        var beta = external_call["pow", Float64](
            self.excursion_energy / (capacity - self.total_energy), exponent
        )
        return 1.0 if abs(beta) >= 1.0 else beta

    fn evaluate(mut self, params: IMKPeakParameters, strain: Float64, has_commit: Bool):
        var d = strain
        var delta = d - self.strain
        if abs(delta) < 1e-18 and strain != 0.0 and has_commit:
            return
        var direction = 0 if delta >= 0.0 else 1
        var sign = 1.0 if direction == 0 else -1.0
        if self.stopped:
            self.strain = d
            self.stress = 0.0
            self.tangent = 0.0
            return
        if self.direction != -1 and self.direction != direction:
            self._reverse(params, direction, sign)
        self.direction = direction
        var stiffness = params.initial_tangent()
        ref side = self.envelopes[direction]
        ref opposite = self.envelopes[1 - direction]
        var force: Float64
        var tangent: Float64
        if sign * d >= sign * side.peak_rotation:
            var response = side.backbone(params.sides[direction], sign, stiffness, d, self.stopped)
            force = response[0]
            tangent = response[1]
            side.peak_rotation = d
            side.peak_force = force
            side.last_rotation = d + sign * 1e-10
            side.last_force = force
        elif abs(side.zero_rotation) > 1e-10:
            var peak_rotation = side.peak_rotation
            var target = side.backbone(
                params.sides[direction], sign, stiffness, peak_rotation, self.stopped
            )
            side.peak_rotation = peak_rotation
            side.peak_force = target[0]
            var zero = side.zero_rotation
            var unload_force = self.stress + self.unloading_stiffness * delta
            if sign * d <= sign * zero:
                tangent = self.unloading_stiffness
                force = unload_force
                if (
                    self.unloading
                    and abs(tangent - self.tangent) > 1e-10
                    and self.strain != opposite.peak_rotation
                ):
                    opposite.last_rotation = self.strain
                    opposite.last_force = self.stress
            else:
                tangent = side.peak_force / (side.peak_rotation - zero)
                force = tangent * (d - zero)
                if (
                    sign * side.last_rotation > sign * zero
                    and sign * side.last_rotation < sign * side.peak_rotation
                ):
                    var slope = side.last_force / (side.last_rotation - zero)
                    if slope > tangent and sign * side.last_force < sign * side.peak_force:
                        if sign * d < sign * side.last_rotation:
                            tangent = slope
                            force = tangent * (d - zero)
                        else:
                            tangent = (side.peak_force - side.last_force) / (
                                side.peak_rotation - side.last_rotation
                            )
                            force = side.last_force + tangent * (d - side.last_rotation)
                if sign * unload_force < sign * force:
                    force = unload_force
                if abs(force - unload_force) < 1e-10:
                    tangent = self.unloading_stiffness
        else:
            var index = 0 if d > 0.0 else 1
            var response = self.envelopes[index].backbone(
                params.sides[index],
                1.0 if index == 0 else -1.0,
                stiffness,
                d,
                self.stopped,
            )
            force = response[0]
            tangent = response[1]
        var energy = 0.5 * (force + self.stress) * delta
        self.excursion_energy += energy
        self.total_energy += energy
        if force * self.stress < 0.0:
            var prior = 0 if self.stress > 0.0 else 1
            var prior_sign = 1.0 if prior == 0 else -1.0
            ref envelope = self.envelopes[prior]
            if prior_sign * envelope.peak_rotation > prior_sign * envelope.yield_force / stiffness:
                self._deteriorate(params, direction, sign)
        self.strain = d
        self.stress = force
        self.tangent = tangent
