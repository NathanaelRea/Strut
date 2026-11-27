# Asymmetric Menegotto–Pinto–Filippou steel from OpenSees 3.7.1 SteelMPF.cpp.
from sys.ffi import external_call


struct SteelMPFParameters(Defaultable, ImplicitlyCopyable, Movable):
    var fyp: Float64
    var fyn: Float64
    var E0: Float64
    var bp: Float64
    var bn: Float64
    var R0: Float64
    var a1: Float64
    var a2: Float64
    var a3: Float64
    var a4: Float64
    var a5: Float64
    var a6: Float64

    fn __init__(out self):
        self.fyp = 0.0
        self.fyn = 0.0
        self.E0 = 0.0
        self.bp = 0.0
        self.bn = 0.0
        self.R0 = 0.0
        self.a1 = 0.0
        self.a2 = 0.0
        self.a3 = 0.0
        self.a4 = 1.0
        self.a5 = 0.0
        self.a6 = 1.0


struct SteelMPFPoint(Defaultable, ImplicitlyCopyable, Movable):
    var strain: Float64
    var stress: Float64

    fn __init__(out self):
        self.strain = 0.0
        self.stress = 0.0

    fn __init__(out self, strain: Float64, stress: Float64):
        self.strain = strain
        self.stress = stress


struct SteelMPFCurve(Defaultable, ImplicitlyCopyable, Movable):
    var reversal: SteelMPFPoint
    var intersection: SteelMPFPoint
    var curvature: Float64

    fn __init__(out self):
        self.reversal = SteelMPFPoint()
        self.intersection = SteelMPFPoint()
        self.curvature = 0.0

    fn __init__(out self, yield_strain: Float64, yield_stress: Float64, curvature: Float64):
        self.reversal = SteelMPFPoint()
        self.intersection = SteelMPFPoint(yield_strain, yield_stress)
        self.curvature = curvature

    fn evaluate(self, strain: Float64, hardening: Float64) -> (Float64, Float64):
        var delta_strain = self.intersection.strain - self.reversal.strain
        var delta_stress = self.intersection.stress - self.reversal.stress
        var normalized = (strain - self.reversal.strain) / delta_strain
        # The previous branch can be evaluated outside its domain. Preserve
        # the signed base and the resulting IEEE comparisons in that case.
        var power = external_call["pow", Float64](normalized, self.curvature)
        var divisor = external_call["pow", Float64](1.0 + power, 1.0 / self.curvature)
        var transition = (1.0 - hardening) / divisor
        var stress = (
            self.reversal.stress + (hardening * normalized + transition * normalized) * delta_stress
        )
        var tangent = (delta_stress / delta_strain) * (
            hardening + transition * (1.0 - power / (1.0 + power))
        )
        return stress, tangent


struct SteelMPFBranch(Defaultable, ImplicitlyCopyable, Movable):
    var current: SteelMPFCurve
    var previous: SteelMPFCurve
    var reversal_extreme: Float64
    var outside_previous: Bool

    fn __init__(out self):
        self.current = SteelMPFCurve()
        self.previous = SteelMPFCurve()
        self.reversal_extreme = 0.0
        self.outside_previous = True

    fn __init__(out self, curve: SteelMPFCurve):
        self.current = curve
        self.previous = curve
        self.reversal_extreme = 0.0
        self.outside_previous = True

    fn update(
        mut self,
        params: SteelMPFParameters,
        direction: Int,
        reversals: Int,
        reversal: SteelMPFPoint,
        opposite_reversal: Float64,
    ):
        var sign = Float64(direction)
        var yield_stress = params.fyp if direction > 0 else -params.fyn
        var yield_strain = yield_stress / params.E0
        var opposite_yield_strain = (params.fyn if direction > 0 else params.fyp) / params.E0
        var hardening = params.bp if direction > 0 else params.bn
        var shift_scale = params.a5 if direction > 0 else params.a3
        var shift_offset = params.a6 if direction > 0 else params.a4

        self.previous = self.current
        self.current.reversal = reversal
        if sign * reversal.strain < sign * self.reversal_extreme:
            self.reversal_extreme = reversal.strain
        var extreme = abs(self.reversal_extreme)
        var shift = (
            sign * yield_stress * shift_scale * (extreme / opposite_yield_strain - shift_offset)
        )
        if extreme < opposite_yield_strain or shift < 0.0:
            shift = 0.0
        yield_stress += sign * shift

        var post_yield_factor = 1.0 - hardening
        var intersection_strain = (
            yield_stress * post_yield_factor + params.E0 * reversal.strain - reversal.stress
        ) / (params.E0 * post_yield_factor)
        self.current.intersection = SteelMPFPoint(
            intersection_strain,
            reversal.stress + params.E0 * (intersection_strain - reversal.strain),
        )
        var reference_strain = opposite_reversal
        if reversals == 1:
            reference_strain = intersection_strain
        elif reversals == 2:
            reference_strain = yield_strain
        var excursion = abs((reference_strain - intersection_strain) / yield_strain)
        self.current.curvature = min(
            params.R0 - (params.a1 * params.R0) * excursion / (params.a2 + excursion),
            self.previous.curvature,
        )
        var previous_response = self.previous.evaluate(reversal.strain, hardening)
        self.outside_previous = sign * reversal.stress > sign * previous_response[0]

    fn evaluate(
        mut self, strain: Float64, hardening: Float64, direction: Int
    ) -> (Float64, Float64):
        var response = self.current.evaluate(strain, hardening)
        var sign = Float64(direction)
        if (
            not self.outside_previous
            and sign * self.current.reversal.strain > sign * self.previous.reversal.strain
        ):
            var limit = self.previous.evaluate(strain, hardening)
            if sign * response[0] > sign * limit[0]:
                self.current = self.previous
                return limit
        return response


struct SteelMPFHistory(Defaultable, ImplicitlyCopyable, Movable):
    var positive: SteelMPFBranch
    var negative: SteelMPFBranch
    var direction: Int
    var reversals: Int

    fn __init__(out self):
        self.positive = SteelMPFBranch()
        self.negative = SteelMPFBranch()
        self.direction = 0
        self.reversals = 0

    fn evaluate(
        mut self,
        params: SteelMPFParameters,
        committed: SteelMPFPoint,
        strain: Float64,
    ) -> (Float64, Float64):
        if self.direction == 0:
            self.positive = SteelMPFBranch(
                SteelMPFCurve(params.fyp / params.E0, params.fyp, params.R0)
            )
            self.negative = SteelMPFBranch(
                SteelMPFCurve(-params.fyn / params.E0, -params.fyn, params.R0)
            )
            self.direction = -1 if strain < 0.0 else 1
            self.reversals = 0 if strain == 0.0 else 1
        elif self.direction == 1 and strain < committed.strain:
            self.direction = -1
            self.reversals += 1
            self.negative.update(
                params, -1, self.reversals, committed, self.positive.current.reversal.strain
            )
        elif self.direction == -1 and strain > committed.strain:
            self.direction = 1
            self.reversals += 1
            self.positive.update(
                params, 1, self.reversals, committed, self.negative.current.reversal.strain
            )
        if self.direction == 1:
            return self.positive.evaluate(strain, params.bp, 1)
        return self.negative.evaluate(strain, params.bn, -1)
