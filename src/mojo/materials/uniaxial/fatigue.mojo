from simd_math import pow_nonneg


struct FatigueParameters(Defaultable, ImplicitlyCopyable, Movable):
    var enabled: Bool
    var dmax: Float64
    var e0: Float64
    var slope: Float64
    var min_strain: Float64
    var max_strain: Float64

    fn __init__(out self):
        self.enabled = False
        self.dmax = 1.0
        self.e0 = 0.191
        self.slope = -0.458
        self.min_strain = -1.0e16
        self.max_strain = 1.0e16

    fn __init__(
        out self,
        dmax: Float64,
        e0: Float64,
        slope: Float64,
        min_strain: Float64,
        max_strain: Float64,
    ):
        self.enabled = True
        self.dmax = dmax if 0.0 <= dmax <= 1.0 else 1.0
        self.e0 = e0
        self.slope = slope
        self.min_strain = min_strain
        self.max_strain = max_strain


fn _cycle_damage(params: FatigueParameters, strain_range: Float64, cycles: Float64) -> Float64:
    return cycles / abs(pow_nonneg(strain_range / params.e0, 1.0 / params.slope))


fn _residual_damage(params: FatigueParameters, strain_range: Float64, cycles: Float64) -> Float64:
    if strain_range < 1.0e-10:
        return 0.0
    return _cycle_damage(params, strain_range, cycles)


struct FatigueHistory(Defaultable, ImplicitlyCopyable, Movable):
    var params: FatigueParameters
    var failed: Bool
    var started: Bool
    var has_second_peak: Bool
    var pending_cycle: Bool
    var first_peak: Float64
    var second_peak: Float64
    var third_peak: Float64
    var previous_range: Float64
    var previous_strain: Float64
    var previous_slope: Float64
    var damage: Float64
    var trial_strain: Float64
    var initial_tangent: Float64

    fn __init__(out self):
        self.params = FatigueParameters()
        self.failed = False
        self.started = False
        self.has_second_peak = False
        self.pending_cycle = False
        self.first_peak = 0.0
        self.second_peak = 0.0
        self.third_peak = 0.0
        self.previous_range = 0.0
        self.previous_strain = 0.0
        self.previous_slope = 0.0
        self.damage = 0.0
        self.trial_strain = 0.0
        self.initial_tangent = 0.0

    fn _count_peak(mut self):
        var peak = self.previous_strain
        if not self.has_second_peak:
            self.second_peak = peak
            self.previous_range = abs(peak - self.first_peak)
            self.has_second_peak = True
            return

        if self.pending_cycle:
            var current_range = abs(peak - self.third_peak)
            if current_range < self.previous_range:
                self.damage += _cycle_damage(self.params, current_range, 1.0)
            else:
                self.damage += _cycle_damage(self.params, self.previous_range, 1.0)
                self.second_peak = peak
            self.third_peak = 0.0
            self.previous_range = abs(self.second_peak - self.first_peak)
            self.pending_cycle = False
            return

        var current_range = abs(peak - self.second_peak)
        if current_range < self.previous_range:
            self.third_peak = peak
            self.pending_cycle = True
        else:
            self.damage += _cycle_damage(self.params, self.previous_range, 0.5)
            self.first_peak = self.second_peak
            self.second_peak = peak
        self.previous_range = current_range

    fn commit(mut self, stress: Float64):
        if (
            self.trial_strain >= self.params.max_strain
            or self.trial_strain <= self.params.min_strain
        ):
            self.failed = True
            self.damage = self.params.dmax
            return

        if not self.started:
            self.first_peak = self.trial_strain
            self.previous_strain = self.trial_strain
            self.started = True

        var slope = self.trial_strain - self.previous_strain
        if slope == 0.0:
            slope = self.previous_slope
        if (self.previous_slope > 0.0 and slope < 0.0) or (
            self.previous_slope < 0.0 and slope > 0.0
        ):
            self._count_peak()
            self.failed = self.damage >= self.params.dmax

        if not self.failed and stress > 0.0:
            # Treat the current endpoint as a possible last turning point.
            var damage_at_endpoint = self.damage
            if self.second_peak == 0.0 and self.third_peak == 0.0:
                damage_at_endpoint += _residual_damage(
                    self.params, abs(self.trial_strain - self.first_peak), 0.5
                )
            elif self.second_peak != 0.0 and self.third_peak == 0.0:
                damage_at_endpoint += _residual_damage(
                    self.params, abs(self.trial_strain - self.second_peak), 0.5
                )
                damage_at_endpoint += _residual_damage(self.params, self.previous_range, 0.5)
            elif self.second_peak != 0.0 and self.third_peak != 0.0:
                var outer_range = abs(self.trial_strain - self.first_peak)
                var first_range = abs(self.first_peak - self.second_peak)
                if outer_range > first_range:
                    damage_at_endpoint += _residual_damage(self.params, self.previous_range, 1.0)
                    damage_at_endpoint += _residual_damage(self.params, outer_range, 0.5)
                else:
                    damage_at_endpoint += _residual_damage(
                        self.params, abs(self.third_peak - self.trial_strain), 1.0
                    )
                    damage_at_endpoint += _residual_damage(self.params, first_range, 0.5)
            if damage_at_endpoint > self.params.dmax:
                self.damage = damage_at_endpoint
                self.failed = True

        self.previous_slope = slope
        self.previous_strain = self.trial_strain
