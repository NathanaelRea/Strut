@fieldwise_init
struct StrainLimits(ImplicitlyCopyable, Movable):
    var minimum: Float64
    var maximum: Float64

    fn intersect(self, other: Self) -> Self:
        return Self(max(self.minimum, other.minimum), min(self.maximum, other.maximum))


struct MinMaxHistory(ImplicitlyCopyable, Movable):
    var limits: StrainLimits
    var trial_failed: Bool
    var committed_failed: Bool
    var stress: Float64
    var tangent: Float64
    var failed_tangent: Float64

    fn __init__(out self, limits: StrainLimits, initial_tangent: Float64):
        self.limits = limits
        self.trial_failed = False
        self.committed_failed = False
        self.stress = 0.0
        self.tangent = 0.0
        self.failed_tangent = 1.0e-8 * initial_tangent

    fn hide_response(mut self, mut stress: Float64, mut tangent: Float64):
        self.stress = stress
        self.tangent = tangent
        stress = 0.0
        tangent = self.failed_tangent

    fn restore_response(self, mut stress: Float64, mut tangent: Float64):
        stress = self.stress
        tangent = self.tangent

    fn set_trial(mut self, strain: Float64, mut stress: Float64, mut tangent: Float64) -> Bool:
        if self.committed_failed:
            return False
        if self.trial_failed:
            self.restore_response(stress, tangent)
        self.trial_failed = strain <= self.limits.minimum or strain >= self.limits.maximum
        if self.trial_failed:
            self.hide_response(stress, tangent)
        return not self.trial_failed
