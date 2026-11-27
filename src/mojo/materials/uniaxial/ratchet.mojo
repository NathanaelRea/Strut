@fieldwise_init
struct RatchetParameters(ImplicitlyCopyable, Movable):
    var stiffness: Float64
    var travel: Float64
    var initial_engagement: Float64
    var mode: Int

    fn initial_tangent(self) -> Float64:
        return 0.0 if self.initial_engagement > self.travel / 1.0e7 else self.stiffness


struct RatchetHistory(ImplicitlyCopyable, Movable):
    var parameters: RatchetParameters
    var engagement: Float64
    var committed_engagement: Float64

    fn __init__(out self, parameters: RatchetParameters):
        self.parameters = parameters
        self.engagement = parameters.initial_engagement
        # Initial stiffness assembly can evaluate and restore a trial before
        # the first step. Preserve the constructor's free travel in that state.
        self.committed_engagement = parameters.initial_engagement

    fn evaluate(self, strain: Float64) -> Tuple[Float64, Float64]:
        if strain < self.engagement - self.parameters.travel / 1.0e7:
            return (0.0, 0.0)
        return (self.parameters.stiffness * (strain - self.engagement), self.parameters.stiffness)

    fn commit(mut self, strain: Float64, committed_strain: Float64):
        var travel = self.parameters.travel
        if strain <= committed_strain and strain <= self.engagement - travel:
            if self.parameters.mode == 1:
                while strain <= self.engagement - travel + travel / 1.0e7:
                    self.engagement -= travel
            elif self.parameters.mode == 2:
                self.engagement = strain + travel
        self.committed_engagement = self.engagement

    fn revert(mut self):
        self.engagement = self.committed_engagement
