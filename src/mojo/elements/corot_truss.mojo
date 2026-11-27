from math import sqrt
from os import abort


alias TrussVector = SIMD[DType.float64, 4]


struct CorotTrussChord(ImplicitlyCopyable, Movable):
    var direction: TrussVector
    var initial_length: Float64
    var length: Float64

    fn __init__(out self, reference: TrussVector, relative_displacement: TrussVector):
        var current = reference + relative_displacement
        self.initial_length = sqrt((reference * reference).reduce_add())
        self.length = sqrt((current * current).reduce_add())
        if self.initial_length == 0.0 or self.length == 0.0:
            abort("corotTruss requires a nonzero chord length")
        self.direction = current / self.length

    fn strain(self) -> Float64:
        return (self.length - self.initial_length) / self.initial_length

    fn strain_rate(self, relative_velocity: TrussVector) -> Float64:
        return (self.direction * relative_velocity).reduce_add() / self.initial_length
