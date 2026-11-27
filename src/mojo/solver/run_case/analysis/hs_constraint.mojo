# Hyperspherical continuation from OpenSees 3.7.1 HSConstraint.cpp.
from math import sqrt


@fieldwise_init
struct HSConstraint(ImplicitlyCopyable, Movable):
    var length: Float64
    var displacement_weight: Float64
    var force_weight: Float64
    var reference_displacement: Float64

    fn __init__(out self):
        self = HSConstraint(0.0, 1.0, 1.0, 1.0)

    fn _weights(self, reference_load: List[Float64]) -> (Float64, Float64):
        var scale = self.displacement_weight / self.reference_displacement
        var load_norm_sq = 0.0
        for load in reference_load:
            load_norm_sq += load * load
        return (scale * scale, self.force_weight * self.force_weight * load_norm_sq)

    fn predict(
        self,
        displacement: List[Float64],
        reference_load: List[Float64],
        last_increment: Float64,
    ) -> Float64:
        var weights = self._weights(reference_load)
        var norm_sq = 0.0
        for value in displacement:
            norm_sq += value * value
        var increment = self.length / sqrt(weights[0] * norm_sq + weights[1])
        return -increment if last_increment < 0.0 else increment

    fn correct(
        self,
        displacement: List[Float64],
        residual: List[Float64],
        prior: List[Float64],
        reference_load: List[Float64],
        lambda_step: Float64,
    ) -> Optional[Float64]:
        var weights = self._weights(reference_load)
        var length_sq = self.length * self.length
        var a1 = weights[1]
        var a2 = weights[1] * lambda_step
        var a3 = 0.0
        var projection = 0.0
        var previous_projection = 0.0
        var shift = -length_sq + lambda_step * lambda_step * weights[1]
        for i in range(len(displacement)):
            var direction = displacement[i]
            var total = prior[i] + residual[i]
            a1 += weights[0] * direction * direction
            a2 += weights[0] * direction * total
            # C++ applies the scalar shift to the RHS Vector before operator^.
            a3 += (weights[0] * total) * (total + shift)
            projection += direction * prior[i]
            previous_projection += total * prior[i]
        a2 *= 2.0
        var radicand = a2 * a2 - a1 * a3
        if radicand < 0.0:
            return None
        if a1 == 0.0:
            if a2 == 0.0:
                return None
            return -a3 / (2.0 * a2)
        var root = sqrt(radicand)
        var first = (-a2 + root) / a1
        var second = (-a2 - root) / a1
        var cosine_first = previous_projection + first * projection
        var cosine_second = previous_projection + second * projection
        return first if cosine_first > cosine_second else second
