# Elastic catenary element from OpenSees 3.7.1 CatenaryCable.cpp.
from collections import InlineArray
from elements.utils import _invert_3x3_values
from math import atan2, cos, cosh, isfinite, log, sin, sinh, sqrt
from os import abort


alias CableVector = SIMD[DType.float64, 4]
alias CableMatrix = InlineArray[CableVector, 3]


@fieldwise_init
struct CatenaryCable(ImplicitlyCopyable, Movable):
    var weight: Float64
    var modulus: Float64
    var area: Float64
    var length: Float64
    var thermal_expansion: Float64
    var temperature_change: Float64
    var density: Float64
    var tolerance: Float64
    var substeps: Int
    var mass_type: Int

    fn __init__(out self):
        self = CatenaryCable(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 0)


@fieldwise_init
struct CableResponse(ImplicitlyCopyable, Movable):
    var start: CableVector
    var end: CableVector
    var stiffness: CableMatrix

    fn force(self) -> List[Float64]:
        return [self.start[0], self.start[1], self.start[2], self.end[0], self.end[1], self.end[2]]

    fn tangent(self) -> List[List[Float64]]:
        var matrix = List[List[Float64]](capacity=6)
        for i in range(6):
            var row = List[Float64](capacity=6)
            for j in range(6):
                var sign = -1.0 if (i < 3) == (j < 3) else 1.0
                row.append(sign * self.stiffness[i % 3][j % 3])
            matrix.append(row^)
        return matrix^


@fieldwise_init
struct CableEquilibrium(ImplicitlyCopyable, Movable):
    var length: Float64
    var axial_rigidity: Float64
    var thermal_factor: Float64
    var weight: CableVector
    var weight_norm: Float64

    fn _log_ratio(
        self,
        projection: Float64,
        t1: Float64,
        t2: Float64,
    ) -> Float64:
        return log(
            (projection / self.weight_norm + t1)
            / (self.length * self.weight_norm + projection / self.weight_norm + t2)
        )

    fn projected_chord(self, start: CableVector) -> CableVector:
        var w = self.weight_norm
        var projection = (start * self.weight).reduce_add()
        var end = -start - self.weight * self.length
        var t1 = sqrt((start * start).reduce_add())
        var t2 = sqrt((end * end).reduce_add())
        return (
            -self.length * start / self.axial_rigidity
            - self.length * self.length * self.weight / (2.0 * self.axial_rigidity)
            + (self.thermal_factor / (w * w * w))
            * (
                w * self.weight * (t1 - t2)
                + (w * w * start - projection * self.weight) * self._log_ratio(projection, t1, t2)
            )
        )

    fn inverse_flexibility(self, start: CableVector) -> CableMatrix:
        var w = self.weight_norm
        var projection = (start * self.weight).reduce_add()
        var end = -start - self.weight * self.length
        var t1 = sqrt((start * start).reduce_add())
        var t2 = sqrt((end * end).reduce_add())
        var logarithm = self._log_ratio(projection, t1, t2)
        var flex = CableMatrix(fill=CableVector(0.0))
        for i in range(3):
            for j in range(3):
                var geometric = -w * self.weight[i] * (end[j] / t2 + start[j] / t1) + (
                    w * w * start[i] - projection * self.weight[i]
                ) * (
                    (w * start[j] + self.weight[j] * (self.length * w + t2))
                    / (t2 * (self.length * w * w + projection + w * t2))
                    - (w * start[j] + self.weight[j] * t1) / (t1 * (projection + w * t1))
                )
                var weight_term = self.weight[i] * self.weight[j]
                if i == j:
                    weight_term -= w * w
                    flex[i][j] = -self.length / self.axial_rigidity
                flex[i][j] -= (
                    self.thermal_factor / (w * w * w) * (geometric + weight_term * logarithm)
                )
        var inverse = _invert_3x3_values(
            flex[0][0],
            flex[0][1],
            flex[0][2],
            flex[1][0],
            flex[1][1],
            flex[1][2],
            flex[2][0],
            flex[2][1],
            flex[2][2],
        )
        if not inverse[0]:
            abort("CatenaryCable flexibility is singular")
        return CableMatrix(
            CableVector(inverse[1], inverse[2], inverse[3], 0.0),
            CableVector(inverse[4], inverse[5], inverse[6], 0.0),
            CableVector(inverse[7], inverse[8], inverse[9], 0.0),
        )


fn catenary_cable_response(
    cable: CatenaryCable, chord: CableVector, weight: CableVector
) -> CableResponse:
    var horizontal_sq = chord[0] * chord[0] + chord[1] * chord[1]
    var vertical_sq = chord[2] * chord[2]
    var shape = 0.2
    if horizontal_sq == 0.0:
        shape = 1e6
    elif cable.length * cable.length > horizontal_sq + vertical_sq:
        shape = sqrt(3.0 * (cable.length * cable.length - vertical_sq) / horizontal_sq - 1.0)
    var azimuth = atan2(chord[1], chord[0])
    var horizontal_force = weight[2] * sqrt(horizontal_sq) / (2.0 * shape)
    var coth = 1.0 if shape > 10.0 else cosh(shape) / sinh(shape)
    var force = CableVector(
        cos(azimuth) * horizontal_force,
        sin(azimuth) * horizontal_force,
        -0.5 * weight[2] * (cable.length - chord[2] * coth),
        0.0,
    )
    var equilibrium = CableEquilibrium(
        cable.length,
        cable.modulus * cable.area,
        1.0 + cable.thermal_expansion * cable.temperature_change,
        weight,
        sqrt((weight * weight).reduce_add()),
    )
    var next_force = force
    var residual = chord - equilibrium.projected_chord(force)
    var s = 1.0 / Float64(cable.substeps)
    var iteration = 0
    while (
        max(abs(residual[0]), max(abs(residual[1]), abs(residual[2])))
        > cable.tolerance * cable.length
    ):
        residual = chord - equilibrium.projected_chord(next_force)
        for _ in range(cable.substeps):
            force = next_force
            var flexibility = equilibrium.inverse_flexibility(force)
            for i in range(3):
                next_force[i] += s * (flexibility[i] * residual).reduce_add()
        iteration += 1
        if iteration > 100:
            abort("CatenaryCable did not converge")
    if not isfinite(force[0]) or not isfinite(force[1]) or not isfinite(force[2]):
        abort("CatenaryCable produced a nonfinite end force")
    var tangent = equilibrium.inverse_flexibility(force)
    return CableResponse(force, -force - weight * cable.length, tangent)
