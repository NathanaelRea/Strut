from math import sqrt
from os import abort
from utils import StaticTuple


# Components are xx, yy, zz, xy, yz, zx. Internally shear strains are tensorial.
alias SoilVector = StaticTuple[Float64, 6]
alias SoilTangent = StaticTuple[Float64, 36]
alias _LOW = 4.440892098500626e-15
alias _HIGH = 1.0e30
alias _PI = 3.14159265358979


fn _dot(a: SoilVector, b: SoilVector) -> Float64:
    var result = 0.0
    for i in range(3):
        result += a[i] * b[i] + 2.0 * a[i + 3] * b[i + 3]
    return result


fn _mean(a: SoilVector) -> Float64:
    return (a[0] + a[1] + a[2]) / 3.0


fn _deviator(a: SoilVector) -> SoilVector:
    var result = a
    var mean = _mean(a)
    for i in range(3):
        result[i] -= mean
    return result


fn _scaled(a: SoilVector, factor: Float64) -> SoilVector:
    var result = SoilVector(fill=0.0)
    for i in range(6):
        result[i] = a[i] * factor
    return result


fn _add_scaled(a: SoilVector, b: SoilVector, factor: Float64) -> SoilVector:
    var result = SoilVector(fill=0.0)
    for i in range(6):
        result[i] = a[i] + b[i] * factor
    return result


fn _equivalent(a: SoilVector) -> Float64:
    return sqrt(1.5 * _dot(a, a))


fn _positive_root(a: Float64, b: Float64, c: Float64) -> Float64:
    if a == 0.0:
        abort("PressureIndependMultiYield: degenerate surface motion")
    if c == 0.0:
        return 0.0
    if b == 0.0:
        if c / a > 0.0:
            abort("PressureIndependMultiYield: complex surface motion roots")
        return sqrt(-c / a)
    var discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        abort("PressureIndependMultiYield: complex surface motion roots")
    var signed_root = sqrt(discriminant)
    if b > 0.0:
        signed_root = -signed_root
    var x1 = (-b + signed_root) / (2.0 * a)
    var x2 = c / (a * x1)
    if x1 < 0.0 and x2 < 0.0:
        if abs(x1) < _LOW or abs(x2) < _LOW:
            return 0.0
        abort("PressureIndependMultiYield: negative surface motion roots")
    if x1 < 0.0:
        return x2
    if x2 < 0.0:
        return x1
    return min(x1, x2)


struct _Surface(ImplicitlyCopyable, Movable):
    var center: SoilVector
    var size: Float64
    var modulus: Float64

    fn __init__(out self, size: Float64, modulus: Float64):
        self.center = SoilVector(fill=0.0)
        self.size = size
        self.modulus = modulus

    fn yield_value(self, deviator: SoilVector) -> Float64:
        var offset = _add_scaled(deviator, self.center, -1.0)
        return 1.5 * _dot(offset, offset) - self.size * self.size


struct SoilTensor(ImplicitlyCopyable, Movable):
    var deviator: SoilVector
    var mean: Float64
    var _vector: SoilVector

    fn __init__(out self):
        self.deviator = SoilVector(fill=0.0)
        self.mean = 0.0
        self._vector = self.deviator

    fn __init__(out self, deviator: SoilVector, mean: Float64):
        self.deviator = _deviator(deviator)
        self.mean = mean
        self._vector = self.deviator
        for i in range(3):
            self._vector[i] += mean

    fn __init__(out self, vector: SoilVector):
        self.deviator = _deviator(vector)
        self.mean = _mean(vector)
        self._vector = vector

    fn vector(self) -> SoilVector:
        return self._vector

    fn elastic_stress_increment(
        self, strain: SoilTensor, shear: Float64, bulk: Float64, factor: Float64
    ) -> SoilTensor:
        var result = self._vector
        var coupling = (bulk - 2.0 * shear / 3.0) * factor
        var scaled_shear = shear * factor
        var diagonal = 2.0 * scaled_shear + coupling
        for j in range(3):
            for i in range(3):
                var modulus = diagonal if i == j else coupling
                result[i] += modulus * strain._vector[j]
            result[j + 3] += scaled_shear * (2.0 * strain._vector[j + 3])
        return SoilTensor(result)

    fn octahedral_strain(self) -> Float64:
        return 2.0 * sqrt(1.0 / 3.0) * sqrt(_dot(self.deviator, self.deviator))
