from collections import List
from math import asin, sin, sqrt
from os import abort
from simd_math import pow_nonneg


from materials.multi_yield import (
    SoilVector,
    SoilTangent,
    SoilTensor,
    _LOW,
    _HIGH,
    _PI,
    _dot,
    _deviator,
    _scaled,
    _add_scaled,
    _equivalent,
    _positive_root,
    _Surface,
)


# Constitutive algorithm of OpenSees PressureIndependMultiYield (see COPYRIGHT.md).
struct PressureIndependMultiYieldDef(Copyable, Movable):
    var density: Float64
    var shear: Float64
    var bulk: Float64
    var reference_pressure: Float64
    var residual_pressure: Float64
    var friction_angle: Float64
    var pressure_exponent: Float64
    var surfaces: List[_Surface]

    fn __init__(
        out self,
        density: Float64,
        shear: Float64,
        bulk: Float64,
        cohesion: Float64,
        peak_shear_strain: Float64,
        friction_angle: Float64,
        reference_pressure: Float64,
        pressure_exponent: Float64,
        num_surfaces: Int,
        reduction_curve: List[Float64],
    ):
        if shear <= 0.0 or bulk <= 0.0 or peak_shear_strain <= 0.0 or reference_pressure <= 0.0:
            abort(
                "PressureIndependMultiYield: moduli, peak strain, and reference pressure must be"
                " positive"
            )
        self.density = max(density, 0.0)
        self.shear = shear
        self.bulk = bulk
        self.reference_pressure = reference_pressure
        self.residual_pressure = 0.0
        self.friction_angle = max(friction_angle, 0.0)
        self.pressure_exponent = max(pressure_exponent, 0.0)
        self.surfaces = [_Surface(0.0, 0.0)]
        var count = min(num_surfaces, 100)
        if count <= 0:
            count = 10
        var strength = max(cohesion, 0.0)
        if self.friction_angle == 0.0 and strength == 0.0:
            abort("PressureIndependMultiYield: cohesion and friction angle cannot both be zero")

        var cone_height = 1.0
        if len(reduction_curve) != 0:
            if count < 2 or len(reduction_curve) != 2 * count:
                abort("PressureIndependMultiYield: invalid reduction curve size")
            var peak = shear * reduction_curve[2 * count - 2] * reduction_curve[2 * count - 1]
            if self.friction_angle > 0.0:
                var ratio = (sqrt(3.0) * peak - 2.0 * strength) / reference_pressure
                if ratio <= 0.0:
                    self.friction_angle = 0.0
                else:
                    var sin_phi = 3.0 * ratio / (6.0 + ratio)
                    if sin_phi > 1.0:
                        abort("PressureIndependMultiYield: invalid reduction curve friction angle")
                    self.friction_angle = asin(sin_phi) * 180.0 / _PI
                    self.residual_pressure = 2.0 * strength / ratio
                    cone_height = reference_pressure + self.residual_pressure
            for i in range(count):
                var strain = reduction_curve[2 * i]
                var stress = shear * reduction_curve[2 * i + 1] * strain
                var modulus = 0.0
                if i + 1 < count:
                    var next_strain = reduction_curve[2 * i + 2]
                    var next_stress = shear * reduction_curve[2 * i + 3] * next_strain
                    modulus = self._plastic_modulus(
                        2.0 * (next_stress - stress) / (next_strain - strain)
                    )
                    if modulus <= 0.0:
                        abort("PressureIndependMultiYield: invalid reduction curve plastic modulus")
                self.surfaces.append(_Surface(sqrt(3.0) * stress / cone_height, modulus))
        else:
            var peak_shear = 2.0 * sqrt(2.0) * strength / 3.0
            if self.friction_angle > 0.0:
                var sin_phi = sin(self.friction_angle * _PI / 180.0)
                var ratio = 6.0 * sin_phi / (3.0 - sin_phi)
                self.residual_pressure = 3.0 * strength / (sqrt(2.0) * ratio)
                cone_height = reference_pressure + self.residual_pressure
                peak_shear = sqrt(2.0) * cone_height * ratio / 3.0
            var ref_strain = (
                peak_shear_strain * peak_shear / (shear * peak_shear_strain - peak_shear)
            )
            var increment = peak_shear / Float64(count)
            for i in range(1, count + 1):
                var stress = Float64(i) * increment
                var next_stress = stress + increment
                var size = 3.0 * stress / (sqrt(2.0) * cone_height)
                var modulus = 0.0
                if i < count:
                    var strain = stress * ref_strain / (shear * ref_strain - stress)
                    var next_strain = next_stress * ref_strain / (shear * ref_strain - next_stress)
                    modulus = max(
                        0.0,
                        self._plastic_modulus(
                            2.0 * (next_stress - stress) / (next_strain - strain)
                        ),
                    )
                self.surfaces.append(_Surface(size, modulus))
        if self.friction_angle == 0.0:
            self.pressure_exponent = 0.0

    fn _plastic_modulus(self, slope: Float64) -> Float64:
        var elastic = 2.0 * self.shear
        if elastic - slope <= 0.0:
            return _HIGH
        return min(_HIGH, elastic * slope / (elastic - slope))


struct PressureIndependMultiYieldState(Copyable, Movable):
    var shear: Float64
    var bulk: Float64
    var stress_c: SoilTensor
    var stress_t: SoilTensor
    var strain_c: SoilVector
    var strain_t: SoilVector
    var _plastic_initialized: Bool
    var _active_c: Int
    var _active_t: Int
    var _surfaces_c: List[_Surface]
    var _surfaces_t: List[_Surface]

    fn __init__(out self, material: PressureIndependMultiYieldDef):
        self.shear = material.shear
        self.bulk = material.bulk
        self.stress_c = SoilTensor()
        self.stress_t = self.stress_c
        self.strain_c = SoilVector(fill=0.0)
        self.strain_t = self.strain_c
        self._plastic_initialized = False
        self._active_c = 0
        self._active_t = 0
        self._surfaces_c = material.surfaces.copy()
        self._surfaces_t = self._surfaces_c.copy()

    fn _initialize_plasticity(mut self, material: PressureIndependMultiYieldDef):
        self._plastic_initialized = True
        var stress = self.stress_c.deviator
        var failure = len(self._surfaces_c) - 1
        if material.friction_angle > 0.0:
            if self.stress_c.mean > 0.0:
                self.stress_c = SoilTensor(stress, 0.0)
                stress = self.stress_c.deviator
            var height = material.residual_pressure - self.stress_c.mean
            var scale = pow_nonneg(
                height / (material.reference_pressure + material.residual_pressure),
                material.pressure_exponent,
            )
            self.shear *= scale
            self.bulk *= scale
            for surface in range(1, failure + 1):
                self._surfaces_c[surface].size *= height
                self._surfaces_c[surface].modulus *= scale
        var equivalent_stress = _equivalent(stress)
        if equivalent_stress != 0.0:
            while self._active_c < failure:
                if self._surfaces_c[self._active_c + 1].yield_value(stress) <= 0.0:
                    break
                self._active_c += 1
            if self._active_c == failure:
                var size = self._surfaces_c[failure].size
                var yield_value = self._surfaces_c[failure].yield_value(stress)
                if abs(yield_value) > _LOW:
                    var scaled_stress = _scaled(stress, size / sqrt(yield_value + size * size))
                    self.stress_c = SoilTensor(scaled_stress, self.stress_c.mean)
                    stress = self.stress_c.deviator
            for i in range(1, min(self._active_c, failure - 1) + 1):
                self._surfaces_c[i].center = _scaled(
                    stress, 1.0 - self._surfaces_c[i].size / _equivalent(stress)
                )

    fn set_trial_strain(
        mut self,
        material: PressureIndependMultiYieldDef,
        stage: Int,
        engineering_strain: SoilVector,
    ):
        self.strain_t = engineering_strain
        if stage == 1 and not self._plastic_initialized:
            self._initialize_plasticity(material)
        var strain_increment = _add_scaled(engineering_strain, self.strain_c, -1.0)
        for i in range(3, 6):
            strain_increment[i] *= 0.5
        var strain_rate = SoilTensor(strain_increment)
        if stage != 1:
            self.stress_t = self.stress_c.elastic_stress_increment(
                strain_rate, self.shear, self.bulk, 1.0
            )
            return
        var pressure = self.stress_c.mean + self.bulk * (3.0 * strain_rate.mean)
        var stress = self.stress_c.deviator
        var shear_increment = strain_rate.deviator
        var elastic = 2.0 * self.shear
        for i in range(1, len(self._surfaces_c)):
            self._surfaces_t[i] = self._surfaces_c[i]
        self._active_t = self._active_c
        var prediction = _deviator(_add_scaled(stress, shear_increment, elastic))
        if self._active_t > 0:
            var active = self._surfaces_t[self._active_t]
            var normal = _add_scaled(stress, active.center, -1.0)
            normal = _scaled(normal, 1.0 / sqrt(_dot(normal, normal)))
            if _dot(_add_scaled(prediction, stress, -1.0), normal) < 0.0:
                for i in range(1, self._active_t):
                    self._surfaces_t[i].center = _add_scaled(
                        stress,
                        _add_scaled(active.center, stress, -1.0),
                        self._surfaces_t[i].size / active.size,
                    )
                self._active_t = 0
        if _dot(strain_increment, strain_increment) == 0.0:
            self.stress_t = SoilTensor(prediction, pressure)
            return
        var failure = len(self._surfaces_t) - 1
        var modulus = elastic
        if self._active_t > 0:
            var plastic = self._surfaces_t[self._active_t].modulus
            modulus = elastic * plastic / (elastic + plastic)
        var surface_spacing = self._surfaces_t[failure].size / Float64(failure)
        var stress_increment = _deviator(_scaled(shear_increment, modulus))
        var crossings = (
            3.0
            * sqrt(1.0 / 3.0)
            * sqrt(_dot(stress_increment, stress_increment))
            / sqrt(2.0)
            / surface_spacing
        )
        var subdivisions = min(failure, Int(crossings) + 1)
        var increment = _deviator(_scaled(strain_increment, 1.0 / Float64(subdivisions)))
        for _ in range(subdivisions):
            stress = _deviator(_add_scaled(stress, increment, elastic))
            if self._active_t == 0:
                if self._surfaces_t[1].yield_value(stress) <= 0.0:
                    continue
                self._active_t = 1
            stress = self._correct_stress(stress, failure)
            if self._active_t < failure:
                self._translate_surface(stress)
        self.stress_t = SoilTensor(stress, pressure)

    fn _correct_stress(mut self, prediction: SoilVector, failure: Int) -> SoilVector:
        var stress = prediction
        var crossed = False
        var elastic = 2.0 * self.shear
        while True:
            var surface = self._surfaces_t[self._active_t]
            var offset = _add_scaled(stress, surface.center, -1.0)
            var contact = _deviator(
                _add_scaled(surface.center, offset, surface.size / _equivalent(offset))
            )
            var normal = _add_scaled(contact, surface.center, -1.0)
            normal = _scaled(normal, 1.0 / sqrt(_dot(normal, normal)))
            var loading = _dot(normal, _add_scaled(stress, contact, -1.0)) / (
                elastic + surface.modulus
            )
            if crossed:
                var previous_modulus = self._surfaces_t[self._active_t - 1].modulus
                loading *= (previous_modulus - surface.modulus) / previous_modulus
            stress = _deviator(_add_scaled(stress, normal, -elastic * loading))

            var yield_value = surface.yield_value(stress)
            if self._active_t == failure:
                if abs(yield_value) > _LOW:
                    stress = _deviator(
                        _scaled(
                            stress, surface.size / sqrt(yield_value + surface.size * surface.size)
                        )
                    )
                break
            while yield_value < 0.0:
                var magnitude = sqrt(surface.size * surface.size + yield_value)
                var correction = max(1.0e-13, (surface.size - magnitude) / magnitude)
                stress = _deviator(
                    _add_scaled(stress, _add_scaled(stress, surface.center, -1.0), correction)
                )
                yield_value = surface.yield_value(stress)
            if self._surfaces_t[self._active_t + 1].yield_value(stress) <= 0.0:
                break
            self._active_t += 1
            crossed = True
        return stress

    fn _translate_surface(mut self, stress: SoilVector):
        var surface = self._surfaces_t[self._active_t]
        var outer = self._surfaces_t[self._active_t + 1]
        var offset = _add_scaled(stress, surface.center, -1.0)
        var centers = _add_scaled(surface.center, outer.center, -1.0)
        var offset_sq = _dot(offset, offset)
        var projection = _positive_root(
            offset_sq,
            2.0 * _dot(offset, centers),
            _dot(centers, centers) - (2.0 / 3.0) * outer.size * outer.size,
        )
        if projection < 1.0 - _LOW:
            abort("PressureIndependMultiYield: outer surface projection < 1")
        if abs(projection - 1.0) < _LOW:
            projection = 1.0
        var ratio = surface.size / outer.size
        var projected = _scaled(_add_scaled(surface.center, offset, projection), 1.0 - ratio)
        var direction = _deviator(
            _add_scaled(projected, _add_scaled(surface.center, outer.center, -ratio), -1.0)
        )
        var a = _dot(direction, direction)
        if sqrt(a) < _LOW:
            return
        var b = -2.0 * _dot(offset, direction)
        if abs(b) < _LOW:
            b = 0.0
        var c = offset_sq - (2.0 / 3.0) * surface.size * surface.size
        if abs(c) < _LOW or abs(c) / offset_sq < _LOW or (abs(b) < 1.0e-10 and abs(c) < 1.0e-10):
            return
        if b > 0.0 or c < 0.0:
            abort("PressureIndependMultiYield: invalid active surface motion")
        self._surfaces_t[self._active_t].center = _add_scaled(
            surface.center, direction, _positive_root(a, b, c)
        )

    fn initial_tangent(self) -> SoilTangent:
        var tangent = SoilTangent(fill=0.0)
        for i in range(6):
            for j in range(6):
                var value = 0.0
                if i == j:
                    value += self.shear
                    if i < 3:
                        value += self.shear
                if i < 3 and j < 3:
                    value += self.bulk - 2.0 * self.shear / 3.0
                tangent[6 * i + j] = value
        return tangent

    fn tangent(mut self, material: PressureIndependMultiYieldDef, stage: Int) -> SoilTangent:
        if stage == 1 and not self._plastic_initialized:
            self._initialize_plasticity(material)
        var tangent = self.initial_tangent()
        if stage == 1 and self._active_t > 0:
            var surface = self._surfaces_t[self._active_t]
            var offset = _add_scaled(self.stress_t.deviator, surface.center, -1.0)
            var coeff = (
                6.0
                * self.shear
                * self.shear
                / (2.0 * self.shear + surface.modulus)
                / surface.size
                / surface.size
            )
            for i in range(6):
                for j in range(6):
                    tangent[6 * i + j] -= coeff * offset[i] * offset[j]
        return tangent

    fn commit(mut self, stage: Int):
        self.stress_c = self.stress_t
        self.strain_c = self.strain_t
        if stage == 1:
            self._active_c = self._active_t
            for i in range(1, len(self._surfaces_c)):
                self._surfaces_c[i] = self._surfaces_t[i]

    fn committed_stress_ratio(self, stage: Int) -> Float64:
        if stage != 1:
            return 0.0
        return (
            _equivalent(self.stress_c.deviator) / self._surfaces_c[len(self._surfaces_c) - 1].size
        )
