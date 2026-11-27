from collections import List
from math import cos, exp, log, sin, sqrt
from os import abort
from simd_math import pow_nonneg
from materials.pressure_depend_multi_yield_02 import (
    PressureDependMultiYield02History,
    PressureDependMultiYield02Parameters,
)

from materials.multi_yield import (
    SoilTangent,
    SoilTensor,
    SoilVector,
    _HIGH,
    _LOW,
    _PI,
    _Surface,
    _add_scaled,
    _deviator,
    _dot,
    _equivalent,
    _positive_root,
    _scaled,
)


# Constitutive algorithm of OpenSees PressureDependMultiYield (see COPYRIGHT.md).
alias _LOCK_STRESS = 1.0e30


fn _divide(vector: SoilVector, divisor: Float64) -> SoilVector:
    var result = vector
    for i in range(6):
        result[i] /= divisor
    return result


struct PressureDependMultiYieldDef(Copyable, Movable):
    var shear: Float64
    var bulk: Float64
    var reference_pressure: Float64
    var residual_pressure: Float64
    var pressure_exponent: Float64
    var phase_ratio: Float64
    var phase_strain: Float64
    var contraction: Float64
    var dilate1: Float64
    var dilate2: Float64
    var liquefy_pressure: Float64
    var liquefy_size: Float64
    var liquefy_translation: Float64
    var void_ratio: Float64
    var vol_limit1: Float64
    var vol_limit2: Float64
    var vol_limit3: Float64
    var atmospheric_pressure: Float64
    var hv: Float64
    var pv: Float64
    var surfaces: List[_Surface]
    var variant_02: PressureDependMultiYield02Parameters

    fn __init__(
        out self,
        shear: Float64,
        bulk: Float64,
        friction_angle: Float64,
        peak_shear_strain: Float64,
        reference_pressure: Float64,
        pressure_exponent: Float64,
        phase_angle: Float64,
        contraction: Float64,
        dilate1: Float64,
        dilate2: Float64,
        liquefy_pressure: Float64,
        liquefy_size: Float64,
        liquefy_translation: Float64,
        count: Int,
        reduction_curve: List[Float64],
        void_ratio: Float64,
        vol_limit1: Float64,
        vol_limit2: Float64,
        vol_limit3: Float64,
        atmospheric_pressure: Float64,
        cohesion: Float64,
        hv: Float64,
        pv: Float64,
        variant_02: PressureDependMultiYield02Parameters = PressureDependMultiYield02Parameters(),
    ):
        if shear <= 0.0 or bulk <= 0.0 or peak_shear_strain <= 0.0 or reference_pressure <= 0.0:
            abort(
                "PressureDependMultiYield requires positive moduli, peak strain and reference"
                " pressure"
            )
        if friction_angle <= 0.0 or friction_angle >= 90.0 or phase_angle <= 0.0:
            abort("PressureDependMultiYield invalid friction or phase transformation angle")
        if void_ratio < 0.0:
            abort("PressureDependMultiYield requires nonnegative initial void ratio")
        self.shear = shear
        self.bulk = bulk
        self.reference_pressure = reference_pressure
        self.variant_02 = variant_02
        self.pressure_exponent = 0.5 if variant_02.enabled and pressure_exponent < 0.0 else max(
            0.0, pressure_exponent
        )
        self.contraction = contraction
        self.dilate1 = dilate1
        self.dilate2 = dilate2
        self.liquefy_pressure = liquefy_pressure
        self.liquefy_size = liquefy_size
        self.liquefy_translation = liquefy_translation
        self.void_ratio = void_ratio
        self.vol_limit1 = vol_limit1 if vol_limit1 >= 0.0 else 0.8
        self.vol_limit2 = vol_limit2
        self.vol_limit3 = vol_limit3
        self.atmospheric_pressure = atmospheric_pressure
        self.hv = hv
        self.pv = pv
        var n = min(100 if variant_02.enabled else 40, count) if count > 0 else 10
        var phi = sin(friction_angle * _PI / 180.0)
        var strength = 6.0 * phi / (3.0 - phi)
        var has_curve = len(reduction_curve) != 0
        var valid_cohesion = 0.3 if variant_02.enabled and cohesion < 0.0 else max(0.0, cohesion)
        if has_curve:
            if n < 2 or len(reduction_curve) != 2 * n:
                abort("PressureDependMultiYield invalid reduction curve length")
            var peak = shear * reduction_curve[2 * n - 2] * reduction_curve[2 * n - 1]
            strength = (sqrt(3.0) * peak - 2.0 * valid_cohesion) / reference_pressure
            phi = 3.0 * strength / (6.0 + strength)
            if phi < 0.0 or phi > 1.0:
                abort("PressureDependMultiYield reduction curve has invalid friction angle")
        var phase_sine = min(phi, sin(min(phase_angle, friction_angle) * _PI / 180.0))
        self.phase_ratio = 6.0 * phase_sine / (3.0 - phase_sine)
        self.phase_strain = 0.0
        self.residual_pressure = max(2.0 * valid_cohesion / strength, 0.0001 * atmospheric_pressure)
        self.surfaces = [_Surface(0.0, 0.0)]
        var height = reference_pressure + self.residual_pressure
        var peak_shear = sqrt(2.0) * height * strength / 3.0
        var reference_strain = (
            peak_shear_strain * peak_shear / (shear * peak_shear_strain - peak_shear)
        )
        var stress_step = peak_shear / Float64(n)
        for i in range(1, n + 1):
            var s1 = Float64(i) * stress_step
            var s2 = s1 + stress_step
            var e1 = s1 * reference_strain / (shear * reference_strain - s1)
            var e2 = s2 * reference_strain / (shear * reference_strain - s2)
            var ratio1 = 3.0 * s1 / sqrt(2.0) / height
            var ratio2 = 3.0 * s2 / sqrt(2.0) / height
            if has_curve:
                e1 = reduction_curve[2 * i - 2]
                s1 = shear * reduction_curve[2 * i - 1] * e1
                if i < n:
                    e2 = reduction_curve[2 * i]
                    s2 = shear * reduction_curve[2 * i + 1] * e2
                ratio1 = sqrt(3.0) * s1 / height
                ratio2 = sqrt(3.0) * s2 / height
            if (
                ratio1 <= self.phase_ratio
                and ratio2 >= self.phase_ratio
                and (not has_curve or i < n)
            ):
                var ratio = (ratio2 - self.phase_ratio) / (ratio2 - ratio1)
                self.phase_strain = e2 - ratio * (e2 - e1)
                if has_curve:
                    self.phase_strain *= sqrt(6.0) / 3.0
            var modulus = 0.0
            if i < n:
                var slope = 2.0 * (s2 - s1) / (e2 - e1)
                var elastic = 2.0 * shear
                modulus = _HIGH
                if elastic - slope > 0.0:
                    modulus = min(_HIGH, elastic * slope / (elastic - slope))
                if has_curve and modulus <= 0.0:
                    abort("PressureDependMultiYield invalid reduction curve plastic modulus")
            self.surfaces.append(_Surface(ratio1, max(0.0, modulus)))

    fn modulus_factor(self, pressure: Float64) -> Float64:
        var ratio = (self.residual_pressure - pressure) / (
            self.reference_pressure + self.residual_pressure
        )
        return max(1.0e-10, pow_nonneg(ratio, self.pressure_exponent))

    fn stress_ratio(self, stress: SoilTensor) -> Float64:
        return _equivalent(stress.deviator) / (abs(stress.mean) + self.residual_pressure)

    fn yield_value(self, stress: SoilTensor, surface: _Surface) -> Float64:
        var height = stress.mean - self.residual_pressure
        var offset = _add_scaled(stress.deviator, surface.center, -height)
        var radius = surface.size * height
        return 1.5 * _dot(offset, offset) - radius * radius

    fn normal(self, stress: SoilTensor, surface: _Surface) -> SoilTensor:
        var height = stress.mean - self.residual_pressure
        var center = surface.center
        var size = surface.size
        var mean = height * (_dot(center, center) - (2.0 / 3.0) * size * size) - _dot(
            stress.deviator, center
        )
        var dev = _scaled(_add_scaled(stress.deviator, center, -height), 3.0)
        var tensor = SoilTensor(dev, mean)
        var vector = tensor.vector()
        return SoilTensor(_divide(vector, max(_LOW, sqrt(_dot(vector, vector)))))

    fn zone_limit(self, pressure: Float64) -> Float64:
        if -pressure >= self.liquefy_pressure:
            return 0.0
        var cosine = cos(-pressure * _PI / self.liquefy_pressure / 2.0)
        return self.liquefy_size * cosine * cosine * cosine

    fn critical_void_ratio(self, pressure: Float64) -> Float64:
        var p = abs(-pressure / self.atmospheric_pressure)
        if self.vol_limit3 == 0.0:
            return self.vol_limit1 - self.vol_limit2 * log(p)
        return self.vol_limit1 - self.vol_limit2 * pow_nonneg(p, self.vol_limit3)


struct _PhaseHistory(ImplicitlyCopyable, Movable):
    var pressure: Float64
    var reversal: SoilTensor
    var on_zone: Int
    var zone_size: Float64
    var dilative_strain: Float64
    var max_dilative_strain: Float64
    var translation: Float64
    var previous_strain: Float64
    var opposite_strain: Float64
    var pivot: SoilTensor
    var center: SoilTensor
    var locked_stress: SoilTensor

    fn __init__(out self):
        self.pressure = 0.0
        self.reversal = SoilTensor()
        self.on_zone = -1
        self.zone_size = 0.0
        self.dilative_strain = 0.0
        self.max_dilative_strain = 0.0
        self.translation = 0.0
        self.previous_strain = 0.0
        self.opposite_strain = 0.0
        self.pivot = SoilTensor()
        self.center = SoilTensor()
        self.locked_stress = SoilTensor()

    fn translate(
        mut self, material: PressureDependMultiYieldDef, pressure: Float64, strain: SoilTensor
    ):
        if material.liquefy_pressure == 0.0:
            return
        var limit = material.zone_limit(pressure)
        if limit == 0.0:
            return
        var increment = SoilTensor(_add_scaled(strain.deviator, self.pivot.deviator, -1.0))
        self.translation = max(self.translation, increment.octahedral_strain())
        var max_translation = material.liquefy_translation * limit
        if self.max_dilative_strain != 0.0:
            max_translation *= self.dilative_strain / self.max_dilative_strain
        self.translation = min(self.translation, max_translation)

    fn update(
        mut self,
        material: PressureDependMultiYieldDef,
        contact: SoilTensor,
        strain: SoilTensor,
        increment: SoilTensor,
    ):
        var strain_vector = strain.vector()
        var pivot_delta = SoilTensor(_add_scaled(strain_vector, self.pivot.vector(), -1.0))
        if material.liquefy_pressure == 0.0:
            if self.on_zone == 2:
                self.dilative_strain = pivot_delta.octahedral_strain()
            else:
                self.on_zone = 2
                self.pivot = strain
                self.dilative_strain = 0.0
            return
        if self.on_zone == 2:
            self.pivot = strain
            self.dilative_strain += increment.octahedral_strain()
            self.max_dilative_strain = max(self.max_dilative_strain, self.dilative_strain)
            return
        var limit = material.zone_limit(contact.mean)
        var translation_limit = material.liquefy_translation * limit
        var pivot_distance = pivot_delta.octahedral_strain()
        if self.on_zone <= 0 and pivot_distance > self.dilative_strain:
            self.opposite_strain = self.previous_strain
            var ratio = (material.residual_pressure - contact.mean) / (
                material.reference_pressure + material.residual_pressure
            )
            self.previous_strain = (
                pow_nonneg(ratio, 1.0 - material.pressure_exponent) * material.phase_strain
            )
            if self.opposite_strain == 0.0:
                self.opposite_strain = self.previous_strain
        var width = self.previous_strain + self.opposite_strain + self.max_dilative_strain
        if self.on_zone > -1:
            width += translation_limit
        self.zone_size = limit + width / 2.0
        if self.on_zone == 0 or self.on_zone == 1:
            var delta = SoilTensor(_add_scaled(self.pivot.vector(), self.center.vector(), -1.0))
            var scale = (self.zone_size - self.translation) / delta.octahedral_strain()
            self.center = SoilTensor(_add_scaled(self.pivot.vector(), delta.vector(), -scale))
        var from_center = SoilTensor(_add_scaled(strain_vector, self.center.vector(), -1.0))
        if limit == 0.0 or (
            from_center.octahedral_strain() > self.zone_size
            and _dot(increment.vector(), from_center.vector()) > 0.0
        ):
            if pivot_distance > self.dilative_strain:
                self.dilative_strain = 0.0
                if limit == 0.0:
                    self.max_dilative_strain = 0.0
            self.on_zone = 2
            self.pivot = strain
            self.translation = 0.0
        else:
            if self.on_zone >= 0:
                self.translate(material, contact.mean, strain)
            if self.on_zone <= 0:
                self.locked_stress = contact
            if self.on_zone == 0:
                self.on_zone = 1


struct PressureDependMultiYieldState(Copyable, Movable):
    var stress_c: SoilTensor
    var strain_c: SoilTensor
    var stress_t: SoilTensor
    var strain_t: SoilTensor
    var _surfaces_c: List[_Surface]
    var _surfaces_t: List[_Surface]
    var _history_c: _PhaseHistory
    var _history_t: _PhaseHistory
    var _history02_c: PressureDependMultiYield02History
    var _history02_t: PressureDependMultiYield02History
    var _updated02_t: SoilTensor
    var _updated02_c: SoilTensor
    var _rate02: SoilTensor
    var _active_c: Int
    var _active_t: Int
    var _plastic_initialized: Bool
    var _initial_pressure: Float64
    var _max_compression: Float64
    var _factor: Float64

    fn __init__(out self, material: PressureDependMultiYieldDef):
        self.stress_c = SoilTensor()
        self.strain_c = SoilTensor()
        self.stress_t = SoilTensor()
        self.strain_t = SoilTensor()
        self._surfaces_c = material.surfaces.copy()
        self._surfaces_t = material.surfaces.copy()
        self._history_c = _PhaseHistory()
        self._history_t = _PhaseHistory()
        self._history02_c = PressureDependMultiYield02History()
        self._history02_t = PressureDependMultiYield02History()
        self._updated02_t = SoilTensor()
        self._updated02_c = SoilTensor()
        self._rate02 = SoilTensor()
        self._active_c = 0
        self._active_t = 0
        self._plastic_initialized = False
        self._initial_pressure = -material.reference_pressure
        self._max_compression = 0.0
        self._factor = 1.0

    fn _initialize_plasticity(mut self, material: PressureDependMultiYieldDef):
        self._plastic_initialized = True
        if material.variant_02.enabled:
            self._initial_pressure = self.stress_c.mean
        if self.stress_c.mean > 0.0:
            self.stress_c = SoilTensor(self.stress_c.deviator, 0.0)
        var magnitude = _equivalent(self.stress_c.deviator)
        if magnitude == 0.0:
            return
        var height = self.stress_c.mean - material.residual_pressure
        var failure = len(self._surfaces_c) - 1
        while self._active_c < failure:
            if material.yield_value(self.stress_c, self._surfaces_c[self._active_c + 1]) <= 0.0:
                break
            self._active_c += 1
        if self._active_c == failure:
            self.stress_c = self._scale_deviator(
                material, self.stress_c, self._surfaces_c[failure], True
            )
            magnitude = _equivalent(self.stress_c.deviator)
        for i in range(1, min(self._active_c, failure - 1) + 1):
            var size = self._surfaces_c[i].size
            self._surfaces_c[i].center = _divide(
                _scaled(self.stress_c.deviator, 1.0 + size * height / magnitude), height
            )
        self._active_t = self._active_c
        self._surfaces_t[self._active_t] = self._surfaces_c[self._active_c]

    fn _scale_deviator(
        self,
        material: PressureDependMultiYieldDef,
        stress: SoilTensor,
        surface: _Surface,
        failure: Bool,
    ) -> SoilTensor:
        var result = stress
        var diff = material.yield_value(result, surface)
        var height = result.mean - material.residual_pressure
        var radius = -surface.size * height
        if failure:
            if abs(diff) > _LOW:
                result = SoilTensor(
                    _scaled(result.deviator, radius / sqrt(diff + radius * radius)), result.mean
                )
        else:
            while diff < 0.0:
                var magnitude = sqrt(radius * radius + diff)
                var correction = max(1.0e-13, (radius - magnitude) / magnitude)
                var offset = _add_scaled(result.deviator, surface.center, -height)
                result = SoilTensor(_add_scaled(result.deviator, offset, correction), result.mean)
                diff = material.yield_value(result, surface)
        return result

    fn _elastic_factor(mut self, material: PressureDependMultiYieldDef, stage: Int) -> Float64:
        if stage == 2:
            if self._initial_pressure == -material.reference_pressure:
                self._initial_pressure = self.stress_c.mean
            var ratio = (material.residual_pressure - self._initial_pressure) / (
                material.reference_pressure + material.residual_pressure
            )
            if ratio <= 1.0e-10:
                return 1.0e-10
            return max(1.0e-10, pow_nonneg(ratio, material.pressure_exponent))
        if stage == 1:
            return material.modulus_factor(self.stress_c.mean)
        return 1.0

    fn _bulk_modulus(
        self, material: PressureDependMultiYieldDef, factor: Float64, strain_mean: Float64
    ) -> Float64:
        var bulk = material.bulk * factor
        if material.hv != 0.0 and self.stress_t.mean <= self._max_compression and strain_mean < 0.0:
            var plastic = material.hv * pow_nonneg(
                abs(self.stress_t.mean - material.residual_pressure), material.pv
            )
            bulk = bulk * plastic / (bulk + plastic)
        return bulk

    fn _predict(
        mut self, material: PressureDependMultiYieldDef, stress: SoilTensor, increment: SoilTensor
    ):
        self._factor = material.modulus_factor(stress.mean)
        var bulk = self._bulk_modulus(material, self._factor, increment.mean)
        var dev = _add_scaled(
            stress.deviator, increment.deviator, 2.0 * material.shear * self._factor
        )
        self.stress_t = SoilTensor(dev, min(0.0, stress.mean + increment.mean * 3.0 * bulk))

    fn _reverse_if_unloading(mut self, material: PressureDependMultiYieldDef, previous: SoilTensor):
        if self._active_t == 0:
            return
        var active = self._surfaces_t[self._active_t]
        var normal = material.normal(previous, active)
        var stress_change = _add_scaled(self.stress_t.vector(), self.stress_c.vector(), -1.0)
        if _dot(stress_change, normal.vector()) >= 0.0:
            return
        var height = self.stress_c.mean - material.residual_pressure
        var dev = self.stress_c.deviator
        var offset = _add_scaled(_scaled(active.center, height), dev, -1.0)
        for i in range(1, self._active_t):
            self._surfaces_t[i].center = _divide(
                _add_scaled(dev, offset, self._surfaces_t[i].size / active.size), height
            )
        self._active_t = 0

    fn _crosses_next(self, material: PressureDependMultiYieldDef) -> Bool:
        return (
            self._active_t + 1 < len(self._surfaces_t)
            and material.yield_value(self.stress_t, self._surfaces_t[self._active_t + 1]) > 0.0
        )

    fn _subdivisions(self, material: PressureDependMultiYieldDef, increment: SoilTensor) -> Int:
        var failure = len(self._surfaces_t) - 1
        if self._active_t == failure and not material.variant_02.enabled:
            return 1
        if _dot(increment.vector(), increment.vector()) == 0.0:
            return 0
        var factor = material.modulus_factor(self.stress_c.mean)
        var modulus = 2.0 * material.shear * factor
        if self._active_t > 0:
            var plastic = self._surfaces_t[self._active_t].modulus * factor
            modulus = modulus * plastic / (modulus + plastic)
        var height = material.residual_pressure - self.stress_c.mean
        var spacing = self._surfaces_t[failure].size * height / Float64(failure)
        var estimated = _deviator(_scaled(increment.deviator, modulus))
        var octahedral = sqrt(1.0 / 3.0) * sqrt(_dot(estimated, estimated))
        var crossings = 3.0 * octahedral / sqrt(2.0) / spacing
        var subdivisions = min(failure, Int(crossings) + 1)
        var octahedral_limit = 1.0e-5 if material.variant_02.enabled else 1.0e-4
        subdivisions = max(subdivisions, Int(increment.octahedral_strain() / octahedral_limit))
        return max(subdivisions, Int(increment.mean / 1.0e-5))

    fn set_trial_strain(
        mut self, material: PressureDependMultiYieldDef, stage: Int, engineering_strain: SoilVector
    ):
        var strain = engineering_strain
        for i in range(3, 6):
            strain[i] *= 0.5
        var increment = SoilTensor(_add_scaled(strain, self.strain_c.vector(), -1.0))
        self.strain_t = SoilTensor(_add_scaled(self.strain_c.vector(), increment.vector(), 1.0))
        if stage != 1:
            var factor = self._elastic_factor(material, stage)
            self.stress_t = self.stress_c.elastic_stress_increment(
                increment, material.shear, material.bulk, factor
            )
            return
        if not self._plastic_initialized:
            self._initialize_plasticity(material)
        for i in range(1, len(self._surfaces_c)):
            self._surfaces_t[i] = self._surfaces_c[i]
        self._active_t = self._active_c
        self._history_t = self._history_c
        self._history02_t = self._history02_c
        self._rate02 = increment
        var committed_stress = self.stress_c
        if material.variant_02.enabled and _dot(increment.vector(), increment.vector()) == 0.0:
            # Reprojecting a zero strain increment can round the reconstructed
            # stress below the active surface and spuriously trigger reversal.
            self.stress_t = committed_stress
            return
        self._predict(material, committed_stress, increment)
        self._reverse_if_unloading(material, committed_stress)
        if self._active_t == 0 and not self._crosses_next(material):
            return
        var subdivisions = self._subdivisions(material, increment)
        if subdivisions == 0:
            return
        var sub_increment = SoilTensor(_scaled(increment.vector(), 1.0 / Float64(subdivisions)))
        var previous = committed_stress
        for i in range(subdivisions):
            var sub_strain = SoilTensor(
                _add_scaled(self.strain_c.vector(), sub_increment.vector(), Float64(i + 1))
            )
            var reversal_stress = previous if i == 0 else SoilTensor(previous.vector())
            if material.variant_02.enabled:
                self._updated02_t = previous
            self._predict(material, previous, sub_increment)
            self._reverse_if_unloading(material, reversal_stress)
            if self._active_t == 0:
                if not self._crosses_next(material):
                    previous = self.stress_t
                    continue
                self._active_t = 1
            if not self._correct_stress(material, sub_strain, sub_increment):
                self._translate_surface(material)
            previous = self.stress_t

    fn _critical_state(
        self, material: PressureDependMultiYieldDef, contact: SoilTensor, strain: SoilTensor
    ) -> Bool:
        var trial_volume = 3.0 * strain.mean
        var committed_volume = 3.0 * self.strain_c.mean
        var trial_void = material.void_ratio + trial_volume + trial_volume * material.void_ratio
        var committed_void = (
            material.void_ratio + committed_volume + committed_volume * material.void_ratio
        )
        var trial_limit = material.critical_void_ratio(contact.mean)
        var committed_pressure = (
            self._updated02_t.mean if material.variant_02.enabled else self.stress_c.mean
        )
        var committed_limit = material.critical_void_ratio(committed_pressure)
        return not (
            (committed_void < committed_limit and trial_void < trial_limit)
            or (committed_void > committed_limit and trial_void > trial_limit)
        )

    fn _plastic_potential(
        mut self,
        material: PressureDependMultiYieldDef,
        contact: SoilTensor,
        normal: SoilTensor,
        strain: SoilTensor,
        increment: SoilTensor,
    ) -> Float64:
        if material.variant_02.enabled:
            var potential = self._history02_t.plastic_potential(
                material.variant_02,
                material.contraction,
                material.dilate1,
                material.dilate2,
                material.residual_pressure,
                material.atmospheric_pressure,
                material.phase_ratio,
                self.stress_c,
                self._updated02_t,
                self.stress_t,
                contact,
                self._max_compression,
                strain,
                self._rate02,
                increment,
                self._history02_c.pivot_rate,
            )
            return 0.0 if self._critical_state(material, contact, strain) else potential
        var ratio = material.stress_ratio(contact) / material.phase_ratio
        var phase = (ratio * ratio - 1.0) / (ratio * ratio + 1.0) / 3.0
        var contract = max(-5.0e4, -abs(phase * material.contraction))
        var unloading = 0.0
        var pressure = self._history_t.pressure
        if self.stress_c.mean < pressure:
            var height = pressure - material.residual_pressure
            var committed_dev = self.stress_c.deviator
            var slope = (
                sqrt(1.5) * sqrt(_dot(committed_dev, committed_dev)) + material.phase_ratio * height
            ) / (pressure - self.stress_c.mean)
            slope = max(slope, self._surfaces_t[len(self._surfaces_t) - 1].size)
            var reversal_ratio = (
                self._history_t.reversal.mean - material.residual_pressure
            ) / height
            unloading = (
                -sqrt(1.5) * sqrt(_dot(normal.deviator, normal.deviator)) * reversal_ratio / slope
            )
        var shear_loading = (
            material.stress_ratio(self.stress_t) > material.stress_ratio(self.stress_c)
            and _dot(self.stress_c.deviator, self.stress_t.deviator) >= 0.0
        )
        if phase >= 0.0 and shear_loading:
            if pressure == 0.0:
                self._history_t.pressure = self.stress_c.mean
            self._history_t.reversal = self.stress_c
            self._history_t.update(material, contact, strain, increment)
            if self._history_t.on_zone == -1 or self._history_t.on_zone == 1:
                return _LOCK_STRESS
            if self._critical_state(material, contact, strain):
                return 0.0
            return min(
                5.0e4,
                phase * material.dilate1 * exp(material.dilate2 * self._history_t.dilative_strain),
            )

        var potential = contract
        if pressure != 0.0:
            if self.stress_t.mean >= pressure:
                self._history_t.pressure = 0.0
            elif not (phase < 0.0 and shear_loading):
                potential = unloading
        if self._history_t.on_zone > 0:
            self._history_t.on_zone = 0
        if self._history_t.on_zone != -1:
            self._history_t.translate(material, contact.mean, strain)
        if self._critical_state(material, contact, strain):
            return 0.0
        return potential

    fn _correct_stress(
        mut self, material: PressureDependMultiYieldDef, strain: SoilTensor, increment: SoilTensor
    ) -> Bool:
        var crossed = False
        var failure = len(self._surfaces_t) - 1
        var shear = material.shear * self._factor
        var bulk = material.bulk * self._factor
        while True:
            var surface = self._surfaces_t[self._active_t]
            var height = self.stress_t.mean - material.residual_pressure
            var offset = _add_scaled(self.stress_t.deviator, surface.center, -height)
            var contact = SoilTensor(
                _add_scaled(
                    _scaled(surface.center, height),
                    offset,
                    -surface.size * height / _equivalent(offset),
                ),
                self.stress_t.mean,
            )
            var normal = material.normal(contact, surface)
            var potential = self._plastic_potential(material, contact, normal, strain, increment)
            if potential == _LOCK_STRESS:
                self.stress_t = self._history_t.locked_stress
                return True
            var normal_work = 2.0 * shear * _dot(normal.deviator, normal.deviator)
            var denominator = (
                normal_work + 9.0 * bulk * normal.mean * potential + surface.modulus * self._factor
            )
            var limiting_modulus = surface.modulus
            if self._active_t == failure:
                limiting_modulus = self._surfaces_t[self._active_t - 1].modulus
            var limit = limiting_modulus * self._factor / 2.0
            if denominator < limit:
                if material.variant_02.enabled:
                    var multiplier = 9.0 * bulk * normal.mean
                    potential = (multiplier * potential + limit - denominator) / multiplier
                denominator = limit
            var loading = max(
                0.0,
                _dot(normal.vector(), _add_scaled(self.stress_t.deviator, contact.deviator, -1.0))
                / denominator,
            )
            if crossed:
                var previous_modulus = self._surfaces_t[self._active_t - 1].modulus
                loading *= (previous_modulus - surface.modulus) / previous_modulus
            var trial_pressure = self.stress_t.mean
            var pressure = trial_pressure - 3.0 * bulk * potential * loading
            var correction = -2.0 * shear * loading
            if pressure > 0.0 and pressure != trial_pressure:
                correction *= trial_pressure / (trial_pressure - pressure)
            self.stress_t = SoilTensor(
                _add_scaled(self.stress_t.deviator, normal.deviator, correction), min(0.0, pressure)
            )
            self.stress_t = self._scale_deviator(
                material, self.stress_t, surface, self._active_t == failure
            )
            if not self._crosses_next(material):
                return False
            self._active_t += 1
            crossed = True

    fn _translate_surface(mut self, material: PressureDependMultiYieldDef):
        if self._active_t == len(self._surfaces_t) - 1:
            return
        var active = self._surfaces_t[self._active_t]
        var outer = self._surfaces_t[self._active_t + 1]
        var height = self.stress_t.mean - material.residual_pressure
        var offset = _add_scaled(self.stress_t.deviator, active.center, -height)
        var centers = _scaled(_add_scaled(active.center, outer.center, -1.0), height)
        var offset_sq = _dot(offset, offset)
        var projection = _positive_root(
            offset_sq,
            2.0 * _dot(offset, centers),
            _dot(centers, centers) - (2.0 / 3.0) * outer.size * outer.size * height * height,
        )
        if abs(projection - 1.0) < _LOW:
            projection = 1.0
        if projection < 1.0:
            return
        var ratio = active.size / outer.size
        var projected = _scaled(
            _add_scaled(_scaled(active.center, height), offset, projection), 1.0 - ratio
        )
        var direction = _deviator(
            _add_scaled(projected, _add_scaled(active.center, outer.center, -ratio), -height)
        )
        var direction_sq = _dot(direction, direction)
        if sqrt(direction_sq) < _LOW:
            return
        var b = 2.0 * height * _dot(offset, direction)
        if abs(b) < _LOW:
            b = 0.0
        var c = offset_sq - (2.0 / 3.0) * active.size * active.size * height * height
        if abs(c) < _LOW or abs(c) / offset_sq < _LOW:
            return
        if b > 0.0 or c < 0.0:
            abort("PressureDependMultiYield: invalid active surface motion")
        var distance = _positive_root(height * height * direction_sq, b, c)
        self._surfaces_t[self._active_t].center = _add_scaled(active.center, direction, -distance)

    fn tangent(
        mut self,
        material: PressureDependMultiYieldDef,
        stage: Int,
        dimensions: Int,
        initial: Bool = False,
    ) -> SoilTangent:
        if stage == 1 and not initial and not self._plastic_initialized:
            self._initialize_plasticity(material)
        var factor = self._elastic_factor(material, stage)
        if material.variant_02.enabled and stage == 1 and not initial:
            var pressure = self.stress_t.mean if self._active_t > 0 else self._updated02_t.mean
            factor = material.modulus_factor(pressure)
        var shear = material.shear * factor
        var bulk = material.bulk * factor
        var normal = SoilTensor()
        var volume_term = 0.0
        var shear_term = 0.0
        if stage == 1 and not initial:
            var modulus = 0.0
            var active = self._active_c > 0
            if material.variant_02.enabled:
                active = self._active_t > 0
                if active:
                    normal = material.normal(self.stress_t, self._surfaces_t[self._active_t])
                    modulus = self._surfaces_t[self._active_t].modulus
                else:
                    bulk = self._bulk_modulus(material, factor, self._rate02.mean)
            else:
                bulk = self._bulk_modulus(material, factor, self.strain_t.mean - self.strain_c.mean)
                if active:
                    normal = material.normal(self.stress_c, self._surfaces_t[self._active_t])
                    modulus = self._surfaces_c[self._active_c].modulus
            if active:
                var denominator = (
                    9.0 * bulk * normal.mean * normal.mean
                    + (2.0 * shear * _dot(normal.deviator, normal.deviator))
                    + factor * modulus
                )
                volume_term = 9.0 * bulk * bulk * normal.mean * normal.mean / denominator
                shear_term = 4.0 * shear * shear / denominator
        var matrix = SoilTangent(fill=0.0)
        for i in range(6):
            for j in range(6):
                var value = -shear_term * normal.deviator[i] * normal.deviator[j]
                if i == j:
                    value += shear if i >= 3 else 2.0 * shear
                if i < 3 and j < 3:
                    value += bulk - 2.0 * shear / 3.0 - volume_term
                if dimensions == 2 and ((i < 3) != (j < 3)):
                    value = 0.0
                matrix[6 * i + j] = value
        return matrix

    fn commit(mut self, stage: Int):
        self.stress_c = self.stress_t
        self.strain_c = self.strain_t
        if stage == 1:
            self._active_c = self._active_t
            self._history_c = self._history_t
            self._history02_c = self._history02_t
            self._updated02_c = self._updated02_t
            for i in range(1, len(self._surfaces_t)):
                self._surfaces_c[i] = self._surfaces_t[i]
            self._max_compression = min(self._max_compression, self.stress_c.mean)

    fn revert(mut self):
        self.stress_t = self.stress_c
        self.strain_t = self.strain_c
        self._active_t = self._active_c
        self._history_t = self._history_c
        self._history02_t = self._history02_c
        self._updated02_t = self._updated02_c
        self._rate02 = SoilTensor()
        for i in range(1, len(self._surfaces_c)):
            self._surfaces_t[i] = self._surfaces_c[i]

    fn committed_stress_ratio(self, material: PressureDependMultiYieldDef, stage: Int) -> Float64:
        if stage != 1:
            return 0.0
        return (
            material.stress_ratio(self.stress_c) / self._surfaces_c[len(self._surfaces_c) - 1].size
        )
