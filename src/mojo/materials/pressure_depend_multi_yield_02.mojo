from math import sqrt
from materials.multi_yield import SoilTensor, SoilVector, _add_scaled, _dot, _equivalent, _scaled
from simd_math import pow_nonneg


# OpenSees PressureDependMultiYield02 phase transformation law (see COPYRIGHT.md).
@fieldwise_init
struct PressureDependMultiYield02Parameters(ImplicitlyCopyable, Movable):
    var enabled: Bool
    var contraction2: Float64
    var contraction3: Float64
    var dilate3: Float64
    var liquefy1: Float64
    var liquefy2: Float64

    fn __init__(out self):
        self.enabled = False
        self.contraction2 = 0.0
        self.contraction3 = 0.0
        self.dilate3 = 0.0
        self.liquefy1 = 0.0
        self.liquefy2 = 0.0


struct PressureDependMultiYield02History(ImplicitlyCopyable, Movable):
    var on_zone: Int
    var dilative_strain: Float64
    var max_dilative_strain: Float64
    var translation: Float64
    var damage: Float64
    var pivot: SoilTensor
    var center: SoilTensor
    var pivot_rate: SoilVector

    fn __init__(out self):
        self.on_zone = -1
        self.dilative_strain = 0.0
        self.max_dilative_strain = 0.0
        self.translation = 0.0
        self.damage = 0.0
        self.pivot = SoilTensor()
        self.center = SoilTensor()
        self.pivot_rate = SoilVector(fill=0.0)

    fn translate(
        mut self,
        material: PressureDependMultiYield02Parameters,
        strain: SoilTensor,
        reversed: Bool,
        damage: Float64,
    ):
        if material.liquefy1 == 0.0:
            return
        self.damage = damage
        if not reversed:
            return
        var shift = SoilTensor(_add_scaled(strain.deviator, self.pivot.deviator, -1.0))
        self.translation = max(
            self.translation, damage * material.liquefy2 * shift.octahedral_strain()
        )

    fn dilate(
        mut self,
        material: PressureDependMultiYield02Parameters,
        strain: SoilTensor,
        rate: SoilTensor,
        sub_increment: SoilTensor,
        reversed: Bool,
        damage: Float64,
    ):
        var octahedral = sub_increment.octahedral_strain()
        if self.on_zone < 1:
            self.damage = damage
        if material.liquefy1 == 0.0:
            self.pivot = strain
            if self.on_zone == 2:
                self.dilative_strain += octahedral
            else:
                self.on_zone = 2
                self.pivot_rate = rate.deviator
                if reversed:
                    self.dilative_strain = 0.0
            return

        if self.on_zone == 2:
            self.pivot = strain
            self.dilative_strain += octahedral
            self.max_dilative_strain += self.damage * material.liquefy1 * octahedral
            return

        var zone_size = 0.5 * (self.translation + self.max_dilative_strain)
        if self.on_zone == 0 or (self.on_zone == 1 and reversed):
            var offset = SoilTensor(_add_scaled(self.pivot.vector(), self.center.vector(), -1.0))
            var magnitude = offset.octahedral_strain()
            var shift = (zone_size - self.translation) / magnitude if magnitude != 0.0 else 0.0
            self.center = SoilTensor(_add_scaled(self.pivot.vector(), offset.vector(), -shift))

        var distance = SoilTensor(_add_scaled(strain.vector(), self.center.vector(), -1.0))
        if distance.octahedral_strain() > zone_size:
            self.dilative_strain = 0.0
            self.translation = 0.0
            self.on_zone = 2
            self.pivot = strain
            self.pivot_rate = rate.deviator
        else:
            if self.on_zone == 0 or self.on_zone == 1:
                self.translate(material, strain, reversed, damage)
            self.on_zone = 1

    fn plastic_potential(
        mut self,
        material: PressureDependMultiYield02Parameters,
        contraction: Float64,
        dilate1: Float64,
        dilate2: Float64,
        residual_pressure: Float64,
        atmospheric_pressure: Float64,
        phase_ratio: Float64,
        committed_stress: SoilTensor,
        previous: SoilTensor,
        trial: SoilTensor,
        contact: SoilTensor,
        max_compression: Float64,
        strain: SoilTensor,
        rate: SoilTensor,
        sub_increment: SoilTensor,
        committed_pivot_rate: SoilVector,
    ) -> Float64:
        var previous_height = abs(previous.mean) + residual_pressure
        var trial_height = abs(trial.mean) + residual_pressure
        var contact_height = abs(contact.mean) + residual_pressure
        var previous_ratio = _equivalent(previous.deviator) / previous_height
        var trial_ratio = _equivalent(trial.deviator) / trial_height
        var phase = _equivalent(contact.deviator) / (contact_height * phase_ratio)
        var reversed = _dot(rate.deviator, committed_pivot_rate) < 0.0
        var damage = pow_nonneg(
            max(
                0.0,
                (max_compression - committed_stress.mean) / (max_compression - residual_pressure),
            ),
            0.25,
        )
        if (
            phase >= 1.0
            and trial_ratio >= previous_ratio
            and _dot(previous.deviator, trial.deviator) >= 0.0
        ):
            self.dilate(material, strain, rate, sub_increment, reversed, damage)
            if self.on_zone == 1:
                return 0.0
            var pressure_scale = pow_nonneg(
                contact_height / atmospheric_pressure, -material.dilate3
            )
            var distance = phase - 1.0
            var potential = (
                pressure_scale
                * distance
                * distance
                * (dilate1 + pow_nonneg(self.dilative_strain, dilate2))
            )
            return min(5.0e4, abs(potential))

        var angle = 1.0
        if previous_ratio != 0.0:
            var change = _add_scaled(
                _scaled(trial.deviator, 1.0 / trial_height),
                previous.deviator,
                -1.0 / previous_height,
            )
            var magnitude = sqrt(_dot(change, change))
            if magnitude != 0.0:
                angle = _dot(previous.deviator, change) / (
                    magnitude * sqrt(_dot(previous.deviator, previous.deviator))
                )
        var distance = phase * angle - 1.0
        var pressure_scale = max(
            0.1, pow_nonneg(contact_height / atmospheric_pressure, material.contraction3)
        )
        var potential = (
            distance
            * distance
            * (contraction + self.max_dilative_strain * material.contraction2)
            * pressure_scale
        )
        if self.on_zone > 0:
            self.on_zone = 0
        if self.on_zone != -1:
            self.translate(material, strain, reversed, damage)
        return -abs(potential)
