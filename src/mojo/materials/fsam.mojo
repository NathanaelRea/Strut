# Fixed Strut Angle Model from OpenSees 3.7.1 FSAM.cpp.
from collections import InlineArray
from math import atan, cos, sin


alias PanelVector = SIMD[DType.float64, 4]
alias PanelMatrix = InlineArray[PanelVector, 3]
alias FSAM_PI = 3.1415926535


fn perpendicular(angle: Float64) -> Float64:
    return angle - 0.5 * FSAM_PI if angle >= 0.0 else angle + 0.5 * FSAM_PI


fn normal_direction(angle: Float64) -> PanelVector:
    var c = cos(2.0 * angle)
    var s = sin(2.0 * angle)
    return PanelVector(0.5 * (1.0 + c), 0.5 * (1.0 - c), 0.5 * s, 0.0)


fn shear_direction(angle: Float64) -> PanelVector:
    var s = sin(2.0 * angle)
    return PanelVector(-s, s, cos(2.0 * angle), 0.0)


@fieldwise_init
struct FSAMInput(ImplicitlyCopyable, Movable):
    var density: Float64
    var steel_x: Int
    var steel_y: Int
    var concrete: Int
    var steel_x_ratio: Float64
    var steel_y_ratio: Float64
    var friction: Float64
    var dowel: Float64


@fieldwise_init
struct FSAMParameters(ImplicitlyCopyable, Movable):
    var density: Float64
    var steel_x_ratio: Float64
    var steel_y_ratio: Float64
    var friction: Float64
    var dowel: Float64
    var concrete_modulus: Float64
    var concrete_peak_strain: Float64
    var cracking_strain: Float64
    var steel_x_modulus: Float64
    var steel_y_modulus: Float64

    fn initial_tangent(self) -> PanelMatrix:
        return PanelMatrix(
            PanelVector(
                self.concrete_modulus + self.steel_x_ratio * self.steel_x_modulus, 0.0, 0.0, 0.0
            ),
            PanelVector(
                0.0, self.concrete_modulus + self.steel_y_ratio * self.steel_y_modulus, 0.0, 0.0
            ),
            PanelVector(0.0, 0.0, self.concrete_modulus / 2.0, 0.0),
        )


@fieldwise_init
struct FSAMCrack(Defaultable, ImplicitlyCopyable, Movable):
    var angle: Float64
    var compression_1: Float64
    var compression_2: Float64
    var slip: Float64
    var interlock: Float64

    fn __init__(out self):
        self = FSAMCrack(10.0, 0.0, 0.0, 0.0, 0.0)


@fieldwise_init
struct FSAMResponse(ImplicitlyCopyable, Movable):
    var strain: PanelVector
    var stress: PanelVector
    var tangent: PanelMatrix
    var concrete_stress: PanelVector
    var steel_stress: PanelVector
    var material_strain: InlineArray[Float64, 8]
    var material_stress: InlineArray[Float64, 8]

    fn __init__(out self, params: FSAMParameters):
        self = FSAMResponse(
            PanelVector(0.0),
            PanelVector(0.0),
            params.initial_tangent(),
            PanelVector(0.0),
            PanelVector(0.0),
            InlineArray[Float64, 8](fill=0.0),
            InlineArray[Float64, 8](fill=0.0),
        )


struct FSAMState(ImplicitlyCopyable, Movable):
    var params: FSAMParameters
    var crack_count: Int
    var principal_angle: Float64
    var cracks: InlineArray[FSAMCrack, 3]
    var trial_cracks: InlineArray[FSAMCrack, 3]
    var trial: FSAMResponse
    var committed: FSAMResponse
    var converged_stress: PanelVector

    fn __init__(out self, params: FSAMParameters):
        self.params = params
        self.crack_count = 0
        self.principal_angle = 0.0
        self.cracks = InlineArray[FSAMCrack, 3](fill=FSAMCrack())
        self.trial_cracks = self.cracks
        self.trial = FSAMResponse(params)
        self.committed = self.trial
        self.converged_stress = PanelVector(0.0)

    fn set_strain(mut self, strain: PanelVector):
        self.trial.strain = strain
        self.trial_cracks = self.cracks
        self.trial.concrete_stress = PanelVector(0.0)
        self.trial.steel_stress = PanelVector(0.0)
        self.trial.tangent = PanelMatrix(fill=PanelVector(0.0))
        var effective = strain
        if effective[2] == 0.0:
            effective[2] = 1e-20
        var difference = effective[0] - effective[1]
        self.principal_angle = 0.5 * atan(effective[2] / difference)
        self.trial.material_strain[0] = effective[0]
        self.trial.material_strain[1] = (normal_direction(-0.5 * FSAM_PI) * effective).reduce_add()
        if self.crack_count == 0:
            var radius = 0.5 * effective[2] / sin(2.0 * self.principal_angle)
            var mean = 0.5 * (effective[0] + effective[1])
            self.trial.material_strain[2] = mean + radius
            self.trial.material_strain[3] = mean - radius
        else:
            for i in range(1, self.crack_count + 1):
                var angle = self.cracks[i].angle
                self.trial.material_strain[2 * i + 2] = (
                    normal_direction(angle) * effective
                ).reduce_add()
                self.trial.material_strain[2 * i + 3] = (
                    normal_direction(perpendicular(angle)) * effective
                ).reduce_add()
                self.trial_cracks[i].slip = (shear_direction(angle) * effective).reduce_add()

    fn _softening(
        self, strain: Float64, stress: Float64, min_strain: Float64
    ) -> (Float64, Float64):
        var peak = self.params.concrete_peak_strain
        var beta = 1.0 / (1.0 + 0.27 * (-strain / peak - 0.37))
        var slope = beta * beta * 0.27 / peak
        if beta > 1.0 or strain < 0.0 or stress > 0.0:
            beta = 1.0
            slope = 0.0
        return (beta * (1.0 - 0.4 * min(min_strain / peak, 1.0)), slope)

    fn _interlock(self, index: Int, normal_stress: Float64) -> (Float64, Float64, Float64):
        ref old = self.cracks[index]
        var slip = self.trial_cracks[index].slip - old.slip
        if normal_stress > 0.0 or (normal_stress == 0.0 and slip >= 0.0):
            return (0.0, 0.0, 0.0)
        var stiffness = 0.4 * self.params.concrete_modulus
        var stress = old.interlock + stiffness * slip
        var limit = self.params.friction * normal_stress
        if slip >= 0.0 and stress > -limit:
            return (-limit, 0.0, -self.params.friction)
        if slip <= 0.0 and stress < limit:
            return (limit, 0.0, self.params.friction)
        return (stress, stiffness, 0.0)

    fn _add_strut(
        mut self,
        index: Int,
        tangent_1: Float64,
        tangent_2: Float64,
    ):
        var k = 2 * index + 2
        var strain_1 = self.trial.material_strain[k]
        var strain_2 = self.trial.material_strain[k + 1]
        var stress_1 = self.trial.material_stress[k]
        var stress_2 = self.trial.material_stress[k + 1]
        var min_1 = min(strain_1, self.cracks[index].compression_1)
        var min_2 = min(strain_2, self.cracks[index].compression_2)
        self.trial_cracks[index].compression_1 = min_1
        self.trial_cracks[index].compression_2 = min_2
        var beta_1 = self._softening(strain_2, stress_1, min_2)
        var beta_2 = self._softening(strain_1, stress_2, min_1)
        var softened_1 = beta_1[0] * stress_1
        var softened_2 = beta_2[0] * stress_2
        var angle = self.principal_angle if index == 0 else self.cracks[index].angle
        var n1 = normal_direction(angle)
        var n2 = PanelVector(n1[1], n1[0], -n1[2], 0.0)
        var shear = shear_direction(angle)
        var d1 = beta_1[0] * tangent_1 * n1 + beta_1[1] * stress_1 * n2
        var d2 = beta_2[1] * stress_2 * n1 + beta_2[0] * tangent_2 * n2
        var rotation = PanelVector(0.0)
        var slip_tangent = PanelVector(0.0)
        var slip_stress = 0.0
        if index == 0:
            var difference = self.trial.strain[0] - self.trial.strain[1]
            var gamma = self.trial.strain[2]
            if gamma == 0.0:
                gamma = 1e-20
            rotation = PanelVector(-gamma, gamma, difference, 0.0) / (
                2.0 * (difference * difference + gamma * gamma)
            )
        else:
            var interlock = self._interlock(index, softened_1)
            slip_stress = interlock[0]
            self.trial_cracks[index].interlock = slip_stress
            if self.crack_count == 2:
                # The two-strut model uses only the compression direction of each
                # strut in the stress resultant and its linearization.
                softened_1 = 0.0
                d1 = PanelVector(0.0)
            slip_tangent = interlock[1] * shear + interlock[2] * d1
            if self.crack_count == 2 and index == 1:
                # The stage-three OpenSees tangent uses the second crack
                # angle for the first strut's interlock contribution to D22.
                var cos_second = cos(2.0 * self.cracks[2].angle)
                self.trial.tangent[2][2] += interlock[1] * (
                    cos_second * cos_second - shear[2] * shear[2]
                )

        self.trial.concrete_stress += softened_1 * n1 + softened_2 * n2 + slip_stress * shear
        for row in range(3):
            self.trial.tangent[row] += (
                n1[row] * d1
                + n2[row] * d2
                + shear[row] * slip_tangent
                + (softened_1 - softened_2) * shear[row] * rotation
            )

    fn set_response(mut self, tangent: InlineArray[Float64, 8]):
        if self.crack_count == 0:
            self._add_strut(0, tangent[2], tangent[3])
        else:
            for i in range(1, self.crack_count + 1):
                self._add_strut(i, tangent[2 * i + 2], tangent[2 * i + 3])

        var rho_x = self.params.steel_x_ratio
        var rho_y = self.params.steel_y_ratio
        self.trial.steel_stress[0] = rho_x * self.trial.material_stress[0]
        self.trial.steel_stress[1] = rho_y * self.trial.material_stress[1]
        self.trial.tangent[0][0] += rho_x * tangent[0]
        self.trial.tangent[1][1] += rho_y * tangent[1]
        var dowel = (
            Float64(self.crack_count) * rho_y * self.params.dowel * self.params.steel_y_modulus
        )
        self.trial.steel_stress[2] = dowel * self.trial.strain[2]
        self.trial.tangent[2][2] += dowel
        self.trial.stress = self.trial.concrete_stress + self.trial.steel_stress
        var strain = self.trial.strain
        if self.crack_count == 0 and strain[0] == 0.0 and strain[1] == 0.0 and strain[2] == 0.0:
            self.trial.tangent = self.params.initial_tangent()

    fn try_first_crack(mut self) -> Bool:
        var first = self.trial.material_strain[2]
        var second = self.trial.material_strain[3]
        if max(first, second) < self.params.cracking_strain:
            return False
        self.crack_count = 1
        self.cracks[1] = self.trial_cracks[0]
        self.cracks[1].angle = self.principal_angle if first >= second else perpendicular(
            self.principal_angle
        )
        return True

    fn try_second_crack(mut self, cyclic_cracking_strain: Float64) -> Bool:
        if self.trial.material_strain[5] < cyclic_cracking_strain:
            return False
        self.crack_count = 2
        self.cracks[2].angle = perpendicular(self.cracks[1].angle)
        return True

    fn revert(mut self):
        self.trial = self.committed
        self.trial_cracks = self.cracks
