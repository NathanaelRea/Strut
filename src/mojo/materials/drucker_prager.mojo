from collections import InlineArray
from math import exp, sqrt

alias DruckerPragerVector = InlineArray[Float64, 6]
alias DruckerPragerTangent = InlineArray[Float64, 36]
alias ROOT_TWO_THIRDS = 0.816496580927726


@fieldwise_init
struct DruckerPragerDefinition(ImplicitlyCopyable, Movable):
    var bulk: Float64
    var shear: Float64
    var yield_stress: Float64
    var friction: Float64
    var dilation: Float64
    var saturation: Float64
    var initial_hardening: Float64
    var saturation_rate: Float64
    var tension_rate: Float64
    var hardening: Float64
    var isotropic_fraction: Float64

    fn elastic_tangent(self) -> DruckerPragerTangent:
        var C = DruckerPragerTangent(fill=0.0)
        for i in range(3):
            for j in range(3):
                C[6 * i + j] = self.bulk + 2.0 * self.shear * _deviator(i, j)
        for i in range(3, 6):
            C[7 * i] = self.shear
        return C

    fn strength(self, alpha: Float64) -> Float64:
        return (
            self.yield_stress
            + self.isotropic_fraction * self.hardening * alpha
            + (self.saturation - self.initial_hardening)
            * (1.0 - exp(-self.saturation_rate * alpha))
        )

    fn strength_derivative(self, alpha: Float64) -> Float64:
        return self.isotropic_fraction * self.hardening + (
            self.saturation - self.initial_hardening
        ) * self.saturation_rate * exp(-self.saturation_rate * alpha)

    fn tensile_strength(self, alpha: Float64) -> Float64:
        var cutoff = 1.0e10
        if self.friction != 0.0:
            cutoff = ROOT_TWO_THIRDS * self.yield_stress / self.friction
        return cutoff * exp(-self.tension_rate * alpha)


fn _deviator(i: Int, j: Int) -> Float64:
    if i < 3 and j < 3:
        return 2.0 / 3.0 if i == j else -1.0 / 3.0
    return 0.5 if i == j else 0.0


fn _tensor_norm(x: DruckerPragerVector, engineering_shear: Bool) -> Float64:
    var normal = x[0] * x[0] + x[1] * x[1] + x[2] * x[2]
    var shear = x[3] * x[3] + x[4] * x[4] + x[5] * x[5]
    return sqrt(normal + (0.5 if engineering_shear else 2.0) * shear)


@fieldwise_init
struct DruckerPragerHistory(ImplicitlyCopyable, Movable):
    var plastic: DruckerPragerVector
    var backstress: DruckerPragerVector
    var alpha_cone: Float64
    var alpha_tension: Float64

    fn __init__(out self):
        self.plastic = DruckerPragerVector(fill=0.0)
        self.backstress = DruckerPragerVector(fill=0.0)
        self.alpha_cone = 0.0
        self.alpha_tension = 0.0


@fieldwise_init
struct DruckerPragerTrial(ImplicitlyCopyable, Movable):
    var history: DruckerPragerHistory
    var stress: DruckerPragerVector
    var strain: DruckerPragerVector
    var tangent: DruckerPragerTangent
    var shifted_deviator_norm: Float64

    fn state(self) -> List[Float64]:
        var p = self.history.plastic
        var trace = p[0] + p[1] + p[2]
        var norm = _tensor_norm(p, True)
        for i in range(3):
            p[i] -= trace / 3.0
        return [
            self.stress[0] + self.stress[1] + self.stress[2],
            self.shifted_deviator_norm,
            trace,
            _tensor_norm(p, True),
            norm,
        ]


fn drucker_prager_trial(
    mat: DruckerPragerDefinition,
    committed: DruckerPragerHistory,
    strain: DruckerPragerVector,
) -> DruckerPragerTrial:
    var C = mat.elastic_tangent()
    var stress = DruckerPragerVector(fill=0.0)
    for i in range(6):
        for j in range(6):
            stress[i] += C[6 * i + j] * (strain[j] - committed.plastic[j])
    var trace = stress[0] + stress[1] + stress[2]
    var eta = DruckerPragerVector(fill=0.0)
    for i in range(6):
        eta[i] = stress[i] - committed.backstress[i]
        if i < 3:
            eta[i] -= trace / 3.0
    var trial_norm = _tensor_norm(eta, False)
    var cone = trial_norm + mat.friction * trace > ROOT_TWO_THIRDS * mat.strength(
        committed.alpha_cone
    )
    var tension = trace > mat.tensile_strength(committed.alpha_tension)
    if not cone and not tension:
        return DruckerPragerTrial(committed, stress, strain, C, trial_norm)

    var n = DruckerPragerVector(fill=0.0)
    if trial_norm >= 1.0e-13:
        for i in range(6):
            n[i] = eta[i] / trial_norm
    var kinematic = (1.0 - mat.isotropic_fraction) * mat.hardening
    var hardening = 2.0 * mat.shear + 2.0 / 3.0 * kinematic
    var bulk9 = 9.0 * mat.bulk
    var gamma_cone = 0.0
    var gamma_tension = 0.0
    var alpha_cone = committed.alpha_cone
    var alpha_tension = committed.alpha_tension
    var inverse = InlineArray[Float64, 4](1.0, 0.0, 0.0, 1.0)
    for _ in range(5):
        gamma_cone = 0.0
        gamma_tension = 0.0
        alpha_cone = committed.alpha_cone
        alpha_tension = committed.alpha_tension
        for iteration in range(11):
            var flow = mat.dilation * gamma_cone + gamma_tension
            var residual = (
                trial_norm
                - hardening * gamma_cone
                + mat.friction * (trace - bulk9 * flow)
                - ROOT_TWO_THIRDS * mat.strength(alpha_cone)
            )
            var cutoff = mat.tensile_strength(alpha_tension)
            # OpenSees evaluates the cone residual for either active flag.
            # The second residual remains zero in the 3.7.1 return map.
            var a = (
                -hardening
                - 2.0 / 3.0 * mat.strength_derivative(alpha_cone)
                - bulk9 * mat.friction * mat.dilation
            )
            var b = -bulk9 * mat.friction if cone and tension else 0.0
            var c = mat.dilation * (-bulk9 + mat.tension_rate * cutoff) if cone and tension else 0.0
            var det = a - b * c
            inverse = InlineArray[Float64, 4](1.0 / det, -b / det, -c / det, a / det)
            if abs(residual) <= 1.0e-10 or iteration == 10:
                break
            gamma_cone -= inverse[0] * residual
            gamma_tension -= inverse[2] * residual
            alpha_cone = committed.alpha_cone + ROOT_TWO_THIRDS * gamma_cone
            alpha_tension = committed.alpha_tension + mat.dilation * gamma_cone + gamma_tension
        var remaining_tension = (
            trace
            - bulk9 * (mat.dilation * gamma_cone + gamma_tension)
            - mat.tensile_strength(alpha_tension)
        )
        if cone and not tension and remaining_tension >= 0.0:
            tension = True
        elif cone and tension and gamma_cone <= -1.0e-10 and gamma_tension > -1.0e-10:
            cone = False
        elif cone and tension and gamma_cone > -1.0e-10 and gamma_tension <= -1.0e-10:
            tension = False
        else:
            break

    var history = committed
    history.alpha_cone = alpha_cone
    history.alpha_tension = alpha_tension
    var flow = mat.dilation * gamma_cone + gamma_tension
    var bulk3 = 3.0 * mat.bulk
    for i in range(6):
        var normal = 1.0 if i < 3 else 0.0
        history.plastic[i] += flow * normal + gamma_cone * n[i] * (1.0 if i < 3 else 2.0)
        history.backstress[i] -= 2.0 / 3.0 * kinematic * gamma_cone * n[i]
        stress[i] -= bulk3 * flow * normal + 2.0 * mat.shear * gamma_cone * n[i]
        eta[i] -= hardening * gamma_cone * n[i]
    var norm = _tensor_norm(eta, False)
    var tangent = C
    var tangent_norm = 0.0
    for i in range(6):
        var normal_i = 1.0 if i < 3 else 0.0
        for j in range(6):
            var normal_j = 1.0 if j < 3 else 0.0
            var b1 = 2.0 * mat.shear * n[j] + bulk3 * mat.friction * normal_j if cone else 0.0
            var b2 = bulk3 * normal_j if tension else 0.0
            var dt1 = inverse[0] * b1 + inverse[1] * b2
            var dt2 = mat.dilation * dt1 + inverse[2] * b1 + inverse[3] * b2
            tangent[6 * i + j] += bulk3 * normal_i * dt2 + 2.0 * mat.shear * n[i] * dt1
            if norm > 1.0e-13:
                tangent[6 * i + j] -= (
                    4.0
                    * mat.shear
                    * mat.shear
                    * gamma_cone
                    / norm
                    * (_deviator(i, j) - n[i] * n[j])
                )
            tangent_norm += tangent[6 * i + j] * tangent[6 * i + j]
    if tangent_norm < 1.0e-10:
        for i in range(36):
            tangent[i] = 0.001 * C[i]
    return DruckerPragerTrial(history, stress, strain, tangent, norm)
