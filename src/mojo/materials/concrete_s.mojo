from math import atan2, cos, exp, sin, sqrt
from utils import StaticTuple


alias PlaneStressVector = StaticTuple[Float64, 3]
alias PlaneStressTangent = StaticTuple[Float64, 9]


struct ConcreteSDef(ImplicitlyCopyable, Movable):
    var E: Float64
    var fc: Float64
    var ft: Float64
    var Ep: Float64
    var inverse_modulus: Float64
    var beta: Float64
    var initial_tangent: PlaneStressTangent

    fn __init__(out self, E: Float64, nu: Float64, fc: Float64, ft: Float64, Es: Float64):
        self.E = E
        self.fc = abs(fc)
        self.ft = ft
        self.Ep = 10.0 * E
        self.inverse_modulus = 1.0 / (E + self.Ep)
        self.beta = -(self.Ep + 11.0 * abs(Es)) / ft
        var modulus = E / (1.0 - nu * nu)
        self.initial_tangent = PlaneStressTangent(fill=0.0)
        self.initial_tangent[0] = modulus
        self.initial_tangent[1] = nu * modulus
        self.initial_tangent[3] = self.initial_tangent[1]
        self.initial_tangent[4] = modulus
        self.initial_tangent[8] = (0.5 * (1.0 - nu)) * modulus


struct ConcreteSState(ImplicitlyCopyable, Movable):
    var strain_t: PlaneStressVector
    var effective_stress_t: PlaneStressVector
    var crack_strain_t: Float64
    var strain_c: PlaneStressVector
    var effective_stress_c: PlaneStressVector
    var crack_strain_c: Float64

    fn __init__(out self):
        self.strain_t = PlaneStressVector(fill=0.0)
        self.effective_stress_t = PlaneStressVector(fill=0.0)
        self.crack_strain_t = 0.0
        self.strain_c = self.strain_t
        self.effective_stress_c = self.effective_stress_t
        self.crack_strain_c = 0.0

    fn commit(mut self):
        self.strain_c = self.strain_t
        self.effective_stress_c = self.effective_stress_t
        self.crack_strain_c = self.crack_strain_t


@always_inline
fn _elastic_product(
    tangent: PlaneStressTangent, x: Float64, y: Float64, xy: Float64
) -> PlaneStressVector:
    var result = PlaneStressVector(fill=0.0)
    result[0] = tangent[0] * x + tangent[1] * y
    result[1] = tangent[3] * x + tangent[4] * y
    result[2] = tangent[8] * xy
    return result


@always_inline
fn _principal_to_global(
    major: Float64, minor: Float64, c: Float64, s: Float64
) -> PlaneStressVector:
    var mean = 0.5 * (major + minor)
    var radius = 0.5 * abs(major - minor)
    var result = PlaneStressVector(fill=0.0)
    var normal = radius * c
    result[0] = mean + normal
    result[1] = mean - normal
    result[2] = radius * s
    return result


fn concrete_s_set_trial_strain(
    material: ConcreteSDef,
    mut state: ConcreteSState,
    x: Float64,
    y: Float64,
    xy: Float64,
    mut stress: PlaneStressVector,
    mut tangent: PlaneStressTangent,
):
    state.strain_t[0] = x
    state.strain_t[1] = y
    state.strain_t[2] = xy
    tangent = material.initial_tangent
    var predictor = _elastic_product(
        tangent, x - state.strain_c[0], y - state.strain_c[1], xy - state.strain_c[2]
    )
    for i in range(3):
        predictor[i] += state.effective_stress_c[i]

    var sx = predictor[0]
    var sy = predictor[1]
    var tau = predictor[2]
    var equivalent = sqrt(sx * sx - sx * sy + sy * sy + tau * tau * 3.0)
    var yield_value = equivalent - material.fc
    var mean = 0.5 * (sx + sy)
    var diff = 0.5 * (sx - sy)
    var radius = sqrt(diff * diff + tau * tau)
    var major = mean + radius
    var minor = mean - radius

    if yield_value > 0.0 and major <= material.ft and minor <= material.ft:
        var inverse_equivalent = 1.0 / equivalent
        var flow_x = (sx - 0.5 * sy) * inverse_equivalent
        var flow_y = (sy - 0.5 * sx) * inverse_equivalent
        var flow_xy = 3.0 * tau * inverse_equivalent
        var cf = _elastic_product(tangent, flow_x, flow_y, flow_xy)
        var fcf = flow_x * cf[0] + flow_y * cf[1] + flow_xy * cf[2]
        var multiplier = yield_value / fcf
        var correction = _elastic_product(
            tangent, multiplier * flow_x, multiplier * flow_y, multiplier * flow_xy
        )
        sx -= correction[0]
        sy -= correction[1]
        tau -= correction[2]
        var denominator = fcf + 1.0e-3 * material.E
        for i in range(3):
            for j in range(3):
                tangent[3 * i + j] -= cf[i] * cf[j] / denominator

    mean = 0.5 * (sx + sy)
    diff = 0.5 * (sx - sy)
    var theta = atan2(tau, diff)
    radius = sqrt(diff * diff + tau * tau)
    major = mean + radius
    minor = mean - radius
    if minor > 0.0 and major < -material.fc:
        major = -material.fc
    if major > 0.0 and minor < -material.fc:
        minor = -material.fc

    # OpenSees keeps the previous trial crack strain when neither surface is
    # exceeded. Its revertToLastCommit() is a no-op, including on retries.
    var tensile_limit = material.ft + material.Ep * state.crack_strain_c
    yield_value = major - tensile_limit
    if yield_value > 0.0:
        state.crack_strain_t = state.crack_strain_c + yield_value * material.inverse_modulus
        major = material.ft + material.Ep * state.crack_strain_t
    yield_value = minor - tensile_limit
    if yield_value > 0.0:
        var crack_strain = state.crack_strain_c + yield_value * material.inverse_modulus
        minor = material.ft + material.Ep * crack_strain
        if state.crack_strain_t < crack_strain:
            state.crack_strain_t = crack_strain

    var c = cos(theta)
    var s = sin(theta)
    state.effective_stress_t = _principal_to_global(major, minor, c, s)
    var damage = exp(material.beta * state.crack_strain_t)
    if major > 0.0:
        major *= damage
    if minor > 0.0:
        minor *= damage
    stress = _principal_to_global(major, minor, c, s)
