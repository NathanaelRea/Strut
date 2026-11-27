from math import sqrt

from materials.uniaxial.core import _abs, _sign, UniMaterialDefinition, UniMaterialPointState
from simd_math import pow_nonneg


struct _VDStepResult(Defaultable, ImplicitlyCopyable, Movable):
    var yt: Float64
    var eps: Float64
    var error: Float64

    fn __init__(out self):
        self.yt = 0.0
        self.eps = 0.0
        self.error = 0.0

    fn __init__(out self, yt: Float64, eps: Float64, error: Float64):
        self.yt = yt
        self.eps = eps
        self.error = error


fn _vd_f(K: Float64, C: Float64, Alpha: Float64, v: Float64, fd: Float64) -> Float64:
    var abs_fd = _abs(fd)
    var rate = 0.0
    if abs_fd > 0.0:
        rate = pow_nonneg(abs_fd / C, 1.0 / Alpha)
    return (v - _sign(fd) * rate) * K


fn _vd_dormand_prince(
    K: Float64,
    C: Float64,
    Alpha: Float64,
    vel0: Float64,
    vel1: Float64,
    y0: Float64,
    h: Float64,
) -> _VDStepResult:
    var dv = vel1 - vel0
    var k1 = _vd_f(K, C, Alpha, vel0, y0) * h
    var k2 = _vd_f(K, C, Alpha, dv * (1.0 / 5.0) + vel0, y0 + (1.0 / 5.0) * k1) * h
    var k3 = (
        _vd_f(
            K,
            C,
            Alpha,
            dv * (3.0 / 10.0) + vel0,
            y0 + (3.0 / 40.0) * k1 + (9.0 / 40.0) * k2,
        )
        * h
    )
    var k4 = (
        _vd_f(
            K,
            C,
            Alpha,
            dv * (4.0 / 5.0) + vel0,
            y0 + (44.0 / 45.0) * k1 + (-56.0 / 15.0) * k2 + (32.0 / 9.0) * k3,
        )
        * h
    )
    var k5 = (
        _vd_f(
            K,
            C,
            Alpha,
            dv * (8.0 / 9.0) + vel0,
            y0
            + (19372.0 / 6561.0) * k1
            + (-25360.0 / 2187.0) * k2
            + (64448.0 / 6561.0) * k3
            + (-212.0 / 729.0) * k4,
        )
        * h
    )
    var k6 = (
        _vd_f(
            K,
            C,
            Alpha,
            vel1,
            y0
            + (9017.0 / 3168.0) * k1
            + (-355.0 / 33.0) * k2
            + (46732.0 / 5247.0) * k3
            + (49.0 / 176.0) * k4
            + (-5103.0 / 18656.0) * k5,
        )
        * h
    )
    var yt = (
        y0
        + (35.0 / 384.0) * k1
        + (500.0 / 1113.0) * k3
        + (125.0 / 192.0) * k4
        + (-2187.0 / 6784.0) * k5
        + (11.0 / 84.0) * k6
    )
    var k7 = _vd_f(K, C, Alpha, vel1, yt) * h
    var error = (
        (71.0 / 57600.0) * k1
        + (-71.0 / 16695.0) * k3
        + (71.0 / 1920.0) * k4
        + (-17253.0 / 339200.0) * k5
        + (22.0 / 525.0) * k6
        + (-1.0 / 40.0) * k7
    )
    var eps = 1.0e300
    if yt != 0.0:
        eps = _abs(error / yt)
    return _VDStepResult(yt, eps, error)


fn _vd_abm6(
    K: Float64,
    C: Float64,
    Alpha: Float64,
    vel0: Float64,
    vel1: Float64,
    y0: Float64,
    h_in: Float64,
) -> _VDStepResult:
    var h = h_in / 6.0
    var dv = vel1 - vel0
    var f0 = _vd_f(K, C, Alpha, vel0, y0)
    var y11 = y0 + h * f0
    var f1 = _vd_f(K, C, Alpha, dv * (1.0 / 6.0) + vel0, y11)
    y11 = y0 + h * f1

    var y2 = y11 + 0.5 * h * (3.0 * f1 - 1.0 * f0)
    var f2 = _vd_f(K, C, Alpha, dv * (2.0 / 6.0) + vel0, y2)
    y2 = y11 + 0.5 * h * (f2 + f1)

    var y3 = y2 + (h / 12.0) * (23.0 * f2 - 16.0 * f1 + 5.0 * f0)
    var f3 = _vd_f(K, C, Alpha, dv * (3.0 / 6.0) + vel0, y3)
    y3 = y2 + (h / 12.0) * (5.0 * f3 + 8.0 * f2 - 1.0 * f1)

    var y4 = y3 + (h / 24.0) * (55.0 * f3 - 59.0 * f2 + 37.0 * f1 - 9.0 * f0)
    var f4 = _vd_f(K, C, Alpha, dv * (4.0 / 6.0) + vel0, y4)
    y4 = y3 + (h / 24.0) * (9.0 * f4 + 19.0 * f3 - 5.0 * f2 + f1)

    var y5 = y4 + (h / 720.0) * (1901.0 * f4 - 2774.0 * f3 + 2616.0 * f2 - 1274.0 * f1 + 251.0 * f0)
    var f5 = _vd_f(K, C, Alpha, dv * (5.0 / 6.0) + vel0, y5)
    y5 = y4 + (h / 720.0) * (251.0 * f5 + 646.0 * f4 - 264.0 * f3 + 106.0 * f2 - 19.0 * f1)

    var yp6 = y5 + (h / 1440.0) * (
        4277.0 * f5 - 7923.0 * f4 + 9982.0 * f3 - 7298.0 * f2 + 2877.0 * f1 - 475.0 * f0
    )
    var f6 = _vd_f(K, C, Alpha, vel1, yp6)
    var y6 = y5 + (h / 1440.0) * (
        475.0 * f6 + 1427.0 * f5 - 798.0 * f4 + 482.0 * f3 - 173.0 * f2 + 27.0 * f1
    )
    var error = yp6 - y6
    var eps = 1.0e300
    if y6 != 0.0:
        eps = _abs(error / y6)
    return _VDStepResult(y6, eps, error)


fn _vd_ros(
    K: Float64,
    C: Float64,
    Alpha: Float64,
    vel0: Float64,
    vel1: Float64,
    y0: Float64,
    h: Float64,
) -> _VDStepResult:
    var J = -K / (Alpha * C)
    var T = K
    var root2 = sqrt(2.0)
    var d = 1.0 / (2.0 + root2)
    var e32 = 6.0 + root2
    var W = 1.0 - h * d * J
    var f0 = _vd_f(K, C, Alpha, vel0, y0)
    var k1 = (f0 + h * d * T) / W
    var f1 = _vd_f(K, C, Alpha, (vel1 - vel0) * 0.5 + vel0, y0 + 0.5 * k1 * h)
    var k2 = (f1 - k1) / W + k1
    var y2 = y0 + h * k2
    var f2 = _vd_f(K, C, Alpha, vel1, y2)
    var k3 = (1.0 / W) * (f2 - e32 * (k2 - f1) - 2.0 * (k1 - f0) + h * d * T)
    var error = (h / 6.0) * (k1 - 2.0 * k2 + k3)
    var y3 = y2 + error
    var eps = 1.0e300
    if y3 != 0.0:
        eps = _abs(error / y3)
    return _VDStepResult(y2, eps, error)


fn _viscous_damper_set_trial(
    mat_def: UniMaterialDefinition,
    mut state: UniMaterialPointState,
    strain: Float64,
    strain_rate: Float64,
    dt: Float64,
):
    var K = mat_def.p0
    var C = mat_def.p1
    var Alpha = mat_def.p2
    var LGap = mat_def.p3
    var NM = mat_def.p4
    var RelTol = mat_def.p5
    var AbsTol = mat_def.p6
    var MaxHalf = mat_def.p7

    # Revert trial variables to last committed state (matches OpenSees ViscousDamper::setTrialStrain).
    state.eps_t = state.eps_c
    state.sig_t = state.sig_c
    state.tangent_t = state.tangent_c
    state.vd_vel_t = state.vd_vel_c
    state.vd_pugr_t = state.vd_pugr_c
    state.vd_nugr_t = state.vd_nugr_c

    var Vel = 0.0
    var acc = 0.0
    if _abs(strain_rate) != 0.0 and dt > 0.0:
        Vel = strain_rate
        acc = (Vel - state.vd_vel_t) / dt

    var smin = pow_nonneg(0.5, MaxHalf)
    var s = 1.0
    var stot = 0.0
    var fd0 = state.sig_t
    var vel0 = state.vd_vel_t

    while stot < 1.0:
        var h = s * dt
        var vel1 = vel0 + acc * h
        var res = _VDStepResult()
        if NM == 1.0:
            res = _vd_dormand_prince(K, C, Alpha, vel0, vel1, fd0, h)
        elif NM == 2.0:
            res = _vd_abm6(K, C, Alpha, vel0, vel1, fd0, h)
        elif NM == 3.0:
            res = _vd_ros(K, C, Alpha, vel0, vel1, fd0, h)

        if (res.eps <= RelTol) or (s == smin) or (_abs(res.error) <= AbsTol):
            vel0 = vel1
            fd0 = res.yt
            stot = stot + s
            if stot >= 1.0:
                break
        else:
            if s > smin:
                s = 0.5 * s
            else:
                s = smin

    if LGap > 0.0:
        var d_strain = strain - state.eps_t
        if fd0 > 0.0 and state.sig_t < 0.0:
            state.vd_pugr_t = state.eps_t + d_strain * _abs(fd0) / _abs(fd0 - state.sig_t)
            state.vd_nugr_t = 0.0
            if _abs(strain - state.vd_pugr_t) < LGap:
                fd0 = 0.0
        if fd0 < 0.0 and state.sig_t > 0.0:
            state.vd_nugr_t = state.eps_t + d_strain * _abs(fd0) / _abs(fd0 - state.sig_t)
            state.vd_pugr_t = 0.0
            if _abs(strain - state.vd_nugr_t) < LGap:
                fd0 = 0.0
        if _abs(state.vd_pugr_t) > 0.0 and state.sig_t == 0.0:
            if strain > state.vd_pugr_t and (strain - state.vd_pugr_t) < LGap:
                fd0 = 0.0
        if _abs(state.vd_nugr_t) > 0.0 and state.sig_t == 0.0:
            if strain < state.vd_nugr_t and (strain - state.vd_nugr_t) > -LGap:
                fd0 = 0.0

    state.sig_t = fd0
    state.tangent_t = 0.0
    state.vd_vel_t = Vel
    state.eps_t = strain
