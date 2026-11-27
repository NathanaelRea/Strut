from os import abort

from materials.uniaxial.py_tz_qz_state import _abs, PyTzQzState


alias _TZ_TOLERANCE: Float64 = 1.0e-12
alias _TZ_MAX_ITERATIONS: Int = 20


struct _TzParams(Copyable, Movable):
    var tz_type: Int
    var tult: Float64
    var z50: Float64
    var dashpot: Float64
    var zref: Float64
    var np: Float64
    var far_tang: Float64

    fn __init__(out self, p0: Float64, p1: Float64, p2: Float64, p3: Float64):
        var tz_type = Int(p0)
        var tult = p1
        var z50 = p2
        var dashpot = p3

        if tz_type == 0:
            tult = 1.0e-12
            z50 = 1.0e12
        if dashpot < 0.0:
            dashpot = 0.0
        if tult <= 0.0 or z50 <= 0.0:
            abort("TzSimple1 requires positive nonzero tult and z50")

        var zref: Float64
        var np: Float64
        var far_tang: Float64
        if tz_type == 0 or tz_type == 1:
            zref = 0.5 * z50
            np = 1.5
            far_tang = 0.70791 * tult / z50
        elif tz_type == 2:
            zref = 0.6 * z50
            np = 0.85
            far_tang = 2.0504 * tult / z50
        else:
            abort("TzSimple1 only accepts tzType of 1 or 2")
            zref = 0.5 * z50
            np = 1.5
            far_tang = 0.0

        self.tz_type = tz_type
        self.tult = tult
        self.z50 = z50
        self.dashpot = dashpot
        self.zref = zref
        self.np = np
        self.far_tang = far_tang


fn tz_simple1_initial_tangent(p0: Float64, p1: Float64, p2: Float64, p3: Float64) -> Float64:
    var p = _TzParams(p0, p1, p2, p3)
    var nf_tang = p.np * p.tult * (p.zref**p.np) * (p.zref ** (-p.np - 1.0))
    return (1.0 / nf_tang + 1.0 / p.far_tang) ** -1.0


fn tz_simple1_init_state(
    p0: Float64,
    p1: Float64,
    p2: Float64,
    p3: Float64,
    mut s: PyTzQzState,
) -> Float64:
    var p = _TzParams(p0, p1, p2, p3)
    s = PyTzQzState()

    s.t_far_p = 0.0
    s.t_far_y = 0.0
    s.t_far_tang = p.far_tang

    # Reuse nf_pinr/nf_yinr/nf_p/nf_y/nf_tang for TNF_tin/TNF_zin/TNF_t/TNF_z/TNF_tang
    s.t_nf_pinr = 0.0
    s.t_nf_yinr = 0.0
    s.t_nf_p = 0.0
    s.t_nf_y = 0.0
    s.t_nf_tang = p.np * p.tult * (p.zref**p.np) * (p.zref ** (-p.np - 1.0))
    s.t_p_raw = 0.0
    s.commit()

    return (1.0 / s.t_nf_tang + 1.0 / s.t_far_tang) ** -1.0


fn _tz_get_near_field(
    p: _TzParams, mut s: PyTzQzState, zlast: Float64, dz_in: Float64, dz_old: Float64
):
    var dz = dz_in
    if dz * dz_old < 0.0 and _abs(dz / dz_old) > 0.5:
        dz = -dz_old / 2.0

    s.t_nf_y = zlast + dz
    var dz_total = s.t_nf_y - s.c_nf_y

    if _abs(dz_total * s.t_nf_tang / p.tult) < 10.0 * _TZ_TOLERANCE:
        s.t_nf_p = s.t_nf_p + dz * s.t_nf_tang
        if _abs(s.t_nf_p) >= (1.0 - _TZ_TOLERANCE) * p.tult:
            s.t_nf_p = (s.t_nf_p / _abs(s.t_nf_p)) * (1.0 - _TZ_TOLERANCE) * p.tult
        return

    if s.t_nf_pinr != s.c_nf_pinr:
        s.t_nf_pinr = s.c_nf_pinr
        s.t_nf_yinr = s.c_nf_yinr

    if s.c_nf_y > s.c_nf_yinr and dz_total < 0.0:
        s.t_nf_pinr = s.c_nf_p
        s.t_nf_yinr = s.c_nf_y

    if s.c_nf_y < s.c_nf_yinr and dz_total > 0.0:
        s.t_nf_pinr = s.c_nf_p
        s.t_nf_yinr = s.c_nf_y

    if dz_total > 0.0:
        s.t_nf_p = p.tult - (p.tult - s.t_nf_pinr) * (p.zref**p.np) * (
            (p.zref + s.t_nf_y - s.t_nf_yinr) ** (-p.np)
        )
        s.t_nf_tang = (
            p.np
            * (p.tult - s.t_nf_pinr)
            * (p.zref**p.np)
            * ((p.zref + s.t_nf_y - s.t_nf_yinr) ** (-p.np - 1.0))
        )

    if dz_total < 0.0:
        s.t_nf_p = -p.tult + (p.tult + s.t_nf_pinr) * (p.zref**p.np) * (
            (p.zref - s.t_nf_y + s.t_nf_yinr) ** (-p.np)
        )
        s.t_nf_tang = (
            p.np
            * (p.tult + s.t_nf_pinr)
            * (p.zref**p.np)
            * ((p.zref - s.t_nf_y + s.t_nf_yinr) ** (-p.np - 1.0))
        )

    if _abs(s.t_nf_p) >= p.tult:
        s.t_nf_p = (s.t_nf_p / _abs(s.t_nf_p)) * (1.0 - _TZ_TOLERANCE) * p.tult
    if s.t_nf_tang <= 1.0e-4 * p.tult / p.z50:
        s.t_nf_tang = 1.0e-4 * p.tult / p.z50


fn tz_simple1_set_trial(
    p0: Float64,
    p1: Float64,
    p2: Float64,
    p3: Float64,
    mut eps_t: Float64,
    mut sig_t: Float64,
    mut tangent_t: Float64,
    mut s: PyTzQzState,
    newz: Float64,
):
    var p = _TzParams(p0, p1, p2, p3)
    var tz = eps_t
    var tt = s.t_p_raw
    var ttangent = tangent_t

    var dz = newz - tz
    var dt = ttangent * dz

    var num_steps = 1
    var temp = _abs(dt / p.tult)
    if temp > 0.5:
        if temp > 50.0:
            num_steps = 100
        else:
            num_steps = 1 + Int(temp * 2.0)

    temp = _abs(dz / p.z50)
    if temp > 1.0:
        if temp > 100.0:
            num_steps = 100
        else:
            num_steps = 1 + Int(temp)

    if num_steps > 100:
        num_steps = 100
    var step_size = 1.0 / Float64(num_steps)
    dz = step_size * dz

    for _ in range(num_steps):
        tz = tz + dz
        dt = ttangent * dz

        var dz_nf_old = ((tt + dt) - s.t_nf_p) / s.t_nf_tang

        for _ in range(1, _TZ_MAX_ITERATIONS):
            tt = tt + dt
            if _abs(tt) > (1.0 - _TZ_TOLERANCE) * p.tult:
                tt = (1.0 - _TZ_TOLERANCE) * p.tult * (tt / _abs(tt))

            var dz_nf = (tt - s.t_nf_p) / s.t_nf_tang
            _tz_get_near_field(p, s, s.t_nf_y, dz_nf, dz_nf_old)
            var t_unbalance = tt - s.t_nf_p
            var zres_nf = (tt - s.t_nf_p) / s.t_nf_tang
            dz_nf_old = dz_nf

            var dz_far = (tt - s.t_far_p) / s.t_far_tang
            s.t_far_y = s.t_far_y + dz_far
            s.t_far_p = s.t_far_tang * s.t_far_y
            var t_unbalance2 = tt - s.t_far_p
            var zres_far = (tt - s.t_far_p) / s.t_far_tang

            ttangent = (1.0 / s.t_nf_tang + 1.0 / s.t_far_tang) ** -1.0

            var dv = tz - (s.t_nf_y + zres_nf) - (s.t_far_y + zres_far)
            dt = ttangent * dv

            var tsum = _abs(t_unbalance) + _abs(t_unbalance2)
            if tsum / p.tult < _TZ_TOLERANCE:
                break

    s.t_p_raw = tt
    eps_t = tz
    tangent_t = ttangent
    if _abs(tt) >= (1.0 - _TZ_TOLERANCE) * p.tult:
        sig_t = (1.0 - _TZ_TOLERANCE) * p.tult * tt / _abs(tt)
    else:
        sig_t = tt
