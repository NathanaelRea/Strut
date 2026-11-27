from os import abort

from materials.uniaxial.py_tz_qz_state import _abs, PyTzQzState


alias _QZ_TOLERANCE: Float64 = 1.0e-12
alias _QZ_MAX_ITERATIONS: Int = 20


struct _QzParams(Copyable, Movable):
    var qz_type: Int
    var qult: Float64
    var z50: Float64
    var suction: Float64
    var dashpot: Float64
    var zref: Float64
    var np: Float64
    var init_elast: Float64
    var max_elast: Float64
    var nd: Float64
    var far_tang: Float64
    var nf_krig: Float64

    fn __init__(
        out self,
        p0: Float64,
        p1: Float64,
        p2: Float64,
        p3: Float64,
        p4: Float64,
    ):
        var qz_type = Int(p0)
        var qult = p1
        var z50 = p2
        var suction = p3
        var dashpot = p4

        if suction <= _QZ_TOLERANCE:
            suction = _QZ_TOLERANCE
        if suction > 0.1:
            suction = 0.1
        if dashpot < 0.0:
            dashpot = 0.0
        if qult <= 0.0 or z50 <= 0.0:
            abort("QzSimple1 requires positive nonzero Qult and z50")

        var zref: Float64
        var np: Float64
        var init_elast: Float64
        var max_elast: Float64
        var nd: Float64
        var far_tang: Float64
        if qz_type == 1:
            zref = 0.35 * z50
            np = 1.2
            init_elast = 0.2
            max_elast = 0.7
            nd = 1.0
            far_tang = 0.525 * qult / z50
        elif qz_type == 2:
            zref = 12.3 * z50
            np = 5.5
            init_elast = 0.3
            max_elast = 0.7
            nd = 1.0
            far_tang = 1.39 * qult / z50
        else:
            abort("QzSimple1 only accepts QzType of 1 or 2")
            zref = 0.35 * z50
            np = 1.2
            init_elast = 0.2
            max_elast = 0.7
            nd = 1.0
            far_tang = 0.0

        self.qz_type = qz_type
        self.qult = qult
        self.z50 = z50
        self.suction = suction
        self.dashpot = dashpot
        self.zref = zref
        self.np = np
        self.init_elast = init_elast
        self.max_elast = max_elast
        self.nd = nd
        self.far_tang = far_tang
        self.nf_krig = 10000.0 * qult / z50


fn qz_simple1_initial_tangent(
    p0: Float64, p1: Float64, p2: Float64, p3: Float64, p4: Float64
) -> Float64:
    var p = _QzParams(p0, p1, p2, p3, p4)
    var nf_tang = p.nf_krig
    var suction_tang = (
        p.nd * (p.qult * p.suction) * ((p.z50 / 2.0) ** p.nd) * ((p.z50 / 2.0) ** (-p.nd - 1.0))
    )
    var close_tang = 100.0 * p.qult / p.z50
    var gap_tang = close_tang + suction_tang
    return (1.0 / gap_tang + 1.0 / nf_tang + 1.0 / p.far_tang) ** -1.0


fn qz_simple1_init_state(
    p0: Float64,
    p1: Float64,
    p2: Float64,
    p3: Float64,
    p4: Float64,
    mut s: PyTzQzState,
) -> Float64:
    var p = _QzParams(p0, p1, p2, p3, p4)
    s = PyTzQzState()

    s.qz_elast = p.init_elast
    s.t_far_p = 0.0
    s.t_far_y = 0.0
    s.t_far_tang = p.far_tang

    s.t_nf_pinr = s.qz_elast * p.qult
    s.t_nf_pinl = -s.t_nf_pinr
    s.t_nf_yinr = s.t_nf_pinr / p.nf_krig
    s.t_nf_yinl = -s.t_nf_yinr
    s.t_nf_p = 0.0
    s.t_nf_y = 0.0
    s.t_nf_tang = p.nf_krig

    s.t_drag_pin = 0.0
    s.t_drag_yin = 0.0
    s.t_drag_p = 0.0
    s.t_drag_y = 0.0
    s.t_drag_tang = (
        p.nd
        * (p.qult * p.suction - s.t_drag_p)
        * ((p.z50 / 2.0) ** p.nd)
        * ((p.z50 / 2.0 - s.t_drag_y + s.t_drag_yin) ** (-p.nd - 1.0))
    )

    s.t_close_p = 0.0
    s.t_close_y = 0.0
    s.t_close_tang = 100.0 * p.qult / p.z50

    s.t_gap_y = 0.0
    s.t_gap_p = 0.0
    s.t_gap_tang = s.t_close_tang + s.t_drag_tang
    s.t_p_raw = 0.0
    s.commit()

    return (1.0 / s.t_gap_tang + 1.0 / s.t_nf_tang + 1.0 / s.t_far_tang) ** -1.0


fn _qz_get_closure(p: _QzParams, mut s: PyTzQzState, zlast: Float64, dz: Float64):
    s.t_close_y = zlast + dz
    if s.t_close_y <= 0.0:
        s.t_close_tang = 1000.0 * p.qult / p.z50
        s.t_close_p = s.t_close_y * s.t_close_tang
    if s.t_close_y > 0.0:
        s.t_close_tang = 1.0e-6 * p.qult / p.z50
        s.t_close_p = s.t_close_y * s.t_close_tang


fn _qz_get_suction(p: _QzParams, mut s: PyTzQzState, zlast: Float64, dz: Float64):
    s.t_drag_y = zlast + dz
    var qmax = p.suction * p.qult
    var dz_total = s.t_drag_y - s.c_drag_y

    if _abs(dz_total * s.t_drag_tang / p.qult) < 3.0 * _QZ_TOLERANCE:
        s.t_drag_p = s.t_drag_p + dz * s.t_drag_tang
        if _abs(s.t_drag_p) >= qmax:
            s.t_drag_p = (s.t_drag_p / _abs(s.t_drag_p)) * (1.0 - 1.0e-8) * qmax
        return

    if s.t_drag_pin != s.c_drag_pin:
        s.t_drag_pin = s.c_drag_pin
        s.t_drag_yin = s.c_drag_yin

    if s.c_drag_y > s.c_drag_yin and dz_total < 0.0:
        s.t_drag_pin = s.c_drag_p
        s.t_drag_yin = s.c_drag_y

    if s.c_drag_y < s.c_drag_yin and dz_total > 0.0:
        s.t_drag_pin = s.c_drag_p
        s.t_drag_yin = s.c_drag_y

    if dz_total >= 0.0:
        s.t_drag_p = qmax - (qmax - s.t_drag_pin) * ((0.5 * p.z50) ** p.nd) * (
            (0.5 * p.z50 + s.t_drag_y - s.t_drag_yin) ** (-p.nd)
        )
        s.t_drag_tang = (
            p.nd
            * (qmax - s.t_drag_pin)
            * ((0.5 * p.z50) ** p.nd)
            * ((0.5 * p.z50 + s.t_drag_y - s.t_drag_yin) ** (-p.nd - 1.0))
        )

    if dz_total < 0.0:
        s.t_drag_p = -qmax + (qmax + s.t_drag_pin) * ((0.5 * p.z50) ** p.nd) * (
            (0.5 * p.z50 - s.t_drag_y + s.t_drag_yin) ** (-p.nd)
        )
        s.t_drag_tang = (
            p.nd
            * (qmax + s.t_drag_pin)
            * ((0.5 * p.z50) ** p.nd)
            * ((0.5 * p.z50 - s.t_drag_y + s.t_drag_yin) ** (-p.nd - 1.0))
        )

    if _abs(s.t_drag_p) >= (1.0 - _QZ_TOLERANCE) * qmax:
        s.t_drag_p = (s.t_drag_p / _abs(s.t_drag_p)) * (1.0 - _QZ_TOLERANCE) * qmax
    if s.t_drag_tang <= 1.0e-4 * p.qult / p.z50:
        s.t_drag_tang = 1.0e-4 * p.qult / p.z50


fn _qz_get_gap(p: _QzParams, mut s: PyTzQzState, zlast: Float64, dz_in: Float64):
    var dz = dz_in
    if zlast > 0.0 and (zlast + dz) < -_QZ_TOLERANCE:
        dz = -_QZ_TOLERANCE - zlast
    if zlast < 0.0 and (zlast + dz) > _QZ_TOLERANCE:
        dz = _QZ_TOLERANCE - zlast
    s.t_gap_y = zlast + dz

    _qz_get_closure(p, s, zlast, dz)
    _qz_get_suction(p, s, zlast, dz)
    s.t_gap_p = s.t_drag_p + s.t_close_p
    s.t_gap_tang = s.t_drag_tang + s.t_close_tang


fn _qz_get_near_field(
    p: _QzParams, mut s: PyTzQzState, zlast: Float64, dz_in: Float64, dz_old: Float64
):
    var dz = dz_in
    if dz * dz_old < 0.0 and _abs(dz / dz_old) > 0.5:
        dz = -dz_old / 2.0
    if dz * dz_old < -p.z50 * p.z50:
        dz = (s.t_nf_yinr + s.t_nf_yinl) / 2.0 - zlast

    s.t_nf_y = zlast + dz
    var nf_dz = s.t_nf_y - s.c_nf_y

    if _abs(nf_dz * s.t_nf_tang / p.qult) < 3.0 * _QZ_TOLERANCE:
        s.t_nf_p = s.t_nf_p + dz * s.t_nf_tang
        if _abs(s.t_nf_p) >= p.qult:
            s.t_nf_p = (s.t_nf_p / _abs(s.t_nf_p)) * (1.0 - _QZ_TOLERANCE) * p.qult
        return

    if s.t_nf_pinr != s.c_nf_pinr or s.t_nf_pinl != s.c_nf_pinl:
        s.t_nf_pinr = s.c_nf_pinr
        s.t_nf_pinl = s.c_nf_pinl
        s.t_nf_yinr = s.c_nf_yinr
        s.t_nf_yinl = s.c_nf_yinl

    var change_direction = False
    if s.c_nf_p > s.c_nf_pinr and nf_dz < 0.0:
        change_direction = True
        if (s.c_nf_p - s.c_nf_pinl) > 2.0 * p.qult * s.qz_elast:
            s.qz_elast = (s.c_nf_p - s.c_nf_pinl) / (2.0 * p.qult)
        if 2.0 * s.qz_elast > p.max_elast:
            s.qz_elast = p.max_elast / 2.0
        s.t_nf_pinr = s.c_nf_p
        s.t_nf_pinl = s.t_nf_pinr - 2.0 * p.qult * s.qz_elast
        s.t_nf_yinr = s.c_nf_y
        s.t_nf_yinl = s.t_nf_yinr - (s.t_nf_pinr - s.t_nf_pinl) / p.nf_krig

    if s.c_nf_p < s.c_nf_pinl and nf_dz > 0.0:
        change_direction = True
        if (s.c_nf_pinr - s.c_nf_p) > 2.0 * p.qult * s.qz_elast:
            s.qz_elast = (s.c_nf_pinr - s.c_nf_p) / (2.0 * p.qult)
        if 2.0 * s.qz_elast > p.max_elast:
            s.qz_elast = p.max_elast / 2.0
        s.t_nf_pinl = s.c_nf_p
        s.t_nf_pinr = s.t_nf_pinl + 2.0 * p.qult * s.qz_elast
        s.t_nf_yinl = s.c_nf_y
        s.t_nf_yinr = s.t_nf_yinl + (s.t_nf_pinr - s.t_nf_pinl) / p.nf_krig

    if change_direction:
        var maxdz = s.qz_elast * p.qult / p.nf_krig
        if _abs(dz) > maxdz:
            dz = (dz / _abs(dz)) * maxdz

    s.t_nf_y = zlast + dz

    if nf_dz >= 0.0:
        if s.t_nf_y <= s.t_nf_yinr:
            s.t_nf_tang = p.nf_krig
            s.t_nf_p = s.t_nf_pinl + (s.t_nf_y - s.t_nf_yinl) * p.nf_krig
        else:
            s.t_nf_tang = (
                p.np
                * (p.qult - s.t_nf_pinr)
                * (p.zref**p.np)
                * ((p.zref - s.t_nf_yinr + s.t_nf_y) ** (-p.np - 1.0))
            )
            s.t_nf_p = p.qult - (p.qult - s.t_nf_pinr) * (
                (p.zref / (p.zref - s.t_nf_yinr + s.t_nf_y)) ** p.np
            )

    if nf_dz < 0.0:
        if s.t_nf_y >= s.t_nf_yinl:
            s.t_nf_tang = p.nf_krig
            s.t_nf_p = s.t_nf_pinr + (s.t_nf_y - s.t_nf_yinr) * p.nf_krig
        else:
            s.t_nf_tang = (
                p.np
                * (p.qult + s.t_nf_pinl)
                * (p.zref**p.np)
                * ((p.zref + s.t_nf_yinl - s.t_nf_y) ** (-p.np - 1.0))
            )
            s.t_nf_p = -p.qult + (p.qult + s.t_nf_pinl) * (
                (p.zref / (p.zref + s.t_nf_yinl - s.t_nf_y)) ** p.np
            )

    if _abs(s.t_nf_p) >= (1.0 - _QZ_TOLERANCE) * p.qult:
        s.t_nf_p = (s.t_nf_p / _abs(s.t_nf_p)) * (1.0 - _QZ_TOLERANCE) * p.qult
        s.t_nf_tang = 1.0e-4 * p.qult / p.z50
    if s.t_nf_tang <= 1.0e-4 * p.qult / p.z50:
        s.t_nf_tang = 1.0e-4 * p.qult / p.z50


fn qz_simple1_set_trial(
    p0: Float64,
    p1: Float64,
    p2: Float64,
    p3: Float64,
    p4: Float64,
    mut eps_t: Float64,
    mut sig_t: Float64,
    mut tangent_t: Float64,
    mut s: PyTzQzState,
    newz: Float64,
):
    var p = _QzParams(p0, p1, p2, p3, p4)
    var tz = eps_t
    var tq = s.t_p_raw
    var ttangent = tangent_t

    var dz = newz - tz
    var dq = ttangent * dz

    var num_steps = 1
    var temp = _abs(dq / p.qult)
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
        dq = ttangent * dz

        var dz_nf_old = ((tq + dq) - s.t_nf_p) / s.t_nf_tang

        for _ in range(1, _QZ_MAX_ITERATIONS):
            tq = tq + dq
            if _abs(tq) > (1.0 - _QZ_TOLERANCE) * p.qult:
                tq = (1.0 - _QZ_TOLERANCE) * p.qult * (tq / _abs(tq))

            var dz_nf = (tq - s.t_nf_p) / s.t_nf_tang
            _qz_get_near_field(p, s, s.t_nf_y, dz_nf, dz_nf_old)
            var q_unbalance = tq - s.t_nf_p
            var zres_nf = (tq - s.t_nf_p) / s.t_nf_tang
            dz_nf_old = dz_nf

            var dz_gap = (tq - s.t_gap_p) / s.t_gap_tang
            _qz_get_gap(p, s, s.t_gap_y, dz_gap)
            var q_unbalance2 = tq - s.t_gap_p
            var zres_gap = (tq - s.t_gap_p) / s.t_gap_tang

            var dz_far = (tq - s.t_far_p) / s.t_far_tang
            s.t_far_y = s.t_far_y + dz_far
            s.t_far_p = s.t_far_tang * s.t_far_y
            var q_unbalance3 = tq - s.t_far_p
            var zres_far = (tq - s.t_far_p) / s.t_far_tang

            ttangent = (1.0 / s.t_gap_tang + 1.0 / s.t_nf_tang + 1.0 / s.t_far_tang) ** -1.0

            var dv = tz - (s.t_gap_y + zres_gap) - (s.t_nf_y + zres_nf) - (s.t_far_y + zres_far)
            dq = ttangent * dv

            var qsum = (_abs(q_unbalance) + _abs(q_unbalance2) + _abs(q_unbalance3)) / 3.0
            if qsum / p.qult < _QZ_TOLERANCE:
                break

    s.t_p_raw = tq
    eps_t = tz
    tangent_t = ttangent
    if _abs(tq) >= (1.0 - _QZ_TOLERANCE) * p.qult:
        sig_t = (1.0 - _QZ_TOLERANCE) * p.qult * tq / _abs(tq)
    else:
        sig_t = tq
